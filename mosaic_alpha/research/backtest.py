"""Transaction-cost-aware long-short portfolio backtester.

Constructs a dollar-neutral sector ETF portfolio from cross-sectional model
predictions and simulates realistic performance accounting for transaction
costs.

Portfolio construction
----------------------
On each rebalance date *t*:

- **Long book**: the top ``quantile`` fraction of tickers ranked by model score,
  equal-weighted so that long weights sum to +1.
- **Short book**: the bottom ``quantile`` fraction, equal-weighted so that short
  weights sum to -1.
- **Dollar-neutral**: net weight = long + short = 0.

With the default ``quantile=0.25`` and 12 tickers:
- n_long = 3 (top 3), n_short = 3 (bottom 3).

Rebalancing
-----------
Portfolios are rebalanced on non-overlapping windows of length ``horizon``
trading days so that each realized return (`fwd_ret_<horizon>`) is used
exactly once.  The unique test dates from the pooled prediction DataFrame are
indexed every ``horizon`` steps: dates[0], dates[horizon], dates[2*horizon], …

Transaction costs
-----------------
Round-trip transaction cost per rebalance:

    cost_t = (cost_bps / 10_000) * one_way_turnover_t

where ``one_way_turnover_t = sum(|w_t - w_{t-1}|) / 2`` (fraction of the
portfolio notional that changes hands).  The first rebalance has turnover 1.0
because the portfolio is built from cash.

Gross vs net returns
--------------------
- **Gross period return**: ``sum(weights_t * fwd_ret_{horizon}_t)`` -- log return.
- **Net period return**: ``gross_t - cost_t``.

All returns are per-period log returns.  Performance metrics annualise by
``periods_per_year = 252 / horizon``.

Limitations
-----------
- Assumes perfect fill at the close on the signal date (no market impact).
- Log returns are additive (approximation valid for small returns).
- Short selling assumed frictionless beyond the explicit cost_bps charge.
- The universe is small (12 ETFs); t-stats on Sharpe ratios should be
  interpreted with wide confidence intervals.
- Results do not constitute a trading strategy recommendation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """All outputs from a single backtest run."""

    score_col: str
    label_col: str
    quantile: float
    cost_bps: float
    horizon: int
    n_periods: int          # number of non-overlapping rebalance periods

    # ── Gross (before costs)
    ann_gross_return: float
    ann_vol_gross: float
    sharpe_gross: float
    max_drawdown_gross: float

    # ── Net (after costs)
    ann_net_return: float
    ann_vol_net: float
    sharpe_net: float
    max_drawdown_net: float

    hit_rate: float         # fraction of periods with positive gross return
    avg_turnover: float     # mean one-way turnover per rebalance

    # ── Period-level time series (date-indexed)
    period_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    # columns: gross_return, net_return, turnover, n_long, n_short, cost

    cumulative_gross: pd.Series = field(default_factory=pd.Series)
    cumulative_net: pd.Series = field(default_factory=pd.Series)
    # Both series represent exp(cumsum(log_returns)) -- dollar growth of $1.


# ── Weight construction ────────────────────────────────────────────────────────

def build_ls_weights(scores: pd.Series, quantile: float = 0.25) -> pd.Series | None:
    """Build equal-weight dollar-neutral L/S weights for one cross-section.

    Parameters
    ----------
    scores:
        Series mapping ticker (index) to model score for this date.
        NaN entries are dropped before ranking.
    quantile:
        Fraction of the cross-section to include in each leg (default 0.25).

    Returns
    -------
    ``pd.Series`` of weights indexed by ticker, or ``None`` if there are too
    few assets to form both a long and a short book.

    Properties of the returned weights
    -----------------------------------
    - Long weights > 0 and sum to +1.
    - Short weights < 0 and sum to -1.
    - Net sum = 0 (dollar-neutral).
    """
    scores = scores.dropna()
    n = len(scores)
    if n < 2:
        return None

    n_side = max(1, int(np.floor(n * quantile)))

    # Rank ascending: rank 1 = lowest score (short candidate)
    ranked = scores.rank(method="first", ascending=True)
    weights = pd.Series(0.0, index=scores.index)

    long_mask = ranked > (n - n_side)   # top n_side by score
    short_mask = ranked <= n_side       # bottom n_side by score

    # Guard against overlap (possible when n == 2 * n_side and there are ties)
    if long_mask.sum() == 0 or short_mask.sum() == 0:
        return None
    if (long_mask & short_mask).any():
        return None

    weights[long_mask] = 1.0 / long_mask.sum()
    weights[short_mask] = -1.0 / short_mask.sum()

    return weights


# ── Turnover ───────────────────────────────────────────────────────────────────

def compute_turnover(
    weights_new: pd.Series,
    weights_old: pd.Series | None,
) -> float:
    """Compute one-way portfolio turnover between two weight vectors.

    Parameters
    ----------
    weights_new:
        Weight vector for the next period (indexed by ticker).
    weights_old:
        Weight vector for the previous period, or ``None`` (first rebalance).

    Returns
    -------
    One-way turnover as a fraction of portfolio notional (0.0 to 1.0+ range).
    A completely new portfolio (from cash) returns 1.0.
    """
    if weights_old is None:
        # First period: build from zero — one-way cost on full gross exposure
        return float(weights_new.abs().sum() / 2)

    # Align on the union of tickers; missing tickers had weight 0
    all_tickers = weights_new.index.union(weights_old.index)
    wn = weights_new.reindex(all_tickers, fill_value=0.0)
    wo = weights_old.reindex(all_tickers, fill_value=0.0)
    return float((wn - wo).abs().sum() / 2)


# ── Performance metrics ────────────────────────────────────────────────────────

def _annualise(returns: np.ndarray, horizon: int) -> tuple[float, float, float]:
    """Return (ann_return, ann_vol, sharpe) for a period-return array."""
    ppy = _TRADING_DAYS_PER_YEAR / horizon
    ann_ret = float(np.mean(returns)) * ppy
    ann_vol = float(np.std(returns, ddof=1)) * np.sqrt(ppy) if len(returns) > 1 else float("nan")
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else float("nan")
    return ann_ret, ann_vol, sharpe


def _max_drawdown(cumulative: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of a cumulative dollar-growth series.

    Returns a negative number (e.g., -0.15 = -15% max drawdown).
    """
    if len(cumulative) == 0:
        return float("nan")
    roll_max = cumulative.expanding().max()
    drawdown = cumulative / roll_max - 1.0
    return float(drawdown.min())


# ── Main backtest runner ───────────────────────────────────────────────────────

def run_backtest(
    pooled_df: pd.DataFrame,
    score_col: str,
    label_col: str,
    *,
    quantile: float = 0.25,
    cost_bps: float = 5.0,
    horizon: int = 5,
) -> BacktestResult:
    """Run a long-short backtest on pooled walk-forward model predictions.

    Parameters
    ----------
    pooled_df:
        Flat DataFrame (no MultiIndex) with at minimum columns:
        ``date``, ``ticker``, *score_col*, *label_col*.
        Typically the ``pooled_predictions`` from :class:`NewsSectorResult`.
    score_col:
        Column containing the model prediction (used for L/S ranking).
    label_col:
        Column containing the realised forward return for each date.
    quantile:
        Fraction of tickers in each leg (default 0.25 = top/bottom 25%).
    cost_bps:
        Round-trip transaction cost in basis points applied to one-way
        turnover each rebalance (default 5 bps).
    horizon:
        Rebalance frequency in trading days.  Must match the forward-return
        horizon encoded in *label_col* (default 5 for ``fwd_ret_5``).

    Returns
    -------
    :class:`BacktestResult` with per-period time series and aggregated metrics.

    Notes
    -----
    The backtest samples non-overlapping rebalance dates from the pooled
    DataFrame: ``unique_dates[::horizon]``.  Dates with too few assets to
    form both legs are silently skipped.
    """
    required = {"date", "ticker", score_col, label_col}
    missing = required - set(pooled_df.columns)
    if missing:
        raise ValueError(f"pooled_df missing columns: {missing}")

    # Sort and collect unique test dates
    unique_dates = pd.DatetimeIndex(
        pooled_df["date"].sort_values().unique()
    )
    rebalance_dates = unique_dates[::horizon]
    logger.info(
        "Backtest: %d rebalance periods (horizon=%d, quantile=%.2f, cost=%.1fbps)",
        len(rebalance_dates), horizon, quantile, cost_bps,
    )

    rows: list[dict] = []
    prev_weights: pd.Series | None = None

    for date in rebalance_dates:
        cross_section = pooled_df[pooled_df["date"] == date].copy()
        cross_section = cross_section.dropna(subset=[score_col, label_col])

        if cross_section.empty:
            logger.debug("Skipping %s: no valid rows", date.date())
            continue

        cross_section = cross_section.set_index("ticker")
        weights = build_ls_weights(cross_section[score_col], quantile=quantile)

        if weights is None:
            logger.debug(
                "Skipping %s: too few assets (%d) for quantile=%.2f",
                date.date(), len(cross_section), quantile,
            )
            continue

        # Realised gross return: sum of weighted log returns
        gross = float((weights * cross_section[label_col]).sum())

        # Turnover and cost
        turnover = compute_turnover(weights, prev_weights)
        cost = (cost_bps / 10_000) * turnover
        net = gross - cost

        n_long = int((weights > 0).sum())
        n_short = int((weights < 0).sum())

        rows.append({
            "date": date,
            "gross_return": gross,
            "net_return": net,
            "turnover": turnover,
            "cost": cost,
            "n_long": n_long,
            "n_short": n_short,
        })

        prev_weights = weights

    if not rows:
        logger.warning("Backtest produced zero periods -- returning empty result")
        return BacktestResult(
            score_col=score_col, label_col=label_col,
            quantile=quantile, cost_bps=cost_bps, horizon=horizon,
            n_periods=0,
            ann_gross_return=float("nan"), ann_vol_gross=float("nan"),
            sharpe_gross=float("nan"), max_drawdown_gross=float("nan"),
            ann_net_return=float("nan"), ann_vol_net=float("nan"),
            sharpe_net=float("nan"), max_drawdown_net=float("nan"),
            hit_rate=float("nan"), avg_turnover=float("nan"),
        )

    period_df = pd.DataFrame(rows).set_index("date")

    gross_arr = period_df["gross_return"].values
    net_arr = period_df["net_return"].values

    ann_gross, vol_gross, sharpe_gross = _annualise(gross_arr, horizon)
    ann_net, vol_net, sharpe_net = _annualise(net_arr, horizon)

    # Cumulative growth: exp(cumsum of log returns)
    cum_gross = np.exp(np.cumsum(gross_arr))
    cum_net = np.exp(np.cumsum(net_arr))
    cum_gross_s = pd.Series(cum_gross, index=period_df.index, name="cum_gross")
    cum_net_s = pd.Series(cum_net, index=period_df.index, name="cum_net")

    hit_rate = float(np.mean(gross_arr > 0))
    avg_turnover = float(period_df["turnover"].mean())

    logger.info(
        "Gross: ann_ret=%.2f%% ann_vol=%.2f%% Sharpe=%.2f MDD=%.2f%%",
        ann_gross * 100, vol_gross * 100, sharpe_gross,
        _max_drawdown(cum_gross_s) * 100,
    )
    logger.info(
        "Net:   ann_ret=%.2f%% ann_vol=%.2f%% Sharpe=%.2f MDD=%.2f%%",
        ann_net * 100, vol_net * 100, sharpe_net,
        _max_drawdown(cum_net_s) * 100,
    )

    return BacktestResult(
        score_col=score_col,
        label_col=label_col,
        quantile=quantile,
        cost_bps=cost_bps,
        horizon=horizon,
        n_periods=len(period_df),
        ann_gross_return=ann_gross,
        ann_vol_gross=vol_gross,
        sharpe_gross=sharpe_gross,
        max_drawdown_gross=_max_drawdown(cum_gross_s),
        ann_net_return=ann_net,
        ann_vol_net=vol_net,
        sharpe_net=sharpe_net,
        max_drawdown_net=_max_drawdown(cum_net_s),
        hit_rate=hit_rate,
        avg_turnover=avg_turnover,
        period_returns=period_df,
        cumulative_gross=cum_gross_s,
        cumulative_net=cum_net_s,
    )


# ── Report renderer ────────────────────────────────────────────────────────────

def render_report(result: BacktestResult, path: Path, data_mode: str = "live_gdelt") -> None:
    """Write a Markdown backtest report to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)

    is_sample = data_mode == "offline_sample"

    def _pct(v: float) -> str:
        return f"{v*100:+.2f}%" if not np.isnan(v) else "N/A"

    def _f2(v: float) -> str:
        return f"{v:+.2f}" if not np.isnan(v) else "N/A"

    def _f4(v: float) -> str:
        return f"{v:.4f}" if not np.isnan(v) else "N/A"

    lines: list[str] = [
        "# Long-Short Sector Backtest",
        "",
    ]

    if is_sample:
        lines += [
            "> **WARNING: SAMPLE DATA -- news features use synthetic counts.**  ",
            "> Backtest results involving news-based scores have no real-world validity.",
            "",
        ]

    lines += [
        f"**Data mode:** {data_mode}  ",
        f"**Score column:** `{result.score_col}`  ",
        f"**Label column:** `{result.label_col}`  ",
        f"**Rebalance horizon:** {result.horizon} trading days  ",
        f"**Quantile (each leg):** {result.quantile:.0%}  ",
        f"**Transaction cost:** {result.cost_bps:.1f} bps (one-way turnover)  ",
        f"**Rebalance periods:** {result.n_periods}  ",
        f"**Average turnover per rebalance:** {result.avg_turnover:.3f}  ",
        "",
        "## Performance summary",
        "",
        "| Metric | Gross | Net |",
        "|---|---|---|",
        f"| Annualised return | {_pct(result.ann_gross_return)} | {_pct(result.ann_net_return)} |",
        f"| Annualised volatility | {_pct(result.ann_vol_gross)} | {_pct(result.ann_vol_net)} |",
        f"| Sharpe ratio | {_f2(result.sharpe_gross)} | {_f2(result.sharpe_net)} |",
        f"| Max drawdown | {_pct(result.max_drawdown_gross)} | {_pct(result.max_drawdown_net)} |",
        f"| Hit rate | {result.hit_rate:.3f} | -- |",
        "",
    ]

    if not result.period_returns.empty:
        lines += [
            "## Period returns (first 10 rebalances)",
            "",
            "| Date | Gross | Net | Turnover | Cost | n_long | n_short |",
            "|---|---|---|---|---|---|---|",
        ]
        for date, row in result.period_returns.head(10).iterrows():
            lines.append(
                f"| {date.date()} "
                f"| {row['gross_return']:+.4f} "
                f"| {row['net_return']:+.4f} "
                f"| {row['turnover']:.4f} "
                f"| {row['cost']:.6f} "
                f"| {int(row['n_long'])} "
                f"| {int(row['n_short'])} |"
            )
        lines.append("")

    lines += [
        "## Limitations",
        "",
        "- This is a research backtest, not a live-tradeable strategy.",
        "- The universe contains only 12 sector ETFs; cross-sectional IC is",
        "  statistically weak at this universe size.",
        "- Perfect execution at closing prices is assumed.",
        "- No market-impact modelling; actual costs for large positions would be higher.",
        "- Short selling assumed frictionless beyond the flat cost_bps charge.",
        "- Log-return additivity holds for small returns; large log returns may",
        "  cause slight discrepancies vs. compound arithmetic returns.",
        "- Walk-forward training ensures no look-ahead in signal generation,",
        "  but the backtest itself was not optimised on a hold-out period.",
        "- Do not interpret these results as evidence of future performance.",
        "",
        "---",
        "_Generated by MosaicAlpha backtest pipeline._",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Backtest report written to %s", path)
