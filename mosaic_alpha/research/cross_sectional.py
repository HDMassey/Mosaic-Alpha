"""Cross-sectional evaluation metrics for panel return-prediction models.

In a cross-sectional framework we evaluate on each *date* separately:
for a given day t, the model has predicted a score for each ticker in
the universe.  The metrics below measure how well those scores rank the
tickers relative to their realised forward returns.

Metrics
-------
daily_cs_ic         : Per-date Pearson IC across tickers.
daily_cs_rank_ic    : Per-date Spearman rank IC across tickers.
cs_hit_rate         : Fraction of dates where IC > 0.
cs_ls_spread        : Daily long-top / short-bottom quantile return spread.
aggregate_cs_metrics: Aggregate a time series of daily metrics.

The panel is expected as a flat DataFrame (not MultiIndex) with columns
``date``, ``ticker``, one or more feature/score columns, and one label column.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


# ── Per-date scalar metrics ────────────────────────────────────────────────────

def _pearson(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation, returning NaN when not computable."""
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    r, _ = stats.pearsonr(a.values, b.values)
    return float(r)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2:
        return float("nan")
    r, _ = stats.spearmanr(a.values, b.values)
    return float(r)


# ── Daily cross-sectional series ──────────────────────────────────────────────

def daily_cs_ic(
    panel: pd.DataFrame,
    score_col: str,
    label_col: str,
    date_col: str = "date",
) -> pd.Series:
    """Compute per-date Pearson IC between *score_col* and *label_col*.

    Parameters
    ----------
    panel:
        Flat DataFrame with columns for date, score, and label.
    score_col:
        Column name of the model score or feature to evaluate.
    label_col:
        Column name of the forward return label.
    date_col:
        Column name of the date field.

    Returns
    -------
    Series indexed by date, one IC value per trading day.
    """
    return (
        panel.groupby(date_col)
        .apply(lambda g: _pearson(g[score_col], g[label_col]))
        .rename("ic")
    )


def daily_cs_rank_ic(
    panel: pd.DataFrame,
    score_col: str,
    label_col: str,
    date_col: str = "date",
) -> pd.Series:
    """Compute per-date Spearman rank IC between *score_col* and *label_col*."""
    return (
        panel.groupby(date_col)
        .apply(lambda g: _spearman(g[score_col], g[label_col]))
        .rename("rank_ic")
    )


def daily_cs_ls_spread(
    panel: pd.DataFrame,
    score_col: str,
    label_col: str,
    date_col: str = "date",
    n_quantiles: int = 3,
) -> pd.Series:
    """Per-date long-top / short-bottom quantile return spread.

    With *n_quantiles* = 3, longs are the top tertile and shorts the bottom
    tertile.  Sensible for small universes (e.g. 12 sector ETFs).

    Returns
    -------
    Series indexed by date, one spread value per trading day.
    """
    def _spread(g: pd.DataFrame) -> float:
        if len(g) < n_quantiles * 2:
            return float("nan")
        try:
            q = pd.qcut(g[score_col], n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            return float("nan")
        top = g.loc[q == q.max(), label_col].mean()
        bot = g.loc[q == q.min(), label_col].mean()
        return float(top - bot)

    return (
        panel.groupby(date_col)
        .apply(_spread)
        .rename("ls_spread")
    )


# ── Aggregated result dataclass ────────────────────────────────────────────────

@dataclass
class CrossSectionalResult:
    """Aggregated cross-sectional metrics for one (feature/score, label) pair."""

    score_col: str
    label_col: str
    n_dates: int

    mean_ic: float
    std_ic: float
    ic_t_stat: float       # mean_ic / (std_ic / sqrt(n_dates))

    mean_rank_ic: float
    std_rank_ic: float

    ic_hit_rate: float     # fraction of days with IC > 0

    mean_ls_spread: float
    std_ls_spread: float

    # Full daily series for plotting / further analysis
    daily_ic: pd.Series = field(default_factory=pd.Series)
    daily_rank_ic: pd.Series = field(default_factory=pd.Series)
    daily_ls_spread: pd.Series = field(default_factory=pd.Series)


def aggregate_cs_metrics(
    panel: pd.DataFrame,
    score_col: str,
    label_col: str,
    date_col: str = "date",
    n_quantiles: int = 3,
) -> CrossSectionalResult:
    """Compute and aggregate all cross-sectional metrics.

    Parameters
    ----------
    panel:
        Flat DataFrame (no MultiIndex) with date, ticker, score, and label columns.
    score_col:
        Column with the model prediction or raw feature to evaluate.
    label_col:
        Column with the forward return label.
    date_col:
        Column name for the date field.
    n_quantiles:
        Number of quantile buckets for the L/S spread (default 3 = terciles).

    Returns
    -------
    :class:`CrossSectionalResult` with per-date series and aggregated scalars.
    """
    ic_series = daily_cs_ic(panel, score_col, label_col, date_col)
    rank_series = daily_cs_rank_ic(panel, score_col, label_col, date_col)
    ls_series = daily_cs_ls_spread(panel, score_col, label_col, date_col, n_quantiles)

    n = int(ic_series.notna().sum())
    mean_ic = float(np.nanmean(ic_series))
    std_ic = float(np.nanstd(ic_series, ddof=1)) if n > 1 else float("nan")
    ic_t = mean_ic / (std_ic / np.sqrt(n)) if n > 1 and std_ic > 0 else float("nan")

    rank_valid = rank_series.dropna()
    ls_valid = ls_series.dropna()

    return CrossSectionalResult(
        score_col=score_col,
        label_col=label_col,
        n_dates=n,
        mean_ic=mean_ic,
        std_ic=std_ic,
        ic_t_stat=ic_t,
        mean_rank_ic=float(np.nanmean(rank_valid)) if len(rank_valid) else float("nan"),
        std_rank_ic=float(np.nanstd(rank_valid, ddof=1)) if len(rank_valid) > 1 else float("nan"),
        ic_hit_rate=float(np.nanmean(ic_series > 0)),
        mean_ls_spread=float(np.nanmean(ls_valid)) if len(ls_valid) else float("nan"),
        std_ls_spread=float(np.nanstd(ls_valid, ddof=1)) if len(ls_valid) > 1 else float("nan"),
        daily_ic=ic_series,
        daily_rank_ic=rank_series,
        daily_ls_spread=ls_series,
    )
