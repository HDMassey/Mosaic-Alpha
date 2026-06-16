"""News-augmented sector ETF cross-sectional baseline.

Extends Milestone 3 by adding GDELT news intensity features alongside price
and macro features.  The experiment runs four models on identical walk-forward
splits:

  1. **price-only**       : 7 price features (Milestone 1/2).
  2. **price+macro**      : price + 6 FRED macro regime features (Milestone 3).
  3. **price+news**       : price + 5 GDELT news intensity features.
  4. **price+macro+news** : price + macro + news (full feature set).

SPY is excluded from the news features (no sector keywords for the benchmark),
so news features are only available for the 11 sector ETFs.  The walk-forward
splits and date intersection logic ensures all four experiments are evaluated
on the same dates and the same fold boundaries.

Offline / sample mode
---------------------
Pass ``offline_sample=True`` (CLI: ``--offline-sample``) to substitute
deterministic synthetic news counts for live GDELT data.  This is useful
for iterating on the pipeline without waiting on the network or hitting
rate limits.  Reports generated in this mode are clearly labelled
**SAMPLE DATA -- not real GDELT output**.

Pipeline
--------
1. Load universe and price panel (may include SPY).
2. Load FRED series and compute macro features.
3. Load or generate GDELT counts and compute news features (sector ETFs only).
4. Build per-ticker price features and labels.
5. Merge macro (broadcast by date) and news (per-ticker) features.
6. Compute the intersection of valid dates across all four feature sets.
7. Run walk-forward splits once on the common date index.
8. For each fold, fit ridge on each of the four feature combinations.
9. Pool test folds, compute cross-sectional IC, report side-by-side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mosaic_alpha.data.fred import fetch_macro_panel
from mosaic_alpha.data.gdelt import (
    fetch_news_panel,
    generate_sample_news_panel,
    load_gdelt_config,
)
from mosaic_alpha.data.loader import load_panel, load_universe
from mosaic_alpha.features.macro_features import (
    MACRO_FEATURE_COLS,
    build_macro_features,
    load_macro_config,
)
from mosaic_alpha.features.news_features import NEWS_FEATURE_COLS, build_news_features
from mosaic_alpha.features.price_features import FEATURE_COLS, build_panel_features
from mosaic_alpha.research.cross_sectional import CrossSectionalResult, aggregate_cs_metrics
from mosaic_alpha.research.labels import LABEL_COLS, build_panel_labels
from mosaic_alpha.research.models import build_ridge_pipeline, fit_predict
from mosaic_alpha.research.validation import walk_forward_splits

logger = logging.getLogger(__name__)

# Sentinel values for data_mode field
DATA_MODE_LIVE = "live_gdelt"
DATA_MODE_SAMPLE = "offline_sample"


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class FourWayComparison:
    """Side-by-side pooled CS metrics across all four models for one label."""

    label_col: str
    price_only: CrossSectionalResult
    price_macro: CrossSectionalResult
    price_news: CrossSectionalResult
    price_macro_news: CrossSectionalResult


@dataclass
class NewsSectorResult:
    tickers: list[str]
    news_tickers: list[str]       # tickers with GDELT/sample coverage (no SPY)
    macro_series: list[str]
    start: str
    end: str
    n_panel_rows: int             # rows in the full-feature (price+macro+news) panel
    n_dates: int                  # common unique dates across all four experiments
    n_folds: int
    data_mode: str = DATA_MODE_LIVE   # DATA_MODE_LIVE or DATA_MODE_SAMPLE
    comparisons: list[FourWayComparison] = field(default_factory=list)
    # Pooled predictions from the price+macro+news model across all test folds.
    # Flat DataFrame: date, ticker, pmn_fwd_ret_1/5/20, fwd_ret_1/5/20.
    # Used by the backtest module.  Empty DataFrame if not populated.
    pooled_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_one_experiment(
    data: pd.DataFrame,
    feature_cols: list[str],
    unique_dates: pd.DatetimeIndex,
    splits: list,
    ridge_alpha: float,
    score_prefix: str,
) -> tuple[pd.DataFrame, list[CrossSectionalResult]]:
    """Run walk-forward ridge on *feature_cols*.

    Returns
    -------
    tuple of (pooled_df, cs_results):
    - pooled_df: flat DataFrame of all test-fold rows with score columns added.
    - cs_results: list of CrossSectionalResult, one per label in LABEL_COLS.
    """
    pooled_rows: list[pd.DataFrame] = []

    for split in splits:
        train_dates = unique_dates[split.train_idx]
        test_dates = unique_dates[split.test_idx]

        train_mask = data.index.get_level_values("date").isin(train_dates)
        test_mask = data.index.get_level_values("date").isin(test_dates)
        train_data = data[train_mask]
        test_data = data[test_mask]

        test_flat = test_data.reset_index()

        for label_col in LABEL_COLS:
            pipeline = build_ridge_pipeline(alpha=ridge_alpha)
            y_pred = fit_predict(
                pipeline,
                train_data[feature_cols],
                train_data[label_col],
                test_data[feature_cols],
            )
            test_flat[f"{score_prefix}_{label_col}"] = y_pred

        pooled_rows.append(test_flat)

    pooled_df = pd.concat(pooled_rows, ignore_index=True)
    cs_results: list[CrossSectionalResult] = []
    for label_col in LABEL_COLS:
        score_col = f"{score_prefix}_{label_col}"
        cs = aggregate_cs_metrics(pooled_df, score_col, label_col, date_col="date")
        cs_results.append(cs)

    return pooled_df, cs_results


# ── Main runner ────────────────────────────────────────────────────────────────

def run_news_sector_baseline(
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    universe_config: Path | None = None,
    macro_config: Path | None = None,
    gdelt_config: Path | None = None,
    ridge_alpha: float = 1.0,
    min_train_periods: int = 252,
    test_periods: int = 63,
    gdelt_sleep_secs: float = 10.0,
    offline_sample: bool = False,
    sectors: list[str] | None = None,
    force_refresh: bool = False,
) -> NewsSectorResult:
    """Run 4-way comparison on the sector ETF panel.

    Parameters
    ----------
    start, end:
        Date range (inclusive).  GDELT coverage is best from 2015 onward;
        default start is 2018-01-01 for cleaner news data.
    universe_config:
        Path to ``configs/universe.yaml``; uses default if None.
    macro_config:
        Path to ``configs/macro.yaml``; uses default if None.
    gdelt_config:
        Path to ``configs/gdelt.yaml``; uses default if None.
    ridge_alpha:
        Ridge regularisation parameter.
    min_train_periods, test_periods:
        Walk-forward parameters in trading days.
    gdelt_sleep_secs:
        Seconds to sleep between GDELT API requests.  Ignored in offline mode.
    offline_sample:
        If True, use :func:`~mosaic_alpha.data.gdelt.generate_sample_news_panel`
        instead of live GDELT calls.  Results will be labelled SAMPLE DATA.
    sectors:
        Optional list of sector tickers to include in the GDELT/sample fetch.
        When None, all tickers defined in ``gdelt_config`` are used.
    force_refresh:
        If True, bypass the Parquet cache and re-download from GDELT.
        Ignored in offline mode.
    """

    # ── 1. Universe and configs ───────────────────────────────────────────────
    tickers = load_universe(universe_config)
    macro_series = load_macro_config(macro_config)
    gdelt_sector_map = load_gdelt_config(gdelt_config)

    # Filter to requested sectors
    if sectors is not None:
        requested = set(sectors)
        gdelt_sector_map = {t: kws for t, kws in gdelt_sector_map.items() if t in requested}
        logger.info("Sector filter applied: %s", sorted(gdelt_sector_map))

    news_tickers = list(gdelt_sector_map.keys())
    data_mode = DATA_MODE_SAMPLE if offline_sample else DATA_MODE_LIVE

    logger.info(
        "Universe: %d tickers | Macro: %d series | News: %d tickers | mode: %s",
        len(tickers), len(macro_series), len(news_tickers), data_mode,
    )

    panel = load_panel(tickers, start, end)

    # ── 2. FRED macro features ────────────────────────────────────────────────
    raw_macro = fetch_macro_panel(macro_series, start, end)
    macro_feat = build_macro_features(raw_macro)

    # ── 3. News counts and features ───────────────────────────────────────────
    if offline_sample:
        logger.info("Offline sample mode: generating synthetic news counts")
        raw_news = generate_sample_news_panel(news_tickers, start, end)
    else:
        logger.info("Fetching GDELT news counts (live)")
        raw_news = fetch_news_panel(
            gdelt_sector_map, start, end,
            use_cache=not force_refresh,
            sleep_secs=gdelt_sleep_secs,
        )

    news_feat = build_news_features(raw_news)

    # ── 4. Price features and labels ──────────────────────────────────────────
    price_features = build_panel_features(panel)
    labels = build_panel_labels(panel)
    base = price_features.join(labels, how="inner")

    # ── 5. Build the four feature sets ───────────────────────────────────────
    # (a) price+macro: broadcast macro by date
    base_flat = base.reset_index(level="ticker")
    combined_macro_flat = base_flat.join(macro_feat, how="left")
    combined_macro = combined_macro_flat.set_index("ticker", append=True).sort_index()

    # (b) price+news: per-ticker join on (date, ticker)
    combined_news = base.join(news_feat, how="left")

    # (c) price+macro+news: broadcast macro then per-ticker join for news
    combined_all_flat = base_flat.join(macro_feat, how="left")
    combined_all = combined_all_flat.set_index("ticker", append=True).sort_index()
    combined_all = combined_all.join(news_feat, how="left")

    # Drop rows with any missing values in the required columns
    data_price = base.dropna(subset=FEATURE_COLS + LABEL_COLS)
    data_macro = combined_macro.dropna(subset=FEATURE_COLS + MACRO_FEATURE_COLS + LABEL_COLS)
    data_news = combined_news.dropna(subset=FEATURE_COLS + NEWS_FEATURE_COLS + LABEL_COLS)
    data_all = combined_all.dropna(
        subset=FEATURE_COLS + MACRO_FEATURE_COLS + NEWS_FEATURE_COLS + LABEL_COLS
    )

    logger.info(
        "Clean rows: price=%d | +macro=%d | +news=%d | +macro+news=%d",
        len(data_price), len(data_macro), len(data_news), len(data_all),
    )

    # ── 6. Common date intersection ───────────────────────────────────────────
    def _unique_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            df.index.get_level_values("date").unique().sort_values()
        )

    common_dates = (
        _unique_dates(data_price)
        .intersection(_unique_dates(data_macro))
        .intersection(_unique_dates(data_news))
        .intersection(_unique_dates(data_all))
    )

    splits = walk_forward_splits(
        common_dates,
        min_train_periods=min_train_periods,
        test_periods=test_periods,
    )
    logger.info(
        "Walk-forward: %d folds over %d common dates", len(splits), len(common_dates)
    )

    # Restrict all datasets to common dates
    def _filter_dates(df: pd.DataFrame) -> pd.DataFrame:
        return df[df.index.get_level_values("date").isin(common_dates)]

    data_price = _filter_dates(data_price)
    data_macro = _filter_dates(data_macro)
    data_news = _filter_dates(data_news)
    data_all = _filter_dates(data_all)

    # ── 7. Run all four experiments ───────────────────────────────────────────
    logger.info("Running price-only experiment...")
    _, po_cs = _run_one_experiment(
        data_price, FEATURE_COLS, common_dates, splits, ridge_alpha, "po"
    )

    logger.info("Running price+macro experiment...")
    _, pm_cs = _run_one_experiment(
        data_macro, FEATURE_COLS + MACRO_FEATURE_COLS, common_dates, splits, ridge_alpha, "pm"
    )

    logger.info("Running price+news experiment...")
    _, pn_cs = _run_one_experiment(
        data_news, FEATURE_COLS + NEWS_FEATURE_COLS, common_dates, splits, ridge_alpha, "pn"
    )

    logger.info("Running price+macro+news experiment...")
    pmn_full_df, pmn_cs = _run_one_experiment(
        data_all,
        FEATURE_COLS + MACRO_FEATURE_COLS + NEWS_FEATURE_COLS,
        common_dates, splits, ridge_alpha, "pmn",
    )

    # Slim the pmn pooled DataFrame to the columns needed for backtesting:
    # date, ticker, model scores, and realised labels.
    _score_cols = [f"pmn_{lc}" for lc in LABEL_COLS]
    _keep_cols = ["date", "ticker"] + _score_cols + LABEL_COLS
    pooled_predictions = pmn_full_df[[c for c in _keep_cols if c in pmn_full_df.columns]].copy()

    # ── 8. Log comparison ─────────────────────────────────────────────────────
    comparisons: list[FourWayComparison] = []
    for label_col, po, pm, pn, pmn in zip(LABEL_COLS, po_cs, pm_cs, pn_cs, pmn_cs):
        comparisons.append(
            FourWayComparison(
                label_col=label_col,
                price_only=po,
                price_macro=pm,
                price_news=pn,
                price_macro_news=pmn,
            )
        )
        logger.info(
            "%s | po IC=%.4f (t=%.2f) | pm IC=%.4f (t=%.2f) | "
            "pn IC=%.4f (t=%.2f) | pmn IC=%.4f (t=%.2f)",
            label_col,
            po.mean_ic, po.ic_t_stat,
            pm.mean_ic, pm.ic_t_stat,
            pn.mean_ic, pn.ic_t_stat,
            pmn.mean_ic, pmn.ic_t_stat,
        )

    return NewsSectorResult(
        tickers=tickers,
        news_tickers=news_tickers,
        macro_series=macro_series,
        start=start,
        end=end,
        n_panel_rows=len(data_all),
        n_dates=len(common_dates),
        n_folds=len(splits),
        data_mode=data_mode,
        comparisons=comparisons,
        pooled_predictions=pooled_predictions,
    )


# ── Report renderer ────────────────────────────────────────────────────────────

def render_report(result: NewsSectorResult, path: Path) -> None:
    """Write a Markdown 4-way comparison report to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)

    is_sample = result.data_mode == DATA_MODE_SAMPLE

    lines: list[str] = [
        "# News-Augmented Sector ETF Baseline",
        "",
    ]

    if is_sample:
        lines += [
            "> **WARNING: SAMPLE DATA -- not real GDELT output.**  ",
            "> News counts were generated synthetically with a fixed random seed.  ",
            "> Results involving news features have no real-world validity.  ",
            "> Re-run without `--offline-sample` to use live GDELT data.",
            "",
        ]

    lines += [
        f"**Data mode:** {result.data_mode}  ",
        f"**Universe:** {', '.join(result.tickers)}  ",
        f"**News tickers (GDELT):** {', '.join(result.news_tickers)}  ",
        f"**Macro series:** {', '.join(result.macro_series)}  ",
        f"**Period:** {result.start} to {result.end}  ",
        f"**Clean panel rows (price+macro+news):** {result.n_panel_rows:,}  ",
        f"**Unique dates:** {result.n_dates:,}  ",
        f"**Walk-forward folds:** {result.n_folds}  ",
        "",
        "## Feature sets",
        "",
        "### Price features",
        "",
        *(f"- `{c}`" for c in FEATURE_COLS),
        "",
        "### Macro features",
        "",
        *(f"- `{c}`" for c in MACRO_FEATURE_COLS),
        "",
        "### News features (GDELT)",
        "",
        *(f"- `{c}`" for c in NEWS_FEATURE_COLS),
        "",
        "> All macro and news features are lagged by one trading day before",
        "> merging with the price panel.",
        "",
        "## Pooled cross-sectional metrics: 4-way comparison",
        "",
        "Daily IC is computed across tickers for each test date; folds are pooled.",
        "",
        "| Label | Model | Mean IC | IC t-stat | Mean Rank IC | IC Hit Rate | Mean L/S Spread |",
        "|---|---|---|---|---|---|---|",
    ]

    model_keys = [
        ("price-only", "price_only"),
        ("price+macro", "price_macro"),
        ("price+news", "price_news"),
        ("price+macro+news", "price_macro_news"),
    ]

    for cmp in result.comparisons:
        for i, (model_name, attr) in enumerate(model_keys):
            cs: CrossSectionalResult = getattr(cmp, attr)
            label_cell = cmp.label_col if i == 0 else ""
            lines.append(
                f"| {label_cell} | {model_name} "
                f"| {cs.mean_ic:+.4f} | {cs.ic_t_stat:+.2f} "
                f"| {cs.mean_rank_ic:+.4f} | {cs.ic_hit_rate:.3f} "
                f"| {cs.mean_ls_spread:+.4f} |"
            )

    lines += [
        "",
        "## IC delta vs price-only",
        "",
        "| Label | +macro delta | +news delta | +macro+news delta |",
        "|---|---|---|---|",
    ]

    for cmp in result.comparisons:
        po = cmp.price_only
        lines.append(
            f"| {cmp.label_col} "
            f"| {cmp.price_macro.mean_ic - po.mean_ic:+.4f} "
            f"| {cmp.price_news.mean_ic - po.mean_ic:+.4f} "
            f"| {cmp.price_macro_news.mean_ic - po.mean_ic:+.4f} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- All models use Ridge regression (alpha=1.0) with standard scaling.",
        "- Walk-forward: expanding training window, 63-trading-day test folds.",
        "- Macro features are cross-sectionally constant within a date (broadcast).",
        "- News features are per-ticker (sector-specific GDELT article counts).",
        "- SPY excluded from news features: no sector keywords defined.",
        "- GDELT article counts are zero-filled for days with no coverage.",
        "- Cross-sectional IC computed daily across tickers (Pearson correlation).",
        "- These results do not constitute a trading strategy.",
        "- No transaction costs modelled.",
        "",
    ]

    if is_sample:
        lines += [
            "## Limitations (offline sample mode)",
            "",
            "- News counts are **synthetic** (deterministic random, seed=42).",
            "- Synthetic counts have no correlation with real sector news flow.",
            "- Any apparent IC contribution from news features is spurious.",
            "- Do not interpret price+news or price+macro+news results as real.",
            "- To get real results, run without `--offline-sample` and ensure",
            "  the GDELT cache is populated (use `--sleep-seconds 30` or higher).",
            "",
        ]

    lines += [
        "---",
        "_Generated by MosaicAlpha news sector baseline pipeline._",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", path)
