"""Macro-augmented sector ETF cross-sectional baseline.

Extends the Milestone 2 sector baseline by adding macroeconomic regime
features from FRED alongside the existing price features.  The experiment
runs two models side-by-side on identical walk-forward splits:

  - **price-only**: same seven price features as Milestone 2.
  - **price + macro**: price features plus the six macro regime features.

This structure makes the marginal contribution of macro features directly
measurable: any difference in cross-sectional IC between the two models is
attributable solely to the macro features.

Pipeline
--------
1. Load universe and price panel.
2. Load FRED series and compute macro features.
3. Build per-ticker price features and labels.
4. Merge macro features into the panel by date (broadcast across tickers).
5. Align all series to a common clean date index, then run walk-forward
   splits once — used identically for both models.
6. For each fold, fit ridge on price-only features and price+macro features
   separately, then evaluate cross-sectional IC on the test window.
7. Pool all test folds and report side-by-side comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mosaic_alpha.data.fred import fetch_macro_panel
from mosaic_alpha.data.loader import load_panel, load_universe
from mosaic_alpha.features.macro_features import (
    MACRO_FEATURE_COLS,
    build_macro_features,
    load_macro_config,
)
from mosaic_alpha.features.price_features import FEATURE_COLS, build_panel_features
from mosaic_alpha.research.cross_sectional import CrossSectionalResult, aggregate_cs_metrics
from mosaic_alpha.research.labels import LABEL_COLS, build_panel_labels
from mosaic_alpha.research.models import build_ridge_pipeline, fit_predict
from mosaic_alpha.research.validation import walk_forward_splits

logger = logging.getLogger(__name__)


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ModelComparison:
    """Side-by-side pooled CS metrics for one label column."""

    label_col: str
    price_only: CrossSectionalResult
    price_macro: CrossSectionalResult


@dataclass
class MacroSectorResult:
    tickers: list[str]
    macro_series: list[str]
    start: str
    end: str
    n_panel_rows: int   # rows in final cleaned panel
    n_dates: int
    n_folds: int
    comparisons: list[ModelComparison] = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_one_experiment(
    data: pd.DataFrame,
    feature_cols: list[str],
    unique_dates: pd.DatetimeIndex,
    splits: list,
    ridge_alpha: float,
    score_prefix: str,
) -> tuple[list[pd.DataFrame], list[CrossSectionalResult]]:
    """Run walk-forward ridge on *feature_cols* and return pooled test rows + pooled CS results."""
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

    return pooled_rows, cs_results


# ── Main runner ────────────────────────────────────────────────────────────────

def run_macro_sector_baseline(
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    universe_config: Path | None = None,
    macro_config: Path | None = None,
    ridge_alpha: float = 1.0,
    min_train_periods: int = 252,
    test_periods: int = 63,
) -> MacroSectorResult:
    """Run price-only vs price+macro comparison on the sector ETF panel.

    Parameters
    ----------
    start, end:
        Date range (inclusive).
    universe_config:
        Path to ``configs/universe.yaml``; uses default if None.
    macro_config:
        Path to ``configs/macro.yaml``; uses default if None.
    ridge_alpha:
        Ridge regularisation parameter.
    min_train_periods, test_periods:
        Walk-forward parameters in trading days.
    """

    # ── 1. Universe and price panel ───────────────────────────────────────────
    tickers = load_universe(universe_config)
    macro_series = load_macro_config(macro_config)
    logger.info("Universe: %d tickers, %d macro series", len(tickers), len(macro_series))

    panel = load_panel(tickers, start, end)

    # ── 2. FRED macro series ──────────────────────────────────────────────────
    logger.info("Fetching macro series: %s", macro_series)
    raw_macro = fetch_macro_panel(macro_series, start, end)
    macro_feat = build_macro_features(raw_macro)

    # ── 3. Per-ticker price features and labels ───────────────────────────────
    price_features = build_panel_features(panel)
    labels = build_panel_labels(panel)

    # ── 4. Merge into a combined panel ────────────────────────────────────────
    # price_features and labels share a (date, ticker) MultiIndex.
    # macro_feat has a flat DatetimeIndex; broadcast across tickers by joining on date.
    base = price_features.join(labels, how="inner")

    # Join macro features by date level: reset ticker to column, join on date, restore index.
    # set_index("ticker", append=True) appends ticker as level 1, giving (date, ticker) order.
    base_flat = base.reset_index(level="ticker")
    combined_flat = base_flat.join(macro_feat, how="left")
    combined = combined_flat.set_index("ticker", append=True).sort_index()

    # Drop any row missing price features, labels, OR macro features
    data_price = base.dropna(subset=FEATURE_COLS + LABEL_COLS)
    data_combined = combined.dropna(subset=FEATURE_COLS + MACRO_FEATURE_COLS + LABEL_COLS)

    logger.info(
        "Price-only clean rows: %d | Price+macro clean rows: %d",
        len(data_price),
        len(data_combined),
    )

    # ── 5. Walk-forward splits on the shared date index ───────────────────────
    # Use the intersection of valid dates from both experiments so splits are identical.
    price_dates = pd.DatetimeIndex(
        data_price.index.get_level_values("date").unique().sort_values()
    )
    macro_dates = pd.DatetimeIndex(
        data_combined.index.get_level_values("date").unique().sort_values()
    )
    common_dates = price_dates.intersection(macro_dates)

    splits = walk_forward_splits(
        common_dates,
        min_train_periods=min_train_periods,
        test_periods=test_periods,
    )
    logger.info("Walk-forward: %d folds over %d common dates", len(splits), len(common_dates))

    # Restrict both datasets to common dates
    data_price = data_price[
        data_price.index.get_level_values("date").isin(common_dates)
    ]
    data_combined = data_combined[
        data_combined.index.get_level_values("date").isin(common_dates)
    ]

    # ── 6. Run both experiments ───────────────────────────────────────────────
    logger.info("Running price-only experiment...")
    _, price_cs = _run_one_experiment(
        data_price, FEATURE_COLS, common_dates, splits, ridge_alpha, score_prefix="po"
    )

    logger.info("Running price+macro experiment...")
    _, macro_cs = _run_one_experiment(
        data_combined, FEATURE_COLS + MACRO_FEATURE_COLS, common_dates, splits,
        ridge_alpha, score_prefix="pm"
    )

    # ── 7. Log comparison ─────────────────────────────────────────────────────
    comparisons: list[ModelComparison] = []
    for label_col, po, pm in zip(LABEL_COLS, price_cs, macro_cs):
        comparisons.append(ModelComparison(label_col=label_col, price_only=po, price_macro=pm))
        logger.info(
            "%s | Price-only IC=%.4f (t=%.2f) | Price+Macro IC=%.4f (t=%.2f)",
            label_col,
            po.mean_ic, po.ic_t_stat,
            pm.mean_ic, pm.ic_t_stat,
        )

    return MacroSectorResult(
        tickers=tickers,
        macro_series=macro_series,
        start=start,
        end=end,
        n_panel_rows=len(data_combined),
        n_dates=len(common_dates),
        n_folds=len(splits),
        comparisons=comparisons,
    )


# ── Report renderer ────────────────────────────────────────────────────────────

def render_report(result: MacroSectorResult, path: Path) -> None:
    """Write a Markdown comparison report to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Macro-Augmented Sector ETF Baseline",
        "",
        f"**Universe:** {', '.join(result.tickers)}  ",
        f"**Macro series:** {', '.join(result.macro_series)}  ",
        f"**Period:** {result.start} to {result.end}  ",
        f"**Clean panel rows (price+macro):** {result.n_panel_rows:,}  ",
        f"**Unique dates:** {result.n_dates:,}  ",
        f"**Walk-forward folds:** {result.n_folds}  ",
        "",
        "## Price features",
        "",
        *(f"- `{c}`" for c in FEATURE_COLS),
        "",
        "## Macro features",
        "",
        *(f"- `{c}`" for c in MACRO_FEATURE_COLS),
        "",
        "> All macro features are lagged by one trading day before merging with the",
        "> price panel, so features at date *t* use only data published on or before *t-1*.",
        "",
        "## Pooled cross-sectional metrics: price-only vs price+macro",
        "",
        "Daily IC is computed across tickers for each test date; folds are pooled.",
        "",
        "| Label | Model | Mean IC | IC t-stat | Mean Rank IC | IC Hit Rate | Mean L/S Spread |",
        "|---|---|---|---|---|---|---|",
    ]

    for cmp in result.comparisons:
        po = cmp.price_only
        pm = cmp.price_macro
        lines.append(
            f"| {cmp.label_col} | price-only "
            f"| {po.mean_ic:+.4f} | {po.ic_t_stat:+.2f} "
            f"| {po.mean_rank_ic:+.4f} | {po.ic_hit_rate:.3f} "
            f"| {po.mean_ls_spread:+.4f} |"
        )
        lines.append(
            f"| {cmp.label_col} | price+macro "
            f"| {pm.mean_ic:+.4f} | {pm.ic_t_stat:+.2f} "
            f"| {pm.mean_rank_ic:+.4f} | {pm.ic_hit_rate:.3f} "
            f"| {pm.mean_ls_spread:+.4f} |"
        )

    lines += [
        "",
        "## IC delta (price+macro minus price-only)",
        "",
        "| Label | Delta IC | Delta IC t-stat | Delta Rank IC |",
        "|---|---|---|---|",
    ]

    for cmp in result.comparisons:
        po = cmp.price_only
        pm = cmp.price_macro
        lines.append(
            f"| {cmp.label_col} "
            f"| {pm.mean_ic - po.mean_ic:+.4f} "
            f"| {pm.ic_t_stat - po.ic_t_stat:+.2f} "
            f"| {pm.mean_rank_ic - po.mean_rank_ic:+.4f} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Both models use Ridge regression (alpha=1.0) with standard scaling.",
        "- Walk-forward: expanding training window, 63-trading-day test folds.",
        "- Macro features broadcast to all tickers on each date (macro regime is",
        "  cross-sectionally constant within a day).",
        "- Cross-sectional IC is Pearson correlation between model score and realised",
        "  forward returns, computed daily across tickers.",
        "- These results do not constitute a trading strategy. They measure whether",
        "  macro regime signals help rank sector ETFs relative to one another.",
        "- No transaction costs modelled.",
        "",
        "---",
        "_Generated by MosaicAlpha macro sector baseline pipeline._",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", path)
