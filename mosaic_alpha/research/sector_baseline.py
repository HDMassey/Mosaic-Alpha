"""Cross-sectional sector ETF baseline orchestrator.

Pipeline
--------
1. Load universe tickers from config.
2. Download / cache panel OHLCV data (one parquet per ticker).
3. Compute per-ticker features (no cross-ticker leakage).
4. Compute per-ticker forward-return labels.
5. Merge into a clean flat panel; drop rows with any NaN.
6. Walk-forward splits on unique dates (expanding train, fixed test window).
7. For each fold:
   a. Fit a ridge model on training (date, ticker) rows.
   b. Predict scores for test rows.
   c. Evaluate cross-sectional IC per test date.
8. Aggregate metrics across folds and write a Markdown report.

Cross-sectional IC is computed per date across tickers, which is the natural
evaluation framework for a ranking model over a universe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic_alpha.data.loader import load_panel, load_universe
from mosaic_alpha.features.price_features import FEATURE_COLS, build_panel_features
from mosaic_alpha.research.cross_sectional import CrossSectionalResult, aggregate_cs_metrics
from mosaic_alpha.research.labels import LABEL_COLS, build_panel_labels
from mosaic_alpha.research.models import build_ridge_pipeline, fit_predict
from mosaic_alpha.research.validation import walk_forward_splits

logger = logging.getLogger(__name__)


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Metrics for one walk-forward fold."""

    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_rows: int
    n_test_rows: int
    # One CrossSectionalResult per label column
    cs_results: list[CrossSectionalResult] = field(default_factory=list)


@dataclass
class SectorBaselineResult:
    tickers: list[str]
    start: str
    end: str
    n_panel_rows: int
    n_dates: int
    n_folds: int
    fold_results: list[FoldResult] = field(default_factory=list)
    # Pooled cross-sectional metrics (all test folds concatenated)
    pooled: list[CrossSectionalResult] = field(default_factory=list)


# ── Main runner ────────────────────────────────────────────────────────────────

def run_sector_baseline(
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    universe_config: Path | None = None,
    ridge_alpha: float = 1.0,
    min_train_periods: int = 252,
    test_periods: int = 63,
) -> SectorBaselineResult:
    """Run the cross-sectional sector ETF baseline.

    Parameters
    ----------
    start, end:
        Date range for the panel.
    universe_config:
        Path to ``universe.yaml``; defaults to ``configs/universe.yaml``.
    ridge_alpha:
        Ridge regularisation parameter.
    min_train_periods, test_periods:
        Walk-forward parameters in trading days.
    """

    # ── 1. Universe ───────────────────────────────────────────────────────────
    tickers = load_universe(universe_config)
    logger.info("Universe: %d tickers — %s", len(tickers), tickers)

    # ── 2. Panel data ─────────────────────────────────────────────────────────
    panel = load_panel(tickers, start, end)

    # ── 3. Features (per ticker, no leakage) ──────────────────────────────────
    features = build_panel_features(panel)

    # ── 4. Labels (per ticker) ────────────────────────────────────────────────
    labels = build_panel_labels(panel)

    # ── 5. Merge and clean ────────────────────────────────────────────────────
    data = features.join(labels, how="inner").dropna()
    logger.info(
        "Clean panel: %d rows, %d unique dates, %d tickers",
        len(data),
        data.index.get_level_values("date").nunique(),
        data.index.get_level_values("ticker").nunique(),
    )

    # Unique sorted dates (used for walk-forward splitting)
    unique_dates: pd.DatetimeIndex = pd.DatetimeIndex(
        data.index.get_level_values("date").unique().sort_values()
    )

    # ── 6. Walk-forward splits ────────────────────────────────────────────────
    splits = walk_forward_splits(
        unique_dates,
        min_train_periods=min_train_periods,
        test_periods=test_periods,
    )
    logger.info("Walk-forward: %d folds over %d unique dates", len(splits), len(unique_dates))

    # ── 7. For each label × fold: fit ridge, compute CS-IC ───────────────────
    fold_results: list[FoldResult] = []
    # Accumulate test rows across folds for pooled metrics
    pooled_rows: list[pd.DataFrame] = []

    for split in splits:
        train_dates = unique_dates[split.train_idx]
        test_dates = unique_dates[split.test_idx]

        # Select panel rows by date level
        train_mask = data.index.get_level_values("date").isin(train_dates)
        test_mask = data.index.get_level_values("date").isin(test_dates)

        train_data = data[train_mask]
        test_data = data[test_mask]

        # Build a flat test DataFrame for CS-IC evaluation
        test_flat = test_data.reset_index()

        cs_results: list[CrossSectionalResult] = []

        for label_col in LABEL_COLS:
            X_train = train_data[FEATURE_COLS]
            y_train = train_data[label_col]
            X_test = test_data[FEATURE_COLS]

            pipeline = build_ridge_pipeline(alpha=ridge_alpha)
            y_pred = fit_predict(pipeline, X_train, y_train, X_test)

            # Attach predictions to the flat test frame
            score_col = f"score_{label_col}"
            test_flat[score_col] = y_pred

            cs = aggregate_cs_metrics(test_flat, score_col, label_col, date_col="date")
            cs_results.append(cs)

        fold_results.append(
            FoldResult(
                fold=split.fold,
                train_start=str(split.train_start.date()),
                train_end=str(split.train_end.date()),
                test_start=str(split.test_start.date()),
                test_end=str(split.test_end.date()),
                n_train_rows=len(train_data),
                n_test_rows=len(test_data),
                cs_results=cs_results,
            )
        )
        pooled_rows.append(test_flat)

        logger.info(
            "Fold %d | train=%d rows | test=%d rows | "
            "IC(fwd1)=%.3f IC(fwd5)=%.3f IC(fwd20)=%.3f",
            split.fold,
            len(train_data),
            len(test_data),
            cs_results[0].mean_ic,
            cs_results[1].mean_ic,
            cs_results[2].mean_ic,
        )

    # ── 8. Pooled cross-sectional metrics ─────────────────────────────────────
    pooled_df = pd.concat(pooled_rows, ignore_index=True)
    pooled: list[CrossSectionalResult] = []
    for label_col in LABEL_COLS:
        score_col = f"score_{label_col}"
        cs = aggregate_cs_metrics(pooled_df, score_col, label_col, date_col="date")
        pooled.append(cs)
        logger.info(
            "Pooled %s | IC=%.4f (t=%.2f) | RankIC=%.4f | Hit=%.3f | LS=%.4f",
            label_col,
            cs.mean_ic,
            cs.ic_t_stat,
            cs.mean_rank_ic,
            cs.ic_hit_rate,
            cs.mean_ls_spread,
        )

    return SectorBaselineResult(
        tickers=tickers,
        start=start,
        end=end,
        n_panel_rows=len(data),
        n_dates=len(unique_dates),
        n_folds=len(splits),
        fold_results=fold_results,
        pooled=pooled,
    )


# ── Report renderer ────────────────────────────────────────────────────────────

def render_report(result: SectorBaselineResult, path: Path) -> None:
    """Write a Markdown report of *result* to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Sector ETF Cross-Sectional Baseline",
        "",
        f"**Universe:** {', '.join(result.tickers)}  ",
        f"**Period:** {result.start} to {result.end}  ",
        f"**Clean panel rows:** {result.n_panel_rows:,}  ",
        f"**Unique dates:** {result.n_dates:,}  ",
        f"**Walk-forward folds:** {result.n_folds}  ",
        "",
        "## Features used",
        "",
        *(f"- `{c}`" for c in FEATURE_COLS),
        "",
        "## Pooled cross-sectional metrics (all test folds)",
        "",
        "Daily IC is computed across tickers for each trading day in the test",
        "period.  All test folds are pooled for the figures below.",
        "",
        "| Label | Mean IC | IC t-stat | Mean Rank IC | IC Hit Rate | Mean L/S Spread |",
        "|---|---|---|---|---|---|",
    ]

    for cs in result.pooled:
        lines.append(
            f"| {cs.label_col} "
            f"| {cs.mean_ic:+.4f} "
            f"| {cs.ic_t_stat:+.2f} "
            f"| {cs.mean_rank_ic:+.4f} "
            f"| {cs.ic_hit_rate:.3f} "
            f"| {cs.mean_ls_spread:+.4f} |"
        )

    lines += [
        "",
        "## Per-fold summary",
        "",
        "| Fold | Test Start | Test End | Train rows | Test rows | "
        "IC fwd1 | IC fwd5 | IC fwd20 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for fr in result.fold_results:
        ic1 = fr.cs_results[0].mean_ic if fr.cs_results else float("nan")
        ic5 = fr.cs_results[1].mean_ic if len(fr.cs_results) > 1 else float("nan")
        ic20 = fr.cs_results[2].mean_ic if len(fr.cs_results) > 2 else float("nan")
        lines.append(
            f"| {fr.fold} | {fr.test_start} | {fr.test_end} "
            f"| {fr.n_train_rows:,} | {fr.n_test_rows:,} "
            f"| {ic1:+.4f} | {ic5:+.4f} | {ic20:+.4f} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Model: Ridge regression (alpha=1.0) with standard scaling, trained on all (date, ticker)",
        "  panel rows in each training window.",
        "- Walk-forward: expanding training window, 63-trading-day test folds.",
        "- Cross-sectional IC is Pearson correlation between model scores and realised",
        "  forward returns, computed daily across tickers.",
        "- L/S spread uses top vs bottom tercile (4 tickers each out of 12).",
        "- No transaction costs modelled.  XLRE history begins 2015-10-08;",
        "  XLC history begins 2018-06-19.  Earlier dates use the remaining tickers.",
        "",
        "---",
        "_Generated by MosaicAlpha sector baseline pipeline._",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", path)
