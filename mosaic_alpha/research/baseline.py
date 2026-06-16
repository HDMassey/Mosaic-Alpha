"""Baseline experiment orchestrator.

Ties together:
  data loading → feature engineering → label construction →
  walk-forward splits → ridge regression → metric aggregation → report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic_alpha.data.loader import load_prices
from mosaic_alpha.features.price_features import FEATURE_COLS, build_features
from mosaic_alpha.research.labels import LABEL_COLS, build_labels
from mosaic_alpha.research.metrics import (
    AggregateMetrics,
    aggregate_metrics,
    compute_period_metrics,
)
from mosaic_alpha.research.models import build_ridge_pipeline, fit_predict
from mosaic_alpha.research.validation import walk_forward_splits

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    ticker: str
    start: str
    end: str
    n_rows: int
    n_folds: int
    aggregate: list[AggregateMetrics] = field(default_factory=list)


def run_baseline(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    ridge_alpha: float = 1.0,
    min_train_periods: int = 252,
    test_periods: int = 63,
) -> BaselineResult:
    """Run the full price-only baseline pipeline and return structured results."""

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading prices for %s (%s to %s)", ticker, start, end)
    prices = load_prices(ticker, start, end)

    # ── 2. Features ───────────────────────────────────────────────────────────
    features = build_features(prices)

    # ── 3. Labels ─────────────────────────────────────────────────────────────
    labels = build_labels(prices)

    # ── 4. Merge & drop NaN rows ──────────────────────────────────────────────
    data = features.join(labels, how="inner")
    # Drop any row where a feature or label is NaN (warm-up + trailing label NaNs)
    data = data.dropna()

    feature_df = data[FEATURE_COLS]
    n_rows = len(data)
    logger.info("Dataset: %d clean rows, %d features", n_rows, len(FEATURE_COLS))

    # ── 5. Walk-forward splits ────────────────────────────────────────────────
    splits = walk_forward_splits(
        data.index,
        min_train_periods=min_train_periods,
        test_periods=test_periods,
    )
    logger.info("Walk-forward: %d folds", len(splits))

    # ── 6. For each label, train ridge and collect metrics ────────────────────
    all_agg: list[AggregateMetrics] = []

    for label_col in LABEL_COLS:
        label_series = data[label_col]
        period_results = []

        for split in splits:
            X_train = feature_df.iloc[split.train_idx]
            y_train = label_series.iloc[split.train_idx]
            X_test = feature_df.iloc[split.test_idx]
            y_test = label_series.iloc[split.test_idx]

            pipeline = build_ridge_pipeline(alpha=ridge_alpha)
            y_pred = fit_predict(pipeline, X_train, y_train, X_test)

            pm = compute_period_metrics(
                fold=split.fold,
                test_start=str(split.test_start.date()),
                test_end=str(split.test_end.date()),
                label=label_col,
                y_true=y_test.values,
                y_pred=y_pred,
            )
            period_results.append(pm)

        agg = aggregate_metrics(label_col, period_results)
        all_agg.append(agg)
        logger.info(
            "%s | IC=%.4f (t=%.2f) | RankIC=%.4f | Hit=%.3f | LS=%.4f",
            label_col,
            agg.mean_ic,
            agg.ic_t_stat,
            agg.mean_rank_ic,
            agg.mean_hit_rate,
            agg.mean_ls_decile,
        )

    return BaselineResult(
        ticker=ticker,
        start=start,
        end=end,
        n_rows=n_rows,
        n_folds=len(splits),
        aggregate=all_agg,
    )


def render_report(result: BaselineResult, path: Path) -> None:
    """Write a Markdown report of *result* to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# Baseline Report — {result.ticker}",
        "",
        f"**Ticker:** {result.ticker}  ",
        f"**Period:** {result.start} to {result.end}  ",
        f"**Clean rows:** {result.n_rows:,}  ",
        f"**Walk-forward folds:** {result.n_folds}  ",
        "",
        "## Features used",
        "",
        *(f"- `{c}`" for c in FEATURE_COLS),
        "",
        "## Aggregate metrics (mean ± std across folds)",
        "",
        "| Label | IC | IC t-stat | Rank IC | Hit Rate | MSE | L/S Decile |",
        "|---|---|---|---|---|---|---|",
    ]

    for agg in result.aggregate:
        lines.append(
            f"| {agg.label} "
            f"| {agg.mean_ic:+.4f} ± {agg.std_ic:.4f} "
            f"| {agg.ic_t_stat:+.2f} "
            f"| {agg.mean_rank_ic:+.4f} ± {agg.std_rank_ic:.4f} "
            f"| {agg.mean_hit_rate:.3f} ± {agg.std_hit_rate:.3f} "
            f"| {agg.mean_mse:.6f} ± {agg.std_mse:.6f} "
            f"| {agg.mean_ls_decile:+.4f} ± {agg.std_ls_decile:.4f} |"
        )

    lines += [
        "",
        "## Per-fold detail",
        "",
    ]

    for agg in result.aggregate:
        lines += [
            f"### {agg.label}",
            "",
            "| Fold | Test Start | Test End | n | IC | Rank IC | Hit Rate | LS Decile |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for pm in agg.period_metrics:
            lines.append(
                f"| {pm.fold} | {pm.test_start} | {pm.test_end} | {pm.n_obs} "
                f"| {pm.ic:+.4f} | {pm.rank_ic:+.4f} "
                f"| {pm.hit_rate:.3f} | {pm.ls_decile_return:+.4f} |"
            )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Model: Ridge regression (α=1.0) with standard scaling.",
        "- Walk-forward: expanding training window, 63-day test folds.",
        "- IC and Rank IC measure predictive power; L/S Decile measures economic value.",
        "- No transaction costs modelled.",
        "",
        "---",
        "_Generated by MosaicAlpha baseline pipeline._",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", path)
