"""Evaluation metrics for return-prediction models.

Metrics
-------
ic          : Pearson correlation between predictions and realised returns.
rank_ic     : Spearman rank correlation (more robust to outliers).
hit_rate    : Fraction of predictions where sign(pred) == sign(actual).
mse         : Mean squared error.
ls_decile   : Average return of top-decile predictions minus bottom-decile.

All metrics accept 1-D array-like inputs and return a float (or NaN when
there is insufficient data).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PeriodMetrics:
    """Metrics for a single walk-forward test period."""

    fold: int
    test_start: str
    test_end: str
    label: str  # e.g. "fwd_ret_1"
    n_obs: int
    ic: float
    rank_ic: float
    hit_rate: float
    mse: float
    ls_decile_return: float


@dataclass
class AggregateMetrics:
    """Mean ± std of per-fold metrics across all walk-forward folds."""

    label: str
    n_folds: int
    mean_ic: float
    std_ic: float
    mean_rank_ic: float
    std_rank_ic: float
    mean_hit_rate: float
    std_hit_rate: float
    mean_mse: float
    std_mse: float
    mean_ls_decile: float
    std_ls_decile: float
    ic_t_stat: float  # t-stat: mean_ic / (std_ic / sqrt(n_folds))
    period_metrics: list[PeriodMetrics] = field(default_factory=list)


def ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson IC."""
    if len(y_true) < 2:
        return float("nan")
    r, _ = stats.pearsonr(y_pred, y_true)
    return float(r)


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank IC."""
    if len(y_true) < 2:
        return float("nan")
    r, _ = stats.spearmanr(y_pred, y_true)
    return float(r)


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with correct sign."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def ls_decile_return(y_true: np.ndarray, y_pred: np.ndarray, n_deciles: int = 10) -> float:
    """Long-top-decile / short-bottom-decile average return.

    Returns the mean realised return of the top-decile predictions minus
    the mean realised return of the bottom-decile predictions.  This is a
    simple measure of whether the model ranks observations correctly.
    """
    if len(y_pred) < n_deciles * 2:
        return float("nan")
    pred_series = pd.Series(y_pred)
    true_series = pd.Series(y_true)
    decile = pd.qcut(pred_series, n_deciles, labels=False, duplicates="drop")
    top = true_series[decile == decile.max()].mean()
    bot = true_series[decile == decile.min()].mean()
    return float(top - bot)


def compute_period_metrics(
    fold: int,
    test_start: str,
    test_end: str,
    label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> PeriodMetrics:
    return PeriodMetrics(
        fold=fold,
        test_start=test_start,
        test_end=test_end,
        label=label,
        n_obs=len(y_true),
        ic=ic(y_true, y_pred),
        rank_ic=rank_ic(y_true, y_pred),
        hit_rate=hit_rate(y_true, y_pred),
        mse=mse(y_true, y_pred),
        ls_decile_return=ls_decile_return(y_true, y_pred),
    )


def aggregate_metrics(label: str, periods: list[PeriodMetrics]) -> AggregateMetrics:
    """Average per-fold metrics and compute IC t-statistic."""
    arr = lambda attr: np.array([getattr(p, attr) for p in periods], dtype=float)

    ics = arr("ic")
    rank_ics = arr("rank_ic")
    hits = arr("hit_rate")
    mses = arr("mse")
    ls = arr("ls_decile_return")

    n = len(periods)
    mean_ic = float(np.nanmean(ics))
    std_ic = float(np.nanstd(ics, ddof=1)) if n > 1 else float("nan")
    ic_t = mean_ic / (std_ic / np.sqrt(n)) if n > 1 and std_ic > 0 else float("nan")

    return AggregateMetrics(
        label=label,
        n_folds=n,
        mean_ic=mean_ic,
        std_ic=std_ic,
        mean_rank_ic=float(np.nanmean(rank_ics)),
        std_rank_ic=float(np.nanstd(rank_ics, ddof=1)) if n > 1 else float("nan"),
        mean_hit_rate=float(np.nanmean(hits)),
        std_hit_rate=float(np.nanstd(hits, ddof=1)) if n > 1 else float("nan"),
        mean_mse=float(np.nanmean(mses)),
        std_mse=float(np.nanstd(mses, ddof=1)) if n > 1 else float("nan"),
        mean_ls_decile=float(np.nanmean(ls)),
        std_ls_decile=float(np.nanstd(ls, ddof=1)) if n > 1 else float("nan"),
        ic_t_stat=ic_t,
        period_metrics=periods,
    )
