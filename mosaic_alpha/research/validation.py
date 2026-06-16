"""Walk-forward train / test split generation.

We use an *expanding-window* scheme:
- Training set grows from a minimum size each fold.
- Test set is a fixed-length window immediately after the training set.
- There is no gap between train and test (for a price-only daily model
  the label horizon is already baked into the label column).

Each split is a pair of integer-location index arrays (train_idx, test_idx)
over the rows of the combined feature + label DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    index: pd.DatetimeIndex,
    *,
    min_train_periods: int = 252,
    test_periods: int = 63,
) -> list[Split]:
    """Generate expanding-window walk-forward splits.

    Parameters
    ----------
    index:
        DatetimeIndex of the dataset (after dropping NaN rows).
    min_train_periods:
        Minimum number of rows in the first training fold (default ≈ 1 year).
    test_periods:
        Number of rows in each test fold (default ≈ 1 quarter).

    Returns
    -------
    List of :class:`Split` objects.
    """
    n = len(index)
    splits: list[Split] = []
    fold = 0
    test_start_loc = min_train_periods

    while test_start_loc + test_periods <= n:
        test_end_loc = test_start_loc + test_periods
        train_idx = np.arange(0, test_start_loc)
        test_idx = np.arange(test_start_loc, test_end_loc)

        splits.append(
            Split(
                fold=fold,
                train_start=index[0],
                train_end=index[test_start_loc - 1],
                test_start=index[test_start_loc],
                test_end=index[test_end_loc - 1],
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )
        test_start_loc += test_periods
        fold += 1

    return splits
