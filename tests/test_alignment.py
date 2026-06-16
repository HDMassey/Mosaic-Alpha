"""Tests for timestamp alignment between features and labels.

Ensures that features[t] and labels[t] refer to the same calendar date and
that the dataset index is monotonically increasing with no duplicate dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic_alpha.features.price_features import build_features
from mosaic_alpha.research.labels import build_labels


def _make_prices(n: int = 100) -> pd.DataFrame:
    """Create a synthetic OHLCV price DataFrame."""
    dates = pd.bdate_range("2020-01-01", periods=n, freq="B")
    np.random.seed(42)
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    volume = np.random.randint(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def test_feature_index_matches_price_index():
    prices = _make_prices()
    features = build_features(prices)
    assert features.index.equals(prices.index), (
        "Feature index does not match price index."
    )


def test_label_index_matches_price_index():
    prices = _make_prices()
    labels = build_labels(prices)
    assert labels.index.equals(prices.index), (
        "Label index does not match price index."
    )


def test_feature_and_label_indices_match():
    prices = _make_prices()
    features = build_features(prices)
    labels = build_labels(prices)
    assert features.index.equals(labels.index), (
        "Feature index and label index do not align."
    )


def test_index_is_monotonically_increasing():
    prices = _make_prices()
    features = build_features(prices)
    assert features.index.is_monotonic_increasing, (
        "Feature index is not monotonically increasing."
    )


def test_no_duplicate_dates():
    prices = _make_prices()
    features = build_features(prices)
    assert not features.index.duplicated().any(), (
        "Feature index contains duplicate dates."
    )


def test_joined_dataset_no_index_gaps():
    """After joining features + labels and dropping NaN, index must remain
    a subset of the original price index with no introduced dates."""
    prices = _make_prices(120)
    features = build_features(prices)
    labels = build_labels(prices)
    data = features.join(labels).dropna()
    assert data.index.isin(prices.index).all(), (
        "Joined dataset index contains dates not in the original price index."
    )


def test_label_horizon_offset():
    """label[t] for horizon h should equal log(close[t+h] / close[t])."""
    prices = _make_prices(50)
    labels = build_labels(prices, horizons=(1, 5))
    close = prices["close"]

    for h in (1, 5):
        col = f"fwd_ret_{h}"
        for i in range(len(prices) - h):
            date = prices.index[i]
            expected = np.log(close.iloc[i + h] / close.iloc[i])
            actual = labels.loc[date, col]
            assert abs(actual - expected) < 1e-10, (
                f"Label mismatch at row {i} for horizon {h}: "
                f"expected {expected}, got {actual}"
            )
