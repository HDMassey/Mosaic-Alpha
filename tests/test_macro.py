"""Tests for Milestone 3: FRED connector, macro features, and macro-sector pipeline.

Categories
----------
1. FRED parser handles dates and missing values correctly.
2. Macro features are lagged before being available for model training.
3. Macro merge does not create future leakage in the combined panel.
4. Price-only vs price+macro comparison returns the expected schema.
5. Missing FRED_API_KEY produces a clear error (not a silent failure).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mosaic_alpha.data.fred import FredApiKeyError, fetch_series
from mosaic_alpha.features.macro_features import (
    MACRO_FEATURE_COLS,
    build_macro_features,
    load_macro_config,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_fred_response(dates: list[str], values: list[str]) -> MagicMock:
    """Build a mock requests.Response that looks like a FRED JSON reply."""
    payload = {
        "observations": [
            {"date": d, "value": v} for d, v in zip(dates, values)
        ]
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = payload
    return mock_resp


def _make_raw_macro(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic raw macro DataFrame matching the expected FRED columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "DGS10": 2.5 + rng.normal(0, 0.3, n).cumsum() * 0.01,
            "DGS2": 2.0 + rng.normal(0, 0.3, n).cumsum() * 0.01,
            "FEDFUNDS": 1.5 + rng.normal(0, 0.1, n).cumsum() * 0.005,
            "CPIAUCSL": 260 + np.arange(n) * 0.05 + rng.normal(0, 0.2, n),
            "UNRATE": 4.0 + rng.normal(0, 0.2, n).cumsum() * 0.002,
            "BAA10Y": 2.0 + rng.normal(0, 0.2, n).cumsum() * 0.005,
        },
        index=dates,
    )


# ── 1. FRED parser correctness ─────────────────────────────────────────────────

@patch("mosaic_alpha.data.fred.requests.get")
def test_fetch_series_parses_dates(mock_get, tmp_path: Path):
    """Returned series must have a DatetimeIndex with the expected dates."""
    mock_get.return_value = _make_fred_response(
        ["2023-01-03", "2023-01-04", "2023-01-05"],
        ["3.88", "3.90", "3.91"],
    )
    with patch.dict("os.environ", {"FRED_API_KEY": "test_key"}):
        s = fetch_series("DGS10", "2023-01-03", "2023-01-05", use_cache=False)

    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.tz is None
    assert pd.Timestamp("2023-01-03") in s.index
    assert pd.Timestamp("2023-01-05") in s.index


@patch("mosaic_alpha.data.fred.requests.get")
def test_fetch_series_dot_becomes_nan(mock_get, tmp_path: Path):
    """FRED '.' missing-value sentinel must be converted to NaN."""
    mock_get.return_value = _make_fred_response(
        ["2023-01-03", "2023-01-04"],
        [".", "3.90"],
    )
    with patch.dict("os.environ", {"FRED_API_KEY": "test_key"}):
        s = fetch_series("DGS10", "2023-01-03", "2023-01-04", use_cache=False)

    assert np.isnan(s.loc[pd.Timestamp("2023-01-03")])
    assert s.loc[pd.Timestamp("2023-01-04")] == pytest.approx(3.90)


@patch("mosaic_alpha.data.fred.requests.get")
def test_fetch_series_forward_fills_to_daily(mock_get):
    """Weekly or monthly FRED data must be reindexed to daily with forward-fill."""
    # Simulate monthly data: just two observations
    mock_get.return_value = _make_fred_response(
        ["2023-01-01", "2023-02-01"],
        ["260.0", "261.0"],
    )
    with patch.dict("os.environ", {"FRED_API_KEY": "test_key"}):
        s = fetch_series("CPIAUCSL", "2023-01-01", "2023-02-28", use_cache=False)

    # Daily index expected
    assert len(s) > 2
    # Values from Jan 1 forward-filled until Feb 1
    assert s.loc["2023-01-15"] == pytest.approx(260.0)
    # Values from Feb 1 forward-filled until end
    assert s.loc["2023-02-15"] == pytest.approx(261.0)


@patch("mosaic_alpha.data.fred.requests.get")
def test_fetch_series_series_name(mock_get):
    """Returned Series must be named after the series_id."""
    mock_get.return_value = _make_fred_response(["2023-01-03"], ["3.88"])
    with patch.dict("os.environ", {"FRED_API_KEY": "test_key"}):
        s = fetch_series("BAA10Y", "2023-01-03", "2023-01-03", use_cache=False)
    assert s.name == "BAA10Y"


# ── 2. Macro features are lagged ──────────────────────────────────────────────

def test_macro_features_are_lagged_by_one_day():
    """The shift(1) in build_macro_features must shift all values forward one day."""
    raw = _make_raw_macro(300)

    # Build once with the standard function
    feat = build_macro_features(raw)

    # Manually compute yield_curve without shift
    unlagged_yc = raw["DGS10"] - raw["DGS2"]

    # Feature at date[5] must equal the unlagged value at date[4]
    date_t = raw.index[5]
    date_t_minus_1 = raw.index[4]

    assert feat["yield_curve_10y_2y"].loc[date_t] == pytest.approx(
        unlagged_yc.loc[date_t_minus_1], rel=1e-9
    )


def test_macro_features_first_row_is_nan():
    """After a shift(1), the very first row of every feature column must be NaN."""
    raw = _make_raw_macro(100)
    feat = build_macro_features(raw)

    # All columns at index[0] should be NaN because shift(1) moves data forward
    for col in ["yield_curve_10y_2y", "credit_spread_level", "fed_funds_change_3m"]:
        assert np.isnan(feat[col].iloc[0]), (
            f"Expected NaN at row 0 for {col} after lag, got {feat[col].iloc[0]}"
        )


def test_macro_features_returns_all_expected_columns():
    """build_macro_features must return exactly MACRO_FEATURE_COLS columns."""
    raw = _make_raw_macro(300)
    feat = build_macro_features(raw)
    assert list(feat.columns) == MACRO_FEATURE_COLS


def test_macro_feature_index_matches_raw_index():
    """Feature index must be identical to the raw macro DataFrame index."""
    raw = _make_raw_macro(200)
    feat = build_macro_features(raw)
    assert feat.index.equals(raw.index)


# ── 3. Macro merge does not leak future data ──────────────────────────────────

def _make_price_panel(
    tickers: list[str] = ("SPY", "XLK"),
    n: int = 300,
    seed: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n, freq="B")
    pieces = []
    for ticker in tickers:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        df = pd.DataFrame(
            {"close": close, "volume": rng.integers(1_000_000, 5_000_000, n).astype(float)},
            index=pd.MultiIndex.from_arrays(
                [dates, [ticker] * n], names=["date", "ticker"]
            ),
        )
        pieces.append(df)
    return pd.concat(pieces).sort_index()


def test_macro_merge_by_date_broadcasts_identically():
    """Macro features for a given date must be the same for all tickers."""
    from mosaic_alpha.features.price_features import build_panel_features
    from mosaic_alpha.research.labels import build_panel_labels

    panel = _make_price_panel()
    raw_macro = _make_raw_macro(n=600)
    macro_feat = build_macro_features(raw_macro)

    price_features = build_panel_features(panel)
    labels = build_panel_labels(panel)
    base = price_features.join(labels, how="inner")

    # Simulate the merge from sector_baseline
    base_flat = base.reset_index(level="ticker")
    combined_flat = base_flat.join(macro_feat, how="left")

    # For any given date, all tickers should have the same macro feature values
    for date, group in combined_flat.groupby("date"):
        for col in MACRO_FEATURE_COLS:
            unique_vals = group[col].dropna().unique()
            assert len(unique_vals) <= 1, (
                f"{col} has different values for different tickers on {date}: {unique_vals}"
            )


def test_macro_merge_no_future_values_in_feature():
    """Macro feature at date t must not include raw series values from date t+1 or later."""
    raw = _make_raw_macro(300)

    # Tamper with raw value on day 100
    day_100 = raw.index[100]
    day_99 = raw.index[99]

    raw_orig = raw.copy()
    raw_perturbed = raw.copy()
    raw_perturbed.loc[day_100, "DGS10"] += 999.0  # large shock on day 100

    feat_orig = build_macro_features(raw_orig)
    feat_pert = build_macro_features(raw_perturbed)

    # Feature at day 99 must be unaffected (perturbed day is 100)
    assert feat_orig["yield_curve_10y_2y"].loc[day_99] == pytest.approx(
        feat_pert["yield_curve_10y_2y"].loc[day_99]
    ), "Day 100 perturbation leaked into day 99 macro feature"

    # Feature at day 100 uses day 99's raw values, so it should also be unaffected
    assert feat_orig["yield_curve_10y_2y"].loc[day_100] == pytest.approx(
        feat_pert["yield_curve_10y_2y"].loc[day_100]
    ), "Day 100 perturbation affected the day 100 feature (should use day 99 data)"

    # Feature at day 101 WILL differ — it uses day 100's perturbed data
    day_101 = raw.index[101]
    assert feat_orig["yield_curve_10y_2y"].loc[day_101] != pytest.approx(
        feat_pert["yield_curve_10y_2y"].loc[day_101]
    ), "Expected day 101 feature to reflect day 100 shock"


# ── 4. Comparison result schema ───────────────────────────────────────────────

def test_macro_config_loads_six_series():
    """The default configs/macro.yaml must contain exactly 6 FRED series."""
    series = load_macro_config()
    assert len(series) == 6
    assert "DGS10" in series
    assert "CPIAUCSL" in series
    assert "BAA10Y" in series


def test_macro_config_from_tmp(tmp_path: Path):
    """load_macro_config correctly reads a custom YAML path."""
    content = (
        "macro:\n"
        "  series:\n"
        "    AAA:\n"
        "      description: Test series A\n"
        "    BBB:\n"
        "      description: Test series B\n"
    )
    p = tmp_path / "macro.yaml"
    p.write_text(content, encoding="utf-8")
    series = load_macro_config(p)
    assert series == ["AAA", "BBB"]


def test_model_comparison_result_has_expected_attributes():
    """MacroSectorResult and ModelComparison must expose the fields needed for the report."""
    from mosaic_alpha.research.macro_sector_baseline import MacroSectorResult, ModelComparison
    from mosaic_alpha.research.cross_sectional import CrossSectionalResult

    def _dummy_cs(label: str) -> CrossSectionalResult:
        return CrossSectionalResult(
            score_col="score",
            label_col=label,
            n_dates=100,
            mean_ic=0.05,
            std_ic=0.1,
            ic_t_stat=1.5,
            mean_rank_ic=0.04,
            std_rank_ic=0.09,
            ic_hit_rate=0.55,
            mean_ls_spread=0.001,
            std_ls_spread=0.002,
        )

    cmp = ModelComparison(
        label_col="fwd_ret_1",
        price_only=_dummy_cs("fwd_ret_1"),
        price_macro=_dummy_cs("fwd_ret_1"),
    )
    result = MacroSectorResult(
        tickers=["SPY", "XLK"],
        macro_series=["DGS10", "BAA10Y"],
        start="2015-01-01",
        end="2024-12-31",
        n_panel_rows=5000,
        n_dates=500,
        n_folds=10,
        comparisons=[cmp],
    )

    assert hasattr(result, "comparisons")
    assert len(result.comparisons) == 1
    assert result.comparisons[0].label_col == "fwd_ret_1"
    assert result.comparisons[0].price_only.mean_ic == pytest.approx(0.05)
    assert result.comparisons[0].price_macro.mean_ic == pytest.approx(0.05)


# ── 5. Missing API key produces a clear error ─────────────────────────────────

def test_missing_api_key_raises_fred_api_key_error():
    """fetch_series must raise FredApiKeyError when FRED_API_KEY is not set."""
    with patch.dict("os.environ", {}, clear=True):
        # Remove key if present
        import os
        os.environ.pop("FRED_API_KEY", None)
        with pytest.raises(FredApiKeyError, match="FRED_API_KEY"):
            fetch_series("DGS10", "2023-01-01", "2023-01-31", use_cache=False)


def test_empty_api_key_raises_fred_api_key_error():
    """An empty-string FRED_API_KEY must also raise FredApiKeyError."""
    with patch.dict("os.environ", {"FRED_API_KEY": "   "}):
        with pytest.raises(FredApiKeyError, match="FRED_API_KEY"):
            fetch_series("DGS10", "2023-01-01", "2023-01-31", use_cache=False)
