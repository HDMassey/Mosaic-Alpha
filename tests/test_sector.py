"""Tests for the cross-sectional sector ETF baseline.

Covers:
1. Per-ticker feature alignment — features align to source prices per ticker.
2. No leakage across tickers — altering ticker A's prices does not affect ticker B.
3. Forward labels computed within ticker only — label[t, A] uses only ticker A's close.
4. Cross-sectional IC calculation correctness — verified against manual computation.
5. Universe config loading — YAML parses to the expected ticker list.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic_alpha.data.loader import load_universe
from mosaic_alpha.features.price_features import build_panel_features
from mosaic_alpha.research.cross_sectional import (
    aggregate_cs_metrics,
    daily_cs_ic,
    daily_cs_rank_ic,
)
from mosaic_alpha.research.labels import build_panel_labels


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_panel(
    tickers: list[str] = ("AAA", "BBB", "CCC"),
    n: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic panel with a (date, ticker) MultiIndex."""
    dates = pd.bdate_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []

    for i, ticker in enumerate(tickers):
        close = 100.0 * np.exp(
            np.cumsum(rng.normal(0, 0.01 + i * 0.002, n))
        )
        volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
        df = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "adj_close": close,
                "volume": volume,
            },
            index=pd.MultiIndex.from_arrays(
                [dates, [ticker] * n],
                names=["date", "ticker"],
            ),
        )
        pieces.append(df)

    return pd.concat(pieces).sort_index()


# ── 1. Per-ticker feature alignment ──────────────────────────────────────────

def test_panel_features_index_matches_panel():
    """Feature index must cover exactly the same (date, ticker) pairs as the panel."""
    panel = _make_panel()
    features = build_panel_features(panel)
    assert features.index.equals(panel.index), (
        "Panel feature index does not match panel index."
    )


def test_panel_features_date_level_matches_per_ticker():
    """For each ticker, the date-level of features must match the source panel."""
    panel = _make_panel(["X", "Y"])
    features = build_panel_features(panel)

    for ticker in ["X", "Y"]:
        panel_dates = panel.xs(ticker, level="ticker").index
        feat_dates = features.xs(ticker, level="ticker").index
        assert feat_dates.equals(panel_dates), (
            f"Feature dates for {ticker} do not match panel dates."
        )


def test_panel_labels_index_matches_panel():
    """Label index must cover exactly the same (date, ticker) pairs as the panel."""
    panel = _make_panel()
    labels = build_panel_labels(panel)
    assert labels.index.equals(panel.index)


# ── 2. No leakage across tickers ─────────────────────────────────────────────

def test_altering_one_ticker_does_not_change_another():
    """Changing prices for ticker AAA must not affect features for ticker BBB."""
    panel = _make_panel(["AAA", "BBB"])
    features_orig = build_panel_features(panel)

    # Double the close for AAA
    panel_perturbed = panel.copy()
    aaa_mask = panel_perturbed.index.get_level_values("ticker") == "AAA"
    panel_perturbed.loc[aaa_mask, "close"] *= 2.0
    features_perturbed = build_panel_features(panel_perturbed)

    # BBB features must be identical
    bbb_orig = features_orig.xs("BBB", level="ticker")
    bbb_pert = features_perturbed.xs("BBB", level="ticker")
    pd.testing.assert_frame_equal(bbb_orig, bbb_pert, obj="BBB features after perturbing AAA")


def test_rolling_windows_do_not_cross_ticker_boundary():
    """The log_return for ticker BBB on day 1 must not involve ticker AAA's prices."""
    panel = _make_panel(["AAA", "BBB"])
    features = build_panel_features(panel)

    # log_return for BBB row 1 = log(BBB.close[1] / BBB.close[0])
    bbb_close = panel.xs("BBB", level="ticker")["close"]
    expected = float(np.log(bbb_close.iloc[1] / bbb_close.iloc[0]))

    bbb_feat = features.xs("BBB", level="ticker")
    actual = float(bbb_feat["log_return"].iloc[1])
    assert abs(actual - expected) < 1e-12, (
        f"BBB log_return[1] = {actual}, expected {expected}"
    )


# ── 3. Forward labels computed within ticker only ─────────────────────────────

def test_label_uses_same_ticker_close():
    """fwd_ret_1[t, ticker] = log(close[t+1, ticker] / close[t, ticker])."""
    panel = _make_panel(["AAA", "BBB"])
    labels = build_panel_labels(panel, horizons=(1,))

    for ticker in ["AAA", "BBB"]:
        close = panel.xs(ticker, level="ticker")["close"]
        lbl = labels.xs(ticker, level="ticker")["fwd_ret_1"]

        for i in range(len(close) - 1):
            expected = float(np.log(close.iloc[i + 1] / close.iloc[i]))
            actual = float(lbl.iloc[i])
            assert abs(actual - expected) < 1e-10, (
                f"{ticker} fwd_ret_1[{i}]: expected {expected}, got {actual}"
            )


def test_label_trailing_nans_per_ticker():
    """The last h rows of fwd_ret_h must be NaN for every ticker."""
    panel = _make_panel(["AAA", "BBB"])
    labels = build_panel_labels(panel, horizons=(1, 5))

    for ticker in ["AAA", "BBB"]:
        lbl = labels.xs(ticker, level="ticker")
        for h in (1, 5):
            tail = lbl[f"fwd_ret_{h}"].iloc[-h:]
            assert tail.isna().all(), (
                f"{ticker} fwd_ret_{h}: expected last {h} rows to be NaN"
            )


def test_label_for_ticker_a_independent_of_ticker_b():
    """Perturbing future closes for BBB must not change AAA's label values."""
    panel = _make_panel(["AAA", "BBB"])
    labels_orig = build_panel_labels(panel, horizons=(5,))

    panel_pert = panel.copy()
    bbb_mask = panel_pert.index.get_level_values("ticker") == "BBB"
    panel_pert.loc[bbb_mask, "close"] *= 10.0
    labels_pert = build_panel_labels(panel_pert, horizons=(5,))

    aaa_orig = labels_orig.xs("AAA", level="ticker")
    aaa_pert = labels_pert.xs("AAA", level="ticker")
    pd.testing.assert_frame_equal(aaa_orig, aaa_pert, obj="AAA labels after perturbing BBB")


# ── 4. Cross-sectional IC correctness ────────────────────────────────────────

def _make_cs_panel(seed: int = 0) -> pd.DataFrame:
    """Make a flat (date, ticker) panel with score and label columns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=20, freq="B")
    tickers = ["A", "B", "C", "D", "E"]
    rows = []
    for d in dates:
        scores = rng.normal(0, 1, len(tickers))
        labels = scores * 0.5 + rng.normal(0, 0.3, len(tickers))  # correlated
        for t, s, lab in zip(tickers, scores, labels):
            rows.append({"date": d, "ticker": t, "score": s, "label": lab})
    return pd.DataFrame(rows)


def test_daily_cs_ic_matches_manual():
    """daily_cs_ic should match manually computed per-date Pearson r."""
    from scipy import stats

    panel = _make_cs_panel()
    computed = daily_cs_ic(panel, "score", "label")

    for date, group in panel.groupby("date"):
        expected, _ = stats.pearsonr(group["score"], group["label"])
        actual = computed.loc[date]
        assert abs(actual - expected) < 1e-12, (
            f"CS-IC mismatch on {date}: expected {expected}, got {actual}"
        )


def test_daily_cs_rank_ic_matches_manual():
    """daily_cs_rank_ic should match manually computed Spearman r."""
    from scipy import stats

    panel = _make_cs_panel()
    computed = daily_cs_rank_ic(panel, "score", "label")

    for date, group in panel.groupby("date"):
        expected, _ = stats.spearmanr(group["score"], group["label"])
        actual = computed.loc[date]
        assert abs(actual - expected) < 1e-12, (
            f"CS Rank-IC mismatch on {date}: expected {expected}, got {actual}"
        )


def test_aggregate_cs_metrics_ic_t_stat():
    """IC t-stat must equal mean_ic / (std_ic / sqrt(n_dates))."""
    panel = _make_cs_panel()
    result = aggregate_cs_metrics(panel, "score", "label")

    expected_t = result.mean_ic / (result.std_ic / math.sqrt(result.n_dates))
    assert abs(result.ic_t_stat - expected_t) < 1e-10, (
        f"IC t-stat: expected {expected_t}, got {result.ic_t_stat}"
    )


def test_aggregate_cs_metrics_hit_rate():
    """IC hit rate must equal fraction of days with positive IC."""
    panel = _make_cs_panel()
    result = aggregate_cs_metrics(panel, "score", "label")

    ic_series = daily_cs_ic(panel, "score", "label")
    expected = float((ic_series > 0).mean())
    assert abs(result.ic_hit_rate - expected) < 1e-10


def test_cs_ic_perfect_predictor():
    """A perfect score (score == label) must yield IC = 1.0 on every date."""
    panel = _make_cs_panel()
    panel["perfect_score"] = panel["label"]
    result = aggregate_cs_metrics(panel, "perfect_score", "label")
    assert abs(result.mean_ic - 1.0) < 1e-10


def test_cs_ic_random_predictor_near_zero():
    """A score uncorrelated with labels should produce mean IC near zero."""
    rng = np.random.default_rng(7)
    panel = _make_cs_panel()
    panel["noise"] = rng.normal(0, 1, len(panel))
    result = aggregate_cs_metrics(panel, "noise", "label")
    # With 20 dates and 5 tickers, mean IC should be close to zero (no strict bound,
    # but absolute value should be well below 0.5 for unrelated noise).
    assert abs(result.mean_ic) < 0.5, (
        f"Random predictor IC unexpectedly large: {result.mean_ic}"
    )


# ── 5. Universe config loading ────────────────────────────────────────────────

def test_load_universe_returns_list_of_strings(tmp_path: Path):
    """load_universe should return a plain list of ticker strings."""
    yaml_content = (
        "universe:\n"
        "  name: test\n"
        "  tickers:\n"
        "    - AAA\n"
        "    - BBB   # inline comment\n"
        "    - CCC\n"
    )
    config = tmp_path / "universe.yaml"
    config.write_text(yaml_content, encoding="utf-8")

    tickers = load_universe(config)
    assert tickers == ["AAA", "BBB", "CCC"]


def test_load_universe_strips_inline_comments(tmp_path: Path):
    """Inline YAML comments after ticker symbols must be stripped."""
    yaml_content = (
        "universe:\n"
        "  name: test\n"
        "  tickers:\n"
        "    - SPY   # S&P 500 benchmark\n"
        "    - XLK   # Technology\n"
    )
    config = tmp_path / "universe.yaml"
    config.write_text(yaml_content, encoding="utf-8")

    tickers = load_universe(config)
    assert tickers == ["SPY", "XLK"]


def test_load_default_universe():
    """The default configs/universe.yaml should load 12 tickers."""
    tickers = load_universe()
    assert len(tickers) == 12
    assert "SPY" in tickers
    assert "XLC" in tickers
    assert "XLRE" in tickers
