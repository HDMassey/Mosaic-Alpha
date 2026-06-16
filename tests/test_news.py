"""Tests for Milestone 4: GDELT connector, news features, and news-sector pipeline.

Categories
----------
1. GDELT config loader returns the expected ticker -> keyword mapping.
2. GDELT timeline parser handles both response shapes and edge cases.
3. fetch_sector_counts fills zeros for missing days and caches correctly.
4. fetch_sector_counts raises RateLimitError after exhausting 429 retries.
5. generate_sample_news_panel produces deterministic, plausible data.
6. News features are shift(1)-lagged with correct column set.
7. News feature values are correct relative to the raw count series.
8. News merge broadcasts correctly across tickers per date.
9. 4-way result schema (FourWayComparison, NewsSectorResult) is correct.

All tests that involve the GDELT API use mocks; no live network calls are made.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mosaic_alpha.data.gdelt import (
    RateLimitError,
    _build_query_string,
    _parse_gdelt_timeline,
    fetch_sector_counts,
    generate_sample_news_panel,
    load_gdelt_config,
)
from mosaic_alpha.features.news_features import (
    NEWS_FEATURE_COLS,
    build_news_features,
    build_ticker_news_features,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_gdelt_response(points: list[dict]) -> MagicMock:
    """Build a mock requests.Response for a GDELT timelineVol reply."""
    payload = {"timeline": [{"data": points}]}
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.status_code = 200
    mock.json.return_value = payload
    return mock


def _make_429_response() -> MagicMock:
    """Build a mock requests.Response that returns HTTP 429."""
    mock = MagicMock()
    mock.status_code = 429
    mock.raise_for_status.side_effect = None  # won't be called -- we check status first
    return mock


def _make_count_series(n: int = 100, seed: int = 0) -> pd.Series:
    """Synthetic daily article count series."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    counts = rng.integers(0, 500, n).astype(float)
    s = pd.Series(counts, index=dates, name="XLK")
    s.index.name = "date"
    return s


# ── 1. GDELT config loader ─────────────────────────────────────────────────────

def test_load_gdelt_config_returns_dict(tmp_path: Path):
    """load_gdelt_config returns a dict mapping ticker to keyword list."""
    content = (
        "gdelt:\n"
        "  sector_keywords:\n"
        "    XLE:\n"
        "      - energy\n"
        "      - oil\n"
        "    XLK:\n"
        "      - technology\n"
        "      - cloud\n"
    )
    p = tmp_path / "gdelt.yaml"
    p.write_text(content, encoding="utf-8")
    cfg = load_gdelt_config(p)
    assert isinstance(cfg, dict)
    assert "XLE" in cfg
    assert "XLK" in cfg
    assert "energy" in cfg["XLE"]
    assert "cloud" in cfg["XLK"]


def test_load_gdelt_config_default_has_eleven_tickers():
    """The default configs/gdelt.yaml must have exactly 11 sector tickers (no SPY)."""
    cfg = load_gdelt_config()
    assert len(cfg) == 11
    assert "SPY" not in cfg
    assert "XLE" in cfg
    assert "XLC" in cfg


def test_build_query_string_quotes_phrases():
    """Multi-word keywords must be double-quoted in the query string."""
    q = _build_query_string(["energy", "oil gas", "crude"])
    assert '"oil gas"' in q
    assert "energy" in q
    assert "crude" in q
    assert " OR " in q


# ── 2. GDELT timeline parser ──────────────────────────────────────────────────

def test_parse_gdelt_timeline_aggregates_to_daily():
    """15-minute buckets from the same date must be summed into one daily value."""
    points = [
        {"date": "20220103000000", "value": 10},
        {"date": "20220103001500", "value": 5},
        {"date": "20220104000000", "value": 20},
    ]
    data = {"timeline": [{"data": points}]}
    daily = _parse_gdelt_timeline(data)
    assert daily["2022-01-03"] == pytest.approx(15.0)
    assert daily["2022-01-04"] == pytest.approx(20.0)


def test_parse_gdelt_timeline_empty_response():
    """An empty or malformed response must return an empty dict without raising."""
    assert _parse_gdelt_timeline({}) == {}
    assert _parse_gdelt_timeline({"timeline": []}) == {}


def test_parse_gdelt_timeline_dict_timeline_shape():
    """The parser must handle a dict-style timeline (not wrapped in a list)."""
    data = {
        "timeline": {
            "data": [
                {"date": "20221005000000", "value": 42},
            ]
        }
    }
    daily = _parse_gdelt_timeline(data)
    assert "2022-10-05" in daily
    assert daily["2022-10-05"] == pytest.approx(42.0)


def test_parse_gdelt_timeline_ignores_short_dates():
    """Entries with date strings shorter than 8 chars must be silently skipped."""
    data = {
        "timeline": [{"data": [
            {"date": "202201", "value": 999},
            {"date": "20220103000000", "value": 7},
        ]}]
    }
    daily = _parse_gdelt_timeline(data)
    assert len(daily) == 1
    assert "2022-01-03" in daily


# ── 3. fetch_sector_counts caching and zero-fill ──────────────────────────────

@patch("mosaic_alpha.data.gdelt.requests.get")
def test_fetch_sector_counts_zero_fills_missing_days(mock_get, tmp_path: Path):
    """Days with no articles must be zero-filled (not NaN) in the returned series."""
    points = [
        {"date": "20230103000000", "value": 50},
        {"date": "20230105000000", "value": 30},
    ]
    mock_get.return_value = _make_gdelt_response(points)

    with patch("mosaic_alpha.data.gdelt._CACHE_DIR", tmp_path):
        s = fetch_sector_counts(
            "XLK", ["technology"], "2023-01-03", "2023-01-07", use_cache=False, sleep_secs=0
        )

    expected_dates = pd.date_range("2023-01-03", "2023-01-07", freq="D")
    assert set(s.index) == set(expected_dates)
    assert s.loc[pd.Timestamp("2023-01-04")] == pytest.approx(0.0)
    assert s.loc[pd.Timestamp("2023-01-06")] == pytest.approx(0.0)
    assert s.loc[pd.Timestamp("2023-01-03")] == pytest.approx(50.0)
    assert s.loc[pd.Timestamp("2023-01-05")] == pytest.approx(30.0)


@patch("mosaic_alpha.data.gdelt.requests.get")
def test_fetch_sector_counts_returns_named_series(mock_get, tmp_path: Path):
    """Returned Series must be named after the ticker."""
    mock_get.return_value = _make_gdelt_response(
        [{"date": "20230103000000", "value": 10}]
    )
    with patch("mosaic_alpha.data.gdelt._CACHE_DIR", tmp_path):
        s = fetch_sector_counts(
            "XLE", ["energy"], "2023-01-03", "2023-01-03", use_cache=False, sleep_secs=0
        )
    assert s.name == "XLE"


@patch("mosaic_alpha.data.gdelt.requests.get")
def test_fetch_sector_counts_uses_cache(mock_get, tmp_path: Path):
    """Second call with use_cache=True must not make another HTTP request."""
    mock_get.return_value = _make_gdelt_response(
        [{"date": "20230103000000", "value": 10}]
    )
    with patch("mosaic_alpha.data.gdelt._CACHE_DIR", tmp_path):
        fetch_sector_counts(
            "XLK", ["technology"], "2023-01-03", "2023-01-03", use_cache=True, sleep_secs=0
        )
        assert mock_get.call_count == 1

        fetch_sector_counts(
            "XLK", ["technology"], "2023-01-03", "2023-01-03", use_cache=True, sleep_secs=0
        )
        assert mock_get.call_count == 1  # no new API call


@patch("mosaic_alpha.data.gdelt.requests.get")
def test_fetch_sector_counts_sends_user_agent(mock_get, tmp_path: Path):
    """Every request must include a User-Agent header."""
    mock_get.return_value = _make_gdelt_response(
        [{"date": "20230103000000", "value": 5}]
    )
    with patch("mosaic_alpha.data.gdelt._CACHE_DIR", tmp_path):
        fetch_sector_counts(
            "XLK", ["tech"], "2023-01-03", "2023-01-03", use_cache=False, sleep_secs=0
        )
    _, kwargs = mock_get.call_args
    headers = kwargs.get("headers", {})
    assert "User-Agent" in headers
    assert headers["User-Agent"]  # non-empty


# ── 4. RateLimitError on HTTP 429 ─────────────────────────────────────────────

@patch("mosaic_alpha.data.gdelt.time.sleep")
@patch("mosaic_alpha.data.gdelt.requests.get")
def test_rate_limit_error_raised_on_persistent_429(mock_get, mock_sleep, tmp_path: Path):
    """fetch_sector_counts must raise RateLimitError when all retries return 429."""
    mock_get.return_value = _make_429_response()

    with patch("mosaic_alpha.data.gdelt._CACHE_DIR", tmp_path):
        with pytest.raises(RateLimitError, match="429"):
            fetch_sector_counts(
                "XLK", ["technology"], "2023-01-03", "2023-01-07",
                use_cache=False, sleep_secs=0, max_retries=2,
            )


@patch("mosaic_alpha.data.gdelt.time.sleep")
@patch("mosaic_alpha.data.gdelt.requests.get")
def test_rate_limit_error_message_contains_suggestions(mock_get, mock_sleep, tmp_path: Path):
    """RateLimitError message must mention --offline-sample and --sleep-seconds."""
    mock_get.return_value = _make_429_response()

    with patch("mosaic_alpha.data.gdelt._CACHE_DIR", tmp_path):
        with pytest.raises(RateLimitError) as exc_info:
            fetch_sector_counts(
                "XLK", ["technology"], "2023-01-03", "2023-01-07",
                use_cache=False, sleep_secs=0, max_retries=2,
            )
    msg = str(exc_info.value)
    assert "--offline-sample" in msg
    assert "--sleep-seconds" in msg


@patch("mosaic_alpha.data.gdelt.time.sleep")
@patch("mosaic_alpha.data.gdelt.requests.get")
def test_no_rate_limit_error_on_eventual_success(mock_get, mock_sleep, tmp_path: Path):
    """If any retry succeeds (non-429), RateLimitError must NOT be raised."""
    # First call returns 429, second returns success
    mock_get.side_effect = [
        _make_429_response(),
        _make_gdelt_response([{"date": "20230103000000", "value": 7}]),
    ]

    with patch("mosaic_alpha.data.gdelt._CACHE_DIR", tmp_path):
        s = fetch_sector_counts(
            "XLK", ["technology"], "2023-01-03", "2023-01-03",
            use_cache=False, sleep_secs=0, max_retries=3,
        )
    assert s.loc[pd.Timestamp("2023-01-03")] == pytest.approx(7.0)


# ── 5. generate_sample_news_panel ─────────────────────────────────────────────

def test_sample_panel_shape():
    """generate_sample_news_panel must return one column per ticker."""
    tickers = ["XLK", "XLE", "XLF"]
    panel = generate_sample_news_panel(tickers, "2022-01-01", "2022-03-31")
    assert list(panel.columns) == tickers
    expected_dates = pd.date_range("2022-01-01", "2022-03-31", freq="D")
    assert len(panel) == len(expected_dates)


def test_sample_panel_deterministic():
    """Two calls with the same seed must return identical DataFrames."""
    tickers = ["XLK", "XLE"]
    p1 = generate_sample_news_panel(tickers, "2021-01-01", "2021-06-30", seed=7)
    p2 = generate_sample_news_panel(tickers, "2021-01-01", "2021-06-30", seed=7)
    pd.testing.assert_frame_equal(p1, p2)


def test_sample_panel_different_seeds_differ():
    """Different seeds must produce different output."""
    tickers = ["XLK"]
    p1 = generate_sample_news_panel(tickers, "2021-01-01", "2021-06-30", seed=1)
    p2 = generate_sample_news_panel(tickers, "2021-01-01", "2021-06-30", seed=2)
    assert not p1["XLK"].equals(p2["XLK"])


def test_sample_panel_non_negative():
    """All synthetic counts must be non-negative."""
    tickers = ["XLE", "XLK", "XLF", "XLV", "XLI"]
    panel = generate_sample_news_panel(tickers, "2018-01-01", "2022-12-31")
    assert (panel >= 0).all().all()


def test_sample_panel_tickers_differ():
    """Different tickers must produce different count series."""
    tickers = ["XLK", "XLE"]
    panel = generate_sample_news_panel(tickers, "2021-01-01", "2021-12-31", seed=42)
    assert not panel["XLK"].equals(panel["XLE"])


# ── 6. News features are lagged ───────────────────────────────────────────────

def test_news_features_are_lagged_by_one_day():
    """news_count at date t must equal the raw count at t-1."""
    counts = _make_count_series(50)
    feat = build_ticker_news_features(counts)

    date_t = counts.index[5]
    date_t_minus_1 = counts.index[4]

    assert feat["news_count"].loc[date_t] == pytest.approx(
        counts.loc[date_t_minus_1]
    )


def test_news_features_first_row_is_nan():
    """After shift(1), the first row of news_count must be NaN."""
    counts = _make_count_series(50)
    feat = build_ticker_news_features(counts)
    assert np.isnan(feat["news_count"].iloc[0])


def test_news_features_returns_all_expected_columns():
    """build_ticker_news_features must return exactly NEWS_FEATURE_COLS."""
    counts = _make_count_series(60)
    feat = build_ticker_news_features(counts)
    assert list(feat.columns) == NEWS_FEATURE_COLS


def test_news_features_index_matches_input():
    """Feature index must equal the input count series index."""
    counts = _make_count_series(80)
    feat = build_ticker_news_features(counts)
    assert feat.index.equals(counts.index)


# ── 7. News feature value correctness ─────────────────────────────────────────

def test_news_count_7d_avg_correct():
    """news_count_7d_avg at date t should be 7-day rolling mean ending at t-1."""
    counts = _make_count_series(60)
    feat = build_ticker_news_features(counts)

    date_t = counts.index[10]
    # Unshifted 7d avg at position 9 = mean of positions 3..9 (7 values)
    expected = counts.iloc[3:10].mean()
    assert feat["news_count_7d_avg"].loc[date_t] == pytest.approx(expected, rel=1e-6)


def test_news_momentum_zero_when_baseline_zero():
    """news_momentum_7d_30d must be NaN when the 30d average is zero."""
    counts = pd.Series(
        [0.0] * 40 + [100.0] * 20,
        index=pd.date_range("2020-01-01", periods=60, freq="D"),
        name="XLK",
    )
    counts.index.name = "date"
    feat = build_ticker_news_features(counts)
    assert np.isnan(feat["news_momentum_7d_30d"].iloc[1])


# ── 8. News merge broadcasts per ticker ──────────────────────────────────────

def test_build_news_features_multiindex():
    """build_news_features must return a (date, ticker) MultiIndex DataFrame."""
    news_panel = pd.DataFrame(
        {
            "XLK": _make_count_series(50),
            "XLE": _make_count_series(50, seed=1),
        }
    )
    feat = build_news_features(news_panel)
    assert feat.index.names == ["date", "ticker"]
    assert set(feat.index.get_level_values("ticker").unique()) == {"XLK", "XLE"}


def test_build_news_features_different_values_per_ticker():
    """Different tickers must have different news feature values on the same date."""
    counts_xlk = _make_count_series(50, seed=42)
    counts_xle = _make_count_series(50, seed=99)
    news_panel = pd.DataFrame({"XLK": counts_xlk, "XLE": counts_xle})
    feat = build_news_features(news_panel)

    date = feat.index.get_level_values("date").unique()[5]
    val_xlk = feat.xs("XLK", level="ticker").loc[date, "news_count"]
    val_xle = feat.xs("XLE", level="ticker").loc[date, "news_count"]
    assert val_xlk != pytest.approx(val_xle)


# ── 9. Result schema ──────────────────────────────────────────────────────────

def test_four_way_comparison_schema():
    """FourWayComparison and NewsSectorResult must expose the expected attributes."""
    from mosaic_alpha.research.cross_sectional import CrossSectionalResult
    from mosaic_alpha.research.news_sector_baseline import (
        DATA_MODE_LIVE,
        DATA_MODE_SAMPLE,
        FourWayComparison,
        NewsSectorResult,
    )

    def _dummy_cs(label: str) -> CrossSectionalResult:
        return CrossSectionalResult(
            score_col="score",
            label_col=label,
            n_dates=100,
            mean_ic=0.03,
            std_ic=0.08,
            ic_t_stat=1.2,
            mean_rank_ic=0.025,
            std_rank_ic=0.07,
            ic_hit_rate=0.53,
            mean_ls_spread=0.0005,
            std_ls_spread=0.001,
        )

    cmp = FourWayComparison(
        label_col="fwd_ret_1",
        price_only=_dummy_cs("fwd_ret_1"),
        price_macro=_dummy_cs("fwd_ret_1"),
        price_news=_dummy_cs("fwd_ret_1"),
        price_macro_news=_dummy_cs("fwd_ret_1"),
    )

    # Live mode
    result_live = NewsSectorResult(
        tickers=["SPY", "XLK"],
        news_tickers=["XLK"],
        macro_series=["DGS10"],
        start="2018-01-01",
        end="2024-12-31",
        n_panel_rows=3000,
        n_dates=300,
        n_folds=8,
        data_mode=DATA_MODE_LIVE,
        comparisons=[cmp],
    )
    assert result_live.data_mode == "live_gdelt"

    # Sample mode
    result_sample = NewsSectorResult(
        tickers=["SPY", "XLK"],
        news_tickers=["XLK"],
        macro_series=["DGS10"],
        start="2018-01-01",
        end="2024-12-31",
        n_panel_rows=3000,
        n_dates=300,
        n_folds=8,
        data_mode=DATA_MODE_SAMPLE,
        comparisons=[cmp],
    )
    assert result_sample.data_mode == "offline_sample"
    assert len(result_sample.comparisons) == 1
    assert result_sample.comparisons[0].price_macro_news.mean_ic == pytest.approx(0.03)
    assert result_sample.news_tickers == ["XLK"]
    assert result_sample.n_folds == 8


def test_report_includes_sample_warning(tmp_path: Path):
    """render_report must include a SAMPLE DATA warning in offline sample mode."""
    from mosaic_alpha.research.cross_sectional import CrossSectionalResult
    from mosaic_alpha.research.news_sector_baseline import (
        DATA_MODE_SAMPLE,
        FourWayComparison,
        NewsSectorResult,
        render_report,
    )

    def _dummy_cs() -> CrossSectionalResult:
        return CrossSectionalResult(
            score_col="s", label_col="fwd_ret_1", n_dates=50,
            mean_ic=0.0, std_ic=0.1, ic_t_stat=0.0,
            mean_rank_ic=0.0, std_rank_ic=0.1,
            ic_hit_rate=0.5, mean_ls_spread=0.0, std_ls_spread=0.0,
        )

    result = NewsSectorResult(
        tickers=["SPY", "XLK"],
        news_tickers=["XLK"],
        macro_series=["DGS10"],
        start="2020-01-01",
        end="2020-12-31",
        n_panel_rows=100,
        n_dates=50,
        n_folds=2,
        data_mode=DATA_MODE_SAMPLE,
        comparisons=[
            FourWayComparison(
                label_col="fwd_ret_1",
                price_only=_dummy_cs(),
                price_macro=_dummy_cs(),
                price_news=_dummy_cs(),
                price_macro_news=_dummy_cs(),
            )
        ],
    )
    out = tmp_path / "report.md"
    render_report(result, out)
    text = out.read_text(encoding="utf-8")
    assert "SAMPLE DATA" in text
    assert "offline_sample" in text
    assert "Limitations" in text
