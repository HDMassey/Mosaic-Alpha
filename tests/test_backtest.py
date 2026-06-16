"""Tests for Milestone 5: transaction-cost-aware portfolio backtester.

Categories
----------
1. build_ls_weights: dollar-neutral, correct long/short selection, edge cases.
2. compute_turnover: first period, unchanged portfolio, full flip.
3. run_backtest: net < gross with non-zero costs, output schema, period count.
4. _max_drawdown: known series, monotone series, single period.
5. Graceful handling of dates with too few assets.
6. render_report: sample-data warning, required sections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic_alpha.research.backtest import (
    BacktestResult,
    _max_drawdown,
    build_ls_weights,
    compute_turnover,
    render_report,
    run_backtest,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_pooled_df(
    n_dates: int = 40,
    n_tickers: int = 8,
    seed: int = 0,
    score_col: str = "pmn_fwd_ret_5",
    label_col: str = "fwd_ret_5",
) -> pd.DataFrame:
    """Synthetic pooled predictions DataFrame for backtest tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    rows = []
    for date in dates:
        for ticker in tickers:
            rows.append({
                "date": date,
                "ticker": ticker,
                score_col: rng.normal(0, 1),
                label_col: rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


def _uniform_scores(tickers: list[str], ascending: bool = True) -> pd.Series:
    """Scores that unambiguously rank tickers 1..n."""
    scores = pd.Series(range(len(tickers)), index=tickers, dtype=float)
    if not ascending:
        scores = scores[::-1]
    return scores


# ── 1. build_ls_weights ────────────────────────────────────────────────────────

def test_weights_are_dollar_neutral():
    """Long weights must sum to +1, short weights to -1, net to 0."""
    tickers = ["A", "B", "C", "D", "E", "F", "G", "H"]
    scores = _uniform_scores(tickers)
    weights = build_ls_weights(scores, quantile=0.25)

    assert weights is not None
    long_sum = weights[weights > 0].sum()
    short_sum = weights[weights < 0].sum()
    net_sum = weights.sum()

    assert long_sum == pytest.approx(1.0, abs=1e-10)
    assert short_sum == pytest.approx(-1.0, abs=1e-10)
    assert net_sum == pytest.approx(0.0, abs=1e-10)


def test_weights_long_selects_top_quantile():
    """Longs must be the tickers with the highest scores."""
    tickers = ["A", "B", "C", "D", "E", "F", "G", "H"]  # 8 tickers
    # Scores: A=0 (lowest) .. H=7 (highest); quantile=0.25 -> top 2 = G, H
    scores = _uniform_scores(tickers)
    weights = build_ls_weights(scores, quantile=0.25)

    assert weights is not None
    long_tickers = set(weights[weights > 0].index)
    assert "H" in long_tickers  # highest score
    assert "G" in long_tickers  # second highest
    assert "A" not in long_tickers
    assert "B" not in long_tickers


def test_weights_short_selects_bottom_quantile():
    """Shorts must be the tickers with the lowest scores."""
    tickers = ["A", "B", "C", "D", "E", "F", "G", "H"]
    scores = _uniform_scores(tickers)
    weights = build_ls_weights(scores, quantile=0.25)

    assert weights is not None
    short_tickers = set(weights[weights < 0].index)
    assert "A" in short_tickers  # lowest score
    assert "B" in short_tickers  # second lowest
    assert "H" not in short_tickers
    assert "G" not in short_tickers


def test_weights_equal_within_each_leg():
    """All long (short) positions must have the same absolute weight."""
    tickers = [f"T{i}" for i in range(12)]
    scores = _uniform_scores(tickers)
    weights = build_ls_weights(scores, quantile=0.25)

    assert weights is not None
    long_w = weights[weights > 0]
    short_w = weights[weights < 0]
    assert long_w.nunique() == 1   # all equal
    assert short_w.nunique() == 1  # all equal


def test_weights_long_short_do_not_overlap():
    """No ticker should appear in both the long and short book."""
    tickers = [f"T{i}" for i in range(10)]
    scores = _uniform_scores(tickers)
    weights = build_ls_weights(scores, quantile=0.3)

    assert weights is not None
    long_set = set(weights[weights > 0].index)
    short_set = set(weights[weights < 0].index)
    assert long_set.isdisjoint(short_set)


def test_weights_returns_none_for_single_asset():
    """A cross-section with only 1 asset cannot form L/S; must return None."""
    scores = pd.Series({"SPY": 1.0})
    assert build_ls_weights(scores, quantile=0.25) is None


def test_weights_returns_none_for_empty_series():
    """An empty scores Series must return None."""
    scores = pd.Series(dtype=float)
    assert build_ls_weights(scores, quantile=0.25) is None


def test_weights_drops_nan_scores():
    """NaN scores should be dropped before constructing the portfolio."""
    scores = pd.Series({"A": 1.0, "B": np.nan, "C": 3.0, "D": 4.0})
    weights = build_ls_weights(scores, quantile=0.25)
    assert weights is not None
    assert "B" not in weights.index or weights["B"] == 0.0


# ── 2. compute_turnover ────────────────────────────────────────────────────────

def test_turnover_first_period_is_one():
    """Building a portfolio from cash must produce one-way turnover of 1.0."""
    scores = _uniform_scores(["A", "B", "C", "D"])
    weights = build_ls_weights(scores, quantile=0.25)
    assert weights is not None
    turnover = compute_turnover(weights, weights_old=None)
    assert turnover == pytest.approx(1.0, abs=1e-10)


def test_turnover_identical_weights_is_zero():
    """Holding the same portfolio produces zero turnover."""
    scores = _uniform_scores(["A", "B", "C", "D"])
    weights = build_ls_weights(scores, quantile=0.25)
    assert weights is not None
    turnover = compute_turnover(weights, weights_old=weights.copy())
    assert turnover == pytest.approx(0.0, abs=1e-10)


def test_turnover_full_flip_is_two():
    """A complete sign-flip of all positions produces one-way turnover of 2.0.

    Going from +1 long / -1 short to -1 long / +1 short means every dollar
    moves: the old long becomes the new short and vice versa.
    """
    # 4 assets: A<B<C<D by score
    scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    w_long_cd = build_ls_weights(scores, quantile=0.5)   # long C,D / short A,B

    # Flip scores so A,B become longs and C,D become shorts
    scores_flipped = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    w_long_ab = build_ls_weights(scores_flipped, quantile=0.5)  # long A,B / short C,D

    assert w_long_cd is not None
    assert w_long_ab is not None
    turnover = compute_turnover(w_long_ab, weights_old=w_long_cd)
    assert turnover == pytest.approx(2.0, abs=1e-10)


def test_turnover_partial_change():
    """Swapping one name in the long book produces a turnover between 0 and 2."""
    # Initial: long D (score 4), short A (score 1)
    scores_1 = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    w1 = build_ls_weights(scores_1, quantile=0.25)

    # Next period: long C (score 4), short A (score 1) -- D moves out, C moves in
    scores_2 = pd.Series({"A": 1.0, "B": 2.0, "C": 4.0, "D": 3.0})
    w2 = build_ls_weights(scores_2, quantile=0.25)

    assert w1 is not None
    assert w2 is not None
    turnover = compute_turnover(w2, weights_old=w1)
    assert 0.0 < turnover < 2.0


# ── 3. run_backtest ────────────────────────────────────────────────────────────

def test_net_returns_less_than_gross_with_positive_costs():
    """With cost_bps > 0, mean net return must be less than mean gross return."""
    df = _make_pooled_df(n_dates=50, n_tickers=8, seed=1)
    result = run_backtest(df, "pmn_fwd_ret_5", "fwd_ret_5", cost_bps=20.0)

    gross_mean = result.period_returns["gross_return"].mean()
    net_mean = result.period_returns["net_return"].mean()
    assert net_mean < gross_mean


def test_zero_cost_gross_equals_net():
    """With cost_bps=0, gross and net returns must be identical."""
    df = _make_pooled_df(n_dates=30, n_tickers=8, seed=2)
    result = run_backtest(df, "pmn_fwd_ret_5", "fwd_ret_5", cost_bps=0.0)

    pd.testing.assert_series_equal(
        result.period_returns["gross_return"],
        result.period_returns["net_return"],
        check_names=False,
    )


def test_backtest_result_has_correct_period_count():
    """With 40 dates and horizon=5, we expect 40//5 = 8 rebalance periods."""
    df = _make_pooled_df(n_dates=40, n_tickers=8, seed=3)
    result = run_backtest(df, "pmn_fwd_ret_5", "fwd_ret_5", horizon=5)
    assert result.n_periods == 8


def test_backtest_result_schema():
    """BacktestResult must expose all required fields with correct types."""
    df = _make_pooled_df(n_dates=20, n_tickers=8, seed=4)
    result = run_backtest(df, "pmn_fwd_ret_5", "fwd_ret_5", cost_bps=5.0)

    for attr in [
        "ann_gross_return", "ann_net_return",
        "ann_vol_gross", "ann_vol_net",
        "sharpe_gross", "sharpe_net",
        "max_drawdown_gross", "max_drawdown_net",
        "hit_rate", "avg_turnover",
    ]:
        val = getattr(result, attr)
        assert isinstance(val, float), f"{attr} should be float, got {type(val)}"

    assert isinstance(result.period_returns, pd.DataFrame)
    assert isinstance(result.cumulative_gross, pd.Series)
    assert isinstance(result.cumulative_net, pd.Series)
    assert "gross_return" in result.period_returns.columns
    assert "net_return" in result.period_returns.columns
    assert "turnover" in result.period_returns.columns


def test_backtest_hit_rate_in_unit_interval():
    """hit_rate must be in [0, 1]."""
    df = _make_pooled_df(n_dates=40, n_tickers=8, seed=5)
    result = run_backtest(df, "pmn_fwd_ret_5", "fwd_ret_5")
    assert 0.0 <= result.hit_rate <= 1.0


def test_backtest_cumulative_starts_at_exp_first_return():
    """Cumulative gross return at the first period equals exp(first gross return)."""
    df = _make_pooled_df(n_dates=20, n_tickers=8, seed=6)
    result = run_backtest(df, "pmn_fwd_ret_5", "fwd_ret_5")

    if result.n_periods > 0:
        first_gross = result.period_returns["gross_return"].iloc[0]
        first_cum = result.cumulative_gross.iloc[0]
        assert first_cum == pytest.approx(np.exp(first_gross), rel=1e-9)


# ── 4. _max_drawdown ──────────────────────────────────────────────────────────

def test_max_drawdown_known_series():
    """Known peak-to-trough drawdown: 1.0 -> 0.5 = -50%."""
    cum = pd.Series([1.0, 1.2, 1.5, 1.0, 0.75, 0.9, 1.1])
    # Peak = 1.5, trough = 0.75 -> drawdown = 0.75/1.5 - 1 = -0.5
    dd = _max_drawdown(cum)
    assert dd == pytest.approx(-0.5, rel=1e-9)


def test_max_drawdown_monotone_increasing_is_zero():
    """A monotonically increasing equity curve has max drawdown of 0."""
    cum = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
    dd = _max_drawdown(cum)
    assert dd == pytest.approx(0.0, abs=1e-10)


def test_max_drawdown_single_period():
    """A one-period series has max drawdown of 0 (only one observation)."""
    cum = pd.Series([0.95])
    # There's no prior peak to compare against -- rolling max of [0.95] is [0.95]
    # drawdown = 0.95/0.95 - 1 = 0
    dd = _max_drawdown(cum)
    assert dd == pytest.approx(0.0, abs=1e-10)


def test_max_drawdown_empty_series():
    """Empty cumulative series must return NaN."""
    import math
    dd = _max_drawdown(pd.Series(dtype=float))
    assert math.isnan(dd)


# ── 5. Graceful handling of too-few assets ─────────────────────────────────────

def test_backtest_skips_dates_with_one_asset():
    """Dates with only one ticker must be silently skipped (no error)."""
    # Build a pooled_df where most dates have 8 tickers but a few have 1
    normal = _make_pooled_df(n_dates=20, n_tickers=8, seed=7)
    # Insert a "bad" date with only 1 ticker
    bad_date = pd.Timestamp("2021-06-15")
    bad_row = pd.DataFrame([{
        "date": bad_date, "ticker": "SOLO",
        "pmn_fwd_ret_5": 1.0, "fwd_ret_5": 0.01,
    }])
    df = pd.concat([normal, bad_row], ignore_index=True)

    # Should not raise; the bad date is skipped
    result = run_backtest(df, "pmn_fwd_ret_5", "fwd_ret_5")
    assert result.n_periods >= 0  # might be 0 if horizon skips all good dates


def test_backtest_with_empty_dataframe_returns_empty_result():
    """An empty pooled_df must return a BacktestResult with n_periods=0."""
    df = pd.DataFrame(columns=["date", "ticker", "pmn_fwd_ret_5", "fwd_ret_5"])
    result = run_backtest(df, "pmn_fwd_ret_5", "fwd_ret_5")
    assert result.n_periods == 0


def test_backtest_raises_on_missing_score_column():
    """run_backtest must raise ValueError when the score column is absent."""
    df = _make_pooled_df(n_dates=10, n_tickers=4)
    with pytest.raises(ValueError, match="missing columns"):
        run_backtest(df, "nonexistent_score", "fwd_ret_5")


# ── 6. render_report ──────────────────────────────────────────────────────────

def _dummy_backtest_result() -> BacktestResult:
    dates = pd.date_range("2022-01-01", periods=5, freq="5B")
    period_df = pd.DataFrame({
        "gross_return": [0.01, -0.005, 0.008, 0.012, -0.002],
        "net_return": [0.0095, -0.0055, 0.0075, 0.0115, -0.0025],
        "turnover": [1.0, 0.3, 0.3, 0.4, 0.2],
        "cost": [0.0005, 0.00015, 0.00015, 0.0002, 0.0001],
        "n_long": [2, 2, 2, 2, 2],
        "n_short": [2, 2, 2, 2, 2],
    }, index=dates)
    cum_g = np.exp(np.cumsum(period_df["gross_return"].values))
    cum_n = np.exp(np.cumsum(period_df["net_return"].values))
    return BacktestResult(
        score_col="pmn_fwd_ret_5",
        label_col="fwd_ret_5",
        quantile=0.25,
        cost_bps=5.0,
        horizon=5,
        n_periods=5,
        ann_gross_return=0.08,
        ann_vol_gross=0.06,
        sharpe_gross=1.33,
        max_drawdown_gross=-0.03,
        ann_net_return=0.075,
        ann_vol_net=0.06,
        sharpe_net=1.25,
        max_drawdown_net=-0.032,
        hit_rate=0.6,
        avg_turnover=0.44,
        period_returns=period_df,
        cumulative_gross=pd.Series(cum_g, index=dates),
        cumulative_net=pd.Series(cum_n, index=dates),
    )


def test_render_report_live_mode_no_warning(tmp_path: Path):
    """Live-mode report must not contain SAMPLE DATA warning."""
    result = _dummy_backtest_result()
    out = tmp_path / "report.md"
    render_report(result, out, data_mode="live_gdelt")
    text = out.read_text(encoding="utf-8")
    assert "SAMPLE DATA" not in text
    assert "Performance summary" in text
    assert "Limitations" in text


def test_render_report_sample_mode_includes_warning(tmp_path: Path):
    """Offline-sample report must contain a SAMPLE DATA warning."""
    result = _dummy_backtest_result()
    out = tmp_path / "report.md"
    render_report(result, out, data_mode="offline_sample")
    text = out.read_text(encoding="utf-8")
    assert "SAMPLE DATA" in text


def test_render_report_contains_period_table(tmp_path: Path):
    """Report must include the period-returns table."""
    result = _dummy_backtest_result()
    out = tmp_path / "report.md"
    render_report(result, out)
    text = out.read_text(encoding="utf-8")
    assert "Period returns" in text
    assert "Gross" in text
    assert "Net" in text
