"""Daily price feature engineering.

All features are constructed from information available *at the close of
that day*, so they are safe to use as predictors for the *next* day's
return without introducing lookahead bias.

Features
--------
log_return          : log(close_t / close_{t-1})
rolling_vol_20      : 20-day rolling std of log returns (annualised)
rolling_vol_5       : 5-day rolling std of log returns (annualised)
momentum_20         : cumulative log return over past 20 trading days
momentum_5          : cumulative log return over past 5 trading days
momentum_60         : cumulative log return over past 60 trading days
volume_zscore_20    : (volume - 20d mean) / 20d std
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of daily features aligned to *prices*.

    Parameters
    ----------
    prices:
        DataFrame with at least ``close`` and ``volume`` columns and a
        DatetimeIndex.  Typically the output of :func:`data.loader.load_prices`.

    Returns
    -------
    DataFrame with one row per trading day.  Rows where any feature is NaN
    (the warm-up period at the start of the series) are **not** dropped here;
    callers are responsible for dropping or filling them.
    """
    close = prices["close"].astype(float)
    volume = prices["volume"].astype(float)

    log_ret = np.log(close / close.shift(1))

    feat = pd.DataFrame(index=prices.index)

    feat["log_return"] = log_ret

    # Volatility: rolling std annualised
    feat["rolling_vol_5"] = log_ret.rolling(5).std() * np.sqrt(_TRADING_DAYS_PER_YEAR)
    feat["rolling_vol_20"] = log_ret.rolling(20).std() * np.sqrt(_TRADING_DAYS_PER_YEAR)

    # Momentum: sum of log returns over past N days (excludes today via shift(1))
    # shift(1) so momentum_20 is the return from t-21 to t-1 (not including today)
    feat["momentum_5"] = log_ret.shift(1).rolling(5).sum()
    feat["momentum_20"] = log_ret.shift(1).rolling(20).sum()
    feat["momentum_60"] = log_ret.shift(1).rolling(60).sum()

    # Volume z-score
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    feat["volume_zscore_20"] = (volume - vol_mean) / vol_std

    feat.index.name = "date"
    return feat


def build_panel_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute price features for every ticker in a panel dataset.

    Features are computed independently per ticker so that rolling windows
    never mix data across tickers.

    Parameters
    ----------
    panel:
        DataFrame with a ``(date, ticker)`` MultiIndex and at least ``close``
        and ``volume`` columns.  Typically the output of
        :func:`data.loader.load_panel`.

    Returns
    -------
    DataFrame with the same ``(date, ticker)`` MultiIndex and feature columns.
    NaN warm-up rows are retained; callers should ``dropna()`` after joining
    with labels.
    """
    tickers = panel.index.get_level_values("ticker").unique()
    pieces: list[pd.DataFrame] = []

    for ticker in tickers:
        # Extract a plain DatetimeIndex DataFrame for this ticker
        prices_t = panel.xs(ticker, level="ticker")
        feat_t = build_features(prices_t)
        # Re-attach the ticker level
        feat_t.index = pd.MultiIndex.from_arrays(
            [feat_t.index, [ticker] * len(feat_t)],
            names=["date", "ticker"],
        )
        pieces.append(feat_t)

    return pd.concat(pieces).sort_index()


FEATURE_COLS = [
    "log_return",
    "rolling_vol_5",
    "rolling_vol_20",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "volume_zscore_20",
]
