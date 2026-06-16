"""News intensity feature engineering from GDELT article counts.

All features are derived from the raw daily article-count panel produced by
:func:`data.gdelt.fetch_news_panel`.  The final step in
:func:`build_news_features` is a ``shift(1)`` that ensures every feature at
date *t* uses only data available on or before *t-1*, preventing lookahead.

Features (per ticker)
---------------------
news_count            : Raw daily article count for the sector (float).
news_count_7d_avg     : 7-day rolling mean of news_count.
news_count_30d_avg    : 30-day rolling mean of news_count.
abnormal_news_zscore  : Z-score of news_count relative to 30-day rolling
                        mean and std -- measures abnormal news intensity.
news_momentum_7d_30d  : Ratio (7d avg / 30d avg) - 1, capturing whether
                        near-term news intensity is elevated vs. baseline.
                        NaN where 30d avg is zero.

All five features are shift(1)-lagged before being returned, so feature[t]
uses data from t-1 and earlier only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"

NEWS_FEATURE_COLS = [
    "news_count",
    "news_count_7d_avg",
    "news_count_30d_avg",
    "abnormal_news_zscore",
    "news_momentum_7d_30d",
]


def load_gdelt_tickers(config_path: Path | None = None) -> list[str]:
    """Return the list of tickers defined in ``configs/gdelt.yaml``."""
    path = config_path or (_CONFIGS_DIR / "gdelt.yaml")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return list(cfg["gdelt"]["sector_keywords"].keys())


def build_ticker_news_features(counts: pd.Series) -> pd.DataFrame:
    """Compute news features for a single ticker's daily count series.

    Parameters
    ----------
    counts:
        Daily article count ``pd.Series`` with a ``DatetimeIndex``.
        Missing values are treated as zero (no articles).

    Returns
    -------
    ``pd.DataFrame`` with :data:`NEWS_FEATURE_COLS` columns and the same
    index as *counts*, lagged by one day.
    """
    s = counts.fillna(0.0).astype(float)

    feat = pd.DataFrame(index=s.index)
    feat["news_count"] = s
    feat["news_count_7d_avg"] = s.rolling(7, min_periods=1).mean()
    feat["news_count_30d_avg"] = s.rolling(30, min_periods=7).mean()

    # Abnormal news: z-score relative to 30-day rolling stats
    rolling_mean = s.rolling(30, min_periods=7).mean()
    rolling_std = s.rolling(30, min_periods=7).std()
    # Avoid division by zero: where std==0, z-score is 0
    feat["abnormal_news_zscore"] = np.where(
        rolling_std > 0,
        (s - rolling_mean) / rolling_std,
        0.0,
    )

    # News momentum: 7d avg relative to 30d avg
    avg_30 = feat["news_count_30d_avg"]
    avg_7 = feat["news_count_7d_avg"]
    feat["news_momentum_7d_30d"] = np.where(
        avg_30 > 0,
        avg_7 / avg_30 - 1.0,
        np.nan,
    )

    # Lag all features by 1 day to prevent lookahead
    feat = feat.shift(1)
    feat.index.name = "date"

    return feat[NEWS_FEATURE_COLS]


def build_news_features(
    news_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compute news features for all tickers in *news_panel*.

    Parameters
    ----------
    news_panel:
        Wide-format DataFrame with a ``DatetimeIndex`` (date) and one column
        per sector ticker.  Typically the output of
        :func:`data.gdelt.fetch_news_panel`.

    Returns
    -------
    ``pd.DataFrame`` with a ``(date, ticker)`` MultiIndex and
    :data:`NEWS_FEATURE_COLS` columns, shift(1)-lagged.
    """
    pieces: list[pd.DataFrame] = []
    for ticker in news_panel.columns:
        feat_t = build_ticker_news_features(news_panel[ticker])
        feat_t.index = pd.MultiIndex.from_arrays(
            [feat_t.index, [ticker] * len(feat_t)],
            names=["date", "ticker"],
        )
        pieces.append(feat_t)

    if not pieces:
        return pd.DataFrame(columns=NEWS_FEATURE_COLS)

    result = pd.concat(pieces).sort_index()
    return result
