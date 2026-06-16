"""Macroeconomic regime feature engineering.

All features are derived from FRED time series that have been forward-filled
to a daily calendar.  The final step in :func:`build_macro_features` is a
``shift(1)`` that ensures every feature value at date *t* uses only data that
was published on or before *t-1*.  This conservative one-day lag is sufficient
for daily market rates (DGS10, BAA10Y) and more than sufficient for monthly
series (CPI, UNRATE) which carry a 2-4 week release lag in practice.

Features
--------
yield_curve_10y_2y    : DGS10 - DGS2  (slope of the yield curve)
fed_funds_change_3m   : FEDFUNDS change over past ~63 trading days
inflation_yoy         : Year-over-year log change in CPIAUCSL (%)
unemployment_change_3m: UNRATE change over past ~63 trading days
credit_spread_level   : BAA10Y (already a spread over Treasuries)
macro_risk_zscore     : Rolling z-score composite of inflation, unemployment
                        change, and credit spread — higher = more stress
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
_ROLLING_WINDOW = 252  # 1 trading year for rolling z-score normalisation
# Approximate trading days for calendar periods
_3M_DAYS = 63
_1Y_DAYS = 252

MACRO_FEATURE_COLS = [
    "yield_curve_10y_2y",
    "fed_funds_change_3m",
    "inflation_yoy",
    "unemployment_change_3m",
    "credit_spread_level",
    "macro_risk_zscore",
]


def load_macro_config(config_path: Path | None = None) -> list[str]:
    """Return the list of FRED series IDs from the macro config."""
    path = config_path or (_CONFIGS_DIR / "macro.yaml")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return list(cfg["macro"]["series"].keys())


def _rolling_zscore(s: pd.Series, window: int = _ROLLING_WINDOW) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    m = s.rolling(window, min_periods=window // 4).mean()
    sd = s.rolling(window, min_periods=window // 4).std()
    return (s - m) / sd


def build_macro_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute macro regime features from raw FRED series.

    Parameters
    ----------
    raw:
        DataFrame with a daily DatetimeIndex and columns for each FRED series:
        ``DGS10``, ``DGS2``, ``FEDFUNDS``, ``CPIAUCSL``, ``UNRATE``, ``BAA10Y``.
        Typically the output of :func:`data.fred.fetch_macro_panel`.

    Returns
    -------
    DataFrame with :data:`MACRO_FEATURE_COLS` columns and the same daily index,
    already lagged by one day so features at date *t* are safe to join with
    return labels for *t* without lookahead.
    """
    feat = pd.DataFrame(index=raw.index)

    # ── Yield curve slope ──────────────────────────────────────────────────────
    feat["yield_curve_10y_2y"] = raw["DGS10"] - raw["DGS2"]

    # ── Fed funds rate change over ~3 months ──────────────────────────────────
    feat["fed_funds_change_3m"] = raw["FEDFUNDS"] - raw["FEDFUNDS"].shift(_3M_DAYS)

    # ── Year-over-year inflation (log % change in CPI) ────────────────────────
    # CPIAUCSL is monthly, forward-filled to daily.
    # shift(252) ≈ one calendar year of daily obs.
    feat["inflation_yoy"] = (
        np.log(raw["CPIAUCSL"] / raw["CPIAUCSL"].shift(_1Y_DAYS)) * 100
    )

    # ── Unemployment rate change over ~3 months ───────────────────────────────
    feat["unemployment_change_3m"] = raw["UNRATE"] - raw["UNRATE"].shift(_3M_DAYS)

    # ── Credit spread level ───────────────────────────────────────────────────
    feat["credit_spread_level"] = raw["BAA10Y"]

    # ── Composite macro risk z-score ──────────────────────────────────────────
    # Positive = elevated stress: high inflation, rising unemployment, wide spreads.
    z_infl = _rolling_zscore(feat["inflation_yoy"])
    z_unemp = _rolling_zscore(feat["unemployment_change_3m"])
    z_credit = _rolling_zscore(feat["credit_spread_level"])
    feat["macro_risk_zscore"] = (z_infl + z_unemp + z_credit) / 3

    # ── Lag all features by 1 trading day ─────────────────────────────────────
    # This ensures that feature[t] uses only data published on or before t-1,
    # preventing same-day lookahead regardless of the underlying series frequency.
    feat = feat.shift(1)
    feat.index.name = "date"

    return feat
