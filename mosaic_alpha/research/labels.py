"""Forward-return label construction.

Labels are *future* log returns, i.e. what the model is trying to predict.
They are shifted *backwards* relative to the feature row, which means:

    label[t]  = log(close[t+h] / close[t])

where h is the horizon in trading days.

IMPORTANT: labels must never be joined to the feature matrix without
removing the last *h* rows of the combined frame.  Failing to do so leaks
future information into training.  The :func:`build_labels` function marks
those trailing rows as NaN to make the danger visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_HORIZONS = (1, 5, 20)


def build_labels(prices: pd.DataFrame, horizons: tuple[int, ...] = _HORIZONS) -> pd.DataFrame:
    """Return forward log-return labels for each horizon.

    Parameters
    ----------
    prices:
        DataFrame with a ``close`` column and a DatetimeIndex.
    horizons:
        Tuple of integer look-ahead horizons in trading days.

    Returns
    -------
    DataFrame with columns ``fwd_ret_{h}`` for each horizon.  The last *h*
    rows of each column are NaN because the future close is not yet known.
    """
    close = prices["close"].astype(float)
    labels = pd.DataFrame(index=prices.index)
    for h in horizons:
        # shift(-h) moves the future close to the current row
        fwd_close = close.shift(-h)
        labels[f"fwd_ret_{h}"] = np.log(fwd_close / close)
    labels.index.name = "date"
    return labels


def build_panel_labels(
    panel: pd.DataFrame,
    horizons: tuple[int, ...] = _HORIZONS,
) -> pd.DataFrame:
    """Compute forward-return labels for every ticker in a panel dataset.

    Labels are computed independently per ticker so that the forward close
    for ticker A never contaminates the label for ticker B.

    Parameters
    ----------
    panel:
        DataFrame with a ``(date, ticker)`` MultiIndex and a ``close`` column.
        Typically the output of :func:`data.loader.load_panel`.
    horizons:
        Tuple of integer look-ahead horizons in trading days.

    Returns
    -------
    DataFrame with the same ``(date, ticker)`` MultiIndex and label columns.
    The last *h* rows of each ``fwd_ret_{h}`` column are NaN per ticker.
    """
    tickers = panel.index.get_level_values("ticker").unique()
    pieces: list[pd.DataFrame] = []

    for ticker in tickers:
        prices_t = panel.xs(ticker, level="ticker")
        lbl_t = build_labels(prices_t, horizons=horizons)
        lbl_t.index = pd.MultiIndex.from_arrays(
            [lbl_t.index, [ticker] * len(lbl_t)],
            names=["date", "ticker"],
        )
        pieces.append(lbl_t)

    return pd.concat(pieces).sort_index()


LABEL_COLS = [f"fwd_ret_{h}" for h in _HORIZONS]
