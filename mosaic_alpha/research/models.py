"""Baseline model: Ridge regression with standard scaling.

Keeping the model thin and explicit here.  The pipeline is:
    1. Fit a StandardScaler on the training features.
    2. Fit a Ridge regressor on the scaled features.
    3. Return raw predictions (not clipped or transformed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """Return an unfitted sklearn Pipeline: scaler → Ridge."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
        ]
    )


def fit_predict(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Fit *pipeline* on training data and return predictions on test data.

    Parameters
    ----------
    pipeline:
        An unfitted (or previously fitted) sklearn pipeline.
    X_train, y_train:
        Training features and labels.
    X_test:
        Test features.

    Returns
    -------
    1-D numpy array of predictions aligned to *X_test*.
    """
    pipeline.fit(X_train.values, y_train.values)
    return pipeline.predict(X_test.values)
