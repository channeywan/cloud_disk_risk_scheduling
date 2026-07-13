"""Predictor-specific absolute-error scale models for conformal intervals."""
from __future__ import annotations

import numpy as np

from models.lightgbm_model import LightGBMForecaster


def apply_scale_floor(predicted_error, a_min: float) -> np.ndarray:
    if a_min <= 0:
        raise ValueError("a_min must be positive")
    values = np.asarray(predicted_error, dtype=float)
    return np.maximum(values, float(a_min))


class ErrorScaleModel:
    """Learn g(X) -> |y - y_hat| and return max(g(X), a_min)."""

    def __init__(self, a_min: float = 1.0, **lightgbm_params):
        self.a_min = float(a_min)
        params = {"objective": "regression_l1", **lightgbm_params}
        self.model = LightGBMForecaster(**params)

    def fit(self, features, absolute_errors) -> "ErrorScaleModel":
        target = np.maximum(0.0, np.asarray(absolute_errors, dtype=float))
        self.model.fit(features, target)
        return self

    def predict(self, features) -> np.ndarray:
        return apply_scale_floor(self.model.predict(features), self.a_min)
