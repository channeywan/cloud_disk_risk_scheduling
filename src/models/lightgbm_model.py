from __future__ import annotations

import numpy as np


class LightGBMForecaster:
    def __init__(self, **params):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("LightGBM is optional; install it with `pip install lightgbm`") from exc
        defaults = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                        max_depth=-1, n_jobs=-1, random_state=42)
        defaults.update(params)
        self.model = LGBMRegressor(**defaults)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LightGBMForecaster":
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, self.model.predict(x))

