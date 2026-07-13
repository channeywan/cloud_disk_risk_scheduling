"""Finite-sample split conformal intervals."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float = 0.1) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        raise ValueError("calibration scores are empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    k = min(len(scores), math.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.partition(scores, k - 1)[k - 1])


@dataclass
class SplitConformal:
    alpha: float = 0.1
    adaptive: bool = False
    q: float | None = None

    def fit(self, truth: np.ndarray, prediction: np.ndarray,
            scale: np.ndarray | None = None) -> "SplitConformal":
        residual = np.abs(np.asarray(truth) - np.asarray(prediction))
        if self.adaptive:
            if scale is None or np.any(np.asarray(scale) <= 0):
                raise ValueError("positive scale is required for adaptive conformal")
            residual = residual / np.asarray(scale)
        self.q = conformal_quantile(residual, self.alpha)
        return self

    def interval(self, prediction: np.ndarray, scale: np.ndarray | None = None):
        if self.q is None:
            raise RuntimeError("conformal calibrator is not fitted")
        radius = self.q
        if self.adaptive:
            if scale is None:
                raise ValueError("scale is required for adaptive conformal")
            radius = radius * np.asarray(scale)
        prediction = np.asarray(prediction, dtype=float)
        return np.maximum(0.0, prediction - radius), prediction + radius

