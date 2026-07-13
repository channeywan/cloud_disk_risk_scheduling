from __future__ import annotations

import numpy as np


def predict_baseline(values: np.ndarray, disk_indices: np.ndarray, decisions: np.ndarray,
                     method: str, points_per_day: int = 288) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    d, t = np.asarray(disk_indices), np.asarray(decisions)
    if method == "persistence":
        return values[d, t]
    if method == "seasonal_persistence":
        return values[d, t + 1 - points_per_day]
    if method == "rolling_mean":
        # Vectorize the 12 lag columns; avoids millions of Python-level slices.
        recent = np.stack([values[d, t - offset] for offset in range(11, -1, -1)], axis=1)
        return recent.mean(axis=1)
    raise ValueError(f"unknown baseline: {method}")
