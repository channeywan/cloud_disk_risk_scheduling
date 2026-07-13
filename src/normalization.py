"""Per-disk robust scaling fitted only on the supplied training interval."""
from __future__ import annotations

import numpy as np


class DiskScaler:
    def __init__(self, min_scale: float = 1.0):
        self.min_scale = float(min_scale)
        self.scales: np.ndarray | None = None

    def fit(self, values: np.ndarray, train_end: int) -> "DiskScaler":
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or not 0 < train_end <= values.shape[1]:
            raise ValueError("values must be [disk, time] and train_end must be valid")
        self.scales = np.maximum(np.nanquantile(values[:, :train_end], 0.95, axis=1), self.min_scale)
        self.scales = np.where(np.isfinite(self.scales), self.scales, self.min_scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.scales is None:
            raise RuntimeError("DiskScaler is not fitted")
        return np.asarray(values, dtype=float) / self.scales[:, None]

    def inverse(self, values: np.ndarray, disk_indices: np.ndarray | None = None) -> np.ndarray:
        if self.scales is None:
            raise RuntimeError("DiskScaler is not fitted")
        scales = self.scales if disk_indices is None else self.scales[np.asarray(disk_indices)]
        array = np.asarray(values, dtype=float)
        return array * scales if array.ndim == 1 else array * scales[:, None]
