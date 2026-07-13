"""Aligned three-day LightGBM features and conformal scale features.

The functions preserve the unit of ``values``. The main experiment passes raw
IOPS/bandwidth, so current, lag, rolling, seasonal features and targets are raw.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import numpy as np


logger = logging.getLogger(__name__)
_WORK_VALUES = None
_WORK_TIMES = None
_WORK_POINTS_PER_DAY = 288


FEATURE_NAMES = [
    "current_load", "lag_1", "lag_3", "lag_6", "lag_12", "rolling_mean_3", "rolling_mean_12",
    "rolling_std_12", "rolling_max_12", "rolling_mean_72", "rolling_std_72",
    "diff_1", "seasonal_lag_1d", "seasonal_lag_2d", "seasonal_lag_3d",
    "same_time_mean", "same_time_std", "same_time_max_3d", "seasonal_deviation_1d",
    "time_sin", "time_cos",
]


def _row_features(series: np.ndarray, t: int, points_per_day: int) -> list[float]:
    seasonal = np.array([series[t + 1 - points_per_day * d] for d in (1, 2, 3)])
    slot = t % points_per_day
    return [
        series[t], series[t], series[t - 2], series[t - 5], series[t - 11],
        series[t - 2:t + 1].mean(), series[t - 11:t + 1].mean(),
        series[t - 11:t + 1].std(ddof=0), series[t - 11:t + 1].max(),
        series[t - 71:t + 1].mean(), series[t - 71:t + 1].std(ddof=0),
        series[t] - series[t - 1], *seasonal,
        seasonal.mean(), seasonal.std(ddof=0), seasonal.max(), seasonal[0] - seasonal.mean(),
        np.sin(2 * np.pi * slot / points_per_day), np.cos(2 * np.pi * slot / points_per_day),
    ]


def build_current_only(values: np.ndarray, decisions, stride: int = 1):
    """Build the unambiguous feature values[i,t] -> values[i,t+1]."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(list(decisions), dtype=np.int32)[::stride]
    times = times[times + 1 < values.shape[1]]
    disks = np.repeat(np.arange(values.shape[0], dtype=np.int32), len(times))
    out_times = np.tile(times, values.shape[0])
    current = values[disks, out_times].astype(np.float32, copy=False)
    target = values[disks, out_times + 1].astype(np.float32, copy=False)
    return current[:, None], target, disks, out_times


def _tabular_chunk(bounds):
    """Build one consecutive disk block; globals are inherited with fork."""
    start, end = bounds
    rows, labels, disks, out_times = [], [], [], []
    for disk in range(start, end):
        series = _WORK_VALUES[disk]
        for t in _WORK_TIMES:
            rows.append(_row_features(series, int(t), _WORK_POINTS_PER_DAY))
            labels.append(series[t + 1])
            disks.append(disk)
            out_times.append(t)
    return (np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.float32),
            np.asarray(disks, dtype=np.int32), np.asarray(out_times, dtype=np.int32))


def _disk_chunks(disk_count: int, chunk_disks: int):
    return [(start, min(start + chunk_disks, disk_count))
            for start in range(0, disk_count, chunk_disks)]


def build_tabular(values: np.ndarray, decisions, points_per_day: int = 288,
                   stride: int = 1, workers: int = 1,
                   chunk_disks: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X, next-step target, disk index and decision index."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(list(decisions), dtype=int)[::stride]
    times = times[times >= 3 * points_per_day - 1]
    times = times[times + 1 < values.shape[1]]
    global _WORK_VALUES, _WORK_TIMES, _WORK_POINTS_PER_DAY
    _WORK_VALUES, _WORK_TIMES, _WORK_POINTS_PER_DAY = values, times, points_per_day
    chunks = _disk_chunks(values.shape[0], max(1, chunk_disks))
    workers = max(1, min(int(workers), len(chunks)))
    logger.info("构造表格特征：%d 个样本，%d 个进程，%d 个云盘/块",
                values.shape[0] * len(times), workers, chunk_disks)
    if workers == 1:
        results = [_tabular_chunk(chunk) for chunk in chunks]
    else:
        # Linux fork lets workers read the workload matrix copy-on-write, avoiding
        # one full serialization per process. Returned blocks stay bounded.
        context = mp.get_context("fork")
        results = []
        with context.Pool(workers) as pool:
            for completed, result in enumerate(pool.imap(_tabular_chunk, chunks), 1):
                results.append(result)
                if completed == 1 or completed == len(chunks) or completed % 10 == 0:
                    logger.info("表格特征进度：%d/%d 块", completed, len(chunks))
    _WORK_VALUES = _WORK_TIMES = None
    if not results:
        empty = np.empty(0, dtype=np.float32)
        return np.empty((0, len(FEATURE_NAMES)), np.float32), empty, empty.astype(np.int32), empty.astype(np.int32)
    return tuple(np.concatenate(parts, axis=0) for parts in zip(*results))


def adaptive_scale(values: np.ndarray, disk_indices: np.ndarray, decisions: np.ndarray,
                   points_per_day: int = 288, gamma: float = 1.0,
                   beta: float = 0.1, epsilon: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    disk_indices, decisions = np.asarray(disk_indices), np.asarray(decisions)
    # Vectorize across samples in bounded columns; this is substantially faster
    # than multiprocessing for the small 12-value calculation.
    recent = np.stack([values[disk_indices, decisions - offset] for offset in range(11, -1, -1)], axis=1)
    seasonal = np.stack([values[disk_indices, decisions + 1 - points_per_day * day]
                         for day in (1, 2, 3)], axis=1)
    result = recent.std(axis=1) + gamma * seasonal.std(axis=1) + beta * recent.mean(axis=1) + epsilon
    return np.maximum(result, epsilon)
