"""Lazy LSTM windows: only (disk, end time) indices are materialized."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def window_indices(disk_count: int, decisions, window: int = 864, stride: int = 1,
                   max_windows: int | None = None, seed: int = 42) -> np.ndarray:
    times = np.asarray(list(decisions), dtype=np.int32)[::stride]
    times = times[times >= window - 1]
    pairs = np.stack(np.meshgrid(np.arange(disk_count, dtype=np.int32), times, indexing="ij"), -1).reshape(-1, 2)
    if max_windows and len(pairs) > max_windows:
        # Stratified ordering covers disks and time slots before deterministic sampling.
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(pairs), max_windows, replace=False))
        pairs = pairs[selected]
    return pairs


class WorkloadWindowDataset(Dataset):
    def __init__(self, workload_values: np.ndarray, indices: np.ndarray,
                 points_per_day: int = 288, window: int = 864):
        """Lazily slice workload windows and append cyclic time channels.

        The main experiment supplies per-disk Q95-normalized values to LSTM.
        """
        self.values = np.asarray(workload_values, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.points_per_day = points_per_day
        self.window = window
        slots = np.arange(self.values.shape[1]) % self.points_per_day
        self.time_sin = np.sin(2 * np.pi * slots / self.points_per_day).astype(np.float32)
        self.time_cos = np.cos(2 * np.pi * slots / self.points_per_day).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        disk, end = self.indices[index]
        start = end - self.window + 1
        load = self.values[disk, start:end + 1]
        x = np.stack([load, self.time_sin[start:end + 1],
                      self.time_cos[start:end + 1]], axis=1)
        return torch.from_numpy(x), torch.tensor(self.values[disk, end + 1]), disk, end
