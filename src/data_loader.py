"""Canonical workload matrix loader built on the existing trace artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _bandwidth_column(frame: pd.DataFrame) -> str:
    for name in ("bandwidth", "Bandwidth"):
        if name in frame.columns:
            return name
    raise ValueError("trace has no bandwidth column")


def traces_to_matrix(
    traces: Mapping[Any, Any], expected_points: int = 2016,
    max_disks: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert nested cluster/disk mappings or a flat disk mapping to [D,T]."""
    flat: list[tuple[str, Any]] = []
    for outer_id, value in traces.items():
        if isinstance(value, pd.DataFrame):
            flat.append((str(outer_id), value))
        elif isinstance(value, Mapping):
            flat.extend((str(disk_id), trace) for disk_id, trace in value.items())
    flat.sort(key=lambda pair: pair[0])
    if max_disks is not None:
        flat = flat[:max_disks]
    ids, rows = [], []
    for disk_id, trace in flat:
        if isinstance(trace, pd.DataFrame):
            column = _bandwidth_column(trace)
            row = trace[column].to_numpy(dtype=float)
        else:
            array = np.asarray(trace, dtype=float)
            row = array[:, 0] if array.ndim == 2 else array
        if len(row) >= expected_points and np.all(np.isfinite(row[:expected_points])):
            ids.append(disk_id)
            rows.append(row[:expected_points])
    if not rows:
        raise ValueError(f"no complete {expected_points}-point disk traces found")
    return np.asarray(ids, dtype=str), np.stack(rows)


def load_workload(path: str | Path, expected_points: int = 2016,
                  max_disks: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        values = np.asarray(data["bandwidth"], dtype=float)
        ids = np.asarray(data["disk_id"] if "disk_id" in data else np.arange(len(values)), dtype=str)
        return ids[:max_disks], values[:max_disks, :expected_points]
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        required = {"disk_id", "timestamp", "bandwidth"}
        lower = {str(c).lower(): c for c in frame.columns}
        if not required.issubset(lower):
            raise ValueError(f"CSV must contain {sorted(required)}")
        frame = frame.rename(columns={lower[k]: k for k in required}).sort_values(["disk_id", "timestamp"])
        traces = {disk: group for disk, group in frame.groupby("disk_id", sort=True)}
        return traces_to_matrix(traces, expected_points, max_disks)
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("loading .pkl artifacts requires joblib") from exc
    return traces_to_matrix(joblib.load(path), expected_points, max_disks)
