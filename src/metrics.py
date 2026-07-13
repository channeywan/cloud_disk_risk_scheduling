from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-8


def point_metrics(truth, prediction, disk_ids, peak_threshold):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    error = np.abs(truth - prediction)
    frame = pd.DataFrame({"disk": disk_ids, "truth": truth, "error": error})
    per_disk = frame.groupby("disk").apply(
        lambda x: x.error.sum() / (x.truth.abs().sum() + EPS), include_groups=False)
    peak = truth > peak_threshold
    return {
        "MAE": float(error.mean()),
        "WAPE": float(error.sum() / (np.abs(truth).sum() + EPS)),
        "Peak-MAE": float(error[peak].mean()) if peak.any() else float("nan"),
        "MedianDiskWAPE": float(per_disk.median()),
    }


def interval_metrics(truth, lower, upper, calibration_range, peak_threshold):
    truth, lower, upper = map(np.asarray, (truth, lower, upper))
    covered = (lower <= truth) & (truth <= upper)
    peak = truth > peak_threshold
    return {
        "PICP": float(covered.mean()),
        "PINAW": float((upper - lower).mean() / (calibration_range + EPS)),
        "PeakCoverage": float(covered[peak].mean()) if peak.any() else float("nan"),
        "UpperViolationRate": float((truth > upper).mean()),
    }


def scheduling_metrics(loads: np.ndarray, capacities: np.ndarray) -> dict[str, float]:
    loads = np.asarray(loads, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    if loads.ndim != 2 or capacities.shape != (loads.shape[0],):
        raise ValueError("loads must be [time,node], capacities must be [time]")
    maxima = loads.max(axis=1)
    mean = loads.mean(axis=1)
    overloaded = loads > capacities[:, None]
    overload_volume = np.maximum(0, loads - capacities[:, None]).sum()
    return {
        "AvgMaxLoad": float(maxima.mean()),
        "P95MaxLoad": float(np.quantile(maxima, 0.95)),
        "AvgImbalance": float(np.mean(maxima / (mean + EPS))),
        "OverloadEventRate": float(overloaded.mean()),
        "OverloadedTimestampRate": float(overloaded.any(axis=1).mean()),
        "NormalizedOverloadVolume": float(overload_volume / (loads.shape[1] * capacities.sum() + EPS)),
    }


def capacity_sensitivity_metrics(loads: np.ndarray, capacities: np.ndarray) -> dict[str, float]:
    """The three metrics required for synthetic-capacity sensitivity."""
    base = scheduling_metrics(loads, capacities)
    normalized_max = loads.max(axis=1) / (capacities + EPS)
    return {
        "OverloadEventRate": base["OverloadEventRate"],
        "NormalizedOverloadVolume": base["NormalizedOverloadVolume"],
        "P95MaxLoad/C_t": float(np.quantile(normalized_max, 0.95)),
    }
