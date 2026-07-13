"""Leakage-safe seven-day split definitions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeSplit:
    train_end: int
    decision_start: int
    decision_end: int  # exclusive; labels are at decision index + 1

    def decisions(self, available_points: int) -> range:
        end = min(self.decision_end, available_points - 1)
        return range(self.decision_start, max(self.decision_start, end))


def seven_day_splits(points_per_day: int = 288) -> dict[str, TimeSplit]:
    p = points_per_day
    return {
        "internal_train": TimeSplit(4 * p, 3 * p - 1, 4 * p - 1),
        "validation": TimeSplit(4 * p, 4 * p, 5 * p - 1),
        "final_train": TimeSplit(5 * p, 3 * p - 1, 5 * p - 1),
        "calibration": TimeSplit(5 * p, 5 * p, 6 * p - 1),
        "test": TimeSplit(5 * p, 6 * p, 7 * p - 1),
    }

