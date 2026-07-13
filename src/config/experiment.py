"""Load the single standalone bandwidth experiment configuration."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    metric = str(config["target_metric"]).lower()
    if metric not in {"iops", "bandwidth"}:
        raise ValueError("target_metric must be 'iops' or 'bandwidth'")
    config["target_metric"] = metric
    return config
