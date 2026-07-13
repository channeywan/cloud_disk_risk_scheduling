"""Core publication-ready plots (matplotlib is imported lazily)."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def representative_interval(truth, prediction, lower, upper, path):
    plt = _plt(); x = np.arange(len(truth))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, truth, label="true", lw=1.2); ax.plot(x, prediction, label="prediction", lw=1)
    ax.fill_between(x, lower, upper, alpha=.25, label="90% interval")
    ax.set(xlabel="5-minute decision", ylabel="load"); ax.legend(); fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def normalized_max_load(loads_by_strategy, capacities, path):
    plt = _plt(); fig, ax = plt.subplots(figsize=(10, 4))
    for name, loads in loads_by_strategy.items():
        ax.plot(np.max(loads, axis=1) / capacities, label=name, lw=.9)
    ax.axhline(1, color="black", ls="--", lw=.8); ax.set(xlabel="decision", ylabel="MaxLoad / C(t)")
    ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)

