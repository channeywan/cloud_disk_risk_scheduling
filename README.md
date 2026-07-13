# Minimal bandwidth experiment

This directory contains only the code required by the complete bandwidth
forecasting, conformal calibration and risk-aware scheduling pipeline.

Run in the `tidal2` environment from this directory:

```bash
python main.py
```

Defaults:

- config: `configs/bandwidth.yaml`
- device: `cuda:0`
- input: `/data/tidal_info/cluster_trace_db/all_aligned_trace.pkl`
- output: `results/bandwidth/`

Optional overrides remain available, for example `python main.py --device cpu`
or `python main.py --data /path/to/traces.npz`.
