#!/usr/bin/env python
"""Run point forecasting, conformal calibration and risk-aware scheduling."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from conformal import SplitConformal
from data_loader import load_workload
from error_scale import ErrorScaleModel
from feature_engineering import build_tabular
from metrics import (capacity_sensitivity_metrics, interval_metrics,
                     point_metrics, scheduling_metrics)
from models.baselines import predict_baseline
from models.lightgbm_model import LightGBMForecaster
from models.lstm_model import GlobalLSTM, predict_lstm, train_lstm
from normalization import DiskScaler
from scheduling import capacities, schedule_custom_day, select_capacity_ratio
from split import seven_day_splits
from visualization import normalized_max_load, representative_interval
from window_dataset import WorkloadWindowDataset, window_indices


BASELINES = ("persistence", "seasonal_persistence", "rolling_mean")
logger = logging.getLogger("workload_experiment")

# Single standalone bandwidth experiment configuration. There is no metric
# switching layer because this repository intentionally supports bandwidth only.
CONFIG = {
    "data": {
        "trace_path": "/data/tidal_info/cluster_trace_db/all_aligned_trace.pkl",
        "points_per_day": 288, "total_days": 7, "min_scale": 1.0,
        "max_disks": None,
    },
    "sampling": {
        "lstm_train_stride": 6, "lightgbm_train_stride": 1,
        "eval_stride": 1, "lstm_max_train_windows": 2_000_000,
        "random_seed": 42,
    },
    "compute": {
        "feature_workers": 16, "feature_chunk_disks": 256,
        "scheduling_workers": 16, "lstm_dataloader_workers": 8,
    },
    "lightgbm": {
        "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
        "max_depth": -1, "n_jobs": 32,
    },
    "lstm": {
        "hidden_size": 64, "num_layers": 1, "dropout": 0.0,
        "batch_size": 256, "epochs": 20, "learning_rate": 0.001,
        "patience": 3, "pin_memory": True,
    },
    "conformal": {
        "alpha": 0.1,
        "error_scale": {
            "a_min": 1.0, "train_stride": 6, "n_estimators": 200,
            "learning_rate": 0.05, "num_leaves": 31,
            "max_depth": -1, "n_jobs": 32,
        },
    },
    "scheduling": {
        "node_number": 100, "risk_lambda": 0.25,
        "capacity_ratio": 1.2,
        "capacity_candidates": [1.1, 1.2, 1.3, 1.4, 1.5],
        "risk_lambda_candidates": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
    "output_root": "results",
}


def stage(number, total, message):
    """Emit a consistent, coarse-grained experiment progress marker."""
    logger.info("[阶段 %d/%d] %s", number, total, message)


def samples(values, decisions):
    times = np.asarray(list(decisions), dtype=int)
    disks = np.repeat(np.arange(values.shape[0]), len(times))
    tiled_times = np.tile(times, values.shape[0])
    return disks, tiled_times, values[disks, tiled_times + 1]


def train_predict(name, model_values, raw, train_decisions, eval_decisions, cfg, device,
                  fixed_lstm_epochs=None, output_scaler=None):
    eval_disks, eval_times, _ = samples(raw, eval_decisions)
    if name in BASELINES:
        return predict_baseline(raw, eval_disks, eval_times, name), None
    if name == "lightgbm":
        compute = cfg["compute"]
        x_train, y_train, _, _ = build_tabular(model_values, train_decisions,
            stride=cfg["sampling"]["lightgbm_train_stride"],
            workers=compute["feature_workers"], chunk_disks=compute["feature_chunk_disks"])
        x_eval, _, _, _ = build_tabular(model_values, eval_decisions,
            workers=compute["feature_workers"], chunk_disks=compute["feature_chunk_disks"])
        logger.info("表格特征构造完成，开始拟合 LightGBM：train=%s, validation=%s",
                    x_train.shape, x_eval.shape)
        model = LightGBMForecaster(**cfg["lightgbm"]).fit(x_train, y_train)
        return model.predict(x_eval), model
    lc = cfg["lstm"]
    train_idx = window_indices(len(raw), train_decisions, stride=cfg["sampling"]["lstm_train_stride"],
        max_windows=cfg["sampling"]["lstm_max_train_windows"], seed=cfg["sampling"]["random_seed"])
    eval_idx = window_indices(len(raw), eval_decisions)
    train_ds = WorkloadWindowDataset(model_values, train_idx)
    eval_ds = WorkloadWindowDataset(model_values, eval_idx)
    model = GlobalLSTM(lc["hidden_size"], lc["num_layers"], lc["dropout"])
    loader_workers = cfg["compute"]["lstm_dataloader_workers"]
    epochs = fixed_lstm_epochs or lc["epochs"]
    validation_ds = None if fixed_lstm_epochs is not None else eval_ds
    model = train_lstm(model, train_ds, validation_ds, epochs, lc["batch_size"],
                       lc["learning_rate"], lc["patience"], device,
                       loader_workers, lc["pin_memory"])
    model.dataloader_workers = loader_workers
    model.pin_memory = lc["pin_memory"]
    pred, _ = predict_lstm(model, eval_ds, device, loader_workers, lc["pin_memory"])
    if output_scaler is not None:
        pred = output_scaler.inverse(pred, eval_idx[:, 0])
    return pred, model


def predict_fitted(name, model, model_values, raw, eval_decisions, device,
                   output_scaler=None):
    disks, times, _ = samples(raw, eval_decisions)
    if name in BASELINES:
        return predict_baseline(raw, disks, times, name)
    if name == "lightgbm":
        # Fitted prediction is called by the experiment runner, which injects
        # parallel settings on the model for a consistent interface.
        workers = getattr(model, "feature_workers", 1)
        chunk_disks = getattr(model, "feature_chunk_disks", 256)
        x, _, _, _ = build_tabular(model_values, eval_decisions,
                                          workers=workers, chunk_disks=chunk_disks)
        return model.predict(x)
    indices = window_indices(len(raw), eval_decisions)
    pred, _ = predict_lstm(model, WorkloadWindowDataset(model_values, indices), device,
                           getattr(model, "dataloader_workers", 0),
                           getattr(model, "pin_memory", True))
    if output_scaler is not None:
        pred = output_scaler.inverse(pred, indices[:, 0])
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Aligned .pkl, long CSV, or matrix .npz")
    parser.add_argument("--device", default="cuda:0", help="Defaults to cuda:0; cpu is also supported")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started_at = time.monotonic()
    stage(1, 8, "初始化 bandwidth 实验配置")
    cfg = CONFIG
    path = args.data or cfg["data"]["trace_path"]
    if not path:
        raise SystemExit("Set CONFIG['data']['trace_path'] or pass --data")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"GPU {device} requested but CUDA is unavailable")
    p = cfg["data"]["points_per_day"]
    stage(2, 8, f"加载 bandwidth 负载数据（设备：{device}）")
    disk_ids, raw = load_workload(path, p * cfg["data"]["total_days"], cfg["data"]["max_disks"])
    logger.info("数据加载完成：%d 块云盘，每块 %d 个时间点", len(disk_ids), raw.shape[1])
    splits = seven_day_splits(p); out = ROOT / cfg["output_root"] / "bandwidth"
    out.mkdir(parents=True, exist_ok=True)

    # LightGBM uses raw units. Global LSTM uses leakage-safe per-disk Q95 scaling.
    model_values = raw
    lstm_scaler4 = DiskScaler(cfg["data"]["min_scale"]).fit(raw, 4*p)
    lstm_values4 = lstm_scaler4.transform(raw)
    val_decisions = splits["validation"].decisions(raw.shape[1])
    val_d, val_t, val_y = samples(raw, val_decisions)
    peak4 = np.quantile(raw[:, :4*p], .9)
    rows, val_predictions, internal_models = [], {}, {}
    methods = (*BASELINES, "lightgbm", "lstm")
    stage(3, 8, "在第 5 天比较五种点预测方法")
    for method in methods:
        logger.info("开始训练/评估点预测方法：%s", method)
        method_values = lstm_values4 if method == "lstm" else model_values
        method_scaler = lstm_scaler4 if method == "lstm" else None
        pred, trained = train_predict(method, method_values, raw,
                                      splits["internal_train"].decisions(4*p),
                                      val_decisions, cfg, device,
                                      output_scaler=method_scaler)
        internal_models[method] = trained
        val_predictions[method] = pred
        rows.append({"workload": "bandwidth", "model": method,
                     **point_metrics(val_y, pred, disk_ids[val_d], peak4)})
        logger.info("方法 %s 完成，验证 WAPE=%.6f", method, rows[-1]["WAPE"])
    forecast_table = pd.DataFrame(rows).sort_values("WAPE")
    forecast_table.to_csv(out / "forecasting_metrics.csv", index=False)
    selected = str(forecast_table.iloc[0].model)
    logger.info("点预测器选择完成：%s", selected)

    # Refit both learned predictors; all three downstream predictors are kept.
    stage(4, 8, "使用第 1–5 天重新训练 LightGBM 和 LSTM")
    cal_decisions = splits["calibration"].decisions(raw.shape[1]); test_decisions = splits["test"].decisions(raw.shape[1])
    train_final = splits["final_train"].decisions(5*p)
    lstm_scaler5 = DiskScaler(cfg["data"]["min_scale"]).fit(raw, 5*p)
    lstm_values5 = lstm_scaler5.transform(raw)
    cal_d, cal_t, cal_y = samples(raw, cal_decisions); test_d, test_t, test_y = samples(raw, test_decisions)
    downstream = ("persistence", "lightgbm", "lstm")
    cal_predictions = {
        "persistence": predict_baseline(raw, cal_d, cal_t, "persistence")
    }
    test_predictions = {
        "persistence": predict_baseline(raw, test_d, test_t, "persistence")
    }
    final_models = {}
    for model_name in ("lightgbm", "lstm"):
        logger.info("最终重训模型：%s", model_name)
        fixed_epochs = (getattr(internal_models["lstm"], "best_epoch", cfg["lstm"]["epochs"])
                        if model_name == "lstm" else None)
        final_values = lstm_values5 if model_name == "lstm" else model_values
        final_scaler = lstm_scaler5 if model_name == "lstm" else None
        cal_prediction, model = train_predict(
            model_name, final_values, raw, train_final, cal_decisions,
            cfg, device, fixed_epochs, final_scaler)
        if model_name == "lightgbm":
            model.feature_workers = cfg["compute"]["feature_workers"]
            model.feature_chunk_disks = cfg["compute"]["feature_chunk_disks"]
        cal_predictions[model_name] = cal_prediction
        test_predictions[model_name] = predict_fitted(
            model_name, model, final_values, raw, test_decisions, device,
            final_scaler)
        final_models[model_name] = model

    cc = cfg["conformal"]
    scale_cfg = dict(cc["error_scale"])
    a_min = scale_cfg.pop("a_min")
    scale_stride = scale_cfg.pop("train_stride")
    compute = cfg["compute"]
    stage(5, 8, "训练预测器专属误差尺度模型并校准 Conformal 区间")
    logger.info("构造误差尺度模型特征：第 5 天训练，第 6 天校准，第 7 天测试")
    x_scale_train, _, scale_train_d, scale_train_t = build_tabular(
        raw, val_decisions, p, scale_stride, compute["feature_workers"],
        compute["feature_chunk_disks"])
    x_scale_cal, _, scale_cal_d, scale_cal_t = build_tabular(
        raw, cal_decisions, p, 1, compute["feature_workers"],
        compute["feature_chunk_disks"])
    x_scale_test, _, scale_test_d, scale_test_t = build_tabular(
        raw, test_decisions, p, 1, compute["feature_workers"],
        compute["feature_chunk_disks"])
    if not (np.array_equal(scale_cal_d, cal_d) and np.array_equal(scale_cal_t, cal_t)):
        raise AssertionError("calibration error-scale features are misaligned")
    if not (np.array_equal(scale_test_d, test_d) and np.array_equal(scale_test_t, test_t)):
        raise AssertionError("test error-scale features are misaligned")
    val_time_count = len(list(val_decisions))
    scale_train_positions = np.concatenate([
        disk * val_time_count + np.arange(0, val_time_count, scale_stride)
        for disk in range(len(disk_ids))
    ])
    if not (np.array_equal(scale_train_d, val_d[scale_train_positions]) and
            np.array_equal(scale_train_t, val_t[scale_train_positions])):
        raise AssertionError("training error-scale features are misaligned")
    interval_rows, adaptive_intervals = [], {}
    for predictor in downstream:
        logger.info("训练 %s 的误差尺度模型 g(X)", predictor)
        validation_error = np.abs(val_y - val_predictions[predictor])
        scale_model = ErrorScaleModel(a_min=a_min, **scale_cfg).fit(
            x_scale_train, validation_error[scale_train_positions])
        cal_scale = scale_model.predict(x_scale_cal)
        test_scale = scale_model.predict(x_scale_test)
        calibrator = SplitConformal(cc["alpha"], adaptive=True).fit(
            cal_y, cal_predictions[predictor], cal_scale)
        lower, upper = calibrator.interval(test_predictions[predictor], test_scale)
        adaptive_intervals[predictor] = (lower, upper)
        interval_rows.append({
            "workload": "bandwidth", "predictor": predictor,
            "method": "learned_error_scale", "q": calibrator.q,
            "mean_cal_scale": float(np.mean(cal_scale)),
            "mean_test_scale": float(np.mean(test_scale)),
            **interval_metrics(test_y, lower, upper, np.ptp(cal_y), np.quantile(cal_y, .9))})
    pd.DataFrame(interval_rows).to_csv(out / "interval_metrics.csv", index=False)

    def td(x): return np.asarray(x).reshape(len(disk_ids), -1).T
    test_current = raw[:, np.asarray(list(test_decisions))].T
    truth_matrix = td(test_y)
    prediction_matrices = {name: td(value) for name, value in test_predictions.items()}
    upper_matrices = {name: td(adaptive_intervals[name][1]) for name in downstream}
    scheduling_cfg = cfg["scheduling"]
    scheduling_workers = cfg["compute"]["scheduling_workers"]
    # Capacity selection uses day-5 validation only and is frozen before test.
    stage(6, 8, "使用第 5 天验证集确定容量比例")
    val_truth = td(val_y)
    val_current = raw[:, np.asarray(list(val_decisions))].T
    val_scores = {
        "Current-load greedy": val_current,
        "Persistence-point greedy": td(val_predictions["persistence"]),
        "LSTM-point greedy": td(val_predictions["lstm"]),
        "lightgbm-point greedy": td(val_predictions["lightgbm"]),
        "Oracle greedy": val_truth,
    }
    val_loads = schedule_custom_day(disk_ids, val_truth, val_scores,
                                    scheduling_cfg["node_number"], scheduling_workers)
    rho = select_capacity_ratio(val_loads, val_truth, scheduling_cfg["node_number"],
                                scheduling_cfg["capacity_candidates"], scheduling_cfg["capacity_ratio"])
    logger.info("容量比例选择完成：rho=%.2f", rho)
    stage(7, 8, "执行第 7 天八种调度策略、λ 消融与 ρ 敏感性实验")
    risk_lambda = scheduling_cfg["risk_lambda"]
    scores = {
        "Current-load greedy": test_current,
        "LSTM-point greedy": prediction_matrices["lstm"],
        "Persistence-risk greedy": prediction_matrices["persistence"] + risk_lambda * (
            upper_matrices["persistence"] - prediction_matrices["persistence"]),
        "LSTM-risk greedy": prediction_matrices["lstm"] + risk_lambda * (
            upper_matrices["lstm"] - prediction_matrices["lstm"]),
        "Oracle greedy": truth_matrix,
        "lightgbm-point greedy": prediction_matrices["lightgbm"],
        "lightgbm-risk greedy": prediction_matrices["lightgbm"] + risk_lambda * (
            upper_matrices["lightgbm"] - prediction_matrices["lightgbm"]),
    }
    loads = schedule_custom_day(disk_ids, truth_matrix, scores,
                                scheduling_cfg["node_number"], scheduling_workers)
    caps = capacities(truth_matrix, scheduling_cfg["node_number"], rho)
    pd.DataFrame([{"strategy": name, **scheduling_metrics(load, caps)} for name, load in loads.items()]).to_csv(out / "scheduling_metrics.csv", index=False)
    ablation = []
    for lambda_value in scheduling_cfg["risk_lambda_candidates"]:
        logger.info("执行风险系数消融：lambda=%.2f", lambda_value)
        lambda_scores = {
            f"{name}-risk greedy": prediction_matrices[name] + lambda_value * (
                upper_matrices[name] - prediction_matrices[name])
            for name in downstream
        }
        lambda_loads = schedule_custom_day(
            disk_ids, truth_matrix, lambda_scores, scheduling_cfg["node_number"],
            scheduling_workers, include_hash=False)
        for strategy, load in lambda_loads.items():
            ablation.append({"predictor": strategy.removesuffix("-risk greedy"),
                             "lambda": lambda_value, **scheduling_metrics(load, caps)})
    pd.DataFrame(ablation).to_csv(out / "risk_ablation.csv", index=False)
    sensitivity = []
    for ratio in scheduling_cfg["capacity_candidates"]:
        ratio_caps = capacities(truth_matrix, scheduling_cfg["node_number"], ratio)
        for strategy, load in loads.items():
            sensitivity.append({"rho": ratio, "strategy": strategy,
                                **capacity_sensitivity_metrics(load, ratio_caps)})
    pd.DataFrame(sensitivity).to_csv(out / "capacity_sensitivity.csv", index=False)
    metadata = {"validation_best_model": selected,
                "downstream_predictors": list(downstream),
                "scheduling_strategies": list(loads),
                "workload": "bandwidth", "disk_count": len(disk_ids),
                "validation_decisions": len(list(val_decisions)), "calibration_decisions": len(list(cal_decisions)),
                "test_decisions": len(list(test_decisions)), "capacity_ratio": rho,
                "capacity_sensitivity": scheduling_cfg["capacity_candidates"],
                "risk_lambda_candidates": scheduling_cfg["risk_lambda_candidates"],
                "conformal_scale": "predictor-specific learned absolute error",
                "error_scale_train_day": 5,
                "error_scale_a_min": a_min,
                "device": device,
                "model_input_scale": {
                    "lightgbm": "raw",
                    "lstm": "per-disk Q95",
                    "persistence": "raw",
                }}
    stage(8, 8, f"保存结果表、元数据和可视化至 {out}")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lstm_lower, lstm_upper = adaptive_intervals["lstm"]
    representative_interval(truth_matrix[:, 0], prediction_matrices["lstm"][:, 0],
                            td(lstm_lower)[:, 0], td(lstm_upper)[:, 0],
                            out / "representative_interval.png")
    normalized_max_load(loads, caps, out / "normalized_max_load.png")
    logger.info("全部实验完成，总耗时 %.1f 分钟", (time.monotonic() - started_at) / 60)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
