"""Independent per-timestamp logical placement strategies."""
from __future__ import annotations

import hashlib
import heapq
import logging
import multiprocessing as mp
import numpy as np


logger = logging.getLogger(__name__)
_SCHEDULE_STATE = None
_CUSTOM_SCHEDULE_STATE = None


def stable_hash_assignment(disk_ids, node_number: int) -> np.ndarray:
    return np.asarray([
        int.from_bytes(hashlib.blake2b(str(d).encode(), digest_size=8).digest(), "little") % node_number
        for d in disk_ids
    ], dtype=np.int32)


def greedy_assignment(scores: np.ndarray, node_number: int) -> np.ndarray:
    scores = np.maximum(0.0, np.asarray(scores, dtype=float))
    assignment = np.empty(len(scores), dtype=np.int32)
    heap = [(0.0, node) for node in range(node_number)]
    heapq.heapify(heap)
    # Stable tie breaking makes experiments reproducible.
    for disk in np.argsort(-scores, kind="stable"):
        load, node = heapq.heappop(heap)
        assignment[disk] = node
        heapq.heappush(heap, (load + scores[disk], node))
    return assignment


def node_loads(assignment, truth, node_number):
    return np.bincount(np.asarray(assignment), weights=np.asarray(truth), minlength=node_number)


def _schedule_timestamp(t):
    current, prediction, upper, truth, node_number, risk_lambda, names, hash_assignment = _SCHEDULE_STATE
    scores = {
        "Current-load greedy": current[t],
        "Point-prediction greedy": prediction[t],
        "Risk-aware greedy": prediction[t] + risk_lambda * (upper[t] - prediction[t]),
        "Oracle greedy": truth[t],
    }
    row = {}
    for name in names:
        assignment = hash_assignment if name == "Hash" else greedy_assignment(scores[name], node_number)
        row[name] = node_loads(assignment, truth[t], node_number)
    return row


def _custom_schedule_timestamp(t):
    truth, scores_by_strategy, node_number = _CUSTOM_SCHEDULE_STATE
    return {name: node_loads(greedy_assignment(scores[t], node_number), truth[t], node_number)
            for name, scores in scores_by_strategy.items()}


def schedule_day(disk_ids, current, prediction, upper, truth, node_number=100,
                 risk_lambda=1.0, workers=1, strategies=None):
    """Inputs are [time,disk]; return strategy -> [time,node] true loads."""
    current, prediction, upper, truth = map(np.asarray, (current, prediction, upper, truth))
    if not (current.shape == prediction.shape == upper.shape == truth.shape):
        raise ValueError("all workload arrays must have identical [time,disk] shape")
    all_names = ("Hash", "Current-load greedy", "Point-prediction greedy", "Risk-aware greedy", "Oracle greedy")
    names = tuple(strategies or all_names)
    unknown = set(names) - set(all_names)
    if unknown:
        raise ValueError(f"unknown scheduling strategies: {sorted(unknown)}")
    output = {name: [] for name in names}
    hash_assignment = stable_hash_assignment(disk_ids, node_number)
    global _SCHEDULE_STATE
    _SCHEDULE_STATE = (current, prediction, upper, truth, node_number,
                       risk_lambda, names, hash_assignment)
    workers = max(1, min(int(workers), truth.shape[0]))
    logger.info("执行调度：%d 个时刻，%d 个策略，%d 个进程", truth.shape[0], len(names), workers)
    if workers == 1:
        rows = map(_schedule_timestamp, range(truth.shape[0]))
        for row in rows:
            for name in names:
                output[name].append(row[name])
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            for completed, row in enumerate(pool.imap(_schedule_timestamp, range(truth.shape[0])), 1):
                for name in names:
                    output[name].append(row[name])
                if completed == 1 or completed == truth.shape[0] or completed % 50 == 0:
                    logger.info("调度进度：%d/%d 个时刻", completed, truth.shape[0])
    _SCHEDULE_STATE = None
    return {name: np.stack(rows) for name, rows in output.items()}


def schedule_custom_day(disk_ids, truth, scores_by_strategy, node_number=100,
                        workers=1, include_hash=True):
    """Schedule arbitrary named score matrices while evaluating all with truth."""
    truth = np.asarray(truth, dtype=float)
    scores = {name: np.asarray(value, dtype=float) for name, value in scores_by_strategy.items()}
    if any(value.shape != truth.shape for value in scores.values()):
        raise ValueError("every strategy score must match truth [time,disk]")
    output = {name: [] for name in scores}
    global _CUSTOM_SCHEDULE_STATE
    _CUSTOM_SCHEDULE_STATE = (truth, scores, node_number)
    workers = max(1, min(int(workers), truth.shape[0]))
    logger.info("执行自定义调度：%d 个时刻，%d 个评分策略，%d 个进程",
                truth.shape[0], len(scores), workers)
    if workers == 1:
        iterator = map(_custom_schedule_timestamp, range(truth.shape[0]))
        for row in iterator:
            for name in scores:
                output[name].append(row[name])
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            for completed, row in enumerate(pool.imap(_custom_schedule_timestamp, range(truth.shape[0])), 1):
                for name in scores:
                    output[name].append(row[name])
                if completed == 1 or completed == truth.shape[0] or completed % 50 == 0:
                    logger.info("自定义调度进度：%d/%d", completed, truth.shape[0])
    _CUSTOM_SCHEDULE_STATE = None
    result = {name: np.stack(rows) for name, rows in output.items()}
    if include_hash:
        assignment = stable_hash_assignment(disk_ids, node_number)
        result = {"Hash": np.stack([node_loads(assignment, row, node_number) for row in truth]), **result}
    return result


def capacities(truth: np.ndarray, node_number: int, ratio: float) -> np.ndarray:
    return ratio * np.asarray(truth).sum(axis=1) / node_number


def select_capacity_ratio(validation_loads: dict[str, np.ndarray], validation_truth: np.ndarray,
                          node_number: int, candidates=(1.1, 1.2, 1.3), preferred=1.2) -> float:
    """Choose moderate pressure using validation only, preferring rho=1.2 on ties."""
    primary = [k for k in validation_loads if k != "Oracle greedy"]
    viable = []
    for ratio in candidates:
        cap = capacities(validation_truth, node_number, ratio)
        rates = [np.mean(validation_loads[k] > cap[:, None]) for k in primary]
        # Avoid virtually no overload and near-universal overload.
        if max(rates) >= 0.001 and min(rates) <= 0.5:
            viable.append(float(ratio))
    if preferred in viable:
        return float(preferred)
    return min(viable, key=lambda x: abs(x - preferred)) if viable else float(preferred)
