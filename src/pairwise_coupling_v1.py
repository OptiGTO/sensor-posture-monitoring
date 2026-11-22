"""Stateless seven-class Wu/Lin/Weng Method2 coupling by direct KKT solve.

Implemented from the equations in Wu, Lin and Weng (2004), section4.1,
equations(19), (22), (23), rather than copied from LIBSVM source. This couples
pair estimates; it does not calibrate forests or learn from an input batch.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json

import numpy as np


PAIR_INDICES = tuple(combinations(range(7), 2))


@dataclass(frozen=True)
class _Configuration:
    version: str = "pairwise_coupling_v1"
    classes: int = 7
    epsilon: float = 1e-7
    tolerance: float = 1e-12
    condition_limit: float = 1e8


_CONFIG = _Configuration()


def pairwise_coupling_rule():
    """Return a fresh complete JSON rule; there are no tunable call arguments."""
    rule = {"algorithm_version": _CONFIG.version, "method": "Wu-Lin-Weng Method2 direct equality KKT",
        "class_ids": list(range(7)), "pair_indices": [list(pair) for pair in PAIR_INDICES],
        "input_shape": ["N", 21], "input_orientation": "column(i,j),i<j,contains P(i|i,j); reverse=1-forward",
        "dtype": "float64", "memory_order": "C", "epsilon": _CONFIG.epsilon,
        "input_validation": "finite real values in[0,1]; nonempty N by21; no input mutation",
        "clipping": "clip only forward r_ij to[epsilon,1-epsilon], then construct r_ji=1-r_ij",
        "objective": "sum_i<j (r_ji*p_i-r_ij*p_j)^2",
        "Q_diagonal": "Q_ii=sum_j!=i r_ji^2", "Q_off_diagonal": "Q_ij=-r_ji*r_ij",
        "solver": "per-row numpy.linalg.solve([[Q,ones],[ones.T,0]],[zeros7,1]); fixed8x8",
        "condition_matrix": "full8x8 KKT, never Q alone", "condition_norm": "2-norm via numpy.linalg.cond",
        "condition_limit": _CONFIG.condition_limit, "tolerance": _CONFIG.tolerance,
        "backward_residual": "norm_inf(A*z-b)/(norm_inf(A)*norm_inf(z)+norm_inf(b))",
        "stationarity_residual": "norm_inf(Q*p-(p.T*Q*p)*ones)/max(1,norm_inf(Q)*norm_inf(p)+abs(p.T*Q*p))",
        "simplex_check": "strictly0<p_i<1; abs(sum(p)-1)<=tolerance before and after normalization",
        "normalization": "after all raw-solve checks, divide p by its sum once; check stationarity again",
        "uniform_tie": "only exact all21 forward inputs==0.5 map to equal np.full(7,1/7) after raw-solve checks; no near-tie handling",
        "failure": "raise ValueError on bad input, solve/condition failure, nonpositive solution or failed residual; no fallback or output clipping",
        "iteration": "none", "batch_statistics": False, "calibrated_probability_claim": False,
        "reference": "https://www.jmlr.org/papers/volume5/wu04a/wu04a.pdf#page=7",
        "reference_sections": "4.1 equations19/22/23,Theorems2/3; direct solver differs from finite-iteration LIBSVM"}
    rule["rule_sha256"] = hashlib.sha256(json.dumps(rule, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return rule


def verify_pairwise_coupling_rule(rule):
    if not isinstance(rule, dict) or rule != pairwise_coupling_rule():
        raise ValueError("Pairwise coupling rule or immutable numerical configuration changed")
    return True


def _input(values):
    source = np.asarray(values)
    if (source.ndim != 2 or source.shape[1] != 21 or not len(source)
            or source.dtype.kind not in "fiu" or not np.isfinite(source).all()
            or np.any(source < 0) or np.any(source > 1)):
        raise ValueError("Pairwise coupling requires finite nonempty real probabilities in[0,1] with shape(N,21)")
    return np.array(source, dtype=np.float64, order="C", copy=True)


def _q_matrix(forward):
    q = np.zeros((7, 7), dtype=np.float64, order="C")
    for value, (i, j) in zip(forward, PAIR_INDICES):
        reverse = 1. - value
        q[i, i] += reverse * reverse
        q[j, j] += value * value
        q[i, j] = q[j, i] = -reverse * value
    return q


def _simplex(probability):
    if (not np.isfinite(probability).all() or np.any(probability <= 0) or np.any(probability >= 1)
            or abs(float(probability.sum()) - 1.) > _CONFIG.tolerance):
        raise ValueError("Pairwise KKT result is not a strictly positive numerical simplex")


def _solve_row(source):
    forward = np.clip(source, _CONFIG.epsilon, 1. - _CONFIG.epsilon)
    q = _q_matrix(forward)
    matrix = np.zeros((8, 8), dtype=np.float64, order="C")
    matrix[:7, :7], matrix[:7, 7], matrix[7, :7] = q, 1., 1.
    rhs = np.zeros(8, dtype=np.float64)
    rhs[7] = 1.
    try:
        condition = float(np.linalg.cond(matrix))
        if not np.isfinite(condition) or condition > _CONFIG.condition_limit:
            raise ValueError("Pairwise KKT condition exceeds the fixed numerical limit")
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError("Pairwise KKT linear solve failed; no fallback is permitted") from error
    if (not isinstance(solution, np.ndarray) or solution.shape != (8,) or solution.dtype != np.dtype(np.float64)
            or not np.isfinite(solution).all()):
        raise ValueError("Pairwise KKT solve produced an invalid finite float64 vector")
    probability = solution[:7].copy()
    _simplex(probability)
    residual = float(np.max(np.abs(matrix @ solution - rhs)))
    scale = float(np.linalg.norm(matrix, ord=np.inf) * np.linalg.norm(solution, ord=np.inf) + 1.)
    if not np.isfinite(residual) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Pairwise KKT backward residual or its scale is not finite")
    backward = residual / scale
    if not np.isfinite(backward) or backward > _CONFIG.tolerance:
        raise ValueError("Pairwise KKT scaled backward residual exceeds tolerance")
    probability /= probability.sum()
    uniform = bool(np.all(source == .5))
    if uniform:
        probability = np.full(7, 1. / 7., dtype=np.float64)
    _simplex(probability)
    gradient = q @ probability
    objective_from_q = float(probability @ gradient)
    stationarity = float(np.max(np.abs(gradient - objective_from_q))) / max(1.,
        float(np.linalg.norm(q, ord=np.inf) * np.linalg.norm(probability, ord=np.inf) + abs(objective_from_q)))
    if not np.isfinite(stationarity) or stationarity > _CONFIG.tolerance:
        raise ValueError("Pairwise normalized stationarity residual exceeds tolerance")
    objective = sum(float(((1. - value) * probability[i] - value * probability[j]) ** 2)
        for value, (i, j) in zip(forward, PAIR_INDICES))
    return probability, {"condition_KKT": condition, "backward_residual": backward,
        "stationarity_residual": stationarity, "pairwise_squared_residual": objective,
        "sum_error": abs(float(probability.sum()) - 1.), "minimum_probability": float(probability.min()),
        "clipped_pairs": int(np.count_nonzero(forward != source)), "exact_uniform_tie": uniform}


def couple_pairwise(pair_probability, *, return_diagnostics=False):
    """Couple N rows of21 lexicographic P(i|i,j) estimates into N by7 values.

    Diagnostics are returned only, never stored in mutable module/model state.
    Each row uses the same8x8 solve regardless of batch size or row order.
    """
    if type(return_diagnostics) is not bool:
        raise ValueError("return_diagnostics must be boolean")
    values = _input(pair_probability)
    result = np.empty((len(values), 7), dtype=np.float64, order="C")
    rows = []
    for index, row in enumerate(values):
        try:
            result[index], detail = _solve_row(row)
        except ValueError as error:
            raise ValueError(f"Pairwise coupling row{index}: {error}") from error
        if return_diagnostics:
            rows.append(detail)
    if not return_diagnostics:
        return result
    return result, {"algorithm_version": _CONFIG.version, "rule_sha256": pairwise_coupling_rule()["rule_sha256"],
        "row_count": len(rows), "maximum_condition_KKT": max(row["condition_KKT"] for row in rows),
        "maximum_backward_residual": max(row["backward_residual"] for row in rows),
        "maximum_stationarity_residual": max(row["stationarity_residual"] for row in rows),
        "maximum_pairwise_squared_residual": max(row["pairwise_squared_residual"] for row in rows),
        "minimum_probability": min(row["minimum_probability"] for row in rows),
        "clipped_pair_count": sum(row["clipped_pairs"] for row in rows),
        "exact_uniform_rows": sum(row["exact_uniform_tie"] for row in rows), "rows": rows}
