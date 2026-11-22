"""Stateless, fixed distance kernels for standardized 596-feature rows.

The caller fits and applies its TRAIN-only scaler. This module learns nothing,
uses no labels or query-batch statistics, and never changes either input.
Distances are accumulated per query row without a BLAS norm/dot shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


KINDS = ("rbf", "laplacian", "rational_quadratic")


@dataclass(frozen=True)
class _Configuration:
    version: str = "distance_kernels_v1"
    features: int = 596


_CONFIG = _Configuration()


def _kind(kind):
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError(f"Distance kernel kind must be one of {KINDS}")
    return kind


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def distance_kernel_rule(kind):
    """Return a fresh complete JSON rule, including its canonical SHA-256."""
    kind = _kind(kind)
    formula = {
        "rbf": "exp(-sum_j((x_j-y_j)^2)/596)",
        "laplacian": "exp(-sum_j(abs(x_j-y_j))/596)",
        "rational_quadratic": "reciprocal(1+sum_j((x_j-y_j)^2)/596)",
    }[kind]
    rule = {
        "schema_version": 1, "version": _CONFIG.version, "kind": kind,
        "source_module": "src.distance_kernels_v1",
        "input_shapes": [["N", _CONFIG.features], ["M", _CONFIG.features]],
        "input_dtypes": ["float32", "float64"],
        "input_validation": "nonempty finite 2D inputs; exactly 596 coordinates",
        "input_conversion": "independent float64 C-order copies of both inputs",
        "output_shape": ["N", "M"], "output_dtype": "float64", "output_order": "C",
        "coordinate_order": "caller supplies the same original 596-feature order for X and Y",
        "preprocessing": "caller-owned TRAIN-only StandardScaler; no scaling in this helper",
        "formula": formula, "dimension_divisor": float(_CONFIG.features),
        "gamma": 1.0 / _CONFIG.features if kind != "rational_quadratic" else None,
        "alpha": 1.0 if kind == "rational_quadratic" else None,
        "length_scale_squared": _CONFIG.features / 2.0 if kind == "rational_quadratic" else None,
        "difference": "for each query row: numpy.subtract(row[None,:], Y_float64_C)",
        "distance_reduction": (
            "numpy.sum(numpy.abs(diff),axis=1,dtype=float64)"
            if kind == "laplacian" else
            "numpy.einsum('ij,ij->i',diff,diff,optimize=False)"),
        "row_policy": "same per-row arithmetic for fit Gram, single query, batch, and reordered rows",
        "blas_norm_shortcut": False, "learned_parameters": False,
        "query_statistics": False, "input_mutation": False,
        "overflow_policy": "ValueError on subtraction/distance overflow or any nonfinite intermediate/output",
        "underflow_policy": "float64 underflow may round to zero; exponential zero is allowed without a floor",
        "output_range": [0.0, 1.0], "diagonal_adjustment": False,
        "jitter": False, "clipping": False, "renormalization": False,
        "kernel_is_class_probability": False,
        "cross_environment_bitwise_guarantee": False,
        "json_hash_encoding": "UTF-8 json.dumps(ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False); exclude rule_sha256",
    }
    rule["rule_sha256"] = hashlib.sha256(_json_text(rule).encode("utf-8")).hexdigest()
    return rule


def verify_distance_kernel_rule(rule, kind):
    """Reject any rule/configuration drift, even if its digest was recomputed."""
    expected = distance_kernel_rule(kind)
    if not isinstance(rule, dict):
        raise ValueError("Distance kernel rule must be the complete JSON object")
    try:
        matches = _json_text(rule) == _json_text(expected)
    except (TypeError, ValueError) as error:
        raise ValueError("Distance kernel rule must contain finite JSON values") from error
    if not matches:
        raise ValueError("Distance kernel rule or immutable numerical configuration changed")
    return True


def _features(values, name):
    try:
        source = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a nonempty finite float32/float64 (N,596) array") from error
    if (source.ndim != 2 or source.shape[1] != _CONFIG.features or not len(source)
            or source.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or not np.isfinite(source).all()):
        raise ValueError(f"{name} must be a nonempty finite float32/float64 (N,596) array")
    return np.array(source, dtype=np.float64, order="C", copy=True)


def distance_kernel(X, Y, kind):
    """Compute a fixed Gram/cross-kernel matrix without fitting or adaptation."""
    kind = _kind(kind)
    queries, reference = _features(X, "X"), _features(Y, "Y")
    result = np.empty((len(queries), len(reference)), dtype=np.float64, order="C")
    for index, row in enumerate(queries):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                difference = np.subtract(row[None, :], reference)
                if not np.isfinite(difference).all():
                    raise ValueError("Distance kernel subtraction produced a nonfinite value")
                if kind == "laplacian":
                    distance = np.sum(np.abs(difference), axis=1, dtype=np.float64)
                else:
                    distance = np.einsum("ij,ij->i", difference, difference, optimize=False)
                if not np.isfinite(distance).all() or np.any(distance < 0):
                    raise ValueError("Distance kernel distance overflow or invalid value")
                scaled = distance / float(_CONFIG.features)
                result[index] = (np.reciprocal(1.0 + scaled)
                    if kind == "rational_quadratic" else np.exp(-scaled))
        except FloatingPointError as error:
            raise ValueError("Distance kernel arithmetic overflow or invalid intermediate") from error
    if (not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1)):
        raise ValueError("Distance kernel output must be finite and in [0,1]")
    return result
