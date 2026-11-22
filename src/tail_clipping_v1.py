"""Versioned TRAIN-fitted quantile clipping of the fixed 596 feature columns.

Clipping is a model hypothesis, not a declaration that extreme values are
measurement errors. Feature names and their canonical order belong to the
enclosing pipeline; this transformer preserves every positional column.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


VERSION = "tail_clipping_v1"
FEATURE_COUNT = 596


def _fraction(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value not in (0, .005, .01)):
        raise ValueError("tail_fraction must be a finite numeric value exactly 0, 0.005 or 0.01")
    return float(value)


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _array_digest(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(json.dumps([values.dtype.str, list(values.shape)]).encode("utf-8"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _features(X):
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] != FEATURE_COUNT or not len(values)
            or values.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or not np.isfinite(values).all()):
        raise ValueError("Tail clipping requires finite nonempty float32 or float64 features with shape (N, 596)")
    return values


def tail_clipping_rule(tail_fraction):
    """Return fresh, JSON-serializable exact transformation metadata."""
    fraction = _fraction(tail_fraction)
    rule = {"schema_version": 1, "version": VERSION, "tail_fraction": fraction,
        "input_features": FEATURE_COUNT, "output_features": FEATURE_COUNT,
        "input_dtypes": ["float32", "float64"], "column_order": "unchanged positional 0..595; canonical names checked by wrapper",
        "quantiles": None if fraction == 0 else [fraction, 1. - fraction],
        "fit_scope": "TRAIN rows only; labels unused", "quantile_computation_dtype": "float64",
        "quantile_method": "numpy.quantile(method='linear', axis=0)",
        "identity": "fraction 0 returns validated input directly; no min/max clipping or arithmetic",
        "positive_transform": "numpy.clip with stored per-feature bounds, then cast to input dtype before downstream StandardScaler",
        "constant_or_tied_bounds": "equal lower/upper bounds retained; positive clipping maps that coordinate to the bound",
        "refit_on_transform": False, "evaluation_batch_statistics": False, "uses_labels": False,
        "row_count_and_order": "unchanged", "training_input_mutated": False,
        "interpretation": "extreme values may contain genuine posture information; no data-error or units claim"}
    rule["rule_sha256"] = _json_digest(rule)
    return rule


def _state(clipper):
    if type(clipper) is not TailClipperV1:
        raise ValueError("Expected exact TailClipperV1 fitted transformer")
    check_is_fitted(clipper, ("tail_fraction_", "n_features_in_", "n_features_out_", "n_samples_seen_",
        "training_dtype_", "training_input_sha256_", "lower_bounds_", "upper_bounds_", "rule_"))
    fraction = _fraction(clipper.tail_fraction_)
    if (type(clipper.n_features_in_) is not int or clipper.n_features_in_ != FEATURE_COUNT
            or type(clipper.n_features_out_) is not int or clipper.n_features_out_ != FEATURE_COUNT
            or type(clipper.n_samples_seen_) is not int or clipper.n_samples_seen_ < 1
            or clipper.training_dtype_ not in ("float32", "float64")
            or not isinstance(clipper.training_input_sha256_, str) or len(clipper.training_input_sha256_) != 64
            or any(character not in "0123456789abcdef" for character in clipper.training_input_sha256_)):
        raise ValueError("Invalid fitted tail-clipping feature/count/dtype/input-hash state")
    lower, upper = clipper.lower_bounds_, clipper.upper_bounds_
    if fraction == 0:
        if lower is not None or upper is not None:
            raise ValueError("Identity clipping must not store min/max or quantile bounds")
    elif (not isinstance(lower, np.ndarray) or not isinstance(upper, np.ndarray)
            or lower.shape != (FEATURE_COUNT,) or upper.shape != (FEATURE_COUNT,)
            or lower.dtype != np.float64 or upper.dtype != np.float64
            or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper)):
        raise ValueError("Invalid fitted tail-clipping bounds")
    return {"version": VERSION, "configured_tail_fraction": _fraction(clipper.tail_fraction),
        "fitted_tail_fraction": fraction, "input_features": clipper.n_features_in_,
        "output_features": clipper.n_features_out_, "training_windows": clipper.n_samples_seen_,
        "training_dtype": clipper.training_dtype_, "training_input_sha256": clipper.training_input_sha256_,
        "lower_bounds_sha256": None if lower is None else _array_digest(lower),
        "upper_bounds_sha256": None if upper is None else _array_digest(upper), "rule": deepcopy(clipper.rule_)}


def tail_clipping_state_sha256(clipper):
    """Hash fitted numeric state, constructor fraction, TRAIN scope and rule."""
    return _json_digest(_state(clipper))


def _training_details(clipper):
    return {"version": VERSION, "tail_fraction": float(clipper.tail_fraction_),
        "training_windows": clipper.n_samples_seen_, "input_features": clipper.n_features_in_,
        "output_features": clipper.n_features_out_, "training_dtype": clipper.training_dtype_,
        "training_input_sha256": clipper.training_input_sha256_,
        "lower_bounds": None if clipper.lower_bounds_ is None else clipper.lower_bounds_.tolist(),
        "upper_bounds": None if clipper.upper_bounds_ is None else clipper.upper_bounds_.tolist(),
        "bounds_dtype": None if clipper.lower_bounds_ is None else "float64", "rule": deepcopy(clipper.rule_),
        "external_validation_used": False, "labels_used": False, "evaluation_batch_statistics": False}


def verify_tail_clipper(clipper, tail_fraction):
    """Verify complete stored bounds/recipe/audit identity without training data."""
    fraction = _fraction(tail_fraction)
    state = _state(clipper)
    if (state["configured_tail_fraction"] != fraction or state["fitted_tail_fraction"] != fraction
            or clipper.rule_ != tail_clipping_rule(fraction)
            or getattr(clipper, "training_details_", None) != _training_details(clipper)
            or getattr(clipper, "state_sha256_", None) != _json_digest(state)):
        raise ValueError("Tail-clipping parameters, fitted state, rule or training audit changed")
    return True


class TailClipperV1(TransformerMixin, BaseEstimator):
    """Fit fixed symmetric TRAIN cutpoints, preserving query row order/dtype."""

    def __init__(self, tail_fraction=0.0):
        self.tail_fraction = tail_fraction

    def fit(self, X, y=None):
        fraction = _fraction(self.tail_fraction)
        values = _features(X)
        lower = upper = None
        if fraction > 0:
            with np.errstate(over="ignore", invalid="ignore"):
                bounds = np.quantile(values.astype(np.float64), [fraction, 1. - fraction], axis=0, method="linear")
            if not np.isfinite(bounds).all():
                raise ValueError("Tail quantile computation exceeded the finite float64 range")
            lower, upper = bounds[0].copy(), bounds[1].copy()
        self.tail_fraction_ = fraction
        self.n_features_in_ = FEATURE_COUNT
        self.n_features_out_ = FEATURE_COUNT
        self.n_samples_seen_ = int(len(values))
        self.training_dtype_ = str(values.dtype)
        self.training_input_sha256_ = _array_digest(values)
        self.lower_bounds_, self.upper_bounds_ = lower, upper
        self.rule_ = tail_clipping_rule(fraction)
        self.training_details_ = _training_details(self)
        self.state_sha256_ = tail_clipping_state_sha256(self)
        return self

    def transform(self, X):
        verify_tail_clipper(self, self.tail_fraction)
        values = _features(X)
        if self.tail_fraction_ == 0:
            return values
        result = np.clip(values, self.lower_bounds_, self.upper_bounds_).astype(values.dtype, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("Tail clipping produced values outside the finite input dtype range")
        return result
