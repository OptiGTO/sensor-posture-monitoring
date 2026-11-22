"""TRAIN-only PCA replacement and LDA feature augmentation for fixed596 input.

The caller owns StandardScaler. This helper never learns from transform inputs
and makes no probability-calibration or independent-sample claim.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import scipy
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted

from .posture_core import feature_names


VERSION = "learned_projection_v1"
MODES = frozenset({"identity", "pca64", "all_plus_lda6"})
WIDTHS = {"identity": 596, "pca64": 64, "all_plus_lda6": 602}


def _mode(mode):
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("Unknown learned projection mode")
    return mode


def _input(X):
    if hasattr(X, "columns") and list(X.columns) != feature_names():
        raise ValueError("Projection input column names differ from canonical596 features")
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] != 596 or not len(values)
            or values.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or not np.isfinite(values).all()):
        raise ValueError("Projection requires finite nonempty float32/float64 input with shape (N,596)")
    return values


def _labels(y, rows):
    labels = np.asarray(y)
    if (labels.ndim != 1 or len(labels) != rows or labels.dtype.kind not in "iu"
            or not np.array_equal(np.unique(labels), np.arange(7))):
        raise ValueError("Projection audit requires aligned integer labels containing all seven posture IDs")
    return labels.astype(np.int64, copy=True)


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_digest(array):
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256(json.dumps([values.dtype.str, list(values.shape)]).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _names(mode):
    if mode == "identity":
        return feature_names()
    if mode == "pca64":
        return [f"learned_pca_axis_{index + 1}" for index in range(64)]
    return [*feature_names(), *[f"learned_lda_axis_{index + 1}" for index in range(6)]]


def learned_projection_rule(mode):
    mode = _mode(mode)
    rule = {"version": VERSION, "schema_version": 1, "mode": mode,
        "input_features": 596, "output_features": WIDTHS[mode], "input_dtypes": ["float32", "float64"],
        "output_dtype": "same input dtype" if mode == "identity" else "float64",
        "scaler_owner": "caller fits one shared StandardScaler on original TRAIN features",
        "fit_scope": "supplied TRAIN rows only", "labels_used": mode == "all_plus_lda6",
        "identity": "Return validated original ndarray unchanged; no arithmetic or fitted bounds",
        "evaluation_batch_statistics": False, "external_validation_used": False,
        "fallback": "none; insufficient rank or singular covariance fails explicitly",
        "solver": "none" if mode == "identity" else "full SVD" if mode == "pca64" else "eigen",
        "parameters": {} if mode == "identity" else {"n_components": 64, "svd_solver": "full", "whiten": False, "copy": True}
            if mode == "pca64" else {"n_components": 6, "solver": "eigen", "shrinkage": "auto", "priors": None},
        "pca_rank_tolerance": "eps(float64)*max(TRAIN rows,596)*largest singular value; require singular_values[63]>tolerance and N>=65",
        "lda_mask": "Drop only columns constant across all TRAIN rows; retain original column order",
        "lda_priors": "Empirical TRAIN posture frequencies; no sample or subject weighting",
        "lda_sign": "For each of six axes, largest absolute loading positive; ties use first retained input column",
        "linear_transform": "identity passthrough" if mode == "identity" else
            "float64 einsum('ni,ji->nj', X64-TRAIN mean, components, optimize=False)" if mode == "pca64" else
            "float64 C-order selected inputs and components; einsum('ni,ji->nj', X64[:,retained], signed first6 eigen scalings transposed, optimize=False); concatenate original X64 first",
        "numerical_environment": "Equal or near-equal eigenvalues may change basis across libraries; require pinned environment and refit/reload checks"}
    rule["rule_sha256"] = _json_digest(rule)
    return rule


def _backend(mode):
    if mode == "pca64":
        return PCA(n_components=64, svd_solver="full", whiten=False, copy=True)
    if mode == "all_plus_lda6":
        return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=6, priors=None)
    return None


def _signs(components):
    anchors = np.argmax(np.abs(components), axis=1)
    return np.where(components[np.arange(len(components)), anchors] < 0., -1., 1.)


def _state(model):
    mode = _mode(model.mode_)
    state = {"version": VERSION, "mode": mode, "rows": model.n_samples_seen_, "input_features": model.n_features_in_,
        "output_features": model.n_features_out_, "training_dtype": model.training_dtype_,
        "training_input_sha256": model.training_input_sha256_, "training_labels_sha256": model.training_labels_sha256_,
        "classes": _array_digest(model.classes_), "class_counts": _array_digest(model.class_counts_),
        "retained_indices": _array_digest(model.retained_indices_), "feature_names_in": model.feature_names_in_.tolist(),
        "feature_names_out": model.feature_names_out_.tolist(), "rule": model.rule_, "rank_tolerance": model.rank_tolerance_,
        "components": None if model.components_ is None else _array_digest(model.components_),
        "signs": None if model.signs_ is None else _array_digest(model.signs_)}
    if mode == "identity":
        state["backend"] = None
    else:
        estimator = model.estimator_
        fields = ("components_", "mean_", "explained_variance_", "explained_variance_ratio_", "singular_values_") if mode == "pca64" else (
            "classes_", "means_", "priors_", "covariance_", "scalings_", "explained_variance_ratio_", "coef_", "intercept_")
        state["backend"] = {"class": type(estimator).__name__, "params": estimator.get_params(deep=False),
            "n_features": int(estimator.n_features_in_), **{name: _array_digest(getattr(estimator, name)) for name in fields},
            "array_schema": {name: {"shape": list(getattr(estimator, name).shape),
                "dtype": str(getattr(estimator, name).dtype)} for name in fields}}
        if mode == "pca64":
            state["backend"].update(n_samples=int(estimator.n_samples_), n_components=int(estimator.n_components_),
                noise_variance=float(estimator.noise_variance_), fitted_solver=estimator._fit_svd_solver)
        else:
            state["backend"].update(max_components=estimator._max_components, output_features=estimator._n_features_out)
    return state


def learned_projection_state_sha256(model):
    if type(model) is not LearnedProjectionV1:
        raise ValueError("Expected the exact versioned learned projection type")
    check_is_fitted(model, "mode_")
    try:
        return _json_digest(_state(model))
    except (AttributeError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Incomplete learned projection state") from error


def _audit(model):
    state = _state(model)
    return {"version": VERSION, "mode": model.mode_, "training_windows": model.n_samples_seen_,
        "input_features": 596, "output_features": model.n_features_out_, "training_dtype": model.training_dtype_,
        "training_input_sha256": model.training_input_sha256_, "training_labels_sha256": model.training_labels_sha256_,
        "class_counts": model.class_counts_.tolist(), "classes": model.classes_.tolist(),
        "retained_indices": model.retained_indices_.tolist(), "feature_names_in": model.feature_names_in_.tolist(),
        "feature_names_out": model.feature_names_out_.tolist(), "rank_tolerance": model.rank_tolerance_,
        "rule": deepcopy(model.rule_), "backend_state": deepcopy(state["backend"]),
        "components_sha256": state["components"], "signs_sha256": state["signs"],
        "learned_state_sha256": _json_digest(state), "external_validation_used": False,
        "labels_used": model.mode_ == "all_plus_lda6", "evaluation_batch_statistics": False,
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "scipy_version": scipy.__version__}


class LearnedProjectionV1(TransformerMixin, BaseEstimator):
    def __init__(self, mode="identity"):
        self.mode = mode

    def fit(self, X, y):
        mode = _mode(self.mode)
        values = _input(X)
        labels = _labels(y, len(values))
        counts = np.bincount(labels, minlength=7)
        estimator = _backend(mode)
        retained, components, signs, tolerance = np.arange(596), None, None, None
        if mode != "identity":
            numeric = np.array(values, dtype=np.float64, order="C", copy=True)
            if mode == "pca64":
                if len(values) < 65:
                    raise ValueError("PCA64 requires at least65 TRAIN rows and centered numerical rank64")
                estimator.fit(numeric)
                tolerance = float(np.finfo(np.float64).eps * max(numeric.shape) * estimator.singular_values_[0])
                if (not np.isfinite(estimator.singular_values_).all()
                        or estimator.singular_values_[-1] <= tolerance):
                    raise ValueError("PCA64 TRAIN numerical rank is below64; no dimension fallback")
                components = estimator.components_.copy()
            else:
                retained = np.flatnonzero(np.any(numeric != numeric[0], axis=0))
                if min(counts) < 2 or len(values) <= 7 or len(retained) < 6:
                    raise ValueError("LDA6 requires two TRAIN rows per class and six nonconstant columns")
                try:
                    estimator.fit(numeric[:, retained], labels)
                except np.linalg.LinAlgError as error:
                    raise ValueError("TRAIN within-class covariance is singular; no LDA fallback") from error
                raw = estimator.scalings_[:, :6].T
                if raw.shape != (6, len(retained)) or not np.isfinite(raw).all():
                    raise ValueError("LDA6 produced invalid discriminant coordinates")
                signs = _signs(raw)
                components = np.ascontiguousarray(raw * signs[:, None])
        self.mode_, self.estimator_ = mode, estimator
        self.n_samples_seen_, self.n_features_in_, self.n_features_out_ = len(values), 596, WIDTHS[mode]
        self.training_dtype_ = str(values.dtype)
        self.training_input_sha256_, self.training_labels_sha256_ = _array_digest(values), _array_digest(labels)
        self.classes_, self.class_counts_ = np.arange(7), counts
        self.retained_indices_, self.components_, self.signs_, self.rank_tolerance_ = retained, components, signs, tolerance
        self.feature_names_in_, self.feature_names_out_ = np.asarray(feature_names()), np.asarray(_names(mode))
        self.rule_ = learned_projection_rule(mode)
        self.training_details_ = _audit(self)
        self.state_sha256_ = learned_projection_state_sha256(self)
        verify_learned_projection(self, mode)
        return self

    def transform(self, X):
        _inference_guard(self)
        values = _input(X)
        if self.mode_ == "identity":
            return values
        numeric = np.array(values, dtype=np.float64, order="C", copy=True)
        if self.mode_ == "pca64":
            result = np.einsum("ni,ji->nj", numeric - self.estimator_.mean_, self.components_, optimize=False)
        else:
            selected = np.ascontiguousarray(numeric[:, self.retained_indices_])
            axes = np.einsum("ni,ji->nj", selected, self.components_, optimize=False)
            result = np.concatenate([numeric, axes], axis=1)
        if result.shape != (len(values), self.n_features_out_) or not np.isfinite(result).all():
            raise ValueError("Learned projection produced invalid output")
        return result

    def get_feature_names_out(self, input_features=None):
        _inference_guard(self)
        if input_features is not None and not np.array_equal(np.asarray(input_features), self.feature_names_in_):
            raise ValueError("Projection input feature names differ")
        return self.feature_names_out_.copy()


def _inference_guard(model):
    """Check executable shapes without hashing unused dense covariance per query.

    Full state/audit integrity is checked at fit completion and by the explicit
    verifier used by checkpoint finalization and explicit audits. Inference does not
    recompute those hashes or learn from a query batch.
    """
    mode = _mode(model.mode)
    check_is_fitted(model, "mode_")
    try:
        if (type(model) is not LearnedProjectionV1 or model.mode_ != mode
                or model.n_features_in_ != 596 or model.n_features_out_ != WIDTHS[mode]
                or not np.array_equal(model.feature_names_in_, feature_names())
                or not np.array_equal(model.feature_names_out_, _names(mode))):
            raise ValueError("Learned projection constructor or input/output schema changed")
        if mode == "identity":
            if model.estimator_ is not None or model.components_ is not None:
                raise ValueError("Identity projection acquired learned coordinates")
            return
        expected = _backend(mode)
        if (type(model.estimator_) is not type(expected)
                or model.estimator_.get_params(deep=False) != expected.get_params(deep=False)
                or model.components_.dtype != np.dtype(np.float64)
                or not model.components_.flags.c_contiguous
                or not np.isfinite(model.components_).all()):
            raise ValueError("Learned projection inference solver, C-order layout or coordinates changed")
        if mode == "pca64":
            if (model.components_.shape != (64, 596) or model.estimator_.mean_.shape != (596,)
                    or not np.isfinite(model.estimator_.mean_).all()):
                raise ValueError("PCA64 inference state has invalid dimensions or mean")
        else:
            indices = model.retained_indices_
            if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) < 6
                    or np.any(indices < 0) or np.any(indices >= 596)
                    or not np.all(indices[1:] > indices[:-1])
                    or model.components_.shape != (6, len(indices))):
                raise ValueError("LDA6 inference state has invalid columns or coordinates")
    except (AttributeError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Incomplete learned projection inference state") from error


def verify_learned_projection(model, mode):
    mode = _mode(mode)
    if type(model) is not LearnedProjectionV1 or model.mode != mode:
        raise ValueError("Learned projection class or constructor mode changed")
    check_is_fitted(model, "mode_")
    try:
        if (model.mode_ != mode or model.n_features_in_ != 596 or model.n_features_out_ != WIDTHS[mode]
                or type(model.n_samples_seen_) is not int or model.n_samples_seen_ < 7
                or model.training_dtype_ not in ("float32", "float64")
                or not np.array_equal(model.classes_, np.arange(7)) or model.class_counts_.shape != (7,)
                or model.class_counts_.dtype.kind not in "iu" or (model.class_counts_ < 1).any()
                or int(model.class_counts_.sum()) != model.n_samples_seen_
                or not np.array_equal(model.feature_names_in_, feature_names())
                or not np.array_equal(model.feature_names_out_, _names(mode))
                or model.rule_ != learned_projection_rule(mode)):
            raise ValueError("Learned projection schema or training audit changed")
        for digest in (model.training_input_sha256_, model.training_labels_sha256_):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Learned projection training digest is invalid")
        indices = model.retained_indices_
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                or np.any(indices < 0) or np.any(indices >= 596) or not np.all(indices[1:] > indices[:-1])):
            raise ValueError("Learned projection retained feature order changed")
        if mode == "identity":
            if (model.estimator_ is not None or model.components_ is not None or model.signs_ is not None
                    or model.rank_tolerance_ is not None or not np.array_equal(indices, np.arange(596))):
                raise ValueError("Identity projection must not learn or apply coordinates")
        else:
            estimator, expected = model.estimator_, _backend(mode)
            if (type(estimator) is not type(expected) or estimator.get_params(deep=False) != expected.get_params(deep=False)
                    or estimator.n_features_in_ != len(indices) or model.components_.dtype != np.dtype(np.float64)
                    or not model.components_.flags.c_contiguous
                    or not np.isfinite(model.components_).all()):
                raise ValueError("Projection solver, C-order layout or coordinate state changed")
            if mode == "pca64":
                tolerance = float(np.finfo(np.float64).eps * max(model.n_samples_seen_, 596) * estimator.singular_values_[0])
                if (model.n_samples_seen_ < 65 or not np.array_equal(indices, np.arange(596))
                        or model.signs_ is not None or estimator._fit_svd_solver != "full"
                        or estimator.n_samples_ != model.n_samples_seen_ or estimator.n_components_ != 64
                        or estimator.singular_values_.shape != (64,) or estimator.singular_values_[-1] <= tolerance
                        or model.rank_tolerance_ != tolerance or model.components_.shape != (64, 596)
                        or not np.array_equal(model.components_, estimator.components_)):
                    raise ValueError("PCA64 rank or fitted coordinate contract changed")
            else:
                raw = estimator.scalings_[:, :6].T
                signs = _signs(raw)
                if (len(indices) < 6 or min(model.class_counts_) < 2 or model.rank_tolerance_ is not None
                        or estimator._max_components != 6 or estimator._n_features_out != 6
                        or not np.array_equal(estimator.classes_, np.arange(7))
                        or not np.array_equal(estimator.priors_, model.class_counts_.astype(float) / model.n_samples_seen_)
                        or not np.array_equal(model.signs_, signs) or model.components_.shape != (6, len(indices))
                        or not np.array_equal(model.components_, raw * signs[:, None])):
                    raise ValueError("LDA6 class priors, signs or fitted axes changed")
        if model.training_details_ != _audit(model) or model.state_sha256_ != learned_projection_state_sha256(model):
            raise ValueError("Learned projection complete fitted state or metadata changed")
    except (AttributeError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Incomplete learned projection state") from error
    return True
