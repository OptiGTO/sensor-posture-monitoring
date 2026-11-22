"""Seven class-conditional diagonal Gaussian mixtures on standardized features.

The caller owns the training-only standardizer and split protocol. Each mixture
sees only the supplied training rows of its posture class; class priors remain
uniform regardless of class frequencies. Failed convergence is an error.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import scipy
from scipy.special import logsumexp
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits


CLASS_COUNT = 7
FEATURE_COUNT = 596
DEFAULT_PARAMS = {"n_components": 2, "reg_covar": .1, "n_init": 3,
    "max_iter": 200, "tol": 1e-3, "random_state": 20260921}
STATE_ARRAYS = ("weights_", "means_", "covariances_", "precisions_", "precisions_cholesky_")


def _parameters(params):
    if not isinstance(params, dict) or set(params) - set(DEFAULT_PARAMS):
        raise ValueError("Unknown class-conditional mixture parameters")
    value = {**DEFAULT_PARAMS, **deepcopy(params)}
    for key in ("n_components", "n_init", "max_iter"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("reg_covar", "tol"):
        number = value[key]
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not np.isfinite(number) or number < 0 or key == "tol" and number == 0):
            raise ValueError(f"{key} must be finite and {'positive' if key == 'tol' else 'nonnegative'}")
        value[key] = float(number)
    seed = value["random_state"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("random_state must be a nonnegative uint32 integer")
    return value


def _features(X):
    array = np.asarray(X)
    if (array.ndim != 2 or array.shape[1] != FEATURE_COUNT or not len(array)
            or array.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or not np.isfinite(array).all()):
        raise ValueError("Expected nonempty finite float32/float64 standardized features with shape (N, 596)")
    return np.array(array, dtype=np.float64, order="C", copy=True)


def _array_digest(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([array.dtype.str, list(array.shape)]).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_digest(estimator):
    fields = {key: _array_digest(getattr(estimator, key)) for key in STATE_ARRAYS}
    fields.update(converged=bool(estimator.converged_), n_iter=int(estimator.n_iter_),
        lower_bound=float(estimator.lower_bound_), lower_bounds=list(map(float, estimator.lower_bounds_)))
    return hashlib.sha256(json.dumps(fields, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _backend(params, class_id):
    return GaussianMixture(**{**params, "random_state": (params["random_state"] + class_id) % 2**32},
        covariance_type="diag", init_params="kmeans", warm_start=False,
        weights_init=None, means_init=None, precisions_init=None, verbose=0, verbose_interval=10)


def _backend_audit(estimator, class_id, count):
    return {"class_id": class_id, "training_rows": int(count),
        "parameters": estimator.get_params(deep=False), "converged": bool(estimator.converged_),
        "n_iter": int(estimator.n_iter_), "lower_bound": float(estimator.lower_bound_),
        "lower_bounds": list(map(float, estimator.lower_bounds_)), "state_sha256": _state_digest(estimator)}


class ClassConditionalMixtureClassifier(ClassifierMixin, BaseEstimator):
    """Fit one converged diagonal mixture per class and use uniform class priors."""

    def __init__(self, n_components=2, reg_covar=.1, n_init=3, max_iter=200, tol=1e-3, random_state=20260921):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        params, values, labels = _parameters(self.get_params(deep=False)), _features(X), np.asarray(y)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu"
                or not np.array_equal(np.unique(labels), np.arange(CLASS_COUNT))):
            raise ValueError("Mixture training requires matching integer labels containing all seven posture classes 0..6")
        labels = labels.astype(np.int64, copy=True)
        counts = np.bincount(labels, minlength=CLASS_COUNT)
        if (counts < max(2, params["n_components"])).any():
            raise ValueError("Every class requires at least two rows and at least n_components training rows")
        estimators, audits, training_hashes = [], [], []
        with threadpool_limits(limits=1):
            for class_id in range(CLASS_COUNT):
                class_values = values[labels == class_id].copy()
                estimator = _backend(params, class_id).fit(class_values)
                if not bool(estimator.converged_):
                    raise ValueError(f"Class {class_id} Gaussian mixture failed to converge within max_iter={params['max_iter']}; no fallback")
                if (not np.isfinite(estimator.lower_bound_) or any(
                        not np.isfinite(getattr(estimator, key)).all() for key in STATE_ARRAYS)):
                    raise ValueError(f"Class {class_id} mixture produced nonfinite fitted density parameters")
                estimators.append(estimator)
                audits.append(_backend_audit(estimator, class_id, counts[class_id]))
                training_hashes.append(_array_digest(class_values))
        self.fitted_params_, self.estimators_ = params, estimators
        self.classes_, self.class_counts_ = np.arange(CLASS_COUNT), counts.astype(np.int64)
        self.class_priors_ = np.repeat(1 / CLASS_COUNT, CLASS_COUNT)
        self.n_features_in_, self.n_samples_seen_ = FEATURE_COUNT, len(values)
        self.effective_seeds_ = [(params["random_state"] + index) % 2**32 for index in range(CLASS_COUNT)]
        self.class_training_sha256_ = training_hashes
        self.state_sha256_ = [entry["state_sha256"] for entry in audits]
        self.training_details_ = {"algorithm": "class_conditional_diagonal_gaussian_mixture_v1",
            "parameters": deepcopy(params), "class_order": self.classes_.tolist(),
            "class_counts": self.class_counts_.tolist(), "class_priors": self.class_priors_.tolist(),
            "class_training_sha256": training_hashes.copy(), "effective_seeds": self.effective_seeds_.copy(),
            "training_rows": len(values), "input_features": FEATURE_COUNT,
            "backend_dtype": "float64", "threads": 1, "external_validation_used": False,
            "class_scope": "each mixture is fitted only to supplied training rows with its class label",
            "prior_policy": "uniform seven-class prior, independent of training class frequencies",
            "posterior": "score_samples + log(1/7); subtract row maximum, then float64 logsumexp normalization",
            "convergence_policy": "every selected GaussianMixture fit must converge; failure raises ValueError without fallback",
            "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "class_models": audits}
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "fitted_params_")
        if _parameters(self.get_params(deep=False)) != self.fitted_params_:
            raise ValueError("Mixture parameters changed after fitting; refit before prediction")
        values = _features(X)
        with threadpool_limits(limits=1), np.errstate(over="ignore", invalid="ignore", under="ignore"):
            log_density = np.column_stack([estimator.score_samples(values) for estimator in self.estimators_])
            if log_density.shape != (len(values), CLASS_COUNT) or not np.isfinite(log_density).all():
                raise ValueError("Mixture density calculation exceeded the finite float64 range")
            log_posterior = log_density + np.log(self.class_priors_)
            shifted = log_posterior - log_posterior.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted - logsumexp(shifted, axis=1, keepdims=True))
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(1), 1., atol=1e-12, rtol=0)):
            raise ValueError("Mixture produced invalid normalized posture probabilities")
        return probabilities

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def verify_class_conditional_mixture_identity(model, expected_params):
    """Reject changed recipes, class scopes, backend seeds or learned densities."""
    params = _parameters(expected_params)
    if (type(model) is not ClassConditionalMixtureClassifier
            or _parameters(model.get_params(deep=False)) != params or getattr(model, "fitted_params_", None) != params
            or getattr(model, "n_features_in_", None) != FEATURE_COUNT
            or not np.array_equal(getattr(model, "classes_", ()), np.arange(CLASS_COUNT))
            or not np.array_equal(getattr(model, "class_priors_", ()), np.repeat(1 / CLASS_COUNT, CLASS_COUNT))):
        raise ValueError("Class-conditional mixture recipe, features or class contract changed")
    counts = np.asarray(getattr(model, "class_counts_", ()))
    seeds = [(params["random_state"] + index) % 2**32 for index in range(CLASS_COUNT)]
    details = getattr(model, "training_details_", {})
    estimators = getattr(model, "estimators_", ())
    training_hashes = getattr(model, "class_training_sha256_", ())
    if (counts.shape != (CLASS_COUNT,) or counts.dtype.kind not in "iu" or (counts < max(2, params["n_components"])).any()
            or getattr(model, "n_samples_seen_", None) != int(counts.sum()) or len(estimators) != CLASS_COUNT
            or getattr(model, "effective_seeds_", None) != seeds or len(training_hashes) != CLASS_COUNT
            or any(not isinstance(item, str) or len(item) != 64 or any(c not in "0123456789abcdef" for c in item) for item in training_hashes)
            or details.get("parameters") != params or details.get("class_counts") != counts.tolist()
            or details.get("class_order") != list(range(CLASS_COUNT)) or details.get("class_priors") != model.class_priors_.tolist()
            or details.get("training_rows") != int(counts.sum()) or details.get("effective_seeds") != seeds
            or details.get("class_training_sha256") != training_hashes or details.get("input_features") != FEATURE_COUNT
            or details.get("backend_dtype") != "float64" or details.get("threads") != 1
            or details.get("external_validation_used") is not False
            or details.get("sklearn_version") != sklearn.__version__ or details.get("numpy_version") != np.__version__
            or details.get("scipy_version") != scipy.__version__):
        raise ValueError("Class-conditional mixture training audit or library versions changed")
    expected_audits = []
    try:
        for class_id, estimator in enumerate(estimators):
            expected = _backend(params, class_id)
            if (type(estimator) is not GaussianMixture or estimator.get_params(deep=False) != expected.get_params(deep=False)
                    or estimator.n_features_in_ != FEATURE_COUNT or not bool(estimator.converged_)
                    or not 1 <= estimator.n_iter_ <= params["max_iter"] or not np.isfinite(estimator.lower_bound_)
                    or estimator.weights_.shape != (params["n_components"],)
                    or not np.isclose(estimator.weights_.sum(), 1., atol=1e-12, rtol=0) or (estimator.weights_ <= 0).any()
                    or any(getattr(estimator, key).shape != (params["n_components"], FEATURE_COUNT)
                        for key in STATE_ARRAYS if key != "weights_")
                    or any(getattr(estimator, key).dtype != np.dtype(np.float64) or not np.isfinite(getattr(estimator, key)).all()
                        for key in STATE_ARRAYS)
                    or (estimator.covariances_ <= 0).any()):
                raise ValueError("Class mixture backend parameters, seed, convergence or density shape changed")
            expected_audits.append(_backend_audit(estimator, class_id, counts[class_id]))
    except (AttributeError, TypeError) as error:
        raise ValueError("Incomplete fitted class mixture density state") from error
    if (details.get("class_models") != expected_audits
            or getattr(model, "state_sha256_", None) != [entry["state_sha256"] for entry in expected_audits]):
        raise ValueError("Class mixture learned density or convergence evidence changed")
    return True
