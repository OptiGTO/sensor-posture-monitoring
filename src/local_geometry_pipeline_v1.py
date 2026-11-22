"""Versioned full-feature record prototypes and class-conditional mixtures."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .posture_core import CLASS_NAMES, feature_names, normalize_text
from .record_gain_features_v1 import validate_training_records


PROTOTYPE = "record_prototype"
MIXTURE = "class_conditional_mixture"
VOTING = "local_soft_voting"
LOCAL_FAMILIES = frozenset({PROTOTYPE, MIXTURE, VOTING})
DATA_POLICY = "include_ordinary_reserved_training_records_v1"
MIXTURE_FIXED = {"covariance_type": "diag", "init_params": "kmeans", "warm_start": False}
MIXTURE_DEFAULTS = {"n_components": 2, "reg_covar": .1, "n_init": 3, "max_iter": 200, "tol": 1e-3}


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Local pipeline seed must be a uint32 integer")
    return value


def validate_local_pipeline(spec):
    if not isinstance(spec, dict):
        raise ValueError("Local pipeline specification must be an object")
    value = deepcopy(spec)
    family = value.get("family")
    if not isinstance(family, str) or family not in LOCAL_FAMILIES or set(value) - {
            "family", "base_method", "window_seconds", "feature_engineering", "preprocessing",
            "model_params", "training", "postprocessing", "members"}:
        raise ValueError("Unknown local pipeline family or field")
    for key, default in (("base_method", "standard"), ("window_seconds", 3),
            ("feature_engineering", {"mode": "all"}), ("preprocessing", {"scaler": "standard"}),
            ("model_params", {}), ("training", {}), ("postprocessing", {"kind": "argmax"})):
        value.setdefault(key, deepcopy(default))
    for key in ("feature_engineering", "preprocessing", "model_params", "training", "postprocessing"):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} must be an object")
    if (value["base_method"] != "standard" or type(value["window_seconds"]) is not int or value["window_seconds"] != 3
            or value["feature_engineering"] != {"mode": "all"} or value["preprocessing"] != {"scaler": "standard"}
            or value["postprocessing"] != {"kind": "argmax"}):
        raise ValueError("Local v1 requires 3-second all596 features, StandardScaler and argmax")
    params, training = value["model_params"], value["training"]
    if set(training) - {"seed", "weighting", "data_policy"}:
        raise ValueError("Unknown local training fields")
    weighting = "equal_class_distance" if family == PROTOTYPE else "uniform_class_prior" if family == MIXTURE else "per_member"
    training.setdefault("seed", 20260921)
    training.setdefault("weighting", weighting)
    _seed(training["seed"])
    if training["weighting"] != weighting or training.get("data_policy") not in (None, DATA_POLICY):
        raise ValueError("Local training weighting or record policy differs from its family")
    if family == VOTING:
        members = value.get("members")
        if params or not isinstance(members, list) or len(members) < 2 or any(
                not isinstance(member, dict) or member.get("family") not in (PROTOTYPE, MIXTURE) for member in members):
            raise ValueError("Local voting requires two or more nonnested leaf members")
        value["members"] = [validate_local_pipeline(member) for member in members]
        if any(member["training"].get("data_policy") != training.get("data_policy")
                or member["training"]["seed"] != training["seed"] for member in value["members"]):
            raise ValueError("Local voting members must share primary seed and training policy")
        if len({json.dumps(member, sort_keys=True) for member in value["members"]}) != len(members):
            raise ValueError("Local voting members must have distinct recipes")
    else:
        if "members" in value:
            raise ValueError("Only local_soft_voting accepts members")
        defaults = {"n_neighbors": 1, "epsilon": 1e-12} if family == PROTOTYPE else {**MIXTURE_DEFAULTS, **MIXTURE_FIXED}
        if set(params) - set(defaults):
            raise ValueError("Unknown local model parameters")
        for key, default in defaults.items():
            params.setdefault(key, default)
        if family == PROTOTYPE:
            if type(params["n_neighbors"]) is not int or params["n_neighbors"] not in (1, 3) or params["epsilon"] != 1e-12:
                raise ValueError("Prototype v1 requires n_neighbors 1 or 3 and epsilon 1e-12")
        else:
            if (type(params["n_components"]) is not int or params["n_components"] not in (1, 2)
                    or any(type(params[key]) is not int or params[key] != MIXTURE_DEFAULTS[key] for key in ("n_init", "max_iter"))
                    or any(type(params[key]) is not float or params[key] != MIXTURE_DEFAULTS[key] for key in ("reg_covar", "tol"))
                    or params["warm_start"] is not False
                    or any(params[key] != expected for key, expected in MIXTURE_FIXED.items())):
                raise ValueError("Mixture v1 requires preregistered components1/2, diagonal covariance and fixed regularization/optimizer")
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("Local recipe must contain finite JSON values") from error
    return value


def local_requires_record_ids(spec):
    value = validate_local_pipeline(spec)
    return value["family"] == PROTOTYPE or value["family"] == VOTING and any(
        member["family"] == PROTOTYPE for member in value["members"])


def _features(X):
    if hasattr(X, "columns") and list(X.columns) != feature_names():
        raise ValueError("Local input columns differ from the canonical feature schema")
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[1] != 596 or not len(values) or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite real features with shape (N,596)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(values, dtype=np.float32 if values.dtype == np.dtype(np.float32) else np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Features exceed supported numeric range")
    return result


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _scaler_digest(scaler):
    return _json_digest({"params": scaler.get_params(deep=False), "samples": int(scaler.n_samples_seen_),
        "features": int(scaler.n_features_in_), **{key: getattr(scaler, key).tolist() for key in ("mean_", "scale_", "var_")}})


def _estimator(spec, seed):
    if spec["family"] == PROTOTYPE:
        from .record_prototype_v1 import RecordPrototypeClassifier
        return RecordPrototypeClassifier(**spec["model_params"])
    from .class_conditional_mixture_v1 import ClassConditionalMixtureClassifier
    return ClassConditionalMixtureClassifier(**{key: spec["model_params"][key] for key in MIXTURE_DEFAULTS}, random_state=seed)


class LocalGeometryPipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, spec, seed=None):
        self.spec = spec
        self.seed = seed

    def fit(self, X, y, *, groups=None, record_ids=None, validation_data=None):
        spec = validate_local_pipeline(self.spec)
        seed = _seed(spec["training"]["seed"] if self.seed is None else self.seed)
        values, labels = _features(X), np.asarray(y)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu"
                or not np.array_equal(np.unique(labels), np.arange(7))):
            raise ValueError("Local training requires matching labels with all seven posture classes")
        people = None
        if groups is not None:
            people = np.asarray(groups)
            if people.ndim != 1 or len(people) != len(values) or any(
                    not isinstance(person, str) or not normalize_text(person) for person in people.tolist()):
                raise ValueError("Local training subject identities must align")
        if local_requires_record_ids(spec):
            validate_training_records(labels, groups, record_ids)
        with threadpool_limits(limits=1):
            if spec["family"] == VOTING:
                members = [build_local_pipeline(member, seed=self.seed).fit(values, labels, groups=groups,
                    record_ids=record_ids if local_requires_record_ids(member) else None) for member in spec["members"]]
            else:
                scaler = StandardScaler()
                transformed = scaler.fit_transform(values)
                estimator = _estimator(spec, seed)
                if spec["family"] == PROTOTYPE:
                    estimator.fit(transformed, labels, groups=groups, record_ids=record_ids)
                else:
                    estimator.fit(transformed, labels)
        self.spec_, self.effective_seed_ = spec, seed
        self.classes_, self.label_names_, self.n_features_in_ = np.arange(7), list(CLASS_NAMES), 596
        self.feature_names_in_ = np.asarray(feature_names())
        self.training_details_ = {"pipeline": deepcopy(spec), "effective_seed": seed,
            "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
            "input_contract": "feature_vector", "training_windows": len(values), "training_feature_dtype": str(values.dtype),
            "training_subjects": None if people is None else sorted({normalize_text(person) for person in people.tolist()}),
            "class_counts": np.bincount(labels.astype(int), minlength=7).tolist(), "class_names": list(CLASS_NAMES),
            "feature_names": feature_names(), "external_validation_used": False, "threads": 1,
            "training_record_policy": spec["training"].get("data_policy"),
            "training_record_policy_application": "caller supplies eligible training rows; no historical caches imported",
            "preprocessing_fit": "unweighted StandardScaler on original training windows only; no evaluation adaptation"}
        if spec["family"] == VOTING:
            self.members_ = members
            self.training_details_.update(combination="equal aligned probability mean", weights=[1 / len(members)] * len(members),
                members=[deepcopy(member.training_details_) for member in members])
        else:
            self.scaler_, self.estimator_ = scaler, estimator
            self.scaler_state_sha256_ = _scaler_digest(scaler)
            self.training_details_.update(scaler_state_sha256=self.scaler_state_sha256_,
                effective_model_params=estimator.get_params(deep=False),
                backend_training_details=deepcopy(getattr(estimator, "training_details_", {})),
                model_input_dtype=str(transformed.dtype), record_ids_used=spec["family"] == PROTOTYPE)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "spec_")
        if validate_local_pipeline(self.spec) != self.spec_ or _seed(
                self.spec_["training"]["seed"] if self.seed is None else self.seed) != self.effective_seed_:
            raise ValueError("Local recipe or seed changed after fitting")
        values = _features(X)
        with threadpool_limits(limits=1):
            if self.spec_["family"] == VOTING:
                probability = np.mean([member.predict_proba(values) for member in self.members_], axis=0)
            else:
                if not np.array_equal(self.estimator_.classes_, np.arange(7)):
                    raise ValueError("Local backend class order changed")
                probability = np.asarray(self.estimator_.predict_proba(self.scaler_.transform(values)), dtype=float)
        if (probability.shape != (len(values), 7) or not np.isfinite(probability).all() or (probability < 0).any()
                or not np.allclose(probability.sum(1), 1., atol=1e-12, rtol=0)):
            raise ValueError("Invalid local backend probabilities")
        return probability

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def build_local_pipeline(spec, seed=None):
    if seed is not None:
        _seed(seed)
    return LocalGeometryPipeline(validate_local_pipeline(spec), seed=seed)


def fit_local_pipeline(model, spec, X, y, *, groups=None, record_ids=None, validation_data=None):
    recipe = validate_local_pipeline(spec)
    if not isinstance(model, LocalGeometryPipeline) or validate_local_pipeline(model.spec) != recipe:
        raise ValueError("Constructed local estimator and requested recipe differ")
    return model.fit(X, y, groups=groups, record_ids=record_ids, validation_data=validation_data)


def verify_fitted_local_identity(model, spec, seed):
    recipe, seed = validate_local_pipeline(spec), _seed(seed)
    if (not isinstance(model, LocalGeometryPipeline) or getattr(model, "spec_", None) != recipe
            or validate_local_pipeline(model.spec) != recipe or getattr(model, "effective_seed_", None) != seed
            or (recipe["training"]["seed"] if model.seed is None else model.seed) != seed
            or not np.array_equal(getattr(model, "classes_", ()), np.arange(7))
            or getattr(model, "label_names_", None) != list(CLASS_NAMES) or getattr(model, "n_features_in_", None) != 596
            or not np.array_equal(getattr(model, "feature_names_in_", ()), feature_names())):
        raise ValueError("Fitted local recipe, seed or feature/class contract changed")
    details = getattr(model, "training_details_", {})
    count = details.get("training_windows")
    if (details.get("pipeline") != recipe or details.get("effective_seed") != seed
            or details.get("sklearn_version") != sklearn.__version__ or details.get("numpy_version") != np.__version__
            or details.get("external_validation_used") is not False or type(count) is not int or count < 7
            or details.get("feature_names") != feature_names() or details.get("class_names") != list(CLASS_NAMES)
            or details.get("training_record_policy") != recipe["training"].get("data_policy")):
        raise ValueError("Local training audit differs from its recipe or library versions")
    if recipe["family"] == VOTING:
        members = getattr(model, "members_", ())
        if len(members) != len(recipe["members"]):
            raise ValueError("Local ensemble member count changed")
        for member, member_recipe in zip(members, recipe["members"]):
            verify_fitted_local_identity(member, member_recipe, seed)
            if any(member.training_details_[key] != details[key] for key in ("training_windows", "training_subjects", "class_counts")):
                raise ValueError("Local ensemble member training scope changed")
        if (details.get("members") != [member.training_details_ for member in members]
                or details.get("weights") != [1 / len(members)] * len(members)
                or details.get("combination") != "equal aligned probability mean"):
            raise ValueError("Local ensemble audit differs from its fitted members")
        return True
    scaler, estimator = getattr(model, "scaler_", None), getattr(model, "estimator_", None)
    expected = _estimator(recipe, seed)
    if (type(scaler) is not StandardScaler or scaler.get_params(deep=False) != StandardScaler().get_params(deep=False)
            or getattr(scaler, "n_features_in_", None) != 596 or getattr(scaler, "n_samples_seen_", None) != count
            or _scaler_digest(scaler) != getattr(model, "scaler_state_sha256_", None)
            or details.get("scaler_state_sha256") != model.scaler_state_sha256_
            or type(estimator) is not type(expected) or estimator.get_params(deep=False) != expected.get_params(deep=False)
            or details.get("effective_model_params") != expected.get_params(deep=False)
            or np.asarray(getattr(estimator, "class_counts_", [])).tolist() != details.get("class_counts")
            or details.get("backend_training_details") != getattr(estimator, "training_details_", {})):
        raise ValueError("Local scaler, model parameters or backend audit changed")
    if recipe["family"] == PROTOTYPE:
        from .record_prototype_v1 import verify_record_prototype_identity
        verify_record_prototype_identity(estimator, **recipe["model_params"])
        if details.get("training_subjects") != estimator.training_details_["training_subjects"]:
            raise ValueError("Local prototype training subject audit changed")
    else:
        from .class_conditional_mixture_v1 import verify_class_conditional_mixture_identity
        verify_class_conditional_mixture_identity(estimator, expected.get_params(deep=False))
    return True
