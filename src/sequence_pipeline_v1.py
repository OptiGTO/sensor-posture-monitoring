"""Versioned, serializable research recipes for 36-by-16 sensor windows.

This module leaves the registered feature-vector pipeline unchanged. All fitted
normalizers, MiniRocket parameters and neural weights live inside the estimator.
The caller owns subject-disjoint partitions and any extra training-record policy.
"""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .posture_core import CLASS_NAMES, SENSOR_COLUMNS


DEFAULT_SEED = 20260921
SEQUENCE_FAMILIES = frozenset({"cnn1d", "minirocket", "sequence_soft_voting"})
DATA_POLICY = "include_ordinary_reserved_training_records_v1"
CNN_DEFAULTS = {"hidden_size": 32, "epochs": 40, "patience": 8, "batch_size": 64,
    "learning_rate": .001, "weight_decay": .01, "dropout": .2,
    "gain_std": 0., "noise_fraction": 0.}


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**31:
        # The vendored MiniRocket bias function has an optional int32 signature.
        raise ValueError("Sequence seed must be a nonnegative signed-int32 integer")
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{label} must be finite and numeric")
    return value


def validate_sequence_pipeline(spec: dict) -> dict:
    """Return a complete JSON recipe without importing Torch or Numba."""
    if not isinstance(spec, dict):
        raise ValueError("Sequence pipeline specification must be an object")
    value = deepcopy(spec)
    allowed = {"family", "base_method", "window_seconds", "feature_engineering",
        "preprocessing", "model_params", "training", "postprocessing", "members"}
    if set(value) - allowed or not isinstance(value.get("family"), str) or value["family"] not in SEQUENCE_FAMILIES:
        raise ValueError("Unknown sequence pipeline field or family")
    value.setdefault("base_method", "standard")
    value.setdefault("window_seconds", 3)
    value.setdefault("feature_engineering", {"mode": "raw_sequence"})
    value.setdefault("preprocessing", {"scaler": "model_internal"})
    value.setdefault("model_params", {})
    value.setdefault("training", {})
    value.setdefault("postprocessing", {"kind": "argmax"})
    for key in ("feature_engineering", "preprocessing", "model_params", "training", "postprocessing"):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} must be an object")
    if value["base_method"] not in ("standard", "regularized"):
        raise ValueError("Unknown base_method; model_params defines the exact sequence recipe")
    if type(value["window_seconds"]) is not int or value["window_seconds"] != 3:
        raise ValueError("Sequence pipeline v1 requires 3-second windows")
    for key, expected in (("feature_engineering", {"mode": "raw_sequence"}),
            ("preprocessing", {"scaler": "model_internal"}), ("postprocessing", {"kind": "argmax"})):
        if value[key] != expected:
            raise ValueError(f"Sequence pipeline v1 requires {key}={expected}")
    training = value["training"]
    if set(training) - {"seed", "weighting", "data_policy"}:
        raise ValueError("Unknown sequence training fields")
    training.setdefault("seed", DEFAULT_SEED)
    training.setdefault("weighting", "class_balanced")
    _seed(training["seed"])
    if training["weighting"] != "class_balanced":
        raise ValueError("Sequence pipeline v1 supports class_balanced training only")
    if training.get("data_policy") not in (None, DATA_POLICY):
        raise ValueError("Unknown training-record policy")
    family, params = value["family"], value["model_params"]
    if family == "sequence_soft_voting":
        members = value.get("members")
        if params or value["base_method"] != "standard" or not isinstance(members, list) or len(members) < 2:
            raise ValueError("Sequence voting requires at least two complete members and no outer model parameters")
        if any(not isinstance(member, dict) or member.get("family") not in ("cnn1d", "minirocket") for member in members):
            raise ValueError("Sequence voting accepts only CNN/MiniRocket members, without nested ensembles")
        value["members"] = [validate_sequence_pipeline(member) for member in members]
        if any(member["training"].get("data_policy") != training.get("data_policy") for member in value["members"]):
            raise ValueError("Sequence voting members must share the training-record policy")
        if any(member["training"]["seed"] != training["seed"] for member in value["members"]):
            raise ValueError("Sequence voting members must declare the same primary seed")
        if len({json.dumps(member, sort_keys=True) for member in value["members"]}) != len(members):
            raise ValueError("Sequence voting members must have distinct recipes")
    else:
        if "members" in value:
            raise ValueError("Only sequence_soft_voting accepts members")
        defaults = CNN_DEFAULTS if family == "cnn1d" else {"num_kernels": 1008, "C": 1.}
        if set(params) - set(defaults):
            raise ValueError("Unknown sequence model parameters; architecture, seed and weighting are fixed by the recipe")
        for key, default in defaults.items():
            params.setdefault(key, default)
        if family == "cnn1d":
            for key in ("hidden_size", "epochs", "patience", "batch_size"):
                if type(params[key]) is not int or params[key] < 1:
                    raise ValueError(f"CNN {key} must be a positive integer")
            for key in ("learning_rate", "weight_decay", "dropout", "gain_std", "noise_fraction"):
                _number(params[key], "CNN " + key)
            if params["learning_rate"] <= 0 or params["weight_decay"] < 0 or not 0 <= params["dropout"] < 1:
                raise ValueError("CNN learning rate, weight decay or dropout is outside its supported range")
            if params["gain_std"] != 0 or params["noise_fraction"] != 0:
                raise ValueError("Sequence pipeline v1 does not apply sensor augmentation")
        else:
            count = params["num_kernels"]
            if type(count) is not int or count < 84 or count % 84:
                raise ValueError("MiniRocket num_kernels must be a positive multiple of 84")
            if _number(params["C"], "MiniRocket C") <= 0:
                raise ValueError("MiniRocket C must be positive")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Sequence recipe must contain finite JSON values") from error
    return value


def _sequence(X):
    values = np.asarray(X)
    if (values.ndim != 3 or values.shape[1:] != (36, 16) or not len(values)
            or values.dtype.kind not in "iuf" or not np.isfinite(values).all()):
        raise ValueError("Expected nonempty finite numeric sensor sequences with shape (N, 36, 16)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(values, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Sensor values exceed the finite float32 input range")
    return result


class SequenceIterativePipeline(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible sequence estimator with an immutable fitted recipe."""

    def __init__(self, spec, seed=None):
        self.spec = spec
        self.seed = seed

    def fit(self, X, y, *, groups=None, validation_data=None):
        spec = validate_sequence_pipeline(self.spec)
        seed = _seed(spec["training"]["seed"] if self.seed is None else self.seed)
        values, labels = _sequence(X), np.asarray(y)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu"
                or not np.array_equal(np.unique(labels), np.arange(len(CLASS_NAMES)))):
            raise ValueError("Sequence training requires matching labels containing all seven posture IDs 0..6")
        labels = labels.astype(np.int64, copy=True)
        subject_count = None
        if groups is not None:
            identities = np.asarray(groups)
            if identities.ndim != 1 or len(identities) != len(values):
                raise ValueError("Training groups must match the sequence rows")
            if any(not isinstance(item, (str, int, np.integer)) or isinstance(item, (bool, np.bool_))
                    or isinstance(item, str) and not item.strip() for item in identities.tolist()):
                raise ValueError("Training group IDs must be nonempty strings or integers")
            subject_count = len(set(identities.tolist()))
        family = spec["family"]
        with threadpool_limits(limits=1):
            if family == "sequence_soft_voting":
                members = [build_sequence_pipeline(member, seed=self.seed).fit(values, labels, groups=groups)
                           for member in spec["members"]]
                estimator = None
            else:
                from .extended_models import MiniRocketClassifier, SmallNeuralClassifier
                if family == "cnn1d":
                    estimator = SmallNeuralClassifier(architecture="cnn1d", random_state=seed, **spec["model_params"])
                    # Deliberately ignore external validation for epoch selection.
                    estimator.fit(values, labels, validation_data=None)
                else:
                    estimator = MiniRocketClassifier(random_state=seed, **spec["model_params"])
                    estimator.fit(values, labels)
                    if estimator.training_details_["transform_features"] != spec["model_params"]["num_kernels"]:
                        raise ValueError("MiniRocket transform feature count differs from its declared recipe")
                if not np.array_equal(estimator.classes_, np.arange(len(CLASS_NAMES))):
                    raise ValueError("Fitted sequence model has a different posture class contract")
        self.spec_, self.effective_seed_ = spec, seed
        self.input_shape_, self.n_features_in_ = (36, 16), 16
        self.classes_ = np.arange(len(CLASS_NAMES))
        self.sensor_columns_ = list(SENSOR_COLUMNS)
        self.training_details_ = {"pipeline": deepcopy(spec), "effective_seed": seed,
            "input_contract": "sensor_sequence", "input_shape": [36, 16],
            "sensor_columns": list(SENSOR_COLUMNS), "sampling_rate": 12,
            "training_windows": len(values), "training_subject_count": subject_count,
            "class_counts": np.bincount(labels, minlength=len(CLASS_NAMES)).tolist(),
            "preprocessing_fit": "training subjects only", "external_validation_used": False,
            "training_record_policy": spec["training"].get("data_policy"),
            "training_record_policy_application": "caller prepares eligible training rows; estimator never imports historical records",
            "device": "cpu", "threads": 1}
        if family == "sequence_soft_voting":
            self.members_ = members
            self.training_details_.update(combination="equal aligned probability mean",
                weights=[1 / len(members)] * len(members),
                members=[deepcopy(member.training_details_) for member in members])
        else:
            self.estimator_ = estimator
            self.training_details_["model_training"] = deepcopy(estimator.training_details_)
            self.training_details_["effective_model_params"] = estimator.get_params(deep=False)
            if family == "cnn1d":
                self.training_details_["normalization"] = {
                    "kind": "per-sensor training mean and standard deviation across rows and time",
                    "mean": estimator.mean_.tolist(), "scale": estimator.scale_.tolist()}
                self.training_details_["epoch_policy"] = "fixed epoch budget; keep minimum training-loss epoch; external validation and patience stopping unused"
            else:
                self.training_details_["normalization"] = {
                    "kind": "training-fitted MiniRocket thresholds, followed by StandardScaler on transformed training features",
                    "requested_transform_features": spec["model_params"]["num_kernels"],
                    "actual_transform_features": estimator.training_details_["transform_features"]}
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "spec_")
        if validate_sequence_pipeline(self.spec) != self.spec_ or _seed(
                self.spec_["training"]["seed"] if self.seed is None else self.seed) != self.effective_seed_:
            raise ValueError("Sequence recipe or seed changed after fitting")
        values = _sequence(X)
        with threadpool_limits(limits=1):
            if self.spec_["family"] == "sequence_soft_voting":
                result = np.mean([member.predict_proba(values) for member in self.members_], axis=0)
            else:
                raw = np.asarray(self.estimator_.predict_proba(values), dtype=float)
                result = np.zeros((len(values), len(CLASS_NAMES)), dtype=float)
                classes = np.asarray(self.estimator_.classes_)
                if classes.dtype.kind not in "iu" or not np.array_equal(np.sort(classes), self.classes_):
                    raise ValueError("Fitted sequence classes changed")
                if raw.shape != (len(values), len(classes)):
                    raise ValueError("Sequence estimator returned an invalid probability shape")
                result[:, classes] = raw
        if (result.shape != (len(values), len(CLASS_NAMES)) or not np.isfinite(result).all()
                or (result < 0).any() or not np.allclose(result.sum(axis=1), 1., rtol=1e-6, atol=1e-7)):
            raise ValueError("Sequence estimator returned invalid posture probabilities")
        return result

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def build_sequence_pipeline(spec: dict, seed: int | None = None):
    if seed is not None:
        _seed(seed)
    return SequenceIterativePipeline(validate_sequence_pipeline(spec), seed=seed)


def fit_sequence_pipeline(model, spec: dict, X, y, *, groups=None, validation_data=None):
    normalized = validate_sequence_pipeline(spec)
    if not isinstance(model, SequenceIterativePipeline) or validate_sequence_pipeline(model.spec) != normalized:
        raise ValueError("Training specification differs from the constructed sequence pipeline")
    return model.fit(X, y, groups=groups, validation_data=validation_data)
