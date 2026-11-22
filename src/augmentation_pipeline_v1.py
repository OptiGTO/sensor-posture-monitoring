"""Versioned record-constant gain recipes with unmodified runtime sequences.

The caller supplies training rows and recording identities. Gain draws replace
training values once, before the frozen 596-feature extractor. Inference always
extracts features from the supplied unaugmented sensor sequence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .posture_core import CLASS_NAMES, SENSOR_COLUMNS, feature_names
from .record_gain_features_v1 import augment_record_gains, sequence_features, validate_sequences


GAIN = "record_gain_extra_trees"
VOTING = "gain_soft_voting"
AUGMENTATION_FAMILIES = frozenset({GAIN, VOTING})
DEFAULT_SEED = 20260921
DATA_POLICY = "include_ordinary_reserved_training_records_v1"
TREE_DEFAULTS = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 8, "max_features": .5}
AUGMENTATION_FIXED = {"distribution": "uniform", "row_policy": "replace", "record_seed_policy": "sha256_record_v1"}


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Gain pipeline seed must be a nonnegative uint32 integer")
    return value


def validate_augmentation_pipeline(spec: dict) -> dict:
    """Normalize every choice without deriving parameters from evaluation data."""
    if not isinstance(spec, dict):
        raise ValueError("Gain pipeline specification must be an object")
    value = deepcopy(spec)
    allowed = {"family", "base_method", "window_seconds", "feature_engineering", "preprocessing",
        "model_params", "training", "postprocessing", "members"}
    family = value.get("family")
    if set(value) - allowed or not isinstance(family, str) or family not in AUGMENTATION_FAMILIES:
        raise ValueError("Unknown gain pipeline field or family")
    value.setdefault("base_method", "standard")
    value.setdefault("window_seconds", 3)
    value.setdefault("feature_engineering", {"mode": "all_from_sequence"})
    value.setdefault("preprocessing", {"scaler": "standard"})
    value.setdefault("model_params", {})
    value.setdefault("training", {})
    value.setdefault("postprocessing", {"kind": "argmax"})
    for key in ("feature_engineering", "preprocessing", "model_params", "training", "postprocessing"):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} must be an object")
    if value["base_method"] != "standard" or type(value["window_seconds"]) is not int or value["window_seconds"] != 3:
        raise ValueError("Gain pipeline requires standard method and 3-second windows")
    for key, expected in (("feature_engineering", {"mode": "all_from_sequence"}),
            ("preprocessing", {"scaler": "standard"}), ("postprocessing", {"kind": "argmax"})):
        if value[key] != expected:
            raise ValueError(f"Gain pipeline requires {key}={expected}")
    training, params = value["training"], value["model_params"]
    if set(training) - {"seed", "weighting", "data_policy", "augmentation"}:
        raise ValueError("Unknown gain training fields")
    training.setdefault("seed", DEFAULT_SEED)
    weighting = "per_member" if family == VOTING else "class_balanced"
    training.setdefault("weighting", weighting)
    _seed(training["seed"])
    if training["weighting"] != weighting or training.get("data_policy") not in (None, DATA_POLICY):
        raise ValueError("Gain training weighting or record policy differs from the declared contract")
    if family == VOTING:
        members = value.get("members")
        if params or "augmentation" in training or not isinstance(members, list) or len(members) < 2:
            raise ValueError("Gain voting requires at least two members, without outer augmentation or model parameters")
        if any(not isinstance(member, dict) or member.get("family") != GAIN for member in members):
            raise ValueError("Gain voting accepts only nonnested record_gain_extra_trees members")
        value["members"] = [validate_augmentation_pipeline(member) for member in members]
        if any(member["training"].get("data_policy") != training.get("data_policy")
                or member["training"]["seed"] != training["seed"] for member in value["members"]):
            raise ValueError("Gain voting members must share training-record policy and primary seed")
        if len({json.dumps(member, sort_keys=True) for member in value["members"]}) != len(members):
            raise ValueError("Gain voting members must have distinct recipes")
    else:
        if "members" in value:
            raise ValueError("Only gain_soft_voting accepts members")
        augmentation = training.setdefault("augmentation", {"mode": "identity", "strength": 0.})
        if not isinstance(augmentation, dict) or set(augmentation) - {"mode", "strength", *AUGMENTATION_FIXED}:
            raise ValueError("Unknown gain augmentation fields")
        augmentation.setdefault("mode", "identity")
        augmentation.setdefault("strength", 0.)
        for key, expected in AUGMENTATION_FIXED.items():
            augmentation.setdefault(key, expected)
            if augmentation[key] != expected:
                raise ValueError(f"Gain augmentation requires {key}={expected}")
        mode, strength = augmentation["mode"], augmentation["strength"]
        if (mode not in ("identity", "common", "per_sensor") or isinstance(strength, bool)
                or not isinstance(strength, (float, int)) or strength not in (0., .02, .05)
                or mode == "identity" and strength != 0):
            raise ValueError("Gain mode/strength must be identity/0 or common/per_sensor with 0, .02 or .05")
        augmentation["strength"] = float(strength)
        if set(params) - set(TREE_DEFAULTS):
            raise ValueError("Unknown gain model parameters; class weights, seeds and threads are fixed")
        for key, default in TREE_DEFAULTS.items():
            params.setdefault(key, default)
        for key in ("n_estimators", "min_samples_leaf"):
            if type(params[key]) is not int or params[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if params["max_depth"] is not None and (type(params["max_depth"]) is not int or params["max_depth"] < 1):
            raise ValueError("max_depth must be null or a positive integer")
        count = params["max_features"]
        if not ((type(count) is int and 1 <= count <= 596)
                or (type(count) is float and np.isfinite(count) and 0 < count <= 1)
                or (isinstance(count, str) and count in ("sqrt", "log2"))):
            raise ValueError("max_features must be a supported positive feature count or fraction")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Gain recipe must contain finite JSON values") from error
    return value


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([array.dtype.str, list(array.shape)]).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _scaler_digest(scaler):
    return _json_digest({"params": scaler.get_params(deep=False),
        "samples": int(scaler.n_samples_seen_), "features": int(scaler.n_features_in_),
        **{key: getattr(scaler, key).tolist() for key in ("mean_", "scale_", "var_")}})


def _estimator(spec, seed):
    return ExtraTreesClassifier(**deepcopy(spec["model_params"]), class_weight="balanced", random_state=seed, n_jobs=1)


class AugmentationIterativePipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, spec, seed=None):
        self.spec = spec
        self.seed = seed

    def fit(self, X, y, *, groups=None, record_ids=None, validation_data=None):
        spec = validate_augmentation_pipeline(self.spec)
        seed = _seed(spec["training"]["seed"] if self.seed is None else self.seed)
        values, labels = validate_sequences(X), np.asarray(y)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu"
                or not np.array_equal(np.unique(labels), np.arange(7))):
            raise ValueError("Gain training requires matching labels with all seven posture IDs 0..6")
        labels = labels.astype(np.int64, copy=True)
        # Identity audit validates record purity for ensembles before any member
        # can fit. It does not import or synthesize missing recording identities.
        _, identity_audit = augment_record_gains(values, labels, groups, record_ids,
            mode="identity", strength=0., seed=seed)
        with threadpool_limits(limits=1):
            if spec["family"] == VOTING:
                members = [build_augmentation_pipeline(member, seed=self.seed).fit(values, labels,
                    groups=groups, record_ids=record_ids) for member in spec["members"]]
            else:
                augmentation = spec["training"]["augmentation"]
                augmented, gain_audit = augment_record_gains(values, labels, groups, record_ids,
                    mode=augmentation["mode"], strength=augmentation["strength"], seed=seed)
                features = sequence_features(augmented)
                scaler = StandardScaler()
                transformed = scaler.fit_transform(features)
                estimator = _estimator(spec, seed).fit(transformed, labels)
        self.spec_, self.effective_seed_ = spec, seed
        self.input_shape_, self.n_features_in_, self.n_transformed_features_ = (36, 16), 16, 596
        self.classes_, self.label_names_, self.sensor_columns_ = np.arange(7), list(CLASS_NAMES), list(SENSOR_COLUMNS)
        self.feature_names_out_ = np.asarray(feature_names())
        self.feature_schema_sha256_ = _json_digest(feature_names())
        self.identity_audit_ = identity_audit
        self.record_metadata_ = deepcopy(identity_audit["records"])
        self.identity_audit_sha256_ = _json_digest(identity_audit)
        self.training_details_ = {"pipeline": deepcopy(spec), "effective_seed": seed,
            "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
            "input_contract": "sensor_sequence", "input_shape": [36, 16], "sensor_columns": list(SENSOR_COLUMNS),
            "sampling_rate": 12, "training_windows": len(values), "training_records": len(self.record_metadata_),
            "class_counts": np.bincount(labels, minlength=7).tolist(), "class_names": list(CLASS_NAMES),
            "training_input_sha256": _array_digest(values), "feature_names": feature_names(),
            "feature_schema_sha256": self.feature_schema_sha256_, "model_input_features": 596,
            "record_metadata": deepcopy(self.record_metadata_), "identity_audit_sha256": self.identity_audit_sha256_,
            "external_validation_used": False, "inference_augmentation": False,
            "preprocessing_fit": "replace training sequences once by record-constant gains; frozen all596 extraction; unweighted train StandardScaler",
            "training_record_policy": spec["training"].get("data_policy"),
            "training_record_policy_application": "caller supplies eligible training records; estimator never imports historical caches",
            "threads": 1}
        if spec["family"] == VOTING:
            self.members_ = members
            self.training_details_.update(combination="equal aligned probability mean", weights=[1 / len(members)] * len(members),
                members=[deepcopy(member.training_details_) for member in members])
        else:
            self.scaler_, self.estimator_, self.gain_audit_ = scaler, estimator, gain_audit
            self.gain_audit_sha256_ = _json_digest(gain_audit)
            self.scaler_state_sha256_, self.training_features_sha256_ = _scaler_digest(scaler), _array_digest(features)
            self.training_details_.update(gain_audit=deepcopy(gain_audit), gain_audit_sha256=self.gain_audit_sha256_,
                scaler_state_sha256=self.scaler_state_sha256_, training_features_sha256=self.training_features_sha256_,
                model_input_dtype=str(transformed.dtype), effective_model_params=estimator.get_params(deep=False))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "spec_")
        if validate_augmentation_pipeline(self.spec) != self.spec_ or _seed(
                self.spec_["training"]["seed"] if self.seed is None else self.seed) != self.effective_seed_:
            raise ValueError("Gain recipe or seed changed after fitting")
        values = validate_sequences(X)
        with threadpool_limits(limits=1):
            if self.spec_["family"] == VOTING:
                result = np.mean([member.predict_proba(values) for member in self.members_], axis=0)
            else:
                features = sequence_features(values)
                classes = np.asarray(self.estimator_.classes_)
                if classes.dtype.kind not in "iu" or not np.array_equal(np.sort(classes), self.classes_):
                    raise ValueError("Gain estimator posture classes changed")
                raw = np.asarray(self.estimator_.predict_proba(self.scaler_.transform(features)), dtype=float)
                if raw.shape != (len(values), 7):
                    raise ValueError("Invalid gain estimator probability shape")
                result = np.empty_like(raw)
                result[:, classes] = raw
        if (result.shape != (len(values), 7) or not np.isfinite(result).all() or (result < 0).any()
                or not np.allclose(result.sum(1), 1., rtol=0, atol=1e-12)):
            raise ValueError("Invalid gain estimator probabilities")
        return result

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def build_augmentation_pipeline(spec: dict, seed: int | None = None):
    if seed is not None:
        _seed(seed)
    return AugmentationIterativePipeline(validate_augmentation_pipeline(spec), seed=seed)


def fit_augmentation_pipeline(model, spec: dict, X, y, *, groups=None, record_ids=None, validation_data=None):
    recipe = validate_augmentation_pipeline(spec)
    if not isinstance(model, AugmentationIterativePipeline) or validate_augmentation_pipeline(model.spec) != recipe:
        raise ValueError("Training specification differs from the constructed gain pipeline")
    return model.fit(X, y, groups=groups, record_ids=record_ids, validation_data=validation_data)


def verify_fitted_augmentation_identity(model, spec: dict, seed: int):
    """Verify complete declared algorithm, record audit and fitted parameters."""
    recipe, seed = validate_augmentation_pipeline(spec), _seed(seed)
    if (not isinstance(model, AugmentationIterativePipeline) or getattr(model, "spec_", None) != recipe
            or validate_augmentation_pipeline(model.spec) != recipe or getattr(model, "effective_seed_", None) != seed
            or (recipe["training"]["seed"] if model.seed is None else model.seed) != seed
            or getattr(model, "input_shape_", None) != (36, 16) or getattr(model, "n_features_in_", None) != 16
            or getattr(model, "n_transformed_features_", None) != 596
            or not np.array_equal(getattr(model, "classes_", ()), np.arange(7))
            or getattr(model, "sensor_columns_", None) != list(SENSOR_COLUMNS)
            or getattr(model, "label_names_", None) != list(CLASS_NAMES)
            or not np.array_equal(getattr(model, "feature_names_out_", ()), feature_names())
            or getattr(model, "feature_schema_sha256_", None) != _json_digest(feature_names())):
        raise ValueError("Fitted gain recipe, seed or input/feature contract changed")
    details = getattr(model, "training_details_", {})
    count = details.get("training_windows")
    if (details.get("pipeline") != recipe or details.get("effective_seed") != seed
            or details.get("sklearn_version") != sklearn.__version__ or details.get("numpy_version") != np.__version__
            or details.get("external_validation_used") is not False or details.get("inference_augmentation") is not False
            or type(count) is not int or count < 7 or details.get("class_names") != list(CLASS_NAMES)
            or details.get("feature_names") != feature_names() or details.get("feature_schema_sha256") != model.feature_schema_sha256_
            or details.get("record_metadata") != getattr(model, "record_metadata_", None)
            or details.get("training_record_policy") != recipe["training"].get("data_policy")):
        raise ValueError("Fitted gain training audit differs from its recipe or library versions")
    _verify_record_audits(model, recipe, seed, count)
    if recipe["family"] == VOTING:
        members = getattr(model, "members_", ())
        if len(members) != len(recipe["members"]):
            raise ValueError("Fitted gain voting member count changed")
        for member, member_recipe in zip(members, recipe["members"]):
            verify_fitted_augmentation_identity(member, member_recipe, seed)
            if member.record_metadata_ != model.record_metadata_ or member.training_details_["training_input_sha256"] != details["training_input_sha256"]:
                raise ValueError("Gain voting members trained on different recording rows")
        return True
    scaler, estimator = getattr(model, "scaler_", None), getattr(model, "estimator_", None)
    if (type(scaler) is not StandardScaler or scaler.get_params(deep=False) != StandardScaler().get_params(deep=False)
            or getattr(scaler, "n_features_in_", None) != 596 or getattr(scaler, "n_samples_seen_", None) != count
            or _scaler_digest(scaler) != getattr(model, "scaler_state_sha256_", None)
            or details.get("scaler_state_sha256") != model.scaler_state_sha256_
            or details.get("training_features_sha256") != getattr(model, "training_features_sha256_", None)):
        raise ValueError("Fitted gain feature or scaler state changed")
    expected = _estimator(recipe, seed)
    if (type(estimator) is not ExtraTreesClassifier or estimator.get_params(deep=False) != expected.get_params(deep=False)
            or details.get("effective_model_params") != expected.get_params(deep=False)
            or getattr(estimator, "n_features_in_", None) != 596
            or not np.array_equal(getattr(estimator, "classes_", ()), np.arange(7))
            or len(getattr(estimator, "estimators_", ())) != recipe["model_params"]["n_estimators"]):
        raise ValueError("Fitted gain Extra Trees parameters, seed, classes or tree count changed")
    return True


def _verify_record_audits(model, recipe, seed, count):
    """Regenerate the full metadata/gain audit without retaining sensor values."""
    details = model.training_details_
    metadata = getattr(model, "record_metadata_", None)
    try:
        if not isinstance(metadata, list) or not metadata or details.get("training_records") != len(metadata):
            raise ValueError("Missing training-record metadata")
        labels = np.full(count, -1, dtype=np.int64)
        subjects, records = np.full(count, "", dtype=object), np.full(count, "", dtype=object)
        covered = np.zeros(count, dtype=bool)
        for row in metadata:
            indices = row["row_indices"]
            if (not isinstance(indices, list) or row["windows"] != len(indices) or not indices
                    or any(type(index) is not int or not 0 <= index < count for index in indices)
                    or len(set(indices)) != len(indices) or covered[indices].any()
                    or type(row["class_id"]) is not int or not 0 <= row["class_id"] < 7):
                raise ValueError("Invalid or overlapping training-record rows")
            labels[indices], subjects[indices], records[indices] = row["class_id"], row["subject"], row["record_id"]
            covered[indices] = True
        if not covered.all() or not np.array_equal(np.unique(labels), np.arange(7)):
            raise ValueError("Record metadata does not cover every training row and class")
        zeros = np.zeros((count, 36, 16), dtype=np.float32)
        _, identity = augment_record_gains(zeros, labels, subjects, records, mode="identity", strength=0., seed=seed)
        if (getattr(model, "identity_audit_", None) != identity or metadata != identity["records"]
                or getattr(model, "identity_audit_sha256_", None) != _json_digest(identity)
                or details.get("identity_audit_sha256") != model.identity_audit_sha256_
                or details.get("class_counts") != np.bincount(labels, minlength=7).tolist()):
            raise ValueError("Training-record identity audit changed")
        if recipe["family"] != VOTING:
            augmentation = recipe["training"]["augmentation"]
            _, expected = augment_record_gains(zeros, labels, subjects, records,
                mode=augmentation["mode"], strength=augmentation["strength"], seed=seed)
            if (getattr(model, "gain_audit_", None) != expected or details.get("gain_audit") != expected
                    or getattr(model, "gain_audit_sha256_", None) != _json_digest(expected)
                    or details.get("gain_audit_sha256") != model.gain_audit_sha256_):
                raise ValueError("Seeded recording gains differ from the declared augmentation")
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Incomplete fitted gain recording audit") from error
