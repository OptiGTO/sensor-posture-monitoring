"""Versioned record-aware Extra Trees recipes with train-only record metadata.

No record is imported by this estimator. The caller supplies the already audited
training rows, including any authorized ordinary-record restoration policy.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .posture_core import CLASS_NAMES, feature_names


BALANCED = "record_balanced_extra_trees"
BAGGED = "record_bagged_extra_trees"
VOTING = "record_soft_voting"
RECORD_FAMILIES = frozenset({BALANCED, BAGGED, VOTING})
DEFAULT_SEED = 20260921
DATA_POLICY = "include_ordinary_reserved_training_records_v1"
TREE_DEFAULTS = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 8, "max_features": .5}


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Record pipeline seed must be a nonnegative uint32 integer")
    return value


def _validate_effective_seed(spec, seed):
    _seed(seed)
    if spec["family"] == BAGGED and seed + spec["model_params"]["bag_count"] - 1 >= 2**32:
        raise ValueError("Bag member seeds would exceed uint32")
    for member in spec.get("members", ()):
        _validate_effective_seed(member, seed)


def validate_record_pipeline(spec: dict) -> dict:
    """Normalize a complete reproducible record recipe without fitting anything."""
    if not isinstance(spec, dict):
        raise ValueError("Record pipeline specification must be an object")
    value = deepcopy(spec)
    allowed = {"family", "base_method", "window_seconds", "feature_engineering", "preprocessing",
        "model_params", "training", "postprocessing", "members"}
    family = value.get("family")
    if set(value) - allowed or not isinstance(family, str) or family not in RECORD_FAMILIES:
        raise ValueError("Unknown record pipeline field or family")
    value.setdefault("base_method", "standard")
    value.setdefault("window_seconds", 3)
    value.setdefault("feature_engineering", {"mode": "all"})
    value.setdefault("preprocessing", {"scaler": "standard"})
    value.setdefault("model_params", {})
    value.setdefault("training", {})
    value.setdefault("postprocessing", {"kind": "argmax"})
    for key in ("feature_engineering", "preprocessing", "model_params", "training", "postprocessing"):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} must be an object")
    if value["base_method"] not in ("standard", "regularized"):
        raise ValueError("Unknown base method")
    if type(value["window_seconds"]) is not int or value["window_seconds"] != 3:
        raise ValueError("Record pipeline v1 requires the 3-second feature contract")
    for key, expected in (("feature_engineering", {"mode": "all"}),
            ("preprocessing", {"scaler": "standard"}), ("postprocessing", {"kind": "argmax"})):
        if value[key] != expected:
            raise ValueError(f"Record pipeline v1 requires {key}={expected}")
    training = value["training"]
    if set(training) - {"seed", "weighting", "data_policy"}:
        raise ValueError("Unknown record training fields")
    training.setdefault("seed", DEFAULT_SEED)
    required_weighting = "class_record_balanced" if family == BALANCED else "class_balanced"
    training.setdefault("weighting", required_weighting)
    if training["weighting"] != required_weighting:
        raise ValueError(f"{family} requires weighting={required_weighting}")
    _seed(training["seed"])
    if training.get("data_policy") not in (None, DATA_POLICY):
        raise ValueError("Unknown training-record policy")
    params = value["model_params"]
    if family == VOTING:
        members = value.get("members")
        if params or value["base_method"] != "standard" or not isinstance(members, list) or len(members) < 2:
            raise ValueError("Record voting requires at least two members and no outer model parameters")
        if any(not isinstance(member, dict) or member.get("family") not in (BALANCED, BAGGED) for member in members):
            raise ValueError("Record voting accepts only nonnested record-aware Extra Trees members")
        value["members"] = [validate_record_pipeline(member) for member in members]
        if any(member["training"].get("data_policy") != training.get("data_policy")
                or member["training"]["seed"] != training["seed"] for member in value["members"]):
            raise ValueError("Record voting members must share the training-record policy and primary seed")
        if len({json.dumps(member, sort_keys=True) for member in value["members"]}) != len(members):
            raise ValueError("Record voting members must have distinct recipes")
    else:
        if "members" in value:
            raise ValueError("Only record_soft_voting accepts members")
        defaults = {**TREE_DEFAULTS, **({"bag_count": 5} if family == BAGGED else {})}
        if set(params) - set(defaults):
            raise ValueError("Unknown record model parameters; seeds, weights and thread count are controlled by the recipe")
        for key, default in defaults.items():
            params.setdefault(key, default)
        for key in ("n_estimators", "min_samples_leaf"):
            if type(params[key]) is not int or params[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if params["max_depth"] is not None and (type(params["max_depth"]) is not int or params["max_depth"] < 1):
            raise ValueError("max_depth must be null or a positive integer")
        count = params["max_features"]
        valid_features = ((type(count) is int and 1 <= count <= 596)
            or (type(count) is float and np.isfinite(count) and 0 < count <= 1)
            or (isinstance(count, str) and count in ("sqrt", "log2")))
        if not valid_features:
            raise ValueError("max_features must be a supported positive feature count or fraction")
        if family == BAGGED:
            bags = params["bag_count"]
            if type(bags) is not int or bags < 2 or params["n_estimators"] % bags:
                raise ValueError("bag_count must be at least two and divide total n_estimators exactly")
    _validate_effective_seed(value, training["seed"])
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("Record recipe must contain finite JSON values") from error
    return value


def _features(X):
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] != 596 or not len(values)
            or values.dtype.kind not in "iuf" or not np.isfinite(values).all()):
        raise ValueError("Expected nonempty finite numeric features with shape (N, 596)")
    # The frozen feature cache is float32. Preserve StandardScaler's original
    # float32 fit/transform precision instead of introducing a second change.
    dtype = np.float32 if values.dtype == np.dtype(np.float32) else np.float64
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(values, dtype=dtype, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Features exceed the finite supported numeric range")
    return result


def _identities(values, count, name):
    identities = np.asarray(values)
    if identities.ndim != 1 or len(identities) != count:
        raise ValueError(f"{name} must be supplied for every training row")
    if any(not isinstance(item, str) or not item.strip() for item in identities.tolist()):
        raise ValueError(f"{name} must contain nonempty string identifiers")
    return np.array([unicodedata.normalize("NFC", item).strip() for item in identities.tolist()])


def _record_metadata(labels, groups, record_ids):
    people = _identities(groups, len(labels), "groups")
    records = _identities(record_ids, len(labels), "record_ids")
    result = []
    for record in sorted(set(records.tolist())):
        indices = np.flatnonzero(records == record)
        if len(set(people[indices])) != 1 or len(set(labels[indices])) != 1:
            raise ValueError("Each training record must belong to exactly one subject and one posture class")
        result.append({"record_id": record, "subject": str(people[indices[0]]),
            "class_id": int(labels[indices[0]]), "window_count": len(indices), "row_indices": indices.tolist()})
    return result


def _metadata_digest(metadata):
    return hashlib.sha256(json.dumps(metadata, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _scaler_digest(scaler):
    return _metadata_digest({"params": scaler.get_params(deep=False),
        "samples": int(scaler.n_samples_seen_), "features": int(scaler.n_features_in_),
        **{key: getattr(scaler, key).tolist() for key in ("mean_", "scale_", "var_")}})


def _weights(metadata, count):
    records_per_class = np.bincount([row["class_id"] for row in metadata], minlength=7)
    result = np.empty(count, dtype=float)
    record_totals = {}
    class_totals = np.zeros(7)
    for row in metadata:
        weight = count / (7 * records_per_class[row["class_id"]] * row["window_count"])
        result[row["row_indices"]] = weight
        record_totals[row["record_id"]] = float(weight * row["window_count"])
        class_totals[row["class_id"]] += record_totals[row["record_id"]]
    return result, {"formula": "N / (7 * records_in_class * windows_in_record)",
        "records_per_class": records_per_class.tolist(), "record_totals": record_totals,
        "class_totals": class_totals.tolist(), "minimum": float(result.min()), "maximum": float(result.max())}


def _draw_records(metadata, seed):
    random = np.random.default_rng(seed)
    lookup = {row["record_id"]: row for row in metadata}
    drawn = []
    for class_id in range(7):
        pool = sorted(row["record_id"] for row in metadata if row["class_id"] == class_id)
        drawn.extend(random.choice(pool, size=len(pool), replace=True).tolist())
    indices = np.array([index for record in drawn for index in lookup[record]["row_indices"]], dtype=np.int64)
    counts = dict(sorted(Counter(drawn).items()))
    class_counts = np.zeros(7, dtype=np.int64)
    for record in drawn:
        class_counts[lookup[record]["class_id"]] += lookup[record]["window_count"]
    return indices, {"seed": seed, "record_ids": drawn, "draw_counts": counts,
        "training_window_count": len(indices), "class_window_counts": class_counts.tolist(),
        "class_record_draw_counts": np.bincount([lookup[record]["class_id"] for record in drawn], minlength=7).tolist()}


def _estimator(spec, seed):
    params = deepcopy(spec["model_params"])
    if spec["family"] == BAGGED:
        params["n_estimators"] //= params.pop("bag_count")
    return ExtraTreesClassifier(**params, random_state=seed, n_jobs=1,
        class_weight=None if spec["family"] == BALANCED else "balanced")


class RecordIterativePipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, spec, seed=None):
        self.spec = spec
        self.seed = seed

    def fit(self, X, y, *, groups=None, record_ids=None, validation_data=None):
        spec = validate_record_pipeline(self.spec)
        seed = spec["training"]["seed"] if self.seed is None else self.seed
        _validate_effective_seed(spec, seed)
        values, labels = _features(X), np.asarray(y)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu"
                or not np.array_equal(np.unique(labels), np.arange(7))):
            raise ValueError("Record training requires matching labels containing all seven posture IDs 0..6")
        labels = labels.astype(np.int64, copy=True)
        metadata = _record_metadata(labels, groups, record_ids)
        family = spec["family"]
        with threadpool_limits(limits=1):
            if family == VOTING:
                members = [build_record_pipeline(member, seed=self.seed).fit(values, labels,
                    groups=groups, record_ids=record_ids) for member in spec["members"]]
            else:
                scaler = StandardScaler()
                transformed = scaler.fit_transform(values)
                if family == BALANCED:
                    weights, weighting_audit = _weights(metadata, len(values))
                    estimator = _estimator(spec, seed).fit(transformed, labels, sample_weight=weights)
                else:
                    estimators, draws, sample_indices = [], [], []
                    for member_index in range(spec["model_params"]["bag_count"]):
                        member_seed = seed + member_index
                        indices, draw = _draw_records(metadata, member_seed)
                        draw["member_index"] = member_index
                        estimators.append(_estimator(spec, member_seed).fit(transformed[indices], labels[indices]))
                        draws.append(draw)
                        sample_indices.append(indices)
        self.spec_, self.effective_seed_ = spec, seed
        self.classes_, self.n_features_in_ = np.arange(7), 596
        self.label_names_ = list(CLASS_NAMES)
        self.feature_names_in_ = np.asarray(feature_names())
        self.record_metadata_, self.training_identity_sha256_ = metadata, _metadata_digest(metadata)
        self.training_details_ = {"pipeline": deepcopy(spec), "effective_seed": seed,
            "input_contract": "feature_vector", "feature_names": feature_names(), "class_names": list(CLASS_NAMES),
            "training_windows": len(values), "training_feature_dtype": str(values.dtype),
            "training_records": len(metadata), "training_subjects": sorted({row["subject"] for row in metadata}),
            "class_counts": np.bincount(labels, minlength=7).tolist(),
            "record_metadata": deepcopy(metadata), "training_identity_sha256": self.training_identity_sha256_,
            "preprocessing_fit": "unweighted StandardScaler on all training rows before record weighting or record bootstrap",
            "external_validation_used": False, "training_record_policy": spec["training"].get("data_policy"),
            "training_record_policy_application": "caller supplies eligible records; historical caches are never imported by estimator",
            "threads": 1}
        if family == VOTING:
            self.members_ = members
            self.training_details_.update(combination="equal aligned probability mean",
                weights=[1 / len(members)] * len(members), members=[deepcopy(member.training_details_) for member in members])
        else:
            self.scaler_ = scaler
            self.scaler_state_sha256_ = _scaler_digest(scaler)
            self.training_details_["scaler_state_sha256"] = self.scaler_state_sha256_
            if family == BALANCED:
                self.estimator_, self.sample_weight_, self.record_weighting_audit_ = estimator, weights, weighting_audit
                self.training_details_["record_weighting"] = deepcopy(weighting_audit)
                self.training_details_["effective_model_params"] = estimator.get_params(deep=False)
            else:
                self.estimators_, self.bag_draws_, self.bag_sample_indices_ = estimators, draws, sample_indices
                self.training_details_.update(bag_draws=deepcopy(draws),
                    total_trees=spec["model_params"]["n_estimators"],
                    trees_per_bag=spec["model_params"]["n_estimators"] // spec["model_params"]["bag_count"],
                    backend_seeds=[seed + index for index in range(len(estimators))],
                    effective_model_params=[estimator.get_params(deep=False) for estimator in estimators])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "spec_")
        if validate_record_pipeline(self.spec) != self.spec_ or (
                self.spec_["training"]["seed"] if self.seed is None else self.seed) != self.effective_seed_:
            raise ValueError("Record recipe or seed changed after fitting")
        values = _features(X)
        with threadpool_limits(limits=1):
            if self.spec_["family"] == VOTING:
                result = np.mean([member.predict_proba(values) for member in self.members_], axis=0)
            else:
                transformed = self.scaler_.transform(values)
                models = [self.estimator_] if self.spec_["family"] == BALANCED else self.estimators_
                probabilities = []
                for estimator in models:
                    classes = np.asarray(estimator.classes_)
                    if classes.dtype.kind not in "iu" or not np.array_equal(np.sort(classes), self.classes_):
                        raise ValueError("Record estimator posture classes changed")
                    raw = np.asarray(estimator.predict_proba(transformed), dtype=float)
                    if raw.shape != (len(values), 7):
                        raise ValueError("Invalid record estimator probability shape")
                    aligned = np.empty_like(raw)
                    aligned[:, classes] = raw
                    probabilities.append(aligned)
                result = np.mean(probabilities, axis=0)
        if (result.shape != (len(values), 7) or not np.isfinite(result).all() or (result < 0).any()
                or not np.allclose(result.sum(1), 1., atol=1e-12, rtol=0)):
            raise ValueError("Invalid record estimator probabilities")
        return result

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def build_record_pipeline(spec: dict, seed: int | None = None):
    normalized = validate_record_pipeline(spec)
    _validate_effective_seed(normalized, normalized["training"]["seed"] if seed is None else seed)
    return RecordIterativePipeline(normalized, seed=seed)


def fit_record_pipeline(model, spec: dict, X, y, *, groups=None, record_ids=None, validation_data=None):
    normalized = validate_record_pipeline(spec)
    if not isinstance(model, RecordIterativePipeline) or validate_record_pipeline(model.spec) != normalized:
        raise ValueError("Training specification differs from the constructed record pipeline")
    return model.fit(X, y, groups=groups, record_ids=record_ids, validation_data=validation_data)


def verify_fitted_record_identity(model, spec: dict, seed: int):
    """Reject changed recipes, estimator parameters or record-sampling evidence."""
    recipe = validate_record_pipeline(spec)
    _validate_effective_seed(recipe, seed)
    if (not isinstance(model, RecordIterativePipeline) or getattr(model, "spec_", None) != recipe
            or validate_record_pipeline(model.spec) != recipe or getattr(model, "effective_seed_", None) != seed
            or (recipe["training"]["seed"] if model.seed is None else model.seed) != seed
            or getattr(model, "n_features_in_", None) != 596
            or not np.array_equal(getattr(model, "classes_", ()), np.arange(7))
            or getattr(model, "label_names_", None) != list(CLASS_NAMES)
            or not np.array_equal(getattr(model, "feature_names_in_", ()), feature_names())):
        raise ValueError("Fitted record pipeline identity differs from the declared recipe")
    metadata = getattr(model, "record_metadata_", None)
    details = getattr(model, "training_details_", {})
    try:
        metadata_hash = _metadata_digest(metadata)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid fitted record metadata") from error
    if (not isinstance(metadata, list) or not metadata or metadata_hash != getattr(model, "training_identity_sha256_", None)
            or details.get("record_metadata") != metadata or details.get("pipeline") != recipe
            or details.get("effective_seed") != seed or details.get("training_records") != len(metadata)
            or details.get("external_validation_used") is not False or details.get("class_names") != list(CLASS_NAMES)
            or details.get("training_record_policy") != recipe["training"].get("data_policy")
            or details.get("training_identity_sha256") != model.training_identity_sha256_):
        raise ValueError("Fitted record metadata or training audit changed")
    try:
        count = details["training_windows"]
        indices = [index for row in metadata for index in row["row_indices"]]
        if (type(count) is not int or sorted(indices) != list(range(count))
                or len({row["record_id"] for row in metadata}) != len(metadata)
                or any(not isinstance(row["record_id"], str) or not row["record_id"].strip()
                    or not isinstance(row["subject"], str) or not row["subject"].strip()
                    or type(row["class_id"]) is not int or row["class_id"] not in range(7)
                    or row["window_count"] != len(row["row_indices"]) or row["window_count"] < 1 for row in metadata)):
            raise ValueError("Invalid record row identities")
        class_counts = [sum(row["window_count"] for row in metadata if row["class_id"] == c) for c in range(7)]
        if (min(class_counts) < 1 or class_counts != details["class_counts"]
                or details.get("training_subjects") != sorted({row["subject"] for row in metadata})):
            raise ValueError("Fitted record class totals changed")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete fitted record metadata") from error
    family = recipe["family"]
    if family == VOTING:
        members = getattr(model, "members_", ())
        if len(members) != len(recipe["members"]):
            raise ValueError("Fitted record voting member count changed")
        for member, member_recipe in zip(members, recipe["members"]):
            verify_fitted_record_identity(member, member_recipe, seed)
            if member.record_metadata_ != metadata:
                raise ValueError("Record voting members trained on different record rows")
        return True
    scaler = getattr(model, "scaler_", None)
    if (type(scaler) is not StandardScaler or scaler.get_params(deep=False) != StandardScaler().get_params(deep=False)
            or getattr(scaler, "n_features_in_", None) != 596 or getattr(scaler, "n_samples_seen_", None) != count
            or any(np.asarray(getattr(scaler, key, ())).shape != (596,) or not np.isfinite(getattr(scaler, key)).all()
                for key in ("mean_", "scale_", "var_")) or (scaler.scale_ <= 0).any() or (scaler.var_ < 0).any()):
        raise ValueError("Fitted record scaler differs from the training-row contract")
    if (_scaler_digest(scaler) != getattr(model, "scaler_state_sha256_", None)
            or details.get("scaler_state_sha256") != model.scaler_state_sha256_):
        raise ValueError("Fitted record scaler statistics changed")
    if family == BALANCED:
        models, seeds = [getattr(model, "estimator_", None)], [seed]
        expected_weights, audit = _weights(metadata, count)
        if (not np.array_equal(getattr(model, "sample_weight_", ()), expected_weights)
                or getattr(model, "record_weighting_audit_", None) != audit or details.get("record_weighting") != audit):
            raise ValueError("Fitted class/record weights differ from the recorded training rows")
    else:
        models = getattr(model, "estimators_", ())
        bags = recipe["model_params"]["bag_count"]
        seeds = [seed + index for index in range(bags)]
        if (len(models) != bags or len(getattr(model, "bag_draws_", ())) != bags
                or len(getattr(model, "bag_sample_indices_", ())) != bags or details.get("backend_seeds") != seeds
                or details.get("bag_draws") != model.bag_draws_
                or details.get("total_trees") != recipe["model_params"]["n_estimators"]
                or details.get("trees_per_bag") != recipe["model_params"]["n_estimators"] // bags):
            raise ValueError("Fitted record bag count or seeds changed")
        for index, member_seed in enumerate(seeds):
            expected_indices, draw = _draw_records(metadata, member_seed)
            draw["member_index"] = index
            if (model.bag_draws_[index] != draw or not np.array_equal(model.bag_sample_indices_[index], expected_indices)
                    or details.get("bag_draws", [])[index] != draw):
                raise ValueError("Fitted record draws differ from the seeded whole-record bootstrap")
    for estimator, member_seed in zip(models, seeds):
        expected = _estimator(recipe, member_seed)
        if (type(estimator) is not ExtraTreesClassifier or estimator.get_params(deep=False) != expected.get_params(deep=False)
                or len(getattr(estimator, "estimators_", ())) != expected.n_estimators
                or getattr(estimator, "n_features_in_", None) != 596
                or not np.array_equal(getattr(estimator, "classes_", ()), np.arange(7))):
            raise ValueError("Fitted record Extra Trees parameters, seed, tree count or classes changed")
    expected_params = ([estimator.get_params(deep=False) for estimator in models] if family == BAGGED
        else models[0].get_params(deep=False))
    if details.get("effective_model_params") != expected_params:
        raise ValueError("Fitted record parameter audit changed")
    return True
