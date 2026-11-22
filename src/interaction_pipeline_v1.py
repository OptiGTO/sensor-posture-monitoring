"""Versioned train-scaled interaction features with deterministic Extra Trees.

All inputs are independent frozen sensor windows. Feature extraction and fixed
post-scaling transforms use no evaluation population statistics or labels.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import scipy
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .posture_core import CLASS_NAMES, SENSOR_COLUMNS, normalize_text
from .record_gain_features_v1 import validate_sequences
from .interaction_features_v1 import (interaction_feature_names, extract_interaction_features,
    transform_standardized_features, interaction_rule)


PIPELINE_VERSION = "interaction_pipeline_v1"
TREE = "interaction_extra_trees"
VOTING = "interaction_soft_voting"
INTERACTION_FAMILIES = frozenset({TREE, VOTING})
MODES = ("identity", "hadamard512", "temporal_corr120")
WIDTHS = {"identity": 596, "hadamard512": 596, "temporal_corr120": 716}
DATA_POLICY = "include_ordinary_reserved_training_records_v1"
TREE_DEFAULTS = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 8, "max_features": .5}


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Interaction seed must be a nonnegative uint32 integer")
    return value


def validate_interaction_pipeline(spec):
    """Return a complete immutable-choice recipe without inspecting any data."""
    if not isinstance(spec, dict):
        raise ValueError("Interaction specification must be an object")
    value = deepcopy(spec)
    family = value.get("family")
    if not isinstance(family, str) or family not in INTERACTION_FAMILIES or set(value) - {
            "family", "base_method", "window_seconds", "feature_engineering", "preprocessing",
            "model_params", "training", "postprocessing", "members"}:
        raise ValueError("Unknown interaction family or field")
    for key, default in (("base_method", "standard"), ("window_seconds", 3),
            ("feature_engineering", {"mode": "per_member" if family == VOTING else "identity"}),
            ("preprocessing", {"scaler": "standard"}), ("model_params", {}),
            ("training", {}), ("postprocessing", {"kind": "argmax"})):
        value.setdefault(key, deepcopy(default))
    for key in ("feature_engineering", "preprocessing", "model_params", "training", "postprocessing"):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} must be an object")
    if (value["base_method"] != "standard" or type(value["window_seconds"]) is not int or value["window_seconds"] != 3
            or value["preprocessing"] != {"scaler": "standard"} or value["postprocessing"] != {"kind": "argmax"}):
        raise ValueError("Interaction v1 requires 3-second windows, standard method/scaler and argmax")
    features, training, params = value["feature_engineering"], value["training"], value["model_params"]
    if set(features) != {"mode"} or features["mode"] not in (("per_member",) if family == VOTING else MODES):
        raise ValueError("Interaction feature mode must be a compatible predeclared mode")
    if set(training) - {"seed", "weighting", "data_policy"}:
        raise ValueError("Unknown interaction training fields")
    training.setdefault("seed", 20260921)
    training.setdefault("weighting", "per_member" if family == VOTING else "class_balanced")
    _seed(training["seed"])
    if (training["weighting"] != ("per_member" if family == VOTING else "class_balanced")
            or training.get("data_policy") not in (None, DATA_POLICY)):
        raise ValueError("Interaction weighting or training-record policy differs from the declared contract")
    if family == VOTING:
        members = value.get("members")
        if params or not isinstance(members, list) or len(members) < 2 or any(
                not isinstance(member, dict) or member.get("family") != TREE for member in members):
            raise ValueError("Interaction voting requires two or more nonnested Extra Trees members")
        value["members"] = [validate_interaction_pipeline(member) for member in members]
        if any(member["training"].get("data_policy") != training.get("data_policy")
                or member["training"]["seed"] != training["seed"] for member in value["members"]):
            raise ValueError("Interaction voting members must share primary seed and training-record policy")
        if len({json.dumps(member, sort_keys=True) for member in value["members"]}) != len(members):
            raise ValueError("Interaction voting members must have distinct recipes")
    else:
        if "members" in value or set(params) - set(TREE_DEFAULTS):
            raise ValueError("Unknown interaction model parameters or unexpected members")
        for key, default in TREE_DEFAULTS.items():
            params.setdefault(key, default)
        for key in ("n_estimators", "min_samples_leaf"):
            if type(params[key]) is not int or params[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if params["max_depth"] is not None and (type(params["max_depth"]) is not int or params["max_depth"] < 1):
            raise ValueError("max_depth must be null or a positive integer")
        count = params["max_features"]
        if not ((type(count) is int and 1 <= count <= WIDTHS[features["mode"]])
                or (type(count) is float and np.isfinite(count) and 0 < count <= 1)
                or (isinstance(count, str) and count in ("sqrt", "log2"))):
            raise ValueError("max_features must be a supported positive count or fraction")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Interaction recipe must contain finite JSON values") from error
    return value


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    values = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([values.dtype.str, list(values.shape)]).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _scaler_digest(scaler):
    return _json_digest({"params": scaler.get_params(deep=False), "samples": int(scaler.n_samples_seen_),
        "features": int(scaler.n_features_in_), **{key: getattr(scaler, key).tolist() for key in ("mean_", "scale_", "var_")}})


def _forest_digest(estimator):
    states = []
    for tree in estimator.estimators_:
        state = tree.tree_.__getstate__()
        # Hash fields separately: structured-array padding bytes are not learned
        # parameters and need not be initialized identically by every process.
        states.append({"seed": tree.random_state, "node_count": state["node_count"],
            "max_depth": state["max_depth"], "values": _array_digest(state["values"]),
            "nodes": {name: _array_digest(state["nodes"][name]) for name in state["nodes"].dtype.names}})
    return _json_digest(states)


def _estimator(spec, seed):
    return ExtraTreesClassifier(**deepcopy(spec["model_params"]), class_weight="balanced", n_jobs=1, random_state=seed)


class InteractionIterativePipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, spec, seed=None):
        self.spec = spec
        self.seed = seed

    def fit(self, X, y, *, groups=None, validation_data=None):
        spec = validate_interaction_pipeline(self.spec)
        seed = _seed(spec["training"]["seed"] if self.seed is None else self.seed)
        values, labels = validate_sequences(X), np.asarray(y)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu"
                or not np.array_equal(np.unique(labels), np.arange(7))):
            raise ValueError("Interaction training requires aligned integer labels with all seven posture IDs")
        labels = labels.astype(np.int64, copy=True)
        people = None
        if groups is not None:
            people = np.asarray(groups)
            if people.ndim != 1 or len(people) != len(values) or any(
                    not isinstance(person, str) or not normalize_text(person) for person in people.tolist()):
                raise ValueError("Interaction training groups must contain aligned nonempty subject names")
            people = [normalize_text(person) for person in people.tolist()]
        with threadpool_limits(limits=1):
            if spec["family"] == VOTING:
                members = [build_interaction_pipeline(member, seed=self.seed).fit(values, labels, groups=groups)
                    for member in spec["members"]]
            else:
                mode = spec["feature_engineering"]["mode"]
                rule = interaction_rule(mode)
                extracted = extract_interaction_features(values, mode)
                scaler = StandardScaler()
                scaled = scaler.fit_transform(extracted)
                transformed = transform_standardized_features(scaled, mode)
                estimator = _estimator(spec, seed).fit(transformed, labels)
        self.spec_, self.effective_seed_, self.classes_ = spec, seed, np.arange(7)
        self.input_shape_, self.n_features_in_, self.sensor_columns_ = (36, 16), 16, list(SENSOR_COLUMNS)
        self.label_names_, self.class_counts_ = list(CLASS_NAMES), np.bincount(labels, minlength=7)
        self.training_subjects_ = None if people is None else sorted(set(people))
        self.training_input_sha256_, self.training_labels_sha256_ = _array_digest(values), _array_digest(labels)
        self.training_groups_sha256_ = _json_digest(people)
        self.training_details_ = {"pipeline": deepcopy(spec), "pipeline_version": PIPELINE_VERSION, "effective_seed": seed,
            "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "input_contract": "sensor_sequence", "input_shape": [36, 16], "sensor_columns": list(SENSOR_COLUMNS),
            "sampling_rate": 12, "training_windows": len(values), "training_subjects": deepcopy(self.training_subjects_),
            "class_counts": self.class_counts_.tolist(), "class_names": list(CLASS_NAMES),
            "training_input_sha256": self.training_input_sha256_, "training_labels_sha256": self.training_labels_sha256_,
            "training_groups_sha256": self.training_groups_sha256_, "external_validation_used": False,
            "training_record_policy": spec["training"].get("data_policy"),
            "training_record_policy_application": "caller supplies eligible training rows; no historical cache import",
            "preprocessing_fit": "window-local extraction; unweighted StandardScaler fit on TRAIN rows; fixed feature transform after scaling",
            "evaluation_population_normalization": False, "threads": 1}
        if spec["family"] == VOTING:
            self.members_ = members
            self.training_details_.update(combination="equal aligned probability mean", weights=[1 / len(members)] * len(members),
                members=[deepcopy(member.training_details_) for member in members])
        else:
            self.scaler_, self.estimator_, self.interaction_rule_ = scaler, estimator, deepcopy(rule)
            self.extraction_feature_names_ = np.asarray(rule["extraction_feature_names"])
            self.feature_names_out_ = np.asarray(interaction_feature_names(mode))
            self.n_extracted_features_, self.n_transformed_features_ = extracted.shape[1], transformed.shape[1]
            self.extraction_state_sha256_, self.transformed_state_sha256_ = _array_digest(extracted), _array_digest(transformed)
            self.scaler_state_sha256_, self.estimator_state_sha256_ = _scaler_digest(scaler), _forest_digest(estimator)
            self.training_details_.update(interaction_rule=deepcopy(rule), extraction_feature_names=self.extraction_feature_names_.tolist(),
                transformed_feature_names=self.feature_names_out_.tolist(), extraction_features=self.n_extracted_features_,
                transformed_features=self.n_transformed_features_, model_input_dtype=str(transformed.dtype),
                extraction_state_sha256=self.extraction_state_sha256_, transformed_state_sha256=self.transformed_state_sha256_,
                scaler_state_sha256=self.scaler_state_sha256_, estimator_state_sha256=self.estimator_state_sha256_,
                effective_model_params=estimator.get_params(deep=False))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "spec_")
        if validate_interaction_pipeline(self.spec) != self.spec_ or _seed(
                self.spec_["training"]["seed"] if self.seed is None else self.seed) != self.effective_seed_:
            raise ValueError("Interaction recipe or seed changed after fitting")
        values = validate_sequences(X)
        with threadpool_limits(limits=1):
            if self.spec_["family"] == VOTING:
                probability = np.mean([member.predict_proba(values) for member in self.members_], axis=0)
            else:
                mode = self.spec_["feature_engineering"]["mode"]
                extracted = extract_interaction_features(values, mode)
                transformed = transform_standardized_features(self.scaler_.transform(extracted), mode)
                classes = np.asarray(self.estimator_.classes_)
                if classes.dtype.kind not in "iu" or not np.array_equal(np.sort(classes), np.arange(7)):
                    raise ValueError("Interaction backend posture classes changed")
                raw = np.asarray(self.estimator_.predict_proba(transformed), dtype=float)
                if raw.shape != (len(values), 7):
                    raise ValueError("Invalid interaction backend probability shape")
                probability = np.empty_like(raw)
                probability[:, classes] = raw
        if (probability.shape != (len(values), 7) or not np.isfinite(probability).all() or (probability < 0).any()
                or not np.allclose(probability.sum(1), 1., atol=1e-12, rtol=0)):
            raise ValueError("Invalid interaction probabilities")
        return probability

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def build_interaction_pipeline(spec, seed=None):
    if seed is not None:
        _seed(seed)
    return InteractionIterativePipeline(validate_interaction_pipeline(spec), seed=seed)


def fit_interaction_pipeline(model, spec, X, y, *, groups=None, validation_data=None):
    recipe = validate_interaction_pipeline(spec)
    if not isinstance(model, InteractionIterativePipeline) or validate_interaction_pipeline(model.spec) != recipe:
        raise ValueError("Constructed interaction estimator and requested recipe differ")
    return model.fit(X, y, groups=groups, validation_data=validation_data)


def verify_fitted_interaction_identity(model, spec, seed):
    """Check fixed feature rules, complete fitted scaler/trees and training audit."""
    recipe, seed = validate_interaction_pipeline(spec), _seed(seed)
    if (type(model) is not InteractionIterativePipeline or getattr(model, "spec_", None) != recipe
            or validate_interaction_pipeline(model.spec) != recipe or getattr(model, "effective_seed_", None) != seed
            or (recipe["training"]["seed"] if model.seed is None else model.seed) != seed
            or getattr(model, "input_shape_", None) != (36, 16) or getattr(model, "n_features_in_", None) != 16
            or not np.array_equal(getattr(model, "classes_", ()), np.arange(7))
            or getattr(model, "label_names_", None) != list(CLASS_NAMES)
            or getattr(model, "sensor_columns_", None) != list(SENSOR_COLUMNS)):
        raise ValueError("Fitted interaction recipe, seed or sensor/class contract changed")
    details = getattr(model, "training_details_", {})
    count, counts = details.get("training_windows"), np.asarray(getattr(model, "class_counts_", ()))
    if (details.get("pipeline") != recipe or details.get("pipeline_version") != PIPELINE_VERSION or details.get("effective_seed") != seed
            or details.get("sklearn_version") != sklearn.__version__ or details.get("numpy_version") != np.__version__
            or details.get("scipy_version") != scipy.__version__ or details.get("external_validation_used") is not False
            or details.get("evaluation_population_normalization") is not False or details.get("threads") != 1
            or type(count) is not int or counts.shape != (7,) or counts.dtype.kind not in "iu"
            or (counts < 1).any() or counts.sum() != count or details.get("class_counts") != counts.tolist()
            or details.get("class_names") != list(CLASS_NAMES) or details.get("input_shape") != [36, 16]
            or details.get("sensor_columns") != list(SENSOR_COLUMNS) or details.get("input_contract") != "sensor_sequence"
            or details.get("training_subjects") != getattr(model, "training_subjects_", None)
            or details.get("training_record_policy") != recipe["training"].get("data_policy")):
        raise ValueError("Interaction training audit or library versions differ from fitted state")
    for name in ("training_input_sha256", "training_labels_sha256", "training_groups_sha256"):
        digest = getattr(model, name + "_", None)
        if (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or details.get(name) != digest):
            raise ValueError("Interaction training scope digest changed")
    if recipe["family"] == VOTING:
        members = getattr(model, "members_", ())
        if len(members) != len(recipe["members"]):
            raise ValueError("Interaction voting member count changed")
        for member, member_recipe in zip(members, recipe["members"]):
            verify_fitted_interaction_identity(member, member_recipe, seed)
            if any(member.training_details_[key] != details[key] for key in ("training_input_sha256", "training_labels_sha256",
                    "training_groups_sha256", "class_counts", "training_subjects", "training_windows")):
                raise ValueError("Interaction members trained on different rows or subject/class scopes")
        if (details.get("members") != [member.training_details_ for member in members]
                or details.get("weights") != [1 / len(members)] * len(members)
                or details.get("combination") != "equal aligned probability mean"):
            raise ValueError("Interaction ensemble audit differs from the fitted members")
        return True
    mode = recipe["feature_engineering"]["mode"]
    rule = interaction_rule(mode)
    if (getattr(model, "interaction_rule_", None) != rule or details.get("interaction_rule") != rule
            or not np.array_equal(getattr(model, "extraction_feature_names_", ()), rule["extraction_feature_names"])
            or not np.array_equal(getattr(model, "feature_names_out_", ()), interaction_feature_names(mode))
            or details.get("extraction_feature_names") != rule["extraction_feature_names"]
            or details.get("transformed_feature_names") != np.asarray(interaction_feature_names(mode)).tolist()
            or getattr(model, "n_extracted_features_", None) != WIDTHS[mode]
            or getattr(model, "n_transformed_features_", None) != WIDTHS[mode]
            or details.get("extraction_features") != WIDTHS[mode] or details.get("transformed_features") != WIDTHS[mode]
            or details.get("model_input_dtype") != "float32"):
        raise ValueError("Interaction feature rule, names, widths or precision changed")
    for name in ("extraction_state_sha256", "transformed_state_sha256"):
        digest = getattr(model, name + "_", None)
        if (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or details.get(name) != digest):
            raise ValueError("Interaction feature training digest changed")
    scaler, estimator = getattr(model, "scaler_", None), getattr(model, "estimator_", None)
    expected = _estimator(recipe, seed)
    try:
        if (type(scaler) is not StandardScaler or scaler.get_params(deep=False) != StandardScaler().get_params(deep=False)
                or scaler.n_features_in_ != WIDTHS[mode] or scaler.n_samples_seen_ != count
                or _scaler_digest(scaler) != getattr(model, "scaler_state_sha256_", None)
                or details.get("scaler_state_sha256") != model.scaler_state_sha256_
                or type(estimator) is not ExtraTreesClassifier or estimator.get_params(deep=False) != expected.get_params(deep=False)
                or details.get("effective_model_params") != expected.get_params(deep=False)
                or estimator.n_features_in_ != WIDTHS[mode] or not np.array_equal(estimator.classes_, np.arange(7))
                or len(estimator.estimators_) != recipe["model_params"]["n_estimators"]
                or any(tree.tree_.n_node_samples[0] != count for tree in estimator.estimators_)
                or _forest_digest(estimator) != getattr(model, "estimator_state_sha256_", None)
                or details.get("estimator_state_sha256") != model.estimator_state_sha256_):
            raise ValueError("Interaction scaler, fitted trees or backend parameters changed")
    except (AttributeError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Incomplete fitted interaction scaler or forest state") from error
    return True
