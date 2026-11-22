"""TRAIN-only learned-coordinate Extra Trees recipes with immutable state.

The original feature scaler is shared within a leaf. PCA64 replaces its axes;
LDA6 adds supervised axes to all standardized original features. Neither uses
external validation, and identity delegates unchanged standardized features.
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

from .posture_core import CLASS_NAMES, feature_names, normalize_text
from .learned_projection_v1 import (LearnedProjectionV1, learned_projection_rule,
    learned_projection_state_sha256, verify_learned_projection)
from sklearn.tree import ExtraTreeClassifier


PIPELINE_VERSION = "learned_projection_pipeline_v1"
TREE = "projection_extra_trees"
VOTING = "projection_soft_voting"
PROJECTION_FAMILIES = frozenset({TREE, VOTING})
DATA_POLICY = "include_ordinary_reserved_training_records_v1"
TREE_DEFAULTS = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 8, "max_features": .5}


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Projection pipeline seed must be a nonnegative uint32 integer")
    return value


def validate_projection_pipeline(spec):
    """Normalize fixed projection and native forest settings."""
    if not isinstance(spec, dict):
        raise ValueError("Projection pipeline specification must be an object")
    value = deepcopy(spec)
    family = value.get("family")
    if not isinstance(family, str) or family not in PROJECTION_FAMILIES or set(value) - {
            "family", "base_method", "window_seconds", "feature_engineering", "preprocessing",
            "model_params", "training", "postprocessing", "members"}:
        raise ValueError("Unknown projection pipeline family or field")
    for key, default in (("base_method", "standard"), ("window_seconds", 3),
            ("feature_engineering", {"mode": "per_member" if family == VOTING else "identity"}), ("preprocessing", {"scaler": "standard"}),
            ("model_params", {}), ("training", {}),
            ("postprocessing", {"kind": "argmax"})):
        value.setdefault(key, deepcopy(default))
    for key in ("feature_engineering", "preprocessing", "model_params", "training", "postprocessing"):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} must be an object")
    if (value["base_method"] != "standard" or type(value["window_seconds"]) is not int or value["window_seconds"] != 3
            or value["preprocessing"] != {"scaler": "standard"} or value["postprocessing"] != {"kind": "argmax"}):
        raise ValueError("Projection v1 requires 3-second canonical596 input, StandardScaler and argmax")
    mode = value["feature_engineering"].get("mode")
    if (set(value["feature_engineering"]) != {"mode"} or not isinstance(mode, str)
            or mode not in ({"per_member"} if family == VOTING else {"identity", "pca64", "all_plus_lda6"})):
        raise ValueError("Unknown or inappropriate fixed projection mode")
    params, training, post = value["model_params"], value["training"], value["postprocessing"]
    if set(training) - {"seed", "weighting", "data_policy"}:
        raise ValueError("Unknown projection training fields")
    training.setdefault("seed", 20260921)
    training.setdefault("weighting", "per_member" if family == VOTING else "class_balanced")
    _seed(training["seed"])
    if (training["weighting"] != ("per_member" if family == VOTING else "class_balanced")
            or training.get("data_policy") not in (None, DATA_POLICY)):
        raise ValueError("Projection weighting or training-record policy differs from the family contract")
    if family == VOTING:
        members = value.get("members")
        if params or post != {"kind": "argmax"} or not isinstance(members, list) or len(members) < 2 or any(
                not isinstance(member, dict) or member.get("family") != TREE for member in members):
            raise ValueError("Projection voting requires two or more nonnested forest members and argmax")
        value["members"] = [validate_projection_pipeline(member) for member in members]
        if any(member["training"].get("data_policy") != training.get("data_policy")
                or member["training"]["seed"] != training["seed"] for member in value["members"]):
            raise ValueError("Projection voting members must share training policy and primary seed")
        if len({json.dumps(member, sort_keys=True) for member in value["members"]}) != len(members):
            raise ValueError("Projection voting members must have distinct recipes")
    else:
        if "members" in value or set(params) - set(TREE_DEFAULTS):
            raise ValueError("Unknown projection forest parameters or unexpected members")
        for key, default in TREE_DEFAULTS.items():
            params.setdefault(key, default)
        for key in ("n_estimators", "min_samples_leaf"):
            if type(params[key]) is not int or params[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if params["max_depth"] is not None and (type(params["max_depth"]) is not int or params["max_depth"] < 1):
            raise ValueError("max_depth must be null or a positive integer")
        count = params["max_features"]
        if not ((type(count) is int and 1 <= count <= {"identity": 596, "pca64": 64, "all_plus_lda6": 602}[mode])
                or (type(count) is float and np.isfinite(count) and 0 < count <= 1)
                or (isinstance(count, str) and count in ("sqrt", "log2"))):
            raise ValueError("max_features must be a supported count or fraction")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Projection recipe must contain finite JSON values") from error
    return value


def _features(X):
    if hasattr(X, "columns") and list(X.columns) != feature_names():
        raise ValueError("Named projection input columns differ from the canonical feature schema")
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] != 596 or not len(values)
            or values.dtype.kind not in "iuf" or not np.isfinite(values).all()):
        raise ValueError("Expected nonempty finite numeric features with shape (N,596)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(values, dtype=np.float32 if values.dtype == np.dtype(np.float32) else np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Features exceed the supported finite numeric range")
    return result


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


def _forest_digest(estimator, dimensions):
    if (type(estimator) is not ExtraTreesClassifier or estimator.n_features_in_ != dimensions or estimator.n_outputs_ != 1
            or not np.array_equal(estimator.classes_, np.arange(7)) or len(estimator.estimators_) != estimator.n_estimators):
        raise ValueError("Projection forest fitted class or feature contract changed")
    trees = []
    for tree in estimator.estimators_:
        if (type(tree) is not ExtraTreeClassifier or tree.n_features_in_ != dimensions or tree.n_outputs_ != 1
                or not np.array_equal(tree.classes_, np.arange(7))):
            raise ValueError("Projection tree fitted class or feature contract changed")
        state = tree.tree_.__getstate__()
        trees.append({"seed": int(tree.random_state), "classes": _array_digest(tree.classes_),
            "n_classes": int(tree.n_classes_), "n_features": int(tree.n_features_in_), "n_outputs": int(tree.n_outputs_),
            "node_count": int(state["node_count"]), "max_depth": int(state["max_depth"]),
            "values": _array_digest(state["values"]),
            "nodes": {name: _array_digest(state["nodes"][name]) for name in state["nodes"].dtype.names}})
    return _json_digest({"classes": _array_digest(estimator.classes_), "trees": trees})


def _estimator(spec, seed):
    return ExtraTreesClassifier(**deepcopy(spec["model_params"]), class_weight="balanced", n_jobs=1, random_state=seed)


class LearnedProjectionPipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, spec, seed=None):
        self.spec = spec
        self.seed = seed

    def fit(self, X, y, *, groups=None, validation_data=None):
        spec = validate_projection_pipeline(self.spec)
        seed = _seed(spec["training"]["seed"] if self.seed is None else self.seed)
        values, labels = _features(X), np.asarray(y)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu"
                or not np.array_equal(np.unique(labels), np.arange(7))):
            raise ValueError("Projection training requires aligned integer labels with all seven posture classes")
        labels = labels.astype(np.int64, copy=True)
        people = None
        if groups is not None:
            people = np.asarray(groups)
            if people.ndim != 1 or len(people) != len(values) or any(
                    not isinstance(person, str) or not normalize_text(person) for person in people.tolist()):
                raise ValueError("Projection training groups must be aligned nonempty subject names")
            people = [normalize_text(person) for person in people.tolist()]
        with threadpool_limits(limits=1):
            if spec["family"] == VOTING:
                members = [build_projection_pipeline(member, seed=self.seed).fit(values, labels, groups=groups)
                    for member in spec["members"]]
            else:
                scaler = StandardScaler()
                standardized = scaler.fit_transform(values)
                projector = LearnedProjectionV1(spec["feature_engineering"]["mode"]).fit(standardized, labels)
                transformed = projector.transform(standardized)
                estimator = _estimator(spec, seed).fit(transformed, labels)
        self.spec_, self.effective_seed_, self.classes_ = spec, seed, np.arange(7)
        self.n_features_in_, self.feature_names_in_, self.label_names_ = 596, np.asarray(feature_names()), list(CLASS_NAMES)
        self.class_counts_, self.training_subjects_ = np.bincount(labels, minlength=7), None if people is None else sorted(set(people))
        self.training_input_sha256_, self.training_labels_sha256_ = _array_digest(values), _array_digest(labels)
        self.training_groups_sha256_, self.feature_schema_sha256_ = _json_digest(people), _json_digest(feature_names())
        self.training_details_ = {"pipeline": deepcopy(spec), "pipeline_version": PIPELINE_VERSION, "effective_seed": seed,
            "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "input_contract": "feature_vector", "input_features": 596, "feature_names": feature_names(),
            "feature_schema_sha256": self.feature_schema_sha256_, "training_feature_dtype": str(values.dtype),
            "training_windows": len(values), "training_subjects": deepcopy(self.training_subjects_),
            "class_counts": self.class_counts_.tolist(), "class_names": list(CLASS_NAMES),
            "training_input_sha256": self.training_input_sha256_, "training_labels_sha256": self.training_labels_sha256_,
            "training_groups_sha256": self.training_groups_sha256_, "external_validation_used": False,
            "training_record_policy": spec["training"].get("data_policy"),
            "training_record_policy_application": "caller supplies eligible training rows; no historical cache import",
            "preprocessing_fit": "unweighted TRAIN StandardScaler; fixed learned projection fit only those same TRAIN rows",
            "prediction_dtype": "stored scaler then stored projector; native ExtraTrees converts final coordinates to float32",
            "evaluation_population_normalization": False, "threads": 1}
        if spec["family"] == VOTING:
            self.members_ = members
            self.training_details_.update(combination="equal aligned probability mean", weights=[1 / len(members)] * len(members),
                members=[deepcopy(member.training_details_) for member in members])
        else:
            self.scaler_, self.projector_, self.estimator_ = scaler, projector, estimator
            self.projection_rule_ = learned_projection_rule(spec["feature_engineering"]["mode"])
            self.projector_state_sha256_ = learned_projection_state_sha256(projector)
            self.scaler_state_sha256_, self.estimator_state_sha256_ = _scaler_digest(scaler), _forest_digest(estimator, transformed.shape[1])
            self.standardized_state_sha256_, self.transformed_state_sha256_ = _array_digest(standardized), _array_digest(transformed)
            self.training_details_.update(projection_rule=deepcopy(self.projection_rule_), scaler_state_sha256=self.scaler_state_sha256_,
                projector_state_sha256=self.projector_state_sha256_, learned_projection=deepcopy(projector.training_details_),
                estimator_state_sha256=self.estimator_state_sha256_, transformed_state_sha256=self.transformed_state_sha256_,
                standardized_state_sha256=self.standardized_state_sha256_, output_features=int(transformed.shape[1]),
                model_input_dtype=str(transformed.dtype), effective_model_params=estimator.get_params(deep=False))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "spec_")
        if validate_projection_pipeline(self.spec) != self.spec_ or _seed(
                self.spec_["training"]["seed"] if self.seed is None else self.seed) != self.effective_seed_:
            raise ValueError("Projection recipe or seed changed after fitting")
        values = _features(X)
        with threadpool_limits(limits=1):
            if self.spec_["family"] == VOTING:
                probability = np.mean([member.predict_proba(values) for member in self.members_], axis=0)
            else:
                transformed = self.projector_.transform(self.scaler_.transform(values))
                probability = self.estimator_.predict_proba(transformed)
        if (probability.shape != (len(values), 7) or not np.isfinite(probability).all() or (probability < 0).any()
                or not np.allclose(probability.sum(1), 1., atol=1e-12, rtol=0)):
            raise ValueError("Invalid projection pipeline probabilities")
        return probability

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def build_projection_pipeline(spec, seed=None):
    if seed is not None:
        _seed(seed)
    return LearnedProjectionPipeline(validate_projection_pipeline(spec), seed=seed)


def fit_projection_pipeline(model, spec, X, y, *, groups=None, validation_data=None):
    recipe = validate_projection_pipeline(spec)
    if not isinstance(model, LearnedProjectionPipeline) or validate_projection_pipeline(model.spec) != recipe:
        raise ValueError("Constructed projection estimator and requested recipe differ")
    return model.fit(X, y, groups=groups, validation_data=validation_data)


def verify_fitted_projection_identity(model, spec, seed):
    """Check learned forest/scaler, rule, exact feature schema and audit links."""
    recipe, seed = validate_projection_pipeline(spec), _seed(seed)
    if (type(model) is not LearnedProjectionPipeline or getattr(model, "spec_", None) != recipe
            or validate_projection_pipeline(model.spec) != recipe or getattr(model, "effective_seed_", None) != seed
            or (recipe["training"]["seed"] if model.seed is None else model.seed) != seed
            or getattr(model, "n_features_in_", None) != 596
            or not np.array_equal(getattr(model, "classes_", ()), np.arange(7))
            or getattr(model, "label_names_", None) != list(CLASS_NAMES)
            or not np.array_equal(getattr(model, "feature_names_in_", ()), feature_names())
            or getattr(model, "feature_schema_sha256_", None) != _json_digest(feature_names())):
        raise ValueError("Fitted projection recipe, seed or feature/class contract changed")
    details = getattr(model, "training_details_", {})
    count, counts = details.get("training_windows"), np.asarray(getattr(model, "class_counts_", ()))
    if (details.get("pipeline") != recipe or details.get("pipeline_version") != PIPELINE_VERSION or details.get("effective_seed") != seed
            or details.get("sklearn_version") != sklearn.__version__ or details.get("numpy_version") != np.__version__
            or details.get("scipy_version") != scipy.__version__ or details.get("external_validation_used") is not False
            or details.get("evaluation_population_normalization") is not False or details.get("threads") != 1
            or type(count) is not int or counts.shape != (7,) or counts.dtype.kind not in "iu"
            or (counts < 1).any() or counts.sum() != count or details.get("class_counts") != counts.tolist()
            or details.get("class_names") != list(CLASS_NAMES) or details.get("feature_names") != feature_names()
            or details.get("feature_schema_sha256") != model.feature_schema_sha256_
            or details.get("input_contract") != "feature_vector" or details.get("input_features") != 596
            or details.get("training_feature_dtype") not in ("float32", "float64")
            or details.get("training_subjects") != getattr(model, "training_subjects_", None)
            or details.get("training_record_policy") != recipe["training"].get("data_policy")):
        raise ValueError("Projection training audit or library versions differ from fitted state")
    for name in ("training_input_sha256", "training_labels_sha256", "training_groups_sha256"):
        _verify_digest_link(model, details, name)
    if recipe["family"] == VOTING:
        members = getattr(model, "members_", ())
        if len(members) != len(recipe["members"]):
            raise ValueError("Projection voting member count changed")
        for member, member_recipe in zip(members, recipe["members"]):
            verify_fitted_projection_identity(member, member_recipe, seed)
            if any(member.training_details_[key] != details[key] for key in ("training_input_sha256", "training_labels_sha256",
                    "training_groups_sha256", "class_counts", "training_subjects", "training_windows", "training_feature_dtype")):
                raise ValueError("Projection members trained on different input/class/subject scopes")
        if (details.get("members") != [member.training_details_ for member in members]
                or details.get("weights") != [1 / len(members)] * len(members)
                or details.get("combination") != "equal aligned probability mean"):
            raise ValueError("Projection ensemble audit differs from its fitted members")
        return True
    mode = recipe["feature_engineering"]["mode"]
    output_features = {"identity": 596, "pca64": 64, "all_plus_lda6": 602}[mode]
    rule = learned_projection_rule(mode)
    if getattr(model, "projection_rule_", None) != rule or details.get("projection_rule") != rule:
        raise ValueError("Fitted projection scoring rule changed")
    for name in ("standardized_state_sha256", "transformed_state_sha256"):
        _verify_digest_link(model, details, name)
    projector = getattr(model, "projector_", None)
    verify_learned_projection(projector, mode)
    if (projector.n_samples_seen_ != count or projector.training_input_sha256_ != model.standardized_state_sha256_
            or projector.training_labels_sha256_ != model.training_labels_sha256_
            or projector.training_dtype_ != details["training_feature_dtype"]
            or projector.n_features_out_ != output_features or details.get("output_features") != output_features
            or projector.rule_ != rule or details.get("learned_projection") != projector.training_details_
            or learned_projection_state_sha256(projector) != getattr(model, "projector_state_sha256_", None)
            or details.get("projector_state_sha256") != model.projector_state_sha256_
            or mode == "identity" and model.standardized_state_sha256_ != model.transformed_state_sha256_):
        raise ValueError("Projection helper or TRAIN input/label scope changed")
    scaler, estimator = getattr(model, "scaler_", None), getattr(model, "estimator_", None)
    expected = _estimator(recipe, seed)
    try:
        if (type(scaler) is not StandardScaler or scaler.get_params(deep=False) != StandardScaler().get_params(deep=False)
                or scaler.n_features_in_ != 596 or scaler.n_samples_seen_ != count
                or _scaler_digest(scaler) != getattr(model, "scaler_state_sha256_", None)
                or details.get("scaler_state_sha256") != model.scaler_state_sha256_
                or details.get("model_input_dtype") != (details["training_feature_dtype"] if mode == "identity" else "float64")
                or type(estimator) is not ExtraTreesClassifier or estimator.get_params(deep=False) != expected.get_params(deep=False)
                or details.get("effective_model_params") != expected.get_params(deep=False)
                or estimator.n_features_in_ != output_features or not np.array_equal(estimator.classes_, np.arange(7))
                or len(estimator.estimators_) != recipe["model_params"]["n_estimators"]
                or any(tree.tree_.n_node_samples[0] != count for tree in estimator.estimators_)
                or _forest_digest(estimator, output_features) != getattr(model, "estimator_state_sha256_", None)
                or details.get("estimator_state_sha256") != model.estimator_state_sha256_):
            raise ValueError("Projection scaler, fitted forest or backend parameters changed")
    except (AttributeError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Incomplete fitted projection scaler or forest state") from error
    return True


def _verify_digest_link(model, details, name):
    digest = getattr(model, name + "_", None)
    if (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            or details.get(name) != digest):
        raise ValueError("Projection training scope or transformed-state digest changed")
