"""Versioned training-only tail clipping and one-vs-rest forest recipes.

The clipping cutpoints and shared StandardScaler see only supplied training
rows. Binary-head normalization is a fixed scoring rule, not calibration.
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
from sklearn.tree import ExtraTreeClassifier
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .posture_core import CLASS_NAMES, feature_names, normalize_text
from .tail_clipping_v1 import (TailClipperV1, tail_clipping_rule,
    tail_clipping_state_sha256, verify_tail_clipper)
from .binary_forest_v1 import (BinaryForestV1, binary_forest_rule,
    binary_forest_state_sha256, verify_binary_forest)


PIPELINE_VERSION = "forest_alternatives_pipeline_v1"
TAIL, OVR, VOTING = "tail_extra_trees", "ovr_extra_trees", "alternatives_soft_voting"
FAMILIES = frozenset({TAIL, OVR, VOTING})
DATA_POLICY = "include_ordinary_reserved_training_records_v1"
TREE_DEFAULTS = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 8, "max_features": .5}


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Alternatives seed must be a nonnegative uint32 integer")
    return value


def validate_alternatives_pipeline(spec):
    """Normalize the complete recipe without consulting data or evaluation."""
    if not isinstance(spec, dict):
        raise ValueError("Alternatives recipe must be an object")
    value = deepcopy(spec)
    family = value.get("family")
    if not isinstance(family, str) or family not in FAMILIES or set(value) - {
            "family", "base_method", "window_seconds", "feature_engineering", "preprocessing",
            "model_params", "training", "postprocessing", "members"}:
        raise ValueError("Unknown alternatives family or recipe field")
    for name, default in (("base_method", "standard"), ("window_seconds", 3),
            ("feature_engineering", {"mode": "all"}),
            ("preprocessing", {"scaler": "standard", "tail_fraction": 0.} if family == TAIL else {"scaler": "standard"}),
            ("model_params", {}), ("training", {}),
            ("postprocessing", {"kind": "ovr_normalized", "zero_row": "uniform"} if family == OVR else {"kind": "argmax"})):
        value.setdefault(name, deepcopy(default))
    for name in ("feature_engineering", "preprocessing", "model_params", "training", "postprocessing"):
        if not isinstance(value[name], dict):
            raise ValueError(f"{name} must be an object")
    if (value["base_method"] != "standard" or type(value["window_seconds"]) is not int or value["window_seconds"] != 3
            or value["feature_engineering"] != {"mode": "all"}):
        raise ValueError("Alternatives v1 requires standard method and all596 features from 3-second windows")
    pre, training, params, post = (value[name] for name in ("preprocessing", "training", "model_params", "postprocessing"))
    if family == TAIL:
        if set(pre) != {"scaler", "tail_fraction"} or pre["scaler"] != "standard":
            raise ValueError("Tail forests require StandardScaler and an explicit tail_fraction")
        q = pre["tail_fraction"]
        if isinstance(q, bool) or not isinstance(q, (int, float)) or q not in (0., .005, .01):
            raise ValueError("Tail fraction must be exactly 0, .005 or .01")
        pre["tail_fraction"] = float(q)
    elif pre != {"scaler": "standard"}:
        raise ValueError("Binary forests and voting use only the shared StandardScaler recipe")
    expected_post = {"kind": "ovr_normalized", "zero_row": "uniform"} if family == OVR else {"kind": "argmax"}
    if post != expected_post:
        raise ValueError("Alternatives postprocessing must match its fixed family rule")
    if set(training) - {"seed", "weighting", "data_policy"}:
        raise ValueError("Unknown alternatives training fields")
    weighting = "per_member" if family == VOTING else "binary_class_balanced" if family == OVR else "class_balanced"
    training.setdefault("seed", 20260921)
    training.setdefault("weighting", weighting)
    _seed(training["seed"])
    if training["weighting"] != weighting or training.get("data_policy") not in (None, DATA_POLICY):
        raise ValueError("Alternatives weighting or training-record policy differs from the family contract")
    if family == VOTING:
        members = value.get("members")
        if params or not isinstance(members, list) or len(members) < 2 or any(
                not isinstance(member, dict) or member.get("family") not in {TAIL, OVR} for member in members):
            raise ValueError("Alternatives voting requires distinct nonnested tail or binary forest members")
        value["members"] = [validate_alternatives_pipeline(member) for member in members]
        if any(member["training"]["seed"] != training["seed"]
                or member["training"].get("data_policy") != training.get("data_policy") for member in value["members"]):
            raise ValueError("Alternatives members must share primary seed and training-record policy")
        if len({json.dumps(member, sort_keys=True) for member in value["members"]}) != len(members):
            raise ValueError("Alternatives voting members must have distinct recipes")
    else:
        if "members" in value or set(params) - set(TREE_DEFAULTS):
            raise ValueError("Unknown alternatives model parameters or unexpected members")
        for name, default in TREE_DEFAULTS.items():
            params.setdefault(name, default)
        for name in ("n_estimators", "min_samples_leaf"):
            if type(params[name]) is not int or params[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        if params["max_depth"] is not None and (type(params["max_depth"]) is not int or params["max_depth"] < 1):
            raise ValueError("max_depth must be null or a positive integer")
        count = params["max_features"]
        if not ((type(count) is int and 1 <= count <= 596)
                or (type(count) is float and np.isfinite(count) and 0 < count <= 1)):
            raise ValueError("max_features must be a positive feature count or fraction")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Alternatives recipe must contain finite JSON values") from error
    return value


def _features(X):
    if hasattr(X, "columns") and list(X.columns) != feature_names():
        raise ValueError("Named alternatives input columns differ from canonical596 features")
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] != 596 or not len(values)
            or values.dtype.kind not in "iuf" or not np.isfinite(values).all()):
        raise ValueError("Expected nonempty finite numeric features with shape (N,596)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(values, dtype=np.float32 if values.dtype == np.dtype(np.float32) else np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Features exceed the supported numeric range")
    return result


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_digest(array):
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256(json.dumps([values.dtype.str, list(values.shape)]).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _scaler_digest(scaler):
    return _json_digest({"params": scaler.get_params(deep=False), "samples": int(scaler.n_samples_seen_),
        "features": int(scaler.n_features_in_), **{name: getattr(scaler, name).tolist() for name in ("mean_", "scale_", "var_")}})


def _forest_digest(estimator):
    if (type(estimator) is not ExtraTreesClassifier or estimator.n_features_in_ != 596 or estimator.n_outputs_ != 1
            or not np.array_equal(estimator.classes_, np.arange(7)) or len(estimator.estimators_) != estimator.n_estimators):
        raise ValueError("Tail forest fitted class or feature contract changed")
    trees = []
    for tree in estimator.estimators_:
        if (type(tree) is not ExtraTreeClassifier or tree.n_features_in_ != 596 or tree.n_outputs_ != 1
                or not np.array_equal(tree.classes_, np.arange(7))):
            raise ValueError("Tail tree fitted class or feature contract changed")
        state = tree.tree_.__getstate__()
        trees.append({"seed": int(tree.random_state), "classes": _array_digest(tree.classes_),
            "n_classes": int(tree.n_classes_), "n_features": int(tree.n_features_in_), "n_outputs": int(tree.n_outputs_),
            "node_count": int(state["node_count"]), "max_depth": int(state["max_depth"]),
            "values": _array_digest(state["values"]),
            "nodes": {name: _array_digest(state["nodes"][name]) for name in state["nodes"].dtype.names}})
    return _json_digest({"classes": _array_digest(estimator.classes_), "trees": trees})


def _estimator(recipe, seed):
    if recipe["family"] == OVR:
        return BinaryForestV1(**deepcopy(recipe["model_params"]), random_state=seed)
    return ExtraTreesClassifier(**deepcopy(recipe["model_params"]), class_weight="balanced", n_jobs=1, random_state=seed)


def _model_rule(recipe):
    if recipe["family"] == OVR:
        return binary_forest_rule()
    return {"kind": "native_seven_class_extra_trees", "class_weight": "balanced", "class_order": list(range(7)),
        "probability": "Native forest.predict_proba, aligned to 0..6; argmax", "n_estimators_scope": "total forest"}


class ForestAlternativesPipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, spec, seed=None):
        self.spec = spec
        self.seed = seed

    def fit(self, X, y, *, groups=None, validation_data=None):
        recipe = validate_alternatives_pipeline(self.spec)
        seed = _seed(recipe["training"]["seed"] if self.seed is None else self.seed)
        values, labels = _features(X), np.asarray(y)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu"
                or not np.array_equal(np.unique(labels), np.arange(7))):
            raise ValueError("Alternatives training requires aligned integer labels with all seven posture IDs")
        labels = labels.astype(np.int64, copy=True)
        people = None
        if groups is not None:
            people = np.asarray(groups)
            if people.ndim != 1 or len(people) != len(values) or any(
                    not isinstance(person, str) or not normalize_text(person) for person in people.tolist()):
                raise ValueError("Alternatives groups must be aligned nonempty subject names")
            people = [normalize_text(person) for person in people.tolist()]
        with threadpool_limits(limits=1):
            if recipe["family"] == VOTING:
                members = [build_alternatives_pipeline(member, seed=self.seed).fit(values, labels, groups=groups)
                    for member in recipe["members"]]
            else:
                clipper = TailClipperV1(recipe["preprocessing"]["tail_fraction"]) if recipe["family"] == TAIL else None
                prepared = clipper.fit_transform(values) if clipper is not None else values
                scaler = StandardScaler()
                transformed = scaler.fit_transform(prepared)
                model_input = transformed.astype(np.float32, copy=False) if recipe["family"] == OVR else transformed
                estimator = _estimator(recipe, seed).fit(model_input, labels)
        self.spec_, self.effective_seed_, self.classes_ = recipe, seed, np.arange(7)
        self.n_features_in_, self.feature_names_in_, self.label_names_ = 596, np.asarray(feature_names()), list(CLASS_NAMES)
        self.class_counts_, self.training_subjects_ = np.bincount(labels, minlength=7), None if people is None else sorted(set(people))
        self.training_input_sha256_, self.training_labels_sha256_ = _array_digest(values), _array_digest(labels)
        self.training_groups_sha256_, self.feature_schema_sha256_ = _json_digest(people), _json_digest(feature_names())
        self.training_details_ = {"pipeline": deepcopy(recipe), "pipeline_version": PIPELINE_VERSION, "effective_seed": seed,
            "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "input_contract": "feature_vector", "input_features": 596, "feature_names": feature_names(),
            "feature_schema_sha256": self.feature_schema_sha256_, "training_feature_dtype": str(values.dtype),
            "training_windows": len(values), "training_subjects": deepcopy(self.training_subjects_),
            "class_counts": self.class_counts_.tolist(), "class_names": list(CLASS_NAMES),
            "training_input_sha256": self.training_input_sha256_, "training_labels_sha256": self.training_labels_sha256_,
            "training_groups_sha256": self.training_groups_sha256_, "external_validation_used": False,
            "training_record_policy": recipe["training"].get("data_policy"),
            "training_record_policy_application": "caller supplies eligible TRAIN rows; no historical cache import",
            "preprocessing_fit": "fixed tail rule, cutpoints and unweighted shared StandardScaler fit only supplied TRAIN rows",
            "evaluation_population_normalization": False, "threads": 1}
        if recipe["family"] == VOTING:
            self.members_ = members
            self.training_details_.update(combination="equal aligned probability mean", weights=[1 / len(members)] * len(members),
                members=[deepcopy(member.training_details_) for member in members])
        else:
            self.scaler_, self.estimator_, self.model_rule_ = scaler, estimator, _model_rule(recipe)
            self.scaler_state_sha256_ = _scaler_digest(scaler)
            self.estimator_state_sha256_ = binary_forest_state_sha256(estimator) if recipe["family"] == OVR else _forest_digest(estimator)
            self.prepared_state_sha256_, self.transformed_state_sha256_ = _array_digest(prepared), _array_digest(transformed)
            self.model_input_sha256_ = _array_digest(model_input)
            self.training_details_.update(model_rule=deepcopy(self.model_rule_), scaler_state_sha256=self.scaler_state_sha256_,
                estimator_state_sha256=self.estimator_state_sha256_, prepared_state_sha256=self.prepared_state_sha256_,
                transformed_state_sha256=self.transformed_state_sha256_, model_input_sha256=self.model_input_sha256_,
                model_input_dtype=str(model_input.dtype), effective_model_params=estimator.get_params(deep=False))
            if clipper is not None:
                self.clipper_, self.clipper_state_sha256_ = clipper, tail_clipping_state_sha256(clipper)
                self.training_details_.update(tail_clipping=deepcopy(clipper.training_details_),
                    clipper_state_sha256=self.clipper_state_sha256_)
            else:
                self.training_details_["binary_forest"] = deepcopy(estimator.training_details_)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "spec_")
        if validate_alternatives_pipeline(self.spec) != self.spec_ or _seed(
                self.spec_["training"]["seed"] if self.seed is None else self.seed) != self.effective_seed_:
            raise ValueError("Alternatives recipe or seed changed after fitting")
        values = _features(X)
        with threadpool_limits(limits=1):
            if self.spec_["family"] == VOTING:
                probability = np.mean([member.predict_proba(values) for member in self.members_], axis=0)
            else:
                prepared = self.clipper_.transform(values) if self.spec_["family"] == TAIL else values
                transformed = self.scaler_.transform(prepared)
                if self.spec_["family"] == OVR:
                    with np.errstate(over="ignore", invalid="ignore"):
                        transformed = np.asarray(transformed, dtype=np.float32)
                raw = self.estimator_.predict_proba(transformed)
                classes = np.asarray(self.estimator_.classes_)
                if classes.dtype.kind not in "iu" or not np.array_equal(np.sort(classes), np.arange(7)):
                    raise ValueError("Alternatives backend classes changed")
                probability = np.empty_like(raw)
                probability[:, classes] = raw
        if (probability.shape != (len(values), 7) or probability.dtype != np.dtype(np.float64)
                or not np.isfinite(probability).all() or (probability < 0).any()
                or not np.allclose(probability.sum(1), 1., atol=1e-12, rtol=0)):
            raise ValueError("Invalid alternatives probabilities")
        return probability

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def build_alternatives_pipeline(spec, seed=None):
    if seed is not None:
        _seed(seed)
    return ForestAlternativesPipeline(validate_alternatives_pipeline(spec), seed=seed)


def fit_alternatives_pipeline(model, spec, X, y, *, groups=None, validation_data=None):
    recipe = validate_alternatives_pipeline(spec)
    if not isinstance(model, ForestAlternativesPipeline) or validate_alternatives_pipeline(model.spec) != recipe:
        raise ValueError("Constructed alternatives estimator and requested recipe differ")
    return model.fit(X, y, groups=groups, validation_data=validation_data)


def _verify_digest_link(model, details, name):
    digest = getattr(model, name + "_", None)
    if (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            or details.get(name) != digest):
        raise ValueError("Alternatives training scope or transformed-state digest changed")


def verify_fitted_alternatives_identity(model, spec, seed):
    """Verify complete helper, fitted-state, preprocessing and training-scope links."""
    recipe, seed = validate_alternatives_pipeline(spec), _seed(seed)
    if (type(model) is not ForestAlternativesPipeline or getattr(model, "spec_", None) != recipe
            or validate_alternatives_pipeline(model.spec) != recipe or getattr(model, "effective_seed_", None) != seed
            or (recipe["training"]["seed"] if model.seed is None else model.seed) != seed
            or getattr(model, "n_features_in_", None) != 596 or not np.array_equal(getattr(model, "classes_", ()), np.arange(7))
            or getattr(model, "label_names_", None) != list(CLASS_NAMES)
            or not np.array_equal(getattr(model, "feature_names_in_", ()), feature_names())
            or getattr(model, "feature_schema_sha256_", None) != _json_digest(feature_names())):
        raise ValueError("Fitted alternatives recipe, seed or feature/class contract changed")
    details = getattr(model, "training_details_", {})
    count, counts = details.get("training_windows"), np.asarray(getattr(model, "class_counts_", ()))
    if (details.get("pipeline") != recipe or details.get("pipeline_version") != PIPELINE_VERSION or details.get("effective_seed") != seed
            or details.get("sklearn_version") != sklearn.__version__ or details.get("numpy_version") != np.__version__
            or details.get("scipy_version") != scipy.__version__ or details.get("external_validation_used") is not False
            or details.get("evaluation_population_normalization") is not False or details.get("threads") != 1
            or type(count) is not int or counts.shape != (7,) or counts.dtype.kind not in "iu" or (counts < 1).any()
            or counts.sum() != count or details.get("class_counts") != counts.tolist()
            or details.get("class_names") != list(CLASS_NAMES) or details.get("feature_names") != feature_names()
            or details.get("feature_schema_sha256") != model.feature_schema_sha256_
            or details.get("input_contract") != "feature_vector" or details.get("input_features") != 596
            or details.get("training_feature_dtype") not in ("float32", "float64")
            or details.get("training_subjects") != getattr(model, "training_subjects_", None)
            or details.get("training_record_policy") != recipe["training"].get("data_policy")):
        raise ValueError("Alternatives training audit or library versions differ from fitted state")
    for name in ("training_input_sha256", "training_labels_sha256", "training_groups_sha256"):
        _verify_digest_link(model, details, name)
    if recipe["family"] == VOTING:
        members = getattr(model, "members_", ())
        if len(members) != len(recipe["members"]):
            raise ValueError("Alternatives voting member count changed")
        for member, member_recipe in zip(members, recipe["members"]):
            verify_fitted_alternatives_identity(member, member_recipe, seed)
            if any(member.training_details_[name] != details[name] for name in ("training_input_sha256", "training_labels_sha256",
                    "training_groups_sha256", "class_counts", "training_subjects", "training_windows", "training_feature_dtype")):
                raise ValueError("Alternatives members trained on different rows or subjects")
        if (details.get("members") != [member.training_details_ for member in members]
                or details.get("weights") != [1 / len(members)] * len(members)
                or details.get("combination") != "equal aligned probability mean"):
            raise ValueError("Alternatives ensemble audit differs from its fitted members")
        return True
    for name in ("prepared_state_sha256", "transformed_state_sha256", "model_input_sha256"):
        _verify_digest_link(model, details, name)
    expected_rule = _model_rule(recipe)
    if getattr(model, "model_rule_", None) != expected_rule or details.get("model_rule") != expected_rule:
        raise ValueError("Alternatives scoring rule changed")
    scaler, estimator = getattr(model, "scaler_", None), getattr(model, "estimator_", None)
    expected = _estimator(recipe, seed)
    try:
        if (type(scaler) is not StandardScaler or scaler.get_params(deep=False) != StandardScaler().get_params(deep=False)
                or scaler.n_features_in_ != 596 or scaler.n_samples_seen_ != count
                or _scaler_digest(scaler) != getattr(model, "scaler_state_sha256_", None)
                or details.get("scaler_state_sha256") != model.scaler_state_sha256_
                or type(estimator) is not type(expected) or estimator.get_params(deep=False) != expected.get_params(deep=False)
                or details.get("effective_model_params") != expected.get_params(deep=False)):
            raise ValueError("Alternatives scaler or backend parameters changed")
        if recipe["family"] == TAIL:
            clipper = model.clipper_
            q = recipe["preprocessing"]["tail_fraction"]
            verify_tail_clipper(clipper, q)
            if (clipper.n_samples_seen_ != count or clipper.training_dtype_ != details["training_feature_dtype"]
                    or clipper.training_input_sha256_ != model.training_input_sha256_
                    or clipper.rule_ != tail_clipping_rule(q) or details.get("tail_clipping") != clipper.training_details_
                    or tail_clipping_state_sha256(clipper) != getattr(model, "clipper_state_sha256_", None)
                    or details.get("clipper_state_sha256") != model.clipper_state_sha256_
                    or details.get("model_input_dtype") != details["training_feature_dtype"]
                    or model.model_input_sha256_ != model.transformed_state_sha256_
                    or q == 0 and model.prepared_state_sha256_ != model.training_input_sha256_
                    or any(tree.tree_.n_node_samples[0] != count for tree in estimator.estimators_)):
                raise ValueError("Tail clipping helper, original training scope or forest count changed")
            estimator_digest = _forest_digest(estimator)
        else:
            verify_binary_forest(estimator, recipe["model_params"], seed)
            backend = estimator.training_details_
            if (estimator.n_samples_seen_ != count or not np.array_equal(estimator.class_counts_, counts)
                    or backend.get("training_input_sha256") != model.model_input_sha256_
                    or backend.get("training_labels_sha256") != model.training_labels_sha256_
                    or details.get("binary_forest") != backend or details.get("model_input_dtype") != "float32"
                    or model.prepared_state_sha256_ != model.training_input_sha256_):
                raise ValueError("Binary forest helper was fitted on a different input or label scope")
            estimator_digest = binary_forest_state_sha256(estimator)
        if (estimator_digest != getattr(model, "estimator_state_sha256_", None)
                or details.get("estimator_state_sha256") != model.estimator_state_sha256_):
            raise ValueError("Alternatives fitted forest state changed")
    except (AttributeError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Incomplete alternatives scaler, helper or fitted forest state") from error
    return True
