"""TRAIN-only pair forests and fixed between-person residual coordinates.

Pairwise coupling is a mathematical aggregation rule, not a calibration claim.
The residual projection uses only supplied TRAIN identities and labels.
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
from sklearn.tree import ExtraTreeClassifier
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .posture_core import CLASS_NAMES, feature_names, normalize_text
from .pairwise_coupling_v1 import (PAIR_INDICES, couple_pairwise, pairwise_coupling_rule,
    verify_pairwise_coupling_rule)
from .subject_residual_v1 import (SubjectResidualV1, subject_residual_rule,
    subject_residual_state_sha256, verify_subject_residual)


VERSION = "structured_forest_pipeline_v1"
PAIR, RESIDUAL, VOTING = "coupled_extra_trees", "subject_residual_extra_trees", "structured_soft_voting"
FAMILIES = frozenset({PAIR, RESIDUAL, VOTING})
POLICY = "include_ordinary_reserved_training_records_v1"
TREE_DEFAULTS = {"n_estimators": 200, "min_samples_leaf": 8, "max_depth": 8, "max_features": .5}


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Structured forest seed must be a uint32 integer")
    return value


def validate_structured_pipeline(spec):
    if not isinstance(spec, dict):
        raise ValueError("Structured forest recipe must be an object")
    value = deepcopy(spec)
    family = value.get("family")
    if not isinstance(family, str) or family not in FAMILIES or set(value) - {
            "family", "base_method", "window_seconds", "feature_engineering", "preprocessing",
            "model_params", "training", "postprocessing", "members"}:
        raise ValueError("Unknown structured forest family or field")
    defaults = {"base_method": "standard", "window_seconds": 3,
        "feature_engineering": {"mode": "all" if family == PAIR else "per_member" if family == VOTING else "remove_subject3"},
        "preprocessing": {"scaler": "standard"}, "model_params": {}, "training": {},
        "postprocessing": {"kind": "pairwise_wu_kkt" if family == PAIR else "argmax"}}
    for key, default in defaults.items():
        value.setdefault(key, deepcopy(default))
    for key in ("feature_engineering", "preprocessing", "model_params", "training", "postprocessing"):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} must be an object")
    if (value["base_method"] != "standard" or type(value["window_seconds"]) is not int or value["window_seconds"] != 3
            or value["preprocessing"] != {"scaler": "standard"}
            or value["postprocessing"] != defaults["postprocessing"]):
        raise ValueError("Structured forest v1 requires fixed3s canonical596 input, StandardScaler and family scoring")
    mode = value["feature_engineering"].get("mode")
    if (set(value["feature_engineering"]) != {"mode"} or not isinstance(mode, str)
            or mode not in ({"all"} if family == PAIR else {"per_member"} if family == VOTING else {"identity", "remove_subject3"})):
        raise ValueError("Unknown structured forest feature mode")
    training, params = value["training"], value["model_params"]
    weighting = "pair_class_balanced" if family == PAIR else "per_member" if family == VOTING else "class_balanced"
    if set(training) - {"seed", "weighting", "data_policy"}:
        raise ValueError("Unknown structured forest training setting")
    training.setdefault("seed", 20260921)
    training.setdefault("weighting", weighting)
    _seed(training["seed"])
    if training["weighting"] != weighting or training.get("data_policy") not in (None, POLICY):
        raise ValueError("Structured forest weighting or record policy changed")
    if family == VOTING:
        members = value.get("members")
        if (params or not isinstance(members, list) or len(members) < 2 or any(
                not isinstance(member, dict) or member.get("family") not in {PAIR, RESIDUAL} for member in members)):
            raise ValueError("Structured voting requires distinct nonnested leaf recipes")
        value["members"] = [validate_structured_pipeline(member) for member in members]
        if any(member["training"]["seed"] != training["seed"] or
                member["training"].get("data_policy") != training.get("data_policy") for member in value["members"]):
            raise ValueError("Structured voting members must share seed and training record policy")
        if len({json.dumps(member, sort_keys=True) for member in value["members"]}) != len(members):
            raise ValueError("Structured voting members must be distinct")
    else:
        if "members" in value or set(params) - set(TREE_DEFAULTS):
            raise ValueError("Unknown structured forest model parameters")
        for key, default in TREE_DEFAULTS.items():
            params.setdefault(key, 32 if family == PAIR and key == "n_estimators" else default)
        for key in ("n_estimators", "min_samples_leaf"):
            if type(params[key]) is not int or params[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if params["max_depth"] is not None and (type(params["max_depth"]) is not int or params["max_depth"] < 1):
            raise ValueError("max_depth must be null or a positive integer")
        count = params["max_features"]
        if not ((type(count) is int and 1 <= count <= 596) or
                (type(count) is float and np.isfinite(count) and 0 < count <= 1)):
            raise ValueError("max_features must be a fixed supported count or fraction")
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("Structured forest recipe must contain finite JSON values") from error
    return value


def _features(X):
    if hasattr(X, "columns") and list(X.columns) != feature_names():
        raise ValueError("Structured forest named input differs from canonical596")
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] != 596 or not len(values) or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()):
        raise ValueError("Structured forest needs finite nonempty numeric input(N,596)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(values, dtype=np.float32 if values.dtype == np.dtype(np.float32) else np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Structured forest input exceeds the supported finite range")
    return result


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    array = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps([array.dtype.str, list(array.shape)]).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def _identities(value, rows, name):
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != rows or any(not isinstance(v, str) or not normalize_text(v) for v in array.tolist()):
        raise ValueError(f"{name} must contain one nonempty string identity per TRAIN row")
    return np.asarray([normalize_text(v) for v in array.tolist()])


def _records(labels, people, records):
    result = []
    for record in sorted(set(records.tolist())):
        idx = np.flatnonzero(records == record)
        if len(np.unique(labels[idx])) != 1 or len(np.unique(people[idx])) != 1:
            raise ValueError("Every TRAIN recording must contain one person and one posture")
        result.append({"record_id": record, "subject": str(people[idx[0]]), "class_id": int(labels[idx[0]]),
            "row_indices": idx.tolist(), "window_count": len(idx)})
    return result


def _backend(params, seed):
    return ExtraTreesClassifier(**params, random_state=seed, class_weight="balanced", n_jobs=1, bootstrap=False)


def _scaler_state(scaler):
    return {"params": scaler.get_params(deep=False), "rows": int(scaler.n_samples_seen_),
        "features": int(scaler.n_features_in_), **{name: getattr(scaler, name).tolist() for name in ("mean_", "var_", "scale_")}}


def _forest_state(forest, params, seed, classes, rows):
    expected = _backend(params, seed)
    classes = np.asarray(classes)
    if (type(forest) is not ExtraTreesClassifier or forest.get_params(deep=False) != expected.get_params(deep=False)
            or not np.array_equal(forest.classes_, classes) or forest.n_features_in_ != 596
            or forest.n_outputs_ != 1 or len(forest.estimators_) != params["n_estimators"]):
        raise ValueError("Structured native forest class, parameters or dimensions changed")
    seeds = np.random.RandomState(seed).randint(np.iinfo(np.int32).max, size=params["n_estimators"])
    states = []
    for tree, tree_seed in zip(forest.estimators_, seeds):
        expected_tree = ExtraTreeClassifier(max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"], random_state=int(tree_seed))
        if (type(tree) is not ExtraTreeClassifier or tree.get_params(deep=False) != expected_tree.get_params(deep=False)
                or tree.n_features_in_ != 596 or tree.n_outputs_ != 1 or not np.array_equal(tree.classes_, np.arange(len(classes)))
                or tree.tree_.n_node_samples[0] != rows
                or not np.isclose(tree.tree_.weighted_n_node_samples[0], rows, atol=1e-9, rtol=1e-12)
                or not np.allclose(tree.tree_.value[0, 0], 1/len(classes), atol=1e-12, rtol=0)):
            raise ValueError("Structured tree parameters, seed or class-balanced TRAIN scope changed")
        state = tree.tree_.__getstate__()
        states.append({"params": tree.get_params(deep=False), "classes": _array_digest(tree.classes_),
            "node_count": int(state["node_count"]), "max_depth": int(state["max_depth"]),
            "values": _array_digest(state["values"]),
            "nodes": {key: _array_digest(state["nodes"][key]) for key in state["nodes"].dtype.names}})
    return _json_digest({"params": forest.get_params(deep=False), "classes": _array_digest(forest.classes_), "trees": states})


def _pair_audit(params, seed, labels, people, records, pair, index, transformed):
    i, j = pair
    take = np.flatnonzero((labels == i) | (labels == j))
    binary = (labels[take] == j).astype(np.int64)
    counts = np.bincount(binary, minlength=2)
    if (counts == 0).any():
        raise ValueError("Each OVO TRAIN pair requires both postures")
    return take, binary, {"pair": [int(i), int(j)], "pair_index": index, "seed": (seed + index) % 2**32,
        "rows": len(take), "row_indices": take.tolist(), "binary_counts": counts.tolist(),
        "subjects": sorted(set(people[take].tolist())), "records": sorted(set(records[take].tolist())),
        "binary_labels_sha256": _array_digest(binary),
        "input_sha256": _array_digest(transformed[take]) if transformed is not None else None,
        "binary_class_weights": (len(take) / (2 * counts)).tolist(), "class_weight_mass": [len(take)/2]*2,
        "binary_label_rule": "original i->0, j->1; coupling uses probability column0 for i"}


class StructuredForestPipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, spec, seed=None):
        self.spec, self.seed = spec, seed

    def fit(self, X, y, *, groups=None, record_ids=None, validation_data=None):
        recipe = validate_structured_pipeline(self.spec)
        seed = _seed(recipe["training"]["seed"] if self.seed is None else self.seed)
        values, labels = _features(X), np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu" or not np.array_equal(np.unique(labels), np.arange(7)):
            raise ValueError("Structured forests require all seven canonical integer TRAIN labels")
        labels = labels.astype(np.int64, copy=True)
        people = _identities(groups, len(values), "groups")
        records = _identities(record_ids, len(values), "record_ids")
        metadata = _records(labels, people, records)
        self.spec_, self.effective_seed_ = recipe, seed
        self.classes_, self.label_names_, self.n_features_in_ = np.arange(7), list(CLASS_NAMES), 596
        self.feature_names_in_, self.record_metadata_ = np.asarray(feature_names()), metadata
        self.training_input_sha256_, self.training_labels_sha256_ = _array_digest(values), _array_digest(labels)
        self.training_groups_sha256_, self.training_records_sha256_ = _json_digest(people.tolist()), _json_digest(records.tolist())
        self.training_details_ = {"version": VERSION, "pipeline": deepcopy(recipe), "effective_seed": seed,
            "input_contract": "feature_vector", "features": 596, "feature_names": feature_names(), "class_names": list(CLASS_NAMES),
            "rows": len(labels), "dtype": str(values.dtype), "class_counts": np.bincount(labels, minlength=7).tolist(),
            "subjects": sorted(set(people.tolist())), "record_metadata": deepcopy(metadata),
            "training_input_sha256": self.training_input_sha256_, "training_labels_sha256": self.training_labels_sha256_,
            "training_groups_sha256": self.training_groups_sha256_, "training_records_sha256": self.training_records_sha256_,
            "external_validation_used": False, "evaluation_batch_adaptation": False, "threads": 1,
            "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
            "training_record_policy": recipe["training"].get("data_policy"), "data_policy_application": "caller supplies eligible TRAIN only"}
        with threadpool_limits(limits=1):
            if recipe["family"] == VOTING:
                self.members_ = [build_structured_pipeline(spec, seed=seed).fit(values, labels, groups=people, record_ids=records)
                    for spec in recipe["members"]]
                self.training_details_.update(combination="equal aligned native probability mean",
                    members=[deepcopy(model.training_details_) for model in self.members_])
            else:
                self.scaler_ = StandardScaler()
                transformed = self.scaler_.fit_transform(values)
                self.scaler_state_sha256_ = _json_digest(_scaler_state(self.scaler_))
                self.training_details_.update(scaler_state_sha256=self.scaler_state_sha256_,
                    standardized_input_sha256=_array_digest(transformed), preprocessing="unweighted original TRAIN StandardScaler; no later rescaling")
                if recipe["family"] == RESIDUAL:
                    self.projector_ = SubjectResidualV1(recipe["feature_engineering"]["mode"]).fit(transformed, labels, groups=people, record_ids=records)
                    processed = self.projector_.transform(transformed)
                    self.estimator_ = _backend(recipe["model_params"], seed).fit(processed, labels)
                    self.training_details_.update(projector_rule=subject_residual_rule(recipe["feature_engineering"]["mode"]),
                        projector_state_sha256=subject_residual_state_sha256(self.projector_),
                        projector=deepcopy(self.projector_.training_details_), transformed_input_sha256=_array_digest(processed),
                        forest_state_sha256=_forest_state(self.estimator_, recipe["model_params"], seed, np.arange(7), len(labels)))
                else:
                    pair_input = np.asarray(transformed, dtype=np.float32, order="C")
                    self.heads_, self.pair_audits_ = [], []
                    self.coupling_rule_ = pairwise_coupling_rule()
                    for index, pair in enumerate(PAIR_INDICES):
                        take, binary, audit = _pair_audit(recipe["model_params"], seed, labels, people, records, pair, index, pair_input)
                        head = _backend(recipe["model_params"], audit["seed"]).fit(pair_input[take], binary)
                        audit["forest_state_sha256"] = _forest_state(head, recipe["model_params"], audit["seed"], [0, 1], len(take))
                        self.heads_.append(head)
                        self.pair_audits_.append(audit)
                    self.training_details_.update(pair_input_sha256=_array_digest(pair_input), pair_audits=deepcopy(self.pair_audits_),
                        coupling_rule=deepcopy(self.coupling_rule_), total_trees=len(PAIR_INDICES)*recipe["model_params"]["n_estimators"],
                        pair_training_row_exposures=sum(row["rows"] for row in self.pair_audits_), calibrated_posterior=False)
        self.fit_state_sha256_ = _fit_state(self)
        self.training_details_["fit_state_sha256"] = self.fit_state_sha256_
        verify_fitted_structured_identity(self, recipe, seed)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "fit_state_sha256_")
        if validate_structured_pipeline(self.spec) != self.spec_ or _seed(
                self.spec_["training"]["seed"] if self.seed is None else self.seed) != self.effective_seed_:
            raise ValueError("Structured forest recipe or seed changed after training")
        values = _features(X)
        with threadpool_limits(limits=1):
            if self.spec_["family"] == VOTING:
                probability = np.mean([member.predict_proba(values) for member in self.members_], axis=0)
            else:
                transformed = self.scaler_.transform(values)
                if self.spec_["family"] == RESIDUAL:
                    probability = self.estimator_.predict_proba(self.projector_.transform(transformed))
                else:
                    verify_pairwise_coupling_rule(self.coupling_rule_)
                    if len(self.heads_) != len(PAIR_INDICES):
                        raise ValueError("OVO head count changed")
                    pair_input = np.asarray(transformed, dtype=np.float32, order="C")
                    scores = []
                    for head in self.heads_:
                        if not np.array_equal(head.classes_, [0, 1]):
                            raise ValueError("OVO binary class orientation changed")
                        scores.append(head.predict_proba(pair_input)[:, 0])
                    probability = couple_pairwise(np.column_stack(scores))
        if (probability.shape != (len(values), 7) or not np.isfinite(probability).all() or np.any(probability < 0)
                or not np.allclose(probability.sum(1), 1., rtol=0, atol=1e-12)):
            raise ValueError("Invalid structured forest output probability")
        return probability

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def _fit_state(model):
    details = deepcopy(model.training_details_)
    details.pop("fit_state_sha256", None)
    # Model-add appends its independently validated method provenance after fit.
    # This field is not learned numerical state; all fitted fields remain hashed.
    details.pop("training_method", None)
    if model.spec_["family"] == VOTING:
        state = [member.fit_state_sha256_ for member in model.members_]
    elif model.spec_["family"] == RESIDUAL:
        state = [subject_residual_state_sha256(model.projector_), _forest_state(model.estimator_, model.spec_["model_params"],
            model.effective_seed_, np.arange(7), model.training_details_["rows"]), _json_digest(_scaler_state(model.scaler_))]
    else:
        state = [_forest_state(head, model.spec_["model_params"], row["seed"], [0, 1], row["rows"])
            for head, row in zip(model.heads_, model.pair_audits_)]
        state += [_json_digest(_scaler_state(model.scaler_)), _json_digest(model.coupling_rule_)]
    return _json_digest({"details": details, "state": state})


def _restore(metadata, rows):
    labels = np.full(rows, -1, dtype=np.int64)
    people, records, seen = [None]*rows, [None]*rows, np.zeros(rows, dtype=np.int64)
    for row in metadata:
        idx = row["row_indices"]
        if (not isinstance(idx, list) or not idx or len(idx) != row["window_count"] or
                any(type(i) is not int or not 0 <= i < rows for i in idx) or
                type(row["class_id"]) is not int or not 0 <= row["class_id"] < 7):
            raise ValueError("Invalid structured forest recording row mapping")
        for i in idx:
            labels[i], people[i], records[i] = row["class_id"], row["subject"], row["record_id"]
            seen[i] += 1
    if not np.all(seen == 1):
        raise ValueError("Recording metadata must cover each TRAIN row exactly once")
    people, records = _identities(people, rows, "groups"), _identities(records, rows, "record_ids")
    if _records(labels, people, records) != metadata or not np.array_equal(np.unique(labels), np.arange(7)):
        raise ValueError("Structured forest record identities or class coverage changed")
    return labels, people, records


def verify_fitted_structured_identity(model, spec, seed):
    recipe, seed = validate_structured_pipeline(spec), _seed(seed)
    if (type(model) is not StructuredForestPipeline or getattr(model, "spec_", None) != recipe
            or validate_structured_pipeline(model.spec) != recipe or getattr(model, "effective_seed_", None) != seed
            or (recipe["training"]["seed"] if model.seed is None else model.seed) != seed
            or getattr(model, "n_features_in_", None) != 596 or not np.array_equal(getattr(model, "classes_", ()), np.arange(7))
            or not np.array_equal(getattr(model, "feature_names_in_", ()), feature_names()) or getattr(model, "label_names_", None) != list(CLASS_NAMES)):
        raise ValueError("Structured forest recipe, seed or canonical schema changed")
    try:
        details = model.training_details_
        rows = details["rows"]
        if type(rows) is not int or rows < 7:
            raise ValueError("Invalid structured forest TRAIN row count")
        labels, people, records = _restore(model.record_metadata_, rows)
        expected = {"version": VERSION, "pipeline": recipe, "effective_seed": seed, "input_contract": "feature_vector",
            "features": 596, "feature_names": feature_names(), "class_names": list(CLASS_NAMES),
            "class_counts": np.bincount(labels, minlength=7).tolist(), "subjects": sorted(set(people.tolist())),
            "record_metadata": model.record_metadata_, "external_validation_used": False, "evaluation_batch_adaptation": False,
            "threads": 1, "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
            "training_record_policy": recipe["training"].get("data_policy"), "data_policy_application": "caller supplies eligible TRAIN only"}
        if any(details.get(key) != value for key, value in expected.items()) or details.get("dtype") not in {"float32", "float64"}:
            raise ValueError("Structured forest training identity metadata changed")
        hashes = {"training_input_sha256": model.training_input_sha256_, "training_labels_sha256": _array_digest(labels),
            "training_groups_sha256": _json_digest(people.tolist()), "training_records_sha256": _json_digest(records.tolist())}
        for name, value in hashes.items():
            if (not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                    or getattr(model, name + "_") != value or details.get(name) != value):
                raise ValueError("Structured forest TRAIN digest changed")
        if recipe["family"] == VOTING:
            if len(model.members_) != len(recipe["members"]):
                raise ValueError("Structured voting member count changed")
            for member, member_recipe in zip(model.members_, recipe["members"]):
                verify_fitted_structured_identity(member, member_recipe, seed)
                if any(member.training_details_[key] != details[key] for key in (*hashes, "record_metadata", "dtype")):
                    raise ValueError("Structured voting members did not share TRAIN rows")
            if details.get("members") != [member.training_details_ for member in model.members_] or details.get("combination") != "equal aligned native probability mean":
                raise ValueError("Structured voting audit changed")
        else:
            scaler = model.scaler_
            if (type(scaler) is not StandardScaler or scaler.get_params(deep=False) != StandardScaler().get_params(deep=False)
                    or scaler.n_samples_seen_ != rows or scaler.n_features_in_ != 596 or
                    _json_digest(_scaler_state(scaler)) != model.scaler_state_sha256_ or details["scaler_state_sha256"] != model.scaler_state_sha256_
                    or details["preprocessing"] != "unweighted original TRAIN StandardScaler; no later rescaling"):
                raise ValueError("Structured forest scaler or fitting scope changed")
            if recipe["family"] == RESIDUAL:
                mode = recipe["feature_engineering"]["mode"]
                verify_subject_residual(model.projector_, mode)
                if (details["projector_rule"] != subject_residual_rule(mode)
                        or details["projector_state_sha256"] != subject_residual_state_sha256(model.projector_)
                        or details["projector"] != model.projector_.training_details_
                        or model.projector_.training_input_sha256_ != details["standardized_input_sha256"]
                        or model.projector_.training_labels_sha256_ != model.training_labels_sha256_
                        or model.projector_.training_groups_sha256_ != model.training_groups_sha256_
                        or model.projector_.training_records_sha256_ != model.training_records_sha256_
                        or details["forest_state_sha256"] != _forest_state(model.estimator_, recipe["model_params"], seed, np.arange(7), rows)):
                    raise ValueError("Subject residual projector, TRAIN scope or forest changed")
            else:
                verify_pairwise_coupling_rule(model.coupling_rule_)
                if (len(model.heads_) != len(PAIR_INDICES) or len(model.pair_audits_) != len(PAIR_INDICES)
                        or details["pair_audits"] != model.pair_audits_ or details["coupling_rule"] != model.coupling_rule_
                        or details["total_trees"] != 21*recipe["model_params"]["n_estimators"]
                        or details["pair_training_row_exposures"] != 6*rows or details["calibrated_posterior"] is not False):
                    raise ValueError("OVO coupling rule, heads or training exposure changed")
                for index, (head, actual, pair) in enumerate(zip(model.heads_, model.pair_audits_, PAIR_INDICES)):
                    take, _, expected_pair = _pair_audit(recipe["model_params"], seed, labels, people, records, pair, index, None)
                    for key, value in expected_pair.items():
                        if key != "input_sha256" and actual.get(key) != value:
                            raise ValueError("OVO pair training rows, labels, people or seeds changed")
                    if actual["forest_state_sha256"] != _forest_state(head, recipe["model_params"], actual["seed"], [0, 1], len(take)):
                        raise ValueError("OVO fitted forest changed")
                    h = actual["input_sha256"]
                    if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                        raise ValueError("OVO input digest malformed")
        if details["fit_state_sha256"] != model.fit_state_sha256_ or _fit_state(model) != model.fit_state_sha256_:
            raise ValueError("Structured forest complete fitted state changed")
    except (AttributeError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Incomplete structured forest fitted state") from error
    return True


def build_structured_pipeline(spec, seed=None):
    if seed is not None:
        _seed(seed)
    return StructuredForestPipeline(validate_structured_pipeline(spec), seed=seed)


def fit_structured_pipeline(model, spec, X, y, *, groups=None, record_ids=None, validation_data=None):
    recipe = validate_structured_pipeline(spec)
    if type(model) is not StructuredForestPipeline or validate_structured_pipeline(model.spec) != recipe:
        raise ValueError("Built structured forest differs from the supplied recipe")
    return model.fit(X, y, groups=groups, record_ids=record_ids, validation_data=validation_data)
