"""TRAIN-only subject-OOF record-difficulty weighting for native forests.

The caller supplies eligible rows. No data import, validation feedback or
adaptive split fallback is performed by this versioned estimator.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .posture_core import CLASS_NAMES, feature_names, normalize_text


PIPELINE_VERSION = "hardness_pipeline_v1"
ET, RF, VOTING = "hardness_extra_trees", "hardness_random_forest", "hardness_soft_voting"
HARDNESS_FAMILIES = frozenset({ET, RF, VOTING})
DATA_POLICY = "include_ordinary_reserved_training_records_v1"
TREE_DEFAULTS = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 8, "max_features": .5}
HARDNESS_DEFAULTS = {"coefficient": .5, "n_splits": 4, "seed_offset": 1000}


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Hardness pipeline seed must be a nonnegative uint32 integer")
    return value


def validate_hardness_pipeline(spec):
    if not isinstance(spec, dict):
        raise ValueError("Hardness pipeline specification must be an object")
    value = deepcopy(spec)
    family = value.get("family")
    if not isinstance(family, str) or family not in HARDNESS_FAMILIES or set(value) - {
            "family", "base_method", "window_seconds", "feature_engineering", "preprocessing",
            "model_params", "training", "postprocessing", "members"}:
        raise ValueError("Unknown hardness family or field")
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
        raise ValueError("Hardness v1 requires canonical596 features, 3 seconds, StandardScaler and argmax")
    training, params = value["training"], value["model_params"]
    allowed_training = {"seed", "weighting", "data_policy"} | ({"hardness"} if family != VOTING else set())
    if set(training) - allowed_training:
        raise ValueError("Unknown hardness training field")
    training.setdefault("seed", 20260921)
    training.setdefault("weighting", "per_member" if family == VOTING else "class_balanced")
    _seed(training["seed"])
    if (training["weighting"] != ("per_member" if family == VOTING else "class_balanced")
            or training.get("data_policy") not in (None, DATA_POLICY)):
        raise ValueError("Hardness weighting or training-record policy is unsupported")
    if family == VOTING:
        members = value.get("members")
        if params or not isinstance(members, list) or len(members) < 2 or any(
                not isinstance(member, dict) or member.get("family") not in (ET, RF) for member in members):
            raise ValueError("Hardness voting requires distinct nonnested ET/RF members")
        value["members"] = [validate_hardness_pipeline(member) for member in members]
        if any(member["training"]["seed"] != training["seed"]
                or member["training"].get("data_policy") != training.get("data_policy") for member in value["members"]):
            raise ValueError("Hardness members must share primary seed and training-record policy")
        if len({json.dumps(member, sort_keys=True) for member in value["members"]}) != len(members):
            raise ValueError("Hardness voting member recipes must be distinct")
    else:
        training.setdefault("hardness", deepcopy(HARDNESS_DEFAULTS))
        hardness = training["hardness"]
        if not isinstance(hardness, dict) or set(hardness) != set(HARDNESS_DEFAULTS):
            raise ValueError("Hardness requires exactly coefficient, n_splits and seed_offset")
        if (type(hardness["coefficient"]) not in (int, float) or hardness["coefficient"] not in (0, .5)
                or type(hardness["n_splits"]) is not int or hardness["n_splits"] != 4
                or type(hardness["seed_offset"]) is not int or hardness["seed_offset"] != 1000):
            raise ValueError("Hardness v1 fixes coefficient0/.5, four folds and seed offset1000")
        hardness["coefficient"] = float(hardness["coefficient"])
        if "members" in value or set(params) - set(TREE_DEFAULTS):
            raise ValueError("Unknown hardness forest parameters or unexpected members")
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
            raise ValueError("max_features must be a supported feature count or fraction")
    json.dumps(value, allow_nan=False)
    return value


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    values = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([values.dtype.str, list(values.shape)]).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _features(X):
    if hasattr(X, "columns") and list(X.columns) != feature_names():
        raise ValueError("Named hardness features differ from canonical596 order")
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] != 596 or not len(values)
            or values.dtype.kind not in "iuf" or not np.isfinite(values).all()):
        raise ValueError("Hardness requires finite nonempty numeric input with shape (N,596)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(values, dtype=np.float32 if values.dtype == np.dtype(np.float32) else np.float64,
            order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Hardness features exceed the supported finite range")
    return result


def _identities(values, rows, name):
    result = np.asarray(values)
    if (result.ndim != 1 or len(result) != rows or any(
            not isinstance(item, str) or not normalize_text(item) for item in result.tolist())):
        raise ValueError(f"{name} must contain aligned nonempty string identities")
    return np.asarray([normalize_text(item) for item in result.tolist()])


def _metadata(labels, people, records):
    result = []
    for record in sorted(set(records.tolist())):
        indices = np.flatnonzero(records == record)
        if len(np.unique(people[indices])) != 1 or len(np.unique(labels[indices])) != 1:
            raise ValueError("Every training record must belong to one subject and one class")
        result.append({"record_id": record, "subject": str(people[indices[0]]), "class_id": int(labels[indices[0]]),
            "window_count": len(indices), "row_indices": indices.tolist()})
    return result


def _estimator(spec, seed):
    cls = ExtraTreesClassifier if spec["family"] == ET else RandomForestClassifier
    return cls(**deepcopy(spec["model_params"]), bootstrap=False, class_weight="balanced", n_jobs=1, random_state=seed)


def _teacher(seed):
    return ExtraTreesClassifier(**TREE_DEFAULTS, bootstrap=False, class_weight="balanced", n_jobs=1, random_state=seed)


def _scaler_payload(scaler):
    return {"params": scaler.get_params(deep=False), "samples": int(scaler.n_samples_seen_),
        "features": int(scaler.n_features_in_), **{key: getattr(scaler, key).tolist() for key in ("mean_", "scale_", "var_")}}


def _forest_digest(estimator):
    if type(estimator) not in (ExtraTreesClassifier, RandomForestClassifier):
        raise ValueError("Unknown fitted native hardness forest class")
    tree_type = ExtraTreeClassifier if type(estimator) is ExtraTreesClassifier else DecisionTreeClassifier
    if (estimator.n_features_in_ != 596 or estimator.n_outputs_ != 1
            or not np.array_equal(estimator.classes_, np.arange(7)) or len(estimator.estimators_) != estimator.n_estimators):
        raise ValueError("Hardness forest classes, feature count or tree count changed")
    trees = []
    for tree in estimator.estimators_:
        if (type(tree) is not tree_type or tree.n_features_in_ != 596 or tree.n_outputs_ != 1
                or not np.array_equal(tree.classes_, np.arange(7))):
            raise ValueError("Hardness tree class or schema changed")
        state = tree.tree_.__getstate__()
        trees.append({"params": tree.get_params(deep=False), "classes": _array_digest(tree.classes_),
            "n_classes": int(tree.n_classes_), "node_count": int(state["node_count"]), "max_depth": int(state["max_depth"]),
            "values": _array_digest(state["values"]),
            "nodes": {name: _array_digest(state["nodes"][name]) for name in state["nodes"].dtype.names}})
    return _json_digest({"class": type(estimator).__name__, "params": estimator.get_params(deep=False),
        "classes": _array_digest(estimator.classes_), "trees": trees})


def hardness_rule(coefficient):
    return {"version": PIPELINE_VERSION, "coefficient": coefficient, "n_splits": 4, "shuffle": False,
        "seed_offset": 1000, "fold_seed": "(effective_seed+1000+zero_based_fold)%2**32",
        "teacher": {"family": "extra_trees", **TREE_DEFAULTS, "bootstrap": False, "class_weight": "balanced", "n_jobs": 1},
        "error": "e_r=mean(OOF_argmax_prediction!=class) across windows of record r",
        "raw_factor": "h_r=1+coefficient*e_r",
        "sample_weight": "a_i=h_record(i)/mean(h_record(j) for TRAIN windows j in class y_i)",
        "normalization_unit": "window mean within class, not equal record mass",
        "final_weight": "native class_weight=balanced multiplied by a_i",
        "scaler": "each OOF scaler fits only innerTRAIN; final scaler fits all outerTRAIN unweighted",
        "zero_coefficient": "skip OOF completely and omit sample_weight argument",
        "final_bootstrap": False, "fit_scope": "supplied TRAIN only", "external_validation_used": False,
        "fallback": "none; positive hardness requires four subjects and all seven classes in every innerTRAIN"}


def _splits(labels, people):
    if len(np.unique(people)) < 4:
        raise ValueError("Positive hardness requires at least four outerTRAIN subjects")
    folds = list(GroupKFold(n_splits=4, shuffle=False).split(np.zeros((len(labels), 1)), labels, people))
    for train, heldout in folds:
        if len(np.unique(people[train])) < 2 or not np.array_equal(np.unique(labels[train]), np.arange(7)):
            raise ValueError("Every hardness innerTRAIN needs at least two subjects and all seven classes")
        if set(people[train]) & set(people[heldout]):
            raise ValueError("Hardness inner subjects overlap")
    return folds


def _record_weights(metadata, labels, predictions, coefficient):
    raw = np.empty(len(labels), dtype=np.float64)
    records = []
    for row in metadata:
        idx = np.asarray(row["row_indices"], dtype=np.int64)
        mistakes = int(np.count_nonzero(predictions[idx] != labels[idx]))
        error = mistakes / len(idx)
        factor = 1. + coefficient * error
        raw[idx] = factor
        records.append({"record_id": row["record_id"], "subject": row["subject"], "class_id": row["class_id"],
            "window_count": len(idx), "error_count": mistakes, "error_rate": error, "raw_factor": factor})
    denominators = np.array([raw[labels == c].mean() for c in range(7)])
    weights = raw / denominators[labels]
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid hardness sample weights")
    counts = np.bincount(labels, minlength=7)
    sums = np.bincount(labels, weights=weights, minlength=7)
    if not np.allclose(sums, counts, rtol=1e-12, atol=1e-12):
        raise ValueError("Hardness class window mass was not preserved")
    for row in records:
        row["sample_weight"] = row["raw_factor"] / denominators[row["class_id"]]
    return weights, {"records": records, "class_denominators": denominators.tolist(),
        "class_sample_weight_sums": sums.tolist(), "minimum": float(weights.min()), "maximum": float(weights.max()),
        "sample_weight_sha256": _array_digest(weights)}


def _oof(values, labels, people, metadata, seed, coefficient):
    prediction = np.full(len(labels), -1, dtype=np.int64)
    probability = np.full((len(labels), 7), np.nan, dtype=np.float64)
    assignments = np.full(len(labels), -1, dtype=np.int64)
    audits = []
    for index, (train, heldout) in enumerate(_splits(labels, people)):
        inner_seed = (seed + 1000 + index) % 2**32
        scaler = StandardScaler()
        estimator = _teacher(inner_seed).fit(scaler.fit_transform(values[train]), labels[train])
        probability[heldout] = estimator.predict_proba(scaler.transform(values[heldout]))
        prediction[heldout], assignments[heldout] = probability[heldout].argmax(1), index
        scaler_state = _scaler_payload(scaler)
        audits.append({"fold": index, "seed": inner_seed, "train_indices": train.tolist(), "heldout_indices": heldout.tolist(),
            "train_subjects": sorted(set(people[train].tolist())), "heldout_subjects": sorted(set(people[heldout].tolist())),
            "train_class_counts": np.bincount(labels[train], minlength=7).tolist(),
            "heldout_class_counts": np.bincount(labels[heldout], minlength=7).tolist(),
            "training_input_sha256": _array_digest(values[train]), "training_labels_sha256": _array_digest(labels[train]),
            "heldout_input_sha256": _array_digest(values[heldout]), "scaler": scaler_state,
            "scaler_state_sha256": _json_digest(scaler_state), "forest_state_sha256": _forest_digest(estimator),
            "model_params": estimator.get_params(deep=False), "prediction_sha256": _array_digest(prediction[heldout])})
    if np.any(prediction < 0) or np.any(assignments < 0):
        raise ValueError("Hardness OOF predictions do not cover every training row")
    weights, weighting = _record_weights(metadata, labels, prediction, coefficient)
    return prediction, probability, assignments, weights, {"oof_performed": True, "folds": audits,
        "oof_prediction": prediction.tolist(), "oof_probability_sha256": _array_digest(probability),
        "oof_fold": assignments.tolist(), "weighting": weighting}


def _fit_state(model):
    state = {"pipeline": model.spec_, "seed": model.effective_seed_, "training_input": model.training_input_sha256_,
        "training_labels": model.training_labels_sha256_, "training_groups": model.training_groups_sha256_,
        "training_records": model.training_records_sha256_, "metadata": model.record_metadata_,
        "training_feature_dtype": model.training_details_["training_feature_dtype"]}
    if model.spec_["family"] == VOTING:
        state["members"] = [member.fit_state_sha256_ for member in model.members_]
    else:
        state.update(scaler=_json_digest(_scaler_payload(model.scaler_)), forest=_forest_digest(model.estimator_),
            hardness=model.hardness_audit_, rule=model.hardness_rule_,
            sample_weight=None if model.sample_weight_ is None else _array_digest(model.sample_weight_),
            oof_prediction=None if model.oof_prediction_ is None else _array_digest(model.oof_prediction_),
            oof_probability=None if model.oof_probability_ is None else _array_digest(model.oof_probability_),
            oof_fold=None if model.oof_fold_ is None else _array_digest(model.oof_fold_))
    return _json_digest(state)


class HardnessPipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, spec, seed=None):
        self.spec, self.seed = spec, seed

    def fit(self, X, y, *, groups=None, record_ids=None, validation_data=None):
        spec = validate_hardness_pipeline(self.spec)
        seed = _seed(spec["training"]["seed"] if self.seed is None else self.seed)
        values, labels = _features(X), np.asarray(y)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu"
                or not np.array_equal(np.unique(labels), np.arange(7))):
            raise ValueError("Hardness training requires aligned integer labels with all seven posture classes")
        labels = labels.astype(np.int64, copy=True)
        people = _identities(groups, len(values), "groups")
        records = _identities(record_ids, len(values), "record_ids")
        metadata = _metadata(labels, people, records)
        with threadpool_limits(limits=1):
            if spec["family"] == VOTING:
                members = [build_hardness_pipeline(member, seed=self.seed).fit(values, labels, groups=people, record_ids=records)
                    for member in spec["members"]]
            else:
                coefficient = spec["training"]["hardness"]["coefficient"]
                prediction, probability, assignments, weights = None, None, None, None
                audit = {"oof_performed": False, "folds": [], "oof_prediction": None, "oof_probability_sha256": None,
                    "oof_fold": None, "weighting": None}
                if coefficient:
                    prediction, probability, assignments, weights, audit = _oof(values, labels, people, metadata, seed, coefficient)
                scaler = StandardScaler()
                transformed = scaler.fit_transform(values)
                estimator = _estimator(spec, seed)
                if weights is None:
                    estimator.fit(transformed, labels)
                else:
                    estimator.fit(transformed, labels, sample_weight=weights)
        self.spec_, self.effective_seed_, self.classes_ = spec, seed, np.arange(7)
        self.n_features_in_, self.feature_names_in_, self.label_names_ = 596, np.asarray(feature_names()), list(CLASS_NAMES)
        self.record_metadata_, self.training_identity_sha256_ = metadata, _json_digest(metadata)
        self.class_counts_ = np.bincount(labels, minlength=7)
        self.training_input_sha256_, self.training_labels_sha256_ = _array_digest(values), _array_digest(labels)
        self.training_groups_sha256_, self.training_records_sha256_ = _json_digest(people.tolist()), _json_digest(records.tolist())
        self.training_details_ = {"pipeline": deepcopy(spec), "pipeline_version": PIPELINE_VERSION, "effective_seed": seed,
            "input_contract": "feature_vector", "input_features": 596, "feature_names": feature_names(), "class_names": list(CLASS_NAMES),
            "training_windows": len(values), "training_feature_dtype": str(values.dtype), "class_counts": self.class_counts_.tolist(),
            "training_records": len(metadata), "record_metadata": deepcopy(metadata), "training_subjects": sorted(set(people.tolist())),
            "training_identity_sha256": self.training_identity_sha256_, "training_input_sha256": self.training_input_sha256_,
            "training_labels_sha256": self.training_labels_sha256_, "training_groups_sha256": self.training_groups_sha256_,
            "training_records_sha256": self.training_records_sha256_, "external_validation_used": False,
            "training_record_policy": spec["training"].get("data_policy"), "training_record_policy_application": "caller supplies eligible TRAIN records only",
            "evaluation_population_normalization": False, "threads": 1, "numpy_version": np.__version__, "sklearn_version": sklearn.__version__}
        if spec["family"] == VOTING:
            self.members_ = members
            self.training_details_.update(combination="equal aligned probability mean", weights=[1 / len(members)] * len(members),
                members=[deepcopy(member.training_details_) for member in members])
        else:
            self.scaler_, self.estimator_ = scaler, estimator
            self.oof_prediction_, self.oof_fold_, self.sample_weight_, self.hardness_audit_ = prediction, assignments, weights, audit
            self.oof_probability_ = probability
            self.hardness_rule_ = hardness_rule(coefficient)
            self.scaler_state_sha256_, self.estimator_state_sha256_ = _json_digest(_scaler_payload(scaler)), _forest_digest(estimator)
            self.training_details_.update(hardness_rule=deepcopy(self.hardness_rule_), hardness=deepcopy(audit),
                scaler_state_sha256=self.scaler_state_sha256_, estimator_state_sha256=self.estimator_state_sha256_,
                effective_model_params=estimator.get_params(deep=False), final_bootstrap=False,
                preprocessing_fit="innerTRAIN scaler per OOF fold; unweighted outerTRAIN scaler before final native forest",
                sample_weight_argument_used=weights is not None)
        self.fit_state_sha256_ = _fit_state(self)
        self.training_details_["fit_state_sha256"] = self.fit_state_sha256_
        verify_fitted_hardness_identity(self, spec, seed)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "spec_")
        if validate_hardness_pipeline(self.spec) != self.spec_ or _seed(
                self.spec_["training"]["seed"] if self.seed is None else self.seed) != self.effective_seed_:
            raise ValueError("Hardness recipe or effective seed changed after fitting")
        values = _features(X)
        with threadpool_limits(limits=1):
            probability = np.mean([member.predict_proba(values) for member in self.members_], axis=0) if self.spec_["family"] == VOTING else self.estimator_.predict_proba(self.scaler_.transform(values))
        if (probability.shape != (len(values), 7) or not np.isfinite(probability).all() or (probability < 0).any()
                or not np.allclose(probability.sum(1), 1., rtol=0, atol=1e-12)):
            raise ValueError("Invalid hardness native forest probabilities")
        return probability

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def build_hardness_pipeline(spec, seed=None):
    if seed is not None:
        _seed(seed)
    return HardnessPipeline(validate_hardness_pipeline(spec), seed=seed)


def fit_hardness_pipeline(model, spec, X, y, *, groups=None, record_ids=None, validation_data=None):
    recipe = validate_hardness_pipeline(spec)
    if type(model) is not HardnessPipeline or validate_hardness_pipeline(model.spec) != recipe:
        raise ValueError("Constructed hardness pipeline differs from requested recipe")
    return model.fit(X, y, groups=groups, record_ids=record_ids, validation_data=validation_data)


def _digest_link(model, details, name):
    value = getattr(model, name + "_", None)
    if (not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            or details.get(name) != value):
        raise ValueError("Hardness training digest or audit link changed")


def _restored_identities(metadata, count):
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Hardness record metadata missing")
    labels = np.full(count, -1, dtype=np.int64)
    people, records, seen = [None] * count, [None] * count, np.zeros(count, dtype=np.int64)
    for row in metadata:
        indices = row["row_indices"]
        if (not isinstance(indices, list) or row["window_count"] != len(indices) or not indices
                or any(type(i) is not int or not 0 <= i < count for i in indices)
                or type(row["class_id"]) is not int or not 0 <= row["class_id"] < 7):
            raise ValueError("Hardness record row mapping is invalid")
        for i in indices:
            labels[i], people[i], records[i] = row["class_id"], row["subject"], row["record_id"]
            seen[i] += 1
    if not np.all(seen == 1):
        raise ValueError("Hardness record metadata must cover each TRAIN row exactly once")
    people, records = _identities(people, count, "groups"), _identities(records, count, "record_ids")
    if _metadata(labels, people, records) != metadata:
        raise ValueError("Hardness record metadata is not a canonical single-subject/class mapping")
    return labels, people, records


def _verify_oof(model, labels, people, seed, coefficient):
    audit, prediction, assignments, weights = model.hardness_audit_, model.oof_prediction_, model.oof_fold_, model.sample_weight_
    probability = model.oof_probability_
    if not coefficient:
        if (prediction is not None or probability is not None or assignments is not None or weights is not None
                or audit != {"oof_performed": False, "folds": [], "oof_prediction": None, "oof_probability_sha256": None,
                    "oof_fold": None, "weighting": None}):
            raise ValueError("Zero hardness must skip OOF and sample weighting entirely")
        return
    if (not isinstance(prediction, np.ndarray) or prediction.shape != labels.shape or prediction.dtype != np.dtype(np.int64)
            or np.any(prediction < 0) or np.any(prediction > 6) or not isinstance(assignments, np.ndarray)
            or assignments.dtype != np.dtype(np.int64) or assignments.shape != labels.shape
            or audit.get("oof_performed") is not True or audit.get("oof_prediction") != prediction.tolist()
            or audit.get("oof_fold") != assignments.tolist() or len(audit.get("folds", ())) != 4):
        raise ValueError("Hardness OOF row predictions or assignment audit changed")
    if (not isinstance(probability, np.ndarray) or probability.dtype != np.dtype(np.float64)
            or probability.shape != (len(labels), 7) or not np.isfinite(probability).all() or (probability < 0).any()
            or not np.allclose(probability.sum(1), 1., rtol=0, atol=1e-12)
            or not np.array_equal(probability.argmax(1), prediction)
            or audit.get("oof_probability_sha256") != _array_digest(probability)):
        raise ValueError("Hardness OOF aligned probabilities changed")
    for index, ((train, heldout), row) in enumerate(zip(_splits(labels, people), audit["folds"])):
        inner_seed = (seed + 1000 + index) % 2**32
        expected = {"fold": index, "seed": inner_seed, "train_indices": train.tolist(), "heldout_indices": heldout.tolist(),
            "train_subjects": sorted(set(people[train].tolist())), "heldout_subjects": sorted(set(people[heldout].tolist())),
            "train_class_counts": np.bincount(labels[train], minlength=7).tolist(),
            "heldout_class_counts": np.bincount(labels[heldout], minlength=7).tolist(),
            "training_labels_sha256": _array_digest(labels[train]), "prediction_sha256": _array_digest(prediction[heldout]),
            "model_params": _teacher(inner_seed).get_params(deep=False)}
        if any(row.get(key) != value for key, value in expected.items()) or not np.all(assignments[heldout] == index):
            raise ValueError("Hardness OOF fold identity, subject exclusion, parameters or seeds changed")
        scaler = row.get("scaler", {})
        if (scaler.get("params") != StandardScaler().get_params(deep=False) or scaler.get("samples") != len(train)
                or scaler.get("features") != 596 or row.get("scaler_state_sha256") != _json_digest(scaler)):
            raise ValueError("Hardness OOF scaler scope or state hash changed")
        for key in ("mean_", "scale_", "var_"):
            values = np.asarray(scaler.get(key, ()))
            if values.shape != (596,) or not np.isfinite(values).all() or (key == "scale_" and np.any(values <= 0)) or (key == "var_" and np.any(values < 0)):
                raise ValueError("Hardness OOF scaler learned statistics are invalid")
        for key in ("training_input_sha256", "heldout_input_sha256", "forest_state_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Hardness OOF fit evidence hash is invalid")
    expected_weights, expected_audit = _record_weights(model.record_metadata_, labels, prediction, coefficient)
    if (not isinstance(weights, np.ndarray) or weights.dtype != np.dtype(np.float64)
            or not np.array_equal(weights, expected_weights) or audit.get("weighting") != expected_audit):
        raise ValueError("Hardness record errors or class-normalized weights changed")


def verify_fitted_hardness_identity(model, spec, seed):
    recipe, seed = validate_hardness_pipeline(spec), _seed(seed)
    if (type(model) is not HardnessPipeline or getattr(model, "spec_", None) != recipe
            or validate_hardness_pipeline(model.spec) != recipe or getattr(model, "effective_seed_", None) != seed
            or (recipe["training"]["seed"] if model.seed is None else model.seed) != seed
            or getattr(model, "n_features_in_", None) != 596 or not np.array_equal(getattr(model, "classes_", ()), np.arange(7))
            or not np.array_equal(getattr(model, "feature_names_in_", ()), feature_names())
            or getattr(model, "label_names_", None) != list(CLASS_NAMES)):
        raise ValueError("Fitted hardness recipe, seed or class/feature schema changed")
    try:
        details = model.training_details_
        count = details.get("training_windows")
        if type(count) is not int or count < 7:
            raise ValueError("Hardness training row count is invalid")
        labels, people, records = _restored_identities(model.record_metadata_, count)
        expected = {"pipeline": recipe, "pipeline_version": PIPELINE_VERSION, "effective_seed": seed,
            "input_contract": "feature_vector", "input_features": 596, "feature_names": feature_names(), "class_names": list(CLASS_NAMES),
            "class_counts": np.bincount(labels, minlength=7).tolist(), "training_records": len(model.record_metadata_),
            "record_metadata": model.record_metadata_, "training_subjects": sorted(set(people.tolist())),
            "external_validation_used": False, "training_record_policy": recipe["training"].get("data_policy"),
            "training_record_policy_application": "caller supplies eligible TRAIN records only", "evaluation_population_normalization": False,
            "threads": 1, "numpy_version": np.__version__, "sklearn_version": sklearn.__version__}
        if (any(details.get(key) != value for key, value in expected.items())
                or details.get("training_feature_dtype") not in ("float32", "float64")
                or not np.array_equal(model.class_counts_, np.bincount(labels, minlength=7))
                or (model.class_counts_ <= 0).any() or model.training_identity_sha256_ != _json_digest(model.record_metadata_)
                or model.training_labels_sha256_ != _array_digest(labels) or model.training_groups_sha256_ != _json_digest(people.tolist())
                or model.training_records_sha256_ != _json_digest(records.tolist())):
            raise ValueError("Hardness TRAIN identities, labels or audit changed")
        for key in ("training_identity_sha256", "training_input_sha256", "training_labels_sha256", "training_groups_sha256", "training_records_sha256", "fit_state_sha256"):
            _digest_link(model, details, key)
        if recipe["family"] == VOTING:
            if len(model.members_) != len(recipe["members"]):
                raise ValueError("Hardness ensemble member count changed")
            for member, member_recipe in zip(model.members_, recipe["members"]):
                verify_fitted_hardness_identity(member, member_recipe, seed)
                if any(member.training_details_[key] != details[key] for key in ("training_input_sha256", "training_labels_sha256",
                        "training_groups_sha256", "training_records_sha256", "record_metadata", "training_feature_dtype")):
                    raise ValueError("Hardness ensemble members trained on different TRAIN scopes")
            if (details.get("members") != [m.training_details_ for m in model.members_]
                    or details.get("weights") != [1 / len(model.members_)] * len(model.members_)
                    or details.get("combination") != "equal aligned probability mean"):
                raise ValueError("Hardness ensemble audit changed")
        else:
            coefficient = recipe["training"]["hardness"]["coefficient"]
            rule = hardness_rule(coefficient)
            if model.hardness_rule_ != rule or details.get("hardness_rule") != rule or details.get("hardness") != model.hardness_audit_:
                raise ValueError("Hardness rule or recorded OOF audit changed")
            _verify_oof(model, labels, people, seed, coefficient)
            scaler, estimator, expected_estimator = model.scaler_, model.estimator_, _estimator(recipe, seed)
            if (type(scaler) is not StandardScaler or scaler.get_params(deep=False) != StandardScaler().get_params(deep=False)
                    or scaler.n_samples_seen_ != count or scaler.n_features_in_ != 596
                    or _json_digest(_scaler_payload(scaler)) != model.scaler_state_sha256_
                    or details.get("scaler_state_sha256") != model.scaler_state_sha256_
                    or type(estimator) is not type(expected_estimator) or estimator.get_params(deep=False) != expected_estimator.get_params(deep=False)
                    or details.get("effective_model_params") != expected_estimator.get_params(deep=False)
                    or details.get("final_bootstrap") is not False or details.get("sample_weight_argument_used") is not bool(coefficient)
                    or details.get("preprocessing_fit") != "innerTRAIN scaler per OOF fold; unweighted outerTRAIN scaler before final native forest"
                    or any(tree.tree_.n_node_samples[0] != count for tree in estimator.estimators_)
                    or _forest_digest(estimator) != model.estimator_state_sha256_
                    or details.get("estimator_state_sha256") != model.estimator_state_sha256_):
                raise ValueError("Hardness final scaler, forest, parameters or row scope changed")
            for tree in estimator.estimators_:
                if (not np.isclose(tree.tree_.weighted_n_node_samples[0], count, atol=1e-9, rtol=1e-12)
                        or not np.allclose(tree.tree_.value[0, 0], np.full(7, 1/7), atol=1e-12, rtol=0)):
                    raise ValueError("Hardness native class-balanced root masses changed")
        if _fit_state(model) != model.fit_state_sha256_:
            raise ValueError("Hardness complete fitted state hash changed")
    except (AttributeError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Incomplete hardness fitted state") from error
    return True
