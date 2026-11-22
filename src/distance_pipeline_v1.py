"""Fixed distance kernels with TRAIN-only scaling and native precomputed SVC.

The complete standardized TRAIN reference is part of the saved model. Native
libsvm probability fitting uses non-grouped folds within TRAIN; this is not a
claim of subject-independent probability calibration.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .distance_kernels_v1 import distance_kernel, distance_kernel_rule, verify_distance_kernel_rule
from .posture_core import CLASS_NAMES, feature_names, normalize_text


VERSION = "distance_pipeline_v1"
SINGLE, VOTING = "distance_svm", "distance_soft_voting"
FAMILIES = frozenset({SINGLE, VOTING})
POLICY = "include_ordinary_reserved_training_records_v1"
KINDS = frozenset({"rbf", "laplacian", "rational_quadratic"})


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Distance SVM seed must be a uint32 integer")
    return value


def validate_distance_pipeline(spec):
    if not isinstance(spec, dict):
        raise ValueError("Distance recipe must be an object")
    value = deepcopy(spec)
    family = value.get("family")
    if not isinstance(family, str) or family not in FAMILIES or set(value) - {
            "family", "base_method", "window_seconds", "feature_engineering", "preprocessing",
            "model_params", "training", "postprocessing", "members"}:
        raise ValueError("Unknown distance recipe family or field")
    for key, default in {"base_method": "standard", "window_seconds": 3,
            "feature_engineering": {"mode": "all"}, "preprocessing": {"scaler": "standard"},
            "model_params": {}, "training": {}, "postprocessing": {"kind": "argmax"}}.items():
        value.setdefault(key, deepcopy(default))
    for key in ("feature_engineering", "preprocessing", "model_params", "training", "postprocessing"):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} must be an object")
    if (value["base_method"] != "standard" or type(value["window_seconds"]) is not int or value["window_seconds"] != 3
            or value["feature_engineering"] != {"mode": "all"} or value["preprocessing"] != {"scaler": "standard"}
            or value["postprocessing"] != {"kind": "argmax"}):
        raise ValueError("Distance v1 requires canonical596, fixed3s, StandardScaler and probability argmax")
    training, params = value["training"], value["model_params"]
    if set(training) - {"seed", "weighting", "data_policy"}:
        raise ValueError("Unknown distance training field")
    training.setdefault("seed", 20260921)
    training.setdefault("weighting", "class_balanced")
    _seed(training["seed"])
    if training["weighting"] != "class_balanced" or training.get("data_policy") not in (None, POLICY):
        raise ValueError("Distance training requires native balanced weights and a known record policy")
    if family == VOTING:
        members = value.get("members")
        if params or not isinstance(members, list) or len(members) < 2 or any(
                not isinstance(member, dict) or member.get("family") != SINGLE for member in members):
            raise ValueError("Distance voting requires at least two nonnested SVM leaf recipes")
        value["members"] = [validate_distance_pipeline(member) for member in members]
        if any(member["training"] != training for member in value["members"]):
            raise ValueError("Distance voting members must share training seed, weights and record policy")
        if len({_json_digest(member) for member in value["members"]}) != len(members):
            raise ValueError("Distance voting members must be distinct")
    else:
        if "members" in value or set(params) - {"C", "kernel"}:
            raise ValueError("Unknown fixed distance model parameters")
        params.setdefault("C", 1.)
        params.setdefault("kernel", "laplacian")
        if type(params["C"]) not in (int, float) or not np.isfinite(params["C"]) or params["C"] <= 0:
            raise ValueError("SVC C must be a finite positive number")
        params["C"] = float(params["C"])
        if not isinstance(params["kernel"], str) or params["kernel"] not in KINDS:
            raise ValueError("Unknown fixed distance kernel")
    _json_digest(value)
    return value


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    array = np.ascontiguousarray(value)
    result = hashlib.sha256(json.dumps([array.dtype.str, list(array.shape)]).encode())
    result.update(array.tobytes())
    return result.hexdigest()


def _features(X):
    if hasattr(X, "columns") and list(X.columns) != feature_names():
        raise ValueError("Distance input columns differ from canonical596")
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] != 596 or not len(values) or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()):
        raise ValueError("Distance input must be finite nonempty numeric(N,596)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(values, dtype=np.float32 if values.dtype == np.dtype(np.float32) else np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Distance input exceeds the supported finite range")
    return result


def _identities(values, rows, name):
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != rows or any(not isinstance(v, str) or not normalize_text(v) for v in array):
        raise ValueError(f"Aligned nonempty TRAIN {name} strings are required")
    return np.asarray([normalize_text(v) for v in array])


def _records(labels, people, records):
    metadata = []
    for record in sorted(set(records.tolist())):
        take = np.flatnonzero(records == record)
        if len(np.unique(labels[take])) != 1 or len(np.unique(people[take])) != 1:
            raise ValueError("A TRAIN recording must have one person and one posture")
        metadata.append({"record_id": record, "subject": str(people[take[0]]), "class_id": int(labels[take[0]]),
            "row_indices": take.tolist(), "window_count": len(take)})
    return metadata


def _backend(C, seed):
    return SVC(C=C, kernel="precomputed", probability=True, class_weight="balanced", random_state=seed,
        decision_function_shape="ovr", shrinking=True, tol=1e-3, cache_size=200., max_iter=-1,
        verbose=False, break_ties=False)


def _state_value(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("Native SVC state must contain finite numeric arrays")
        return {"array_sha256": _array_digest(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_state_value(v) for v in value]
    if isinstance(value, dict):
        return {key: _state_value(v) for key, v in value.items()}
    if value is None or type(value) in (int, float, str, bool):
        return value
    raise ValueError("Unexpected native SVC serialized state type")


def _scaler_state(scaler):
    return {"params": scaler.get_params(deep=False), "rows": int(scaler.n_samples_seen_),
        "features": int(scaler.n_features_in_), **{name: _array_digest(getattr(scaler, name)) for name in ("mean_", "var_", "scale_")}}


def _svc_state(model, C, seed, labels):
    rows = len(labels)
    if (type(model) is not SVC or model.get_params(deep=False) != _backend(C, seed).get_params(deep=False)
            or not np.array_equal(model.classes_, np.arange(7)) or model.n_features_in_ != rows
            or model.shape_fit_ != (rows, rows) or model.fit_status_ != 0 or model._sparse is not False or model._gamma != 0.
            or model.support_vectors_.shape != (0, 0) or model.support_vectors_.dtype != np.dtype(np.float64)):
        raise ValueError("Native precomputed SVC parameters, fit status or training shape changed")
    counts = np.bincount(labels, minlength=7)
    if not np.array_equal(model.class_weight_, rows/(7*counts)):
        raise ValueError("Native SVC balanced class mass changed")
    support = model.support_
    if (not isinstance(support, np.ndarray) or support.ndim != 1 or support.dtype.kind not in "iu"
            or not len(support) or len(np.unique(support)) != len(support) or np.any(support < 0) or np.any(support >= rows)
            or model._n_support.shape != (7,) or np.any(model._n_support <= 0) or model._n_support.sum() != len(support)):
        raise ValueError("Native SVC support reference indices changed")
    start = 0
    for label, count in enumerate(model._n_support):
        block = support[start:start+count]
        if not np.all(labels[block] == label) or np.any(np.diff(block) <= 0):
            raise ValueError("Native SVC support order does not match the TRAIN labels")
        start += count
    for name, shape in {"dual_coef_": (6, len(support)), "_dual_coef_": (6, len(support)),
            "intercept_": (21,), "_intercept_": (21,), "_probA": (21,), "_probB": (21,)}.items():
        value = getattr(model, name)
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.dtype(np.float64) or not np.isfinite(value).all():
            raise ValueError("Native SVC support/decision/probability state changed")
    if (not np.array_equal(model.dual_coef_, model._dual_coef_) or not np.array_equal(model.intercept_, model._intercept_)
            or not np.array_equal(model.n_iter_, model._num_iter)):
        raise ValueError("Native SVC internal/public fitted states disagree")
    return _json_digest({key: _state_value(value) for key, value in vars(model).items()})


class DistancePipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, spec, seed=None):
        self.spec, self.seed = spec, seed

    def fit(self, X, y, *, groups=None, record_ids=None, validation_data=None):
        recipe = validate_distance_pipeline(self.spec)
        seed = _seed(recipe["training"]["seed"] if self.seed is None else self.seed)
        values, labels = _features(X), np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu" or not np.array_equal(np.unique(labels), np.arange(7)):
            raise ValueError("Distance SVM needs all seven canonical integer TRAIN classes")
        labels = labels.astype(np.int64, copy=True)
        people, records = _identities(groups, len(labels), "groups"), _identities(record_ids, len(labels), "record_ids")
        metadata = _records(labels, people, records)
        self.spec_, self.effective_seed_ = recipe, seed
        self.classes_, self.label_names_, self.n_features_in_ = np.arange(7), list(CLASS_NAMES), 596
        self.feature_names_in_, self.record_metadata_ = np.asarray(feature_names()), metadata
        self.training_labels_ = labels
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
                self.members_ = [build_distance_pipeline(member, seed=seed).fit(values, labels, groups=people, record_ids=records)
                    for member in recipe["members"]]
                self.training_details_.update(combination="equal aligned native probability mean",
                    members=[deepcopy(member.training_details_) for member in self.members_])
            else:
                self.scaler_ = StandardScaler()
                transformed = self.scaler_.fit_transform(values)
                self.reference_ = np.array(transformed, dtype=np.float64, order="C", copy=True)
                kind = recipe["model_params"]["kernel"]
                self.kernel_rule_ = distance_kernel_rule(kind)
                gram = distance_kernel(self.reference_, self.reference_, kind)
                if not np.array_equal(gram, gram.T) or not np.array_equal(gram.diagonal(), np.ones(len(labels))):
                    raise ValueError("TRAIN kernel must be exactly symmetric with unit diagonal")
                self.estimator_ = _backend(recipe["model_params"]["C"], seed).fit(gram, labels)
                self.training_details_.update(scaler_state_sha256=_json_digest(_scaler_state(self.scaler_)),
                    standardized_input_sha256=_array_digest(transformed), reference_sha256=_array_digest(self.reference_),
                    reference_rows=len(labels), reference_dtype="float64", reference_order="C",
                    gram_sha256=_array_digest(gram), gram_shape=list(gram.shape), gram_bytes=gram.nbytes,
                    kernel_rule=deepcopy(self.kernel_rule_),
                    native_svc_state_sha256=_svc_state(self.estimator_, recipe["model_params"]["C"], seed, labels),
                    probability_calibration="native libsvm non-grouped cross-validation within TRAIN; not subject-independent calibration",
                    probability_postprocessing="native precomputed SVC predict_proba; canonical7 class argmax; no threshold/temperature tuning",
                    native_gamma="ignored by precomputed SVC; actual coefficient fixed in the kernel rule")
        self.fit_state_sha256_ = _fit_state(self)
        self.training_details_["fit_state_sha256"] = self.fit_state_sha256_
        verify_fitted_distance_identity(self, recipe, seed)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "fit_state_sha256_")
        if validate_distance_pipeline(self.spec) != self.spec_ or _seed(
                self.spec_["training"]["seed"] if self.seed is None else self.seed) != self.effective_seed_:
            raise ValueError("Distance recipe or seed changed after fitting")
        values = _features(X)
        with threadpool_limits(limits=1):
            if self.spec_["family"] == VOTING:
                probability = np.mean([member.predict_proba(values) for member in self.members_], axis=0)
            else:
                kind = self.spec_["model_params"]["kernel"]
                verify_distance_kernel_rule(self.kernel_rule_, kind)
                if (self.reference_.shape != (self.training_details_["rows"], 596) or self.reference_.dtype != np.dtype(np.float64)
                        or not self.reference_.flags.c_contiguous or not np.array_equal(self.estimator_.classes_, np.arange(7))):
                    raise ValueError("Distance TRAIN reference or native class order changed")
                query = self.scaler_.transform(values)
                probability = self.estimator_.predict_proba(distance_kernel(query, self.reference_, kind))
        if (probability.shape != (len(values), 7) or not np.isfinite(probability).all() or np.any(probability < 0)
                or not np.allclose(probability.sum(1), 1., rtol=0, atol=1e-12)):
            raise ValueError("Invalid distance SVM output probability")
        return probability

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def _fit_state(model):
    details = deepcopy(model.training_details_)
    details.pop("fit_state_sha256", None)
    details.pop("training_method", None)
    if model.spec_["family"] == VOTING:
        state = [member.fit_state_sha256_ for member in model.members_]
    else:
        state = [_json_digest(_scaler_state(model.scaler_)), _array_digest(model.reference_),
            _json_digest(model.kernel_rule_), _svc_state(model.estimator_, model.spec_["model_params"]["C"],
                model.effective_seed_, model.training_labels_)]
    return _json_digest({"details": details, "state": state, "labels": _array_digest(model.training_labels_)})


def verify_fitted_distance_identity(model, spec, seed):
    recipe, seed = validate_distance_pipeline(spec), _seed(seed)
    if (type(model) is not DistancePipeline or getattr(model, "spec_", None) != recipe
            or validate_distance_pipeline(model.spec) != recipe or getattr(model, "effective_seed_", None) != seed
            or (recipe["training"]["seed"] if model.seed is None else model.seed) != seed
            or getattr(model, "n_features_in_", None) != 596 or not np.array_equal(getattr(model, "classes_", ()), np.arange(7))
            or not np.array_equal(getattr(model, "feature_names_in_", ()), feature_names()) or getattr(model, "label_names_", None) != list(CLASS_NAMES)):
        raise ValueError("Distance fitted recipe, seed or canonical schema changed")
    try:
        details, labels = model.training_details_, model.training_labels_
        rows = details["rows"]
        if (type(rows) is not int or rows < 7 or not isinstance(labels, np.ndarray) or labels.shape != (rows,)
                or labels.dtype != np.dtype(np.int64) or not np.array_equal(np.unique(labels), np.arange(7))):
            raise ValueError("Distance TRAIN dimensions or labels changed")
        people, records, seen = [None]*rows, [None]*rows, np.zeros(rows, dtype=np.int64)
        for row in model.record_metadata_:
            idx = row["row_indices"]
            if (not isinstance(idx, list) or not idx or len(idx) != row["window_count"]
                    or any(type(i) is not int or not 0 <= i < rows for i in idx)
                    or type(row["class_id"]) is not int or not 0 <= row["class_id"] < 7
                    or not np.all(labels[idx] == row["class_id"])):
                raise ValueError("Distance record mapping differs from TRAIN labels")
            for i in idx:
                people[i], records[i] = row["subject"], row["record_id"]
                seen[i] += 1
        if not np.all(seen == 1):
            raise ValueError("Distance record mapping must cover every TRAIN row exactly once")
        people, records = _identities(people, rows, "groups"), _identities(records, rows, "record_ids")
        if _records(labels, people, records) != model.record_metadata_:
            raise ValueError("Distance canonical recording metadata changed")
        expected = {"version": VERSION, "pipeline": recipe, "effective_seed": seed, "input_contract": "feature_vector",
            "features": 596, "feature_names": feature_names(), "class_names": list(CLASS_NAMES),
            "class_counts": np.bincount(labels, minlength=7).tolist(), "subjects": sorted(set(people.tolist())),
            "record_metadata": model.record_metadata_, "external_validation_used": False, "evaluation_batch_adaptation": False,
            "threads": 1, "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
            "training_record_policy": recipe["training"].get("data_policy"), "data_policy_application": "caller supplies eligible TRAIN only"}
        if any(details.get(key) != value for key, value in expected.items()) or details.get("dtype") not in {"float32", "float64"}:
            raise ValueError("Distance training metadata changed")
        hashes = {"training_input_sha256": model.training_input_sha256_, "training_labels_sha256": _array_digest(labels),
            "training_groups_sha256": _json_digest(people.tolist()), "training_records_sha256": _json_digest(records.tolist())}
        for key, value in hashes.items():
            if (not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                    or getattr(model, key + "_") != value or details.get(key) != value):
                raise ValueError("Distance TRAIN identity digest changed")
        if recipe["family"] == VOTING:
            if len(model.members_) != len(recipe["members"]):
                raise ValueError("Distance voting member count changed")
            for member, member_recipe in zip(model.members_, recipe["members"]):
                verify_fitted_distance_identity(member, member_recipe, seed)
                if any(member.training_details_[key] != details[key] for key in (*hashes, "record_metadata", "dtype")):
                    raise ValueError("Distance voting members did not share TRAIN rows")
            if details["members"] != [member.training_details_ for member in model.members_] or details["combination"] != "equal aligned native probability mean":
                raise ValueError("Distance voting audit changed")
        else:
            scaler, ref = model.scaler_, model.reference_
            if (type(scaler) is not StandardScaler or scaler.get_params(deep=False) != StandardScaler().get_params(deep=False)
                    or scaler.n_samples_seen_ != rows or scaler.n_features_in_ != 596
                    or details["scaler_state_sha256"] != _json_digest(_scaler_state(scaler))
                    or ref.shape != (rows, 596) or ref.dtype != np.dtype(np.float64) or not ref.flags.c_contiguous or not np.isfinite(ref).all()
                    or details["reference_sha256"] != _array_digest(ref) or details["reference_rows"] != rows
                    or details["reference_dtype"] != "float64" or details["reference_order"] != "C"
                    or details["gram_shape"] != [rows, rows] or details["gram_bytes"] != 8*rows*rows):
                raise ValueError("Distance fitted scaler or TRAIN reference changed")
            for key in ("standardized_input_sha256", "gram_sha256"):
                h = details[key]
                if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                    raise ValueError("Distance numerical input digest is invalid")
            kind = recipe["model_params"]["kernel"]
            verify_distance_kernel_rule(model.kernel_rule_, kind)
            if (details["kernel_rule"] != model.kernel_rule_ or details["native_svc_state_sha256"] !=
                    _svc_state(model.estimator_, recipe["model_params"]["C"], seed, labels)):
                raise ValueError("Distance kernel rule or native SVC state changed")
        if details["fit_state_sha256"] != model.fit_state_sha256_ or _fit_state(model) != model.fit_state_sha256_:
            raise ValueError("Distance complete fitted state changed")
    except (AttributeError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Incomplete distance fitted state") from error
    return True


def build_distance_pipeline(spec, seed=None):
    if seed is not None:
        _seed(seed)
    return DistancePipeline(validate_distance_pipeline(spec), seed=seed)


def fit_distance_pipeline(model, spec, X, y, *, groups=None, record_ids=None, validation_data=None):
    recipe = validate_distance_pipeline(spec)
    if type(model) is not DistancePipeline or validate_distance_pipeline(model.spec) != recipe:
        raise ValueError("Built distance model differs from the supplied recipe")
    return model.fit(X, y, groups=groups, record_ids=record_ids, validation_data=validation_data)
