"""Seven independent train-only binary ExtraTrees heads on standardized features.

The caller owns the shared TRAIN StandardScaler. Each head sees all the same
float32 rows, with its own one-vs-rest target and balanced binary class weights.
Row normalization is a fixed score rule, not posterior calibration.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits


VERSION = "binary_forest_v1"
FEATURE_COUNT = 596
CLASS_COUNT = 7
DEFAULT_PARAMS = {"n_estimators": 200, "min_samples_leaf": 8,
                  "max_depth": 8, "max_features": .5}


def _parameters(params, seed):
    if not isinstance(params, dict) or set(params) - set(DEFAULT_PARAMS):
        raise ValueError("Unknown binary forest parameters")
    value = {**DEFAULT_PARAMS, **deepcopy(params)}
    for key in ("n_estimators", "min_samples_leaf"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if value["max_depth"] is not None and (type(value["max_depth"]) is not int or value["max_depth"] < 1):
        raise ValueError("max_depth must be a positive integer or None")
    features = value["max_features"]
    if not ((type(features) is int and 1 <= features <= FEATURE_COUNT)
            or (type(features) is float and np.isfinite(features) and 0 < features <= 1)):
        raise ValueError("max_features must be a fraction in (0, 1] or an integer in [1, 596]")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("random_state must be a uint32 integer")
    return {**value, "random_state": seed}


def _model_parameters(model):
    params = model.get_params(deep=False)
    seed = params.pop("random_state")
    return _parameters(params, seed)


def _features(X):
    value = np.asarray(X)
    if (value.ndim != 2 or value.shape[1] != FEATURE_COUNT or not len(value)
            or value.dtype != np.dtype(np.float32) or not np.isfinite(value).all()):
        raise ValueError("Binary forest requires nonempty finite float32 standardized features (N, 596)")
    return np.array(value, order="C", copy=True)


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _array_digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([value.dtype.str, list(value.shape)]).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def binary_forest_rule():
    """Return fresh JSON metadata for the fixed binary learning/score contract."""
    rule = {"schema_version": 1, "version": VERSION,
        "input_shape": [None, FEATURE_COUNT], "input_dtype": "float32",
        "classes": list(range(CLASS_COUNT)), "binary_classes": [0, 1],
        "target": "head c: (original_label == c).astype(int64); all supplied TRAIN rows in original order",
        "weighting": "per-head class_weight='balanced': N/(2*n_binary_class); no other sample weights",
        "seed_rule": "head c seed=(primary_seed+c)%2**32, c=0..6",
        "preprocessing": "shared TRAIN-only StandardScaler belongs to the outer pipeline; no internal scaling",
        "positive_column": "locate label 1 in each fitted head.classes_; preserve outer class order 0..6",
        "normalization": "float64 positive scores divided by their row sum when positive",
        "zero_sum_policy": "exact all-zero positive-score row becomes uniform probability 1/7",
        "prediction": "argmax in fixed class order 0..6; ties choose the first class",
        "calibrated_posterior": False, "external_validation_used": False,
        "evaluation_batch_adaptation": False, "mutable_inference_counters": False,
        "n_jobs_per_head": 1, "head_execution": "sequential in class order",
        "state_hash": "hash structured tree node fields individually; never hash node padding bytes"}
    rule["rule_sha256"] = _json_digest(rule)
    return rule


def _backend(params, class_id):
    return ExtraTreesClassifier(**{**params, "random_state": (params["random_state"] + class_id) % 2**32},
        criterion="gini", min_samples_split=2, min_weight_fraction_leaf=0., max_leaf_nodes=None,
        min_impurity_decrease=0., bootstrap=False, oob_score=False, n_jobs=1,
        verbose=0, warm_start=False, class_weight="balanced", ccp_alpha=0.,
        max_samples=None, monotonic_cst=None)


def _binary_classes(classes):
    values = np.asarray(classes)
    if (values.shape != (2,) or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all() or not np.array_equal(np.sort(values), [0, 1])):
        raise ValueError("Binary head classes must be distinct numeric labels 0 and 1")
    return values.astype(np.int64)


def _head_schema(head, params, class_id):
    if (type(head) is not ExtraTreesClassifier
            or head.get_params(deep=False) != _backend(params, class_id).get_params(deep=False)
            or getattr(head, "n_features_in_", None) != FEATURE_COUNT
            or getattr(head, "n_outputs_", None) != 1 or getattr(head, "n_classes_", None) != 2
            or not isinstance(getattr(head, "estimators_", None), list)
            or len(head.estimators_) != params["n_estimators"]):
        raise ValueError("Binary forest head parameters, seed or fitted dimensions changed")
    return _binary_classes(head.classes_)


def _tree_parameters(params, seed):
    return ExtraTreeClassifier(criterion="gini", splitter="random", max_depth=params["max_depth"],
        min_samples_split=2, min_samples_leaf=params["min_samples_leaf"], min_weight_fraction_leaf=0.,
        max_features=params["max_features"], random_state=int(seed), max_leaf_nodes=None,
        min_impurity_decrease=0., class_weight=None, ccp_alpha=0., monotonic_cst=None).get_params(deep=False)


def _head_state(head, params, class_id, training_rows):
    _head_schema(head, params, class_id)
    seeds = np.random.RandomState((params["random_state"] + class_id) % 2**32).randint(
        np.iinfo(np.int32).max, size=params["n_estimators"])
    trees = []
    max_features = params["max_features"]
    expected_features = max_features if type(max_features) is int else max(1, int(max_features * FEATURE_COUNT))
    for tree, seed in zip(head.estimators_, seeds):
        if (type(tree) is not ExtraTreeClassifier
                or tree.get_params(deep=False) != _tree_parameters(params, seed)
                or getattr(tree, "n_features_in_", None) != FEATURE_COUNT
                or getattr(tree, "n_outputs_", None) != 1 or getattr(tree, "n_classes_", None) != 2
                or getattr(tree, "max_features_", None) != expected_features or not hasattr(tree, "tree_")):
            raise ValueError("Binary forest tree parameters, seed or fitted dimensions changed")
        _binary_classes(tree.classes_)
        learned = tree.tree_
        if (learned.n_features != FEATURE_COUNT or learned.n_outputs != 1
                or not np.array_equal(learned.n_classes, [2])
                or learned.n_node_samples[0] != training_rows
                or not np.isclose(learned.weighted_n_node_samples[0], training_rows, rtol=1e-10, atol=1e-10)):
            raise ValueError("Binary forest tree TRAIN row scope changed")
        state = learned.__getstate__()
        if set(state) != {"max_depth", "node_count", "nodes", "values"} or not state["nodes"].dtype.names:
            raise ValueError("Unsupported fitted tree state schema")
        trees.append({"parameters": tree.get_params(deep=False),
            "classes": _array_digest(tree.classes_), "n_classes": int(tree.n_classes_),
            "n_features": int(tree.n_features_in_), "n_outputs": int(tree.n_outputs_),
            "max_features": int(tree.max_features_), "node_count": int(state["node_count"]),
            "max_depth": int(state["max_depth"]), "values": _array_digest(state["values"]),
            "nodes": {name: _array_digest(state["nodes"][name]) for name in state["nodes"].dtype.names}})
    return _json_digest({"parameters": head.get_params(deep=False), "classes": _array_digest(head.classes_),
        "n_classes": int(head.n_classes_), "n_features": int(head.n_features_in_),
        "n_outputs": int(head.n_outputs_), "tree_count": len(trees), "trees": trees})


def _audit(model):
    params, counts = model.fitted_params_, model.class_counts_
    n = int(model.n_samples_seen_)
    heads = []
    for class_id, head in enumerate(model.estimators_):
        positive, negative = int(counts[class_id]), n - int(counts[class_id])
        heads.append({"class_id": class_id, "seed": (params["random_state"] + class_id) % 2**32,
            "training_rows": n, "positive_rows": positive, "negative_rows": negative,
            "binary_class_counts": [negative, positive], "binary_class_weights": [n / (2 * negative), n / (2 * positive)],
            "binary_weight_mass": [n / 2, n / 2], "parameters": head.get_params(deep=False),
            "binary_labels_sha256": model.binary_labels_sha256_[class_id],
            "state_sha256": _head_state(head, params, class_id, n)})
    return {"algorithm": VERSION, "parameters": deepcopy(params), "rule": binary_forest_rule(),
        "effective_seed": params["random_state"], "effective_seeds": model.effective_seeds_.copy(),
        "class_order": model.classes_.tolist(), "class_counts": counts.tolist(), "training_rows": n,
        "input_features": FEATURE_COUNT, "input_dtype": "float32", "backend_dtype": "float32",
        "training_input_sha256": model.training_input_sha256_, "training_labels_sha256": model.training_labels_sha256_,
        "external_validation_used": False, "threads": 1,
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "heads": heads}


class BinaryForestV1(ClassifierMixin, BaseEstimator):
    """Seven balanced one-vs-rest forests; n_estimators counts trees per head."""

    def __init__(self, n_estimators=200, min_samples_leaf=8, max_depth=8, max_features=.5, random_state=20260921):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        params, values, labels = _model_parameters(self), _features(X), np.asarray(y)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu"
                or not np.array_equal(np.unique(labels), np.arange(CLASS_COUNT))):
            raise ValueError("Binary forest requires aligned integer labels containing all seven classes 0..6")
        labels = labels.astype(np.int64, copy=True)
        estimators, binary_hashes = [], []
        with threadpool_limits(limits=1):
            for class_id in range(CLASS_COUNT):
                binary = (labels == class_id).astype(np.int64)
                estimator = _backend(params, class_id).fit(values, binary)
                _head_schema(estimator, params, class_id)
                estimators.append(estimator)
                binary_hashes.append(_array_digest(binary))
        self.estimators_, self.fitted_params_ = estimators, params
        self.classes_, self.class_counts_ = np.arange(CLASS_COUNT), np.bincount(labels, minlength=CLASS_COUNT)
        self.n_features_in_, self.n_samples_seen_ = FEATURE_COUNT, len(values)
        self.effective_seeds_ = [(params["random_state"] + c) % 2**32 for c in range(CLASS_COUNT)]
        self.training_input_sha256_, self.training_labels_sha256_ = _array_digest(values), _array_digest(labels)
        self.binary_labels_sha256_ = binary_hashes
        self.training_details_ = _audit(self)
        self.fitted_state_sha256_ = binary_forest_state_sha256(self)
        return self

    def _positive_scores(self, X):
        check_is_fitted(self, ("fitted_params_", "estimators_", "classes_"))
        if _model_parameters(self) != self.fitted_params_:
            raise ValueError("Binary forest parameters changed after fitting; refit before prediction")
        if (not np.array_equal(self.classes_, np.arange(CLASS_COUNT))
                or len(self.estimators_) != CLASS_COUNT or self.n_features_in_ != FEATURE_COUNT):
            raise ValueError("Binary forest fitted class or input schema changed")
        values, scores = _features(X), []
        with threadpool_limits(limits=1):
            for class_id, head in enumerate(self.estimators_):
                classes = _head_schema(head, self.fitted_params_, class_id)
                probability = np.asarray(head.predict_proba(values), dtype=np.float64)
                if (probability.shape != (len(values), 2) or not np.isfinite(probability).all()
                        or (probability < 0).any() or (probability > 1 + 1e-12).any()
                        or not np.allclose(probability.sum(axis=1), 1., atol=1e-12, rtol=0)):
                    raise ValueError("Binary head returned invalid two-column probabilities")
                scores.append(probability[:, int(np.flatnonzero(classes == 1)[0])])
        return np.column_stack(scores)

    def predict_proba(self, X):
        scores = self._positive_scores(X)
        totals = scores.sum(axis=1, keepdims=True)
        probability = np.full_like(scores, 1 / CLASS_COUNT)
        np.divide(scores, totals, out=probability, where=totals > 0)
        return probability

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def probability_diagnostics(self, X):
        """Report fixed-rule zero rows for this call without retaining any state."""
        scores = self._positive_scores(X)
        return {"rows": len(scores), "all_zero_rows": int(np.count_nonzero(scores.sum(axis=1) == 0)),
            "zero_sum_policy": binary_forest_rule()["zero_sum_policy"]}


def binary_forest_state_sha256(model):
    """Hash actual learned forest state, without circular saved-hash references."""
    if type(model) is not BinaryForestV1:
        raise ValueError("Expected exact BinaryForestV1 estimator")
    check_is_fitted(model, ("fitted_params_", "estimators_", "n_samples_seen_"))
    if len(model.estimators_) != CLASS_COUNT:
        raise ValueError("Binary forest must retain seven fitted heads")
    return _json_digest({"version": VERSION, "parameters": _model_parameters(model),
        "fitted_parameters": model.fitted_params_, "classes": _array_digest(model.classes_),
        "class_counts": _array_digest(model.class_counts_), "n_features": model.n_features_in_,
        "training_rows": model.n_samples_seen_, "heads": [_head_state(head, model.fitted_params_, c,
            model.n_samples_seen_) for c, head in enumerate(model.estimators_)]})


def verify_binary_forest(model, expected_params, seed):
    """Validate recipe, class/row audit, deterministic seeds and every tree field."""
    params = _parameters(expected_params, seed)
    try:
        if (type(model) is not BinaryForestV1 or _model_parameters(model) != params
                or model.fitted_params_ != params or model.n_features_in_ != FEATURE_COUNT
                or not np.array_equal(model.classes_, np.arange(CLASS_COUNT))
                or model.class_counts_.shape != (CLASS_COUNT,) or model.class_counts_.dtype.kind not in "iu"
                or (model.class_counts_ < 1).any() or model.n_samples_seen_ != int(model.class_counts_.sum())
                or not isinstance(model.estimators_, list) or len(model.estimators_) != CLASS_COUNT
                or model.effective_seeds_ != [(seed + c) % 2**32 for c in range(CLASS_COUNT)]
                or len(model.binary_labels_sha256_) != CLASS_COUNT):
            raise ValueError("Binary forest recipe, class counts, seeds or row scope changed")
        hashes = [model.training_input_sha256_, model.training_labels_sha256_, *model.binary_labels_sha256_]
        if any(not isinstance(item, str) or len(item) != 64 or any(c not in "0123456789abcdef" for c in item) for item in hashes):
            raise ValueError("Binary forest training scope digests are invalid")
        if model.training_details_ != _audit(model):
            raise ValueError("Binary forest training audit or learned head state changed")
        if model.fitted_state_sha256_ != binary_forest_state_sha256(model):
            raise ValueError("Binary forest learned state changed")
    except (AttributeError, TypeError, IndexError, KeyError) as error:
        raise ValueError("Incomplete or invalid binary forest fitted state") from error
    return True
