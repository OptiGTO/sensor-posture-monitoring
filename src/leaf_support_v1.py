"""Fixed post-fit leaf-support shrinkage for seven-class Extra Trees.

The support is an unweighted count of training windows reaching a leaf. Windows
can overlap and share a recording/person; this is not independent sample size
and the adjusted probabilities are not a calibrated Bayesian posterior.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Real

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.utils.validation import check_is_fitted


VERSION = "leaf_support_v1"
FEATURE_COUNT = 596
CLASS_COUNT = 7


def _alpha(alpha):
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or not np.isfinite(alpha) or alpha not in (0, 8, 32)):
        raise ValueError("alpha must be a finite numeric value exactly 0, 8 or 32")
    return float(alpha)


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _array_digest(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(json.dumps([value.dtype.str, list(value.shape)]).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _class_indices(classes, *, canonical=False):
    values = np.asarray(classes)
    if (values.shape != (CLASS_COUNT,) or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
            or not np.array_equal(np.sort(values), np.arange(CLASS_COUNT))
            or canonical and not np.array_equal(values, np.arange(CLASS_COUNT))):
        raise ValueError("Leaf-support classes must be integral numeric posture IDs 0..6")
    return values.astype(np.int64)


def _forest(estimator):
    if type(estimator) is not ExtraTreesClassifier:
        raise ValueError("Leaf support requires the exact sklearn ExtraTreesClassifier")
    check_is_fitted(estimator, ("estimators_", "classes_", "n_features_in_", "n_outputs_"))
    if (estimator.n_features_in_ != FEATURE_COUNT or estimator.n_outputs_ != 1
            or not isinstance(estimator.estimators_, list) or not estimator.estimators_
            or len(estimator.estimators_) != estimator.n_estimators):
        raise ValueError("Leaf-support forest must have a fitted single-output 596-feature schema")
    _class_indices(estimator.classes_, canonical=True)
    for tree in estimator.estimators_:
        if (type(tree) is not ExtraTreeClassifier or getattr(tree, "n_features_in_", None) != FEATURE_COUNT
                or getattr(tree, "n_outputs_", None) != 1 or not hasattr(tree, "tree_")):
            raise ValueError("Leaf-support tree schema differs from its fitted forest")
        _class_indices(tree.classes_)
        if tree.tree_.n_features != FEATURE_COUNT or tree.tree_.n_outputs != 1:
            raise ValueError("Leaf-support learned tree dimensions differ")
    return estimator


def _features(X):
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] != FEATURE_COUNT or not len(values)
            or values.dtype != np.dtype(np.float32) or not np.isfinite(values).all()):
        raise ValueError("Leaf support requires nonempty finite float32 standardized features (N, 596)")
    return values


def _check_probability(probability, rows):
    if (not isinstance(probability, np.ndarray) or probability.dtype != np.dtype(np.float64)
            or probability.shape != (rows, CLASS_COUNT) or not np.isfinite(probability).all()
            or np.any(probability < 0) or not np.allclose(probability.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError("Leaf-support probabilities must be finite normalized float64 (N, 7)")


def leaf_support_rule(alpha):
    """Return a fresh JSON rule with normalized alpha and an exact rule digest."""
    alpha = _alpha(alpha)
    rule = {
        "schema_version": 1, "version": VERSION, "alpha": alpha,
        "input_shape": [None, FEATURE_COUNT], "input_dtype": "float32",
        "classes": list(range(CLASS_COUNT)), "target_probability": [1 / CLASS_COUNT] * CLASS_COUNT,
        "support_source": "tree.tree_.n_node_samples[tree.apply(X)]: unweighted training-window count",
        "tree_probability": "Native normalized tree.predict_proba, already reflecting training class/sample weights; align integral numeric tree.classes_ to 0..6",
        "formula": "q_t=(n_t*p_t+alpha/7)/(n_t+alpha); output=sum(q_t in estimator.estimators_ order)/number_of_trees",
        "alpha_zero": "Return the original forest.predict_proba result unchanged after schema/output validation; no probability adjustment",
        "tree_probability_divided_by_support": False,
        "training_or_tree_state_modified": False,
        "evaluation_record_or_subject_normalization": False,
        "evaluation_batch_adaptation": False,
        "support_is_independent_sample_size": False,
        "calibrated_posterior": False,
        "limitation": "Training windows may overlap and share records/people; this fixed shrinkage is not a calibrated posterior or an independent-sample uncertainty estimate",
        "json_hash_encoding": "UTF-8 json.dumps(ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)",
    }
    rule["rule_sha256"] = _json_digest(rule)
    return rule


def leaf_support_probabilities(estimator, X, alpha):
    """Predict each window independently, with no learned post-processing."""
    alpha = _alpha(alpha)
    estimator, values = _forest(estimator), _features(X)
    if alpha == 0:
        probability = estimator.predict_proba(values)
        _check_probability(probability, len(values))
        return probability
    probability = np.zeros((len(values), CLASS_COUNT), dtype=np.float64)
    for tree in estimator.estimators_:
        native = tree.predict_proba(values, check_input=False)
        _check_probability(native, len(values))
        aligned = np.empty_like(native)
        aligned[:, _class_indices(tree.classes_)] = native
        leaves = tree.apply(values, check_input=False)
        support = tree.tree_.n_node_samples[leaves]
        if support.shape != (len(values),) or support.dtype.kind not in "iu" or np.any(support < 1):
            raise ValueError("Reached leaves must have positive unweighted training-window counts")
        n = support.astype(np.float64)[:, None]
        probability += (n * aligned + alpha / CLASS_COUNT) / (n + alpha)
    probability /= len(estimator.estimators_)
    _check_probability(probability, len(values))
    return probability


def forest_state_sha256(estimator):
    """Hash deterministic fitted state, excluding structured node padding bytes."""
    estimator = _forest(estimator)
    trees = []
    for tree in estimator.estimators_:
        state = tree.tree_.__getstate__()
        trees.append({
            "seed": int(tree.random_state) if tree.random_state is not None else None,
            "classes": _array_digest(tree.classes_), "n_classes": int(tree.n_classes_),
            "n_features": int(tree.n_features_in_), "n_outputs": int(tree.n_outputs_),
            "node_count": int(state["node_count"]), "max_depth": int(state["max_depth"]),
            "values": _array_digest(state["values"]),
            "nodes": {name: _array_digest(state["nodes"][name]) for name in state["nodes"].dtype.names},
        })
    return _json_digest({"version": VERSION, "classes": _array_digest(estimator.classes_),
        "n_features": int(estimator.n_features_in_), "n_outputs": int(estimator.n_outputs_),
        "tree_count": len(trees), "trees": trees})
