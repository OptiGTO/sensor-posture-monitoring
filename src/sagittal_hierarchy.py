"""A train-only soft hierarchy over the existing seven posture classes.

Input is the existing 596-dimensional feature vector, already scaled by its
outer training pipeline. The fixed gate combines labels 0/1/2 into one class;
the specialist estimates their conditional probabilities. No validation data,
new features, calibration layer, or saved candidate predictions are consumed.
"""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


INPUT_FEATURES = 596
GATE_CLASSES = (0, 3, 4, 5, 6)
SPECIALIST_CLASSES = (0, 1, 2)
DEFAULT_PARAMS = {
    "extra_trees": {"n_estimators": 200, "min_samples_leaf": 8,
                    "max_depth": 8, "max_features": 0.5},
    "rbf_svm": {"C": 1.0, "gamma": 0.0005},
}
CONTROLLED_PARAMS = frozenset({"random_state", "class_weight", "sample_weight", "n_jobs",
                              "probability", "kernel", "decision_function_shape"})


def _aligned_probability(estimator, values, expected_classes):
    """Align actual estimator columns, rejecting absent/duplicate/wrong labels."""
    classes = np.asarray(estimator.classes_)
    if (classes.ndim != 1 or classes.dtype.kind not in "iu"
            or not np.array_equal(np.sort(classes), np.asarray(expected_classes))):
        raise ValueError("Hierarchy head class labels do not match its fixed contract")
    raw = np.asarray(estimator.predict_proba(values), dtype=np.float64)
    if (raw.shape != (len(values), len(classes)) or not np.isfinite(raw).all()
            or np.any(raw < 0) or np.any(raw > 1 + 1e-9)
            or not np.allclose(raw.sum(axis=1), 1, rtol=1e-7, atol=1e-9)):
        raise ValueError("Hierarchy head returned invalid probability rows")
    aligned = np.zeros((len(values), 7), dtype=np.float64)
    aligned[:, classes] = raw / raw.sum(axis=1, keepdims=True)
    return aligned


def _weight_summary(labels, weights):
    return {"samples": len(labels), "minimum": float(weights.min()),
            "maximum": float(weights.max()), "mean": float(weights.mean()),
            "class_counts": {str(label): int((labels == label).sum()) for label in np.unique(labels)},
            "class_weight_mass": {str(label): float(weights[labels == label].sum())
                                  for label in np.unique(labels)}}


class SagittalHierarchyClassifier(ClassifierMixin, BaseEstimator):
    """Five-way gate plus three-way specialist, with soft probability routing.

    Both heads use the same family and model parameters. ``random_state`` seeds
    the gate, and ``(random_state + 1) % 2**32`` seeds the specialist. Internal
    SVC probability fitting uses only supplied training rows; its native folds
    are not subject-disjoint calibration folds.
    """

    def __init__(self, family="extra_trees", model_params=None, random_state=20260921):
        self.family = family
        self.model_params = model_params
        self.random_state = random_state

    def _configuration(self):
        if self.family not in DEFAULT_PARAMS:
            raise ValueError("Hierarchy family must be extra_trees or rbf_svm")
        if type(self.random_state) is not int or not 0 <= self.random_state < 2**32:
            raise ValueError("Hierarchy random_state must be a uint32 integer")
        if self.model_params is not None and not isinstance(self.model_params, dict):
            raise ValueError("Hierarchy model_params must be a dictionary or null")
        params = deepcopy(DEFAULT_PARAMS[self.family] if self.model_params is None else self.model_params)
        if set(params) & CONTROLLED_PARAMS:
            raise ValueError("Hierarchy seeds, weighting, threads and probability settings are controlled")
        try:
            json.dumps(params, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Hierarchy model_params must contain finite JSON values") from error
        return params

    def _estimator(self, params, seed):
        if self.family == "extra_trees":
            return ExtraTreesClassifier(**params, class_weight=None, random_state=seed, n_jobs=1)
        return SVC(**params, class_weight=None, random_state=seed, kernel="rbf",
                   probability=True, decision_function_shape="ovr")

    def fit(self, X, y):
        params = self._configuration()
        values, labels = check_X_y(X, y, dtype=np.float64)
        if values.shape[1] != INPUT_FEATURES:
            raise ValueError("Hierarchy expects exactly 596 already-scaled features")
        if labels.dtype.kind not in "iu" or not np.array_equal(np.unique(labels), np.arange(7)):
            raise ValueError("Hierarchy requires all seven original integer class labels 0..6")
        gate_labels = np.where(labels <= 2, 0, labels)
        specialist_mask = labels <= 2
        specialist_labels = labels[specialist_mask]
        # Balance ORIGINAL leaves, then merge 0/1/2. Balancing the five coarse
        # labels instead would silently reduce the triplet's prior mass to 1/5.
        gate_weights = compute_sample_weight("balanced", labels)
        specialist_weights = compute_sample_weight("balanced", specialist_labels)
        gate_seed = self.random_state
        specialist_seed = (gate_seed + 1) % 2**32
        gate = self._estimator(params, gate_seed)
        specialist = self._estimator(params, specialist_seed)
        gate.fit(values, gate_labels, sample_weight=gate_weights)
        specialist.fit(values[specialist_mask], specialist_labels, sample_weight=specialist_weights)
        if (not np.array_equal(np.sort(gate.classes_), GATE_CLASSES)
                or not np.array_equal(np.sort(specialist.classes_), SPECIALIST_CLASSES)):
            raise ValueError("Fitted hierarchy head lost a required class")
        self.gate_, self.specialist_ = gate, specialist
        self.classes_ = np.arange(7)
        self.n_features_in_ = values.shape[1]
        self.training_details_ = {
            "architecture": "sagittal_soft_hierarchy_v1", "base_family": self.family,
            "base_params": params, "input_features": INPUT_FEATURES,
            "preprocessing": "Already scaled by outer training-only pipeline; no internal preprocessing",
            "effective_seed": gate_seed, "seed_rule": "gate=s; specialist=(s+1)%2**32",
            "gate_class_mapping": {str(i): (0 if i <= 2 else i) for i in range(7)},
            "original_training": _weight_summary(labels, gate_weights),
            "gate": {"seed": gate_seed, "classes": gate.classes_.astype(int).tolist(),
                     "weighting": "sample_weight from original seven-class balanced labels; class_weight=None",
                     "weight_summary": _weight_summary(gate_labels, gate_weights),
                     "estimator_params": gate.get_params(deep=False)},
            "specialist": {"seed": specialist_seed, "classes": specialist.classes_.astype(int).tolist(),
                           "weighting": "balanced sample_weight on training labels 0/1/2 only; class_weight=None",
                           "weight_summary": _weight_summary(specialist_labels, specialist_weights),
                           "estimator_params": specialist.get_params(deep=False)},
            "probability_rule": "p[0:3]=gate[0]*specialist[0:3]; p[3:7]=gate[3:7]; row normalization",
            "routing": "soft; both heads predict every input; no gate argmax or threshold",
            "validation_data_used": False,
            "probability_calibration": ("SVC native five-fold fitting on supplied training rows only; not subject-disjoint"
                                        if self.family == "rbf_svm" else "Uncalibrated tree class probabilities"),
        }
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["gate_", "specialist_", "classes_", "n_features_in_"])
        values = check_array(X, dtype=np.float64)
        if values.shape[1] != INPUT_FEATURES:
            raise ValueError("Hierarchy expects exactly 596 already-scaled features")
        gate = _aligned_probability(self.gate_, values, GATE_CLASSES)
        specialist = _aligned_probability(self.specialist_, values, SPECIALIST_CLASSES)
        result = gate.copy()
        result[:, :3] = gate[:, [0]] * specialist[:, :3]
        totals = result.sum(axis=1, keepdims=True)
        if not np.isfinite(result).all() or np.any(result < 0) or np.any(totals <= 0):
            raise ValueError("Hierarchy produced invalid combined probabilities")
        return result / totals

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
