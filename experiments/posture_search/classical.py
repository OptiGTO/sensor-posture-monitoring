"""Isolated, reproducible classifiers for the controlled posture experiments.

Every preprocessing estimator is fitted inside ``fit`` on its supplied training
rows. Group-aware estimators require subject IDs explicitly. No estimator reads
the dataset, validation labels, or test labels. Probabilities always have seven
columns in posture-ID order, even when a bootstrap omits a class.

SVC's probability=True uses libsvm's internal, non-grouped probability calibration
on training data only. This does not cross the outer train/validation/test split,
but its calibration is not subject-independent. The limitation is recorded in
each applicable configuration. ``predict`` preserves SVC's native decision rule;
it may differ from the argmax of its calibrated probabilities.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


MODEL_VERSION = "posture-search-classical-v1"
DEFAULT_BASELINE_PARAMS = {"C": 1.0, "gamma": 0.0005}
KINDS = (
    "baseline", "robust_svm", "feature_select", "pca_svm", "extra_trees",
    "random_forest", "histogram_boosting", "shrinkage_lda", "knn", "logistic",
    "hierarchical_correct", "hierarchical_direction", "subject_balanced_svm",
    "group_bagging", "stacking", "soft_vote", "weighted_vote", "hard_vote",
)
ALIASES = {"svm": "baseline", "hist_boost": "histogram_boosting", "hist": "histogram_boosting",
           "lda": "shrinkage_lda", "knn15": "knn", "logistic_regression": "logistic"}
SVC_CALIBRATION = "libsvm_internal_5fold_training_rows_only_not_subject_grouped"


def _aligned_probability(estimator, X, n_classes: int = 7) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    aligned = np.zeros((len(X), n_classes), dtype=np.float64)
    aligned[:, np.asarray(estimator.classes_, dtype=int)] = raw
    return aligned


def _json_value(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return getattr(value, "__name__", str(value))


def _estimator_config(estimator) -> dict:
    if isinstance(estimator, Pipeline):
        return {"class": "Pipeline", "steps": [
            {"name": name, **_estimator_config(step)} for name, step in estimator.steps
        ]}
    return {"class": type(estimator).__name__,
            "parameters": _json_value(estimator.get_params(deep=False))}


class ClassicalClassifier(ClassifierMixin, BaseEstimator):
    """Common interface for single, hierarchical, and grouped ensemble models."""

    def __init__(self, kind: str, seed: int = 42, baseline_params: dict | None = None,
                 ensemble_members: list[str] | tuple[str, ...] | None = None):
        self.kind = kind
        self.seed = seed
        self.baseline_params = baseline_params
        self.ensemble_members = ensemble_members

    def __setstate__(self, state):
        # Previously saved study checkpoints predate the optional ablation argument.
        super().__setstate__(state)
        self.__dict__.setdefault("ensemble_members", None)

    def _svm(self):
        return SVC(**self.svm_params_, kernel="rbf", probability=True,
                   class_weight="balanced", random_state=self.seed, cache_size=256)

    def _single(self, kind: str, n_features: int, n_rows: int):
        if kind in {"baseline", "subject_balanced_svm"}:
            return Pipeline([("scale", StandardScaler()), ("model", self._svm())])
        if kind == "robust_svm":
            return Pipeline([("scale", RobustScaler(quantile_range=(25.0, 75.0))),
                             ("model", self._svm())])
        if kind == "feature_select":
            return Pipeline([("select", SelectKBest(f_classif, k=min(96, n_features))),
                             ("scale", StandardScaler()), ("model", self._svm())])
        if kind == "pca_svm":
            return Pipeline([("scale", StandardScaler()),
                             ("pca", PCA(n_components=.95, svd_solver="full")),
                             ("model", self._svm())])
        if kind == "extra_trees":
            return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                                        max_features="sqrt", class_weight="balanced",
                                        random_state=self.seed, n_jobs=1)
        if kind == "random_forest":
            return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                          max_features="sqrt", class_weight="balanced_subsample",
                                          random_state=self.seed, n_jobs=1)
        if kind == "histogram_boosting":
            # Fixed iteration count avoids an internal row-random validation set.
            return HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                                                   learning_rate=.06, l2_regularization=1.,
                                                   class_weight="balanced", early_stopping=False,
                                                   random_state=self.seed)
        if kind == "shrinkage_lda":
            return Pipeline([("scale", StandardScaler()),
                             ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])
        if kind == "knn":
            return Pipeline([("scale", StandardScaler()),
                             ("model", KNeighborsClassifier(n_neighbors=min(15, n_rows),
                                                            weights="distance", n_jobs=1))])
        if kind == "logistic":
            return Pipeline([("scale", StandardScaler()),
                             ("model", LogisticRegression(C=1.0, max_iter=2500,
                                                           class_weight="balanced", random_state=self.seed))])
        raise ValueError(f"Not a single-classifier kind: {kind}")

    def _fit_branch(self, X, y):
        if len(np.unique(y)) == 1:
            return DummyClassifier(strategy="most_frequent").fit(X, y)
        return self._single("baseline", X.shape[1], len(X)).fit(X, y)

    def _child(self, kind, seed=None):
        return ClassicalClassifier(kind, self.seed if seed is None else seed, self.svm_params_)

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or len(y) != len(X) or len(X) == 0:
            raise ValueError("Expected non-empty 2D features and one label per row")
        if not np.isfinite(X).all() or not np.isin(y, np.arange(7)).all():
            raise ValueError("Features must be finite and posture IDs must be in 0..6")
        kind = ALIASES.get(self.kind, self.kind)
        if kind not in KINDS:
            raise ValueError(f"Unknown classifier {self.kind!r}; choose from {KINDS}")
        if groups is not None:
            groups = np.asarray(groups)
            if len(groups) != len(y):
                raise ValueError("Expected one group ID per training row")
        if kind in {"subject_balanced_svm", "group_bagging", "stacking"} and groups is None:
            raise ValueError(f"{kind} requires explicit subject grouping")
        # A repeated fit (including after set_params) must not retain an old route.
        for attribute in list(vars(self)):
            if attribute.endswith("_"):
                delattr(self, attribute)
        supplied = self.baseline_params or DEFAULT_BASELINE_PARAMS
        self.svm_params_ = {key: supplied.get(key, DEFAULT_BASELINE_PARAMS[key]) for key in ("C", "gamma")}
        self.kind_ = kind
        self.classes_ = np.arange(7)
        self.n_features_in_ = X.shape[1]
        self.config_ = {"model_version": MODEL_VERSION, "kind": kind, "seed": int(self.seed),
                        "training_rows": len(X), "input_features": X.shape[1],
                        "baseline_svm_parameters": self.svm_params_,
                        "preprocessing_fit_scope": "training_rows_only",
                        "svc_probability_calibration": SVC_CALIBRATION,
                        "predict_rule": "native_estimator_predict_for_single_models_probability_argmax_for_composites"}
        if groups is not None:
            self.config_["training_groups"] = _json_value(np.unique(groups))
        if kind in {"hierarchical_correct", "hierarchical_direction"}:
            self.branches_ = [[0], [1, 2, 3, 4, 5, 6]] if kind == "hierarchical_correct" else [[0, 1, 2], [3, 4], [5, 6]]
            routing = {label: index for index, branch in enumerate(self.branches_) for label in branch}
            group_y = np.asarray([routing[int(label)] for label in y])
            self.gate_ = self._fit_branch(X, group_y)
            self.branch_models_ = []
            for branch in self.branches_:
                mask = np.isin(y, branch)
                self.branch_models_.append(self._fit_branch(X[mask], y[mask]) if mask.any() else None)
            self.config_.update({"branches": self.branches_, "routing": "soft_conditional_probability_product",
                                 "gate": _estimator_config(self.gate_),
                                 "branch_models": [_estimator_config(m) if m is not None else None for m in self.branch_models_]})
        elif kind == "group_bagging":
            unique = np.unique(groups)
            rng = np.random.default_rng(self.seed)
            self.models_, draws = [], []
            for replicate in range(7):
                sampled = rng.choice(unique, size=len(unique), replace=True)
                indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
                model = self._child("baseline", self.seed + replicate + 1)
                model.fit(X[indices], y[indices], groups[indices])
                self.models_.append(model)
                draws.append(_json_value(sampled))
            self.config_.update({"bootstrap_unit": "subject", "bootstrap_group_draws": draws,
                                 "members": [m.config_ for m in self.models_], "ensemble_size": 7})
        elif kind in {"soft_vote", "weighted_vote", "hard_vote", "stacking"}:
            default_members = ["baseline", "extra_trees", "logistic"]
            supplied_members = getattr(self, "ensemble_members", None)
            constituent_kinds = list(default_members if supplied_members is None else supplied_members)
            if (not constituent_kinds or len(set(constituent_kinds)) != len(constituent_kinds)
                    or any(name not in default_members for name in constituent_kinds)):
                raise ValueError("ensemble_members must be a nonempty, unique subset of baseline, extra_trees, logistic")
            self.config_["ensemble_members"] = constituent_kinds
            if kind == "stacking":
                n_folds = min(3, len(np.unique(groups)))
                if n_folds < 2:
                    raise ValueError("Group-OOF stacking needs at least two distinct training subjects")
                oof = np.zeros((len(X), 7 * len(constituent_kinds)), dtype=np.float64)
                fold_records = []
                for fold, (train, valid) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
                    for i, name in enumerate(constituent_kinds):
                        model = self._child(name, self.seed + fold + 1).fit(X[train], y[train], groups[train])
                        oof[valid, i * 7:(i + 1) * 7] = model.predict_proba(X[valid])
                    fold_records.append({"fold": fold,
                                         "train_subjects": _json_value(np.unique(groups[train])),
                                         "validation_subjects": _json_value(np.unique(groups[valid]))})
                meta = (DummyClassifier(strategy="most_frequent") if len(np.unique(y)) == 1 else
                        LogisticRegression(C=1., max_iter=2500, class_weight="balanced", random_state=self.seed))
                self.meta_ = meta.fit(oof, y)
                self.config_.update({"meta_model": _estimator_config(self.meta_), "meta_input_features": 7 * len(constituent_kinds),
                                     "oof_splitter": "GroupKFold", "oof_folds": fold_records})
            self.models_ = [self._child(name).fit(X, y, groups) for name in constituent_kinds]
            if kind == "weighted_vote":
                prior = dict(zip(default_members, (.5, .3, .2)))
                self.weights_ = np.asarray([prior[name] for name in constituent_kinds])
                self.weights_ /= self.weights_.sum()
            else:
                self.weights_ = np.full(len(constituent_kinds), 1 / len(constituent_kinds))
            self.config_.update({"members": [m.config_ for m in self.models_],
                                 "weights": self.weights_.tolist(), "weights_source": "fixed_a_priori"})
        else:
            if len(np.unique(y)) == 1:
                self.estimator_ = DummyClassifier(strategy="most_frequent").fit(X, y)
            else:
                self.estimator_ = self._single(kind, X.shape[1], len(X))
                if kind == "subject_balanced_svm":
                    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
                    weights = len(y) / (len(unique) * counts[inverse])
                    # Weight the scaler and classifier identically, avoiding large-subject domination.
                    self.estimator_.fit(X, y, scale__sample_weight=weights, model__sample_weight=weights)
                    self.config_["sample_weight"] = "n_rows / (n_subjects * rows_in_subject)"
                else:
                    self.estimator_.fit(X, y)
            self.config_["estimator"] = _estimator_config(self.estimator_)
            if kind == "feature_select" and isinstance(self.estimator_, Pipeline):
                self.config_["selected_feature_indices"] = self.estimator_["select"].get_support(indices=True).tolist()
            if kind == "pca_svm" and isinstance(self.estimator_, Pipeline):
                self.config_["retained_pca_components"] = int(self.estimator_["pca"].n_components_)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "kind_")
        X = np.asarray(X, dtype=np.float64)
        if self.kind_ in {"hierarchical_correct", "hierarchical_direction"}:
            gate = _aligned_probability(self.gate_, X, len(self.branches_))
            probabilities = np.zeros((len(X), 7), dtype=np.float64)
            for i, model in enumerate(self.branch_models_):
                if model is not None:
                    probabilities += gate[:, i:i + 1] * _aligned_probability(model, X)
            return probabilities
        if self.kind_ in {"soft_vote", "weighted_vote", "hard_vote", "stacking", "group_bagging"}:
            probabilities = np.asarray([m.predict_proba(X) for m in self.models_])
            if self.kind_ == "stacking":
                return _aligned_probability(self.meta_, np.concatenate(list(probabilities), axis=1))
            if self.kind_ == "hard_vote":
                votes = np.asarray([m.predict(X) for m in self.models_])
                counts = np.mean(np.eye(7)[votes], axis=0)
                # Fixed probability-based tie breaker retains strict majority decisions.
                return (counts + 1e-9 * np.mean(probabilities, axis=0)) / (1 + 1e-9)
            if self.kind_ == "group_bagging":
                return np.mean(probabilities, axis=0)
            return np.average(probabilities, axis=0, weights=self.weights_)
        return _aligned_probability(self.estimator_, X)

    def predict(self, X):
        check_is_fitted(self, "kind_")
        if hasattr(self, "estimator_"):
            return np.asarray(self.estimator_.predict(X), dtype=int)
        return np.argmax(self.predict_proba(X), axis=1)


def build_classical(kind: str, seed: int = 42, baseline_params: dict | None = None,
                    ensemble_members: list[str] | tuple[str, ...] | None = None):
    """Build an unfitted estimator; caller owns features and all outer splits."""
    return ClassicalClassifier(kind=kind, seed=seed, baseline_params=baseline_params,
                               ensemble_members=ensemble_members)
