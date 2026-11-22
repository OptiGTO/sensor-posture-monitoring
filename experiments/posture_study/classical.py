"""Isolated, reproducible classical hypotheses for the posture study.

All fitted transformations use the supplied training partition only.  Validation
is accepted for a common estimator interface but is deliberately unused here.
Ensemble weights and stacking features use subject-disjoint training OOF folds.
The caller owns the outer train/validation/test split and all temporal policies.
"""

from __future__ import annotations

import time
import warnings
from collections import Counter
from typing import Mapping

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

VERSION = "posture-study-classical-v1"
METHODS = (
    "baseline", "relative", "spatial", "robust", "select", "pca", "personal",
    "multiscale", "subject_weight", "gain_aug", "hierarchy", "soft_hierarchy",
    "extra_trees", "histboost", "lda", "knn", "random_conv", "logreg",
    "majority", "weighted", "stacking",
)
_FEATURE_METHODS = {
    "relative", "spatial", "robust", "select", "pca", "personal", "multiscale",
    "subject_weight",
}
_COMPONENTS = _FEATURE_METHODS | {"gain_aug"}


def _subset(data: Mapping[str, np.ndarray], indices: np.ndarray) -> dict:
    n = len(data["features"])
    return {
        key: value[indices] if isinstance(value, np.ndarray) and len(value) == n else value
        for key, value in data.items()
    }


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        # Binary SVC scores are signed margins, unlike its multiclass OVR scores.
        scores = np.column_stack((-scores, scores))
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _probabilities(model, features: np.ndarray, classes: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probability = np.asarray(model.predict_proba(features), dtype=np.float64)
    else:
        probability = _softmax(model.decision_function(features))
    result = np.zeros((len(features), len(classes)), dtype=np.float64)
    positions = {int(label): i for i, label in enumerate(classes)}
    for column, label in enumerate(model.classes_):
        result[:, positions[int(label)]] = probability[:, column]
    return result


def _spatial_features(sequences: np.ndarray) -> np.ndarray:
    """Compact within-frame pressure shape; orientation remains row/column only."""
    seq = np.asarray(sequences, dtype=np.float64)
    # A common offset/gain within each frame does not change this representation.
    relative = (seq - seq.mean(axis=2, keepdims=True)) / (
        seq.std(axis=2, keepdims=True) + 1e-8
    )
    grids = relative.reshape(len(seq), seq.shape[1], 4, 4)
    nonnegative = seq - seq.min(axis=2, keepdims=True)
    distribution = nonnegative / (nonnegative.sum(axis=2, keepdims=True) + 1e-8)
    rows = np.repeat(np.arange(4, dtype=float), 4)
    columns = np.tile(np.arange(4, dtype=float), 4)
    center_row = (distribution * rows).sum(axis=2)
    center_column = (distribution * columns).sum(axis=2)
    asymmetry_row = (grids[:, :, :2].sum(axis=(2, 3))
                     - grids[:, :, 2:].sum(axis=(2, 3)))
    asymmetry_column = (grids[:, :, :, :2].sum(axis=(2, 3))
                        - grids[:, :, :, 2:].sum(axis=(2, 3)))
    entropy = -(distribution * np.log(distribution + 1e-8)).sum(axis=2)
    summaries = np.stack((center_row, center_column, asymmetry_row,
                          asymmetry_column, entropy), axis=2)
    return np.concatenate((
        relative.mean(axis=1), relative.std(axis=1),
        grids.mean(axis=3).mean(axis=1), grids.mean(axis=3).std(axis=1),
        grids.mean(axis=2).mean(axis=1), grids.mean(axis=2).std(axis=1),
        summaries.mean(axis=1), summaries.std(axis=1),
        np.abs(np.diff(relative, axis=1)).mean(axis=1),
    ), axis=1)


def _multiscale_features(sequences: np.ndarray) -> np.ndarray:
    seq = np.asarray(sequences, dtype=np.float64)
    features = []
    # At the study's 12 Hz rate, the final 12 and 24 frames are 1 s and 2 s.
    for frames in (12, 24):
        tail = seq[:, -min(frames, seq.shape[1]):]
        relative = (tail - tail.mean(axis=2, keepdims=True)) / (
            tail.std(axis=2, keepdims=True) + 1e-8
        )
        for values in (tail, relative):
            features.extend((values.mean(axis=1), values.std(axis=1),
                             values[:, -1] - values[:, 0]))
    return np.concatenate(features, axis=1)


class _RandomConvolution:
    """ROCKET-inspired fixed random kernels with max/positive-proportion pooling.

    This is an independent small implementation, not a dependency on ROCKET.
    Kernel parameters are generated once from the recorded seed.  Channel scale
    statistics are fitted on training windows only and are stored in checkpoint.
    """

    def __init__(self, seed: int, n_kernels: int = 128):
        self.seed = seed
        self.n_kernels = n_kernels

    def fit(self, seq: np.ndarray):
        rng = np.random.default_rng(self.seed)
        self.mean_ = seq.mean(axis=(0, 1), keepdims=True)
        self.scale_ = seq.std(axis=(0, 1), keepdims=True) + 1e-6
        self.kernels_ = []
        for _ in range(self.n_kernels):
            length = int(rng.choice([3, 5, 7]))
            dilation = int(rng.choice([1, 2, 4]))
            channels = np.sort(rng.choice(seq.shape[2], size=int(rng.choice([1, 4, 8, 16])), replace=False))
            weights = rng.normal(size=(len(channels), length))
            weights -= weights.mean(axis=1, keepdims=True)
            weights /= np.linalg.norm(weights) + 1e-8
            self.kernels_.append((channels, weights, dilation, float(rng.uniform(-1, 1))))
        return self

    def transform(self, seq: np.ndarray) -> np.ndarray:
        from numpy.lib.stride_tricks import sliding_window_view

        values = (np.asarray(seq, dtype=np.float64) - self.mean_) / self.scale_
        output = np.empty((len(seq), 2 * self.n_kernels), dtype=np.float64)
        for index, (channels, weights, dilation, bias) in enumerate(self.kernels_):
            extent = 1 + dilation * (weights.shape[1] - 1)
            if extent > values.shape[1]:
                raise ValueError("Random convolution requires at least 25 temporal frames")
            windows = sliding_window_view(values[:, :, channels], extent, axis=1)
            responses = np.einsum("ntck,ck->nt", windows[:, :, :, ::dilation], weights) + bias
            output[:, index * 2] = responses.max(axis=1)
            output[:, index * 2 + 1] = (responses > 0).mean(axis=1)
        return output


class ClassicalEstimator:
    """Factory-compatible estimator accepting aligned study data dictionaries.

    ``components`` supports progressive feature/learning additions without
    changing ``method``. ``relative`` and ``spatial`` replace legacy features;
    ``personal`` and ``multiscale`` append features; the remaining components
    change fitted scaling, projection, selection or training weights.  The two
    replacement feature representations cannot be requested simultaneously.
    """

    def __init__(self, method: str = "baseline", seed: int = 20260913,
                 components: list[str] | tuple[str, ...] | None = None,
                 ensemble_members: list[str] | tuple[str, ...] | None = None):
        if method not in METHODS:
            raise ValueError(f"Unknown classical method: {method}")
        self.method = method
        self.seed = int(seed)
        self.ensemble_members = tuple(ensemble_members or ("baseline", "logreg", "extra_trees"))
        if len(self.ensemble_members) < 2 or len(set(self.ensemble_members)) != len(self.ensemble_members):
            raise ValueError("Ensembles need at least two distinct base learner methods")
        if any(member not in METHODS or member in {"majority", "weighted", "stacking"}
               for member in self.ensemble_members):
            raise ValueError("Ensemble members must be non-ensemble classical methods")
        component_list = list(components or [])
        if method in _FEATURE_METHODS or method == "gain_aug":
            component_list.insert(0, method)
        self.components = list(dict.fromkeys(component_list))
        unknown = set(self.components) - _COMPONENTS
        if unknown:
            raise ValueError(f"Unknown classical components: {sorted(unknown)}")
        if {"relative", "spatial"} <= set(self.components):
            raise ValueError("relative and spatial are alternative base feature representations")
        if {"select", "pca"} <= set(self.components):
            raise ValueError("select and pca are alternative dimension reduction hypotheses")
        self.classes_ = np.arange(7)

    def _features(self, data: Mapping[str, np.ndarray], fit: bool = False) -> np.ndarray:
        if self.method == "random_conv":
            if fit:
                self.random_transform_ = _RandomConvolution(self.seed).fit(np.asarray(data["seq"]))
            base = self.random_transform_.transform(np.asarray(data["seq"]))
        elif "spatial" in self.components:
            base = _spatial_features(data["seq"])
        elif "relative" in self.components:
            base = np.asarray(data["features"])[:, 298:]
        else:
            base = np.asarray(data["features"])
        features = [base]
        if "personal" in self.components:
            features.append(np.asarray(data["personal"]))
        if "multiscale" in self.components:
            features.append(_multiscale_features(data["seq"]))
        result = np.concatenate(features, axis=1).astype(np.float64, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("Study features must be finite before fitting or prediction")
        return result

    def _pipeline(self, n_features: int, classifier: str | None = None) -> Pipeline:
        classifier = classifier or self.method
        scaler = RobustScaler(quantile_range=(10, 90)) if "robust" in self.components else StandardScaler()
        steps = [("scale", scaler)]
        if "select" in self.components:
            steps.append(("select", SelectKBest(f_classif, k=min(96, n_features))))
        if "pca" in self.components:
            steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
        if classifier == "extra_trees":
            model = ExtraTreesClassifier(n_estimators=250, min_samples_leaf=3,
                max_features="sqrt", class_weight="balanced", n_jobs=1, random_state=self.seed)
        elif classifier == "histboost":
            model = HistGradientBoostingClassifier(max_iter=150, max_depth=5,
                max_leaf_nodes=15, l2_regularization=1, early_stopping=False,
                random_state=self.seed)
        elif classifier == "lda":
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        elif classifier == "knn":
            model = KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=1)
        elif classifier == "random_conv":
            model = RidgeClassifier(alpha=10, class_weight="balanced")
        elif classifier == "logreg":
            model = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000,
                                       random_state=self.seed)
        else:
            # Match the audited production family including libsvm's internal
            # probability calibration. Its internal folds stay within the outer
            # training data but are not subject-grouped; record that limitation.
            model = SVC(C=1.0, gamma=0.0005, kernel="rbf", probability=True,
                        class_weight="balanced", random_state=self.seed)
        return Pipeline(steps + [("model", model)])

    def fit(self, data: Mapping[str, np.ndarray], y: np.ndarray, validation=None):
        started = time.perf_counter()
        labels = np.asarray(y, dtype=np.int64)
        if set(np.unique(labels)) != set(self.classes_):
            raise ValueError("Every study training partition must contain all seven classes")
        self.n_training_windows_ = len(labels)
        self.training_groups_ = sorted(str(value) for value in np.unique(data["groups"]))
        if self.method in {"majority", "weighted", "stacking"}:
            self._fit_ensemble(data, labels)
            self.n_features_ = int(np.asarray(data["features"]).shape[1])
        else:
            fit_data = data
            if "gain_aug" in self.components:
                fit_data, labels = self._augment(data, labels)
            features = self._features(fit_data, fit=True)
            self.n_features_ = int(features.shape[1])
            sample_weights = None
            if "subject_weight" in self.components:
                counts = Counter(np.asarray(fit_data["groups"]).tolist())
                sample_weights = np.asarray([1 / counts[group] for group in fit_data["groups"]])
                sample_weights *= len(sample_weights) / sample_weights.sum()
            if self.method in {"hierarchy", "soft_hierarchy"}:
                incorrect = labels != 0
                self.gate_ = self._pipeline(self.n_features_, classifier="baseline")
                self.detail_ = self._pipeline(self.n_features_, classifier="baseline")
                gate_kwargs = {} if sample_weights is None else {"model__sample_weight": sample_weights}
                detail_kwargs = {} if sample_weights is None else {"model__sample_weight": sample_weights[incorrect]}
                self.gate_.fit(features, incorrect.astype(int), **gate_kwargs)
                self.detail_.fit(features[incorrect], labels[incorrect], **detail_kwargs)
            else:
                self.model_ = self._pipeline(self.n_features_)
                kwargs = {} if sample_weights is None else {"model__sample_weight": sample_weights}
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Features .* are constant")
                    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
                    self.model_.fit(features, labels, **kwargs)
        self.fit_seconds_ = float(time.perf_counter() - started)
        return self

    def _augment(self, data: Mapping[str, np.ndarray], labels: np.ndarray) -> tuple[dict, np.ndarray]:
        from src.posture_core import extract_feature_vector

        if "personal" in self.components:
            raise ValueError("gain_aug with personal needs corresponding baseline gain augmentation")
        rng = np.random.default_rng(self.seed)
        seq = np.asarray(data["seq"])
        augmented = seq * rng.uniform(0.85, 1.15, size=(len(seq), 1, seq.shape[2]))
        augmented_features = np.stack([extract_feature_vector(window) for window in augmented])
        combined = {
            key: np.concatenate((value, value), axis=0)
            if isinstance(value, np.ndarray) and len(value) == len(labels) else value
            for key, value in data.items()
        }
        combined["seq"] = np.concatenate((seq, augmented), axis=0)
        combined["features"] = np.concatenate((data["features"], augmented_features), axis=0)
        return combined, np.concatenate((labels, labels))

    def _fit_ensemble(self, data: Mapping[str, np.ndarray], labels: np.ndarray):
        # getattr keeps previously written default checkpoints usable.
        methods = getattr(self, "ensemble_members", ("baseline", "logreg", "extra_trees"))
        self.ensemble_weights_ = np.ones(len(methods), dtype=float) / len(methods)
        self.oof_macro_f1_ = None
        self.oof_fold_groups_ = []
        if self.method in {"weighted", "stacking"}:
            groups = np.asarray(data["groups"])
            n_splits = min(3, len(np.unique(groups)))
            if n_splits < 2:
                raise ValueError("Subject-disjoint OOF ensembles need at least two training subjects")
            oof = np.zeros((len(labels), len(methods), 7), dtype=float)
            cv = GroupKFold(n_splits=n_splits)
            for train, held in cv.split(data["features"], labels, groups):
                self.oof_fold_groups_.append({
                    "train": sorted(str(v) for v in np.unique(groups[train])),
                    "validation": sorted(str(v) for v in np.unique(groups[held])),
                })
                for index, method in enumerate(methods):
                    estimator = ClassicalEstimator(method, self.seed, self.components)
                    estimator.fit(_subset(data, train), labels[train])
                    oof[held, index] = estimator.predict_proba(_subset(data, held))
            scores = np.asarray([f1_score(labels, oof[:, index].argmax(axis=1),
                                labels=self.classes_, average="macro", zero_division=0)
                                for index in range(len(methods))])
            self.oof_macro_f1_ = scores.tolist()
            if self.method == "weighted":
                self.ensemble_weights_ = (scores + 1e-8) / (scores + 1e-8).sum()
            else:
                self.meta_ = Pipeline([
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(C=0.1, class_weight="balanced",
                     max_iter=2000, random_state=self.seed)),
                ])
                self.meta_.fit(oof.reshape(len(labels), -1), labels)
        self.members_ = [ClassicalEstimator(method, self.seed, self.components).fit(data, labels)
                         for method in methods]

    def predict_proba(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.method in {"majority", "weighted", "stacking"}:
            probabilities = np.stack([member.predict_proba(data) for member in self.members_], axis=1)
            if self.method == "stacking":
                return _probabilities(self.meta_, probabilities.reshape(len(probabilities), -1), self.classes_)
            if self.method == "majority":
                votes = np.eye(7)[probabilities.argmax(axis=2)].mean(axis=1)
                # Probability is only a deterministic secondary tie-breaker.
                result = votes + 1e-6 * probabilities.mean(axis=1)
                return result / result.sum(axis=1, keepdims=True)
            return (probabilities * self.ensemble_weights_[None, :, None]).sum(axis=1)
        features = self._features(data)
        if self.method in {"hierarchy", "soft_hierarchy"}:
            gate = _probabilities(self.gate_, features, np.arange(2))
            detail = _probabilities(self.detail_, features, np.arange(1, 7))
            result = np.zeros((len(features), 7), dtype=float)
            if self.method == "hierarchy":
                incorrect = gate.argmax(axis=1) == 1
                result[~incorrect, 0] = 1.0
                result[incorrect, 1:] = detail[incorrect]
            else:
                result[:, 0] = gate[:, 0]
                result[:, 1:] = gate[:, 1, None] * detail
            return result
        return _probabilities(self.model_, features, self.classes_)

    def predict(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.predict_proba(data).argmax(axis=1)

    def complexity(self) -> dict:
        def describe(pipeline):
            model = pipeline.named_steps["model"]
            result = {"estimator": type(model).__name__, "hyperparameters": model.get_params(deep=False)}
            if hasattr(model, "support_vectors_"):
                result["support_vectors"] = int(len(model.support_vectors_))
                result["support_vector_dimensions"] = int(model.support_vectors_.shape[1])
                result["numeric_parameter_count"] = int(model.support_vectors_.size + model.dual_coef_.size + model.intercept_.size)
            elif hasattr(model, "coef_"):
                result["numeric_parameter_count"] = int(model.coef_.size + model.intercept_.size)
            if hasattr(model, "estimators_"):
                trees = [tree for tree in np.asarray(model.estimators_, dtype=object).flat if hasattr(tree, "tree_")]
                result["trees"] = len(trees)
                result["tree_nodes"] = int(sum(tree.tree_.node_count for tree in trees))
            elif hasattr(model, "_predictors"):
                trees = [tree for stage in model._predictors for tree in stage]
                result["trees"] = len(trees)
                result["tree_nodes"] = int(sum(len(tree.nodes) for tree in trees))
            if hasattr(model, "_fit_X"):
                result["stored_training_vectors"] = int(model._fit_X.shape[0])
                result["stored_numeric_values"] = int(model._fit_X.size)
            if "pca" in pipeline.named_steps:
                result["pca_components"] = int(pipeline.named_steps["pca"].n_components_)
            if "select" in pipeline.named_steps:
                result["selected_features"] = int(pipeline.named_steps["select"].get_support().sum())
            result["preprocessing"] = [name for name, _ in pipeline.steps[:-1]]
            result["preprocessing_configuration"] = {
                name: {key: (value.__name__ if callable(value) else value)
                       for key, value in step.get_params(deep=False).items()}
                for name, step in pipeline.steps[:-1]
            }
            return result

        result = {"version": VERSION, "method": self.method, "seed": self.seed,
                  "components": list(self.components), "input_features": self.n_features_,
                  "training_windows": self.n_training_windows_, "training_subjects": self.training_groups_,
                  "fit_seconds": self.fit_seconds_, "probability_semantics": "native probability or uncalibrated decision softmax",
                  "svc_calibration": "libsvm probability=True; training-only internal non-grouped calibration"}
        if hasattr(self, "members_"):
            result["ensemble_member_methods"] = [member.method for member in self.members_]
            result["members"] = [member.complexity() for member in self.members_]
            result["weights"] = self.ensemble_weights_.tolist()
            result["oof_macro_f1"] = self.oof_macro_f1_
            result["oof_fold_groups"] = self.oof_fold_groups_
            if hasattr(self, "meta_"):
                result["meta_model"] = describe(self.meta_)
        elif hasattr(self, "gate_"):
            result["gate"] = describe(self.gate_)
            result["detail"] = describe(self.detail_)
        else:
            result["model"] = describe(self.model_)
        if hasattr(self, "random_transform_"):
            result["random_kernels"] = self.random_transform_.n_kernels
            result["random_kernel_seed"] = self.random_transform_.seed
            result["kernel_lengths"] = [3, 5, 7]
            result["kernel_dilations"] = [1, 2, 4]
        return result
