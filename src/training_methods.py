"""Reusable training recipes, with evidence-gated public choices.

Research calls may compare any supported recipe. The app and train_comparison
must additionally validate the selected recipe against the checked-in evidence
registry. Every transform stays inside the serialized estimator so inference
continues to receive the original feature vector or sensor sequence.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted

from .model_catalog import MODEL_NAMES
from .posture_core import feature_names
from .training_windows import MIN_WINDOW_SECONDS, MAX_WINDOW_SECONDS
from .checkpoint_registry import USER_ADOPTION_STATUS, checkpoint_method_entries, get_checkpoint, is_checkpoint


METHOD_LABELS = {
    "standard": "기본 학습",
    "regularized": "규제 재튜닝",
    "relative": "상대압력 특징",
    "selected": "핵심 특징 96개",
    "subject_balanced": "사용자 균형 학습",
    "cnn_augmentation": "CNN 센서 증강",
}
REGISTRY_PATH = Path(__file__).with_name("training_improvements.json")
CLASSICAL_FAMILIES = frozenset({"rbf_svm", "logistic_regression", "random_forest", "extra_trees", "xgboost", "catboost"})
REGULARIZED_FAMILIES = frozenset({"rbf_svm", "random_forest", "extra_trees", "xgboost"})


def _check_supported(family: str, method: str) -> None:
    if is_checkpoint(method):
        checkpoint = get_checkpoint(method)
        if family != checkpoint["pipeline"]["family"]:
            raise ValueError("체크포인트에 저장된 모델 구성과 다릅니다.")
        return
    if family not in MODEL_NAMES or method not in METHOD_LABELS:
        raise ValueError(f"Unknown training method/family: {method}/{family}")
    allowed = (method == "standard"
        or method in {"relative", "selected", "subject_balanced"} and family in CLASSICAL_FAMILIES
        or method == "regularized" and family in REGULARIZED_FAMILIES
        or method == "cnn_augmentation" and family == "cnn1d")
    if not allowed:
        raise ValueError(f"Training method {method} does not support {family}")


def method_details(method: str) -> dict:
    if is_checkpoint(method):
        checkpoint = get_checkpoint(method)
        adopted = checkpoint["status"] == USER_ADOPTION_STATUS
        details = {"id": method, "label": checkpoint["name"], "checkpoint": checkpoint,
                "pipeline": deepcopy(checkpoint["pipeline"]),
                "checkpoint_status": checkpoint["status"], "preprocessing_fit": "training subjects only",
                "policy": ("exact user-adopted research checkpoint recipe; no parameter search; adoption does not certify meaningful improvement"
                    if adopted else "exact retrospectively validated checkpoint recipe; no parameter search"),
                "training_data": "User-selected eligible records with subject-disjoint splits; original research cache is not appended"}
        if checkpoint["pipeline"]["training"].get("data_policy") == "include_ordinary_reserved_training_records_v1":
            details["data_policy_provenance"] = ("The original research cohort restored ordinary correct-posture records "
                "only for training subjects. This records the saved artifact's training provenance; new app training "
                "uses the user's eligible selected records without importing the historical supplemental cache.")
        return details
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown training method: {method}")
    details = {"id": method, "label": METHOD_LABELS[method], "preprocessing_fit": "training subjects only"}
    policies = {
        "standard": "Existing candidate grid and preprocessing",
        "regularized": "Prespecified regularization grid; validation chooses parameters",
        "relative": "Keep rel_ columns inside estimator; original 596-feature runtime input",
        "selected": "ANOVA SelectKBest(k=96) fitted only on training labels, before scaling",
        "subject_balanced": "Equal total weight per training subject; equal observed class mass within each subject; weighted scaler; no additional class weighting",
        "cnn_augmentation": "Training-only channel gain and additive noise in raw sensor space; noise scale from training data; deterministic inference",
    }
    details["policy"] = policies[method]
    return details


def available_training_methods(registry_path: Path | None = None) -> list[dict]:
    """Return standard, evidenced improvements and explicit research adoptions.

    A missing/invalid method registry contributes no optional methods.
    Independently verified checkpoints remain available. Registry evidence
    controls availability of a training recipe, never model approval/activation.
    """
    result = [{"id": "standard", "label": METHOD_LABELS["standard"],
        "families": list(MODEL_NAMES), "window_seconds": list(range(MIN_WINDOW_SECONDS, MAX_WINDOW_SECONDS + 1)), "evidence": {}}]
    result.extend(checkpoint_method_entries())
    try:
        payload = json.loads(Path(registry_path or REGISTRY_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return result
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("methods"), list):
        return result
    seen = {"standard"}
    for entry in payload["methods"]:
        if not isinstance(entry, dict):
            continue
        method = entry.get("id")
        families, windows, evidence = entry.get("families"), entry.get("window_seconds"), entry.get("evidence")
        if not isinstance(method, str) or method not in METHOD_LABELS or method in seen:
            continue
        if not isinstance(families, list) or not families or any(not isinstance(family, str) for family in families):
            continue
        if not isinstance(windows, list) or not windows or any(type(value) is not int
                or not MIN_WINDOW_SECONDS <= value <= MAX_WINDOW_SECONDS for value in windows):
            continue
        if not isinstance(evidence, dict) or evidence.get("improved") is not True:
            continue
        try:
            for family in families:
                _check_supported(family, method)
        except ValueError:
            continue
        clean = deepcopy(entry)
        clean["label"] = str(entry.get("label") or METHOD_LABELS[method])
        clean["families"] = list(dict.fromkeys(families))
        clean["window_seconds"] = list(dict.fromkeys(windows))
        result.append(clean)
        seen.add(method)
    return result


def validate_training_method(method: str, families, windows) -> dict:
    """Fail before training/output writes if a nonstandard recipe lacks evidence."""
    selected = next((entry for entry in available_training_methods() if entry["id"] == method), None)
    if selected is None:
        raise ValueError(f"성능 개선 근거가 등록되지 않은 학습 방식입니다: {method}")
    if method == "standard" and any(family not in MODEL_NAMES for family in families):
        raise ValueError("완전한 체크포인트 구성 없이 사용할 수 없는 모델입니다.")
    if method != "standard":
        if any(family not in selected["families"] for family in families):
            raise ValueError(f"선택한 모델에서 검증되지 않은 학습 방식입니다: {method}")
        if any(seconds not in selected["window_seconds"] for seconds in windows):
            raise ValueError(f"선택한 시간 창에서 검증되지 않은 학습 방식입니다: {method}")
    return selected


class RelativeFeatureSelector(TransformerMixin, BaseEstimator):
    """Select named relative features, retaining the external 596-column input."""

    def fit(self, X, y=None):
        values = check_array(X)
        names = feature_names()
        if values.shape[1] != len(names):
            raise ValueError(f"Expected original {len(names)}-feature input")
        self.n_features_in_ = values.shape[1]
        self.indices_ = np.array([i for i, name in enumerate(names) if name.startswith("rel_")])
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        values = check_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature input differs from training contract")
        return values[:, self.indices_]


def subject_balanced_weights(y, groups) -> np.ndarray:
    """Mean-one weights with equal subject mass and within-subject class mass.

    Using global class_weight='balanced' on top would destroy equal subject
    totals when subjects have different class coverage, so that is disabled.
    """
    labels, subjects = np.asarray(y), np.asarray(groups)
    if labels.ndim != 1 or subjects.ndim != 1 or not len(labels) or len(labels) != len(subjects):
        raise ValueError("Training subject IDs must match a nonempty label vector")
    weights = np.zeros(len(labels), dtype=np.float64)
    identities = np.unique(subjects)
    for subject in identities:
        subject_mask = subjects == subject
        classes, counts = np.unique(labels[subject_mask], return_counts=True)
        for label, count in zip(classes, counts):
            weights[subject_mask & (labels == label)] = len(labels) / (len(identities) * len(classes) * count)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Training subject weights must be finite and positive")
    return weights


def method_candidates(family: str, method: str = "standard") -> list[dict]:
    _check_supported(family, method)
    if is_checkpoint(method):
        return [deepcopy(get_checkpoint(method)["pipeline"]["model_params"])]
    from .model_comparison import MODEL_CANDIDATES
    if method == "regularized":
        if family == "rbf_svm":
            return [{"C": c, "gamma": gamma} for c in (0.3, 1.0, 3.0, 10.0) for gamma in (0.0005, 0.001)]
        if family in {"random_forest", "extra_trees"}:
            return [{"n_estimators": 200, "min_samples_leaf": leaf, "max_depth": depth, "max_features": features}
                for leaf in (3, 8) for depth in (8, None) for features in ("sqrt", 0.5)]
        return [{"n_estimators": 200, "max_depth": depth, "learning_rate": 0.05,
            "min_child_weight": 8, "reg_lambda": penalty, "reg_alpha": 1.0,
            "subsample": 0.8, "colsample_bytree": 0.8}
            for depth in (2, 3) for penalty in (10.0, 30.0)]
    if method == "cnn_augmentation":
        return [{**params, "gain_std": gain, "noise_fraction": noise}
            for params in MODEL_CANDIDATES[family] for gain, noise in ((0.03, 0.01), (0.08, 0.03))]
    return deepcopy(MODEL_CANDIDATES[family])


def build_method_model(family: str, params: dict, seed: int, method: str = "standard"):
    _check_supported(family, method)
    if is_checkpoint(method):
        from .pipeline_recipes import build_pipeline
        pipeline = get_checkpoint(method)["pipeline"]
        if params != pipeline["model_params"]:
            raise ValueError("체크포인트의 고정 하이퍼파라미터를 변경할 수 없습니다.")
        # The caller's seed controls dataset splitting; the trained recipe keeps
        # its original estimator seed, recorded separately in the full pipeline.
        return build_pipeline(pipeline)
    from .model_comparison import build_model
    model = build_model(family, dict(params), seed)
    if method == "relative":
        model = Pipeline([("relative", RelativeFeatureSelector()), *model.steps])
    elif method == "selected":
        model = Pipeline([("selection", SelectKBest(f_classif, k=96)), *model.steps])
    elif method == "subject_balanced":
        estimator = model.named_steps["model"]
        if family == "catboost":
            estimator.set_params(auto_class_weights=None)
        elif family != "xgboost":
            estimator.set_params(class_weight=None)
    return model


def fit_method_model(model, family: str, method: str, X, y, *, groups=None, record_ids=None, validation_data=None):
    _check_supported(family, method)
    if is_checkpoint(method):
        from .pipeline_recipes import fit_pipeline
        pipeline = get_checkpoint(method)["pipeline"]
        fit_pipeline(model, pipeline, X, y, groups=groups, record_ids=record_ids, validation_data=validation_data)
        model.training_details_ = {**getattr(model, "training_details_", {}),
                                   "training_method": method_details(method)}
        return model
    options = {}
    if method == "subject_balanced":
        if groups is None:
            raise ValueError("Subject-balanced training requires training subject IDs")
        weights = subject_balanced_weights(y, groups)
        options.update(scale__sample_weight=weights, model__sample_weight=weights)
    elif family in {"cnn1d", "mlp"}:
        options["validation_data"] = validation_data
    elif family == "xgboost":
        from sklearn.utils.class_weight import compute_sample_weight
        options["model__sample_weight"] = compute_sample_weight("balanced", y)
    model.fit(X, y, **options)
    if method != "standard":
        details = {**getattr(model, "training_details_", {}), "training_method": method_details(method)}
        if method == "relative":
            details["selected_feature_names"] = [feature_names()[i] for i in model.named_steps["relative"].indices_]
        elif method == "selected":
            details["selected_feature_names"] = [feature_names()[i] for i in model.named_steps["selection"].get_support(indices=True)]
        elif method == "subject_balanced":
            details["training_subject_count"] = int(len(np.unique(groups)))
            details["weight_min"], details["weight_max"] = float(weights.min()), float(weights.max())
        model.training_details_ = details
    return model
