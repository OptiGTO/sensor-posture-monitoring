"""Shared model loading and inference helpers for batch and live prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .posture_core import SENSOR_COLUMNS, extract_feature_vector, resample_sequence
from .window_pipeline import WindowSpec, segment_sensor_frame
from .model_runtime import AdapterFactory, ModelRegistry
from .model_runtime.window_assembler import WindowAssembler
from .model_catalog import MODEL_NAMES, CHECKPOINT_MODEL_NAMES


def _model_classes(model: object) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_, dtype=int)
    if hasattr(model, "named_steps") and hasattr(model.named_steps["model"], "classes_"):
        return np.asarray(model.named_steps["model"].classes_, dtype=int)
    raise ValueError("모델의 클래스 순서를 확인할 수 없습니다.")


def _aligned_probabilities(
    model: object,
    features: np.ndarray,
    class_count: int,
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=float)
    classes = _model_classes(model)
    aligned = np.zeros((len(features), class_count), dtype=float)
    aligned[:, classes] = raw
    return aligned


def predict_probabilities(bundle: dict, features: np.ndarray) -> np.ndarray:
    """Return one probability row per feature row in the fixed label order."""

    labels = bundle["label_names"]
    if bundle.get("signal_preprocessing") is not None:
        # Features were extracted by the caller, so the raw-signal rule cannot be
        # verified here; use PosturePredictor/SklearnAdapter on raw frames instead.
        raise ValueError("신호 전처리 계약이 있는 모델은 원본 센서 프레임 경로로만 추론할 수 있습니다.")
    if bundle.get("input_representation") == "sensor_sequence":
        if np.asarray(features).ndim != 3:
            raise ValueError("이 모델에는 (창, 시점, 센서) 시계열 입력이 필요합니다.")
        return _aligned_probabilities(bundle["model"], features, len(labels))
    selected_model = bundle.get("selected_model", "RBF SVM")
    if selected_model in {"Logistic Regression", "Random Forest"}:
        return _aligned_probabilities(bundle["model"], features, len(labels))
    if selected_model == "RBF SVM":
        return _aligned_probabilities(bundle["svm_model"], features, len(labels))
    if selected_model == "Extra Trees":
        return _aligned_probabilities(bundle["extra_trees_model"], features, len(labels))
    if selected_model == "Soft Voting":
        extra_trees = _aligned_probabilities(
            bundle["extra_trees_model"], features, len(labels)
        )
        svm = _aligned_probabilities(bundle["svm_model"], features, len(labels))
        weight = float(bundle["extra_trees_weight"])
        return weight * extra_trees + (1.0 - weight) * svm
    if selected_model in (MODEL_NAMES | CHECKPOINT_MODEL_NAMES).values() and "model" in bundle:
        return _aligned_probabilities(bundle["model"], features, len(labels))
    raise ValueError(f"지원하지 않는 최종 모델 계열: {selected_model}")


def predict_probability(bundle: dict, features: np.ndarray) -> np.ndarray:
    """Return the first probability row for the legacy single-record path."""

    return predict_probabilities(bundle, features)[0]


class PosturePredictor:
    """Load the deployed bundle once and run its exact preprocessing contract."""

    def __init__(self, model_path: str | Path, window_seconds: float | None = None, *, registry: ModelRegistry | None = None) -> None:
        self.model_path = Path(model_path)
        registry = registry or ModelRegistry(Path(__file__).resolve().parents[1])
        self.descriptor = registry.descriptor_for_path(self.model_path)
        if self.descriptor is None:
            raise ValueError(f"매니페스트가 없거나 검증되지 않은 모델입니다: {self.model_path}")
        self.adapter = AdapterFactory.create(self.descriptor)
        self.bundle = getattr(self.adapter, "bundle", {})
        self.labels = self.descriptor.labels
        self.target_length = int(self.bundle.get("target_length", self.descriptor.window.target_samples(self.descriptor.window.default_seconds, self.descriptor.sampling_rate)))
        self.expected_features = len(self.bundle.get("feature_names", []))
        self.sampling_rate = self.descriptor.sampling_rate
        self.is_window_model = "window_seconds" in self.bundle
        self.confidence_semantics = self.descriptor.confidence_semantics
        self.window_seconds = self.descriptor.window.validate(window_seconds if window_seconds is not None else self.descriptor.window.default_seconds)
        self.min_tail_fraction = float(
            self.bundle.get("min_tail_fraction", 0.975)
        )
        bundled_columns = tuple(self.bundle.get("sensor_columns", SENSOR_COLUMNS))
        if bundled_columns != tuple(SENSOR_COLUMNS):
            raise ValueError(
                "모델 센서 채널이 현재 프로젝트와 다릅니다: "
                f"{list(bundled_columns)}"
            )

    @property
    def model_name(self) -> str:
        return str(self.bundle.get("selected_model", self.descriptor.model_family))

    @property
    def minimum_window_seconds(self) -> float:
        return self.window_seconds * self.min_tail_fraction

    def predict_windows(self, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Predict every non-overlapping window using the saved model contract."""

        if not self.is_window_model:
            raise ValueError("3초·5초 창 모델에서만 창별 예측을 사용할 수 있습니다.")
        return self.adapter.predict_windows(frame, self.window_seconds)

    def _validate_features(self, features: np.ndarray) -> None:
        if self.expected_features and features.shape[1] != self.expected_features:
            raise ValueError(
                "특징 수가 모델과 다릅니다: "
                f"입력 {features.shape[1]}, 모델 {self.expected_features}"
            )

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if self.descriptor.backend == "pytorch":
            sequence = WindowAssembler().assemble(frame, self.descriptor, self.window_seconds)
            return self.adapter.predict_proba(sequence, self.window_seconds)[0]
        probabilities = self.adapter.predict_proba(frame, self.window_seconds)
        return probabilities.mean(axis=0)
