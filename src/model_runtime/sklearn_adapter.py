"""Compatibility adapter for immutable sklearn joblib bundles."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .contracts import ModelDescriptor
from ..posture_core import SENSOR_COLUMNS, extract_feature_vector, resample_sequence
from ..signal_preprocessing import SignalPreprocessing
from ..window_pipeline import WindowSpec, segment_sensor_frame
from ..model_catalog import MODEL_NAMES, CHECKPOINT_MODEL_NAMES


def _model_classes(model: object) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_, dtype=int)
    if hasattr(model, "named_steps") and hasattr(model.named_steps["model"], "classes_"):
        return np.asarray(model.named_steps["model"].classes_, dtype=int)
    raise ValueError("모델의 클래스 순서를 확인할 수 없습니다.")


def _aligned_probabilities(model: object, features: np.ndarray, class_count: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=float)
    classes = _model_classes(model)
    aligned = np.zeros((len(features), class_count), dtype=float)
    aligned[:, classes] = raw
    return aligned


class SklearnAdapter:
    def __init__(self, descriptor: ModelDescriptor) -> None:
        self.descriptor = descriptor
        self.bundle = joblib.load(descriptor.artifact_path)
        self.labels = descriptor.labels
        bundled_columns = tuple(self.bundle.get("sensor_columns", SENSOR_COLUMNS))
        if bundled_columns != tuple(descriptor.input_channels):
            raise ValueError("모델 센서 채널이 매니페스트와 다릅니다.")
        if tuple(map(str, self.bundle.get("label_names", ()))) != self.labels:
            raise ValueError("모델 라벨 순서가 매니페스트와 다릅니다.")
        self.expected_features = len(self.bundle.get("feature_names", []))
        self.input_representation = self.bundle.get("input_representation", "feature_vector")
        if self.input_representation not in {"feature_vector", "sensor_sequence"}:
            raise ValueError("지원하지 않는 모델 입력 형식입니다.")
        # Fail closed: an unknown or altered rule must not silently run unprocessed.
        self.signal_preprocessing = SignalPreprocessing.from_bundle(self.bundle)

    def validate_runtime(self, window_seconds: float) -> float:
        return self.descriptor.window.validate(window_seconds)

    def _validate_output(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.labels) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("모델 출력 계약 검증에 실패했습니다.")
        if self.descriptor.output_type == "probabilities" and not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("확률 출력의 합이 1이 아닙니다.")
        return values

    def predict_proba(self, frame: pd.DataFrame, window_seconds: float | None = None) -> np.ndarray:
        selected = self.validate_runtime(window_seconds if window_seconds is not None else self.descriptor.window.default_seconds)
        if self.bundle.get("window_seconds") is not None:
            return self.predict_windows(frame, selected)[0]
        target = self.descriptor.window.target_samples(selected, self.descriptor.sampling_rate)
        frame = self.signal_preprocessing.apply_inference(frame)
        features = extract_feature_vector(resample_sequence(frame, target_length=target)).reshape(1, -1)
        return self._predict_features(features)

    def _predict_features(self, features: np.ndarray) -> np.ndarray:
        if self.input_representation != "feature_vector":
            raise ValueError("이 모델에는 특징 벡터 대신 센서 시계열이 필요합니다.")
        if self.expected_features and features.shape[1] != self.expected_features:
            raise ValueError(f"특징 수가 모델과 다릅니다: 입력 {features.shape[1]}, 모델 {self.expected_features}")
        selected_model = self.bundle.get("selected_model", "RBF SVM")
        if selected_model in {"Logistic Regression", "Random Forest"}:
            result = _aligned_probabilities(self.bundle["model"], features, len(self.labels))
        elif selected_model == "RBF SVM":
            result = _aligned_probabilities(self.bundle["svm_model"], features, len(self.labels))
        elif selected_model == "Extra Trees":
            result = _aligned_probabilities(self.bundle["extra_trees_model"], features, len(self.labels))
        elif selected_model == "Soft Voting":
            extra = _aligned_probabilities(self.bundle["extra_trees_model"], features, len(self.labels))
            svm = _aligned_probabilities(self.bundle["svm_model"], features, len(self.labels))
            result = float(self.bundle["extra_trees_weight"]) * extra + (1 - float(self.bundle["extra_trees_weight"])) * svm
        elif selected_model in (MODEL_NAMES | CHECKPOINT_MODEL_NAMES).values() and "model" in self.bundle:
            result = _aligned_probabilities(self.bundle["model"], features, len(self.labels))
        else:
            raise ValueError(f"지원하지 않는 최종 모델 계열: {selected_model}")
        return self._validate_output(result)

    def predict_windows(self, frame: pd.DataFrame, window_seconds: float | None = None) -> tuple[np.ndarray, pd.DataFrame]:
        if self.bundle.get("window_seconds") is None:
            raise ValueError("고정 20초 모델에서는 창별 예측을 사용할 수 없습니다.")
        selected = self.validate_runtime(window_seconds if window_seconds is not None else self.descriptor.window.default_seconds)
        spec = WindowSpec(window_seconds=selected, sampling_rate=self.descriptor.sampling_rate, min_tail_fraction=float(self.bundle.get("min_tail_fraction", 0.975)))
        windows, metadata = segment_sensor_frame(self.signal_preprocessing.apply_inference(frame), spec)
        if self.input_representation == "sensor_sequence":
            return self._validate_output(_aligned_probabilities(self.bundle["model"], windows, len(self.labels))), metadata
        features = np.stack([extract_feature_vector(window) for window in windows])
        return self._predict_features(features), metadata
