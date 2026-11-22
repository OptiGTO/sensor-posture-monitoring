"""Personal-reference conditioned binary gate followed by an incorrect-only classifier.

Feature and artifact contracts are shared by offline fitting and live inference.
Scores are explicitly uncalibrated; no per-user model is fitted from ten seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .personal_guidance import CalibrationProfile
from .posture_core import CLASS_NAMES, SENSOR_COLUMNS

FEATURE_VERSION = "personal-ratio-window-v1"
WINDOW_SECONDS = 3.0
SAMPLING_RATE = 12.0
TARGET_SAMPLES = 36


def features(sequence: np.ndarray, profile: CalibrationProfile) -> tuple[np.ndarray, np.ndarray]:
    """Offset-corrected ratios and personal deltas; never change legacy model input."""
    x = np.asarray(sequence, dtype=float)
    if x.shape != (TARGET_SAMPLES, 16) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("개인 기준 모델은 3초·12 Hz·16채널의 유효한 입력이 필요합니다.")
    pressure = profile.context.pressure(x)
    total = pressure.sum(axis=1)
    if np.any(total <= 1e-6):
        raise ValueError("빈 좌석 기준보다 충분한 착석 압력이 필요합니다.")
    ratio = pressure / total[:, None]
    center = np.asarray(profile.ratio_center)
    # The floor is part of the feature contract, not a learned normal boundary.
    scale = np.maximum(np.asarray(profile.ratio_mad) * 1.4826, .008)
    mean, spread = ratio.mean(axis=0), ratio.std(axis=0)
    delta = mean - center
    grid = mean.reshape(4, 4)
    base_total = profile.context.pressure(np.asarray(profile.pressure_center)).sum()
    total_ratio = float(np.median(total) / max(base_total, 1e-6))
    spatial = np.r_[grid.sum(axis=0), grid.sum(axis=1),
                    np.dot(grid.sum(axis=0), np.arange(4)), np.dot(grid.sum(axis=1), np.arange(4))]
    summary = np.array([np.abs(delta).sum(), np.sqrt(np.mean((delta / scale) ** 2)),
                        total_ratio, np.std(total) / np.mean(total)])
    gate = np.r_[mean, spread, center, delta, delta / scale, spatial, summary]
    detail = np.r_[mean, spread, delta, spatial, summary]
    return gate, detail


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PersonalPrediction:
    state: str
    posture: str | None
    incorrect_score: float
    detail_score: float | None
    stage2_executed: bool


class PersonalPostureModel:
    def __init__(self, bundle: dict, *, identity: dict | None = None):
        if (bundle.get("feature_version") != FEATURE_VERSION
                or tuple(bundle.get("labels", ())) != tuple(CLASS_NAMES)
                or tuple(bundle.get("sensor_columns", ())) != tuple(SENSOR_COLUMNS)
                or bundle.get("window_seconds") != WINDOW_SECONDS
                or bundle.get("sampling_rate") != SAMPLING_RATE):
            raise ValueError("개인 기준 모델의 특징·센서·자세·윈도우 계약이 호환되지 않습니다.")
        self.gate, self.detail = bundle["gate"], bundle["detail"]
        if list(self.gate.classes_) != [0, 1] or list(self.detail.classes_) != list(range(1, 7)):
            raise ValueError("1단계는 정상/비정상, 2단계는 비정상 6개 자세를 포함해야 합니다.")
        if getattr(self.gate, "n_features_in_", 94) != 94 or getattr(self.detail, "n_features_in_", 62) != 62:
            raise ValueError("개인 기준 모델의 학습 특징 수가 일치하지 않습니다.")
        self.low, self.high = map(float, bundle["gate_thresholds"])
        self.detail_threshold = float(bundle["detail_threshold"])
        if not (0 < self.low < self.high < 1 and 0 <= self.detail_threshold <= 1):
            raise ValueError("개인 기준 모델 임계값이 유효하지 않습니다.")
        self.identity = identity or {}
        self.stage2_calls = 0

    @staticmethod
    def _scores(model, feature, size):
        scores = np.asarray(model.predict_proba(feature.reshape(1, -1)), dtype=float)
        if (scores.shape != (1, size) or not np.isfinite(scores).all()
                or (scores < 0).any() or not np.allclose(scores.sum(axis=1), 1)):
            raise ValueError("모델 출력 점수가 유효하지 않습니다.")
        return scores[0]

    def predict(self, sequence: np.ndarray, profile: CalibrationProfile) -> PersonalPrediction:
        gate, detail = features(sequence, profile)
        score = float(self._scores(self.gate, gate, 2)[1])
        # The gray region is explicit. A stale state is never accepted as a new
        # correct observation; temporal confirmation belongs to the live engine.
        if score <= self.low:
            return PersonalPrediction("correct", CLASS_NAMES[0], score, None, False)
        if score < self.high:
            return PersonalPrediction("uncertain", None, score, None, False)
        self.stage2_calls += 1
        scores = self._scores(self.detail, detail, 6)
        index = int(np.argmax(scores))
        confidence = float(scores[index])
        label = CLASS_NAMES[int(self.detail.classes_[index])] if confidence >= self.detail_threshold else None
        return PersonalPrediction("incorrect", label, score, confidence, True)

    def snapshot(self, session_id, input_source="serial"):
        from datetime import datetime, timezone
        from .model_runtime import SessionModelSnapshot
        stamp = datetime.now(timezone.utc).isoformat()
        token = hashlib.sha256(f"{session_id}:{self.identity['model_sha256']}:{stamp}".encode()).hexdigest()[:24]
        return SessionModelSnapshot(session_id, token, "personal-" + self.identity["model_sha256"][:16],
            "개인 기준 2단계 자세 판별", "1", "sklearn", "personalized_hierarchy",
            self.identity["model_sha256"], self.identity["manifest_sha256"], WINDOW_SECONDS, TARGET_SAMPLES,
            SAMPLING_RATE, tuple(SENSOR_COLUMNS), tuple(CLASS_NAMES), "uncalibrated_score", "not_calibrated", input_source, stamp)

    @classmethod
    def load(cls, manifest_path: Path) -> "PersonalPostureModel":
        try:
            return cls._load_checked(manifest_path)
        except (OSError, ValueError):
            raise
        except Exception as error:
            raise ValueError(f"개인 기준 모델을 읽을 수 없습니다: {error}") from error

    @classmethod
    def _load_checked(cls, manifest_path: Path) -> "PersonalPostureModel":
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1 or manifest.get("kind") != "personalized_hierarchy"
                or manifest.get("status") != "experimental_validated"
                or (path.parent / ".incomplete").exists()):
            raise ValueError("완료된 실험용 개인 기준 모델이 아닙니다.")
        files = {}
        for role in ("model", "report"):
            part = manifest[role]
            candidate = (path.parent / part["file"]).resolve()
            if candidate.parent != path.parent or sha256(candidate) != part["sha256"]:
                raise ValueError("개인 기준 모델 또는 평가 근거의 해시가 일치하지 않습니다.")
            files[role] = candidate
        report = json.loads(files["report"].read_text(encoding="utf-8"))
        if report.get("final_evaluation_used_for_selection") is not False:
            raise ValueError("독립 평가 계약이 없습니다.")
        partitions = [set(report["partitions"][key]) for key in ("train", "validation", "final_evaluation")]
        if any(not group for group in partitions) or any(partitions[i] & partitions[j] for i in range(3) for j in range(i)):
            raise ValueError("학습·검증·최종 평가 피험자가 겹칩니다.")
        subjects = set.union(*partitions)
        if {row["subject"] for row in report["source_records"]} != subjects:
            raise ValueError("학습 원본의 피험자와 고정된 분할이 일치하지 않습니다.")
        model = cls(joblib.load(files["model"]), identity={"manifest": str(path), "manifest_sha256": sha256(path),
                    "model_sha256": manifest["model"]["sha256"], "feature_version": FEATURE_VERSION,
                    "window_seconds": WINDOW_SECONDS, "confidence_semantics": "uncalibrated_score",
                    "evidence_kind": report["evidence_kind"]})
        return model


def available_models(*roots: Path) -> list[Path]:
    result = []
    for root in dict.fromkeys(Path(root).resolve() for root in roots):
        for path in sorted((root / "artifacts" / "personalized").glob("*/personalized.manifest.json"), reverse=True):
            try:
                PersonalPostureModel.load(path)
            except Exception:
                continue
            result.append(path)
    return result
