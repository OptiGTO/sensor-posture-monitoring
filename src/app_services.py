"""Services used by the no-terminal posture monitoring application.

The services deliberately keep raw measurement CSVs independent from inference
logs: an operator-selected label is the only label ever written to a CSV.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import shutil
import subprocess
import threading
import queue
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

from .posture_core import CLASS_NAMES, SENSOR_COLUMNS, make_subject_pairs, read_sensor_csv, recording_label, normalize_text, standardize_label, sensor_identity
from .training_windows import QUICK_WINDOW_SECONDS, validate_training_windows
from .window_pipeline import WindowSpec, window_start_times
from .model_catalog import DEFAULT_MODEL_FAMILIES, MODEL_NAMES, missing_model_packages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_COLUMNS = ("elapsed_s", "label", *SENSOR_COLUMNS)


@dataclass(frozen=True)
class OccupancyConfig:
    preset: str = "보통"
    enter_score: float = 4.0
    exit_score: float = 2.0
    enter_hold_seconds: float = 0.75
    exit_hold_seconds: float = 0.75
    smoothing_frames: int = 5

    def __post_init__(self) -> None:
        if self.enter_score <= self.exit_score or min(self.enter_hold_seconds, self.exit_hold_seconds) < 0 or self.smoothing_frames < 1:
            raise ValueError("착석 임계값은 복귀 임계값보다 커야 하며 유지 시간과 평활화 창은 유효해야 합니다.")


OCCUPANCY_PRESETS = {
    "민감": OccupancyConfig("민감", 2.8, 1.4, 0.4, 0.5, 3),
    "보통": OccupancyConfig(),
    "보수적": OccupancyConfig("보수적", 5.5, 3.0, 1.2, 1.0, 7),
}


def quality_check(path: Path, window_seconds: float = 3.0) -> dict[str, object]:
    """Assess recording integrity only; prediction correctness never affects QC."""
    try:
        frame = read_sensor_csv(path)
    except Exception as error:
        return {"status": "fail", "issues": [f"CSV 검사 실패: {error}"], "window_count": 0}
    return _frame_quality(frame, window_seconds)


def _frame_quality(frame, window_seconds: float = 3.0) -> dict[str, object]:
    spec = WindowSpec(window_seconds)
    issues: list[str] = []
    elapsed = frame["elapsed_s"].to_numpy(dtype=float)
    values = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
    duration = float(elapsed[-1] - elapsed[0]) if len(frame) > 1 else 0.0
    intervals = np.diff(elapsed)
    windows = len(window_start_times(duration, spec.window_seconds, spec.min_tail_fraction))
    if not windows: return {"status": "fail", "issues": ["모델 창을 만들 수 있을 만큼 기록되지 않았습니다."], "row_count": len(frame), "duration_s": duration, "window_count": 0}
    if np.any(intervals > 1.0): issues.append("긴 센서 공백이 있습니다.")
    if len(intervals) and np.std(intervals) > max(np.median(intervals) * 0.4, 0.02): issues.append("샘플링 간격 흔들림이 큽니다.")
    if np.any(np.ptp(values, axis=0) < 1e-6): issues.append("변화가 없는 센서가 있습니다.")
    return {"status": "warning" if issues else "pass", "issues": issues, "row_count": len(frame), "duration_s": duration, "window_count": windows, "occupied_ratio": None}


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


@dataclass(frozen=True)
class ModelDescriptor:
    path: Path
    name: str
    version: str
    window_seconds: float
    sampling_rate: float
    labels: tuple[str, ...]
    created_at: str
    metadata: dict[str, object]

    @property
    def window_label(self) -> str:
        return f"{self.window_seconds:g}초"

    @property
    def input_label(self) -> str:
        required = set(map(str, self.metadata.get("required_sensors", ["pressure"])))
        if required == {"accelerometer"}:
            return "가속도계"
        if "accelerometer" in required:
            return "압력 센서 + 가속도계"
        return "압력 센서"

    @property
    def requires_accelerometer(self) -> bool:
        return "가속도계" in self.input_label


class _LegacyModelRegistry:
    """Discover bundled and user-trained pressure-only models at application start."""

    def __init__(self, root: Path = PROJECT_ROOT, state_root: Path | None = None) -> None:
        self.root = root
        self.state_root = state_root or root
        self.active_path = self.state_root / "artifacts" / "active_model.json"

    def discover(self) -> list[ModelDescriptor]:
        candidates = [
            self.root / "artifacts" / "posture_classifier_window_3s.joblib",
            self.root / "artifacts" / "posture_classifier_window_5s.joblib",
            self.root / "artifacts" / "posture_classifier.joblib",
        ]
        candidates.extend((self.root / "artifacts" / "models").glob("**/*.joblib"))
        if self.state_root != self.root:
            candidates.extend((self.state_root / "artifacts" / "models").glob("**/*.joblib"))
        models: list[ModelDescriptor] = []
        seen: set[Path] = set()
        for path in candidates:
            path = path.resolve()
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            bundle = joblib.load(path)
            sampling_rate = float(bundle.get("sampling_rate", 12.0))
            target_length = int(bundle.get("target_length", 240))
            window = float(bundle.get("window_seconds", target_length / sampling_rate))
            models.append(ModelDescriptor(
                path=path,
                name=str(bundle.get("selected_model", "RBF SVM")) + " 자세 분류 모델",
                version=str(bundle.get("version", "1")),
                window_seconds=window,
                sampling_rate=sampling_rate,
                labels=tuple(map(str, bundle["label_names"])),
                created_at=datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                metadata={key: value for key, value in bundle.items() if key not in {"svm_model", "extra_trees_model"}},
            ))
        return sorted(models, key=lambda item: item.window_seconds)

    def active_model(self, models: list[ModelDescriptor]) -> ModelDescriptor | None:
        if not models:
            return None
        if self.active_path.is_file():
            try:
                selected = Path(json.loads(self.active_path.read_text(encoding="utf-8"))["model_path"]).resolve()
                return next((model for model in models if model.path == selected), models[0])
            except (OSError, KeyError, json.JSONDecodeError):
                pass
        return models[0]

    def activate(self, model: ModelDescriptor) -> None:
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = joblib.load(model.path)
        if tuple(bundle.get("label_names", ())) != model.labels or not bundle.get("feature_names"):
            raise ValueError("모델 메타데이터 검증에 실패했습니다.")
        record = {"model_path": str(model.path), "model_sha256": _digest(model.path), "previous_active_model": self.active_path.exists() and json.loads(self.active_path.read_text(encoding="utf-8")).get("model_path"), "activated_at": datetime.now().isoformat()}
        temporary = self.active_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.active_path)


# New application code uses manifest-first runtime objects.  The legacy
# implementation remains above solely for compatibility with older bundles.
from .model_runtime import ModelDescriptor, ModelRegistry  # noqa: E402


@dataclass(frozen=True)
class StandardProfile:
    center: np.ndarray
    deviation: np.ndarray
    signature: str
    file_count: int


def build_standard_profile(standard_root: Path) -> StandardProfile | None:
    """Build a per-recording-balanced empty-seat profile, without hard-coded counts."""
    files = sorted(path for path in standard_root.rglob("*.csv") if not path.name.startswith("._")) if standard_root.exists() else []
    if not files:
        return None
    summaries: list[np.ndarray] = []
    noise: list[np.ndarray] = []
    signatures: list[str] = []
    for path in files:
        frame = read_sensor_csv(path)
        values = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
        median = np.median(values, axis=0)
        summaries.append(median)
        noise.append(np.median(np.abs(values - median), axis=0) * 1.4826)
        signatures.append(f"{path.relative_to(standard_root)}:{path.stat().st_mtime_ns}:{_digest(path)}")
    records = np.stack(summaries)
    center = np.median(records, axis=0)
    # Preserve the sensor's units: a fixed floor of 1 suppresses occupancy for
    # voltage streams (roughly 1–3 V). Account for both recording-to-recording
    # drift and within-recording noise, with a 1% relative noise floor so tiny
    # baseline fluctuations do not become evidence of occupancy.
    scale = np.maximum(np.abs(center), np.median(np.abs(center)))
    relative_floor = np.maximum(scale * 0.01, np.finfo(float).eps)
    deviation = np.maximum.reduce([
        np.median(np.abs(records - center), axis=0) * 1.4826,
        np.median(np.stack(noise), axis=0),
        relative_floor,
    ])
    return StandardProfile(center, deviation, hashlib.sha256("\n".join(signatures).encode()).hexdigest(), len(files))


def occupancy_profile_for_input(
    input_source: str,
    standard_root: Path,
) -> StandardProfile | None:
    """Return an empty-seat profile only for a live sensor session.

    A saved CSV is an already-recorded model input.  Re-applying a profile
    collected from another hardware calibration can classify every frame as an
    empty seat before inference gets a chance to run.
    """
    if input_source == "csv_replay":
        return None
    if input_source == "serial":
        return build_standard_profile(standard_root)
    raise ValueError(f"지원하지 않는 센서 입력 방식입니다: {input_source}")


@dataclass
class OccupancyDetector:
    profile: StandardProfile | None
    config: OccupancyConfig = field(default_factory=OccupancyConfig)
    occupied: bool = False
    _candidate: bool | None = field(default=None, init=False)
    _candidate_since: float | None = field(default=None, init=False)
    _frames: deque = field(default_factory=deque, init=False)

    def update(self, timestamp_s: float, values: Iterable[float]) -> bool | None:
        """Return a changed occupancy state, or None. No profile means unknown."""
        if self.profile is None:
            return None
        data = np.asarray(tuple(values), dtype=float)
        self._frames.append(data)
        while len(self._frames) > self.config.smoothing_frames:
            self._frames.popleft()
        data = np.mean(self._frames, axis=0)
        z = np.abs(data - self.profile.center) / self.profile.deviation
        score = float(np.median(z) + np.mean(z >= 3.0) + np.abs(data - self.profile.center).sum() / np.maximum(self.profile.deviation.sum(), 1e-12))
        desired = score >= (self.config.exit_score if self.occupied else self.config.enter_score)
        if desired == self.occupied:
            self._candidate = None
            self._candidate_since = None
            return None
        if desired != self._candidate:
            self._candidate, self._candidate_since = desired, timestamp_s
            return None
        hold = self.config.enter_hold_seconds if desired else self.config.exit_hold_seconds
        if self._candidate_since is not None and timestamp_s - self._candidate_since >= hold:
            self.occupied = desired
            self._candidate = None
            self._candidate_since = None
            return self.occupied
        return None


@dataclass
class RiskAssessment:
    level: str = "정상"
    message: str = ""
    last_alert_s: float = -math.inf
    poor_since_s: float | None = None

    def update(self, timestamp_s: float, posture: str | None, confidence: float | None) -> "RiskAssessment":
        poor = posture not in {None, "정자세"} and (confidence or 0.0) >= 0.45
        if poor and self.poor_since_s is None:
            self.poor_since_s = timestamp_s
        if not poor:
            self.poor_since_s = None
            self.level, self.message = "정상", ""
            return self
        duration = timestamp_s - self.poor_since_s
        self.level = "주의" if duration < 120 else "위험"
        if duration >= 60 and timestamp_s - self.last_alert_s >= 120:
            self.message = f"{posture} 자세가 {int(duration)}초간 지속되고 있습니다. 자세를 조정하세요."
            self.last_alert_s = timestamp_s
        return self

    def reset(self) -> None:
        self.level, self.message, self.poor_since_s = "정상", "", None


class MeasurementRecorder:
    """Write compatible CSV and separate prediction/manifest sidecars for one session."""

    def __init__(
        self,
        root: Path,
        *,
        mode: str,
        subject: str = "",
        label: str = "",
        model_snapshot: object | None = None,
        data_directory: Path | None = None,
        guidance_metadata: dict[str, object] | None = None,
    ) -> None:
        if mode not in {"labeled", "measurement"}:
            raise ValueError("지원하지 않는 측정 모드입니다.")
        subject = normalize_text(subject.strip()) if mode == "labeled" else ""
        label = standardize_label(label) if mode == "labeled" else ""
        self.mode, self.subject, self.label = mode, subject, label
        self.model_snapshot = model_snapshot
        self.guidance_metadata = guidance_metadata
        self.data_directory = data_directory or root / "data"
        now = datetime.now()
        if mode == "labeled":
            safe_subject = subject.strip()
            if not safe_subject or not label:
                raise ValueError("라벨 수집에는 측정 대상 이름과 실제 자세가 필요합니다.")
            if label not in CLASS_NAMES:
                raise ValueError("지원하는 실제 자세를 선택하세요.")
            if safe_subject.casefold() in {"standard", "measurements", "__macosx"}:
                raise ValueError("시스템 데이터 폴더 이름은 측정 대상 이름으로 사용할 수 없습니다.")
            for name in (safe_subject, label):
                if name in {".", ".."} or any(char in name for char in ("/", "\\", "\0")):
                    raise ValueError("이름과 자세에는 경로 구분자를 사용할 수 없습니다.")
            directory = self.data_directory / safe_subject
            prefix = f"{safe_subject}_{label}"
            number = 1
            while (directory / f"{prefix}_{number:03d}.csv").exists():
                number += 1
            self.csv_path = directory / f"{prefix}_{number:03d}.csv"
        else:
            directory = self.data_directory / "measurements" / now.strftime("%Y-%m-%d")
            self.csv_path = directory / f"measurement_{now.strftime('%H%M%S')}.csv"
        directory.mkdir(parents=True, exist_ok=True)
        self.prediction_path = self.csv_path.with_suffix(".predictions.jsonl")
        self.imu_path = self.csv_path.with_suffix(".imu.csv")
        self.manifest_path = self.csv_path.with_suffix(".manifest.json")
        # Reserve a new file atomically; starting twice in one second must not truncate data.
        base_path = self.csv_path
        suffix = 0
        while True:
            try:
                if any(self.csv_path.with_suffix(suffix).exists() for suffix in (".predictions.jsonl", ".guidance.jsonl", ".manifest.json", ".imu.csv")):
                    raise FileExistsError(self.csv_path)
                self._csv = self.csv_path.open("x", newline="", encoding="utf-8-sig")
                break
            except FileExistsError:
                suffix += 1
                self.csv_path = base_path.with_stem(f"{base_path.stem}_{suffix:03d}")
        self.prediction_path = self.csv_path.with_suffix(".predictions.jsonl")
        self.imu_path = self.csv_path.with_suffix(".imu.csv")
        self.manifest_path = self.csv_path.with_suffix(".manifest.json")
        self._writer = csv.writer(self._csv)
        self._writer.writerow(CSV_COLUMNS)
        try:
            self._prediction = self.prediction_path.open("x", encoding="utf-8")
        except OSError:
            self._csv.close()
            raise
        self._imu = None
        self.guidance_path = self.csv_path.with_suffix(".guidance.jsonl")
        self._guidance = None
        self.started = time.monotonic()
        self.rows = 0
        self._last_elapsed = -math.inf
        self._first_elapsed: float | None = None
        self._closed = False

    def write_frame(self, timestamp_s: float, values: Iterable[float]) -> None:
        values = tuple(float(value) for value in values)
        if len(values) != len(SENSOR_COLUMNS) or not np.isfinite(values).all():
            raise ValueError("16개의 유한한 센서값이 필요합니다.")
        timestamp_s = float(f"{timestamp_s:.6f}")
        if not math.isfinite(timestamp_s) or timestamp_s < 0 or timestamp_s <= self._last_elapsed:
            raise ValueError("elapsed_s는 계속 증가해야 합니다.")
        self._writer.writerow([f"{timestamp_s:.6f}", self.label if self.mode == "labeled" else "", *values])
        self._csv.flush()
        self.rows += 1
        if self._first_elapsed is None:
            self._first_elapsed = timestamp_s
        self._last_elapsed = timestamp_s

    def write_prediction(self, record: dict[str, object]) -> None:
        self._prediction.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._prediction.flush()

    def write_guidance(self, record: dict[str, object]) -> None:
        if self._guidance is None:
            self._guidance = self.guidance_path.open("x", encoding="utf-8")
        self._guidance.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._guidance.flush()

    def write_imu(self, timestamp_s: float, values: tuple[float, float, float]) -> None:
        """Store real accelerometer observations separately; missing data is never synthesized."""
        if len(values) != 3 or not np.isfinite(values).all():
            raise ValueError("가속도계 값은 유한한 ax, ay, az 세 값이어야 합니다.")
        if self._imu is None:
            self._imu = self.imu_path.open("x", newline="", encoding="utf-8-sig")
            csv.writer(self._imu).writerow(["elapsed_s", "ax", "ay", "az"])
        csv.writer(self._imu).writerow([f"{timestamp_s:.6f}", *values])
        self._imu.flush()

    def close(self, status: str = "저장만", *, error: str | None = None, completion_reason: str = "completed") -> Path:
        if self._closed:
            return self.manifest_path
        self._csv.close(); self._prediction.close()
        if self._guidance is not None: self._guidance.close()
        if self._imu is not None: self._imu.close()
        qc = quality_check(self.csv_path)
        payload = {"mode": self.mode, "subject": self.subject, "ground_truth_label": self.label, "csv": self.csv_path.name, "predictions": self.prediction_path.name, "imu": self.imu_path.name if self.imu_path.exists() else None, "rows": self.rows, "duration_s": round(time.monotonic() - self.started, 3), "qc": qc, "dataset_status": status, "source_type": "collected" if self.mode == "labeled" else "measurement", "label_status": "labeled" if self.mode == "labeled" else "unlabeled", "training_role": "exclude", "created_at": datetime.now().isoformat()}
        if self.model_snapshot is not None:
            payload["model_snapshot"] = self.model_snapshot.as_dict() if hasattr(self.model_snapshot, "as_dict") else self.model_snapshot
        if self.guidance_metadata is not None:
            payload["guidance"] = self.guidance_metadata
            payload["guidance_log"] = self.guidance_path.name if self.guidance_path.exists() else None
        payload["duration_s"] = round(self._last_elapsed - self._first_elapsed, 6) if self._first_elapsed is not None else 0.0
        payload["session_status"] = "failed" if error else completion_reason
        payload["session_error"] = error
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._closed = True
        return self.manifest_path


@dataclass(frozen=True)
class SensorPacket:
    """Common future-facing packet; pressure-only hardware keeps imu as None."""

    timestamp_s: float
    pressure: tuple[float, ...]
    imu: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if len(self.pressure) != len(SENSOR_COLUMNS) or not np.isfinite(self.pressure).all():
            raise ValueError("압력 센서값은 유한한 16개 값이어야 합니다.")
        if self.imu is not None and (len(self.imu) != 3 or not np.isfinite(self.imu).all()):
            raise ValueError("가속도 값은 누락 없이 ax, ay, az 세 값이어야 합니다.")


@dataclass(frozen=True)
class DatasetRecord:
    csv_path: Path
    manifest_path: Path | None
    subject: str
    posture: str
    mode: str
    status: str
    rows: int
    duration_s: float
    quality: str
    created_at: str
    source_type: str = "baseline"
    label_status: str = "labeled"
    training_role: str = "include"
    qc_status: str = "not_checked"


@dataclass
class TrainingWizardState:
    """Persist the supported first-stage training choices across wizard pages."""

    current_step: int = 1
    backend: str = "sklearn"
    preset: str = "rbf_svm"
    window_policy: str = "fixed"
    windows: tuple[int, ...] = (3,)
    models: tuple[str, ...] = DEFAULT_MODEL_FAMILIES
    training_method: str = "standard"
    configuration_type: str = "individual"
    experiment_name: str = "새 모델 실험"
    source_experiment: str | None = None
    voter_selection_initialized: bool = False
    soft_voting_enabled: bool = False
    soft_voting_members: tuple[str, ...] = ()
    selection_metric: str = "loso"
    seed: int = 20260826
    selected_record_paths: tuple[Path, ...] = ()

    QUICK_WINDOWS = QUICK_WINDOW_SECONDS
    # Backward-compatible name for callers that build the quick-choice UI.
    SUPPORTED_WINDOWS = QUICK_WINDOWS
    SUPPORTED_MODELS = tuple(MODEL_NAMES)

    def configure(self, windows: Iterable[int], seed: int, models: Iterable[str] | None = None) -> None:
        selected = validate_training_windows(windows)
        if isinstance(seed, bool):
            raise ValueError("난수 시드는 정수여야 합니다.")
        if models is not None:
            from .model_comparison import validate_model_families
            self.models = validate_model_families(models, training_method=self.training_method)
        self.windows = selected
        self.seed = int(seed)

    def remember_records(self, records: Iterable[DatasetRecord]) -> None:
        self.selected_record_paths = tuple(record.csv_path.resolve() for record in records)


class DatasetService:
    """Index recordings without placing measurement-only or standard files in training."""

    def __init__(self, root: Path = PROJECT_ROOT, *, data_directory: Path | None = None,
                 additional_data_directories: Iterable[Path] = ()) -> None:
        self.root = root
        self.data_directory = data_directory or root / "data"
        directories: dict[Path, Path] = {}
        for path in (self.data_directory, *additional_data_directories):
            directories.setdefault(path.resolve(), path)
        self.data_directories = tuple(directories.values())
        self._record_cache: dict[Path, tuple[object, DatasetRecord]] = {}

    @staticmethod
    def _file_signature(path: Path) -> tuple[int, int, int] | None:
        try:
            stat = path.stat()
            return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        except FileNotFoundError:
            return None

    def records(self) -> list[DatasetRecord]:
        exclusions = self._training_exclusion_policies()
        indexed: list[DatasetRecord] = []
        seen: set[Path] = set()
        for directory, csv_path in ((directory, csv_path) for directory in self.data_directories
                                    for csv_path in sorted(directory.glob("**/*.csv"))):
            relative = csv_path.relative_to(directory)
            if {"standard", "__macosx"} & {part.casefold() for part in relative.parts[:-1]} or csv_path.name.startswith("._") or csv_path.name.endswith(".imu.csv"):
                continue
            canonical = csv_path.resolve()
            if canonical in seen:
                continue
            seen.add(canonical)
            manifest = csv_path.with_suffix(".manifest.json")
            signature = (self._file_signature(csv_path), self._file_signature(manifest))
            cached = self._record_cache.get(csv_path)
            if cached is not None and cached[0] == signature:
                indexed.append(self._apply_training_exclusion(cached[1], exclusions))
                continue
            data = self._manifest(manifest)
            default_status = "측정 전용" if relative.parts[0].casefold() == "measurements" else "기본 학습 데이터"
            record = self._from_paths(csv_path, manifest if manifest.exists() else None, data, default_status)
            self._record_cache[csv_path] = signature, record
            indexed.append(self._apply_training_exclusion(record, exclusions))
        current_paths = {record.csv_path for record in indexed}
        self._record_cache = {path: item for path, item in self._record_cache.items() if path in current_paths}
        return indexed

    def _training_exclusion_policies(self, records=()):
        """Read hash-bound exclusion lists without editing source sidecars."""
        from .training_exclusions import load_training_exclusions
        directories = {path.resolve() for path in self.data_directories}
        # Jobs can receive sources from both packaged state and this checkout.
        # The known data ancestor preserves the original manifest's path base.
        for record in records:
            directories.update(parent for parent in record.csv_path.resolve().parents if parent.name == 'data')
        return {directory: load_training_exclusions(directory) for directory in directories}

    @staticmethod
    def _apply_training_exclusion(record, policies):
        from .training_exclusions import is_training_excluded
        for directory, exclusions in policies.items():
            if exclusions and record.csv_path.resolve().is_relative_to(directory):
                if is_training_excluded(record.csv_path, directory, exclusions):
                    return replace(record, training_role='exclude', status='학습 제외 목록')
        return record

    def _manifest(self, path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {"training_role": "exclude"}
        except (OSError, json.JSONDecodeError):
            return {"training_role": "exclude"}

    def _from_paths(self, csv_path: Path, manifest: Path | None, data: dict[str, object], default_status: str) -> DatasetRecord:
        rows, duration, posture = 0, 0.0, ""
        label_valid = False
        try:
            frame = read_sensor_csv(csv_path)
            rows = len(frame)
            duration = float(frame["elapsed_s"].iloc[-1] - frame["elapsed_s"].iloc[0])
            qc = _frame_quality(frame)
            try:
                _, posture = recording_label(frame)
                label_valid = True
            except ValueError:
                pass
        except (OSError, ValueError) as error:
            qc = {"status": "fail", "issues": [str(error)]}
        measurement_only = (data.get("mode") == "measurement" or data.get("source_type") == "measurement"
                            or "measurements" in {part.casefold() for part in csv_path.parts[:-1]})
        source = "measurement" if measurement_only else str(data.get("source_type", "collected" if manifest else "baseline"))
        label_status = str(data.get("label_status", "unlabeled" if source == "measurement" else "labeled"))
        if not label_valid:
            label_status = "unlabeled"
        role = str(data.get("training_role", "exclude" if source == "measurement" else "include"))
        if measurement_only:
            label_status, role = "unlabeled", "exclude"
        qc_status = str(qc.get("status", "not_checked"))
        quality_text = {"pass": "통과", "warning": "경고", "fail": "실패", "not_checked": "미검사"}.get(qc_status, "미검사")
        return DatasetRecord(csv_path, manifest, str(data.get("subject", csv_path.parent.name)), posture, str(data.get("mode", "measurement" if source == "measurement" else "labeled")), str(data.get("dataset_status", default_status)), rows, duration, quality_text, str(data.get("created_at", datetime.fromtimestamp(csv_path.stat().st_mtime).isoformat())), source, label_status, role, qc_status)

    def set_status(self, record: DatasetRecord, status: str) -> None:
        if record.manifest_path is None:
            manifest = record.csv_path.with_suffix(".manifest.json")
            data: dict[str, object] = {
                "csv": record.csv_path.name,
                "mode": record.mode,
                "subject": record.subject,
                "ground_truth_label": record.posture,
                "rows": record.rows,
                "duration_s": record.duration_s,
                "qc": quality_check(record.csv_path),
                "source_type": record.source_type,
                "label_status": record.label_status,
                "created_at": record.created_at,
            }
        else:
            manifest = record.manifest_path; data = self._manifest(manifest)
        if not isinstance(data.get("qc"), dict) or data["qc"].get("status") not in {"pass", "warning", "fail"}:
            data["qc"] = quality_check(record.csv_path)
        roles = {"학습 대기열에 추가": "include", "학습에 포함": "include", "학습에서 제외": "exclude", "평가 전용": "evaluation_only", "평가 전용으로 보관": "evaluation_only", "저장만": "exclude", "측정 전용": "exclude", "폐기": "exclude"}
        data["dataset_status"] = status; data["training_role"] = roles.get(status, data.get("training_role", "exclude"))
        if record.source_type == "measurement" or record.mode == "measurement":
            data.update(mode="measurement", source_type="measurement", label_status="unlabeled", training_role="exclude")
        manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def is_training_eligible(record: DatasetRecord) -> bool:
        return (
            record.source_type != "measurement"
            and record.label_status == "labeled"
            and record.training_role == "include"
            and record.qc_status == "pass"
        )

    def stage_for_training(self, selected: Iterable[DatasetRecord], target: Path) -> Path:
        selected = tuple(selected)
        exclusions = self._training_exclusion_policies(selected)
        if target.exists() and any(target.iterdir()):
            raise ValueError("학습용 임시 폴더가 비어 있지 않습니다. 새 실행 폴더가 필요합니다.")
        target.mkdir(parents=True, exist_ok=True)
        copied = 0
        staged_sources = []
        content_sources = {}
        skipped_duplicates = []
        for record in selected:
            manifest = record.csv_path.with_suffix(".manifest.json")
            record = self._from_paths(record.csv_path, manifest if manifest.exists() else None, self._manifest(manifest), record.status)
            record = self._apply_training_exclusion(record, exclusions)
            if not self.is_training_eligible(record):
                raise ValueError(f"학습에 사용할 수 없는 기록입니다: {record.csv_path.name}")
            # macOS filenames may use decomposed Hangul, while sidecars use NFC.
            # Keep one subject identity across all selected data directories.
            record = replace(record, subject=normalize_text(record.subject))
            content_hash = sensor_identity(read_sensor_csv(record.csv_path))
            previous = content_sources.get(content_hash)
            if previous is not None:
                if (previous["subject"], previous["label"]) != (record.subject, record.posture):
                    raise ValueError("동일한 센서 기록의 피험자 또는 자세 라벨이 다릅니다: "
                                     f"{previous['source_file']} / {record.csv_path}")
                skipped_duplicates.append({"source_file": str(record.csv_path.resolve()),
                    "sha256": _digest(record.csv_path), "sensor_content_sha256": content_hash,
                    "kept_source_file": previous["source_file"],
                    "staged_file": previous["staged_file"],
                    "reason": "same_subject_label_and_sensor_content"})
                continue
            destination = target / record.subject / record.csv_path.name
            if target.resolve() not in destination.resolve().parents:
                raise ValueError("학습 기록의 대상 이름이 유효하지 않습니다.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                source_hash = _digest(record.csv_path)
                if _digest(destination) == source_hash:
                    raise ValueError(f"동일한 학습 기록이 중복됩니다: {destination.name}")
                # Separate sessions can have the same basename in data/<user>
                # and data/etc/<user>. Keep the subject unchanged for splitting.
                destination = destination.with_name(f"{destination.stem}_{source_hash[:12]}{destination.suffix}")
                if destination.exists():
                    raise ValueError(f"학습 기록 파일명이 중복됩니다: {destination.name}")
            shutil.copy2(record.csv_path, destination); copied += 1
            if manifest.is_file():
                metadata = self._manifest(manifest)
                metadata["csv"] = destination.name
                metadata["subject"] = record.subject
                destination.with_suffix(".manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            staged_sources.append({"source_file": str(record.csv_path.resolve()),
                "staged_file": str(destination.relative_to(target)), "subject": record.subject,
                "label": record.posture, "sensor_content_sha256": content_hash,
                "sha256": _digest(destination)})
            content_sources[content_hash] = staged_sources[-1]
        if not copied:
            raise ValueError("학습에 포함할 품질 통과 라벨 데이터가 없습니다.")
        (target / "staged_sources.json").write_text(json.dumps(staged_sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "staging_summary.json").write_text(json.dumps({
            "selected_count": len(selected), "staged_count": copied,
            "skipped_duplicates": skipped_duplicates,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target


class TrainingJob:
    """Own a cancellable child process and persist a reproducible run manifest."""

    def __init__(self, root: Path, records: Iterable[DatasetRecord], seed: int, *, windows: tuple[int, ...] = (3,), models: tuple[str, ...] = DEFAULT_MODEL_FAMILIES, state_root: Path | None = None, frozen: bool = False, personalized: bool = False, allow_retrospective: bool = False, soft_voting_members: tuple[str, ...] = (), experiment_config: dict | None = None, training_method: str = "standard", selection_metric: str = "loso", signal_preprocessing=None) -> None:
        self.experiment_config = experiment_config
        # Optional versioned raw-signal preprocessing (src.signal_preprocessing);
        # None keeps the existing training command exactly.
        self.signal_preprocessing = signal_preprocessing
        self.selection_metric = selection_metric
        self.training_method = training_method
        self.soft_voting_members = tuple(soft_voting_members)
        self.personalized, self.allow_retrospective = personalized, allow_retrospective
        self.root, self.state_root, self.records, self.seed, self.windows, self.frozen = root, state_root or root, tuple(records), seed, windows, frozen
        self.models = tuple(models)
        self.run_dir = self.state_root / "artifacts" / "models" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.run_dir.with_name(f"{self.run_dir.name}_{uuid.uuid4().hex[:8]}")
        if self.personalized:
            self.run_dir = self.state_root / "artifacts" / "personalized" / self.run_dir.name
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.candidate_manifest_paths: tuple[Path, ...] = ()
        self.validation_results: list[dict[str, object]] = []
        self._reader: threading.Thread | None = None
        self._cancel_requested = threading.Event()

    @property
    def comparison_report_path(self) -> Path:
        if self.personalized:
            return self.run_dir / "personalized.report.json"
        return self.run_dir / "artifacts" / "model_comparison.json"

    def comparison_results(self) -> list[dict[str, object]]:
        if not self.comparison_report_path.is_file():
            return []
        return list(json.loads(self.comparison_report_path.read_text(encoding="utf-8")).get("models", []))

    @property
    def is_running(self) -> bool:
        return bool((self.process and self.process.poll() is None) or (self._reader and self._reader.is_alive()))

    def start(self) -> None:
        if self.process is not None:
            raise ValueError("이미 시작한 학습 작업입니다.")
        if self.personalized and self.windows != (3,):
            raise ValueError("개인 기준 2단계 모델은 3초 윈도우를 사용합니다.")
        self.windows = validate_training_windows(self.windows)
        from .model_comparison import validate_model_families
        self.models = validate_model_families(self.models, training_method=self.training_method)
        from .training_methods import validate_training_method
        if self.personalized and self.training_method != "standard":
            raise ValueError("개인 표준화 구성은 기본 학습 방식만 지원합니다.")
        method = validate_training_method(self.training_method, self.models, self.windows)
        from .model_catalog import missing_pipeline_packages, model_display_name
        if not self.personalized:
            for family in self.models:
                missing = (missing_pipeline_packages(method["checkpoint"]["pipeline"])
                           if "checkpoint" in method else missing_model_packages(family))
                if missing:
                    raise ValueError(f"{model_display_name(family)}: {', '.join(missing)} 설치가 필요합니다. requirements.txt를 설치하세요.")
        from .soft_voting import validate_members
        self.soft_voting_members = validate_members(self.soft_voting_members, self.models, self.windows)
        if self.personalized and self.soft_voting_members:
            raise ValueError("Soft voting은 일반 모델 학습에서 실행하세요.")
        if self.experiment_config is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            snapshot = {**self.experiment_config, "training_method": self.training_method,
                "training_method_label": method["label"], "training_method_evidence": method.get("evidence"),
                **({"checkpoint": method["checkpoint"]} if "checkpoint" in method else {}),
                "created_at": datetime.now().astimezone().isoformat(),
                "selected_records": [str(record.csv_path.resolve()) for record in self.records],
                "final_evaluation_used_for_selection": False}
            (self.run_dir / "experiment.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged = DatasetService(self.state_root).stage_for_training(self.records, self.run_dir / "training_data")
        staging_summary = json.loads((staged / "staging_summary.json").read_text(encoding="utf-8"))
        if staging_summary["skipped_duplicates"]:
            self.events.put(("log", f"동일한 측정 기록 {len(staging_summary['skipped_duplicates'])}건은 한 번만 포함했습니다. "
                             f"총 {staging_summary['staged_count']}건으로 학습합니다. 원본은 보존됩니다."))
        subjects = []
        for path in staged.glob("*/*.csv"):
            frame = read_sensor_csv(path)
            qc = _frame_quality(frame, max(self.windows))
            if qc["status"] != "pass":
                raise ValueError(f"{path.name}: 선택한 {max(self.windows)}초 모델에 사용할 수 없습니다. {'; '.join(qc['issues'])}")
            subjects.append(path.parent.name)
        make_subject_pairs(subjects, self.seed)
        command = [str(self.root / ".venv" / "bin" / "python"), str(self.root / "train_window_models.py"), "--deployment", "--data", str(staged), "--output-dir", str(self.run_dir), "--seed", str(self.seed), "--windows", *(str(window) for window in self.windows), "--models", *self.models]
        command.extend(["--training-method", self.training_method])
        if not self.personalized:
            from .model_comparison import validate_selection_metric
            command.extend(["--selection-metric", validate_selection_metric(self.selection_metric)])
        if self.soft_voting_members:
            command.extend(["--soft-voting-members", *self.soft_voting_members])
        if self.signal_preprocessing is not None and not self.signal_preprocessing.is_identity:
            if self.personalized:
                raise ValueError("개인 기준 2단계 학습에는 신호 전처리 옵션을 사용할 수 없습니다.")
            command.extend(["--despike", self.signal_preprocessing.despike or "none",
                            "--training-trim-start-seconds", repr(self.signal_preprocessing.training_trim_start_seconds)])
        if self.personalized:
            command.extend(["--personalized", *(["--allow-retrospective"] if self.allow_retrospective else [])])
            (self.run_dir / ".incomplete").touch()
        if self.frozen:
            command = [str(__import__("sys").executable), "--training-worker", *command[2:]]
        python = Path(command[0]); command[0] = str(python if python.exists() else Path(__import__("sys").executable))
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        environment["NUMBA_NUM_THREADS"] = "1"
        plot_cache = self.state_root / ".cache" / "matplotlib"
        plot_cache.mkdir(parents=True, exist_ok=True)
        environment["MPLCONFIGDIR"] = str(plot_cache)
        environment["MPLBACKEND"] = "Agg"
        numba_cache = self.state_root / ".cache" / "numba"
        numba_cache.mkdir(parents=True, exist_ok=True)
        environment["NUMBA_CACHE_DIR"] = str(numba_cache)
        self.process = subprocess.Popen(command, cwd=self.root, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    @staticmethod
    def pending_candidate_manifests(state_root: Path) -> tuple[Path, ...]:
        from .model_runtime.lifecycle import DeploymentStore
        from .model_runtime.validation import ArtifactValidator

        candidates = []
        artifact_root = Path(state_root) / "artifacts"
        models_root = artifact_root / "models"
        validator = ArtifactValidator((artifact_root,))
        store = DeploymentStore(Path(state_root))
        for path in sorted(models_root.glob("*/artifacts/manifests/*.json"), reverse=True):
            discovery = validator.discover_manifest(path)
            if not discovery.available or discovery.descriptor is None:
                continue
            record = store.get(discovery.descriptor)
            if record is None or (
                record.approval_status == "pending" and record.revocation_status == "active"
            ):
                candidates.append(path)
        return tuple(candidates)

    def _read(self) -> None:
        assert self.process and self.process.stdout
        recent_output: list[str] = []
        for line in self.process.stdout:
            message = line.rstrip()
            recent_output.append(message)
            recent_output = recent_output[-20:]
            self.events.put(("log", message))
        self.process.stdout.close()
        code = self.process.wait()
        packaging_error = None
        if code == 0 and not self._cancel_requested.is_set():
            try:
                if self.personalized:
                    from .personalized_model import PersonalPostureModel
                    path = self.run_dir / "personalized.manifest.json"
                    PersonalPostureModel.load(path)
                    self.candidate_manifest_paths = (path,)
                    self.validation_results = [{"manifest": str(path), "runtime": "passed", "deployment": "experimental", "passed": True}]
                else:
                    self.candidate_manifest_paths = self._write_pending_candidate_manifests()
                    self.validation_results = self._validate_candidates()
            except Exception as error:
                packaging_error = str(error)
        succeeded = code == 0 and packaging_error is None and not self._cancel_requested.is_set()
        if not succeeded and self.personalized:
            (self.run_dir / ".incomplete").touch()
        if not succeeded and (self.run_dir / "artifacts").is_dir():
            (self.run_dir / "artifacts" / ".incomplete").touch()
        if self._cancel_requested.is_set() and self.comparison_report_path.is_file():
            try:
                comparison = json.loads(self.comparison_report_path.read_text(encoding="utf-8"))
                comparison.update(status="cancelled", error="Cancelled by user; outputs were not packaged or activated")
                self.comparison_report_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except (OSError, ValueError) as error:
                self.events.put(("log", f"취소 상태 저장 실패: {error}"))
        manifest = {
            "command": self.process.args,
            "selected_records": [str(record.csv_path) for record in self.records],
            "seed": self.seed,
            "windows": list(self.windows),
            "models": list(self.models),
            "training_method": self.training_method,
            "selection_metric": None if self.personalized else self.selection_metric,
            "soft_voting_members": list(self.soft_voting_members),
            "experiment_config": self.experiment_config,
            "comparison_report": str(self.comparison_report_path) if self.comparison_report_path.is_file() else None,
            "completed_at": datetime.now().isoformat(),
            "exit_code": code,
            "status": ("완료 · 검증 실패 포함" if any(not item["passed"] for item in self.validation_results) else "완료 · 승인 대기") if succeeded else "취소" if self._cancel_requested.is_set() else "실패",
            "approval_status": "pending" if succeeded else "not_created",
            "candidate_manifests": [str(path) for path in self.candidate_manifest_paths],
            "validation_results": self.validation_results,
            "failure_reason": None if succeeded else "Cancelled by user" if self._cancel_requested.is_set() else packaging_error or "\n".join(recent_output),
        }
        if self.personalized and succeeded:
            manifest.update(status="완료 · 실험용 개인 기준 모델", approval_status="experimental_only")
        if packaging_error is not None:
            manifest["packaging_error"] = packaging_error
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "training.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            succeeded = False
            self.events.put(("log", f"학습 결과 기록 실패: {error}"))
        if packaging_error:
            self.events.put(("log", f"후보 모델 저장 실패: {packaging_error}"))
        self.events.put(("finished", "0" if succeeded else "1"))

    def _validate_candidates(self) -> list[dict[str, object]]:
        """Validate only this run's candidates on the background reader thread."""
        from .model_runtime import ModelRegistry

        registry = ModelRegistry(self.root, self.state_root)
        results = []
        for path in self.candidate_manifest_paths:
            if self._cancel_requested.is_set():
                break
            result = {"manifest": str(path), "runtime": "unvalidated",
                      "deployment": "unvalidated", "passed": False, "reason": None}
            try:
                discovery = registry.validator.discover_manifest(path)
                if not discovery.available or discovery.descriptor is None:
                    raise ValueError(discovery.reason or "후보 매니페스트를 읽을 수 없습니다.")
                model = discovery.descriptor
                self.events.put(("status", f"{model.name} · Runtime 자동 검증 중"))
                model = registry.validate_runtime(model)
                result["runtime"] = model.validation_status
                if model.validation_status == "passed" and model.runtime_compatible and model.sensor_compatible:
                    if self._cancel_requested.is_set():
                        result["reason"] = "Cancelled by user"
                        results.append(result)
                        break
                    self.events.put(("status", f"{model.name} · 배포 자동 검증 중"))
                    model = registry.validate_deployment(model)
                    result["deployment"] = model.deployment_validation_status
                result["passed"] = result["runtime"] == result["deployment"] == "passed"
                result["reason"] = model.lifecycle_reason
            except Exception as error:
                # One broken candidate must not prevent checking the remaining models.
                result["reason"] = str(error)
            results.append(result)
            self.events.put(("log", f"{path.name}: Runtime={result['runtime']}, 배포={result['deployment']} · "
                             + ("승인 대기" if result["passed"] else str(result["reason"] or "검증 미통과"))))
        return results

    def _write_pending_candidate_manifests(self) -> tuple[Path, ...]:
        """Package successful outputs without making them production-selectable."""

        artifacts_dir = self.run_dir / "artifacts"
        manifests_dir = artifacts_dir / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        incomplete = artifacts_dir / ".incomplete"
        incomplete.touch()
        golden_source = self.root / "example" / "sample_input.csv"
        golden_target = artifacts_dir / "golden" / "sample_input.csv"
        if golden_source.is_file():
            golden_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(golden_source, golden_target)
        created_at = datetime.now().isoformat()
        created: list[Path] = []
        config_paths = sorted(artifacts_dir.glob("window_*_model_config.json"))
        if not config_paths:
            raise ValueError("학습 결과 설정 파일이 없습니다.")
        for config_path in config_paths:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            # Legacy single-SVM outputs retain their previous filenames.
            seconds = int(config.get("window_seconds", config_path.name.split("_")[1][:-1]))
            artifact = artifacts_dir / str(config.get("model_file", f"posture_classifier_window_{seconds}s.joblib"))
            if not artifact.is_file() or not config_path.is_file():
                raise ValueError(f"{seconds}초 학습 결과 파일이 완전하지 않습니다.")
            family = str(config.get("model_family", "rbf_svm"))
            name = str(config.get("selected_model", "RBF SVM"))
            evaluation_path = artifacts_dir / str(config.get("evaluation_file", f"window_{seconds}s_evaluation.json"))
            has_evaluation = config.get("evaluation_file") == evaluation_path.name and evaluation_path.is_file()
            labels = list(map(str, config["label_names"]))
            channels = list(map(str, config["sensor_columns"]))
            summary = dict(config.get("fixed_configuration_group_cv_summary", {}))
            manifest = {
                "schema_version": 2,
                "artifact_id": f"posture-classifier-{family.replace('_', '-')}-{seconds}s-{self.run_dir.name}",
                "display_name": f"{name} 자세 분류 후보 ({seconds}초)",
                "version": str(config.get("version", "1")),
                "backend": "sklearn",
                "model_family": family,
                "artifact_file": f"../{artifact.name}",
                "artifact_sha256": _digest(artifact),
                "components": [
                    {
                        "role": "training_config",
                        "file": f"../{config_path.name}",
                        "sha256": _digest(config_path),
                        "required": True,
                    },
                    *([{
                        "role": "golden_replay_input",
                        "file": "../golden/sample_input.csv",
                        "sha256": _digest(golden_target),
                        "required": True,
                    }] if golden_target.is_file() else []),
                ],
                "labels": labels,
                "input_channels": channels,
                "sampling_rate": float(config["sampling_rate"]),
                "window": {
                    "policy": "fixed",
                    "values_seconds": [seconds],
                    "target_sample_rule": "round(window_seconds * sampling_rate)",
                    "padding_policy": "interpolate_time_axis",
                },
                "preprocessing": {
                    "kind": config.get("input_representation", "feature_vector"),
                    "feature_count": len(config.get("feature_names", ())),
                    "channel_count": len(channels),
                    "resampling": "linear_interpolation_on_elapsed_s",
                    **({"signal_preprocessing": config["signal_preprocessing"]}
                       if config.get("signal_preprocessing") is not None else {}),
                },
                "output_type": "probabilities",
                "confidence_semantics": str(config.get("confidence_semantics", "score")),
                "calibration_status": "pending_validation",
                "created_at": created_at,
                "provenance": {
                    "source": "training_run",
                    "created_by": "Training Wizard",
                    "training_run": self.run_dir.name,
                },
                "runtime": {
                    "python": ">=3.10",
                    "packages": {
                        "joblib": ">=1.3",
                        "numpy": ">=1.26",
                        "pandas": ">=2.1",
                        "scikit-learn": ">=1.3",
                        **config.get("runtime_packages", {}),
                    },
                },
                "sensors": {
                    "required": ["pressure"],
                    "supported_devices": ["cpu"],
                },
                "validation": {
                    "evaluation": {
                        "split_strategy": "grouped by subject",
                        "integrity_status": "pending",
                        "evidence_files": [],
                    },
                    "semantic_mapping": {label: label for label in labels},
                    "device_tolerances": {"cpu_probability_atol": 0.03},
                    **({"golden_replay": {
                        "input_file": "../golden/sample_input.csv",
                        "expected_label": "왼쪽 다리 꼬기",
                    }} if golden_target.is_file() else {}),
                },
                "metadata": {
                    "required_sensors": ["pressure"],
                    "minimum_window_seconds": float(seconds) * float(config.get("min_tail_fraction", 0.95)),
                    "training_subject_count": config.get("training_subject_count"),
                    "training_record_count": config.get("training_record_count"),
                    "training_window_count": config.get("training_window_count"),
                    "fixed_configuration_group_cv_summary": summary,
                    "nested_cv_summary": config.get("nested_cv_summary", {}),
                    "training_run": self.run_dir.name,
                    "random_seed": self.seed,
                    "validation_macro_f1": config.get("validation_macro_f1"),
                    "inference_ms_median": config.get("inference_ms_median"),
                    "model_size_bytes": config.get("model_size_bytes"),
                    "estimator_backend": config.get("estimator_backend", "sklearn"),
                    "input_representation": config.get("input_representation", "feature_vector"),
                },
            }
            suffix = f"_{family}" if "model_family" in config else ""
            manifest_path = manifests_dir / f"posture_classifier_{seconds}s{suffix}_candidate.json"
            if has_evaluation:
                manifest["components"].append({
                    "role": "evaluation", "file": f"../{evaluation_path.name}",
                    "sha256": _digest(evaluation_path), "required": True,
                })
                manifest["provenance"]["dataset_fingerprint"] = config["dataset_fingerprint"]
                manifest["validation"]["evaluation"] = {
                    "split_strategy": "subject-disjoint train/validation/final_evaluation",
                    "integrity_status": "passed", "evidence_files": [f"../{evaluation_path.name}"],
                }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            from .model_runtime import ArtifactValidator, DeploymentRecord, DeploymentStore
            discovery = ArtifactValidator().discover_manifest(manifest_path)
            if not discovery.available or discovery.descriptor is None:
                raise ValueError(discovery.reason or "학습 후보 매니페스트가 유효하지 않습니다")
            if has_evaluation:
                from .model_runtime.deployment_checks import check_evaluation
                from .model_runtime.sklearn_adapter import SklearnAdapter
                check_evaluation(discovery.descriptor, manifest["validation"]["evaluation"], manifests_dir)
                if not golden_target.is_file():
                    raise ValueError("배포 후보의 Golden Replay 입력 파일이 없습니다.")
                probabilities = SklearnAdapter(discovery.descriptor).predict_proba(read_sensor_csv(golden_target), seconds)
                scores = probabilities.mean(axis=0)
                manifest["validation"]["golden_replay"].update(
                    expected_label=labels[int(scores.argmax())], expected_scores=scores.tolist(),
                    label_order=labels, window_count=len(probabilities), window_seconds=seconds,
                )
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                discovery = ArtifactValidator().discover_manifest(manifest_path)
            DeploymentStore(self.state_root).save(DeploymentRecord.pending(discovery.descriptor))
            created.append(manifest_path)
        incomplete.unlink()
        return tuple(created)

    def cancel(self) -> None:
        self._cancel_requested.set()
        if self.process and self.process.poll() is None:
            process = self.process
            try:
                process.terminate()
            except ProcessLookupError:
                return
            def ensure_stopped() -> None:
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
            threading.Thread(target=ensure_stopped, daemon=True).start()
