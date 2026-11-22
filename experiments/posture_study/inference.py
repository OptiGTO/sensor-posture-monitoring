"""CSV and stateful inference for frozen study checkpoints, outside production.

Example::

    .venv/bin/python -m experiments.posture_study.inference \
        --checkpoint results/posture_study_20260913/B00/primary/checkpoint.joblib \
        --csv /path/to/new_recording.csv --output /tmp/predictions.json

Personalized models additionally require ``--registration-csv`` from a separate
correct-posture registration recording. Prediction labels are never read and a
registration reference is never estimated from the prediction recording.

``StreamingPredictor.predict_proba`` takes one already-resampled, complete
36-by-16 window. Call ``reset`` for a new recording; observable recording changes,
nonincreasing timestamps and gaps longer than 3.01 s also reset temporal state.
The reference mean must belong to the current user and precede their predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np

from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, read_sensor_csv, recording_label
from src.window_pipeline import WindowSpec, extract_window_features, segment_sensor_frame

WINDOW_SECONDS = 3.0
SAMPLE_RATE = 12.0
WINDOW_SHAPE = (36, 16)
VERSION = "posture-study-inference-v1"


def requires_registration(model: Any, _seen: set[int] | None = None) -> bool:
    """Inspect nested study models without looking at prediction data or labels."""
    seen = set() if _seen is None else _seen
    if id(model) in seen:
        return False
    seen.add(id(model))
    if getattr(model, "method", None) == "personal" or "personal" in getattr(model, "components", []):
        return True
    children = []
    for name in ("base", "encoder"):
        child = getattr(model, name, None)
        if child is not None:
            children.append(child)
    for name in ("models", "members_"):
        children.extend(getattr(model, name, []))
    return any(requires_registration(child, seen) for child in children)


def registration_mean_from_csv(path: str | Path) -> np.ndarray:
    """Exactly the study's first-ten-seconds mean of a separate registration."""
    frame = read_sensor_csv(path)
    if "label" in frame and recording_label(frame)[1] != CLASS_NAMES[0]:
        raise ValueError("등록 CSV의 라벨은 정자세여야 합니다.")
    elapsed = frame.elapsed_s.to_numpy() - frame.elapsed_s.iloc[0]
    return frame.loc[elapsed <= 10, SENSOR_COLUMNS].to_numpy(float).mean(axis=0)


def _validate_reference(reference: np.ndarray | None, required: bool) -> np.ndarray | None:
    if reference is None:
        if required:
            raise ValueError("이 모델에는 별도 정자세 등록 CSV 또는 16채널 등록 평균이 필요합니다.")
        return None
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape != (16,) or not np.isfinite(reference).all() or np.any(reference <= -1):
        raise ValueError("등록 평균은 로그 비율을 계산할 수 있는 유한한 16채널 값이어야 합니다.")
    return reference.copy()


def build_inputs(sequences: np.ndarray, *, registration_mean: np.ndarray | None = None,
                 record_ids: np.ndarray | None = None, start_s: np.ndarray | None = None,
                 require_registration: bool = False) -> dict[str, np.ndarray]:
    """Build the precise study feature schema, without fitting any transforms."""
    seq = np.asarray(sequences, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[1:] != WINDOW_SHAPE or len(seq) == 0:
        raise ValueError("입력은 하나 이상의 (36, 16) 센서 창이어야 합니다.")
    if not np.isfinite(seq).all():
        raise ValueError("센서 창에 비유한 수치가 있습니다.")
    reference = _validate_reference(registration_mean, require_registration)
    if reference is None:
        personal = np.zeros((len(seq), 49), dtype=np.float32)
    else:
        # Preserve preparation's float32 window-mean arithmetic and final cast.
        mean = seq.mean(axis=1)
        if np.any(mean <= -1):
            raise ValueError("센서 평균의 로그 비율을 계산할 수 없습니다.")
        ratio = mean / np.maximum(mean.sum(axis=1, keepdims=True), 1e-6)
        reference_ratio = reference / max(reference.sum(), 1e-6)
        personal = np.concatenate((
            np.tile(reference_ratio, (len(seq), 1)), ratio - reference_ratio,
            np.log((mean + 1) / (reference + 1)),
            ((mean.sum(axis=1) + 1) / (reference.sum() + 1))[:, None],
        ), axis=1).astype(np.float32)
    records = np.full(len(seq), "local") if record_ids is None else np.asarray(record_ids, dtype=str)
    starts = np.arange(len(seq), dtype=float) * WINDOW_SECONDS if start_s is None else np.asarray(start_s, dtype=float)
    if records.shape != (len(seq),) or starts.shape != (len(seq),) or not np.isfinite(starts).all():
        raise ValueError("기록 ID와 시작 시간은 센서 창마다 하나씩 필요합니다.")
    return {"seq": seq, "features": extract_window_features(seq).astype(np.float32),
            "personal": personal, "groups": np.full(len(seq), "local"),
            "record_ids": records, "start_s": starts}


class _StatefulNode:
    """Keep one bounded temporal state per node in a nested inference graph."""

    def __init__(self, model: Any):
        from .run import FusionModel, TemporalModel

        self.model = model
        self.kind = "temporal" if isinstance(model, TemporalModel) else "fusion" if isinstance(model, FusionModel) else "leaf"
        self.children = ([_StatefulNode(model.base)] if self.kind == "temporal" else
                         [_StatefulNode(child) for child in model.models] if self.kind == "fusion" else [])
        if self.kind == "temporal" and model.method not in {"ema", "hysteresis"}:
            raise ValueError(f"지원하지 않는 시간 후처리: {model.method}")
        self.reset()

    def reset(self) -> None:
        self.state = None
        self.pending = None
        self.count = 0
        self.previous_record = None
        self.last_time = None
        for child in self.children:
            child.reset()

    def predict(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        if self.kind == "leaf":
            return np.asarray(self.model.predict_proba(inputs))[0].copy()
        if self.kind == "fusion":
            return np.mean([child.predict(inputs) for child in self.children], axis=0)
        probability = self.children[0].predict(inputs)
        record, start = str(inputs["record_ids"][0]), float(inputs["start_s"][0])
        if (record != self.previous_record or self.last_time is None or
                start <= self.last_time or start - self.last_time > 3.01):
            self.state = probability.copy()
            self.pending, self.count = None, 0
        elif self.model.method == "ema":
            self.state = 0.6 * probability + 0.4 * self.state
        else:
            current, candidate = int(self.state.argmax()), int(probability.argmax())
            if candidate == current:
                self.pending, self.count = None, 0
                self.state = probability.copy()
            else:
                self.count = self.count + 1 if self.pending == candidate else 1
                self.pending = candidate
                if self.count >= 2:
                    self.state = probability.copy()
                    self.pending, self.count = None, 0
        self.previous_record, self.last_time = record, start
        return self.state.copy()


class StreamingPredictor:
    """Inference on successive complete windows with persistent causal state.

    ``start_s`` defaults to consecutive 3 s starts. Supply observed starts to
    detect gaps and restarted clocks. No prediction call updates the registered
    reference. Passing a different explicit reference resets temporal history.
    """

    def __init__(self, model_or_checkpoint: Any, *, registration_mean: np.ndarray | None = None,
                 record_id: str = "stream"):
        self.model = joblib.load(model_or_checkpoint) if isinstance(model_or_checkpoint, (str, Path)) else model_or_checkpoint
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("자세 실험의 predict_proba 체크포인트가 필요합니다.")
        self.needs_registration = requires_registration(self.model)
        self.registration_mean = _validate_reference(registration_mean, self.needs_registration)
        self.node = _StatefulNode(self.model)
        self.record_id = str(record_id)
        self.next_start_s = 0.0
        self.last_start_s = None

    def reset(self, *, record_id: str | None = None,
              registration_mean: np.ndarray | None = None) -> None:
        """Reset temporal history; an omitted reference preserves this user's reference."""
        if registration_mean is not None:
            self.registration_mean = _validate_reference(registration_mean, self.needs_registration)
        if record_id is not None:
            self.record_id = str(record_id)
        self.next_start_s = 0.0
        self.last_start_s = None
        self.node.reset()

    def predict_proba(self, window: np.ndarray, *, start_s: float | None = None,
                      record_id: str | None = None,
                      registration_mean: np.ndarray | None = None) -> np.ndarray:
        if registration_mean is not None:
            reference = _validate_reference(registration_mean, self.needs_registration)
            if self.registration_mean is None or not np.array_equal(reference, self.registration_mean):
                self.node.reset()
            self.registration_mean = reference
        if record_id is not None and str(record_id) != self.record_id:
            self.record_id = str(record_id)
            self.next_start_s = 0.0
        start = self.next_start_s if start_s is None else float(start_s)
        inputs = build_inputs(np.asarray(window)[None], registration_mean=self.registration_mean,
                              record_ids=np.array([self.record_id]), start_s=np.array([start]),
                              require_registration=self.needs_registration)
        probability = self.node.predict(inputs)
        if probability.shape != (7,) or not np.isfinite(probability).all() or not np.isclose(probability.sum(), 1, atol=1e-4):
            raise ValueError("모델 확률 출력이 7개 클래스 규격과 일치하지 않습니다.")
        self.last_start_s, self.next_start_s = start, start + WINDOW_SECONDS
        return probability

    def predict(self, window: np.ndarray, **kwargs) -> dict:
        probability = self.predict_proba(window, **kwargs)
        return _prediction_row(probability, self.record_id, self.last_start_s)


def _prediction_row(probability: np.ndarray, record_id: str, start_s: float) -> dict:
    label_id = int(np.argmax(probability))
    return {"record_id": str(record_id), "start_s": float(start_s),
            "nominal_end_s": float(start_s + WINDOW_SECONDS), "class_id": label_id,
            "label": CLASS_NAMES[label_id], "score": float(probability[label_id]),
            "probabilities": np.asarray(probability, dtype=float).tolist()}


def predict_csv(checkpoint: str | Path, csv_path: str | Path,
                registration_csv: str | Path | None = None) -> dict:
    checkpoint, csv_path = Path(checkpoint).resolve(), Path(csv_path).resolve()
    registration_path = None if registration_csv is None else Path(registration_csv).resolve()
    model = joblib.load(checkpoint)
    if not hasattr(model, "predict_proba"):
        raise ValueError("자세 실험의 predict_proba 체크포인트가 필요합니다.")
    needs_registration = requires_registration(model)
    reference = None
    if registration_path is not None:
        if registration_path == csv_path or registration_path.read_bytes() == csv_path.read_bytes():
            raise ValueError("예측 CSV와 정자세 등록 CSV는 별도 기록이어야 합니다.")
        reference = registration_mean_from_csv(registration_path)
    _validate_reference(reference, needs_registration)
    started = time.perf_counter()
    prediction_frame = read_sensor_csv(csv_path)
    if registration_path is not None:
        registration_frame = read_sensor_csv(registration_path)
        if np.array_equal(prediction_frame[SENSOR_COLUMNS].to_numpy(), registration_frame[SENSOR_COLUMNS].to_numpy()):
            raise ValueError("예측 CSV와 등록 CSV의 센서 시계열이 동일합니다. 독립 등록 기록을 사용하세요.")
    sequences, metadata = segment_sensor_frame(prediction_frame, WindowSpec(WINDOW_SECONDS))
    inputs = build_inputs(sequences, registration_mean=reference,
                          record_ids=np.full(len(sequences), str(csv_path)),
                          start_s=metadata["start_s"].to_numpy(), require_registration=needs_registration)
    prepared = time.perf_counter()
    probability = np.asarray(model.predict_proba(inputs))
    finished = time.perf_counter()
    if probability.shape != (len(sequences), 7) or not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-4):
        raise ValueError("모델 확률 출력이 7개 클래스 규격과 일치하지 않습니다.")
    return {"version": VERSION, "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "csv": str(csv_path), "class_names": CLASS_NAMES,
            "window_seconds": WINDOW_SECONDS, "sampling_rate": SAMPLE_RATE,
            "frames_per_window": WINDOW_SHAPE[0], "legacy_feature_dimension": 596,
            "registration_required": needs_registration,
            "registration_csv": None if registration_path is None else str(registration_path),
            "registration_rule": "separate record; first <=10 seconds mean; no prediction-derived calibration",
            "probability_note": "모델 결정 점수이며 실제 정확도 또는 보정된 확신도로 검증되지 않았습니다.",
            "preprocessing_seconds": prepared - started, "model_inference_seconds": finished - prepared,
            "predictions": [_prediction_row(p, record, start) for p, record, start in
                            zip(probability, inputs["record_ids"], inputs["start_s"])]}


def main() -> None:
    parser = argparse.ArgumentParser(description="분리된 자세 실험 체크포인트로 새 CSV를 추론합니다.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--registration-csv", type=Path)
    parser.add_argument("--output", type=Path, help="생략하면 JSON을 표준 출력으로 보냅니다.")
    args = parser.parse_args()
    inputs = {path.resolve() for path in (args.checkpoint, args.csv, args.registration_csv) if path is not None}
    if args.output is not None and args.output.resolve() in inputs:
        parser.error("출력 경로는 입력 CSV 및 체크포인트와 달라야 합니다.")
    try:
        result = predict_csv(args.checkpoint, args.csv, args.registration_csv)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
