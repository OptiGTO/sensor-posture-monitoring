from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .posture_core import (
    CLASS_NAMES,
    LABEL_TO_ID,
    SENSOR_COLUMNS,
    extract_feature_vector,
    normalize_text,
    read_sensor_csv,
    recording_label,
    training_csv_paths,
    validate_sensor_frame,
    standardize_label,
)

DEFAULT_SAMPLING_RATE = 12.0
DEFAULT_MIN_TAIL_FRACTION = 0.95


@dataclass(frozen=True)
class WindowSpec:
    window_seconds: float
    sampling_rate: float = DEFAULT_SAMPLING_RATE
    min_tail_fraction: float = DEFAULT_MIN_TAIL_FRACTION

    def __post_init__(self) -> None:
        if not np.isfinite([self.window_seconds, self.sampling_rate, self.min_tail_fraction]).all():
            raise ValueError("창 설정은 유한한 숫자여야 합니다.")
        if self.window_seconds <= 0 or self.sampling_rate <= 0 or self.target_length < 2:
            raise ValueError("창 길이와 샘플링 주파수는 양수이며 최소 두 시점이 필요합니다.")
        if not 0 < self.min_tail_fraction <= 1:
            raise ValueError("min_tail_fraction은 0보다 크고 1 이하여야 합니다.")

    @property
    def target_length(self) -> int:
        return int(round(self.window_seconds * self.sampling_rate))

    @property
    def stride_seconds(self) -> float:
        return self.window_seconds


def window_start_times(
    duration_s: float,
    window_seconds: float,
    min_tail_fraction: float = DEFAULT_MIN_TAIL_FRACTION,
) -> list[float]:
    if not np.isfinite([duration_s, window_seconds, min_tail_fraction]).all() or not 0 < min_tail_fraction <= 1:
        raise ValueError("창 설정과 입력 길이가 유효하지 않습니다.")
    if duration_s <= 0 or window_seconds <= 0:
        raise ValueError("duration_s와 window_seconds는 양수여야 합니다.")
    full_count = int(np.floor((duration_s + 1e-9) / window_seconds))
    starts = [float(index * window_seconds) for index in range(full_count)]
    tail_start = float(full_count * window_seconds)
    tail_duration = float(duration_s - tail_start)
    if tail_duration >= min_tail_fraction * window_seconds:
        starts.append(tail_start)
    return starts


def segment_sensor_frame(
    frame: pd.DataFrame,
    spec: WindowSpec,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create non-overlapping fixed-rate windows using the original elapsed time.

    A near-complete final window is retained when at least min_tail_fraction of its
    nominal duration is available. np.interp performs constant edge extension only
    for the tiny missing tail, if any.
    """
    validate_sensor_frame(frame)
    time = frame["elapsed_s"].to_numpy(dtype=np.float64)
    values = frame[SENSOR_COLUMNS].to_numpy(dtype=np.float64)
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("elapsed_s는 엄격히 증가해야 합니다.")
    if np.any(np.diff(time) > 1.0):
        raise ValueError("1초를 넘는 센서 공백이 있습니다. 연결이 연속된 기록을 사용하세요.")
    relative_time = time - time[0]
    duration_s = float(relative_time[-1])
    starts = window_start_times(
        duration_s,
        spec.window_seconds,
        spec.min_tail_fraction,
    )
    if not starts:
        raise ValueError(
            f"입력 길이 {duration_s:.3f}초가 {spec.window_seconds:g}초 창보다 짧습니다."
        )

    sample_offsets = np.arange(spec.target_length, dtype=np.float64) / spec.sampling_rate
    windows: list[np.ndarray] = []
    rows: list[dict[str, float | int]] = []
    for window_index, start_s in enumerate(starts, start=1):
        grid = start_s + sample_offsets
        sequence = np.column_stack(
            [np.interp(grid, relative_time, values[:, channel]) for channel in range(values.shape[1])]
        ).astype(np.float32)
        windows.append(sequence)
        rows.append(
            {
                "window_index": window_index,
                "start_s": float(start_s),
                "nominal_end_s": float(start_s + spec.window_seconds),
                "last_sample_s": float(grid[-1]),
                "available_duration_s": float(max(0.0, duration_s - start_s)),
                "target_length": spec.target_length,
            }
        )
    return np.stack(windows), pd.DataFrame(rows)


def _record_session(filename: str) -> str:
    import re

    match = re.search(r"_(\d{3})\.csv$", filename)
    return match.group(1) if match else "미상"


def load_window_dataset(root: str | Path, spec: WindowSpec,
                        frame_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None):
    """Window every training record; ``frame_transform`` (e.g. versioned signal
    preprocessing) is applied to each raw recording before segmentation. Record
    identity, label and duration always describe the untransformed source file."""
    root = Path(root)
    files = training_csv_paths(root)

    all_windows: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    for record_number, path in enumerate(files, start=1):
        frame = read_sensor_csv(path)
        raw_label, label = recording_label(frame)
        subject = normalize_text(path.parent.name)
        filename = normalize_text(path.name)
        record_id = f"R{record_number:03d}"
        try:
            segmented = frame if frame_transform is None else frame_transform(frame)
            windows, window_meta = segment_sensor_frame(segmented, spec)
        except ValueError as error:
            raise ValueError(f"{path.name}: {error}") from error
        all_windows.extend(windows)
        duration_s = float(frame["elapsed_s"].iloc[-1] - frame["elapsed_s"].iloc[0])
        record_rows.append(
            {
                "record_id": record_id,
                "path": str(path),
                "file": filename,
                "subject": subject,
                "raw_label": raw_label,
                "label": label,
                "class_id": LABEL_TO_ID[label],
                "session": _record_session(filename),
                "duration_s": duration_s,
                "window_count": len(windows),
            }
        )
        for window_row in window_meta.to_dict(orient="records"):
            rows.append(
                {
                    "record_id": record_id,
                    "path": str(path),
                    "file": filename,
                    "subject": subject,
                    "label": label,
                    "class_id": LABEL_TO_ID[label],
                    "duration_s": duration_s,
                    **window_row,
                }
            )

    return np.stack(all_windows), pd.DataFrame(rows), pd.DataFrame(record_rows)


def extract_window_features(windows: np.ndarray) -> np.ndarray:
    return np.stack([extract_feature_vector(window) for window in windows])


def model_classes(model) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_, dtype=int)
    if hasattr(model, "named_steps") and hasattr(model.named_steps["model"], "classes_"):
        return np.asarray(model.named_steps["model"].classes_, dtype=int)
    raise ValueError("모델 클래스 순서를 확인할 수 없습니다.")


def aligned_predict_proba(model, features: np.ndarray, class_count: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = model_classes(model)
    aligned = np.zeros((len(features), class_count), dtype=np.float64)
    aligned[:, classes] = raw
    return aligned


def aggregate_probabilities(
    probabilities: np.ndarray,
    record_ids: Iterable[str],
    y: np.ndarray,
    subjects: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        {
            "record_id": np.asarray(list(record_ids), dtype=str),
            "y": np.asarray(y, dtype=int),
            "subject": np.asarray(list(subjects), dtype=str),
        }
    )
    probability_columns = [f"p_{index}" for index in range(probabilities.shape[1])]
    probability_frame = pd.DataFrame(probabilities, columns=probability_columns)
    joined = pd.concat([frame.reset_index(drop=True), probability_frame], axis=1)
    grouped_probability = joined.groupby("record_id", sort=False)[probability_columns].mean()
    grouped_label = joined.groupby("record_id", sort=False)["y"].first()
    grouped_subject = joined.groupby("record_id", sort=False)["subject"].first()
    if not (grouped_probability.index.equals(grouped_label.index) and grouped_label.index.equals(grouped_subject.index)):
        raise RuntimeError("기록 단위 집계 인덱스가 일치하지 않습니다.")
    return (
        grouped_probability.to_numpy(dtype=np.float64),
        grouped_label.to_numpy(dtype=int),
        grouped_probability.index.to_numpy(dtype=str),
        grouped_subject.to_numpy(dtype=str),
    )


def class_probability_dict(probability: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(CLASS_NAMES, probability)}
