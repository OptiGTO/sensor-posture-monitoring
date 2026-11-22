from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

SENSOR_COLUMNS = [f"R{i}{j}" for i in range(1, 5) for j in range(1, 5)]
CLASS_NAMES = [
    "정자세",
    "앞으로 숙이기",
    "뒤로 기대기",
    "왼쪽 기울임",
    "오른쪽 기울임",
    "왼쪽 다리 꼬기",
    "오른쪽 다리 꼬기",
]
LABEL_TO_ID = {label: idx for idx, label in enumerate(CLASS_NAMES)}

_LABEL_ALIASES = {
    "정자세": "정자세",
    "앞으로숙이기": "앞으로 숙이기",
    "앞으로기대기": "앞으로 숙이기",
    "뒤로기대기": "뒤로 기대기",
    "뒤대기": "뒤로 기대기",
    "왼쪽기울임": "왼쪽 기울임",
    "왼쪽기율임": "왼쪽 기울임",
    "오른쪽기울임": "오른쪽 기울임",
    "왼쪽꼬기": "왼쪽 다리 꼬기",
    "왼쪽다리꼬기": "왼쪽 다리 꼬기",
    "오른쪽꼬기": "오른쪽 다리 꼬기",
    "오른쪽다리꼬기": "오른쪽 다리 꼬기",
}


def normalize_text(value: object) -> str:
    return unicodedata.normalize("NFC", str(value)).strip()


def standardize_label(value: object) -> str:
    key = normalize_text(value).replace(" ", "")
    if key not in _LABEL_ALIASES:
        raise ValueError(f"알 수 없는 라벨 표기: {value!r}")
    return _LABEL_ALIASES[key]


def sensor_identity(frame: pd.DataFrame) -> str:
    """Identify a recording independently of label spelling and start time."""
    raw = frame[["elapsed_s", *SENSOR_COLUMNS]].to_numpy(dtype="<f8", copy=True)
    raw[:, 0] -= raw[0, 0]
    return hashlib.sha256(np.round(raw, 8).tobytes()).hexdigest()


def read_sensor_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [normalize_text(c) for c in df.columns]
    required = ["elapsed_s", *SENSOR_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 열 누락({path.name}): {missing}")
    df = df.assign(
        **{col: pd.to_numeric(df[col], errors="raise") for col in required}
    )
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError(f"비유한 수치가 포함됨: {path.name}")
    validate_sensor_frame(df)
    return df


def validate_sensor_frame(frame: pd.DataFrame) -> None:
    values = frame[["elapsed_s", *SENSOR_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("센서값과 elapsed_s는 유한한 숫자여야 합니다.")
    if len(values) < 2 or np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("elapsed_s는 최소 두 행에서 엄격히 증가해야 합니다.")


def recording_label(frame: pd.DataFrame) -> tuple[str, str]:
    """Require a real, consistent human label on every training row."""
    if "label" not in frame or frame["label"].isna().any():
        raise ValueError("학습 기록의 모든 행에 실제 자세 라벨이 필요합니다.")
    raw = [normalize_text(value) for value in frame["label"].unique()]
    labels = {standardize_label(value) for value in raw}
    if len(labels) != 1:
        raise ValueError("파일 내 실제 자세 라벨은 하나여야 합니다.")
    return raw[0], labels.pop()


def training_csv_paths(root: str | Path) -> list[Path]:
    """Use the same training boundary for analysis and both training entrypoints."""
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*.csv")):
        parts = {part.casefold() for part in path.relative_to(root).parts[:-1]}
        if parts & {"standard", "measurements", "__macosx"} or path.name.startswith("._") or path.name.endswith(".imu.csv"):
            continue
        manifest = path.with_suffix(".manifest.json")
        if manifest.exists():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"잘못된 데이터 매니페스트: {manifest.name}")
            if (metadata.get("source_type") == "measurement"
                    or metadata.get("label_status", "labeled") != "labeled"
                    or metadata.get("training_role", "include") != "include"):
                continue
        files.append(path)
    if not files:
        raise FileNotFoundError(f"학습용 CSV 파일을 찾지 못했습니다: {root}")
    return files


def resample_sequence(df: pd.DataFrame, target_length: int = 240) -> np.ndarray:
    """Irregular timestamps are interpolated to a fixed, uniform grid."""
    validate_sensor_frame(df)
    time = df["elapsed_s"].to_numpy(dtype=np.float64)
    values = df[SENSOR_COLUMNS].to_numpy(dtype=np.float64)
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("elapsed_s는 엄격히 증가해야 합니다.")
    grid = np.linspace(time[0], time[-1], target_length)
    sequence = np.column_stack(
        [np.interp(grid, time, values[:, idx]) for idx in range(values.shape[1])]
    )
    return sequence.astype(np.float32)


def _base_feature_names(prefix: str = "") -> list[str]:
    stat_names = [
        "mean", "std", "median", "min", "max", "range", "iqr", "rms",
        "mean_abs_diff", "slope", "skew", "kurtosis", "dominant_frequency",
        "spectral_entropy", "spectral_centroid", "low_frequency_ratio",
    ]
    names = [f"{prefix}{stat}_{channel}" for stat in stat_names for channel in SENSOR_COLUMNS]
    for axis_name in ("row", "column"):
        for stat in ("mean", "std", "mean_abs_diff"):
            names.extend(f"{prefix}{axis_name}_{stat}_{idx}" for idx in range(1, 5))
    derived = ("global", "left_right", "front_back", "diagonal", "outer_row", "outer_column")
    for stat in ("mean", "std", "mean_abs_diff"):
        names.extend(f"{prefix}{stat}_{item}" for item in derived)
    return names


def feature_names() -> list[str]:
    return _base_feature_names("abs_") + _base_feature_names("rel_")


def _extract_base_features(sequence: np.ndarray, sampling_rate: float = 12.0) -> np.ndarray:
    x = np.asarray(sequence, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError(f"예상 shape=(time, 16), 실제 shape={x.shape}")
    eps = 1e-9
    out: list[float] = []

    temporal = [
        x.mean(axis=0),
        x.std(axis=0),
        np.median(x, axis=0),
        x.min(axis=0),
        x.max(axis=0),
        np.ptp(x, axis=0),
        np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0),
        np.sqrt(np.mean(np.square(x), axis=0)),
        np.mean(np.abs(np.diff(x, axis=0)), axis=0),
    ]
    normalized_time = np.linspace(-1.0, 1.0, len(x))
    temporal.append(
        (normalized_time[:, None] * x).mean(axis=0)
        / (np.mean(np.square(normalized_time)) + eps)
    )
    temporal.extend(
        [
            np.nan_to_num(skew(x, axis=0, bias=False), nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(kurtosis(x, axis=0, bias=False), nan=0.0, posinf=0.0, neginf=0.0),
        ]
    )

    centered = x - x.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(len(x), d=1.0 / sampling_rate)
    spectrum_without_dc = spectrum.copy()
    spectrum_without_dc[0] = 0.0
    probability = spectrum / (spectrum.sum(axis=0, keepdims=True) + eps)
    temporal.extend(
        [
            frequencies[np.argmax(spectrum_without_dc, axis=0)],
            -(probability * np.log(probability + eps)).sum(axis=0) / np.log(len(frequencies)),
            (frequencies[:, None] * spectrum).sum(axis=0) / (spectrum.sum(axis=0) + eps),
            spectrum[frequencies < 1.0].sum(axis=0) / (spectrum.sum(axis=0) + eps),
        ]
    )
    for array in temporal:
        out.extend(np.asarray(array).tolist())

    matrix = x.reshape(len(x), 4, 4)
    row_mean = matrix.mean(axis=2)
    column_mean = matrix.mean(axis=1)
    for array in (row_mean, column_mean):
        out.extend(array.mean(axis=0).tolist())
        out.extend(array.std(axis=0).tolist())
        out.extend(np.mean(np.abs(np.diff(array, axis=0)), axis=0).tolist())

    derived = np.column_stack(
        [
            matrix.mean(axis=(1, 2)),
            matrix[:, :, :2].mean(axis=(1, 2)) - matrix[:, :, 2:].mean(axis=(1, 2)),
            matrix[:, :2, :].mean(axis=(1, 2)) - matrix[:, 2:, :].mean(axis=(1, 2)),
            np.diagonal(matrix, axis1=1, axis2=2).mean(axis=1)
            - np.diagonal(matrix[:, :, ::-1], axis1=1, axis2=2).mean(axis=1),
            matrix[:, 0, :].mean(axis=1) - matrix[:, 3, :].mean(axis=1),
            matrix[:, :, 0].mean(axis=1) - matrix[:, :, 3].mean(axis=1),
        ]
    )
    out.extend(derived.mean(axis=0).tolist())
    out.extend(derived.std(axis=0).tolist())
    out.extend(np.mean(np.abs(np.diff(derived, axis=0)), axis=0).tolist())
    return np.asarray(out, dtype=np.float32)


def extract_feature_vector(sequence: np.ndarray) -> np.ndarray:
    """Concatenate absolute-signal and frame-relative spatial-distribution features."""
    x = np.asarray(sequence, dtype=np.float32)
    frame_mean = x.mean(axis=1, keepdims=True)
    frame_std = x.std(axis=1, keepdims=True) + 1e-6
    relative = (x - frame_mean) / frame_std
    features = np.concatenate(
        [_extract_base_features(x), _extract_base_features(relative)], axis=0
    )
    expected = len(feature_names())
    if features.size != expected:
        raise RuntimeError(f"특징 수 불일치: {features.size} != {expected}")
    return features


def load_dataset(root: str | Path, target_length: int = 240):
    root = Path(root)
    files = training_csv_paths(root)

    sequences: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for path in files:
        df = read_sensor_csv(path)
        raw_label, label = recording_label(df)
        subject = normalize_text(path.parent.name)
        filename = normalize_text(path.name)
        match = re.search(r"_(\d{3})\.csv$", filename)
        session = match.group(1) if match else "미상"
        sequence = resample_sequence(df, target_length=target_length)
        sequences.append(sequence)
        rows.append(
            {
                "path": str(path),
                "file": filename,
                "subject": subject,
                "raw_label": raw_label,
                "label": label,
                "class_id": LABEL_TO_ID[label],
                "session": session,
                "rows": len(df),
                "duration_s": float(df["elapsed_s"].iloc[-1] - df["elapsed_s"].iloc[0]),
            }
        )
    metadata = pd.DataFrame(rows)
    return np.stack(sequences), metadata


def make_subject_pairs(subjects: Iterable[str], seed: int = 20260826) -> list[list[str]]:
    unique = sorted(set(map(str, subjects)))
    if len(unique) < 4:
        raise ValueError("피험자 단위 중첩 검증에는 최소 4명의 데이터가 필요합니다.")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return [unique[idx : idx + 2] for idx in range(0, len(unique), 2)]
