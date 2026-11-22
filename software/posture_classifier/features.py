"""Windowing and handcrafted pressure-map feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .data import Recording


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration shared by offline training and live inference.

    ``window_size`` is measured in frames. With the current dataset's stable
    sampling rate (about 12 Hz), the default 12-frame window is about one second.
    A full-window stride avoids near-duplicate training examples.
    """

    window_size: int = 12
    stride: int = 12
    minimum_regular_gap_s: float = 0.04

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.minimum_regular_gap_s <= 0:
            raise ValueError("minimum_regular_gap_s must be positive")


@dataclass(frozen=True)
class FeatureDataset:
    """Feature matrix plus provenance needed for leakage-safe evaluation."""

    features: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    session_ids: np.ndarray


def feature_names() -> tuple[str, ...]:
    """Return the ordered feature schema used by the trained SVM."""

    sensor_names = tuple(f"sensor_mean_{row}{column}" for row in range(1, 5) for column in range(1, 5))
    sensor_std_names = tuple(f"sensor_std_{row}{column}" for row in range(1, 5) for column in range(1, 5))
    aggregate_names = (
        "total_pressure_ratio_mean",
        "total_pressure_ratio_std",
        "pressure_center_row_mean",
        "pressure_center_row_std",
        "pressure_center_column_mean",
        "pressure_center_column_std",
        "row_balance_mean",
        "row_balance_std",
        "column_balance_mean",
        "column_balance_std",
        "row_reflection_error_mean",
        "row_reflection_error_std",
        "column_reflection_error_mean",
        "column_reflection_error_std",
        "frame_change_mean",
        "frame_change_std",
    )
    return sensor_names + sensor_std_names + aggregate_names


def trim_startup_frames(recording: Recording, minimum_regular_gap_s: float) -> Recording:
    """Discard the initial high-frequency burst before stable sampling begins.

    The current logger emits a short startup burst followed by the regular
    approximately 82 ms cadence. The first sample after the first regular-sized
    timestamp gap is retained. Recordings without such a gap are left unchanged.
    """

    gaps = np.diff(recording.timestamps)
    regular_gap_indexes = np.flatnonzero(gaps >= minimum_regular_gap_s)
    if regular_gap_indexes.size == 0:
        return recording

    start = int(regular_gap_indexes[0] + 1)
    return Recording(
        source=recording.source,
        subject=recording.subject,
        label=recording.label,
        timestamps=recording.timestamps[start:],
        frames=recording.frames[start:],
    )


def extract_window_features(window: np.ndarray, *, reference_total_pressure: float) -> np.ndarray:
    """Create an orientation-agnostic, 48-value feature vector from a 4x4 window.

    Each frame is divided by its total pressure, which controls for absolute
    body weight and sensor gain while preserving the pressure distribution. The
    total pressure is represented only relative to the recording's median total.
    Row/column features intentionally do not assume which physical direction is
    front, rear, left, or right; that mapping belongs to sensor installation.
    """

    values = np.asarray(window, dtype=float)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError("window must have shape (frames, 16)")
    if values.shape[0] < 2:
        raise ValueError("window must contain at least two frames")
    if not np.isfinite(values).all():
        raise ValueError("window values must be finite")
    if not np.isfinite(reference_total_pressure) or reference_total_pressure <= 0:
        raise ValueError("reference_total_pressure must be positive and finite")

    totals = values.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("each frame must have a positive total pressure")

    normalized = (values / totals[:, np.newaxis]).reshape(-1, 4, 4)
    sensor_mean = normalized.mean(axis=0).ravel()
    sensor_std = normalized.std(axis=0).ravel()

    row_axis = np.arange(4, dtype=float)[np.newaxis, :, np.newaxis]
    column_axis = np.arange(4, dtype=float)[np.newaxis, np.newaxis, :]
    row_center = (normalized * row_axis).sum(axis=(1, 2))
    column_center = (normalized * column_axis).sum(axis=(1, 2))

    row_balance = normalized[:, :2, :].sum(axis=(1, 2)) - normalized[:, 2:, :].sum(axis=(1, 2))
    column_balance = normalized[:, :, :2].sum(axis=(1, 2)) - normalized[:, :, 2:].sum(axis=(1, 2))
    row_reflection_error = np.abs(normalized - normalized[:, ::-1, :]).mean(axis=(1, 2))
    column_reflection_error = np.abs(normalized - normalized[:, :, ::-1]).mean(axis=(1, 2))
    frame_changes = np.abs(np.diff(normalized, axis=0)).mean(axis=(1, 2))

    summary = np.asarray(
        [
            np.mean(totals / reference_total_pressure),
            np.std(totals / reference_total_pressure),
            np.mean(row_center),
            np.std(row_center),
            np.mean(column_center),
            np.std(column_center),
            np.mean(row_balance),
            np.std(row_balance),
            np.mean(column_balance),
            np.std(column_balance),
            np.mean(row_reflection_error),
            np.std(row_reflection_error),
            np.mean(column_reflection_error),
            np.std(column_reflection_error),
            np.mean(frame_changes),
            np.std(frame_changes),
        ],
        dtype=float,
    )
    return np.concatenate((sensor_mean, sensor_std, summary))


def recording_to_features(recording: Recording, config: FeatureConfig) -> np.ndarray:
    """Convert one recording into non-overlapping feature windows."""

    stable_recording = trim_startup_frames(recording, config.minimum_regular_gap_s)
    frames = stable_recording.frames
    if len(frames) < config.window_size:
        raise ValueError(
            f"{recording.source}: {len(frames)} stable frames are insufficient for "
            f"a {config.window_size}-frame window"
        )

    reference_total_pressure = float(np.median(frames.sum(axis=1)))
    rows = [
        extract_window_features(frames[start : start + config.window_size], reference_total_pressure=reference_total_pressure)
        for start in range(0, len(frames) - config.window_size + 1, config.stride)
    ]
    return np.vstack(rows)


def build_feature_dataset(recordings: Iterable[Recording], config: FeatureConfig | None = None) -> FeatureDataset:
    """Build a feature matrix while retaining subject and session group labels."""

    config = config or FeatureConfig()
    feature_blocks: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []
    session_ids: list[str] = []

    for recording in recordings:
        block = recording_to_features(recording, config)
        feature_blocks.append(block)
        count = len(block)
        labels.extend([recording.label] * count)
        subjects.extend([recording.subject] * count)
        session_ids.extend([recording.session_id] * count)

    if not feature_blocks:
        raise ValueError("At least one recording is required")
    return FeatureDataset(
        features=np.vstack(feature_blocks),
        labels=np.asarray(labels, dtype=str),
        subjects=np.asarray(subjects, dtype=str),
        session_ids=np.asarray(session_ids, dtype=str),
    )
