"""CSV loading and validation for the 4x4 posture dataset."""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SENSOR_COLUMNS = tuple(f"R{row}{column}" for row in range(1, 5) for column in range(1, 5))
CANONICAL_LABELS = (
    "정자세",
    "왼쪽기울임",
    "오른쪽기울임",
    "앞으로숙이기",
    "뒤로기대기",
    "왼쪽꼬기",
    "오른쪽꼬기",
)

# Keep aliases at the ingestion boundary so older recordings can still be read.
LABEL_ALIASES = {
    "왼기울임": "왼쪽기울임",
}
LABEL_SUSPECT_MARKER = "__label-suspect"


class DatasetError(ValueError):
    """Raised when a recording cannot safely be used for model training."""


@dataclass(frozen=True)
class Recording:
    """One labelled, contiguous sensor recording from a single subject."""

    source: Path
    subject: str
    label: str
    timestamps: np.ndarray
    frames: np.ndarray

    @property
    def session_id(self) -> str:
        """A stable identifier used to keep windows from one file together."""

        return str(self.source)


def normalize_label(label: str) -> str:
    """Return a canonical label, accepting only documented legacy aliases."""

    normalized = unicodedata.normalize("NFC", label).strip()
    normalized = LABEL_ALIASES.get(normalized, normalized)
    if normalized not in CANONICAL_LABELS:
        expected = ", ".join(CANONICAL_LABELS)
        raise DatasetError(f"Unknown label {label!r}. Expected one of: {expected}")
    return normalized


def _validate_headers(path: Path, fieldnames: list[str] | None) -> None:
    expected = {"elapsed_s", "label", *SENSOR_COLUMNS}
    actual = set(fieldnames or [])
    missing = expected - actual
    if missing:
        raise DatasetError(f"{path}: missing CSV columns: {', '.join(sorted(missing))}")


def load_recording(path: Path, *, subject: str | None = None) -> Recording:
    """Load one CSV recording and verify its schema, numbers, and label consistency."""

    path = Path(path)
    timestamps: list[float] = []
    frames: list[list[float]] = []
    labels: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_headers(path, reader.fieldnames)
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["elapsed_s"])
                frame = [float(row[column]) for column in SENSOR_COLUMNS]
            except (TypeError, ValueError) as error:
                raise DatasetError(f"{path}:{line_number}: invalid numeric sensor value") from error

            values = np.asarray([timestamp, *frame], dtype=float)
            if not np.isfinite(values).all():
                raise DatasetError(f"{path}:{line_number}: values must be finite")

            timestamps.append(timestamp)
            frames.append(frame)
            labels.add(normalize_label(row["label"]))

    if not frames:
        raise DatasetError(f"{path}: recording is empty")
    if len(labels) != 1:
        raise DatasetError(f"{path}: a recording must have exactly one label, found {labels}")

    timestamp_array = np.asarray(timestamps, dtype=float)
    if np.any(np.diff(timestamp_array) <= 0):
        raise DatasetError(f"{path}: elapsed_s values must be strictly increasing")

    subject_name = unicodedata.normalize("NFC", subject or path.parent.name)
    return Recording(
        source=path,
        subject=subject_name,
        label=labels.pop(),
        timestamps=timestamp_array,
        frames=np.asarray(frames, dtype=float),
    )


def is_label_suspect_path(path: Path) -> bool:
    """Return whether a filename marks a recording excluded from default training."""

    return LABEL_SUSPECT_MARKER in path.stem


def load_dataset(data_dir: Path, *, include_label_suspect: bool = False) -> list[Recording]:
    """Load recordings below ``data_dir`` in a deterministic order.

    Files whose title includes ``__label-suspect`` are excluded by default. This
    preserves raw recordings for later review without allowing questionable
    labels to affect the default training or evaluation results.
    """

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DatasetError(f"Dataset directory does not exist: {data_dir}")

    paths = sorted(data_dir.rglob("*.csv"))
    if not include_label_suspect:
        paths = [path for path in paths if not is_label_suspect_path(path)]
    if not paths:
        raise DatasetError(f"No CSV files found below: {data_dir}")
    return [load_recording(path) for path in paths]
