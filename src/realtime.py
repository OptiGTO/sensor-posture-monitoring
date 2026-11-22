"""Time-windowed live inference using the deployed posture model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .inference_core import PosturePredictor
from .posture_core import SENSOR_COLUMNS


@dataclass(frozen=True)
class LiveConfig:
    window_seconds: float = 20.0
    minimum_window_seconds: float = 19.5
    prediction_interval_seconds: float = 1.0
    max_gap_seconds: float = 1.0
    smoothing_history: int = 3
    confidence_threshold: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite([self.window_seconds, self.minimum_window_seconds, self.prediction_interval_seconds, self.max_gap_seconds]).all():
            raise ValueError("live timing settings must be finite")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not 0 < self.minimum_window_seconds <= self.window_seconds:
            raise ValueError("minimum_window_seconds must be in (0, window_seconds]")
        if self.prediction_interval_seconds <= 0 or self.max_gap_seconds <= 0:
            raise ValueError("prediction_interval_seconds and max_gap_seconds must be positive")
        if self.smoothing_history < 1:
            raise ValueError("smoothing_history must be positive")
        if self.confidence_threshold is not None and not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between zero and one")


@dataclass(frozen=True)
class PredictionEvent:
    timestamp_s: float
    state: str
    window_start_s: float | None
    window_end_s: float | None
    sample_count: int
    posture: str | None
    confidence: float | None
    probabilities: dict[str, float]
    detail: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp_s": self.timestamp_s,
            "state": self.state,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "sample_count": self.sample_count,
            "posture": self.posture,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "detail": self.detail,
        }


class TimeWindowBuffer:
    """Keep every valid sensor frame needed for one recent time interval."""

    def __init__(self, config: LiveConfig) -> None:
        self.config = config
        self._timestamps: deque[float] = deque()
        self._values: deque[np.ndarray] = deque()
        self.last_reset_reason: str | None = None

    @property
    def sample_count(self) -> int:
        return len(self._timestamps)

    @property
    def duration_seconds(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        return self._timestamps[-1] - self._timestamps[0]

    @property
    def end_timestamp_s(self) -> float | None:
        return self._timestamps[-1] if self._timestamps else None

    def clear(self, reason: str | None = None) -> None:
        self._timestamps.clear()
        self._values.clear()
        self.last_reset_reason = reason

    def append(self, timestamp_s: float, values: Iterable[float]) -> None:
        timestamp_s = float(timestamp_s)
        frame = np.asarray(tuple(values), dtype=float)
        if not np.isfinite(timestamp_s):
            raise ValueError("timestamp must be finite")
        if frame.shape != (len(SENSOR_COLUMNS),):
            raise ValueError(f"sensor frame must contain {len(SENSOR_COLUMNS)} values")
        if not np.isfinite(frame).all():
            raise ValueError("sensor frame must contain only finite values")

        if self._timestamps:
            gap = timestamp_s - self._timestamps[-1]
            if gap <= 0:
                raise ValueError("timestamps must be strictly increasing")
            if gap > self.config.max_gap_seconds:
                self.clear(reason=f"sensor gap {gap:.3f}s")

        self._timestamps.append(timestamp_s)
        self._values.append(frame)
        self._trim(timestamp_s)

    def _trim(self, timestamp_s: float) -> None:
        # Retain one frame before the cutoff for precise interpolation at the start.
        cutoff = timestamp_s - self.config.window_seconds
        while len(self._timestamps) > 1 and self._timestamps[1] < cutoff:
            self._timestamps.popleft()
            self._values.popleft()

    def is_ready(self) -> bool:
        return self.duration_seconds >= self.config.minimum_window_seconds

    def snapshot(self) -> pd.DataFrame:
        if not self.is_ready():
            raise RuntimeError("live buffer has not reached the minimum window duration")
        timestamps = np.asarray(self._timestamps, dtype=float)
        values = np.stack(tuple(self._values))
        end = timestamps[-1]
        start = end - self.config.window_seconds

        # During the initial warm-up, preserve the original interval exactly.
        if timestamps[0] <= start:
            start_values = np.asarray(
                [np.interp(start, timestamps, values[:, index]) for index in range(values.shape[1])]
            )
            mask = timestamps > start
            timestamps = np.concatenate(([start], timestamps[mask]))
            values = np.vstack((start_values, values[mask]))

        relative_time = timestamps - timestamps[0]
        return pd.DataFrame(values, columns=SENSOR_COLUMNS).assign(elapsed_s=relative_time)[
            ["elapsed_s", *SENSOR_COLUMNS]
        ]


class RealtimePosturePredictor:
    """Turn an ordered stream of 16-channel frames into stable posture events."""

    def __init__(self, predictor: PosturePredictor, config: LiveConfig | None = None) -> None:
        self.predictor = predictor
        self.config = config or LiveConfig()
        self.buffer = TimeWindowBuffer(self.config)
        self._probability_history: deque[np.ndarray] = deque(maxlen=self.config.smoothing_history)
        self._last_prediction_s: float | None = None

    def reset(self, reason: str | None = None) -> None:
        self.buffer.clear(reason)
        self._probability_history.clear()
        self._last_prediction_s = None

    def push(self, timestamp_s: float, values: Iterable[float]) -> PredictionEvent | None:
        had_gap = (
            self.buffer.end_timestamp_s is not None
            and timestamp_s - self.buffer.end_timestamp_s > self.config.max_gap_seconds
        )
        self.buffer.append(timestamp_s, values)
        if had_gap:
            self._probability_history.clear()
            self._last_prediction_s = None

        if not self.buffer.is_ready():
            return None
        if (
            self._last_prediction_s is not None
            and timestamp_s - self._last_prediction_s < self.config.prediction_interval_seconds
        ):
            return None

        snapshot = self.buffer.snapshot()
        raw = self.predictor.predict_frame(snapshot)
        self._probability_history.append(raw)
        averaged = np.mean(np.stack(tuple(self._probability_history)), axis=0)
        predicted_index = int(np.argmax(averaged))
        confidence = float(averaged[predicted_index])
        state = "predicting"
        posture: str | None = self.predictor.labels[predicted_index]
        if self.config.confidence_threshold is not None and confidence < self.config.confidence_threshold:
            state = "uncertain"
            posture = None

        self._last_prediction_s = float(timestamp_s)
        return PredictionEvent(
            timestamp_s=float(timestamp_s),
            state=state,
            window_start_s=float(timestamp_s - snapshot["elapsed_s"].iloc[-1]),
            window_end_s=float(timestamp_s),
            sample_count=len(snapshot),
            posture=posture,
            confidence=confidence,
            probabilities={
                label: float(value)
                for label, value in zip(self.predictor.labels, averaged)
            },
            detail=self.buffer.last_reset_reason,
        )
