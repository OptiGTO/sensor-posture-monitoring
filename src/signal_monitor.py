"""Display-only data for the live 16-channel sensor monitor.

The session worker pushes the same frames it already passes to the recorder and
the model. Nothing here feeds back into recording, inference or training: a
rejected or dropped display sample never changes the measurement CSV.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .posture_core import SENSOR_COLUMNS

DEFAULT_SPAN_S = 10.0
# Same threshold the session worker uses to report a sensor gap.
DEFAULT_BREAK_S = 1.0


@dataclass(frozen=True)
class SignalSnapshot:
    """Copy of the buffered samples, safe to read on the Tk thread."""

    times: np.ndarray
    values: np.ndarray
    origin_s: float | None
    age_s: float | None
    total_samples: int

    @property
    def latest(self) -> np.ndarray | None:
        return self.values[-1] if len(self.values) else None

    def sample_rate_hz(self, recent: int = 12) -> float | None:
        """Steady input rate from the latest intervals (a start-up burst of
        buffered serial lines should not dominate the reading)."""
        if len(self.times) < 3:
            return None
        intervals = np.diff(self.times)[-recent:]
        intervals = intervals[intervals <= DEFAULT_BREAK_S]
        if not len(intervals):
            return None
        median = float(np.median(intervals))
        return 1.0 / median if median > 0 else None


class SignalBuffer:
    """Thread-safe rolling window of the most recent sensor frames."""

    def __init__(
        self,
        span_s: float = DEFAULT_SPAN_S,
        channels: int = len(SENSOR_COLUMNS),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(span_s) or span_s <= 0:
            raise ValueError("표시 구간은 0보다 큰 유한한 초 단위여야 합니다.")
        self.span_s = float(span_s)
        self.channels = int(channels)
        self._clock = clock
        self._lock = threading.Lock()
        self._samples: deque[tuple[float, np.ndarray]] = deque()
        self._origin: float | None = None
        self._last_push: float | None = None
        self._total = 0

    def push(self, elapsed_s: float, values: Iterable[float]) -> bool:
        """Add one frame; invalid or out-of-order frames are ignored, never raised."""
        try:
            timestamp = float(elapsed_s)
            data = np.asarray(tuple(values), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return False
        if data.size != self.channels or not math.isfinite(timestamp) or not np.isfinite(data).all():
            return False
        with self._lock:
            if self._samples and timestamp <= self._samples[-1][0]:
                return False
            if self._origin is None:
                self._origin = timestamp
            self._samples.append((timestamp, data.copy()))
            while self._samples and self._samples[0][0] < timestamp - self.span_s:
                self._samples.popleft()
            self._last_push = self._clock()
            self._total += 1
        return True

    def snapshot(self) -> SignalSnapshot:
        with self._lock:
            samples = list(self._samples)
            origin, last_push, total = self._origin, self._last_push, self._total
        if samples:
            times = np.fromiter((item[0] for item in samples), dtype=float, count=len(samples))
            values = np.vstack([item[1] for item in samples])
        else:
            times = np.empty(0, dtype=float)
            values = np.empty((0, self.channels), dtype=float)
        age = None if last_push is None else max(0.0, self._clock() - last_push)
        return SignalSnapshot(times, values, origin, age, total)


def sweep_segments(
    times: np.ndarray,
    span_s: float,
    *,
    origin_s: float = 0.0,
    erase_fraction: float = 0.04,
    break_s: float = DEFAULT_BREAK_S,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Heart-monitor sweep layout.

    Returns horizontal positions in [0, 1) for every sample and contiguous
    ``(start, stop)`` index ranges to draw. Samples older than one sweep minus a
    small erase gap are hidden; lines break where the sweep wraps to the left
    edge and across sensor gaps longer than ``break_s``.
    """
    times = np.asarray(times, dtype=float)
    positions = np.empty(0, dtype=float)
    if not len(times):
        return positions, []
    visible_after = times[-1] - span_s * (1.0 - erase_fraction)
    first = int(np.searchsorted(times, visible_after, side="right"))
    positions = np.mod(times - origin_s, span_s) / span_s
    segments: list[tuple[int, int]] = []
    start = first
    for index in range(first + 1, len(times)):
        wrapped = positions[index] < positions[index - 1]
        gap = times[index] - times[index - 1] > break_s
        if wrapped or gap:
            segments.append((start, index))
            start = index
    segments.append((start, len(times)))
    return positions, [(a, b) for a, b in segments if b > a]


class CommonScale:
    """One vertical range shared by all sensors so heights stay comparable.

    The range grows immediately when a larger signal arrives and shrinks
    gradually, so the traces do not jump on every frame. A minimum span keeps a
    steady empty-seat signal flat instead of magnifying its noise.
    """

    def __init__(self, min_span_fraction: float = 0.25, padding_fraction: float = 0.08, decay: float = 0.15) -> None:
        self.min_span_fraction = min_span_fraction
        self.padding_fraction = padding_fraction
        self.decay = decay
        self.low: float | None = None
        self.high: float | None = None

    def target(self, values: np.ndarray) -> tuple[float, float] | None:
        values = np.asarray(values, dtype=float)
        if not values.size:
            return None
        low, high = float(values.min()), float(values.max())
        minimum = max(abs(low), abs(high), 1e-9) * self.min_span_fraction
        if high - low < minimum:
            middle = (low + high) / 2.0
            low, high = middle - minimum / 2.0, middle + minimum / 2.0
        padding = (high - low) * self.padding_fraction
        return low - padding, high + padding

    def update(self, values: np.ndarray) -> tuple[float, float]:
        target = self.target(values)
        if target is None:
            return (self.low, self.high) if self.low is not None else (0.0, 1.0)
        low, high = target
        if self.low is None or self.high is None:
            self.low, self.high = low, high
        else:
            self.low = low if low < self.low else self.low + (low - self.low) * self.decay
            self.high = high if high > self.high else self.high + (high - self.high) * self.decay
        return self.low, self.high


def format_signal(value: float, low: float, high: float) -> str:
    """Two decimals for voltage-like streams, integers for raw ADC counts."""
    return f"{value:.0f}" if max(abs(low), abs(high)) >= 100 else f"{value:.2f}"
