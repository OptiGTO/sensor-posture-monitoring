"""Personal reference calibration and nonblocking, clock-driven posture guidance.

This module never changes the samples supplied to a classification model. Personal
ratios are computed independently, after empty-seat compensation. Default decision
thresholds are provisional and must be validated on separate sensor recordings.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .posture_core import CLASS_NAMES, SENSOR_COLUMNS


PROFILE_SCHEMA_VERSION = 1
REGISTRATION_GUIDANCE = (
    "Sit upright with your back supported, feet flat, legs uncrossed, and weight "
    "distributed comfortably. Confirm this posture yourself before recording for "
    "10 seconds. Stable pressure alone cannot establish correct posture. "
    "Register again whenever the chair or sensor placement changes."
)


def _vector(values: Iterable[float], count: int = len(SENSOR_COLUMNS)) -> np.ndarray:
    try:
        result = np.asarray(tuple(values), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Sensor values must be finite numeric values.") from error
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {count} finite sensor values.")
    if np.any(result < 0):
        raise ValueError("Raw pressure sensor values must be nonnegative.")
    return result


@dataclass(frozen=True)
class CalibrationContext:
    sensor_channels: tuple[str, ...] = tuple(SENSOR_COLUMNS)
    sensor_configuration: str = "pressure-4x4-v1"
    chair_id: str = "default-chair"
    placement_id: str = "default-placement"
    empty_seat_center: tuple[float, ...] = (0.0,) * len(SENSOR_COLUMNS)
    pressure_direction: str = "absolute_delta"

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_channels", tuple(self.sensor_channels))
        object.__setattr__(self, "empty_seat_center", tuple(float(x) for x in self.empty_seat_center))
        if self.sensor_channels != tuple(SENSOR_COLUMNS):
            raise ValueError("Personal calibration requires the ordered 4x4 pressure channels.")
        if not all(isinstance(x, str) and x.strip() for x in (self.sensor_configuration, self.chair_id, self.placement_id)):
            raise ValueError("Sensor configuration, chair, and placement identifiers are required.")
        _vector(self.empty_seat_center)
        if self.pressure_direction not in {"absolute_delta", "increasing", "decreasing"}:
            raise ValueError("Unknown pressure direction.")

    def compatibility_key(self) -> tuple[object, ...]:
        # A fresh empty-seat check can change the offset without changing the chair.
        return (self.sensor_channels, self.sensor_configuration, self.chair_id, self.placement_id, self.pressure_direction)

    def pressure(self, raw: np.ndarray) -> np.ndarray:
        difference = raw - np.asarray(self.empty_seat_center)
        if self.pressure_direction == "increasing":
            return np.maximum(difference, 0.0)
        if self.pressure_direction == "decreasing":
            return np.maximum(-difference, 0.0)
        return np.abs(difference)


@dataclass(frozen=True)
class CalibrationConfig:
    duration_s: float = 10.0
    minimum_samples: int = 60
    max_gap_s: float = 1.0
    max_ratio_step: float = 0.12
    max_ratio_spread: float = 0.10
    max_total_cv: float = 0.10
    max_total_step_fraction: float = 0.20
    minimum_total_pressure: float = 1e-6
    ratio_tolerance_floor: float = 0.008

    def __post_init__(self) -> None:
        numeric = [v for k, v in asdict(self).items() if k != "minimum_samples"]
        if not np.isfinite(numeric).all() or min(numeric) <= 0 or self.minimum_samples < 2:
            raise ValueError("Calibration timing and stability settings must be positive and finite.")


@dataclass(frozen=True)
class CalibrationProfile:
    user_id: str
    profile_version: str
    created_at: str
    context: CalibrationContext
    pressure_center: tuple[float, ...]
    pressure_lower: tuple[float, ...]
    pressure_upper: tuple[float, ...]
    ratio_center: tuple[float, ...]
    ratio_tolerance: tuple[float, ...]
    ratio_mad: tuple[float, ...]
    stability_metrics: dict[str, float]
    guidance_confirmed: bool
    empty_seat_checked: bool
    schema_version: int = PROFILE_SCHEMA_VERSION
    calibration_method: str = "personal-pressure-ratio-v1"
    thresholds_status: str = "provisional-requires-sensor-validation"
    calibration_settings: dict[str, object] = field(default_factory=lambda: asdict(CalibrationConfig()))

    def validate(self, context: CalibrationContext | None = None) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION or self.calibration_method != "personal-pressure-ratio-v1":
            raise ValueError("Unsupported calibration version; register the reference again.")
        if not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("A user identifier is required.")
        if not isinstance(self.profile_version, str) or not re.fullmatch(r"[a-f0-9]{32}", self.profile_version):
            raise ValueError("Invalid calibration profile version; register again.")
        try:
            timestamp = datetime.fromisoformat(self.created_at)
            if timestamp.tzinfo is None:
                raise ValueError("Registration time must include a timezone.")
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid calibration registration time.") from error
        if not self.guidance_confirmed or not self.empty_seat_checked:
            raise ValueError("Correct-posture confirmation and a separate empty-seat check are required.")
        center, lower, upper, ratios, tolerance, ratio_mad = (
            _vector(getattr(self, name)) for name in
            ("pressure_center", "pressure_lower", "pressure_upper", "ratio_center", "ratio_tolerance", "ratio_mad")
        )
        if np.any(lower > center) or np.any(center > upper):
            raise ValueError("Invalid pressure variation range.")
        if not np.isclose(ratios.sum(), 1.0, atol=1e-6) or np.any(tolerance <= 0):
            raise ValueError("Invalid personal pressure ratios or allowable deviations.")
        required = {"duration_s", "sample_count", "total_cv", "max_ratio_step", "ratio_spread", "max_total_step_fraction"}
        if not isinstance(self.stability_metrics, dict) or not required.issubset(self.stability_metrics):
            raise ValueError("Missing calibration stability metrics.")
        if not np.isfinite(list(self.stability_metrics.values())).all() or min(self.stability_metrics.values()) < 0:
            raise ValueError("Invalid calibration stability metrics.")
        config = CalibrationConfig(**self.calibration_settings)
        expected_tolerance = np.maximum(3 * 1.4826 * ratio_mad, config.ratio_tolerance_floor)
        if not np.allclose(tolerance, expected_tolerance, atol=1e-10, rtol=1e-8):
            raise ValueError("Personal ratio tolerances do not match the recorded registration variation.")
        metrics = self.stability_metrics
        if (metrics["duration_s"] + 1e-9 < config.duration_s or metrics["sample_count"] < config.minimum_samples
                or metrics["total_cv"] > config.max_total_cv or metrics["max_ratio_step"] > config.max_ratio_step
                or metrics["ratio_spread"] > config.max_ratio_spread or metrics["max_total_step_fraction"] > config.max_total_step_fraction):
            raise ValueError("The calibration profile does not meet its registration stability requirements.")
        if context is not None and self.context.compatibility_key() != context.compatibility_key():
            raise ValueError("Sensor configuration, chair, or placement changed; register the reference again.")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, object]) -> "CalibrationProfile":
        try:
            data = dict(record)
            data["context"] = CalibrationContext(**data["context"])
            for name in ("pressure_center", "pressure_lower", "pressure_upper", "ratio_center", "ratio_tolerance", "ratio_mad"):
                data[name] = tuple(data[name])
            profile = cls(**data)
            profile.validate()
            return profile
        except (TypeError, KeyError, ValueError) as error:
            raise ValueError(f"Invalid calibration profile; register again: {error}") from error


class CalibrationSession:
    """A registration attempt fails as a whole if input or stability checks fail."""

    def __init__(self, user_id: str, context: CalibrationContext, *, guidance_confirmed: bool,
                 empty_seat_checked: bool, config: CalibrationConfig | None = None) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("Select a user before registering a correct posture.")
        if not guidance_confirmed or not empty_seat_checked:
            raise ValueError("Confirm the posture guidance and perform the separate empty-seat check first.")
        self.user_id = user_id.strip()
        self.context = context
        self.config = config or CalibrationConfig()
        self._timestamps: list[float] = []
        self._raw: list[np.ndarray] = []
        self._ratios: list[np.ndarray] = []
        self._totals: list[float] = []
        self.failed_reason: str | None = None
        self._profile: CalibrationProfile | None = None

    @property
    def duration_s(self) -> float:
        return self._timestamps[-1] - self._timestamps[0] if len(self._timestamps) >= 2 else 0.0

    @property
    def progress(self) -> float:
        return min(1.0, self.duration_s / self.config.duration_s)

    def _fail(self, reason: str) -> None:
        self.failed_reason = reason
        raise ValueError(f"{reason} Repeat the correct-posture registration.")

    def push(self, timestamp_s: float, values: Iterable[float], *, seated: bool | None = True,
             valid: bool = True, moving: bool = False) -> float:
        if self.failed_reason:
            self._fail(self.failed_reason)
        if self._profile is not None:
            raise ValueError("This registration is complete; start a new session to re-register.")
        if not valid or seated is not True or moving:
            self._fail("Movement, an empty seat, or invalid sensor input was detected.")
        try:
            timestamp_s = float(timestamp_s)
            raw = _vector(values)
            if not math.isfinite(timestamp_s):
                raise ValueError("Invalid sensor timestamp.")
            if self._timestamps and not 0 < timestamp_s - self._timestamps[-1] <= self.config.max_gap_s:
                raise ValueError("Sensor timestamps are out of order or contain a gap.")
            pressure = self.context.pressure(raw)
            total = float(pressure.sum())
            if total <= self.config.minimum_total_pressure:
                raise ValueError("Insufficient pressure above the empty-seat baseline.")
            ratios = pressure / total
            if self._ratios:
                if np.abs(ratios - self._ratios[-1]).sum() > self.config.max_ratio_step:
                    raise ValueError("Pressure distribution moved during registration.")
                if abs(total - self._totals[-1]) / self._totals[-1] > self.config.max_total_step_fraction:
                    raise ValueError("Total pressure moved during registration.")
        except (TypeError, ValueError) as error:
            self._fail(str(error))
        self._timestamps.append(timestamp_s)
        self._raw.append(raw)
        self._ratios.append(ratios)
        self._totals.append(total)
        return self.progress

    def finish(self) -> CalibrationProfile:
        if self.failed_reason:
            self._fail(self.failed_reason)
        if self._profile is not None:
            return self._profile
        if self.duration_s + 1e-9 < self.config.duration_s or len(self._raw) < self.config.minimum_samples:
            raise ValueError(f"Registration needs at least {self.config.duration_s:g} seconds and {self.config.minimum_samples} valid samples.")
        raw, ratios, totals = np.stack(self._raw), np.stack(self._ratios), np.asarray(self._totals)
        center = np.median(ratios, axis=0)
        center /= center.sum()
        total_cv = float(np.std(totals) / np.mean(totals))
        ratio_spread = float(np.percentile(np.abs(ratios - center).sum(axis=1), 95))
        metrics = {
            "duration_s": self.duration_s, "sample_count": len(raw), "total_cv": total_cv,
            "max_ratio_step": float(np.max(np.abs(np.diff(ratios, axis=0)).sum(axis=1))),
            "ratio_spread": ratio_spread,
            "max_total_step_fraction": float(np.max(np.abs(np.diff(totals)) / totals[:-1])),
        }
        if total_cv > self.config.max_total_cv or ratio_spread > self.config.max_ratio_spread:
            self._fail("The recorded posture was not stable enough.")
        ratio_mad = np.median(np.abs(ratios - center), axis=0)
        tolerance = np.maximum(3 * 1.4826 * ratio_mad, self.config.ratio_tolerance_floor)
        self._profile = CalibrationProfile(
            user_id=self.user_id, profile_version=uuid.uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(), context=self.context,
            pressure_center=tuple(np.median(raw, axis=0)), pressure_lower=tuple(np.percentile(raw, 5, axis=0)),
            pressure_upper=tuple(np.percentile(raw, 95, axis=0)), ratio_center=tuple(center),
            ratio_tolerance=tuple(tolerance), ratio_mad=tuple(ratio_mad), stability_metrics=metrics,
            guidance_confirmed=True, empty_seat_checked=True,
            calibration_settings=asdict(self.config),
        )
        self._profile.validate()
        return self._profile


class CalibrationStore:
    """Versioned registration history plus an atomically replaced current profile."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _user_directory(self, user_id: str) -> Path:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("A user identifier is required.")
        return self.root / hashlib.sha256(user_id.strip().encode("utf-8")).hexdigest()

    def save(self, profile: CalibrationProfile) -> Path:
        profile.validate()
        directory = self._user_directory(profile.user_id)
        directory.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(profile.as_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        archive = directory / f"{profile.profile_version}.json"
        if archive.exists():
            if archive.read_text(encoding="utf-8") != serialized:
                raise ValueError("A different profile already uses this version identifier.")
        else:
            with archive.open("x", encoding="utf-8") as output:
                output.write(serialized)
        current = directory / "current.json"
        temporary = directory / f"current-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as output:
                output.write(serialized)
            temporary.replace(current)
        finally:
            temporary.unlink(missing_ok=True)
        return current

    def load(self, user_id: str, context: CalibrationContext | None = None) -> CalibrationProfile | None:
        path = self._user_directory(user_id) / "current.json"
        if not path.is_file():
            return None
        try:
            profile = CalibrationProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if profile.user_id != user_id.strip():
                raise ValueError("The stored reference belongs to a different user.")
            profile.validate(context)
            return profile
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot use the saved reference; register again: {error}") from error

    def list_profiles(self) -> list[CalibrationProfile]:
        profiles = []
        for path in sorted(self.root.glob("*/current.json")):
            try:
                profiles.append(CalibrationProfile.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
        return sorted(profiles, key=lambda profile: profile.user_id)


@dataclass(frozen=True)
class AlertSettings:
    mode: str = "deployment"
    initial_alert_s: float = 60.0
    realert_interval_s: float = 120.0
    correct_return_confirmation_s: float = 2.0
    uncertainty_tolerance_s: float = 3.0

    def __post_init__(self) -> None:
        timings = (self.initial_alert_s, self.realert_interval_s, self.correct_return_confirmation_s, self.uncertainty_tolerance_s)
        if self.mode not in {"development", "deployment"} or not np.isfinite(timings).all() or min(timings) <= 0:
            raise ValueError("Alert settings require a known mode and positive finite timings.")

    @classmethod
    def for_mode(cls, mode: str, **overrides: float) -> "AlertSettings":
        if mode not in {"development", "deployment"}:
            raise ValueError("Alert mode must be development or deployment.")
        return cls(mode=mode, **{
            "initial_alert_s": 12.0 if mode == "development" else 60.0,
            "realert_interval_s": 24.0 if mode == "development" else 120.0,
            **overrides,
        })

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GuidanceConfig:
    smoothing_s: float = 0.5
    minimum_samples: int = 3
    max_gap_s: float = 1.0
    leave_correct_score: float = 1.5
    return_correct_score: float = 1.0
    consecutive_observations: int = 3
    movement_ratio_range: float = 0.15
    movement_total_fraction: float = 0.25
    model_confidence_threshold: float = 0.45
    max_model_age_s: float = 2.0
    minimum_total_pressure: float = 1e-6

    def __post_init__(self) -> None:
        numeric = [v for k, v in asdict(self).items() if k not in {"minimum_samples", "consecutive_observations"}]
        if not np.isfinite(numeric).all() or min(numeric) <= 0:
            raise ValueError("Guidance thresholds and timing must be positive and finite.")
        if self.minimum_samples < 1 or self.consecutive_observations < 1 or self.leave_correct_score <= self.return_correct_score:
            raise ValueError("Invalid observation counts or reference hysteresis thresholds.")
        if self.model_confidence_threshold > 1:
            raise ValueError("Model confidence threshold must be in (0, 1].")


@dataclass(frozen=True)
class AlertDecision:
    incorrect_duration_s: float
    popup_action: str | None
    popup_open: bool
    returned_to_correct: bool
    events: tuple[str, ...]


class PostureAlertTimer:
    """Pure state/time policy; callers display a nonmodal popup only on `show`."""

    UNCERTAIN_STATES = {"uncertain", "movement", "insufficient", "classifying"}
    RESET_STATES = {"empty", "input_error", "disconnected", "profile_required"}

    def __init__(self, settings: AlertSettings | None = None) -> None:
        self.settings = settings or AlertSettings()
        self.incorrect_duration_s = 0.0
        self.incorrect_since_s: float | None = None
        self.last_popup_s: float | None = None
        self.popup_open = False
        self._last_timestamp_s: float | None = None
        self._previous_state: str | None = None
        self._uncertain_since_s: float | None = None
        self._correct_since_s: float | None = None

    def _reset_duration(self) -> None:
        self.incorrect_duration_s = 0.0
        self.incorrect_since_s = None
        self._uncertain_since_s = None
        self._correct_since_s = None

    def close_popup(self, timestamp_s: float | None = None) -> None:
        # Deliberately leave duration and most recent display time untouched.
        self.popup_open = False

    def update(self, timestamp_s: float, state: str, *, incorrect_since_s: float | None = None) -> AlertDecision:
        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s) or (self._last_timestamp_s is not None and timestamp_s < self._last_timestamp_s):
            raise ValueError("Alert timestamps must be finite and monotonic.")
        if state not in self.UNCERTAIN_STATES | self.RESET_STATES | {"correct", "incorrect"}:
            raise ValueError(f"Unknown posture state: {state}")
        elapsed = 0.0 if self._last_timestamp_s is None else timestamp_s - self._last_timestamp_s
        events: list[str] = []
        popup_action = None
        returned = False
        if (state not in self.UNCERTAIN_STATES and self._uncertain_since_s is not None
                and timestamp_s - self._uncertain_since_s >= self.settings.uncertainty_tolerance_s):
            if self.incorrect_since_s is not None:
                events.append("uncertainty_timeout")
            self._reset_duration()
            if self.popup_open:
                self.popup_open = False
                popup_action = "dismiss"
        if state in self.RESET_STATES:
            if self.incorrect_since_s is not None:
                events.append("incorrect_duration_reset")
            self._reset_duration()
            if self.popup_open:
                self.popup_open = False
                popup_action = "dismiss"
        elif state in self.UNCERTAIN_STATES:
            self._correct_since_s = None
            if self._uncertain_since_s is None:
                self._uncertain_since_s = timestamp_s
            if timestamp_s - self._uncertain_since_s >= self.settings.uncertainty_tolerance_s:
                if self.incorrect_since_s is not None:
                    events.append("uncertainty_timeout")
                self._reset_duration()
                self._uncertain_since_s = timestamp_s
                if self.popup_open:
                    self.popup_open = False
                    popup_action = "dismiss"
        elif state == "correct":
            self._uncertain_since_s = None
            if self._correct_since_s is None:
                self._correct_since_s = timestamp_s
            if timestamp_s - self._correct_since_s >= self.settings.correct_return_confirmation_s:
                returned = True
                if self.incorrect_since_s is not None:
                    events.append("returned_to_correct")
                    self._reset_duration()
                    self._correct_since_s = timestamp_s - self.settings.correct_return_confirmation_s
                if self.popup_open:
                    self.popup_open = False
                    popup_action = "dismiss"
        else:
            self._correct_since_s = None
            self._uncertain_since_s = None
            if self.incorrect_since_s is None:
                onset = timestamp_s if incorrect_since_s is None else float(incorrect_since_s)
                if not math.isfinite(onset) or onset > timestamp_s:
                    raise ValueError("Incorrect-posture onset must not be later than this observation.")
                self.incorrect_since_s = onset
                self.incorrect_duration_s = timestamp_s - onset
                events.append("incorrect_posture_started")
            elif self._previous_state == "incorrect":
                self.incorrect_duration_s += elapsed
            if (self.incorrect_duration_s + 1e-9 >= self.settings.initial_alert_s
                    and not self.popup_open
                    and (self.last_popup_s is None or timestamp_s - self.last_popup_s + 1e-9 >= self.settings.realert_interval_s)):
                self.popup_open = True
                self.last_popup_s = timestamp_s
                popup_action = "show"
                events.append("popup_shown")
        self._last_timestamp_s = timestamp_s
        self._previous_state = state
        if popup_action == "dismiss":
            events.append("popup_dismissed")
        return AlertDecision(self.incorrect_duration_s, popup_action, self.popup_open, returned, tuple(events))


_CORRECTION = {
    CLASS_NAMES[1]: "Gently bring your torso upright and support your back.",
    CLASS_NAMES[2]: "Bring your torso back to a comfortable upright position.",
    CLASS_NAMES[3]: "Bring your weight back toward the center of the seat.",
    CLASS_NAMES[4]: "Bring your weight back toward the center of the seat.",
    CLASS_NAMES[5]: "Uncross your left leg and place both feet flat on the floor.",
    CLASS_NAMES[6]: "Uncross your right leg and place both feet flat on the floor.",
}


@dataclass(frozen=True)
class GuidanceEvent:
    timestamp_s: float
    state: str
    message: str
    posture: str | None
    reference_score: float | None
    incorrect_duration_s: float
    popup_action: str | None
    popup_open: bool
    correction_guidance: str
    profile_version: str | None
    mode: str
    alert_settings: dict[str, object]
    events: tuple[str, ...] = ()
    detail: str | None = None
    gate_score: float | None = None
    detail_score: float | None = None
    stage2_executed: bool = False
    model_identity: dict | None = None
    window_timestamp_s: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class GuidanceEngine:
    """Evaluate pressure distribution independently from model posture typing."""

    def __init__(self, profile: CalibrationProfile | None, settings: AlertSettings | None = None,
                 config: GuidanceConfig | None = None, context: CalibrationContext | None = None) -> None:
        if profile is not None:
            profile.validate(context)
        self.profile = profile
        self.context = context or (profile.context if profile else CalibrationContext())
        self.settings = settings or AlertSettings()
        self.config = config or GuidanceConfig()
        self.timer = PostureAlertTimer(self.settings)
        self._frames: deque[tuple[float, np.ndarray, float]] = deque()
        self._last_timestamp_s: float | None = None
        self._candidate: str | None = None
        self._candidate_count = 0
        self._candidate_since_s: float | None = None
        self._confirmed: str | None = None
        self._last_state: str | None = None

    def close_popup(self, timestamp_s: float | None = None) -> None:
        self.timer.close_popup(timestamp_s)

    def _clear_observations(self) -> None:
        self._frames.clear()
        self._candidate = None
        self._candidate_count = 0
        self._candidate_since_s = None
        self._confirmed = None

    def update(self, timestamp_s: float, values: Iterable[float], *, seated: bool | None = True,
               valid: bool = True, connected: bool = True, model_posture: str | None = None,
               model_confidence: float | None = None, model_timestamp_s: float | None = None,
               moving: bool = False) -> GuidanceEvent:
        try:
            timestamp_s = float(timestamp_s)
            invalid_timestamp = not math.isfinite(timestamp_s) or (self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s)
        except (TypeError, ValueError):
            invalid_timestamp = True
        if invalid_timestamp:
            # There is no trustworthy elapsed-time increment. Use the most recent
            # valid instant solely to deliver an error/reset/dismiss event.
            timestamp_s = self._last_timestamp_s if self._last_timestamp_s is not None else 0.0
        gap = self._last_timestamp_s is not None and timestamp_s - self._last_timestamp_s > self.config.max_gap_s
        if not invalid_timestamp:
            self._last_timestamp_s = timestamp_s
        score = None
        detail = None
        onset = None
        state = "insufficient"
        pending_transition = False
        try:
            raw = _vector(values)
        except ValueError as error:
            valid = False
            detail = str(error)
            raw = np.zeros(len(SENSOR_COLUMNS))
        if not connected or not valid or gap or invalid_timestamp:
            state = "input_error"
            detail = detail or ("Invalid or out-of-order sensor timestamp." if invalid_timestamp else "Sensor disconnected." if not connected else "Sensor input gap." if gap else "Invalid sensor input.")
            self._clear_observations()
        elif seated is False:
            state = "empty"
            self._clear_observations()
        elif self.profile is None:
            state = "profile_required"
            self._clear_observations()
        elif seated is None:
            detail = "Waiting for a confirmed seating state."
            self._clear_observations()
        else:
            pressure = self.context.pressure(raw)
            total = float(pressure.sum())
            if total <= self.config.minimum_total_pressure:
                state, detail = "input_error", "Insufficient pressure above the empty-seat baseline."
                self._clear_observations()
            else:
                self._frames.append((timestamp_s, pressure / total, total))
                while self._frames and self._frames[0][0] < timestamp_s - self.config.smoothing_s:
                    self._frames.popleft()
                if len(self._frames) >= self.config.minimum_samples:
                    ratios = np.stack([entry[1] for entry in self._frames])
                    totals = np.asarray([entry[2] for entry in self._frames])
                    ratio_movement = float(np.ptp(ratios, axis=0).sum())
                    total_movement = float(np.ptp(totals) / np.mean(totals))
                    if moving or ratio_movement > self.config.movement_ratio_range or total_movement > self.config.movement_total_fraction:
                        state = "movement"
                    else:
                        difference = np.abs(ratios.mean(axis=0) - np.asarray(self.profile.ratio_center))
                        score = float(np.max(difference / np.asarray(self.profile.ratio_tolerance)))
                        if score <= self.config.return_correct_score:
                            candidate = "correct"
                        elif score >= self.config.leave_correct_score:
                            candidate = "incorrect"
                        elif self._confirmed == "correct":
                            candidate = "correct"  # Leave only through the outer threshold.
                        elif self._confirmed == "incorrect":
                            candidate = "incorrect"  # Return only through the inner threshold.
                        else:
                            candidate = "uncertain"
                        if candidate == "uncertain":
                            state = "uncertain"
                        else:
                            if candidate != self._candidate:
                                self._candidate, self._candidate_count, self._candidate_since_s = candidate, 1, timestamp_s
                            else:
                                self._candidate_count += 1
                            if self._candidate_count >= self.config.consecutive_observations:
                                self._confirmed = candidate
                                state = candidate
                                if state == "incorrect" and self._candidate_count == self.config.consecutive_observations:
                                    onset = self._candidate_since_s
                            else:
                                state = "uncertain"
                                pending_transition = True
                elif moving:
                    state = "movement"
        if state in {"movement", "insufficient", "uncertain"}:
            # Preserve a partially confirmed candidate only across its own consecutive
            # observations; actual movement and missing input interrupt confirmation.
            if not pending_transition:
                self._candidate = None
                self._candidate_count = 0
                self._candidate_since_s = None
        decision = self.timer.update(timestamp_s, state, incorrect_since_s=onset)
        if state == "correct" and self.timer.incorrect_since_s is not None and not decision.returned_to_correct:
            display_state = "uncertain"
            detail = "Confirming return to the personal reference posture."
        else:
            display_state = state
        posture = None
        if display_state == "correct":
            posture = CLASS_NAMES[0]
        elif display_state == "incorrect":
            confidence_valid = model_confidence is not None and math.isfinite(model_confidence) and model_confidence >= self.config.model_confidence_threshold
            model_fresh = model_timestamp_s is None or (math.isfinite(model_timestamp_s) and 0 <= timestamp_s - model_timestamp_s <= self.config.max_model_age_s)
            if model_posture in CLASS_NAMES[1:] and confidence_valid and model_fresh:
                posture = model_posture
        messages = {
            "correct": "Posture is correct.",
            "incorrect": posture or "Deviated from reference posture · Verifying posture type.",
            "uncertain": "Classifying", "movement": "Classifying · Movement detected.",
            "insufficient": "Classifying · Insufficient input.", "empty": "Empty seat · Classification and alerts stopped.",
            "input_error": "Input error · Classification and alerts stopped.",
            "profile_required": "Register or load a correct-posture reference for this user.",
        }
        events = list(decision.events)
        if display_state != self._last_state:
            events.insert(0, f"state_transition:{self._last_state or 'initial'}->{display_state}")
        self._last_state = display_state
        return GuidanceEvent(
            timestamp_s=timestamp_s, state=display_state, message=messages[display_state], posture=posture,
            reference_score=score, incorrect_duration_s=decision.incorrect_duration_s,
            popup_action=decision.popup_action, popup_open=decision.popup_open,
            correction_guidance=_CORRECTION.get(posture, "Return to your registered upright posture, uncross your legs, and place both feet flat on the floor."),
            profile_version=self.profile.profile_version if self.profile else None,
            mode=self.settings.mode, alert_settings=self.settings.as_dict(), events=tuple(events), detail=detail,
        )
