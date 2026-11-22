"""Versioned raw-signal preprocessing shared by training and every inference path.

Two independent, opt-in steps:

* ``despike_v1`` (training **and** inference): repairs isolated one-sample
  up-and-back excursions in the raw 16-channel series before time resampling.
  The rule is purely local (previous repaired sample, current sample, next raw
  sample) with fixed thresholds, so a recording and a live buffer produce the
  same repaired values away from their first/last sample.
* ``training_trim_start_seconds`` (training only): drops the first seconds of
  each labelled training recording (seat settling / serial start-up). It is not
  applied at inference, because a rolling live window has no recording start;
  evaluation and replay therefore see the untrimmed signal.

The default ``SignalPreprocessing()`` is the identity and is never written into
model bundles, so existing training runs and models are unchanged. Bundles that
carry a rule are validated fail-closed: an unknown version, altered parameter or
mismatched rule hash raises instead of silently running without the step.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from .posture_core import SENSOR_COLUMNS, validate_sensor_frame

SCHEMA = "signal_preprocessing_v1"
DESPIKE_V1 = "despike_v1"
DESPIKE_VERSIONS = (DESPIKE_V1,)
# Fixed before any model comparison from the label-free distribution of raw
# consecutive differences (99th percentile |diff| ~= 0.083 V over 725 training
# recordings). Values are the recorded sensor units (0-5 V ADC scale).
DESPIKE_V1_PARAMETERS = {
    "min_excursion": 0.10,
    "max_return_ratio": 0.5,
    "max_interval_seconds": 0.25,
}
MAX_TRIM_SECONDS = 5.0
BUNDLE_KEY = "signal_preprocessing"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def despike_v1(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return repaired float64 values and a boolean mask of repaired samples.

    Sample ``i`` (never the first or last) of a channel is replaced by the mean
    of its neighbours when all of these hold:

    * it lies on the same side of the previous (already repaired) sample and the
      next raw sample, by at least ``min_excursion`` on both sides;
    * the neighbours agree: ``|next - previous| <= max_return_ratio * excursion``
      (the signal returns, so level steps and ramps are never modified);
    * both adjacent sampling intervals are at most ``max_interval_seconds``.

    Processing is sequential in time so an alternating flicker (A, B, A, B) is
    repaired to a constant level instead of being phase-inverted.
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or time_s.shape != (len(source),):
        raise ValueError("despike_v1 requires time (N,) and values (N, channels).")
    if not np.isfinite(time_s).all() or not np.isfinite(source).all():
        raise ValueError("despike_v1 requires finite time and sensor values.")
    if len(time_s) > 1 and np.any(np.diff(time_s) <= 0):
        raise ValueError("despike_v1 requires strictly increasing time.")
    repaired = source.copy()
    mask = np.zeros(source.shape, dtype=bool)
    if len(source) < 3:
        return repaired, mask
    threshold = DESPIKE_V1_PARAMETERS["min_excursion"]
    ratio = DESPIKE_V1_PARAMETERS["max_return_ratio"]
    close = np.diff(time_s) <= DESPIKE_V1_PARAMETERS["max_interval_seconds"]
    for index in range(1, len(source) - 1):
        if not (close[index - 1] and close[index]):
            continue
        previous, current, following = repaired[index - 1], source[index], source[index + 1]
        left, right = current - previous, current - following
        excursion = np.minimum(np.abs(left), np.abs(right))
        spike = ((left * right) > 0) & (excursion >= threshold) & (np.abs(following - previous) <= ratio * excursion)
        if spike.any():
            repaired[index, spike] = 0.5 * (previous[spike] + following[spike])
            mask[index, spike] = True
    return repaired, mask


@dataclass(frozen=True)
class SignalPreprocessing:
    despike: str | None = None
    training_trim_start_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.despike is not None and self.despike not in DESPIKE_VERSIONS:
            raise ValueError(f"Unsupported despike version: {self.despike!r}")
        trim = self.training_trim_start_seconds
        if (isinstance(trim, (bool, np.bool_)) or not isinstance(trim, Real) or not np.isfinite(trim)
                or not 0 <= float(trim) <= MAX_TRIM_SECONDS):
            raise ValueError(f"training_trim_start_seconds must be between 0 and {MAX_TRIM_SECONDS:g}.")
        object.__setattr__(self, "training_trim_start_seconds", float(trim))

    @property
    def is_identity(self) -> bool:
        return self.despike is None and self.training_trim_start_seconds == 0

    def rule(self) -> dict[str, Any]:
        """JSON rule stored in bundles/manifests; includes its own SHA-256."""
        payload = {
            "schema": SCHEMA,
            "despike": None if self.despike is None else {
                "version": self.despike, "parameters": dict(DESPIKE_V1_PARAMETERS),
                "applies_to": "training and inference, raw samples before time resampling",
                "edges": "first and last sample of each supplied frame are never modified",
            },
            "training_trim_start_seconds": self.training_trim_start_seconds,
            "training_trim_scope": "labelled training recordings only; never applied at inference or evaluation",
            "order": "despike whole recording, then training trim, then windowing",
        }
        payload["rule_sha256"] = _digest(payload)
        return payload

    @classmethod
    def from_rule(cls, rule: Any) -> "SignalPreprocessing":
        if not isinstance(rule, dict) or rule.get("schema") != SCHEMA:
            raise ValueError("Unsupported signal preprocessing schema.")
        despike = rule.get("despike")
        version = None
        if despike is not None:
            if not isinstance(despike, dict):
                raise ValueError("Invalid despike rule.")
            version = despike.get("version")
            if version not in DESPIKE_VERSIONS or despike.get("parameters") != DESPIKE_V1_PARAMETERS:
                raise ValueError("Unsupported despike version or parameters.")
        result = cls(version, rule.get("training_trim_start_seconds", 0.0))
        if result.rule() != rule:
            raise ValueError("Signal preprocessing rule does not match its recorded hash.")
        return result

    @classmethod
    def from_bundle(cls, bundle: dict) -> "SignalPreprocessing":
        rule = bundle.get(BUNDLE_KEY)
        return cls() if rule is None else cls.from_rule(rule)

    def _despike_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        validate_sensor_frame(frame)
        repaired, mask = despike_v1(frame["elapsed_s"].to_numpy(dtype=np.float64),
                                    frame[SENSOR_COLUMNS].to_numpy(dtype=np.float64))
        if not mask.any():
            return frame, 0
        result = frame.copy()
        result[SENSOR_COLUMNS] = repaired
        return result, int(mask.sum())

    def apply_inference(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform used by replay, golden checks and live buffers."""
        if self.despike is None:
            return frame
        return self._despike_frame(frame)[0]

    def apply_training(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform for one labelled training recording (despike, then trim)."""
        result = self.apply_inference(frame)
        if self.training_trim_start_seconds == 0:
            return result
        time_s = result["elapsed_s"].to_numpy(dtype=np.float64)
        keep = (time_s - time_s[0]) >= self.training_trim_start_seconds - 1e-9
        if keep.sum() < 2:
            raise ValueError("Training trim leaves fewer than two samples.")
        return result.loc[keep].reset_index(drop=True)

    def count_repairs(self, frame: pd.DataFrame) -> int:
        if self.despike is None:
            return 0
        return self._despike_frame(frame)[1]


def bundle_entry(preprocessing: SignalPreprocessing | None) -> dict[str, Any]:
    """Bundle/config fields: empty for the identity so legacy outputs stay byte-compatible."""
    if preprocessing is None or preprocessing.is_identity:
        return {}
    return {BUNDLE_KEY: preprocessing.rule()}
