"""Per-sensor output calibration from a recorded finger-press comparison.

The measurement file (``hardware/calibration/finger_press_calibration.json``)
holds, for each of the 16 sensors, the range seen while pressing the sensor
centre with a finger at roughly constant maximal force. It is expected to be
replaced by newer measurements; every calibration carries the SHA-256 of its
source file so a model or experiment can record exactly which values it used.

Two transforms are defined, both applied to raw sensor values before any
feature extraction:

* ``gain``: ``x * g_c`` with ``g_c = median(m) / m_c`` (m = representative
  pressed value). This is the transform of the 2026-09-23 sensor_gain study.
* ``offset_gain``: ``b_c + (x - b_c) * g_c`` with
  ``g_c = median(m - b) / (m_c - b_c)`` where ``b`` is the empty-seat output.
  The unloaded level stays fixed and only the load response is equalized.

Neither transform clips, reorders or quantizes values.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .posture_core import SENSOR_COLUMNS

CALIBRATION_MODES = ("none", "gain", "offset_gain")
DEFAULT_MEASUREMENT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "calibration" / "finger_press_calibration.json"


def _finite_positive(value: object, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{what}: 숫자가 아닙니다 ({value!r})") from error
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{what}: 양의 유한한 값이어야 합니다 ({value!r})")
    return number


def representative_values(measurement: dict) -> np.ndarray:
    """Pressed output per sensor in SENSOR_COLUMNS order."""
    sensors = measurement.get("sensors")
    if not isinstance(sensors, dict) or set(sensors) != set(SENSOR_COLUMNS):
        raise ValueError("sensors에는 R11~R44 16개 센서가 정확히 있어야 합니다.")
    values = []
    for name in SENSOR_COLUMNS:
        entry = sensors[name]
        readings = entry.get("readings")
        if readings is not None:
            if not isinstance(readings, list) or not readings:
                raise ValueError(f"{name}.readings는 비어 있지 않은 목록이어야 합니다.")
            values.append(float(np.mean([_finite_positive(v, f"{name}.readings") for v in readings])))
            continue
        low = _finite_positive(entry.get("min"), f"{name}.min")
        high = _finite_positive(entry.get("max"), f"{name}.max")
        if low > high:
            raise ValueError(f"{name}: min이 max보다 큽니다.")
        values.append((low + high) / 2)
    return np.asarray(values, dtype=np.float64)


def empty_seat_baseline(measurement: dict) -> np.ndarray:
    values = (measurement.get("empty_seat_baseline") or {}).get("values")
    if not isinstance(values, dict) or set(values) != set(SENSOR_COLUMNS):
        raise ValueError("empty_seat_baseline.values에는 R11~R44 16개 값이 필요합니다.")
    return np.asarray([_finite_positive(values[name], f"baseline.{name}") for name in SENSOR_COLUMNS])


@dataclass(frozen=True)
class SensorCalibration:
    mode: str
    gains: tuple[float, ...]
    offsets: tuple[float, ...]
    calibration_id: str
    source_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in CALIBRATION_MODES:
            raise ValueError(f"지원하지 않는 보정 방식: {self.mode}")
        gains, offsets = np.asarray(self.gains, float), np.asarray(self.offsets, float)
        if gains.shape != (16,) or offsets.shape != (16,):
            raise ValueError("gains와 offsets는 16개여야 합니다.")
        if not np.isfinite(gains).all() or not np.isfinite(offsets).all() or np.any(gains <= 0):
            raise ValueError("gains는 양의 유한값, offsets는 유한값이어야 합니다.")
        if self.mode == "none" and (np.any(gains != 1) or np.any(offsets != 0)):
            raise ValueError("none 방식은 항등 변환이어야 합니다.")

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Calibrate an array whose last axis is the 16 sensors in SENSOR_COLUMNS order."""
        x = np.asarray(values, dtype=np.float64)
        if x.shape[-1:] != (16,) or not np.isfinite(x).all():
            raise ValueError("마지막 축이 16채널인 유한한 센서값이 필요합니다.")
        if self.mode == "none":
            return x.astype(np.float32)
        offsets, gains = np.asarray(self.offsets), np.asarray(self.gains)
        return (offsets + (x - offsets) * gains).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mode": self.mode, "calibration_id": self.calibration_id, "source_sha256": self.source_sha256,
                "sensor_columns": list(SENSOR_COLUMNS), "gains": list(self.gains), "offsets": list(self.offsets)}

    @classmethod
    def from_dict(cls, payload: dict) -> "SensorCalibration":
        if list(payload.get("sensor_columns", SENSOR_COLUMNS)) != list(SENSOR_COLUMNS):
            raise ValueError("보정 센서 순서가 R11~R44 행 우선 순서와 다릅니다.")
        return cls(mode=str(payload["mode"]), gains=tuple(map(float, payload["gains"])),
                   offsets=tuple(map(float, payload["offsets"])),
                   calibration_id=str(payload["calibration_id"]), source_sha256=str(payload["source_sha256"]))


def identity_calibration() -> SensorCalibration:
    return SensorCalibration("none", (1.0,) * 16, (0.0,) * 16, "none", "")


def build_calibration(measurement: dict, mode: str, source_sha256: str = "") -> SensorCalibration:
    if mode not in CALIBRATION_MODES:
        raise ValueError(f"지원하지 않는 보정 방식: {mode}")
    if mode == "none":
        return identity_calibration()
    pressed = representative_values(measurement)
    if mode == "gain":
        offsets = np.zeros(16)
    else:
        offsets = empty_seat_baseline(measurement)
        if np.any(pressed <= offsets):
            raise ValueError("눌렀을 때 값이 빈 좌석 기준 이하인 센서가 있어 offset_gain을 만들 수 없습니다.")
    response = pressed - offsets
    gains = np.median(response) / response
    return SensorCalibration(mode, tuple(map(float, gains)), tuple(map(float, offsets)),
                             str(measurement.get("calibration_id", "")), source_sha256)


def load_measurement(path: str | Path = DEFAULT_MEASUREMENT_PATH) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    measurement = json.loads(raw.decode("utf-8"))
    if not isinstance(measurement, dict) or measurement.get("schema_version") != 1:
        raise ValueError("보정 측정 파일 schema_version=1이 필요합니다.")
    if not str(measurement.get("calibration_id", "")).strip():
        raise ValueError("calibration_id가 필요합니다.")
    representative_values(measurement)
    return measurement, hashlib.sha256(raw).hexdigest()


def load_calibration(mode: str, path: str | Path = DEFAULT_MEASUREMENT_PATH) -> SensorCalibration:
    if mode == "none":
        return identity_calibration()
    measurement, digest = load_measurement(path)
    return build_calibration(measurement, mode, digest)
