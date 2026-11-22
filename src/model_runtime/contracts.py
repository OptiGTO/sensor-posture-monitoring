"""Explicit contracts shared by model discovery, inference, and sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
from typing import Any, Literal


WindowPolicy = Literal["fixed", "discrete", "range"]


@dataclass(frozen=True)
class WindowContract:
    policy: WindowPolicy
    values: tuple[float, ...] = ()
    minimum_seconds: float | None = None
    maximum_seconds: float | None = None
    step_seconds: float | None = None
    target_sample_rule: str = "round(window_seconds * sampling_rate)"
    padding_policy: str = "reject"

    def __post_init__(self) -> None:
        if self.policy not in {"fixed", "discrete", "range"}:
            raise ValueError("지원하지 않는 윈도우 정책입니다.")
        if self.policy in {"fixed", "discrete"} and (
            any(not math.isfinite(value) or value <= 0 for value in self.values) or len(set(self.values)) != len(self.values)
        ):
            raise ValueError("윈도우 길이는 양수이고 중복이 없어야 합니다.")
        if self.policy == "fixed" and len(self.values) != 1:
            raise ValueError("fixed 창 계약은 정확히 하나의 길이를 가져야 합니다.")
        if self.policy == "discrete" and not self.values:
            raise ValueError("discrete 창 계약은 하나 이상의 길이를 가져야 합니다.")
        if self.policy == "range":
            if self.minimum_seconds is None or self.maximum_seconds is None or self.step_seconds is None:
                raise ValueError("range 창 계약에는 최소·최대·간격이 필요합니다.")
            if not all(math.isfinite(value) for value in (self.minimum_seconds, self.maximum_seconds, self.step_seconds)) or self.minimum_seconds <= 0 or self.maximum_seconds < self.minimum_seconds or self.step_seconds <= 0:
                raise ValueError("range 창 계약 값이 유효하지 않습니다.")

    def validate(self, seconds: float) -> float:
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("윈도우 길이는 양수여야 합니다.")
        if self.policy in {"fixed", "discrete"}:
            if not any(abs(seconds - value) <= 1e-9 for value in self.values):
                raise ValueError("선택한 모델에서 검증되지 않은 윈도우 길이입니다.")
            return seconds
        assert self.minimum_seconds is not None and self.maximum_seconds is not None and self.step_seconds is not None
        offset = (seconds - self.minimum_seconds) / self.step_seconds
        if not self.minimum_seconds <= seconds <= self.maximum_seconds or abs(offset - round(offset)) > 1e-9:
            raise ValueError("선택한 모델에서 검증되지 않은 윈도우 길이입니다.")
        return seconds

    def target_samples(self, seconds: float, sampling_rate: float) -> int:
        self.validate(seconds)
        return int(round(seconds * sampling_rate))

    @property
    def default_seconds(self) -> float:
        if self.policy in {"fixed", "discrete"}:
            return self.values[0]
        assert self.minimum_seconds is not None
        return self.minimum_seconds

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"policy": self.policy, "target_sample_rule": self.target_sample_rule, "padding_policy": self.padding_policy}
        if self.policy in {"fixed", "discrete"}:
            result["values_seconds"] = list(self.values)
        else:
            result.update({"minimum_seconds": self.minimum_seconds, "maximum_seconds": self.maximum_seconds, "step_seconds": self.step_seconds})
        return result


@dataclass(frozen=True)
class ModelDescriptor:
    artifact_id: str
    display_name: str
    version: str
    backend: str
    model_family: str
    artifact_path: Path
    manifest_path: Path
    artifact_hash: str
    manifest_hash: str
    labels: tuple[str, ...]
    input_channels: tuple[str, ...]
    sampling_rate: float
    window: WindowContract
    preprocessing: dict[str, Any]
    output_type: str
    confidence_semantics: str
    calibration_status: str
    deployment_status: str
    created_at: str
    metadata: dict[str, Any]
    schema_version: int = 1
    validation_status: str = "unvalidated"
    deployment_validation_status: str = "unvalidated"
    approval_status: str = "pending"
    revocation_status: str = "active"
    runtime_compatible: bool = False
    sensor_compatible: bool = False
    availability_status: str = "unchecked"
    lifecycle_reason: str | None = None

    @property
    def path(self) -> Path:  # legacy UI compatibility
        return self.artifact_path

    @property
    def name(self) -> str:  # legacy UI compatibility
        return self.display_name

    @property
    def window_seconds(self) -> float:  # legacy UI compatibility
        return self.window.default_seconds

    @property
    def window_label(self) -> str:
        if self.window.policy == "fixed":
            return f"{self.window.default_seconds:g}초"
        if self.window.policy == "discrete":
            return " / ".join(f"{value:g}초" for value in self.window.values)
        return f"{self.window.minimum_seconds:g}~{self.window.maximum_seconds:g}초"

    @property
    def input_label(self) -> str:
        required = set(map(str, self.metadata.get("required_sensors", ["pressure"])))
        if required == {"accelerometer"}:
            return "가속도계"
        if "accelerometer" in required:
            return "압력 센서 + 가속도계"
        return "압력 센서"

    @property
    def requires_accelerometer(self) -> bool:
        return "가속도계" in self.input_label

    @property
    def minimum_window_seconds(self) -> float:
        return float(self.metadata.get("minimum_window_seconds", self.window.default_seconds))

    @property
    def production_selectable(self) -> bool:
        """Whether this exact immutable artifact can appear in measurement UI."""

        return (
            self.validation_status == "passed"
            and self.deployment_validation_status == "passed"
            and self.approval_status == "approved"
            and self.revocation_status == "active"
            and self.runtime_compatible
            and self.sensor_compatible
        )

    @property
    def management_status(self) -> str:
        if self.revocation_status == "revoked":
            return "revoked"
        if self.approval_status == "rejected":
            return "rejected"
        if "failed" in {self.validation_status, self.deployment_validation_status}:
            return "failed"
        if self.availability_status == "environment_unavailable" or (
            self.validation_status == "passed" and (not self.runtime_compatible or not self.sensor_compatible)
        ):
            return "environment_unavailable"
        if self.production_selectable:
            return "usable"
        return "pending"


@dataclass(frozen=True)
class SessionModelSnapshot:
    session_id: str
    model_snapshot_id: str
    artifact_id: str
    display_name: str
    version: str
    backend: str
    model_family: str
    artifact_hash: str
    manifest_hash: str
    selected_window_seconds: float
    target_samples: int
    sampling_rate: float
    channel_order: tuple[str, ...]
    label_order: tuple[str, ...]
    output_semantics: str
    calibration_status: str
    input_source: str
    created_at: str

    @classmethod
    def create(cls, descriptor: ModelDescriptor, *, session_id: str, window_seconds: float, input_source: str) -> "SessionModelSnapshot":
        import hashlib

        selected = descriptor.window.validate(window_seconds)
        stamp = datetime.now(timezone.utc).isoformat()
        token = f"{session_id}:{descriptor.artifact_hash}:{selected}:{stamp}".encode()
        return cls(session_id, hashlib.sha256(token).hexdigest()[:24], descriptor.artifact_id, descriptor.display_name, descriptor.version, descriptor.backend, descriptor.model_family, descriptor.artifact_hash, descriptor.manifest_hash, selected, descriptor.window.target_samples(selected, descriptor.sampling_rate), descriptor.sampling_rate, descriptor.input_channels, descriptor.labels, descriptor.confidence_semantics, descriptor.calibration_status, input_source, stamp)

    def as_dict(self) -> dict[str, Any]:
        return {key: (list(value) if isinstance(value, tuple) else value) for key, value in self.__dict__.items()}
