"""Three-level validation for immutable posture-model artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from .contracts import ModelDescriptor, WindowContract
from .lifecycle import ValidationCache
from .schema_validation import read_json_bytes, validate_schema


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_BACKENDS = {"sklearn", "pytorch"}
LIFECYCLE_KEYS = {
    "approval_status", "approved_at", "approved_by", "deployment_status",
    "revocation_status", "revoked", "revoked_at", "validation_status", "deployment_validation_status",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


class RuntimeUnavailable(RuntimeError):
    """A missing/incompatible environment can be retried without changing bytes."""


def runtime_environment_hash(backend: str, extra_packages: Iterable[str] = ()) -> str:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "jsonschema", "packaging"]
    if backend == "pytorch":
        packages.append("torch")
    versions: dict[str, str] = {}
    for package in sorted(set(packages) | set(extra_packages)):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    source_root = Path(__file__).resolve().parents[1]
    sources = [*Path(__file__).parent.glob("*.py"), *[source_root / name for name in ("posture_core.py", "window_pipeline.py", "realtime.py")], *source_root.parent.glob("schemas/model_artifact_*.schema.json")]
    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
        # A validator/adapter update must invalidate old successful probes too.
        "runtime_code": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources) if path.is_file()
        },
    }
    import sys
    if getattr(sys, "frozen", False):
        stat = Path(sys.executable).stat()
        payload["bundled_executable"] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    if backend == "pytorch":
        try:
            import torch
            payload["accelerators"] = {"cuda": torch.cuda.is_available(), "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}
        except (ImportError, OSError, RuntimeError):
            payload["accelerators"] = "unavailable"
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ValidationResult:
    descriptor: ModelDescriptor | None
    reason: str | None = None
    level: int = 1
    status: str = "passed"
    actual_artifact_hash: str | None = None
    runtime_environment_hash: str | None = None
    runtime_compatible: bool = False
    sensor_compatible: bool = False
    cache_hit: bool = False

    @property
    def available(self) -> bool:
        return self.descriptor is not None and self.reason is None and self.status == "passed"


def _is_inside(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or root in path.parents for root in (item.resolve() for item in roots))


def _require_nonempty_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 값이 비어 있습니다")
    return value


def _parse_version(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    if not match:
        raise ValueError(f"버전을 해석할 수 없습니다: {value}")
    return tuple(int(part) for part in match.group(1).split("."))


def _meets_requirement(installed: str, requirement: str) -> bool:
    return Version(installed) in SpecifierSet(requirement)


class ArtifactValidator:
    supported_backends = SUPPORTED_BACKENDS

    def __init__(
        self,
        allowed_artifact_roots: Iterable[Path] | None = None,
        cache: ValidationCache | None = None,
    ) -> None:
        self.allowed_artifact_roots = tuple(Path(path).resolve() for path in (allowed_artifact_roots or ()))
        self.cache = cache

    def _allowed_roots(self, manifest_path: Path, override: Iterable[Path] | None) -> tuple[Path, ...]:
        if override is not None:
            configured = tuple(Path(path).resolve() for path in override)
        elif self.allowed_artifact_roots:
            configured = self.allowed_artifact_roots
        else:
            configured = ()
        if configured:
            resolved_manifest = manifest_path.resolve()
            owners = [root for root in configured if root == resolved_manifest or root in resolved_manifest.parents]
            if not owners:
                raise ValueError("매니페스트가 허용된 artifact 루트 밖에 있습니다")
            owner = max(owners, key=lambda root: len(root.parts))
            package = resolved_manifest.parent
            if package.name == "manifests":
                package = package.parent
            return (package if _is_inside(package, (owner,)) else owner,)
        parent = manifest_path.resolve().parent
        return ((parent.parent if parent.name == "manifests" else parent),)

    def discover_manifest(
        self,
        manifest_path: Path,
        *,
        allowed_artifact_roots: Iterable[Path] | None = None,
    ) -> ValidationResult:
        """Level 1: parse metadata and paths without hashing or loading weights."""

        try:
            manifest_path = Path(manifest_path).resolve()
            roots = self._allowed_roots(manifest_path, allowed_artifact_roots)
            raw_bytes = manifest_path.read_bytes()
            payload = read_json_bytes(raw_bytes)
            if not isinstance(payload, dict):
                raise ValueError("매니페스트는 JSON 객체여야 합니다")
            schema_version = payload.get("schema_version")
            if schema_version not in {1, 2}:
                raise ValueError("지원하지 않는 매니페스트 버전")
            if schema_version == 1:
                validate_schema(payload, "model_artifact_v1.schema.json")
            if schema_version == 2:
                embedded = sorted(LIFECYCLE_KEYS.intersection(payload))
                if embedded:
                    raise ValueError(f"v2 매니페스트에는 가변 수명주기 상태를 저장할 수 없습니다: {', '.join(embedded)}")
                validate_schema(payload, "model_artifact_v2.schema.json")
                for key in ("provenance", "runtime", "sensors", "validation"):
                    if not isinstance(payload.get(key), dict):
                        raise ValueError(f"v2 매니페스트에 {key} 객체가 필요합니다")
                for key in ("preprocessing", "metadata"):
                    if not isinstance(payload.get(key), dict):
                        raise ValueError(f"v2 매니페스트에 {key} 객체가 필요합니다")
                for key in ("confidence_semantics", "calibration_status", "created_at"):
                    _require_nonempty_string(payload, key)
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(payload.get("artifact_id", ""))):
                    raise ValueError("artifact_id 형식이 올바르지 않습니다")
                provenance = payload["provenance"]
                if provenance.get("source") not in {"training_run", "imported", "built_in_baseline"}:
                    raise ValueError("지원하지 않는 provenance.source")
                _require_nonempty_string(provenance, "created_by")
                runtime = payload["runtime"]
                _require_nonempty_string(runtime, "python")
                if not isinstance(runtime.get("packages"), dict):
                    raise ValueError("runtime.packages 객체가 필요합니다")
                sensors = payload["sensors"]
                if not isinstance(sensors.get("supported_devices"), list) or not sensors["supported_devices"]:
                    raise ValueError("하나 이상의 supported device가 필요합니다")
                validation = payload["validation"]
                evaluation = validation.get("evaluation")
                if not isinstance(evaluation, dict) or evaluation.get("integrity_status") not in {"pending", "passed", "failed"}:
                    raise ValueError("validation.evaluation 무결성 상태가 필요합니다")
                _require_nonempty_string(evaluation, "split_strategy")
                if not isinstance(validation.get("semantic_mapping"), dict):
                    raise ValueError("validation.semantic_mapping 객체가 필요합니다")
                tolerances = validation.get("device_tolerances")
                if not isinstance(tolerances, dict) or any(
                    not isinstance(value, (int, float)) or value < 0 for value in tolerances.values()
                ):
                    raise ValueError("validation.device_tolerances 값이 올바르지 않습니다")

            artifact_id = _require_nonempty_string(payload, "artifact_id")
            display_name = _require_nonempty_string(payload, "display_name")
            version = _require_nonempty_string(payload, "version")
            backend = _require_nonempty_string(payload, "backend")
            model_family = _require_nonempty_string(payload, "model_family")
            if backend not in self.supported_backends:
                raise ValueError("지원하지 않는 백엔드")

            expected_hash = _require_nonempty_string(payload, "artifact_sha256").lower()
            if not SHA256_PATTERN.fullmatch(expected_hash):
                raise ValueError("artifact_sha256 형식이 올바르지 않습니다")
            roots = self._allowed_roots(manifest_path, allowed_artifact_roots)
            artifact_path = (manifest_path.parent / _require_nonempty_string(payload, "artifact_file")).resolve()
            if not _is_inside(artifact_path, roots):
                raise ValueError("모델 파일 경로가 허용된 artifact 루트 밖에 있습니다")
            if not artifact_path.is_file():
                raise ValueError("모델 파일이 없습니다")

            if not isinstance(payload["labels"], list) or not isinstance(payload["input_channels"], list):
                raise ValueError("labels와 input_channels는 배열이어야 합니다")
            labels = tuple(map(str, payload["labels"]))
            channels = tuple(map(str, payload["input_channels"]))
            if not labels or len(labels) != len(set(labels)) or any(not item for item in labels):
                raise ValueError("라벨 목록은 비어 있지 않고 중복이 없어야 합니다")
            if not channels or len(channels) != len(set(channels)) or any(not item for item in channels):
                raise ValueError("입력 채널 목록은 비어 있지 않고 중복이 없어야 합니다")
            sampling_rate = float(payload["sampling_rate"])
            if not np.isfinite(sampling_rate) or sampling_rate <= 0:
                raise ValueError("sampling_rate는 양수여야 합니다")

            window_data = payload["window"]
            if not isinstance(window_data, dict):
                raise ValueError("window 객체가 필요합니다")
            window = WindowContract(
                policy=str(window_data["policy"]),
                values=tuple(float(value) for value in window_data.get("values_seconds", [])),
                minimum_seconds=window_data.get("minimum_seconds"),
                maximum_seconds=window_data.get("maximum_seconds"),
                step_seconds=window_data.get("step_seconds"),
                target_sample_rule=str(window_data.get("target_sample_rule", "round(window_seconds * sampling_rate)")),
                padding_policy=str(window_data.get("padding_policy", "reject")),
            )
            if window.target_sample_rule != "round(window_seconds * sampling_rate)":
                raise ValueError("지원하지 않는 target_sample_rule")
            if window.padding_policy not in {"reject", "interpolate_time_axis"}:
                raise ValueError("지원하지 않는 padding_policy")
            output_type = str(payload.get("output_type", "probabilities"))
            if output_type not in {"probabilities", "logits"}:
                raise ValueError("지원하지 않는 output_type")

            component_payload = payload.get("components", [])
            if not isinstance(component_payload, list):
                raise ValueError("components는 배열이어야 합니다")
            components: list[dict[str, Any]] = []
            for component in component_payload:
                if not isinstance(component, dict):
                    raise ValueError("component 항목은 객체여야 합니다")
                component_path = (manifest_path.parent / _require_nonempty_string(component, "file")).resolve()
                if not _is_inside(component_path, roots):
                    raise ValueError("구성요소 경로가 허용된 artifact 루트 밖에 있습니다")
                if component.get("required", True) and not component_path.is_file():
                    raise ValueError(f"필수 구성요소 파일이 없습니다: {component_path.name}")
                component_hash = _require_nonempty_string(component, "sha256").lower()
                if not SHA256_PATTERN.fullmatch(component_hash):
                    raise ValueError("구성요소 SHA-256 형식이 올바르지 않습니다")
                components.append({**component, "path": str(component_path), "sha256": component_hash})

            metadata = dict(payload.get("metadata", {}))
            sensors = dict(payload.get("sensors", {}))
            required_sensors = sensors.get("required", metadata.get("required_sensors", ["pressure"]))
            if not isinstance(required_sensors, list) or not required_sensors:
                raise ValueError("하나 이상의 required sensor가 필요합니다")
            metadata.update({
                "_package_root": str(roots[0]),
                "required_sensors": list(map(str, required_sensors)),
                "_components": components,
                "_runtime": dict(payload.get("runtime", {})),
                "_validation": dict(payload.get("validation", {})),
                "_provenance": dict(payload.get("provenance", {})),
                "supported_devices": list(map(str, sensors.get("supported_devices", metadata.get("supported_devices", ["cpu"])))),
            })
            descriptor = ModelDescriptor(
                artifact_id=artifact_id, display_name=display_name, version=version,
                backend=backend, model_family=model_family, artifact_path=artifact_path,
                manifest_path=manifest_path, artifact_hash=expected_hash,
                manifest_hash=hashlib.sha256(raw_bytes).hexdigest(), labels=labels,
                input_channels=channels, sampling_rate=sampling_rate, window=window,
                preprocessing=dict(payload.get("preprocessing", {})), output_type=output_type,
                confidence_semantics=str(payload.get("confidence_semantics", "score")),
                calibration_status=str(payload.get("calibration_status", "not_validated")),
                deployment_status=str(payload.get("deployment_status", "external_lifecycle")),
                created_at=str(payload.get("created_at", "")), metadata=metadata,
                schema_version=int(schema_version),
            )
            return ValidationResult(descriptor, level=1)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            return ValidationResult(None, str(error), level=1, status="failed")

    def _check_runtime_requirements(self, descriptor: ModelDescriptor) -> None:
        requirements = descriptor.metadata.get("_runtime", {})
        python_requirement = requirements.get("python")
        if python_requirement and not _meets_requirement(platform.python_version(), str(python_requirement)):
            raise RuntimeUnavailable(f"Python 실행 환경이 요구사항과 다릅니다: {python_requirement}")
        packages = requirements.get("packages", {})
        if not isinstance(packages, dict):
            raise ValueError("runtime.packages는 객체여야 합니다")
        for package, requirement in packages.items():
            try:
                installed = importlib.metadata.version(str(package))
            except importlib.metadata.PackageNotFoundError:
                raise RuntimeUnavailable(f"필수 런타임 패키지가 없습니다: {package}") from None
            if not _meets_requirement(installed, str(requirement)):
                raise RuntimeUnavailable(f"{package} 버전이 요구사항과 다릅니다: {requirement}")

    def validate_runtime(
        self,
        descriptor: ModelDescriptor,
        *,
        available_sensors: Iterable[str] = ("pressure",),
        use_cache: bool = True,
    ) -> ValidationResult:
        """Level 2: verify bytes, runtime, controlled load, and I/O contract."""

        environment_hash = runtime_environment_hash(descriptor.backend, descriptor.metadata.get("_runtime", {}).get("packages", {}))
        required_sensors = set(map(str, descriptor.metadata.get("required_sensors", ["pressure"])))
        sensor_compatible = required_sensors.issubset(set(map(str, available_sensors)))
        sensor_reason = None if sensor_compatible else "현재 연결된 센서가 모델 요구사항과 호환되지 않습니다"
        try:
            if hashlib.sha256(descriptor.manifest_path.read_bytes()).hexdigest() != descriptor.manifest_hash:
                raise ValueError("매니페스트가 변경되었습니다. 모델 목록을 새로 고치세요")
        except (OSError, ValueError) as error:
            return ValidationResult(descriptor, str(error), level=2, status="failed", runtime_environment_hash=environment_hash)
        if use_cache and self.cache is not None:
            cached = self.cache.get(descriptor, environment_hash)
            if cached is not None:
                status = str(cached.get("status", "failed"))
                reason = cached.get("reason")
                return ValidationResult(
                    descriptor, str(reason) if reason else sensor_reason, level=2, status=status,
                    actual_artifact_hash=str(cached.get("actual_artifact_hash", "")) or None,
                    runtime_environment_hash=environment_hash,
                    runtime_compatible=bool(cached.get("runtime_compatible", False)),
                    sensor_compatible=sensor_compatible, cache_hit=True,
                )

        actual_hash: str | None = None
        runtime_compatible = False
        status, reason = "passed", None
        component_hashes: dict[str, str] = {}
        try:
            actual_hash = digest(descriptor.artifact_path)
            if actual_hash != descriptor.artifact_hash:
                raise ValueError("모델 파일 해시가 일치하지 않습니다")
            for component in descriptor.metadata.get("_components", ()):
                component_path = Path(component["path"])
                if not component.get("required", True) and not component_path.exists():
                    continue
                actual_component_hash = digest(component_path)
                component_hashes[str(component.get("role", component["path"]))] = actual_component_hash
                if actual_component_hash != component["sha256"]:
                    raise ValueError(f"구성요소 해시가 일치하지 않습니다: {component.get('role', component['path'])}")
            self._check_runtime_requirements(descriptor)
            if descriptor.backend == "pytorch":
                try:
                    import torch
                except (ImportError, OSError) as error:
                    raise RuntimeUnavailable(f"PyTorch 실행 환경을 사용할 수 없습니다: {error}") from error

            from .factory import AdapterFactory

            adapter = AdapterFactory.create(descriptor)
            self._check_preprocessing_contract(descriptor, adapter)
            if descriptor.backend == "sklearn":
                import pandas as pd
                from ..posture_core import SENSOR_COLUMNS

                if tuple(descriptor.input_channels) != tuple(SENSOR_COLUMNS):
                    raise ValueError("현재 sklearn 전처리는 16개 압력 센서 채널만 지원합니다")
                sample_count = descriptor.window.target_samples(descriptor.window.default_seconds, descriptor.sampling_rate) + 1
                frame = pd.DataFrame(np.zeros((sample_count, len(SENSOR_COLUMNS))), columns=SENSOR_COLUMNS)
                frame.insert(0, "elapsed_s", np.arange(sample_count) / descriptor.sampling_rate)
                output = adapter.predict_proba(frame, descriptor.window.default_seconds)
            else:
                sample_count = descriptor.window.target_samples(descriptor.window.default_seconds, descriptor.sampling_rate)
                sample = np.zeros((1, sample_count, len(descriptor.input_channels)), dtype=np.float32)
                output = adapter.predict_proba(sample, descriptor.window.default_seconds)
            output = np.asarray(output, dtype=float)
            if output.ndim != 2 or not len(output) or output.shape[1] != len(descriptor.labels):
                raise ValueError("모델 출력 shape이 라벨 계약과 다릅니다")
            if not np.isfinite(output).all() or (output < 0).any():
                raise ValueError("모델 출력에 NaN 또는 Inf가 있습니다")
            if not np.allclose(output.sum(axis=1), 1.0, rtol=0, atol=1e-6):
                raise ValueError("확률 출력의 합이 1이 아닙니다")
            runtime_compatible = True
        except RuntimeUnavailable as error:
            status, reason = "environment_unavailable", str(error)
        except Exception as error:
            status, reason = "failed", str(error)

        result_payload = {
            "status": status, "reason": reason, "actual_artifact_hash": actual_hash,
            "component_hashes": component_hashes, "runtime_compatible": runtime_compatible,
            "sensor_compatible": sensor_compatible,
        }
        if self.cache is not None and status != "environment_unavailable":
            self.cache.put(descriptor, environment_hash, result_payload)
        return ValidationResult(
            descriptor, reason or sensor_reason, level=2, status=status, actual_artifact_hash=actual_hash,
            runtime_environment_hash=environment_hash, runtime_compatible=runtime_compatible,
            sensor_compatible=sensor_compatible,
        )

    @staticmethod
    def _check_preprocessing_contract(descriptor: ModelDescriptor, adapter: Any) -> None:
        if descriptor.backend == "sklearn":
            bundle = adapter.bundle
            if descriptor.window.policy != "fixed":
                raise ValueError("현재 sklearn artifact는 fixed 창만 검증할 수 있습니다")
            seconds = descriptor.window.default_seconds
            if "window_seconds" in bundle and float(bundle["window_seconds"]) != seconds:
                raise ValueError("저장된 모델과 매니페스트의 윈도우 계약이 다릅니다")
            if "sampling_rate" in bundle and float(bundle["sampling_rate"]) != descriptor.sampling_rate:
                raise ValueError("저장된 모델과 매니페스트의 sampling 계약이 다릅니다")
            if "target_length" in bundle and int(bundle["target_length"]) != descriptor.window.target_samples(seconds, descriptor.sampling_rate):
                raise ValueError("저장된 모델과 매니페스트의 target sample 계약이 다릅니다")
            kind = bundle.get("input_representation", "feature_vector")
            if kind not in {"feature_vector", "sensor_sequence"} or descriptor.preprocessing.get("kind") != kind:
                raise ValueError("지원하지 않는 sklearn 전처리 계약입니다")
            if descriptor.preprocessing.get("feature_count") != len(bundle.get("feature_names", ())):
                raise ValueError("저장된 모델과 매니페스트의 feature 계약이 다릅니다")
            from ..signal_preprocessing import BUNDLE_KEY, SignalPreprocessing
            if descriptor.preprocessing.get(BUNDLE_KEY) != bundle.get(BUNDLE_KEY):
                raise ValueError("저장된 모델과 매니페스트의 신호 전처리 계약이 다릅니다")
            SignalPreprocessing.from_bundle(bundle)
            if kind == "sensor_sequence":
                from ..model_catalog import SEQUENCE_MODELS
                if descriptor.model_family not in SEQUENCE_MODELS or bundle.get("model_family") != descriptor.model_family:
                    raise ValueError("시계열 모델 계열이 매니페스트와 다릅니다")
                expected = (descriptor.window.target_samples(seconds, descriptor.sampling_rate), len(descriptor.input_channels))
                if tuple(getattr(bundle.get("model"), "input_shape_", ())) != expected or descriptor.preprocessing.get("channel_count") != expected[1]:
                    raise ValueError("시계열 입력 크기가 모델 계약과 다릅니다")
        else:
            if descriptor.preprocessing.get("signal_preprocessing") is not None:
                raise ValueError("PyTorch 실행 경로는 신호 전처리 계약을 지원하지 않습니다")
            normalization = descriptor.preprocessing.get("normalization", {})
            if normalization:
                mean = np.asarray(normalization.get("mean", []), dtype=float)
                std = np.asarray(normalization.get("std", []), dtype=float)
                if mean.shape != (len(descriptor.input_channels),) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                    raise ValueError("PyTorch 정규화 채널·평균·표준편차 계약이 유효하지 않습니다")

    def validate_deployment(
        self,
        descriptor: ModelDescriptor,
        *,
        validation_config: dict[str, Any] | None = None,
        legacy_compatibility: bool = False,
        available_sensors: Iterable[str] = ("pressure",),
    ) -> ValidationResult:
        """Level 3: verify evaluation integrity, semantics, devices, and golden replay."""

        config = dict(validation_config if validation_config is not None else descriptor.metadata.get("_validation", {}))
        try:
            runtime = self.validate_runtime(descriptor, available_sensors=available_sensors, use_cache=False)
            if not runtime.available:
                raise ValueError(runtime.reason or "Runtime 검증을 통과하지 못했습니다")
            from .deployment_checks import check_evaluation, check_golden, check_semantics, evidence_path

            evaluation = config.get("evaluation", {})
            base_dir = Path(str(config.get("_base_dir", descriptor.manifest_path.parent))) if legacy_compatibility else descriptor.manifest_path.parent
            if not legacy_compatibility:
                check_evaluation(descriptor, evaluation, base_dir)
            check_semantics(descriptor, config.get("semantic_mapping", {}))
            tolerances = config.get("device_tolerances", {})
            if not isinstance(tolerances, dict) or not tolerances:
                raise ValueError("지원 장치별 허용 오차가 정의되지 않았습니다")
            from .factory import AdapterFactory
            from ..posture_core import read_sensor_csv

            adapter = AdapterFactory.create(descriptor)
            devices = descriptor.metadata.get("supported_devices", ["cpu"])
            if "cpu" not in devices or (descriptor.backend == "sklearn" and devices != ["cpu"]):
                raise ValueError("CPU 기준 검증이 필요하며 sklearn은 CPU 장치만 지원합니다")
            goldens = config.get("golden_replays", [config.get("golden_replay", {})])
            if not isinstance(goldens, list) or not goldens or not all(isinstance(item, dict) and item.get("input_file") for item in goldens):
                raise ValueError("Golden Replay 기준이 정의되지 않았습니다")
            window = descriptor.window
            if window.policy == "range":
                count = int(np.floor((window.maximum_seconds - window.minimum_seconds) / window.step_seconds + 1e-9)) + 1
                if count > 1000:
                    raise ValueError("검증 범위가 너무 큽니다. 검증 가능한 discrete 창으로 제한하세요")
                required_windows = [window.minimum_seconds + index * window.step_seconds for index in range(count)]
            else:
                required_windows = list(window.values)
            covered = [window.validate(item.get("window_seconds", window.default_seconds)) for item in goldens]
            if any(not any(abs(seconds - value) < 1e-9 for value in covered) for seconds in required_windows):
                raise ValueError("운영에 노출할 모든 윈도우 길이의 Golden Replay 근거가 필요합니다")
            for device in devices:
                tolerance = float(tolerances[f"{device}_probability_atol"])
                if not np.isfinite(tolerance) or not 0 <= tolerance <= 1:
                    raise ValueError("장치 허용 오차가 유효하지 않습니다")
                if descriptor.backend == "pytorch":
                    torch = adapter.torch
                    if device == "cuda" and not torch.cuda.is_available() or device == "mps" and not torch.backends.mps.is_available():
                        raise ValueError(f"{device} 장치 검증을 수행할 수 없습니다. 검증된 장치만 선언하세요")
                    adapter.device = torch.device(device)
                    adapter.model.to(adapter.device).eval()
                for golden, seconds in zip(goldens, covered):
                    if legacy_compatibility:
                        input_path = (base_dir / str(golden["input_file"])).resolve()
                        if digest(input_path) != golden.get("input_sha256"):
                            raise ValueError("기본 모델 Golden Replay 입력 해시가 다릅니다")
                    else:
                        input_path = evidence_path(descriptor, base_dir, str(golden["input_file"]))
                    frame = read_sensor_csv(input_path)
                    probabilities = check_golden(adapter, descriptor, frame, golden, tolerance, seconds)
                    if legacy_compatibility and seconds == 20:
                        self._check_legacy_realtime(adapter, descriptor, frame, probabilities, tolerance)
            return ValidationResult(descriptor, level=3, runtime_environment_hash=runtime.runtime_environment_hash, runtime_compatible=True, sensor_compatible=True)
        except Exception as error:
            return ValidationResult(descriptor, str(error), level=3, status="failed")

    @staticmethod
    def _check_legacy_realtime(adapter: Any, descriptor: ModelDescriptor, frame: Any, probabilities: np.ndarray, tolerance: float) -> None:
        from types import SimpleNamespace
        from ..realtime import LiveConfig, RealtimePosturePredictor

        predictor = SimpleNamespace(labels=descriptor.labels, predict_frame=lambda data: adapter.predict_proba(data, 20).mean(axis=0))
        engine = RealtimePosturePredictor(predictor, LiveConfig(window_seconds=20, minimum_window_seconds=19.99, prediction_interval_seconds=100, smoothing_history=1))
        event = None
        for row in frame.itertuples(index=False):
            event = engine.push(row.elapsed_s, [getattr(row, name) for name in descriptor.input_channels]) or event
        if event is None or not np.allclose([event.probabilities[label] for label in descriptor.labels], probabilities.mean(axis=0), rtol=0, atol=tolerance):
            raise ValueError("20초 기본 모델의 batch/realtime Golden Replay가 다릅니다")

    def validate_manifest(self, manifest_path: Path, *, production_only: bool = True) -> ValidationResult:
        """Compatibility facade: Level 1 only, or legacy-v1 Level 2 production check."""

        discovery = self.discover_manifest(manifest_path)
        if not discovery.available or discovery.descriptor is None or not production_only:
            return discovery
        descriptor = discovery.descriptor
        if descriptor.schema_version != 1:
            return ValidationResult(
                descriptor, "v2 모델의 운영 상태는 DeploymentRecord에서 확인해야 합니다",
                level=2, status="failed",
            )
        try:
            payload = json.loads(descriptor.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return ValidationResult(descriptor, str(error), level=2, status="failed")
        if payload.get("deployment_status") != "validated" or payload.get("revoked", False):
            return ValidationResult(descriptor, "운영 승인되지 않은 모델", level=2, status="failed")
        return self.validate_runtime(descriptor)
