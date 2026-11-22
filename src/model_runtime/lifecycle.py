"""Mutable deployment lifecycle state kept outside immutable model packages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import ModelDescriptor


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as destination:
        temporary = Path(destination.name)
        try:
            json.dump(payload, destination, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class DeploymentRecord:
    artifact_id: str
    version: str
    artifact_hash: str
    manifest_hash: str | None = None
    validation_status: str = "unvalidated"
    deployment_validation_status: str = "unvalidated"
    approval_status: str = "pending"
    approved_at: str | None = None
    approved_by: str | None = None
    revocation_status: str = "active"
    revoked_at: str | None = None
    reason: str | None = None
    runtime_environment_hash: str | None = None
    runtime_compatible: bool = False
    sensor_compatible: bool = False
    availability_status: str = "unchecked"
    deployment_environment_hash: str | None = None
    validated_at: str | None = None
    deployment_validated_at: str | None = None
    provenance: str = "model_artifact_v2"
    record_schema_version: int = 1

    @classmethod
    def pending(cls, descriptor: ModelDescriptor) -> "DeploymentRecord":
        return cls(
            descriptor.artifact_id,
            descriptor.version,
            descriptor.artifact_hash,
            manifest_hash=descriptor.manifest_hash,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DeploymentRecord":
        if not isinstance(payload, dict):
            raise ValueError("DeploymentRecord must be an object")
        allowed = cls.__dataclass_fields__.keys()
        for key, choices in {
            "validation_status": {"unvalidated", "passed", "failed"},
            "deployment_validation_status": {"unvalidated", "passed", "failed"},
            "approval_status": {"pending", "approved", "rejected"},
            "revocation_status": {"active", "revoked"},
        }.items():
            if key in payload and payload[key] not in choices:
                raise ValueError(f"Invalid lifecycle state: {key}")
        for key in ("runtime_compatible", "sensor_compatible"):
            if key in payload and not isinstance(payload[key], bool):
                raise ValueError(f"Invalid compatibility value: {key}")
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def updated(self, **changes: Any) -> "DeploymentRecord":
        return replace(self, **changes)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeploymentStore:
    def __init__(self, state_root: Path) -> None:
        self.root = Path(state_root) / "artifacts" / "deployment_records"

    @staticmethod
    def _key(artifact_id: str, version: str, artifact_hash: str) -> str:
        identity = f"{artifact_id}\0{version}\0{artifact_hash}".encode("utf-8")
        return hashlib.sha256(identity).hexdigest()

    def path_for(self, descriptor: ModelDescriptor) -> Path:
        return self.root / f"{self._key(descriptor.artifact_id, descriptor.version, descriptor.artifact_hash)}.json"

    def get(self, descriptor: ModelDescriptor) -> DeploymentRecord | None:
        path = self.path_for(descriptor)
        try:
            record = DeploymentRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return DeploymentRecord.pending(descriptor).updated(reason="배포 상태 파일을 읽을 수 없어 재검증과 승인이 필요합니다")
        expected = (descriptor.artifact_id, descriptor.version, descriptor.artifact_hash)
        if (record.artifact_id, record.version, record.artifact_hash) != expected:
            return DeploymentRecord.pending(descriptor).updated(reason="배포 상태 식별자가 모델과 다릅니다")
        if record.manifest_hash != descriptor.manifest_hash:
            return record.updated(
                manifest_hash=descriptor.manifest_hash, validation_status="unvalidated",
                deployment_validation_status="unvalidated", approval_status="pending",
                approved_at=None, approved_by=None, runtime_compatible=False,
                sensor_compatible=False, deployment_environment_hash=None,
                reason="매니페스트가 변경되어 재검증과 승인이 필요합니다",
            )
        return record

    def save(self, record: DeploymentRecord) -> None:
        key = self._key(record.artifact_id, record.version, record.artifact_hash)
        _atomic_json_write(self.root / f"{key}.json", record.as_dict())


class ValidationCache:
    """Cache Level 2 results until a manifest, artifact, or runtime changes."""

    def __init__(self, state_root: Path) -> None:
        self.path = Path(state_root) / "artifacts" / "validation_cache.json"

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) and payload.get("cache_schema_version") == 2 and isinstance(payload.get("entries"), dict) else {"cache_schema_version": 2, "entries": {}}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {"cache_schema_version": 2, "entries": {}}

    @staticmethod
    def _fingerprint(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"path": str(path.resolve()), "missing": True}
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }

    def fingerprint(self, descriptor: ModelDescriptor) -> list[dict[str, Any]]:
        paths = [descriptor.artifact_path]
        for component in descriptor.metadata.get("_components", ()):
            if isinstance(component, dict) and component.get("path"):
                paths.append(Path(str(component["path"])))
        return [self._fingerprint(path) for path in paths]

    @staticmethod
    def _identity(descriptor: ModelDescriptor) -> str:
        raw = f"{descriptor.artifact_id}\0{descriptor.version}\0{descriptor.manifest_path}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, descriptor: ModelDescriptor, runtime_environment_hash: str) -> dict[str, Any] | None:
        payload = self._load()
        entry = payload.get("entries", {}).get(self._identity(descriptor))
        if not isinstance(entry, dict):
            return None
        try:
            current_fingerprint = self.fingerprint(descriptor)
        except OSError:
            return None
        if (
            entry.get("manifest_hash") != descriptor.manifest_hash
            or entry.get("expected_artifact_hash") != descriptor.artifact_hash
            or entry.get("runtime_environment_hash") != runtime_environment_hash
            or entry.get("file_fingerprint") != current_fingerprint
        ):
            return None
        return entry

    def put(
        self,
        descriptor: ModelDescriptor,
        runtime_environment_hash: str,
        result: dict[str, Any],
    ) -> None:
        payload = self._load()
        entries = payload.setdefault("entries", {})
        actual_hash = str(result.get("actual_artifact_hash") or "")
        cache_key = hashlib.sha256(
            f"{actual_hash}:{descriptor.manifest_hash}:{runtime_environment_hash}".encode("utf-8")
        ).hexdigest()
        entries[self._identity(descriptor)] = {
            **result,
            "cache_key": cache_key,
            "artifact_hash": actual_hash,
            "manifest_hash": descriptor.manifest_hash,
            "expected_artifact_hash": descriptor.artifact_hash,
            "runtime_environment_hash": runtime_environment_hash,
            "file_fingerprint": self.fingerprint(descriptor),
            "cached_at": utc_now(),
        }
        _atomic_json_write(self.path, payload)
