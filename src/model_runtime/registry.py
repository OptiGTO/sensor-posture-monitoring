"""Model catalog plus external validation, approval, and revocation lifecycle."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .catalog import ArtifactCatalog
from .contracts import ModelDescriptor
from .lifecycle import DeploymentRecord, DeploymentStore, ValidationCache, utc_now
from .validation import ArtifactValidator, ValidationResult


class ModelRegistry:
    def __init__(
        self,
        root: Path,
        state_root: Path | None = None,
        validator: ArtifactValidator | None = None,
        *,
        available_sensors: Iterable[str] = ("pressure",),
        use_bundled_release: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.state_root = Path(state_root or root).resolve()
        self.persist_lifecycle = state_root is not None
        self.use_bundled_release = use_bundled_release
        self.builtin_artifact_root = self.root / "artifacts"
        self.user_artifact_root = self.state_root / "artifacts"
        roots = tuple(dict.fromkeys((self.builtin_artifact_root.resolve(), self.user_artifact_root.resolve())))
        self.manifest_root = self.builtin_artifact_root / "manifests"
        self.catalog = ArtifactCatalog(roots)
        self.active_path = self.user_artifact_root / "active_model.json"
        self.store = DeploymentStore(self.state_root)
        self.cache = ValidationCache(self.state_root) if self.persist_lifecycle else None
        self.validator = validator or ArtifactValidator(roots, self.cache)
        self.available_sensors = tuple(map(str, available_sensors))
        self.last_results: list[ValidationResult] = []
        self._ephemeral_records: dict[tuple[str, str, str], DeploymentRecord] = {}
        self._legacy_baseline = self._load_legacy_baseline()

    @staticmethod
    def _record_key(descriptor: ModelDescriptor) -> tuple[str, str, str]:
        return descriptor.artifact_id, descriptor.version, descriptor.artifact_hash

    def _get_record(self, descriptor: ModelDescriptor) -> DeploymentRecord | None:
        if self.persist_lifecycle:
            record = self.store.get(descriptor)
            if record is None and self._is_bundled_release(descriptor):
                bundled = DeploymentStore(self.root).get(descriptor)
                # A previously reviewed release can have stale environment-bound
                # evidence. Import its decision, then inspect() must rerun runtime
                # and deployment checks before it becomes selectable. Never import
                # failed evidence or a model that has never passed deployment.
                if (bundled and bundled.approval_status == "approved"
                        and (bundled.deployment_validation_status == "passed" or (
                            bundled.deployment_validation_status == "unvalidated"
                            and bundled.deployment_validated_at is not None
                            and bundled.deployment_environment_hash is not None))
                        and bundled.revocation_status != "revoked"):
                    record = bundled.updated(provenance="bundled-reviewed-release-v1")
                    self.store.save(record)
            return record
        record = self._ephemeral_records.get(self._record_key(descriptor))
        if record is not None and record.manifest_hash == descriptor.manifest_hash:
            return record
        return self.store.get(descriptor)

    def _is_bundled_release(self, descriptor: ModelDescriptor) -> bool:
        if not self.use_bundled_release or self.builtin_artifact_root.resolve() not in descriptor.manifest_path.parents:
            return False
        try:
            selected = json.loads((self.builtin_artifact_root / "release_selection.json").read_text(encoding="utf-8"))
            if not isinstance(selected, dict):
                return False
            return (selected.get("artifact_id"), selected.get("artifact_hash"), selected.get("manifest_hash")) == (descriptor.artifact_id, descriptor.artifact_hash, descriptor.manifest_hash)
        except (OSError, ValueError):
            return False

    def _save_record(self, record: DeploymentRecord) -> None:
        if self.persist_lifecycle:
            self.store.save(record)
        else:
            self._ephemeral_records[(record.artifact_id, record.version, record.artifact_hash)] = record

    def _load_legacy_baseline(self) -> dict[str, object]:
        path = self.builtin_artifact_root / "golden" / "legacy_baseline_v1.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {}
            payload["_base_dir"] = str(path.parent)
            return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}

    def _legacy_validation_config(self, descriptor: ModelDescriptor) -> dict[str, object] | None:
        models = self._legacy_baseline.get("models", {})
        golden = models.get(descriptor.artifact_id) if isinstance(models, dict) else None
        if not isinstance(golden, dict) or descriptor.schema_version != 1 or descriptor.manifest_path.parent != self.manifest_root.resolve():
            return None
        if (golden.get("artifact_sha256"), golden.get("manifest_sha256"), golden.get("version")) != (descriptor.artifact_hash, descriptor.manifest_hash, descriptor.version):
            return None
        return {
            "_base_dir": self._legacy_baseline.get("_base_dir", descriptor.manifest_path.parent),
            "evaluation": self._legacy_baseline.get("evaluation", {}),
            "semantic_mapping": self._legacy_baseline.get("semantic_mapping", {}),
            "device_tolerances": self._legacy_baseline.get("device_tolerances", {}),
            "golden_replay": {**golden, "label_order": list(self._legacy_baseline.get("semantic_mapping", {}))},
        }

    def _migrate_legacy(self, descriptor: ModelDescriptor) -> DeploymentRecord:
        """Import a v1 `validated` decision once after current runtime checks."""

        validation_config = self._legacy_validation_config(descriptor)
        if validation_config is None:
            return DeploymentRecord.pending(descriptor)
        runtime = self.validator.validate_runtime(
            descriptor, available_sensors=self.available_sensors
        )
        stamp = utc_now()
        if not runtime.available:
            return DeploymentRecord.pending(descriptor).updated(
                validation_status=runtime.status if runtime.status in {"passed", "failed"} else "unvalidated",
                availability_status="environment_unavailable" if runtime.status == "environment_unavailable" or not runtime.sensor_compatible else "available",
                reason=runtime.reason,
                runtime_environment_hash=runtime.runtime_environment_hash,
                runtime_compatible=runtime.runtime_compatible,
                sensor_compatible=runtime.sensor_compatible,
                validated_at=stamp,
                provenance="legacy_v1_migration",
            )

        deployment = self.validator.validate_deployment(descriptor, validation_config=validation_config, legacy_compatibility=True, available_sensors=self.available_sensors)
        if not deployment.available:
            return DeploymentRecord.pending(descriptor).updated(
                validation_status="passed",
                deployment_validation_status="failed",
                reason=deployment.reason,
                runtime_environment_hash=runtime.runtime_environment_hash,
                runtime_compatible=True,
                sensor_compatible=True,
                validated_at=stamp,
                deployment_validated_at=stamp,
                provenance="legacy_v1_migration",
            )
        return DeploymentRecord.pending(descriptor).updated(
            validation_status="passed",
            deployment_validation_status="passed",
            approval_status="approved",
            approved_at=stamp,
            approved_by="legacy-v1-migration",
            runtime_environment_hash=runtime.runtime_environment_hash,
            runtime_compatible=True,
            sensor_compatible=True,
            validated_at=stamp,
            deployment_validated_at=stamp,
            deployment_environment_hash=runtime.runtime_environment_hash,
            availability_status="available",
            provenance="built-in-baseline-v2",
        )

    @staticmethod
    def _apply_record(descriptor: ModelDescriptor, record: DeploymentRecord) -> ModelDescriptor:
        return replace(
            descriptor,
            validation_status=record.validation_status,
            deployment_validation_status=record.deployment_validation_status,
            approval_status=record.approval_status,
            revocation_status=record.revocation_status,
            runtime_compatible=record.runtime_compatible,
            sensor_compatible=record.sensor_compatible,
            availability_status=record.availability_status,
            lifecycle_reason=record.reason,
        )

    def inspect(self) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for path in self.catalog.manifests():
            discovery = self.validator.discover_manifest(path)
            if not discovery.available or discovery.descriptor is None:
                results.append(discovery)
                continue
            descriptor = discovery.descriptor
            record = self._get_record(descriptor)
            if record is not None and record.approved_by == "legacy-v1-migration" and record.provenance != "built-in-baseline-v2" and record.revocation_status != "revoked":
                record = None
            if record is None and self._legacy_validation_config(descriptor) is not None:
                try:
                    legacy_payload = json.loads(descriptor.manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    legacy_payload = {}
                if legacy_payload.get("deployment_status") == "validated" and not legacy_payload.get("revoked", False):
                    record = self._migrate_legacy(descriptor)
                    self._save_record(record)
            if record is None:
                record = DeploymentRecord.pending(descriptor)
            elif record.validated_at is not None:
                runtime = self.validator.validate_runtime(
                    descriptor, available_sensors=self.available_sensors
                )
                preserved_reason = record.reason if (
                    runtime.status == "passed"
                    and (
                        record.deployment_validation_status == "failed"
                        or record.approval_status == "rejected"
                        or record.revocation_status == "revoked"
                    )
                ) else runtime.reason
                updated = self._with_runtime(record, runtime, reason=preserved_reason)
                if (
                    runtime.available
                    and updated.deployment_validation_status == "unvalidated"
                    and updated.approval_status == "approved"
                    and updated.approved_by == "legacy-v1-migration"
                    and updated.revocation_status != "revoked"
                    and self._legacy_validation_config(descriptor) is not None
                ):
                    # A rebuilt app invalidates environment-bound evidence. Re-run
                    # the pinned built-in replay instead of hiding the model forever.
                    deployment = self.validator.validate_deployment(
                        descriptor,
                        validation_config=self._legacy_validation_config(descriptor),
                        legacy_compatibility=True,
                        available_sensors=self.available_sensors,
                    )
                    updated = updated.updated(
                        deployment_validation_status=deployment.status,
                        deployment_environment_hash=deployment.runtime_environment_hash if deployment.available else None,
                        deployment_validated_at=utc_now(),
                        reason=deployment.reason,
                    )
                if updated != record:
                    record = updated.updated(validated_at=utc_now())
                    self._save_record(record)
                if (runtime.available and record.deployment_validation_status == "unvalidated"
                        and record.approval_status == "approved" and record.revocation_status != "revoked"
                        and record.provenance == "bundled-reviewed-release-v1" and self._is_bundled_release(descriptor)):
                    deployment = self.validator.validate_deployment(descriptor, available_sensors=self.available_sensors)
                    record = record.updated(deployment_validation_status=deployment.status,
                        deployment_environment_hash=deployment.runtime_environment_hash if deployment.available else None,
                        deployment_validated_at=utc_now(), reason=deployment.reason)
                    self._save_record(record)
            merged = self._apply_record(descriptor, record)
            results.append(
                ValidationResult(
                    merged,
                    record.reason,
                    level=1,
                    status="passed",
                    runtime_environment_hash=record.runtime_environment_hash,
                    runtime_compatible=record.runtime_compatible,
                    sensor_compatible=record.sensor_compatible,
                )
            )
        self.last_results = results
        return results

    def management_models(self) -> list[ModelDescriptor]:
        return [result.descriptor for result in self.inspect() if result.descriptor is not None]

    def discover(self) -> list[ModelDescriptor]:
        return sorted(
            (model for model in self.management_models() if model.production_selectable),
            key=lambda item: (item.window.default_seconds, item.display_name),
        )

    def _fallback_model(self, models: list[ModelDescriptor]) -> ModelDescriptor | None:
        released = next((model for model in models if self._is_bundled_release(model)), None)
        if released is not None:
            return released
        builtin = [model for model in models if self.builtin_artifact_root.resolve() in model.manifest_path.parents]
        if not builtin:
            return None
        return sorted(builtin, key=lambda item: (item.window.default_seconds, item.display_name))[0]

    def active_model(self, models: list[ModelDescriptor]) -> ModelDescriptor | None:
        if not models:
            return None
        try:
            saved = json.loads(self.active_path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                return self._fallback_model(models)
            selected = next(
                (
                    model for model in models
                    if model.artifact_id == saved.get("artifact_id")
                    and model.artifact_hash == saved.get("artifact_hash")
                ),
                None,
            )
            return selected or self._fallback_model(models)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return self._fallback_model(models)

    def _record_for_update(self, model: ModelDescriptor) -> DeploymentRecord:
        record = self._get_record(model)
        return record or DeploymentRecord.pending(model)

    def _current_model(self, model: ModelDescriptor) -> ModelDescriptor:
        result = self.validator.discover_manifest(model.manifest_path)
        current = result.descriptor
        if not result.available or current is None or (current.manifest_hash, current.artifact_hash, current.artifact_id, current.version) != (model.manifest_hash, model.artifact_hash, model.artifact_id, model.version):
            raise ValueError("모델 또는 매니페스트가 변경되었습니다. 목록을 새로 고친 뒤 다시 선택하세요")
        return current

    @staticmethod
    def _with_runtime(record: DeploymentRecord, result: ValidationResult, *, reason: str | None = None) -> DeploymentRecord:
        deployment_status = record.deployment_validation_status
        if result.status != "passed" or (deployment_status == "passed" and record.deployment_environment_hash != result.runtime_environment_hash):
            deployment_status = "unvalidated"
        return record.updated(
            validation_status=result.status if result.status in {"passed", "failed"} else "unvalidated",
            deployment_validation_status=deployment_status,
            runtime_environment_hash=result.runtime_environment_hash,
            runtime_compatible=result.runtime_compatible, sensor_compatible=result.sensor_compatible,
            availability_status="environment_unavailable" if result.status == "environment_unavailable" or not result.sensor_compatible else "available",
            reason=reason if reason is not None else result.reason,
        )

    def validate_runtime(self, model: ModelDescriptor) -> ModelDescriptor:
        model = self._current_model(model)
        result = self.validator.validate_runtime(model, available_sensors=self.available_sensors)
        record = self._with_runtime(self._record_for_update(model), result).updated(validated_at=utc_now())
        self._save_record(record)
        return self._apply_record(model, record)

    def validate_deployment(self, model: ModelDescriptor) -> ModelDescriptor:
        model = self._current_model(model)
        record = self._record_for_update(model)
        if record.validation_status != "passed" or not record.runtime_compatible or not record.sensor_compatible:
            raise ValueError("Level 2 Runtime 검증을 먼저 통과해야 합니다")
        legacy_config = self._legacy_validation_config(model)
        result = self.validator.validate_deployment(model, validation_config=legacy_config, legacy_compatibility=legacy_config is not None, available_sensors=self.available_sensors)
        preserve_decision = (record.approval_status == "rejected" or (
            result.available and record.deployment_validation_status == "passed"
            and record.deployment_environment_hash == result.runtime_environment_hash))
        record = record.updated(
            deployment_validation_status=result.status,
            approval_status=record.approval_status if preserve_decision else "pending",
            approved_at=record.approved_at if preserve_decision else None,
            approved_by=record.approved_by if preserve_decision else None,
            reason=result.reason,
            deployment_validated_at=utc_now(),
            deployment_environment_hash=result.runtime_environment_hash if result.available else None,
        )
        self._save_record(record)
        return self._apply_record(model, record)

    def approve(self, model: ModelDescriptor, *, approved_by: str) -> ModelDescriptor:
        if not approved_by.strip():
            raise ValueError("승인자 식별자가 필요합니다")
        model = self._current_model(model)
        record = self._record_for_update(model)
        runtime = self.validator.validate_runtime(model, available_sensors=self.available_sensors, use_cache=False)
        record = self._with_runtime(record, runtime)
        self._save_record(record)
        if record.validation_status != "passed" or record.deployment_validation_status != "passed":
            raise ValueError("Level 2와 Level 3 검증을 모두 통과해야 승인할 수 있습니다")
        if record.revocation_status == "revoked":
            raise ValueError("폐기된 모델은 승인할 수 없습니다")
        if not runtime.available:
            raise ValueError(runtime.reason or "현재 실행 환경에서 사용할 수 없습니다")
        record = record.updated(
            approval_status="approved", approved_at=utc_now(),
            approved_by=approved_by.strip(), reason=None,
        )
        self._save_record(record)
        return self._apply_record(model, record)

    def reject(self, model: ModelDescriptor, *, reason: str) -> ModelDescriptor:
        if not reason.strip():
            raise ValueError("거절 사유가 필요합니다")
        model = self._current_model(model)
        record = self._record_for_update(model).updated(
            approval_status="rejected", approved_at=None, approved_by=None, reason=reason.strip()
        )
        self._save_record(record)
        return self._apply_record(model, record)

    def revoke(self, model: ModelDescriptor, *, reason: str) -> ModelDescriptor:
        if not reason.strip():
            raise ValueError("폐기 사유가 필요합니다")
        model = self._current_model(model)
        record = self._record_for_update(model).updated(
            revocation_status="revoked", revoked_at=utc_now(), reason=reason.strip()
        )
        self._save_record(record)
        return self._apply_record(model, record)

    def activate(self, model: ModelDescriptor) -> None:
        selectable = next(
            (
                item for item in self.discover()
                if item.artifact_id == model.artifact_id and item.artifact_hash == model.artifact_hash
            ),
            None,
        )
        if selectable is None:
            raise ValueError("승인·검증을 통과한 현재 artifact만 측정 모델로 활성화할 수 있습니다")
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        previous: dict[str, object] | None = None
        try:
            previous = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        # Keep one useful rollback target; repeatedly choosing the same model
        # must not recursively embed an ever-growing activation history.
        if not isinstance(previous, dict):
            previous = None
        elif (previous.get("artifact_id"), previous.get("artifact_hash")) == (selectable.artifact_id, selectable.artifact_hash):
            return
        else:
            previous = {key: value for key, value in previous.items() if key != "previous_active_model"}
        record = {
            "artifact_id": selectable.artifact_id,
            "version": selectable.version,
            "artifact_hash": selectable.artifact_hash,
            "default_window_seconds": selectable.window.default_seconds,
            "activated_at": utc_now(),
            "previous_active_model": previous,
        }
        temporary = self.active_path.with_name(f".{self.active_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(record, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(self.active_path)

    def descriptor_for_path(self, path: str | Path) -> ModelDescriptor | None:
        resolved = Path(path).resolve()
        return next((item for item in self.discover() if item.artifact_path == resolved), None)

    def rollback(self) -> ModelDescriptor:
        """Restore a previous approved model without bypassing current validation."""
        try:
            saved = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("No saved model activation history is available.") from error
        if not isinstance(saved, dict):
            raise ValueError("Invalid model activation history.")
        previous = saved.get("previous_active_model")
        available = self.discover()
        while isinstance(previous, dict):
            model = next((item for item in available if item.artifact_id == previous.get("artifact_id") and item.artifact_hash == previous.get("artifact_hash")), None)
            if model is not None and (model.artifact_id, model.artifact_hash) != (saved.get("artifact_id"), saved.get("artifact_hash")):
                self.activate(model)
                return model
            previous = previous.get("previous_active_model")
        raise ValueError("No previously active model still passes approval and validation.")
