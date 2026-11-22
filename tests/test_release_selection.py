"""Release bootstrapping keeps review decisions bound to immutable artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.model_runtime import ModelRegistry
from src.model_runtime.validation import ValidationResult
from tests.test_model_lifecycle import write_v2_candidate


class ReleaseSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.bundle = self.directory / "bundle"
        self.state = self.directory / "user_state"
        self.manifest = write_v2_candidate(self.bundle, artifact_id="reviewed-release")
        self.publisher = ModelRegistry(self.bundle, self.bundle)
        model = self.publisher.validator.discover_manifest(self.manifest).descriptor
        self.assertIsNotNone(model)
        self.model = self.publisher.approve(
            self.publisher.validate_deployment(self.publisher.validate_runtime(model)),
            approved_by="release-test-reviewer",
        )
        self.selection_path = self.bundle / "artifacts/release_selection.json"
        self.selection_path.write_text(json.dumps({
            "artifact_id": self.model.artifact_id,
            "artifact_hash": self.model.artifact_hash,
            "manifest_hash": self.model.manifest_hash,
        }), encoding="utf-8")

    def registry(self, **kwargs) -> ModelRegistry:
        return ModelRegistry(self.bundle, self.state, use_bundled_release=True, **kwargs)

    def test_fresh_state_imports_only_opted_in_exact_reviewed_release(self) -> None:
        ordinary = ModelRegistry(self.bundle, self.state)
        self.assertEqual(ordinary.discover(), [])
        self.assertIsNone(ordinary.store.get(self.model))
        registry = self.registry()
        with mock.patch.object(registry.validator, "validate_runtime", wraps=registry.validator.validate_runtime) as runtime:
            models = registry.discover()
        self.assertEqual([model.artifact_id for model in models], [self.model.artifact_id])
        self.assertTrue(runtime.called)
        imported = registry.store.get(models[0])
        self.assertEqual(imported.provenance, "bundled-reviewed-release-v1")
        self.assertEqual(imported.approval_status, "approved")
        self.assertEqual(imported.approved_by, "release-test-reviewer")
        self.assertEqual(registry.active_model(models).artifact_id, self.model.artifact_id)

    def test_symlinked_frozen_artifact_root_bootstraps_exact_reviewed_release(self) -> None:
        # PyInstaller exposes Resources/artifacts through Frameworks/artifacts.
        # Discovery resolves manifest paths, so release ownership must resolve
        # the configured artifact root before comparing parent directories.
        frameworks = self.directory / "Application.app/Contents/Frameworks"
        frameworks.mkdir(parents=True)
        (frameworks / "artifacts").symlink_to(self.bundle / "artifacts", target_is_directory=True)
        registry = ModelRegistry(frameworks, self.state, use_bundled_release=True)
        models = registry.discover()
        self.assertEqual([model.artifact_id for model in models], [self.model.artifact_id])
        self.assertEqual(models[0].manifest_path, self.manifest.resolve())
        self.assertEqual(registry.active_model(models).artifact_id, self.model.artifact_id)
        self.assertEqual(registry.store.get(models[0]).provenance, "bundled-reviewed-release-v1")

    def test_user_rejection_revocation_and_corrupt_state_override_packaged_approval(self) -> None:
        registry = self.registry()
        model, = registry.discover()
        initial = registry.store.get(model)
        for changes in ({"approval_status": "rejected", "reason": "User rejected release"},
                        {"revocation_status": "revoked", "reason": "User recalled release"}):
            with self.subTest(changes=changes):
                registry.store.save(initial.updated(**changes))
                self.assertEqual(self.registry().discover(), [])
                current = registry.store.get(model)
                for key, value in changes.items():
                    self.assertEqual(getattr(current, key), value)
        registry.store.path_for(model).write_text("[]", encoding="utf-8")
        self.assertEqual(self.registry().discover(), [])
        self.assertEqual(registry.store.get(model).approval_status, "pending")

    def test_changed_manifest_hash_does_not_inherit_release_approval(self) -> None:
        self.manifest.write_bytes(self.manifest.read_bytes() + b"\n")
        registry = self.registry()
        self.assertEqual(registry.discover(), [])
        model, = registry.management_models()
        self.assertNotEqual(model.manifest_hash, self.model.manifest_hash)
        self.assertEqual(model.approval_status, "pending")

    def test_changed_artifact_bytes_fail_runtime_even_when_release_pointer_matches(self) -> None:
        self.model.artifact_path.write_bytes(self.model.artifact_path.read_bytes() + b"altered")
        registry = self.registry()
        self.assertEqual(registry.discover(), [])
        model, = registry.management_models()
        self.assertEqual(model.validation_status, "failed")
        self.assertFalse(model.runtime_compatible)
        self.assertIn("해시", model.lifecycle_reason or "")

    def test_wrong_selected_identity_or_unapproved_package_cannot_bootstrap(self) -> None:
        selection = json.loads(self.selection_path.read_text())
        selection["artifact_hash"] = "0" * 64
        self.selection_path.write_text(json.dumps(selection))
        self.assertEqual(self.registry().discover(), [])
        selection["artifact_hash"] = self.model.artifact_hash
        self.selection_path.write_text(json.dumps(selection))
        self.publisher.reject(self.model, reason="Release approval withdrawn")
        self.assertEqual(self.registry().discover(), [])

    def test_changed_runtime_replays_golden_once_before_model_becomes_usable(self) -> None:
        registry = self.registry()
        with mock.patch("src.model_runtime.validation.runtime_environment_hash", return_value="f" * 64), \
             mock.patch.object(registry.validator, "validate_deployment", wraps=registry.validator.validate_deployment) as deployment:
            models = registry.discover()
            self.assertEqual(len(models), 1)
            self.assertEqual(deployment.call_count, 1)
            record = registry.store.get(models[0])
            self.assertEqual(record.runtime_environment_hash, "f" * 64)
            self.assertEqual(record.deployment_environment_hash, "f" * 64)
            self.assertEqual(record.deployment_validation_status, "passed")
            self.assertEqual(len(registry.discover()), 1)
            self.assertEqual(deployment.call_count, 1)

    def test_changed_runtime_replay_failure_keeps_reviewed_model_unavailable(self) -> None:
        registry = self.registry()
        failure = ValidationResult(self.model, reason="Golden Replay failed", level=3,
            status="failed", runtime_environment_hash="f" * 64)
        with mock.patch("src.model_runtime.validation.runtime_environment_hash", return_value="f" * 64), \
             mock.patch.object(registry.validator, "validate_deployment", return_value=failure) as deployment:
            self.assertEqual(registry.discover(), [])
            deployment.assert_called_once()
            current = registry.store.get(self.model)
            self.assertEqual(current.approval_status, "approved")
            self.assertEqual(current.deployment_validation_status, "failed")
            self.assertEqual(current.reason, "Golden Replay failed")

    def add_approved_identity(self, registry: ModelRegistry, artifact_id: str):
        # Different reviewed package identities may legitimately reuse the same
        # weights. Rollback must compare the identity, not just weight bytes.
        payload = json.loads(self.manifest.read_text())
        payload["artifact_id"] = artifact_id
        path = self.manifest.with_name(f"{artifact_id}.json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        model = registry.validator.discover_manifest(path).descriptor
        return registry.approve(registry.validate_deployment(registry.validate_runtime(model)), approved_by="local-reviewer")

    def test_rollback_returns_previous_approved_identity_even_with_identical_weights(self) -> None:
        registry = self.registry()
        previous, = registry.discover()
        current = self.add_approved_identity(registry, "newer-reviewed-release")
        self.assertEqual(current.artifact_hash, previous.artifact_hash)
        registry.activate(previous)
        registry.activate(current)
        rolled_back = registry.rollback()
        self.assertEqual(rolled_back.artifact_id, previous.artifact_id)
        self.assertEqual(registry.active_model(registry.discover()).artifact_id, previous.artifact_id)

    def test_rollback_never_restores_revoked_previous_identity(self) -> None:
        registry = self.registry()
        previous, = registry.discover()
        current = self.add_approved_identity(registry, "current-reviewed-release")
        registry.activate(previous)
        registry.activate(current)
        registry.revoke(previous, reason="Withdrawn before rollback")
        with self.assertRaisesRegex(ValueError, "No previously active model"):
            registry.rollback()
        self.assertEqual(registry.active_model(registry.discover()).artifact_id, current.artifact_id)

    def test_malformed_release_selection_and_activation_history_fail_closed(self) -> None:
        for raw in ("[]", "null", "42", '"text"'):
            with self.subTest(raw=raw):
                self.selection_path.write_text(raw, encoding="utf-8")
                self.assertEqual(self.registry().discover(), [])
        registry = self.registry()
        registry.active_path.parent.mkdir(parents=True, exist_ok=True)
        registry.active_path.write_text("[]", encoding="utf-8")
        with self.assertRaises(ValueError):
            registry.rollback()
