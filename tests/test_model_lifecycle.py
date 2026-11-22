from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.model_runtime import ArtifactValidator, ModelRegistry, ValidationCache
from src.model_runtime.validation import runtime_environment_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_v2_candidate(state_root: Path, *, artifact_id: str = "candidate-v2") -> Path:
    artifact_root = state_root / "artifacts"
    manifest_root = artifact_root / "manifests"
    golden_root = artifact_root / "golden"
    manifest_root.mkdir(parents=True)
    golden_root.mkdir(parents=True)
    artifact = artifact_root / "candidate.joblib"
    shutil.copy2(PROJECT_ROOT / "artifacts" / "posture_classifier_window_3s.joblib", artifact)
    shutil.copy2(PROJECT_ROOT / "example" / "sample_input.csv", golden_root / "sample_input.csv")
    labels = [
        "정자세", "앞으로 숙이기", "뒤로 기대기", "왼쪽 기울임", "오른쪽 기울임",
        "왼쪽 다리 꼬기", "오른쪽 다리 꼬기",
    ]
    channels = [f"R{row}{column}" for row in range(1, 5) for column in range(1, 5)]
    fingerprint = hashlib.sha256(b"test dataset provenance").hexdigest()
    evaluation = {
        "schema_version": 1, "labels": labels, "dataset_fingerprint": fingerprint,
        "partitions": {name: {"subjects": [f"subject-{name}"], "record_ids": [f"record-{name}"]} for name in ("train", "validation", "final_evaluation")},
        "final_evaluation_used_for_selection": False,
        "class_coverage": labels,
        "metrics": {"macro_f1": 1, "per_class_f1": dict.fromkeys(labels, 1), "confusion_matrix": np.eye(len(labels), dtype=int).tolist()},
    }
    evaluation_path = artifact_root / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    baseline = json.loads((PROJECT_ROOT / "artifacts/golden/legacy_baseline_v1.json").read_text())["models"]["posture-classifier-rbf-svm-3s"]
    payload = {
        "schema_version": 2,
        "artifact_id": artifact_id,
        "display_name": "v2 후보",
        "version": "1",
        "backend": "sklearn",
        "model_family": "rbf_svm",
        "artifact_file": "../candidate.joblib",
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "components": [
            {"role": "evaluation", "file": "../evaluation.json", "sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest()},
            {"role": "golden_input", "file": "../golden/sample_input.csv", "sha256": hashlib.sha256((golden_root / "sample_input.csv").read_bytes()).hexdigest()},
        ],
        "labels": labels,
        "input_channels": channels,
        "sampling_rate": 12,
        "window": {
            "policy": "fixed",
            "values_seconds": [3],
            "target_sample_rule": "round(window_seconds * sampling_rate)",
            "padding_policy": "interpolate_time_axis",
        },
        "preprocessing": {"kind": "feature_vector", "feature_count": 596},
        "output_type": "probabilities",
        "confidence_semantics": "calibrated_probability",
        "calibration_status": "pending_validation",
        "created_at": "2026-09-03T00:00:00+09:00",
        "provenance": {"source": "training_run", "created_by": "test", "training_run": "test-run", "dataset_fingerprint": fingerprint},
        "runtime": {"python": ">=3.10", "packages": {}},
        "sensors": {"required": ["pressure"], "supported_devices": ["cpu"]},
        "validation": {
            "evaluation": {"split_strategy": "grouped by subject", "integrity_status": "passed", "evidence_files": ["../evaluation.json"]},
            "semantic_mapping": {label: label for label in labels},
            "device_tolerances": {"cpu_probability_atol": 0.03},
            "golden_replay": {"input_file": "../golden/sample_input.csv", "expected_label": "왼쪽 다리 꼬기", "expected_scores": baseline["expected_scores"], "window_count": 6, "label_order": labels},
        },
        "metadata": {"required_sensors": ["pressure"], "minimum_window_seconds": 2.85},
    }
    manifest = manifest_root / "candidate.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return manifest


class ModelLifecycleTests(unittest.TestCase):
    def test_bundled_stale_approved_release_is_revalidated_without_new_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            write_v2_candidate(root)
            registry = ModelRegistry(root, root)
            model = registry.management_models()[0]
            model = registry.validate_runtime(model)
            model = registry.validate_deployment(model)
            model = registry.approve(model, approved_by="original-reviewer")
            original = registry.store.get(model)
            (root / "artifacts/release_selection.json").write_text(json.dumps({
                "artifact_id": model.artifact_id, "artifact_hash": model.artifact_hash,
                "manifest_hash": model.manifest_hash,
            }))
            stale = original.updated(deployment_validation_status="unvalidated",
                                     deployment_environment_hash="previous-build")
            registry.store.save(stale)
            fresh = ModelRegistry(root, Path(temporary) / "fresh", use_bundled_release=True)
            with mock.patch.object(fresh.validator, "validate_deployment", wraps=fresh.validator.validate_deployment) as check:
                selected = fresh.active_model(fresh.discover())
            self.assertIsNotNone(selected)
            self.assertEqual(selected.artifact_id, model.artifact_id)
            check.assert_called_once()
            self.assertEqual(fresh.store.get(selected).approved_by, "original-reviewer")
            self.assertEqual(fresh.store.get(selected).deployment_validation_status, "passed")
            for name, updates in (
                ("failed", {"deployment_validation_status": "failed"}),
                ("never-validated", {"deployment_validated_at": None}),
                ("pending", {"approval_status": "pending"}),
                ("revoked", {"revocation_status": "revoked"}),
            ):
                with self.subTest(name=name):
                    registry.store.save(stale.updated(**updates))
                    rejected = ModelRegistry(root, Path(temporary) / name, use_bundled_release=True)
                    self.assertEqual(rejected.discover(), [])

    def test_builtin_models_revalidate_after_app_environment_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ModelRegistry(PROJECT_ROOT, Path(temporary))
            models = registry.discover()
            self.assertEqual(len(models), 3)
            for index, model in enumerate(models):
                record = registry.store.get(model)
                registry.store.save(record.updated(
                    deployment_environment_hash="previous-app-build",
                    deployment_validation_status="unvalidated" if index == 0 else "passed",
                ))
            restored = registry.discover()
            self.assertEqual(len(restored), 3)
            for model in restored:
                record = registry.store.get(model)
                self.assertEqual(record.deployment_validation_status, "passed")
                self.assertEqual(record.deployment_environment_hash, record.runtime_environment_hash)
            with mock.patch.object(registry.validator, "validate_deployment", side_effect=AssertionError("unnecessary replay")):
                self.assertEqual(len(registry.discover()), 3)

    def test_builtin_recovery_preserves_revocation_rejection_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ModelRegistry(PROJECT_ROOT, Path(temporary))
            models = registry.discover()
            for model, changes in zip(models, (
                {"revocation_status": "revoked"},
                {"approval_status": "rejected"},
                {"deployment_validation_status": "failed", "reason": "replay failed"},
            )):
                record = registry.store.get(model)
                registry.store.save(record.updated(deployment_environment_hash="previous-app-build", **changes))
            with mock.patch.object(registry.validator, "validate_deployment", side_effect=AssertionError("must not recover")):
                self.assertEqual(registry.discover(), [])

    def test_v2_requires_runtime_deployment_and_human_approval_then_revocation_blocks_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest = write_v2_candidate(state)
            original_manifest = manifest.read_bytes()
            registry = ModelRegistry(PROJECT_ROOT, state)
            candidate = next(model for model in registry.management_models() if model.artifact_id == "candidate-v2")

            self.assertEqual(candidate.management_status, "pending")
            self.assertNotIn(candidate.artifact_id, {model.artifact_id for model in registry.discover()})
            candidate = registry.validate_runtime(candidate)
            self.assertEqual(candidate.validation_status, "passed")
            self.assertNotIn(candidate.artifact_id, {model.artifact_id for model in registry.discover()})
            candidate = registry.validate_deployment(candidate)
            self.assertEqual(candidate.deployment_validation_status, "passed")
            candidate = registry.approve(candidate, approved_by="reviewer")
            self.assertIn(candidate.artifact_id, {model.artifact_id for model in registry.discover()})
            self.assertNotEqual(registry.active_model(registry.discover()).artifact_id, candidate.artifact_id)

            registry.revoke(candidate, reason="superseded")
            self.assertNotIn(candidate.artifact_id, {model.artifact_id for model in registry.discover()})
            self.assertEqual(manifest.read_bytes(), original_manifest)
            self.assertTrue(any((state / "artifacts" / "deployment_records").glob("*.json")))

    def test_level_one_does_not_hash_or_deserialize_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest = write_v2_candidate(state)
            validator = ArtifactValidator((state / "artifacts",))
            with mock.patch("src.model_runtime.validation.digest", side_effect=AssertionError("hash called")), mock.patch(
                "joblib.load", side_effect=AssertionError("load called")
            ):
                result = validator.discover_manifest(manifest)
            self.assertTrue(result.available)
            self.assertEqual(result.level, 1)

    def test_runtime_cache_hits_then_invalidates_when_artifact_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest = write_v2_candidate(state)
            cache = ValidationCache(state)
            validator = ArtifactValidator((state / "artifacts",), cache)
            descriptor = validator.discover_manifest(manifest).descriptor
            assert descriptor is not None

            first = validator.validate_runtime(descriptor)
            second = validator.validate_runtime(descriptor)
            self.assertTrue(first.available)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            descriptor.artifact_path.write_bytes(descriptor.artifact_path.read_bytes() + b"changed")
            changed = validator.validate_runtime(descriptor)
            self.assertFalse(changed.available)
            self.assertFalse(changed.cache_hit)
            self.assertIn("해시", changed.reason or "")

    def test_runtime_cache_key_includes_environment_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest = write_v2_candidate(state)
            validator = ArtifactValidator((state / "artifacts",), ValidationCache(state))
            descriptor = validator.discover_manifest(manifest).descriptor
            assert descriptor is not None
            self.assertTrue(validator.validate_runtime(descriptor).available)
            with mock.patch("src.model_runtime.validation.runtime_environment_hash", return_value="f" * 64):
                refreshed = validator.validate_runtime(descriptor)
            self.assertFalse(refreshed.cache_hit)
            self.assertEqual(refreshed.runtime_environment_hash, "f" * 64)
            self.assertNotEqual(refreshed.runtime_environment_hash, runtime_environment_hash("sklearn"))

    def test_path_escape_and_embedded_v2_lifecycle_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest = write_v2_candidate(state)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["artifact_file"] = "../../../outside.joblib"
            (state / "outside.joblib").write_bytes(b"outside")
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            result = ArtifactValidator((state / "artifacts",)).discover_manifest(manifest)
            self.assertFalse(result.available)
            self.assertIn("루트 밖", result.reason or "")

            payload["artifact_file"] = "../candidate.joblib"
            payload["approval_status"] = "approved"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            result = ArtifactValidator((state / "artifacts",)).discover_manifest(manifest)
            self.assertFalse(result.available)
            self.assertIn("수명주기", result.reason or "")

    def test_schema_is_enforced_for_nested_and_unexpected_fields(self) -> None:
        changes = (
            lambda p: p.update(unexpected=True),
            lambda p: p["sensors"].update(required="pressure"),
            lambda p: p["runtime"].update(packages={"joblib": 123}),
            lambda p: p.update(labels=[1, 2]),
            lambda p: p["window"].update(policy="range", values_seconds=[]),
            lambda p: p.update(sampling_rate=float("nan")),
        )
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                manifest = write_v2_candidate(Path(temporary))
                payload = json.loads(manifest.read_text())
                change(payload)
                manifest.write_text(json.dumps(payload))
                self.assertFalse(ArtifactValidator().discover_manifest(manifest).available)

    def test_discovery_rejects_non_objects_and_duplicate_keys(self) -> None:
        for raw in ('[]', 'null', '{"schema_version":2,"schema_version":1}'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "manifest.json"
                path.write_text(raw)
                self.assertFalse(ArtifactValidator().discover_manifest(path).available)

    def test_sensor_changes_are_checked_even_on_a_runtime_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest = write_v2_candidate(state)
            validator = ArtifactValidator((state / "artifacts",), ValidationCache(state))
            model = validator.discover_manifest(manifest).descriptor
            self.assertTrue(validator.validate_runtime(model).available)
            with mock.patch("src.model_runtime.validation.digest", side_effect=AssertionError("unexpected rehash")):
                missing = validator.validate_runtime(model, available_sensors=())
                restored = validator.validate_runtime(model, available_sensors=("pressure",))
            self.assertTrue(missing.cache_hit)
            self.assertEqual(missing.status, "passed")
            self.assertFalse(missing.sensor_compatible)
            self.assertFalse(missing.available)
            self.assertTrue(restored.available)

    def test_unknown_v1_cannot_self_approve_and_missing_baseline_is_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest = write_v2_candidate(state)
            payload = json.loads(manifest.read_text())
            payload.update(schema_version=1, deployment_status="validated")
            manifest.write_text(json.dumps(payload))
            registry = ModelRegistry(state, state)
            with mock.patch("joblib.load", side_effect=AssertionError("unapproved discovery loaded weights")):
                self.assertEqual(registry.discover(), [])
            self.assertEqual(registry.management_models()[0].approval_status, "pending")

    def test_catalog_finds_package_manifests_and_excludes_incomplete_directories(self) -> None:
        from src.model_runtime import ArtifactCatalog
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / name / "manifest.json" for name in ("ready", ".tmp/run", "run.part", "run.tmp", "incomplete")]
            for path in paths:
                path.parent.mkdir(parents=True)
                path.write_text("{}")
            (root / "incomplete/.incomplete").touch()
            self.assertEqual(ArtifactCatalog(root).manifests(), [paths[0].resolve()])

    def test_approval_rechecks_bytes_and_refuses_stale_manifests(self) -> None:
        for target in ("manifest", "artifact", "evidence"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                state = Path(temporary)
                manifest = write_v2_candidate(state)
                registry = ModelRegistry(PROJECT_ROOT, state)
                model = registry.validator.discover_manifest(manifest).descriptor
                model = registry.validate_deployment(registry.validate_runtime(model))
                self.assertEqual(model.deployment_validation_status, "passed")
                path = {"manifest": manifest, "artifact": model.artifact_path, "evidence": state / "artifacts/evaluation.json"}[target]
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaises(ValueError):
                    registry.approve(model, approved_by="reviewer")
                self.assertNotIn(model.artifact_id, {item.artifact_id for item in registry.discover()})

    def test_revocation_survives_a_manifest_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest = write_v2_candidate(state)
            registry = ModelRegistry(PROJECT_ROOT, state)
            model = registry.validator.discover_manifest(manifest).descriptor
            registry.revoke(model, reason="recalled")
            payload = json.loads(manifest.read_text())
            payload["display_name"] = "renamed"
            manifest.write_text(json.dumps(payload))
            current = next(item for item in registry.management_models() if item.artifact_id == model.artifact_id)
            self.assertEqual(current.revocation_status, "revoked")
            self.assertFalse(current.production_selectable)

    def test_deployment_checks_actual_evaluation_integrity_and_metrics(self) -> None:
        changes = {
            "subject leak": lambda p: p["partitions"]["validation"].update(subjects=p["partitions"]["train"]["subjects"]),
            "record leak": lambda p: p["partitions"]["final_evaluation"].update(record_ids=p["partitions"]["train"]["record_ids"]),
            "selected on holdout": lambda p: p.update(final_evaluation_used_for_selection=True),
            "wrong label order": lambda p: p.update(labels=list(reversed(p["labels"]))),
            "wrong provenance": lambda p: p.update(dataset_fingerprint="0" * 64),
            "nonfinite metrics": lambda p: p["metrics"].update(macro_f1=float("nan")),
            "inconsistent metrics": lambda p: p["metrics"].update(macro_f1=0.5),
        }
        for name, change in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                state = Path(temporary)
                manifest = write_v2_candidate(state)
                evidence = state / "artifacts/evaluation.json"
                report = json.loads(evidence.read_text())
                change(report)
                evidence.write_text(json.dumps(report))
                payload = json.loads(manifest.read_text())
                payload["components"][0]["sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
                manifest.write_text(json.dumps(payload))
                validator = ArtifactValidator()
                result = validator.validate_deployment(validator.discover_manifest(manifest).descriptor)
                self.assertFalse(result.available, name)

    def test_low_metrics_are_not_a_deployment_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest = write_v2_candidate(state)
            evidence = state / "artifacts/evaluation.json"
            report = json.loads(evidence.read_text())
            report["metrics"] = {"macro_f1": 0, "per_class_f1": dict.fromkeys(report["labels"], 0), "confusion_matrix": np.roll(np.eye(7, dtype=int), 1, axis=1).tolist()}
            evidence.write_text(json.dumps(report))
            payload = json.loads(manifest.read_text())
            payload["components"][0]["sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(payload))
            validator = ArtifactValidator()
            result = validator.validate_deployment(validator.discover_manifest(manifest).descriptor)
            self.assertTrue(result.available, result.reason)

    def test_golden_checks_full_scores_label_order_and_window_count(self) -> None:
        for key, value in (("expected_scores", [1, 0, 0, 0, 0, 0, 0]), ("window_count", 1), ("label_order", ["wrong"] * 7)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                manifest = write_v2_candidate(Path(temporary))
                payload = json.loads(manifest.read_text())
                payload["validation"]["golden_replay"][key] = value
                manifest.write_text(json.dumps(payload))
                validator = ArtifactValidator()
                model = validator.discover_manifest(manifest).descriptor
                if model is not None:
                    self.assertFalse(validator.validate_deployment(model).available)

    def test_builtin_migration_runs_golden_only_once_and_preserves_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ModelRegistry(PROJECT_ROOT, Path(temporary))
            models = registry.discover()
            self.assertEqual(len(models), 3)
            for model in models:
                self.assertEqual(hashlib.sha256(model.artifact_path.read_bytes()).hexdigest(), model.artifact_hash)
                self.assertEqual(registry.store.get(model).provenance, "built-in-baseline-v2")
            again = ModelRegistry(PROJECT_ROOT, Path(temporary))
            with mock.patch.object(again.validator, "validate_deployment", side_effect=AssertionError("golden rerun")), mock.patch("joblib.load", side_effect=AssertionError("cache missed")):
                self.assertEqual(len(again.discover()), 3)

    def test_approved_user_model_loads_through_existing_predictor_and_revocation_blocks_it(self) -> None:
        from src.inference_core import PosturePredictor
        from src.posture_core import read_sensor_csv
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest = write_v2_candidate(state)
            registry = ModelRegistry(PROJECT_ROOT, state)
            model = registry.validator.discover_manifest(manifest).descriptor
            model = registry.approve(registry.validate_deployment(registry.validate_runtime(model)), approved_by="reviewer")
            predictor = PosturePredictor(model.artifact_path, registry=registry)
            self.assertEqual(predictor.predict_frame(read_sensor_csv(PROJECT_ROOT / "example/sample_input.csv")).shape, (7,))
            registry.revoke(model, reason="retired")
            with self.assertRaises(ValueError):
                PosturePredictor(model.artifact_path, registry=registry)


if __name__ == "__main__":
    unittest.main()
