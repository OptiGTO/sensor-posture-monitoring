from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib

from src.model_runtime import ModelRegistry, WindowContract
from src.model_runtime.validation import ArtifactValidator


class ModelRuntimeTests(unittest.TestCase):
    def test_window_contract_rejects_unsupported_windows(self) -> None:
        fixed = WindowContract("fixed", values=(3.0,))
        self.assertEqual(fixed.validate(3), 3.0)
        with self.assertRaises(ValueError):
            fixed.validate(5)
        ranged = WindowContract("range", minimum_seconds=3, maximum_seconds=5, step_seconds=1)
        self.assertEqual(ranged.target_samples(4, 12), 48)
        with self.assertRaises(ValueError):
            ranged.validate(3.5)

    def test_discovery_reads_manifest_and_default_uses_stable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(__file__).resolve().parents[1]
            root = Path(temporary) / "resources"; state = Path(temporary) / "state"
            for relative in (
                Path("artifacts/posture_classifier_window_3s.joblib"),
                Path("artifacts/manifests/posture_classifier_3s.json"),
                Path("artifacts/golden/legacy_baseline_v1.json"),
                Path("example/sample_input.csv"),
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / relative, destination)
            registry = ModelRegistry(root, state)
            model = registry.discover()[0]
            registry.activate(model)
            self.assertEqual(registry.active_model(registry.discover()).artifact_id, model.artifact_id)
            self.assertTrue((state / "artifacts" / "active_model.json").exists())

    def test_hash_mismatch_is_not_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); artifact = root / "model.joblib"; artifact.write_bytes(b"changed")
            manifest = root / "model.json"
            manifest.write_text(json.dumps({"schema_version": 1, "artifact_id": "bad", "display_name": "bad", "version": "1", "backend": "sklearn", "model_family": "x", "artifact_file": "model.joblib", "artifact_sha256": "0" * 64, "labels": ["a"], "input_channels": ["R11"], "sampling_rate": 12, "window": {"policy": "fixed", "values_seconds": [3]}, "deployment_status": "validated"}), encoding="utf-8")
            result = ArtifactValidator().validate_manifest(manifest)
            self.assertFalse(result.available)
            self.assertIn("해시", result.reason or "")
