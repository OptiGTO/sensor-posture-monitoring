from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd

from src.app_services import TrainingJob
from src.model_runtime import ModelRegistry
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS
from train_window_models import train_deployment

ROOT = Path(__file__).resolve().parents[1]


class TrainingEvidenceTests(unittest.TestCase):
    def make_data(self, root):
        rng = np.random.default_rng(17)
        for subject in range(4):
            folder = root / f"subject{subject}"
            folder.mkdir(parents=True)
            for index, label in enumerate(CLASS_NAMES):
                frame = pd.DataFrame(rng.uniform(10, 20, (37, 16)) + index * 20, columns=SENSOR_COLUMNS)
                frame.insert(0, "label", label)
                frame.insert(0, "elapsed_s", np.arange(37) / 12)
                frame.to_csv(folder / f"posture{index}.csv", index=False)

    def test_real_training_packaging_and_deployment_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            data = state / "data"
            self.make_data(data)
            job = TrainingJob(ROOT, (), 41, windows=(3,), state_root=state)
            with mock.patch("train_window_models.SVM_CANDIDATES", [{"C": 1.0, "gamma": "scale"}, {"C": 3.0, "gamma": "scale"}]):
                train_deployment(data, job.run_dir, (3,), 41)
            manifest, = job._write_pending_candidate_manifests()
            report_path = job.run_dir / "artifacts/window_3s_evaluation.json"
            report = json.loads(report_path.read_text())
            parts = report["partitions"]
            for key in ("subjects", "record_ids"):
                train, validation, final = (set(parts[name][key]) for name in ("train", "validation", "final_evaluation"))
                self.assertFalse(train & validation or train & final or validation & final)
            bundle = joblib.load(job.run_dir / "artifacts/posture_classifier_window_3s.joblib")
            self.assertEqual(bundle["training_subject_count"], 2)
            self.assertEqual(bundle["training_record_count"], 14)
            self.assertEqual(bundle["svm_model"].named_steps["model"].shape_fit_[0], 14)
            self.assertEqual(report["dataset_fingerprint"], bundle["dataset_fingerprint"])
            registry = ModelRegistry(ROOT, state)
            model = next(item for item in registry.management_models() if item.manifest_path == manifest.resolve())
            job.process = mock.Mock(stdout=io.StringIO("Training completed\n"), args=["training-worker"])
            job.process.wait.return_value = 0
            with mock.patch.object(job, "_write_pending_candidate_manifests", return_value=(manifest,)):
                job._read()
            saved = json.loads((job.run_dir / "training.manifest.json").read_text())
            self.assertTrue(saved["validation_results"][0]["passed"], saved)
            self.assertEqual(saved["approval_status"], "pending")
            events = list(job.events.queue)
            self.assertEqual(events[-1], ("finished", "0"))
            self.assertEqual(len([event for event in events if event[0] == "status"]), 2)
            model = next(item for item in registry.management_models() if item.manifest_path == manifest.resolve())
            self.assertEqual(model.deployment_validation_status, "passed", str(model))
            self.assertEqual(model.approval_status, "pending")
            # Evidence cannot be modified after packaging without invalidating hashes.
            report_path.write_text('{}')
            self.assertFalse(registry.validator.validate_deployment(model).available)
            job.candidate_manifest_paths = (manifest, manifest)
            failed = job._validate_candidates()
            self.assertEqual(len(failed), 2)
            self.assertTrue(all(not result["passed"] and result["reason"] for result in failed))
            with self.assertRaises(ValueError):
                registry.approve(model, approved_by="reviewer")
            job._cancel_requested.set()
            self.assertEqual(job._validate_candidates(), [])

    def test_missing_class_and_duplicate_content_fail_before_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            self.make_data(data)
            for path in data.glob("*/posture0.csv"):
                path.unlink()
            with mock.patch("train_window_models.build_svm") as fit:
                with self.assertRaisesRegex(ValueError, "분할에 자세 데이터가 부족"):
                    train_deployment(data, root / "missing", (3,), 41)
                fit.assert_not_called()
            self.make_data_for_duplicate(data)
            with mock.patch("train_window_models.build_svm") as fit:
                with self.assertRaisesRegex(ValueError, "중복된 CSV"):
                    train_deployment(data, root / "duplicate", (3,), 41)
                fit.assert_not_called()

    def make_data_for_duplicate(self, data):
        # Restore the missing class, then repeat one recording under another subject.
        frame = pd.read_csv(data / "subject0/posture1.csv")
        frame["label"] = CLASS_NAMES[0]
        for subject in range(4):
            frame.to_csv(data / f"subject{subject}/posture0.csv", index=False)
