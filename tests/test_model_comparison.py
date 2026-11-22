from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd

from src.app_services import DatasetService, TrainingJob, TrainingWizardState
from src.model_comparison import DEFAULT_MODEL_FAMILIES, MODEL_NAMES, _dataset, aligned_predict_proba, build_model, train_comparison
from src.model_runtime import ModelRegistry
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SMALL_GRID = {"rbf_svm": [{"C": 1.0, "gamma": "scale"}],
    "logistic_regression": [{"C": 1.0}],
    "random_forest": [{"n_estimators": 12, "min_samples_leaf": 1, "max_features": "sqrt"}],
    "extra_trees": [{"n_estimators": 12, "min_samples_leaf": 1, "max_features": "sqrt"}]}


class ModelComparisonTests(unittest.TestCase):
    def make_data(self, root: Path, rows: int = 37) -> None:
        rng = np.random.default_rng(37)
        for subject in range(4):
            directory = root / f"subject{subject}"
            directory.mkdir(parents=True)
            for i, label in enumerate(CLASS_NAMES):
                values = rng.uniform(10, 15, (rows, len(SENSOR_COLUMNS))) + i * 15 + subject * 3
                frame = pd.DataFrame(values, columns=SENSOR_COLUMNS)
                frame.insert(0, "label", label)
                frame.insert(0, "elapsed_s", np.arange(rows) / 12)
                frame.to_csv(directory / f"posture{i}.csv", index=False)

    def test_custom_four_second_window_trains_and_is_packaged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_data(root / "data", rows=49)
            with mock.patch("src.model_comparison.MODEL_CANDIDATES", SMALL_GRID):
                report = train_comparison(
                    root / "data", root / "run", (4,), 41,
                    model_families=("rbf_svm",), progress=lambda _: None,
                )
            self.assertEqual(report["windows"], [4])
            self.assertEqual(report["models"][0]["window_seconds"], 4)
            self.assertTrue((root / "run/artifacts/window_4s_rbf_svm_model_config.json").is_file())

    def test_four_models_share_data_train_only_scaling_and_final_selection_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            data_root = state / "data"
            self.make_data(data_root)
            dataset = _dataset(data_root, 3, 41)
            final_X = dataset["X"][dataset["masks"]["final_evaluation"]]
            job = TrainingJob(ROOT, (), 41, windows=(3,), state_root=state)
            checked_final = []
            def checked_probability(model, features, count):
                if np.array_equal(features, final_X):
                    lock = json.loads(job.comparison_report_path.read_text())
                    self.assertTrue(lock["selection_locked"])
                    self.assertFalse(lock["final_evaluation_used_for_selection"])
                    self.assertEqual(len(lock["candidates"]), 4)
                    checked_final.append(lock["selected"])
                return aligned_predict_proba(model, features, count)
            with mock.patch("src.model_comparison.MODEL_CANDIDATES", SMALL_GRID), \
                 mock.patch("src.model_comparison.aligned_predict_proba", side_effect=checked_probability):
                report = train_comparison(data_root, job.run_dir, (3,), 41, progress=lambda _: None)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(checked_final), 4)
            self.assertTrue(all(item == report["selected"] for item in checked_final))
            self.assertEqual({row["family"] for row in report["models"]}, set(DEFAULT_MODEL_FAMILIES))
            fingerprints = set()
            train_X = dataset["X"][dataset["masks"]["train"]]
            for row in report["models"]:
                bundle = joblib.load(job.run_dir / "artifacts" / row["artifact_file"])
                fingerprints.add(bundle["dataset_fingerprint"])
                np.testing.assert_allclose(bundle["model"].named_steps["scale"].mean_, train_X.mean(axis=0, dtype=np.float64))
                self.assertEqual(bundle["training_window_count"], len(train_X))
                self.assertGreater(row["inference_ms_median"], 0)
                self.assertEqual(set(row["final_evaluation"]["per_class"]), set(CLASS_NAMES))
            self.assertEqual(fingerprints, {dataset["fingerprint"]})
            manifests = job._write_pending_candidate_manifests()
            self.assertEqual(len(manifests), 4)
            registry = ModelRegistry(ROOT, state)
            candidates = [model for model in registry.management_models() if model.manifest_path in {p.resolve() for p in manifests}]
            self.assertEqual(len(candidates), 4)
            for model in candidates:
                model = registry.validate_deployment(registry.validate_runtime(model))
                self.assertEqual(model.deployment_validation_status, "passed", str(model))
                self.assertEqual(model.approval_status, "pending")

    def test_cancellation_preserves_failure_reason_without_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            self.make_data(data)
            with self.assertRaisesRegex(InterruptedError, "cancelled"):
                train_comparison(data, root / "run", (3,), 41, cancelled=lambda: True)
            report = json.loads((root / "run/artifacts/model_comparison.json").read_text())
            self.assertEqual(report["status"], "cancelled")
            self.assertEqual(report["models"], [])
            self.assertFalse(report["selection_locked"])

    def test_exhausted_model_family_fails_before_final_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_data(root / "data")
            def with_failure(family, params, seed):
                if family == "logistic_regression":
                    raise ValueError("Injected optimizer failure")
                return build_model(family, params, seed)
            with mock.patch("src.model_comparison.MODEL_CANDIDATES", SMALL_GRID), \
                 mock.patch("src.model_comparison.build_model", side_effect=with_failure), \
                 self.assertRaisesRegex(ValueError, "Comparison incomplete"):
                train_comparison(root / "data", root / "run", (3,), 41, progress=lambda _: None)
            report = json.loads((root / "run/artifacts/model_comparison.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["models"], [])
            self.assertFalse(report["selection_locked"])
            self.assertIn("Injected optimizer failure", report["failures"][0]["reason"])

    def test_cancelled_subprocess_records_terminal_status_and_does_not_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = TrainingJob(ROOT, (), 41, state_root=Path(temporary))
            job.comparison_report_path.parent.mkdir(parents=True)
            job.comparison_report_path.write_text(json.dumps({"status": "running", "models": []}))
            job.process = mock.Mock(stdout=io.StringIO("Fitting model\n"), args=["training-worker"])
            job.process.wait.return_value = -15
            job._cancel_requested.set()
            with mock.patch.object(job, "_write_pending_candidate_manifests") as package:
                job._read()
            package.assert_not_called()
            manifest = json.loads((job.run_dir / "training.manifest.json").read_text())
            self.assertEqual(manifest["failure_reason"], "Cancelled by user")
            self.assertEqual(json.loads(job.comparison_report_path.read_text())["status"], "cancelled")

    def test_duplicate_sensor_content_with_changed_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_data(root / "data")
            original = pd.read_csv(root / "data/subject0/posture0.csv")
            original["label"] = CLASS_NAMES[1]
            original = original.assign(elapsed_s=original["elapsed_s"] + 42)
            original.to_csv(root / "data/subject1/extra.csv", index=False)
            with mock.patch("src.model_comparison.build_model") as build, self.assertRaisesRegex(ValueError, "Duplicate sensor"):
                train_comparison(root / "data", root / "run", (3,), 41, progress=lambda _: None)
            build.assert_not_called()

    def test_model_multiselect_validation_and_backward_compatible_configure(self) -> None:
        state = TrainingWizardState()
        state.configure((3,), 42, ("random_forest", "rbf_svm", "rbf_svm"))
        self.assertEqual(state.models, ("random_forest", "rbf_svm"))
        state.configure((5,), 17)
        self.assertEqual(state.models, ("random_forest", "rbf_svm"))
        with self.assertRaises(ValueError):
            state.configure((3,), 42, ())
        with self.assertRaises(ValueError):
            state.configure((3,), 42, ("unknown",))

    def test_distinct_same_named_recordings_keep_subject_identity_when_staged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_data(root / "data")
            extra = root / "data/etc/subject0"
            extra.mkdir(parents=True)
            frame = pd.read_csv(root / "data/subject0/posture0.csv")
            frame.loc[:, SENSOR_COLUMNS] = frame[SENSOR_COLUMNS] + 0.2
            frame.to_csv(extra / "posture0.csv", index=False)
            service = DatasetService(root)
            staged = service.stage_for_training(service.records(), root / "stage")
            self.assertEqual(len(list((staged / "subject0").glob("*.csv"))), 8)
            mapping = json.loads((staged / "staged_sources.json").read_text())
            self.assertEqual(len(mapping), 29)
            self.assertEqual({row["subject"] for row in mapping}, {f"subject{i}" for i in range(4)})
