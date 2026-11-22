from __future__ import annotations

import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import joblib
import numpy as np

from src.app_services import TrainingJob, TrainingWizardState
from src.extended_models import MiniRocketClassifier, SmallNeuralClassifier
from src.model_catalog import DEFAULT_MODEL_FAMILIES, MODEL_NAMES, input_contract, missing_model_packages, model_input
from functools import partial
from src.model_comparison import _dataset, aligned_predict_proba, build_model, train_comparison as _train_comparison

# These contracts count fits on the single validation split; LOSO selection has
# its own tests in test_loso_selection.py.
train_comparison = partial(_train_comparison, selection_metric="validation")
from src.model_runtime import ModelRegistry
from src.model_runtime.sklearn_adapter import SklearnAdapter
from src.model_runtime.validation import ArtifactValidator
from src.posture_core import CLASS_NAMES, read_sensor_csv
from tests import test_model_comparison as fixtures

ROOT, SMALL_GRID = fixtures.ROOT, fixtures.SMALL_GRID


EXTENDED_GRID = {**SMALL_GRID,
    "catboost": [{"iterations": 8, "depth": 3}],
    "xgboost": [{"n_estimators": 8, "max_depth": 2}],
    "minirocket": [{"num_kernels": 84, "C": 0.1}],
    "cnn1d": [{"hidden_size": 8, "epochs": 3, "patience": 2}],
    "mlp": [{"hidden_size": 8, "epochs": 3, "patience": 2}],
}


class ExtendedPipelineTests(unittest.TestCase):
    @unittest.skipUnless(all(not missing_model_packages(family) for family in MODEL_NAMES), "Install requirements.txt for the extended runtime tests")
    def test_nine_models_share_split_lock_reload_and_candidate_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures.ModelComparisonTests().make_data(root / "data")
            data = _dataset(root / "data", 3, 41)
            job = TrainingJob(ROOT, (), 41, windows=(3,), models=tuple(MODEL_NAMES), state_root=root)
            final_features = data["X"][data["masks"]["final_evaluation"]]
            final_sequence = data["sequences"][data["masks"]["final_evaluation"]]
            final_calls = []
            def predict(model, values, count):
                if np.array_equal(values, final_features) or np.array_equal(values, final_sequence):
                    lock = json.loads(job.comparison_report_path.read_text())
                    self.assertTrue(lock["selection_locked"])
                    self.assertFalse(lock["final_evaluation_used_for_selection"])
                    self.assertEqual(len(lock["candidates"]), 9)
                    final_calls.append(lock["selected"])
                return aligned_predict_proba(model, values, count)
            original_fit = SmallNeuralClassifier.fit
            def neural_fit(model, values, labels, *, validation_data=None):
                expected = model_input(model.architecture, data["sequences"], data["X"])
                np.testing.assert_array_equal(values, expected[data["masks"]["train"]])
                np.testing.assert_array_equal(validation_data[0], expected[data["masks"]["validation"]])
                np.testing.assert_array_equal(validation_data[1], data["y"][data["masks"]["validation"]])
                return original_fit(model, values, labels, validation_data=validation_data)
            with mock.patch("src.model_comparison.MODEL_CANDIDATES", EXTENDED_GRID), \
                 mock.patch("src.model_comparison.aligned_predict_proba", side_effect=predict), \
                 mock.patch.object(SmallNeuralClassifier, "fit", neural_fit):
                report = train_comparison(root / "data", job.run_dir, (3,), 41,
                                          model_families=MODEL_NAMES, progress=lambda _: None)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(final_calls), 9)
            self.assertTrue(all(item == report["selected"] for item in final_calls))
            self.assertEqual(set(report["model_families"]), set(MODEL_NAMES))
            manifests = job._write_pending_candidate_manifests()
            self.assertEqual(len(manifests), 9)
            registry = ModelRegistry(ROOT, root)
            for manifest in manifests:
                descriptor = ArtifactValidator().discover_manifest(manifest).descriptor
                self.assertIsNotNone(descriptor)
                model = registry.validate_deployment(registry.validate_runtime(descriptor))
                self.assertEqual(model.deployment_validation_status, "passed", str(model))
                self.assertEqual(model.approval_status, "pending")
                adapter = SklearnAdapter(descriptor)
                bundle = adapter.bundle
                family = bundle["model_family"]
                self.assertEqual(bundle["input_representation"], input_contract(family))
                self.assertEqual(bundle["dataset_fingerprint"], data["fingerprint"])
                frame = read_sensor_csv(root / "data/subject0/posture0.csv")
                window_index = int(np.flatnonzero(data["meta"]["path"].to_numpy() == str(root / "data/subject0/posture0.csv"))[0])
                expected_input = model_input(family, data["sequences"], data["X"])[window_index:window_index + 1]
                expected = aligned_predict_proba(bundle["model"], expected_input, len(CLASS_NAMES))
                np.testing.assert_allclose(adapter.predict_proba(frame), expected, atol=1e-6)
                if family in {"cnn1d", "mlp"}:
                    train_input = model_input(family, data["sequences"], data["X"])[data["masks"]["train"]]
                    axes = (0, 1) if family == "cnn1d" else 0
                    np.testing.assert_allclose(bundle["model"].mean_, train_input.mean(axis=axes, dtype=np.float64), rtol=1e-6)
                    self.assertIn("torch", descriptor.metadata["_runtime"]["packages"])
                if family in {"cnn1d", "minirocket"}:
                    with self.assertRaisesRegex(ValueError, "전처리"):
                        ArtifactValidator._check_preprocessing_contract(replace(descriptor, preprocessing={"kind": "feature_vector", "feature_count": 0}), adapter)

    @unittest.skipUnless(all(not missing_model_packages(family) for family in ("minirocket", "cnn1d", "mlp")), "Install Numba and PyTorch for sequence/neural tests")
    def test_short_windows_seed_reproducibility_and_serialization(self):
        rng = np.random.default_rng(19)
        y = np.tile(np.arange(7), 3)
        X = rng.uniform(1, 3, (21, 12, 16)).astype(np.float32)
        for family in ("minirocket", "cnn1d", "mlp"):
            with self.subTest(family=family):
                values = model_input(family, X)
                first = build_model(family, EXTENDED_GRID[family][0], 12).fit(values, y)
                second = build_model(family, EXTENDED_GRID[family][0], 12).fit(values, y)
                probability = first.predict_proba(values)
                np.testing.assert_allclose(probability, second.predict_proba(values), atol=1e-7)
                np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
                buffer = io.BytesIO()
                joblib.dump(first, buffer)
                buffer.seek(0)
                np.testing.assert_allclose(probability, joblib.load(buffer).predict_proba(values), atol=1e-7)
                invalid = values.copy()
                invalid.flat[0] = np.nan
                with self.assertRaises(ValueError):
                    first.predict_proba(invalid)
                with self.assertRaises(ValueError):
                    first.predict_proba(values[..., :-1])

    def test_optional_dependency_failure_is_recorded_before_reading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch("src.model_comparison.missing_model_packages", return_value=("torch",)), \
                 mock.patch("src.model_comparison._dataset") as dataset, \
                 self.assertRaisesRegex(RuntimeError, "requires torch"):
                train_comparison(root / "data", root / "run", (3,), 41, ("cnn1d",), progress=lambda _: None)
            dataset.assert_not_called()
            report = json.loads((root / "run/artifacts/model_comparison.json").read_text())
            self.assertFalse(report["selection_locked"])
            self.assertEqual(report["status"], "failed")

    def test_wizard_accepts_all_nine_and_preserves_existing_default(self):
        wizard = TrainingWizardState()
        self.assertEqual(wizard.models, DEFAULT_MODEL_FAMILIES)
        wizard.configure((1, 3), 41, MODEL_NAMES)
        self.assertEqual(wizard.models, tuple(MODEL_NAMES))


if __name__ == "__main__":
    unittest.main()
