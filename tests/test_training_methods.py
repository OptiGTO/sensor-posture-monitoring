from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

from src.model_comparison import build_model, train_comparison
from src.posture_core import feature_names
from src.training_methods import (available_training_methods, build_method_model, fit_method_model,
    method_candidates, subject_balanced_weights, validate_training_method)


class TrainingMethodsTests(unittest.TestCase):
    def setUp(self):
        # Checkpoint registration has separate fixtures in test_checkpoint_registry.
        checkpoints = mock.patch("src.training_methods.checkpoint_method_entries", return_value=[])
        checkpoints.start()
        self.addCleanup(checkpoints.stop)
        rng = np.random.default_rng(59)
        self.X = rng.normal(size=(40, len(feature_names())))
        self.y = np.tile([0, 1], 20)
        self.X[:, 11] += self.y * 4
        self.X[:, 301] += self.y * 4

    def test_standard_predictions_and_grid_are_unchanged(self):
        for family, params in {
            "rbf_svm": {"C": 1, "gamma": "scale"},
            "random_forest": {"n_estimators": 5},
            "extra_trees": {"n_estimators": 5},
        }.items():
            expected = build_model(family, params, 42).fit(self.X, self.y)
            model = build_method_model(family, params, 42, "standard")
            fit_method_model(model, family, "standard", self.X, self.y)
            np.testing.assert_array_equal(model.predict_proba(self.X), expected.predict_proba(self.X))
        from src.model_comparison import MODEL_CANDIDATES
        self.assertEqual(method_candidates("rbf_svm", "standard"), MODEL_CANDIDATES["rbf_svm"])

    def test_relative_and_selected_serialize_with_original_input_contract(self):
        for method in ("relative", "selected"):
            model = build_method_model("rbf_svm", {"C": 1, "gamma": "scale"}, 4, method)
            fit_method_model(model, "rbf_svm", method, self.X, self.y,
                validation_data=(self.X + 1e6, 1 - self.y))
            self.assertEqual(model.n_features_in_, 596)
            transformed = model[:-1].transform(self.X)
            self.assertEqual(transformed.shape[1], 298 if method == "relative" else 96)
            if method == "relative":
                expected_indices = [i for i, name in enumerate(feature_names()) if name.startswith("rel_")]
                np.testing.assert_array_equal(model.named_steps["relative"].indices_, expected_indices)
                expected_scaler_mean = self.X[:, expected_indices].mean(axis=0)
            else:
                # The selected columns must equal training-only ANOVA despite an
                # adversarial validation distribution passed to the generic API.
                reference = SelectKBest(f_classif, k=96).fit(self.X, self.y)
                np.testing.assert_array_equal(model.named_steps["selection"].get_support(), reference.get_support())
                expected_scaler_mean = self.X[:, reference.get_support()].mean(axis=0)
            np.testing.assert_allclose(model.named_steps["scale"].mean_, expected_scaler_mean)
            stream = io.BytesIO()
            joblib.dump(model, stream)
            stream.seek(0)
            restored = joblib.load(stream)
            np.testing.assert_allclose(restored.predict_proba(self.X), model.predict_proba(self.X))
            self.assertEqual(restored.training_details_["training_method"]["id"], method)

    def test_subject_weights_equalize_subject_and_within_subject_classes(self):
        y = np.array([0, 0, 0, 1, 0, 1, 1])
        groups = np.array(["a"] * 4 + ["b"] * 3)
        weights = subject_balanced_weights(y, groups)
        self.assertAlmostEqual(weights.mean(), 1)
        for subject in ("a", "b"):
            self.assertAlmostEqual(weights[groups == subject].sum(), len(y) / 2)
            for label in (0, 1):
                self.assertAlmostEqual(weights[(groups == subject) & (y == label)].sum(), len(y) / 4)
        X = self.X[:len(y)]
        model = build_method_model("rbf_svm", {"C": 1, "gamma": "scale"}, 0, "subject_balanced")
        fit_method_model(model, "rbf_svm", "subject_balanced", X, y, groups=groups)
        self.assertIsNone(model.named_steps["model"].class_weight)
        np.testing.assert_allclose(model.named_steps["scale"].mean_, np.average(X, axis=0, weights=weights))
        self.assertEqual(model.training_details_["training_subject_count"], 2)
        with self.assertRaisesRegex(ValueError, "subject IDs"):
            fit_method_model(model, "rbf_svm", "subject_balanced", X, y)

    def test_registry_fails_closed_and_limits_families_and_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "registry.json"
            for text in ("not JSON", "[]", '{"schema_version":1,"methods":[]}'):
                registry.write_text(text)
                self.assertEqual([row["id"] for row in available_training_methods(registry)], ["standard"])
            registry.write_text(json.dumps({"schema_version": 1, "methods": [
                {"id": "selected", "families": ["rbf_svm"], "window_seconds": [3], "evidence": {"improved": False}},
                {"id": "relative", "families": ["cnn1d"], "window_seconds": [3], "evidence": {"improved": True}},
                {"id": "subject_balanced", "families": ["rbf_svm"], "window_seconds": [3], "evidence": {"improved": True}},
            ]}))
            choices = available_training_methods(registry)
            self.assertEqual([row["id"] for row in choices], ["standard", "subject_balanced"])
            with mock.patch("src.training_methods.REGISTRY_PATH", registry):
                validate_training_method("subject_balanced", ["rbf_svm"], [3])
                with self.assertRaises(ValueError):
                    validate_training_method("subject_balanced", ["xgboost"], [3])
                with self.assertRaises(ValueError):
                    validate_training_method("subject_balanced", ["rbf_svm"], [5])
                with self.assertRaises(ValueError):
                    train_comparison(Path(directory) / "missing_data", Path(directory) / "output",
                        [3], 42, ["rbf_svm"], training_method="selected")
                self.assertFalse((Path(directory) / "output").exists())

    def test_nonstandard_bundle_contains_method_and_runtime_input_is_unchanged(self):
        from tests.test_model_comparison import ModelComparisonTests
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ModelComparisonTests().make_data(root / "data")
            entry = {"id": "relative", "label": "상대압력 특징", "families": ["rbf_svm"],
                "window_seconds": [3], "evidence": {"improved": True, "source": "synthetic unit fixture"}}
            with mock.patch("src.training_methods.available_training_methods", return_value=[entry]), \
                    mock.patch("src.model_comparison.MODEL_CANDIDATES", {"rbf_svm": [{"C": 1, "gamma": "scale"}]}):
                report = train_comparison(root / "data", root / "run", (3,), 41, ("rbf_svm",),
                    training_method="relative", progress=lambda _: None)
            self.assertEqual(report["training_method"], "relative")
            artifact = root / "run/artifacts" / report["models"][0]["artifact_file"]
            bundle = joblib.load(artifact)
            self.assertEqual(bundle["training_method"], "relative")
            self.assertEqual(bundle["training_method_evidence"], entry["evidence"])
            self.assertEqual(len(bundle["feature_names"]), 596)
            self.assertEqual(bundle["model"].n_features_in_, 596)
            config = json.loads((artifact.parent / report["models"][0]["config_file"]).read_text())
            self.assertEqual(config["training_method"], "relative")
            self.assertEqual(len(config["training_details"]["selected_feature_names"]), 298)

    def test_neural_augmentation_is_training_only_and_survives_serialization(self):
        from importlib.util import find_spec
        if find_spec("torch") is None:
            self.skipTest("Torch is optional")
        from src.extended_models import SmallNeuralClassifier
        rng = np.random.default_rng(8)
        X = rng.uniform(10, 20, (12, 12, 16)).astype(np.float32)
        y = np.tile([0, 1], 6)
        params = {"hidden_size": 4, "epochs": 2, "patience": 2, "gain_std": 0.03, "noise_fraction": 0.01}
        model = build_method_model("cnn1d", params, 3, "cnn_augmentation")
        fit_method_model(model, "cnn1d", "cnn_augmentation", X, y, validation_data=(X + 1000, y))
        np.testing.assert_allclose(model.mean_, X.mean(axis=(0, 1), dtype=np.float64), rtol=1e-6)
        np.testing.assert_allclose(model.augmentation_noise_scale_, X.std(axis=(0, 1), dtype=np.float64), rtol=1e-6)
        self.assertTrue(model.training_details_["augmentation"]["training_only"])
        with mock.patch.object(SmallNeuralClassifier, "_augment_training_values", side_effect=AssertionError("Inference augmented")):
            expected = model.predict_proba(X)
            np.testing.assert_array_equal(model.predict_proba(X), expected)
            stream = io.BytesIO()
            joblib.dump(model, stream)
            stream.seek(0)
            restored = joblib.load(stream)
            np.testing.assert_array_equal(restored.predict_proba(X), expected)
        # Calling the helper must also leave the input batch byte-for-byte intact.
        original = X.copy()
        self.assertFalse(np.array_equal(model._augment_training_values(X, np.random.default_rng(3)), X))
        np.testing.assert_array_equal(X, original)


if __name__ == "__main__":
    unittest.main()
