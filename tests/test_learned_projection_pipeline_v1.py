"""Synthetic TRAIN-routing and persistence contracts for learned coordinates."""
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src import learned_projection_pipeline_v1 as subject
from src.iterative_pipeline import build_pipeline as legacy_pipeline
from src.posture_core import feature_names


def recipe(mode="identity"):
    return subject.validate_projection_pipeline({"family": "projection_extra_trees",
        "feature_engineering": {"mode": mode},
        "model_params": {"n_estimators": 3, "min_samples_leaf": 1}, "training": {"seed": 53}})


class LearnedProjectionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.y = np.repeat(np.arange(7), 14)
        self.X = np.random.default_rng(1701).normal(size=(len(self.y), 596)).astype(np.float32)
        self.X[:, 0] += self.y
        self.X[:, 1] = 0
        self.groups = np.asarray([f"person-{i % 3}" for i in range(len(self.y))])
        self.pca, self.lda = recipe("pca64"), recipe("all_plus_lda6")
        self.ensemble = subject.validate_projection_pipeline({"family": "projection_soft_voting",
            "members": [self.pca, self.lda], "training": {"seed": 53}})

    def fit(self, spec, X=None, seed=None):
        return subject.build_projection_pipeline(spec, seed=seed).fit(
            self.X if X is None else X, self.y, groups=self.groups)

    def test_only_fixed_modes_complete_settings_and_distinct_nonnested_members(self):
        original = deepcopy(self.ensemble)
        normalized = subject.validate_projection_pipeline(original)
        normalized["members"][0]["feature_engineering"]["mode"] = "identity"
        self.assertEqual(original, self.ensemble)
        for changed in (None, {**self.pca, "window_seconds": True}, {**self.pca, "unexpected": 1},
                {**self.pca, "feature_engineering": {"mode": "pca32"}},
                {**self.pca, "feature_engineering": {"mode": "pca64", "n_components": 32}},
                {**self.pca, "model_params": {"max_features": 65}},
                {**self.pca, "preprocessing": {"scaler": "quantile_normal"}},
                {**self.pca, "postprocessing": {"kind": "softmax"}},
                {**self.ensemble, "members": [self.pca, self.pca]},
                {**self.ensemble, "members": [self.pca, self.ensemble]},
                {**self.ensemble, "members": [self.pca, {**self.lda, "training": {"seed": 54}}]}):
            with self.subTest(value=changed), self.assertRaises(ValueError):
                subject.validate_projection_pipeline(changed)

    def test_identity_matches_native_200_tree_baseline_bytes_in_both_dtypes(self):
        spec = subject.validate_projection_pipeline({"family": "projection_extra_trees", "training": {"seed": 53}})
        reference = {"family": "extra_trees", "model_params": spec["model_params"], "training": {"seed": 53}}
        for dtype in (np.float32, np.float64):
            X = self.X.astype(dtype)
            candidate, baseline = self.fit(spec, X), legacy_pipeline(reference).fit(X, self.y, groups=self.groups)
            query = np.concatenate([X, X[:2] * 10000]).astype(dtype)
            self.assertEqual(candidate.predict_proba(query).tobytes(), baseline.predict_proba(query).tobytes())
            self.assertEqual(candidate.scaler_.mean_.tobytes(), baseline.named_steps["scale"].mean_.tobytes())
            self.assertEqual(candidate.standardized_state_sha256_, candidate.transformed_state_sha256_)
            self.assertTrue(subject.verify_fitted_projection_identity(candidate, spec, 53))

    def test_scaler_then_projector_receive_only_actual_train_rows_and_labels(self):
        class UnreadableValidation:
            def __array__(self, *args, **kwargs):
                raise AssertionError("Validation must not enter fitting")
        original = subject.LearnedProjectionV1.fit
        for spec in (self.pca, self.lda):
            captured = []
            def capture(projector, X, y):
                captured.append((X.copy(), y.copy()))
                return original(projector, X, y)
            with patch.object(subject.LearnedProjectionV1, "fit", capture):
                model = subject.build_projection_pipeline(spec)
                subject.fit_projection_pipeline(model, spec, self.X, self.y, groups=self.groups,
                    validation_data=UnreadableValidation())
            self.assertEqual(len(captured), 1)
            np.testing.assert_array_equal(captured[0][0], model.scaler_.transform(self.X))
            np.testing.assert_array_equal(captured[0][1], self.y.astype(np.int64))
            np.testing.assert_allclose(model.scaler_.mean_, self.X.mean(axis=0, dtype=np.float64))
            self.assertEqual(model.scaler_.n_samples_seen_, len(self.X))
            self.assertEqual(model.projector_.training_input_sha256_, model.standardized_state_sha256_)
            self.assertEqual(model.projector_.training_labels_sha256_, model.training_labels_sha256_)
            self.assertEqual(model.estimator_.n_features_in_, 64 if spec == self.pca else 602)
            self.assertTrue(subject.verify_fitted_projection_identity(model, spec, 53))

    def test_native_forest_and_stored_projection_define_prediction_without_recalibration(self):
        for spec in (self.pca, self.lda):
            model = self.fit(spec)
            before = deepcopy(model.training_details_)
            transformed = model.projector_.transform(model.scaler_.transform(self.X))
            np.testing.assert_array_equal(model.predict_proba(self.X), model.estimator_.predict_proba(transformed))
            first = model.predict_proba(self.X[:1])
            joined = model.predict_proba(np.concatenate([self.X[:1], np.full_like(self.X, 10000)]))
            self.assertEqual(first.tobytes(), joined[:1].tobytes())
            self.assertEqual(before, model.training_details_)

    def test_equal_ensemble_and_seed_override_preserve_member_training_scope(self):
        model = self.fit(self.ensemble, seed=54)
        expected = np.mean([member.predict_proba(self.X) for member in model.members_], axis=0)
        np.testing.assert_array_equal(model.predict_proba(self.X), expected)
        self.assertEqual([member.effective_seed_ for member in model.members_], [54, 54])
        self.assertEqual({m.training_input_sha256_ for m in model.members_}, {model.training_input_sha256_})
        self.assertTrue(subject.verify_fitted_projection_identity(model, self.ensemble, 54))

    def test_refit_reload_and_partition_predictions_are_exact(self):
        for spec in (self.pca, self.lda, self.ensemble):
            model = self.fit(spec)
            expected = model.predict_proba(self.X)
            stream = io.BytesIO()
            joblib.dump(model, stream)
            stream.seek(0)
            restored = joblib.load(stream)
            self.assertEqual(restored.predict_proba(self.X).tobytes(), expected.tobytes())
            self.assertEqual(self.fit(spec).predict_proba(self.X).tobytes(), expected.tobytes())
            self.assertEqual(np.concatenate([model.predict_proba(self.X[:1]), model.predict_proba(self.X[1:])]).tobytes(), expected.tobytes())
            np.testing.assert_array_equal(model.predict_proba(self.X[::-1]), expected[::-1])
            self.assertTrue(subject.verify_fitted_projection_identity(restored, spec, 53))

    def test_bad_schema_labels_groups_and_constructed_recipe_are_rejected(self):
        for X in (self.X[:, :-1], self.X[:0], self.X[0], np.full_like(self.X, np.nan), self.X.astype(str)):
            with self.assertRaises(ValueError):
                self.fit(self.pca, X)
        model = self.fit(self.pca)
        named = pd.DataFrame(self.X, columns=feature_names())
        np.testing.assert_array_equal(model.predict_proba(named), model.predict_proba(self.X))
        with self.assertRaises(ValueError):
            model.predict_proba(named.iloc[:, ::-1])
        for y in (self.y.astype(float), self.y[:-1], np.zeros_like(self.y)):
            with self.assertRaises(ValueError):
                subject.build_projection_pipeline(self.pca).fit(self.X, y)
        for groups in (self.groups[:-1], [""] * len(self.X), np.arange(len(self.X))):
            with self.assertRaises(ValueError):
                subject.build_projection_pipeline(self.pca).fit(self.X, self.y, groups=groups)
        with self.assertRaises(ValueError):
            subject.fit_projection_pipeline(subject.build_projection_pipeline(self.pca), self.lda, self.X, self.y)
        model.spec["model_params"]["n_estimators"] += 1
        with self.assertRaises(ValueError):
            model.predict_proba(self.X)

    def test_fitted_state_tampering_and_different_scope_member_replacement_fail(self):
        original = self.fit(self.pca)
        for mutate in (lambda m: m.scaler_.mean_.__setitem__(0, 1000.),
                lambda m: m.estimator_.estimators_[0].tree_.value.__setitem__((0, 0, 0), .12345),
                lambda m: m.training_details_.update(external_validation_used=True),
                lambda m: m.training_details_["learned_projection"].update(training_windows=1),
                lambda m: setattr(m, "projector_state_sha256_", "f" * 64),
                lambda m: setattr(m, "feature_names_in_", np.asarray(feature_names()[::-1]))):
            changed = deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                subject.verify_fitted_projection_identity(changed, self.pca, 53)
        model = self.fit(self.ensemble)
        model.members_[1] = self.fit(self.lda, self.X + np.float32(1))
        model.training_details_["members"][1] = deepcopy(model.members_[1].training_details_)
        with self.assertRaisesRegex(ValueError, "different input"):
            subject.verify_fitted_projection_identity(model, self.ensemble, 53)


if __name__ == "__main__":
    unittest.main()
