from __future__ import annotations

from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.augmentation_pipeline_v1 import (GAIN, VOTING, build_augmentation_pipeline,
    fit_augmentation_pipeline, validate_augmentation_pipeline, verify_fitted_augmentation_identity)
from src.iterative_pipeline import build_pipeline
from src.record_gain_features_v1 import augment_record_gains, sequence_features
from src.window_pipeline import extract_window_features


def recipe(mode="identity", strength=0., **changes):
    value = {"family": GAIN, "model_params": {"n_estimators": 8, "min_samples_leaf": 1},
        "training": {"seed": 53, "augmentation": {"mode": mode, "strength": strength}}}
    value.update(changes)
    return value


def fixture():
    y = np.repeat(np.arange(7), 6)
    groups = np.tile(np.repeat(["person-a", "person-b"], 3), 7)
    records = np.asarray([f"{person}/class-{label}/record" for person, label in zip(groups, y)])
    X = np.random.default_rng(924).normal(200, 30, size=(len(y), 36, 16)).astype(np.float32)
    X[:, :, :7] += np.eye(7, dtype=np.float32)[y, None, :] * 30
    X[0, 0, :2] = [0., -0.]
    return X, y, groups, records


class AugmentationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.groups, self.records = fixture()

    def fit(self, spec=None, **kwargs):
        spec = recipe() if spec is None else spec
        return fit_augmentation_pipeline(build_augmentation_pipeline(spec), spec, self.X, self.y,
            groups=self.groups, record_ids=self.records, **kwargs)

    def test_identity_path_is_exactly_equal_to_legacy_and_raw_feature_scaler_et(self):
        spec = recipe()
        features = sequence_features(self.X)
        np.testing.assert_array_equal(features, extract_window_features(self.X))
        model = self.fit(spec)
        baseline_spec = {"family": "extra_trees", "base_method": "standard", "window_seconds": 3,
            "feature_engineering": {"mode": "all"}, "preprocessing": {"scaler": "standard"},
            "model_params": model.spec_["model_params"], "training": {"seed": 53, "weighting": "class_balanced"},
            "postprocessing": {"kind": "argmax"}}
        legacy = build_pipeline(baseline_spec).fit(features, self.y, groups=self.groups)
        explicit = Pipeline([("scale", StandardScaler()), ("model", ExtraTreesClassifier(
            **model.spec_["model_params"], class_weight="balanced", random_state=53, n_jobs=1))]).fit(features, self.y)
        np.testing.assert_array_equal(model.scaler_.transform(features), explicit.named_steps["scale"].transform(features))
        self.assertEqual(model.scaler_.transform(features).dtype, np.float32)
        p = model.predict_proba(self.X)
        np.testing.assert_array_equal(p, legacy.predict_proba(features))
        np.testing.assert_array_equal(p, explicit.predict_proba(features))
        self.assertTrue(verify_fitted_augmentation_identity(model, spec, 53))

    def test_gain_modes_replace_each_train_row_once_before_feature_extraction(self):
        before = self.X.copy()
        for mode in ("common", "per_sensor"):
            with self.subTest(mode=mode):
                spec = recipe(mode, .05)
                original_spec = deepcopy(spec)
                augmented, audit = augment_record_gains(self.X, self.y, self.groups, self.records,
                    mode=mode, strength=.05, seed=53)
                features = sequence_features(augmented)
                expected_scaler = StandardScaler().fit(features)
                model = self.fit(spec, validation_data=(np.full((2, 36, 16), np.nan), [-1, 9]))
                self.assertEqual(model.gain_audit_, audit)
                self.assertEqual(model.scaler_.n_samples_seen_, len(self.y))
                np.testing.assert_array_equal(model.scaler_.mean_, expected_scaler.mean_)
                np.testing.assert_array_equal(model.scaler_.transform(features), expected_scaler.transform(features))
                self.assertFalse(np.array_equal(features, sequence_features(self.X)))
                self.assertFalse(model.training_details_["external_validation_used"])
                self.assertFalse(model.training_details_["inference_augmentation"])
                self.assertEqual(model.training_details_["model_input_dtype"], "float32")
                self.assertEqual(spec, original_spec)
                self.assertTrue(verify_fitted_augmentation_identity(model, spec, 53))
                json.dumps(model.training_details_, allow_nan=False)
        np.testing.assert_array_equal(self.X, before)

    def test_inference_never_generates_gains_or_uses_record_ids(self):
        model = self.fit(recipe("per_sensor", .05))
        untouched = self.X[:5].copy()
        raw_features = extract_window_features(untouched)
        expected = model.estimator_.predict_proba(model.scaler_.transform(raw_features))
        with patch("src.augmentation_pipeline_v1.augment_record_gains", side_effect=AssertionError("Inference must not draw gains")):
            np.testing.assert_array_equal(model.predict_proba(untouched), expected)
        np.testing.assert_array_equal(untouched, self.X[:5])

    def test_refit_reload_and_batch_predictions_are_exact_and_inputs_immutable(self):
        spec = recipe("per_sensor", .02)
        first = self.fit(spec)
        second = self.fit(spec, validation_data=(self.X * 10, (self.y + 1) % 7))
        p = first.predict_proba(self.X)
        np.testing.assert_array_equal(p, second.predict_proba(self.X))
        self.assertEqual(first.gain_audit_, second.gain_audit_)
        np.testing.assert_array_equal(p, np.concatenate([first.predict_proba(self.X[:9]), first.predict_proba(self.X[9:])]))
        stream = io.BytesIO(); joblib.dump(first, stream); stream.seek(0)
        restored = joblib.load(stream)
        np.testing.assert_array_equal(p, restored.predict_proba(self.X))
        np.testing.assert_array_equal(restored.predict(self.X), p.argmax(1))
        self.assertTrue(verify_fitted_augmentation_identity(restored, spec, 53))
        self.assertEqual(clone(first).spec, validate_augmentation_pipeline(spec))

    def test_ensemble_preserves_shared_record_contract_and_propagates_override_seed(self):
        first, second = recipe("common", .02), recipe("per_sensor", .02)
        spec = {"family": VOTING, "members": [first, second], "training": {"seed": 53}}
        model = build_augmentation_pipeline(spec, seed=71).fit(self.X, self.y, groups=self.groups, record_ids=self.records)
        self.assertEqual([member.effective_seed_ for member in model.members_], [71, 71])
        self.assertEqual([member.estimator_.random_state for member in model.members_], [71, 71])
        self.assertTrue(all(member.record_metadata_ == model.record_metadata_ for member in model.members_))
        expected = np.mean([member.predict_proba(self.X) for member in model.members_], axis=0)
        np.testing.assert_array_equal(model.predict_proba(self.X), expected)
        self.assertTrue(verify_fitted_augmentation_identity(model, spec, 71))
        stream = io.BytesIO(); joblib.dump(model, stream); stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).predict_proba(self.X), expected)
        for members in ([first], [first, first], [first, spec],
                [first, {**second, "training": {**second["training"], "seed": 54}}],
                [first, {**second, "training": {**second["training"], "data_policy": "include_ordinary_reserved_training_records_v1"}}]):
            with self.subTest(members=len(members)), self.assertRaises(ValueError):
                validate_augmentation_pipeline({**spec, "members": members})

    def test_record_identity_purity_and_shape_validation_precede_backend_fit(self):
        changed_groups = self.groups.astype(object)
        changed_groups[0] = "different-person"
        mixed_records = self.records.astype(object)
        mixed_records[6] = " " + self.records[0] + " "
        cases = [(self.X, self.y, None, self.records), (self.X, self.y, self.groups, None),
            (self.X, self.y, changed_groups, self.records), (self.X, self.y, self.groups, mixed_records),
            (self.X, self.y, self.groups[:-1], self.records),
            (self.X, self.y, self.groups, np.repeat("", len(self.y))),
            (self.X[:, :-1], self.y, self.groups, self.records),
            (np.full_like(self.X, np.nan), self.y, self.groups, self.records),
            (self.X, self.y.astype(float), self.groups, self.records),
            (self.X, np.zeros_like(self.y), self.groups, self.records)]
        with patch.object(ExtraTreesClassifier, "fit") as backend_fit:
            for X, y, groups, records in cases:
                with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                    build_augmentation_pipeline(recipe()).fit(X, y, groups=groups, record_ids=records)
        backend_fit.assert_not_called()

    def test_strict_normalized_schema_rejects_undeclared_changes(self):
        normalized = validate_augmentation_pipeline(recipe("common", .02))
        self.assertEqual(normalized["training"]["augmentation"], {"mode": "common", "strength": .02,
            "distribution": "uniform", "row_policy": "replace", "record_seed_policy": "sha256_record_v1"})
        cases = [recipe("identity", .02), recipe("per_sensor", .1), recipe("common", True),
            recipe(training={"augmentation": {"mode": "common", "strength": .02, "row_policy": "append"}}),
            recipe(training={"augmentation": {"mode": "common", "strength": .02, "record_seed_policy": "row_index"}}),
            recipe(training={"augmentation": {"mode": "common", "strength": .02, "distribution": "normal"}}),
            recipe(training={"weighting": "subject_class_balanced"}), recipe(model_params={"random_state": 1}),
            recipe(model_params={"n_estimators": True}), recipe(model_params={"max_features": float("nan")}),
            recipe(window_seconds=True), recipe(feature_engineering={"mode": "all"}),
            recipe(preprocessing={"scaler": "none"}), recipe(postprocessing={"kind": "macro_f1_bias"}),
            {"family": VOTING, "members": [recipe("common", .02), recipe("per_sensor", .02)],
                "training": {"seed": 53, "augmentation": {"mode": "identity", "strength": 0}}}]
        for spec in cases:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_augmentation_pipeline(spec)
        with self.assertRaises(ValueError):
            fit_augmentation_pipeline(build_augmentation_pipeline(recipe()), recipe("per_sensor", .02),
                self.X, self.y, groups=self.groups, record_ids=self.records)

    def test_fitted_identity_rejects_gains_records_features_scaler_and_backend_tampering(self):
        spec = recipe("per_sensor", .02)
        original = self.fit(spec)
        first_record = sorted(original.gain_audit_["record_gains"])[0]
        mutations = [lambda m: setattr(m, "effective_seed_", 999),
            lambda m: m.gain_audit_["record_gains"][first_record].__setitem__(0, 1.9),
            lambda m: m.record_metadata_[0].update(subject="other-person"),
            lambda m: m.record_metadata_[0]["row_indices"].__setitem__(0, 999),
            lambda m: m.scaler_.mean_.__setitem__(0, m.scaler_.mean_[0] + 1),
            lambda m: m.feature_names_out_.__setitem__(0, "changed"),
            lambda m: m.estimator_.set_params(random_state=999),
            lambda m: m.estimator_.set_params(class_weight=None),
            lambda m: m.estimator_.classes_.__setitem__(0, 99)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = deepcopy(original)
                mutation(changed)
                with self.assertRaises(ValueError):
                    verify_fitted_augmentation_identity(changed, spec, 53)
        original.spec["training"]["augmentation"]["strength"] = .05
        with self.assertRaises(ValueError):
            original.predict_proba(self.X)


if __name__ == "__main__":
    unittest.main()
