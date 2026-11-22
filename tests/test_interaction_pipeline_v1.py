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

from src.interaction_features_v1 import (extract_interaction_features, interaction_feature_names,
    interaction_rule, transform_standardized_features)
from src.interaction_pipeline_v1 import (TREE, VOTING, build_interaction_pipeline,
    fit_interaction_pipeline, validate_interaction_pipeline, verify_fitted_interaction_identity)
from src.iterative_pipeline import build_pipeline
from src.record_gain_features_v1 import sequence_features


def recipe(mode="identity", **changes):
    value = {"family": TREE, "feature_engineering": {"mode": mode},
        "model_params": {"n_estimators": 8, "min_samples_leaf": 1}, "training": {"seed": 53}}
    value.update(changes)
    return value


def fixture():
    y = np.repeat(np.arange(7), 6)
    X = np.random.default_rng(1414).normal(200, 30, size=(len(y), 36, 16)).astype(np.float32)
    X[:, :, :7] += np.eye(7, dtype=np.float32)[y, None, :] * 30
    groups = np.tile(np.repeat(["person-a", "person-b"], 3), 7)
    return X, y, groups


class InteractionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.groups = fixture()

    def fit(self, spec=None, **kwargs):
        spec = recipe() if spec is None else spec
        return fit_interaction_pipeline(build_interaction_pipeline(spec), spec,
            self.X, self.y, groups=self.groups, **kwargs)

    def test_identity_control_is_exactly_equal_to_legacy_and_explicit_float32_pipeline(self):
        spec = recipe()
        features = sequence_features(self.X)
        np.testing.assert_array_equal(extract_interaction_features(self.X, "identity"), features)
        model = self.fit(spec)
        legacy_spec = {"family": "extra_trees", "feature_engineering": {"mode": "all"},
            "model_params": model.spec_["model_params"], "training": {"seed": 53, "weighting": "class_balanced"}}
        legacy = build_pipeline(legacy_spec).fit(features, self.y, groups=self.groups)
        explicit = Pipeline([("scale", StandardScaler()), ("model", ExtraTreesClassifier(
            **model.spec_["model_params"], class_weight="balanced", random_state=53, n_jobs=1))]).fit(features, self.y)
        np.testing.assert_array_equal(model.scaler_.transform(features), explicit.named_steps["scale"].transform(features))
        self.assertEqual(model.scaler_.transform(features).dtype, np.float32)
        p = model.predict_proba(self.X)
        np.testing.assert_array_equal(p, legacy.predict_proba(features))
        np.testing.assert_array_equal(p, explicit.predict_proba(features))
        self.assertTrue(verify_fitted_interaction_identity(model, spec, 53))

    def test_feature_extraction_then_train_scaler_then_fixed_transform_order(self):
        before = self.X.copy()
        for mode, width in (("hadamard512", 596), ("temporal_corr120", 716)):
            with self.subTest(mode=mode):
                spec, original_spec = recipe(mode), recipe(mode)
                extracted = extract_interaction_features(self.X, mode)
                control_scaler = StandardScaler()
                scaled = control_scaler.fit_transform(extracted)
                expected_input = transform_standardized_features(scaled, mode)
                calls, original_fit = [], ExtraTreesClassifier.fit
                def observe(estimator, values, labels, **kwargs):
                    calls.append((values.copy(), labels.copy(), kwargs))
                    return original_fit(estimator, values, labels, **kwargs)
                with patch.object(ExtraTreesClassifier, "fit", autospec=True, side_effect=observe):
                    model = self.fit(spec, validation_data=(np.full((7, 36, 16), np.nan), np.arange(7)))
                self.assertEqual(len(calls), 1)
                np.testing.assert_array_equal(calls[0][0], expected_input)
                np.testing.assert_array_equal(calls[0][1], self.y)
                self.assertEqual(calls[0][2], {})
                self.assertEqual(calls[0][0].dtype, np.float32)
                np.testing.assert_array_equal(model.scaler_.mean_, extracted.mean(0, dtype=np.float64))
                np.testing.assert_array_equal(model.scaler_.transform(extracted), scaled)
                self.assertEqual(model.scaler_.n_samples_seen_, len(self.y))
                self.assertEqual(model.estimator_.n_features_in_, width)
                self.assertEqual(model.n_extracted_features_, width)
                self.assertEqual(model.n_transformed_features_, width)
                np.testing.assert_array_equal(model.feature_names_out_, interaction_feature_names(mode))
                self.assertEqual(model.interaction_rule_, interaction_rule(mode))
                self.assertFalse(model.training_details_["external_validation_used"])
                self.assertFalse(model.training_details_["evaluation_population_normalization"])
                self.assertEqual(spec, original_spec)
                self.assertTrue(verify_fitted_interaction_identity(model, spec, 53))
                json.dumps(model.training_details_, allow_nan=False)
        np.testing.assert_array_equal(self.X, before)

    def test_same_seed_refit_reload_and_independent_window_predictions_are_exact(self):
        for mode in ("identity", "hadamard512", "temporal_corr120"):
            with self.subTest(mode=mode):
                spec = recipe(mode)
                first, second = self.fit(spec), self.fit(spec, validation_data=(self.X * 100, (self.y + 1) % 7))
                p = first.predict_proba(self.X)
                np.testing.assert_array_equal(p, second.predict_proba(self.X))
                self.assertEqual(first.estimator_state_sha256_, second.estimator_state_sha256_)
                np.testing.assert_array_equal(p, np.concatenate([first.predict_proba(self.X[:9]), first.predict_proba(self.X[9:])]))
                np.testing.assert_array_equal(p[:1], first.predict_proba(self.X[:1]))
                stream = io.BytesIO(); joblib.dump(first, stream); stream.seek(0)
                restored = joblib.load(stream)
                np.testing.assert_array_equal(p, restored.predict_proba(self.X))
                np.testing.assert_array_equal(restored.predict(self.X), p.argmax(1))
                self.assertTrue(verify_fitted_interaction_identity(restored, spec, 53))
                self.assertEqual(clone(first).spec, validate_interaction_pipeline(spec))

    def test_inference_uses_frozen_train_scaler_and_does_not_adapt_to_query_population(self):
        model = self.fit(recipe("hadamard512"))
        original_mean = model.scaler_.mean_.copy()
        before_details = deepcopy(model.training_details_)
        query = self.X[:1].copy()
        alone = model.predict_proba(query)
        batch = np.concatenate([query, self.X[1:8] * 100])
        np.testing.assert_array_equal(alone, model.predict_proba(batch)[:1])
        expected = model.estimator_.predict_proba(transform_standardized_features(
            model.scaler_.transform(extract_interaction_features(query, "hadamard512")), "hadamard512"))
        np.testing.assert_array_equal(alone, expected)
        np.testing.assert_array_equal(model.scaler_.mean_, original_mean)
        self.assertEqual(model.training_details_, before_details)
        np.testing.assert_array_equal(query, self.X[:1])

    def test_ensemble_has_exact_equal_probabilities_shared_scope_and_seed_override(self):
        first, second = recipe("hadamard512"), recipe("temporal_corr120")
        spec = {"family": VOTING, "members": [first, second], "training": {"seed": 53}}
        model = build_interaction_pipeline(spec, seed=71).fit(self.X, self.y, groups=self.groups)
        self.assertEqual([member.effective_seed_ for member in model.members_], [71, 71])
        self.assertEqual([member.estimator_.random_state for member in model.members_], [71, 71])
        self.assertTrue(all(member.training_input_sha256_ == model.training_input_sha256_ for member in model.members_))
        p = np.mean([member.predict_proba(self.X) for member in model.members_], axis=0)
        np.testing.assert_array_equal(model.predict_proba(self.X), p)
        self.assertTrue(verify_fitted_interaction_identity(model, spec, 71))
        stream = io.BytesIO(); joblib.dump(model, stream); stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).predict_proba(self.X), p)
        for members in ([first], [first, first], [first, spec],
                [first, {**second, "training": {"seed": 54}}],
                [first, {**second, "training": {"seed": 53, "data_policy": "include_ordinary_reserved_training_records_v1"}}]):
            with self.subTest(members=len(members)), self.assertRaises(ValueError):
                validate_interaction_pipeline({**spec, "members": members})

    def test_schema_input_and_training_label_contracts_fail_before_backend_fit(self):
        cases = [recipe("unknown"), recipe(window_seconds=True), recipe(base_method="regularized"),
            recipe(feature_engineering={"mode": "identity", "fit_components": 16}),
            recipe(preprocessing={"scaler": "none"}), recipe(postprocessing={"kind": "decision_softmax"}),
            recipe(training={"weighting": "subject_class_balanced"}), recipe(training={"data_policy": "unknown"}),
            recipe(model_params={"random_state": 1}), recipe(model_params={"n_estimators": True}),
            recipe(model_params={"max_features": 597}), recipe(model_params={"max_depth": 0}),
            recipe(model_params={"max_features": float("nan")})]
        for spec in cases:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_interaction_pipeline(spec)
        with patch.object(ExtraTreesClassifier, "fit") as backend_fit:
            for values, labels, groups in ((self.X[:, :-1], self.y, self.groups),
                    (self.X.astype(np.float64), self.y, self.groups), (np.full_like(self.X, np.nan), self.y, self.groups),
                    (self.X[:0], self.y[:0], self.groups[:0]), (self.X, np.zeros_like(self.y), self.groups),
                    (self.X, self.y.astype(float), self.groups), (self.X, self.y, self.groups[:-1]),
                    (self.X, self.y, np.repeat("", len(self.y)))):
                with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                    build_interaction_pipeline(recipe()).fit(values, labels, groups=groups)
        backend_fit.assert_not_called()
        with self.assertRaises(ValueError):
            fit_interaction_pipeline(build_interaction_pipeline(recipe()), recipe("hadamard512"), self.X, self.y)
        model = build_interaction_pipeline(recipe()).fit(self.X, self.y)
        self.assertIsNone(model.training_subjects_)
        self.assertTrue(verify_fitted_interaction_identity(model, recipe(), 53))

    def test_identity_checks_fixed_rule_learned_scaler_complete_trees_and_audit_consistency(self):
        spec = recipe("hadamard512")
        original = self.fit(spec)
        mutations = [lambda m: setattr(m, "effective_seed_", 999),
            lambda m: m.interaction_rule_.update(mode="identity"),
            lambda m: m.feature_names_out_.__setitem__(0, "changed"),
            lambda m: m.extraction_feature_names_.__setitem__(0, "changed"),
            lambda m: m.scaler_.mean_.__setitem__(0, 999.),
            lambda m: m.estimator_.set_params(random_state=999),
            lambda m: m.estimator_.estimators_[0].tree_.threshold.__setitem__(0, 999.),
            lambda m: m.estimator_.estimators_[0].set_params(random_state=999),
            lambda m: m.training_details_["class_counts"].__setitem__(0, 999),
            lambda m: m.training_details_.update(training_subjects=["wrong"]),
            lambda m: m.training_details_.update(transformed_state_sha256="0" * 64)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                model = deepcopy(original)
                mutation(model)
                with self.assertRaises(ValueError):
                    verify_fitted_interaction_identity(model, spec, 53)
        original.spec["feature_engineering"]["mode"] = "identity"
        with self.assertRaisesRegex(ValueError, "changed after fitting"):
            original.predict_proba(self.X)

    def test_ensemble_identity_rejects_saved_members_weights_and_training_scope_drift(self):
        spec = {"family": VOTING, "members": [recipe("hadamard512"), recipe("temporal_corr120")],
            "training": {"seed": 53}}
        original = self.fit(spec)
        mutations = [lambda m: m.training_details_.update(weights=[.1, .9]),
            lambda m: m.training_details_.update(combination="weighted probability mean"),
            lambda m: m.training_details_["members"][0].update(effective_seed=999),
            lambda m: setattr(m.members_[0], "training_input_sha256_", "0" * 64)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                model = deepcopy(original)
                mutation(model)
                with self.assertRaises(ValueError):
                    verify_fitted_interaction_identity(model, spec, 53)


if __name__ == "__main__":
    unittest.main()
