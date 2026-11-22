from __future__ import annotations

from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from src.forest_support_pipeline_v1 import (TREE, VOTING, build_support_pipeline,
    fit_support_pipeline, validate_support_pipeline, verify_fitted_support_identity)
from src.iterative_pipeline import build_pipeline
from src.leaf_support_v1 import forest_state_sha256, leaf_support_probabilities, leaf_support_rule
from src.posture_core import feature_names


def recipe(alpha=0., **changes):
    value = {"family": TREE, "model_params": {"n_estimators": 8, "min_samples_leaf": 1},
        "training": {"seed": 53}, "postprocessing": {"kind": "uniform_leaf_shrinkage", "alpha": alpha}}
    value.update(changes)
    return value


def fixture():
    y = np.repeat(np.arange(7), np.arange(8, 15))
    X = np.random.default_rng(1515).normal(size=(len(y), 596)).astype(np.float32)
    X[:, :7] += np.eye(7, dtype=np.float32)[y] * 1.2
    groups = np.where(np.arange(len(y)) % 2, "person-a", "person-b")
    return X, y, groups


class ForestSupportPipelineTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.groups = fixture()

    def fit(self, spec=None, **kwargs):
        spec = recipe() if spec is None else spec
        return fit_support_pipeline(build_support_pipeline(spec), spec,
            self.X, self.y, groups=self.groups, **kwargs)

    def test_alpha_zero_exactly_matches_legacy200_forest_for_float32_and_float64(self):
        spec = recipe(model_params={})
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                values = self.X.astype(dtype)
                model = build_support_pipeline(spec).fit(values, self.y, groups=self.groups)
                legacy_spec = {"family": "extra_trees", "feature_engineering": {"mode": "all"},
                    "model_params": model.spec_["model_params"], "training": {"seed": 53, "weighting": "class_balanced"}}
                legacy = build_pipeline(legacy_spec).fit(values, self.y, groups=self.groups)
                self.assertEqual(model.estimator_.n_estimators, 200)
                expected_scaler = StandardScaler().fit(values)
                np.testing.assert_array_equal(model.scaler_.mean_, expected_scaler.mean_)
                np.testing.assert_array_equal(model.scaler_.transform(values), expected_scaler.transform(values))
                self.assertEqual(model.scaler_.transform(values).dtype, dtype)
                np.testing.assert_array_equal(model.predict_proba(values), legacy.predict_proba(values))
                np.testing.assert_array_equal(model.predict_proba(values),
                    model.estimator_.predict_proba(model.scaler_.transform(values)))
                self.assertTrue(verify_fitted_support_identity(model, spec, 53))

    def test_train_only_scaler_and_backend_inputs_ignore_external_validation(self):
        original_X, original_y = self.X.copy(), self.y.copy()
        spec, original_spec = recipe(8), recipe(8)
        expected = StandardScaler().fit_transform(self.X)
        calls, original_fit = [], ExtraTreesClassifier.fit
        def observe(estimator, values, labels, **kwargs):
            calls.append((values.copy(), labels.copy(), kwargs))
            return original_fit(estimator, values, labels, **kwargs)
        with patch.object(ExtraTreesClassifier, "fit", autospec=True, side_effect=observe):
            model = self.fit(spec, validation_data=(np.full((7, 596), np.nan), np.arange(7)))
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0], expected)
        np.testing.assert_array_equal(calls[0][1], self.y)
        self.assertEqual(calls[0][2], {})
        self.assertEqual(calls[0][0].dtype, np.float32)
        self.assertEqual(model.scaler_.n_samples_seen_, len(self.y))
        self.assertFalse(model.training_details_["external_validation_used"])
        self.assertFalse(model.training_details_["evaluation_population_normalization"])
        self.assertEqual(spec, original_spec)
        np.testing.assert_array_equal(self.X, original_X)
        np.testing.assert_array_equal(self.y, original_y)
        json.dumps(model.training_details_, allow_nan=False)

    def test_alpha_changes_only_scoring_and_matches_unweighted_support_formula(self):
        models = [self.fit(recipe(alpha)) for alpha in (0., 8., 32.)]
        self.assertEqual(len({model.estimator_state_sha256_ for model in models}), 1)
        self.assertEqual(len({model.scaler_state_sha256_ for model in models}), 1)
        self.assertEqual(len({model.transformed_state_sha256_ for model in models}), 1)
        native = models[0].predict_proba(self.X)
        for alpha, model in zip((8., 32.), models[1:]):
            scaled = model.scaler_.transform(self.X)
            expected = np.zeros((len(self.y), 7), dtype=np.float64)
            for tree in model.estimator_.estimators_:
                nodes = tree.apply(scaled)
                counts = tree.tree_.n_node_samples[nodes, None]
                probability = tree.predict_proba(scaled)
                expected += (counts * probability + alpha / 7) / (counts + alpha)
            expected /= len(model.estimator_.estimators_)
            np.testing.assert_allclose(model.predict_proba(self.X), expected, atol=1e-15, rtol=1e-14)
            self.assertFalse(np.array_equal(model.predict_proba(self.X), native))
            self.assertEqual(model.support_rule_, leaf_support_rule(alpha))
            self.assertEqual(forest_state_sha256(model.estimator_), models[0].estimator_state_sha256_)
            self.assertTrue(verify_fitted_support_identity(model, recipe(alpha), 53))

    def test_more_trees_retains_existing_seeded_tree_prefix_without_retraining_changes(self):
        small = self.fit(recipe())
        large_spec = recipe(model_params={"n_estimators": 16, "min_samples_leaf": 1})
        large = self.fit(large_spec)
        np.testing.assert_array_equal(small.scaler_.mean_, large.scaler_.mean_)
        for first, second in zip(small.estimator_.estimators_, large.estimator_.estimators_):
            self.assertEqual(first.random_state, second.random_state)
            np.testing.assert_array_equal(first.tree_.threshold, second.tree_.threshold)
            np.testing.assert_array_equal(first.tree_.feature, second.tree_.feature)
            np.testing.assert_array_equal(first.tree_.value, second.tree_.value)
        self.assertEqual(len(large.estimator_.estimators_), 16)
        self.assertTrue(verify_fitted_support_identity(large, large_spec, 53))

    def test_refit_reload_batch_and_independent_query_predictions_are_exact(self):
        spec = recipe(32.)
        first = self.fit(spec)
        second = self.fit(spec, validation_data=(self.X * 100, (self.y + 1) % 7))
        p = first.predict_proba(self.X)
        np.testing.assert_array_equal(p, second.predict_proba(self.X))
        self.assertEqual(first.estimator_state_sha256_, second.estimator_state_sha256_)
        np.testing.assert_array_equal(p, np.concatenate([first.predict_proba(self.X[:9]), first.predict_proba(self.X[9:])]))
        np.testing.assert_array_equal(p[:1], first.predict_proba(np.concatenate([self.X[:1], self.X[1:7] * 100]))[:1])
        stream = io.BytesIO(); joblib.dump(first, stream); stream.seek(0)
        restored = joblib.load(stream)
        np.testing.assert_array_equal(restored.predict_proba(self.X), p)
        np.testing.assert_array_equal(restored.predict(self.X), p.argmax(1))
        self.assertTrue(verify_fitted_support_identity(restored, spec, 53))
        self.assertEqual(clone(first).spec, validate_support_pipeline(spec))

    def test_float64_runtime_cast_matches_native_tree_cast_before_support_scoring(self):
        values = self.X.astype(np.float64)
        values += np.arange(len(values))[:, None] * 1e-9
        spec = recipe(8.)
        model = build_support_pipeline(spec).fit(values, self.y)
        scaled = model.scaler_.transform(values)
        self.assertEqual(scaled.dtype, np.float64)
        self.assertEqual(model.training_details_["model_input_dtype"], "float64")
        expected = leaf_support_probabilities(model.estimator_, scaled.astype(np.float32), 8.)
        np.testing.assert_array_equal(model.predict_proba(values), expected)
        self.assertIsNone(model.training_subjects_)
        self.assertTrue(verify_fitted_support_identity(model, spec, 53))

    def test_ensemble_seed_probability_scope_and_recipe_contracts(self):
        first, second = recipe(8.), recipe(32.)
        spec = {"family": VOTING, "members": [first, second], "training": {"seed": 53}}
        model = build_support_pipeline(spec, seed=71).fit(self.X, self.y, groups=self.groups)
        self.assertEqual([member.effective_seed_ for member in model.members_], [71, 71])
        self.assertEqual([member.estimator_.random_state for member in model.members_], [71, 71])
        self.assertTrue(all(member.training_input_sha256_ == model.training_input_sha256_ for member in model.members_))
        expected = np.mean([member.predict_proba(self.X) for member in model.members_], axis=0)
        np.testing.assert_array_equal(model.predict_proba(self.X), expected)
        self.assertTrue(verify_fitted_support_identity(model, spec, 71))
        stream = io.BytesIO(); joblib.dump(model, stream); stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).predict_proba(self.X), expected)
        for members in ([first], [first, first], [first, spec],
                [first, {**second, "training": {"seed": 54}}],
                [first, {**second, "training": {"seed": 53, "data_policy": "include_ordinary_reserved_training_records_v1"}}]):
            with self.subTest(members=len(members)), self.assertRaises(ValueError):
                validate_support_pipeline({**spec, "members": members})

    def test_schema_input_and_named_feature_contracts_fail_before_fitting(self):
        invalid = [recipe(True), recipe(1.), recipe(float("nan")), recipe(-8),
            recipe(postprocessing={"kind": "argmax"}), recipe(window_seconds=True),
            recipe(feature_engineering={"mode": "static"}), recipe(preprocessing={"scaler": "none"}),
            recipe(training={"weighting": "subject_class_balanced"}), recipe(training={"data_policy": "unknown"}),
            recipe(model_params={"class_weight": "balanced"}), recipe(model_params={"n_estimators": 0}),
            recipe(model_params={"max_features": 597}), recipe(model_params={"max_depth": True})]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_support_pipeline(spec)
        normalized = validate_support_pipeline(recipe(8))
        self.assertIs(type(normalized["postprocessing"]["alpha"]), float)
        with patch.object(ExtraTreesClassifier, "fit") as backend_fit:
            for values, labels, groups in ((self.X[:, :-1], self.y, self.groups),
                    (np.full_like(self.X, np.nan), self.y, self.groups), (self.X[:0], self.y[:0], self.groups[:0]),
                    (self.X, np.zeros_like(self.y), self.groups), (self.X, self.y.astype(float), self.groups),
                    (self.X, self.y, self.groups[:-1]), (self.X, self.y, np.repeat("", len(self.y)))):
                with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                    build_support_pipeline(recipe()).fit(values, labels, groups=groups)
        backend_fit.assert_not_called()
        frame = pd.DataFrame(self.X, columns=feature_names())
        with self.assertRaises(ValueError):
            build_support_pipeline(recipe()).fit(frame[frame.columns[::-1]], self.y)
        with self.assertRaises(ValueError):
            fit_support_pipeline(build_support_pipeline(recipe()), recipe(8), self.X, self.y)
        model = self.fit()
        with self.assertRaises(ValueError):
            model.predict_proba(frame[frame.columns[::-1]])

    def test_identity_rejects_rule_scaler_complete_tree_and_audit_tampering(self):
        spec = recipe(8.)
        original = self.fit(spec)
        mutations = [lambda m: setattr(m, "effective_seed_", 999),
            lambda m: m.support_rule_.update(alpha=32.),
            lambda m: m.feature_names_in_.__setitem__(0, "changed"),
            lambda m: m.scaler_.mean_.__setitem__(0, 999.),
            lambda m: m.estimator_.set_params(random_state=999),
            lambda m: m.estimator_.estimators_[0].tree_.threshold.__setitem__(0, 999.),
            lambda m: m.estimator_.estimators_[0].tree_.n_node_samples.__setitem__(-1, 999),
            lambda m: m.estimator_.estimators_[0].set_params(random_state=999),
            lambda m: m.training_details_["class_counts"].__setitem__(0, 999),
            lambda m: m.training_details_.update(training_subjects=["wrong"]),
            lambda m: m.training_details_.update(transformed_state_sha256="0" * 64)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = deepcopy(original)
                mutation(changed)
                with self.assertRaises(ValueError):
                    verify_fitted_support_identity(changed, spec, 53)
        original.spec["postprocessing"]["alpha"] = 32.
        with self.assertRaisesRegex(ValueError, "changed after fitting"):
            original.predict_proba(self.X)
        ensemble = {"family": VOTING, "members": [recipe(8.), recipe(32.)], "training": {"seed": 53}}
        original = self.fit(ensemble)
        for mutation in (lambda m: m.training_details_.update(weights=[.1, .9]),
                lambda m: m.training_details_["members"][0].update(effective_seed=999),
                lambda m: setattr(m.members_[0], "training_input_sha256_", "0" * 64)):
            changed = deepcopy(original)
            mutation(changed)
            with self.assertRaises(ValueError):
                verify_fitted_support_identity(changed, ensemble, 53)


if __name__ == "__main__":
    unittest.main()
