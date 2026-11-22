from __future__ import annotations

import io
import unittest
from copy import deepcopy
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone

from src.iterative_pipeline import (build_pipeline, fit_pipeline,
    selected_feature_names, validate_pipeline, IterativePipeline)
from src.model_comparison import build_model
from src.posture_core import feature_names


def recipe(family="extra_trees", **changes):
    result = {"family": family, "base_method": "standard", "window_seconds": 3,
        "feature_engineering": {"mode": "all"}, "preprocessing": {"scaler": "standard"},
        "model_params": {"n_estimators": 8, "min_samples_leaf": 2},
        "training": {"seed": 43, "weighting": "class_balanced"},
        "postprocessing": {"kind": "argmax"}}
    result.update(changes)
    return result


class IterativePipelineTests(unittest.TestCase):
    def test_hierarchy_pipeline_preserves_input_and_complete_learned_structure(self):
        spec = recipe("sagittal_hierarchy", model_params={"base_family": "extra_trees",
            "base_params": {"n_estimators": 8, "min_samples_leaf": 2}})
        model = build_pipeline(spec).fit(self.X, self.y)
        self.assertEqual(model.n_features_in_, 596)
        self.assertIn("hierarchy_training", model.training_details_)
        self.assertNotIn("neural_normalization", model.training_details_)
        expected = model.predict_proba(self.X)
        self.assertEqual(expected.shape, (28, 7))
        stream = io.BytesIO()
        joblib.dump(model, stream); stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).predict_proba(self.X), expected)
        with self.assertRaises(ValueError):
            validate_pipeline({**spec, "feature_engineering": {"mode": "contrast"}})
        with self.assertRaises(ValueError):
            validate_pipeline({**spec, "model_params": {"base_family": "extra_trees", "base_params": {"class_weight": "balanced"}}})

    def test_quantile_mapping_uses_only_training_distribution(self):
        spec = recipe(preprocessing={"scaler": "quantile_normal"})
        first = build_pipeline(spec).fit(self.X, self.y)
        second = build_pipeline(spec).fit(self.X, self.y, validation_data=(self.X + 1e9, self.y))
        np.testing.assert_array_equal(first.named_steps["scale"].quantiles_, second.named_steps["scale"].quantiles_)
        np.testing.assert_array_equal(first.named_steps["scale"].quantiles_[0], self.X.min(0))
        np.testing.assert_array_equal(first.named_steps["scale"].quantiles_[-1], self.X.max(0))
        np.testing.assert_array_equal(first.predict_proba(self.X), second.predict_proba(self.X))
        with self.assertRaises(ValueError):
            validate_pipeline(recipe(preprocessing={"scaler": "quantile_normal"},
                training={"seed": 43, "weighting": "subject_class_balanced"}))

    def test_bias_uses_inner_held_out_people_and_ignores_external_validation(self):
        groups = np.repeat(np.array(["a", "b", "c", "d"]), 7)
        spec = recipe(postprocessing={"kind": "macro_f1_bias", "inner_folds": 4,
            "max_sweeps": 3, "regularization": .01})
        calls = []
        original = IterativePipeline.fit
        def observe(model, X, y, **kwargs):
            calls.append((model.spec["postprocessing"]["kind"], X.copy(), np.array(kwargs.get("groups"))))
            return original(model, X, y, **kwargs)
        with patch.object(IterativePipeline, "fit", autospec=True, side_effect=observe):
            model = build_pipeline(spec).fit(self.X, self.y, groups=groups,
                validation_data=(self.X + 1e9, (self.y + 1) % 7))
        self.assertEqual(len(calls), 6)
        folds = model.training_details_["inner_group_folds"]
        for (_, X, seen_groups), fold in zip(calls[1:5], folds):
            self.assertFalse(set(fold["training_subjects"]) & set(fold["held_out_subjects"]))
            mask = np.isin(groups, fold["training_subjects"])
            np.testing.assert_array_equal(X, self.X[mask])
            np.testing.assert_array_equal(seen_groups, groups[mask])
        self.assertEqual(sorted(v for f in folds for v in f["held_out_subjects"]), ["a", "b", "c", "d"])
        np.testing.assert_array_equal(calls[-1][1], self.X)
        expected = model.predict_proba(self.X)
        self.assertEqual(expected.shape, (28, 7))
        np.testing.assert_allclose(expected.sum(1), 1)
        stream = io.BytesIO()
        joblib.dump(model, stream); stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).predict_proba(self.X), expected)
        with self.assertRaises(ValueError):
            build_pipeline(spec).fit(self.X, self.y, groups=np.repeat("a", len(self.y)))

    def test_bias_recipe_requires_explicit_bounded_search_and_single_estimator(self):
        post = {"kind": "macro_f1_bias", "inner_folds": 4, "max_sweeps": 3, "regularization": .01}
        validate_pipeline(recipe(postprocessing=post))
        for change in ({"inner_folds": 1}, {"max_sweeps": 4}, {"regularization": -1}, {"temperature": 1}):
            with self.assertRaises(ValueError):
                validate_pipeline(recipe(postprocessing={**post, **change}))
        with self.assertRaises(ValueError):
            validate_pipeline(recipe(postprocessing={"kind": "macro_f1_bias"}))

    def test_pressure_contrast_recipe_preserves_input_and_serialized_inference(self):
        import json
        for mode, dimension in (("contrast", 75), ("all_plus_contrast", 671)):
            spec = recipe(feature_engineering={"mode": mode})
            model = fit_pipeline(build_pipeline(spec), spec, self.X, self.y)
            self.assertEqual(model.n_features_in_, 596)
            self.assertEqual(model.named_steps["model"].n_features_in_, dimension)
            self.assertEqual(len(json.loads(json.dumps(model.training_details_))["selected_feature_names"]), dimension)
            stream = io.BytesIO()
            joblib.dump(model, stream)
            stream.seek(0)
            restored = joblib.load(stream)
            np.testing.assert_array_equal(restored.predict_proba(self.X), model.predict_proba(self.X))

    def setUp(self):
        rng = np.random.default_rng(891)
        self.X = rng.normal(size=(28, len(feature_names())))
        self.y = np.tile(np.arange(7), 4)
        self.X[:, :7] += np.eye(7)[self.y] * 5

    def test_complete_baseline_recipe_reproduces_existing_probabilities(self):
        for family, params in (("rbf_svm", {"C": 1., "gamma": .0005}),
                ("random_forest", {"n_estimators": 7}),
                ("extra_trees", {"n_estimators": 7})):
            spec = recipe(family, model_params=params)
            expected = build_model(family, params, 43).fit(self.X, self.y)
            model = fit_pipeline(build_pipeline(spec), spec, self.X, self.y)
            np.testing.assert_array_equal(model.predict_proba(self.X), expected.predict_proba(self.X))
            np.testing.assert_array_equal(model.predict(self.X), expected.predict_proba(self.X).argmax(1))

    def test_feature_subsets_and_scaler_ignore_validation_data(self):
        original = self.X.copy()
        for mode, count in (("all", 596), ("static", 312), ("means", 92), ("relative_static", 156)):
            spec = recipe(feature_engineering={"mode": mode})
            model = fit_pipeline(build_pipeline(spec), spec, self.X, self.y,
                validation_data=(self.X + 1e9, (self.y + 1) % 7))
            retained = selected_feature_names(mode)
            self.assertEqual(len(retained), count)
            if mode != "all":
                self.assertFalse(any("frequency" in name or "slope" in name or "kurtosis" in name for name in retained))
            indices = [feature_names().index(name) for name in retained]
            np.testing.assert_allclose(model.named_steps["scale"].mean_, self.X[:, indices].mean(0))
            self.assertEqual(model.n_features_in_, 596)
            stream = io.BytesIO()
            joblib.dump(model, stream)
            stream.seek(0)
            restored = joblib.load(stream)
            np.testing.assert_array_equal(restored.predict_proba(self.X), model.predict_proba(self.X))
            with self.assertRaises(ValueError):
                restored.predict_proba(self.X[:, :-1])
        np.testing.assert_array_equal(self.X, original)

    def test_subject_balancing_weights_both_scaler_and_classifier_without_double_weighting(self):
        y = np.array([0, 0, 0, 1, 0, 1, 1])
        groups = np.array(["a", "a", "a", "a", "b", "b", "b"])
        X = self.X[:len(y)]
        spec = recipe(training={"seed": 43, "weighting": "subject_class_balanced"})
        model = fit_pipeline(build_pipeline(spec), spec, X, y, groups=groups)
        expected_weights = np.array([7/12, 7/12, 7/12, 7/4, 7/4, 7/8, 7/8])
        self.assertIsNone(model.named_steps["model"].class_weight)
        np.testing.assert_allclose(model.named_steps["scale"].mean_, np.average(X, axis=0, weights=expected_weights))
        reference = build_model("extra_trees", spec["model_params"], 43)
        reference.named_steps["model"].set_params(class_weight=None)
        reference.fit(X, y, scale__sample_weight=expected_weights, model__sample_weight=expected_weights)
        np.testing.assert_array_equal(model.predict_proba(X)[:, :2], reference.predict_proba(X))
        with self.assertRaisesRegex(ValueError, "subject IDs"):
            fit_pipeline(build_pipeline(spec), spec, X, y)

    def test_whitening_and_decision_scores_are_serialized_and_train_fitted(self):
        spec = recipe("rbf_svm", model_params={"C": 1., "gamma": "scale"},
            feature_engineering={"mode": "means"},
            preprocessing={"scaler": "shrinkage_whiten"},
            postprocessing={"kind": "decision_softmax", "temperature": 1.})
        model = fit_pipeline(build_pipeline(spec), spec, self.X, self.y,
            validation_data=(self.X + 1e9, self.y))
        self.assertFalse(model.named_steps["model"].probability)
        np.testing.assert_allclose(model.named_steps["whiten"].location_, 0, atol=1e-14)
        probability = model.predict_proba(self.X)
        np.testing.assert_allclose(probability.sum(1), 1)
        np.testing.assert_array_equal(probability.argmax(1), model.estimator_.decision_function(self.X).argmax(1))
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).predict_proba(self.X), probability)

    def test_equal_voting_aligns_missing_classes_and_preserves_member_seeds(self):
        a = recipe(training={"seed": 1, "weighting": "class_balanced"})
        b = recipe(feature_engineering={"mode": "means"}, training={"seed": 2, "weighting": "class_balanced"})
        spec = {"family": "equal_soft_voting", "members": [a, b]}
        y = np.where(self.y % 2, 6, 2)
        model = fit_pipeline(build_pipeline(spec), spec, self.X, y)
        probability = model.predict_proba(self.X)
        expected = np.mean([build_pipeline(member).fit(self.X, y).predict_proba(self.X) for member in (a, b)], axis=0)
        np.testing.assert_array_equal(probability, expected)
        self.assertTrue(np.all(probability[:, [0, 1, 3, 4, 5]] == 0))
        self.assertEqual([member.effective_seed_ for member in model.members_], [1, 2])
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).predict_proba(self.X), probability)

    def test_spec_clone_and_seed_override_do_not_mutate_recipe(self):
        spec = recipe()
        original = deepcopy(spec)
        model = clone(build_pipeline(spec, seed=2))
        model.fit(self.X, self.y)
        self.assertEqual(spec, original)
        self.assertEqual(model.effective_seed_, 2)
        self.assertEqual(model.spec_["training"]["seed"], 43)
        changed = deepcopy(spec)
        changed["model_params"]["min_samples_leaf"] = 9
        with self.assertRaisesRegex(ValueError, "differs"):
            fit_pipeline(model, changed, self.X, self.y)

    def test_subject_invariant_recipe_routes_training_groups_and_serializes_selection(self):
        from src.domain_feature_selection import SubjectInvariantSelector
        groups = np.repeat(np.arange(4), 7)
        spec = recipe(feature_engineering={"mode": "subject_invariant", "k": 16, "strength": 0.5})
        before = self.X.copy()
        model = fit_pipeline(build_pipeline(spec), spec, self.X, self.y, groups=groups,
            validation_data=(self.X + 1e9, (self.y + 1) % 7))
        expected = SubjectInvariantSelector(k=16, strength=0.5).fit(self.X, self.y, groups=groups)
        np.testing.assert_array_equal(model.named_steps["features"].indices_, expected.indices_)
        np.testing.assert_allclose(model.named_steps["scale"].mean_, self.X[:, expected.indices_].mean(0))
        self.assertEqual(model.training_details_["feature_learning"]["training_subjects"], 4)
        self.assertEqual(model.training_details_["selected_feature_names"], expected.get_feature_names_out().tolist())
        self.assertEqual(model.n_features_in_, 596)
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).predict_proba(self.X), model.predict_proba(self.X))
        np.testing.assert_array_equal(self.X, before)
        with self.assertRaisesRegex(ValueError, "subject IDs"):
            fit_pipeline(build_pipeline(spec), spec, self.X, self.y)
        model.named_steps["features"].set_params(strength=2.)
        with self.assertRaisesRegex(ValueError, "changed after fitting"):
            model.predict_proba(self.X)

    def test_shrinkage_lda_recipe_keeps_input_contract_and_train_only_axes(self):
        from src.domain_feature_selection import ShrinkageDiscriminantProjection
        spec = recipe(feature_engineering={"mode": "shrinkage_lda", "n_components": 3})
        model = fit_pipeline(build_pipeline(spec), spec, self.X, self.y,
            validation_data=(self.X * 1e9, (self.y + 1) % 7))
        expected = ShrinkageDiscriminantProjection(n_components=3).fit(self.X, self.y)
        np.testing.assert_allclose(model.named_steps["features"].transform(self.X), expected.transform(self.X))
        np.testing.assert_allclose(model.named_steps["scale"].mean_, expected.transform(self.X).mean(0))
        self.assertEqual(model.named_steps["scale"].n_features_in_, 3)
        self.assertEqual(model.n_features_in_, 596)
        self.assertEqual(model.training_details_["selected_feature_names"], expected.get_feature_names_out().tolist())
        self.assertEqual(model.training_details_["feature_learning"]["n_components"], 3)
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        np.testing.assert_array_equal(restored.predict_proba(self.X), model.predict_proba(self.X))
        with self.assertRaises(ValueError):
            restored.predict_proba(self.X[:, :-1])
        invalid_data = np.zeros_like(self.X)
        with self.assertRaises(ValueError):
            fit_pipeline(build_pipeline(spec), spec, invalid_data, self.y)

    def test_learned_feature_schema_is_explicit_and_mode_specific(self):
        for feature in ({"mode": "subject_invariant"}, {"mode": "subject_invariant", "k": 96},
                {"mode": "subject_invariant", "k": True, "strength": 1},
                {"mode": "subject_invariant", "k": 597, "strength": 1},
                {"mode": "subject_invariant", "k": 96, "strength": -1},
                {"mode": "subject_invariant", "k": 96, "strength": float('inf')},
                {"mode": "shrinkage_lda"}, {"mode": "shrinkage_lda", "n_components": 7},
                {"mode": "shrinkage_lda", "n_components": True},
                {"mode": "shrinkage_lda", "n_components": 2, "k": 96},
                {"mode": "all", "strength": 1}):
            with self.assertRaises(ValueError, msg=str(feature)):
                validate_pipeline(recipe(feature_engineering=feature))
        for feature in ({"mode": "subject_invariant", "k": 96, "strength": 1.},
                {"mode": "shrinkage_lda", "n_components": None}):
            self.assertEqual(validate_pipeline(recipe(feature_engineering=feature))["feature_engineering"], feature)
            with self.assertRaisesRegex(ValueError, "Learned feature names"):
                selected_feature_names(feature["mode"])

    def test_invalid_specs_fail_before_fit(self):
        invalid = [recipe(window_seconds=5), recipe(window_seconds=True),
            recipe(feature_engineering={"mode": "unknown"}),
            recipe(training={"seed": 2**32}), recipe(postprocessing={"kind": "decision_softmax"}),
            recipe(model_params={"random_state": 9}), recipe(unused=True),
            recipe(preprocessing={"scaler": "shrinkage_whiten"}, training={"weighting": "subject_class_balanced"}),
            {"family": "equal_soft_voting", "members": [recipe(), recipe()]}]
        for spec in invalid:
            with self.assertRaises(ValueError, msg=str(spec)):
                validate_pipeline(spec)

    def test_mlp_fixed_budget_ignores_external_validation_and_reproduces_after_reload(self):
        from importlib.util import find_spec
        if find_spec("torch") is None:
            self.skipTest("Torch is optional")
        from src.extended_models import SmallNeuralClassifier
        spec = recipe("mlp", model_params={"hidden_size": 4, "epochs": 2, "batch_size": 14,
            "dropout": .3, "weight_decay": .03})
        before = self.X.copy()
        original_fit = SmallNeuralClassifier.fit
        calls = []
        def checked_fit(estimator, X, y, *, validation_data=None):
            self.assertIsNone(validation_data)
            calls.append(len(X))
            return original_fit(estimator, X, y, validation_data=validation_data)
        with patch.object(SmallNeuralClassifier, "fit", checked_fit):
            first = fit_pipeline(build_pipeline(spec), spec, self.X, self.y,
                validation_data=(self.X + 1e9, (self.y + 1) % 7))
            repeated = fit_pipeline(build_pipeline(spec), spec, self.X, self.y)
        self.assertEqual(calls, [len(self.X), len(self.X)])
        first_probability = first.predict_proba(self.X)
        np.testing.assert_array_equal(first_probability, repeated.predict_proba(self.X))
        np.testing.assert_allclose(first_probability.sum(1), 1, atol=1e-6)
        self.assertEqual(first.n_features_in_, 596)
        neural = first.named_steps["model"]
        np.testing.assert_allclose(first.named_steps["scale"].mean_, self.X.mean(0))
        scaled = first.named_steps["scale"].transform(self.X).astype(np.float32)
        np.testing.assert_array_equal(neural.mean_, scaled.mean(0, dtype=np.float64).astype(np.float32))
        self.assertEqual(neural.random_state, 43)
        self.assertEqual(neural.architecture, "mlp")
        self.assertEqual(first.training_details_["neural_training"]["epochs_run"], 2)
        self.assertEqual(first.training_details_["neural_training"]["epoch_selection"], "training loss")
        self.assertTrue(all("validation_macro_f1" not in row for row in neural.training_details_["history"]))
        self.assertFalse(first.training_details_["neural_normalization"]["raw_sensor_augmentation"])
        stream = io.BytesIO()
        joblib.dump(first, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        self.assertFalse(hasattr(restored.named_steps["model"], "_network"))
        np.testing.assert_array_equal(restored.predict_proba(self.X), first_probability)
        np.testing.assert_array_equal(self.X, before)

    def test_mlp_parameters_are_complete_and_unsupported_training_fails_early(self):
        valid = validate_pipeline(recipe("mlp", model_params={"hidden_size": 16, "epochs": 40,
            "dropout": .3, "weight_decay": .03}))
        self.assertEqual(valid["model_params"]["batch_size"], 64)
        self.assertEqual(valid["model_params"]["learning_rate"], .001)
        self.assertEqual(valid["model_params"]["gain_std"], 0.)
        self.assertEqual(valid["model_params"]["noise_fraction"], 0.)
        for params in ({"architecture": "mlp"}, {"gain_std": .01}, {"noise_fraction": .01},
                {"hidden_size": True}, {"epochs": 0}, {"dropout": 1.}, {"dropout": -.1},
                {"learning_rate": 0}, {"weight_decay": -1.}, {"gain_std": False},
                {"batch_size": 1.5}, {"learning_rate": float('nan')}, {"unsupported": 1}):
            with self.assertRaises(ValueError, msg=str(params)):
                validate_pipeline(recipe("mlp", model_params=params))
        with self.assertRaisesRegex(ValueError, "class_balanced"):
            validate_pipeline(recipe("mlp", model_params={}, training={"seed": 43, "weighting": "subject_class_balanced"}))
        with self.assertRaisesRegex(ValueError, "outer StandardScaler"):
            validate_pipeline(recipe("mlp", model_params={}, preprocessing={"scaler": "quantile_normal"}))


if __name__ == "__main__":
    unittest.main()
