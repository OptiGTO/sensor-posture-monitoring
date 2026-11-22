"""Synthetic-only contracts for immutable alternative forest pipelines."""
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src import forest_alternatives_pipeline_v1 as subject
from src.iterative_pipeline import build_pipeline as legacy_pipeline
from src.posture_core import feature_names


def recipe(family="tail_extra_trees", q=.01):
    spec = {"family": family, "model_params": {"n_estimators": 3, "min_samples_leaf": 1}, "training": {"seed": 53}}
    if family == "tail_extra_trees":
        spec["preprocessing"] = {"scaler": "standard", "tail_fraction": q}
    return subject.validate_alternatives_pipeline(spec)


class ForestAlternativesTests(unittest.TestCase):
    def setUp(self):
        self.y = np.repeat(np.arange(7), np.arange(3, 10))
        self.X = np.random.default_rng(1601).normal(size=(len(self.y), 596)).astype(np.float32)
        self.X[:, 0] += self.y
        self.X[:, 1] = 0
        self.groups = np.asarray([f"person-{index % 3}" for index in range(len(self.y))])
        self.tail, self.ovr = recipe(), recipe("ovr_extra_trees")
        self.ensemble = subject.validate_alternatives_pipeline({"family": "alternatives_soft_voting",
            "members": [self.tail, self.ovr], "training": {"seed": 53}})

    def fit(self, spec, X=None, seed=None):
        return subject.build_alternatives_pipeline(spec, seed=seed).fit(
            self.X if X is None else X, self.y, groups=self.groups)

    def test_complete_recipes_are_copied_and_only_supported_contracts_are_accepted(self):
        original = deepcopy(self.ensemble)
        normalized = subject.validate_alternatives_pipeline(original)
        normalized["members"][0]["preprocessing"]["tail_fraction"] = .005
        self.assertEqual(original, self.ensemble)
        self.assertEqual(self.ovr["training"]["weighting"], "binary_class_balanced")
        self.assertEqual(self.ovr["postprocessing"], {"kind": "ovr_normalized", "zero_row": "uniform"})
        cases = [None, {**self.tail, "window_seconds": True}, {**self.tail, "unexpected": 1},
            {**self.tail, "preprocessing": {"scaler": "standard", "tail_fraction": .02}},
            {**self.tail, "preprocessing": {"scaler": "standard", "tail_fraction": True}},
            {**self.ovr, "postprocessing": {"kind": "argmax"}},
            {**self.ovr, "training": {"seed": 53, "weighting": "class_balanced"}},
            {**self.ensemble, "members": [self.tail, self.tail]},
            {**self.ensemble, "members": [self.tail, self.ensemble]},
            {**self.ensemble, "members": [self.tail, {**self.ovr, "training": {"seed": 54}}]},
            {**self.ensemble, "members": [self.tail, {**self.ovr, "training": {"seed": 53, "data_policy": subject.DATA_POLICY}}]}]
        for changed in cases:
            with self.subTest(value=changed), self.assertRaises(ValueError):
                subject.validate_alternatives_pipeline(changed)

    def test_tail_zero_reproduces_legacy_probability_bytes_for_float32_and_float64(self):
        spec = subject.validate_alternatives_pipeline({"family": "tail_extra_trees",
            "preprocessing": {"scaler": "standard", "tail_fraction": 0.}, "training": {"seed": 53}})
        self.assertEqual(spec["model_params"], {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 8, "max_features": .5})
        baseline = {"family": "extra_trees", "model_params": spec["model_params"], "training": {"seed": 53}}
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                X = self.X.astype(dtype)
                original = X.tobytes()
                candidate = self.fit(spec, X)
                reference = legacy_pipeline(baseline).fit(X, self.y, groups=self.groups)
                query = np.concatenate([X, X[:2] * 1e4]).astype(dtype)
                self.assertEqual(candidate.predict_proba(query).tobytes(), reference.predict_proba(query).tobytes())
                self.assertEqual(candidate.scaler_.mean_.tobytes(), reference.named_steps["scale"].mean_.tobytes())
                self.assertEqual(original, X.tobytes())
                self.assertIsNone(candidate.clipper_.lower_bounds_)
                self.assertEqual(candidate.prepared_state_sha256_, candidate.training_input_sha256_)
                self.assertTrue(subject.verify_fitted_alternatives_identity(candidate, spec, 53))

    def test_tail_bounds_and_scaler_fit_only_supplied_training_rows(self):
        class UnreadableValidation:
            def __array__(self, *args, **kwargs):
                raise AssertionError("External validation must not be consulted")
        model = subject.build_alternatives_pipeline(self.tail)
        subject.fit_alternatives_pipeline(model, self.tail, self.X, self.y, groups=self.groups,
            validation_data=UnreadableValidation())
        lower, upper = np.quantile(self.X.astype(np.float64), [.01, .99], axis=0, method="linear")
        clipped = np.clip(self.X, lower, upper).astype(np.float32)
        np.testing.assert_array_equal(model.clipper_.lower_bounds_, lower)
        np.testing.assert_array_equal(model.clipper_.upper_bounds_, upper)
        np.testing.assert_allclose(model.scaler_.mean_, np.mean(clipped, axis=0, dtype=np.float64))
        self.assertEqual(model.scaler_.n_samples_seen_, len(self.X))
        before = deepcopy(model.training_details_)
        first = model.predict_proba(self.X[:1])
        joined = model.predict_proba(np.concatenate([self.X[:1], np.full_like(self.X, 1e10)]))
        self.assertEqual(first.tobytes(), joined[:1].tobytes())
        self.assertEqual(before, model.training_details_)
        self.assertTrue(subject.verify_fitted_alternatives_identity(model, self.tail, 53))

    def test_binary_heads_receive_shared_training_scaler_float32_and_canonical_labels(self):
        original = subject.BinaryForestV1.fit
        captured = []
        def capture(model, X, y):
            captured.append((X.copy(), y.copy()))
            return original(model, X, y)
        with patch.object(subject.BinaryForestV1, "fit", capture):
            model = self.fit(self.ovr, self.X.astype(np.float64))
        self.assertEqual(len(captured), 1)
        transformed, labels = captured[0]
        self.assertEqual(transformed.dtype, np.float32)
        np.testing.assert_array_equal(transformed, model.scaler_.transform(self.X.astype(np.float64)).astype(np.float32))
        np.testing.assert_array_equal(labels, self.y.astype(np.int64))
        self.assertEqual(model.model_input_sha256_, model.estimator_.training_details_["training_input_sha256"])
        self.assertEqual(model.training_labels_sha256_, model.estimator_.training_details_["training_labels_sha256"])
        self.assertEqual(len(model.estimator_.estimators_), 7)
        self.assertTrue(subject.verify_fitted_alternatives_identity(model, self.ovr, 53))

    def test_ensemble_means_exact_aligned_probabilities_and_seed_override_is_recorded(self):
        original = deepcopy(self.ensemble)
        model = self.fit(self.ensemble, seed=54)
        expected = np.mean([member.predict_proba(self.X) for member in model.members_], axis=0)
        np.testing.assert_array_equal(model.predict_proba(self.X), expected)
        self.assertEqual(self.ensemble, original)
        self.assertEqual([member.effective_seed_ for member in model.members_], [54, 54])
        self.assertEqual([head.random_state for head in model.members_[1].estimator_.estimators_], list(range(54, 61)))
        self.assertEqual({member.training_input_sha256_ for member in model.members_}, {model.training_input_sha256_})
        self.assertTrue(subject.verify_fitted_alternatives_identity(model, self.ensemble, 54))

    def test_reload_refit_and_batch_partition_predictions_are_reproducible(self):
        for spec in (self.tail, self.ovr, self.ensemble):
            with self.subTest(family=spec["family"]):
                model = self.fit(spec)
                before = deepcopy(model.training_details_)
                expected = model.predict_proba(self.X)
                stream = io.BytesIO()
                joblib.dump(model, stream)
                stream.seek(0)
                restored = joblib.load(stream)
                self.assertEqual(restored.predict_proba(self.X).tobytes(), expected.tobytes())
                self.assertEqual(self.fit(spec).predict_proba(self.X).tobytes(), expected.tobytes())
                pieces = np.concatenate([model.predict_proba(self.X[:1]), model.predict_proba(self.X[1:])])
                self.assertEqual(pieces.tobytes(), expected.tobytes())
                np.testing.assert_array_equal(model.predict_proba(self.X[::-1]), expected[::-1])
                self.assertEqual(model.training_details_, before)
                self.assertTrue(subject.verify_fitted_alternatives_identity(restored, spec, 53))

    def test_invalid_input_labels_groups_and_changed_constructed_recipe_fail_closed(self):
        for X in (self.X[:, :-1], self.X[:0], self.X[0], np.full_like(self.X, np.nan),
                np.full_like(self.X, np.inf), self.X.astype(str)):
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                self.fit(self.tail, X)
        model = self.fit(self.tail)
        named = pd.DataFrame(self.X, columns=feature_names())
        np.testing.assert_array_equal(model.predict_proba(named), model.predict_proba(self.X))
        with self.assertRaises(ValueError):
            model.predict_proba(named.iloc[:, ::-1])
        for y in (self.y.astype(float), self.y[:-1], np.zeros_like(self.y)):
            with self.assertRaises(ValueError):
                subject.build_alternatives_pipeline(self.tail).fit(self.X, y)
        for groups in (self.groups[:-1], [""] * len(self.X), np.arange(len(self.X))):
            with self.assertRaises(ValueError):
                subject.build_alternatives_pipeline(self.tail).fit(self.X, self.y, groups=groups)
        with self.assertRaises(ValueError):
            subject.fit_alternatives_pipeline(subject.build_alternatives_pipeline(self.tail), self.ovr, self.X, self.y)
        model.spec["model_params"]["n_estimators"] += 1
        with self.assertRaises(ValueError):
            model.predict_proba(self.X)

    def test_tail_state_and_scope_tampering_are_rejected(self):
        original = self.fit(self.tail)
        mutations = [lambda m: m.clipper_.lower_bounds_.__setitem__(0, -1000.),
            lambda m: m.scaler_.mean_.__setitem__(0, 1000.),
            lambda m: m.estimator_.estimators_[0].tree_.value.__setitem__((0, 0, 0), .12345),
            lambda m: m.training_details_.update(external_validation_used=True),
            lambda m: m.training_details_.update(training_input_sha256="f" * 64),
            lambda m: m.model_rule_.update(class_weight=None),
            lambda m: m.training_details_["tail_clipping"].update(training_windows=1),
            lambda m: setattr(m, "feature_names_in_", np.asarray(feature_names()[::-1]))]
        for mutate in mutations:
            changed = deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                subject.verify_fitted_alternatives_identity(changed, self.tail, 53)

    def test_binary_and_ensemble_state_or_cross_scope_tampering_are_rejected(self):
        original = self.fit(self.ovr)
        for mutate in (lambda m: m.estimator_.estimators_[0].set_params(random_state=99),
                lambda m: m.training_details_["binary_forest"].update(training_rows=1),
                lambda m: setattr(m, "model_input_sha256_", "f" * 64)):
            changed = deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                subject.verify_fitted_alternatives_identity(changed, self.ovr, 53)
        model = self.fit(self.ensemble)
        # An individually valid helper trained on different rows cannot be
        # substituted for a member even if its own complete audit is copied.
        model.members_[1] = self.fit(self.ovr, self.X + np.float32(1))
        model.training_details_["members"][1] = deepcopy(model.members_[1].training_details_)
        with self.assertRaisesRegex(ValueError, "different rows"):
            subject.verify_fitted_alternatives_identity(model, self.ensemble, 53)


if __name__ == "__main__":
    unittest.main()
