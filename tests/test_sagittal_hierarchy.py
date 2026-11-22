import io
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import NotFittedError

from src.sagittal_hierarchy import SagittalHierarchyClassifier


class FixedProbabilities:
    def __init__(self, classes, probability):
        self.classes_ = np.asarray(classes)
        self.probability = np.asarray(probability, dtype=float)

    def predict_proba(self, X):
        return np.tile(self.probability, (len(X), 1))


class SagittalHierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(190)
        cls.y = np.repeat(np.arange(7), [6, 7, 8, 6, 7, 8, 9])
        cls.X = rng.normal(size=(len(cls.y), 596))
        cls.X[np.arange(len(cls.y)), cls.y] += 4
        cls.tree_params = {"n_estimators": 13, "max_depth": 4, "min_samples_leaf": 1,
                           "max_features": 0.5}

    def fixed(self):
        model = SagittalHierarchyClassifier()
        # Coarse A=0 is most likely, but its probability is spread across
        # three leaves. Class 3 must win instead of forcing the A branch.
        model.gate_ = FixedProbabilities([6, 0, 5, 3, 4], [.05, .4, .1, .35, .1])
        model.specialist_ = FixedProbabilities([2, 0, 1], [.33, .34, .33])
        model.classes_ = np.arange(7)
        model.n_features_in_ = 596
        return model

    def test_soft_routing_and_actual_class_alignment(self):
        model = self.fixed()
        probability = model.predict_proba(self.X[:3])
        expected = [.136, .132, .132, .35, .1, .1, .05]
        np.testing.assert_allclose(probability, np.tile(expected, (3, 1)), atol=1e-15)
        np.testing.assert_array_equal(model.predict(self.X[:3]), [3, 3, 3])
        self.assertEqual(probability.dtype, np.float64)
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-15)

    def test_original_leaf_weights_and_specialist_training_scope(self):
        captures = []
        original_fit = ExtraTreesClassifier.fit

        def capture(model, X, y, sample_weight=None):
            captures.append((np.asarray(X).copy(), np.asarray(y).copy(), np.asarray(sample_weight).copy()))
            return original_fit(model, X, y, sample_weight=sample_weight)

        original_X, original_y = self.X.copy(), self.y.copy()
        model = SagittalHierarchyClassifier(model_params=self.tree_params, random_state=123)
        with patch.object(ExtraTreesClassifier, "fit", capture):
            model.fit(self.X, self.y)
        self.assertEqual(len(captures), 2)
        gate_X, gate_y, gate_weights = captures[0]
        child_X, child_y, child_weights = captures[1]
        np.testing.assert_array_equal(gate_X, self.X)
        np.testing.assert_array_equal(gate_y, np.where(self.y <= 2, 0, self.y))
        np.testing.assert_array_equal(child_X, self.X[self.y <= 2])
        np.testing.assert_array_equal(child_y, self.y[self.y <= 2])
        for label in range(7):
            self.assertAlmostEqual(gate_weights[self.y == label].sum(), len(self.y) / 7)
        for label in range(3):
            self.assertAlmostEqual(child_weights[child_y == label].sum(), len(child_y) / 3)
        self.assertAlmostEqual(gate_weights[gate_y == 0].sum(), 3 * len(self.y) / 7)
        self.assertIsNone(model.gate_.class_weight)
        self.assertIsNone(model.specialist_.class_weight)
        self.assertEqual(model.gate_.random_state, 123)
        self.assertEqual(model.specialist_.random_state, 124)
        self.assertEqual(model.training_details_["original_training"]["class_counts"],
                         {str(i): int((self.y == i).sum()) for i in range(7)})
        self.assertEqual(model.training_details_["specialist"]["weight_summary"]["samples"], 21)
        json.dumps(model.training_details_, allow_nan=False)
        np.testing.assert_array_equal(self.X, original_X)
        np.testing.assert_array_equal(self.y, original_y)

    def test_deterministic_clone_serialization_and_both_families(self):
        for family, params in [("extra_trees", self.tree_params), ("rbf_svm", {"C": 1., "gamma": .01})]:
            with self.subTest(family=family):
                model = SagittalHierarchyClassifier(family, params, random_state=20260921)
                other = clone(model)
                model.fit(self.X, self.y)
                other.fit(self.X, self.y)
                probability = model.predict_proba(self.X)
                self.assertEqual(probability.shape, (len(self.X), 7))
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue((probability >= 0).all())
                np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-14)
                np.testing.assert_array_equal(probability, other.predict_proba(self.X))
                buffer = io.BytesIO()
                joblib.dump(model, buffer)
                buffer.seek(0)
                restored = joblib.load(buffer)
                np.testing.assert_array_equal(probability, restored.predict_proba(self.X))
                np.testing.assert_array_equal(model.classes_, np.arange(7))
                np.testing.assert_array_equal(model.predict(self.X), probability.argmax(axis=1))

    def test_seed_wrap_and_shared_params_without_mutating_input_config(self):
        params = self.tree_params.copy()
        model = SagittalHierarchyClassifier(model_params=params, random_state=2**32 - 1).fit(self.X, self.y)
        self.assertEqual(model.specialist_.random_state, 0)
        self.assertEqual(params, self.tree_params)
        for name, value in params.items():
            self.assertEqual(model.gate_.get_params()[name], value)
            self.assertEqual(model.specialist_.get_params()[name], value)

    def test_missing_bad_labels_and_invalid_features_rejected_before_fit(self):
        bad_labels = [self.y.astype(str), self.y.astype(float) + .25, self.y == 1,
                      np.where(self.y == 6, 7, self.y), np.where(self.y == 6, -1, self.y)]
        for labels in bad_labels:
            with self.subTest(labels=str(labels.dtype)), self.assertRaises(ValueError):
                SagittalHierarchyClassifier().fit(self.X, labels)
        for absent in range(7):
            mask = self.y != absent
            with self.subTest(absent=absent), self.assertRaises(ValueError):
                SagittalHierarchyClassifier().fit(self.X[mask], self.y[mask])
        with self.assertRaises(ValueError):
            SagittalHierarchyClassifier().fit(self.X[:, :-1], self.y)
        invalid = self.X.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            SagittalHierarchyClassifier().fit(invalid, self.y)
        with self.assertRaises(NotFittedError):
            SagittalHierarchyClassifier().predict_proba(self.X)
        with self.assertRaises(ValueError):
            self.fixed().predict_proba(self.X[:, :-1])

    def test_invalid_configuration_cannot_override_weighting_or_seed(self):
        for family in ("xgboost", "", None):
            with self.subTest(family=family), self.assertRaises(ValueError):
                SagittalHierarchyClassifier(family=family).fit(self.X, self.y)
        for seed in (-1, 2**32, 1.1, True, None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                SagittalHierarchyClassifier(random_state=seed).fit(self.X, self.y)
        for key in ("random_state", "class_weight", "sample_weight", "n_jobs", "probability", "kernel", "decision_function_shape"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                SagittalHierarchyClassifier(model_params={key: None}).fit(self.X, self.y)
        for params in ([], {"max_features": float("nan")}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                SagittalHierarchyClassifier(model_params=params).fit(self.X, self.y)

    def test_probability_contract_rejects_broken_head_outputs(self):
        for head in [FixedProbabilities([0, 3, 4, 5, 5], [.4, .3, .1, .1, .1]),
                     FixedProbabilities([0, 3, 4, 5, 7], [.4, .3, .1, .1, .1]),
                     FixedProbabilities([0, 3, 4, 5, 6], [.4, .3, .1, .1, -.1]),
                     FixedProbabilities([0, 3, 4, 5, 6], [0, 0, 0, 0, 0]),
                     FixedProbabilities([0, 3, 4, 5, 6], [float("nan"), .3, .1, .1, .1]),
                     FixedProbabilities([0, 3, 4, 5, 6], [.4, .3, .1, .1])]:
            model = self.fixed()
            model.gate_ = head
            with self.assertRaises(ValueError):
                model.predict_proba(self.X[:1])
        model = self.fixed()
        model.specialist_ = FixedProbabilities([0, 1, 3], [.3, .3, .4])
        with self.assertRaises(ValueError):
            model.predict_proba(self.X[:1])


if __name__ == "__main__":
    unittest.main()
