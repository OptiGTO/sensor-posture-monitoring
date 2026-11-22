"""Synthetic-only fixed leaf-support probabilities and immutable tree evidence."""
from copy import deepcopy
import hashlib
import io
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from src import leaf_support_v1 as subject


def fixture(trees=3, depth=2):
    y = np.repeat(np.arange(7), np.arange(1, 8))
    X = np.zeros((len(y), 596), dtype=np.float32)
    X[:, 0] = y
    forest = ExtraTreesClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=1,
        max_features=1.0, class_weight="balanced", random_state=15, n_jobs=1).fit(X, y)
    return forest, X, y


class LeafSupportTests(unittest.TestCase):
    def setUp(self):
        self.forest, self.X, self.y = fixture()

    def test_alpha0_returns_native_result_object_and_exact_probability_bytes(self):
        expected = self.forest.predict_proba(self.X)
        for alpha in (0, 0., np.float64(0)):
            actual = subject.leaf_support_probabilities(self.forest, self.X, alpha)
            self.assertEqual(actual.tobytes(), expected.tobytes())
        with patch.object(self.forest, "predict_proba", return_value=expected) as native:
            actual = subject.leaf_support_probabilities(self.forest, self.X, 0)
        self.assertIs(actual, expected)
        native.assert_called_once()

    def test_formula_uses_unweighted_counts_and_does_not_divide_native_probability(self):
        for alpha in (8, 32):
            expected = np.zeros((len(self.X), 7), np.float64)
            weighted_wrong = expected.copy()
            for tree in self.forest.estimators_:
                p = tree.predict_proba(self.X)
                leaves = tree.apply(self.X)
                n = tree.tree_.n_node_samples[leaves].astype(float)[:, None]
                w = tree.tree_.weighted_n_node_samples[leaves][:, None]
                expected += (n * p + alpha / 7) / (n + alpha)
                weighted_wrong += (w * p + alpha / 7) / (w + alpha)
            expected /= len(self.forest.estimators_)
            weighted_wrong /= len(self.forest.estimators_)
            actual = subject.leaf_support_probabilities(self.forest, self.X, alpha)
            self.assertEqual(actual.tobytes(), expected.tobytes())
            self.assertGreater(float(np.max(np.abs(actual - weighted_wrong))), .001)
            np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-15)

    def test_single_tree_uniform_contraction_strength_and_integral_float_class_alignment(self):
        forest, values, _ = fixture(trees=1, depth=None)
        tree = forest.estimators_[0]
        native = tree.predict_proba(values)
        n = tree.tree_.n_node_samples[tree.apply(values)][:, None]
        outputs = {}
        for alpha in (8, 32):
            actual = subject.leaf_support_probabilities(forest, values, alpha)
            np.testing.assert_allclose(actual - 1 / 7, (native - 1 / 7) * n / (n + alpha), atol=1e-16)
            self.assertTrue(np.all(actual >= np.minimum(native, 1 / 7) - 1e-16))
            self.assertTrue(np.all(actual <= np.maximum(native, 1 / 7) + 1e-16))
            outputs[alpha] = actual
        self.assertLess(np.abs(outputs[32] - 1 / 7).max(), np.abs(outputs[8] - 1 / 7).max())
        permutation = np.asarray([3, 0, 6, 2, 4, 1, 5])
        tree.classes_ = permutation.astype(np.float64)
        raw = np.arange(1, 8, dtype=np.float64)
        raw /= raw.sum()
        raw = np.broadcast_to(raw, (len(values), 7)).copy()
        aligned = np.empty_like(raw)
        aligned[:, permutation] = raw
        with patch.object(tree, "predict_proba", return_value=raw):
            actual = subject.leaf_support_probabilities(forest, values, 8)
        np.testing.assert_array_equal(actual, (n * aligned + 8 / 7) / (n + 8))

    def test_uniform_native_leaf_remains_uniform_for_both_strengths(self):
        y = np.arange(7)
        X = np.zeros((7, 596), np.float32)
        forest = ExtraTreesClassifier(n_estimators=1, class_weight="balanced", random_state=3,
            n_jobs=1).fit(X, y)
        for alpha in (0, 8, 32):
            np.testing.assert_allclose(subject.leaf_support_probabilities(forest, X, alpha), 1 / 7, atol=1e-16)

    def test_batch_order_reload_and_prediction_leave_input_and_tree_state_unchanged(self):
        before_X, before_state = self.X.tobytes(), subject.forest_state_sha256(self.forest)
        for alpha in (0, 8, 32):
            expected = subject.leaf_support_probabilities(self.forest, self.X, alpha)
            parts = np.concatenate([subject.leaf_support_probabilities(self.forest, self.X[:1], alpha),
                subject.leaf_support_probabilities(self.forest, self.X[1:13], alpha),
                subject.leaf_support_probabilities(self.forest, self.X[13:], alpha)])
            self.assertEqual(parts.tobytes(), expected.tobytes())
            np.testing.assert_array_equal(subject.leaf_support_probabilities(self.forest, self.X[::-1], alpha), expected[::-1])
            query = np.concatenate([self.X[:1], self.X * 100])
            np.testing.assert_array_equal(subject.leaf_support_probabilities(self.forest, query, alpha)[:1], expected[:1])
            stream = io.BytesIO()
            joblib.dump(self.forest, stream)
            stream.seek(0)
            restored = joblib.load(stream)
            self.assertEqual(subject.forest_state_sha256(restored), before_state)
            self.assertEqual(subject.leaf_support_probabilities(restored, self.X, alpha).tobytes(), expected.tobytes())
        self.assertEqual(self.X.tobytes(), before_X)
        self.assertEqual(subject.forest_state_sha256(self.forest), before_state)

    def test_forest_digest_is_reproducible_and_sensitive_to_each_learned_input(self):
        expected = subject.forest_state_sha256(self.forest)
        self.assertEqual(subject.forest_state_sha256(fixture()[0]), expected)
        mutations = [
            lambda f: f.estimators_[0].tree_.threshold.__setitem__(0, 10.),
            lambda f: f.estimators_[0].tree_.n_node_samples.__setitem__(0, 999),
            lambda f: f.estimators_[0].tree_.weighted_n_node_samples.__setitem__(0, 999.),
            lambda f: f.estimators_[0].tree_.value.__setitem__((0, 0, 0), .5),
            lambda f: f.estimators_[0].set_params(random_state=100),
            lambda f: setattr(f.estimators_[0], "classes_", np.arange(7, dtype=float)[::-1]),
            lambda f: f.estimators_.reverse(),
        ]
        for mutation in mutations:
            changed = deepcopy(self.forest)
            mutation(changed)
            self.assertNotEqual(subject.forest_state_sha256(changed), expected)

    def test_rule_normalizes_alpha_hashes_and_reports_support_limitations(self):
        for alpha in (0, 8, 32):
            rule = subject.leaf_support_rule(alpha)
            self.assertEqual(rule, subject.leaf_support_rule(float(alpha)))
            self.assertIs(type(rule["alpha"]), float)
            self.assertFalse(rule["support_is_independent_sample_size"])
            self.assertFalse(rule["calibrated_posterior"])
            self.assertFalse(rule["tree_probability_divided_by_support"])
            self.assertIn("n_node_samples", rule["support_source"])
            own_hash = rule.pop("rule_sha256")
            actual = hashlib.sha256(json.dumps(rule, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            self.assertEqual(own_hash, actual)
            rule["classes"][0] = 99
            self.assertEqual(subject.leaf_support_rule(alpha)["classes"][0], 0)

    def test_invalid_inputs_alphas_classes_and_support_fail_closed(self):
        for alpha in (-1, 1, 9, 33, np.nan, np.inf, True, np.bool_(False), "8", None):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                subject.leaf_support_probabilities(self.forest, self.X, alpha)
            with self.assertRaises(ValueError):
                subject.leaf_support_rule(alpha)
        for X in (self.X.astype(np.float64), self.X[:0], self.X[:, :-1], self.X[0],
                np.full_like(self.X, np.nan), np.full_like(self.X, np.inf)):
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                subject.leaf_support_probabilities(self.forest, X, 8)
        with self.assertRaises(ValueError):
            subject.leaf_support_probabilities(RandomForestClassifier(), self.X, 8)
        for classes in (np.arange(6), np.arange(7) + .5, np.arange(7).astype(str), np.zeros(7),
                np.full(7, np.nan), np.zeros(7, dtype=bool)):
            changed = deepcopy(self.forest)
            changed.estimators_[0].classes_ = classes
            with self.assertRaises(ValueError):
                subject.leaf_support_probabilities(changed, self.X, 8)
        changed = deepcopy(self.forest)
        changed.classes_ = np.arange(7)[::-1]
        with self.assertRaises(ValueError):
            subject.leaf_support_probabilities(changed, self.X, 0)
        changed = deepcopy(self.forest)
        leaf = changed.estimators_[0].apply(self.X[:1])[0]
        changed.estimators_[0].tree_.n_node_samples[leaf] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            subject.leaf_support_probabilities(changed, self.X, 8)


if __name__ == "__main__":
    unittest.main()
