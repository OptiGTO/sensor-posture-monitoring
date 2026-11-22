from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier

from src.binary_forest_v1 import (BinaryForestV1, binary_forest_rule,
    binary_forest_state_sha256, verify_binary_forest)


def fixture():
    y = np.repeat(np.arange(7), np.arange(10, 17))
    X = np.random.default_rng(1616).normal(size=(len(y), 596)).astype(np.float32)
    X[:, :7] += np.eye(7, dtype=np.float32)[y] * 1.5
    return X, y


def model(**changes):
    return BinaryForestV1(**{"n_estimators": 4, "min_samples_leaf": 2,
        "max_depth": 5, "random_state": 53, **changes})


def parameters(estimator):
    result = estimator.get_params(deep=False)
    result.pop("random_state")
    return result


class BinaryForestTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = fixture()

    def test_seven_heads_receive_identical_train_rows_and_balanced_binary_labels(self):
        calls = []
        original_fit = ExtraTreesClassifier.fit
        def observe(estimator, X, y, *args, **kwargs):
            calls.append((X.copy(), y.copy(), estimator.get_params(deep=False), args, kwargs))
            return original_fit(estimator, X, y, *args, **kwargs)
        before, labels_before = self.X.copy(), self.y.copy()
        with patch.object(ExtraTreesClassifier, "fit", autospec=True, side_effect=observe):
            trained = model().fit(self.X, self.y)
        self.assertEqual(len(calls), 7)
        for c, (values, binary, params, args, kwargs) in enumerate(calls):
            np.testing.assert_array_equal(values, self.X)
            np.testing.assert_array_equal(binary, (self.y == c).astype(np.int64))
            self.assertEqual(values.dtype, np.float32)
            self.assertEqual(binary.dtype, np.int64)
            self.assertEqual(params["class_weight"], "balanced")
            self.assertEqual(params["random_state"], 53 + c)
            self.assertEqual(params["n_jobs"], 1)
            self.assertFalse(params["bootstrap"])
            self.assertEqual(args, ())
            self.assertEqual(kwargs, {})
            audit = trained.training_details_["heads"][c]
            counts = np.bincount(binary, minlength=2)
            self.assertEqual(audit["binary_class_counts"], counts.tolist())
            np.testing.assert_array_equal(audit["binary_class_weights"], len(binary) / (2 * counts))
            self.assertEqual(audit["binary_weight_mass"], [len(binary) / 2] * 2)
            self.assertEqual(audit["training_rows"], len(self.X))
            for tree in trained.estimators_[c].estimators_:
                self.assertEqual(tree.tree_.n_node_samples[0], len(self.X))
                np.testing.assert_allclose(tree.tree_.value[0, 0], [.5, .5], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(trained.class_counts_, np.bincount(self.y))
        np.testing.assert_array_equal(self.X, before)
        np.testing.assert_array_equal(self.y, labels_before)
        self.assertTrue(verify_binary_forest(trained, parameters(trained), 53))
        json.dumps(trained.training_details_, allow_nan=False)

    def test_probabilities_equal_manual_positive_head_normalization(self):
        trained = model().fit(self.X, self.y)
        query = self.X[[1, 17, 28, 45, 69, 81]]
        scores = np.column_stack([head.predict_proba(query)[:, np.flatnonzero(head.classes_ == 1)[0]]
            for head in trained.estimators_])
        totals = scores.sum(1, keepdims=True)
        expected = np.full_like(scores, 1 / 7)
        np.divide(scores, totals, out=expected, where=totals > 0)
        np.testing.assert_array_equal(trained.predict_proba(query), expected)
        np.testing.assert_array_equal(trained.predict(query), expected.argmax(1))
        np.testing.assert_allclose(expected.sum(1), 1., atol=1e-12, rtol=0)

    def test_positive_class_alignment_exact_zero_policy_and_no_inference_counters(self):
        trained = model().fit(self.X, self.y)
        query = self.X[:3]
        before = deepcopy(trained.training_details_)
        # The first row has no positive scores, the second a class6 maximum,
        # and the third ties every head. Reversed column order must be honored.
        with ExitStack() as stack:
            for c, head in enumerate(trained.estimators_):
                positive = np.asarray([0., (c + 1) / 8, .3])
                probability = np.column_stack((1 - positive, positive))
                if c % 2:
                    stack.enter_context(patch.object(head, "classes_", np.asarray([1, 0])))
                    probability = probability[:, ::-1]
                stack.enter_context(patch.object(head, "predict_proba", return_value=probability))
            p = trained.predict_proba(query)
            np.testing.assert_array_equal(p[0], np.full(7, 1 / 7))
            np.testing.assert_allclose(p[1], np.arange(1, 8) / 28, atol=1e-15, rtol=0)
            np.testing.assert_array_equal(trained.predict(query), [0, 6, 0])
            diagnostic = trained.probability_diagnostics(query)
            self.assertEqual(diagnostic["rows"], 3)
            self.assertEqual(diagnostic["all_zero_rows"], 1)
        self.assertEqual(before, trained.training_details_)
        self.assertFalse(hasattr(trained, "all_zero_rows_"))
        self.assertTrue(verify_binary_forest(trained, parameters(trained), 53))

    def test_rejects_invalid_head_probability_and_class_schema(self):
        trained = model().fit(self.X, self.y)
        cases = [np.ones((3, 1)), np.full((3, 2), np.nan),
            np.tile([-0.1, 1.1], (3, 1)), np.tile([.7, .7], (3, 1)),
            np.tile([0., 0.], (3, 1))]
        for invalid in cases:
            with self.subTest(probability=invalid), patch.object(trained.estimators_[2], "predict_proba", return_value=invalid):
                with self.assertRaisesRegex(ValueError, "invalid two-column"):
                    trained.predict_proba(self.X[:3])
        for classes in ([0, 0], [0, 2], [.25, 1], [0, np.nan]):
            with self.subTest(classes=classes), patch.object(trained.estimators_[2], "classes_", np.asarray(classes)):
                with self.assertRaisesRegex(ValueError, "classes"):
                    trained.predict_proba(self.X[:3])

    def test_same_seed_refit_reload_batch_and_prediction_state_are_identical(self):
        first = model().fit(self.X, self.y)
        second = clone(first).fit(self.X, self.y)
        p = first.predict_proba(self.X)
        np.testing.assert_array_equal(p, second.predict_proba(self.X))
        self.assertEqual(first.training_details_, second.training_details_)
        self.assertEqual(first.fitted_state_sha256_, second.fitted_state_sha256_)
        batch = np.concatenate([first.predict_proba(self.X[:5]), first.predict_proba(self.X[5:27]), first.predict_proba(self.X[27:])])
        np.testing.assert_array_equal(p, batch)
        stream = io.BytesIO(); joblib.dump(first, stream); stream.seek(0)
        restored = joblib.load(stream)
        np.testing.assert_array_equal(p, restored.predict_proba(self.X))
        self.assertTrue(verify_binary_forest(restored, parameters(first), 53))
        before = joblib.hash(first)
        extreme = np.stack([np.full(596, 1e30), np.full(596, -1e30)]).astype(np.float32)
        self.assertTrue(np.isfinite(first.predict_proba(extreme)).all())
        first.probability_diagnostics(extreme)
        self.assertEqual(before, joblib.hash(first))

    def test_seed_wrap_and_unsigned_valid_labels_preserve_seven_class_order(self):
        trained = model(random_state=2**32 - 2).fit(self.X, self.y.astype(np.uint64))
        seeds = [2**32 - 2, 2**32 - 1, 0, 1, 2, 3, 4]
        self.assertEqual(trained.effective_seeds_, seeds)
        self.assertEqual([head.random_state for head in trained.estimators_], seeds)
        np.testing.assert_array_equal(trained.classes_, np.arange(7))
        self.assertTrue(verify_binary_forest(trained, parameters(trained), 2**32 - 2))

    def test_strict_parameters_feature_label_and_training_interface_contracts(self):
        params = [{"n_estimators": 0}, {"n_estimators": True}, {"n_estimators": 2.5},
            {"min_samples_leaf": 0}, {"min_samples_leaf": .5}, {"max_depth": 0}, {"max_depth": False},
            {"max_features": 0.}, {"max_features": 1.1}, {"max_features": True}, {"max_features": "sqrt"},
            {"max_features": np.nan}, {"max_features": 597}, {"random_state": -1},
            {"random_state": 2**32}, {"random_state": True}, {"random_state": None}]
        invalid_data = [(self.X.astype(np.float64), self.y), (self.X.astype(int), self.y),
            (self.X[:, :-1], self.y), (self.X[:0], self.y[:0]), (np.full_like(self.X, np.nan), self.y),
            (self.X, self.y.astype(float)), (self.X, self.y.astype(str)), (self.X, self.y[:-1]),
            (self.X[self.y != 6], self.y[self.y != 6]), (self.X, self.y + 1)]
        with patch.object(ExtraTreesClassifier, "fit") as fit:
            for changes in params:
                with self.subTest(params=changes), self.assertRaises(ValueError):
                    model(**changes).fit(self.X, self.y)
            for X, y in invalid_data:
                with self.subTest(dtype=X.dtype, shape=X.shape), self.assertRaises(ValueError):
                    model().fit(X, y)
            with self.assertRaises(TypeError):
                model().fit(self.X, self.y, validation_data=(self.X, self.y))
            with self.assertRaises(TypeError):
                model().fit(self.X, self.y, sample_weight=np.ones(len(self.y)))
        fit.assert_not_called()
        trained = model(max_features=13, max_depth=None).fit(self.X, self.y)
        self.assertTrue(verify_binary_forest(trained, parameters(trained), 53))

    def test_rule_is_fresh_and_state_hash_ignores_structured_node_padding(self):
        rule = binary_forest_rule()
        rule["classes"][0] = 99
        self.assertEqual(binary_forest_rule()["classes"], list(range(7)))
        json.dumps(binary_forest_rule(), allow_nan=False)
        trained = model().fit(self.X, self.y)
        before = binary_forest_state_sha256(trained)
        nodes = trained.estimators_[0].estimators_[0].tree_.__getstate__()["nodes"]
        occupied = set()
        for name in nodes.dtype.names:
            dtype, offset = nodes.dtype.fields[name][:2]
            occupied.update(range(offset, offset + dtype.itemsize))
        padding = sorted(set(range(nodes.dtype.itemsize)) - occupied)
        if padding:
            byte_view = nodes.view(np.uint8).reshape(len(nodes), nodes.dtype.itemsize)
            byte_view[:, padding] ^= np.uint8(255)
        self.assertEqual(before, binary_forest_state_sha256(trained))
        self.assertTrue(verify_binary_forest(trained, parameters(trained), 53))

    def test_identity_rejects_backend_tree_seed_state_scope_and_audit_tampering(self):
        original = model().fit(self.X, self.y)
        mutations = [lambda m: m.set_params(n_estimators=5),
            lambda m: m.estimators_[0].set_params(random_state=999),
            lambda m: m.estimators_[0].set_params(class_weight=None),
            lambda m: m.estimators_[0].set_params(bootstrap=True),
            lambda m: m.estimators_[0].estimators_[0].set_params(random_state=999),
            lambda m: m.estimators_[0].estimators_[0].set_params(min_samples_leaf=1),
            lambda m: m.estimators_[0].estimators_[0].tree_.value.__setitem__((0, 0, 0), .125),
            lambda m: m.estimators_[0].estimators_[0].tree_.threshold.__setitem__(0, 998.),
            lambda m: m.estimators_[0].estimators_[0].tree_.n_node_samples.__setitem__(0, 1),
            lambda m: m.estimators_[0].estimators_[0].tree_.weighted_n_node_samples.__setitem__(0, 1.),
            lambda m: m.estimators_[0].classes_.__setitem__(0, 2),
            lambda m: m.classes_.__setitem__(0, 99),
            lambda m: m.class_counts_.__setitem__(0, 99),
            lambda m: m.effective_seeds_.__setitem__(0, 999),
            lambda m: setattr(m, "training_input_sha256_", "f" * 64),
            lambda m: m.binary_labels_sha256_.__setitem__(0, "f" * 64),
            lambda m: m.training_details_["heads"][0].update(positive_rows=999),
            lambda m: m.training_details_.update(external_validation_used=True)]
        for mutation in mutations:
            changed = deepcopy(original)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_binary_forest(changed, parameters(original), 53)
        original.set_params(max_depth=6)
        with self.assertRaisesRegex(ValueError, "changed after fitting"):
            original.predict_proba(self.X)


if __name__ == "__main__":
    unittest.main()
