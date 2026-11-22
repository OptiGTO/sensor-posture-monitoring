from __future__ import annotations

from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.base import clone
from sklearn.mixture import GaussianMixture

from src.class_conditional_mixture_v1 import (ClassConditionalMixtureClassifier,
    verify_class_conditional_mixture_identity)


def fixture():
    y = np.repeat(np.arange(7), np.arange(18, 25))
    X = np.random.default_rng(1313).normal(0, .6, (len(y), 596)).astype(np.float32)
    X[:, :7] += np.eye(7, dtype=np.float32)[y] * 1.2
    X[:, 7:13] += np.where(np.arange(len(y))[:, None] % 2, .7, -.7)
    return X, y


def classifier(**changes):
    return ClassConditionalMixtureClassifier(**{"n_init": 2, "random_state": 53, **changes})


class ClassConditionalMixtureTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = fixture()

    def test_each_backend_sees_only_its_training_class_in_float64(self):
        calls = []
        original_fit = GaussianMixture.fit
        def observe(estimator, X, *args, **kwargs):
            calls.append((X.copy(), estimator.get_params(deep=False), args, kwargs))
            return original_fit(estimator, X, *args, **kwargs)
        before, labels_before = self.X.copy(), self.y.copy()
        with patch.object(GaussianMixture, "fit", autospec=True, side_effect=observe):
            model = classifier().fit(self.X, self.y)
        self.assertEqual(len(calls), 7)
        for class_id, (values, params, args, kwargs) in enumerate(calls):
            np.testing.assert_array_equal(values, self.X[self.y == class_id].astype(np.float64))
            self.assertEqual(values.dtype, np.float64)
            self.assertEqual(params["covariance_type"], "diag")
            self.assertEqual(params["init_params"], "kmeans")
            self.assertFalse(params["warm_start"])
            self.assertEqual(params["random_state"], 53 + class_id)
            self.assertEqual(args, ())
            self.assertEqual(kwargs, {})
        np.testing.assert_array_equal(model.class_counts_, np.bincount(self.y))
        np.testing.assert_array_equal(model.class_priors_, np.repeat(1 / 7, 7))
        self.assertFalse(np.allclose(model.class_priors_, model.class_counts_ / len(self.y)))
        np.testing.assert_array_equal(self.X, before)
        np.testing.assert_array_equal(self.y, labels_before)
        self.assertTrue(verify_class_conditional_mixture_identity(model, model.get_params()))
        json.dumps(model.training_details_, allow_nan=False)

    def test_uniform_prior_posterior_matches_density_formula_and_normalizes_extremes(self):
        model = classifier().fit(self.X, self.y)
        values = self.X[:9].astype(np.float64)
        log_density = np.column_stack([item.score_samples(values) for item in model.estimators_]) + np.log(1 / 7)
        expected = np.exp(log_density - logsumexp(log_density, axis=1, keepdims=True))
        np.testing.assert_allclose(model.predict_proba(values), expected, atol=1e-13, rtol=1e-12)
        extreme = np.stack([np.full(596, 1e100), np.full(596, -1e100), np.zeros(596)])
        p = model.predict_proba(extreme)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((p >= 0).all())
        np.testing.assert_allclose(p.sum(1), 1., atol=1e-12, rtol=0)
        # Subtracting the row maximum before logsumexp retains the 1/7 answer
        # when absolute log densities are too large to represent a log(7) delta.
        with patch.object(GaussianMixture, "score_samples", return_value=np.full(3, -1e250)):
            np.testing.assert_allclose(model.predict_proba(extreme), np.full((3, 7), 1 / 7), atol=1e-15, rtol=0)
        with self.assertRaisesRegex(ValueError, "finite float64 range"):
            model.predict_proba(np.full((1, 596), 1e300))

    def test_one_component_control_is_one_regularized_gaussian_per_class(self):
        model = classifier(n_components=1).fit(self.X, self.y.astype(np.uint64))
        for class_id, estimator in enumerate(model.estimators_):
            values = self.X[self.y == class_id].astype(np.float64)
            np.testing.assert_allclose(estimator.means_[0], values.mean(0), atol=1e-14, rtol=1e-13)
            np.testing.assert_allclose(estimator.covariances_[0], values.var(0) + .1, atol=1e-14, rtol=1e-13)
            np.testing.assert_array_equal(estimator.weights_, [1.])
        self.assertTrue(verify_class_conditional_mixture_identity(model, model.get_params()))

    def test_refit_reload_batch_predictions_and_train_state_are_reproducible(self):
        first, second = classifier().fit(self.X, self.y), classifier().fit(self.X, self.y)
        p = first.predict_proba(self.X)
        np.testing.assert_array_equal(p, second.predict_proba(self.X))
        self.assertEqual(first.state_sha256_, second.state_sha256_)
        batches = np.concatenate([first.predict_proba(self.X[:9]), first.predict_proba(self.X[9:])])
        np.testing.assert_allclose(p, batches, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(p.argmax(1), batches.argmax(1))
        stream = io.BytesIO(); joblib.dump(first, stream); stream.seek(0)
        restored = joblib.load(stream)
        np.testing.assert_array_equal(p, restored.predict_proba(self.X))
        np.testing.assert_array_equal(restored.predict(self.X), p.argmax(1))
        self.assertTrue(verify_class_conditional_mixture_identity(restored, first.get_params()))
        hashes = first.state_sha256_.copy()
        first.predict_proba(self.X[:3] * 1e6)
        self.assertEqual(first.state_sha256_, hashes)
        self.assertTrue(verify_class_conditional_mixture_identity(first, first.get_params()))
        self.assertEqual(clone(first).get_params(), first.get_params())

    def test_class_seed_wrap_is_explicit_and_validated(self):
        model = classifier(n_components=1, n_init=1, random_state=2**32 - 2).fit(self.X, self.y)
        expected = [2**32 - 2, 2**32 - 1, 0, 1, 2, 3, 4]
        self.assertEqual(model.effective_seeds_, expected)
        self.assertEqual([item.random_state for item in model.estimators_], expected)
        self.assertTrue(verify_class_conditional_mixture_identity(model, model.get_params()))

    def test_nonconvergence_raises_without_fallback_or_partial_fitted_state(self):
        original_fit = GaussianMixture.fit
        calls = []
        def fail_convergence(estimator, X, *args, **kwargs):
            result = original_fit(estimator, X, *args, **kwargs)
            calls.append(estimator)
            result.converged_ = False
            return result
        model = classifier()
        with patch.object(GaussianMixture, "fit", autospec=True, side_effect=fail_convergence):
            with self.assertRaisesRegex(ValueError, "Class 0.*failed to converge.*no fallback"):
                model.fit(self.X, self.y)
        self.assertEqual(len(calls), 1)
        self.assertFalse(hasattr(model, "estimators_"))
        self.assertFalse(hasattr(model, "fitted_params_"))

    def test_invalid_parameters_features_labels_and_external_validation_are_rejected(self):
        cases = [{"n_components": 0}, {"n_components": True}, {"n_init": 0}, {"max_iter": 0},
            {"tol": 0}, {"tol": float("nan")}, {"reg_covar": -1}, {"reg_covar": True},
            {"random_state": None}, {"random_state": -1}, {"random_state": 2**32}]
        with patch.object(GaussianMixture, "fit") as fit:
            for params in cases:
                with self.subTest(params=params), self.assertRaises(ValueError):
                    classifier(**params).fit(self.X, self.y)
            data_cases = [(self.X[:, :-1], self.y), (self.X[:0], self.y[:0]),
                (self.X.astype(np.int64), self.y), (np.full_like(self.X, np.nan), self.y),
                (self.X, self.y.astype(float)), (self.X[self.y != 6], self.y[self.y != 6]),
                (self.X, self.y[:-1]), (self.X[:7], np.arange(7))]
            for values, labels in data_cases:
                with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                    classifier().fit(values, labels)
            with self.assertRaises(TypeError):
                classifier().fit(self.X, self.y, validation_data=(self.X, self.y))
        fit.assert_not_called()

    def test_identity_rejects_recipe_seed_class_density_and_audit_tampering(self):
        original = classifier().fit(self.X, self.y)
        expected_params = original.get_params()
        mutations = [lambda m: m.set_params(reg_covar=.2),
            lambda m: m.estimators_[0].set_params(random_state=999),
            lambda m: m.estimators_[0].set_params(covariance_type="spherical"),
            lambda m: setattr(m.estimators_[0], "converged_", False),
            lambda m: m.estimators_[0].means_.__setitem__((0, 0), 999.),
            lambda m: m.estimators_[0].covariances_.__setitem__((0, 0), 999.),
            lambda m: m.estimators_[0].precisions_.__setitem__((0, 0), 999.),
            lambda m: m.estimators_[0].lower_bounds_.__setitem__(0, 999.),
            lambda m: m.classes_.__setitem__(0, 99),
            lambda m: m.class_counts_.__setitem__(0, 99),
            lambda m: m.class_priors_.__setitem__(0, .2),
            lambda m: m.training_details_["class_models"][0].update(training_rows=999)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = deepcopy(original)
                mutation(changed)
                with self.assertRaises(ValueError):
                    verify_class_conditional_mixture_identity(changed, expected_params)
        original.set_params(n_components=1)
        with self.assertRaisesRegex(ValueError, "changed after fitting"):
            original.predict_proba(self.X)


if __name__ == "__main__":
    unittest.main()
