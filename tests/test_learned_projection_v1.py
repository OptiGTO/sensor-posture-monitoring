"""Synthetic numerical, TRAIN-scope, persistence and integrity contracts."""
from copy import deepcopy
import io
import unittest
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits

from src import learned_projection_v1 as subject
from src.posture_core import feature_names


class LearnedProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def setUp(self):
        self.y = np.repeat(np.arange(7), 14)
        rng = np.random.default_rng(1703)
        self.pca_X = np.zeros((len(self.y), 596), dtype=np.float32)
        self.pca_X[:, :80] = rng.normal(size=(len(self.y), 80))
        self.lda_X = np.zeros((len(self.y), 596), dtype=np.float64)
        self.active = np.asarray([0, 2, 5, 9, 13, 21, 33, 45, 70, 100, 222, 595])
        self.lda_X[:, self.active] = rng.normal(size=(len(self.y), len(self.active)))
        self.lda_X[:, self.active[:6]] += rng.normal(size=(7, 6))[self.y]
        self.lda_X[:, 1] = 3.5

    def fit(self, mode):
        return subject.LearnedProjectionV1(mode).fit(self.lda_X if mode == "all_plus_lda6" else self.pca_X, self.y)

    def test_identity_preserves_original_object_dtype_and_bytes_without_arithmetic(self):
        for dtype in (np.float32, np.float64):
            train = self.pca_X.astype(dtype)
            train[0, 0] = -0.
            model = subject.LearnedProjectionV1().fit(train, self.y)
            query = train[::-1]
            before = query.tobytes()
            result = model.transform(query)
            self.assertIs(result, query)
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.tobytes(), before)
            self.assertIsNone(model.estimator_)
            self.assertIsNone(model.components_)
            self.assertEqual(model.training_input_sha256_, subject._array_digest(train))
            self.assertEqual(model.training_labels_sha256_, subject._array_digest(self.y.astype(np.int64)))
            self.assertFalse(model.training_details_["labels_used"])
            self.assertTrue(subject.verify_learned_projection(model, "identity"))

    def test_pca_full_svd_centering_matches_independent_float64_calculation(self):
        model = self.fit("pca64")
        expected = PCA(n_components=64, svd_solver="full", whiten=False, copy=True).fit(self.pca_X.astype(np.float64))
        np.testing.assert_array_equal(model.components_, expected.components_)
        query = self.pca_X[3:8].astype(np.float64) + .125
        result = model.transform(query)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        self.assertEqual(result.shape, (5, 64))
        np.testing.assert_allclose(result, expected.transform(query), rtol=1e-12, atol=1e-12)
        self.assertEqual(model.rank_tolerance_, np.finfo(float).eps * 596 * expected.singular_values_[0])
        # Labels are recorded but never passed to PCA; changing them cannot change fitted coordinates.
        permuted = subject.LearnedProjectionV1("pca64").fit(self.pca_X, self.y[::-1])
        np.testing.assert_array_equal(model.components_, permuted.components_)
        self.assertNotEqual(model.training_labels_sha256_, permuted.training_labels_sha256_)

    def test_pca_insufficient_rows_and_centered_rank_fail_without_dimension_fallback(self):
        rows = np.arange(64)
        with self.assertRaisesRegex(ValueError, "at least65"):
            subject.LearnedProjectionV1("pca64").fit(self.pca_X[rows], np.arange(64) % 7)
        rank63 = self.pca_X.copy()
        rank63[:, 63:] = 0
        with self.assertRaisesRegex(ValueError, "rank is below64"):
            subject.LearnedProjectionV1("pca64").fit(rank63, self.y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, "rank is below64"):
                subject.LearnedProjectionV1("pca64").fit(np.zeros_like(self.pca_X), self.y)

    def test_lda_uses_train_constant_mask_empirical_priors_and_signed_eigen_axes(self):
        # Unequal frequencies distinguish empirical priors from uniform-class priors.
        keep = np.ones(len(self.y), dtype=bool)
        keep[1:4] = False
        X, y = self.lda_X[keep], self.y[keep]
        model = subject.LearnedProjectionV1("all_plus_lda6").fit(X, y)
        expected = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=6, priors=None)
        expected.fit(X[:, self.active], y)
        raw = expected.scalings_[:, :6].T
        signs = np.where(raw[np.arange(6), np.argmax(np.abs(raw), axis=1)] < 0., -1., 1.)
        np.testing.assert_array_equal(model.retained_indices_, self.active)
        np.testing.assert_array_equal(model.signs_, signs)
        np.testing.assert_array_equal(model.components_, raw * signs[:, None])
        np.testing.assert_array_equal(model.estimator_.priors_, np.bincount(y) / len(y))
        query = X[:3].astype(np.float32)
        result = model.transform(query)
        self.assertEqual(result.shape, (3, 602))
        self.assertEqual(result.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(result[:, :596], query.astype(np.float64))
        np.testing.assert_allclose(result[:, 596:], expected.transform(query[:, self.active].astype(np.float64)) * signs,
            rtol=1e-12, atol=1e-12)
        anchors = np.argmax(np.abs(model.components_), axis=1)
        self.assertTrue(np.all(model.components_[np.arange(6), anchors] >= 0.))
        self.assertTrue(model.training_details_["labels_used"])

    def test_lda_singular_within_class_covariance_and_inadequate_train_scope_fail(self):
        # Variation between postures is not a substitute for within-class covariance.
        singular = np.zeros_like(self.lda_X)
        singular[:, :7] = np.eye(7)[self.y]
        with self.assertRaisesRegex(ValueError, "singular; no LDA fallback"):
            subject.LearnedProjectionV1("all_plus_lda6").fit(singular, self.y)
        too_few_columns = self.lda_X.copy()
        too_few_columns[:, 6:] = 0
        with self.assertRaisesRegex(ValueError, "six nonconstant"):
            subject.LearnedProjectionV1("all_plus_lda6").fit(too_few_columns, self.y)
        keep = (self.y != 0) | (np.arange(len(self.y)) == 0)
        with self.assertRaisesRegex(ValueError, "two TRAIN rows"):
            subject.LearnedProjectionV1("all_plus_lda6").fit(self.lda_X[keep], self.y[keep])

    def test_query_batches_never_refit_and_permutations_have_exact_row_outputs(self):
        for mode, X in (("pca64", self.pca_X), ("all_plus_lda6", self.lda_X)):
            model = self.fit(mode)
            query = np.concatenate([X[:3], np.full_like(X[:2], 1000)])
            raw = query.tobytes()
            audit = deepcopy(model.training_details_)
            state = subject.learned_projection_state_sha256(model)
            with patch.object(type(model.estimator_), "fit", side_effect=AssertionError("query fit")):
                expected = model.transform(query)
                per_row = np.concatenate([model.transform(query[i:i + 1]) for i in range(len(query))])
                reversed_rows = model.transform(query[::-1])
            self.assertEqual(expected.tobytes(), per_row.tobytes())
            self.assertEqual(expected[::-1].tobytes(), reversed_rows.tobytes())
            self.assertEqual(query.tobytes(), raw)
            self.assertEqual(audit, model.training_details_)
            self.assertEqual(state, subject.learned_projection_state_sha256(model))
            self.assertTrue(subject.verify_learned_projection(model, mode))

    def test_reload_refit_and_feature_names_preserve_exact_outputs_and_fitted_state(self):
        for mode, X in (("identity", self.pca_X), ("pca64", self.pca_X), ("all_plus_lda6", self.lda_X)):
            model = self.fit(mode)
            memory = io.BytesIO()
            joblib.dump(model, memory)
            memory.seek(0)
            restored = joblib.load(memory)
            self.assertEqual(model.transform(X).tobytes(), restored.transform(X).tobytes())
            self.assertEqual(model.transform(X).tobytes(), self.fit(mode).transform(X).tobytes())
            self.assertEqual(model.state_sha256_, restored.state_sha256_)
            self.assertTrue(subject.verify_learned_projection(restored, mode))
            names = model.get_feature_names_out(feature_names())
            self.assertEqual(len(names), subject.WIDTHS[mode])
            names[0] = "changed"
            self.assertNotEqual(model.get_feature_names_out()[0], "changed")
            with self.assertRaises(ValueError):
                model.get_feature_names_out(feature_names()[::-1])

    def test_invalid_input_mode_labels_and_named_schema_fail_explicitly(self):
        with self.assertRaises(ValueError):
            subject.LearnedProjectionV1("pca32").fit(self.pca_X, self.y)
        for X in (self.pca_X[:, :-1], self.pca_X[:0], self.pca_X[0], self.pca_X.astype(np.int64),
                self.pca_X.astype(str), np.full_like(self.pca_X, np.inf), np.full_like(self.pca_X, np.nan)):
            with self.subTest(dtype=X.dtype, shape=X.shape), self.assertRaises(ValueError):
                subject.LearnedProjectionV1().fit(X, self.y)
        for y in (self.y.astype(float), self.y[:-1], np.zeros_like(self.y), self.y[:, None]):
            with self.assertRaises(ValueError):
                subject.LearnedProjectionV1().fit(self.pca_X, y)
        model = self.fit("pca64")
        named = pd.DataFrame(self.pca_X, columns=feature_names())
        np.testing.assert_array_equal(model.transform(named), model.transform(self.pca_X))
        with self.assertRaises(ValueError):
            model.transform(named.iloc[:, ::-1])
        with self.assertRaises(ValueError):
            model.transform(self.pca_X[:, :-1])

    def test_full_verifier_detects_solver_coordinates_covariance_mask_and_audit_tampering(self):
        pca, lda = self.fit("pca64"), self.fit("all_plus_lda6")
        for original, mode, mutate in (
                (pca, "pca64", lambda m: setattr(m, "mode", "identity")),
                (pca, "pca64", lambda m: m.components_.__setitem__((0, 0), 123.)),
                (pca, "pca64", lambda m: m.estimator_.mean_.__setitem__(0, 123.)),
                (pca, "pca64", lambda m: m.estimator_.set_params(whiten=True)),
                (pca, "pca64", lambda m: m.training_details_.update(external_validation_used=True)),
                (pca, "pca64", lambda m: setattr(m, "training_input_sha256_", "a" * 64)),
                (pca, "pca64", lambda m: setattr(m, "state_sha256_", "a" * 64)),
                (lda, "all_plus_lda6", lambda m: m.estimator_.covariance_.__setitem__((0, 0), 123.)),
                (lda, "all_plus_lda6", lambda m: m.estimator_.scalings_.__setitem__((0, 0), 123.)),
                (lda, "all_plus_lda6", lambda m: m.retained_indices_.__setitem__(0, 1)),
                (lda, "all_plus_lda6", lambda m: m.rule_.update(labels_used=False))):
            changed = deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                subject.verify_learned_projection(changed, mode)

    def test_inference_uses_cheap_guards_while_explicit_verifier_keeps_complete_hash_checks(self):
        model = self.fit("all_plus_lda6")
        with patch.object(subject, "_audit", side_effect=AssertionError("expensive audit in inference")), \
                patch.object(subject, "learned_projection_state_sha256", side_effect=AssertionError("expensive hash in inference")):
            self.assertEqual(model.transform(self.lda_X[:1]).shape, (1, 602))
            self.assertEqual(len(model.get_feature_names_out()), 602)
        for mutate in (lambda m: setattr(m, "mode", "pca64"),
                lambda m: setattr(m, "n_features_out_", 601),
                lambda m: m.components_.__setitem__((0, 0), np.nan),
                lambda m: setattr(m, "retained_indices_", np.array([0, 1, 1, 3, 4, 5]))):
            changed = deepcopy(model)
            mutate(changed)
            with self.assertRaises(ValueError):
                changed.transform(self.lda_X[:1])

    def test_same_value_fortran_components_are_rejected_and_c_order_restores_exact_batch_bytes(self):
        for mode, X in (("pca64", self.pca_X), ("all_plus_lda6", self.lda_X)):
            model = self.fit(mode)
            query = np.concatenate([X[:3], np.full_like(X[:2], 1000)])
            expected = model.transform(query)
            expected_rows = np.concatenate([model.transform(row[None, :]) for row in query])
            self.assertEqual(expected.tobytes(), expected_rows.tobytes())
            model.components_ = np.asfortranarray(model.components_)
            self.assertTrue(model.components_.flags.f_contiguous)
            self.assertFalse(model.components_.flags.c_contiguous)
            # C-byte digests alone cannot distinguish equal-valued memory layouts.
            self.assertEqual(model.state_sha256_, subject.learned_projection_state_sha256(model))
            with self.assertRaisesRegex(ValueError, "C-order layout"):
                subject.verify_learned_projection(model, mode)
            with self.assertRaisesRegex(ValueError, "C-order layout"):
                model.transform(query)
            model.components_ = np.ascontiguousarray(model.components_)
            self.assertTrue(subject.verify_learned_projection(model, mode))
            self.assertEqual(model.transform(query).tobytes(), expected.tobytes())
            actual_rows = np.concatenate([model.transform(row[None, :]) for row in query])
            self.assertEqual(actual_rows.tobytes(), expected.tobytes())


if __name__ == "__main__":
    unittest.main()
