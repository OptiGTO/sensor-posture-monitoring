from __future__ import annotations

import io
import unittest

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import f_classif

from src.domain_feature_selection import SubjectInvariantSelector, ShrinkageDiscriminantProjection


class DomainFeatureSelectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(917)
        self.groups = np.repeat(np.arange(6), 28)
        self.y = np.tile(np.repeat(np.arange(7), 4), 6)
        self.X = rng.normal(size=(len(self.y), 596))
        self.X[:, 0] = self.y * 2 + rng.normal(0, .08, len(self.y))
        self.X[:, 1] = self.groups * 100 + self.y * 2 + rng.normal(0, .08, len(self.y))

    def test_posture_signal_retained_subject_distractor_penalized(self):
        model = SubjectInvariantSelector(k=8, strength=1).fit(self.X, self.y, groups=self.groups)
        self.assertIn(0, model.indices_)
        self.assertNotIn(1, model.indices_)
        self.assertGreater(model.subject_f_scores_[1], model.posture_f_scores_[1])
        expected = model.posture_f_scores_ / (1 + model.subject_f_scores_)
        np.testing.assert_allclose(model.scores_, expected)
        self.assertEqual(model.n_subjects_seen_, 6)
        self.assertEqual(model.n_classes_seen_, 7)
        self.assertEqual(model.n_samples_seen_, len(self.y))

    def test_training_statistics_are_unchanged_by_adversarial_heldout_transform(self):
        model = SubjectInvariantSelector(k=12).fit(self.X, self.y, groups=self.groups)
        original_scores = model.scores_.copy()
        original_indices = model.indices_.copy()
        heldout = self.X[:10] * 1e9
        model.transform(heldout)
        np.testing.assert_array_equal(model.scores_, original_scores)
        np.testing.assert_array_equal(model.indices_, original_indices)
        expected, _ = f_classif(self.X, self.y)
        np.testing.assert_allclose(model.posture_f_scores_, expected)
        self.assertEqual(model.n_samples_seen_, len(self.y))

    def test_subject_identity_renaming_does_not_change_selected_features(self):
        original = SubjectInvariantSelector(k=24).fit(self.X, self.y, groups=self.groups)
        names = np.asarray(['f', 'b', 'e', 'a', 'd', 'c'])[self.groups]
        renamed = SubjectInvariantSelector(k=24).fit(self.X, self.y, groups=names)
        np.testing.assert_allclose(original.scores_, renamed.scores_, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(original.indices_, renamed.indices_)

    def test_constants_ties_perfect_separation_and_extreme_strength_are_finite(self):
        X = np.zeros_like(self.X)
        model = SubjectInvariantSelector(k=6).fit(X, self.y, groups=self.groups)
        np.testing.assert_array_equal(model.indices_, np.arange(6))
        self.assertTrue(np.isfinite(model.scores_).all())
        X[:, 10] = self.y
        X[:, 11] = self.groups
        for strength in (0., 1., np.finfo(float).max):
            model = SubjectInvariantSelector(k=6, strength=strength).fit(X, self.y, groups=self.groups)
            self.assertTrue(np.isfinite(model.scores_).all())
            self.assertIn(10, model.indices_)

    def test_serialization_clone_parameter_mutation_and_input_preservation(self):
        before = self.X.copy()
        model = clone(SubjectInvariantSelector(k=12)).fit(self.X, self.y, groups=self.groups)
        expected = model.transform(self.X)
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        np.testing.assert_array_equal(restored.transform(self.X), expected)
        np.testing.assert_array_equal(restored.get_feature_names_out(), model.get_feature_names_out())
        np.testing.assert_array_equal(self.X, before)
        for parameters in ({'k': 13}, {'strength': 2.}):
            modified = joblib.load(io.BytesIO(stream.getvalue()))
            modified.set_params(**parameters)
            with self.assertRaisesRegex(ValueError, 'changed after fitting'):
                modified.transform(self.X)
        with self.assertRaises(ValueError):
            restored.transform(self.X[:, :-1])

    def test_invalid_training_contracts_fail(self):
        for groups in (None, self.groups[:-1], np.zeros(len(self.y)), np.arange(len(self.y))):
            with self.assertRaises(ValueError):
                SubjectInvariantSelector().fit(self.X, self.y, groups=groups)
        for parameters in ({'k': 0}, {'k': 597}, {'k': True}, {'strength': -1}, {'strength': float('inf')}):
            with self.assertRaises(ValueError):
                SubjectInvariantSelector(**parameters).fit(self.X, self.y, groups=self.groups)
        with self.assertRaises(ValueError):
            SubjectInvariantSelector().fit(self.X, np.zeros(len(self.y)), groups=self.groups)

    def test_shrinkage_projection_high_dimension_train_only_serialization(self):
        X = self.X.copy()
        X[:, -2:] = 0
        model = ShrinkageDiscriminantProjection().fit(X, self.y)
        projected = model.transform(X)
        self.assertEqual(projected.shape, (len(X), 6))
        self.assertTrue(np.isfinite(projected).all())
        np.testing.assert_allclose(model.scaler_.mean_, X.mean(0))
        before = model.scaler_.mean_.copy()
        model.transform(X[:5] + 1e6)
        np.testing.assert_array_equal(model.scaler_.mean_, before)
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).transform(X), projected)
        model.set_params(n_components=2)
        with self.assertRaisesRegex(ValueError, 'changed after fitting'):
            model.transform(X)
        with self.assertRaises(ValueError):
            ShrinkageDiscriminantProjection(n_components=7).fit(X, self.y)
        with self.assertRaises(ValueError):
            ShrinkageDiscriminantProjection().fit(np.zeros_like(X), self.y)


if __name__ == '__main__':
    unittest.main()
