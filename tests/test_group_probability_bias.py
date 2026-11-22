from __future__ import annotations

import io
import unittest

import joblib
import numpy as np
from sklearn.base import clone

from src.group_probability_bias import GroupProbabilityBias, LOG_BIAS_GRID


class GroupProbabilityBiasTests(unittest.TestCase):
    def setUp(self):
        self.y = np.tile(np.arange(7), 4)
        self.groups = np.repeat(np.arange(4), 7)
        self.p = np.full((len(self.y), 7), .04)
        self.p[np.arange(len(self.y)), self.y] = .76
        self.p[self.y == 0] = [.35, .4, .05, .05, .05, .05, .05]

    def test_bounded_deterministic_search_has_complete_objective_audit(self):
        a = GroupProbabilityBias().fit(self.p, self.y, groups=self.groups)
        b = clone(GroupProbabilityBias()).fit(self.p, self.y, groups=self.groups)
        np.testing.assert_array_equal(a.log_bias_, b.log_bias_)
        self.assertTrue(np.isin(a.log_bias_, LOG_BIAS_GRID).all())
        self.assertLessEqual(a.n_sweeps_, 3)
        self.assertGreater(a.objective_, a.initial_objective_)
        self.assertEqual(a.macro_f1_, 1.)
        self.assertEqual(len(a.training_details_["objective_audit"]), a.n_sweeps_ * 7)
        self.assertEqual(a.n_subjects_seen_, 4)
        self.assertAlmostEqual(a.objective_, a.macro_f1_ - .01 * np.mean(a.log_bias_ ** 2))

    def test_identity_when_predictions_are_correct_and_ties_prefer_zero_norm(self):
        perfect = np.eye(7)[self.y]
        model = GroupProbabilityBias().fit(perfect, self.y, groups=self.groups)
        np.testing.assert_array_equal(model.log_bias_, np.zeros(7))
        self.assertEqual(model.n_sweeps_, 1)
        result = model.transform(perfect)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result > 0).all())
        np.testing.assert_allclose(result.sum(1), 1.)
        np.testing.assert_array_equal(result.argmax(1), self.y)

    def test_transform_is_row_independent_serializable_and_does_not_mutate_input(self):
        original = self.p.copy()
        labels, groups = self.y.copy(), self.groups.copy()
        model = GroupProbabilityBias().fit(self.p, self.y, groups=self.groups)
        probability = model.transform(self.p)
        np.testing.assert_array_equal(probability[:3], model.transform(self.p[:3]))
        np.testing.assert_array_equal(self.p, original)
        np.testing.assert_array_equal(self.y, labels)
        np.testing.assert_array_equal(self.groups, groups)
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).transform(self.p), probability)
        before = model.log_bias_.copy()
        model.transform(np.full((10, 7), 1/7))
        np.testing.assert_array_equal(model.log_bias_, before)
        self.assertEqual(model.n_samples_seen_, len(self.p))
        model.set_params(regularization=.1)
        with self.assertRaisesRegex(ValueError, 'changed after fitting'):
            model.transform(self.p)

    def test_subject_renaming_does_not_change_objective_or_output(self):
        a = GroupProbabilityBias().fit(self.p, self.y, groups=self.groups)
        names = np.asarray(['d', 'a', 'c', 'b'])[self.groups]
        b = GroupProbabilityBias().fit(self.p, self.y, groups=names)
        np.testing.assert_array_equal(a.log_bias_, b.log_bias_)
        self.assertEqual(a.objective_, b.objective_)

    def test_rejects_invalid_probabilities_labels_groups_and_parameters(self):
        nan = self.p.copy(); nan[0, 0] = np.nan
        negative = self.p.copy(); negative[0, 0] = -.01
        for p in (nan, negative, self.p * 2, self.p[:, :-1], self.p[:0], self.p[0]):
            with self.assertRaises(ValueError):
                GroupProbabilityBias().fit(p, self.y, groups=self.groups)
        for y in (self.y[:-1], np.zeros_like(self.y), self.y + .5, self.y + 1):
            with self.assertRaises(ValueError):
                GroupProbabilityBias().fit(self.p, y, groups=self.groups)
        for groups in (None, self.groups[:-1], np.zeros(len(self.y)), np.full(len(self.y), np.nan)):
            with self.assertRaises(ValueError):
                GroupProbabilityBias().fit(self.p, self.y, groups=groups)
        for kwargs in ({'regularization': -1}, {'regularization': float('nan')}, {'regularization': True},
                {'max_sweeps': 0}, {'max_sweeps': 4}, {'max_sweeps': True}):
            with self.assertRaises(ValueError):
                GroupProbabilityBias(**kwargs).fit(self.p, self.y, groups=self.groups)


if __name__ == '__main__':
    unittest.main()
