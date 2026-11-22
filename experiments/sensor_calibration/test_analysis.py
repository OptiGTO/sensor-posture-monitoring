import unittest

import numpy as np
from sklearn.metrics import f1_score

from experiments.sensor_calibration import analysis as subject

CRITERIA = {"minimum_mean_delta": 0.01, "maximum_per_class_drop": 0.05}


class ScoreTests(unittest.TestCase):
    def test_macro_f1_matches_sklearn_including_absent_classes(self):
        rng = np.random.default_rng(4)
        for _ in range(20):
            truth = rng.integers(0, 7, 200)
            pred = rng.integers(0, 5, 200)  # classes 5, 6 never predicted
            ref = f1_score(truth, pred, labels=np.arange(7), average="macro", zero_division=0)
            self.assertAlmostEqual(subject.macro_f1(truth, pred), ref, places=12)
            np.testing.assert_allclose(subject.per_class_f1(truth, pred),
                                       f1_score(truth, pred, labels=np.arange(7), average=None, zero_division=0))

    def test_present_class_macro(self):
        truth = np.array([0, 0, 1, 1]); pred = np.array([0, 0, 1, 2])
        ref = f1_score(truth, pred, labels=[0, 1], average="macro", zero_division=0)
        self.assertAlmostEqual(subject.present_class_macro_f1(truth, pred), ref)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.repeat(np.arange(7), 8)
        self.groups = np.tile(list("abcdefgh"), 7)

    def test_identical_predictions_give_zero(self):
        p = np.stack([self.truth, self.truth])
        r = subject.seed_mean_bootstrap(self.truth, p, p, self.groups, replicates=300, seed=1, comparisons=6)
        self.assertEqual((r["observed"], r["lower_2_5"], r["bonferroni_upper"]), (0.0, 0.0, 0.0))

    def test_uniformly_better_has_positive_bounds_and_bonferroni_is_wider(self):
        rng = np.random.default_rng(0)
        worse = np.where(rng.random((3, len(self.truth))) < .5, (self.truth + 1) % 7, self.truth)
        better = np.where(rng.random((3, len(self.truth))) < .1, (self.truth + 1) % 7, self.truth)
        r = subject.seed_mean_bootstrap(self.truth, worse, better, self.groups, replicates=2000, seed=2, comparisons=6)
        self.assertGreater(r["lower_2_5"], 0)
        self.assertLessEqual(r["bonferroni_lower"], r["lower_2_5"])
        self.assertGreaterEqual(r["bonferroni_upper"], r["upper_97_5"])
        expected = np.mean([subject.macro_f1(self.truth, b) - subject.macro_f1(self.truth, a) for a, b in zip(worse, better)])
        self.assertAlmostEqual(r["observed"], expected)
        self.assertEqual((r["subjects"], r["seeds"]), (8, 3))

    def test_matches_explicit_resampling(self):
        rng = np.random.default_rng(5)
        a = rng.integers(0, 7, (2, len(self.truth))); b = rng.integers(0, 7, (2, len(self.truth)))
        r = subject.seed_mean_bootstrap(self.truth, a, b, self.groups, replicates=50, seed=9)
        subjects = np.unique(self.groups)
        weights = np.random.default_rng(9).multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), 50)
        deltas = []
        for w in weights:
            take = np.concatenate([np.repeat(np.flatnonzero(self.groups == s), k) for s, k in zip(subjects, w)])
            deltas.append(np.mean([subject.macro_f1(self.truth[take], pb[take]) - subject.macro_f1(self.truth[take], pa[take])
                                   for pa, pb in zip(a, b)]))
        self.assertAlmostEqual(r["lower_2_5"], float(np.percentile(deltas, 2.5)), places=12)

    def test_misaligned_rejected(self):
        with self.assertRaises(ValueError):
            subject.seed_mean_bootstrap([0, 1], [[0]], [[0, 1]], ["a", "b"], replicates=1, seed=0)


class DecisionTests(unittest.TestCase):
    boot_pos = {"bonferroni_lower": 0.002}
    boot_neg = {"bonferroni_lower": -0.001}

    def test_all_criteria_pass(self):
        d = subject.decide([.60, .61, .60], [.62, .625, .615], np.full((3, 7), .6), np.full((3, 7), .61),
                           self.boot_pos, CRITERIA)
        self.assertTrue(d["significant"])
        self.assertAlmostEqual(d["mean_delta"], 0.0166666, places=5)

    def test_each_criterion_can_fail(self):
        base = ([.60, .61, .60], [.62, .625, .615], np.full((3, 7), .6), np.full((3, 7), .61))
        self.assertFalse(subject.decide(*base, self.boot_neg, CRITERIA)["checks"]["bonferroni_lower_positive"])
        small = subject.decide([.60] * 3, [.605] * 3, *base[2:], self.boot_pos, CRITERIA)
        self.assertFalse(small["checks"]["mean_delta"])
        mixed = subject.decide([.60, .60, .60], [.64, .64, .599], *base[2:], self.boot_pos, CRITERIA)
        self.assertTrue(mixed["checks"]["mean_delta"]); self.assertFalse(mixed["checks"]["all_seeds_positive"])
        drop = np.full((3, 7), .61); drop[:, 2] = .54
        dropped = subject.decide(*base[:3], drop, self.boot_pos, CRITERIA)
        self.assertFalse(dropped["checks"]["per_class_drop"]); self.assertFalse(dropped["significant"])


if __name__ == "__main__":
    unittest.main()
