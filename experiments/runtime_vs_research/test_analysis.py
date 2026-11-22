import unittest

import numpy as np

from experiments.runtime_vs_research.analysis import decide, factorial_effects, macro_f1, paired_subject_bootstrap

CRITERIA = {"minimum_development_mean_delta": 0.01, "minimum_positive_splits": 2,
            "maximum_per_class_drop": 0.05, "minimum_holdout_delta": 0.01}


class FactorialTests(unittest.TestCase):
    def test_known_effects(self):
        s = {("svm", "standard"): 0.60, ("svm", "ordinary"): 0.62, ("et", "standard"): 0.70, ("et", "ordinary"): 0.71}
        e = factorial_effects(s)
        self.assertAlmostEqual(e["model_effect"], 0.095)
        self.assertAlmostEqual(e["policy_effect"], 0.015)
        self.assertAlmostEqual(e["interaction"], -0.01)
        self.assertAlmostEqual(e["total_research_minus_runtime"], 0.11)


class BootstrapTests(unittest.TestCase):
    def test_identical_predictions_give_zero(self):
        truth = np.repeat(np.arange(7), 4)
        groups = np.tile(["a", "b", "c", "d"], 7)
        r = paired_subject_bootstrap(truth, truth, truth, groups, replicates=200, seed=1)
        self.assertEqual((r["observed"], r["lower_2_5"], r["upper_97_5"]), (0.0, 0.0, 0.0))

    def test_uniformly_better_b_has_positive_interval(self):
        truth = np.repeat(np.arange(7), 4)
        groups = np.tile(["a", "b", "c", "d"], 7)
        worse = (truth + 1) % 7
        r = paired_subject_bootstrap(truth, worse, truth, groups, replicates=200, seed=1)
        self.assertEqual(r["observed"], 1.0)
        self.assertGreater(r["lower_2_5"], 0)
        self.assertEqual(r["subjects"], 4)

    def test_misaligned_inputs_rejected(self):
        with self.assertRaises(ValueError):
            paired_subject_bootstrap([0, 1], [0], [0, 1], ["a", "b"], replicates=1, seed=0)

    def test_macro_f1_counts_absent_classes_as_zero(self):
        self.assertAlmostEqual(macro_f1([0, 1], [0, 1]), 2 / 7)


class DecisionTests(unittest.TestCase):
    def _call(self, dev_delta, hold_delta, lower, class_drop=0.0):
        a = np.full((3, 3), 0.6)
        b = a + np.asarray(dev_delta)
        ca = np.full((3, 3, 7), 0.6)
        cb = ca.copy()
        cb[..., 0] -= class_drop
        ha, hb = np.full(3, 0.6), np.full(3, 0.6) + hold_delta
        hca, hcb = np.full((3, 7), 0.6), np.full((3, 7), 0.6)
        return decide(a, b, ca, cb, ha, hb, hca, hcb, lower, CRITERIA)

    def test_all_pass(self):
        self.assertTrue(self._call(np.full((3, 3), 0.02), 0.02, 0.001)["replace"])

    def test_holdout_failure_blocks(self):
        r = self._call(np.full((3, 3), 0.02), -0.01, -0.03)
        self.assertFalse(r["replace"])
        self.assertFalse(r["checks"]["holdout_mean_delta"])
        self.assertTrue(r["checks"]["development_mean_delta"])

    def test_only_one_positive_split_blocks(self):
        d = np.tile([0.06, -0.001, -0.001], (3, 1))
        r = self._call(d, 0.02, 0.01)
        self.assertFalse(r["checks"]["development_positive_splits"])

    def test_class_drop_blocks(self):
        r = self._call(np.full((3, 3), 0.02), 0.02, 0.01, class_drop=0.06)
        self.assertFalse(r["checks"]["development_per_class_drop"])
        self.assertFalse(r["replace"])


if __name__ == "__main__":
    unittest.main()
