"""Known-answer tests for the despike study decision rules and live metrics.

Run: .venv/bin/python -m unittest experiments.despike_study.test_analysis
"""
from __future__ import annotations

import unittest

import numpy as np

from experiments.despike_study.analysis import decide, label_changes, live_stability, smoothed_predictions
from experiments.despike_study.run import CRITERIA


class SmoothingTests(unittest.TestCase):
    def test_running_mean_of_last_three(self):
        p = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0]], float)
        # means: [1,0], [.5,.5]->0 (tie, first), [1/3,2/3]->1, [1/3,2/3]->1, [2/3,1/3]->0
        self.assertEqual(smoothed_predictions(p).tolist(), [0, 0, 1, 1, 0])
        self.assertEqual(smoothed_predictions(p, 1).tolist(), [0, 1, 1, 0, 0])

    def test_changes_ignore_session_boundaries(self):
        self.assertEqual(label_changes(np.array([0, 1, 1, 2, 2]), np.array([0, 0, 0, 1, 1])), 1)

    def test_live_stability_resets_history_per_session(self):
        p = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], float)
        truth = np.array([0, 0, 1, 1])
        result = live_stability(p, truth, np.array(["a", "a", "b", "b"]))
        self.assertEqual(result["smoothed_changes"], 0)
        self.assertEqual(result["raw_changes"], 0)
        self.assertAlmostEqual(result["prediction_minutes"], 4 / 60)
        with self.assertRaises(ValueError):
            live_stability(p, truth, np.array(["a", "b", "a", "b"]))


class DecisionTests(unittest.TestCase):
    base_class = np.full((3, 7), .6)

    def run_rule(self, arm_f1, lower, arm_changes, class_shift=0.0):
        arm_class = self.base_class + class_shift
        return decide([.65, .66, .64], arm_f1, self.base_class, arm_class, lower,
                      [4.0, 4.0, 4.0], arm_changes, CRITERIA)

    def test_clear_improvement_passes_performance(self):
        result = self.run_rule([.67, .675, .655], .002, [4.0, 4.0, 4.0])
        self.assertTrue(result["performance_pass"])
        self.assertFalse(result["stability_pass"])

    def test_small_gain_fails_performance_but_can_pass_stability(self):
        result = self.run_rule([.652, .661, .641], -.008, [3.0, 3.2, 3.4])
        self.assertFalse(result["performance_pass"])
        self.assertTrue(result["stability_pass"])

    def test_stability_requires_non_inferiority_and_consistent_reduction(self):
        self.assertFalse(self.run_rule([.64, .65, .63], -.02, [3.0, 3.0, 3.0])["stability_pass"])
        self.assertFalse(self.run_rule([.65, .66, .64], 0., [3.0, 4.1, 3.0])["stability_pass"])
        self.assertFalse(self.run_rule([.65, .66, .64], 0., [3.8, 3.8, 3.8])["stability_pass"])

    def test_per_class_drop_blocks_both(self):
        arm_class = self.base_class.copy()
        arm_class[:, 2] -= .06
        result = decide([.65, .66, .64], [.67, .68, .66], self.base_class, arm_class, .01,
                        [4, 4, 4], [3, 3, 3], CRITERIA)
        self.assertFalse(result["performance_pass"])
        self.assertFalse(result["stability_pass"])


if __name__ == "__main__":
    unittest.main()
