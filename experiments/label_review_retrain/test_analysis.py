"""Known-answer tests for the label-review retrain helpers.

Run: .venv/bin/python -B -m unittest experiments.label_review_retrain.test_analysis -v
"""
from __future__ import annotations

import unittest

import numpy as np

from experiments.label_review_retrain import analysis as A
from experiments.runtime_vs_research.analysis import paired_subject_bootstrap as reference_bootstrap


def decision(kind, normal=(), abnormal=()):
    return {"decision": kind, "normal_intervals": [list(v) for v in normal],
            "abnormal_intervals": [list(v) for v in abnormal]}


STARTS = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]


def valid_starts(kind, normal=(), abnormal=()):
    return [s for s in STARTS if A.window_is_valid(s, kind, [list(v) for v in normal], [list(v) for v in abnormal])]


class WindowValidityTests(unittest.TestCase):
    def test_delete_without_normal_segment_drops_everything(self):
        self.assertEqual(valid_starts("delete"), [])

    def test_hold_and_mismatch_keep_everything(self):
        self.assertEqual(valid_starts("hold"), STARTS)
        self.assertEqual(valid_starts("posture_mismatch"), STARTS)

    def test_normal_interval_requires_full_containment(self):
        self.assertEqual(valid_starts("delete", normal=[(0.0, 7.5)]), [0.0, 3.0])  # LR001
        self.assertEqual(valid_starts("delete", normal=[(7.5, 13.0)]), [9.0])  # LR047
        self.assertEqual(valid_starts("delete", normal=[(0.0, 10.0)]), [0.0, 3.0, 6.0])  # LR079
        self.assertEqual(valid_starts("delete", normal=[(15.0, None)]), [15.0, 18.0])  # LR063
        self.assertEqual(valid_starts("hold", normal=[(5.0, None)]), [6.0, 9.0, 12.0, 15.0, 18.0])  # LR048
        self.assertEqual(valid_starts("hold", normal=[(7.0, None)]), [9.0, 12.0, 15.0, 18.0])  # LR065

    def test_abnormal_interval_drops_overlapping_windows(self):
        self.assertEqual(valid_starts("hold", abnormal=[(0.0, 3.0)]), STARTS[1:])  # LR002: touching edge kept
        self.assertEqual(valid_starts("hold", abnormal=[(0.0, 2.5), (15.0, 18.0)]),
                         [3.0, 6.0, 9.0, 12.0, 18.0])  # LR003
        self.assertEqual(valid_starts("posture_mismatch", abnormal=[(0.0, 5.0)]), STARTS[2:])  # LR005

    def test_unknown_decision_rejected(self):
        with self.assertRaises(ValueError):
            A.window_is_valid(0.0, "relabel", [], [])


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.records = np.array(["a", "a", "b", "b", "c", "c", "d"])
        self.starts = np.array([0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0])
        self.decisions = {"a": decision("delete", normal=[(3.0, None)]), "b": decision("hold", abnormal=[(0, 3)]),
                          "c": decision("posture_mismatch")}
        self.masks = A.review_masks(self.records, self.starts, self.decisions)

    def test_review_masks(self):
        m = self.masks
        self.assertEqual(m["reviewed"].tolist(), [1, 1, 1, 1, 1, 1, 0])
        self.assertEqual(m["delete_record"].tolist(), [1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(m["mismatch_record"].tolist(), [0, 0, 0, 0, 1, 1, 0])
        self.assertEqual(m["invalid_window"].tolist(), [1, 0, 1, 0, 0, 0, 0])

    def test_condition_and_evaluation_masks(self):
        fit = A.condition_fit_masks(self.masks)
        self.assertEqual(set(fit), set(A.CONDITIONS))
        self.assertEqual(fit["baseline"].tolist(), [1] * 7)
        self.assertEqual(fit["drop_delete"].tolist(), [0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(fit["drop_invalid_segments"].tolist(), [0, 1, 0, 1, 1, 1, 1])
        self.assertEqual(fit["drop_delete_mismatch"].tolist(), [0, 0, 1, 1, 0, 0, 1])
        ev = A.evaluation_masks(self.masks)
        self.assertEqual(ev["valid"].tolist(), [0, 1, 0, 1, 1, 1, 1])
        self.assertEqual(ev["valid_no_mismatch"].tolist(), [0, 1, 0, 1, 0, 0, 1])
        self.assertEqual(ev["unreviewed"].tolist(), [0, 0, 0, 0, 0, 0, 1])


class MetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.truth = rng.integers(0, 7, 400)
        self.a = np.where(rng.random(400) < .6, self.truth, rng.integers(0, 7, 400))
        self.b = np.where(rng.random(400) < .7, self.truth, rng.integers(0, 7, 400))
        self.groups = np.array([f"s{i % 13}" for i in range(400)])

    def test_confusion_macro_f1_matches_sklearn(self):
        _, matrices = A.confusion_by_group(self.truth, self.a, self.groups)
        self.assertAlmostEqual(float(A.macro_f1_from_confusion(matrices.sum(0))), A.macro_f1(self.truth, self.a), 12)
        # absent labels count as 0 (sklearn labels=range(7), zero_division=0)
        self.assertAlmostEqual(float(A.macro_f1_from_confusion(np.diag([2, 0, 0, 0, 0, 0, 0]))), 1 / 7, 12)

    def test_bootstrap_matches_reference_implementation(self):
        fast = A.paired_subject_bootstrap(self.truth, self.a, self.b, self.groups, replicates=300, seed=7)
        slow = reference_bootstrap(self.truth, self.a, self.b, self.groups, replicates=300, seed=7)
        for key in ("observed", "lower_2_5", "upper_97_5"):
            self.assertAlmostEqual(fast[key], slow[key], 12)

    def test_present_macro_f1(self):
        truth, pred = np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
        # class 0: P=1 R=.5 F=2/3; class 1: P=2/3 R=1 F=.8
        self.assertAlmostEqual(A.present_macro_f1(truth, pred), (2 / 3 + .8) / 2, 12)

    def test_smoothing_restarts_per_session(self):
        p = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [1, 0]], float)
        # session a: [1,0]->0, [.5,.5]->0 (tie), [1/3,2/3]->1; session b restarts: [0,1]->1, [.5,.5]->0
        self.assertEqual(A.smoothed_by_session(p, np.array(list("aaabb"))).tolist(), [0, 0, 1, 1, 0])
        with self.assertRaises(ValueError):
            A.smoothed_by_session(p, np.array(list("ababa")))

    def test_record_accuracy(self):
        p = np.eye(7)[[0, 1, 1, 2]]
        self.assertEqual(A.record_accuracy(p, np.array([0, 0, 1, 1]), np.array(["x", "x", "y", "y"])),
                         {"x": .5, "y": .5})


class DecisionTests(unittest.TestCase):
    criteria = {"minimum_mean_delta": .01, "maximum_per_class_drop": .05,
                "non_inferiority_mean_margin": .005, "non_inferiority_bootstrap_margin": .015}
    base_class = np.full((3, 7), .6)

    def rule(self, arm, lower, shift=0.0):
        return A.decide([.6, .6, .6], arm, self.base_class, self.base_class + shift, lower, self.criteria)

    def test_pass(self):
        result = self.rule([.615, .612, .62], .001)
        self.assertTrue(result["performance_pass"])
        self.assertTrue(result["non_inferior"])

    def test_each_performance_check_can_fail(self):
        self.assertFalse(self.rule([.605, .605, .605], .001)["performance_pass"])  # mean < +1%p
        self.assertFalse(self.rule([.64, .64, .599], .001)["performance_pass"])  # one seed negative
        self.assertFalse(self.rule([.62, .62, .62], -.001)["performance_pass"])  # bootstrap
        self.assertFalse(self.rule([.62, .62, .62], .001, shift=-.06)["performance_pass"])  # class drop

    def test_non_inferiority(self):
        self.assertTrue(self.rule([.596, .596, .596], -.014)["non_inferior"])
        self.assertFalse(self.rule([.594, .594, .594], -.01)["non_inferior"])
        self.assertFalse(self.rule([.6, .6, .6], -.016)["non_inferior"])


if __name__ == "__main__":
    unittest.main()
