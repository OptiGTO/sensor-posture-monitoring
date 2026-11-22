"""Known-answer tests for the alert-policy post-hoc analysis helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.alert_policy.run import (  # noqa: E402
    RATES_COLUMNS, alert_score, bootstrap_difference, loso_thresholds, never_reset, pooled, smooth, subject_counts)


def one_hot(labels, k=7, value=1.0):
    p = np.zeros((len(labels), k))
    p[np.arange(len(labels)), labels] = value
    return p


class SmoothTest(unittest.TestCase):
    def test_running_mean_restarts_per_record(self):
        p = one_hot([0, 1, 1, 1, 0])
        out = smooth(p, np.array(["A", "A", "A", "B", "B"]), 3)
        np.testing.assert_allclose(out[2, :2], [1 / 3, 2 / 3])
        np.testing.assert_allclose(out[3, 1], 1.0)           # B restarts
        np.testing.assert_allclose(out[4, :2], [0.5, 0.5])

    def test_non_contiguous_records_rejected(self):
        with self.assertRaises(ValueError):
            smooth(one_hot([0, 0, 0]), np.array(["A", "B", "A"]), 3)


class ScoreTest(unittest.TestCase):
    def test_current_rule_matches_risk_assessment(self):
        s = np.array([[0.5, 0.3, 0.2, 0, 0, 0, 0],      # argmax upright -> 0
                      [0.2, 0.44, 0.36, 0, 0, 0, 0],    # below gate
                      [0.1, 0.46, 0.44, 0, 0, 0, 0]])
        score = alert_score(s, "current")
        np.testing.assert_allclose(score, [0.0, 0.44, 0.46])
        self.assertEqual((score >= 0.45).tolist(), [False, False, True])
        np.testing.assert_allclose(alert_score(s, "not_upright"), [0.5, 0.8, 0.9])
        with self.assertRaises(ValueError):
            alert_score(s, "other")


class NeverResetTest(unittest.TestCase):
    def test_app_rule_requires_every_event(self):
        self.assertTrue(never_reset([True, True, True]))
        self.assertFalse(never_reset([True, False, True]))
        self.assertFalse(never_reset([False, True, True]))
        self.assertFalse(never_reset([]))

    def test_tolerance(self):
        self.assertTrue(never_reset([True, False, False, True], tolerance=3))
        self.assertFalse(never_reset([True, False, False, False, True], tolerance=3))
        self.assertTrue(never_reset([True, True, False, False], tolerance=3))
        with self.assertRaises(ValueError):
            never_reset([True], tolerance=0)


class ThresholdTest(unittest.TestCase):
    def test_threshold_excludes_the_held_out_subject(self):
        score = np.array([0.1, 0.2, 0.9, 0.95, 0.3, 0.4])
        abnormal = np.array([True, True, True, True, True, True])
        subject = np.array(["a", "a", "b", "b", "c", "c"])
        t = loso_thresholds(score, abnormal, subject, 0.5)
        self.assertAlmostEqual(t[0], np.quantile([0.9, 0.95, 0.3, 0.4], 0.5))
        self.assertAlmostEqual(t[2], np.quantile([0.1, 0.2, 0.3, 0.4], 0.5))
        with self.assertRaises(ValueError):
            loso_thresholds(score, abnormal, subject, 1.0)


class CountsTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"record_id": ["U1"] * 3 + ["A1"] * 3 + ["U2"] * 2,
                             "subject": ["s1"] * 6 + ["s2"] * 2, "y": [0, 0, 0, 2, 2, 2, 0, 0]})

    def test_pooled_rates(self):
        poor = np.array([True, True, True, True, False, True, False, False])
        counts = subject_counts(self.frame(), poor, 1)
        self.assertEqual(list(counts.columns), list(RATES_COLUMNS))
        rates = pooled(counts)
        self.assertAlmostEqual(rates["upright_poor"], 3 / 5)
        self.assertAlmostEqual(rates["detection"], 2 / 3)
        self.assertAlmostEqual(rates["upright_record_never_reset"], 1 / 2)
        self.assertAlmostEqual(rates["abnormal_record_never_reset"], 0.0)
        self.assertAlmostEqual(pooled(subject_counts(self.frame(), poor, 2))["abnormal_record_never_reset"], 1.0)

    def test_bootstrap_identical_policies_have_zero_difference(self):
        poor = np.array([True, True, True, True, False, True, False, False])
        counts = subject_counts(self.frame(), poor, 1)
        result = bootstrap_difference(counts, counts, 200, 1)
        self.assertEqual(result["upright_poor"]["observed"], 0.0)
        self.assertEqual(result["upright_poor"]["ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
