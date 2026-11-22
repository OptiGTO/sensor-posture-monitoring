"""Unit tests for the spike-repair / start-trim study (experiment code only)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.spike_trim_study.signal_rules import (  # noqa: E402
    apply_repair, median3, repair_isolated_spikes, review_spike_mask, start_trim_keep)

STAGED = ROOT / "research/models/models/20260929_221617_db0a4f57/training_data"


def series(values, rate=12.0):
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    return np.arange(len(values)) / rate, values


class IsolatedRepairTest(unittest.TestCase):
    def test_single_spike_is_replaced_by_neighbour_mean(self):
        t, x = series([1.0, 1.0, 1.6, 1.02, 1.0])
        repaired, mask = repair_isolated_spikes(t, x)
        self.assertAlmostEqual(repaired[2, 0], 1.01)
        self.assertEqual(mask[:, 0].tolist(), [False, False, True, False, False])

    def test_negative_spike_repaired(self):
        t, x = series([2.0, 2.0, 1.5, 2.0, 2.0])
        repaired, mask = repair_isolated_spikes(t, x)
        self.assertAlmostEqual(repaired[2, 0], 2.0)
        self.assertTrue(mask[2, 0])

    def test_steps_ramps_small_changes_and_edges_are_kept(self):
        for values in ([1.0, 1.0, 1.5, 1.5, 1.5],      # level step
                       [1.0, 1.2, 1.4, 1.6, 1.8],      # ramp
                       [1.0, 1.0, 1.08, 1.0, 1.0],     # below 0.10 V
                       [1.6, 1.0, 1.0, 1.0, 1.6]):     # first/last sample
            t, x = series(values)
            repaired, mask = repair_isolated_spikes(t, x)
            np.testing.assert_array_equal(repaired, x)
            self.assertFalse(mask.any(), values)

    def test_two_sample_plateau_is_not_an_isolated_spike(self):
        t, x = series([1.0, 1.0, 1.6, 1.6, 1.0, 1.0])
        _, mask = repair_isolated_spikes(t, x)
        self.assertFalse(mask.any())

    def test_alternating_flicker_becomes_constant(self):
        t, x = series([1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0])
        repaired, _ = repair_isolated_spikes(t, x)
        np.testing.assert_allclose(repaired[:-1, 0], 1.0)

    def test_long_sampling_gap_blocks_repair(self):
        t = np.array([0.0, 0.08, 0.5, 0.58, 0.66])
        x = np.array([[1.0], [1.0], [1.6], [1.0], [1.0]])
        _, mask = repair_isolated_spikes(t, x)
        self.assertFalse(mask.any())

    def test_channels_are_independent_and_input_unchanged(self):
        t = np.arange(5) / 12
        x = np.array([[1.0, 2.0], [1.0, 2.0], [1.6, 2.0], [1.0, 2.0], [1.0, 2.0]])
        original = x.copy()
        repaired, mask = repair_isolated_spikes(t, x)
        self.assertEqual(mask.sum(axis=0).tolist(), [1, 0])
        np.testing.assert_array_equal(repaired[:, 1], x[:, 1])
        np.testing.assert_array_equal(x, original)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            repair_isolated_spikes(np.array([0.0, 0.0, 0.1]), np.ones((3, 1)))
        with self.assertRaises(ValueError):
            repair_isolated_spikes(np.arange(3.0), np.array([[1.0], [np.nan], [1.0]]))


class OtherRulesTest(unittest.TestCase):
    def test_median3_keeps_edges(self):
        t, x = series([5.0, 1.0, 9.0, 1.0, 5.0])
        repaired, mask = median3(t, x)
        self.assertEqual(repaired[:, 0].tolist(), [5.0, 5.0, 1.0, 5.0, 5.0])
        self.assertEqual(mask[:, 0].tolist(), [False, True, True, True, False])

    def test_identity_repair_copies(self):
        t, x = series([1.0, 2.0, 3.0])
        repaired, mask = apply_repair(None, t, x)
        np.testing.assert_array_equal(repaired, x)
        self.assertFalse(mask.any())
        with self.assertRaises(ValueError):
            apply_repair("unknown", t, x)

    def test_start_trim(self):
        t = np.array([10.0, 10.25, 10.5, 10.75, 11.0, 11.25])
        self.assertEqual(start_trim_keep(t, 0.5).tolist(), [False, False, True, True, True, True])
        self.assertTrue(start_trim_keep(t, 0.0).all())
        with self.assertRaises(ValueError):
            start_trim_keep(t, 1.2)
        with self.assertRaises(ValueError):
            start_trim_keep(t, -1)


@unittest.skipUnless(STAGED.exists(), "staged submission training data not available")
class ReviewRuleConsistencyTest(unittest.TestCase):
    def test_matches_data_review_counts_on_real_records(self):
        from src.data_review import inspect_record
        from src.posture_core import SENSOR_COLUMNS, read_sensor_csv
        paths = sorted(STAGED.rglob("*.csv"))[::60]
        self.assertGreater(len(paths), 5)
        for path in paths:
            record, _ = inspect_record(path, STAGED)
            frame = read_sensor_csv(path)
            t = frame["elapsed_s"].to_numpy(dtype=float)
            mask = review_spike_mask(frame[SENSOR_COLUMNS].to_numpy(dtype=float))
            rows = mask.any(axis=1)
            self.assertEqual(int(mask.sum()), record["spike_count"], path.name)
            self.assertEqual(int(rows[(t - t[0]) <= 0.5].sum()), record["initial_spike_count"], path.name)


class RunHelpersTest(unittest.TestCase):
    def test_holm_adjustment(self):
        from experiments.spike_trim_study.analyze import holm
        adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)   # monotone: max(0.06, 1 * 0.04)
        self.assertIsNone(adjusted["d"])

    def test_feature_groups_cover_all_features(self):
        from experiments.spike_trim_study.run import feature_group
        from src.posture_core import feature_names
        groups = {feature_group(name) for name in feature_names()}
        for expected in ("abs:max", "abs:min", "abs:range", "abs:std", "abs:kurtosis", "abs:spectral_entropy",
                         "rel:kurtosis", "abs:row_mean", "abs:derived_std"):
            self.assertIn(expected, groups)
        self.assertEqual(len(feature_names()), 596)

    def test_rolling_metrics_smoothing_and_flips(self):
        from experiments.spike_trim_study.run import rolling_metrics
        k = 7
        probability = np.full((5, k), 0.0)
        for row, label in enumerate([0, 0, 1, 1, 1]):
            probability[row, label] = 1.0
        meta = {"record_id": np.array(["R1"] * 5), "subject": np.array(["A"] * 5), "y": np.zeros(5, dtype=int),
                "end_s": np.arange(5, dtype=float) + 3, "has_spike": np.array([False, False, True, False, False]),
                "initial": np.array([True, False, False, False, False])}
        metrics, frame = rolling_metrics(probability, meta, np.arange(5))
        # Mean of the last 3: [0], [0], [0,0,1]->0, [0,1,1]->1, [1,1,1]->1
        self.assertEqual(frame.smooth_pred.tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(frame.raw_pred.tolist(), [0, 0, 1, 1, 1])
        summary = metrics["summary"]
        self.assertAlmostEqual(summary["flips_per_minute"], 1 / (4 / 60))
        self.assertAlmostEqual(summary["upright_false_alarm"], 0.4)
        self.assertAlmostEqual(summary["raw_accuracy_spike_events"], 0.0)
        self.assertAlmostEqual(summary["raw_change_rate_spike_events"], 1.0)

    @unittest.skipUnless(STAGED.exists(), "staged submission training data not available")
    def test_live_replay_matches_realtime_engine(self):
        """rolling_set must issue the same events and features as RealtimePosturePredictor."""
        from experiments.spike_trim_study.run import (HOP_SECONDS, MIN_TAIL_FRACTION, SMOOTHING, WINDOW_SECONDS,
                                                      load_records, rolling_set)
        from src.posture_core import CLASS_NAMES, extract_feature_vector
        from src.realtime import LiveConfig, RealtimePosturePredictor
        from src.window_pipeline import WindowSpec, segment_sensor_frame

        class Capture:
            labels = CLASS_NAMES

            def __init__(self):
                self.features = []

            def predict_frame(self, frame):
                spec = WindowSpec(window_seconds=WINDOW_SECONDS, min_tail_fraction=MIN_TAIL_FRACTION)
                windows, _ = segment_sensor_frame(frame, spec)
                self.features.append(np.stack([extract_feature_vector(w) for w in windows]).mean(axis=0))
                return np.full(len(CLASS_NAMES), 1 / len(CLASS_NAMES))

        subset = STAGED / sorted(p.name for p in STAGED.iterdir() if p.is_dir())[0]
        records = load_records(subset)[:3]
        live, _ = rolling_set(records, None)
        expected = []
        for record in records:
            capture = Capture()
            engine = RealtimePosturePredictor(capture, LiveConfig(
                window_seconds=WINDOW_SECONDS, minimum_window_seconds=WINDOW_SECONDS * MIN_TAIL_FRACTION,
                prediction_interval_seconds=HOP_SECONDS, smoothing_history=SMOOTHING))
            for timestamp, values in zip(record["time"], record["values"]):
                engine.push(timestamp, values)
            expected.extend(capture.features)
        np.testing.assert_allclose(live["X"], np.stack(expected), rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
