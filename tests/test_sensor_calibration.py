"""Contract of the finger-press sensor calibration transform (synthetic + recorded file)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import sensor_calibration as subject
from src.posture_core import SENSOR_COLUMNS

# Gains printed in docs/reports/2026-09-23_sensor_gain_2s3s_report.md (3.05 / midpoint).
HISTORICAL_GAINS = [1.648649, 1.089286, 0.847222, 0.953125, 0.953125, 0.924242, 1.033898, 0.871429,
                    1.016667, 1.089286, 0.968254, 0.938462, 1.051724, 1.109091, 1.016667, 0.983871]


class SensorCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.measurement, self.digest = subject.load_measurement(subject.DEFAULT_MEASUREMENT_PATH)

    def test_recorded_file_matches_user_table(self):
        mids = subject.representative_values(self.measurement)
        expected = [1.85, 2.80, 3.60, 3.20, 3.20, 3.30, 2.95, 3.50,
                    3.00, 2.80, 3.15, 3.25, 2.90, 2.75, 3.00, 3.10]
        np.testing.assert_allclose(mids, expected, atol=1e-12)
        self.assertEqual(len(self.digest), 64)

    def test_gain_mode_reproduces_2026_09_23_study(self):
        calibration = subject.build_calibration(self.measurement, "gain", self.digest)
        np.testing.assert_allclose(calibration.gains, HISTORICAL_GAINS, atol=5e-7)
        self.assertAlmostEqual(calibration.gains[0], 3.05 / 1.85)
        np.testing.assert_array_equal(calibration.offsets, np.zeros(16))

    def test_offset_gain_keeps_empty_seat_and_equalizes_pressed_value(self):
        calibration = subject.build_calibration(self.measurement, "offset_gain", self.digest)
        baseline = subject.empty_seat_baseline(self.measurement)
        pressed = subject.representative_values(self.measurement)
        np.testing.assert_allclose(calibration.apply(baseline), baseline, atol=1e-6)
        out = calibration.apply(pressed)
        target = baseline + np.median(pressed - baseline)
        np.testing.assert_allclose(out, target, atol=1e-6)

    def test_apply_is_elementwise_on_last_axis_and_does_not_mutate(self):
        calibration = subject.build_calibration(self.measurement, "offset_gain")
        x = np.random.default_rng(3).uniform(1.0, 3.0, (4, 36, 16)).astype(np.float32)
        before = x.copy()
        y = calibration.apply(x)
        np.testing.assert_array_equal(x, before)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float32)
        c = 5
        manual = calibration.offsets[c] + (x[..., c].astype(np.float64) - calibration.offsets[c]) * calibration.gains[c]
        np.testing.assert_allclose(y[..., c], manual.astype(np.float32))

    def test_identity_is_exact(self):
        x = np.random.default_rng(1).normal(size=(10, 16)).astype(np.float32)
        np.testing.assert_array_equal(subject.identity_calibration().apply(x), x)
        np.testing.assert_array_equal(subject.load_calibration("none").apply(x), x)

    def test_round_trip_dict(self):
        calibration = subject.build_calibration(self.measurement, "offset_gain", self.digest)
        restored = subject.SensorCalibration.from_dict(json.loads(json.dumps(calibration.to_dict())))
        self.assertEqual(restored, calibration)
        bad = {**calibration.to_dict(), "sensor_columns": SENSOR_COLUMNS[::-1]}
        with self.assertRaises(ValueError):
            subject.SensorCalibration.from_dict(bad)

    def test_readings_override_range_for_updated_measurements(self):
        updated = copy.deepcopy(self.measurement)
        updated["sensors"]["R11"] = {"readings": [1.9, 2.0, 2.1]}
        self.assertAlmostEqual(subject.representative_values(updated)[0], 2.0)

    def test_invalid_measurements_rejected(self):
        cases = []
        missing = copy.deepcopy(self.measurement); missing["sensors"].pop("R44"); cases.append(missing)
        swapped = copy.deepcopy(self.measurement); swapped["sensors"]["R12"] = {"min": 3.0, "max": 2.0}; cases.append(swapped)
        nan = copy.deepcopy(self.measurement); nan["sensors"]["R13"]["min"] = float("nan"); cases.append(nan)
        for case in cases:
            with self.assertRaises(ValueError):
                subject.build_calibration(case, "gain")
        below = copy.deepcopy(self.measurement); below["sensors"]["R11"] = {"min": 1.0, "max": 1.1}
        subject.build_calibration(below, "gain")
        with self.assertRaises(ValueError):
            subject.build_calibration(below, "offset_gain")
        with self.assertRaises(ValueError):
            subject.build_calibration(self.measurement, "clip")
        with self.assertRaises(ValueError):
            subject.identity_calibration().apply(np.ones((3, 15)))

    def test_load_rejects_wrong_schema_and_hash_tracks_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps({**self.measurement, "schema_version": 2}), encoding="utf-8")
            with self.assertRaises(ValueError):
                subject.load_measurement(path)
            path.write_text(json.dumps(self.measurement), encoding="utf-8")
            _, first = subject.load_measurement(path)
            changed = copy.deepcopy(self.measurement); changed["sensors"]["R11"]["max"] = 2.1
            path.write_text(json.dumps(changed), encoding="utf-8")
            _, second = subject.load_measurement(path)
            self.assertNotEqual(first, second)
            self.assertNotEqual(subject.load_calibration("gain", path).gains[0],
                                subject.build_calibration(self.measurement, "gain").gains[0])


if __name__ == "__main__":
    unittest.main()
