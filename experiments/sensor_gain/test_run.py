import unittest
import numpy as np
from experiments.sensor_gain.run import RANGES, gains, calibrate, partition_indices


class GainContract(unittest.TestCase):
    def test_midpoints_equalize_and_input_unchanged(self):
        x = np.stack([np.zeros(16), RANGES.mean(1)])
        before = x.copy(); y = calibrate(x, gains())
        np.testing.assert_array_equal(x, before)
        np.testing.assert_allclose(y[0], 0)
        np.testing.assert_allclose(y[1], 3.05)
        self.assertAlmostEqual(gains()[0], 3.05 / 1.85)
        self.assertAlmostEqual(gains()[2], 3.05 / 3.6)

    def test_invalid_gain_or_mapping_rejected(self):
        for g in ([1] * 15, [0] * 16, [float('nan')] * 16):
            with self.assertRaises(ValueError): calibrate(np.ones((2, 16)), g)

    def test_subject_and_record_leakage_rejected(self):
        d = {"groups": np.array(['A'] * 7 + ['B'] * 7), "y": np.tile(np.arange(7), 2),
             "record_ids": np.array([str(i) for i in range(14)])}
        p = {"train": ['A'], "validation": ['B'], "test": ['C']}
        tr, te = partition_indices(d, p)
        self.assertEqual(len(tr), 7); self.assertEqual(len(te), 7)
        with self.assertRaises(ValueError): partition_indices(d, {**p, "validation": ['A']})
        d['record_ids'][7] = '0'
        with self.assertRaises(ValueError): partition_indices(d, p)


if __name__ == '__main__': unittest.main()
