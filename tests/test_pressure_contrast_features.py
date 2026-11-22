"""Pressure geometry, nuisance invariance and train-independent reuse checks."""
from __future__ import annotations

import io
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.posture_core import SENSOR_COLUMNS, feature_names
from src.pressure_contrast_features import PressureContrastFeatures, contrast_feature_names


def make_features(means, medians=None, std=None):
    means = np.atleast_2d(np.asarray(means, dtype=float))
    names = feature_names()
    values = np.zeros((len(means), len(names)), dtype=float)
    medians = means if medians is None else np.broadcast_to(medians, means.shape)
    std = np.ones_like(means) if std is None else np.broadcast_to(std, means.shape)
    for stat, data in (("mean", means), ("median", medians), ("std", std)):
        indices = [names.index(f"abs_{stat}_{sensor}") for sensor in SENSOR_COLUMNS]
        values[:, indices] = data
    return values


class PressureContrastFeaturesTests(unittest.TestCase):
    def test_known_single_sensor_pressure_has_correct_grid_geometry(self):
        pressure = np.zeros((2, 16))
        pressure[0, 0] = 8
        pressure[1, -1] = 3
        X = make_features(pressure, std=0)
        result = PressureContrastFeatures().fit_transform(X)
        self.assertEqual(result.shape, (2, 75))
        np.testing.assert_array_equal(result[:, 32:48], pressure / pressure.sum(axis=1, keepdims=True))
        np.testing.assert_array_equal(result[:, 64:68], [[1, 0, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_array_equal(result[:, 68:72], [[1, 0, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_array_equal(result[:, 72:], [[-1, -1, 1], [1, 1, 1]])
        np.testing.assert_allclose(result[:, :16].sum(axis=1), 0, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(result[:, :16], axis=1), 1)

    def test_positive_gain_and_offset_cancel_from_all_contrast_columns(self):
        rng = np.random.default_rng(907)
        mean = rng.uniform(2, 40, (6, 16))
        median = mean + rng.uniform(-1, 1, mean.shape)
        std = rng.uniform(.1, 2, mean.shape)
        X = make_features(mean, median, std)
        gains = np.array([.25, .5, 1, 2, 5, 8])[:, None]
        offsets = np.array([30, -1, 0, 120, -8, 11])[:, None]
        adjusted = make_features(mean * gains + offsets, median * gains + offsets, std * gains)
        transformer = PressureContrastFeatures().fit(X)
        np.testing.assert_allclose(transformer.transform(adjusted), transformer.transform(X), rtol=2e-13, atol=2e-13)

    def test_temporal_std_uses_centered_mean_l2_norm(self):
        means = np.arange(16, dtype=float)[None, :]
        std = np.linspace(0, 3, 16)[None, :]
        result = PressureContrastFeatures().fit_transform(make_features(means, std=std))
        expected = std / np.linalg.norm(means - means.mean(), axis=1, keepdims=True)
        np.testing.assert_allclose(result[:, 48:64], expected, rtol=1e-14)

    def test_constant_and_zero_pressure_have_finite_zero_contrast(self):
        X = make_features(np.repeat([[0.], [17.], [-4.]], 16, axis=1), std=10)
        result = PressureContrastFeatures().fit_transform(X)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result, np.zeros((3, 75)))

    def test_large_finite_inputs_do_not_overflow_shape_features(self):
        values = np.linspace(-1, 1, 16)[None, :] * 1e308
        X = make_features(values, std=1e308)
        result = PressureContrastFeatures().fit_transform(X)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[:, 32:48].sum(axis=1), 1)

    def test_append_mode_preserves_original_values_and_names(self):
        X = make_features(np.arange(32).reshape(2, 16))
        X[:, -17:] = np.arange(17)
        before = X.copy()
        transformer = PressureContrastFeatures(mode="all_plus_contrast").fit(X)
        result = transformer.transform(X)
        self.assertEqual(result.shape, (2, 671))
        np.testing.assert_array_equal(result[:, :596], X)
        np.testing.assert_array_equal(X, before)
        self.assertEqual(transformer.get_feature_names_out().tolist(), feature_names() + contrast_feature_names())
        self.assertEqual(len(set(transformer.get_feature_names_out())), 671)

    def test_fit_learns_no_population_statistics_or_labels(self):
        train = make_features(np.arange(48).reshape(3, 16))
        evaluation = make_features(np.arange(32).reshape(2, 16) ** 2, std=2)
        a = PressureContrastFeatures().fit(train, y=object())
        b = PressureContrastFeatures().fit(train * 1000 + 1e6, y="unused labels")
        np.testing.assert_array_equal(a.transform(evaluation), b.transform(evaluation))
        alone = a.transform(evaluation[:1])
        together = a.transform(np.vstack((evaluation[:1], train)))
        np.testing.assert_array_equal(alone, together[:1])
        self.assertFalse(hasattr(a, "mean_"))
        self.assertFalse(hasattr(a, "scale_"))

    def test_clone_and_joblib_roundtrip_keep_exact_output(self):
        X = make_features(np.arange(48).reshape(3, 16), std=2)
        transformer = clone(PressureContrastFeatures("contrast")).fit(X)
        expected = transformer.transform(X)
        stream = io.BytesIO()
        joblib.dump(transformer, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        np.testing.assert_array_equal(restored.transform(X), expected)
        np.testing.assert_array_equal(restored.get_feature_names_out(), transformer.get_feature_names_out())

    def test_names_shapes_nonfinite_and_unfitted_use_fail_closed(self):
        X = make_features(np.arange(16))
        with self.assertRaises(NotFittedError):
            PressureContrastFeatures().transform(X)
        for bad in (X[:, :-1], np.full_like(X, np.nan), np.full_like(X, np.inf)):
            with self.assertRaises(ValueError):
                PressureContrastFeatures().fit(bad)
        with self.assertRaises(ValueError):
            PressureContrastFeatures("unknown").fit(X)
        frame = pd.DataFrame(X, columns=feature_names())
        transformer = PressureContrastFeatures().fit(frame)
        np.testing.assert_array_equal(transformer.transform(frame), transformer.transform(X))
        with self.assertRaises(ValueError):
            transformer.transform(frame[frame.columns[::-1]])
        with self.assertRaises(ValueError):
            transformer.get_feature_names_out(feature_names()[::-1])
        transformer.set_params(mode="all_plus_contrast")
        with self.assertRaises(ValueError):
            transformer.transform(X)


if __name__ == "__main__":
    unittest.main()
