"""Synthetic spatial-order contracts; no project measurements or model fitting."""
from __future__ import annotations

import io
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.posture_core import SENSOR_COLUMNS, feature_names
from src.spatial_features_v1 import SpatialFeatureTransformer, spatial_feature_names


def features(means, medians=None, dtype=np.float32):
    means = np.atleast_2d(np.asarray(means, dtype=dtype))
    X = np.zeros((len(means), 596), dtype=dtype)
    names = feature_names()
    medians = means if medians is None else np.broadcast_to(medians, means.shape)
    for statistic, values in (("mean", means), ("median", medians)):
        X[:, [names.index(f"abs_{statistic}_{sensor}") for sensor in SENSOR_COLUMNS]] = values
    return X


class SpatialFeatureTests(unittest.TestCase):
    def test_canonical_pair_counts_names_signs_and_ties(self):
        X = features(np.stack((np.arange(16), np.repeat(4, 16))),
                     medians=np.stack((np.arange(16)[::-1], np.repeat(4, 16))))
        for mode, count in (("all_plus_mean_order", 716), ("all_plus_mean_median_order", 836)):
            transformer = SpatialFeatureTransformer(mode).fit(X)
            result = transformer.transform(X)
            self.assertEqual(result.shape, (2, count))
            self.assertEqual(transformer.n_features_out_, count)
            names = transformer.get_feature_names_out().tolist()
            self.assertEqual(names[:596], feature_names())
            self.assertEqual(names[596], "abs_mean_order_R11_vs_R12")
            self.assertEqual(names[610], "abs_mean_order_R11_vs_R44")
            self.assertEqual(names[611], "abs_mean_order_R12_vs_R13")
            self.assertEqual(names[715], "abs_mean_order_R43_vs_R44")
            self.assertEqual(len(set(names)), count)
            self.assertEqual(transformer.pair_indices_.shape, (120, 2))
            np.testing.assert_array_equal(result[0, 596:716], -np.ones(120))
            np.testing.assert_array_equal(result[1, 596:], np.zeros(count - 596))
            if count == 836:
                self.assertEqual(names[716], "abs_median_order_R11_vs_R12")
                self.assertEqual(names[-1], "abs_median_order_R43_vs_R44")
                np.testing.assert_array_equal(result[0, 716:], np.ones(120))

    def test_abs_rel_means_select_exact_sensor_columns_in_declared_order(self):
        X = np.tile(np.arange(596, dtype=np.float32), (2, 1))
        transformer = SpatialFeatureTransformer("abs_rel_sensor_means").fit(X)
        actual = transformer.transform(X)
        np.testing.assert_array_equal(actual, X[:, np.r_[0:16, 298:314]])
        self.assertEqual(actual.shape, (2, 32))
        self.assertEqual(transformer.get_feature_names_out().tolist(),
            [f"{prefix}_mean_R{row}{column}" for prefix in ("abs", "rel")
             for row in range(1, 5) for column in range(1, 5)])
        np.testing.assert_array_equal(transformer.selected_indices_, np.r_[0:16, 298:314])

    def test_append_preserves_float32_float64_original_bits_and_does_not_mutate(self):
        for dtype in (np.float32, np.float64):
            X = np.random.default_rng(57).normal(size=(3, 596)).astype(dtype)
            X[0, 100], X[1, 100] = -0., +0.
            before = X.tobytes()
            for mode in ("all_plus_mean_order", "all_plus_mean_median_order"):
                result = SpatialFeatureTransformer(mode).fit_transform(X)
                self.assertEqual(result.dtype, X.dtype)
                self.assertEqual(result[:, :596].tobytes(), before)
                self.assertEqual(X.tobytes(), before)
                self.assertFalse(np.shares_memory(result, X))
            means = SpatialFeatureTransformer("abs_rel_sensor_means").fit_transform(X)
            self.assertEqual(means.dtype, X.dtype)
            self.assertFalse(np.shares_memory(means, X))

    def test_order_is_invariant_to_representable_common_offset_and_positive_gain(self):
        means = np.arange(48, dtype=np.float32).reshape(3, 16) % 9
        medians = means[:, ::-1]
        X = features(means, medians)
        gains = np.array([.25, 2., 8.], dtype=np.float32)[:, None]
        offsets = np.array([-4., 32., 100.], dtype=np.float32)[:, None]
        changed = features(means * gains + offsets, medians * gains + offsets)
        transform = SpatialFeatureTransformer("all_plus_mean_median_order").fit(X)
        np.testing.assert_array_equal(transform.transform(X)[:, 596:], transform.transform(changed)[:, 596:])

    def test_extreme_finite_values_and_signed_zero_do_not_overflow_or_break_ties(self):
        for dtype in (np.float32, np.float64):
            row = np.zeros(16, dtype=dtype)
            row[:4] = [np.finfo(dtype).max, -np.finfo(dtype).max, -0., +0.]
            X = features(row, dtype=dtype)
            with np.errstate(all="raise"):
                result = SpatialFeatureTransformer().fit_transform(X)
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_array_equal(result[0, 596:599], [1., 1., 1.])
            names = spatial_feature_names("all_plus_mean_order")
            self.assertEqual(result[0, names.index("abs_mean_order_R13_vs_R14")], 0.)

    def test_fit_uses_no_population_statistics_labels_or_other_evaluation_rows(self):
        train = features(np.arange(48).reshape(3, 16))
        evaluation = features(np.arange(32).reshape(2, 16) % 7)
        for mode in ("all_plus_mean_order", "all_plus_mean_median_order", "abs_rel_sensor_means"):
            first = SpatialFeatureTransformer(mode).fit(train, y=object())
            second = SpatialFeatureTransformer(mode).fit(train * 23 + 999, y="unused")
            self.assertEqual(set(first.__dict__), set(second.__dict__))
            for key in first.__dict__:
                if isinstance(first.__dict__[key], np.ndarray):
                    np.testing.assert_array_equal(first.__dict__[key], second.__dict__[key])
                else:
                    self.assertEqual(first.__dict__[key], second.__dict__[key])
            np.testing.assert_array_equal(first.transform(evaluation), second.transform(evaluation))
            np.testing.assert_array_equal(first.transform(evaluation[:1]),
                first.transform(np.concatenate((evaluation[:1], train)))[:1])

    def test_clone_and_serialization_preserve_fitted_indices_names_and_output(self):
        X = features(np.arange(48).reshape(3, 16))
        for mode in ("all_plus_mean_order", "all_plus_mean_median_order", "abs_rel_sensor_means"):
            transformer = clone(SpatialFeatureTransformer(mode)).fit(X)
            output = transformer.transform(X)
            stream = io.BytesIO()
            joblib.dump(transformer, stream)
            stream.seek(0)
            restored = joblib.load(stream)
            self.assertEqual(restored.transform(X).tobytes(), output.tobytes())
            for key in ("mean_indices_", "median_indices_", "selected_indices_", "pair_indices_",
                        "feature_names_in_", "feature_names_out_"):
                np.testing.assert_array_equal(getattr(restored, key), getattr(transformer, key))
            names = restored.get_feature_names_out()
            names[0] = "modified-copy"
            self.assertNotEqual(restored.get_feature_names_out()[0], "modified-copy")

    def test_wrong_shapes_nonfinite_dtypes_and_unfitted_mode_drift_fail_closed(self):
        X = features(np.arange(16))
        with self.assertRaises(NotFittedError):
            SpatialFeatureTransformer().transform(X)
        for mode in ("unknown", None, []):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                SpatialFeatureTransformer(mode).fit(X)
        for values in (X[:, :-1], X[0], X[:0], X[:, None], np.full_like(X, np.nan),
                np.full_like(X, np.inf), X.astype(complex), X.astype(str), X.astype(bool)):
            with self.subTest(shape=values.shape, dtype=values.dtype), self.assertRaises(ValueError):
                SpatialFeatureTransformer().fit(values)
        transformer = SpatialFeatureTransformer().fit(X)
        with self.assertRaises(ValueError):
            transformer.transform(np.full_like(X, np.nan))
        transformer.set_params(mode="abs_rel_sensor_means")
        with self.assertRaisesRegex(ValueError, "changed after fitting"):
            transformer.transform(X)

    def test_named_column_permutation_or_renaming_is_rejected(self):
        X = features(np.arange(16))
        frame = pd.DataFrame(X, columns=feature_names())
        transformer = SpatialFeatureTransformer().fit(frame)
        np.testing.assert_array_equal(transformer.transform(frame), transformer.transform(X))
        for changed in (frame.iloc[:, ::-1], frame.rename(columns={"abs_mean_R11": "wrong"}), frame.iloc[:, :-1]):
            with self.subTest(columns=len(changed.columns)), self.assertRaises(ValueError):
                SpatialFeatureTransformer().fit(changed)
            with self.assertRaises(ValueError):
                transformer.transform(changed)
        with self.assertRaises(ValueError):
            transformer.get_feature_names_out(feature_names()[::-1])
        np.testing.assert_array_equal(transformer.get_feature_names_out(feature_names()), transformer.get_feature_names_out())


if __name__ == "__main__":
    unittest.main()
