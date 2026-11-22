"""Synthetic-only numerical feature rules and independent window inference."""
from copy import deepcopy
import hashlib
import io
from itertools import combinations
import json
import unittest
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd

from src import interaction_features_v1 as subject
from src.posture_core import SENSOR_COLUMNS, feature_names
from src.record_gain_features_v1 import sequence_features


class InteractionFeatureTests(unittest.TestCase):
    def setUp(self):
        self.sequences = np.random.default_rng(551).normal(10, 2, (5, 36, 16)).astype(np.float32)

    def test_identity_and_hadamard_extraction_exactly_preserve_original596(self):
        original = sequence_features(self.sequences)
        before = self.sequences.tobytes()
        for mode in subject.MODES:
            actual = subject.extract_interaction_features(self.sequences, mode)
            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(actual[:, :596].tobytes(), original.tobytes())
            self.assertEqual(actual.shape, (5, 716 if mode == "temporal_corr120" else 596))
            self.assertFalse(np.shares_memory(actual, self.sequences))
            self.assertEqual(self.sequences.tobytes(), before)

    def test_known_correlations_constants_pair_sign_and_full_pair_order(self):
        values = np.zeros((1, 36, 16), np.float32)
        time = np.arange(36, dtype=np.float32) - 17.5
        values[0, :, 0], values[0, :, 1], values[0, :, 2] = time, 2 * time + 9, -3 * time + 7
        values[0, :, 3] = 4
        values[0, :, 4] = time ** 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = subject.extract_interaction_features(values, "temporal_corr120")
        pairs = list(combinations(range(16), 2))
        correlations = result[0, 596:]
        self.assertEqual(correlations[pairs.index((0, 1))], 1.)
        self.assertEqual(correlations[pairs.index((0, 2))], -1.)
        self.assertEqual(correlations[pairs.index((1, 2))], -1.)
        self.assertAlmostEqual(float(correlations[pairs.index((0, 4))]), 0., delta=1e-15)
        for pair, value in zip(pairs, correlations):
            if pair[0] == 3 or pair[1] == 3 or pair[0] >= 5 or pair[1] >= 5:
                self.assertEqual(value, 0.)
        names = subject.interaction_feature_names("temporal_corr120").tolist()
        self.assertEqual(names[:596], feature_names())
        self.assertEqual(names[596:], [f"raw_temporal_corr_{SENSOR_COLUMNS[i]}_vs_{SENSOR_COLUMNS[j]}" for i,j in pairs])
        expected = np.corrcoef(self.sequences[0].astype(np.float64), rowvar=False)
        actual = subject.extract_interaction_features(self.sequences[:1], "temporal_corr120")[0, 596:]
        np.testing.assert_allclose(actual, np.asarray([expected[i,j] for i,j in pairs]), atol=2e-8, rtol=1e-7)

    def test_correlation_feature_gain_offset_and_time_permutation_invariance(self):
        # Exact integer arithmetic makes the transformed inputs representable.
        values = np.random.default_rng(752).integers(-8, 9, (2, 36, 16)).astype(np.float32)
        gain = np.arange(1, 17, dtype=np.float32)
        changed = values * gain + np.arange(16, dtype=np.float32) * 4
        original = subject.extract_interaction_features(values, "temporal_corr120")[:, 596:]
        adjusted = subject.extract_interaction_features(changed, "temporal_corr120")[:, 596:]
        np.testing.assert_allclose(adjusted, original, atol=1e-7)
        reversed_time = subject.extract_interaction_features(values[:, ::-1], "temporal_corr120")[:, 596:]
        np.testing.assert_allclose(reversed_time, original, atol=1e-7)

    def test_hadamard_exact_blocks_orthogonality_approximate_inverse_and_untouched84(self):
        matrix = subject._hadamard16()
        np.testing.assert_array_equal(matrix @ matrix.T, np.eye(16))
        values = np.random.default_rng(154).normal(size=(3, 596)).astype(np.float32)
        values[0, 256], values[0, 554] = -0., +0.
        before = values.tobytes()
        result = subject.transform_standardized_features(values, "hadamard512")
        rule = subject.interaction_rule("hadamard512")
        self.assertEqual(len(rule["hadamard"]["block_order"]), 32)
        self.assertEqual(len(rule["hadamard"]["unchanged_indices"]), 84)
        for block in rule["hadamard"]["block_order"]:
            columns = slice(block["start"], block["stop"])
            expected = (values[:, columns].astype(np.float64) @ matrix.T).astype(np.float32)
            self.assertEqual(result[:, columns].tobytes(), expected.tobytes())
        indices = rule["hadamard"]["unchanged_indices"]
        self.assertEqual(indices, list(range(256, 298)) + list(range(554, 596)))
        self.assertEqual(result[:, indices].tobytes(), values[:, indices].tobytes())
        restored = subject.transform_standardized_features(result, "hadamard512")
        np.testing.assert_allclose(restored, values, atol=2e-7, rtol=2e-7)
        self.assertEqual(values.tobytes(), before)
        self.assertFalse(np.shares_memory(result, values))
        # An impulse proves the sensor/block and transformed-name mapping.
        impulse = np.zeros((1, 596), np.float32)
        impulse[0, 298 + 16 + 3] = 4
        result = subject.transform_standardized_features(impulse, "hadamard512")
        np.testing.assert_array_equal(result[0, 314:330], 4 * matrix[:, 3])
        self.assertEqual(subject.interaction_feature_names("hadamard512")[314], "rel_std_hadamard_00")
        self.assertEqual(np.count_nonzero(result), 16)

    def test_other_standardized_modes_are_exact_noops_including_signed_zero(self):
        for mode, width in (("identity", 596), ("temporal_corr120", 716)):
            values = np.random.default_rng(61).normal(size=(2, width)).astype(np.float32)
            values[0, :2] = [-0., +0.]
            before = values.tobytes()
            result = subject.transform_standardized_features(values, mode)
            self.assertEqual(result.tobytes(), before)
            self.assertEqual(values.tobytes(), before)
            self.assertFalse(np.shares_memory(result, values))

    def test_window_row_batch_order_and_serialization_consistency_without_adaptation(self):
        for mode in subject.MODES:
            expected = subject.extract_interaction_features(self.sequences, mode)
            chunks = np.concatenate([subject.extract_interaction_features(self.sequences[:1], mode),
                                     subject.extract_interaction_features(self.sequences[1:], mode)])
            self.assertEqual(chunks.tobytes(), expected.tobytes())
            np.testing.assert_array_equal(subject.extract_interaction_features(self.sequences[::-1], mode), expected[::-1])
            transformed = subject.transform_standardized_features(expected, mode)
            chunks = np.concatenate([subject.transform_standardized_features(expected[:1], mode),
                                     subject.transform_standardized_features(expected[1:], mode)])
            self.assertEqual(chunks.tobytes(), transformed.tobytes())
            np.testing.assert_array_equal(subject.transform_standardized_features(expected[::-1], mode), transformed[::-1])
            changed = np.concatenate([self.sequences[:1], self.sequences[1:] * 11 - 99])
            self.assertEqual(subject.extract_interaction_features(changed, mode)[:1].tobytes(), expected[:1].tobytes())
            buffer = io.BytesIO()
            joblib.dump((mode, subject.interaction_rule(mode), expected), buffer)
            buffer.seek(0)
            restored_mode, rule, restored = joblib.load(buffer)
            self.assertEqual(rule, subject.interaction_rule(restored_mode))
            self.assertEqual(subject.transform_standardized_features(restored, restored_mode).tobytes(), transformed.tobytes())

    def test_rule_schema_names_hashes_and_return_values_cannot_change_future_rules(self):
        def digest(value):
            return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        for mode in subject.MODES:
            rule = subject.interaction_rule(mode)
            original = deepcopy(rule)
            check = dict(rule)
            own_hash = check.pop("rule_sha256")
            self.assertEqual(own_hash, digest(check))
            for key in ("sensor_order", "extraction_feature_names", "transformed_feature_names"):
                self.assertEqual(rule[key + "_sha256"], digest(rule[key]))
            self.assertEqual(subject.interaction_feature_names(mode).tolist(), rule["transformed_feature_names"])
            self.assertFalse(rule["physical_rotation_or_reflection_claim"])
            self.assertFalse(rule["population_or_label_fitting"])
            if mode == "hadamard512":
                self.assertEqual(rule["hadamard"]["matrix_sha256"], hashlib.sha256(np.asarray(rule["hadamard"]["matrix"], dtype="<f8").tobytes()).hexdigest())
                self.assertEqual(rule["hadamard"]["block_order_sha256"], digest(rule["hadamard"]["block_order"]))
            if mode == "temporal_corr120":
                self.assertEqual(rule["correlation"]["pair_order_sha256"], digest(rule["correlation"]["pair_indices"]))
            rule["sensor_order"][0] = "changed"
            rule["transformed_feature_names"][0] = "changed"
            self.assertEqual(subject.interaction_rule(mode), original)

    def test_invalid_modes_shapes_types_nonfinite_named_order_and_schema_fail(self):
        for mode in (None, [], "other"):
            for function in (subject.interaction_feature_names, subject.interaction_rule):
                with self.assertRaises(ValueError):
                    function(mode)
            with self.assertRaises(ValueError):
                subject.extract_interaction_features(self.sequences, mode)
        for values in (self.sequences.astype(np.float64), self.sequences[:0], self.sequences[:, :-1],
                np.full_like(self.sequences, np.nan), np.full_like(self.sequences, np.inf)):
            with self.assertRaises(ValueError):
                subject.extract_interaction_features(values, "identity")
        features = sequence_features(self.sequences)
        for mode in subject.MODES:
            width = 716 if mode == "temporal_corr120" else 596
            values = np.zeros((2, width), np.float32)
            for bad in (values.astype(np.float64), values[:0], values[:, :-1], values[0],
                    np.full_like(values, np.nan), np.full_like(values, np.inf)):
                with self.assertRaises(ValueError):
                    subject.transform_standardized_features(bad, mode)
            names = subject.interaction_rule(mode)["extraction_feature_names"]
            frame = pd.DataFrame(values, columns=names)
            np.testing.assert_array_equal(subject.transform_standardized_features(frame, mode),
                                          subject.transform_standardized_features(values, mode))
            with self.assertRaisesRegex(ValueError, "ordering"):
                subject.transform_standardized_features(frame.iloc[:, ::-1], mode)
        with patch.object(subject, "feature_names", return_value=feature_names()[::-1]):
            with self.assertRaisesRegex(ValueError, "ordering"):
                subject.interaction_rule("hadamard512")
        with patch.object(subject, "SENSOR_COLUMNS", SENSOR_COLUMNS[::-1]):
            with self.assertRaisesRegex(ValueError, "canonical"):
                subject.transform_standardized_features(features, "identity")
        overflowing = np.full((1, 596), np.finfo(np.float32).max, np.float32)
        before = overflowing.tobytes()
        with self.assertRaisesRegex(ValueError, "finite float32"):
            subject.transform_standardized_features(overflowing, "hadamard512")
        self.assertEqual(overflowing.tobytes(), before)


if __name__ == "__main__":
    unittest.main()
