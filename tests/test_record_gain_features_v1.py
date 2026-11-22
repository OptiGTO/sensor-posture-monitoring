"""Synthetic-only gain determinism, record purity and canonical feature parity."""
from copy import deepcopy
import hashlib
import io
import json
import unittest

import joblib
import numpy as np

from src import record_gain_features_v1 as subject
from src.posture_core import extract_feature_vector


class RecordGainFeatureTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(317).uniform(-20, 100, (6, 36, 16)).astype(np.float32)
        self.X[0, 0, 0], self.X[1, 0, 0] = -0., +0.
        self.y = np.array([1, 0, 2, 1, 2, 0])
        self.groups = np.array(["가", "나", "다", "가", "다", "나"])
        self.records = np.array(["가/r.csv", "나/r.csv", "다/r.csv", "가/r.csv", "다/r.csv", "나/r.csv"])

    def augment(self, **kwargs):
        return subject.augment_record_gains(self.X, self.y, self.groups, self.records, **kwargs)

    def test_identity_is_bitwise_unchanged_including_signed_zero_and_no_shared_memory(self):
        before = self.X.tobytes()
        for mode in ("identity", "common", "per_sensor"):
            augmented, audit = self.augment(mode=mode, strength=0., seed=5)
            self.assertEqual(augmented.dtype, np.float32)
            self.assertEqual(augmented.tobytes(), before)
            self.assertEqual(self.X.tobytes(), before)
            self.assertFalse(np.shares_memory(augmented, self.X))
            self.assertEqual(audit["record_gains"], {r: [1.] * 16 for r in sorted(set(self.records))})
            self.assertEqual(audit["input_shape"], audit["output_shape"])
            self.assertFalse(audit["labels_used_for_gain_generation"])

    def test_record_constant_positive_bounds_and_exact_once_rounded_multiplication(self):
        before = self.X.tobytes()
        for mode in ("common", "per_sensor"):
            augmented, audit = self.augment(mode=mode, strength=.2, seed=19)
            self.assertEqual(list(audit["record_gains"]), sorted(set(self.records)))
            for row in audit["records"]:
                take = np.asarray(row["row_indices"])
                factors = np.asarray(audit["record_gains"][row["record_id"]])
                self.assertTrue(np.all((factors >= .8) & (factors <= 1.2)))
                self.assertEqual(len(set(factors)), 1 if mode == "common" else 16)
                expected = (self.X[take].astype(np.float64) * factors[None, None, :]).astype(np.float32)
                self.assertEqual(augmented[take].tobytes(), expected.tobytes())
                self.assertEqual(row["windows"], len(take))
            np.testing.assert_array_equal(augmented == 0, self.X == 0)
            np.testing.assert_array_equal(np.signbit(augmented), np.signbit(self.X))
            self.assertEqual(self.X.tobytes(), before)

    def test_factors_ignore_row_batch_label_and_subject_order_and_normalize_aliases(self):
        augmented, audit = self.augment(mode="per_sensor", strength=.1, seed=31)
        order = np.array([5, 0, 4, 1, 3, 2])
        permuted, changed = subject.augment_record_gains(self.X[order], self.y[order], self.groups[order], self.records[order],
            mode="per_sensor", strength=.1, seed=31)
        np.testing.assert_array_equal(permuted, augmented[order])
        self.assertEqual(changed["record_gains"], audit["record_gains"])
        for rows in (np.array([0]), np.array([5, 1]), np.array([2, 4])):
            chunk, partial = subject.augment_record_gains(self.X[rows], self.y[rows], self.groups[rows], self.records[rows],
                mode="per_sensor", strength=.1, seed=31)
            np.testing.assert_array_equal(chunk, augmented[rows])
            for record, gains in partial["record_gains"].items():
                self.assertEqual(gains, audit["record_gains"][record])
        aliases = self.records.astype(object)
        aliases[0] = " 가/r.csv "
        changed_y = (self.y + 1) % 7
        changed_groups = np.array([f"renamed-{value}" for value in self.groups])
        output, renamed = subject.augment_record_gains(self.X, changed_y, changed_groups, aliases,
            mode="per_sensor", strength=.1, seed=31)
        np.testing.assert_array_equal(output, augmented)
        self.assertEqual(renamed["record_gains"], audit["record_gains"])

    def test_generator_rule_digest_exact_seed_formula_and_reload_reproduction(self):
        first, audit = self.augment(mode="common", strength=.03, seed=71)
        again, repeated = self.augment(mode="common", strength=.03, seed=71)
        self.assertEqual(first.tobytes(), again.tobytes())
        self.assertEqual(audit, repeated)
        self.assertEqual(audit["generator_rule_sha256"], hashlib.sha256(audit["generator_rule"].encode()).hexdigest())
        for row in audit["records"]:
            payload = {"generator_version": "record_gain_v1", "seed": 71, "mode": "common", "record_id": row["record_id"]}
            digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).digest()
            self.assertEqual(row["generator_sha256"], digest.hex())
            scalar = np.random.Generator(np.random.PCG64(int.from_bytes(digest, "big"))).uniform(.97, 1.03)
            self.assertEqual(audit["record_gains"][row["record_id"]], [scalar] * 16)
        stream = io.BytesIO()
        joblib.dump((first, audit), stream)
        stream.seek(0)
        restored, metadata = joblib.load(stream)
        self.assertEqual(restored.tobytes(), first.tobytes())
        self.assertEqual(metadata, audit)
        self.assertEqual(subject.expected_record_gains(self.records, mode="common", strength=.03, seed=71), audit["record_gains"])
        self.assertNotEqual(self.augment(mode="common", strength=.03, seed=72)[1]["record_gains"], audit["record_gains"])

    def test_record_validation_rejects_cross_class_cross_subject_and_bad_identities(self):
        before = deepcopy((self.X, self.y, self.groups, self.records))
        bad_y, bad_groups = self.y.copy(), self.groups.copy()
        bad_y[0] = 2
        bad_groups[0] = "나"
        bad_records = self.records.astype(object)
        bad_records[0] = " "
        for y, groups, records in ((bad_y, self.groups, self.records), (self.y, bad_groups, self.records),
                (self.y, self.groups, bad_records), (self.y[:-1], self.groups, self.records),
                (self.y.astype(float), self.groups, self.records), (self.y, self.groups, np.arange(6))):
            with self.subTest(), self.assertRaises(ValueError):
                subject.augment_record_gains(self.X, y, groups, records, mode="common", strength=.03, seed=7)
        aliases = self.groups.astype(object)
        aliases[0] = " 가 "
        normalized, metadata = subject.validate_training_records(self.y, aliases, self.records)
        np.testing.assert_array_equal(normalized, self.records)
        self.assertEqual(metadata[0], {"record_id": "가/r.csv", "subject": "가", "class_id": 1, "windows": 2, "row_indices": [0, 3]})
        for actual, expected in zip((self.X, self.y, self.groups, self.records), before):
            np.testing.assert_array_equal(actual, expected)

    def test_sequence_and_gain_contracts_reject_invalid_values_without_input_mutation(self):
        for invalid in (self.X.astype(np.float64), self.X[:, :-1], self.X[:0], self.X[0],
                np.full_like(self.X, np.nan), np.full_like(self.X, np.inf)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(ValueError):
                subject.validate_sequences(invalid)
            with self.assertRaises(ValueError):
                subject.sequence_features(invalid)
        for options in ({"mode": "bad"}, {"mode": "identity", "strength": .01},
                {"mode": "common", "strength": 1}, {"mode": "common", "strength": -.1},
                {"mode": "common", "strength": float("nan")}, {"strength": True}, {"seed": True}, {"seed": 2**32}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.augment(**options)
        with self.assertRaisesRegex(ValueError, "align"):
            subject.augment_record_gains(self.X[:1], self.y, self.groups, self.records)

    def test_nonfinite_or_underflowing_augmented_output_is_rejected_without_clipping(self):
        for kind in ("overflow", "underflow"):
            seed = next(seed for seed in range(100) if
                (subject.expected_record_gains(["record"], mode="common", strength=.9, seed=seed)["record"][0] > 1
                 if kind == "overflow" else subject.expected_record_gains(["record"], mode="common", strength=.9, seed=seed)["record"][0] < .5))
            value = np.finfo(np.float32).max if kind == "overflow" else np.nextafter(np.float32(0), np.float32(1))
            sequence = np.full((1, 36, 16), value, dtype=np.float32)
            before = sequence.tobytes()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                subject.augment_record_gains(sequence, [0], ["person"], ["record"], mode="common", strength=.9, seed=seed)
            self.assertEqual(sequence.tobytes(), before)

    def test_all596_features_match_canonical_extraction_bitwise_and_batch_independently(self):
        augmented, audit = self.augment(mode="per_sensor", strength=.03, seed=17)
        for sequences in (self.X, augmented):
            expected = np.stack([extract_feature_vector(row) for row in sequences])
            actual = subject.sequence_features(sequences)
            self.assertEqual(actual.shape, (6, 596))
            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(actual.tobytes(), expected.tobytes())
            self.assertEqual(subject.sequence_features(sequences[:1]).tobytes(), actual[:1].tobytes())
        # Identity reconstruction uses the same immutable feature extractor.
        identity, _ = self.augment()
        self.assertEqual(subject.sequence_features(identity).tobytes(), subject.sequence_features(self.X).tobytes())
        json.dumps(audit, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
