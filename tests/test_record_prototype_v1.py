"""Synthetic recording-prototype contracts; no project measurements are read."""
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone

from src import record_prototype_v1 as subject


def fixture(dtype=np.float32):
    labels = np.repeat(np.arange(7), 6)
    records = np.asarray([f"class-{c}/record-{r}" for c in range(7) for r in range(3) for _ in range(2)])
    groups = np.asarray([f"person-{c}" for c in labels])
    features = np.zeros((len(labels), 596), dtype=dtype)
    features[:, 0] = [c * 10 + r * 2 + w * 2 for c in range(7) for r in range(3) for w in range(2)]
    return features, labels, groups, records


class RecordPrototypeTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.groups, self.records = fixture()

    def fit(self, k=1, X=None, **kwargs):
        return subject.RecordPrototypeClassifier(n_neighbors=k, **kwargs).fit(
            self.X if X is None else X, self.y, groups=self.groups, record_ids=self.records)

    def test_exact_record_means_class_alignment_and_expected_k1_k3_scores(self):
        query = np.zeros((2, 596), dtype=np.float32)
        query[:, 0] = [1, 15]
        for k in (1, 3):
            model = self.fit(k)
            expected_means = np.asarray([c * 10 + r * 2 + 1 for c in range(7) for r in range(3)])
            np.testing.assert_array_equal(model.prototypes_[:, 0], expected_means)
            self.assertEqual(model.prototypes_.dtype, np.float64)
            self.assertFalse(model.prototypes_[:, 1:].any())
            np.testing.assert_array_equal(model.prototype_labels_, np.repeat(np.arange(7), 3))
            np.testing.assert_array_equal(model.classes_, np.arange(7))
            np.testing.assert_array_equal(model.class_record_counts_, np.full(7, 3))
            np.testing.assert_array_equal(model.class_counts_, np.full(7, 6))
            self.assertEqual(model.training_details_["input_shape"], [42, 596])
            self.assertEqual(model.training_details_["training_subjects"], [f"person-{i}" for i in range(7)])
            self.assertEqual(model.training_details_["record_metadata"], model.record_metadata_)
            self.assertEqual(model.input_sha256_, subject._input_digest(self.X))
            distances = np.abs(query[:, :1] - expected_means[None, :]).reshape(2, 7, 3)
            scores = 1 / (1e-12 + np.sort(distances, axis=2)[:, :, :k].mean(axis=2))
            expected = scores / scores.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(model.predict_proba(query), expected, rtol=1e-15, atol=0)
            np.testing.assert_array_equal(model.predict(query), expected.argmax(axis=1))
            subject.verify_record_prototype_identity(model, k, 1e-12)

    def test_nfc_aliases_are_one_record_and_subject_but_cross_identity_is_rejected(self):
        records, groups = self.records.astype(object), self.groups.astype(object)
        records[:2], groups[:2] = [" 가/record ", "가/record"], [" 가 ", "가"]
        model = subject.RecordPrototypeClassifier().fit(self.X, self.y, groups=groups, record_ids=records)
        row = next(row for row in model.record_metadata_ if row["record_id"] == "가/record")
        self.assertEqual(row, {"record_id": "가/record", "subject": "가", "class_id": 0,
                               "windows": 2, "row_indices": [0, 1]})
        for changed_y, changed_groups, changed_records in (
                (np.concatenate(([1], self.y[1:])), groups, records),
                (self.y, np.concatenate((["another"], groups[1:])), records),
                (self.y, groups, np.concatenate(([" "], records[1:])))):
            candidate = subject.RecordPrototypeClassifier()
            with self.assertRaises(ValueError):
                candidate.fit(self.X, changed_y, groups=changed_groups, record_ids=changed_records)
            self.assertFalse(hasattr(candidate, "prototypes_"))

    def test_missing_labels_insufficient_distinct_records_and_bad_parameters_fail(self):
        with self.assertRaisesRegex(ValueError, "seven"):
            subject.RecordPrototypeClassifier().fit(self.X[:-6], self.y[:-6],
                groups=self.groups[:-6], record_ids=self.records[:-6])
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.fit(4)
        for options in ({"n_neighbors": 0}, {"n_neighbors": 1.0}, {"n_neighbors": True},
                {"epsilon": 0}, {"epsilon": -1}, {"epsilon": np.nan}, {"epsilon": np.inf}, {"epsilon": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                subject.RecordPrototypeClassifier(**options).fit(self.X, self.y,
                    groups=self.groups, record_ids=self.records)
        with self.assertRaisesRegex(ValueError, "align"):
            self.fit(X=self.X[:-1])
        with self.assertRaises(ValueError):
            subject.RecordPrototypeClassifier().fit(self.X, self.y.astype(float),
                groups=self.groups, record_ids=self.records)

    def test_row_permutation_query_order_batching_and_reload_are_bitwise_consistent(self):
        rng = np.random.default_rng(147)
        values = rng.normal(size=(42, 596)).astype(np.float64)
        # Deliberate cancellation: canonical feature reduction order must survive
        # a permutation of the complete training input, even for float64 input.
        records = np.repeat([f"record-{i}" for i in range(7)], 6)
        values[:6, 0] = [1e15, -1e15, 1, 2, 3, 4]
        model = subject.RecordPrototypeClassifier().fit(values, self.y, groups=self.groups, record_ids=records)
        order = rng.permutation(len(values))
        reordered = subject.RecordPrototypeClassifier().fit(values[order], self.y[order],
            groups=self.groups[order], record_ids=records[order])
        self.assertEqual(model.prototypes_.tobytes(), reordered.prototypes_.tobytes())
        query = rng.normal(size=(265, 596)).astype(np.float64)
        expected = model.predict_proba(query)
        self.assertEqual(expected.tobytes(), reordered.predict_proba(query).tobytes())
        parts = np.concatenate([model.predict_proba(query[:1]), model.predict_proba(query[1:97]),
                                model.predict_proba(query[97:])])
        self.assertEqual(parts.tobytes(), expected.tobytes())
        np.testing.assert_array_equal(model.predict_proba(query[::-1]), expected[::-1])
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        self.assertEqual(restored.predict_proba(query).tobytes(), expected.tobytes())
        subject.verify_record_prototype_identity(restored, 1, 1e-12)
        self.assertEqual(clone(model).get_params(), {"n_neighbors": 1, "epsilon": 1e-12})

    def test_no_mutation_aliasing_or_evaluation_batch_adaptation(self):
        before = [value.tobytes() for value in (self.X, self.y, self.groups, self.records)]
        model = self.fit(3)
        fitted = deepcopy(model.__dict__)
        first = model.predict_proba(self.X[:1])
        together = model.predict_proba(np.concatenate([self.X[:1], self.X + 5000]))
        np.testing.assert_array_equal(first, together[:1])
        self.assertEqual(before, [value.tobytes() for value in (self.X, self.y, self.groups, self.records)])
        self.assertFalse(np.shares_memory(model.prototypes_, self.X))
        self.assertEqual(model.state_sha256_, fitted["state_sha256_"])
        self.assertEqual(model.record_metadata_, fitted["record_metadata_"])
        self.assertEqual(model.prototypes_.tobytes(), fitted["prototypes_"].tobytes())

    def test_validation_finite_range_and_all_zero_class_ties(self):
        model = self.fit()
        for invalid in (self.X[:, :-1], self.X[0], self.X[:0], self.X.astype(int),
                self.X.astype(np.float16), np.full_like(self.X, np.nan), np.full_like(self.X, np.inf)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(ValueError):
                model.predict_proba(invalid)
            with self.assertRaises(ValueError):
                self.fit(X=invalid)
        zero_model = self.fit(X=np.zeros_like(self.X), epsilon=np.nextafter(0., 1.))
        expected = np.full((2, 7), 1 / 7)
        np.testing.assert_array_equal(zero_model.predict_proba(np.zeros((2, 596), np.float64)), expected)
        with self.assertRaisesRegex(ValueError, "distances"):
            model.predict_proba(np.full((1, 596), 1e300, np.float64))

    def test_distance_batches_remain_two_dimensional_and_bounded(self):
        model = self.fit(3)
        real = subject.cdist
        calls = []
        def observed(query, prototypes, **kwargs):
            calls.append((query.shape, prototypes.shape))
            return real(query, prototypes, **kwargs)
        with patch.object(subject, "cdist", side_effect=observed):
            result = model.predict_proba(np.zeros((259, 596), np.float32))
        self.assertEqual(len(calls), 21)
        self.assertEqual(result.shape, (259, 7))
        self.assertTrue(all(a[0] <= 128 and a[1] == 596 and b == (3, 596) for a, b in calls))
        np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-15)

    def test_identity_rejects_parameter_state_hash_and_metadata_drift(self):
        original = self.fit(3)
        mutations = [
            lambda m: setattr(m, "n_neighbors", 1),
            lambda m: setattr(m, "epsilon_fit_", 1e-8),
            lambda m: m.prototypes_.__setitem__((0, 0), 999),
            lambda m: m.prototype_labels_.__setitem__(0, 2),
            lambda m: m.class_record_counts_.__setitem__(0, 99),
            lambda m: m.class_counts_.__setitem__(0, 99),
            lambda m: setattr(m, "n_features_in_", 595),
            lambda m: setattr(m, "state_sha256_", "0" * 64),
            lambda m: setattr(m, "metadata_sha256_", "0" * 64),
            lambda m: setattr(m, "input_sha256_", "1" * 64),
            lambda m: m.training_details_.__setitem__("input_shape", [99, 596]),
            lambda m: m.record_metadata_[0].__setitem__("subject", "another"),
            lambda m: m.record_metadata_[0].__setitem__("row_indices", [0, 0]),
            lambda m: m.record_metadata_[0].__setitem__("record_id", "noncanonical "),
        ]
        for mutate in mutations:
            changed = deepcopy(original)
            mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                subject.verify_record_prototype_identity(changed, 3, 1e-12)
        # Even rehashing malformed metadata cannot bypass the structural checks.
        changed = deepcopy(original)
        changed.record_metadata_[0]["row_indices"] = [0, 0]
        changed.metadata_sha256_ = subject._metadata_digest(changed.record_metadata_)
        changed.state_sha256_ = subject._state_digest(changed)
        with self.assertRaisesRegex(ValueError, "unique"):
            subject.verify_record_prototype_identity(changed, 3, 1e-12)


if __name__ == "__main__":
    unittest.main()
