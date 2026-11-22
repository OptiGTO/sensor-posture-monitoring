"""Independent synthetic contracts for exact-input collision diagnostics."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.iterative_improvement import exact_input_collision_audit as audit


class ExactInputCollisionAuditTests(unittest.TestCase):
    def metadata(self, n):
        return (np.arange(n, dtype=np.int64) % 7, np.asarray([f"p{i}" for i in range(n)]),
                np.asarray([f"r{i}" for i in range(n)]), [{"cache": "base", "index": i} for i in range(n)])

    def summarize(self, values, labels=None, mode="exact_bytes"):
        y, people, records, origins = self.metadata(len(values))
        return audit.collision_summary(values, y if labels is None else labels, people, records, origins, mode=mode)

    def test_reader_selects_only_requested_rows_all_supported_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            data = {"one": np.arange(11, dtype=np.int64), "two": np.arange(44, dtype=np.float32).reshape(11, 4),
                    "three": np.arange(264, dtype=np.float32).reshape(11, 3, 8)}
            np.savez_compressed(path, **data)
            for key, values in data.items():
                for rows in ([0, 3, 4, 10], [0], [10]):
                    actual = audit.selected_rows(path, key, np.asarray(rows, dtype=np.int64), 11)
                    self.assertEqual(actual.dtype, values.dtype)
                    self.assertEqual(actual.shape, values[rows].shape)
                    self.assertEqual(actual.tobytes(), values[rows].tobytes())

    def test_reader_rejects_invalid_indices_and_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            np.savez_compressed(path, X=np.arange(20, dtype=np.float32).reshape(5, 4),
                                F=np.asfortranarray(np.arange(20, dtype=np.float32).reshape(5, 4)))
            for indices in (np.asarray([], dtype=np.int64), np.array([1, 1]), np.array([3, 1]),
                            np.array([3, 1], dtype=np.uint64), np.array([-1]), np.array([5]), np.array([1.])):
                with self.subTest(indices=indices), self.assertRaises(ValueError):
                    audit.selected_rows(path, "X", indices, 5)
            with self.assertRaises(ValueError):
                audit.selected_rows(path, "X", np.array([0]), 6)
            with self.assertRaises(ValueError):
                audit.selected_rows(path, "F", np.array([0]), 5)

    def test_counts_distinguish_repeated_rows_from_excess_and_label_conflicts(self):
        X = np.array([[1, 2], [1, 2], [3, 4], [3, 4], [3, 4], [5, 6]], dtype=np.float32)
        result = self.summarize(X, np.array([0, 0, 1, 2, 2, 3], dtype=np.int64))
        expected = {"unique_inputs": 3, "duplicate_groups": 2, "same_label_duplicate_groups": 1,
                    "conflict_groups": 1, "duplicate_rows": 5, "conflicting_rows": 3, "excess_duplicate_rows": 3}
        for key, value in expected.items():
            self.assertEqual(result[key], value, key)
        self.assertEqual(sorted(row["rows"] for row in result["group_details"]), [2, 3])
        self.assertEqual(sum(sum(row["class_counts"]) for row in result["group_details"]), 5)

    def test_forced_hash_collision_still_compares_bytes(self):
        X = np.array([[1, 2], [3, 4], [1, 2], [5, 6]], dtype=np.float32)
        with patch.object(audit, "_content_hash", return_value="0" * 64):
            result = self.summarize(X)
        self.assertEqual(result["unique_inputs"], 3)
        self.assertEqual(result["duplicate_groups"], 1)
        self.assertEqual(result["duplicate_rows"], 2)

    def test_signed_zero_modes_differ_and_input_bytes_are_untouched(self):
        X = np.array([[0., 1.], [-0., 1.], [0., 1.]], dtype=np.float32)
        before = X.tobytes()
        strict = self.summarize(X, np.array([0, 1, 0], dtype=np.int64))
        normalized = self.summarize(X, np.array([0, 1, 0], dtype=np.int64), "signed_zero_normalized")
        self.assertEqual((strict["unique_inputs"], strict["conflict_groups"]), (2, 0))
        self.assertEqual((normalized["unique_inputs"], normalized["conflict_groups"]), (1, 1))
        self.assertEqual(normalized["conflicting_rows"], 3)
        self.assertEqual(X.tobytes(), before)

    def test_sequence_and_feature_rows_share_exact_duplicate_semantics(self):
        X = np.arange(36 * 16, dtype=np.float32).reshape(1, 36, 16)
        values = np.concatenate((X, X, X + 1), axis=0)
        a, b = self.summarize(values), self.summarize(values.reshape(3, -1))
        for key in ("unique_inputs", "duplicate_groups", "duplicate_rows", "conflicting_rows", "conflict_groups"):
            self.assertEqual(a[key], b[key])
        self.assertEqual(a["conflicting_rows"], 2)

    def test_origin_and_member_metadata_remain_aligned(self):
        X = np.zeros((3, 2), dtype=np.float32)
        labels, people, records, origins = self.metadata(3)
        origins[1] = {"cache": "ordinary", "index": 17}
        result = audit.collision_summary(X, labels, people, records, origins)
        members = sorted(result["group_details"][0]["members"], key=lambda row: row["row"])
        for index, member in enumerate(members):
            self.assertEqual(member["origin"], origins[index])
            self.assertEqual(member["class_id"], int(labels[index]))
            self.assertEqual(len(member["subject_sha256"]), 64)
            self.assertEqual(len(member["record_sha256"]), 64)

    def test_input_validation_fails_without_mutation(self):
        X = np.ones((3, 2), dtype=np.float32)
        metadata = list(self.metadata(3))
        for invalid in (X.astype(np.float64), X[0], X[:0], np.full_like(X, np.nan), np.full_like(X, np.inf)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                audit.collision_summary(invalid, *metadata)
        with self.assertRaises(ValueError):
            audit.collision_summary(X, *metadata, mode="rounded")
        for labels in (np.array([0, 1, 7]), np.array([0., 1., 2.]), np.array([0, 1])):
            with self.assertRaises(ValueError):
                audit.collision_summary(X, labels, *metadata[1:])
        metadata[-1][1] = metadata[-1][0].copy()
        with self.assertRaises(ValueError):
            audit.collision_summary(X, *metadata)
        self.assertTrue(np.array_equal(X, np.ones((3, 2), dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
