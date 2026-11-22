"""Synthetic recording alignment and leakage checks; no measurements or fitting."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from experiments.iterative_improvement import record_training as subject
from experiments.iterative_improvement import supplemental as base


class RecordTrainingTests(unittest.TestCase):
    def setUp(self):
        # Interleave subjects, classes and the two windows of each recording.
        rows = [(person, label, window) for window in range(2)
                for label in range(7) for person in ("가", "나", "다")]
        self.data = {
            "X": np.repeat(np.arange(len(rows), dtype=np.float32)[:, None], 596, axis=1),
            "y": np.asarray([label for _, label, _ in rows], dtype=np.int64),
            "groups": np.asarray([person for person, _, _ in rows]),
            "record_ids": np.asarray([f"{person}/class_{label}.csv" for person, label, _ in rows]),
        }
        extra_groups = np.asarray(["다", "가", "나", "가", "다", "나"])
        self.supplement = {
            "X": np.repeat((100 + np.arange(6, dtype=np.float32))[:, None], 596, axis=1),
            "y": np.zeros(6, dtype=np.int64),
            "groups": extra_groups,
            "record_ids": np.asarray([f"{person}/restored.csv" for person in extra_groups]),
        }
        self.train = np.flatnonzero(np.isin(self.data["groups"], ["가", "나"]))[::-1]

    def assert_inputs_unchanged(self, before):
        for actual, expected in zip((self.data, self.supplement), before):
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])

    def test_interleaved_rows_train_only_alignment_and_exact_original_audit(self):
        before = deepcopy((self.data, self.supplement))
        expected_X, expected_y, expected_groups, expected_audit = base.append_training(
            self.data, self.supplement, self.train)
        X, y, groups, records, audit = subject.append_record_training(self.data, self.supplement, self.train)
        take = np.asarray([1, 2, 3, 5])
        for actual, expected in ((X, expected_X), (y, expected_y), (groups, expected_groups),
                (records, np.concatenate((self.data["record_ids"][self.train],
                                          self.supplement["record_ids"][take])))):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(audit, expected_audit)
        self.assertEqual(set(groups), {"가", "나"})
        self.assertEqual(len(records), len(self.train) + len(take))
        self.assert_inputs_unchanged(before)
        for returned in (X, y, groups, records):
            self.assertFalse(any(np.shares_memory(returned, original)
                                 for arrays in (self.data, self.supplement) for original in arrays.values()))

    def test_normalized_subject_selection_preserves_original_identity_strings(self):
        self.supplement["groups"] = np.asarray(["다", " 가 ", "나", "가", "다", "나"])
        X, y, groups, records, audit = subject.append_record_training(self.data, self.supplement, self.train)
        self.assertEqual(audit["training_subjects"], ["가", "나"])
        self.assertEqual(groups[len(self.train)], " 가 ")
        self.assertEqual(records[len(self.train)], "가/restored.csv")
        np.testing.assert_array_equal(X[-4:], self.supplement["X"][[1, 2, 3, 5]])

    def test_original_train_and_evaluation_overlap_guards_are_retained(self):
        bad_cases = []
        for identity in ("가/class_0.csv", "다/class_0.csv", " 가/class_0.csv "):
            changed = deepcopy(self.supplement)
            changed["record_ids"] = changed["record_ids"].astype("U80")
            changed["record_ids"][1] = identity
            bad_cases.append((self.train, changed))
        bad_cases.extend((train, self.supplement) for train in (
            self.train[:-1], np.asarray([0, 0]), np.asarray([999]), np.asarray([1.5]),
            np.asarray([], dtype=int)))
        for train, supplement in bad_cases:
            before = deepcopy((self.data, supplement))
            with self.subTest(train=train, records=supplement["record_ids"]), self.assertRaises(ValueError):
                subject.append_record_training(self.data, supplement, train)
            for actual, expected in zip((self.data, supplement), before):
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key])

    def test_record_with_multiple_posture_labels_is_rejected_without_mutation(self):
        # One existing record receives another label while remaining in one person.
        self.data["y"][0] = 1
        before = deepcopy((self.data, self.supplement))
        with self.assertRaisesRegex(ValueError, "multiple posture labels"):
            subject.append_record_training(self.data, self.supplement, self.train)
        self.assert_inputs_unchanged(before)

    def test_record_with_multiple_subjects_is_rejected_without_mutation(self):
        for target in (self.data, self.supplement):
            original = target["record_ids"].copy()
            target["record_ids"][0] = target["record_ids"][1]
            before = deepcopy((self.data, self.supplement))
            with self.subTest(target_is_base=target is self.data), self.assertRaisesRegex(ValueError, "multiple subjects"):
                subject.append_record_training(self.data, self.supplement, self.train)
            self.assert_inputs_unchanged(before)
            target["record_ids"] = original

    def test_delegate_row_count_or_order_drift_is_rejected(self):
        original = base.append_training(self.data, self.supplement, self.train)
        bad_results = [
            (original[0][:-1], *original[1:]),
            (original[0], original[1][::-1], original[2], original[3]),
            (original[0], original[1], original[2][::-1], original[3]),
            (*original[:3], {**original[3], "added_windows": 999}),
        ]
        for changed in bad_results:
            with self.subTest(), patch.object(base, "append_training", return_value=changed):
                with self.assertRaisesRegex(ValueError, "feature training selection"):
                    subject.append_record_training(self.data, self.supplement, self.train)

    def test_ordinary_identity_validation_rejects_invalid_or_ambiguous_rows(self):
        subject.validate_record_identities(self.data["y"], self.data["groups"], self.data["record_ids"])
        subject.validate_record_identities([0, 0], ["가", " 가 "], ["가/x.csv", " 가/x.csv "])
        for args in (([0, 1], ["가", "가"], ["가/x.csv", " 가/x.csv "]),
                     ([0, 0], ["가", "나"], ["x", "x"]),
                     ([0], ["가", "나"], ["x"]), ([0.5], ["가"], ["x"]),
                     ([7], ["가"], ["x"]), ([0], [" "], ["x"]),
                     ([0], ["가"], [""]), ([], [], [])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                subject.validate_record_identities(*args)


if __name__ == "__main__":
    unittest.main()
