import unittest
import numpy as np
from experiments.consensus_filter.run import (
    CONDITIONS, common_records, random_control, record_errors, subject_folds, validate_split,
)


class ProtocolTests(unittest.TestCase):
    def test_nested_people_never_overlap_outer_test(self):
        people = [f"s{i:02d}" for i in range(32)]
        coverage = []
        for outer in subject_folds(people, 5, 20260921):
            self.assertFalse(set(outer["train"]) & set(outer["test"]))
            coverage += outer["test"]
            inner_coverage = []
            for inner in subject_folds(outer["train"], 3, 20261021):
                self.assertFalse(set(inner["train"]) & set(inner["test"]))
                self.assertFalse((set(inner["train"]) | set(inner["test"])) & set(outer["test"]))
                inner_coverage += inner["test"]
            self.assertEqual(sorted(inner_coverage), sorted(outer["train"]))
        self.assertEqual(sorted(coverage), people)

    def test_common_screen_requires_all_six_and_matching_records(self):
        tables = {c: [{"record_id": "a", "error_rate": .5}, {"record_id": "b", "error_rate": 1.}]
                  for c in CONDITIONS}
        tables[CONDITIONS[-1]][1]["error_rate"] = .49
        self.assertEqual(common_records(tables), ["a"])
        tables[CONDITIONS[-1]].pop()
        with self.assertRaisesRegex(ValueError, "inventories"):
            common_records(tables)

    def test_unpredicted_test_rows_never_become_candidates(self):
        rows = record_errors(np.array(["a", "a", "b", "c"]), np.array([0, 0, 1, 2]),
                             np.array([0, 1, 0, -1]))
        self.assertEqual([r["record_id"] for r in rows], ["a", "b"])
        self.assertEqual(rows[0]["error_rate"], .5)

    def test_random_removal_preserves_class_record_counts(self):
        rows = [{"record_id": f"c{c}r{i}", "label_id": c} for c in range(7) for i in range(5)]
        removed = ["c0r0", "c0r1", "c6r4"]
        control = random_control(rows, removed, 1)
        self.assertEqual(control, random_control(rows, removed, 1))
        self.assertEqual(len(control), len(set(control)))
        lookup = {r["record_id"]: r["label_id"] for r in rows}
        self.assertEqual(sorted(lookup[r] for r in control), sorted(lookup[r] for r in removed))

    def test_split_rejects_overlap_and_missing_classes(self):
        y = np.tile(np.arange(7), 2)
        g = np.repeat(["a", "b"], 7)
        validate_split(g, y, np.arange(7), np.arange(7, 14))
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_split(g, y, np.arange(8), np.arange(7, 14))
        with self.assertRaisesRegex(ValueError, "Missing class"):
            validate_split(g, y, np.arange(6), np.arange(7, 14))


if __name__ == "__main__":
    unittest.main()
