"""Checks for registration separation, fixed subject splits and immutable evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments.improvement_retest.prepare import (
    assert_protected, protected_snapshot, reserve_registration, sensor_digest,
    subject_partitions, write_json,
)
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS


class FrozenCohortTests(unittest.TestCase):
    def test_default_keeps_all_records_including_single_correct_record(self):
        rows = [{"subject": "a", "label": CLASS_NAMES[0], "source_relative_path": "a/01.csv"}]
        self.assertEqual(reserve_registration(rows), ([], rows, []))

    def test_registration_is_whole_first_record_and_single_record_subject_has_no_evaluation(self):
        rows = [
            {"subject": "a", "label": CLASS_NAMES[0], "source_relative_path": "a/02.csv"},
            {"subject": "a", "label": CLASS_NAMES[1], "source_relative_path": "a/00.csv"},
            {"subject": "a", "label": CLASS_NAMES[0], "source_relative_path": "a/01.csv"},
            {"subject": "b", "label": CLASS_NAMES[0], "source_relative_path": "b/01.csv"},
        ]
        registrations, retained, excluded = reserve_registration(rows, enabled=True)
        self.assertEqual([r["source_relative_path"] for r in registrations], ["a/01.csv", "b/01.csv"])
        self.assertEqual({r["source_relative_path"] for r in retained}, {"a/00.csv", "a/02.csv"})
        self.assertEqual(excluded[0]["reason"], "subject_has_only_reserved_registration_record")

    def test_three_fixed_splits_cover_classes_and_never_use_holdout_for_development(self):
        groups = np.repeat([f"subject_{index:02}" for index in range(32)], 7)
        labels = np.tile(np.arange(7), 32)
        ids = np.asarray([f"record_{index}" for index in range(len(labels))])
        splits = subject_partitions(groups, labels, ids)
        self.assertEqual(splits, subject_partitions(groups[::-1], labels[::-1], ids[::-1]))
        for split in splits:
            self.assertEqual([len(split[k]) for k in ("train", "validation", "test")], [20, 6, 6])
            self.assertEqual(split["test"], splits[0]["test"])
            self.assertFalse(set(split["test"]) & (set(split["train"]) | set(split["validation"])))
        with self.assertRaisesRegex(ValueError, "at least 13"):
            subject_partitions(groups[:84], labels[:84], ids[:84])

    def test_split_includes_subject_restored_by_unreserving_correct_record(self):
        groups = np.repeat([f"subject_{index:02}" for index in range(33)], 7)
        labels = np.tile(np.arange(7), 33)
        ids = np.asarray([f"record_{index}" for index in range(len(labels))])
        for split in subject_partitions(groups, labels, ids):
            self.assertEqual([len(split[k]) for k in ("train", "validation", "test")], [21, 6, 6])

    def test_sensor_hash_is_label_and_timestamp_independent_but_content_sensitive(self):
        frame = pd.DataFrame(np.arange(48).reshape(3, 16), columns=SENSOR_COLUMNS)
        frame["elapsed_s"], frame["label"] = [0, 1, 2], "old"
        changed_metadata = frame.copy()
        changed_metadata["elapsed_s"], changed_metadata["label"] = [10, 11, 12], "new"
        self.assertEqual(sensor_digest(frame), sensor_digest(changed_metadata))
        changed_metadata.loc[0, SENSOR_COLUMNS[0]] = 2
        self.assertNotEqual(sensor_digest(frame), sensor_digest(changed_metadata))

    def test_snapshot_detects_data_changes_and_new_model_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "research/models").mkdir(parents=True)
            (root / "data/raw.csv").write_text("original")
            out = root / "run"
            write_json(out / "protected_before.json", protected_snapshot(root))
            assert_protected(out, root)
            (root / "research/models/new.json").write_text(json.dumps({"active": True}))
            with self.assertRaisesRegex(ValueError, "inventory/hash/mtime"):
                assert_protected(out, root)


if __name__ == "__main__":
    unittest.main()
