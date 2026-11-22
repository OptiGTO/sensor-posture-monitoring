from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unicodedata
import unittest

import numpy as np
import pandas as pd

from src.app_services import DatasetService
from src.model_comparison import train_comparison
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS


class TrainingDeduplicationTests(unittest.TestCase):
    def make_data(self, root):
        rng = np.random.default_rng(41)
        for subject in ("가상피험자", "subject1", "subject2", "subject3"):
            directory = root / subject
            directory.mkdir(parents=True)
            for index, label in enumerate(CLASS_NAMES):
                frame = pd.DataFrame(rng.uniform(10, 20, (37, 16)) + index * 20,
                                     columns=SENSOR_COLUMNS)
                frame.insert(0, "label", label)
                frame.insert(0, "elapsed_s", np.arange(37) / 12)
                frame.to_csv(directory / f"posture{index}.csv", index=False)

    def test_two_roots_unicode_names_and_shifted_copy_train_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_data(root / "data")
            source = root / "data/가상피험자/posture2.csv"
            other = root / "other_data" / unicodedata.normalize("NFD", "가상피험자")
            other.mkdir(parents=True)
            frame = pd.read_csv(source)
            frame["elapsed_s"] = frame["elapsed_s"] + 42
            frame["label"] = "뒤로기대기"
            frame.to_csv(other / "뒤로 기대기.csv", index=False)
            # A distinct recording for the same decomposed subject must also
            # land in the original subject's partition.
            frame[SENSOR_COLUMNS] = frame[SENSOR_COLUMNS] + 0.25
            frame.to_csv(other / "distinct.csv", index=False)
            service = DatasetService(root, additional_data_directories=[root / "other_data"])
            records = service.records()
            before = {r.csv_path: (hashlib.sha256(r.csv_path.read_bytes()).hexdigest(),
                                  r.csv_path.stat().st_mtime_ns) for r in records}
            staged = service.stage_for_training(records, root / "staged")
            summary = json.loads((staged / "staging_summary.json").read_text())
            self.assertEqual(summary["selected_count"], 30)
            self.assertEqual(summary["staged_count"], 29)
            self.assertEqual(len(summary["skipped_duplicates"]), 1)
            self.assertEqual(len(list(staged.glob("*/*.csv"))), 29)
            report = train_comparison(staged, root / "run", (3,), 41,
                                      model_families=("logistic_regression",), progress=lambda _: None)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["datasets"]["3"]["source_records"]), 29)
            subjects = [s for p in report["datasets"]["3"]["partitions"].values() for s in p["subjects"]]
            self.assertEqual(len(subjects), 4)
            self.assertEqual(before, {p: (hashlib.sha256(p.read_bytes()).hexdigest(),
                                         p.stat().st_mtime_ns) for p in before})

    def test_conflicting_subject_or_label_is_rejected_before_worker(self):
        for conflict in ("subject", "label"):
            with self.subTest(conflict=conflict), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.make_data(root / "data")
                frame = pd.read_csv(root / "data/가상피험자/posture2.csv")
                subject = "subject1" if conflict == "subject" else "가상피험자"
                if conflict == "label":
                    frame["label"] = CLASS_NAMES[0]
                frame.to_csv(root / "data" / subject / "duplicate.csv", index=False)
                service = DatasetService(root)
                with self.assertRaisesRegex(ValueError, "피험자 또는 자세 라벨"):
                    service.stage_for_training(service.records(), root / "stage")

    def test_repeated_selection_keeps_one_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_data(root / "data")
            service = DatasetService(root)
            record = service.records()[0]
            stage = service.stage_for_training([record, record], root / "stage")
            summary = json.loads((stage / "staging_summary.json").read_text())
            self.assertEqual(summary["staged_count"], 1)
            self.assertEqual(len(summary["skipped_duplicates"]), 1)
