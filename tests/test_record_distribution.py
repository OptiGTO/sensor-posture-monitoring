import csv
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.app_services import CSV_COLUMNS
from src.record_distribution import (
    centered_cosine, default_comparison, distribution_figure, load_distribution, posture_name,
    similarity_summary,
)


@dataclass
class FakeRecord:
    csv_path: Path
    subject: str
    posture: str


def write_csv(path, label, pattern, rows=24):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        for step in range(rows):
            writer.writerow([step / 12, label, *(np.asarray(pattern) + .001 * step)])
    return path


class RecordDistributionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        front, back = np.ones(16), np.ones(16)
        front[:4], back[-4:] = 2., 2.
        self.target = FakeRecord(write_csv(root / "a/a_뒤로 기대기_001.csv", "뒤로기대기", front), "a", "뒤로기대기")
        self.records = [
            self.target,
            FakeRecord(write_csv(root / "a/a_앞으로 숙이기_001.csv", "앞으로숙이기", front), "a", "앞으로 숙이기"),
            FakeRecord(write_csv(root / "a/a_뒤로 기대기_002.csv", "뒤로 기대기", back), "a", "뒤로 기대기"),
            FakeRecord(write_csv(root / "b/b_정자세_001.csv", "정자세", front), "b", "정자세"),
            FakeRecord(root / "a/missing.csv", "a", "정자세"),
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def test_references_are_same_person_and_exclude_self(self):
        distribution = load_distribution(self.target, self.records)
        self.assertEqual(distribution.posture, "뒤로 기대기")
        self.assertEqual({k: v.count for k, v in distribution.references.items()}, {"앞으로 숙이기": 1, "뒤로 기대기": 1})
        self.assertEqual(distribution.skipped_references, 1)
        self.assertEqual(default_comparison(distribution), "앞으로 숙이기")
        similarities = distribution.similarities()
        self.assertGreater(similarities["앞으로 숙이기"], similarities["뒤로 기대기"])

    def test_interval_and_figure(self):
        distribution = load_distribution(self.target, self.records)
        np.testing.assert_allclose(distribution.interval_mean(0, .5), distribution.values[:7].mean(0))
        with self.assertRaises(ValueError):
            distribution.interval_mean(1.5, 1.0)
        figure = distribution_figure(distribution, "앞으로 숙이기", 0, 1)
        self.assertEqual(len(figure.axes), 5)
        self.assertIn("참고용", similarity_summary(distribution))

    def test_labels_and_similarity_helpers(self):
        self.assertEqual(posture_name("앞으로숙이기"), "앞으로 숙이기")
        self.assertEqual(posture_name(""), "")
        self.assertEqual(posture_name("미지"), "")
        self.assertAlmostEqual(centered_cosine(np.arange(16.), np.arange(16.) + 3), 1.0)
        self.assertIsNone(centered_cosine(np.ones(16), np.arange(16.)))


if __name__ == "__main__":
    unittest.main()
