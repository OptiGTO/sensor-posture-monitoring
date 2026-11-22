"""Unit tests for CSV validation, window features, and live smoothing."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from posture_classifier.data import SENSOR_COLUMNS, load_dataset, load_recording
from posture_classifier.evaluation import evaluate_leave_one_subject_out
from posture_classifier.features import FeatureConfig, build_feature_dataset, feature_names
from posture_classifier.model import ProbabilitySmoother


class PostureClassifierTests(unittest.TestCase):
    def _write_recording(self, path: Path, label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["elapsed_s", "label", *SENSOR_COLUMNS])
            writer.writeheader()
            for frame_index in range(30):
                timestamp = 0.001 * (frame_index + 1) if frame_index < 3 else 0.1 + 0.082 * (frame_index - 3)
                row = {"elapsed_s": timestamp, "label": label}
                row.update({column: 1.0 + frame_index * 0.01 + sensor_index * 0.001 for sensor_index, column in enumerate(SENSOR_COLUMNS)})
                writer.writerow(row)

    def test_legacy_label_is_normalized_when_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "subject" / "recording.csv"
            self._write_recording(path, "왼기울임")
            recording = load_recording(path)
        self.assertEqual(recording.label, "왼쪽기울임")
        self.assertEqual(recording.frames.shape, (30, 16))

    def test_feature_dataset_has_expected_shape_after_startup_trim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "subject" / "recording.csv"
            self._write_recording(path, "정자세")
            recording = load_recording(path)
            dataset = build_feature_dataset([recording], FeatureConfig(window_size=12, stride=12))
        self.assertEqual(dataset.features.shape, (2, len(feature_names())))
        self.assertTrue(np.isfinite(dataset.features).all())
        self.assertEqual(dataset.labels.tolist(), ["정자세", "정자세"])

    def test_label_suspect_titles_are_excluded_unless_explicitly_included(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_recording(root / "subject" / "normal.csv", "정자세")
            self._write_recording(root / "subject" / "suspect__label-suspect.csv", "왼쪽기울임")
            default_dataset = load_dataset(root)
            full_dataset = load_dataset(root, include_label_suspect=True)
        self.assertEqual(len(default_dataset), 1)
        self.assertEqual(len(full_dataset), 2)

    def test_probability_smoother_uses_mean_probability(self) -> None:
        smoother = ProbabilitySmoother(["정자세", "왼쪽기울임"], history_size=3)
        label, _ = smoother.update(np.asarray([0.8, 0.2]))
        self.assertEqual(label, "정자세")
        label, probabilities = smoother.update(np.asarray([0.2, 0.8]))
        self.assertEqual(label, "정자세")
        self.assertTrue(np.allclose(probabilities, [0.5, 0.5]))

    def test_loso_evaluation_preserves_full_korean_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recordings = []
            for subject in ("subject_one", "subject_two"):
                for label in ("정자세", "왼쪽기울임"):
                    path = root / subject / f"{label}.csv"
                    self._write_recording(path, label)
                    recordings.append(load_recording(path))
            result = evaluate_leave_one_subject_out(
                recordings,
                feature_config=FeatureConfig(window_size=6, stride=6),
            )
        self.assertEqual(result.labels, ("정자세", "왼쪽기울임"))
        self.assertEqual(len(result.folds), 2)
        self.assertEqual(int(result.confusion_matrix.sum()), 16)


if __name__ == "__main__":
    unittest.main()
