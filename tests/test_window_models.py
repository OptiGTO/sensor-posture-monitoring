from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from inference_window import predict_one
from src.inference_core import PosturePredictor
from src.posture_core import read_sensor_csv
from src.window_pipeline import WindowSpec, segment_sensor_frame, window_start_times


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WindowModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.input_path = PROJECT_ROOT / "example" / "sample_input.csv"
        cls.frame = read_sensor_csv(cls.input_path)
        cls.expected = json.loads(
            (PROJECT_ROOT / "example" / "example_prediction_window_both.json").read_text(
                encoding="utf-8"
            )
        )["결과"]

    def test_expected_window_counts_follow_training_segmentation(self) -> None:
        duration = float(
            self.frame["elapsed_s"].iloc[-1] - self.frame["elapsed_s"].iloc[0]
        )
        self.assertEqual(
            window_start_times(duration, 3.0),
            [0.0, 3.0, 6.0, 9.0, 12.0, 15.0],
        )
        self.assertEqual(
            window_start_times(duration, 5.0),
            [0.0, 5.0, 10.0, 15.0],
        )
        windows_3s, _ = segment_sensor_frame(self.frame, WindowSpec(3.0))
        windows_5s, _ = segment_sensor_frame(self.frame, WindowSpec(5.0))
        self.assertEqual(windows_3s.shape, (6, 36, 16))
        self.assertEqual(windows_5s.shape, (4, 60, 16))

    def test_integrated_inference_matches_validated_example(self) -> None:
        for seconds in (3, 5):
            with self.subTest(seconds=seconds):
                model_path = (
                    PROJECT_ROOT
                    / "artifacts"
                    / f"posture_classifier_window_{seconds}s.joblib"
                )
                result = predict_one(self.input_path, model_path)
                expected = self.expected[f"{seconds}초"]
                self.assertEqual(result["창_개수"], expected["창_개수"])
                self.assertEqual(
                    result["전체_기록_예측_자세"],
                    expected["전체_기록_예측_자세"],
                )
                actual = np.asarray(
                    list(result["전체_기록_클래스별_평균확률"].values())
                )
                wanted = np.asarray(
                    list(expected["전체_기록_클래스별_평균확률"].values())
                )
                np.testing.assert_allclose(actual, wanted, rtol=1e-9, atol=1e-9)

    def test_shared_predictor_aggregates_all_windows(self) -> None:
        predictor = PosturePredictor(
            PROJECT_ROOT / "artifacts" / "posture_classifier_window_3s.joblib"
        )
        self.assertTrue(predictor.is_window_model)
        self.assertEqual(predictor.window_seconds, 3.0)
        self.assertAlmostEqual(predictor.minimum_window_seconds, 2.85)
        probability = predictor.predict_frame(self.frame)
        expected = np.asarray(
            list(self.expected["3초"]["전체_기록_클래스별_평균확률"].values())
        )
        np.testing.assert_allclose(probability, expected, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
