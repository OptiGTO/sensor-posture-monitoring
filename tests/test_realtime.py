from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from software.posture_realtime import (
    build_parser,
    find_serial_port,
    parse_sensor_line,
    prediction_display_text,
    resolve_model_path,
    serial_frames,
)
from src.inference_core import PosturePredictor
from src.posture_core import SENSOR_COLUMNS, read_sensor_csv
from src.realtime import LiveConfig, PredictionEvent, RealtimePosturePredictor, TimeWindowBuffer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakePredictor:
    labels = ("정자세", "왼쪽 기울임")

    def __init__(self) -> None:
        self.frames = []

    def predict_frame(self, frame):
        self.frames.append(frame)
        return np.asarray([0.2, 0.8])


class RealtimeInferenceTests(unittest.TestCase):
    def test_gui_flag_and_single_posture_display(self) -> None:
        args = build_parser().parse_args(["--port", "auto", "--gui"])
        self.assertTrue(args.gui)
        event = PredictionEvent(
            timestamp_s=20.0,
            state="predicting",
            window_start_s=0.0,
            window_end_s=20.0,
            sample_count=240,
            posture="오른쪽 기울임",
            confidence=0.8,
            probabilities={"정자세": 0.2, "오른쪽 기울임": 0.8},
        )
        self.assertEqual(prediction_display_text(event), "오른쪽 기울임")

    def test_window_model_flag_selects_bundled_artifact(self) -> None:
        args = build_parser().parse_args(["--port", "auto", "--window-model", "3"])
        self.assertEqual(
            resolve_model_path(args),
            PROJECT_ROOT / "artifacts" / "posture_classifier_window_3s.joblib",
        )

    def test_serial_port_resolution_supports_auto_and_explicit_modes(self) -> None:
        class Port:
            def __init__(self, device: str, description: str) -> None:
                self.device = device
                self.description = description

        ports = [Port("/dev/cu.other", "Other"), Port("/dev/cu.usbmodem42", "USB")]
        self.assertEqual(find_serial_port("auto", ports), "/dev/cu.usbmodem42")
        self.assertEqual(find_serial_port("/dev/cu.fixed", []), "/dev/cu.fixed")
        self.assertIsNone(find_serial_port("auto", []))

    def test_serial_stream_reconnects_after_a_disconnect(self) -> None:
        import serial

        class Port:
            device = "/dev/cu.usbmodem42"
            description = "Arduino"

        class Connection:
            def __init__(self, result) -> None:
                self.result = result

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def readline(self) -> bytes:
                if isinstance(self.result, Exception):
                    raise self.result
                return self.result

        valid = ",".join(str(value) for value in range(16)).encode()
        connections = [
            Connection(serial.SerialException("disconnected")),
            Connection(valid),
        ]
        with (
            patch("serial.Serial", side_effect=connections) as open_serial,
            patch("serial.tools.list_ports.comports", return_value=[Port()]),
            patch("software.posture_realtime.time.sleep") as sleep,
        ):
            stream = serial_frames("auto", 115200, reconnect_delay_seconds=0.0)
            _, values = next(stream)
            stream.close()

        self.assertEqual(values.shape, (16,))
        self.assertEqual(open_serial.call_count, 2)
        sleep.assert_called_once_with(0.0)

    def test_sensor_line_requires_sixteen_finite_values(self) -> None:
        valid = ",".join(str(value) for value in range(16))
        self.assertEqual(parse_sensor_line(valid).shape, (16,))
        self.assertIsNone(parse_sensor_line("1,2,3"))
        self.assertIsNone(parse_sensor_line(",".join(["nan"] * 16)))

    def test_buffer_uses_a_recent_time_window(self) -> None:
        config = LiveConfig(window_seconds=2.0, minimum_window_seconds=1.5)
        buffer = TimeWindowBuffer(config)
        for timestamp in np.arange(0.0, 2.51, 0.5):
            buffer.append(timestamp, np.arange(16, dtype=float))
        snapshot = buffer.snapshot()
        self.assertGreaterEqual(float(snapshot["elapsed_s"].iloc[-1]), 1.99)
        self.assertEqual(list(snapshot.columns), ["elapsed_s", *SENSOR_COLUMNS])
        self.assertTrue(np.isfinite(snapshot.to_numpy(dtype=float)).all())

    def test_gap_resets_the_history_before_a_new_prediction(self) -> None:
        config = LiveConfig(
            window_seconds=1.0,
            minimum_window_seconds=0.8,
            prediction_interval_seconds=0.1,
            max_gap_seconds=0.5,
            smoothing_history=2,
        )
        predictor = FakePredictor()
        engine = RealtimePosturePredictor(predictor, config)
        for timestamp in (0.0, 0.4, 0.8):
            engine.push(timestamp, np.arange(16, dtype=float))
        self.assertEqual(len(predictor.frames), 1)
        engine.push(2.0, np.arange(16, dtype=float))
        self.assertEqual(engine.buffer.sample_count, 1)
        self.assertEqual(engine.buffer.last_reset_reason, "sensor gap 1.200s")

    def test_replay_matches_batch_inference_for_the_complete_example(self) -> None:
        path = PROJECT_ROOT / "example" / "sample_input.csv"
        frame = read_sensor_csv(path)
        predictor = PosturePredictor(PROJECT_ROOT / "artifacts" / "posture_classifier.joblib")
        batch_probabilities = predictor.predict_frame(frame)
        config = LiveConfig(
            window_seconds=20.0,
            minimum_window_seconds=19.99,
            prediction_interval_seconds=100.0,
            smoothing_history=1,
        )
        engine = RealtimePosturePredictor(predictor, config)
        event = None
        for row in frame.itertuples(index=False):
            event = engine.push(row.elapsed_s, [getattr(row, name) for name in SENSOR_COLUMNS]) or event
        self.assertIsNotNone(event)
        actual = np.asarray([event.probabilities[label] for label in predictor.labels])
        np.testing.assert_allclose(actual, batch_probabilities, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
