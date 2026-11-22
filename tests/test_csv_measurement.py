"""Exercise the GUI worker, including CSV loading and prediction sidecars."""
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np

from software import posture_app as app
from software.posture_realtime import replay_csv
from src.app_services import MeasurementRecorder
from src.model_runtime import ModelRegistry, SessionModelSnapshot

ROOT = Path(__file__).resolve().parents[1]


class CsvMeasurementTests(TestCase):
    def run_csv(self, registry, model, root):
        snapshot = SessionModelSnapshot.create(model, session_id="csv-test", window_seconds=model.window_seconds, input_source="csv_replay")
        recorder = MeasurementRecorder(root, mode="measurement", model_snapshot=snapshot)
        session = SimpleNamespace(selected_model=model, registry=registry, session_snapshot=snapshot,
                                  _recorder=recorder, _session_input=(str(ROOT / "example/sample_input.csv"), "auto"),
                                  _stop=threading.Event(), _messages=queue.Queue())
        with mock.patch.object(app, "replay_csv", side_effect=lambda path, speed, **kw: replay_csv(path, 0, **kw)), \
             mock.patch.object(app, "OccupancyDetector", side_effect=AssertionError("CSV must not use live occupancy")):
            app.PostureApplication._run_session(session)
        messages = list(session._messages.queue)
        self.assertFalse([item for item in messages if item[0] in {"error", "empty"}])
        predictions = [payload[0] for kind, payload in messages if kind == "prediction"]
        self.assertTrue(predictions)
        rows = [json.loads(line) for line in recorder.prediction_path.read_text().splitlines()]
        self.assertEqual(len(rows), len(predictions))
        self.assertTrue(all(row["posture"] for row in rows))
        self.assertEqual(messages[-1][0], "finished")
        return predictions

    def test_csv_produces_predictions_with_each_builtin_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(ROOT, Path(tmp))
            models = registry.discover()
            self.assertEqual({model.window_seconds for model in models}, {3, 5, 20})
            for model in models:
                with self.subTest(window=model.window_seconds):
                    self.run_csv(registry, model, Path(tmp))

    def test_low_confidence_still_displays_highest_scoring_posture(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(ROOT, Path(tmp))
            model = next(model for model in registry.discover() if model.window_seconds == 3)
            scores = np.array([0.28, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
            predictor = SimpleNamespace(minimum_window_seconds=2.85, labels=model.labels, predict_frame=lambda frame: scores)
            with mock.patch.object(app, "PosturePredictor", return_value=predictor):
                predictions = self.run_csv(registry, model, Path(tmp))
            self.assertTrue(all(event.posture == model.labels[0] for event in predictions))
            self.assertTrue(all(event.confidence < 0.35 for event in predictions))
