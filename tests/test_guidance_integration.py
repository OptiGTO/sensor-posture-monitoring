"""Synthetic controller/worker integration; these tests are not physical sensor evidence."""
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from src.app_services import MeasurementRecorder
from src.guidance_session import GuidanceSession
from src.personal_guidance import AlertSettings, CalibrationContext
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, read_sensor_csv
from src.model_runtime import ModelRegistry, SessionModelSnapshot


def controller_reference(root, *, baseline=1.2, gain=1.0, mode="deployment", recorder=None):
    messages = []
    session = GuidanceSession(root, "synthetic-test-user", CalibrationContext(),
                              AlertSettings.for_mode(mode), messages.append, recorder)
    session.command("baseline")
    for step in range(31):
        session.push(step / 10, np.full(16, baseline))
    for step in range(31, 46):
        session.push(step / 10, np.full(16, baseline + gain))
    session.command("register")
    for step in range(46, 150):
        session.push(step / 10, np.full(16, baseline + gain))
    if session.engine is None:
        raise AssertionError(messages[-1])
    return session, messages


class GuidanceIntegrationTests(unittest.TestCase):
    def test_real_worker_synthetic_serial_frames_continue_while_single_popup_is_open(self):
        from software import posture_app as app

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = Path(__file__).resolve().parents[1]
            registry = ModelRegistry(project, root)
            model = next(model for model in registry.discover() if model.window_seconds == 3)
            snapshot = SessionModelSnapshot.create(model, session_id="synthetic-integration",
                                                    window_seconds=3, input_source="serial")
            recorder = MeasurementRecorder(root, mode="measurement", model_snapshot=snapshot,
                                           guidance_metadata={"synthetic_test": True})
            session = SimpleNamespace(selected_model=model, registry=registry, session_snapshot=snapshot,
                                      _recorder=recorder, _session_input=("", "synthetic"),
                                      _stop=threading.Event(), _messages=queue.Queue(),
                                      _session_guidance_config=("synthetic-test-user", CalibrationContext(),
                                                                AlertSettings.for_mode("development")))
            original = []

            def synthetic_frames(*args, **kwargs):
                origin = time.monotonic()
                controller = session._guidance_session
                controller.command("baseline")
                for step in range(661):
                    if step == 46:
                        controller.command("register")
                    if step == 340:
                        controller.command("close_popup")
                    values = np.full(16, 1.2 if step < 31 or step >= 640 else 2.2)
                    if 150 <= step < 600:
                        values[:8] += 1.0
                    values += .0001 * np.sin(step * .113 + np.arange(16) * .37)
                    original.append(values.copy())
                    yield origin + step / 10, values

            with mock.patch.object(app, "serial_frames", side_effect=synthetic_frames), \
                 mock.patch.object(app, "STATE_ROOT", root):
                app.PostureApplication._run_session(session)
            messages = list(session._messages.queue)
            self.assertFalse([payload for kind, payload in messages if kind == "error"])
            self.assertEqual(messages[-1][0], "finished")
            self.assertTrue(all(payload[2] for kind, payload in messages if kind == "frame"))
            guidance = [payload for kind, payload in messages if kind == "guidance"]
            shows = [event for event in guidance if event.popup_action == "show"]
            self.assertEqual(len(shows), 2)
            self.assertTrue(any(event.popup_open and event.timestamp_s > shows[0].timestamp_s + 4 for event in guidance))
            self.assertTrue(any(event.state == "correct" and event.timestamp_s > 62 for event in guidance))
            self.assertTrue(any(event.state == "empty" for event in guidance))
            self.assertGreater(len([payload for kind, payload in messages if kind == "prediction"]), 20)
            saved = read_sensor_csv(recorder.csv_path)
            self.assertEqual(len(saved), 661)
            np.testing.assert_allclose(saved[SENSOR_COLUMNS].to_numpy(), original, rtol=0, atol=1e-12)

    def test_empty_baseline_registration_loading_and_context_invalidation_for_voltage_and_adc(self):
        for baseline, gain in ((1.2, 1.0), (200.0, 100.0)):
            with self.subTest(scale=baseline), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                session, messages = controller_reference(root, baseline=baseline, gain=gain)
                self.assertTrue(session.detector.occupied)
                self.assertTrue(any(kind == "guidance_status" and value["state"] == "profile_loaded"
                                    for kind, value in messages))
                profile = session.engine.profile
                np.testing.assert_allclose(profile.context.empty_seat_center, baseline)
                np.testing.assert_allclose(profile.ratio_center, np.full(16, 1 / 16))
                loaded_messages = []
                loaded = GuidanceSession(root, "synthetic-test-user", CalibrationContext(),
                                         AlertSettings(), loaded_messages.append)
                loaded.command("baseline")
                for step in range(31):
                    loaded.push(step / 10, np.full(16, baseline))
                self.assertEqual(loaded.engine.profile.profile_version, profile.profile_version)
                changed = GuidanceSession(root, "synthetic-test-user", CalibrationContext(placement_id="moved"),
                                          AlertSettings(), loaded_messages.append)
                changed.command("baseline")
                for step in range(31):
                    changed.push(step / 10, np.full(16, baseline))
                self.assertIsNone(changed.engine)
                self.assertEqual(loaded_messages[-1][1]["state"], "profile_required")

    def test_long_synthetic_replay_keeps_type_changes_popup_close_and_return_separate(self):
        for mode in ("development", "deployment"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                session, messages = controller_reference(Path(temporary), mode=mode)
                good = np.full(16, 2.2)
                bad = good.copy()
                bad[:8] += 1.0
                shows = []
                first_show = None
                for step in range(150, 2251):
                    timestamp = step / 10
                    prediction = SimpleNamespace(posture=CLASS_NAMES[1 if timestamp < 120 else 2],
                                                 confidence=.9, timestamp_s=timestamp)
                    session.push(timestamp, bad, prediction)
                    event = session.last_event
                    if event.popup_action == "show":
                        shows.append(timestamp)
                        if first_show is None:
                            first_show = timestamp
                    # Keeping the initial popup open suppresses duplicate shows.
                    if first_show is not None and abs(timestamp - (first_show + 5.0)) < .001:
                        session.command("close_popup")
                settings = AlertSettings.for_mode(mode)
                self.assertEqual(len(shows), 2)
                self.assertGreaterEqual(shows[0] - 15.0, settings.initial_alert_s)
                self.assertLess(shows[0] - 15.0, settings.initial_alert_s + 1.0)
                self.assertAlmostEqual(shows[1] - shows[0], settings.realert_interval_s, delta=.11)
                self.assertGreater(session.last_event.incorrect_duration_s, 200)
                return_started = 225.1
                dismissed = None
                for step in range(2251, 2291):
                    session.push(step / 10, good)
                    if session.last_event.popup_action == "dismiss":
                        dismissed = step / 10
                self.assertIsNotNone(dismissed)
                self.assertLess(dismissed - return_started, 3.1)
                self.assertEqual(session.last_event.state, "correct")
                self.assertEqual(session.last_event.incorrect_duration_s, 0)
                for step in range(2291, 2311):
                    session.push(step / 10, np.full(16, 1.2))
                self.assertEqual(session.last_event.state, "empty")
                self.assertEqual(session.last_event.incorrect_duration_s, 0)

    def test_disconnect_and_invalid_input_reset_active_guidance(self):
        with tempfile.TemporaryDirectory() as temporary:
            session, _ = controller_reference(Path(temporary), mode="development")
            bad = np.full(16, 2.2)
            bad[:8] += 1
            for step in range(150, 301):
                session.push(step / 10, bad)
            self.assertTrue(session.last_event.popup_open)
            session.interrupt("disconnected")
            self.assertEqual(session.last_event.state, "input_error")
            self.assertEqual(session.last_event.incorrect_duration_s, 0)
            self.assertFalse(session.last_event.popup_open)
            session.push(30.2, [np.nan] * 16)
            self.assertEqual(session.last_event.state, "input_error")

    def test_rejected_frames_block_model_inference_and_reset_an_open_warning(self):
        for case in ("negative", "nonfinite", "nonfinite_timestamp", "duplicate_timestamp", "backward_timestamp", "gap"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                session, _ = controller_reference(Path(temporary), mode="development")
                bad = np.full(16, 2.2)
                bad[:8] += 1
                for step in range(150, 301):
                    session.push(step / 10, bad)
                self.assertTrue(session.last_event.popup_open)
                self.assertGreater(session.last_event.incorrect_duration_s, 12)
                timestamp, values = 30.1, bad.copy()
                if case == "negative": values[0] = -1.0
                elif case == "nonfinite": values[0] = np.nan
                elif case == "nonfinite_timestamp": timestamp = np.nan
                elif case == "duplicate_timestamp": timestamp = 30.0
                elif case == "backward_timestamp": timestamp = 29.0
                elif case == "gap": timestamp = 32.0
                # The worker uses this result to decide whether to call its
                # model engine. A displayed error alone must not allow inference.
                self.assertFalse(session.push(timestamp, values))
                self.assertEqual(session.last_event.state, "input_error")
                self.assertEqual(session.last_event.incorrect_duration_s, 0)
                self.assertFalse(session.last_event.popup_open)
                self.assertEqual(session.last_event.popup_action, "dismiss")

    def test_recording_preserves_raw_samples_and_guidance_is_a_separate_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = SimpleNamespace(as_dict=lambda: {"model_id": "synthetic-test-model"})
            recorder = MeasurementRecorder(root, mode="measurement", model_snapshot=snapshot,
                                           guidance_metadata={"mode": "deployment", "synthetic_test": True})
            session, messages = controller_reference(root, recorder=recorder)
            raw = np.linspace(1.6, 2.6, 16)
            originals = []
            for step in range(150, 350):
                values = raw + np.sin(step / 10) * .0001
                originals.append(values.copy())
                recorder.write_frame(step / 10, values)
                session.push(step / 10, values)
            manifest = json.loads(recorder.close().read_text())
            saved = read_sensor_csv(recorder.csv_path)
            np.testing.assert_allclose(saved[SENSOR_COLUMNS].to_numpy(), originals, rtol=0, atol=1e-12)
            events = [json.loads(line) for line in recorder.guidance_path.read_text().splitlines()]
            self.assertTrue(any(event["kind"] == "empty_seat_baseline" for event in events))
            self.assertTrue(any(event["kind"] == "registration_complete" for event in events))
            self.assertTrue(any(event["kind"] == "guidance" for event in events))
            self.assertEqual(manifest["guidance_log"], recorder.guidance_path.name)
            self.assertTrue(all(event["model_snapshot"]["model_id"] == "synthetic-test-model" for event in events))


if __name__ == "__main__":
    unittest.main()
