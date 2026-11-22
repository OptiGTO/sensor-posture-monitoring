"""Registration runs independently of measurement recording and model inference."""
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from software import posture_app as app
from src.guidance_session import GuidanceSession
from src.personal_guidance import AlertSettings, CalibrationContext, CalibrationStore


class RegistrationModeTests(unittest.TestCase):
    def run_worker(self, root, *, cancel=False, gap=False):
        session = SimpleNamespace(
            _calibration_context=("registration-test", CalibrationContext()),
            _calibration_port="synthetic", _calibration_result=None,
            _messages=queue.Queue(), _stop=threading.Event())

        def frames(*args, **kwargs):
            controller = session._guidance_session
            controller.command("baseline")
            for step in range(200):
                if step == 46:
                    controller.command("register")
                if cancel and step == 80:
                    session._stop.set()
                timestamp = step / 10 + (2 if gap and step >= 80 else 0)
                yield timestamp, np.full(16, 1.2 if step < 31 else 2.2)

        with mock.patch.object(app, "STATE_ROOT", root), \
             mock.patch.object(app, "serial_frames", side_effect=frames), \
             mock.patch.object(app, "MeasurementRecorder", side_effect=AssertionError("No measurement recording")), \
             mock.patch.object(app, "PosturePredictor", side_effect=AssertionError("No model inference")):
            app.PostureApplication._run_calibration(session)
        return session

    def test_registration_saves_reusable_profile_without_measurement_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.run_worker(root)
            self.assertTrue(session._calibration_result.is_file())
            self.assertFalse(list(root.rglob("*.csv")))
            self.assertIsNone(session._guidance_session.engine)
            self.assertFalse([item for item in session._messages.queue if item[0] in {"guidance", "error"}])
            loaded = GuidanceSession(root, "registration-test", CalibrationContext(), AlertSettings(),
                                     lambda item: None, allow_registration=False)
            loaded.command("baseline")
            for step in range(31):
                loaded.push(step / 10, np.full(16, 1.2))
            self.assertIsNotNone(loaded.engine)
            previous = session._calibration_result.read_bytes()
            loaded.command("register")
            loaded.push(3.1, np.full(16, 2.2))
            self.assertIsNone(loaded.registration)
            self.assertEqual(previous, session._calibration_result.read_bytes())

    def test_cancelled_reregistration_preserves_previous_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.run_worker(root)
            previous = first._calibration_result.read_bytes()
            cancelled = self.run_worker(root, cancel=True)
            self.assertIsNone(cancelled._calibration_result)
            self.assertEqual(previous, first._calibration_result.read_bytes())

    def test_sensor_gap_discards_incomplete_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.run_worker(root, gap=True)
            self.assertIsNone(session._calibration_result)
            self.assertIsNone(CalibrationStore(root / "calibration_profiles").load("registration-test"))


if __name__ == "__main__":
    unittest.main()
