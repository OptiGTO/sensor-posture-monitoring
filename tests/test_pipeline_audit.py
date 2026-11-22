"""Failure recovery and integrity boundaries across the integrated pipeline."""
import io
import json
import queue
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
import pandas as pd

from software import posture_app as app
from software.posture_realtime import replay_csv
from src.app_paths import data_directory, writable_data_directory
from src.app_services import DatasetService, MeasurementRecorder, TrainingJob
from src.model_runtime import ModelRegistry, SessionModelSnapshot
from src.posture_core import SENSOR_COLUMNS, read_sensor_csv
from src.session_events import SessionEventQueue
from src.window_pipeline import WindowSpec, segment_sensor_frame
from src.personal_guidance import AlertSettings, CalibrationContext

ROOT = Path(__file__).resolve().parents[1]


class AuditRegressions(TestCase):
    def test_versioned_bundles_and_onedir_share_source_data(self):
        root = Path('/tmp/posture-project')
        for name in ('dist/release-123/Posture.app/Contents/MacOS/Posture', 'dist/release-123/Posture/Posture'):
            self.assertEqual(data_directory(root, executable=root / name), root.resolve() / 'data')

    def test_installed_app_can_fall_back_to_durable_user_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unavailable = root / 'file'
            unavailable.write_text('not a directory')
            self.assertEqual(writable_data_directory(unavailable / 'data', root / 'state'), root / 'state/data')

    def test_reserved_subjects_cannot_create_hidden_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            for subject in ('standard', 'Measurements', '__MACOSX'):
                with self.subTest(subject=subject), self.assertRaises(ValueError):
                    MeasurementRecorder(Path(tmp), mode='labeled', subject=subject, label='정자세')

    def test_measurement_metadata_discards_stale_label_and_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = MeasurementRecorder(root, mode='measurement', subject='previous person', label='정자세')
            for index in range(37): recorder.write_frame(index / 12, np.arange(16) + index * .1)
            manifest = recorder.close()
            payload = json.loads(manifest.read_text())
            self.assertEqual((payload['subject'], payload['ground_truth_label']), ('', ''))
            frame = pd.read_csv(recorder.csv_path).assign(label='정자세')
            frame.to_csv(recorder.csv_path, index=False)
            payload.update(mode='labeled', source_type='collected', label_status='labeled', training_role='include')
            manifest.write_text(json.dumps(payload))
            service = DatasetService(root)
            record = service.records()[0]
            self.assertFalse(service.is_training_eligible(record))
            service.set_status(record, '학습에 포함')
            self.assertEqual(json.loads(manifest.read_text())['training_role'], 'exclude')

    def test_failed_session_stays_failed_after_repeated_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = MeasurementRecorder(Path(tmp), mode='measurement')
            recorder.write_frame(0, range(16)); recorder.write_frame(1, range(16))
            manifest = recorder.close(error='sensor failed')
            recorder.close()
            self.assertEqual(json.loads(manifest.read_text())['session_status'], 'failed')
            self.assertEqual(json.loads(manifest.read_text())['session_error'], 'sensor failed')

    def test_ui_backlog_keeps_latest_telemetry_and_lossless_terminal_events(self):
        events = SessionEventQueue()
        for index in range(10000):
            events.put(('frame', index))
            events.put(('guidance', SimpleNamespace(popup_action=None, timestamp_s=index)))
        popup = SimpleNamespace(popup_action='show')
        events.put(('guidance', popup)); events.put(('error', 'failure')); events.put(('finished', None))
        self.assertEqual(events.qsize(), 5)
        self.assertEqual(events.get_nowait(), ('frame', 9999))
        self.assertEqual(events.get_nowait()[1].timestamp_s, 9999)
        self.assertIs(events.get_nowait()[1], popup)
        self.assertEqual(events.get_nowait(), ('error', 'failure'))
        self.assertEqual(events.get_nowait(), ('finished', None))
        for _ in range(5): events.task_done()
        events.join()

    def test_playback_accounts_for_time_spent_processing_frames(self):
        frame = pd.DataFrame([[index, *range(16)] for index in range(3)], columns=['elapsed_s', *SENSOR_COLUMNS])
        now = [0.]
        sleeps = []
        def sleep(delay):
            sleeps.append(delay); now[0] += delay
        with mock.patch('software.posture_realtime.read_sensor_csv', return_value=frame), \
             mock.patch('software.posture_realtime.time.monotonic', side_effect=lambda: now[0]), \
             mock.patch('software.posture_realtime.time.sleep', side_effect=sleep):
            for _ in replay_csv(Path('unused'), 1): now[0] += .3
        self.assertAlmostEqual(sum(sleeps), 1.4)

    def test_training_does_not_interpolate_across_disconnected_sensor(self):
        frame = pd.DataFrame([[value, *range(16)] for value in (0, .1, 2.9, 3.)], columns=['elapsed_s', *SENSOR_COLUMNS])
        with self.assertRaisesRegex(ValueError, '센서 공백'):
            segment_sensor_frame(frame, WindowSpec(3))

    def test_packaging_runtime_failure_still_emits_finished_and_persists_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = TrainingJob(ROOT, (), 42, state_root=Path(tmp))
            job.process = mock.Mock(stdout=io.StringIO('done\n'), args=['worker'])
            job.process.wait.return_value = 0
            with mock.patch.object(job, '_write_pending_candidate_manifests', side_effect=RuntimeError('broken estimator')):
                job._read()
            self.assertEqual(list(job.events.queue)[-1], ('finished', '1'))
            manifest = json.loads((job.run_dir / 'training.manifest.json').read_text())
            self.assertEqual(manifest['failure_reason'], 'broken estimator')

    def test_corrupt_active_state_and_repeated_activation_are_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(ROOT, Path(tmp))
            models = registry.discover()
            registry.active_path.write_text('[]')
            self.assertIsNotNone(registry.active_model(models))
            registry.activate(models[0])
            for index in range(20): registry.activate(models[index % 2])
            saved = json.loads(registry.active_path.read_text())
            self.assertNotIn('previous_active_model', saved['previous_active_model'])
            registry.activate(models[1])
            self.assertEqual(saved, json.loads(registry.active_path.read_text()))
            self.assertEqual(registry.rollback().artifact_id, models[0].artifact_id)

    def test_revalidation_preserves_existing_approval_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(ROOT, Path(tmp))
            model = registry.discover()[0]
            self.assertEqual(registry.validate_deployment(model).approval_status, 'approved')
            rejected = registry.reject(model, reason='reviewer declined')
            self.assertEqual(registry.validate_deployment(rejected).approval_status, 'rejected')

    def test_worker_always_saves_raw_data_even_if_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = ModelRegistry(ROOT, root)
            model = registry.discover()[0]
            snapshot = SessionModelSnapshot.create(model, session_id='cleanup-test', window_seconds=3, input_source='csv_replay')
            recorder = MeasurementRecorder(root, mode='measurement', model_snapshot=snapshot)
            session = SimpleNamespace(selected_model=model, registry=registry, session_snapshot=snapshot,
                _recorder=recorder, _session_input=('ignored.csv', 'auto'), _stop=threading.Event(), _messages=queue.Queue())
            class BrokenCleanup:
                def __iter__(self): return iter([(0, np.ones(16)), (.1, np.ones(16))])
                def close(self): raise RuntimeError('cleanup failed')
            with mock.patch.object(app, 'replay_csv', return_value=BrokenCleanup()):
                app.PostureApplication._run_session(session)
            self.assertEqual(list(session._messages.queue)[-1][0], 'finished')
            self.assertEqual(json.loads(recorder.manifest_path.read_text())['session_error'], 'cleanup failed')
            self.assertEqual(len(read_sensor_csv(recorder.csv_path)), 2)

    def test_labeled_live_capture_excludes_baseline_registration_and_empty_seat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = ModelRegistry(ROOT, root)
            model = registry.discover()[0]
            snapshot = SessionModelSnapshot.create(model, session_id='label-test', window_seconds=3, input_source='serial')
            recorder = MeasurementRecorder(root, mode='labeled', subject='qa', label='정자세', model_snapshot=snapshot)
            session = SimpleNamespace(selected_model=model, registry=registry, session_snapshot=snapshot,
                _recorder=recorder, _session_input=('', 'synthetic'), _stop=threading.Event(), _messages=queue.Queue(),
                _capture_requested=threading.Event(), _session_guidance_config=('qa', CalibrationContext(), AlertSettings()))
            def stream(*args, **kwargs):
                guidance = session._guidance_session
                guidance.command('baseline')
                for step in range(230):
                    if step == 46: guidance.command('register')
                    if step == 170: session._capture_requested.set()
                    yield step / 10, np.full(16, 1.2 if step < 31 else 2.2) + .0001 * np.sin(step + np.arange(16))
            with mock.patch.object(app, 'serial_frames', side_effect=stream), mock.patch.object(app, 'STATE_ROOT', root):
                app.PostureApplication._run_session(session)
            frame = read_sensor_csv(recorder.csv_path)
            self.assertEqual(frame['elapsed_s'].iloc[0], 17.)
            self.assertEqual(len(frame), 60)
            self.assertEqual(set(frame['label']), {'정자세'})
            self.assertFalse([item for item in session._messages.queue if item[0] == 'error'])
