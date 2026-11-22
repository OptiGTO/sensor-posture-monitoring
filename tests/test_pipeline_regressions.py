"""Regression coverage for collection, dataset selection, training, and inference."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
import json
import tempfile
import threading

import numpy as np
import pandas as pd

from software.posture_realtime import replay_csv, serial_frames
from src.app_services import CSV_COLUMNS, DatasetService, MeasurementRecorder, OccupancyConfig, OccupancyDetector, StandardProfile, TrainingJob, quality_check
from src.inference_core import PosturePredictor
from src.posture_core import load_dataset, make_subject_pairs, read_sensor_csv, training_csv_paths
from src.realtime import LiveConfig, RealtimePosturePredictor
from src.window_pipeline import WindowSpec, load_window_dataset


ROOT = Path(__file__).resolve().parents[1]


def write_record(path, timestamps=(0, 1, 2, 3), label="정자세"):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[t, label, *[v + t for v in range(16)]] for t in timestamps], columns=CSV_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
    return path


class RecordingRegressions(TestCase):
    def test_same_second_sessions_keep_both_csvs_and_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("src.app_services.datetime") as clock:
            clock.now.return_value = datetime(2026, 9, 5, 12, 0, 0)
            a = MeasurementRecorder(Path(tmp), mode="measurement")
            a.write_frame(0, range(16))
            original = a.csv_path.read_bytes()
            b = MeasurementRecorder(Path(tmp), mode="measurement")
            b.write_frame(1, range(16))
            a.close(); b.close()
            self.assertNotEqual(a.csv_path, b.csv_path)
            self.assertEqual(a.csv_path.read_bytes(), original)
            self.assertNotEqual(a.prediction_path, b.prediction_path)
            self.assertTrue(a.manifest_path.exists() and b.manifest_path.exists())

    def test_recorder_rejects_nonfinite_and_rounded_duplicate_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = MeasurementRecorder(Path(tmp), mode="measurement")
            recorder.write_frame(0, range(16))
            for timestamp in (float("nan"), float("inf"), -1, 0.0000001):
                with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                    recorder.write_frame(timestamp, range(16))
            recorder.close()
            self.assertEqual(recorder.rows, 1)

    def test_duration_uses_sensor_time_even_during_fast_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = MeasurementRecorder(Path(tmp), mode="measurement")
            recorder.write_frame(0, range(16))
            recorder.write_frame(20, range(16))
            result = json.loads(recorder.close().read_text())
            self.assertEqual(result["duration_s"], 20)

    def test_subject_cannot_escape_data_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                MeasurementRecorder(Path(tmp), mode="labeled", subject="../outside", label="정자세")


class DatasetRegressions(TestCase):
    def test_multiple_roots_list_once_and_keep_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            state, legacy = root / 'state/data', root / 'project/data'
            paths = [write_record(directory / 'person/sample.csv') for directory in (state, legacy)]
            write_record(legacy / 'standard/baseline.csv')
            write_record(legacy / 'person/sample.imu.csv')
            measurement = write_record(legacy / 'measurements/person/record.csv')
            alias = root / 'alias'
            alias.symlink_to(legacy, target_is_directory=True)
            before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob('*') if path.is_file()}
            service = DatasetService(root / 'state', data_directory=state,
                                     additional_data_directories=(legacy, alias, legacy / 'person'))
            records = service.records()
            self.assertEqual({record.csv_path for record in records}, {*paths, measurement})
            self.assertEqual(len(records), 3)
            self.assertEqual(sum(service.is_training_eligible(record) for record in records), 2)
            self.assertEqual(service.records(), records)
            self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns)
                                     for path in root.rglob('*') if path.is_file()})
            write_record(legacy / 'person/new.csv')
            self.assertEqual(len(service.records()), 4)

    def test_duplicate_time_is_failure_and_near_complete_tail_counts_as_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_record(Path(tmp) / "sample.csv", (0, 1, 1, 2, 3))
            self.assertEqual(quality_check(path)["status"], "fail")
            write_record(path, (0, 0.97, 1.94, 2.9))
            self.assertEqual(quality_check(path)["window_count"], 1)

    def test_csv_and_dataframe_paths_reject_invalid_time_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            for times in ((0, 0), (2, 1), (0, float("inf"))):
                path = write_record(Path(tmp) / "bad.csv", times)
                with self.subTest(times=times), self.assertRaises(ValueError):
                    read_sensor_csv(path)

    def test_all_training_loaders_share_exclusion_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = write_record(root / "person" / "good.csv")
            for folder in ("standard", "measurements", "__MACOSX"):
                write_record(root / folder / "ignored.csv", label="")
            write_record(root / "person" / "good.imu.csv")
            excluded = write_record(root / "person" / "evaluation.csv")
            excluded.with_suffix(".manifest.json").write_text(json.dumps({"training_role": "evaluation_only"}))
            self.assertEqual(training_csv_paths(root), [valid])
            self.assertEqual(len(load_dataset(root)[1]), 1)
            self.assertEqual(len(load_window_dataset(root, WindowSpec(3))[2]), 1)

    def test_missing_or_mixed_labels_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_record(root / "data" / "person" / "sample.csv")
            frame = pd.read_csv(path)
            for bad_label in (None, "뒤로 기대기"):
                frame.loc[1, "label"] = bad_label
                frame.to_csv(path, index=False)
                record = DatasetService(root).records()[0]
                self.assertFalse(DatasetService.is_training_eligible(record))
                with self.assertRaises(ValueError):
                    load_window_dataset(root / "data", WindowSpec(3))

    def test_cache_reuses_unchanged_records_and_refreshes_csv_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_record(root / "data" / "person" / "sample.csv")
            service = DatasetService(root)
            first = service.records()[0]
            self.assertEqual(first.posture, "정자세")
            with mock.patch("src.app_services.read_sensor_csv", side_effect=AssertionError("unnecessary reread")):
                self.assertEqual(service.records(), [first])
            manifest = path.with_suffix(".manifest.json")
            manifest.write_text(json.dumps({"training_role": "exclude"}))
            self.assertFalse(service.is_training_eligible(service.records()[0]))
            write_record(path, (0, 1, 1, 2, 3))
            self.assertEqual(service.records()[0].qc_status, "fail")
            path.unlink()
            self.assertEqual(service.records(), [])

    def test_old_manifest_pass_does_not_override_corrupted_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_record(root / "data" / "person" / "sample.csv", (0, 1, 1, 2, 3))
            path.with_suffix(".manifest.json").write_text(json.dumps({"rows": 5, "qc": {"status": "pass"}}))
            record = DatasetService(root).records()[0]
            self.assertEqual(record.qc_status, "fail")
            self.assertFalse(DatasetService.is_training_eligible(record))

    def test_staging_rechecks_selection_and_rejects_existing_staging_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_record(root / "data" / "person" / "sample.csv")
            service = DatasetService(root)
            selected = service.records()
            service.set_status(selected[0], "학습에서 제외")
            with self.assertRaises(ValueError):
                service.stage_for_training(selected, root / "staged")
            write_record(root / "stale" / "old.csv")
            with self.assertRaisesRegex(ValueError, "비어 있지"):
                service.stage_for_training(selected, root / "stale")

    def test_listing_omits_imu_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_record(root / "data" / "person" / "sample.csv")
            write_record(root / "data" / "person" / "sample.imu.csv")
            self.assertEqual(len(DatasetService(root).records()), 1)


class StreamRegressions(TestCase):
    def test_occupancy_smoothing_filters_short_pressure_spikes(self):
        profile = StandardProfile(np.zeros(16), np.ones(16), "test", 1)
        detector = OccupancyDetector(profile, OccupancyConfig(enter_hold_seconds=0))
        for timestamp in range(5):
            detector.update(timestamp, np.zeros(16))
        detector.update(5, np.full(16, 3.0))
        detector.update(6, np.full(16, 3.0))
        self.assertFalse(detector.occupied)

    def test_stop_without_connected_port_interrupts_reconnect_wait(self):
        stop = threading.Event()
        waiting = threading.Event()
        def no_ports():
            waiting.set()
            return []
        with mock.patch("serial.tools.list_ports.comports", side_effect=no_ports):
            stream = serial_frames("auto", 115200, 30, stop_event=stop)
            worker = threading.Thread(target=lambda: list(stream), daemon=True)
            worker.start()
            self.assertTrue(waiting.wait(1))
            stop.set(); worker.join(1)
            self.assertFalse(worker.is_alive())

    def test_stop_while_connected_but_silent_closes_serial_port(self):
        stop = threading.Event()
        connection = mock.MagicMock()
        connection.__enter__.return_value = connection
        def empty_line():
            stop.set()
            return b""
        connection.readline.side_effect = empty_line
        with mock.patch("serial.Serial", return_value=connection):
            self.assertEqual(list(serial_frames("test", 115200, 30, stop_event=stop)), [])
        connection.__exit__.assert_called_once()

    def test_replay_wait_can_be_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_record(Path(tmp) / "replay.csv", (0, 30))
            stop = threading.Event()
            stream = replay_csv(path, 1, stop_event=stop)
            next(stream)
            timer = threading.Timer(0.02, stop.set)
            timer.start()
            self.assertEqual(list(stream), [])
            timer.join()

    def test_empty_seat_reset_discards_previous_probability_history(self):
        predictor = SimpleNamespace(labels=("a", "b"), predict_frame=mock.Mock(side_effect=[np.array([1., 0.]), np.array([0., 1.])]))
        engine = RealtimePosturePredictor(predictor, LiveConfig(window_seconds=1, minimum_window_seconds=1, smoothing_history=3))
        for t in (0, 0.5, 1):
            engine.push(t, range(16))
        engine.reset("empty seat")
        for t in (2, 2.5, 3):
            event = engine.push(t, range(16))
        self.assertEqual(event.posture, "b")
        self.assertEqual(event.confidence, 1.0)

    def test_prediction_builds_one_snapshot(self):
        predictor = SimpleNamespace(labels=("a", "b"), predict_frame=lambda frame: np.array([1., 0.]))
        engine = RealtimePosturePredictor(predictor, LiveConfig(window_seconds=1, minimum_window_seconds=1))
        with mock.patch.object(engine.buffer, "snapshot", wraps=engine.buffer.snapshot) as snapshot:
            for t in (0, 0.5, 1):
                event = engine.push(t, range(16))
        self.assertIsNotNone(event)
        snapshot.assert_called_once()

    def test_window_prediction_segments_once_and_keeps_probabilities(self):
        from src.model_runtime.sklearn_adapter import segment_sensor_frame
        predictor = PosturePredictor(ROOT / "artifacts" / "posture_classifier_window_3s.joblib")
        frame = read_sensor_csv(ROOT / "example" / "sample_input.csv")
        expected = predictor.predict_frame(frame)
        with mock.patch("src.model_runtime.sklearn_adapter.segment_sensor_frame", wraps=segment_sensor_frame) as segment:
            probabilities, metadata = predictor.predict_windows(frame)
        segment.assert_called_once()
        self.assertEqual(len(probabilities), len(metadata))
        np.testing.assert_allclose(probabilities.mean(axis=0), expected)


class TrainingRegressions(TestCase):
    def test_training_saves_graphs_without_a_display(self):
        from train_window_models import save_figures
        comparison = pd.DataFrame([{"evaluation": "fixed", "window_seconds": 3, "window_macro_f1": 0.5, "record_macro_f1": 0.6}])
        result = {3: {"record_y": np.arange(7), "record_prediction": np.arange(7)}}
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            save_figures(destination, comparison, result)
            for name in ("window_length_macro_f1_comparison.png", "window_3s_record_confusion_normalized.png"):
                self.assertTrue((destination / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_five_subjects_are_each_held_out_once_and_old_even_split_stays_same(self):
        for count in (4, 5, 14):
            subjects = [f"s{i}" for i in range(count)]
            pairs = make_subject_pairs(subjects)
            self.assertEqual(sorted(sum(pairs, [])), sorted(subjects))
            self.assertTrue(all(len(pair) in (1, 2) for pair in pairs))
            if count % 2 == 0:
                expected = sorted(subjects)
                np.random.default_rng(20260826).shuffle(expected)
                self.assertEqual(pairs, [expected[i:i + 2] for i in range(0, count, 2)])
        with self.assertRaisesRegex(ValueError, "최소 4명"):
            make_subject_pairs(["a", "b", "c"])

    def test_training_rejects_insufficient_subjects_before_starting_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_record(root / "data" / "person" / "sample.csv")
            job = TrainingJob(root, DatasetService(root).records(), 42, windows=(3,))
            with mock.patch("subprocess.Popen") as process, self.assertRaisesRegex(ValueError, "최소 4명"):
                job.start()
            process.assert_not_called()

    def test_short_record_cannot_start_long_window_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_record(root / "data" / "person" / "sample.csv")
            job = TrainingJob(root, DatasetService(root).records(), 42, windows=(20,))
            with mock.patch("subprocess.Popen") as process, self.assertRaisesRegex(ValueError, "20초"):
                job.start()
            process.assert_not_called()

    def test_simultaneous_training_jobs_use_different_directories(self):
        a = TrainingJob(ROOT, (), 42)
        b = TrainingJob(ROOT, (), 42)
        self.assertNotEqual(a.run_dir, b.run_dir)

    def test_nested_training_accepts_odd_subject_count(self):
        from train_window_models import evaluate_nested
        groups = np.repeat(["a", "b", "c", "d", "e"], 8)
        y = np.tile([0, 1], 20)
        features = np.column_stack([y, y * 2, y * 3]).astype(float)
        with mock.patch("train_window_models.SVM_CANDIDATES", [{"C": 1.0, "gamma": "scale"}]):
            result = evaluate_nested(features, y, groups, np.arange(40).astype(str), {g: g for g in groups}, 42)
        self.assertEqual(len(result["folds"]), 3)
        np.testing.assert_allclose(result["window_probability"].sum(axis=1), 1.0)

    def test_window_spec_rejects_invalid_resampling_settings(self):
        for args in ((0,), (3, 0), (3, 12, 0), (float("inf"),), (3, 12, 1.1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                WindowSpec(*args)


class GuiLifecycleRegressions(TestCase):
    def test_close_waits_for_measurement_and_training_before_destroying(self):
        from software.posture_app import PostureApplication
        worker = mock.Mock()
        worker.is_alive.return_value = True
        app = SimpleNamespace(_stop=threading.Event(), _worker=worker, training_job=SimpleNamespace(is_running=False, cancel=mock.Mock()), after=mock.Mock(), destroy=mock.Mock())
        app._finish_close = lambda: PostureApplication._finish_close(app)
        PostureApplication._close(app)
        self.assertTrue(app._stop.is_set())
        app.destroy.assert_not_called()
        worker.is_alive.return_value = False
        app.training_job.is_running = True
        PostureApplication._finish_close(app)
        app.destroy.assert_not_called()
        app.training_job.is_running = False
        PostureApplication._finish_close(app)
        app.destroy.assert_called_once()

    def test_finished_message_is_handled_after_user_requests_stop(self):
        import queue
        from software.posture_app import PostureApplication
        messages = queue.Queue()
        messages.put(("finished", Path("result.manifest.json")))
        app = SimpleNamespace(_closing=False, _messages=messages, _stop=threading.Event(), finish_measurement=mock.Mock())
        app._stop.set()
        PostureApplication.poll_messages(app)
        app.finish_measurement.assert_called_once_with(Path("result.manifest.json"))
