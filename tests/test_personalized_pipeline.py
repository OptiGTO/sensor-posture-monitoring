"""Routing, provenance, leakage guards and live worker integration (synthetic sensors)."""
from dataclasses import replace
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from src.app_services import DatasetService, MeasurementRecorder, TrainingJob
from src.guidance_session import GuidanceSession
from src.personal_guidance import AlertSettings, CalibrationContext, CalibrationSession, CalibrationStore
from src.personalized_model import PersonalPostureModel, features
from src.personalized_guidance import PersonalGuidanceEngine
from src.personalized_training import make_bundle, metric, split_subjects
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS


class FixedScores:
    def __init__(self, classes, scores):
        self.classes_ = np.asarray(classes)
        self.scores = scores
        self.calls = 0

    def predict_proba(self, x):
        self.calls += 1
        return np.tile(self.scores, (len(x), 1))


def reference(user="synthetic-user"):
    context = CalibrationContext(empty_seat_center=(1.,) * 16)
    registration = CalibrationSession(user, context, guidance_confirmed=True, empty_seat_checked=True)
    for step in range(122):
        registration.push(step / 12, np.full(16, 2.) + .00001 * np.sin(step))
    return registration.finish()


def model(score=.9, detail=None):
    return PersonalPostureModel(make_bundle(FixedScores([0, 1], [1-score, score]),
        FixedScores(range(1, 7), detail or [.9, .02, .02, .02, .02, .02]), [.4, .6]),
        identity={"model_sha256": "test-hash", "manifest_sha256": "test-manifest"})


class PersonalizedPipelineTests(unittest.TestCase):
    def test_correct_and_unknown_never_execute_detail_classifier(self):
        profile = reference()
        for score, state in ((.1, "correct"), (.5, "uncertain")):
            classifier = model(score)
            event = classifier.predict(np.full((36, 16), 2.), profile)
            self.assertEqual(event.state, state)
            self.assertEqual(classifier.detail.calls, 0)
            self.assertFalse(event.stage2_executed)
        classifier = model()
        event = classifier.predict(np.full((36, 16), 2.), profile)
        self.assertEqual(event.posture, CLASS_NAMES[1])
        self.assertEqual(classifier.detail.calls, 1)

    def test_low_detail_confidence_keeps_incorrect_state_without_forced_label(self):
        classifier = model(detail=[1/6] * 6)
        event = classifier.predict(np.full((36, 16), 2.), reference())
        self.assertEqual(event.state, "incorrect")
        self.assertIsNone(event.posture)

    def test_class_order_and_window_contract_rejected(self):
        classifier = model()
        with self.assertRaises(ValueError): classifier.predict(np.full((35, 16), 2.), reference())
        bundle = make_bundle(FixedScores([1, 0], [.1, .9]), classifier.detail, [.4, .6])
        with self.assertRaises(ValueError): PersonalPostureModel(bundle)

    def test_offset_invariance_and_personal_reference_sensitivity(self):
        profile = reference()
        raw = np.full((36, 16), 2.)
        x1, _ = features(raw, profile)
        shifted = replace(profile, context=replace(profile.context, empty_seat_center=(11.,) * 16),
                          pressure_center=tuple(np.asarray(profile.pressure_center) + 10))
        np.testing.assert_allclose(x1, features(raw + 10, shifted)[0])
        center = np.array(profile.ratio_center); center[:8] += .01; center[8:] -= .01
        self.assertFalse(np.allclose(x1, features(raw, replace(profile, ratio_center=tuple(center)))[0]))

    def test_live_window_is_shared_alert_continues_and_gap_clears_cached_prediction(self):
        classifier = model()
        engine = PersonalGuidanceEngine(reference(), classifier, settings=AlertSettings.for_mode("development"))
        events = [engine.update(step / 12, np.full(16, 2.)) for step in range(241)]
        self.assertIsNone(events[35].window_timestamp_s)
        self.assertEqual(events[36].window_timestamp_s, 3.)
        self.assertTrue(any(e.popup_action == "show" for e in events))
        # One head invocation per second, not one per sensor sample.
        self.assertEqual(classifier.detail.calls, 18)
        event = engine.update(22, np.full(16, 2.))
        self.assertEqual(event.state, "input_error")
        self.assertEqual(event.popup_action, "dismiss")
        self.assertIsNone(event.gate_score)
        self.assertIsNone(event.window_timestamp_s)
        self.assertEqual(engine.update(22.1, np.full(16, 2.)).state, "insufficient")
        self.assertEqual(engine.update(22.1, np.full(16, 2.)).state, "input_error")
        self.assertEqual(engine.update(22.2, np.full(16, 2.), seated=False).state, "empty")

    def test_worker_uses_personal_model_identity_and_never_calls_legacy_predictor(self):
        from software import posture_app as app
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = reference(); CalibrationStore(root / "calibration_profiles").save(profile)
            classifier = model()
            snapshot = classifier.snapshot("test-session")
            recorder = MeasurementRecorder(root, mode="measurement", model_snapshot=snapshot)
            session = SimpleNamespace(selected_model=object(), _recorder=recorder, session_snapshot=snapshot,
                _session_personal_model=classifier, _session_guidance_config=(profile.user_id, profile.context, AlertSettings.for_mode("development")),
                _session_input=("", "test"), _stop=threading.Event(), _messages=queue.Queue(), _allow_registration=False)
            closed = []
            def frames(*args, **kwargs):
                try:
                    session._guidance_session.command("baseline")
                    for step in range(401):
                        yield step / 12, np.full(16, 1. if step < 38 else 2.)
                finally: closed.append(True)
            with mock.patch.object(app, "STATE_ROOT", root), mock.patch.object(app, "serial_frames", side_effect=frames), \
                 mock.patch.object(app, "PosturePredictor", side_effect=AssertionError("Legacy inference must not run")):
                app.PostureApplication._run_session(session)
            errors = [payload for kind, payload in session._messages.queue if kind == "error"]
            self.assertEqual(errors, [])
            self.assertTrue(closed)
            self.assertEqual(recorder.rows, 401)
            predictions = [json.loads(line) for line in recorder.prediction_path.read_text().splitlines()]
            self.assertTrue(predictions)
            self.assertTrue(all(p["model_snapshot"]["model_family"] == "personalized_hierarchy" for p in predictions))
            self.assertEqual(len({p["window_timestamp_s"] for p in predictions}), len(predictions))
            self.assertTrue(any(p["stage2_executed"] for p in predictions))

    def test_unknowns_remain_in_metrics_and_subjects_are_disjoint(self):
        y = np.array([0, 0, 1, 2])
        scores = metric(y, np.array([0, -1, -1, 2]))
        self.assertEqual(scores["accuracy"], .5)
        self.assertEqual(scores["unknown_rate"], .5)
        groups = np.repeat(["a", "b", "c", "d"], 7)
        labels = np.tile(np.arange(7), 4)
        partitions, masks = split_subjects(groups, labels, 42)
        self.assertFalse(set(partitions["train"]) & set(partitions["final_evaluation"]))
        self.assertTrue(np.all(sum(masks.values()) == 1))
        with self.assertRaises(ValueError): split_subjects(groups[:21], labels[:21], 42)

    def test_staging_preserves_reference_snapshot_but_measurement_is_ineligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = model().snapshot("staging")
            profile = reference()
            recorder = MeasurementRecorder(root, mode="labeled", subject="subject", label=CLASS_NAMES[0], model_snapshot=snapshot,
                                           guidance_metadata={"reference_profile": profile.as_dict()})
            for step in range(241): recorder.write_frame(step / 12, np.full(16, 2.) + step * .00001)
            recorder.close(status="학습 대기열에 추가")
            service = DatasetService(root)
            record = next(r for r in service.records() if r.csv_path == recorder.csv_path)
            service.set_status(record, "학습 대기열에 추가")
            staged = service.stage_for_training([record], root / "staging")
            manifest = next(staged.glob("*/*.manifest.json"))
            self.assertEqual(json.loads(manifest.read_text())["guidance"]["reference_profile"], json.loads(json.dumps(profile.as_dict())))
            measurement = MeasurementRecorder(root, mode="measurement", model_snapshot=snapshot)
            measurement.write_frame(0, np.full(16, 2.)); measurement.close()
            record = next(r for r in service.records() if r.csv_path == measurement.csv_path)
            with self.assertRaises(ValueError): service.stage_for_training([record], root / "rejected")


class StoredPersonalModelTests(unittest.TestCase):
    def test_artifact_tampering_and_incomplete_run_are_rejected_before_unpickling(self):
        from src.personalized_model import sha256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "personalized.joblib"; artifact.write_bytes(b"broken pickle")
            report = root / "personalized.report.json"; report.write_text("{}")
            path = root / "personalized.manifest.json"
            payload = {"schema_version": 1, "kind": "personalized_hierarchy", "status": "experimental_validated",
                       "model": {"file": artifact.name, "sha256": "wrong"},
                       "report": {"file": report.name, "sha256": sha256(report)}}
            path.write_text(json.dumps(payload))
            with mock.patch("src.personalized_model.joblib.load", side_effect=AssertionError("Must validate hash first")):
                with self.assertRaises(ValueError): PersonalPostureModel.load(path)
                (root / ".incomplete").touch()
                with self.assertRaises(ValueError): PersonalPostureModel.load(path)


class PersonalizedTrainingTests(unittest.TestCase):
    def dataset(self, root):
        import pandas as pd
        from src.app_services import CSV_COLUMNS
        for subject in range(4):
            directory = root / f"S{subject}"; directory.mkdir(parents=True)
            for label in range(7):
                for repeat in range(2 if label == 0 else 1):
                    times = np.arange(241) / 12
                    sensor = np.ones((241, 16)) * (2 + subject * .2)
                    sensor[:, label:label+3] += label * .1
                    sensor += .001 * np.sin(times[:, None] * (.1 + repeat * .01) + np.arange(16)[None, :] * .17)
                    frame = pd.DataFrame(np.column_stack((times, np.full(len(times), CLASS_NAMES[label]), sensor)), columns=CSV_COLUMNS)
                    frame.to_csv(directory / f"{label}_{repeat}.csv", index=False)

    def test_training_reserves_entire_registration_and_checks_loaded_runtime_parity(self):
        from src.personalized_training import prepare_data, train
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data_root = root / "data"
            self.dataset(data_root)
            with self.assertRaisesRegex(ValueError, "연결되지"):
                prepare_data(data_root)
            prepared = prepare_data(data_root, allow_retrospective=True)
            reserved = {row["file"] for row in prepared["registrations"]}
            self.assertFalse(reserved & {row["file"] for row in prepared["sources"]})
            repeated = prepare_data(data_root, allow_retrospective=True)
            self.assertEqual(prepared["sources"], repeated["sources"])
            path = train(data_root, root / "model", 42, allow_retrospective=True)
            restored = PersonalPostureModel.load(path)
            report = json.loads((path.parent / "personalized.report.json").read_text())
            self.assertFalse(report["final_evaluation_used_for_selection"])
            self.assertEqual(report["serialization_parity"], "passed")
            for index, sample in enumerate(prepared["windows"]):
                if prepared["groups"][index] not in report["partitions"]["final_evaluation"]: continue
                result = restored.predict(sample, prepared["profiles"][index])
                expected = next(row["personal_hierarchy"] for row in report["final_windows"]
                                if row["file"] == prepared["rows"][index]["file"]
                                and row["window_end_s"] == prepared["rows"][index]["window_end_s"])
                self.assertEqual(CLASS_NAMES.index(result.posture) if result.posture else -1, expected)
    def test_personal_training_job_completes_and_is_not_a_generic_approved_model(self):
        import time
        from src.personalized_model import available_models
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.dataset(root / "data")
            records = DatasetService(root).records()
            job = TrainingJob(Path(__file__).resolve().parents[1], records, 42, windows=(3,),
                              state_root=root, personalized=True, allow_retrospective=True)
            job.start()
            deadline = time.monotonic() + 30
            while job.is_running and time.monotonic() < deadline: time.sleep(.05)
            if job.is_running:
                job.cancel()
                # The reader persists cancellation metadata. Let it finish
                # before TemporaryDirectory removes its run directory.
                if job.process:
                    job.process.wait(timeout=10)
                if job._reader:
                    job._reader.join(timeout=10)
                self.fail("Personal training worker timed out")
            events = list(job.events.queue)
            self.assertEqual(events[-1], ("finished", "0"), events[-8:])
            self.assertEqual(len(available_models(root)), 1)
            manifest = json.loads((job.run_dir / "training.manifest.json").read_text())
            self.assertEqual(manifest["approval_status"], "experimental_only")
            self.assertFalse((root / "artifacts" / "active_model.json").exists())
            self.assertTrue(job.validation_results[0]["passed"])

    def test_personal_training_cancel_does_not_publish_a_selectable_artifact(self):
        import time
        from src.personalized_model import available_models
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.dataset(root / "data")
            job = TrainingJob(Path(__file__).resolve().parents[1], DatasetService(root).records(), 42,
                              windows=(3,), state_root=root, personalized=True, allow_retrospective=True)
            job.start(); job.cancel()
            deadline = time.monotonic() + 10
            while job.is_running and time.monotonic() < deadline: time.sleep(.05)
            self.assertFalse(job.is_running)
            self.assertEqual(available_models(root), [])
            self.assertTrue((job.run_dir / ".incomplete").exists())


if __name__ == "__main__": unittest.main()
