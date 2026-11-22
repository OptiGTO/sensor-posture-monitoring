from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS
from validate_personalization import (
    ReplayRecord, assert_separate, compare_selected_model, evaluate_personalization, register_from_record,
    registration_plan, replay_record, summarize_records,
)


class PersonalizationEvaluationTests(unittest.TestCase):
    def record(self, name, label=CLASS_NAMES[0], *, subject="S1", digest=None, sensor=None):
        return ReplayRecord(Path("/tmp") / name, name, subject, label, 200, 20.0,
                            digest or name, sensor or name)

    def test_default_does_not_assign_any_record(self):
        self.assertEqual(registration_plan([self.record("correct.csv")]), [])
        with self.assertRaisesRegex(ValueError, "해제"):
            evaluate_personalization(Path("/tmp/unused"))

    def test_registration_choice_is_fixed_and_records_do_not_overlap(self):
        first, second = self.record("S1/correct_001.csv"), self.record("S1/correct_002.csv")
        poor = self.record("S1/incorrect.csv", CLASS_NAMES[1])
        for records in ([poor, second, first], [second, first, poor]):
            [(registration, evaluation)] = registration_plan(records, enabled=True)
            self.assertEqual(registration, first)
            self.assertEqual(set(evaluation), {second, poor})

    def test_different_name_or_changed_label_does_not_hide_registration_leakage(self):
        registration = self.record("registration.csv", sensor="same-sensor-data")
        copy = self.record("evaluation.csv", CLASS_NAMES[1], sensor="same-sensor-data")
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_separate(registration, [copy])
        with self.assertRaisesRegex(ValueError, "same subject"):
            assert_separate(registration, [self.record("other.csv", subject="S2")])

    def test_one_correct_record_never_reused_as_correct_evaluation(self):
        first, poor = self.record("correct.csv"), self.record("incorrect.csv", CLASS_NAMES[1])
        [(registration, evaluation)] = registration_plan([first, poor], enabled=True)
        self.assertEqual(registration, first)
        self.assertEqual(evaluation, [poor])

    def test_metrics_keep_uncertainty_and_unavailable_continuous_evidence_explicit(self):
        report = summarize_records([
            {"duration_s": 20.0, "ground_truth": CLASS_NAMES[0],
             "state_duration_s": {"correct": 10.0, "incorrect": 4.0, "uncertain": 6.0},
             "popup_timestamps_s": [18.0]},
            {"duration_s": 10.0, "ground_truth": CLASS_NAMES[1],
             "state_duration_s": {"correct": 2.0, "incorrect": 5.0, "uncertain": 3.0},
             "popup_timestamps_s": []},
        ])
        self.assertAlmostEqual(report["incorrect_missed_as_correct_rate"], 0.2)
        self.assertAlmostEqual(report["correct_classified_incorrect_rate"], 0.2)
        self.assertEqual(report["state_duration_s"]["uncertain"], 9.0)
        self.assertEqual(report["false_alerts_per_correct_hour"], 180.0)
        self.assertIsNone(report["actual_posture_onset_to_popup_s"])
        self.assertIsNone(report["actual_correct_return_to_warning_dismissal_s"])
        self.assertIsNone(summarize_records([])["false_alerts_per_correct_hour"])

    def test_failed_registration_is_reported_without_retry_or_reusing_evaluation(self):
        records = [self.record("correct_001.csv"), self.record("correct_002.csv"),
                   self.record("incorrect.csv", CLASS_NAMES[1])]
        with mock.patch("validate_personalization.audit_dataset", return_value=(records, {"records": []})), \
             mock.patch("validate_personalization.register_from_record", side_effect=ValueError("Movement")) as register:
            report, _ = evaluate_personalization(Path("/tmp/unused"), allow_retrospective=True)
        register.assert_called_once_with(records[0])
        self.assertEqual(report["failed_registration_count"], 1)
        self.assertEqual(report["registration_outcomes"][0]["reason"], "Movement")
        self.assertEqual(report["modes"]["deployment"]["summary"]["record_count"], 0)

    def test_real_core_replays_separate_stable_record_and_accounts_for_entire_clip(self):
        timestamps = np.arange(241) / 12.0
        values = np.full((len(timestamps), 16), 10.0)
        values += np.sin(timestamps[:, None]) * 0.001
        frame = pd.DataFrame(values, columns=SENSOR_COLUMNS).assign(elapsed_s=timestamps)
        with mock.patch("validate_personalization.read_sensor_csv", return_value=frame):
            profile = register_from_record(self.record("registration.csv"))
            result = replay_record(self.record("evaluation.csv"), profile, "deployment")
        self.assertAlmostEqual(profile.stability_metrics["duration_s"], 10.0)
        self.assertAlmostEqual(sum(result["state_duration_s"].values()), 20.0)
        self.assertGreater(result["state_duration_s"]["correct"], 19.0)
        self.assertEqual(result["popup_timestamps_s"], [])

    def test_before_after_comparison_rejects_tampered_models_and_subject_leakage_before_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.joblib"
            model.write_bytes(b"not-loaded-in-this-test")
            evidence = root / "evaluation.json"
            evidence.write_text(json.dumps({"partitions": {
                "train": {"subjects": ["same"]}, "validation": {"subjects": []},
                "final_evaluation": {"subjects": ["same"]}},
                "final_evaluation_used_for_selection": False}))
            manifest = {"artifact_file": model.name, "artifact_sha256": "wrong", "components": [
                {"role": "evaluation", "file": evidence.name,
                 "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}]}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "model hash"):
                compare_selected_model({}, path)
            manifest["artifact_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "independent final subjects"):
                compare_selected_model({}, path)


if __name__ == "__main__":
    unittest.main()
