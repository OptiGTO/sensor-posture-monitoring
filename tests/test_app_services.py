from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.app_services import (
    CSV_COLUMNS, DatasetService, MeasurementRecorder, OccupancyDetector,
    SensorPacket, StandardProfile, TrainingJob, TrainingWizardState,
    build_standard_profile, occupancy_profile_for_input,
)
from src.posture_core import SENSOR_COLUMNS
from src.window_pipeline import WindowSpec, load_window_dataset
from src.model_runtime import ModelRegistry, SessionModelSnapshot


class AppServiceTests(unittest.TestCase):
    def test_measurement_only_csv_preserves_contract_and_keeps_label_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = MeasurementRecorder(Path(temporary), mode="measurement")
            recorder.write_frame(0.0, range(16))
            recorder.write_prediction({"posture": "정자세", "confidence": 0.9})
            manifest = recorder.close("측정 전용")
            frame = pd.read_csv(recorder.csv_path, encoding="utf-8-sig")
            self.assertEqual(tuple(frame.columns), CSV_COLUMNS)
            self.assertTrue(frame["label"].isna().all())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_status"], "측정 전용")
            self.assertEqual(payload["source_type"], "measurement")
            self.assertEqual(payload["label_status"], "unlabeled")
            self.assertEqual(payload["training_role"], "exclude")

    def test_session_snapshot_is_written_without_changing_raw_csv_columns(self) -> None:
        descriptor = ModelRegistry(Path(__file__).resolve().parents[1]).discover()[0]
        snapshot = SessionModelSnapshot.create(descriptor, session_id="session-test", window_seconds=descriptor.window_seconds, input_source="csv_replay")
        with tempfile.TemporaryDirectory() as temporary:
            recorder = MeasurementRecorder(Path(temporary), mode="measurement", model_snapshot=snapshot)
            recorder.write_frame(0.0, range(16))
            manifest = recorder.close()
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_snapshot"]["artifact_id"], descriptor.artifact_id)
            self.assertEqual(payload["model_snapshot"]["selected_window_seconds"], descriptor.window_seconds)
            self.assertEqual(pd.read_csv(recorder.csv_path, encoding="utf-8-sig").columns.tolist(), list(CSV_COLUMNS))

    def test_accelerometer_is_written_as_a_separate_timestamped_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = MeasurementRecorder(Path(temporary), mode="measurement")
            recorder.write_imu(0.0, (0.1, 0.2, 0.3))
            manifest = recorder.close()
            self.assertEqual(pd.read_csv(recorder.imu_path, encoding="utf-8-sig").columns.tolist(), ["elapsed_s", "ax", "ay", "az"])
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["imu"], recorder.imu_path.name)

    def test_standard_profile_and_hysteresis_use_discovered_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            standard = Path(temporary) / "data" / "standard"
            standard.mkdir(parents=True)
            for index in range(2):
                pd.DataFrame([[0.0, "", *([index] * 16)], [1.0, "", *([index] * 16)]], columns=CSV_COLUMNS).to_csv(standard / f"empty_{index}.csv", index=False, encoding="utf-8-sig")
            profile = build_standard_profile(standard)
            self.assertIsNotNone(profile)
            self.assertEqual(profile.file_count, 2)
            detector = OccupancyDetector(profile, config=__import__("src.app_services", fromlist=["OccupancyConfig"]).OccupancyConfig(enter_score=1.0, exit_score=0.5, enter_hold_seconds=0.0, exit_hold_seconds=0.0))
            self.assertIsNone(detector.update(0.0, np.full(16, 50.0)))
            self.assertTrue(detector.update(0.1, np.full(16, 50.0)))

    def test_csv_replay_bypasses_empty_seat_profile(self) -> None:
        """A saved recording must reach inference despite another calibration."""
        with tempfile.TemporaryDirectory() as temporary:
            standard = Path(temporary) / "data" / "standard"
            standard.mkdir(parents=True)
            pd.DataFrame(
                [[0.0, "", *([0.0] * 16)], [1.0, "", *([0.0] * 16)]],
                columns=CSV_COLUMNS,
            ).to_csv(standard / "empty.csv", index=False, encoding="utf-8-sig")
            self.assertIsNotNone(occupancy_profile_for_input("serial", standard))
            self.assertIsNone(occupancy_profile_for_input("csv_replay", standard))

    def test_voltage_occupancy_rejects_noise_detects_load_and_returns_to_empty(self):
        # The same physical signal in V and mV must have the same decisions.
        for scale in (1., 1000.):
            with self.subTest(scale=scale), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows = [[i / 12, "", *([scale * (1.154 + (i % 3 - 1) * .002)] * 16)] for i in range(48)]
                pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(root / 'empty.csv', index=False)
                detector = OccupancyDetector(build_standard_profile(root))
                empty = np.full(16, 1.157) * scale
                loaded = np.array([1.18] * 8 + [1.6, 1.9, 2.1, 1.4, 1.6, 2.3, 2.1, 1.9]) * scale
                for i in range(24):
                    detector.update(i / 12, empty)
                self.assertFalse(detector.occupied)
                for i in range(24, 48):
                    detector.update(i / 12, loaded)
                self.assertTrue(detector.occupied)
                for i in range(48, 72):
                    detector.update(i / 12, empty)
                self.assertFalse(detector.occupied)

    def test_standard_profile_accounts_for_within_recording_noise(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [[i / 12, "", *([1.2 + (-.1 if i % 2 else .1)] * 16)] for i in range(48)]
            pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(root / 'empty.csv', index=False)
            profile = build_standard_profile(root)
            self.assertTrue(np.all(profile.deviation > .14))

    def test_window_training_loader_excludes_standard_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder, label in ((root / "person", "정자세"), (root / "standard", "정자세")):
                folder.mkdir(parents=True)
                rows = [[index / 12, label, *range(16)] for index in range(37)]
                pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(folder / "sample_001.csv", index=False, encoding="utf-8-sig")
            _windows, _meta, records = load_window_dataset(root, WindowSpec(3.0))
            self.assertEqual(len(records), 1)
            self.assertEqual(records.iloc[0]["subject"], "person")

    def test_dataset_service_stages_only_selected_labeled_quality_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            person = root / "data" / "홍길동"; person.mkdir(parents=True)
            path = person / "홍길동_정자세_001.csv"
            rows = [[float(second), "정자세", *(value + second for value in range(16))] for second in range(4)]
            pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
            service = DatasetService(root)
            record = service.records()[0]
            self.assertEqual(record.status, "기본 학습 데이터")
            self.assertEqual(record.qc_status, "pass")
            service.set_status(record, "학습 대기열에 추가")
            manifest = path.with_suffix(".manifest.json")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["qc"]["status"], "pass")
            staged = service.stage_for_training(service.records(), root / "staged")
            self.assertTrue((staged / "홍길동" / path.name).exists())

            service.set_status(service.records()[0], "폐기")
            discarded = service.records()[0]
            self.assertEqual(discarded.training_role, "exclude")
            self.assertFalse(service.is_training_eligible(discarded))

    def test_recording_and_listing_share_a_configured_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_root = root / "state"
            data_directory = root / "requested" / "data"

            labeled = MeasurementRecorder(
                state_root,
                mode="labeled",
                subject="홍길동",
                label="정자세",
                data_directory=data_directory,
            )
            labeled.write_frame(0.0, range(16))
            labeled.close()

            measurement = MeasurementRecorder(
                state_root,
                mode="measurement",
                data_directory=data_directory,
            )
            measurement.write_frame(0.0, range(16))
            measurement.close("측정 전용")

            self.assertEqual(labeled.csv_path.relative_to(data_directory).parts[0], "홍길동")
            self.assertEqual(measurement.csv_path.relative_to(data_directory).parts[0], "measurements")
            records = DatasetService(state_root, data_directory=data_directory).records()
            self.assertEqual({record.csv_path for record in records}, {labeled.csv_path, measurement.csv_path})

    def test_sensor_packet_never_uses_zero_filled_missing_accelerometer(self) -> None:
        packet = SensorPacket(0.0, tuple(range(16)))
        self.assertIsNone(packet.imu)
        with self.assertRaises(ValueError):
            SensorPacket(0.0, tuple(range(16)), (0.0, float("nan"), 0.0))

    def test_training_wizard_state_accepts_custom_fixed_windows_in_range(self) -> None:
        state = TrainingWizardState()
        state.configure((20, 4, 3, 4), 17)
        self.assertEqual(state.windows, (3, 4, 20))
        self.assertEqual(state.seed, 17)
        with self.assertRaises(ValueError):
            state.configure((), 17)
        with self.assertRaises(ValueError):
            state.configure((0,), 17)
        with self.assertRaises(ValueError):
            state.configure((61,), 17)

    def test_training_outputs_are_packaged_as_pending_not_production(self) -> None:
        from src.model_runtime.validation import ArtifactValidator

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = TrainingJob(root, (), 41, windows=(3, 20), state_root=root / "state")
            artifacts = job.run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            for seconds in job.windows:
                (artifacts / f"posture_classifier_window_{seconds}s.joblib").write_bytes(f"candidate-{seconds}".encode())
                config = {
                    "version": 4,
                    "label_names": ["정자세"],
                    "sensor_columns": ["R11"],
                    "sampling_rate": 12,
                    "feature_names": ["mean_R11"],
                    "min_tail_fraction": 0.95,
                    "training_subject_count": 2,
                    "training_record_count": 4,
                    "training_window_count": 8,
                    "confidence_semantics": "calibrated_probability",
                    "fixed_configuration_group_cv_summary": {"record_macro_f1": 0.8},
                }
                (artifacts / f"window_{seconds}s_model_config.json").write_text(json.dumps(config), encoding="utf-8")

            manifests = job._write_pending_candidate_manifests()
            self.assertEqual(len(manifests), 2)
            self.assertEqual(set(TrainingJob.pending_candidate_manifests(job.state_root)), set(manifests))
            for manifest in manifests:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                self.assertEqual(payload["schema_version"], 2)
                self.assertNotIn("approval_status", payload)
                self.assertNotIn("deployment_status", payload)
                self.assertTrue(ArtifactValidator().validate_manifest(manifest, production_only=False).available)
                self.assertFalse(ArtifactValidator().validate_manifest(manifest).available)
