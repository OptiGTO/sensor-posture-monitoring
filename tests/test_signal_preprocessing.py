"""Versioned raw-signal preprocessing: rule, training/inference parity, fail-closed contracts."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd

import src.model_comparison as mc
from src.app_services import TrainingJob
from src.model_comparison import _dataset, aligned_predict_proba, loso_evaluation, train_comparison
from src.model_runtime import ModelRegistry
from src.model_runtime.sklearn_adapter import SklearnAdapter
from src.model_runtime.validation import ArtifactValidator
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, read_sensor_csv
from src.signal_preprocessing import (DESPIKE_V1_PARAMETERS, SignalPreprocessing, bundle_entry, despike_v1)
from src.window_pipeline import WindowSpec, extract_window_features, load_window_dataset, segment_sensor_frame

ROOT = Path(__file__).resolve().parents[1]
DT = 0.082
SMALL_GRID = {"extra_trees": [{"n_estimators": 12, "min_samples_leaf": 1, "max_features": "sqrt"}],
              "rbf_svm": [{"C": 1.0, "gamma": "scale"}]}


def volt_like_data(root: Path, subjects: int = 5, rows: int = 86, spikes: bool = True) -> None:
    """Posture-dependent levels in the recorded 1-3 V range with isolated spikes."""
    rng = np.random.default_rng(29)
    for subject in range(subjects):
        directory = root / f"subject{subject}"
        directory.mkdir(parents=True)
        for index, label in enumerate(CLASS_NAMES):
            level = 1.1 + 0.2 * index + 0.03 * subject + 0.05 * np.arange(len(SENSOR_COLUMNS)) / 16
            values = level + rng.normal(0, 0.004, (rows, len(SENSOR_COLUMNS)))
            if spikes:
                for row, channel in zip(rng.integers(2, rows - 2, 6), rng.integers(0, 16, 6)):
                    values[row, channel] += 0.4
            frame = pd.DataFrame(values, columns=SENSOR_COLUMNS)
            frame.insert(0, "label", label)
            frame.insert(0, "elapsed_s", np.arange(rows) * DT)
            frame.to_csv(directory / f"posture{index}.csv", index=False)


class DespikeRuleTests(unittest.TestCase):
    def series(self, values):
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        return np.arange(len(values)) * DT, values

    def test_isolated_spike_is_repaired_to_neighbour_mean(self):
        repaired, mask = despike_v1(*self.series([1.2, 1.2, 1.6, 1.21, 1.2]))
        self.assertAlmostEqual(repaired[2, 0], (1.2 + 1.21) / 2)
        self.assertEqual(mask[:, 0].tolist(), [False, False, True, False, False])

    def test_negative_spike_is_repaired(self):
        repaired, mask = despike_v1(*self.series([1.5, 1.5, 1.17, 1.5, 1.5]))
        self.assertAlmostEqual(repaired[2, 0], 1.5)
        self.assertEqual(int(mask.sum()), 1)

    def test_steps_ramps_small_noise_and_edges_are_unchanged(self):
        for values in ([1.2, 1.2, 1.6, 1.6, 1.6],        # posture step
                       [1.2, 1.4, 1.6, 1.8, 2.0],        # ramp
                       [1.2, 1.25, 1.2, 1.25, 1.2],      # below threshold
                       [1.9, 1.2, 1.2, 1.2, 1.9],        # first/last sample
                       [1.2, 1.6, 1.9, 1.2, 1.2]):       # one-sided return
            time_s, source = self.series(values)
            repaired, mask = despike_v1(time_s, source)
            np.testing.assert_array_equal(repaired, source)
            self.assertFalse(mask.any(), values)

    def test_alternating_flicker_is_repaired_without_phase_inversion(self):
        repaired, mask = despike_v1(*self.series([1.5, 1.5, 1.17, 1.52, 1.16, 1.5, 1.5]))
        np.testing.assert_allclose(repaired[:, 0], [1.5, 1.5, 1.51, 1.52, 1.51, 1.5, 1.5])
        self.assertEqual(np.flatnonzero(mask[:, 0]).tolist(), [2, 4])

    def test_large_sampling_gap_blocks_repair(self):
        time_s = np.array([0, DT, DT + 0.5, 2 * DT + 0.5, 3 * DT + 0.5])
        source = np.array([[1.2], [1.2], [1.6], [1.2], [1.2]])
        repaired, mask = despike_v1(time_s, source)
        self.assertFalse(mask.any())
        np.testing.assert_array_equal(repaired, source)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            despike_v1(np.array([0, 0, 1.0]), np.ones((3, 1)))
        with self.assertRaises(ValueError):
            despike_v1(np.arange(3.0), np.array([[1.0], [np.nan], [1.0]]))

    def test_live_snapshot_matches_whole_recording_away_from_its_edges(self):
        rng = np.random.default_rng(3)
        time_s = np.arange(200) * DT
        values = 1.3 + rng.normal(0, 0.004, (200, 16))
        for row, channel in zip(rng.integers(1, 199, 40), rng.integers(0, 16, 40)):
            values[row, channel] += rng.choice([-0.3, 0.3])
        whole, _ = despike_v1(time_s, values)
        for start in (0, 17, 90, 163):
            stop = start + 37
            part, _ = despike_v1(time_s[start:stop], values[start:stop])
            np.testing.assert_allclose(part[1:-1], whole[start + 1:stop - 1])


class SignalPreprocessingContractTests(unittest.TestCase):
    def frame(self, rows=40):
        values = np.full((rows, 16), 1.3)
        values[10, 3] = 1.8
        frame = pd.DataFrame(values, columns=SENSOR_COLUMNS)
        frame.insert(0, "elapsed_s", np.arange(rows) * DT + 5.0)
        frame["label"] = CLASS_NAMES[0]
        return frame

    def test_identity_is_default_and_writes_no_bundle_fields(self):
        identity = SignalPreprocessing()
        frame = self.frame()
        self.assertTrue(identity.is_identity)
        self.assertIs(identity.apply_inference(frame), frame)
        self.assertIs(identity.apply_training(frame), frame)
        self.assertEqual(bundle_entry(None), {})
        self.assertEqual(bundle_entry(identity), {})
        self.assertEqual(SignalPreprocessing.from_bundle({}), identity)

    def test_rule_round_trip_and_fail_closed_on_changes(self):
        rule = SignalPreprocessing("despike_v1", 1.0).rule()
        self.assertEqual(SignalPreprocessing.from_rule(json.loads(json.dumps(rule))), SignalPreprocessing("despike_v1", 1.0))
        altered = json.loads(json.dumps(rule))
        altered["despike"]["parameters"]["min_excursion"] = 0.05
        unknown = json.loads(json.dumps(rule))
        unknown["despike"]["version"] = "despike_v2"
        wrong_hash = {**rule, "rule_sha256": "0" * 64}
        for bad in (altered, unknown, wrong_hash, {**rule, "schema": "other"}, "despike_v1"):
            with self.assertRaises(ValueError):
                SignalPreprocessing.from_rule(bad)
        for trim in (-1, 5.5, float("nan"), True):
            with self.assertRaises(ValueError):
                SignalPreprocessing(None, trim)
        with self.assertRaises(ValueError):
            SignalPreprocessing("median")

    def test_training_despikes_then_trims_while_inference_never_trims(self):
        processing = SignalPreprocessing("despike_v1", 1.0)
        frame = self.frame()
        inference = processing.apply_inference(frame)
        training = processing.apply_training(frame)
        self.assertEqual(len(inference), len(frame))
        channel = SENSOR_COLUMNS[3]
        self.assertAlmostEqual(inference.loc[10, channel], 1.3)
        self.assertEqual(frame.loc[10, channel], 1.8, "input frame must not be mutated")
        relative = training["elapsed_s"] - frame["elapsed_s"].iloc[0]
        self.assertGreaterEqual(relative.min(), 1.0 - 1e-9)
        self.assertEqual(len(training), int(np.sum(np.arange(40) * DT >= 1.0 - 1e-9)))
        self.assertTrue((training[SENSOR_COLUMNS] == 1.3).all().all())
        self.assertEqual(training["label"].unique().tolist(), [CLASS_NAMES[0]])
        self.assertEqual(processing.count_repairs(frame), 1)
        trim_only = SignalPreprocessing(None, 1.0)
        self.assertIs(trim_only.apply_inference(frame), frame)

    def test_dataset_loader_identity_transform_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volt_like_data(root / "data", subjects=2)
            base = load_window_dataset(root / "data", WindowSpec(3))
            same = load_window_dataset(root / "data", WindowSpec(3), None)
            np.testing.assert_array_equal(base[0], same[0])
            pd.testing.assert_frame_equal(base[1], same[1])
            despiked = load_window_dataset(root / "data", WindowSpec(3), SignalPreprocessing("despike_v1").apply_inference)
            self.assertEqual(despiked[0].shape, base[0].shape)
            self.assertFalse(np.array_equal(despiked[0], base[0]))
            pd.testing.assert_frame_equal(base[2], despiked[2])

    def test_loso_fits_separate_training_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volt_like_data(root / "data")
            processing = SignalPreprocessing("despike_v1", 1.0)
            data = _dataset(root / "data", 3, 41, None, processing)
            fit = data["fit"]
            self.assertIsNot(fit["X"], data["X"])
            subjects = data["meta"]["subject"].to_numpy(dtype=str)
            development = data["masks"]["train"] | data["masks"]["validation"]
            fit_subjects = fit["meta"]["subject"].to_numpy(dtype=str)
            fit_development = fit["masks"]["train"] | fit["masks"]["validation"]
            sizes, real_fit = [], mc.fit_method_model
            def spy(model, family, method, X, y, **kwargs):
                sizes.append(len(X))
                return real_fit(model, family, method, X, y, **kwargs)
            with mock.patch.object(mc, "fit_method_model", side_effect=spy):
                result = loso_evaluation("extra_trees", SMALL_GRID["extra_trees"][0], 41, "standard", data["X"], data["y"],
                    subjects, data["meta"]["record_id"].to_numpy(dtype=str), development,
                    fit=(fit["X"], fit["y"], fit_subjects, fit["meta"]["record_id"].to_numpy(dtype=str), fit_development))
            people = sorted(set(subjects[development]))
            self.assertEqual(sizes, [int(np.sum(fit_development & (fit_subjects != person))) for person in people])
            self.assertEqual(result["windows"], int(development.sum()))


class TrainingInferenceParityTests(unittest.TestCase):
    def test_default_training_output_carries_no_signal_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volt_like_data(root / "data", spikes=False)
            with mock.patch("src.model_comparison.MODEL_CANDIDATES", SMALL_GRID):
                report = train_comparison(root / "data", root / "run", (3,), 41, ("extra_trees",),
                                          progress=lambda _: None, selection_metric="validation")
            self.assertNotIn("signal_preprocessing", report["preprocessing"])
            bundle = joblib.load(root / "run/artifacts" / report["models"][0]["artifact_file"])
            self.assertNotIn("signal_preprocessing", bundle)
            self.assertNotIn("fit_window_count", report["datasets"]["3"]["partitions"]["train"])

    def test_bundle_manifest_and_adapter_apply_the_trained_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            volt_like_data(state / "data")
            processing = SignalPreprocessing("despike_v1", 1.0)
            job = TrainingJob(ROOT, (), 41, windows=(3,), models=("extra_trees",), state_root=state)
            with mock.patch("src.model_comparison.MODEL_CANDIDATES", SMALL_GRID):
                report = train_comparison(state / "data", job.run_dir, (3,), 41, ("extra_trees",),
                                          progress=lambda _: None, signal_preprocessing=processing)
            self.assertEqual(report["preprocessing"]["signal_preprocessing"], processing.rule())
            data = _dataset(state / "data", 3, 41, None, processing)
            bundle = joblib.load(job.run_dir / "artifacts" / report["models"][0]["artifact_file"])
            self.assertEqual(bundle["signal_preprocessing"], processing.rule())
            fit_train = data["fit"]["masks"]["train"]
            self.assertEqual(bundle["training_window_count"], int(fit_train.sum()))
            np.testing.assert_allclose(bundle["model"].named_steps["scale"].mean_,
                                       data["fit"]["X"][fit_train].mean(axis=0, dtype=np.float64))
            partitions = report["datasets"]["3"]["partitions"]
            untrimmed = _dataset(state / "data", 3, 41)
            for name in ("train", "validation", "final_evaluation"):
                self.assertEqual(partitions[name]["window_count"], untrimmed["partitions"][name]["window_count"])
            self.assertLess(partitions["train"]["fit_window_count"], partitions["train"]["window_count"] + 1)
            evidence = json.loads((job.run_dir / "artifacts" / bundle["evaluation_file"]).read_text())
            self.assertEqual(evidence["signal_preprocessing"], processing.rule())

            manifests = job._write_pending_candidate_manifests()
            descriptor = ArtifactValidator().discover_manifest(manifests[0]).descriptor
            self.assertEqual(descriptor.preprocessing["signal_preprocessing"], processing.rule())
            registry = ModelRegistry(ROOT, state)
            model = registry.validate_deployment(registry.validate_runtime(descriptor))
            self.assertEqual(model.deployment_validation_status, "passed", str(model))
            adapter = SklearnAdapter(descriptor)
            frame = read_sensor_csv(state / "data/subject0/posture3.csv")
            windows, _ = segment_sensor_frame(processing.apply_inference(frame), WindowSpec(3))
            expected = aligned_predict_proba(bundle["model"], extract_window_features(windows), len(CLASS_NAMES))
            np.testing.assert_allclose(adapter.predict_proba(frame), expected, atol=1e-12)
            raw_windows, _ = segment_sensor_frame(frame, WindowSpec(3))
            self.assertFalse(np.array_equal(windows, raw_windows), "fixture must contain repaired spikes")

            # Live path: the realtime engine feeds rolling snapshots through the same adapter.
            from types import SimpleNamespace
            from src.realtime import LiveConfig, RealtimePosturePredictor
            seen = []
            def live_predict(snapshot):
                seen.append(snapshot)
                return adapter.predict_proba(snapshot, 3).mean(axis=0)
            live = RealtimePosturePredictor(SimpleNamespace(labels=descriptor.labels, predict_frame=live_predict),
                LiveConfig(window_seconds=3, minimum_window_seconds=2.85, smoothing_history=1))
            events = [live.push(row.elapsed_s, [getattr(row, c) for c in SENSOR_COLUMNS])
                      for row in frame.itertuples()]
            last = [event for event in events if event is not None][-1]
            live_windows, _ = segment_sensor_frame(processing.apply_inference(seen[-1]), WindowSpec(3))
            live_expected = aligned_predict_proba(bundle["model"], extract_window_features(live_windows),
                                                  len(CLASS_NAMES)).mean(axis=0)
            np.testing.assert_allclose([last.probabilities[label] for label in descriptor.labels], live_expected, atol=1e-12)
            with self.assertRaisesRegex(ValueError, "신호 전처리"):
                from src.inference_core import predict_probabilities
                predict_probabilities(bundle, extract_window_features(windows))

            # A manifest that drops the rule no longer matches the bundle.
            manifest_path = manifests[0]
            payload = json.loads(manifest_path.read_text())
            payload["preprocessing"].pop("signal_preprocessing")
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            tampered = ArtifactValidator().discover_manifest(manifest_path).descriptor
            result = ArtifactValidator().validate_runtime(tampered, use_cache=False)
            self.assertEqual(result.status, "failed")
            self.assertIn("신호 전처리", result.reason)

    def test_adapter_rejects_unknown_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.joblib"
            joblib.dump({"label_names": list(CLASS_NAMES), "sensor_columns": list(SENSOR_COLUMNS),
                         "signal_preprocessing": {"schema": "signal_preprocessing_v1", "despike": {"version": "x"}}}, path)
            descriptor = mock.Mock(artifact_path=path, labels=tuple(CLASS_NAMES), input_channels=tuple(SENSOR_COLUMNS))
            with self.assertRaises(ValueError):
                SklearnAdapter(descriptor)

    def test_cli_rejects_preprocessing_outside_deployment_comparison(self):
        import train_window_models
        with mock.patch.object(sys, "argv", ["train_window_models.py", "--data", "/nonexistent", "--despike", "despike_v1"]):
            with self.assertRaisesRegex(ValueError, "--deployment"):
                train_window_models.main()

    def test_parameters_are_the_documented_constants(self):
        self.assertEqual(DESPIKE_V1_PARAMETERS, {"min_excursion": 0.10, "max_return_ratio": 0.5,
                                                 "max_interval_seconds": 0.25})


if __name__ == "__main__":
    unittest.main()
