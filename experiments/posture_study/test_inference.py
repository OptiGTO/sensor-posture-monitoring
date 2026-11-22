"""Synthetic inference contracts; these tests make no model-accuracy claims.

Run with::

    .venv/bin/python -m unittest experiments.posture_study.test_inference -v

The optional B00 check loads its checkpoint only and predicts generated windows.
No prepared dataset, held-out recording or held-out labels are accessed.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from experiments.posture_study.classical import ClassicalEstimator
from experiments.posture_study.inference import (
    StreamingPredictor, build_inputs, predict_csv, registration_mean_from_csv,
    requires_registration,
)
from experiments.posture_study.run import FusionModel, TemporalModel
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS
from src.window_pipeline import extract_window_features

ROOT = Path(__file__).resolve().parents[2]
BASELINE_CHECKPOINT = ROOT / "results/posture_study_20260913/B00/primary/checkpoint.joblib"


class _ChangingProbabilityModel:
    """Known class changes derived from input only, to exercise temporal state."""

    def predict_proba(self, inputs):
        winners = inputs["seq"][:, 0, 0].astype(int) % 7
        return np.eye(7)[winners] * 0.93 + 0.01


class StreamingInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(729)
        cls.sequences = rng.uniform(30, 850, (12, 36, 16)).astype(np.float32)
        cls.sequences[:, 0, 0] = [0, 1, 1, 2, 3, 0, 4, 5, 5, 6, 6, 0]
        cls.records = np.array(["record_A"] * 6 + ["record_B"] * 6)
        cls.starts = np.array([0, 3, 6, 12, 15, 0, 0, 3, 6, 9, 12, 15.0])

    def _assert_batch_stream_match(self, model):
        inputs = build_inputs(self.sequences, record_ids=self.records, start_s=self.starts)
        streaming = StreamingPredictor(model)
        actual = np.stack([
            streaming.predict_proba(window, record_id=record, start_s=start)
            for window, record, start in zip(self.sequences, self.records, self.starts)
        ])
        np.testing.assert_allclose(actual, model.predict_proba(inputs), rtol=1e-12, atol=1e-12)
        streaming.reset(record_id="reset_record")
        first = streaming.predict_proba(self.sequences[0])
        np.testing.assert_allclose(first, model.predict_proba(build_inputs(self.sequences[:1]))[0])
        self.assertEqual(streaming.last_start_s, 0)
        self.assertEqual(streaming.next_start_s, 3)

    def test_leaf_and_nested_temporal_fusion_match_batch_with_resets(self):
        base = _ChangingProbabilityModel()
        models = [
            base,
            TemporalModel(base, "ema"),
            TemporalModel(base, "hysteresis"),
            FusionModel([TemporalModel(base, "ema"), TemporalModel(base, "hysteresis")]),
            TemporalModel(FusionModel([base, TemporalModel(base, "hysteresis")]), "ema"),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__, method=getattr(model, "method", None)):
                self._assert_batch_stream_match(model)

    def test_hysteresis_retains_state_and_confirms_second_matching_window(self):
        stream = StreamingPredictor(TemporalModel(_ChangingProbabilityModel(), "hysteresis"))
        windows = self.sequences[:3]
        actual = [stream.predict_proba(window).argmax() for window in windows]
        self.assertEqual(actual, [0, 0, 1])
        # Supplying a new recording starts its local clock and clears history.
        result = stream.predict(windows[1], record_id="new_record")
        self.assertEqual(result["class_id"], 1)
        self.assertEqual(result["start_s"], 0)
        self.assertEqual(result["label"], CLASS_NAMES[1])

    def test_bad_window_shapes_and_nonfinite_values_are_rejected(self):
        stream = StreamingPredictor(_ChangingProbabilityModel())
        for shape in ((35, 16), (36, 15), (1, 36, 16)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                stream.predict_proba(np.ones(shape))
        bad = self.sequences[0].copy()
        bad[3, 2] = np.nan
        with self.assertRaises(ValueError):
            stream.predict_proba(bad)

    @unittest.skipUnless(BASELINE_CHECKPOINT.exists(), "Study B00 checkpoint is not present")
    def test_current_b00_checkpoint_replay_uses_synthetic_windows_only(self):
        model = joblib.load(BASELINE_CHECKPOINT)
        self._assert_batch_stream_match(model)
        self._assert_batch_stream_match(TemporalModel(model, "ema"))


class PersonalizationAndEnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(951)
        cls.seq = rng.uniform(1, 500, (42, 36, 16)).astype(np.float32)
        cls.reference = rng.uniform(1, 500, 16)
        cls.inputs = build_inputs(cls.seq, registration_mean=cls.reference)
        cls.inputs["groups"] = np.repeat(np.arange(6), 7)
        cls.labels = np.tile(np.arange(7), 6)
        cls.baseline = ClassicalEstimator("baseline", seed=951).fit(cls.inputs, cls.labels)
        cls.personal = ClassicalEstimator("personal", seed=951).fit(cls.inputs, cls.labels)

    def test_legacy_and_49_registration_features_match_preparation_exactly(self):
        mean = self.seq.mean(axis=1)
        ratio = mean / np.maximum(mean.sum(axis=1, keepdims=True), 1e-6)
        reference_ratio = self.reference / max(self.reference.sum(), 1e-6)
        expected = np.concatenate([
            np.tile(reference_ratio, (len(self.seq), 1)), ratio - reference_ratio,
            np.log((mean + 1) / (self.reference + 1)),
            ((mean.sum(axis=1) + 1) / (self.reference.sum() + 1))[:, None],
        ], axis=1).astype(np.float32)
        np.testing.assert_array_equal(self.inputs["personal"], expected)
        np.testing.assert_array_equal(self.inputs["features"], extract_window_features(self.seq).astype(np.float32))
        self.assertEqual(self.inputs["features"].shape, (42, 596))
        self.assertEqual(self.inputs["personal"].shape, (42, 49))

    def test_nested_personalization_requires_reference_and_matches_batch(self):
        model = TemporalModel(FusionModel([self.personal, self.baseline]), "ema")
        self.assertTrue(requires_registration(model))
        self.assertFalse(requires_registration(self.baseline))
        with self.assertRaises(ValueError):
            StreamingPredictor(model)
        stream = StreamingPredictor(model, registration_mean=self.reference)
        actual = np.stack([stream.predict_proba(window) for window in self.seq])
        np.testing.assert_allclose(actual, model.predict_proba(self.inputs), rtol=1e-12, atol=1e-12)

    def test_reference_is_copied_and_explicit_change_resets_temporal_history(self):
        model = TemporalModel(self.personal, "ema")
        caller_reference = self.reference.copy()
        stream = StreamingPredictor(model, registration_mean=caller_reference)
        caller_reference[:] += 100
        np.testing.assert_array_equal(stream.registration_mean, self.reference)
        stream.predict_proba(self.seq[0])
        changed_reference = self.reference + 10
        result = stream.predict_proba(self.seq[1], registration_mean=changed_reference)
        expected = self.personal.predict_proba(build_inputs(self.seq[1:2], registration_mean=changed_reference))[0]
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_two_member_stacking_uses_disjoint_oof_and_checkpoint_roundtrip(self):
        stack = ClassicalEstimator("stacking", seed=951, ensemble_members=["logreg", "extra_trees"])
        stack.fit(self.inputs, self.labels)
        self.assertEqual(len(stack.members_), 2)
        self.assertEqual(stack.complexity()["ensemble_member_methods"], ["logreg", "extra_trees"])
        held_subjects = []
        for fold in stack.oof_fold_groups_:
            self.assertFalse(set(fold["train"]) & set(fold["validation"]))
            self.assertEqual(set(fold["train"]) | set(fold["validation"]), {str(i) for i in range(6)})
            held_subjects.extend(fold["validation"])
        self.assertCountEqual(held_subjects, [str(i) for i in range(6)])
        probabilities = stack.predict_proba(self.inputs)
        self.assertEqual(probabilities.shape, (42, 7))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        checkpoint = io.BytesIO()
        joblib.dump(stack, checkpoint)
        checkpoint.seek(0)
        np.testing.assert_array_equal(joblib.load(checkpoint).predict_proba(self.inputs), probabilities)
        self.assertEqual(ClassicalEstimator("stacking").ensemble_members, ("baseline", "logreg", "extra_trees"))


class CsvInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1983)
        seq = rng.uniform(20, 600, (28, 36, 16)).astype(np.float32)
        inputs = build_inputs(seq)
        inputs["groups"] = np.repeat(np.arange(4), 7)
        cls.baseline = ClassicalEstimator("baseline", seed=1983).fit(inputs, np.tile(np.arange(7), 4))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="posture_inference_test_")
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        rng = np.random.default_rng(1984)
        self.frame = pd.DataFrame(rng.uniform(20, 600, (150, 16)), columns=SENSOR_COLUMNS)
        self.frame.insert(0, "elapsed_s", np.arange(150) / 12)
        # This intentionally invalid annotation must never affect prediction.
        self.frame["label"] = "prediction labels are unavailable"
        self.csv = self.folder / "prediction.csv"
        self.frame.to_csv(self.csv, index=False)
        self.checkpoint = self.folder / "baseline.joblib"
        joblib.dump(self.baseline, self.checkpoint)

    def test_csv_inference_has_four_windows_and_never_uses_prediction_labels(self):
        result = predict_csv(self.checkpoint, self.csv)
        self.assertEqual(len(result["predictions"]), 4)
        self.assertEqual([row["start_s"] for row in result["predictions"]], [0, 3, 6, 9])
        self.assertFalse(result["registration_required"])
        self.assertEqual(result["legacy_feature_dimension"], 596)
        self.assertGreaterEqual(result["preprocessing_seconds"], 0)
        self.assertGreaterEqual(result["model_inference_seconds"], 0)

    def test_registration_uses_first_ten_seconds_and_checks_correct_label(self):
        registration = self.frame.copy()
        registration["label"] = CLASS_NAMES[0]
        elapsed = registration.elapsed_s - registration.elapsed_s.iloc[0]
        expected = registration.loc[elapsed <= 10, SENSOR_COLUMNS].to_numpy(float).mean(axis=0)
        registration.loc[elapsed > 10, SENSOR_COLUMNS] = 100000
        path = self.folder / "registration.csv"
        registration.to_csv(path, index=False)
        np.testing.assert_allclose(registration_mean_from_csv(path), expected, rtol=1e-14)
        registration["label"] = CLASS_NAMES[1]
        registration.to_csv(path, index=False)
        with self.assertRaises(ValueError):
            registration_mean_from_csv(path)

    def test_same_record_and_duplicate_sensor_content_cannot_be_registration(self):
        with self.assertRaises(ValueError):
            predict_csv(self.checkpoint, self.csv, self.csv)
        renamed = self.folder / "renamed_registration.csv"
        copied = self.frame.copy()
        copied["label"] = CLASS_NAMES[0]
        copied.to_csv(renamed, index=False)
        with self.assertRaisesRegex(ValueError, "센서 시계열이 동일"):
            predict_csv(self.checkpoint, self.csv, renamed)

    def test_cli_emits_json_and_preserves_input_paths(self):
        destination = self.folder / "output.json"
        command = [sys.executable, "-m", "experiments.posture_study.inference",
                   "--checkpoint", str(self.checkpoint), "--csv", str(self.csv),
                   "--output", str(destination)]
        completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(len(json.loads(destination.read_text())["predictions"]), 4)
        original = self.csv.read_bytes()
        rejected = subprocess.run(command[:-1] + [str(self.csv)], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(self.csv.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
