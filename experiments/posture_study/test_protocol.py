"""Validate the frozen study boundary against source evidence, without training.

Run after preparation with ``python -m unittest
experiments.posture_study.test_protocol -v``. POSTURE_STUDY_OUTPUT can point to
another reproduced run. These tests never create or change a dataset.
"""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.posture_study.prepare import OUT, ROOT
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, normalize_text, read_sensor_csv


class _RecordingEstimator:
    """Synthetic learner that records the actual arrays presented by the runner."""
    def fit(self, x, y, validation=None):
        self.training_groups = set(x["groups"])
        self.validation_groups = set(validation[0]["groups"])
        return self

    def predict_proba(self, x):
        self.last_prediction_groups = set(x["groups"])
        return np.full((len(x["y"]), 7), 1 / 7)


class FrozenProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = Path(os.environ.get("POSTURE_STUDY_OUTPUT", str(OUT)))
        paths = [cls.output / "data.npz", cls.output / "dataset.json"]
        if not all(path.is_file() for path in paths):
            raise unittest.SkipTest("Prepare the study dataset before running frozen-data checks")
        cls.manifest = json.loads(paths[1].read_text(encoding="utf-8"))
        with np.load(paths[0], allow_pickle=False) as archive:
            cls.data = {key: archive[key] for key in archive.files}

    def test_each_record_and_subject_stays_in_one_partition(self):
        data, manifest = self.data, self.manifest
        subjects = set(data["groups"])
        by_record = defaultdict(set)
        for record, subject in zip(data["record_ids"], data["groups"]):
            by_record[str(record)].add(str(subject))
        self.assertTrue(all(len(value) == 1 for value in by_record.values()))
        original_test = set(manifest["partitions"][0]["test"])
        self.assertTrue(original_test)
        for partition in manifest["partitions"]:
            with self.subTest(partition=partition["id"]):
                sets = {name: set(partition[name]) for name in ("train", "validation", "test")}
                self.assertEqual(sets["test"], original_test)
                self.assertFalse(sets["train"] & sets["validation"])
                self.assertFalse((sets["train"] | sets["validation"]) & original_test)
                self.assertEqual(set.union(*sets.values()), subjects)
                record_sets = {}
                for name, members in sets.items():
                    self.assertEqual(len(partition[name]), len(members))
                    mask = np.isin(data["groups"], list(members))
                    record_sets[name] = set(data["record_ids"][mask])
                    self.assertEqual(set(data["y"][mask]), set(range(len(CLASS_NAMES))))
                    recorded = partition["counts"][name]
                    self.assertEqual(recorded["windows"], int(mask.sum()))
                    self.assertEqual(recorded["records"], len(record_sets[name]))
                    self.assertEqual(recorded["subjects"], len(members))
                    self.assertEqual(recorded["class_counts"], np.bincount(data["y"][mask], minlength=7).tolist())
                self.assertFalse(record_sets["train"] & record_sets["validation"])
                self.assertFalse(record_sets["train"] & record_sets["test"])
                self.assertFalse(record_sets["validation"] & record_sets["test"])

    def test_registration_is_entirely_separate_by_path_and_content(self):
        registrations = self.manifest["registrations"]
        self.assertEqual(len(registrations), len({row["subject"] for row in registrations}))
        self.assertEqual({row["subject"] for row in registrations}, set(self.data["groups"]))
        rows = self.manifest["rows"]
        registration_paths = {(ROOT / "data" / row["relative_path"]).resolve() for row in registrations}
        target_paths = {(ROOT / "data" / row["record"]).resolve() for row in rows}
        self.assertFalse(registration_paths & target_paths)
        for digest_key in ("sha256", "sensor_sha256"):
            self.assertFalse({row[digest_key] for row in registrations} & {row[digest_key] for row in rows})
        eligible = self.manifest["audit"]["records"]
        expected_records = {row["relative_path"] for row in eligible}
        self.assertEqual(expected_records, {row["relative_path"] for row in registrations} | set(self.data["record_ids"]))
        self.assertFalse(self.manifest["audit"]["duplicate_raw_files"])
        self.assertFalse(self.manifest["audit"]["duplicate_sensor_sequences"])

    def test_dataset_rows_features_and_temporal_endpoints_match(self):
        data, manifest = self.data, self.manifest
        count = len(data["y"])
        self.assertEqual(len(manifest["rows"]), count)
        self.assertEqual(data["seq"].shape, (count, 36, 16))
        self.assertEqual(data["features"].shape, (count, manifest["feature_dimension"]))
        self.assertEqual(data["personal"].shape, (count, 49))
        self.assertEqual(manifest["class_names"], CLASS_NAMES)
        for key, value in data.items():
            self.assertEqual(len(value), count, key)
            if np.issubdtype(value.dtype, np.number):
                self.assertTrue(np.isfinite(value).all(), key)
        seen = set()
        by_record = defaultdict(list)
        for index, row in enumerate(manifest["rows"]):
            self.assertEqual(row["subject"], normalize_text(row["subject"]))
            self.assertEqual(row["subject"], data["groups"][index])
            self.assertEqual(row["record"], data["record_ids"][index])
            self.assertEqual(row["start_s"], data["start_s"][index])
            self.assertEqual(row["label"], CLASS_NAMES[data["y"][index]])
            identity = row["record"], row["start_s"]
            self.assertNotIn(identity, seen)
            seen.add(identity)
            by_record[row["record"]].append(row["start_s"])
        for starts in by_record.values():
            self.assertEqual(starts[0], 0)
            np.testing.assert_allclose(np.diff(starts), manifest["stride_seconds"])

    def test_source_hashes_and_fingerprint_are_current(self):
        manifest = self.manifest
        for row in manifest["audit"]["records"]:
            path = ROOT / "data" / row["relative_path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"], str(path))
        for row in manifest["sidecars"]:
            self.assertEqual(hashlib.sha256((ROOT / row["file"]).read_bytes()).hexdigest(), row["sha256"])
        self.assertEqual(hashlib.sha256(Path(manifest["baseline"]["active_artifact"]).read_bytes()).hexdigest(), manifest["baseline"]["sha256"])
        basis = {key: manifest[key] for key in ("audit", "partitions", "registrations")}
        actual = hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(actual, manifest["fingerprint"])

    def test_personal_reference_uses_only_reserved_first_ten_seconds(self):
        for registration in self.manifest["registrations"]:
            with self.subTest(subject=registration["subject"]):
                frame = read_sensor_csv(ROOT / "data" / registration["relative_path"])
                time = frame.elapsed_s.to_numpy() - frame.elapsed_s.iloc[0]
                reference = frame.loc[time <= 10, SENSOR_COLUMNS].to_numpy(float).mean(0)
                reference_ratio = reference / max(reference.sum(), 1e-6)
                np.testing.assert_allclose(reference, registration["reference_mean"])
                np.testing.assert_allclose(reference_ratio, registration["reference_ratio"])
                indices = np.flatnonzero(self.data["groups"] == registration["subject"])
                expected = np.tile(reference_ratio, (len(indices), 1))
                np.testing.assert_allclose(self.data["personal"][indices, :16], expected, rtol=1e-6, atol=1e-7)


class TemporalProtocolTests(unittest.TestCase):
    def setUp(self):
        from experiments.posture_study.run import postprocess
        self.postprocess = postprocess

    def test_predictions_are_prefix_invariant_and_future_changes_do_not_rewrite_past(self):
        rng = np.random.default_rng(381)
        probability = rng.dirichlet(np.ones(7), size=30)
        records = np.array(["session_A"] * 15 + ["session_B"] * 15)
        starts = np.tile(np.arange(15) * 3.0, 2)
        for method in ("ema", "hysteresis"):
            full = self.postprocess(probability, records, starts, method)
            for endpoint in (1, 2, 7, 14, 15, 16, 23, 29):
                with self.subTest(method=method, endpoint=endpoint):
                    prefix = self.postprocess(probability[:endpoint], records[:endpoint], starts[:endpoint], method)
                    np.testing.assert_allclose(prefix, full[:endpoint])
                    changed = probability.copy()
                    changed[endpoint:] = rng.dirichlet(np.ones(7), size=len(changed) - endpoint)
                    np.testing.assert_allclose(self.postprocess(changed, records, starts, method)[:endpoint], prefix)

    def test_observed_boundaries_and_gaps_reset_state(self):
        a, b = np.eye(7)[0], np.eye(7)[1]
        probability = np.stack([a, b, b, a, b])
        for method in ("ema", "hysteresis"):
            # A new recording, a missing-window gap, and a restarted clock each reset.
            for records, starts, reset_index in (
                (["A", "B", "B", "B", "B"], [0, 0, 3, 6, 9], 1),
                (["A"] * 5, [0, 3, 9, 12, 15], 2),
                (["A"] * 5, [0, 3, 0, 3, 6], 2),
            ):
                with self.subTest(method=method, starts=starts):
                    result = self.postprocess(probability, records, starts, method)
                    np.testing.assert_allclose(result[reset_index], probability[reset_index])

    def test_hysteresis_requires_two_consecutive_matching_candidates(self):
        a, b, c = np.eye(7)[:3]
        probability = np.stack([a, b, c, b, b, a, a])
        result = self.postprocess(probability, ["A"] * 7, np.arange(7) * 3, "hysteresis")
        np.testing.assert_array_equal(result.argmax(1), [0, 0, 0, 0, 1, 1, 0])

    def test_ema_preserves_probability_mass_and_has_documented_weight(self):
        a, b = np.eye(7)[:2]
        result = self.postprocess(np.stack([a, b, b]), ["A"] * 3, [0, 3, 6], "ema")
        np.testing.assert_allclose(result[1], 0.6 * b + 0.4 * a)
        np.testing.assert_allclose(result[2], 0.84 * b + 0.16 * a)
        np.testing.assert_allclose(result.sum(1), 1)


class SelectionProtocolTests(unittest.TestCase):
    @staticmethod
    def summary(macro=0.5, splits=(0.5, 0.5, 0.5), worst=0.3, classes=None):
        return {"macro_f1": macro, "split_f1": dict(zip(("primary", "robustness_1", "robustness_2"), splits)),
                "worst_class_f1": worst, "class_f1": [0.5] * 7 if classes is None else classes,
                "accuracy": 1.0}

    def test_success_requires_macro_robustness_and_class_coverage_not_accuracy(self):
        from experiments.posture_study.run import is_improvement
        baseline = self.summary()
        self.assertTrue(is_improvement(self.summary(.52, (.52, .52, .52), .31, [.52] * 7), baseline))
        self.assertFalse(is_improvement(self.summary(.50), baseline))
        self.assertFalse(is_improvement(self.summary(.52, (.56, .5, .5)), baseline))
        self.assertFalse(is_improvement(self.summary(.52, (.52, .52, .52), .27), baseline))
        self.assertFalse(is_improvement(self.summary(.52, (.52, .52, .52), .31, [.6] * 3 + [.49] * 4), baseline))

    def test_experimental_baseline_matches_production_architecture_and_probability_decision(self):
        from experiments.posture_study.classical import ClassicalEstimator
        from src.model_comparison import build_model
        rng = np.random.default_rng(2471)
        features = rng.normal(size=(84, 12))
        labels = np.tile(np.arange(7), 12)
        training = {"features": features, "groups": np.repeat(["A", "B", "C", "D"], 21)}
        validation = {"features": np.full((7, 12), 1e9), "groups": np.repeat("V", 7)}
        model = ClassicalEstimator("baseline", seed=20260826).fit(training, labels, validation=(validation, np.arange(7)))
        reference = build_model("rbf_svm", {"C": 1.0, "gamma": 0.0005}, 20260826).fit(features, labels)
        observed = rng.normal(size=(19, 12))
        actual = model.predict_proba({"features": observed})
        expected = reference.predict_proba(observed)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
        np.testing.assert_allclose(model.model_.named_steps["scale"].mean_, features.mean(0))
        params = model.model_.named_steps["model"].get_params()
        for key, value in {"C": 1.0, "gamma": 0.0005, "kernel": "rbf", "probability": True,
                           "class_weight": "balanced", "random_state": 20260826}.items():
            self.assertEqual(params[key], value)

    def test_train_stage_never_passes_test_subjects_to_fit_or_prediction(self):
        from experiments.posture_study.run import run_one
        groups = np.repeat(["train_A", "validation_B", "test_C"], 7)
        data = {"groups": groups, "y": np.tile(np.arange(7), 3), "features": np.ones((21, 4)),
                "record_ids": np.array([f"record_{index}" for index in range(21)]), "start_s": np.zeros(21)}
        manifest = {"fingerprint": "synthetic-frozen-data", "partitions": [
            {"id": "primary", "seed": 42, "train": ["train_A"], "validation": ["validation_B"], "test": ["test_C"]}]}
        model = _RecordingEstimator()
        with tempfile.TemporaryDirectory() as temporary:
            with patch("experiments.posture_study.run.build", return_value=model), patch("experiments.posture_study.run.benchmark", return_value={}):
                run_one({"id": "X00", "family": "classical", "method": "synthetic"}, data, manifest, temporary)
        self.assertEqual(model.training_groups, {"train_A"})
        self.assertEqual(model.validation_groups, {"validation_B"})
        self.assertEqual(model.last_prediction_groups, {"validation_B"})

    def test_test_stage_requires_lock_and_rejects_changed_evidence_before_access(self):
        from experiments.posture_study.run import evaluate_test
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = {"fingerprint": "frozen-dataset"}
            with patch("experiments.posture_study.run.take", side_effect=RuntimeError("test data accessed")):
                with self.assertRaises(FileNotFoundError):
                    evaluate_test({}, manifest, output)
                (output / "checkpoint.joblib").write_bytes(b"changed checkpoint")
                (output / "protocol.json").write_text("{}")
                lock = {"dataset_fingerprint": manifest["fingerprint"], "source_hashes": {},
                        "protocol_sha256": hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest(),
                        "checkpoints": {"checkpoint.joblib": hashlib.sha256((output / "checkpoint.joblib").read_bytes()).hexdigest()}}
                mutations = (
                    ({"dataset_fingerprint": "different-data"}, "Locked dataset differs"),
                    ({"protocol_sha256": "different-protocol"}, "Locked protocol differs"),
                    ({"source_hashes": {"experiments/posture_study/prepare.py": "changed-source"}}, "Post-lock implementation changed"),
                    ({"checkpoints": {"checkpoint.joblib": "wrong-hash"}}, "Post-lock model changed"),
                )
                for mutation, expected in mutations:
                    with self.subTest(evidence=expected):
                        (output / "selection_lock.json").write_text(json.dumps({**lock, **mutation}))
                        with self.assertRaisesRegex(AssertionError, expected):
                            evaluate_test({}, manifest, output)


if __name__ == "__main__":
    unittest.main()
