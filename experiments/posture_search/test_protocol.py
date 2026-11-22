"""Leakage and causal-inference checks for the isolated posture study.

Run after dataset preparation:
    .venv/bin/python -m unittest experiments.posture_search.test_protocol -v

Frozen-artifact checks intentionally skip when preparation has not completed.
The source-mutation guard remains testable without creating study artifacts.
"""

from collections import defaultdict
from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from experiments.posture_search import data as study_data
from src.posture_core import CLASS_NAMES, normalize_text


class TestSourceFreeze(unittest.TestCase):
    def test_cached_data_rejects_changed_source_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            np.savez_compressed(output / "data.npz", y=np.array([0]))
            manifest = {
                "source_inventory": {"data/example.csv": "original-content-hash"},
                "cache_sha256": study_data.digest(output / "data.npz"),
            }
            (output / "dataset.json").write_text(json.dumps(manifest))
            with patch.object(study_data, "source_inventory", return_value={
                "data/example.csv": "modified-content-hash"
            }):
                with self.assertRaisesRegex(AssertionError, "Source files/sidecars changed"):
                    study_data.prepare(output)

    def test_cached_data_rejects_modified_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            np.savez_compressed(output / "data.npz", y=np.array([0]))
            manifest = {"source_inventory": {}, "cache_sha256": "wrong-cache-hash"}
            (output / "dataset.json").write_text(json.dumps(manifest))
            with patch.object(study_data, "source_inventory", return_value={}):
                with self.assertRaises(AssertionError):
                    study_data.prepare(output)


class TestFrozenStudyProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        output = study_data.OUT
        if not (output / "dataset.json").is_file() or not (output / "data.npz").is_file():
            raise unittest.SkipTest("Run experiments.posture_search.data before artifact checks")
        cls.manifest = json.loads((output / "dataset.json").read_text())
        with np.load(output / "data.npz", allow_pickle=False) as archive:
            cls.data = {name: archive[name] for name in archive.files}

    def test_source_and_cache_hashes_still_match(self):
        self.assertEqual(study_data.source_inventory(), self.manifest["source_inventory"])
        self.assertEqual(study_data.digest(study_data.OUT / "data.npz"), self.manifest["cache_sha256"])
        self.assertEqual(study_data.digest(self.manifest["baseline"]["artifact"]),
                         self.manifest["baseline"]["sha256"])

    def test_registration_records_are_separate_by_path_and_content(self):
        registrations = self.manifest["registrations"]
        rows = self.manifest["rows"]
        self.assertEqual(len(registrations), len({r["subject"] for r in registrations}))
        self.assertEqual({r["subject"] for r in registrations}, set(self.data["groups"]))
        for key in ("relative_path", "sha256", "sensor_sha256"):
            calibration = {r[key] for r in registrations}
            scored = {r[key] for r in rows}
            self.assertFalse(calibration & scored, f"Calibration overlaps scored data by {key}")
        reserved = {r["relative_path"] for r in registrations}
        scored_paths = set(self.data["record_ids"])
        eligible = {r["relative_path"] for r in self.manifest["audit"]["records"]}
        self.assertEqual(reserved | scored_paths, eligible)
        self.assertFalse(self.manifest["audit"]["duplicate_raw_files"])
        self.assertFalse(self.manifest["audit"]["duplicate_sensor_sequences"])

    def test_calibration_choice_is_fixed_without_evaluation_outcomes(self):
        correct = defaultdict(list)
        for record in self.manifest["audit"]["records"]:
            if record["label"] == CLASS_NAMES[0]:
                correct[record["subject"]].append(record["relative_path"])
        for registration in self.manifest["registrations"]:
            expected = min(correct[registration["subject"]], key=normalize_text)
            self.assertEqual(registration["relative_path"], expected)

    def test_all_group_splits_are_disjoint_and_cover_identical_dataset(self):
        groups = self.data["groups"]
        expected_subjects = set(groups)
        self.assertTrue(all(s == normalize_text(s) for s in expected_subjects))
        records = {r["relative_path"]: r for r in self.manifest["rows"]}
        registrations = {r["subject"]: r for r in self.manifest["registrations"]}
        for partition in self.manifest["partitions"]:
            sets = [set(partition[name]) for name in ("train", "validation", "test")]
            self.assertEqual(set.union(*sets), expected_subjects)
            for left in range(3):
                for right in range(left + 1, 3):
                    self.assertFalse(sets[left] & sets[right])
            partition_content = []
            for name, subjects in zip(("train", "validation", "test"), sets):
                mask = np.isin(groups, list(subjects))
                self.assertEqual(set(self.data["y"][mask]), set(range(len(CLASS_NAMES))))
                paths = set(self.data["record_ids"][mask])
                self.assertEqual(len(paths), partition["counts"][name]["records"])
                # Include reserved calibration sources when checking cross-partition leakage.
                sources = [records[path] for path in paths] + [registrations[s] for s in subjects]
                partition_content.append({key: {r[key] for r in sources}
                                          for key in ("relative_path", "sha256", "sensor_sha256")})
            for left in range(3):
                for right in range(left + 1, 3):
                    for key in ("relative_path", "sha256", "sensor_sha256"):
                        self.assertFalse(partition_content[left][key] & partition_content[right][key],
                                         f"{partition['id']}: cross-partition {key} leakage")

    def test_window_inputs_and_provenance_align(self):
        data = self.data
        count = len(data["y"])
        self.assertEqual(data["seq"].shape, (count, 36, 16))
        self.assertEqual(data["features"].shape, (count, len(self.manifest["feature_names"])))
        self.assertEqual(len(self.manifest["rows"]), count)
        for value in data.values():
            self.assertEqual(len(value), count)
        for key in ("seq", "personal", "features", "start_s"):
            self.assertTrue(np.isfinite(data[key]).all(), key)
        for index, row in enumerate(self.manifest["rows"]):
            self.assertEqual(data["record_ids"][index], row["relative_path"])
            self.assertEqual(data["groups"][index], row["subject"])
            self.assertEqual(CLASS_NAMES[data["y"][index]], row["label"])
            self.assertEqual(data["start_s"][index], row["start_s"])
        for record in set(data["record_ids"]):
            starts = data["start_s"][data["record_ids"] == record]
            self.assertEqual(starts[0], 0)
            np.testing.assert_allclose(np.diff(starts), 3, atol=1e-9)


class TestCausalPostprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            module = importlib.import_module("experiments.posture_search.run")
        except ModuleNotFoundError as error:
            if error.name != "experiments.posture_search.run":
                raise
            raise unittest.SkipTest("Study runner is not yet available") from error
        cls.postprocess = staticmethod(module.postprocess)

    def setUp(self):
        rng = np.random.default_rng(187)
        self.probability = rng.dirichlet(np.ones(7), size=12)
        self.probability[[0, 3, 6, 10]] = np.eye(7)[[1, 2, 3, 4]]
        self.records = np.array(["record_a"] * 6 + ["record_b"] * 6)
        self.starts = np.tile(np.arange(6) * 3.0, 2)

    def test_future_predictions_do_not_change_past_output(self):
        for method in ("ema", "vote", "confidence"):
            full = self.postprocess(self.probability, self.records, self.starts, method)
            for cutoff in range(1, len(self.probability)):
                with self.subTest(method=method, cutoff=cutoff):
                    prefix = self.postprocess(self.probability[:cutoff], self.records[:cutoff],
                                              self.starts[:cutoff], method)
                    np.testing.assert_allclose(full[:cutoff], prefix)

    def test_state_resets_at_independent_recording_boundary(self):
        for method in ("ema", "vote", "confidence"):
            with self.subTest(method=method):
                full = self.postprocess(self.probability, self.records, self.starts, method)
                alone = self.postprocess(self.probability[6:], self.records[6:], self.starts[6:], method)
                np.testing.assert_allclose(full[6:], alone)

    def test_state_resets_after_gap_or_nonincreasing_time(self):
        for method in ("ema", "vote", "confidence"):
            for changed_time in (18.1, 12.0):
                with self.subTest(method=method, changed_time=changed_time):
                    times = np.arange(12) * 3.0
                    times[6:] = changed_time + np.arange(6) * 3.0
                    records = np.full(12, "same_record")
                    full = self.postprocess(self.probability, records, times, method)
                    alone = self.postprocess(self.probability[6:], records[6:], times[6:], method)
                    np.testing.assert_allclose(full[6:], alone)


class TestEnsembleAblationProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.posture_search.classical import build_classical
        rng = np.random.default_rng(482)
        cls.y = np.tile(np.arange(7), 6)
        cls.groups = np.repeat(np.arange(6), 7)
        cls.features = np.eye(7)[cls.y] + rng.normal(0, .05, (len(cls.y), 7))
        cls.stack = build_classical("stacking", seed=53,
                                   ensemble_members=["baseline", "logistic"])
        cls.stack.fit(cls.features, cls.y, cls.groups)

    def test_removed_stacking_member_reduces_group_oof_meta_width(self):
        model = self.stack
        self.assertEqual(model.config_["ensemble_members"], ["baseline", "logistic"])
        self.assertEqual(model.config_["meta_input_features"], 14)
        self.assertEqual(model.meta_.n_features_in_, 14)
        self.assertEqual(len(model.models_), 2)
        for fold in model.config_["oof_folds"]:
            train, validation = set(fold["train_subjects"]), set(fold["validation_subjects"])
            self.assertFalse(train & validation)
            self.assertEqual(train | validation, set(self.groups))
        probability = model.predict_proba(self.features)
        self.assertEqual(probability.shape, (len(self.y), 7))
        np.testing.assert_allclose(probability.sum(axis=1), 1)

    def test_weighted_vote_renormalizes_retained_prior_weights(self):
        from experiments.posture_search.classical import build_classical
        model = build_classical("weighted_vote", seed=53,
                                ensemble_members=["baseline", "logistic"])
        model.fit(self.features, self.y, self.groups)
        np.testing.assert_allclose(model.weights_, [5 / 7, 2 / 7])
        np.testing.assert_allclose(model.predict_proba(self.features).sum(axis=1), 1)

    def test_checkpoint_before_optional_argument_remains_loadable(self):
        from experiments.posture_search.classical import build_classical
        model = build_classical("baseline", seed=53).fit(self.features, self.y, self.groups)
        expected = model.predict_proba(self.features)
        del model.ensemble_members
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "older_checkpoint.joblib"
            joblib.dump(model, path)
            loaded = joblib.load(path)
        self.assertIsNone(loaded.get_params()["ensemble_members"])
        np.testing.assert_array_equal(loaded.predict_proba(self.features), expected)


class TestRunnerArtifactOrchestration(unittest.TestCase):
    def test_fusion_loads_member_checkpoints_and_saves_evaluable_artifacts(self):
        from sklearn.dummy import DummyClassifier
        from experiments.posture_search.run import Arm, run_one
        count = 21
        synthetic = {
            "seq": np.random.default_rng(19).uniform(10, 100, (count, 36, 16)).astype(np.float32),
            "features": np.zeros((count, 596), dtype=np.float32),
            "personal": np.zeros((count, 49), dtype=np.float32),
            "y": np.tile(np.arange(7), 3),
            "groups": np.repeat(["train", "validation", "test"], 7),
            "record_ids": np.asarray([f"record_{index}" for index in range(count)]),
            "start_s": np.zeros(count),
        }
        split = {"id": "primary", "seed": 19, "train": ["train"],
                 "validation": ["validation"], "test": ["test"]}
        manifest = {"partitions": [split], "fingerprint": "synthetic_smoke_only"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for identifier in ("LEFT", "RIGHT"):
                folder = output / identifier / "primary"
                folder.mkdir(parents=True)
                member = Arm({"id": identifier, "family": "classical", "kind": "baseline",
                              "features": "baseline"}, seed=19)
                member.model = DummyClassifier(strategy="prior").fit(synthetic["features"][:7], synthetic["y"][:7])
                joblib.dump(member, folder / "model.joblib")
            spec = {"id": "FUSION", "family": "fusion", "kind": "soft_vote", "features": "mixed",
                    "members": ["LEFT", "RIGHT"], "hypothesis": "synthetic artifact smoke"}
            with redirect_stdout(io.StringIO()):
                run_one(spec, synthetic, manifest, output)
            folder = output / "FUSION" / "primary"
            record = json.loads((folder / "validation.json").read_text())
            self.assertEqual(record["checkpoint_sha256"], study_data.digest(folder / "model.joblib"))
            with np.load(folder / "validation_predictions.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["proba"].shape, (7, 7))
                np.testing.assert_allclose(saved["proba"], np.full((7, 7), 1 / 7))
                self.assertEqual(set(saved["groups"]), {"validation"})


class TestVerificationContract(unittest.TestCase):
    def test_no_outcomes_are_opened_before_lock_and_completed_test(self):
        from experiments.posture_search import verify as verification
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(verification, "source_inventory") as inventory:
                with self.assertRaisesRegex(verification.VerificationError, "Selection is not locked"):
                    verification.verify(directory)
                inventory.assert_not_called()
            (Path(directory) / "selection_lock.json").write_text("{}")
            with patch.object(verification, "read_json") as reader:
                with self.assertRaisesRegex(verification.VerificationError, "Test stage is incomplete"):
                    verification.verify(directory)
                reader.assert_not_called()

    def test_metric_check_rejects_changed_class_counts_and_confusion(self):
        from experiments.posture_search.verify import VerificationError, compare_metrics, recompute_metrics
        expected = recompute_metrics(np.arange(7), np.arange(7))
        saved = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in expected.items()}
        compare_metrics(saved, expected, "synthetic")
        saved["confusion_matrix"][0][0] += 1
        with self.assertRaisesRegex(VerificationError, "confusion_matrix"):
            compare_metrics(saved, expected, "synthetic")


if __name__ == "__main__":
    unittest.main()
