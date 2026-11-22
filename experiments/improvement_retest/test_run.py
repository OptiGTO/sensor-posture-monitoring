"""Numerical and evidence-bound selection checks, without fitting models."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from experiments.improvement_retest import audit, run


class PairedBootstrapTests(unittest.TestCase):
    def test_vectorized_macro_f1_matches_sklearn_with_missing_classes(self):
        truth = np.array([0, 0, 1, 2, 2, 2])
        prediction = np.array([0, 1, 1, 0, 2, 3])
        matrix = confusion_matrix(truth, prediction, labels=np.arange(7))
        expected = f1_score(truth, prediction, labels=np.arange(7), average="macro", zero_division=0)
        self.assertAlmostEqual(float(run.f1_from_confusion(matrix)), expected, places=14)
        np.testing.assert_allclose(run.f1_from_confusion(np.stack([matrix, np.zeros((7, 7))])), [expected, 0])

    def test_paired_subject_bootstrap_matches_explicit_row_replication(self):
        truth = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 0, 0, 6])
        baseline = np.array([0, 0, 2, 3, 0, 5, 6, 1, 1, 2, 1, 0, 6])
        candidate = np.array([0, 1, 2, 3, 4, 5, 6, 0, 2, 2, 0, 1, 6])
        groups = np.array(["a"] * 7 + ["b"] * 3 + ["c"] * 3)
        subjects = np.unique(groups)
        sampled = np.random.default_rng(67).integers(0, len(subjects), (97, len(subjects)))
        values = []
        for sample in sampled:
            indices = np.concatenate([np.flatnonzero(groups == subjects[index]) for index in sample])
            score = lambda prediction: f1_score(truth[indices], prediction[indices], labels=np.arange(7),
                                                 average="macro", zero_division=0)
            values.append(score(candidate) - score(baseline))
        expected = np.quantile(values, [.025, .975])
        actual = run.paired_interval(truth, baseline, candidate, groups, seed=67, repetitions=97)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        self.assertEqual(run.paired_interval(truth, baseline, baseline, groups, repetitions=97), [0, 0])


class LockedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        truth = np.tile(np.arange(7), 4)
        self.labels = np.tile(truth, 3)
        self.groups = np.repeat(["train_subject", "validation_subject", "holdout_subject"], len(truth))
        self.indices = np.flatnonzero(self.groups == "validation_subject")
        np.savez_compressed(self.out / "data.npz", y=self.labels, groups=self.groups,
            record_ids=np.array([f"record_{index}" for index in range(len(self.labels))]),
            X=np.zeros((len(self.labels), 596)), seq=np.zeros((len(self.labels), 36, 16)))
        run.write_json(self.out / "dataset.json", {"fixture": True})
        parts = [{"id": f"split_{index}", "train": ["train_subject"],
            "validation": ["validation_subject"], "test": ["holdout_subject"]} for index in range(3)]
        self.protocol = {"candidates": {"rbf_svm": {"standard": [{"C": 1}, {"C": 2}, {"C": 3}],
            "relative": [{"C": 1}], "selected": [{"C": 1}]}}, "dataset_partitions": parts,
            "criteria": run.CRITERIA, "cache_sha256": run.digest(self.out / "data.npz"),
            "dataset_sha256": run.digest(self.out / "dataset.json"), "source_hashes": {}}
        run.write_json(self.out / "protocol.json", self.protocol)
        run.write_json(self.out / "protocol_lock.json", {"sha256": run.digest(self.out / "protocol.json")})
        for method, grid in self.protocol["candidates"]["rbf_svm"].items():
            for part_number, part in enumerate(parts):
                for candidate_index, params in enumerate(grid):
                    prediction = truth.copy()
                    errors = (14 if candidate_index == 0 else 7) if method == "standard" else (
                        0 if method == "relative" or part_number == 0 else 7)
                    prediction[-errors:] = (prediction[-errors:] + 1) % 7 if errors else prediction[-errors:]
                    directory = self.out / "development/rbf_svm" / method / part["id"]
                    directory.mkdir(parents=True, exist_ok=True)
                    prediction_path = directory / f"{candidate_index:02}.npz"
                    np.savez_compressed(prediction_path, indices=self.indices, truth=truth,
                        prediction=prediction, probability=np.eye(7)[prediction], groups=self.groups[self.indices])
                    run.write_json(directory / f"{candidate_index:02}.json", {"family": "rbf_svm",
                        "method": method, "split": part["id"], "candidate_index": candidate_index, "params": params,
                        "metrics": run.scores(truth, prediction), "training_details": {},
                        "per_subject": {"validation_subject": run.scores(truth, prediction)},
                        "prediction_sha256": run.digest(prediction_path),
                        "protocol_sha256": run.digest(self.out / "protocol.json")})

    def select(self):
        with redirect_stdout(io.StringIO()):
            run.select(self.out)
        return run.load_selection(self.out)

    def test_selection_uses_best_baseline_fixed_ties_and_two_split_improvement_gate(self):
        selected = self.select()["selected"]
        self.assertEqual(selected["rbf_svm/standard"]["candidate_index"], 1)
        self.assertTrue(selected["rbf_svm/relative"]["development_qualified"])
        one_split = selected["rbf_svm/selected"]
        self.assertGreater(one_split["mean_development_delta"], run.CRITERIA["minimum_development_mean_delta"])
        self.assertFalse(one_split["development_qualified"])
        with self.assertRaisesRegex(ValueError, "already locked"):
            self.select()

    def test_selection_rejects_existing_final_artifacts(self):
        (self.out / "final").mkdir()
        with self.assertRaisesRegex(ValueError, "Final artifacts"):
            self.select()

    def test_selection_lock_rejects_modified_bytes(self):
        self.select()
        path = self.out / "selection_lock.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "Selection lock changed"):
            run.load_selection(self.out)

    def test_protocol_and_feature_cache_tampering_fail_before_selection(self):
        path = self.out / "protocol.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "Protocol changed"):
            self.select()
        path.write_bytes(original)
        with (self.out / "data.npz").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            self.select()

    def test_train_validation_and_recording_overlap_are_rejected(self):
        data = dict(np.load(self.out / "data.npz"))
        part = dict(self.protocol["dataset_partitions"][0])
        part["validation"] = part["train"]
        with self.assertRaisesRegex(ValueError, "Subject overlap"):
            run.train_eval_indices(data, part)
        part = self.protocol["dataset_partitions"][0]
        data["record_ids"][self.indices[0]] = data["record_ids"][0]
        with self.assertRaisesRegex(ValueError, "Recording overlap"):
            run.train_eval_indices(data, part)

    def test_independent_audit_recomputes_metrics_before_locking_evidence(self):
        path = self.out / "development/rbf_svm/relative/split_0/00.json"
        value = run.read(path)
        value["metrics"]["macro_f1"] = .123
        run.write_json(path, value)
        with self.assertRaisesRegex(ValueError, "Metrics differ"):
            audit.audit_development(self.out)

    def test_independent_audit_checks_validation_holdout_disjointness(self):
        protocol = run.read(self.out / "protocol.json")
        protocol["dataset_partitions"][0]["test"] = ["validation_subject", "holdout_subject"]
        run.write_json(self.out / "protocol.json", protocol)
        run.write_json(self.out / "protocol_lock.json", {"sha256": run.digest(self.out / "protocol.json")})
        with self.assertRaisesRegex(ValueError, "Subject partitions overlap"):
            audit.audit_development(self.out)

    def test_independent_audit_binds_all_development_files_before_selection(self):
        result = audit.audit_development(self.out)
        self.assertTrue(result["passed"])
        self.assertEqual(result["details"]["fits"], 15)
        self.select()
        path = self.out / "development/rbf_svm/relative/split_0/00.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "audited evidence changed"):
            audit.audit_selection(self.out)

    def test_resealed_incorrect_selection_is_rejected_by_recomputed_audit(self):
        audit.audit_development(self.out)
        self.select()
        path = self.out / "selection_lock.json"
        selection = run.read(path)
        selection["selected"]["rbf_svm/selected"]["development_qualified"] = True
        run.write_json(path, selection)
        run.write_json(self.out / "selection_lock.sha256.json", {"sha256": run.digest(path)})
        with self.assertRaisesRegex(ValueError, "Selected candidates or criteria differ"):
            audit.audit_selection(self.out)

    def build_final_fixtures(self):
        audit.audit_development(self.out)
        lock = self.select()
        audit.audit_selection(self.out)
        with np.load(self.out / "data.npz") as data:
            indices = np.flatnonzero(data["groups"] == "holdout_subject")
            truth, groups, record_ids = data["y"][indices], data["groups"][indices], data["record_ids"][indices]
        for key, row in lock["selected"].items():
            if row["method"] != "standard" and not row["development_qualified"]:
                continue
            directory = self.out / "final" / key
            directory.mkdir(parents=True)
            prediction = truth.copy()
            if row["method"] == "standard":
                prediction[-7:] = (prediction[-7:] + 1) % 7
            np.savez_compressed(directory / "predictions.npz", indices=indices, truth=truth, prediction=prediction,
                probability=np.eye(7)[prediction], groups=groups, record_ids=record_ids)
            (directory / "model.joblib").write_bytes(b"opaque fixture; no deserialization or model fitting")
            run.write_json(directory / "metrics.json", {"family": row["family"], "method": row["method"],
                "params": row["params"], "metrics": run.scores(truth, prediction),
                "per_subject": {"holdout_subject": run.scores(truth, prediction)},
                "prediction_sha256": run.digest(directory / "predictions.npz"),
                "model_sha256": run.digest(directory / "model.joblib"),
                "model_size_bytes": (directory / "model.joblib").stat().st_size, "prediction_parity": True,
                "training_subjects": ["train_subject", "validation_subject"], "evaluation_subjects": ["holdout_subject"],
                "inference_ms_median": 1.0})
        with redirect_stdout(io.StringIO()):
            run.summarize(self.out)

    def test_independent_final_audit_recomputes_metrics_and_positive_ui_gate(self):
        self.build_final_fixtures()
        verified = audit.audit_final(self.out)
        self.assertEqual(verified["details"]["enabled_methods"], [{"family": "rbf_svm", "method": "relative"}])
        self.assertEqual(audit.audit_final(self.out), verified)
        path = self.out / "summary.json"
        value = run.read(path)
        value["enabled_methods"].append({"family": "rbf_svm", "method": "selected"})
        run.write_json(path, value)
        with self.assertRaisesRegex(ValueError, "Enabled methods differ"):
            audit.audit_final(self.out)

    def test_independent_final_audit_rejects_model_bytes_tampered_after_scoring(self):
        self.build_final_fixtures()
        with (self.out / "final/rbf_svm/relative/model.joblib").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Final model_sha256 differs"):
            audit.audit_final(self.out)

    def test_final_gate_rejects_macro_gain_with_unacceptable_single_class_loss(self):
        self.build_final_fixtures()
        directory = self.out / "final/rbf_svm/relative"
        values = dict(np.load(directory / "predictions.npz", allow_pickle=False))
        prediction = values["truth"].copy()
        prediction[prediction == 0] = 1
        values.update(prediction=prediction, probability=np.eye(7)[prediction])
        np.savez_compressed(directory / "predictions.npz", **values)
        row = run.read(directory / "metrics.json")
        row.update(metrics=run.scores(values["truth"], prediction),
            per_subject={"holdout_subject": run.scores(values["truth"], prediction)},
            prediction_sha256=run.digest(directory / "predictions.npz"))
        run.write_json(directory / "metrics.json", row)
        with redirect_stdout(io.StringIO()):
            run.summarize(self.out)
        summary = run.read(self.out / "summary.json")
        result = next(value for value in summary["comparisons"] if value["method"] == "relative")
        self.assertGreater(result["heldout_delta"], run.CRITERIA["minimum_holdout_delta"])
        self.assertFalse(result["improved"])
        self.assertEqual(audit.audit_final(self.out)["details"]["enabled_methods"], [])


if __name__ == "__main__":
    unittest.main()
