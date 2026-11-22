"""Independent checks of the research runner's leakage and evidence gates."""
from __future__ import annotations

import json
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import f1_score

from experiments.iterative_improvement import run


def fixture_data():
    labels = np.tile(np.arange(7), 3)
    groups = np.repeat(np.array(["train", "validation", "test"]), 7)
    return {
        "X": np.arange(21 * 596, dtype=float).reshape(21, 596),
        "y": labels,
        "groups": groups,
        "record_ids": np.array([f"record-{i}" for i in range(21)]),
    }


class PartitionContractTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture_data()
        self.part = {"id": "one", "train": ["train"],
            "validation": ["validation"], "test": ["test"]}

    def test_development_and_final_have_disjoint_subjects(self):
        train, valid = run.indices(self.data, self.part)
        np.testing.assert_array_equal(train, np.arange(7))
        np.testing.assert_array_equal(valid, np.arange(7, 14))
        train, valid = run.indices(self.data, self.part, final=True)
        np.testing.assert_array_equal(train, np.arange(14))
        np.testing.assert_array_equal(valid, np.arange(14, 21))

    def test_development_rejects_train_subject_in_holdout(self):
        self.part["test"] = ["train"]
        with self.assertRaises(ValueError):
            run.indices(self.data, self.part)

    def test_development_rejects_validation_subject_in_holdout(self):
        self.part["test"] = ["validation"]
        with self.assertRaises(ValueError):
            run.indices(self.data, self.part)

    def test_development_rejects_recording_shared_across_people(self):
        self.data["record_ids"][7] = self.data["record_ids"][0]
        with self.assertRaises(ValueError):
            run.indices(self.data, self.part)

    def test_final_rejects_missing_posture_class(self):
        self.data["y"][20] = 0
        with self.assertRaises(ValueError):
            run.indices(self.data, self.part, final=True)


class BootstrapContractTests(unittest.TestCase):
    def test_cluster_confusion_bootstrap_matches_explicit_subject_resampling(self):
        groups = np.repeat(np.array(["a", "b", "c"]), [7, 14, 21])
        truth = np.concatenate([np.arange(7), np.tile(np.arange(7), 2),
            np.tile(np.arange(7), 3)])
        baseline = truth.copy()
        baseline[::3] = (baseline[::3] + 1) % 7
        candidate = truth.copy()
        candidate[1::7] = (candidate[1::7] + 2) % 7
        repeats = 71
        subjects = sorted(set(groups))
        draws = np.random.default_rng(20260922).integers(0, len(subjects),
            (repeats, len(subjects)))
        differences = []
        for draw in draws:
            indices = np.concatenate([np.flatnonzero(groups == subjects[i]) for i in draw])
            scores = [f1_score(truth[indices], predictions[indices], labels=np.arange(7),
                average="macro", zero_division=0) for predictions in (baseline, candidate)]
            differences.append(scores[1] - scores[0])
        actual = run.cluster_bootstrap(truth, baseline, candidate, groups, repeats)
        self.assertAlmostEqual(actual["lower"], np.quantile(differences, .025), places=14)
        self.assertAlmostEqual(actual["upper"], np.quantile(differences, .975), places=14)
        self.assertEqual(actual["unit"], "subject")

    def test_identical_predictions_have_zero_interval(self):
        data = fixture_data()
        result = run.cluster_bootstrap(data["y"], data["y"], data["y"],
            data["groups"], repeats=53)
        self.assertEqual((result["lower"], result["upper"]), (0., 0.))


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        run.write(self.out / "protocol.json", {"schema_version": 1})
        self.data = fixture_data()
        self.part = {"id": "one", "train": ["train"],
            "validation": ["validation"], "test": ["test"]}
        self.pipeline = run.spec("extra_trees", {"n_estimators": 2})
        self.protocol = {"partitions": [self.part], "baseline": self.pipeline,
            "branches": {"branch": {"candidate": self.pipeline}},
            "seeds": run.SEEDS, "criteria": run.CRITERIA}

    def make_fit(self, name):
        train, valid = run.indices(self.data, self.part)
        probability = np.eye(7)[self.data["y"][valid]]
        with patch.object(run, "build_pipeline", return_value={"fixture_model": True}), \
                patch.object(run, "fit_pipeline") as fit, \
                patch.object(run, "aligned_predict_proba", return_value=probability):
            row = run.fit_one(self.out, self.data, self.pipeline, train, valid,
                self.out / "development" / name / self.part["id"], run.SEEDS[0])
        return row, fit

    def test_fit_receives_training_rows_and_groups_only(self):
        row, fit = self.make_fit("baseline")
        arguments = fit.call_args
        np.testing.assert_array_equal(arguments.args[2], self.data["X"][:7])
        np.testing.assert_array_equal(arguments.args[3], self.data["y"][:7])
        np.testing.assert_array_equal(arguments.kwargs["groups"], self.data["groups"][:7])
        self.assertNotIn("validation_data", arguments.kwargs)
        self.assertEqual(row["training_subjects"], ["train"])
        self.assertEqual(row["evaluation_subjects"], ["validation"])

    def test_resume_rejects_corrupted_predictions(self):
        self.make_fit("baseline")
        destination = self.out / "development/baseline/one"
        with (destination / "predictions.npz").open("ab") as stream:
            stream.write(b"tampered")
        train, valid = run.indices(self.data, self.part)
        with self.assertRaises(ValueError):
            run.fit_one(self.out, self.data, self.pipeline, train, valid,
                destination, run.SEEDS[0])

    def test_selection_summary_rejects_corrupted_prediction_evidence(self):
        self.make_fit("baseline")
        self.make_fit("candidate")
        with (self.out / "development/candidate/one/predictions.npz").open("ab") as stream:
            stream.write(b"tampered")
        with patch.object(run, "verify", return_value=self.protocol):
            with self.assertRaises(ValueError):
                run.development_summary(self.out, "candidate")

    def test_selection_summary_rejects_recipe_not_in_protocol(self):
        self.make_fit("baseline")
        self.make_fit("candidate")
        path = self.out / "development/candidate/one/metrics.json"
        row = json.loads(path.read_text())
        row["pipeline"]["model_params"]["n_estimators"] = 999
        run.write(path, row)
        run.write(path.parent / "metrics_seal.json", {"sha256": run.digest(path)})
        with patch.object(run, "verify", return_value=self.protocol):
            with self.assertRaises(ValueError):
                run.development_summary(self.out, "candidate")

    def test_saved_scores_are_recomputed_even_when_metrics_hash_matches(self):
        self.make_fit("baseline")
        destination = self.out / "development/baseline/one"
        path = destination / "metrics.json"
        row = run.read(path)
        row["metrics"]["macro_f1"] = .12345
        run.write(path, row)
        run.write(destination / "metrics_seal.json", {"sha256": run.digest(path)})
        with self.assertRaises(ValueError):
            run.verify_fit(self.out, destination, self.pipeline, run.SEEDS[0])

    def test_confirmation_rejects_changed_selection_before_fitting(self):
        lock = {"protocol_sha256": run.digest(self.out / "protocol.json"),
            "selected": None, "reason": "No qualifying candidate"}
        path = self.out / "selection_lock.json"
        run.write(path, lock)
        run.write(self.out / "selection_seal.json", {"sha256": run.digest(path)})
        lock["reason"] = "Changed after locking"
        run.write(path, lock)
        with patch.object(run, "verify", return_value=self.protocol), \
                patch.object(run, "load_data", return_value=self.data), \
                patch.object(run, "fit_one") as fit:
            with self.assertRaises(ValueError):
                run.confirm(self.out)
        fit.assert_not_called()

    def test_rejected_development_does_not_open_holdout_or_retrain(self):
        self.make_fit("baseline")
        self.make_fit("candidate")
        with patch.object(run, "verify", return_value=self.protocol):
            run.select(self.out)
        self.assertIsNone(run.read(self.out / "selection_lock.json")["selected"])
        with patch.object(run, "verify", return_value=self.protocol), \
                patch.object(run, "load_data", return_value=self.data), \
                patch.object(run, "check_originals"), \
                patch.object(run, "fit_one") as fit:
            run.confirm(self.out)
        result = run.read(self.out / "evaluation.json")
        self.assertFalse(result["improved"])
        self.assertFalse(result["holdout_evaluated"])
        fit.assert_not_called()

    def test_combination_cannot_nest_ensemble_or_displace_better_candidate(self):
        from copy import deepcopy
        single = deepcopy(self.pipeline)
        other = run.spec("random_forest", {"n_estimators": 2})
        ensemble = run.validate_pipeline({"family": "equal_soft_voting", "members": [single, other]})
        self.protocol["branches"] = {"branch": {"best_ensemble": ensemble, "first": single, "second": other}}
        def summary(out, name, baseline="baseline"):
            recipe = ensemble if name in {"best_ensemble", "combined"} else (single if name == "first" else other)
            return {"id": name, "pipeline": recipe, "qualified": True,
                "mean_f1": {"best_ensemble": .9, "combined": .85, "first": .8, "second": .79}[name],
                "worst_split": .7, "mean_delta": .02}
        with patch.object(run, "verify", return_value=self.protocol), \
                patch.object(run, "development_summary", side_effect=summary) as summaries, \
                patch.object(run, "load_data", return_value=self.data), \
                patch.object(run, "fit_one"):
            run.select(self.out)
        combo = run.read(self.out / "combination_lock.json")
        self.assertEqual(combo["members"], ["first", "second"])
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "best_ensemble")
        self.assertEqual(summaries.call_args.args, (self.out, "combined", "first"))


class UserAdoptionInitializationTests(unittest.TestCase):
    """An explicit exception is bound to one observed recipe and one run."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / "research/results/origin"
        self.previous = self.root / "research/results/previous"
        self.origin.mkdir(parents=True)
        self.previous.mkdir(parents=True)
        (self.origin / "data.npz").write_bytes(b"immutable synthetic input")
        self.dataset = {"cache_sha256": run.digest(self.origin / "data.npz"),
            "input_originals": {}, "partitions": [], "cohort": {},
            "registration_policy": "synthetic fixture"}
        run.write(self.origin / "dataset.json", self.dataset)
        for relative in ("src/example.py", "experiments/__init__.py", "AGENTS.md", *run.pipeline_sources(run.spec("extra_trees", {"n_estimators": 2}))):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic source fixture\n")
        self.pipeline = run.spec("extra_trees", {"n_estimators": 2})
        self.baseline = run.spec("extra_trees", {"n_estimators": 3})
        self.source_protocol = {"baseline": self.baseline, "partitions": self.dataset["partitions"],
            "dataset_sha256": run.digest(self.origin / "dataset.json"),
            "cache_sha256": run.digest(self.origin / "data.npz")}
        run.write(self.previous / "protocol.json", self.source_protocol)
        self.source_selection = {"candidates": [{"id": "candidate", "pipeline": self.pipeline,
            "mean_delta": .00941093, "qualified": False}], "selected": None}
        run.write(self.previous / "selection_lock.json", self.source_selection)
        self.original_selection = (self.previous / "selection_lock.json").read_bytes()
        self.authorization = {"kind": "explicit_user_research_adoption", "candidate_id": "candidate",
            "checkpoint_id": "synthetic-user-adopted-v1",
            "pipeline": self.pipeline, "source_run": "research/results/previous",
            "source_selection_sha256": run.digest(self.previous / "selection_lock.json"),
            "source_protocol_sha256": run.digest(self.previous / "protocol.json"),
            "observed_development_delta": .00941093, "minimum_development_delta": .009,
            "user_instruction": "Use this specific candidate as the research baseline."}
        self.authorization_path = self.root / "authorization.json"
        self.plan = {"baseline": self.baseline, "branches": {"adoption": {"candidate": self.pipeline}}}
        self.plan_path = self.root / "plan.json"

    def initialize(self, suffix="adopted", *, authorized=True):
        plan = deepcopy(self.plan)
        if authorized:
            run.write(self.authorization_path, self.authorization)
            plan["user_authorization"] = {"path": "authorization.json",
                "sha256": run.digest(self.authorization_path)}
        run.write(self.plan_path, plan)
        out = self.root / "research/results" / suffix
        def command(args, **kwargs):
            if args[:2] == ["git", "rev-parse"]:
                return "synthetic-revision\n"
            return ""
        with patch.object(run, "ROOT", self.root), patch.object(run, "ORIGIN", self.origin), \
                patch.object(run, "snapshot_protected", return_value={}), \
                patch.object(run.subprocess, "check_output", side_effect=command):
            run.initialize(out, self.plan_path)
        return out

    def test_authorized_exception_is_local_and_original_failure_is_immutable(self):
        original_criteria = deepcopy(run.CRITERIA)
        out = self.initialize()
        protocol = run.read(out / "protocol.json")
        self.assertEqual(protocol["criteria"]["minimum_development_delta"], .009)
        self.assertEqual(protocol["user_adoption"]["original_criteria"], original_criteria)
        self.assertEqual(protocol["branches"], self.plan["branches"])
        self.assertEqual(protocol["user_adoption"]["authorization_sha256"],
            run.digest(out / "user_authorization.json"))
        self.assertEqual(run.read(out / "user_authorization.json"), self.authorization)
        self.assertEqual((self.previous / "selection_lock.json").read_bytes(), self.original_selection)
        self.assertFalse(run.read(self.previous / "selection_lock.json")["candidates"][0]["qualified"])
        next_out = self.initialize("ordinary-next-iteration", authorized=False)
        self.assertEqual(run.read(next_out / "protocol.json")["criteria"], original_criteria)
        self.assertEqual(run.CRITERIA, original_criteria)

    def test_other_candidate_or_changed_recipe_cannot_use_authorization(self):
        for alteration in ("extra_candidate", "changed_recipe", "changed_candidate_id"):
            with self.subTest(alteration=alteration):
                original = deepcopy(self.plan)
                if alteration == "extra_candidate":
                    self.plan["branches"]["adoption"]["other"] = self.pipeline
                elif alteration == "changed_recipe":
                    changed = deepcopy(self.pipeline)
                    changed["model_params"]["n_estimators"] = 5
                    self.plan["branches"]["adoption"]["candidate"] = changed
                else:
                    self.plan["branches"]["adoption"] = {"other": self.pipeline}
                with self.assertRaises(ValueError):
                    self.initialize(alteration)
                self.assertFalse((self.root / "research/results" / alteration).exists())
                self.plan = original

    def test_changed_original_selection_cannot_be_adopted(self):
        changed = deepcopy(self.source_selection)
        changed["candidates"][0]["mean_delta"] += .0001
        run.write(self.previous / "selection_lock.json", changed)
        with self.assertRaisesRegex(ValueError, "Original development selection changed"):
            self.initialize()
        self.assertFalse((self.root / "research/results/adopted").exists())

    def test_exception_cannot_claim_a_threshold_above_the_observed_gain(self):
        self.authorization["minimum_development_delta"] = .0095
        with self.assertRaisesRegex(ValueError, "Invalid near-threshold"):
            self.initialize()
        self.assertFalse((self.root / "research/results/adopted").exists())

    def test_changed_baseline_or_partitions_cannot_reuse_the_observed_gain(self):
        self.plan["baseline"] = run.spec("random_forest", {"n_estimators": 3})
        with self.assertRaisesRegex(ValueError, "baseline or frozen evaluation differs"):
            self.initialize("changed-baseline")
        self.plan["baseline"] = self.baseline
        self.source_protocol["partitions"] = [{"id": "changed partition"}]
        run.write(self.previous / "protocol.json", self.source_protocol)
        self.authorization["source_protocol_sha256"] = run.digest(self.previous / "protocol.json")
        with self.assertRaisesRegex(ValueError, "baseline or frozen evaluation differs"):
            self.initialize("changed-partition")
        self.assertFalse((self.root / "research/results/changed-baseline").exists())
        self.assertFalse((self.root / "research/results/changed-partition").exists())


class UserAdoptionConfirmationTests(unittest.TestCase):
    """Adoption can finish measurement without converting failure into success."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.data = fixture_data()
        self.part = {"id": "one", "train": ["train"],
            "validation": ["validation"], "test": ["test"]}
        self.pipeline = run.spec("extra_trees", {"n_estimators": 2})
        self.baseline = run.spec("extra_trees", {"n_estimators": 3})

    def confirm(self, *, adopted=True, negative_seed=False, negative_final=False,
                uncertain=False, reproduction_error=False):
        criteria = deepcopy(run.CRITERIA)
        if adopted:
            criteria["minimum_development_delta"] = .009
        protocol = {"partitions": [self.part], "baseline": self.baseline,
            "branches": {"branch": {"candidate": self.pipeline}},
            "seeds": run.SEEDS, "criteria": criteria,
            "user_adoption": ({"candidate_id": "candidate", "original_criteria": run.CRITERIA}
                if adopted else None)}
        run.write(self.out / "protocol.json", protocol)
        selected = {"id": "candidate", "pipeline": self.pipeline, "qualified": True,
            "f1": [.71], "mean_delta": .00941093 if adopted else .02}
        lock = {"protocol_sha256": run.digest(self.out / "protocol.json"),
            "selected": selected, "reason": "Synthetic locked candidate"}
        run.write(self.out / "selection_lock.json", lock)
        run.write(self.out / "selection_seal.json",
            {"sha256": run.digest(self.out / "selection_lock.json")})
        paths = []
        def fit(out, data, recipe, train, valid, destination, seed):
            paths.append(destination.relative_to(out).as_posix())
            destination.mkdir(parents=True, exist_ok=True)
            candidate = recipe == self.pipeline
            truth = data["y"][valid]
            predicted = truth.copy()
            in_robustness = "robustness" in destination.parts
            failing_candidate = negative_seed if in_robustness else negative_final
            if candidate == failing_candidate:
                predicted[0] = 1
            probability = np.full((len(valid), 7), .025)
            probability[np.arange(len(valid)), predicted] = .85
            if reproduction_error and "reproduction" in destination.parts:
                probability[0, 0] += .001
                probability[0, 2] -= .001
            np.savez_compressed(destination / "predictions.npz", probability=probability,
                prediction=probability.argmax(1), truth=truth)
            run.joblib.dump({"probability": probability}, destination / "model.joblib")
            row = {"metrics": run.scores(truth, probability.argmax(1))}
            run.write(destination / "metrics.json", row)
            return row
        interval = {"lower": -.03 if uncertain else .02, "upper": .04,
            "unit": "subject", "replicates": criteria["bootstrap_replicates"]}
        with ExitStack() as stack:
            stack.enter_context(patch.object(run, "verify", return_value=protocol))
            stack.enter_context(patch.object(run, "load_data", return_value=self.data))
            stack.enter_context(patch.object(run, "development_summary", return_value=selected))
            stack.enter_context(patch.object(run, "fit_one", side_effect=fit))
            stack.enter_context(patch.object(run, "cluster_bootstrap", return_value=interval))
            stack.enter_context(patch.object(run, "check_originals"))
            stack.enter_context(patch.object(run, "aligned_predict_proba",
                side_effect=lambda model, features, classes: model["probability"]))
            run.confirm(self.out)
        evaluation = run.read(self.out / "evaluation.json")
        self.assertEqual(run.read(self.out / "evaluation_seal.json")["sha256"],
            run.digest(self.out / "evaluation.json"))
        return evaluation, paths

    def test_adoption_finishes_final_and_refit_after_negative_seed_without_green_result(self):
        evaluation, paths = self.confirm(negative_seed=True)
        self.assertTrue(evaluation["holdout_evaluated"])
        self.assertTrue(evaluation["reproduced"])
        self.assertFalse(evaluation["improved"])
        self.assertFalse(evaluation["meaningful_gain"])
        self.assertFalse(evaluation["original_autonomous_criteria_met"])
        self.assertEqual(evaluation["user_adoption"]["candidate_id"], "candidate")
        self.assertTrue(evaluation["baseline_preserved"])
        self.assertTrue(all(row["mean_delta"] < 0 for row in evaluation["development_seed_robustness"]))
        for seed in run.SEEDS:
            self.assertIn(f"final/{seed}/candidate", paths)
        self.assertIn("reproduction/candidate", paths)

    def test_ordinary_iteration_still_stops_before_holdout_on_negative_seed(self):
        evaluation, paths = self.confirm(adopted=False, negative_seed=True)
        self.assertFalse(evaluation["holdout_evaluated"])
        self.assertFalse(evaluation["improved"])
        self.assertFalse(any(path.startswith(("final/", "reproduction/")) for path in paths))

    def test_adoption_does_not_override_negative_final_or_uncertainty_gates(self):
        for field in ("negative_final", "uncertain"):
            with self.subTest(field=field):
                # Each confirmation writes an immutable, independent result.
                self.out = Path(self.temp.name) / field
                evaluation, paths = self.confirm(**{field: True})
                self.assertTrue(evaluation["holdout_evaluated"])
                self.assertTrue(evaluation["reproduced"])
                self.assertFalse(evaluation["improved"])
                self.assertFalse(evaluation["meaningful_gain"])
                self.assertIn("reproduction/candidate", paths)

    def test_adoption_still_fails_exact_reproduction(self):
        evaluation, _ = self.confirm(reproduction_error=True)
        self.assertFalse(evaluation["reproduced"])
        self.assertFalse(evaluation["improved"])
        self.assertFalse(evaluation["meaningful_gain"])
        self.assertGreater(run.read(self.out / "reproducibility.json")["max_probability_error"],
            run.CRITERIA["refit_probability_atol"])

    def test_passing_confirmation_does_not_relabel_the_original_development_exception(self):
        evaluation, _ = self.confirm()
        self.assertTrue(evaluation["reproduced"])
        self.assertTrue(all(row["mean_delta"] > 0 for row in evaluation["development_seed_robustness"]))
        self.assertTrue(all(row["delta"] > 0 for row in evaluation["final"]))
        self.assertFalse(evaluation["original_autonomous_criteria_met"])
        self.assertEqual(evaluation["user_adoption"]["original_criteria"], run.CRITERIA)


if __name__ == "__main__":
    unittest.main()
