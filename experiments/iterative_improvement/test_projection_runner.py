"""Training-only learned-coordinate forest routing and explicit identity/combination gates."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.iterative_improvement import run, finalize, supplemental, report


class ProjectionRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        rng = np.random.default_rng(1501)
        self.data = {"X": rng.normal(size=(21, 596)).astype(np.float32),
            "seq": np.zeros((21, 36, 16), dtype=np.float32), "y": np.tile(np.arange(7), 3),
            "groups": np.repeat(["train", "validation", "test"], 7),
            "record_ids": np.asarray([f"record-{i}.csv" for i in range(21)])}
        self.part = {"id": "one", "train": ["train"], "validation": ["validation"], "test": ["test"]}
        self.train, self.valid = np.arange(6, -1, -1), np.asarray([13, 7, 12, 8, 11, 9, 10])
        self.recipes = {mode: run.validate_pipeline({"family": "projection_extra_trees",
            "feature_engineering": {"mode": mode}, "model_params": {"n_estimators": 4}})
            for mode in ("identity", "pca64", "all_plus_lda6")}
        self.baseline = run.spec("extra_trees", self.recipes["identity"]["model_params"])
        run.write(self.out / "protocol.json", {"synthetic": True})

    def test_leaves_and_ensemble_route_exact_feature_rows_without_metadata_or_eval_fit(self):
        recipes = [*self.recipes.values(), run.validate_pipeline({"family": "projection_soft_voting",
            "members": [self.recipes["identity"], self.recipes["pca64"]]})]
        for i, recipe in enumerate(recipes):
            with self.subTest(i=i), patch.object(run, "build_pipeline", return_value={}), \
                    patch.object(run, "fit_pipeline") as fit, patch.object(run, "aligned_predict_proba",
                    return_value=np.eye(7)[self.data["y"][self.valid]]) as predict:
                row = run.fit_one(self.out, self.data, recipe, self.train, self.valid, self.out / str(i), run.SEEDS[1])
            np.testing.assert_array_equal(fit.call_args.args[2], self.data["X"][self.train])
            np.testing.assert_array_equal(fit.call_args.args[3], self.data["y"][self.train])
            np.testing.assert_array_equal(fit.call_args.kwargs["groups"], self.data["groups"][self.train])
            self.assertNotIn("record_ids", fit.call_args.kwargs)
            self.assertNotIn("validation_data", fit.call_args.kwargs)
            np.testing.assert_array_equal(predict.call_args.args[1], self.data["X"][self.valid])
            self.assertEqual(row["input_representation"], "feature_vector")
            with np.load(self.out / str(i) / "predictions.npz") as saved:
                for key in ("groups", "record_ids"):
                    np.testing.assert_array_equal(saved[key], self.data[key][self.valid])

    def test_restore_uses_legacy_train_feature_helper_and_excludes_other_people(self):
        extra = {"X": np.ones((3, 596), dtype=np.float32), "y": np.zeros(3, dtype=int),
            "groups": np.asarray(["validation", "train", "test"]),
            "record_ids": np.asarray(["extra-validation", "extra-train", "extra-test"])}
        np.savez_compressed(self.out / "supplemental_training.npz", **extra)
        features, labels, groups, audit = supplemental.append_training(self.data, extra, self.train)
        for mode, original in self.recipes.items():
            recipe = deepcopy(original)
            recipe["training"]["data_policy"] = supplemental.POLICY
            with patch.object(run, "build_pipeline", return_value={}), patch.object(run, "fit_pipeline") as fit, \
                    patch.object(run, "aligned_predict_proba", return_value=np.eye(7)[self.data["y"][self.valid]]):
                row = run.fit_one(self.out, self.data, recipe, self.train, self.valid, self.out / mode, run.SEEDS[0])
            np.testing.assert_array_equal(fit.call_args.args[2], features)
            np.testing.assert_array_equal(fit.call_args.args[3], labels)
            np.testing.assert_array_equal(fit.call_args.kwargs["groups"], groups)
            self.assertNotIn("record_ids", fit.call_args.kwargs)
            self.assertEqual(row["supplemental_training_audit"], audit)
            self.assertEqual(audit["added_record_paths"], ["extra-train"])

    def test_only_qualified_projection_leaves_combine_and_existing_ensemble_is_not_nested(self):
        left, right = self.recipes["pca64"], self.recipes["all_plus_lda6"]
        existing = run.validate_pipeline({"family": "projection_soft_voting", "members": [left, right]})
        recipes = {"left": left, "right": right, "control": self.recipes["identity"], "existing": existing, "legacy": self.baseline}
        means = {"left": .85, "right": .84, "control": .99, "existing": .95, "legacy": .83, "combined": .88}
        protocol = {"baseline": self.baseline, "branches": {"support": recipes}, "partitions": [self.part],
            "seeds": run.SEEDS, "criteria": run.CRITERIA}
        def summary(out, name, baseline="baseline"):
            recipe = run.read(out / "combination_lock.json")["pipeline"] if name == "combined" else recipes[name]
            return {"id": name, "pipeline": recipe, "qualified": name != "control", "mean_f1": means[name],
                "worst_split": .7, "mean_delta": .02}
        with patch.object(run, "verify", return_value=protocol), patch.object(run, "development_summary", side_effect=summary), \
                patch.object(run, "load_data", return_value=self.data), patch.object(run, "fit_one"):
            run.select(self.out)
        combo = run.read(self.out / "combination_lock.json")
        self.assertEqual(combo["members"], ["left", "right"])
        self.assertEqual(combo["pipeline"]["family"], "projection_soft_voting")
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "existing")

    def identity_fixture(self):
        protocol = {"baseline": self.baseline, "branches": {"control": {"identity": self.recipes["identity"]}},
            "identity_controls": ["identity"], "partitions": [self.part], "seeds": run.SEEDS}
        truth = np.arange(7)
        for name in ("baseline", "identity"):
            folder = self.out / "development" / name / "one"
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "predictions.npz", truth=truth, prediction=truth, probability=.3 / 7 + .7 * np.eye(7))
        return protocol

    def test_identity_control_requires_exact_scores_and_rejects_projection_or_new_treecount(self):
        protocol = self.identity_fixture()
        with patch.object(run, "verify_fit", return_value={"metrics": {"macro_f1": 1.}}):
            run.verify_identity_controls(self.out, protocol)
        for field, value in (("mode", "pca64"), ("n_estimators", 8)):
            changed = deepcopy(protocol)
            candidate = changed["branches"]["control"]["identity"]
            candidate["feature_engineering" if field == "mode" else "model_params"][field] = value
            with patch.object(run, "verify_fit") as fit, self.assertRaises(ValueError):
                run.verify_identity_controls(self.out, changed)
            fit.assert_not_called()
        file = self.out / "development/identity/one/predictions.npz"
        values = dict(np.load(file))
        values["probability"][0, 1] += .01
        values["probability"][0, 2] -= .01
        np.savez_compressed(file, **values)
        with patch.object(run, "verify_fit", return_value={"metrics": {"macro_f1": 1.}}), self.assertRaises(ValueError):
            run.verify_identity_controls(self.out, protocol)

    def test_finalizer_calls_projection_state_verifier_for_leaves_and_ensemble(self):
        recipes = [*self.recipes.values(), run.validate_pipeline({"family": "projection_soft_voting",
            "members": [self.recipes["identity"], self.recipes["pca64"]]})]
        for recipe in recipes:
            model = type("Model", (), {"spec_": recipe, "effective_seed_": run.SEEDS[0]})()
            with patch("src.learned_projection_pipeline_v1.verify_fitted_projection_identity") as verify:
                finalize._verify_fitted_identity(model, recipe, run.SEEDS[0])
            verify.assert_called_once_with(model, recipe, run.SEEDS[0])

    def test_projection_identity_is_explicit_and_report_preserves_distinct_objectives(self):
        for mode in ("pca64", "all_plus_lda6"):
            with self.assertRaises(ValueError):
                run.validate_identity_control(self.baseline, self.recipes[mode])
        self.assertIn("무변형 경로 검증", report.hypothesis(self.recipes["identity"], identity_control=True))
        self.assertIn("라벨을 보지 않는", report.hypothesis(self.recipes["pca64"]))
        self.assertIn("005", report.hypothesis(self.recipes["all_plus_lda6"]))
        self.assertIn("full SVD", report.recipe_description(self.recipes["pca64"]))
        self.assertIn("602", report.recipe_description(self.recipes["all_plus_lda6"]))
        self.assertTrue(any("재탐색하지" in row for row in report.next_experiments(
            [{"pipeline": self.recipes["pca64"]}])))


if __name__ == "__main__":
    unittest.main()
