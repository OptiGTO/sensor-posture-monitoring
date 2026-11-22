"""Frozen outer-TRAIN routing and selection for record OOF forest recipes."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.iterative_improvement import run, finalize, report
from experiments.iterative_improvement.record_training import append_record_training


class HardnessRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        rng = np.random.default_rng(1801)
        self.data = {"X": rng.normal(size=(21, 596)).astype(np.float32),
            "seq": np.zeros((21, 36, 16), dtype=np.float32), "y": np.tile(np.arange(7), 3),
            "groups": np.repeat(["train", "validation", "test"], 7),
            "record_ids": np.asarray([f"record-{i}.csv" for i in range(21)])}
        self.part = {"id": "one", "train": ["train"], "validation": ["validation"], "test": ["test"]}
        self.train, self.valid = np.arange(6, -1, -1), np.asarray([13, 7, 12, 8, 11, 9, 10])
        self.recipes = {}
        for name, family, coefficient in (("identity", "hardness_extra_trees", 0.),
                ("hard", "hardness_extra_trees", .5), ("rf", "hardness_random_forest", 0.)):
            self.recipes[name] = run.validate_pipeline({"family": family,
                "model_params": {"n_estimators": 4},
                "training": {"seed": run.SEEDS[0], "hardness": {
                    "coefficient": coefficient, "n_splits": 4, "seed_offset": 1000}}})
        self.baseline = run.spec("extra_trees", self.recipes["identity"]["model_params"])
        run.write(self.out / "protocol.json", {"synthetic": True})

    def combined(self):
        return run.validate_pipeline({"family": "hardness_soft_voting",
            "members": [self.recipes["hard"], self.recipes["rf"]]})

    def test_exact_feature_labels_people_record_rows_and_no_evaluation_fit(self):
        for i, recipe in enumerate([*self.recipes.values(), self.combined()]):
            with self.subTest(i=i), patch.object(run, "build_pipeline", return_value={}), \
                    patch.object(run, "fit_pipeline") as fit, patch.object(run, "aligned_predict_proba",
                    return_value=np.eye(7)[self.data["y"][self.valid]]) as predict:
                row = run.fit_one(self.out, self.data, recipe, self.train, self.valid, self.out / str(i), run.SEEDS[0])
            np.testing.assert_array_equal(fit.call_args.args[2], self.data["X"][self.train])
            np.testing.assert_array_equal(fit.call_args.args[3], self.data["y"][self.train])
            for key in ("groups", "record_ids"):
                np.testing.assert_array_equal(fit.call_args.kwargs[key], self.data[key][self.train])
            self.assertNotIn("validation_data", fit.call_args.kwargs)
            np.testing.assert_array_equal(predict.call_args.args[1], self.data["X"][self.valid])
            self.assertEqual(row["input_representation"], "feature_vector")

    def test_ordinary_restore_preserves_record_alignment_and_train_only_scope(self):
        extra = {"X": np.ones((3, 596), dtype=np.float32), "y": np.zeros(3, dtype=int),
            "groups": np.asarray(["validation", "train", "test"]),
            "record_ids": np.asarray(["extra-validation", "extra-train", "extra-test"])}
        np.savez_compressed(self.out / "supplemental_training.npz", **extra)
        X, y, groups, records, audit = append_record_training(self.data, extra, self.train)
        for name, original in self.recipes.items():
            recipe = deepcopy(original)
            recipe["training"]["data_policy"] = "include_ordinary_reserved_training_records_v1"
            with patch.object(run, "build_pipeline", return_value={}), patch.object(run, "fit_pipeline") as fit, \
                    patch.object(run, "aligned_predict_proba", return_value=np.eye(7)[self.data["y"][self.valid]]):
                row = run.fit_one(self.out, self.data, recipe, self.train, self.valid, self.out / name, run.SEEDS[0])
            for actual, expected in ((fit.call_args.args[2], X), (fit.call_args.args[3], y),
                    (fit.call_args.kwargs["groups"], groups), (fit.call_args.kwargs["record_ids"], records)):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(row["supplemental_training_audit"], audit)
            self.assertEqual(audit["added_record_paths"], ["extra-train"])

    def test_combination_uses_only_qualified_compatible_leaves(self):
        recipes = {"hard": self.recipes["hard"], "rf": self.recipes["rf"],
            "control": self.recipes["identity"], "existing": self.combined(), "legacy": self.baseline}
        means = {"hard": .85, "rf": .84, "control": .99, "existing": .95, "legacy": .83, "combined": .88}
        protocol = {"baseline": self.baseline, "branches": {"forest": recipes}, "partitions": [self.part],
            "seeds": run.SEEDS, "criteria": run.CRITERIA}
        def summary(out, name, baseline="baseline"):
            recipe = run.read(out / "combination_lock.json")["pipeline"] if name == "combined" else recipes[name]
            return {"id": name, "pipeline": recipe, "qualified": name != "control", "mean_f1": means[name],
                "worst_split": .7, "mean_delta": .02}
        with patch.object(run, "verify", return_value=protocol), patch.object(run, "development_summary", side_effect=summary), \
                patch.object(run, "load_data", return_value=self.data), patch.object(run, "fit_one"):
            run.select(self.out)
        combo = run.read(self.out / "combination_lock.json")
        self.assertEqual(combo["members"], ["hard", "rf"])
        self.assertEqual(combo["pipeline"]["family"], "hardness_soft_voting")
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "existing")

    def test_identity_control_rejects_weighting_rf_and_tree_drift(self):
        from src.pipeline_recipes import validate_identity_control
        validate_identity_control(self.baseline, self.recipes["identity"])
        for recipe in (self.recipes["hard"], self.recipes["rf"]):
            with self.assertRaises(ValueError):
                validate_identity_control(self.baseline, recipe)
        changed = deepcopy(self.recipes["identity"])
        changed["model_params"]["n_estimators"] += 1
        with self.assertRaises(ValueError):
            validate_identity_control(self.baseline, changed)

    def test_finalizer_dispatches_full_hardness_verification_before_generic_record_path(self):
        for recipe in [*self.recipes.values(), self.combined()]:
            model = type("Model", (), {"spec_": recipe, "effective_seed_": run.SEEDS[0]})()
            with patch("src.hardness_pipeline_v1.verify_fitted_hardness_identity") as verify, \
                    patch("src.record_pipeline_v1.verify_fitted_record_identity") as old:
                finalize._verify_fitted_identity(model, recipe, run.SEEDS[0])
            verify.assert_called_once_with(model, recipe, run.SEEDS[0])
            old.assert_not_called()

    def test_report_explains_train_only_weights_and_rf_bootstrap_without_gain_claim(self):
        hard, rf = self.recipes["hard"], self.recipes["rf"]
        self.assertIn("클래스 안의 창 평균", report.recipe_description(hard))
        self.assertIn("bootstrap=False", report.recipe_description(rf))
        self.assertIn("외부 평가 사람", report.hypothesis(hard))
        self.assertIn("성능 개선 후보가 아니다", report.hypothesis(self.recipes["identity"], identity_control=True))
        self.assertIn("강도나 inner fold 수", " ".join(report.next_experiments([{"pipeline": hard}])))


if __name__ == "__main__":
    unittest.main()
