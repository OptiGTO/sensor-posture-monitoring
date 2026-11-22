"""Spatial runner contracts with synthetic rows and mocked estimator fitting."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.iterative_improvement import run, supplemental
from experiments.iterative_improvement.test_run import fixture_data


class SpatialRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.data = fixture_data()
        self.data["X"] = self.data["X"].astype(np.float32)
        self.data["y"][7:14] = np.roll(np.arange(7), 2)
        # Distinct shape and contents make accidental sequence routing visible.
        self.data["seq"] = np.full((21, 36, 16), -999, dtype=np.float32)
        self.part = {"id": "one", "train": ["train"], "validation": ["validation"], "test": ["test"]}
        self.baseline = run.spec("extra_trees", {"n_estimators": 4})
        self.ordinal = run.validate_pipeline({"family": "ordinal_extra_trees",
            "feature_engineering": {"mode": "all_plus_mean_order"},
            "model_params": {"n_estimators": 4, "min_samples_leaf": 1},
            "training": {"seed": run.SEEDS[0], "weighting": "class_balanced"}})
        self.qda = run.validate_pipeline({"family": "shrinkage_qda",
            "feature_engineering": {"mode": "abs_rel_sensor_means"},
            "model_params": {"reg_param": .5},
            "training": {"seed": run.SEEDS[0], "weighting": "uniform_class_prior"}})
        self.ensemble = run.validate_pipeline({"family": "spatial_soft_voting",
            "members": [self.ordinal, self.qda], "feature_engineering": {"mode": "per_member"},
            "training": {"seed": run.SEEDS[0], "weighting": "per_member"}})
        run.write(self.out / "protocol.json", {"synthetic": True})

    def fit_mocked(self, recipe, name, train, valid):
        probability = np.eye(7)[self.data["y"][valid]]
        with patch.object(run, "build_pipeline", return_value={"synthetic_estimator": name}) as builder, \
                patch.object(run, "fit_pipeline") as fitter, \
                patch.object(run, "aligned_predict_proba", return_value=probability) as predictor:
            result = run.fit_one(self.out, self.data, recipe, train, valid, self.out / name, run.SEEDS[1])
        builder.assert_called_once_with(recipe, seed=run.SEEDS[1])
        return result, fitter, predictor

    def supplement(self):
        extra = {"X": np.repeat(-np.arange(1, 5, dtype=np.float32)[:, None], 596, axis=1),
            "y": np.zeros(4, dtype=int),
            "groups": np.asarray(["validation", "train", "test", "train"]),
            "record_ids": np.asarray(["validation/ordinary.csv", "train/ordinary-a.csv",
                                      "test/ordinary.csv", "train/ordinary-b.csv"])}
        np.savez_compressed(self.out / "supplemental_training.npz", **extra)
        return extra

    def test_model_input_is_untransformed_596_features_for_all_spatial_families(self):
        rows = np.asarray([13, 7, 12, 8, 11, 9, 10])
        for recipe in (self.ordinal, self.qda, self.ensemble):
            with self.subTest(family=recipe["family"]):
                actual = run.model_input(self.data, recipe, rows)
                self.assertEqual(actual.shape, (7, 596))
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_array_equal(actual, self.data["X"][rows])
                self.assertEqual(run.input_representation(recipe), "feature_vector")
                self.assertFalse(run.requires_record_ids(recipe))
                with self.assertRaisesRegex(ValueError, "rows differ"):
                    run.model_input({**self.data, "X": self.data["X"][:-1]}, recipe, rows)

    def test_fit_predict_and_saved_eval_identities_preserve_nonmonotonic_rows(self):
        train = np.arange(6, -1, -1)
        valid = np.asarray([13, 7, 12, 8, 11, 9, 10])
        for recipe in (self.ordinal, self.qda, self.ensemble):
            with self.subTest(family=recipe["family"]):
                name = recipe["family"]
                row, fit, predict = self.fit_mocked(recipe, name, train, valid)
                np.testing.assert_array_equal(fit.call_args.args[2], self.data["X"][train])
                np.testing.assert_array_equal(fit.call_args.args[3], self.data["y"][train])
                np.testing.assert_array_equal(fit.call_args.kwargs["groups"], self.data["groups"][train])
                self.assertEqual(set(fit.call_args.kwargs), {"groups"})
                np.testing.assert_array_equal(predict.call_args.args[1], self.data["X"][valid])
                self.assertEqual(predict.call_args.args[2], 7)
                self.assertEqual(row["input_representation"], "feature_vector")
                self.assertEqual(row["training_subjects"], ["train"])
                self.assertEqual(row["evaluation_subjects"], ["validation"])
                with np.load(self.out / name / "predictions.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["indices"], valid)
                    for key, source in (("truth", "y"), ("groups", "groups"), ("record_ids", "record_ids")):
                        np.testing.assert_array_equal(saved[key], self.data[source][valid])
                self.assertEqual(run.verify_fit(self.out, self.out / name, recipe, run.SEEDS[1]), row)

    def test_ordinary_restoration_uses_same_train_only_rows_and_audit_as_baseline(self):
        extra = self.supplement()
        train, valid = run.indices(self.data, self.part)
        train = train[::-1]
        expected_X = np.concatenate((self.data["X"][train], extra["X"][[1, 3]]))
        expected_y = np.concatenate((self.data["y"][train], extra["y"][[1, 3]]))
        audits = []
        for original in (self.baseline, self.ordinal, self.qda):
            recipe = deepcopy(original)
            recipe["training"]["data_policy"] = supplemental.POLICY
            with self.subTest(family=recipe["family"]):
                row, fit, predict = self.fit_mocked(recipe, "restored_" + recipe["family"], train, valid)
                np.testing.assert_array_equal(fit.call_args.args[2], expected_X)
                np.testing.assert_array_equal(fit.call_args.args[3], expected_y)
                np.testing.assert_array_equal(fit.call_args.kwargs["groups"], np.repeat("train", 9))
                self.assertEqual(set(fit.call_args.kwargs), {"groups"})
                np.testing.assert_array_equal(predict.call_args.args[1], self.data["X"][valid])
                audit = row["supplemental_training_audit"]
                self.assertEqual(audit["training_subjects"], ["train"])
                self.assertEqual(audit["added_windows"], 2)
                self.assertEqual(audit["added_record_paths"], ["train/ordinary-a.csv", "train/ordinary-b.csv"])
                self.assertTrue(audit["evaluation_rows_unchanged"])
                audits.append(audit)
        self.assertEqual(audits[0], audits[1])
        self.assertEqual(audits[1], audits[2])

    def test_supplement_overlap_with_base_evaluation_record_fails_before_fit(self):
        extra = self.supplement()
        extra["record_ids"][0] = self.data["record_ids"][7]
        np.savez_compressed(self.out / "supplemental_training.npz", **extra)
        recipe = deepcopy(self.ordinal)
        recipe["training"]["data_policy"] = supplemental.POLICY
        train, valid = run.indices(self.data, self.part)
        with patch.object(run, "build_pipeline", return_value={}), patch.object(run, "fit_pipeline") as fit, \
                patch.object(run, "aligned_predict_proba") as predict:
            with self.assertRaisesRegex(ValueError, "evaluation recording"):
                run.fit_one(self.out, self.data, recipe, train, valid, self.out / "rejected", run.SEEDS[0])
        fit.assert_not_called()
        predict.assert_not_called()
        self.assertFalse((self.out / "rejected/model.joblib").exists())
        self.assertFalse((self.out / "rejected/predictions.npz").exists())

    def select_mocked(self, recipes, scores, qualified=None, versus_member=True):
        protocol = {"baseline": self.baseline, "branches": {"candidates": recipes},
                    "partitions": [self.part], "seeds": run.SEEDS, "criteria": run.CRITERIA}
        qualified = set(recipes) if qualified is None else set(qualified)
        def summary(out, name, baseline="baseline"):
            recipe = run.read(out / "combination_lock.json")["pipeline"] if name == "combined" else recipes[name]
            passed = (baseline == "baseline" or versus_member) if name == "combined" else name in qualified
            return {"id": name, "pipeline": recipe, "qualified": passed, "mean_f1": scores[name],
                    "worst_split": scores[name] - .02, "mean_delta": .02, "baseline": baseline}
        with patch.object(run, "verify", return_value=protocol), \
                patch.object(run, "development_summary", side_effect=summary), \
                patch.object(run, "load_data", return_value=self.data), \
                patch.object(run, "fit_one") as fit, redirect_stdout(io.StringIO()):
            run.select(self.out)
        return fit

    def test_only_qualified_spatial_candidates_with_identical_policy_are_combined(self):
        failed = deepcopy(self.ordinal)
        failed["feature_engineering"]["mode"] = "all_plus_mean_median_order"
        other_policy = deepcopy(self.qda)
        other_policy["training"]["data_policy"] = supplemental.POLICY
        recipes = {"ordinal": self.ordinal, "qda": self.qda, "legacy": self.baseline,
                   "other_policy": other_policy, "failed": failed}
        scores = {"ordinal": .90, "qda": .89, "legacy": .88, "other_policy": .87, "failed": .99, "combined": .93}
        fit = self.select_mocked(recipes, scores, qualified=set(recipes) - {"failed"})
        combo = run.read(self.out / "combination_lock.json")
        self.assertEqual(combo["members"], ["ordinal", "qda"])
        self.assertEqual(combo["pipeline"]["family"], "spatial_soft_voting")
        self.assertEqual(combo["pipeline"]["members"], [self.ordinal, self.qda])
        self.assertEqual(run.input_representation(fit.call_args.args[2]), "feature_vector")
        train, valid = run.indices(self.data, self.part)
        np.testing.assert_array_equal(fit.call_args.args[3], train)
        np.testing.assert_array_equal(fit.call_args.args[4], valid)
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "combined")

    def test_same_596_input_does_not_make_legacy_and_spatial_recipes_compatible(self):
        fit = self.select_mocked({"ordinal": self.ordinal, "legacy": self.baseline},
                                {"ordinal": .90, "legacy": .89})
        fit.assert_not_called()
        self.assertFalse((self.out / "combination_lock.json").exists())
        with self.assertRaises(ValueError):
            run.validate_pipeline({"family": "spatial_soft_voting", "members": [self.ordinal, self.baseline]})

    def test_existing_spatial_ensemble_is_not_nested_or_displaced_by_weaker_combination(self):
        self.select_mocked({"existing": self.ensemble, "ordinal": self.ordinal, "qda": self.qda},
                           {"existing": .95, "ordinal": .90, "qda": .89, "combined": .93})
        combo = run.read(self.out / "combination_lock.json")
        self.assertEqual(combo["members"], ["ordinal", "qda"])
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "existing")
        with self.assertRaises(ValueError):
            run.validate_pipeline({"family": "spatial_soft_voting", "members": [self.ordinal, self.ensemble]})

    def test_combination_must_also_improve_over_best_individual_member(self):
        self.select_mocked({"ordinal": self.ordinal, "qda": self.qda},
                           {"ordinal": .90, "qda": .89, "combined": .91}, versus_member=False)
        selected = run.read(self.out / "selection_lock.json")
        self.assertEqual(selected["selected"]["id"], "ordinal")
        combined = next(row for row in selected["candidates"] if row["id"] == "combined")
        self.assertFalse(combined["qualified"])
        self.assertFalse(combined["versus_best_member"]["qualified"])

    def test_finalizer_checks_spatial_identity_before_replay_including_voting(self):
        from experiments.iterative_improvement import finalize
        for recipe in (self.ordinal, self.qda, self.ensemble):
            model = SimpleNamespace(spec_=recipe, effective_seed_=run.SEEDS[1])
            with self.subTest(family=recipe["family"]), \
                    patch("src.spatial_pipeline_v1.verify_fitted_spatial_identity") as verify:
                finalize._verify_fitted_identity(model, recipe, run.SEEDS[1])
                verify.assert_called_once_with(model, recipe, run.SEEDS[1])
            with patch("src.spatial_pipeline_v1.verify_fitted_spatial_identity",
                       side_effect=ValueError("Fitted spatial identity changed")), \
                    patch.object(finalize, "aligned_predict_proba") as predict:
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    finalize.verified_model_probabilities(model, recipe, run.SEEDS[1], self.data["X"])
                predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
