"""Local-geometry runner routing using synthetic arrays and mocked fitting."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.iterative_improvement import run, supplemental, record_training
from experiments.iterative_improvement.test_run import fixture_data


class LocalRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.data = fixture_data()
        self.data["X"] = self.data["X"].astype(np.float32)
        self.data["y"][7:14] = np.roll(np.arange(7), 2)
        self.data["seq"] = np.full((21, 36, 16), -999, dtype=np.float32)
        self.part = {"id": "one", "train": ["train"], "validation": ["validation"], "test": ["test"]}
        self.baseline = run.spec("extra_trees", {"n_estimators": 4})
        self.prototype = run.validate_pipeline({"family": "record_prototype",
            "model_params": {"n_neighbors": 1, "epsilon": 1e-12}, "training": {"seed": run.SEEDS[0]}})
        self.mixture = run.validate_pipeline({"family": "class_conditional_mixture",
            "model_params": {"n_components": 1}, "training": {"seed": run.SEEDS[0]}})
        self.other_mixture = deepcopy(self.mixture)
        self.other_mixture["model_params"]["n_components"] = 2
        self.mixed = run.validate_pipeline({"family": "local_soft_voting",
            "members": [self.prototype, self.mixture], "training": {"seed": run.SEEDS[0]}})
        self.mixtures = run.validate_pipeline({"family": "local_soft_voting",
            "members": [self.mixture, self.other_mixture], "training": {"seed": run.SEEDS[0]}})
        self.cases = (("prototype", self.prototype, True), ("mixture", self.mixture, False),
                      ("mixed", self.mixed, True), ("mixtures", self.mixtures, False))
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
            "y": np.zeros(4, dtype=int), "groups": np.asarray(["validation", "train", "test", "train"]),
            "record_ids": np.asarray(["validation/ordinary.csv", "train/ordinary-a.csv",
                                      "test/ordinary.csv", "train/ordinary-b.csv"])}
        np.savez_compressed(self.out / "supplemental_training.npz", **extra)
        return extra

    def test_all_local_inputs_are_raw_596_features_at_exact_requested_indices(self):
        rows = np.asarray([13, 7, 12, 8, 11, 9, 10])
        for name, recipe, needs_records in self.cases:
            with self.subTest(name=name):
                actual = run.model_input(self.data, recipe, rows)
                self.assertEqual(actual.shape, (7, 596))
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_array_equal(actual, self.data["X"][rows])
                self.assertEqual(run.input_representation(recipe), "feature_vector")
                self.assertEqual(run.requires_record_ids(recipe), needs_records)
                with self.assertRaisesRegex(ValueError, "rows differ"):
                    run.model_input({**self.data, "X": self.data["X"][:-1]}, recipe, rows)

    def test_actual_training_record_ids_are_passed_only_to_prototype_compositions(self):
        train = np.arange(6, -1, -1)
        valid = np.asarray([13, 7, 12, 8, 11, 9, 10])
        for name, recipe, needs_records in self.cases:
            with self.subTest(name=name):
                row, fit, predict = self.fit_mocked(recipe, name, train, valid)
                np.testing.assert_array_equal(fit.call_args.args[2], self.data["X"][train])
                np.testing.assert_array_equal(fit.call_args.args[3], self.data["y"][train])
                np.testing.assert_array_equal(fit.call_args.kwargs["groups"], self.data["groups"][train])
                self.assertEqual(set(fit.call_args.kwargs), {"groups", "record_ids"} if needs_records else {"groups"})
                if needs_records:
                    np.testing.assert_array_equal(fit.call_args.kwargs["record_ids"], self.data["record_ids"][train])
                np.testing.assert_array_equal(predict.call_args.args[1], self.data["X"][valid])
                self.assertEqual(row["input_representation"], "feature_vector")
                self.assertEqual(row["training_subjects"], ["train"])
                self.assertEqual(row["evaluation_subjects"], ["validation"])
                with np.load(self.out / name / "predictions.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["indices"], valid)
                    for key, source in (("truth", "y"), ("groups", "groups"), ("record_ids", "record_ids")):
                        np.testing.assert_array_equal(saved[key], self.data[source][valid])
                self.assertEqual(run.verify_fit(self.out, self.out / name, recipe, run.SEEDS[1]), row)

    def test_ordinary_restoration_routes_helper_and_keeps_train_rows_records_and_audit_aligned(self):
        extra = self.supplement()
        train = np.arange(6, -1, -1)
        valid = np.asarray([13, 7, 12, 8, 11, 9, 10])
        expected_X = np.concatenate((self.data["X"][train], extra["X"][[1, 3]]))
        expected_y = np.concatenate((self.data["y"][train], extra["y"][[1, 3]]))
        expected_records = np.concatenate((self.data["record_ids"][train], extra["record_ids"][[1, 3]]))
        audits = []
        for name, original, needs_records in self.cases:
            recipe = deepcopy(original)
            recipe["training"]["data_policy"] = supplemental.POLICY
            for member in recipe.get("members", ()):
                member["training"]["data_policy"] = supplemental.POLICY
            with self.subTest(name=name), \
                    patch.object(record_training, "append_record_training", wraps=record_training.append_record_training) as record_append, \
                    patch.object(supplemental, "append_training", wraps=supplemental.append_training) as ordinary_append:
                row, fit, predict = self.fit_mocked(recipe, "restored_" + name, train, valid)
                self.assertEqual(record_append.call_count, int(needs_records))
                # The record-aware helper delegates the identical row selection
                # to the original helper before attaching recording identities.
                ordinary_append.assert_called_once()
                np.testing.assert_array_equal(ordinary_append.call_args.args[2], train)
                np.testing.assert_array_equal(fit.call_args.args[2], expected_X)
                np.testing.assert_array_equal(fit.call_args.args[3], expected_y)
                np.testing.assert_array_equal(fit.call_args.kwargs["groups"], np.repeat("train", 9))
                self.assertEqual(set(fit.call_args.kwargs), {"groups", "record_ids"} if needs_records else {"groups"})
                if needs_records:
                    np.testing.assert_array_equal(fit.call_args.kwargs["record_ids"], expected_records)
                np.testing.assert_array_equal(predict.call_args.args[1], self.data["X"][valid])
                audit = row["supplemental_training_audit"]
                self.assertEqual(audit["training_subjects"], ["train"])
                self.assertEqual(audit["added_windows"], 2)
                self.assertEqual(audit["added_record_paths"], ["train/ordinary-a.csv", "train/ordinary-b.csv"])
                self.assertTrue(audit["evaluation_rows_unchanged"])
                audits.append(audit)
        self.assertTrue(all(audit == audits[0] for audit in audits))

    def select_mocked(self, recipes, scores, qualified=None, versus_member=True):
        protocol = {"baseline": self.baseline, "branches": {"candidates": recipes},
                    "partitions": [self.part], "seeds": run.SEEDS, "criteria": run.CRITERIA}
        qualified = set(recipes) if qualified is None else set(qualified)
        def summary(out, name, baseline="baseline"):
            recipe = run.read(self.out / "combination_lock.json")["pipeline"] if name == "combined" else recipes[name]
            passed = (baseline == "baseline" or versus_member) if name == "combined" else name in qualified
            return {"id": name, "pipeline": recipe, "qualified": passed, "mean_f1": scores[name],
                    "worst_split": scores[name] - .02, "mean_delta": .02, "baseline": baseline}
        with patch.object(run, "verify", return_value=protocol), \
                patch.object(run, "development_summary", side_effect=summary), \
                patch.object(run, "load_data", return_value=self.data), \
                patch.object(run, "fit_one") as fit, redirect_stdout(io.StringIO()):
            run.select(self.out)
        return fit

    def test_only_qualified_local_candidates_with_shared_policy_form_a_combination(self):
        failed = deepcopy(self.prototype)
        failed["model_params"]["n_neighbors"] = 3
        other_policy = deepcopy(self.other_mixture)
        other_policy["training"]["data_policy"] = supplemental.POLICY
        recipes = {"prototype": self.prototype, "mixture": self.mixture, "legacy": self.baseline,
                   "other_policy": other_policy, "failed": failed}
        fit = self.select_mocked(recipes,
            {"prototype": .90, "mixture": .89, "legacy": .88, "other_policy": .87, "failed": .99, "combined": .93},
            qualified=set(recipes) - {"failed"})
        combo = run.read(self.out / "combination_lock.json")
        self.assertEqual(combo["members"], ["prototype", "mixture"])
        self.assertEqual(combo["pipeline"]["family"], "local_soft_voting")
        self.assertEqual(combo["pipeline"]["members"], [self.prototype, self.mixture])
        self.assertTrue(run.requires_record_ids(fit.call_args.args[2]))
        train, valid = run.indices(self.data, self.part)
        np.testing.assert_array_equal(fit.call_args.args[3], train)
        np.testing.assert_array_equal(fit.call_args.args[4], valid)
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "combined")

    def test_same_feature_shape_does_not_allow_local_and_legacy_mixing(self):
        fit = self.select_mocked({"prototype": self.prototype, "legacy": self.baseline},
                                {"prototype": .90, "legacy": .89})
        fit.assert_not_called()
        self.assertFalse((self.out / "combination_lock.json").exists())
        with self.assertRaises(ValueError):
            run.validate_pipeline({"family": "local_soft_voting", "members": [self.prototype, self.baseline]})

    def test_qualified_mixture_only_combination_requires_no_record_identities(self):
        fit = self.select_mocked({"mixture1": self.mixture, "mixture2": self.other_mixture},
                                {"mixture1": .90, "mixture2": .89, "combined": .93})
        combo = run.read(self.out / "combination_lock.json")
        self.assertEqual(combo["members"], ["mixture1", "mixture2"])
        self.assertEqual(combo["pipeline"]["family"], "local_soft_voting")
        self.assertFalse(run.requires_record_ids(combo["pipeline"]))
        self.assertEqual(fit.call_args.args[2], combo["pipeline"])
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "combined")

    def test_local_ensemble_cannot_be_nested_or_displaced_by_a_weaker_combination(self):
        self.select_mocked({"existing": self.mixed, "prototype": self.prototype, "mixture": self.mixture},
                           {"existing": .95, "prototype": .90, "mixture": .89, "combined": .93})
        combo = run.read(self.out / "combination_lock.json")
        self.assertEqual(combo["members"], ["prototype", "mixture"])
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "existing")
        with self.assertRaises(ValueError):
            run.validate_pipeline({"family": "local_soft_voting", "members": [self.prototype, self.mixed]})

    def test_combination_must_improve_over_best_qualified_member(self):
        self.select_mocked({"prototype": self.prototype, "mixture": self.mixture},
                           {"prototype": .90, "mixture": .89, "combined": .93}, versus_member=False)
        selection = run.read(self.out / "selection_lock.json")
        self.assertEqual(selection["selected"]["id"], "prototype")
        combined = next(row for row in selection["candidates"] if row["id"] == "combined")
        self.assertFalse(combined["qualified"])
        self.assertEqual(combined["versus_best_member"]["baseline"], "prototype")
        self.assertFalse(combined["versus_best_member"]["qualified"])

    def test_finalizer_uses_local_identity_before_generic_record_route_or_replay(self):
        from experiments.iterative_improvement import finalize
        for name, recipe, _ in self.cases:
            model = SimpleNamespace(spec_=recipe, effective_seed_=run.SEEDS[1])
            with self.subTest(name=name), \
                    patch("src.local_geometry_pipeline_v1.verify_fitted_local_identity") as verify, \
                    patch("src.record_pipeline_v1.verify_fitted_record_identity",
                          side_effect=AssertionError("Local recipe entered generic record verification")):
                finalize._verify_fitted_identity(model, recipe, run.SEEDS[1])
                verify.assert_called_once_with(model, recipe, run.SEEDS[1])
            with patch("src.local_geometry_pipeline_v1.verify_fitted_local_identity",
                       side_effect=ValueError("Fitted local identity changed")), \
                    patch.object(finalize, "aligned_predict_proba") as predict:
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    finalize.verified_model_probabilities(model, recipe, run.SEEDS[1], self.data["X"])
                predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
