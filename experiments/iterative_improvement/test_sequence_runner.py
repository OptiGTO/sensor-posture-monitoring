"""Synthetic runner routing checks; all estimator construction/fitting is mocked."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.iterative_improvement import run
from experiments.iterative_improvement import supplemental, supplemental_sequence
from src.window_pipeline import extract_window_features


class SequenceRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        rng = np.random.default_rng(810)
        seq = rng.uniform(10, 90, (21, 36, 16)).astype(np.float32)
        self.data = {"seq": seq, "X": extract_window_features(seq),
                     "y": np.tile(np.arange(7), 3),
                     "groups": np.repeat(["train", "validation", "test"], 7),
                     "record_ids": np.asarray([f"base-{i}.csv" for i in range(21)])}
        self.part = {"id": "one", "train": ["train"], "validation": ["validation"], "test": ["test"]}
        self.feature = run.spec("extra_trees", {"n_estimators": 2})
        self.cnn = run.validate_pipeline({"family": "cnn1d", "model_params": {"epochs": 1},
                                          "training": {"seed": run.SEEDS[0]}})
        self.rocket = run.validate_pipeline({"family": "minirocket", "model_params": {"num_kernels": 84},
                                             "training": {"seed": run.SEEDS[0]}})
        run.write(self.out / "protocol.json", {"synthetic": True})

    def fit_mocked(self, recipe, name, train, valid):
        probability = np.eye(7)[self.data["y"][valid]]
        with patch.object(run, "build_pipeline", return_value={"synthetic_estimator": name}) as builder, \
                patch.object(run, "fit_pipeline") as fitter, \
                patch.object(run, "aligned_predict_proba", return_value=probability) as predictor:
            result = run.fit_one(self.out, self.data, recipe, train, valid,
                                 self.out / name, run.SEEDS[1])
        builder.assert_called_once_with(recipe, seed=run.SEEDS[1])
        return result, fitter, predictor

    def test_model_input_preserves_identical_nonmonotonic_row_selection(self):
        rows = np.asarray([13, 7, 12, 8, 11, 9, 10])
        for recipe, key in ((self.feature, "X"), (self.cnn, "seq"), (self.rocket, "seq")):
            with self.subTest(family=recipe["family"]):
                np.testing.assert_array_equal(run.model_input(self.data, recipe, rows), self.data[key][rows])
                changed = {**self.data, key: self.data[key][:-1]}
                with self.assertRaisesRegex(ValueError, "rows differ"):
                    run.model_input(changed, recipe, rows)

    def test_fit_and_prediction_use_sequence_rows_with_original_eval_identities(self):
        train = np.arange(6, -1, -1)
        valid = np.asarray([13, 7, 12, 8, 11, 9, 10])
        for recipe in (self.cnn, self.rocket):
            with self.subTest(family=recipe["family"]):
                row, fit, predict = self.fit_mocked(recipe, recipe["family"], train, valid)
                np.testing.assert_array_equal(fit.call_args.args[2], self.data["seq"][train])
                np.testing.assert_array_equal(fit.call_args.args[3], self.data["y"][train])
                np.testing.assert_array_equal(fit.call_args.kwargs["groups"], self.data["groups"][train])
                self.assertNotIn("validation_data", fit.call_args.kwargs)
                np.testing.assert_array_equal(predict.call_args.args[1], self.data["seq"][valid])
                self.assertEqual(predict.call_args.args[2], 7)
                self.assertEqual(row["input_representation"], "sensor_sequence")
                self.assertEqual(row["training_subjects"], ["train"])
                self.assertEqual(row["evaluation_subjects"], ["validation"])
                folder = self.out / recipe["family"]
                with np.load(folder / "predictions.npz", allow_pickle=False) as prediction:
                    np.testing.assert_array_equal(prediction["indices"], valid)
                    for saved, original in (("truth", "y"), ("groups", "groups"), ("record_ids", "record_ids")):
                        np.testing.assert_array_equal(prediction[saved], self.data[original][valid])
                self.assertEqual(run.verify_fit(self.out, folder, recipe, run.SEEDS[1]), row)

    def test_supplemented_sequence_and_feature_fits_have_identical_labels_groups_audits(self):
        rng = np.random.default_rng(811)
        sequence = rng.uniform(20, 80, (3, 36, 16)).astype(np.float32)
        extra = {"seq": sequence, "X": extract_window_features(sequence), "y": np.zeros(3, dtype=int),
                 "groups": np.asarray(["train", "validation", "test"]),
                 "record_ids": np.asarray([f"{g}/ordinary.csv" for g in ("train", "validation", "test")])}
        np.savez_compressed(self.out / "supplemental_training.npz", **extra)
        train, valid = run.indices(self.data, self.part)
        expected_X, expected_y, expected_groups, expected_audit = supplemental.append_training(self.data, extra, train)
        expected_seq, _, _, sequence_audit = supplemental_sequence.append_sequence_training(self.data, extra, train)
        self.assertEqual(expected_audit, sequence_audit)
        results = []
        for original, expected in ((self.feature, expected_X), (self.cnn, expected_seq)):
            recipe = deepcopy(original)
            recipe["training"]["data_policy"] = supplemental.POLICY
            row, fit, predict = self.fit_mocked(recipe, "supplemented_" + recipe["family"], train, valid)
            np.testing.assert_array_equal(fit.call_args.args[2], expected)
            np.testing.assert_array_equal(fit.call_args.args[3], expected_y)
            np.testing.assert_array_equal(fit.call_args.kwargs["groups"], expected_groups)
            np.testing.assert_array_equal(predict.call_args.args[1], run.model_input(self.data, recipe, valid))
            self.assertEqual(row["supplemental_training_audit"], expected_audit)
            self.assertEqual(row["supplemental_training_audit"]["added_record_paths"], ["train/ordinary.csv"])
            results.append(row["supplemental_training_audit"])
        self.assertEqual(results[0], results[1])

    def select_mocked(self, recipes, scores, qualified=None):
        protocol = {"baseline": self.feature, "branches": {"candidates": recipes},
                    "partitions": [self.part], "seeds": run.SEEDS, "criteria": run.CRITERIA}
        qualified = set(recipes) if qualified is None else set(qualified)

        def summary(out, name, baseline="baseline"):
            recipe = (run.read(out / "combination_lock.json")["pipeline"] if name == "combined" else recipes[name])
            return {"id": name, "pipeline": recipe, "qualified": name == "combined" or name in qualified,
                    "mean_f1": scores[name], "worst_split": scores[name] - .02,
                    "mean_delta": .02, "baseline": baseline}

        with patch.object(run, "verify", return_value=protocol), \
                patch.object(run, "development_summary", side_effect=summary), \
                patch.object(run, "load_data", return_value=self.data), \
                patch.object(run, "fit_one") as fit, redirect_stdout(io.StringIO()):
            run.select(self.out)
        return fit

    def test_only_individually_qualified_compatible_sequences_enter_combination(self):
        failed = deepcopy(self.cnn)
        failed["model_params"]["hidden_size"] = 8
        recipes = {"cnn": self.cnn, "rocket": self.rocket, "feature": self.feature, "failed": failed}
        fit = self.select_mocked(recipes, {"cnn": .84, "rocket": .83, "feature": .82, "failed": .99, "combined": .87},
                                 qualified={"cnn", "rocket", "feature"})
        combo = run.read(self.out / "combination_lock.json")
        self.assertEqual(combo["members"], ["cnn", "rocket"])
        self.assertEqual(combo["pipeline"]["family"], "sequence_soft_voting")
        self.assertEqual(combo["pipeline"]["members"], [self.cnn, self.rocket])
        self.assertEqual(run.input_representation(fit.call_args.args[2]), "sensor_sequence")
        train, valid = run.indices(self.data, self.part)
        np.testing.assert_array_equal(fit.call_args.args[3], train)
        np.testing.assert_array_equal(fit.call_args.args[4], valid)
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "combined")

    def test_feature_and_sequence_pair_never_combines(self):
        fit = self.select_mocked({"cnn": self.cnn, "feature": self.feature}, {"cnn": .84, "feature": .83})
        fit.assert_not_called()
        self.assertFalse((self.out / "combination_lock.json").exists())
        for family in ("equal_soft_voting", "sequence_soft_voting"):
            with self.subTest(family=family), self.assertRaises(ValueError):
                run.validate_pipeline({"family": family, "members": [self.feature, self.cnn]})

    def test_existing_sequence_ensemble_is_not_nested_or_displaced_by_weaker_combo(self):
        ensemble = run.validate_pipeline({"family": "sequence_soft_voting", "members": [self.cnn, self.rocket]})
        self.select_mocked({"existing": ensemble, "cnn": self.cnn, "rocket": self.rocket},
                           {"existing": .95, "cnn": .84, "rocket": .83, "combined": .87})
        self.assertEqual(run.read(self.out / "combination_lock.json")["members"], ["cnn", "rocket"])
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "existing")


if __name__ == "__main__":
    unittest.main()
