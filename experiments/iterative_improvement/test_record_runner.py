"""Training record identities must follow exactly the frozen training rows."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.iterative_improvement import run
from experiments.iterative_improvement.test_run import fixture_data


class RecordRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.data = fixture_data()
        self.part = {"id": "one", "train": ["train"], "validation": ["validation"], "test": ["test"]}
        self.recipe = run.validate_pipeline({"family": "record_balanced_extra_trees", "model_params": {"n_estimators": 4}})
        run.write(self.out / "protocol.json", {"synthetic": True})

    def test_record_identity_follows_nonmonotonic_training_indices_only(self):
        train, valid = run.indices(self.data, self.part)
        train = train[::-1]
        probability = np.eye(7)[self.data["y"][valid]]
        with patch.object(run, "build_pipeline", return_value={}), patch.object(run, "fit_pipeline") as fit, \
                patch.object(run, "aligned_predict_proba", return_value=probability) as predict:
            run.fit_one(self.out, self.data, self.recipe, train, valid, self.out / "fit", run.SEEDS[0])
        np.testing.assert_array_equal(fit.call_args.args[2], self.data["X"][train])
        np.testing.assert_array_equal(fit.call_args.kwargs["record_ids"], self.data["record_ids"][train])
        np.testing.assert_array_equal(fit.call_args.kwargs["groups"], self.data["groups"][train])
        self.assertNotIn("validation_data", fit.call_args.kwargs)
        np.testing.assert_array_equal(predict.call_args.args[1], self.data["X"][valid])
        with np.load(self.out / "fit/predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["record_ids"], self.data["record_ids"][valid])
            np.testing.assert_array_equal(saved["indices"], valid)

    def test_same_feature_shape_does_not_make_legacy_and_record_recipes_compatible(self):
        other = run.spec("extra_trees", {"n_estimators": 4})
        protocol = {"baseline": other, "branches": {"records": {"first": self.recipe}, "legacy": {"second": other}},
                    "partitions": [self.part], "seeds": run.SEEDS, "criteria": run.CRITERIA}
        def summary(out, name, baseline="baseline"):
            return {"id": name, "pipeline": self.recipe if name == "first" else other,
                    "qualified": True, "mean_f1": .9 if name == "first" else .8,
                    "worst_split": .7, "mean_delta": .02}
        with patch.object(run, "verify", return_value=protocol), \
                patch.object(run, "development_summary", side_effect=summary), patch.object(run, "fit_one") as fit:
            run.select(self.out)
        self.assertFalse((self.out / "combination_lock.json").exists())
        fit.assert_not_called()

    def test_only_qualified_record_candidates_are_combined(self):
        bagged = run.validate_pipeline({"family": "record_bagged_extra_trees", "model_params": {"n_estimators": 4, "bag_count": 2}})
        third = run.validate_pipeline({"family": "record_balanced_extra_trees", "model_params": {"n_estimators": 6}})
        recipes = {"first": self.recipe, "second": bagged, "failed": third}
        protocol = {"baseline": run.spec("extra_trees", {"n_estimators": 4}), "branches": {"records": recipes},
                    "partitions": [self.part], "seeds": run.SEEDS, "criteria": run.CRITERIA}
        def summary(out, name, baseline="baseline"):
            pipeline = run.read(self.out / "combination_lock.json")["pipeline"] if name == "combined" else recipes[name]
            return {"id": name, "pipeline": pipeline, "qualified": name != "failed",
                    "mean_f1": {"first": .9, "second": .85, "failed": .95, "combined": .92}[name],
                    "worst_split": .7, "mean_delta": .02}
        with patch.object(run, "verify", return_value=protocol), patch.object(run, "development_summary", side_effect=summary), \
                patch.object(run, "load_data", return_value=self.data), patch.object(run, "fit_one"):
            run.select(self.out)
        combo = run.read(self.out / "combination_lock.json")
        self.assertEqual(combo["members"], ["first", "second"])
        self.assertEqual(combo["pipeline"]["family"], "record_soft_voting")
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "combined")


if __name__ == "__main__":
    unittest.main()
