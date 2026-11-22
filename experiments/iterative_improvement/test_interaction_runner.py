"""Frozen row routing, no-op proof and qualified interaction combinations."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.iterative_improvement import run, finalize, supplemental, supplemental_sequence
from src.window_pipeline import extract_window_features


class InteractionRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        rng = np.random.default_rng(1401)
        sequence = rng.uniform(10, 90, (21, 36, 16)).astype(np.float32)
        self.data = {"seq": sequence, "X": extract_window_features(sequence), "y": np.tile(np.arange(7), 3),
            "groups": np.repeat(["train", "validation", "test"], 7),
            "record_ids": np.asarray([f"record-{i}.csv" for i in range(21)])}
        self.part = {"id": "one", "train": ["train"], "validation": ["validation"], "test": ["test"]}
        self.train = np.arange(6, -1, -1)
        self.valid = np.asarray([13, 7, 12, 8, 11, 9, 10])
        self.recipes = {mode: run.validate_pipeline({"family": "interaction_extra_trees",
            "feature_engineering": {"mode": mode}, "model_params": {"n_estimators": 4}})
            for mode in ("identity", "hadamard512", "temporal_corr120")}
        self.baseline = run.spec("extra_trees", self.recipes["identity"]["model_params"])
        run.write(self.out / "protocol.json", {"synthetic": True})

    def test_all_modes_and_ensemble_take_exact_sequence_rows_without_record_ids(self):
        recipes = [*self.recipes.values(), run.validate_pipeline({"family": "interaction_soft_voting",
            "members": [self.recipes["hadamard512"], self.recipes["temporal_corr120"]]})]
        for i, recipe in enumerate(recipes):
            with self.subTest(i=i), patch.object(run, "build_pipeline", return_value={}), \
                    patch.object(run, "fit_pipeline") as fit, patch.object(run, "aligned_predict_proba",
                    return_value=np.eye(7)[self.data["y"][self.valid]]) as predict:
                row = run.fit_one(self.out, self.data, recipe, self.train, self.valid, self.out / str(i), run.SEEDS[1])
            np.testing.assert_array_equal(fit.call_args.args[2], self.data["seq"][self.train])
            np.testing.assert_array_equal(fit.call_args.args[3], self.data["y"][self.train])
            np.testing.assert_array_equal(fit.call_args.kwargs["groups"], self.data["groups"][self.train])
            self.assertNotIn("record_ids", fit.call_args.kwargs)
            self.assertNotIn("validation_data", fit.call_args.kwargs)
            np.testing.assert_array_equal(predict.call_args.args[1], self.data["seq"][self.valid])
            self.assertEqual(row["input_representation"], "sensor_sequence")
            with np.load(self.out / str(i) / "predictions.npz") as saved:
                for key in ("groups", "record_ids"):
                    np.testing.assert_array_equal(saved[key], self.data[key][self.valid])

    def test_ordinary_restoration_uses_only_training_sequences_and_same_feature_audit(self):
        rng = np.random.default_rng(1402)
        sequence = rng.uniform(10, 90, (4, 36, 16)).astype(np.float32)
        extra = {"seq": sequence, "X": extract_window_features(sequence), "y": np.zeros(4, dtype=int),
            "groups": np.asarray(["validation", "train", "test", "train"]),
            "record_ids": np.asarray(["extra-validation", "extra-train1", "extra-test", "extra-train2"])}
        np.savez_compressed(self.out / "supplemental_training.npz", **extra)
        expected, labels, groups, audit = supplemental_sequence.append_sequence_training(self.data, extra, self.train)
        features, old_labels, old_groups, old_audit = supplemental.append_training(self.data, extra, self.train)
        np.testing.assert_array_equal(extract_window_features(expected), features)
        np.testing.assert_array_equal(labels, old_labels)
        np.testing.assert_array_equal(groups, old_groups)
        self.assertEqual(audit, old_audit)
        for mode, original in self.recipes.items():
            recipe = deepcopy(original)
            recipe["training"]["data_policy"] = supplemental.POLICY
            with patch.object(run, "build_pipeline", return_value={}), patch.object(run, "fit_pipeline") as fit, \
                    patch.object(run, "aligned_predict_proba", return_value=np.eye(7)[self.data["y"][self.valid]]):
                row = run.fit_one(self.out, self.data, recipe, self.train, self.valid, self.out / mode, run.SEEDS[0])
            np.testing.assert_array_equal(fit.call_args.args[2], expected)
            np.testing.assert_array_equal(fit.call_args.args[3], labels)
            np.testing.assert_array_equal(fit.call_args.kwargs["groups"], groups)
            self.assertNotIn("record_ids", fit.call_args.kwargs)
            self.assertEqual(row["supplemental_training_audit"], audit)

    def test_only_qualified_compatible_leaves_combine_and_never_nest(self):
        left, right = self.recipes["hadamard512"], self.recipes["temporal_corr120"]
        existing = run.validate_pipeline({"family": "interaction_soft_voting", "members": [left, right]})
        cnn = run.validate_pipeline({"family": "cnn1d", "model_params": {"epochs": 1}})
        recipes = {"left": left, "right": right, "control": self.recipes["identity"], "existing": existing, "cnn": cnn}
        means = {"left": .85, "right": .84, "control": .99, "existing": .95, "cnn": .83, "combined": .88}
        protocol = {"baseline": self.baseline, "branches": {"features": recipes}, "partitions": [self.part],
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
        self.assertEqual(combo["pipeline"]["family"], "interaction_soft_voting")
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

    def test_exact_control_is_audited_but_same_labels_with_changed_probabilities_fail(self):
        protocol = self.identity_fixture()
        with patch.object(run, "verify_fit", return_value={"metrics": {"macro_f1": 1.}}):
            run.verify_identity_controls(self.out, protocol)
        audit = run.read(self.out / "audits/identity_controls.json")
        self.assertTrue(audit)
        file = self.out / "development/identity/one/predictions.npz"
        values = dict(np.load(file))
        values["probability"][0, 1] += .01
        values["probability"][0, 2] -= .01
        np.savez_compressed(file, **values)
        with patch.object(run, "verify_fit", return_value={"metrics": {"macro_f1": 1.}}), self.assertRaises(ValueError):
            run.verify_identity_controls(self.out, protocol)

    def test_nonidentity_features_fail_control_semantics_before_reading_fits(self):
        for mode in ("hadamard512", "temporal_corr120"):
            protocol = {"baseline": self.baseline, "branches": {"control": {"identity": self.recipes[mode]}},
                "identity_controls": ["identity"], "partitions": [self.part], "seeds": run.SEEDS}
            with self.subTest(mode=mode), patch.object(run, "verify_fit") as fit, self.assertRaises(ValueError):
                run.verify_identity_controls(self.out, protocol)
            fit.assert_not_called()

    def test_finalizer_routes_all_interaction_families_to_fitted_state_verifier(self):
        recipes = [*self.recipes.values(), run.validate_pipeline({"family": "interaction_soft_voting",
            "members": [self.recipes["hadamard512"], self.recipes["temporal_corr120"]]})]
        for recipe in recipes:
            model = type("Model", (), {"spec_": recipe, "effective_seed_": run.SEEDS[0]})()
            with patch("src.interaction_pipeline_v1.verify_fitted_interaction_identity") as verify:
                finalize._verify_fitted_identity(model, recipe, run.SEEDS[0])
            verify.assert_called_once_with(model, recipe, run.SEEDS[0])


if __name__ == "__main__":
    unittest.main()
