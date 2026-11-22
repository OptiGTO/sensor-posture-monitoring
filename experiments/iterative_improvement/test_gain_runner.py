"""Joint sequence/record routing and no-op control gates on synthetic data."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.iterative_improvement import run, record_sequence_training as helper
from experiments.iterative_improvement import supplemental, supplemental_sequence, finalize
from src.window_pipeline import extract_window_features


class GainRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        rows = [(person, label, window) for window in range(2) for label in range(7)
                for person in ("train", "validation", "test")]
        rng = np.random.default_rng(912)
        sequence = rng.uniform(10, 100, (len(rows), 36, 16)).astype(np.float32)
        self.data = {"seq": sequence, "X": extract_window_features(sequence),
            "y": np.asarray([label for _, label, _ in rows]),
            "groups": np.asarray([person for person, _, _ in rows]),
            "record_ids": np.asarray([f"{person}/class-{label}.csv" for person, label, _ in rows])}
        extra_groups = np.asarray(["test", "train", "validation", "train", "test", "validation"])
        extra_seq = rng.uniform(10, 100, (len(extra_groups), 36, 16)).astype(np.float32)
        self.extra = {"seq": extra_seq, "X": extract_window_features(extra_seq),
            "y": np.zeros(len(extra_groups), dtype=int), "groups": extra_groups,
            "record_ids": np.asarray([f"{person}/ordinary.csv" for person in extra_groups])}
        self.part = {"id": "one", "train": ["train"], "validation": ["validation"], "test": ["test"]}
        self.train, self.valid = run.indices(self.data, self.part)
        self.train = self.train[::-1]
        self.recipe = run.validate_pipeline({"family": "record_gain_extra_trees", "model_params": {"n_estimators": 4},
            "training": {"augmentation": {"mode": "common", "strength": .02}}})
        run.write(self.out / "protocol.json", {"synthetic": True})

    def test_restored_sequences_records_and_legacy_audit_have_identical_train_rows(self):
        before = deepcopy((self.data, self.extra))
        X, y, groups, audit = supplemental.append_training(self.data, self.extra, self.train)
        seq, actual_y, actual_groups, records, actual_audit = helper.append_record_sequence_training(self.data, self.extra, self.train)
        np.testing.assert_array_equal(extract_window_features(seq), X)
        np.testing.assert_array_equal(actual_y, y)
        np.testing.assert_array_equal(actual_groups, groups)
        np.testing.assert_array_equal(records, np.concatenate((self.data["record_ids"][self.train], self.extra["record_ids"][[1, 3]])))
        self.assertEqual(actual_audit, audit)
        self.assertEqual(set(groups), {"train"})
        for actual, expected in zip((self.data, self.extra), before):
            for key in expected:
                self.assertEqual(actual[key].tobytes(), expected[key].tobytes())

    def test_restore_rejects_overlap_and_impure_record_without_fitting(self):
        changed = deepcopy(self.extra)
        changed["record_ids"] = changed["record_ids"].astype("U100")
        changed["record_ids"][1] = "validation/class-0.csv"
        with self.assertRaises(ValueError):
            helper.append_record_sequence_training(self.data, changed, self.train)
        changed = deepcopy(self.data)
        changed["y"][self.train[0]] = 0
        with self.assertRaises(ValueError):
            helper.append_record_sequence_training(changed, self.extra, self.train)

    def test_restore_rejects_delegate_sequence_or_label_order_drift(self):
        result = supplemental_sequence.append_sequence_training(self.data, self.extra, self.train)
        for changed in ((result[0][:-1], *result[1:]),
                        (result[0], result[1][::-1], result[2], result[3]),
                        (*result[:3], {**result[3], "added_windows": 999})):
            with self.subTest(), patch.object(helper, "append_sequence_training", return_value=changed), self.assertRaises(ValueError):
                helper.append_record_sequence_training(self.data, self.extra, self.train)

    def test_runner_uses_sequence_and_actual_train_records_without_eval_fit(self):
        for restored in (False, True):
            with self.subTest(restored=restored):
                recipe = deepcopy(self.recipe)
                if restored:
                    recipe["training"]["data_policy"] = supplemental.POLICY
                    np.savez_compressed(self.out / "supplemental_training.npz", **self.extra)
                    seq, y, groups, records, _ = helper.append_record_sequence_training(self.data, self.extra, self.train)
                else:
                    seq, y, groups, records = (self.data[key][self.train] for key in ("seq", "y", "groups", "record_ids"))
                probability = np.eye(7)[self.data["y"][self.valid]]
                with patch.object(run, "build_pipeline", return_value={}), patch.object(run, "fit_pipeline") as fit, \
                        patch.object(run, "aligned_predict_proba", return_value=probability) as predict:
                    run.fit_one(self.out, self.data, recipe, self.train, self.valid, self.out / str(restored), run.SEEDS[0])
                np.testing.assert_array_equal(fit.call_args.args[2], seq)
                np.testing.assert_array_equal(fit.call_args.args[3], y)
                np.testing.assert_array_equal(fit.call_args.kwargs["groups"], groups)
                np.testing.assert_array_equal(fit.call_args.kwargs["record_ids"], records)
                self.assertNotIn("validation_data", fit.call_args.kwargs)
                np.testing.assert_array_equal(predict.call_args.args[1], self.data["seq"][self.valid])

    def test_gain_compatibility_and_qualified_members_only(self):
        second = deepcopy(self.recipe)
        second["training"]["augmentation"]["mode"] = "per_sensor"
        failed = deepcopy(self.recipe)
        failed["training"]["augmentation"]["strength"] = .05
        existing = run.validate_pipeline({"family": "gain_soft_voting", "members": [self.recipe, second]})
        cnn = run.validate_pipeline({"family": "cnn1d", "model_params": {"epochs": 1}})
        recipes = {"common": self.recipe, "sensor": second, "failed": failed, "existing": existing, "cnn": cnn}
        means = {"common": .85, "sensor": .84, "failed": .99, "existing": .90, "cnn": .83, "combined": .88}
        protocol = {"baseline": run.spec("extra_trees", {"n_estimators": 4}), "branches": {"gain": recipes},
            "partitions": [self.part], "seeds": run.SEEDS, "criteria": run.CRITERIA}
        def summary(out, name, baseline="baseline"):
            recipe = run.read(self.out / "combination_lock.json")["pipeline"] if name == "combined" else recipes[name]
            return {"id": name, "pipeline": recipe, "qualified": name != "failed", "mean_f1": means[name],
                "worst_split": .7, "mean_delta": .02}
        with patch.object(run, "verify", return_value=protocol), patch.object(run, "development_summary", side_effect=summary), \
                patch.object(run, "load_data", return_value=self.data), patch.object(run, "fit_one"):
            run.select(self.out)
        combo = run.read(self.out / "combination_lock.json")
        self.assertEqual(combo["members"], ["common", "sensor"])
        self.assertEqual(combo["pipeline"]["family"], "gain_soft_voting")
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "existing")

    def control_fixture(self, *, changed_probability=False):
        control = deepcopy(self.recipe)
        control["training"]["augmentation"].update(mode="identity", strength=0.)
        protocol = {"baseline": run.spec("extra_trees", control["model_params"]), "branches": {"control": {"identity": control}},
            "identity_controls": ["identity"], "partitions": [self.part], "seeds": run.SEEDS}
        truth = np.arange(7)
        for name in ("baseline", "identity"):
            directory = self.out / "development" / name / "one"
            directory.mkdir(parents=True)
            probability = .3 / 7 + .7 * np.eye(7)
            if name == "identity" and changed_probability:
                probability[0, 1] += .01
                probability[0, 2] -= .01
            np.savez_compressed(directory / "predictions.npz", truth=truth, prediction=truth, probability=probability)
        return protocol

    def test_identity_gate_records_equal_arrays_and_is_idempotent(self):
        protocol = self.control_fixture()
        with patch.object(run, "verify_fit", return_value={"metrics": {"macro_f1": 1.}}):
            run.verify_identity_controls(self.out, protocol)
            original = (self.out / "audits/identity_controls.json").read_bytes()
            run.verify_identity_controls(self.out, protocol)
        self.assertEqual(original, (self.out / "audits/identity_controls.json").read_bytes())

    def test_identity_gate_rejects_probability_change_even_when_labels_and_metrics_match(self):
        protocol = self.control_fixture(changed_probability=True)
        with patch.object(run, "verify_fit", return_value={"metrics": {"macro_f1": 1.}}), self.assertRaisesRegex(ValueError, "Identity control"):
            run.verify_identity_controls(self.out, protocol)
        self.assertFalse((self.out / "audits/identity_controls.json").exists())

    def test_identity_gate_rejects_nonidentity_config_even_with_identical_arrays(self):
        protocol = self.control_fixture()
        for updates in ({"mode": "common", "strength": .02}, {"mode": "per_sensor", "strength": 0.}):
            changed = deepcopy(protocol)
            changed["branches"]["control"]["identity"]["training"]["augmentation"].update(updates)
            with self.subTest(updates=updates), patch.object(run, "verify_fit") as fit, self.assertRaises(ValueError):
                run.verify_identity_controls(self.out, changed)
            fit.assert_not_called()
        self.assertFalse((self.out / "audits/identity_controls.json").exists())

    def test_finalizer_uses_gain_specific_fitted_identity(self):
        model = type("Model", (), {"spec_": self.recipe, "effective_seed_": run.SEEDS[0]})()
        with patch("src.augmentation_pipeline_v1.verify_fitted_augmentation_identity") as verify:
            finalize._verify_fitted_identity(model, self.recipe, run.SEEDS[0])
        verify.assert_called_once_with(model, self.recipe, run.SEEDS[0])


if __name__ == "__main__":
    unittest.main()
