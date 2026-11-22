"""Frozen outer-TRAIN routing and selection for structured forest recipes."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.iterative_improvement import run, finalize, report
from experiments.iterative_improvement.record_training import append_record_training


class StructuredRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        rng = np.random.default_rng(1901)
        self.data = {"X": rng.normal(size=(21, 596)).astype(np.float32),
            "seq": np.zeros((21, 36, 16), dtype=np.float32), "y": np.tile(np.arange(7), 3),
            "groups": np.repeat(["train", "validation", "test"], 7),
            "record_ids": np.asarray([f"record-{i}.csv" for i in range(21)])}
        self.part = {"id": "one", "train": ["train"], "validation": ["validation"], "test": ["test"]}
        self.train, self.valid = np.arange(6, -1, -1), np.asarray([13, 7, 12, 8, 11, 9, 10])
        self.recipes = {}
        for name, family, mode in (("identity", "subject_residual_extra_trees", "identity"),
                ("residual", "subject_residual_extra_trees", "remove_subject3"), ("pair", "coupled_extra_trees", "all")):
            self.recipes[name] = run.validate_pipeline({"family": family,
                "feature_engineering": {"mode": mode}, "model_params": {"n_estimators": 4},
                "training": {"seed": run.SEEDS[0]}})
        self.baseline = run.spec("extra_trees", self.recipes["identity"]["model_params"])
        run.write(self.out / "protocol.json", {"synthetic": True})

    def combined(self):
        return run.validate_pipeline({"family": "structured_soft_voting",
            "members": [self.recipes["residual"], self.recipes["pair"]]})

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
        recipes = {"hard": self.recipes["residual"], "rf": self.recipes["pair"],
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
        self.assertEqual(combo["pipeline"]["family"], "structured_soft_voting")
        self.assertEqual(run.read(self.out / "selection_lock.json")["selected"]["id"], "existing")

    def test_identity_control_rejects_nonidentity_and_tree_drift(self):
        from src.pipeline_recipes import validate_identity_control
        validate_identity_control(self.baseline, self.recipes["identity"])
        for recipe in (self.recipes["residual"], self.recipes["pair"]):
            with self.assertRaises(ValueError):
                validate_identity_control(self.baseline, recipe)
        changed = deepcopy(self.recipes["identity"])
        changed["model_params"]["n_estimators"] += 1
        with self.assertRaises(ValueError):
            validate_identity_control(self.baseline, changed)

    def test_finalizer_dispatches_full_structured_verification_before_generic_record_path(self):
        for recipe in [*self.recipes.values(), self.combined()]:
            model = type("Model", (), {"spec_": recipe, "effective_seed_": run.SEEDS[0]})()
            with patch("src.structured_forest_pipeline_v1.verify_fitted_structured_identity") as verify, \
                    patch("src.record_pipeline_v1.verify_fitted_record_identity") as old:
                finalize._verify_fitted_identity(model, recipe, run.SEEDS[0])
            verify.assert_called_once_with(model, recipe, run.SEEDS[0])
            old.assert_not_called()

    def test_report_explains_train_only_residual_and_coupling_without_gain_claim(self):
        residual, pair = self.recipes["residual"], self.recipes["pair"]
        self.assertIn("TRAIN 기록", report.recipe_description(residual))
        self.assertIn("equality-KKT", report.recipe_description(pair))
        self.assertIn("평가 사람", report.hypothesis(residual))
        self.assertIn("성능 개선 후보가 아니다", report.hypothesis(self.recipes["identity"], identity_control=True))
        self.assertIn("제거 차원", " ".join(report.next_experiments([{"pipeline": residual}])))

    def test_coupled_synthetic_confidence_reaches_existing_realtime_and_risk_thresholds(self):
        from src.pairwise_coupling_v1 import PAIR_INDICES, couple_pairwise
        from src.posture_core import CLASS_NAMES
        from src.realtime import LiveConfig, RealtimePosturePredictor
        from src.app_services import RiskAssessment
        known = np.full(7, .05)
        known[1] = .7
        pairs = np.asarray([[known[i] / (known[i] + known[j]) for i, j in PAIR_INDICES]])
        probability = couple_pairwise(pairs)[0]
        class SyntheticPredictor:
            labels = list(CLASS_NAMES)
            def predict_frame(self, frame):
                return probability.copy()
        realtime = RealtimePosturePredictor(SyntheticPredictor(), LiveConfig(
            window_seconds=3., minimum_window_seconds=3., confidence_threshold=.45))
        events = [realtime.push(float(t), np.ones(16)) for t in np.arange(0., 4., 1/12)]
        event = next(value for value in reversed(events) if value is not None)
        self.assertEqual(event.state, "predicting")
        self.assertEqual(event.posture, CLASS_NAMES[1])
        self.assertAlmostEqual(event.confidence, .7)
        risk = RiskAssessment()
        risk.update(0., event.posture, event.confidence)
        risk.update(60., event.posture, event.confidence)
        self.assertTrue(risk.message)
        risk.update(120., event.posture, event.confidence)
        self.assertEqual(risk.level, "위험")


if __name__ == "__main__":
    unittest.main()
