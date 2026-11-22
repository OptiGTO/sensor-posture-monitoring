"""LOSO as the default model/window selection metric."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from src.app_services import TrainingJob, TrainingWizardState
from src.comparison_charts import default_split_label, model_value
from src.experiment_workflow import apply_configuration, build_configuration
from src.model_comparison import _dataset, loso_evaluation, train_comparison
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SMALL_GRID = {"rbf_svm": [{"C": 1.0, "gamma": "scale"}],
              "extra_trees": [{"n_estimators": 12, "min_samples_leaf": 1, "max_features": "sqrt"}]}


def make_data(root: Path, subjects: int = 5, rows: int = 37) -> None:
    rng = np.random.default_rng(37)
    for subject in range(subjects):
        directory = root / f"subject{subject}"
        directory.mkdir(parents=True)
        for i, label in enumerate(CLASS_NAMES):
            values = rng.uniform(10, 15, (rows, len(SENSOR_COLUMNS))) + i * 15 + subject * 3
            frame = pd.DataFrame(values, columns=SENSOR_COLUMNS)
            frame.insert(0, "label", label)
            frame.insert(0, "elapsed_s", np.arange(rows) / 12)
            frame.to_csv(directory / f"posture{i}.csv", index=False)


class LosoSelectionTests(unittest.TestCase):
    def test_loso_never_reads_final_subjects_and_predicts_each_development_window_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_data(root / "data")
            data = _dataset(root / "data", 3, 41)
            subjects = data["meta"]["subject"].to_numpy(dtype=str)
            development = data["masks"]["train"] | data["masks"]["validation"]
            final_rows = set(np.flatnonzero(data["masks"]["final_evaluation"]).tolist())
            seen_fit, seen_predict = [], []
            import src.model_comparison as mc
            real_fit, real_predict = mc.fit_method_model, mc.aligned_predict_proba
            def fit(model, family, method, X, y, **kwargs):
                seen_fit.append(set(map(tuple, X.tolist())))
                self.assertIsNone(kwargs["validation_data"])
                return real_fit(model, family, method, X, y, **kwargs)
            def predict(model, X, count):
                seen_predict.append(len(X))
                return real_predict(model, X, count)
            with mock.patch.object(mc, "fit_method_model", side_effect=fit), \
                 mock.patch.object(mc, "aligned_predict_proba", side_effect=predict):
                result = loso_evaluation("rbf_svm", {"C": 1.0, "gamma": "scale"}, 41, "standard", data["X"], data["y"],
                                         subjects, data["meta"]["record_id"].to_numpy(dtype=str), development)
            final_vectors = {tuple(data["X"][i].tolist()) for i in final_rows}
            self.assertTrue(all(not (fold & final_vectors) for fold in seen_fit))
            self.assertEqual(result["subjects"], len(set(subjects[development])))
            self.assertEqual(result["windows"], int(development.sum()))
            self.assertEqual(sum(seen_predict), int(development.sum()))
            self.assertEqual(sum(np.asarray(result["confusion_matrix"]).ravel()), int(development.sum()))

    def test_default_selection_uses_loso_and_locks_before_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_data(root / "data")
            with mock.patch("src.model_comparison.MODEL_CANDIDATES", SMALL_GRID):
                report = train_comparison(root / "data", root / "run", (3,), 41, ("rbf_svm", "extra_trees"),
                                          progress=lambda _: None)
            self.assertEqual(report["selection_metric"], "loso")
            best = max(report["models"], key=lambda r: (r["loso"]["macro_f1"], r["loso"]["worst_class_f1"], -r["inference_ms_median"]))
            self.assertEqual(report["selected"]["family"], best["family"])
            self.assertEqual(report["selected"]["loso_macro_f1"], best["loso"]["macro_f1"])
            evidence = json.loads((root / "run/artifacts" / f"window_3s_{best['family']}_evaluation.json").read_text())
            self.assertEqual(evidence["selection_metric"], "loso")
            self.assertFalse(evidence["final_evaluation_used_for_selection"])
            self.assertIn("loso_metrics", evidence)
            self.assertIn("LOSO Macro F1", (root / "run/artifacts/model_comparison.md").read_text())
            self.assertEqual(default_split_label(report), "LOSO (기본)")
            self.assertAlmostEqual(model_value(best, "loso", "Macro F1"), best["loso"]["macro_f1"] * 100)

    def test_validation_metric_keeps_previous_behaviour(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_data(root / "data")
            with mock.patch("src.model_comparison.MODEL_CANDIDATES", SMALL_GRID):
                report = train_comparison(root / "data", root / "run", (3,), 41, ("rbf_svm",),
                                          progress=lambda _: None, selection_metric="validation")
            self.assertNotIn("loso", report["models"][0])
            self.assertEqual(default_split_label(report), "검증")
            with self.assertRaises(ValueError):
                train_comparison(root / "data", root / "run2", (3,), 41, ("rbf_svm",), selection_metric="holdout")

    def test_configuration_defaults_to_loso_and_old_clones_use_validation(self):
        state = TrainingWizardState()
        config = build_configuration(state)
        self.assertEqual(config["selection_metric"], "loso")
        state.selection_metric = "validation"
        self.assertEqual(build_configuration(state)["selection_metric"], "validation")
        old = {k: v for k, v in config.items() if k not in {"selection_metric", "selection_metric_label"}}
        clone = TrainingWizardState()
        apply_configuration(clone, old)
        self.assertEqual(clone.selection_metric, "validation")
        state.selection_metric = "nope"
        with self.assertRaises(ValueError):
            build_configuration(state)

    def test_training_job_passes_selection_metric_to_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = TrainingJob(ROOT, (), 41, state_root=Path(temporary), selection_metric="loso")
            captured = {}
            record = SimpleNamespace(csv_path=Path(temporary) / "x.csv")
            job.records = (record,)
            staged = Path(temporary) / "staged"
            staged.mkdir()
            (staged / "staging_summary.json").write_text(json.dumps({"skipped_duplicates": [], "staged_count": 0}))
            class FakeProcess:
                stdout = None
                def __init__(self, command, **_kwargs):
                    captured["command"] = command
            with mock.patch("src.app_services.DatasetService.stage_for_training", return_value=staged), \
                 mock.patch("src.app_services.make_subject_pairs"), \
                 mock.patch("src.app_services.subprocess.Popen", FakeProcess), \
                 mock.patch("src.app_services.threading.Thread"):
                job.start()
            command = captured["command"]
            self.assertEqual(command[command.index("--selection-metric") + 1], "loso")


if __name__ == "__main__":
    unittest.main()
