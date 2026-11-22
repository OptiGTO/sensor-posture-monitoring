#!/usr/bin/env python3
"""Exercise every published recipe through the actual isolated training worker.

Synthetic records verify integration and serialization, never accuracy claims.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import queue
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
from src.app_services import DatasetService, TrainingJob, TrainingWizardState
from src.experiment_workflow import build_configuration
from src.model_catalog import input_contract, model_input
from src.model_comparison import _dataset
from src.model_runtime import ModelRegistry
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, feature_names
from src.training_methods import available_training_methods, method_candidates, subject_balanced_weights


def check_serialized_recipe(job, comparison, method):
    """Check actual fitted objects rather than trusting report-level labels."""
    families = set(method["families"])
    rows = comparison["models"]
    assert len(rows) == len(families) and {row["family"] for row in rows} == families
    artifacts = job.run_dir / "artifacts"
    dataset = _dataset(job.run_dir / "training_data", 3, job.seed)
    train = dataset["masks"]["train"]
    checks = []
    for row in rows:
        family, recipe = row["family"], method["id"]
        config = json.loads((artifacts / row["config_file"]).read_text())
        bundle = joblib.load(artifacts / row["artifact_file"])
        assert row["training_method"] == recipe
        for payload in (config, bundle):
            assert payload["training_method"] == recipe
            assert payload["training_method_evidence"] == method["evidence"]
            assert payload["model_family"] == family and payload["window_seconds"] == 3
            assert payload["dataset_fingerprint"] == dataset["fingerprint"]
            assert payload["training_details"]["training_method"]["id"] == recipe
            assert payload["model_params"] == row["params"]
        assert row["params"] in method_candidates(family, recipe)
        model = bundle["model"]
        assert model.training_details_["training_method"]["id"] == recipe
        estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
        for name, value in row["params"].items():
            assert estimator.get_params()[name] == value, f"{family}/{recipe}: fitted parameter mismatch: {name}"
        if input_contract(family) == "feature_vector":
            assert config["feature_names"] == bundle["feature_names"] == feature_names()
            assert model.n_features_in_ == len(feature_names()) == 596
        if recipe == "relative":
            indices = [i for i, name in enumerate(feature_names()) if name.startswith("rel_")]
            assert len(indices) == 298
            np.testing.assert_array_equal(model.named_steps["relative"].indices_, indices)
            np.testing.assert_allclose(model.named_steps["scale"].mean_, dataset["X"][train][:, indices].mean(axis=0))
        elif recipe == "selected":
            selector = model.named_steps["selection"]
            assert selector.k == 96 and selector.get_support().sum() == 96
            expected_names = [feature_names()[i] for i in selector.get_support(indices=True)]
            assert expected_names == config["training_details"]["selected_feature_names"]
            np.testing.assert_allclose(model.named_steps["scale"].mean_, selector.transform(dataset["X"][train]).mean(axis=0))
        elif recipe == "subject_balanced":
            groups = dataset["meta"].loc[train, "subject"].to_numpy()
            weights = subject_balanced_weights(dataset["y"][train], groups)
            np.testing.assert_allclose(model.named_steps["scale"].mean_, np.average(dataset["X"][train], axis=0, weights=weights))
            assert model.training_details_["training_subject_count"] == len(set(groups))
            if family == "catboost":
                assert estimator.get_params().get("auto_class_weights") is None
            elif family != "xgboost":
                assert estimator.class_weight is None
        elif recipe == "cnn_augmentation":
            assert family == "cnn1d" and model.architecture == "cnn1d"
            assert model.gain_std > 0 and model.noise_fraction > 0
            assert model.training_details_["augmentation"]["training_only"] is True
            np.testing.assert_allclose(model.augmentation_noise_scale_,
                dataset["sequences"][train].std(axis=(0, 1), dtype=np.float64), rtol=1e-6)
        elif recipe == "regularized":
            assert list(model.named_steps) == ["scale", "model"]
        else:
            raise AssertionError(f"No serialized recipe check for {recipe}")
        probe = model_input(family, dataset["sequences"], dataset["X"])[dataset["masks"]["final_evaluation"]]
        first, second = model.predict_proba(probe), model.predict_proba(probe)
        np.testing.assert_array_equal(first, second)
        assert np.isfinite(first).all() and first.shape == (len(probe), len(CLASS_NAMES))
        np.testing.assert_allclose(first.sum(axis=1), 1, atol=1e-6)
        checks.append({"family": family, "method": recipe, "params": row["params"],
            "config_and_bundle_evidence_match": True, "fitted_recipe_verified": True,
            "runtime_input": input_contract(family), "deterministic_serialized_inference": True})
    return checks


def validate(output):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(41)
    state = output / "state"
    for subject in range(6):
        directory = state / "data" / f"synthetic_{subject}"
        directory.mkdir(parents=True)
        for label_id, label in enumerate(CLASS_NAMES):
            # Unequal subject window counts make the weighted-scaler assertion
            # meaningful; every subject still supplies all seven labels.
            values = rng.uniform(.1, .15, (37 + subject * 12, 16))
            values[:, label_id * 2:(label_id + 1) * 2] += 1 + subject * .005
            frame = pd.DataFrame(values, columns=SENSOR_COLUMNS)
            frame.insert(0, "label", label)
            frame.insert(0, "elapsed_s", np.arange(len(values)) / 12)
            frame.to_csv(directory / f"{label_id}.csv", index=False)
    records = DatasetService(state).records()
    methods = [m for m in available_training_methods() if m["id"] != "standard"]
    if not methods:
        raise ValueError("No evidence-qualified recipe is published")
    checks = []
    for method in methods:
        families = tuple(method["families"])
        config = build_configuration(TrainingWizardState(models=families, windows=(3,), training_method=method["id"]))
        job = TrainingJob(ROOT, records, 41, windows=(3,), models=families, state_root=state,
                          training_method=method["id"], experiment_config=config)
        job.start()
        deadline = time.monotonic() + 600
        logs, code = [], None
        while time.monotonic() < deadline:
            try:
                kind, value = job.events.get(timeout=.25)
            except queue.Empty:
                continue
            logs.append(f"{kind}: {value}")
            if kind == "finished":
                code = value
                break
        if code is None:
            job.cancel()
            raise TimeoutError("Synthetic worker did not complete")
        if job._reader:
            job._reader.join(5)
        (output / f"{method['id']}.log").write_text("\n".join(logs) + "\n")
        if (code != "0" or len(job.validation_results) != len(families)
                or not all(row["passed"] and row["runtime"] == row["deployment"] == "passed"
                           for row in job.validation_results)):
            raise AssertionError(f"Worker/validation failed: {method['id']}: {job.validation_results}")
        comparison = json.loads(job.comparison_report_path.read_text())
        assert comparison["training_method"] == method["id"]
        assert comparison["selection_locked"] and not comparison["final_evaluation_used_for_selection"]
        snapshot = json.loads((job.run_dir / "experiment.json").read_text())
        assert snapshot["training_method"] == method["id"] and snapshot["training_method_evidence"]["improved"]
        serialized_checks = check_serialized_recipe(job, comparison, method)
        registry = ModelRegistry(ROOT, state)
        paths = {path.resolve() for path in job.candidate_manifest_paths}
        candidates = [item for item in registry.management_models() if item.manifest_path.resolve() in paths]
        assert len(candidates) == len(families)
        assert {item.model_family for item in candidates} == set(families)
        assert all(item.approval_status == "pending" and item.validation_status == "passed"
                   and item.deployment_validation_status == "passed" for item in candidates)
        assert not registry.active_path.exists(), "Validation must not activate a model"
        checks.append({"method": method["id"], "families": list(families), "worker_exit": code,
            "run_dir": str(job.run_dir), "validation_results": job.validation_results,
            "candidate_count": len(candidates), "approval": "pending", "active_changed": False,
            "selection_locked": True, "configuration_and_model_method_persisted": True,
            "serialized_model_checks": serialized_checks})
        print("WORKER PASSED", method["id"], list(families), flush=True)
    result = {"passed": True, "scope": "synthetic worker/runtime/deployment integration only, not accuracy", "checks": checks}
    (output / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    validate(parser.parse_args().output.resolve())
