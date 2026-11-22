"""Extend the frozen gain study with pre-restoration regularized Extra Trees."""
from __future__ import annotations
import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from experiments.sensor_gain.run import (
    ROOT, now, read, write_json, digest, verify, protected_snapshot,
    condition_id, launch_phase, lock_selection,
)

PREVIOUS = ROOT / "research/results/2026-09-23_150000_sensor_gain_2s3s"
PARAMS = {"n_estimators": 200, "min_samples_leaf": 8, "max_depth": 8, "max_features": .5}


def fixed_model_seed_partitions(parts, seed):
    """Preserve membership and split provenance; restore the historical fit seed."""
    return [{**part, "split_seed": part.get("split_seed", part["seed"]), "seed": seed} for part in parts]


def prepare(out, previous):
    prior = verify(previous)
    origin = Path(prior["origin"])
    chosen = read(origin / "selection_lock.json")["selected"]["extra_trees/regularized"]
    if chosen["params"] != PARAMS:
        raise ValueError("Historical regularized ET recipe differs")
    old = read(previous / "origin_dataset.json")
    for row in old["records"]:
        if digest(ROOT / "data" / row["source_relative_path"]) != row["sha256"]:
            raise ValueError("Historical original changed")
        if digest(origin / "training_data" / row["staged_file"]) != row["staged_sha256"]:
            raise ValueError("Historical staging changed")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protected_before.json", protected_snapshot())
    for name in ("data_2s.npz", "data_3s.npz", "origin_dataset.json", "origin_dataset_lock.json"):
        (out / name).symlink_to(os.path.relpath(previous / name, out))
    shutil.copytree(previous / "source_snapshot", out / "source_snapshot",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source_hashes = dict(prior["source_hashes"])
    for name in ("et_followup.py", "summarize.py", "test_et_followup.py"):
        source = Path(__file__).with_name(name)
        dest = out / "source_snapshot/experiments/sensor_gain" / name
        shutil.copy2(source, dest)
        source_hashes[str(source.relative_to(ROOT))] = digest(dest)
    # Record that the shared runtime tested in the preceding run is unchanged.
    drift = [r for r, sha in prior["source_hashes"].items() if r.startswith("src/") and digest(ROOT / r) != sha]
    if drift: raise ValueError(f"Shared runtime changed: {drift}")
    stamp = now()
    p = {**prior, "started_at": stamp, "locked_at": stamp, "source_hashes": source_hashes,
        "models": {"extra_trees": PARAMS}, "conditions": [
            {"id": condition_id(s, a, "extra_trees"), "seconds": s, "arm": a, "family": "extra_trees"}
            for s in (2, 3) for a in ("raw", "calibrated")],
        "partitions": fixed_model_seed_partitions(prior["partitions"], prior["seed"]),
        "model_seed_policy": "Historical ET used seed 20260921 for every fit; split membership unchanged, original split-generation seeds retained as split_seed",
        "extension_of": str(previous), "parent_protocol_sha256": digest(previous / "protocol.json"),
        "historical_et_selection_sha256": digest(origin / "selection_lock.json"),
        "historical_et_params": chosen["params"],
        "historical_et_final_metrics_sha256": digest(origin / "final/extra_trees/regularized/metrics.json"),
        "command": [sys.executable, *sys.argv],
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "tuning": "None; historical pre-restoration selected regularized Extra Trees parameters fixed for both durations and gain arms",
        "restored_ordinary_records": 0, "ordinary_restoration_checkpoint_used": False,
        "shared_runtime_drift_from_parent": drift,
        "runner_module": "experiments.sensor_gain.et_followup",
        "report_filename": "2026-09-23_sensor_gain_regularized_et_2s3s_report.md",
        "report_title": "정자세 복원 전 규제 Extra Trees: 센서 보정 전후 2초·3초 재실험"}
    write_json(out / "protocol.json", p)
    write_json(out / "protocol_lock.json", {"sha256": digest(out / "protocol.json"), "locked_at": now()})
    validation = out / "validation"; validation.mkdir()
    (validation / "project_unit_tests.log").symlink_to(os.path.relpath(previous / "validation/project_unit_tests.log", validation))
    write_json(validation / "checks.json", {"checked_at": now(),
        "inherited_project_tests": {"count": 588, "source": str(previous / "validation/project_unit_tests.log"),
            "scope": "preceding run; all shared src hashes verified unchanged"},
        "cohort": "same 32 subjects/692 recordings; no ordinary-correct-posture restoration",
        "unit_test_log": "et_contract_tests.log"})
    verify(out)


def historical_parity(out):
    p = verify(out); origin = Path(p["origin"])
    prior = read(origin / "selection_lock.json")["selected"]["extra_trees/regularized"]
    actual = [read(out / "development/3s_raw_extra_trees" / part["id"] / "metrics.json")["evaluation"]["macro_f1"] for part in p["partitions"]]
    np.testing.assert_allclose(actual, prior["development_f1"], rtol=0, atol=1e-12)
    old_pred = dict(np.load(origin / "final/extra_trees/regularized/predictions.npz", allow_pickle=False))
    new_pred = dict(np.load(out / "final/3s_raw_extra_trees/holdout/predictions.npz", allow_pickle=False))
    for key in ("indices", "truth", "prediction", "groups"):
        np.testing.assert_array_equal(new_pred[key], old_pred[key])
    np.testing.assert_allclose(new_pred["probability"], old_pred["probability"], rtol=0, atol=1e-12)
    write_json(out / "historical_et_reproduction.json", {"checked_at": now(),
        "three_development_f1_exact": True, "holdout_predictions_exact": True,
        "holdout_probabilities_equal_at_1e12": True,
        "origin": str(origin), "old_predictions_sha256": digest(origin / "final/extra_trees/regularized/predictions.npz")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--previous", type=Path, default=PREVIOUS)
    args = parser.parse_args()
    out = (args.output or ROOT / "research/results" / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_sensor_gain_regularized_et_2s3s")).resolve()
    prepare(out, args.previous.resolve())
    print("OUTPUT", out, flush=True)
    launch_phase(out, "development"); lock_selection(out)
    launch_phase(out, "final")
    historical_parity(out)
    if protected_snapshot() != read(out / "protected_before.json"):
        raise ValueError("Original data or active models changed")
    write_json(out / "training_complete.json", {"finished_at": now(), "fits": 16,
        "data_and_existing_models_unchanged": True, "approved": False,
        "historical_3s_uncalibrated_reproduced": True})


if __name__ == "__main__": main()
