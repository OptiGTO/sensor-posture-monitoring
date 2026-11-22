"""Finger-press sensor calibration: 32-subject LOSO retraining study (preregistered).

Question: does equalizing per-sensor output with the user's finger-press
comparison improve subject-independent posture classification?

Cohort: exactly the records/partitions of the active submission model run
(724 records, 32 subjects, 3 s windows), read from that run's staged copy.
Arms: none / gain (x*g, 2026-09-23 transform) / offset_gain (empty-seat level
kept, response above it equalized). Families with fixed parameters: RBF-SVM
(active model's selected params), regularized Extra Trees (research baseline),
XGBoost (2026-09-23 recipe). Seeds: 3. Every subject is predicted once by a
model fitted on the other 31. The decision rule is written to protocol.json
before the first fit and is not changed afterwards.

The app's own 20/6/6 subject split is also replayed for each arm (including the
app's RBF-SVM parameter grid). This is descriptive and must reproduce the
active model exactly for the uncalibrated arm.

Nothing under data/ or research/models/ is written; both are hashed before and
after. No model is registered, approved or activated.

Usage (new output folder only; ~1 h on 8 workers):
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.sensor_calibration.run \
      --output research/results/<YYYY-MM-DD_HHMMSS>_finger_press_calibration_loso \
      [--calibration hardware/calibration/finger_press_calibration.json]
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sensor_calibration.analysis import (  # noqa: E402
    decide, macro_f1, per_class_f1, present_class_macro_f1, seed_mean_bootstrap)
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS  # noqa: E402
from src.sensor_calibration import (  # noqa: E402
    CALIBRATION_MODES, DEFAULT_MEASUREMENT_PATH, build_calibration, load_measurement)

COHORT_RUN = ROOT / "research/models/models/20260929_221617_db0a4f57"
ACTIVE_POINTER = ROOT / "research/models/active_model.json"
WINDOW_SECONDS = 3
SPLIT_SEED = 20260826  # the app's training seed; defines the stored 20/6/6 split
SEEDS = [20260826, 20260827, 20260828]
FAMILIES = {
    "rbf_svm": {"C": 1.0, "gamma": "scale"},
    "extra_trees": {"n_estimators": 200, "min_samples_leaf": 8, "max_depth": 8, "max_features": 0.5},
    "xgboost": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 3,
                "reg_lambda": 5.0, "subsample": 0.8, "colsample_bytree": 0.8},
}
CALIBRATED = [mode for mode in CALIBRATION_MODES if mode != "none"]
CRITERIA = {"minimum_mean_delta": 0.01, "maximum_per_class_drop": 0.05,
            "bootstrap_replicates": 10000, "bootstrap_seed": 20260929,
            "bonferroni_comparisons": len(CALIBRATED) * len(FAMILIES)}
SOURCES = ["experiments/sensor_calibration/run.py", "experiments/sensor_calibration/analysis.py",
           "experiments/sensor_calibration/test_analysis.py", "src/sensor_calibration.py",
           "src/model_comparison.py", "src/posture_core.py", "src/window_pipeline.py",
           "src/training_methods.py", "tests/test_sensor_calibration.py"]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def protected_state() -> dict:
    """Listing, SHA-256, size and mtime of data/ and research/models/ (read only)."""
    state = {}
    for base in ("data", "research/models"):
        for dirpath, _, files in os.walk(ROOT / base, followlinks=False):
            for name in files:
                path = Path(dirpath) / name
                if name == ".DS_Store" or not path.exists():
                    continue
                stat = path.stat()
                state[str(path.relative_to(ROOT))] = [sha(path), stat.st_size, stat.st_mtime_ns]
    return state


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout


# ---------------------------------------------------------------- cohort

def load_cohort(cohort_run: Path) -> tuple[dict, dict, dict]:
    """Rebuild the active run's windows through the app's own dataset function."""
    from src.model_comparison import _dataset
    evaluation = read(cohort_run / "artifacts/window_3s_rbf_svm_evaluation.json")
    partitions = {name: part["subjects"] for name, part in evaluation["partitions"].items()}
    staged_root = cohort_run / "training_data"
    staged = read(staged_root / "staged_sources.json")
    audit = {"records": len(staged), "original_matches": 0, "staged_matches": 0}
    for row in staged:
        original = Path(row["source_file"])
        if sha(staged_root / row["staged_file"]) != row["sha256"]:
            raise ValueError(f"Staged copy changed: {row['staged_file']}")
        audit["staged_matches"] += 1
        if original.is_file() and sha(original) == row["sha256"]:
            audit["original_matches"] += 1
    data = _dataset(staged_root, WINDOW_SECONDS, SPLIT_SEED, partitions)
    if data["fingerprint"] != evaluation["dataset_fingerprint"]:
        raise ValueError("Cohort fingerprint differs from the active run")
    return data, evaluation, audit


def build_caches(data: dict, measurement: dict, digest: str, cache_dir: Path) -> dict:
    from src.window_pipeline import extract_window_features
    cache_dir.mkdir(parents=True)
    meta = data["meta"]
    info = {}
    for mode in CALIBRATION_MODES:
        calibration = build_calibration(measurement, mode, digest)
        sequences = calibration.apply(data["sequences"])
        X = extract_window_features(sequences)
        if mode == "none" and not np.array_equal(X, data["X"]):
            raise ValueError("Uncalibrated features differ from the app dataset features")
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite features for {mode}")
        path = cache_dir / f"{mode}.npz"
        np.savez_compressed(path, X=X, y=data["y"], groups=meta["subject"].to_numpy(dtype=str),
                            record_ids=meta["record_id"].to_numpy(dtype=str),
                            **{f"mask_{k}": v for k, v in data["masks"].items()})
        info[mode] = {"path": str(path.relative_to(cache_dir.parent)), "sha256": sha(path),
                      "calibration": calibration.to_dict(),
                      "calibrated_sequence_range": [float(sequences.min()), float(sequences.max())]}
    return info


_CACHE: dict[str, dict] = {}


def _load(path: str) -> dict:
    if path not in _CACHE:
        with np.load(path, allow_pickle=False) as handle:
            _CACHE[path] = {key: handle[key] for key in handle.files}
    return _CACHE[path]


# ---------------------------------------------------------------- jobs

def fold_job(job: dict) -> dict:
    from src.model_comparison import build_model
    from src.window_pipeline import aligned_predict_proba
    data = _load(job["cache"])
    held = np.flatnonzero(data["groups"] == job["subject"])
    train = np.flatnonzero(data["groups"] != job["subject"])
    if set(data["record_ids"][held]) & set(data["record_ids"][train]):
        raise ValueError("Recording leakage")
    if set(data["y"][train]) != set(range(len(CLASS_NAMES))):
        raise ValueError("Training fold lacks a class")
    tick = time.perf_counter()
    model = build_model(job["family"], job["params"], job["seed"])
    model.fit(data["X"][train], data["y"][train])
    probability = aligned_predict_proba(model, data["X"][held], len(CLASS_NAMES))
    if not np.allclose(probability.sum(1), 1):
        raise ValueError("Probabilities do not sum to one")
    return {"kind": "loso", "mode": job["mode"], "family": job["family"], "seed": job["seed"],
            "subject": job["subject"], "indices": held.tolist(), "probability": probability.tolist(),
            "training_windows": int(len(train)), "training_seconds": time.perf_counter() - tick}


def app_split_job(job: dict) -> dict:
    """Replay the app's 20/6/6 split: fit on train, score validation and final evaluation."""
    from src.model_comparison import build_model, scores
    from src.window_pipeline import aligned_predict_proba
    data = _load(job["cache"])
    train, validation, final = data["mask_train"], data["mask_validation"], data["mask_final_evaluation"]
    tick = time.perf_counter()
    model = build_model(job["family"], job["params"], job["seed"])
    model.fit(data["X"][train], data["y"][train])
    seconds = time.perf_counter() - tick
    v_prob = aligned_predict_proba(model, data["X"][validation], len(CLASS_NAMES))
    f_prob = aligned_predict_proba(model, data["X"][final], len(CLASS_NAMES))
    return {"kind": "app_split", "mode": job["mode"], "family": job["family"], "seed": job["seed"],
            "params": job["params"], "grid": job.get("grid", False),
            "validation": scores(data["y"][validation], v_prob.argmax(1)),
            "final": scores(data["y"][final], f_prob.argmax(1)),
            "final_prediction": f_prob.argmax(1).tolist(), "training_seconds": seconds}


def run_job(job: dict) -> dict:
    return fold_job(job) if job["kind"] == "loso" else app_split_job(job)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="New research/results run directory")
    parser.add_argument("--calibration", default=str(DEFAULT_MEASUREMENT_PATH), help="Finger-press measurement JSON")
    parser.add_argument("--cohort-run", default=str(COHORT_RUN), help="Training run whose staged records/split to reuse")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
    if out.exists():
        raise SystemExit(f"Output exists; use a new folder: {out}")
    cohort_run = Path(args.cohort_run).resolve()
    calibration_path = Path(args.calibration).resolve()
    out.mkdir(parents=True)
    started = now()
    before = protected_state()
    write(out / "protected_before.json", before)

    measurement, measurement_sha = load_measurement(calibration_path)
    shutil.copy2(calibration_path, out / "calibration_measurement.json")
    for relative in SOURCES:
        target = out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)

    print("Loading cohort and extracting features...", flush=True)
    data, evaluation, staging_audit = load_cohort(cohort_run)
    caches = build_caches(data, measurement, measurement_sha, out / "cache")
    write(out / "calibration.json", {"measurement_file": str(calibration_path.relative_to(ROOT))
                                     if calibration_path.is_relative_to(ROOT) else str(calibration_path),
                                     "measurement_sha256": measurement_sha,
                                     "calibration_id": measurement["calibration_id"],
                                     "modes": {mode: caches[mode]["calibration"] for mode in CALIBRATION_MODES}})
    subjects = sorted(set(map(str, data["meta"]["subject"])))
    active = read(ACTIVE_POINTER)
    protocol = {
        "schema_version": 1, "started_at": started,
        "question": ("Does per-sensor calibration from the finger-press comparison improve 32-subject LOSO "
                     "Macro F1 over uncalibrated input for the same model recipe?"),
        "cohort": {"run": str(cohort_run.relative_to(ROOT)), "dataset_fingerprint": data["fingerprint"],
                   "records": len(data["source_records"]), "subjects": subjects, "windows": int(len(data["y"])),
                   "window_seconds": WINDOW_SECONDS, "staging_audit": staging_audit,
                   "app_partitions": {k: v["subjects"] for k, v in evaluation["partitions"].items()},
                   "active_model_at_start": active},
        "calibration": {"measurement_sha256": measurement_sha, "calibration_id": measurement["calibration_id"],
                        "modes": CALIBRATION_MODES, "applied_to": "resampled 12 Hz window sequences before feature extraction "
                        "(per-channel affine, identical to applying on raw samples because resampling is linear interpolation)",
                        "clipping": None},
        "families": FAMILIES, "seeds": SEEDS, "criteria": CRITERIA,
        "primary_endpoint": "pooled 32-fold LOSO Macro F1 (all 4,706 windows predicted once), mean over 3 seeds",
        "decision_rule": (
            "For each calibrated mode (gain, offset_gain) x family (rbf_svm, extra_trees, xgboost) versus the same "
            "family with mode none: significant only if (1) seed-mean pooled LOSO Macro F1 delta >= +1.0%p, "
            "(2) delta > 0 for all 3 seeds, (3) Bonferroni-adjusted (6 comparisons, two-sided alpha 0.05) "
            "subject-cluster bootstrap lower bound of the seed-mean delta > 0 (10,000 replicates), and (4) no "
            "posture's seed-mean pooled F1 drops by more than 5%p. The calibration is called useful if at least one "
            "comparison is significant; GUI integration follows only then. App 20/6/6 split replay, Wilcoxon test, "
            "record-level F1 and per-subject counts are descriptive and do not change the decision."),
        "hyperparameters": "Fixed; not tuned inside LOSO. Chosen earlier on data including these subjects (shared bias across arms).",
        "exposure": "All 32 subjects were used in earlier experiments; retrospective, not new-user evidence.",
        "measurement_limitations": ("Finger force/contact area not controlled; only ranges from 3-4 repetitions; "
                                    "measured near maximal press, above typical seated load; acquisition date unknown."),
        "source_revision": git("rev-parse", "HEAD").strip(), "dirty_state": git("status", "--short"),
        "source_hashes": {relative: sha(ROOT / relative) for relative in SOURCES},
        "cache_hashes": {mode: caches[mode]["sha256"] for mode in CALIBRATION_MODES},
        "command": [sys.executable, *sys.argv],
        "environment": {"python": sys.version, "platform": platform.platform(),
                        **{name: package_version(name) for name in ("numpy", "scikit-learn", "scipy", "xgboost", "joblib")}},
        "models_saved": False, "active_model_changed": False, "raw_data_mutated": False,
    }
    write(out / "protocol.json", protocol)
    write(out / "protocol_lock.json", {"sha256": sha(out / "protocol.json"), "locked_at": now()})

    cache_path = {mode: str(out / caches[mode]["path"]) for mode in CALIBRATION_MODES}
    from src.model_comparison import MODEL_CANDIDATES
    jobs = []
    for family in ("xgboost", "extra_trees", "rbf_svm"):  # longest first
        for mode in CALIBRATION_MODES:
            for seed in SEEDS:
                jobs.append({"kind": "app_split", "mode": mode, "family": family, "seed": seed,
                             "params": FAMILIES[family], "cache": cache_path[mode]})
                jobs.extend({"kind": "loso", "mode": mode, "family": family, "seed": seed, "subject": s,
                             "params": FAMILIES[family], "cache": cache_path[mode]} for s in subjects)
    for mode in CALIBRATION_MODES:
        jobs.extend({"kind": "app_split", "mode": mode, "family": "rbf_svm", "seed": SPLIT_SEED, "grid": True,
                     "params": params, "cache": cache_path[mode]} for params in MODEL_CANDIDATES["rbf_svm"])
    n = len(data["y"])
    pooled = {(m, f, s): np.full((n, len(CLASS_NAMES)), np.nan) for m in CALIBRATION_MODES for f in FAMILIES for s in SEEDS}
    folds, app_rows = [], []
    print(f"{len(jobs)} fits on {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row["kind"] == "loso":
                key, idx = (row["mode"], row["family"], row["seed"]), np.asarray(row["indices"])
                if not np.isnan(pooled[key][idx]).all():
                    raise ValueError("A window was predicted twice")
                pooled[key][idx] = np.asarray(row["probability"])
                folds.append({k: row[k] for k in ("mode", "family", "seed", "subject", "training_windows", "training_seconds")})
            else:
                app_rows.append(row)
            if done % 50 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} fits done ({now()})", flush=True)
    training_finished = now()
    for key, probability in pooled.items():
        if np.isnan(probability).any():
            raise ValueError(f"Missing LOSO predictions for {key}")
    write(out / "folds.json", folds)
    truth = data["y"]
    groups = data["meta"]["subject"].to_numpy(dtype=str)
    records = data["meta"]["record_id"].to_numpy(dtype=str)
    np.savez_compressed(out / "loso_predictions.npz", truth=truth, groups=groups, record_ids=records,
                        **{f"{m}__{f}__{s}": p for (m, f, s), p in pooled.items()})

    summary = analyse(out, data, evaluation, pooled, app_rows, subjects, cohort_run)
    summary.update(started_at=started, training_finished_at=training_finished)
    after = protected_state()
    write(out / "protected_after.json", after)
    summary["protected_unchanged"] = before == after
    summary["sources_unchanged"] = all(sha(ROOT / r) == protocol["source_hashes"][r] for r in SOURCES)
    summary["protocol_unchanged"] = sha(out / "protocol.json") == read(out / "protocol_lock.json")["sha256"]
    summary["calibration_file_unchanged"] = sha(calibration_path) == measurement_sha
    summary["finished_at"] = now()
    write(out / "summary.json", summary)
    print(json.dumps({"useful": summary["decision"]["useful"], "significant": summary["decision"]["significant"],
                      "protected_unchanged": summary["protected_unchanged"],
                      "reproduction": summary["app_split"]["reproduction"]["passed"]}, ensure_ascii=False), flush=True)


def analyse(out: Path, data: dict, evaluation: dict, pooled: dict, app_rows: list, subjects: list, cohort_run: Path) -> dict:
    from scipy.stats import wilcoxon
    truth = data["y"]
    groups = data["meta"]["subject"].to_numpy(dtype=str)
    records = data["meta"]["record_id"].to_numpy(dtype=str)
    unique_records = list(dict.fromkeys(records))
    record_index = [np.flatnonzero(records == r) for r in unique_records]
    record_truth = np.array([truth[i[0]] for i in record_index])
    pred = {k: v.argmax(1) for k, v in pooled.items()}
    arms = {}
    for mode in CALIBRATION_MODES:
        for family in FAMILIES:
            p = [pred[mode, family, s] for s in SEEDS]
            arms[f"{mode}__{family}"] = {
                "mode": mode, "family": family,
                "pooled_macro_f1": [macro_f1(truth, x) for x in p],
                "pooled_accuracy": [float((x == truth).mean()) for x in p],
                "pooled_per_class_f1": [per_class_f1(truth, x).tolist() for x in p],
                "record_level_macro_f1": [macro_f1(record_truth, np.array([pooled[mode, family, s][i].mean(0).argmax()
                                                                           for i in record_index])) for s in SEEDS],
                "subject_macro_f1": {subject: float(np.mean([present_class_macro_f1(truth[groups == subject], x[groups == subject])
                                                             for x in p])) for subject in subjects},
            }
            arm = arms[f"{mode}__{family}"]
            for key in ("pooled_macro_f1", "pooled_accuracy", "record_level_macro_f1"):
                arm[f"{key}_mean"] = float(np.mean(arm[key]))
            arm["pooled_per_class_f1_mean"] = np.mean(arm["pooled_per_class_f1"], 0).tolist()
            values = list(arm["subject_macro_f1"].values())
            arm["subject_distribution"] = {"min": min(values), "median": float(np.median(values)), "max": max(values)}
    comparisons, significant = {}, []
    for mode in CALIBRATED:
        for family in FAMILIES:
            a, b = arms[f"none__{family}"], arms[f"{mode}__{family}"]
            boot = seed_mean_bootstrap(truth, np.stack([pred["none", family, s] for s in SEEDS]),
                                       np.stack([pred[mode, family, s] for s in SEEDS]), groups,
                                       replicates=CRITERIA["bootstrap_replicates"], seed=CRITERIA["bootstrap_seed"],
                                       comparisons=CRITERIA["bonferroni_comparisons"])
            decision = decide(a["pooled_macro_f1"], b["pooled_macro_f1"], a["pooled_per_class_f1"],
                              b["pooled_per_class_f1"], boot, CRITERIA)
            subject_delta = np.array([b["subject_macro_f1"][s] - a["subject_macro_f1"][s] for s in subjects])
            nonzero = subject_delta[np.abs(subject_delta) > 1e-12]
            test = wilcoxon(nonzero) if len(nonzero) else None
            name = f"{mode}__{family}"
            comparisons[name] = {**decision, "bootstrap": boot,
                                 "subjects_improved": int((subject_delta > 1e-12).sum()),
                                 "subjects_worse": int((subject_delta < -1e-12).sum()),
                                 "subject_delta_median": float(np.median(subject_delta)),
                                 "wilcoxon_two_sided_p": None if test is None else float(test.pvalue),
                                 "record_level_delta_mean": b["record_level_macro_f1_mean"] - a["record_level_macro_f1_mean"]}
            if decision["significant"]:
                significant.append(name)
    return {"arms": arms, "comparisons": comparisons,
            "decision": {"significant": significant, "useful": bool(significant),
                         "rule": "at least one preregistered comparison significant"},
            "app_split": app_split_summary(data, evaluation, app_rows, cohort_run),
            "fold_training_seconds_median": {f: float(np.median([r["training_seconds"] for r in read(out / "folds.json")
                                                                 if r["family"] == f])) for f in FAMILIES}}


def app_split_summary(data: dict, evaluation: dict, rows: list, cohort_run: Path) -> dict:
    import joblib
    fixed = {f"{r['mode']}__{r['family']}__{r['seed']}": {"validation_macro_f1": r["validation"]["macro_f1"],
                                                          "final_macro_f1": r["final"]["macro_f1"],
                                                          "final_per_class_f1": r["final"]["per_class_f1"]}
             for r in rows if not r["grid"]}
    grid = {}
    for mode in CALIBRATION_MODES:
        candidates = [r for r in rows if r["grid"] and r["mode"] == mode]
        best = max(candidates, key=lambda r: (r["validation"]["macro_f1"], r["validation"]["worst_class_f1"]))
        grid[mode] = {"selected_params": best["params"], "validation_macro_f1": best["validation"]["macro_f1"],
                      "final_macro_f1": best["final"]["macro_f1"], "final_accuracy": best["final"]["accuracy"],
                      "final_per_class_f1": best["final"]["per_class_f1"],
                      "candidates": [{"params": r["params"], "validation_macro_f1": r["validation"]["macro_f1"]} for r in candidates],
                      "_prediction": best["final_prediction"]}
    bundle = joblib.load(cohort_run / "artifacts/posture_classifier_window_3s_rbf_svm.joblib")
    final = data["masks"]["final_evaluation"]
    stored_prediction = np.asarray(bundle["model"].predict_proba(data["X"][final])).argmax(1)
    classes = np.asarray(bundle["model"].classes_)
    stored_prediction = classes[stored_prediction]
    replay = np.asarray(grid["none"].pop("_prediction"))
    for mode in CALIBRATED:
        grid[mode].pop("_prediction")
    checks = {"selected_params": grid["none"]["selected_params"] == evaluation["selected_params"],
              "validation_macro_f1": abs(grid["none"]["validation_macro_f1"] - evaluation["validation_metrics"]["macro_f1"]) < 1e-12,
              "final_macro_f1": abs(grid["none"]["final_macro_f1"] - evaluation["metrics"]["macro_f1"]) < 1e-12,
              "final_predictions_equal_stored_model": bool(np.array_equal(replay, stored_prediction))}
    return {"fixed_params": fixed, "app_grid_rbf_svm": grid,
            "reproduction": {"checks": checks, "passed": all(checks.values()),
                             "reference": str((cohort_run / "artifacts/window_3s_rbf_svm_evaluation.json").relative_to(ROOT))},
            "note": "Descriptive: 20 training subjects, 6 validation, 6 final-evaluation subjects (the app's stored split)."}


if __name__ == "__main__":
    main()
