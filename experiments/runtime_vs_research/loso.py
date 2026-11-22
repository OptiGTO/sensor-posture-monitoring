"""Leave-one-subject-out (LOSO) re-comparison of the four runtime-vs-research arms.

Every one of the 32 subjects is predicted exactly once by a model trained on the
other 31, so the estimate rests on 32 subjects instead of the six-person holdout.
Hyperparameters stay fixed (v5 SVM / research ET); nothing is tuned inside the
loop. The replacement rule is written to protocol.json before any fit.

Usage (new output folder only):
  .venv/bin/python -B -m experiments.runtime_vs_research.loso --output research/results/<YYYY-MM-DD_HHMMSS>_runtime_vs_research_loso
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.runtime_vs_research.analysis import factorial_effects, macro_f1, paired_subject_bootstrap, per_class_f1  # noqa: E402
from experiments.runtime_vs_research.run import (  # noqa: E402
    ADOPT, ARMS, ORDINARY, SEEDS, load_frozen, now, protected_state, read, recipe, sha, write)

SOURCES = ["experiments/runtime_vs_research/loso.py", "experiments/runtime_vs_research/run.py",
           "experiments/runtime_vs_research/analysis.py", "src/iterative_pipeline.py",
           "experiments/iterative_improvement/supplemental.py", "src/posture_core.py"]
CRITERIA = {"minimum_pooled_macro_f1_delta": 0.01, "all_seed_deltas_positive": True,
            "bootstrap_lower_must_exceed": 0.0, "maximum_per_class_drop": 0.05,
            "bootstrap_replicates": 10000, "bootstrap_seed": 20260928}


def present_macro_f1(truth, prediction) -> float:
    from sklearn.metrics import f1_score
    labels = sorted(set(np.asarray(truth).tolist()))
    return float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0))


def fold_job(job: dict) -> dict:
    from src.iterative_pipeline import build_pipeline, fit_pipeline
    from experiments.iterative_improvement.supplemental import append_training
    _, data, supplement = load_frozen()
    held = job["subject"]
    train = np.flatnonzero(data["groups"] != held)
    valid = np.flatnonzero(data["groups"] == held)
    spec = job["pipeline"]
    X, y, groups = data["X"][train], data["y"][train], data["groups"][train]
    added = 0
    if spec["training"].get("data_policy") == ORDINARY:
        X, y, groups, audit = append_training(data, supplement, train)
        added = audit["added_windows"]
        if held in set(audit["training_subjects"]):
            raise ValueError("Held-out subject leaked into supplement")
    tick = time.perf_counter()
    model = build_pipeline(spec, seed=job["seed"])
    fit_pipeline(model, spec, X, y, groups=groups)
    probability = model.predict_proba(data["X"][valid])
    if probability.shape != (len(valid), 7) or not np.allclose(probability.sum(1), 1):
        raise ValueError("Invalid probability contract")
    return {"arm": job["arm"], "seed": job["seed"], "subject": held, "indices": valid.tolist(),
            "probability": probability.tolist(), "training_windows": int(len(y)), "added_windows": added,
            "training_seconds": time.perf_counter() - tick}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.output) if Path(args.output).is_absolute() else (ROOT / args.output)
    if out.exists():
        raise SystemExit(f"Output exists; use a new folder: {out}")
    out.mkdir(parents=True)
    started = now()
    before = protected_state()
    write(out / "protected_before.json", before)
    _, data, _ = load_frozen()
    subjects = sorted(set(map(str, data["groups"])))
    for relative in SOURCES:
        target = out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    protocol = {
        "schema_version": 1, "started_at": started,
        "question": "Under 32-subject LOSO, does the research ET baseline justify replacing the runtime RBF-SVM v5 recipe?",
        "comparison": {"A_runtime": "svm_standard", "B_research": "et_ordinary"},
        "arms": {name: recipe(*spec) for name, spec in ARMS.items()}, "seeds": SEEDS, "criteria": CRITERIA,
        "decision_rule": ("Recommend replacement only if pooled LOSO Macro F1 (B-A) averaged over seeds >= +1%p, "
                          "every seed delta > 0, primary-seed paired subject bootstrap 2.5% bound > 0, and no per-class "
                          "pooled F1 drop > 5%p (seed mean). Otherwise keep v5. Fixed before fitting."),
        "evaluation": "Each subject held out once; train on the other 31; ordinary-policy supplement only for the 31 training subjects.",
        "hyperparameters": "Fixed, not tuned inside LOSO. They were chosen earlier on data that includes these subjects; both arms share this bias.",
        "cohort": {"source_run": str(ADOPT.relative_to(ROOT)), "data_sha256": sha(ADOPT / "data.npz"),
                   "supplemental_sha256": sha(ADOPT / "supplemental_training.npz"), "subjects": subjects,
                   "windows": int(len(data["y"]))},
        "exposure": "All 32 subjects appeared in earlier development or holdout evaluations; retrospective, not new-user evidence.",
        "source_revision": git("rev-parse", "HEAD").strip(), "dirty_state": git("status", "--short"),
        "source_hashes": {r: sha(ROOT / r) for r in SOURCES}, "command": [sys.executable, *sys.argv],
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "packages": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.split()},
        "models_saved": False, "active_model_changed": False, "raw_data_mutated": False,
    }
    write(out / "protocol.json", protocol)
    write(out / "protocol_lock.json", {"sha256": sha(out / "protocol.json"), "locked_at": now()})

    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    jobs = [{"arm": arm, "seed": seed, "subject": s, "pipeline": recipe(*spec)}
            for seed in SEEDS for arm, spec in ARMS.items() for s in subjects]
    pooled = {(arm, seed): np.full((len(data["y"]), 7), np.nan) for arm in ARMS for seed in SEEDS}
    folds = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(fit_job_wrapper, jobs), 1):
            key = (row["arm"], row["seed"])
            idx = np.asarray(row["indices"])
            if not np.isnan(pooled[key][idx]).all():
                raise ValueError("A window was predicted twice")
            pooled[key][idx] = np.asarray(row["probability"])
            folds.append({k: row[k] for k in ("arm", "seed", "subject", "training_windows", "added_windows", "training_seconds")})
            if i % 32 == 0:
                print(f"{i}/{len(jobs)} folds done ({row['arm']} seed={row['seed']})", flush=True)
    write(out / "folds.json", folds)
    training_finished = now()
    truth, groups = data["y"], data["groups"]
    for key, prob in pooled.items():
        if np.isnan(prob).any():
            raise ValueError(f"Missing LOSO predictions for {key}")
    np.savez_compressed(out / "loso_predictions.npz", truth=truth, groups=groups, record_ids=data["record_ids"],
                        **{f"{arm}__{seed}": pooled[arm, seed] for arm in ARMS for seed in SEEDS})

    pred = {k: v.argmax(1) for k, v in pooled.items()}
    pooled_f1 = {arm: [macro_f1(truth, pred[arm, s]) for s in SEEDS] for arm in ARMS}
    pooled_acc = {arm: [float((pred[arm, s] == truth).mean()) for s in SEEDS] for arm in ARMS}
    class_f1 = {arm: np.mean([per_class_f1(truth, pred[arm, s]) for s in SEEDS], 0) for arm in ARMS}
    subject_f1 = {arm: {s: float(np.mean([present_macro_f1(truth[groups == s], pred[arm, seed][groups == s]) for seed in SEEDS]))
                        for s in subjects} for arm in ARMS}
    # record-level: mean window probability per recording, then argmax (auxiliary)
    records = data["record_ids"]
    unique_records = sorted(set(map(str, records)))
    record_truth = np.array([truth[records == r][0] for r in unique_records])
    record_f1 = {arm: [macro_f1(record_truth, np.array([pooled[arm, s][records == r].mean(0).argmax() for r in unique_records]))
                       for s in SEEDS] for arm in ARMS}

    A, B = "svm_standard", "et_ordinary"
    deltas = np.array(pooled_f1[B]) - np.array(pooled_f1[A])
    boot = paired_subject_bootstrap(truth, pred[A, SEEDS[0]], pred[B, SEEDS[0]], groups,
                                    replicates=CRITERIA["bootstrap_replicates"], seed=CRITERIA["bootstrap_seed"])
    class_delta = class_f1[B] - class_f1[A]
    wins = sum(subject_f1[B][s] > subject_f1[A][s] for s in subjects)
    checks = {"pooled_mean_delta": bool(deltas.mean() >= CRITERIA["minimum_pooled_macro_f1_delta"]),
              "all_seed_deltas_positive": bool((deltas > 0).all()),
              "bootstrap_lower_positive": bool(boot["lower_2_5"] > CRITERIA["bootstrap_lower_must_exceed"]),
              "per_class_drop": bool(class_delta.min() >= -CRITERIA["maximum_per_class_drop"])}
    pairs = {}
    for a, b in (("svm_standard", "et_standard"), ("svm_standard", "svm_ordinary"), ("et_standard", "et_ordinary"),
                 ("svm_ordinary", "et_ordinary")):
        pairs[f"{b}_minus_{a}"] = {
            "pooled_delta_by_seed": (np.array(pooled_f1[b]) - np.array(pooled_f1[a])).tolist(),
            "bootstrap_primary_seed": paired_subject_bootstrap(truth, pred[a, SEEDS[0]], pred[b, SEEDS[0]], groups,
                                                               replicates=CRITERIA["bootstrap_replicates"], seed=CRITERIA["bootstrap_seed"]),
            "subjects_improved": int(sum(subject_f1[b][s] > subject_f1[a][s] for s in subjects))}
    # agreement with earlier six-person holdout: LOSO restricted to those six subjects (different training size: 31 vs 26)
    holdout6 = read(ADOPT / "protocol.json")["partitions"][0]["test"]
    mask6 = np.isin(groups, holdout6)
    summary = {
        "started_at": started, "training_finished_at": training_finished,
        "pooled_macro_f1": {arm: {"by_seed": pooled_f1[arm], "mean": float(np.mean(pooled_f1[arm]))} for arm in ARMS},
        "pooled_accuracy": {arm: {"by_seed": pooled_acc[arm], "mean": float(np.mean(pooled_acc[arm]))} for arm in ARMS},
        "record_level_macro_f1": {arm: {"by_seed": record_f1[arm], "mean": float(np.mean(record_f1[arm]))} for arm in ARMS},
        "pooled_per_class_f1_seed_mean": {arm: class_f1[arm].tolist() for arm in ARMS},
        "subject_present_class_macro_f1_seed_mean": subject_f1,
        "subject_distribution": {arm: {"min": min(v.values()), "median": float(np.median(list(v.values()))), "max": max(v.values())}
                                 for arm, v in subject_f1.items()},
        "factorial_pooled": factorial_effects({ARMS[a]: np.mean(pooled_f1[a]) for a in ARMS}),
        "decision_B_vs_A": {"checks": checks, "replace": all(checks.values()), "pooled_deltas": deltas.tolist(),
                            "mean_delta": float(deltas.mean()), "bootstrap": boot, "class_deltas": class_delta.tolist(),
                            "subjects_improved": int(wins), "subjects": len(subjects)},
        "pairwise": pairs,
        "holdout6_subset": {"subjects": holdout6, "windows": int(mask6.sum()),
                            "macro_f1_seed_mean": {arm: float(np.mean([macro_f1(truth[mask6], pred[arm, s][mask6]) for s in SEEDS])) for arm in ARMS},
                            "note": "LOSO models train on 31 subjects; earlier holdout models trained on 26."},
        "non_holdout26_subset": {"macro_f1_seed_mean": {arm: float(np.mean([macro_f1(truth[~mask6], pred[arm, s][~mask6]) for s in SEEDS])) for arm in ARMS}},
        "fold_training_seconds_median": {arm: float(np.median([f["training_seconds"] for f in folds if f["arm"] == arm])) for arm in ARMS},
    }
    after = protected_state()
    write(out / "protected_after.json", after)
    summary["protected_unchanged"] = before == after
    summary["sources_unchanged"] = all(sha(ROOT / r) == protocol["source_hashes"][r] for r in SOURCES)
    summary["protocol_unchanged"] = sha(out / "protocol.json") == read(out / "protocol_lock.json")["sha256"]
    summary["finished_at"] = now()
    write(out / "summary.json", summary)
    print({"replace": summary["decision_B_vs_A"]["replace"], "checks": checks,
           "protected_unchanged": summary["protected_unchanged"]}, flush=True)


def fit_job_wrapper(job):
    return fold_job(job)


if __name__ == "__main__":
    main()
