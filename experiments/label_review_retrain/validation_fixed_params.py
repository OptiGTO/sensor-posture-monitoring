"""Post-hoc: RBF-SVM with fixed parameters on the same single split (after validation_run).

The validation step switched gamma between 'scale' and 0.0005 across conditions/seeds.
This refits C=1 with each gamma fixed, per condition and seed, on the same train fit
rows, and scores validation/final on the same sets. Result is written into the
existing run folder as posthoc_fixed_params.json (the run's own files are untouched).

Usage: .venv/bin/python -B -W ignore -m experiments.label_review_retrain.validation_fixed_params --run <validation run folder>
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.label_review_retrain import validation_run as V  # noqa: E402
from experiments.label_review_retrain.analysis import macro_f1  # noqa: E402
from experiments.runtime_vs_research.run import now, protected_state, read, sha, write  # noqa: E402

FIXED = {"gamma_scale": {"C": 1.0, "gamma": "scale"}, "gamma_0.0005": {"C": 1.0, "gamma": 0.0005}}


def fit_job(job: dict) -> dict:
    from src.posture_core import CLASS_NAMES
    from src.training_methods import build_method_model, fit_method_model
    from src.window_pipeline import aligned_predict_proba
    cache = V._cache(job["cache"])
    X, y, groups, records = (np.asarray(cache[k]) for k in ("X", "y", "groups", "records"))
    train = np.flatnonzero(np.asarray(cache[f"fit__{job['condition']}"]))
    validation = np.flatnonzero(np.asarray(cache["part__validation"]))
    model = build_method_model("rbf_svm", FIXED[job["params"]], job["seed"], "standard")
    fit_method_model(model, "rbf_svm", "standard", X[train], y[train], groups=groups[train],
                     record_ids=records[train], validation_data=(X[validation], y[validation]))
    result = {k: job[k] for k in ("condition", "params", "seed")}
    for part in ("validation", "final_evaluation"):
        rows = np.flatnonzero(np.asarray(cache[f"part__{part}"]))
        prediction = aligned_predict_proba(model, X[rows], len(CLASS_NAMES)).argmax(axis=1)
        for name in ("all", "valid", "review_ineligible"):
            keep = np.asarray(cache[f"set__{name}"])[rows]
            result[f"{part}_{name}"] = macro_f1(y[rows][keep], prediction[keep])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    run = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    target = run / "posthoc_fixed_params.json"
    if target.exists():
        raise SystemExit("posthoc_fixed_params.json exists")
    started = now()
    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    before = protected_state()
    work = Path(tempfile.mkdtemp(prefix="label_review_fixed_"))
    stage, sensor5, _ = V.stage_cohort(work / "training_data")
    cohort = V.prepare(work, stage, sensor5, read(ROOT / V.REVIEW_DECISIONS)["records"])
    if cohort["cache_sha256"] != read(run / "cohort.json")["cache_sha256"]:
        raise ValueError("Rebuilt arrays differ from the validation run")
    jobs = [{"condition": c, "params": p, "seed": s, "cache": str(work / "cache")}
            for c in V.CONDITIONS for p in FIXED for s in V.SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(fit_job, jobs))
    summary = {}
    for p in FIXED:
        for c in V.CONDITIONS:
            picked = [r for r in rows if r["params"] == p and r["condition"] == c]
            summary[f"{p}:{c}"] = {key: float(np.mean([r[key] for r in picked])) for key in picked[0]
                                   if key not in ("condition", "params", "seed")}
            summary[f"{p}:{c}"]["final_evaluation_valid_by_seed"] = [r["final_evaluation_valid"] for r in picked]
    shutil.rmtree(work)
    after = protected_state()
    write(target, {"started_at": started, "finished_at": now(), "post_hoc": True, "fixed_params": FIXED,
                   "seeds": V.SEEDS, "source_sha256": sha(Path(__file__)), "rows": rows, "seed_means": summary,
                   "protected_unchanged": before == after, "arrays_match_run": True})
    print({k: round(v["final_evaluation_valid"] * 100, 2) for k, v in summary.items()}, before == after)


if __name__ == "__main__":
    main()
