"""Runtime RBF-SVM v5 recipe vs research-baseline Extra Trees under identical conditions.

2x2 factorial: model {v5 RBF-SVM recipe, research ET recipe} x training data policy
{standard, ordinary reserved-record restore}. Same frozen 32-subject cohort, same three
development splits, same historical holdout, three seeds. The protocol and decision
rule are written before any fit. The historical holdout is reused, not new evidence.

Usage (new output folder only):
  .venv/bin/python -B -m experiments.runtime_vs_research.run --output research/results/<YYYY-MM-DD_HHMMSS>_runtime_vs_research
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.runtime_vs_research.analysis import (  # noqa: E402
    decide, factorial_effects, macro_f1, paired_subject_bootstrap, per_class_f1)

ADOPT = ROOT / "research/results/2026-09-22_013650_adopt_et_ordinary"
V5_DIR = ROOT / "research/models/models/integrated_20260908_121321_070b045a/artifacts"
V5_MODEL = V5_DIR / "posture_classifier_window_3s_rbf_svm.joblib"
V5_EVAL = V5_DIR / "window_3s_rbf_svm_evaluation.json"
ACTIVE = ROOT / "research/models/active_model.json"
BASELINE_POINTER = ROOT / "research/models/checkpoints/research_baseline.json"
SEEDS = [20260921, 20260922, 20260923]
ORDINARY = "include_ordinary_reserved_training_records_v1"
SOURCES = ["experiments/runtime_vs_research/run.py", "experiments/runtime_vs_research/analysis.py",
           "src/iterative_pipeline.py", "experiments/iterative_improvement/supplemental.py",
           "src/posture_core.py", "src/window_pipeline.py"]
CRITERIA = {"minimum_development_mean_delta": 0.01, "minimum_positive_splits": 2,
            "maximum_per_class_drop": 0.05, "minimum_holdout_delta": 0.01,
            "bootstrap_replicates": 10000, "bootstrap_seed": 20260928,
            "reproduction_probability_atol": 1e-12}


def recipe(model: str, policy: str) -> dict:
    training = {"seed": SEEDS[0], "weighting": "class_balanced"}
    if policy == "ordinary":
        training["data_policy"] = ORDINARY
    if model == "svm":  # exact v5 hyperparameters (model_config.json svm_params)
        return {"family": "rbf_svm", "base_method": "standard", "window_seconds": 3,
                "feature_engineering": {"mode": "all"}, "preprocessing": {"scaler": "standard"},
                "model_params": {"C": 1.0, "gamma": 0.0005}, "training": training,
                "postprocessing": {"kind": "argmax"}}
    return {"family": "extra_trees", "base_method": "standard", "window_seconds": 3,
            "feature_engineering": {"mode": "all"}, "preprocessing": {"scaler": "standard"},
            "model_params": {"n_estimators": 200, "min_samples_leaf": 8, "max_depth": 8, "max_features": 0.5},
            "training": training, "postprocessing": {"kind": "argmax"}}


ARMS = {f"{m}_{p}": (m, p) for m in ("svm", "et") for p in ("standard", "ordinary")}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
                st = path.stat()
                state[str(path.relative_to(ROOT))] = [sha(path), st.st_size, st.st_mtime_ns]
    return state


def load_frozen() -> tuple[dict, dict, dict]:
    protocol = read(ADOPT / "protocol.json")
    if sha(ADOPT / "data.npz") != protocol["cache_sha256"]:
        raise ValueError("Frozen cohort cache changed")
    if sha(ADOPT / "supplemental_training.npz") != protocol["supplemental_training"]["cache_sha256"]:
        raise ValueError("Frozen supplemental cache changed")
    data = dict(np.load(ADOPT / "data.npz", allow_pickle=False))
    supplement = dict(np.load(ADOPT / "supplemental_training.npz", allow_pickle=False))
    return protocol, data, supplement


def split_rows(data: dict, part: dict, final: bool):
    train_subjects = part["train"] + part["validation"] if final else part["train"]
    eval_subjects = part["test"] if final else part["validation"]
    if set(train_subjects) & set(eval_subjects):
        raise ValueError("Subject leakage")
    train = np.flatnonzero(np.isin(data["groups"], train_subjects))
    valid = np.flatnonzero(np.isin(data["groups"], eval_subjects))
    return train, valid


def fit_job(job: dict) -> dict:
    from src.iterative_pipeline import build_pipeline, fit_pipeline
    from experiments.iterative_improvement.supplemental import append_training
    out = Path(job["out"])
    dest = out / job["relative"]
    dest.mkdir(parents=True, exist_ok=True)
    _, data, supplement = load_frozen()
    part = job["partition"]
    train, valid = split_rows(data, part, job["final"])
    spec = job["pipeline"]
    X, y, groups, audit = data["X"][train], data["y"][train], data["groups"][train], None
    if spec["training"].get("data_policy") == ORDINARY:
        X, y, groups, audit = append_training(data, supplement, train)
    started, tick = now(), time.perf_counter()
    model = build_pipeline(spec, seed=job["seed"])
    fit_pipeline(model, spec, X, y, groups=groups)
    seconds = time.perf_counter() - tick
    probability = model.predict_proba(data["X"][valid])
    if probability.shape != (len(valid), 7) or not np.allclose(probability.sum(1), 1):
        raise ValueError("Invalid probability contract")
    prediction = probability.argmax(1)
    joblib.dump(model, dest / "model.joblib", compress=3)
    np.savez_compressed(dest / "predictions.npz", indices=valid, truth=data["y"][valid], groups=data["groups"][valid],
                        record_ids=data["record_ids"][valid], probability=probability, prediction=prediction)
    truth = data["y"][valid]
    row = {"arm": job["arm"], "seed": job["seed"], "partition": part["id"], "final": job["final"], "pipeline": spec,
           "started_at": started, "finished_at": now(), "training_seconds": seconds,
           "training_windows": int(len(y)), "training_subjects": sorted(set(map(str, groups))),
           "evaluation_subjects": sorted(set(map(str, data["groups"][valid]))), "evaluation_windows": int(len(valid)),
           "macro_f1": macro_f1(truth, prediction), "accuracy": float((truth == prediction).mean()),
           "per_class_f1": per_class_f1(truth, prediction).tolist(),
           "supplemental_audit": audit, "predictions_sha256": sha(dest / "predictions.npz"),
           "model_sha256": sha(dest / "model.joblib")}
    write(dest / "metrics.json", row)
    return row


def reproduction(out: Path) -> dict:
    """ET arms at every seed must reproduce the earlier adoption run exactly."""
    rows, ok = [], True
    mapping = {"et_ordinary": "ET_ordinary", "et_standard": "baseline"}
    for arm, old in mapping.items():
        for seed in SEEDS:
            pairs = [(out / f"final/{seed}/{arm}", ADOPT / f"final/{seed}/{old}")]
            for k in (1, 2, 3):
                base = ADOPT / (f"development/{old}" if seed == SEEDS[0] else f"robustness/{seed}/{old}")
                pairs.append((out / f"development/{seed}/{arm}/development_{k}", base / f"development_{k}"))
            for new, ref in pairs:
                if not (ref / "predictions.npz").exists():
                    rows.append({"new": str(new.relative_to(out)), "reference": str(ref.relative_to(ROOT)), "status": "reference_missing"})
                    continue
                a, b = np.load(new / "predictions.npz"), np.load(ref / "predictions.npz")
                same = (np.array_equal(a["indices"], b["indices"]) and np.array_equal(a["prediction"], b["prediction"])
                        and np.allclose(a["probability"], b["probability"], rtol=0, atol=CRITERIA["reproduction_probability_atol"]))
                ok &= bool(same)
                rows.append({"new": str(new.relative_to(out)), "reference": str(ref.relative_to(ROOT)),
                             "status": "equal" if same else "DIFFERENT",
                             "max_abs_probability_difference": float(np.abs(a["probability"] - b["probability"]).max())})
    return {"all_available_equal": ok, "compared": sum(r["status"] != "reference_missing" for r in rows), "rows": rows}


def v5_artifact_check(out: Path, data: dict, holdout_part: dict) -> dict:
    """Descriptive: shipped v5 bundle on holdout subjects it never trained on."""
    evaluation = read(V5_EVAL)
    seen_train = set(evaluation["partitions"]["train"]["subjects"])
    seen_any = seen_train | set(evaluation["partitions"]["validation"]["subjects"]) | set(evaluation["partitions"]["final_evaluation"]["subjects"])
    bundle = joblib.load(V5_MODEL)
    from src.posture_core import feature_names
    if bundle["feature_names"] != feature_names():
        raise ValueError("v5 feature contract differs from frozen cache")
    result = {"artifact_sha256": sha(V5_MODEL), "active_artifact_hash": read(ACTIVE)["artifact_hash"],
              "v5_training_subjects": sorted(seen_train), "groups": {}}
    for label, subjects in (("never_seen_by_v5", [s for s in holdout_part["test"] if s not in seen_any]),
                            ("not_in_v5_training", [s for s in holdout_part["test"] if s not in seen_train])):
        rows = np.flatnonzero(np.isin(data["groups"], subjects))
        if not len(rows):
            result["groups"][label] = {"subjects": subjects, "windows": 0}
            continue
        model = bundle["model"]
        raw = model.predict_proba(data["X"][rows])
        prob = np.zeros((len(rows), 7))
        prob[:, np.asarray(model.classes_, dtype=int)] = raw
        truth = data["y"][rows]
        entry = {"subjects": subjects, "windows": int(len(rows)), "classes_present": sorted(set(truth.tolist())),
                 "v5_artifact_macro_f1_present_classes": None, "v5_artifact_accuracy": float((prob.argmax(1) == truth).mean())}
        present = sorted(set(truth.tolist()))
        from sklearn.metrics import f1_score
        entry["v5_artifact_macro_f1_present_classes"] = float(f1_score(truth, prob.argmax(1), labels=present, average="macro", zero_division=0))
        for arm in ARMS:
            p = np.load(out / f"final/{SEEDS[0]}/{arm}/predictions.npz")
            pos = np.flatnonzero(np.isin(p["groups"], subjects))
            entry[f"{arm}_accuracy"] = float((p["prediction"][pos] == p["truth"][pos]).mean())
            entry[f"{arm}_macro_f1_present_classes"] = float(f1_score(p["truth"][pos], p["prediction"][pos], labels=present, average="macro", zero_division=0))
        result["groups"][label] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    if out.exists():
        raise SystemExit(f"Output exists; use a new folder: {out}")
    out.mkdir(parents=True)
    started = now()
    before = protected_state()
    write(out / "protected_before.json", before)
    adopt_protocol, data, _ = load_frozen()
    for relative in SOURCES:
        target = out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    protocol = {
        "schema_version": 1, "started_at": started,
        "question": "Does the research ET baseline justify replacing the runtime RBF-SVM v5 recipe under identical conditions?",
        "comparison": {"A_runtime": "svm_standard", "B_research": "et_ordinary"},
        "arms": {name: recipe(*spec) for name, spec in ARMS.items()},
        "seeds": SEEDS, "criteria": CRITERIA,
        "decision_rule": "Recommend replacement only if every check in analysis.decide passes for B vs A; otherwise keep v5. Rule fixed before fitting.",
        "partitions": adopt_protocol["partitions"],
        "cohort": {"source_run": str(ADOPT.relative_to(ROOT)), "data_sha256": sha(ADOPT / "data.npz"),
                   "supplemental_sha256": sha(ADOPT / "supplemental_training.npz"),
                   "subjects": int(len(set(data["groups"]))), "windows": int(len(data["y"]))},
        "v5_artifact": {"path": str(V5_MODEL.relative_to(ROOT)), "sha256": sha(V5_MODEL),
                        "active_artifact_hash": read(ACTIVE)["artifact_hash"],
                        "note": "v5 was trained on a different 15-subject set in 2026-09-08; retrained recipe is compared, artifact check is descriptive."},
        "research_baseline_pointer_sha256": sha(BASELINE_POINTER),
        "exposure": "Historical six-subject holdout was already observed in prior experiments; results are retrospective, not new-user evidence.",
        "source_revision": git("rev-parse", "HEAD").strip(), "dirty_state": git("status", "--short"),
        "source_hashes": {r: sha(ROOT / r) for r in SOURCES},
        "command": [sys.executable, *sys.argv],
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "packages": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.split()},
        "active_model_changed": False, "raw_data_mutated": False,
    }
    write(out / "protocol.json", protocol)
    write(out / "protocol_lock.json", {"sha256": sha(out / "protocol.json"), "locked_at": now()})

    jobs = []
    for seed in SEEDS:
        for arm, spec in ARMS.items():
            pipeline = recipe(*spec)
            for part in adopt_protocol["partitions"]:
                jobs.append({"out": str(out), "arm": arm, "seed": seed, "final": False, "partition": part, "pipeline": pipeline,
                             "relative": f"development/{seed}/{arm}/{part['id']}"})
            jobs.append({"out": str(out), "arm": arm, "seed": seed, "final": True, "partition": adopt_protocol["partitions"][0],
                         "pipeline": pipeline, "relative": f"final/{seed}/{arm}"})
    env_threads = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    os.environ.update(env_threads)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for row in pool.map(fit_job, jobs):
            rows.append(row)
            print(f"{row['arm']:12s} seed={row['seed']} {row['partition'] if not row['final'] else 'holdout':14s} F1={row['macro_f1']:.4f}", flush=True)
    write(out / "fits.json", rows)
    training_finished = now()

    def pick(arm, final, seed=None):
        return [r for r in rows if r["arm"] == arm and r["final"] == final and (seed is None or r["seed"] == seed)]

    dev = {arm: np.array([[r["macro_f1"] for r in sorted(pick(arm, False, s), key=lambda r: r["partition"])] for s in SEEDS]) for arm in ARMS}
    dev_cls = {arm: np.array([[r["per_class_f1"] for r in sorted(pick(arm, False, s), key=lambda r: r["partition"])] for s in SEEDS]) for arm in ARMS}
    hold = {arm: np.array([pick(arm, True, s)[0]["macro_f1"] for s in SEEDS]) for arm in ARMS}
    hold_cls = {arm: np.array([pick(arm, True, s)[0]["per_class_f1"] for s in SEEDS]) for arm in ARMS}

    pa = np.load(out / f"final/{SEEDS[0]}/svm_standard/predictions.npz")
    pb = np.load(out / f"final/{SEEDS[0]}/et_ordinary/predictions.npz")
    if not np.array_equal(pa["indices"], pb["indices"]):
        raise ValueError("Holdout rows misaligned")
    boot = paired_subject_bootstrap(pa["truth"], pa["prediction"], pb["prediction"], pa["groups"],
                                    replicates=CRITERIA["bootstrap_replicates"], seed=CRITERIA["bootstrap_seed"])
    decision = decide(dev["svm_standard"], dev["et_ordinary"], dev_cls["svm_standard"], dev_cls["et_ordinary"],
                      hold["svm_standard"], hold["et_ordinary"], hold_cls["svm_standard"], hold_cls["et_ordinary"],
                      boot["lower_2_5"], CRITERIA)
    secondary = {}
    for a, b in (("svm_standard", "et_standard"), ("svm_standard", "svm_ordinary"), ("et_standard", "et_ordinary")):
        qa = np.load(out / f"final/{SEEDS[0]}/{a}/predictions.npz")
        qb = np.load(out / f"final/{SEEDS[0]}/{b}/predictions.npz")
        secondary[f"{b}_minus_{a}"] = {
            "development_primary_seed_mean": float((dev[b][0] - dev[a][0]).mean()),
            "development_all_seed_mean": float((dev[b] - dev[a]).mean()),
            "holdout_seed_mean": float((hold[b] - hold[a]).mean()),
            "holdout_bootstrap_primary_seed": paired_subject_bootstrap(qa["truth"], qa["prediction"], qb["prediction"], qa["groups"],
                                                                       replicates=CRITERIA["bootstrap_replicates"], seed=CRITERIA["bootstrap_seed"])}
    per_subject = {}
    for arm in ARMS:
        p = np.load(out / f"final/{SEEDS[0]}/{arm}/predictions.npz")
        per_subject[arm] = {str(g): macro_f1(p["truth"][p["groups"] == g], p["prediction"][p["groups"] == g]) for g in sorted(set(p["groups"]))}
    summary = {
        "started_at": started, "training_finished_at": training_finished,
        "development_macro_f1": {arm: {"by_seed_split": dev[arm].tolist(), "primary_seed_mean": float(dev[arm][0].mean()),
                                       "all_seed_mean": float(dev[arm].mean())} for arm in ARMS},
        "holdout_macro_f1": {arm: {"by_seed": hold[arm].tolist(), "mean": float(hold[arm].mean())} for arm in ARMS},
        "holdout_per_class_f1_seed_mean": {arm: hold_cls[arm].mean(0).tolist() for arm in ARMS},
        "development_per_class_f1_mean": {arm: dev_cls[arm].mean((0, 1)).tolist() for arm in ARMS},
        "factorial_development_all_seeds": factorial_effects({ARMS[a]: dev[a].mean() for a in ARMS}),
        "factorial_holdout_seed_mean": factorial_effects({ARMS[a]: hold[a].mean() for a in ARMS}),
        "primary_bootstrap": boot, "decision_B_vs_A": decision, "secondary": secondary,
        "holdout_per_subject_primary_seed": per_subject,
        "training_seconds_median": {arm: float(np.median([r["training_seconds"] for r in rows if r["arm"] == arm])) for arm in ARMS},
        "model_bytes_holdout_primary_seed": {arm: (out / f"final/{SEEDS[0]}/{arm}/model.joblib").stat().st_size for arm in ARMS},
    }
    summary["reproduction"] = reproduction(out)
    summary["v5_artifact_check"] = v5_artifact_check(out, data, adopt_protocol["partitions"][0])
    after = protected_state()
    write(out / "protected_after.json", after)
    summary["protected_unchanged"] = before == after
    summary["sources_unchanged"] = all(sha(ROOT / r) == protocol["source_hashes"][r] for r in SOURCES)
    summary["protocol_unchanged"] = sha(out / "protocol.json") == read(out / "protocol_lock.json")["sha256"]
    summary["finished_at"] = now()
    write(out / "summary.json", summary)
    print(json.dumps({"replace": decision["replace"], "checks": decision["checks"],
                      "reproduction": summary["reproduction"]["all_available_equal"],
                      "protected_unchanged": summary["protected_unchanged"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
