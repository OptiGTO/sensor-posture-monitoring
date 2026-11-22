"""Bounded, preregistered recipe comparison with a locked historical holdout.

Use separate family worker processes so optional native model libraries do not
share initialization state. This never approves or activates a runtime model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import joblib
import numpy as np
from src.model_comparison import scores, write_json
from src.model_catalog import model_input
from src.posture_core import CLASS_NAMES
from src.training_methods import method_candidates, build_method_model, fit_method_model, METHOD_LABELS
from src.window_pipeline import aligned_predict_proba

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260921
PLAN = {
    "rbf_svm": ["standard", "regularized", "relative", "selected", "subject_balanced"],
    "random_forest": ["standard", "regularized"],
    "extra_trees": ["standard", "regularized"],
    "xgboost": ["standard", "regularized", "relative", "selected", "subject_balanced"],
    "cnn1d": ["standard", "cnn_augmentation"],
}
CRITERIA = {"minimum_development_mean_delta": .01,
    "minimum_positive_development_splits": 2, "minimum_holdout_delta": .01,
    "maximum_per_class_f1_drop": .05,
    "uncertainty": "paired subject bootstrap descriptive; no population superiority or multiplicity-adjusted claim",
    "ui_scope": "observed retrospective improvement only; supported family and 3-second window; normal model approval remains required"}


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lock_protocol(out):
    if (out / "protocol.json").exists():
        raise ValueError("Protocol already exists; use run/resume without rewriting it")
    dataset = read(out / "dataset.json")
    sources = [Path(__file__), Path(__file__).with_name("prepare.py"),
        ROOT / "src/training_methods.py", ROOT / "src/extended_models.py",
        ROOT / "src/model_comparison.py", ROOT / "src/model_catalog.py",
        ROOT / "src/posture_core.py", ROOT / "src/window_pipeline.py"]
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = out / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = digest(source)
    candidates = {family: {method: method_candidates(family, method) for method in methods}
                  for family, methods in PLAN.items()}
    p = {"schema_version": 1, "started_at": now(), "seed": SEED,
        "candidates": candidates, "criteria": CRITERIA, "dataset_sha256": digest(out / "dataset.json"),
        "cache_sha256": digest(out / "data.npz"), "source_hashes": source_hashes,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "command": [sys.executable, *sys.argv],
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()},
        "selection": "Maximize mean validation Macro F1 across three preregistered development splits, ties worst-class F1 then fixed candidate order; compare against equally tuned standard family",
        "final_refit": "All development subjects; neural maximum epochs fixed to median development best epochs, preserving existing minimum-training-loss checkpoint behavior; no holdout early stopping",
        "final_evaluation": "Historical six-subject holdout reused at project level; selection locked before its predictions; only development-qualified methods evaluated",
        "activation": "Never change active model or approvals", "window_seconds": 3,
        "dataset_partitions": dataset["partitions"]}
    write_json(out / "protocol.json", p)
    write_json(out / "protocol_lock.json", {"sha256": digest(out / "protocol.json"), "locked_at": now()})
    print("PROTOCOL LOCKED", sum(len(grid) for methods in candidates.values() for grid in methods.values()) * 3, "development fits", flush=True)


def verify(out):
    p = read(out / "protocol.json")
    if digest(out / "protocol.json") != read(out / "protocol_lock.json")["sha256"]:
        raise ValueError("Protocol changed")
    if digest(out / "data.npz") != p["cache_sha256"] or digest(out / "dataset.json") != p["dataset_sha256"]:
        raise ValueError("Frozen input changed")
    for source, expected in p["source_hashes"].items():
        if digest(ROOT / source) != expected:
            raise ValueError(f"Frozen model source changed: {source}")
    return p


def inputs(data, family):
    return model_input(family, data["seq"], data["X"])


def partitions(p):
    value = p["dataset_partitions"]
    return list(value.values()) if isinstance(value, dict) else value


def train_eval_indices(data, part):
    train = np.flatnonzero(np.isin(data["groups"], part["train"]))
    valid = np.flatnonzero(np.isin(data["groups"], part["validation"]))
    test = part.get("test", part.get("final_evaluation"))
    if set(data["groups"][train]) & (set(data["groups"][valid]) | set(test)):
        raise ValueError("Subject overlap")
    if set(data["record_ids"][train]) & set(data["record_ids"][valid]):
        raise ValueError("Recording overlap")
    return train, valid


def development_worker(out, family):
    p = verify(out)
    data = dict(np.load(out / "data.npz", allow_pickle=False))
    X, y = inputs(data, family), data["y"]
    for method, grid in p["candidates"][family].items():
        for part in partitions(p):
            train, valid = train_eval_indices(data, part)
            dest = out / "development" / family / method / part["id"]
            dest.mkdir(parents=True, exist_ok=True)
            for index, params in enumerate(grid):
                record = dest / f"{index:02d}.json"
                prediction_file = dest / f"{index:02d}.npz"
                if record.exists():
                    row = read(record)
                    if digest(prediction_file) != row["prediction_sha256"]:
                        raise ValueError("Completed predictions changed")
                    continue
                started, tick = now(), time.perf_counter()
                model = build_method_model(family, params, SEED, method)
                fit_method_model(model, family, method, X[train], y[train],
                    groups=data["groups"][train], validation_data=(X[valid], y[valid]))
                probability = aligned_predict_proba(model, X[valid], 7)
                prediction = probability.argmax(1)
                np.savez_compressed(prediction_file, indices=valid, truth=y[valid],
                    prediction=prediction, probability=probability, groups=data["groups"][valid])
                row = {"family": family, "method": method, "split": part["id"], "candidate_index": index,
                    "params": params, "metrics": scores(y[valid], prediction),
                    "per_subject": {s: scores(y[valid][data["groups"][valid] == s], prediction[data["groups"][valid] == s])
                                    for s in sorted(set(data["groups"][valid]))},
                    "training_details": getattr(model, "training_details_", {}),
                    "started_at": started, "finished_at": now(), "seconds": time.perf_counter() - tick,
                    "prediction_sha256": digest(prediction_file), "protocol_sha256": digest(out / "protocol.json")}
                write_json(record, row)
            print(f"DEV {family}/{method}/{part['id']} completed {len(grid)} candidates", flush=True)


def dispatch(out, phase, families):
    def one(family):
        destination = out / "logs"
        destination.mkdir(exist_ok=True)
        with (destination / f"{phase}_{family}.log").open("a", encoding="utf-8") as log:
            done = subprocess.run([sys.executable, "-m", "experiments.improvement_retest.run",
                "--output", str(out), "--phase", phase + "-worker", "--family", family],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if done.returncode:
            raise RuntimeError(f"{family} failed; inspect {destination / f'{phase}_{family}.log'}")
        print(f"{phase.upper()} {family} complete", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(one, family) for family in families]
        for future in as_completed(futures):
            future.result()


def select(out):
    p = verify(out)
    if (out / "selection_lock.json").exists():
        raise ValueError("Selection is already locked")
    selected = {}
    for family, methods in p["candidates"].items():
        for method, grid in methods.items():
            candidates = []
            for index, params in enumerate(grid):
                rows = [read(out / "development" / family / method / part["id"] / f"{index:02d}.json") for part in partitions(p)]
                f1 = [row["metrics"]["macro_f1"] for row in rows]
                worst = [row["metrics"]["worst_class_f1"] for row in rows]
                epoch = [row["training_details"].get("best_epoch") for row in rows]
                candidates.append({"candidate_index": index, "params": params, "development_f1": f1,
                    "mean_macro_f1": float(np.mean(f1)), "mean_worst_f1": float(np.mean(worst)),
                    "refit_epochs": int(np.median(epoch)) if all(e is not None for e in epoch) else None})
            best = max(candidates, key=lambda row: (row["mean_macro_f1"], row["mean_worst_f1"], -row["candidate_index"]))
            selected[f"{family}/{method}"] = {"family": family, "method": method, **best}
    for row in selected.values():
        baseline = selected[f"{row['family']}/standard"]
        deltas = np.array(row["development_f1"]) - baseline["development_f1"]
        row["development_deltas"] = deltas.tolist()
        row["mean_development_delta"] = float(deltas.mean())
        row["development_qualified"] = bool(row["method"] != "standard" and deltas.mean() >= CRITERIA["minimum_development_mean_delta"]
            and int(np.sum(deltas > 0)) >= CRITERIA["minimum_positive_development_splits"])
    lock = {"locked_at": now(), "protocol_sha256": digest(out / "protocol.json"),
            "heldout_predictions_exist": False, "selected": selected}
    if (out / "final").exists():
        raise ValueError("Final artifacts already exist before selection")
    write_json(out / "selection_lock.json", lock)
    write_json(out / "selection_lock.sha256.json", {"sha256": digest(out / "selection_lock.json")})
    for key, row in selected.items():
        print(key, f"dev={row['mean_macro_f1']:.4f} delta={row['mean_development_delta']:+.4f} qualified={row['development_qualified']}", flush=True)


def load_selection(out):
    if digest(out / "selection_lock.json") != read(out / "selection_lock.sha256.json")["sha256"]:
        raise ValueError("Selection lock changed")
    return read(out / "selection_lock.json")


def final_worker(out, family):
    p = verify(out)
    lock = load_selection(out)
    data = dict(np.load(out / "data.npz", allow_pickle=False))
    part = partitions(p)[0]
    holdout = part.get("test", part.get("final_evaluation"))
    test = np.flatnonzero(np.isin(data["groups"], holdout))
    train = np.flatnonzero(~np.isin(data["groups"], holdout))
    X, y = inputs(data, family), data["y"]
    for row in lock["selected"].values():
        if row["family"] != family or (row["method"] != "standard" and not row["development_qualified"]):
            continue
        method, params = row["method"], dict(row["params"])
        if row["refit_epochs"] is not None:
            params["epochs"] = row["refit_epochs"]
        dest = out / "final" / family / method
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "metrics.json").exists():
            saved = read(dest / "metrics.json")
            if digest(dest / "model.joblib") != saved["model_sha256"] or digest(dest / "predictions.npz") != saved["prediction_sha256"]:
                raise ValueError("Completed final artifacts changed")
            continue
        started, tick = now(), time.perf_counter()
        model = build_method_model(family, params, SEED, method)
        fit_method_model(model, family, method, X[train], y[train], groups=data["groups"][train])
        training_seconds = time.perf_counter() - tick
        model_file = dest / "model.joblib"
        joblib.dump({"model": model, "family": family, "training_method": method,
            "params": params, "approved": False, "protocol_sha256": digest(out / "protocol.json"),
            "selection_sha256": digest(out / "selection_lock.json")}, model_file, compress=3)
        probability = aligned_predict_proba(model, X[test], 7)
        restored = joblib.load(model_file)["model"]
        np.testing.assert_allclose(aligned_predict_proba(restored, X[test], 7), probability, rtol=1e-9, atol=1e-10)
        prediction = probability.argmax(1)
        np.savez_compressed(dest / "predictions.npz", truth=y[test], prediction=prediction,
            probability=probability, indices=test, groups=data["groups"][test], record_ids=data["record_ids"][test])
        elapsed = []
        for _ in range(20):
            tick = time.perf_counter()
            aligned_predict_proba(model, model_input(family, data["seq"][test[:1]]), 7)
            elapsed.append((time.perf_counter() - tick) * 1000)
        write_json(dest / "metrics.json", {"family": family, "method": method, "params": params,
            "metrics": scores(y[test], prediction), "training_seconds": training_seconds,
            "inference_ms_median": float(np.median(elapsed)), "inference_ms_p95": float(np.percentile(elapsed, 95)),
            "per_subject": {s: scores(y[test][data["groups"][test] == s], prediction[data["groups"][test] == s]) for s in sorted(set(data["groups"][test]))},
            "model_sha256": digest(model_file), "model_size_bytes": model_file.stat().st_size,
            "prediction_sha256": digest(dest / "predictions.npz"), "prediction_parity": True,
            "training_details": getattr(model, "training_details_", {}), "started_at": started, "finished_at": now(),
            "training_subjects": sorted(set(data["groups"][train])), "evaluation_subjects": sorted(set(data["groups"][test]))})
        print(f"FINAL {family}/{method} F1={scores(y[test], prediction)['macro_f1']:.4f}", flush=True)


def f1_from_confusion(matrix):
    diag = matrix.diagonal(axis1=-2, axis2=-1)
    denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    return np.divide(2 * diag, denominator, out=np.zeros_like(diag, dtype=float), where=denominator != 0).mean(axis=-1)


def paired_interval(truth, baseline, candidate, groups, seed=SEED, repetitions=4000):
    subjects = np.unique(groups)
    def matrices(prediction):
        return np.array([np.bincount(7 * truth[groups == s] + prediction[groups == s], minlength=49).reshape(7, 7) for s in subjects])
    sampled = np.random.default_rng(seed).integers(0, len(subjects), (repetitions, len(subjects)))
    deltas = f1_from_confusion(matrices(candidate)[sampled].sum(1)) - f1_from_confusion(matrices(baseline)[sampled].sum(1))
    return np.quantile(deltas, [.025, .975]).tolist()


def summarize(out):
    p = verify(out)
    lock = load_selection(out)
    summary = {"protocol_sha256": digest(out / "protocol.json"), "selection_sha256": digest(out / "selection_lock.json"),
        "finished_at": now(), "criteria": CRITERIA, "comparisons": [], "enabled_methods": []}
    for key, row in lock["selected"].items():
        if row["method"] == "standard":
            continue
        result = {**row, "improved": False, "heldout_evaluated": False}
        if row["development_qualified"]:
            base_dir = out / "final" / row["family"] / "standard"
            current_dir = out / "final" / row["family"] / row["method"]
            b, c = read(base_dir / "metrics.json"), read(current_dir / "metrics.json")
            bp, cp = np.load(base_dir / "predictions.npz"), np.load(current_dir / "predictions.npz")
            np.testing.assert_array_equal(bp["indices"], cp["indices"])
            np.testing.assert_array_equal(bp["truth"], cp["truth"])
            delta = c["metrics"]["macro_f1"] - b["metrics"]["macro_f1"]
            class_deltas = {label: c["metrics"]["per_class_f1"][label] - b["metrics"]["per_class_f1"][label] for label in CLASS_NAMES}
            result.update(heldout_evaluated=True, baseline=b["metrics"], candidate=c["metrics"],
                heldout_delta=delta, per_class_deltas=class_deltas,
                paired_subject_95_interval=paired_interval(bp["truth"], bp["prediction"], cp["prediction"], bp["groups"]),
                baseline_inference_ms=b["inference_ms_median"], candidate_inference_ms=c["inference_ms_median"],
                improved=bool(delta >= CRITERIA["minimum_holdout_delta"] and min(class_deltas.values()) >= -CRITERIA["maximum_per_class_f1_drop"]))
        summary["comparisons"].append(result)
        if result["improved"]:
            summary["enabled_methods"].append({"family": row["family"], "method": row["method"]})
    write_json(out / "summary.json", summary)
    print(json.dumps({"enabled_methods": summary["enabled_methods"], "comparison_count": len(summary["comparisons"])}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=["lock", "development", "development-worker", "select", "final", "final-worker", "summarize"])
    parser.add_argument("--family", choices=list(PLAN))
    args = parser.parse_args()
    out = args.output.resolve()
    if args.phase == "lock": lock_protocol(out)
    elif args.phase == "development": verify(out); dispatch(out, "development", PLAN)
    elif args.phase == "development-worker": development_worker(out, args.family)
    elif args.phase == "select": select(out)
    elif args.phase == "final": verify(out); load_selection(out); dispatch(out, "final", PLAN)
    elif args.phase == "final-worker": final_worker(out, args.family)
    else: summarize(out)


if __name__ == "__main__":
    main()
