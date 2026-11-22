"""Consistent all-record learning curves and leave-one-person-out evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.learning_curve.run import ROOT, MODELS, SEED, SIZES, make_splits, protected_snapshot, now, read, digest
import joblib
import numpy as np
from src.model_comparison import build_model, scores, write_json
from src.window_pipeline import aligned_predict_proba


def assert_no_leakage(data, train, test):
    if set(train) & set(test) or set(data["groups"][train]) & set(data["groups"][test]):
        raise ValueError("Training/evaluation person overlap")
    if set(data["record_ids"][train]) & set(data["record_ids"][test]):
        raise ValueError("Training/evaluation recording overlap")
    if set(data["y"][train]) != set(range(7)):
        raise ValueError("Training lacks a posture class")


def verify(out):
    p = read(out / "protocol.json")
    if digest(out / "protocol.json") != read(out / "protocol_lock.json")["sha256"]:
        raise ValueError("Protocol changed")
    if digest(out / "data.npz") != p["cache_sha256"]:
        raise ValueError("Cache changed")
    for relative, sha in p["source_hashes"].items():
        if digest(ROOT / relative) != sha:
            raise ValueError(f"Frozen source changed: {relative}")
    return p


def prepare(out, source, report):
    out.mkdir(parents=True, exist_ok=False)
    before = protected_snapshot()
    write_json(out / "protected_before.json", before)
    previous = read(source / "protocol.json")
    # Reuse the already audited all-record cache only if every source byte still matches.
    for row in read(source / "training_data/staged_sources.json"):
        if digest(row["source_file"]) != row["source_sha256"]:
            raise ValueError("Full-cohort source changed; rebuild current cohort first")
    if digest(ROOT / "research/models/training_label_corrections.json") != previous["correction_manifest_sha256"]:
        raise ValueError("Label correction changed")
    for name in ["src/posture_core.py", "src/window_pipeline.py", "src/app_services.py"]:
        if digest(ROOT / name) != previous["source_hashes"][name]:
            raise ValueError("Dataset construction code changed; rebuild cache first")
    if digest(source / "data.npz") != previous["cache_sha256"]:
        raise ValueError("Original full-cohort cache changed")
    (out / "data.npz").symlink_to(os.path.relpath(source / "data.npz", out))
    data = np.load(out / "data.npz", allow_pickle=False)
    splits = make_splits(data["groups"], data["y"], data["record_ids"])
    for i, subject in enumerate(np.unique(data["groups"])):
        tr = np.flatnonzero(data["groups"] != subject)
        te = np.flatnonzero(data["groups"] == subject)
        splits.append({"id": f"loo_{i:02d}", "kind": "leave_one_subject_out", "seed": SEED,
                       "repeat": 0, "evaluation_subjects": [str(subject)], "evaluation_indices": te.tolist(),
                       "arms": [{"id": "subjects_31", "axis": "subjects", "nominal": 31,
                                 "train_indices": tr.tolist(), "subject_count": len(np.unique(data["groups"][tr])),
                                 "record_count": len(np.unique(data["record_ids"][tr])), "window_count": len(tr)}]})
    for split in splits:
        split.setdefault("kind", "repeated_5fold")
        for arm in split["arms"]:
            assert_no_leakage(data, np.array(arm["train_indices"]), np.array(split["evaluation_indices"]))
    hashes = {}
    for path in [Path(__file__), Path(__file__).with_name("run.py"), ROOT / "src/model_comparison.py",
                 ROOT / "src/window_pipeline.py", ROOT / "src/posture_core.py"]:
        relative = path.relative_to(ROOT)
        dest = out / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest); hashes[str(relative)] = digest(path)
    shutil.copy2(report, out / "source_snapshot/report_before.md")
    protocol = {"started_at": now(), "models": MODELS, "seed": SEED, "source_hashes": hashes,
                "cache_sha256": digest(out / "data.npz"), "cohort": previous["cohort"],
                "full_cohort_source": str(source), "report": str(report), "report_before_sha256": digest(report),
                "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "dirty_state": subprocess.check_output(["git", "status", "--short"], text=True),
                "command": [sys.executable, *sys.argv], "environment": previous["environment"],
                "splits": splits, "sizes": SIZES, "registration_records_included": True,
                "score_policy": "Primary metrics use only held-out people; full32 resubstitution is diagnosis only",
                "forecast": "Same two fixed saturation forms, fit only repeated 5fold 4..25; LOSO31 is prospective-to-the-fit extrapolation check, never used to fit curve",
                "uncertainty": "Paired evaluation-subject bootstrap conditional on fitted models; overlapping training sets are not independent repeats",
                "fresh_external_test": False, "approved": False}
    write_json(out / "protocol.json", protocol)
    write_json(out / "protocol_lock.json", {"sha256": digest(out / "protocol.json"), "locked_at": now()})
    print("PREPARED", len(splits), "split tasks per family; expected 366 fits", flush=True)


def worker(out, family, split_id):
    p = verify(out)
    data = np.load(out / "data.npz", allow_pickle=False)
    split = next(s for s in p["splits"] if s["id"] == split_id)
    test = np.asarray(split["evaluation_indices"])
    for arm in split["arms"]:
        dest = out / "fits" / family / split_id / arm["id"]
        if (dest / "metrics.json").exists():
            m = read(dest / "metrics.json")
            if digest(dest / "model.joblib") != m["model_sha256"] or digest(dest / "predictions.npz") != m["predictions_sha256"]:
                raise ValueError("Completed artifact changed")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        train = np.asarray(arm["train_indices"])
        assert_no_leakage(data, train, test)
        model = build_model(family, p["models"][family], split["seed"])
        kwargs = {}
        if family == "xgboost":
            from sklearn.utils.class_weight import compute_sample_weight
            kwargs["model__sample_weight"] = compute_sample_weight("balanced", data["y"][train])
        started, tick = now(), time.perf_counter()
        model.fit(data["X"][train], data["y"][train], **kwargs)
        elapsed = time.perf_counter() - tick
        np.testing.assert_allclose(model.named_steps["scale"].mean_, data["X"][train].mean(axis=0, dtype=np.float64), atol=1e-10, rtol=1e-10)
        pred = aligned_predict_proba(model, data["X"][test], 7).argmax(1)
        train_pred = aligned_predict_proba(model, data["X"][train], 7).argmax(1)
        joblib.dump({"model": model, "family": family, "protocol_sha256": digest(out / "protocol.json"),
                     "train_subjects": sorted(set(data["groups"][train])), "evaluation_subjects": split["evaluation_subjects"],
                     "approved": False}, dest / "model.joblib", compress=3)
        np.savez_compressed(dest / "predictions.npz", indices=test, truth=data["y"][test], prediction=pred)
        metric = {"family": family, "split": split_id, "arm": arm["id"], "subjects": arm["subject_count"],
                  "records": arm["record_count"], "windows": arm["window_count"], "fit_seconds": elapsed,
                  "evaluation": scores(data["y"][test], pred), "training": scores(data["y"][train], train_pred),
                  "score_kind": "heldout_subjects", "started_at": started, "finished_at": now(),
                  "model_sha256": digest(dest / "model.joblib"), "predictions_sha256": digest(dest / "predictions.npz")}
        write_json(dest / "metrics.json", metric)
        print(f"{family} {split_id} {arm['id']}: heldout F1 {metric['evaluation']['macro_f1']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); out = args.output.resolve()
    if args.worker:
        worker(out, *args.worker); return
    if not args.resume:
        prepare(out, args.source.resolve(), args.report.resolve())
    p = verify(out)
    # Isolated single-thread model workers; distribute long XGBoost work first.
    jobs = [(family, s["id"]) for family in MODELS for s in p["splits"]]
    def launch(job):
        family, split = job
        logs = out / "logs"; logs.mkdir(exist_ok=True)
        with (logs / f"{family}_{split}.log").open("a") as log:
            subprocess.run([sys.executable, "-m", "experiments.learning_curve.integrated", "--output", str(out),
                            "--worker", family, split], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        return job
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            print("DONE", *future.result(), flush=True)
    if protected_snapshot() != read(out / "protected_before.json"):
        raise ValueError("Original data/model inventory changed; inspect before completing")
    write_json(out / "training_complete.json", {"finished_at": now(), "fits": len(list((out / "fits").rglob("metrics.json"))),
        "protected_data_and_models_unchanged": True})


if __name__ == "__main__":
    main()
