"""Preregistered sensor-gain ablation; never writes raw data or active models."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import joblib
import numpy as np
from experiments.improvement_retest.prepare import (
    ROOT, SEED, now, digest, read, write_json, protected_snapshot, sensor_digest,
)
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, normalize_text, read_sensor_csv
from src.model_comparison import build_model, scores
from src.window_pipeline import WindowSpec, segment_sensor_frame, extract_window_features, aligned_predict_proba

RANGES = np.array([[1.7, 2.0], [2.7, 2.9], [3.4, 3.8], [2.9, 3.5],
                   [3.0, 3.4], [3.2, 3.4], [2.9, 3.0], [3.4, 3.6],
                   [2.9, 3.1], [2.7, 2.9], [3.0, 3.3], [3.0, 3.5],
                   [2.8, 3.0], [2.7, 2.8], [2.9, 3.1], [3.0, 3.2]])
MODELS = {
    "rbf_svm": {"C": 1.0, "gamma": 0.0005},
    "xgboost": {"n_estimators": 200, "max_depth": 3, "learning_rate": .05,
                "min_child_weight": 3, "reg_lambda": 5., "subsample": .8, "colsample_bytree": .8},
    "random_forest": {"n_estimators": 200, "min_samples_leaf": 1, "max_features": "sqrt"},
}


def gains():
    mids = RANGES.mean(axis=1)
    return np.median(mids) / mids


def calibrate(sequence, factors):
    x, g = np.asarray(sequence), np.asarray(factors)
    if x.shape[-1] != 16 or g.shape != (16,) or not np.isfinite(x).all() or not np.isfinite(g).all() or np.any(g <= 0):
        raise ValueError("Expected finite row-major 16-channel inputs and positive gains")
    return (x * g).astype(np.float32)


def condition_id(seconds, arm, family):
    return f"{seconds}s_{arm}_{family}"


def partition_indices(data, part, final=False):
    train_people = part["train"] + part["validation"] if final else part["train"]
    eval_people = part["test"] if final else part["validation"]
    if set(train_people) & set(eval_people):
        raise ValueError("Subject leakage")
    train = np.flatnonzero(np.isin(data["groups"], train_people))
    test = np.flatnonzero(np.isin(data["groups"], eval_people))
    if set(data["record_ids"][train]) & set(data["record_ids"][test]):
        raise ValueError("Recording leakage")
    if set(data["y"][train]) != set(range(7)) or set(data["y"][test]) != set(range(7)):
        raise ValueError("Missing classes")
    return train, test


def prepare(out, source):
    out.mkdir(parents=True, exist_ok=False)
    started = now()
    write_json(out / "protected_before.json", protected_snapshot())
    old = read(source / "dataset.json")
    if digest(source / "dataset.json") != read(source / "dataset_lock.json")["sha256"]:
        raise ValueError("Historical dataset manifest changed")
    if digest(source / "data.npz") != old["cache_sha256"]:
        raise ValueError("Historical cache changed")
    for name in ("dataset.json", "dataset_lock.json"):
        shutil.copy2(source / name, out / ("origin_" + name))
    frames, raw_hashes, sensor_hashes = {}, set(), set()
    for record in old["records"]:
        raw = ROOT / "data" / record["source_relative_path"]
        staged = source / "training_data" / record["staged_file"]
        if digest(raw) != record["sha256"] or digest(staged) != record["staged_sha256"]:
            raise ValueError(f"Frozen recording changed: {raw}")
        frame = read_sensor_csv(staged)
        sh = sensor_digest(frame)
        if sh != record["sensor_sha256"] or sh in sensor_hashes or record["sha256"] in raw_hashes:
            raise ValueError("Duplicate or changed sensor sequence")
        raw_hashes.add(record["sha256"]); sensor_hashes.add(sh)
        frames[record["source_relative_path"]] = frame
    cache_hashes, counts = {}, {}
    for seconds in (2, 3):
        seq, labels, people, ids, starts = [], [], [], [], []
        for record in old["records"]:
            windows, meta = segment_sensor_frame(frames[record["source_relative_path"]], WindowSpec(seconds))
            seq.extend(windows); labels.extend([CLASS_NAMES.index(record["label"])] * len(windows))
            people.extend([record["subject"]] * len(windows))
            ids.extend([normalize_text(record["source_relative_path"])] * len(windows))
            starts.extend(meta.start_s.tolist())
        seq = np.asarray(seq)
        data = {"seq": seq, "y": np.asarray(labels), "groups": np.asarray(people),
                "record_ids": np.asarray(ids), "start_s": np.asarray(starts)}
        data["raw"] = extract_window_features(seq)
        data["calibrated"] = extract_window_features(calibrate(seq, gains()))
        if seconds == 3:
            previous = np.load(source / "data.npz", allow_pickle=False)
            for key in ("seq", "y", "groups", "record_ids", "start_s"):
                np.testing.assert_array_equal(data[key], previous[key])
            np.testing.assert_array_equal(data["raw"], previous["X"])
        for arm in ("raw", "calibrated"):
            if data[arm].shape != (len(labels), 596) or not np.isfinite(data[arm]).all():
                raise ValueError("Invalid features")
        for part in old["partitions"]:
            partition_indices(data, part)
            partition_indices(data, part, final=True)
        path = out / f"data_{seconds}s.npz"
        np.savez_compressed(path, **data); cache_hashes[path.name] = digest(path)
        counts[str(seconds)] = {"subjects": len(set(people)), "records": len(set(ids)), "windows": len(labels),
            "class_windows": np.bincount(labels, minlength=7).tolist(), "partitions": {p["id"]: {
                stage: int(np.isin(data["groups"], p[stage]).sum()) for stage in ("train", "validation", "test")}
                for p in old["partitions"]}}
        print("PREPARED", seconds, "seconds", len(labels), "windows", flush=True)
    snap = out / "source_snapshot"
    sources = [*ROOT.glob("*.py"), *(ROOT / "src").rglob("*.py"),
               Path(__file__), ROOT / "experiments/improvement_retest/prepare.py", ROOT / "AGENTS.md"]
    source_hashes = {}
    for path in sources:
        rel = path.relative_to(ROOT); dest = snap / rel
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        source_hashes[str(rel)] = digest(dest)
    conditions = [{"id": condition_id(s, a, f), "seconds": s, "arm": a, "family": f}
                  for s in (2, 3) for a in ("raw", "calibrated") for f in MODELS]
    p = {"started_at": started, "locked_at": now(), "root": str(ROOT), "origin": str(source),
        "origin_dataset_sha256": digest(source / "dataset.json"), "cache_hashes": cache_hashes,
        "source_hashes": source_hashes, "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], text=True),
        "command": [sys.executable, *sys.argv], "seed": SEED, "class_names": CLASS_NAMES,
        "sensor_columns": SENSOR_COLUMNS, "ranges": RANGES.tolist(), "midpoints": RANGES.mean(1).tolist(),
        "reference": float(np.median(RANGES.mean(1))), "gains": gains().tolist(),
        "calibration": "raw signal times median(midpoints)/sensor midpoint before feature extraction; no offset, clipping or nonlinear fit",
        "calibration_evidence": "User supplied finger maximal-pressure ranges repeated 3-4 times; equal force unverified; unit and acquisition date unknown; range midpoints are not sample means",
        "hardware_assumption": "Same sensor identities and row-major mapping as all historical recordings; not verified by hardware metadata",
        "models": MODELS, "conditions": conditions, "counts": counts, "partitions": old["partitions"],
        "sampling_rate_hz": 12, "stride": "equal to window duration", "min_tail_fraction": .95,
        "inference": "train-only StandardScaler and predict_proba argmax; SVM probability calibration uses only training data",
        "tuning": "None; fixed prior learning-curve parameters isolate gain correction effects",
        "selection": "Rank conditions by mean development Macro F1 before any final predictions; report all preregistered conditions",
        "success": {"development_mean_delta_min": .01, "development_positive_splits_min": 2,
                    "historical_test_delta_min": .01, "maximum_class_f1_drop": .05},
        "uncertainty": "2000 paired subject-cluster bootstrap draws; descriptive, conditional on trained models; no multiple-comparison correction",
        "final_fit": "26 development subjects, evaluate historical fixed six people once per condition",
        "registration_policy": "Reuse historical 692-record cohort, including its 33 reserved-record exclusions; explicitly registered records not restored",
        "fresh_external_validation": False, "approved": False, "hardware_checked": False,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "packages": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")}}}
    write_json(out / "protocol.json", p)
    write_json(out / "protocol_lock.json", {"sha256": digest(out / "protocol.json"), "locked_at": now()})
    verify(out)


def verify(out):
    p = read(out / "protocol.json")
    if digest(out / "protocol.json") != read(out / "protocol_lock.json")["sha256"]:
        raise ValueError("Protocol changed")
    for name, sha in p["cache_hashes"].items():
        if digest(out / name) != sha: raise ValueError("Cache changed")
    for name, sha in p["source_hashes"].items():
        if digest(out / "source_snapshot" / name) != sha: raise ValueError("Frozen source changed")
    return p


def worker(out, cid, phase):
    p = verify(out); c = next(c for c in p["conditions"] if c["id"] == cid)
    if phase == "final":
        lock = read(out / "selection_lock.json")
        if lock["protocol_sha256"] != digest(out / "protocol.json"): raise ValueError("Selection protocol mismatch")
        for name, sha in lock["development_artifacts"].items():
            if digest(out / name) != sha: raise ValueError("Development evidence changed after selection")
    data = np.load(out / f"data_{c['seconds']}s.npz", allow_pickle=False)
    parts = p["partitions"][:1] if phase == "final" else p["partitions"]
    for part in parts:
        dest = out / phase / cid / ("holdout" if phase == "final" else part["id"])
        if (dest / "metrics.json").exists():
            m = read(dest / "metrics.json")
            for name in ("model.joblib", "predictions.npz"):
                if digest(dest / name) != m["artifacts"][name]: raise ValueError("Completed fit changed")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        tr, te = partition_indices(data, part, final=phase == "final")
        model = build_model(c["family"], p["models"][c["family"]], part["seed"])
        kwargs = {}
        if c["family"] == "xgboost":
            from sklearn.utils.class_weight import compute_sample_weight
            kwargs["model__sample_weight"] = compute_sample_weight("balanced", data["y"][tr])
        started, tick = now(), time.perf_counter()
        model.fit(data[c["arm"]][tr], data["y"][tr], **kwargs)
        fit_seconds = time.perf_counter() - tick
        np.testing.assert_allclose(model.named_steps["scale"].mean_, data[c["arm"]][tr].mean(0, dtype=np.float64), rtol=1e-10, atol=1e-10)
        tick = time.perf_counter(); prob = aligned_predict_proba(model, data[c["arm"]][te], 7)
        latency_ms = (time.perf_counter() - tick) * 1000 / len(te)
        prediction = prob.argmax(1)
        if not np.isfinite(prob).all(): raise ValueError("Invalid probabilities")
        np.testing.assert_allclose(prob.sum(1), 1, atol=1e-6)
        factor = p["gains"] if c["arm"] == "calibrated" else [1.] * 16
        joblib.dump({"model": model, "gains": factor, "condition": c, "label_names": CLASS_NAMES,
            "sensor_columns": SENSOR_COLUMNS, "sampling_rate_hz": 12, "window_seconds": c["seconds"],
            "preprocessing": "multiply resampled raw sequences by gains then extract_window_features",
            "protocol_sha256": digest(out / "protocol.json"), "approved": False,
            "train_subjects": sorted(set(data["groups"][tr])), "test_subjects": sorted(set(data["groups"][te]))},
            dest / "model.joblib", compress=3)
        bundle = joblib.load(dest / "model.joblib")
        replay_x = extract_window_features(calibrate(data["seq"][te], bundle["gains"]))
        np.testing.assert_array_equal(replay_x, data[c["arm"]][te])
        np.testing.assert_allclose(aligned_predict_proba(bundle["model"], replay_x, 7), prob, atol=1e-12)
        np.savez_compressed(dest / "predictions.npz", indices=te, truth=data["y"][te], prediction=prediction,
            probability=prob, groups=data["groups"][te], record_ids=data["record_ids"][te])
        metric = {"condition": c, "phase": phase, "partition": part["id"], "started_at": started,
            "finished_at": now(), "training_windows": len(tr), "evaluation_windows": len(te),
            "training_subjects": len(set(data["groups"][tr])), "evaluation_subjects": len(set(data["groups"][te])),
            "fit_seconds": fit_seconds, "batch_prediction_ms_per_window": latency_ms,
            "evaluation": scores(data["y"][te], prediction), "saved_raw_sequence_replay_parity": True,
            "per_subject": {str(s): scores(data["y"][te][data["groups"][te] == s], prediction[data["groups"][te] == s])
                            for s in np.unique(data["groups"][te])},
            "artifacts": {name: digest(dest / name) for name in ("model.joblib", "predictions.npz")}}
        write_json(dest / "metrics.json", metric)
        print(phase, cid, part["id"], round(metric["evaluation"]["macro_f1"], 5), flush=True)


def launch_phase(out, phase):
    p = verify(out)
    def launch(c):
        logs = out / "logs"; logs.mkdir(exist_ok=True)
        with (logs / f"{phase}_{c['id']}.log").open("a") as stream:
            subprocess.run([sys.executable, "-m", "experiments.sensor_gain.run", "--output", str(out),
                "--phase", phase, "--worker", c["id"]], cwd=out / "source_snapshot", stdout=stream,
                stderr=subprocess.STDOUT, check=True)
        return c["id"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(launch, c) for c in p["conditions"]]):
            print("DONE", phase, future.result(), flush=True)


def lock_selection(out):
    p = verify(out); rows, artifacts = [], {}
    for c in p["conditions"]:
        metrics = []
        for part in p["partitions"]:
            dest = out / "development" / c["id"] / part["id"]
            m = read(dest / "metrics.json"); metrics.append(m["evaluation"]["macro_f1"])
            for name in ("metrics.json", "model.joblib", "predictions.npz"):
                path = dest / name; artifacts[str(path.relative_to(out))] = digest(path)
        rows.append({**c, "development_f1": metrics, "mean_development_f1": float(np.mean(metrics))})
    ranked = sorted(rows, key=lambda r: (-r["mean_development_f1"], r["id"]))
    write_json(out / "selection_lock.json", {"locked_at": now(), "protocol_sha256": digest(out / "protocol.json"),
        "development_artifacts": artifacts, "ranking": ranked, "winner": ranked[0]["id"],
        "note": "All twelve preregistered conditions will be evaluated without further tuning; winner fixed using development only"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "research/results/2026-09-21_200011_improvement_retest")
    parser.add_argument("--phase", choices=("all", "prepare", "development", "final"), default="all")
    parser.add_argument("--worker")
    args = parser.parse_args(); out = args.output.resolve()
    if args.worker:
        worker(out, args.worker, args.phase); return
    if args.phase in ("all", "prepare"):
        prepare(out, args.source.resolve())
    if args.phase in ("all", "development"):
        launch_phase(out, "development"); lock_selection(out)
    if args.phase in ("all", "final"):
        launch_phase(out, "final")
        if protected_snapshot() != read(out / "protected_before.json"):
            raise ValueError("Protected data/models changed")
        write_json(out / "training_complete.json", {"finished_at": now(), "fits": 48,
            "data_and_existing_models_unchanged": True, "approved": False})


if __name__ == "__main__":
    main()
