"""Frozen E2 offset/gain ablation; development gates precede historical evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import joblib
import numpy as np

ARMS = ("P0", "P1", "P2", "P3")
NAMES = {"rbf_svm": "RBF SVM", "xgboost": "XGBoost", "random_forest": "Random Forest"}


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def transform(sequence, arm, baseline, gains):
    x, b, g = np.asarray(sequence), np.asarray(baseline, dtype=np.float64), np.asarray(gains, dtype=np.float64)
    if (arm not in ARMS or x.ndim != 3 or x.shape[1:] != (36, 16)
            or b.shape != (16,) or g.shape != (16,) or not np.isfinite(x).all()
            or not np.isfinite(b).all() or not np.isfinite(g).all() or np.any(g <= 0)):
        raise ValueError("Expected finite 3-second sequences and ordered 16-channel calibration")
    if arm == "P0":
        return x.astype(np.float32, copy=True)
    result = x.astype(np.float64, copy=True)
    if arm in ("P1", "P3"):
        result -= b
    if arm in ("P2", "P3"):
        result *= g
    return result.astype(np.float32)


def gate(candidate, reference):
    """Use only corresponding development splits, including every class guardrail."""
    if len(candidate) != 3 or len(reference) != 3:
        raise ValueError("Exactly three development splits required")
    delta = [c["macro_f1"] - r["macro_f1"] for c, r in zip(candidate, reference)]
    class_delta = [[c["per_class_f1"][name] - r["per_class_f1"][name]
                    for name in r["per_class_f1"]] for c, r in zip(candidate, reference)]
    worst = float(np.min(class_delta))
    mean = float(np.mean(delta))
    positive = sum(d > 0 for d in delta)
    return {"split_deltas": delta, "mean_delta": mean, "positive_splits": positive,
            "worst_split_class_delta": worst, "per_split_class_deltas": class_delta,
            "passed": bool(mean >= .01 - 1e-12 and positive >= 2 and worst >= -.05 - 1e-12)}


def bootstrap(root, out, source):
    """Copy verified previous feature/model code, without changing any shared module."""
    from experiments.improvement_retest.prepare import protected_snapshot
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protected_before.json", protected_snapshot(root))
    historical = read(source / "protocol.json")
    if digest(source / "protocol.json") != read(source / "protocol_lock.json")["sha256"]:
        raise ValueError("Historical protocol changed")
    snap = out / "source_snapshot"
    for name, sha in historical["source_hashes"].items():
        src = source / "source_snapshot" / name
        if digest(src) != sha:
            raise ValueError(f"Historical frozen source changed: {name}")
        dst = snap / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if digest(dst) != sha:
            raise ValueError("Snapshot copy mismatch")
    for src in sorted(Path(__file__).parent.glob("*.py")):
        dst = snap / "experiments/empty_seat_e2" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if digest(src) != digest(dst):
            raise ValueError("Experiment implementation changed during snapshot")
    prior_test = root / "experiments/sensor_gain/test_run.py"
    test_dst = snap / "experiments/sensor_gain/test_run.py"
    shutil.copy2(prior_test, test_dst)
    if digest(prior_test) != digest(test_dst):
        raise ValueError("Prior contract tests changed during snapshot")
    plan = root / "docs/08_experiments_and_results/2026-09-23_standard_empty_seat_experiment_plan.md"
    dst = snap / "docs/08_experiments_and_results" / plan.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan, dst)
    write(out / "bootstrap.json", {"started_at": now(), "root": str(root), "source": str(source),
        "source_protocol_sha256": digest(source / "protocol.json"),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "command": [sys.executable, *sys.argv], "plan_sha256": digest(dst),
        "source_policy": "Reuse verified historical snapshot for feature and model definitions; new E2 runner only",
        "source_hashes": {str(p.relative_to(snap)): digest(p) for p in sorted(snap.rglob('*')) if p.is_file()}})
    env = dict(os.environ, NUMBA_CACHE_DIR=str(out / "runtime_cache/numba"),
               MPLCONFIGDIR=str(out / "runtime_cache/matplotlib"))
    subprocess.run([sys.executable, "-B", "-m", "experiments.empty_seat_e2.run", "--output", str(out),
                    "--frozen"], cwd=snap, env=env, check=True)


def prepare(out):
    from experiments.improvement_retest.prepare import sensor_digest
    from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, normalize_text, read_sensor_csv
    from src.window_pipeline import WindowSpec, segment_sensor_frame, extract_window_features
    from experiments.sensor_gain.run import partition_indices
    boot = read(out / "bootstrap.json")
    root, source = Path(boot["root"]), Path(boot["source"])
    previous = read(source / "protocol.json")
    if digest(source / "protocol.json") != boot["source_protocol_sha256"]:
        raise ValueError("Historical protocol drift")
    old = read(source / "origin_dataset.json")
    if digest(source / "origin_dataset.json") != previous["origin_dataset_sha256"]:
        raise ValueError("Historical dataset manifest drift")
    source_cache = source / "data_3s.npz"
    if digest(source_cache) != previous["cache_hashes"][source_cache.name]:
        raise ValueError("Historical cache changed")
    shutil.copy2(source / "origin_dataset.json", out / "origin_dataset.json")
    previous_data = np.load(source_cache, allow_pickle=False)
    origin = Path(previous["origin"])
    if digest(origin / "dataset.json") != previous["origin_dataset_sha256"]:
        raise ValueError("Original manifest drift")
    seq, labels, people, ids, starts = [], [], [], [], []
    raw_hashes, sensor_hashes = set(), set()
    future_reference_count, max_lookahead = 0, 0.0
    for rec in old["records"]:
        raw, staged = root / "data" / rec["source_relative_path"], origin / "training_data" / rec["staged_file"]
        if digest(raw) != rec["sha256"] or digest(staged) != rec["staged_sha256"]:
            raise ValueError(f"Frozen data changed: {raw.name}")
        frame = read_sensor_csv(staged)
        sh = sensor_digest(frame)
        if sh != rec["sensor_sha256"] or sh in sensor_hashes or rec["sha256"] in raw_hashes:
            raise ValueError("Changed or duplicated sensor input")
        sensor_hashes.add(sh); raw_hashes.add(rec["sha256"])
        windows, meta = segment_sensor_frame(frame, WindowSpec(3))
        seq.extend(windows); labels.extend([CLASS_NAMES.index(rec["label"])] * len(windows))
        people.extend([rec["subject"]] * len(windows))
        ids.extend([normalize_text(rec["source_relative_path"])] * len(windows))
        starts.extend(meta.start_s.tolist())
        t = frame.elapsed_s.to_numpy(dtype=float); t -= t[0]
        for start in meta.start_s:
            grid = start + np.arange(36) / 12
            upper = t[np.minimum(np.searchsorted(t, grid, side="left"), len(t) - 1)]
            lookahead = np.maximum(upper - (start + 3), 0)
            future_reference_count += int(np.count_nonzero(lookahead > 1e-10))
            max_lookahead = max(max_lookahead, float(lookahead.max()))
    data = {"seq": np.asarray(seq), "y": np.asarray(labels), "groups": np.asarray(people),
            "record_ids": np.asarray(ids), "start_s": np.asarray(starts)}
    for key, val in data.items():
        np.testing.assert_array_equal(val, previous_data[key])
    files = sorted((root / "data/standard").glob("*.csv"))
    if len(files) != 5:
        raise ValueError("Plan expects exactly five standard files; inspect changed inventory")
    standard = []
    for path in files:
        frame = read_sensor_csv(path)
        if set(frame.label) != {"standard"}:
            raise ValueError("Unexpected standard label")
        x = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
        standard.append({"path": str(path.relative_to(root)), "sha256": digest(path),
            "rows": len(frame), "duration_s": float(frame.elapsed_s.iloc[-1] - frame.elapsed_s.iloc[0]),
            "median": np.median(x, axis=0).tolist(),
            "mad_sigma": (1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)).tolist()})
    b = np.median([r["median"] for r in standard], axis=0)
    g = np.asarray(previous["gains"])
    feature_diagnostics = {}
    dev_mask = ~np.isin(data["groups"], old["partitions"][0]["test"])
    for arm in ARMS:
        transformed = transform(data["seq"], arm, b, g)
        data[arm] = extract_window_features(transformed)
        if data[arm].shape != (len(labels), 596) or not np.isfinite(data[arm]).all():
            raise ValueError("Invalid extracted features")
        feature_diagnostics[arm] = {"development_negative_value_fraction": float((transformed[dev_mask] < 0).mean()),
            "development_min": float(transformed[dev_mask].min()), "development_max": float(transformed[dev_mask].max()),
            "feature_count": data[arm].shape[1]}
        print("PREPARED", arm, len(labels), "windows", flush=True)
    np.testing.assert_array_equal(data["P0"], previous_data["raw"])
    np.testing.assert_array_equal(data["P2"], previous_data["calibrated"])
    for arm in ("P1", "P3"):
        ref = "P0" if arm == "P1" else "P2"
        feature_diagnostics[arm]["development_changed_feature_columns"] = int(np.any(data[arm][dev_mask] != data[ref][dev_mask], axis=0).sum())
    for part in old["partitions"]:
        partition_indices(data, part); partition_indices(data, part, final=True)
    np.savez_compressed(out / "data.npz", **data)
    calibration = {"standard_files": standard, "baseline": b.tolist(), "gains": g.tolist(),
        "sensor_columns": SENSOR_COLUMNS, "baseline_rule": "median of five per-file sensor medians; equal file weight",
        "no_clipping": True, "signed_residuals": True, "hardware_match_verified": False}
    write(out / "calibration.json", calibration)
    conditions = [{"id": f"{a}_{f}", "arm": a, "family": f} for a in ARMS for f in previous["models"]]
    protocol = {**boot, "locked_at": now(), "origin_dataset_sha256": digest(out / "origin_dataset.json"),
        "models": previous["models"], "partitions": old["partitions"], "conditions": conditions,
        "class_names": CLASS_NAMES, "sensor_columns": SENSOR_COLUMNS, "baseline": b.tolist(), "gains": g.tolist(),
        "source_cache_sha256": digest(source_cache), "cache_sha256": digest(out / "data.npz"),
        "calibration_sha256": digest(out / "calibration.json"), "seed_policy": "Each model uses corresponding partition seed, as historical SVM/XGB/RF",
        "window_seconds": 3, "sampling_rate_hz": 12, "min_tail_fraction": .95, "stride_seconds": 3,
        "transform": {"P0": "x", "P1": "x-b", "P2": "g*x", "P3": "g*(x-b)"},
        "precision": "Transform float32 resampled input in float64, then round once to float32 before features",
        "primary_comparisons": [["P1", "P0"], ["P3", "P2"]],
        "selection": {"mean_delta_min": .01, "positive_splits_min": 2, "per_split_max_class_drop": .05,
            "rank": "highest mean development F1 among passing P1/P3; ties P1 before P3, then family ID",
            "no_passing_candidate": "Stop without historical test predictions or final model fits",
            "passing_candidate": "Lock winner, evaluate winner and paired reference once; five leave-one-standard-file sensitivity refits on development only"},
        "counts": {"subjects": len(set(people)), "records": len(set(ids)), "windows": len(labels),
            "class_windows": np.bincount(labels, minlength=7).tolist()},
        "registration_policy": previous["registration_policy"], "feature_diagnostics": feature_diagnostics,
        "causal_input_audit": {"sample_points_using_source_after_nominal_window_end": future_reference_count,
            "maximum_seconds_after_window_end": max_lookahead, "policy": "Preserve original interpolation for parity; no physical streaming timing claim"},
        "environment": {"python": sys.version, "platform": platform.platform(),
            "packages": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")}},
        "fresh_external_validation": False, "hardware_checked": False, "approved": False}
    write(out / "protocol.json", protocol)
    write(out / "protocol_lock.json", {"sha256": digest(out / "protocol.json"), "locked_at": now()})


def verify(out):
    p = read(out / "protocol.json")
    if digest(out / "protocol.json") != read(out / "protocol_lock.json")["sha256"]:
        raise ValueError("Protocol changed")
    for name, key in (("data.npz", "cache_sha256"), ("calibration.json", "calibration_sha256"),
                      ("origin_dataset.json", "origin_dataset_sha256")):
        if digest(out / name) != p[key]:
            raise ValueError(f"Frozen input changed: {name}")
    for name, sha in p["source_hashes"].items():
        if digest(out / "source_snapshot" / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    return p


def audit_score(truth, prediction):
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (truth, prediction), 1)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * np.diag(cm), denominator, out=np.zeros(7), where=denominator != 0)
    return {"macro_f1": float(f1.mean()), "accuracy": float(np.trace(cm) / cm.sum()),
            "per_class_f1": f1.tolist(), "confusion_matrix": cm.tolist()}


def worker(out, cid, phase, omit=None):
    from experiments.sensor_gain.run import partition_indices
    from src.model_comparison import build_model, scores
    from src.window_pipeline import extract_window_features, aligned_predict_proba
    p = verify(out)
    c = next(c for c in p["conditions"] if c["id"] == cid)
    if phase != "development":
        lock = read(out / "selection_lock.json")
        allowed = [lock["winner"], lock["reference"]] if phase == "final" else [lock["winner"]]
        if not lock["winner"] or cid not in allowed or lock["protocol_sha256"] != digest(out / "protocol.json"):
            raise ValueError("Selection gate does not authorize this fit")
        for name, sha in lock["development_artifacts"].items():
            if digest(out / name) != sha:
                raise ValueError("Development evidence changed after selection")
    data = np.load(out / "data.npz", allow_pickle=False)
    baseline = p["baseline"]
    features = data[c["arm"]]
    if phase == "sensitivity":
        medians = [r["median"] for r in read(out / "calibration.json")["standard_files"]]
        baseline = np.median(np.delete(medians, omit, axis=0), axis=0).tolist()
        features = extract_window_features(transform(data["seq"], c["arm"], baseline, p["gains"]))
    parts = p["partitions"][:1] if phase == "final" else p["partitions"]
    for part in parts:
        name = "holdout" if phase == "final" else part["id"]
        dest = out / phase / cid
        if omit is not None:
            dest /= f"omit_{omit+1:03d}"
        dest /= name
        dest.mkdir(parents=True, exist_ok=False)
        tr, te = partition_indices(data, part, final=phase == "final")
        model = build_model(c["family"], p["models"][c["family"]], part["seed"])
        kwargs = {}
        if c["family"] == "xgboost":
            from sklearn.utils.class_weight import compute_sample_weight
            kwargs["model__sample_weight"] = compute_sample_weight("balanced", data["y"][tr])
        started, tick = now(), time.perf_counter()
        model.fit(features[tr], data["y"][tr], **kwargs)
        fit_seconds = time.perf_counter() - tick
        np.testing.assert_allclose(model.named_steps["scale"].mean_, features[tr].mean(0, dtype=np.float64), rtol=1e-10, atol=1e-10)
        tick = time.perf_counter()
        prob = aligned_predict_proba(model, features[te], 7)
        latency = (time.perf_counter() - tick) * 1000 / len(te)
        if not np.isfinite(prob).all():
            raise ValueError("Nonfinite probabilities")
        np.testing.assert_allclose(prob.sum(1), 1, atol=1e-6)
        pred = prob.argmax(1)
        bundle = {"model": model, "baseline": baseline, "gains": p["gains"], "condition": c,
            "label_names": p["class_names"], "sensor_columns": p["sensor_columns"],
            "window_seconds": 3, "sampling_rate_hz": 12, "preprocessing": p["transform"][c["arm"]],
            "runner": "experiments.empty_seat_e2.run.transform", "approved": False,
            "protocol_sha256": digest(out / "protocol.json"),
            "train_subjects": sorted(set(data["groups"][tr])), "evaluation_subjects": sorted(set(data["groups"][te]))}
        joblib.dump(bundle, dest / "model.joblib", compress=3)
        loaded = joblib.load(dest / "model.joblib")
        replay_x = extract_window_features(transform(data["seq"][te], loaded["condition"]["arm"], loaded["baseline"], loaded["gains"]))
        np.testing.assert_array_equal(replay_x, features[te])
        np.testing.assert_allclose(aligned_predict_proba(loaded["model"], replay_x, 7), prob, atol=1e-12)
        np.savez_compressed(dest / "predictions.npz", indices=te, truth=data["y"][te], prediction=pred,
            probability=prob, groups=data["groups"][te], record_ids=data["record_ids"][te])
        score = scores(data["y"][te], pred)
        audit = audit_score(data["y"][te], pred)
        np.testing.assert_allclose(score["macro_f1"], audit["macro_f1"], atol=1e-12)
        np.testing.assert_array_equal(score["confusion_matrix"], audit["confusion_matrix"])
        write(dest / "metrics.json", {"condition": c, "phase": phase, "omit_standard_index": omit,
            "partition": part["id"], "started_at": started, "finished_at": now(),
            "training_subjects": len(set(data["groups"][tr])), "evaluation_subjects": len(set(data["groups"][te])),
            "training_windows": len(tr), "evaluation_windows": len(te), "fit_seconds": fit_seconds,
            "batch_prediction_ms_per_window": latency, "evaluation": score,
            "train_only_scaler_checked": True, "saved_raw_sequence_replay_parity": True,
            "per_subject": {str(s): scores(data["y"][te][data["groups"][te] == s], pred[data["groups"][te] == s])
                            for s in np.unique(data["groups"][te])},
            "artifacts": {n: digest(dest / n) for n in ("model.joblib", "predictions.npz")}})
        print(phase, cid, name, round(score["macro_f1"], 6), flush=True)


def launch(out, jobs):
    def execute(job):
        cid, phase, omit = job
        (out / "logs").mkdir(exist_ok=True)
        name = f"{phase}_{cid}" + (f"_omit_{omit+1:03d}" if omit is not None else "")
        command = [sys.executable, "-B", "-m", "experiments.empty_seat_e2.run", "--output", str(out),
                   "--worker", cid, "--phase", phase]
        if omit is not None:
            command += ["--omit", str(omit)]
        with (out / "logs" / f"{name}.log").open("w") as stream:
            subprocess.run(command, cwd=out / "source_snapshot", stdout=stream, stderr=subprocess.STDOUT, check=True)
        return name
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(execute, j) for j in jobs]):
            print("DONE", future.result(), flush=True)


def lock_selection(out):
    p = verify(out); rows, artifacts, metrics = [], {}, {}
    for c in p["conditions"]:
        values = []
        for part in p["partitions"]:
            dest = out / "development" / c["id"] / part["id"]
            m = read(dest / "metrics.json")
            values.append(m["evaluation"])
            for name in ("metrics.json", "model.joblib", "predictions.npz"):
                path = dest / name
                artifacts[str(path.relative_to(out))] = digest(path)
        metrics[c["id"]] = values
        rows.append({**c, "split_f1": [v["macro_f1"] for v in values],
            "mean_f1": float(np.mean([v["macro_f1"] for v in values]))})
    comparisons = []
    for c in rows:
        if c["arm"] not in ("P1", "P3"):
            continue
        ref = ("P0" if c["arm"] == "P1" else "P2") + "_" + c["family"]
        comparisons.append({**c, "reference": ref, **gate(metrics[c["id"]], metrics[ref])})
    eligible = sorted([c for c in comparisons if c["passed"]], key=lambda c: (-c["mean_f1"], c["arm"], c["family"]))
    winner = eligible[0] if eligible else None
    lock = {"locked_at": now(), "protocol_sha256": digest(out / "protocol.json"),
        "development_artifacts": artifacts, "rows": rows, "comparisons": comparisons,
        "winner": winner["id"] if winner else None, "reference": winner["reference"] if winner else None,
        "historical_evaluation_authorized": bool(winner)}
    write(out / "selection_lock.json", lock)
    print("SELECTION", lock["winner"], flush=True)
    return lock


def historical_parity(out):
    p = verify(out); source = Path(p["source"]); results = []
    for arm, historical_arm in (("P0", "raw"), ("P2", "calibrated")):
        for family in p["models"]:
            for part in p["partitions"]:
                old_dir = source / "development" / f"3s_{historical_arm}_{family}" / part["id"]
                old_m = read(old_dir / "metrics.json")
                old_pred = old_dir / "predictions.npz"
                if digest(old_pred) != old_m["artifacts"]["predictions.npz"]:
                    raise ValueError("Historical prediction artifact changed")
                old = np.load(old_pred, allow_pickle=False)
                current = np.load(out / "development" / f"{arm}_{family}" / part["id"] / "predictions.npz", allow_pickle=False)
                for key in ("indices", "truth", "prediction", "groups", "record_ids"):
                    np.testing.assert_array_equal(current[key], old[key])
                np.testing.assert_allclose(current["probability"], old["probability"], rtol=0, atol=1e-12)
                results.append({"id": f"{arm}_{family}", "partition": part["id"], "predictions_exact": True,
                    "max_probability_error": float(np.max(np.abs(current["probability"] - old["probability"])))})
    write(out / "historical_reproduction.json", {"passed": True, "fits": results})


def final_comparison(out, lock):
    a = np.load(out / "final" / lock["reference"] / "holdout/predictions.npz", allow_pickle=False)
    b = np.load(out / "final" / lock["winner"] / "holdout/predictions.npz", allow_pickle=False)
    for key in ("indices", "truth", "groups", "record_ids"):
        np.testing.assert_array_equal(a[key], b[key])
    subjects = np.unique(a["groups"])
    rng = np.random.default_rng(20260921); delta = []
    for _ in range(2000):
        indices = np.concatenate([np.flatnonzero(a["groups"] == s) for s in rng.choice(subjects, len(subjects), replace=True)])
        delta.append(audit_score(b["truth"][indices], b["prediction"][indices])["macro_f1"] -
                     audit_score(a["truth"][indices], a["prediction"][indices])["macro_f1"])
    before, after = audit_score(a["truth"], a["prediction"]), audit_score(b["truth"], b["prediction"])
    result = {"reference": lock["reference"], "candidate": lock["winner"], "before": before, "after": after,
        "delta": after["macro_f1"] - before["macro_f1"], "subject_bootstrap_95": np.percentile(delta, [2.5,97.5]).tolist(),
        "fresh_external_validation": False, "subjects": len(subjects), "windows": len(a["truth"])}
    write(out / "historical_evaluation.json", result)
    return result


def finish(out, lock):
    from experiments.improvement_retest.prepare import protected_snapshot
    p = verify(out); root = Path(p["root"])
    audited = []
    for path in sorted(out.glob("*/P*/*/metrics.json")) + sorted(out.glob("sensitivity/P*/*/*/metrics.json")):
        m = read(path); pred = np.load(path.parent / "predictions.npz", allow_pickle=False)
        for name, sha in m["artifacts"].items():
            if digest(path.parent / name) != sha:
                raise ValueError("Fit artifact hash mismatch")
        np.testing.assert_array_equal(pred["prediction"], pred["probability"].argmax(1))
        score = audit_score(pred["truth"], pred["prediction"])
        np.testing.assert_allclose(score["macro_f1"], m["evaluation"]["macro_f1"], atol=1e-12)
        np.testing.assert_allclose(score["accuracy"], m["evaluation"]["accuracy"], atol=1e-12)
        np.testing.assert_allclose(score["per_class_f1"], [m["evaluation"]["per_class_f1"][n] for n in p["class_names"]], atol=1e-12)
        np.testing.assert_array_equal(score["confusion_matrix"], m["evaluation"]["confusion_matrix"])
        audited.append(str(path.relative_to(out)))
    expected = 53 if lock["winner"] else 36
    if len(audited) != expected:
        raise ValueError(f"Expected {expected} fits, got {len(audited)}")
    after = protected_snapshot(root)
    write(out / "protected_after.json", after)
    unchanged = after == read(out / "protected_before.json")
    if not unchanged:
        raise ValueError("Protected data/models changed during this run; investigate concurrent activity")
    final = final_comparison(out, lock) if lock["winner"] else None
    sensitivity = []
    if lock["winner"]:
        for omit in range(5):
            ms = [read(out / "sensitivity" / lock["winner"] / f"omit_{omit+1:03d}" / part["id"] / "metrics.json")["evaluation"] for part in p["partitions"]]
            refs = [read(out / "development" / lock["reference"] / part["id"] / "metrics.json")["evaluation"] for part in p["partitions"]]
            sensitivity.append({"omit_standard_index": omit, "mean_f1": float(np.mean([m["macro_f1"] for m in ms])), **gate(ms, refs)})
    summary = {"finished_at": now(), "fits": len(audited), "rows": lock["rows"], "comparisons": lock["comparisons"],
        "winner": lock["winner"], "historical_evaluation": final, "sensitivity": sensitivity,
        "data_and_existing_models_unchanged": unchanged, "approved": False, "hardware_checked": False}
    write(out / "summary.json", summary)
    write(out / "validation/audit.json", {"verified_fits": len(audited), "independently_recomputed_metrics": True,
        "historical_raw_and_gain_reproduction": True, "fits": audited, "finished_at": now()})
    report = root / "docs/reports" / f"{out.name}_report.md"
    rel = "../../research/results/" + out.name
    lines = ["# 공석 영점 보정 E2 실험 결과", "",
        f"실행 시작: {p['started_at']}; 종료: {summary['finished_at']} (Asia/Seoul).",
        "", "standard 공석 5개 파일의 센서별 중앙값을 파일 균등 가중 중앙값으로 결합했다. 원본 32명·692기록·4,502창, 7자세를 사용했다. 3초·12 Hz·596특징이며 기존 등록용 33기록 제외 정책을 유지했다.",
        "개발 분할은 학습 20명/검증 6명 3회이며 고정 6명은 별도다. 모델 시드는 분할별 20260921/20260922/20260923이고, 각 조건의 학습 전용 StandardScaler와 모델을 새로 적합했다. 아래 점수는 학습 점수가 아니다.",
        "", "## 개발 결과", "", "P0=x, P1=x−b, P2=g×x, P3=g×(x−b). b는 공석 영점이고 g는 기존 손가락 범위 기반 이득이다. 부호를 유지하고 클리핑하지 않았다.",
        "", "| 모델 | P0 무보정 | P1 영점 | P2 이득 | P3 영점+이득 | P1−P0 (%p) | P3−P2 (%p) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    by_id = {r["id"]: r for r in lock["rows"]}
    for f in p["models"]:
        v = [by_id[f"{a}_{f}"]["mean_f1"] for a in ARMS]
        lines.append(f"| {NAMES[f]} | " + " | ".join(f"{100*x:.2f}%" for x in v) + f" | {100*(v[1]-v[0]):+.2f} | {100*(v[3]-v[2]):+.2f} |")
    lines += ["", "## 사전 기준 판정", "",
        "개발 평균 +1%p 이상, 3분할 중 2개 이상 개선, 모든 분할에서 어떤 자세도 F1이 5%p 초과 하락하지 않는 조건을 모두 요구했다. 기준을 만족한 후보만 최종 평가 대상으로 허용했다.",
        "", "| 후보/비교군 | 평균 Δ (%p) | 개선 분할 | 최악 분할·자세 Δ (%p) | 통과 |", "| --- | ---: | ---: | ---: | --- |"]
    for c in lock["comparisons"]:
        lines.append(f"| {c['id']} / {c['reference']} | {100*c['mean_delta']:+.2f} | {c['positive_splits']}/3 | {100*c['worst_split_class_delta']:+.2f} | {'예' if c['passed'] else '아니오'} |")
    lines += ["", f"선택 후보: **{lock['winner'] or '없음'}**.", ""]
    if final:
        lo, hi = final["subject_bootstrap_95"]
        lines += [f"과거 고정 6명에서 {100*final['before']['macro_f1']:.2f}% → {100*final['after']['macro_f1']:.2f}% (Δ {100*final['delta']:+.2f}%p), 사람 단위 2,000회 bootstrap 95% 구간 [{100*lo:+.2f}, {100*hi:+.2f}]%p.",
            "이 평가자들의 과거 결과는 이미 알려져 있으며 독립적인 새 사용자 검증이 아니다. 구간은 학습 재실행이나 후보 선택 불확실성을 포함하지 않는다.", "",
            "공석 파일 하나씩 제외한 기준으로 선택 후보를 개발 3분할에서 각각 재학습했다. 민감도 결과는 summary.json에 있으며 후보를 재선택하거나 임계값을 바꾸지 않았다.", ""]
    else:
        lines += ["통과 후보가 없어 계획대로 36회 개발 학습에서 종료했다. 과거 고정 6명의 예측·최종 재학습과 선택 후보의 영점 민감도 실험은 수행하지 않았다.", ""]
    lines += ["## 분할별·자세별 근거", "", "개발 분할별 Macro-F1(%):", "",
        "| 조건 | 분할 1 | 분할 2 | 분할 3 |", "| --- | ---: | ---: | ---: |"]
    for row in lock["rows"]:
        lines.append(f"| {row['id']} | " + " | ".join(f"{100*v:.2f}" for v in row["split_f1"]) + " |")
    lines += ["", "자세별 F1은 각 분할의 F1을 평균한 보조 요약이다. 개발 분할 간 사람 재사용이 있어 독립 반복으로 해석하지 않는다.", "",
        "| 조건 | " + " | ".join(p["class_names"]) + " |", "| --- | " + " | ".join(["---:"]*7) + " |"]
    for row in lock["rows"]:
        ms = [read(out / "development" / row["id"] / part["id"] / "metrics.json")["evaluation"] for part in p["partitions"]]
        vals = [np.mean([m["per_class_f1"][label] for m in ms]) for label in p["class_names"]]
        lines.append(f"| {row['id']} | " + " | ".join(f"{100*v:.2f}" for v in vals) + " |")
    lines += ["", "## 검증 및 해석 범위", "",
        f"- {len(audited)}개 학습 모두 사람·기록 분리, 학습 전용 scaler, 확률 합, 저장 모델의 원시 창→보정→특징→예측 일치를 확인했다. 저장 예측으로 혼동행렬·Macro-F1·Accuracy·각 자세 F1을 별도 재계산했다.",
        "- P0/P2의 입력 특징과 개발 18회 예측을 이전 센서 이득 실험과 대조했다. 예측 클래스는 정확히, 확률은 절대 오차 1e-12 이내로 재현했다.",
        "- 입력 원본·스테이징·센서 수치 해시와 중복 부재를 확인했다. data/와 기존 research/models/의 목록·SHA-256·mtime은 실행 전후 동일하다. 활성 모델과 앱은 변경하지 않았다.",
        f"- 기존 보간에서 창 종료 이후 원본을 참조하는 표본 지점은 {p['causal_input_audit']['sample_points_using_source_after_nominal_window_end']}개, 최대 {p['causal_input_audit']['maximum_seconds_after_window_end']:.6f}초였다. 기존 입력 재현을 유지했으며 실제 센서 처리 지연은 시험하지 않았다.",
        "- 5개 공석 파일과 자세 기록이 같은 장비·설정인지 확인할 metadata가 없다. 따라서 물리적 영점 교정이나 센서 하중 감도의 개선으로 확정할 수 없다.",
        "- 유한 특징값을 확인했다. 일정한 채널의 skew/kurtosis 정밀도 경고는 기존 추출기의 비유한값 처리 계약을 유지했다.",
        "- 이번 범위는 E2의 CSV 실험이다. E1 공석 감지, 물리 센서, 실제 Tk, 앱 재빌드, 새 사람/새 세션 평가는 수행하지 않았다.", "",
        "## 재현 자료", "",
        f"- [프로토콜]({rel}/protocol.json), [선택 고정]({rel}/selection_lock.json), [요약]({rel}/summary.json)",
        f"- [영점·이득과 standard 해시]({rel}/calibration.json), [기존 결과 재현]({rel}/historical_reproduction.json)",
        f"- [평가 재계산 검사]({rel}/validation/audit.json), [실험 코드 검사 로그]({rel}/validation/contract_tests.log)",
        f"- [원본 자료 명세]({rel}/origin_dataset.json), [원본 보존 전]({rel}/protected_before.json), [원본 보존 후]({rel}/protected_after.json)",
        f"- 개발 모델과 예측은 `{rel}/development/<조건>/<분할>/`에 보존했다. 연구용 미승인 모델이다.",
        "- feature/model 코드는 이전 검증된 source_snapshot을 해시 확인 후 복사했고, 이번 실행기는 새 사본에서 실행했다. 실행 중 공유 소스 변경이 섞이지 않는다.", "",
        "```sh", ".venv/bin/python -B -m experiments.empty_seat_e2.run --output research/results/<새실행폴더>", "```", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    write(out / "training_complete.json", {"finished_at": now(), "fits": len(audited),
        "report": str(report), "report_sha256": digest(report), "summary_sha256": digest(out / "summary.json"),
        "data_and_existing_models_unchanged": True, "approved": False})
    print("COMPLETE", report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--phase", choices=("development", "final", "sensitivity"))
    parser.add_argument("--omit", type=int)
    args = parser.parse_args(); out = args.output.resolve()
    if args.worker:
        worker(out, args.worker, args.phase, args.omit)
    elif args.frozen:
        (out / "validation").mkdir(exist_ok=True)
        with (out / "validation/contract_tests.log").open("w") as stream:
            subprocess.run([sys.executable, "-B", "-m", "unittest", "experiments.empty_seat_e2.test_run", "experiments.sensor_gain.test_run", "-v"],
                stdout=stream, stderr=subprocess.STDOUT, check=True)
        prepare(out)
        p = verify(out)
        launch(out, [(c["id"], "development", None) for c in p["conditions"]])
        historical_parity(out)
        lock = lock_selection(out)
        if lock["winner"]:
            launch(out, [(lock["winner"], "final", None), (lock["reference"], "final", None)])
            launch(out, [(lock["winner"], "sensitivity", i) for i in range(5)])
        finish(out, lock)
    else:
        root = Path(__file__).resolve().parents[2]
        source = args.source.resolve() if args.source else root / "research/results/2026-09-23_150000_sensor_gain_2s3s"
        bootstrap(root, out, source)


if __name__ == "__main__":
    main()
