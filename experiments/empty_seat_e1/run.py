"""Frozen-source E1 occupancy replay. Writes only a new experiment directory."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import types
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def protected(root):
    result = {}
    for folder in ("data", "research/models"):
        for p in sorted((root / folder).rglob("*")):
            if p.is_file():
                stat = p.stat()
                result[str(p.relative_to(root))] = dict(sha256=digest(p), size=stat.st_size,
                                                       mtime_ns=stat.st_mtime_ns)
    return result


def load_runtime(snapshot):
    """Execute exact frozen occupancy AST nodes, avoiding unrelated app initialization."""
    spec = importlib.util.spec_from_file_location("e1_frozen_core", snapshot / "posture_core.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    names = {"OccupancyConfig", "StandardProfile", "build_standard_profile", "OccupancyDetector", "_digest"}
    tree = ast.parse((snapshot / "app_services.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    module = types.ModuleType("e1_frozen_occupancy")
    sys.modules[module.__name__] = module
    module.__dict__.update(np=np, Path=Path, deque=deque, dataclass=dataclass,
                           field=field, Iterable=Iterable, hashlib=hashlib,
                           read_sensor_csv=core.read_sensor_csv, SENSOR_COLUMNS=core.SENSOR_COLUMNS)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(snapshot / "app_services.py"), "exec"), module.__dict__)
    return core, module


def smooth(t, x):
    result = np.empty_like(x)
    start = 0
    for i in range(len(t)):
        if i and t[i] - t[i - 1] > 1:
            start = i
        result[i] = np.mean(x[max(start, i - 4):i + 1], axis=0)
    return result


def score_values(record, profile, kind):
    delta = np.abs(record["smooth"] - profile.center)
    z = delta / profile.deviation
    score = np.median(z, axis=1)
    if kind == "A":
        score += np.mean(z >= 3, axis=1) + delta.sum(axis=1) / max(profile.deviation.sum(), 1e-12)
    return score


def states_from_scores(t, scores, enter):
    """Causal hysteresis matching the runtime; large gaps reset all state."""
    occupied, candidate, since = False, None, None
    states = np.empty(len(t), dtype=bool)
    transitions = []
    for i, (stamp, score) in enumerate(zip(t, scores)):
        if i and stamp - t[i - 1] > 1:
            occupied, candidate, since = False, None, None
        desired = bool(score >= (enter / 2 if occupied else enter))
        if desired == occupied:
            candidate, since = None, None
        elif desired != candidate:
            candidate, since = desired, stamp
        elif since is not None and stamp - since >= .75:
            occupied = desired
            candidate, since = None, None
            transitions.append((i, occupied))
        states[i] = occupied
    return states, transitions


def runtime_states(record, profile, runtime):
    detector = runtime.OccupancyDetector(profile)
    states, transitions = [], []
    for i, (t, x) in enumerate(zip(record["t"], record["x"])):
        if i and t - record["t"][i - 1] > 1:
            detector = runtime.OccupancyDetector(profile)
        change = detector.update(float(t), x)
        if change is not None:
            transitions.append((i, bool(change)))
        states.append(detector.occupied)
    return np.asarray(states, dtype=bool), transitions


def measures(t, states, occupied_truth, transitions):
    dt = np.diff(t)
    valid = dt <= 1
    all_weight = np.where(valid, dt, 0)
    post_weight = np.where(valid, np.maximum(t[1:] - np.maximum(t[:-1], t[0] + 1), 0), 0)
    error = states[:-1] != occupied_truth
    output = {"unobserved_gap_s": float(dt[~valid].sum()), "gap_count": int((~valid).sum()),
              "false_entries": sum(state for _, state in transitions) if not occupied_truth else 0,
              "first_occupied_s": float(t[np.flatnonzero(states)[0]] - t[0]) if states.any() else None}
    for name, weight in (("full", all_weight), ("post1", post_weight)):
        seconds, errors = float(weight.sum()), float(weight[error].sum())
        output[name + "_seconds"] = seconds
        output[name + "_error_seconds"] = errors
        output[name + "_error_rate"] = errors / seconds if seconds else None
    return output


def aggregate(rows):
    output = {"records": len(rows), "false_entries": sum(r["false_entries"] for r in rows),
              "gap_count": sum(r["gap_count"] for r in rows),
              "unobserved_gap_s": sum(r["unobserved_gap_s"] for r in rows)}
    for name in ("full", "post1"):
        seconds = sum(r[name + "_seconds"] for r in rows)
        errors = sum(r[name + "_error_seconds"] for r in rows)
        output[name + "_seconds"] = seconds
        output[name + "_error_seconds"] = errors
        output[name + "_error_rate"] = errors / seconds if seconds else None
        output[name + "_worst_record_rate"] = max((r[name + "_error_rate"] for r in rows), default=None)
    return output


def self_checks(runtime):
    t = np.array([0., .5, 1., 1.5, 2.])
    states, changes = states_from_scores(t, np.ones(5) * 10, 4)
    assert states.tolist() == [False, False, True, True, True]
    m = measures(t, states, True, changes)
    assert m["full_error_rate"] == .5 and m["post1_error_rate"] == 0
    t = np.array([0., .25, 1., 3., 3.25, 4.])
    states, changes = states_from_scores(t, np.ones(6) * 10, 4)
    assert states.tolist() == [False, False, True, False, False, True]
    m = measures(t, states, True, changes)
    assert m["unobserved_gap_s"] == 2 and m["full_seconds"] == 2
    t = np.array([0., .8, 1.2])
    m = measures(t, np.zeros(3, dtype=bool), True, [])
    assert abs(m["post1_seconds"] - .2) < 1e-12 and m["post1_error_rate"] == 1
    # Exact production hysteresis agreement, including drop, brief spike and gap.
    rng = np.random.default_rng(20260923)
    t = np.arange(160) * .1
    t[100:] += 2
    x = np.ones((160, 16)) + rng.normal(0, .001, (160, 16))
    x[10:60] += .5
    x[80:82] += .5
    x[95:130] += .5
    r = {"t": t, "x": x, "smooth": smooth(t, x)}
    p = runtime.StandardProfile(np.ones(16), np.ones(16) * .01, "synthetic", 1)
    a, ac = runtime_states(r, p, runtime)
    b, bc = states_from_scores(t, score_values(r, p, "A"), 4)
    assert np.array_equal(a, b) and ac == bc
    return {"duration_weighting": "passed", "warmup_boundary": "passed",
            "gap_reset": "passed", "runtime_hysteresis_parity_synthetic": "passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = now()
    source = root / "research/results/2026-09-23_150000_sensor_gain_2s3s"
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    sources = [root / "src/app_services.py", root / "src/posture_core.py", Path(__file__).resolve()]
    for p in sources:
        shutil.copy2(p, snapshot / p.name)
    source_hashes = {str(p.relative_to(root)): digest(p) for p in sources}
    core, runtime = load_runtime(snapshot)
    checks = self_checks(runtime)
    before = protected(root)
    write(out / "protected_before.json", before)
    original = json.loads((source / "origin_dataset.json").read_text())
    lock = json.loads((source / "origin_dataset_lock.json").read_text())
    assert digest(source / "origin_dataset.json") == lock["sha256"]
    historical = json.loads((source / "protocol.json").read_text())
    assert historical["origin_dataset_sha256"] == lock["sha256"]
    partitions = historical["partitions"]
    for filename in ("origin_dataset.json", "origin_dataset_lock.json", "protocol.json"):
        shutil.copy2(source / filename, out / ("origin_" + filename))
    inner_groups = {}
    for part in partitions:
        people = np.random.default_rng(part["seed"]).permutation(sorted(part["train"])).tolist()
        inner_groups[part["id"]] = {"seed": part["seed"], "fit_people": people[5:], "tuning_people": people[:5]}
        assert not set(part["train"]) & set(part["validation"])
        assert not (set(part["train"]) | set(part["validation"])) & set(part["test"])
    protocol = {"started_at": started, "locked_at": now(), "experiment": "E1 occupancy only",
                "root": str(root), "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
                "source_hashes": source_hashes, "origin_dataset_sha256": lock["sha256"],
                "command": sys.argv, "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                "partitions": partitions, "inner_groups": inner_groups,
                "inner_protocol": "For each outer training set, seeded person holdout: 5 tuning / 15 fit-reserved. No person model is fitted. Cross with four inner empty-file holdouts; empty profile fits remaining three files.",
                "A": "Exact frozen production occupancy AST; 4 enter / 2 exit, .75s holds, 5-frame mean",
                "B": "median(abs(smoothed x - center) / deviation); otherwise same state logic",
                "B_enter_candidates": [2, 3, 4, 6], "B_exit_ratio": .5,
                "selection": "Full-duration inner occupied error <= .01, then least empty error seconds, least occupied error rate, lowest threshold. No eligible threshold -> B rejected, no outer B evaluation.",
                "time_weight": "State after row i applies to [t_i,t_i+1); gaps >1s excluded and state reset. No extension past last row.",
                "startup": "Empty initial state each file; full is selection/primary; post1 is secondary only, clipped interval at t0+1.",
                "outer_targets": {"empty_rate_max": .01, "occupied_rate_max": .01, "class_miss_max": .02},
                "holdout_used": False, "hardware_verified": False, "approved": False}
    write(out / "protocol.json", protocol)
    print("protocol locked", flush=True)
    data, manifests, sensor_hashes, raw_hashes = {}, [], set(), set()
    test_people = set(partitions[0]["test"])
    dev_records = []
    for row in original["records"]:
        path = root / "data" / row["source_relative_path"]
        assert digest(path) == row["sha256"], str(path)
        if row["subject"] in test_people:
            continue  # bytes checked; no holdout numeric values or predictions opened
        frame = core.read_sensor_csv(path)
        assert core.recording_label(frame)[1] == row["label"]
        x = frame[core.SENSOR_COLUMNS].to_numpy(dtype="<f8", copy=True)
        x[x == 0] = 0
        sh = hashlib.sha256(np.asarray(x.shape, dtype="<i8").tobytes() + x.tobytes()).hexdigest()
        assert sh == row["sensor_sha256"] and sh not in sensor_hashes
        assert row["sha256"] not in raw_hashes
        sensor_hashes.add(sh); raw_hashes.add(row["sha256"])
        t = frame.elapsed_s.to_numpy(dtype=float)
        rid = row["source_relative_path"]
        data[rid] = {"t": t, "x": x, "smooth": smooth(t, x), "occupied": True,
                     "subject": row["subject"], "label": row["label"]}
        manifests.append(dict(row, role="development_occupied"))
        dev_records.append(rid)
    empty_paths = sorted((root / "data/standard").glob("*.csv"))
    assert len(empty_paths) == 5
    empty_ids = []
    for path in empty_paths:
        frame = core.read_sensor_csv(path)
        assert set(frame.label) == {"standard"}
        rid = str(path.relative_to(root / "data"))
        t, x = frame.elapsed_s.to_numpy(dtype=float), frame[core.SENSOR_COLUMNS].to_numpy(dtype=float)
        data[rid] = {"t": t, "x": x, "smooth": smooth(t, x), "occupied": False,
                     "subject": "standard", "label": "empty"}
        empty_ids.append(rid)
        manifests.append({"source_relative_path": rid, "sha256": digest(path), "rows": len(t),
                          "duration_s": float(t[-1]-t[0]), "role": "empty", "session": "unknown"})
    write(out / "source_manifest.json", {"records": manifests, "source_cohort_records": len(original["records"]),
                                         "development_records": len(dev_records), "historical_holdout_excluded": len(original["records"])-len(dev_records)})
    profile_cache, prediction_cache = {}, {}

    def profile(ids):
        key = tuple(sorted(ids))
        if key not in profile_cache:
            name = "_".join(Path(i).stem[-3:] for i in key)
            links = out / "profile_inputs" / name
            links.mkdir(parents=True)
            for rid in key:
                (links / Path(rid).name).symlink_to(root / "data" / rid)
            p = runtime.build_standard_profile(links)
            write(out / "profiles" / (name + ".json"), {"sources": list(key), "center": p.center.tolist(),
                                                        "deviation": p.deviation.tolist(), "signature": p.signature})
            profile_cache[key] = p
        return key, profile_cache[key]

    parity_count = 0
    def evaluate(profile_ids, rid, kind, threshold):
        nonlocal parity_count
        key, p = profile(profile_ids)
        cache_key = key, rid, kind, threshold
        if cache_key not in prediction_cache:
            r = data[rid]
            score = score_values(r, p, kind)
            states, changes = states_from_scores(r["t"], score, threshold)
            if kind == "A":
                actual, actual_changes = runtime_states(r, p, runtime)
                assert np.array_equal(states, actual) and changes == actual_changes, rid
                states, changes = actual, actual_changes
                parity_count += 1
            m = measures(r["t"], states, r["occupied"], changes)
            m.update(record_id=rid, subject=r["subject"], label=r["label"], kind=kind, threshold=threshold)
            prediction_cache[cache_key] = m
            if len(key) == 4:
                name = hashlib.sha256(repr(cache_key).encode()).hexdigest()[:20]
                dest = out / "predictions" / (name + ".npz")
                dest.parent.mkdir(exist_ok=True)
                np.savez_compressed(dest, t=r["t"], states=states, scores=score,
                                    occupied_truth=np.array(r["occupied"]), record_id=np.array(rid))
                m["predictions"] = str(dest.relative_to(out))
        return prediction_cache[cache_key]

    locks, inner_rows, outer_rows, fold_results = [], [], [], []
    for empty_test in empty_ids:
        outer_empty_train = [i for i in empty_ids if i != empty_test]
        for part in partitions:
            fold = Path(empty_test).stem + "_" + part["id"]
            tune_people = set(inner_groups[part["id"]]["tuning_people"])
            tuning_ids = [i for i in dev_records if data[i]["subject"] in tune_people]
            validation_ids = [i for i in dev_records if data[i]["subject"] in part["validation"]]
            assert not set(tuning_ids) & set(validation_ids)
            assert set(data[i]["label"] for i in tuning_ids) == set(core.CLASS_NAMES)
            candidates = []
            for threshold in (2, 3, 4, 6):
                erows, orows = [], []
                for empty_inner in outer_empty_train:
                    inner_fit = [i for i in outer_empty_train if i != empty_inner]
                    e = evaluate(inner_fit, empty_inner, "B", threshold)
                    erows.append(e)
                    for rid in tuning_ids:
                        o = evaluate(inner_fit, rid, "B", threshold)
                        orows.append(o)
                    inner_rows.append({"fold": fold, "inner_empty_test": empty_inner,
                                       "threshold": threshold, "empty": e,
                                       "occupied": aggregate(orows[-len(tuning_ids):])})
                es, os = aggregate(erows), aggregate(orows)
                candidates.append({"threshold": threshold, "empty": es, "occupied": os,
                                   "eligible": os["full_error_rate"] <= .01})
            eligible = [c for c in candidates if c["eligible"]]
            chosen = min(eligible, key=lambda c: (c["empty"]["full_error_seconds"], c["occupied"]["full_error_rate"], c["threshold"])) if eligible else None
            selection = {"fold": fold, "locked_at": now(), "candidates": candidates,
                         "selected_threshold": chosen["threshold"] if chosen else None,
                         "status": "selected" if chosen else "no_eligible_candidate"}
            locks.append(selection)
            write(out / "selection_locks" / (fold + ".json"), selection)
            result = {"fold": fold, "empty_test": empty_test, "person_split": part["id"]}
            methods = [("A", 4)] + ([("B", chosen["threshold"])] if chosen else [])
            for kind, threshold in methods:
                e = evaluate(outer_empty_train, empty_test, kind, threshold)
                rows = [evaluate(outer_empty_train, rid, kind, threshold) for rid in validation_ids]
                outer_rows.extend(dict(r, fold=fold) for r in [e, *rows])
                result[kind] = {"empty": aggregate([e]), "occupied": aggregate(rows),
                                "by_class": {label: aggregate([r for r in rows if r["label"] == label]) for label in core.CLASS_NAMES},
                                "by_subject": {person: aggregate([r for r in rows if r["subject"] == person]) for person in part["validation"]}}
            fold_results.append(result)
            write(out / "progress.json", {"completed": len(fold_results), "total": 15, "last": fold})
            print(fold, "B", selection["status"], "A occupied", result["A"]["occupied"]["full_error_rate"], flush=True)
    write(out / "inner_metrics.json", inner_rows)
    write(out / "selection_lock.json", locks)
    write(out / "outer_metrics.json", fold_results)
    write(out / "record_metrics.json", outer_rows)
    keys = sorted(set().union(*(r.keys() for r in outer_rows)))
    with (out / "record_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(outer_rows)
    # Dedupe repeated person-split appearances; profile-varying occupied replays stay separate.
    unique = { (r["kind"], r["record_id"], Path(r["predictions"]).name): r for r in outer_rows }
    empty_unique = [r for r in unique.values() if r["kind"] == "A" and r["label"] == "empty"]
    occupied_unique = [r for r in unique.values() if r["kind"] == "A" and r["label"] != "empty"]
    summary = {"started_at": started, "finished_at": now(), "experiment": "E1",
               "empty_unique_files": len(empty_ids), "development_records": len(dev_records),
               "outer_unique_people": len(set(r["subject"] for r in occupied_unique)),
               "outer_unique_occupied_records": len(set(r["record_id"] for r in occupied_unique)),
               "outer_occupied_profile_record_pairs": len(occupied_unique),
               "fold_count": len(fold_results), "B_selected_folds": sum(l["status"] == "selected" for l in locks),
               "A_empty_unique": aggregate(empty_unique), "A_occupied_unique_profile_record_pairs": aggregate(occupied_unique),
               "A_by_class": {label: aggregate([r for r in occupied_unique if r["label"] == label]) for label in core.CLASS_NAMES},
               "B_inner_full_miss_range": [min(c["occupied"]["full_error_rate"] for l in locks for c in l["candidates"]), max(c["occupied"]["full_error_rate"] for l in locks for c in l["candidates"])],
               "B_inner_post1_miss_range": [min(c["occupied"]["post1_error_rate"] for l in locks for c in l["candidates"]), max(c["occupied"]["post1_error_rate"] for l in locks for c in l["candidates"])],
               "interpretation": "Static CSV replay only; five baseline records are not known independent sessions; no real transition latency or hardware validation.",
               "historical_holdout_predictions": False, "active_model_changed_by_experiment": False}
    checks["runtime_parity_actual_record_profile_pairs"] = parity_count
    checks["source_snapshot_unchanged"] = all(digest(snapshot / p.name) == source_hashes[str(p.relative_to(root))] for p in sources)
    after = protected(root)
    write(out / "protected_after.json", after)
    changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
    checks["protected_files_unchanged"] = not changed
    checks["protected_changes"] = changed
    checks["original_source_changes_during_run"] = [str(p.relative_to(root)) for p in sources if digest(p) != source_hashes[str(p.relative_to(root))]]
    assert checks["source_snapshot_unchanged"]
    write(out / "checks.json", checks)
    write(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
