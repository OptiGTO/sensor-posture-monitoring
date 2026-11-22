"""Run a locked nested subject CV experiment without editing production data.

Each outer training pool gets its own inner OOF predictions. Records wrong in
at least half their windows under all six conditions are removed from training
only. Outer test people and records are never filtered. A class-matched random
record-removal control separates targeted filtering from sample-size effects.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/posture-consensus-mpl")
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.app_services import DatasetService
from src.model_comparison import build_model, scores, write_json
from src.posture_core import CLASS_NAMES, read_sensor_csv
from src.window_pipeline import WindowSpec, segment_sensor_frame, extract_window_features, aligned_predict_proba
from validate_personalization import audit_dataset

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260921
MODELS = {
    "random_forest": {"n_estimators": 200, "min_samples_leaf": 1, "max_features": "sqrt"},
    "rbf_svm": {"C": 1., "gamma": .0005},
    "xgboost": {"n_estimators": 200, "max_depth": 3, "learning_rate": .05,
                "min_child_weight": 3, "reg_lambda": 5., "subsample": .8, "colsample_bytree": .8},
}
CONDITIONS = [f"{family}_{seconds}s" for family in MODELS for seconds in (2, 3)]


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def protected_snapshot():
    return {str(p.relative_to(ROOT)): {"sha256": digest(p), "size": p.stat().st_size,
                                       "mtime_ns": p.stat().st_mtime_ns}
            for root in (ROOT / "data", ROOT / "research/models")
            for p in sorted(root.rglob("*")) if p.is_file()}


def subject_folds(subjects, n_splits, seed):
    people = np.array(sorted(set(subjects)))
    return [{"train": people[tr].tolist(), "test": people[te].tolist()}
            for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(people)]


def validate_split(groups, y, train, test):
    if set(groups[train]) & set(groups[test]):
        raise ValueError("Subject leakage")
    for idx in (train, test):
        if set(y[idx]) != set(range(len(CLASS_NAMES))):
            raise ValueError("Missing class; do not resample based on scores")


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    started = now()
    write_json(out / "protected_before.json", protected_snapshot())
    service = DatasetService(ROOT)
    indexed = service.records()
    # Explicit calibration/registration metadata is never treated as labeled fit data.
    def explicit_registration(r):
        if not r.manifest_path:
            return False
        m = read(r.manifest_path)
        return any(str(m.get(k, "")).lower() in {"registration", "calibration", "personal_registration"}
                   for k in ("mode", "source_type", "training_role", "purpose"))
    selected = [r for r in indexed if service.is_training_eligible(r) and not explicit_registration(r)]
    service.stage_for_training(selected, out / "training_data")
    records, audit = audit_dataset(out / "training_data")
    if audit["excluded_records"] or audit["duplicate_raw_files"] or audit["duplicate_sensor_sequences"]:
        raise ValueError("Staged cohort quality/duplicate audit failed")
    if len(records) != len(selected):
        raise ValueError("Eligible records were lost")
    write_json(out / "audit.json", {"staged_audit": audit, "excluded_indexed": [
        {"file": str(r.csv_path.relative_to(ROOT)), "source_type": r.source_type,
         "training_role": r.training_role, "label_status": r.label_status, "qc_status": r.qc_status,
         "explicit_registration": explicit_registration(r)} for r in indexed if r not in selected]})
    frames = {r.relative_path: read_sensor_csv(r.path) for r in records}
    cohorts = {}
    for sec in (2, 3):
        seq, ys, groups, ids, starts = [], [], [], [], []
        for r in records:
            windows, meta = segment_sensor_frame(frames[r.relative_path], WindowSpec(sec))
            seq.extend(windows)
            ys.extend([CLASS_NAMES.index(r.label)] * len(windows))
            groups.extend([r.subject] * len(windows))
            ids.extend([r.relative_path] * len(windows))
            starts.extend(meta.start_s.tolist())
        data = dict(X=extract_window_features(np.asarray(seq)), y=np.asarray(ys),
                    groups=np.asarray(groups), record_ids=np.asarray(ids), start_s=np.asarray(starts))
        if not np.isfinite(data["X"]).all():
            raise ValueError("Nonfinite features")
        np.savez_compressed(out / f"data_{sec}s.npz", **data)
        cohorts[str(sec)] = {"windows": len(ys), "features": data["X"].shape[1],
                            "class_window_counts": np.bincount(ys, minlength=7).tolist(),
                            "cache_sha256": digest(out / f"data_{sec}s.npz")}
    subjects = sorted({r.subject for r in records})
    outer = subject_folds(subjects, 5, SEED)
    for i, split in enumerate(outer):
        split["id"] = f"outer_{i}"
        split["inner"] = subject_folds(split["train"], 3, SEED + 100 + i)
    sources = [*ROOT.glob("*.py"), *(ROOT / "src").rglob("*.py"),
               *Path(__file__).parent.glob("*.py")]
    source_hashes = {}
    for src in sorted(set(sources)):
        rel = src.relative_to(ROOT)
        dest = out / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        source_hashes[str(rel)] = digest(src)
    shutil.copy2(service.correction_manifest, out / "label_corrections.json")
    protocol = {"started_at": started, "locked_at": now(), "seed": SEED,
        "purpose": "Retrospective nested-CV ablation, no production selection or fresh external test",
        "models": MODELS, "conditions": CONDITIONS, "cohorts": cohorts,
        "subjects": subjects, "records": len(records), "labels": list(CLASS_NAMES), "outer": outer,
        "consensus": {"unit": "complete original recording", "error_rate_threshold": .5,
                      "required_conditions": 6, "comparison": ">=", "relabel": False,
                      "justification": "Prespecified majority-error screen, not a verified anomaly diagnosis"},
        "arms": ["before", "filtered", "random_control"],
        "random_control": "Same record count per class as consensus removal; fixed random seed; training pool only",
        "metric_primary": "Pooled untouched outer-OOF seven-class Macro F1; filtered minus before",
        "success_criterion": "Per condition: Macro F1 delta >0, paired subject bootstrap 95% lower bound >0, no class F1 loss over 0.05, and positive filtered-minus-random delta. Exploratory, no multiplicity correction or deployment approval.",
        "window": "12 Hz, nonoverlapping stride equal to window; retain tails at >=95%; 16 pressure channels",
        "prediction": "aligned predict_proba argmax; SVC calibration uses only its current training rows",
        "scaling": "StandardScaler fit only on each model training subset",
        "tuning": "No hyperparameter or threshold search; same fixed settings for both durations and all arms",
        "registration_policy": "Exclude explicit registration/calibration metadata. No synthetic first-correct-posture reservation; ordinary eligible correct-posture records included per current dataset policy.",
        "scope": "Historical data previously used in other studies; subject identity follows normalized directory; chair/session/actual posture unverified",
        "uncertainty": "2000 paired bootstrap draws of whole subjects, conditional on these trained folds; not independent training-set uncertainty",
        "source_hashes": source_hashes, "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "command": [sys.executable, *sys.argv], "acquisition_time": "Not inferred from file timestamps",
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "packages": {k: version(k) for k in ["numpy", "scikit-learn", "xgboost", "joblib", "pandas", "matplotlib"]}}}
    write_json(out / "protocol.json", protocol)
    write_json(out / "protocol_lock.json", {"sha256": digest(out / "protocol.json"), "locked_at": now()})
    print("PREPARED", len(subjects), len(records), cohorts, flush=True)


def verify(out):
    p = read(out / "protocol.json")
    if digest(out / "protocol.json") != read(out / "protocol_lock.json")["sha256"]:
        raise ValueError("Protocol changed")
    for sec in (2, 3):
        if digest(out / f"data_{sec}s.npz") != p["cohorts"][str(sec)]["cache_sha256"]:
            raise ValueError("Frozen data cache changed")
    for path, sha in p["source_hashes"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Source changed after lock: {path}")
    return p


def fit(family, X, y, seed):
    if set(y) != set(range(7)):
        raise ValueError("Filtering removed a class")
    model = build_model(family, MODELS[family], seed)
    kwargs = {}
    if family == "xgboost":
        from sklearn.utils.class_weight import compute_sample_weight
        kwargs["model__sample_weight"] = compute_sample_weight("balanced", y)
    t = time.perf_counter()
    model.fit(X, y, **kwargs)
    np.testing.assert_allclose(model.named_steps["scale"].mean_, X.mean(axis=0, dtype=np.float64), rtol=1e-10, atol=1e-10)
    return model, time.perf_counter() - t


def predict(model, X):
    return aligned_predict_proba(model, X, 7).argmax(1)


def worker(out, scope, stage, condition):
    p = verify(out)
    family, seconds = condition.rsplit("_", 1)
    d = np.load(out / f"data_{seconds}.npz", allow_pickle=False)
    X, y, groups, ids = (d[k] for k in ("X", "y", "groups", "record_ids"))
    split = next((s for s in p["outer"] if s["id"] == scope), None)
    dest = out / scope / stage / condition
    dest.mkdir(parents=True, exist_ok=True)
    if stage == "inner":
        pred = np.full(len(y), -1, dtype=int)
        timings = []
        for i, s in enumerate(split["inner"]):
            tr, te = (np.flatnonzero(np.isin(groups, s[k])) for k in ("train", "test"))
            validate_split(groups, y, tr, te)
            if set(groups[tr]) & set(split["test"]) or set(groups[te]) & set(split["test"]):
                raise ValueError("Outer test leaked into exclusion discovery")
            model, elapsed = fit(family, X[tr], y[tr], SEED + i)
            pred[te] = predict(model, X[te])
            timings.append(elapsed)
        expected = np.isin(groups, split["train"])
        np.testing.assert_array_equal(pred >= 0, expected)
        np.savez_compressed(dest / "predictions.npz", prediction=pred)
        write_json(dest / "metrics.json", {"fit_seconds": timings, "finished_at": now(),
            "oof": scores(y[expected], pred[expected]), "predictions_sha256": digest(dest / "predictions.npz")})
    else:
        selected = read(out / scope / "exclusions.json")
        if digest(out / scope / "exclusions.json") != read(out / scope / "exclusions_lock.json")["sha256"]:
            raise ValueError("Exclusions changed after lock")
        pool = np.ones(len(y), dtype=bool) if scope == "full" else np.isin(groups, split["train"])
        test = np.arange(len(y)) if scope == "full" else np.flatnonzero(np.isin(groups, split["test"]))
        arms = ["before", "filtered"] if scope == "full" else p["arms"]
        for arm in arms:
            excluded = [] if arm == "before" else selected["consensus" if arm == "filtered" else "random_control"]
            tr = np.flatnonzero(pool & ~np.isin(ids, excluded))
            if scope != "full":
                validate_split(groups, y, tr, test)
            model, elapsed = fit(family, X[tr], y[tr], SEED)
            t = time.perf_counter()
            pred = predict(model, X[test])
            predict_ms = (time.perf_counter() - t) * 1000 / len(test)
            path = dest / arm
            path.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "approved": False, "family": family,
                         "window_seconds": int(seconds[0]), "label_names": list(CLASS_NAMES),
                         "scope": scope, "arm": arm, "training_indices": tr,
                         "protocol_sha256": digest(out / "protocol.json"),
                         "exclusions_sha256": digest(out / scope / "exclusions.json")}, path / "model.joblib", compress=3)
            loaded = joblib.load(path / "model.joblib")["model"]
            np.testing.assert_array_equal(pred, predict(loaded, X[test]))
            np.savez_compressed(path / "predictions.npz", indices=test, truth=y[test], prediction=pred)
            train_pred = predict(model, X[tr])
            write_json(path / "metrics.json", {"started_scope": scope, "finished_at": now(),
                "fit_seconds": elapsed, "predict_ms_per_window_batched": predict_ms,
                "train_windows": len(tr), "train_records": len(set(ids[tr])), "train_subjects": len(set(groups[tr])),
                "training": scores(y[tr], train_pred), "evaluation": scores(y[test], pred),
                "score_kind": "resubstitution_original_full_cohort_not_generalization" if scope == "full" else "untouched_outer_subject_oof",
                "model_sha256": digest(path / "model.joblib"), "predictions_sha256": digest(path / "predictions.npz"),
                "saved_prediction_parity": True})
    print(scope, stage, condition, "DONE", now(), flush=True)


def record_errors(ids, y, pred):
    rows = []
    for rid in np.unique(ids[pred >= 0]):
        idx = (ids == rid) & (pred >= 0)
        labels = np.unique(y[idx])
        if len(labels) != 1:
            raise ValueError("Multiple truth labels within one recording")
        hist = np.bincount(pred[idx], minlength=7)
        wrong = int((pred[idx] != y[idx]).sum())
        rows.append({"record_id": rid, "label_id": int(labels[0]), "windows": int(idx.sum()),
                     "wrong": wrong, "error_rate": wrong / int(idx.sum()),
                     "dominant_prediction": int(hist.argmax()), "prediction_histogram": hist.tolist()})
    return rows


def common_records(tables, threshold=.5):
    if len(tables) != 6:
        raise ValueError("Six conditions required")
    inventories = [{r["record_id"] for r in rows} for rows in tables.values()]
    if any(x != inventories[0] for x in inventories[1:]):
        raise ValueError("Record inventories must match across window lengths")
    return sorted(set.intersection(*[{r["record_id"] for r in rows if r["error_rate"] >= threshold}
                                    for rows in tables.values()]))


def random_control(rows, excluded, seed):
    rng = np.random.default_rng(seed)
    by_id = {r["record_id"]: r for r in rows}
    result = []
    for label in range(7):
        count = sum(by_id[r]["label_id"] == label for r in excluded)
        pool = sorted(r["record_id"] for r in rows if r["label_id"] == label)
        result.extend(rng.choice(pool, count, replace=False).tolist())
    return sorted(result)


def exclusions(out, scope):
    tables = {}
    for condition in CONDITIONS:
        sec = condition.rsplit("_", 1)[1]
        d = np.load(out / f"data_{sec}.npz", allow_pickle=False)
        if scope == "full":
            pred = np.full(len(d["y"]), -1, dtype=int)
            for split in read(out / "protocol.json")["outer"]:
                z = np.load(out / split["id"] / "evaluate" / condition / "before/predictions.npz")
                if (pred[z["indices"]] >= 0).any():
                    raise ValueError("Duplicate OOF test predictions")
                pred[z["indices"]] = z["prediction"]
            if (pred < 0).any():
                raise ValueError("Incomplete full OOF coverage")
        else:
            pred = np.load(out / scope / "inner" / condition / "predictions.npz")["prediction"]
        tables[condition] = record_errors(d["record_ids"], d["y"], pred)
    common = common_records(tables)
    result = {"created_at": now(), "scope": scope, "consensus": common,
              "random_control": random_control(next(iter(tables.values())), common, SEED + len(scope)),
              "record_error_tables": tables, "threshold": .5,
              "test_predictions_used": scope == "full",
              "note": "Full-cohort list only for review/research refit; never reused for outer CV filtering" if scope == "full" else "Inner OOF only; outer test untouched"}
    write_json(out / scope / "exclusions.json", result)
    write_json(out / scope / "exclusions_lock.json", {"locked_at": now(), "sha256": digest(out / scope / "exclusions.json")})
    print(scope, "LOCKED EXCLUSIONS", len(common), flush=True)


def launch(out, scope, stage):
    def one(condition):
        logfile = out / "logs" / f"{scope}_{stage}_{condition}.log"
        logfile.parent.mkdir(exist_ok=True)
        with logfile.open("w") as log:
            subprocess.run([sys.executable, "-m", "experiments.consensus_filter.run", "--output", str(out),
                            "--worker", condition, "--scope", scope, "--stage", stage],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(logfile.read_text().strip(), flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one, CONDITIONS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=CONDITIONS)
    parser.add_argument("--scope")
    parser.add_argument("--stage", choices=["inner", "evaluate"])
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        worker(out, args.scope, args.stage, args.worker)
        return
    prepare(out)
    for split in read(out / "protocol.json")["outer"]:
        launch(out, split["id"], "inner")
        exclusions(out, split["id"])
        launch(out, split["id"], "evaluate")
    exclusions(out, "full")
    launch(out, "full", "evaluate")
    verify(out)
    if protected_snapshot() != read(out / "protected_before.json"):
        raise ValueError("Canonical data/models changed during experiment")
    write_json(out / "execution_complete.json", {"finished_at": now(), "protected_data_models_unchanged": True,
                                                "fits_expected": 192})
    print("COMPLETE", now(), flush=True)


if __name__ == "__main__":
    main()
