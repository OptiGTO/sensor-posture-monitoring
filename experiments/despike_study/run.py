"""LOSO study: versioned despike_v1 and training start trim vs the unprocessed baseline.

Four preprocessing arms share one staged cohort (the app's training-eligible
records, subjects covering all seven postures; the submission-retrain rule):

  baseline       no signal preprocessing (current runtime behaviour)
  despike        despike_v1 on every training, evaluation and live input
  trim1          first 1.0 s of each *training* recording removed; evaluation untouched
  despike_trim1  both

Models: the active runtime recipe (RBF-SVM 3 s, C=1, gamma="scale") and the
research Extra Trees recipe, three seeds each, 32-subject leave-one-subject-out.
Every held-out subject is scored on the same untrimmed windows in every arm, and
on a live replay (3 s buffer, 1 s prediction interval, 3-step smoothing) built by
src.realtime.TimeWindowBuffer exactly like the app. Arrays come from the
production code (src.model_comparison._dataset with SignalPreprocessing), and two
configurations are re-run through src.model_comparison.loso_evaluation to prove
the fold loop is the same computation. protocol.json (arms, criteria, cohort
hashes) is locked before any fit. data/ and research/models/ are read only and
compared before/after. No model is saved, registered or activated.

Usage (new output folder only):
  .venv/bin/python -B -m experiments.despike_study.run --output research/results/<YYYY-MM-DD_HHMMSS>_despike_study
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.despike_study.analysis import (  # noqa: E402
    decide, live_stability, macro_f1, paired_subject_bootstrap, per_class_f1)
from experiments.runtime_vs_research.run import now, protected_state, read, sha, write  # noqa: E402

WINDOW_SECONDS = 3
TRIM_SECONDS = 1.0
SEEDS = [20260826, 20260827, 20260828]
PARTITION_SEED = 20260826  # only used by _dataset's partition bookkeeping; LOSO ignores it
MODELS = {
    "rbf_svm": {"C": 1.0, "gamma": "scale"},  # active runtime model's selected parameters
    "extra_trees": {"n_estimators": 200, "min_samples_leaf": 8, "max_depth": 8, "max_features": 0.5},
}
ARM_NAMES = ("baseline", "despike", "trim1", "despike_trim1")
ARM_INPUTS = {  # cache keys: evaluation X, (fit X, fit metadata prefix), live X
    "baseline": ("eval_raw_X", ("eval_raw_X", "eval"), "live_raw_X"),
    "despike": ("eval_despike_X", ("eval_despike_X", "eval"), "live_despike_X"),
    "trim1": ("eval_raw_X", ("fit_trim_raw_X", "fit_trim"), "live_raw_X"),
    "despike_trim1": ("eval_despike_X", ("fit_trim_despike_X", "fit_trim"), "live_despike_X"),
}
PRIMARY = {"model": "rbf_svm", "arm": "despike"}
CRITERIA = {
    "minimum_mean_delta": 0.01, "maximum_per_class_drop": 0.05,
    "non_inferiority_mean_margin": 0.005, "non_inferiority_bootstrap_margin": 0.015,
    "minimum_relative_change_reduction": 0.10,
    "bootstrap_replicates": 10000, "bootstrap_seed": 20260930,
}
SOURCES = ["experiments/despike_study/run.py", "experiments/despike_study/analysis.py",
           "experiments/runtime_vs_research/analysis.py", "experiments/submission_retrain/run.py",
           "src/signal_preprocessing.py", "src/model_comparison.py", "src/window_pipeline.py",
           "src/realtime.py", "src/posture_core.py", "src/training_methods.py", "src/app_services.py"]


def arms():
    from src.signal_preprocessing import SignalPreprocessing
    return {"baseline": SignalPreprocessing(), "despike": SignalPreprocessing("despike_v1"),
            "trim1": SignalPreprocessing(None, TRIM_SECONDS),
            "despike_trim1": SignalPreprocessing("despike_v1", TRIM_SECONDS)}


def live_windows(frame, processing, spec):
    """Replay one recording through the app's live buffer; return (end times, windows)."""
    from src.posture_core import SENSOR_COLUMNS
    from src.realtime import LiveConfig, TimeWindowBuffer
    from src.window_pipeline import segment_sensor_frame
    config = LiveConfig(window_seconds=spec.window_seconds,
                        minimum_window_seconds=spec.window_seconds * spec.min_tail_fraction)
    buffer = TimeWindowBuffer(config)
    times, windows, last = [], [], None
    values = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
    for timestamp, row in zip(frame["elapsed_s"].to_numpy(dtype=float), values):
        buffer.append(timestamp, row)
        if not buffer.is_ready() or (last is not None and timestamp - last < config.prediction_interval_seconds):
            continue
        window, _ = segment_sensor_frame(processing.apply_inference(buffer.snapshot()), spec)
        if len(window) != 1:
            raise ValueError("A live snapshot must yield exactly one window")
        times.append(float(timestamp - frame["elapsed_s"].iloc[0]))
        windows.append(window[0])
        last = timestamp
    return times, windows


def prepare(out: Path, stage: Path) -> dict:
    """Build every evaluation/fit/live array with production code; write the feature cache."""
    from src.model_comparison import _dataset
    from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, read_sensor_csv
    from src.window_pipeline import WindowSpec, extract_window_features
    processing = arms()
    data = {}
    for name in ARM_NAMES:
        tick = time.perf_counter()
        data[name] = _dataset(stage, WINDOW_SECONDS, PARTITION_SEED, None, processing[name])
        print(f"[prepare] {name}: {len(data[name]['y'])} evaluation windows, "
              f"{len(data[name]['fit']['y'])} fit windows ({time.perf_counter() - tick:.0f}s)", flush=True)
    base = data["baseline"]
    for name in ARM_NAMES[1:]:
        d = data[name]
        if d["fingerprint"] != base["fingerprint"] or not np.array_equal(d["y"], base["y"]):
            raise ValueError(f"{name}: source records or labels differ from baseline")
        if not d["meta"][["record_id", "start_s"]].equals(base["meta"][["record_id", "start_s"]]):
            raise ValueError(f"{name}: evaluation windows differ from baseline")
    # Evaluation inputs never depend on the training-only trim.
    if not np.array_equal(data["trim1"]["X"], base["X"]) or \
            not np.array_equal(data["despike_trim1"]["X"], data["despike"]["X"]):
        raise ValueError("Training trim changed evaluation inputs")
    if not data["despike"]["fit"]["X"] is data["despike"]["X"] or not base["fit"]["X"] is base["X"]:
        raise ValueError("Untrimmed arms must fit the evaluation rows")
    trim_raw, trim_despike = data["trim1"]["fit"], data["despike_trim1"]["fit"]
    if not trim_raw["meta"][["record_id", "start_s"]].equals(trim_despike["meta"][["record_id", "start_s"]]):
        raise ValueError("Trimmed fit rows differ between raw and despiked arms")

    meta, records = base["meta"], base["source_records"]
    spec = WindowSpec(window_seconds=float(WINDOW_SECONDS))
    live = {"raw": [], "despike": []}
    live_meta = {"y": [], "groups": [], "records": [], "time": []}
    repairs, repaired_records, channel_repairs, samples = 0, 0, Counter(), 0
    tick = time.perf_counter()
    for record in records:
        frame = read_sensor_csv(stage / record["file"])
        count = processing["despike"].count_repairs(frame)
        samples += frame[SENSOR_COLUMNS].size
        repairs += count
        repaired_records += count > 0
        if count:
            from src.signal_preprocessing import despike_v1
            _, mask = despike_v1(frame["elapsed_s"].to_numpy(float), frame[SENSOR_COLUMNS].to_numpy(float))
            channel_repairs.update(dict(zip(SENSOR_COLUMNS, mask.sum(axis=0).astype(int).tolist())))
        times, raw = live_windows(frame, processing["baseline"], spec)
        times_d, despiked = live_windows(frame, processing["despike"], spec)
        if times != times_d:
            raise ValueError("Live replay timing differs between arms")
        live["raw"].extend(raw)
        live["despike"].extend(despiked)
        label = CLASS_NAMES.index(record["label"])
        live_meta["y"] += [label] * len(times)
        live_meta["groups"] += [record["subject"]] * len(times)
        live_meta["records"] += [record["record_id"]] * len(times)
        live_meta["time"] += times
    print(f"[prepare] live replay: {len(live_meta['y'])} predictions ({time.perf_counter() - tick:.0f}s)", flush=True)
    tick = time.perf_counter()
    live_raw_seq, live_despike_seq = np.stack(live["raw"]), np.stack(live["despike"])
    cache = {
        "eval_raw_X": base["X"], "eval_despike_X": data["despike"]["X"],
        "eval_y": base["y"], "eval_groups": meta["subject"].to_numpy(dtype=str),
        "eval_records": meta["record_id"].to_numpy(dtype=str),
        "fit_trim_raw_X": trim_raw["X"], "fit_trim_despike_X": trim_despike["X"],
        "fit_trim_y": trim_raw["y"], "fit_trim_groups": trim_raw["meta"]["subject"].to_numpy(dtype=str),
        "fit_trim_records": trim_raw["meta"]["record_id"].to_numpy(dtype=str),
        "live_raw_X": extract_window_features(live_raw_seq), "live_despike_X": extract_window_features(live_despike_seq),
        "live_y": np.asarray(live_meta["y"]), "live_groups": np.asarray(live_meta["groups"], dtype=str),
        "live_records": np.asarray(live_meta["records"], dtype=str), "live_time": np.asarray(live_meta["time"]),
    }
    print(f"[prepare] live features ({time.perf_counter() - tick:.0f}s)", flush=True)
    cache_dir = out / "cache"
    cache_dir.mkdir()
    for key, value in cache.items():
        np.save(cache_dir / f"{key}.npy", value, allow_pickle=False)
    np.savez_compressed(out / "windows.npz", eval_raw=base["sequences"], eval_despike=data["despike"]["sequences"],
                        fit_trim_raw=trim_raw["sequences"], fit_trim_despike=trim_despike["sequences"],
                        eval_records=cache["eval_records"], eval_start_s=meta["start_s"].to_numpy(),
                        fit_trim_records=cache["fit_trim_records"], fit_trim_start_s=trim_raw["meta"]["start_s"].to_numpy(),
                        live_records=cache["live_records"], live_time=cache["live_time"])
    changed_windows = int(np.any(base["sequences"] != data["despike"]["sequences"], axis=(1, 2)).sum())
    changed_live = int(np.any(live_raw_seq != live_despike_seq, axis=(1, 2)).sum())
    return {
        "dataset_fingerprint": base["fingerprint"], "source_records": records,
        "subjects": sorted(set(cache["eval_groups"].tolist())),
        "records": len(records), "evaluation_windows": int(len(base["y"])),
        "fit_windows": {"untrimmed": int(len(base["y"])), "trimmed": int(len(trim_raw["y"]))},
        "live_predictions": int(len(cache["live_y"])),
        "class_window_counts": {label: int(np.sum(base["y"] == i)) for i, label in enumerate(CLASS_NAMES)},
        "despike_v1_repairs": {"repaired_samples": int(repairs), "total_samples": int(samples),
                               "repaired_fraction": repairs / samples, "records_with_repairs": int(repaired_records),
                               "per_channel": dict(channel_repairs),
                               "evaluation_windows_changed": changed_windows,
                               "live_windows_changed": changed_live},
        "cache_sha256": {key: sha(cache_dir / f"{key}.npy") for key in cache},
        "windows_npz_sha256": sha(out / "windows.npz"),
    }


_CACHE: dict | None = None


def _cache(directory: str) -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = {path.stem: np.load(path, mmap_mode="r", allow_pickle=False) for path in Path(directory).glob("*.npy")}
    return _CACHE


def fit_inputs(cache: dict, arm: str):
    fit_key, prefix = ARM_INPUTS[arm][1]
    return (cache[fit_key], cache[f"{prefix}_y"], cache[f"{prefix}_groups"], cache[f"{prefix}_records"])


def fold_job(job: dict) -> dict:
    from src.posture_core import CLASS_NAMES
    from src.training_methods import build_method_model, fit_method_model
    from src.window_pipeline import aligned_predict_proba
    cache = _cache(job["cache"])
    arm, family, seed, held = job["arm"], job["family"], job["seed"], job["subject"]
    X_fit, y_fit, groups_fit, records_fit = fit_inputs(cache, arm)
    train = np.flatnonzero(groups_fit != held)
    evaluation = np.flatnonzero(cache["eval_groups"] == held)
    live = np.flatnonzero(cache["live_groups"] == held)
    if held in set(groups_fit[train].tolist()) or not len(evaluation) or not len(live):
        raise ValueError("Held-out subject leaked into training or has no rows")
    tick = time.perf_counter()
    # Identical calls to src.model_comparison.loso_evaluation.
    model = build_method_model(family, MODELS[family], seed, "standard")
    fit_method_model(model, family, "standard", np.asarray(X_fit[train]), np.asarray(y_fit[train]),
                     groups=np.asarray(groups_fit[train]), record_ids=np.asarray(records_fit[train]), validation_data=None)
    trained = time.perf_counter() - tick
    probability = aligned_predict_proba(model, np.asarray(cache[ARM_INPUTS[arm][0]][evaluation]), len(CLASS_NAMES))
    live_probability = aligned_predict_proba(model, np.asarray(cache[ARM_INPUTS[arm][2]][live]), len(CLASS_NAMES))
    for values in (probability, live_probability):
        if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1.0):
            raise ValueError("Invalid probability contract")
    return {"kind": "fold", "arm": arm, "family": family, "seed": seed, "subject": held,
            "evaluation_rows": evaluation.tolist(), "probability": probability.tolist(),
            "live_rows": live.tolist(), "live_probability": live_probability.tolist(),
            "training_windows": int(len(train)), "training_seconds": trained}


def reproduction_job(job: dict) -> dict:
    """Run the production loso_evaluation for one configuration (serial, all 32 subjects)."""
    from src.model_comparison import loso_evaluation
    cache = _cache(job["cache"])
    arm, family, seed = job["arm"], job["family"], job["seed"]
    X_fit, y_fit, groups_fit, records_fit = fit_inputs(cache, arm)
    X_eval = np.asarray(cache[ARM_INPUTS[arm][0]])
    everyone = np.ones(len(X_eval), dtype=bool)
    fit = None if ARM_INPUTS[arm][1][1] == "eval" else (
        np.asarray(X_fit), np.asarray(y_fit), np.asarray(groups_fit), np.asarray(records_fit), np.ones(len(y_fit), dtype=bool))
    result = loso_evaluation(family, MODELS[family], seed, "standard", X_eval, np.asarray(cache["eval_y"]),
                             np.asarray(cache["eval_groups"]), np.asarray(cache["eval_records"]), everyone, fit=fit)
    return {"kind": "reproduction", "arm": arm, "family": family, "seed": seed,
            "macro_f1": result["macro_f1"], "confusion_matrix": result["confusion_matrix"]}


def run_job(job: dict) -> dict:
    return reproduction_job(job) if job["kind"] == "reproduction" else fold_job(job)


def present_macro_f1(truth, prediction) -> float:
    from sklearn.metrics import f1_score
    labels = sorted(set(np.asarray(truth).tolist()))
    return float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0))


def analyse(cache: dict, pooled: dict, pooled_live: dict, subjects: list[str]) -> dict:
    from sklearn.metrics import confusion_matrix
    truth, groups, records = (np.asarray(cache[k]) for k in ("eval_y", "eval_groups", "eval_records"))
    live_y, live_records = np.asarray(cache["live_y"]), np.asarray(cache["live_records"])
    unique_records = sorted(set(records.tolist()))
    record_rows = [np.flatnonzero(records == r) for r in unique_records]
    record_truth = np.array([truth[rows[0]] for rows in record_rows])
    metrics = {}
    for (family, arm, seed), probability in pooled.items():
        prediction = probability.argmax(axis=1)
        record_prediction = np.array([probability[rows].mean(axis=0).argmax() for rows in record_rows])
        metrics[family, arm, seed] = {
            "macro_f1": macro_f1(truth, prediction), "accuracy": float((prediction == truth).mean()),
            "per_class_f1": per_class_f1(truth, prediction).tolist(),
            "confusion_matrix": confusion_matrix(truth, prediction, labels=np.arange(7)).tolist(),
            "record_macro_f1": macro_f1(record_truth, record_prediction),
            "subject_present_macro_f1": {s: present_macro_f1(truth[groups == s], prediction[groups == s]) for s in subjects},
            "live": live_stability(pooled_live[family, arm, seed], live_y, live_records),
        }
    comparisons = {}
    for family in MODELS:
        base = [metrics[family, "baseline", s] for s in SEEDS]
        base_pred = pooled[family, "baseline", SEEDS[0]].argmax(axis=1)
        for arm in ARM_NAMES[1:]:
            rows = [metrics[family, arm, s] for s in SEEDS]
            bootstrap = paired_subject_bootstrap(truth, base_pred, pooled[family, arm, SEEDS[0]].argmax(axis=1), groups,
                                                 replicates=CRITERIA["bootstrap_replicates"], seed=CRITERIA["bootstrap_seed"])
            decision = decide([m["macro_f1"] for m in base], [m["macro_f1"] for m in rows],
                              [m["per_class_f1"] for m in base], [m["per_class_f1"] for m in rows],
                              bootstrap["lower_2_5"],
                              [m["live"]["smoothed_changes_per_minute"] for m in base],
                              [m["live"]["smoothed_changes_per_minute"] for m in rows], CRITERIA)
            subject_delta = {s: float(np.mean([r["subject_present_macro_f1"][s] - b["subject_present_macro_f1"][s]
                                               for r, b in zip(rows, base)])) for s in subjects}
            comparisons[f"{family}:{arm}_minus_baseline"] = {
                **decision, "bootstrap": bootstrap,
                "record_macro_f1_delta_by_seed": [r["record_macro_f1"] - b["record_macro_f1"] for r, b in zip(rows, base)],
                "live_smoothed_macro_f1_delta_by_seed": [r["live"]["smoothed_macro_f1"] - b["live"]["smoothed_macro_f1"]
                                                         for r, b in zip(rows, base)],
                "subjects_improved": int(sum(v > 1e-12 for v in subject_delta.values())),
                "subjects_worse": int(sum(v < -1e-12 for v in subject_delta.values())),
                "subject_mean_delta": subject_delta}
    return {"metrics": {f"{f}:{a}:{s}": v for (f, a, s), v in metrics.items()}, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    out = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
    if out.exists():
        raise SystemExit(f"Output exists; use a new folder: {out}")
    out.mkdir(parents=True)
    started = now()
    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    before = protected_state()
    write(out / "protected_before.json", before)
    for relative in SOURCES:
        target = out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)

    from src.app_services import DatasetService
    from experiments.submission_retrain.run import select_records
    service = DatasetService(ROOT, data_directory=ROOT / "data")
    eligible, selected, rule_excluded = select_records(service)
    stage_parent = Path(tempfile.mkdtemp(prefix="despike_study_stage_"))
    stage = service.stage_for_training(selected, stage_parent / "training_data")
    (out / "staging").mkdir()
    for name in ("staged_sources.json", "staging_summary.json"):
        shutil.copy2(stage / name, out / "staging" / name)
    cohort = prepare(out, stage)
    cohort.update(eligible_records=len(eligible), selected_records=len(selected), rule_excluded=rule_excluded,
                  selection_rule="app training-eligible labeled records (DatasetService, training_exclusions applied); "
                                 "subjects without all seven postures excluded (submission-retrain rule)")
    write(out / "cohort.json", cohort)

    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    processing = arms()
    protocol = {
        "schema_version": 1, "started_at": started,
        "question": ("Does versioned despike_v1 (train+inference) and/or removing the first 1 s of each training "
                     "recording improve 32-subject LOSO Macro F1 or live prediction stability?"),
        "arms": {name: processing[name].rule() for name in ARM_NAMES},
        "arm_inputs": ARM_INPUTS, "models": MODELS, "window_seconds": WINDOW_SECONDS, "seeds": SEEDS,
        "primary_comparison": PRIMARY, "criteria": CRITERIA,
        "performance_rule": ("Arm improves a model if pooled LOSO Macro F1 (arm-baseline) averaged over seeds >= +1.0%p, "
                             "every seed delta > 0, primary-seed paired subject bootstrap 2.5% bound > 0, and no "
                             "seed-mean per-class F1 drop > 5%p."),
        "stability_rule": ("Secondary: arm is a non-inferior stability measure if mean delta >= -0.5%p, bootstrap 2.5% "
                           "bound >= -1.5%p, smoothed live label changes per minute lower than baseline in every seed with "
                           "mean relative reduction >= 10%, and no per-class F1 drop > 5%p."),
        "adoption": ("Only the RBF-SVM (runtime recipe) comparisons can support a runtime recommendation; Extra Trees is "
                     "secondary. Six comparisons are not multiplicity-corrected; the preregistered primary is "
                     "rbf_svm despike vs baseline. A pass is a recommendation to train a candidate through the "
                     "TrainingJob/validation/approval workflow, never an automatic model change."),
        "evaluation": ("Each subject held out once; fitted on the other 31. Held-out windows are identical across arms "
                       "except for the despike transform (inference contract); trim never touches evaluation. "
                       "Live replay: src.realtime.TimeWindowBuffer, 3 s window, 2.85 s warm-up, 1 s interval, "
                       "3-step probability smoothing restarted per recording."),
        "hyperparameters": ("Fixed: RBF-SVM = active runtime model's selected params; ET = research baseline params. "
                            "Both were chosen earlier on data including these subjects; all arms share this bias."),
        "parameter_origin": ("despike_v1 thresholds fixed from the label-free raw-difference distribution before any "
                             "model fit (99th pct |diff| ~= 0.083); trim 1.0 s fixed from the audit's concentration of "
                             "spikes in the first 0.5-1 s. No other values are tried."),
        "cohort": {k: cohort[k] for k in ("dataset_fingerprint", "records", "evaluation_windows", "fit_windows",
                                           "live_predictions", "subjects", "cache_sha256")},
        "exposure": "All subjects appeared in earlier evaluations; retrospective CSV replay, not new-user or physical-sensor evidence.",
        "source_revision": git("rev-parse", "HEAD").strip(), "dirty_state": git("status", "--short"),
        "source_hashes": {r: sha(ROOT / r) for r in SOURCES}, "command": [sys.executable, *sys.argv],
        "environment": {"python": sys.version, "platform": platform.platform(), "workers": args.workers,
                        "threads_per_worker": 1,
                        "packages": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                                   text=True).stdout.split()},
        "models_saved": False, "active_model_changed": False, "raw_data_mutated": False,
    }
    write(out / "protocol.json", protocol)
    write(out / "protocol_lock.json", {"sha256": sha(out / "protocol.json"), "locked_at": now()})

    cache_dir = str(out / "cache")
    cache = _cache(cache_dir)
    subjects = cohort["subjects"]
    jobs = [{"kind": "reproduction", "arm": arm, "family": "extra_trees", "seed": SEEDS[0], "cache": cache_dir}
            for arm in ("baseline", "despike_trim1")]
    jobs += [{"kind": "fold", "arm": arm, "family": family, "seed": seed, "subject": subject, "cache": cache_dir}
             for family in MODELS for seed in SEEDS for arm in ARM_NAMES for subject in subjects]
    n_eval, n_live = len(cache["eval_y"]), len(cache["live_y"])
    pooled = {(f, a, s): np.full((n_eval, 7), np.nan) for f in MODELS for a in ARM_NAMES for s in SEEDS}
    pooled_live = {(f, a, s): np.full((n_live, 7), np.nan) for f in MODELS for a in ARM_NAMES for s in SEEDS}
    folds, reproductions = [], []
    tick = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row["kind"] == "reproduction":
                reproductions.append(row)
                continue
            key = (row["family"], row["arm"], row["seed"])
            for target, rows, values in ((pooled[key], row["evaluation_rows"], row["probability"]),
                                         (pooled_live[key], row["live_rows"], row["live_probability"])):
                if not np.isnan(target[rows]).all():
                    raise ValueError("A row was predicted twice")
                target[rows] = np.asarray(values)
            folds.append({k: row[k] for k in ("family", "arm", "seed", "subject", "training_windows", "training_seconds")})
            if done % 32 == 0:
                print(f"{done}/{len(jobs)} jobs done ({time.perf_counter() - tick:.0f}s)", flush=True)
    training_finished = now()
    for key, values in [*pooled.items(), *pooled_live.items()]:
        if np.isnan(values).any():
            raise ValueError(f"Missing LOSO predictions for {key}")
    write(out / "folds.json", folds)
    np.savez_compressed(out / "loso_predictions.npz", truth=np.asarray(cache["eval_y"]),
                        groups=np.asarray(cache["eval_groups"]), record_ids=np.asarray(cache["eval_records"]),
                        live_truth=np.asarray(cache["live_y"]), live_groups=np.asarray(cache["live_groups"]),
                        live_record_ids=np.asarray(cache["live_records"]), live_time=np.asarray(cache["live_time"]),
                        **{f"{f}__{a}__{s}": v for (f, a, s), v in pooled.items()},
                        **{f"live__{f}__{a}__{s}": v for (f, a, s), v in pooled_live.items()})

    summary = {"started_at": started, "training_finished_at": training_finished, **analyse(cache, pooled, pooled_live, subjects)}
    checks = []
    for row in reproductions:
        mine = summary["metrics"][f"{row['family']}:{row['arm']}:{row['seed']}"]
        checks.append({**row, "fold_loop_macro_f1": mine["macro_f1"],
                       "identical": bool(abs(row["macro_f1"] - mine["macro_f1"]) < 1e-12
                                         and row["confusion_matrix"] == mine["confusion_matrix"])})
    summary["production_loso_reproduction"] = checks
    primary = summary["comparisons"][f"{PRIMARY['model']}:{PRIMARY['arm']}_minus_baseline"]
    summary["primary_decision"] = {"comparison": PRIMARY, "performance_pass": primary["performance_pass"],
                                   "stability_pass": primary["stability_pass"]}
    summary["fold_training_seconds_median"] = {
        f"{f}:{a}": float(np.median([r["training_seconds"] for r in folds if r["family"] == f and r["arm"] == a]))
        for f in MODELS for a in ARM_NAMES}
    after = protected_state()
    write(out / "protected_after.json", after)
    summary["protected_unchanged"] = before == after
    summary["sources_unchanged"] = all(sha(ROOT / r) == protocol["source_hashes"][r] for r in SOURCES)
    summary["protocol_unchanged"] = sha(out / "protocol.json") == read(out / "protocol_lock.json")["sha256"]
    summary["finished_at"] = now()
    write(out / "summary.json", summary)
    # The feature cache is regenerable from windows.npz / source code; its hashes are in cohort.json.
    global _CACHE
    _CACHE = None
    shutil.rmtree(out / "cache")
    shutil.rmtree(stage_parent)
    print({"primary": summary["primary_decision"], "reproduction": [c["identical"] for c in checks],
           "protected_unchanged": summary["protected_unchanged"]}, flush=True)


if __name__ == "__main__":
    main()
