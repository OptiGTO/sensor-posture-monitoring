"""32-subject LOSO retrain after the user's review of the 80 all-wrong recordings.

Input: research/results/2026-09-28_130815_label_review_all_wrong/review_decisions_2026-09-30.json
(delete 32 / hold 38 / posture_mismatch 10, with the reviewer's normal/abnormal intervals).

Training conditions (only fitted rows change; data/ is never modified):

  baseline               all 724 records (same cohort and recipe as the despike study baseline)
  drop_delete            the 32 'delete' recordings removed from training
  drop_invalid_segments  full review: 'delete' recordings removed except reviewer-stated normal
                         segments; stated abnormal segments of hold/mismatch recordings removed
  drop_delete_mismatch   'delete' and the 10 'posture_mismatch' recordings removed

Every held-out subject (fitted on the other 31) is predicted on all its original
windows in every condition. Metrics are then reported on fixed evaluation sets:

  all                    every window, recorded labels (comparable with earlier LOSO reports)
  valid                  windows the review did not mark invalid (primary)
  valid_no_mismatch      valid minus posture_mismatch recordings
  review_ineligible      the 18 subjects never shown in the 2026-09-22 008 development
                         validation, so none of their recordings could be reviewed

Models: active runtime RBF-SVM (C=1, gamma=scale) and research Extra Trees, three
seeds. A live replay (3 s buffer, 1 s interval, 3-step smoothing) is predicted by
the same fold models. protocol.json (conditions, criteria, review hash, cohort
hashes) is locked before any fit. data/ and research/models/ are compared before
and after. No model is saved, registered or activated.

Usage (new output folder only):
  .venv/bin/python -B -W ignore -m experiments.label_review_retrain.run --output research/results/<YYYY-MM-DD_HHMMSS>_label_review_retrain
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.label_review_retrain.analysis import (  # noqa: E402
    CONDITIONS, condition_fit_masks, decide, evaluation_masks, macro_f1, paired_subject_bootstrap,
    per_class_f1, present_macro_f1, review_masks, smoothed_by_session)
from experiments.runtime_vs_research.run import now, protected_state, read, sha, write  # noqa: E402

REVIEW_DIR = "research/results/2026-09-28_130815_label_review_all_wrong"
REVIEW_DECISIONS = f"{REVIEW_DIR}/review_decisions_2026-09-30.json"
REVIEW_DECISIONS_SHA256 = "9a5fc64fe8b65a3b4ca25ca9810e56922b825098fce105962351aa49caeabf51"
ORIGIN_PROTOCOL = "research/results/2026-09-22_015526_iterative_008/protocol.json"
DESPIKE_RUN = "research/results/2026-09-30_002935_despike_study"
EXPECTED_FINGERPRINT = "00ed992fa24c6e8265712d7c761b2e0d14f8b78750f09ecc7deed25c71aa3cc4"
WINDOW_SECONDS = 3
SEEDS = [20260826, 20260827, 20260828]
PARTITION_SEED = 20260826  # only used by _dataset's partition bookkeeping; LOSO ignores it
MODELS = {
    "rbf_svm": {"C": 1.0, "gamma": "scale"},
    "extra_trees": {"n_estimators": 200, "min_samples_leaf": 8, "max_depth": 8, "max_features": 0.5},
}
EVALUATION_SETS = ("all", "valid", "valid_no_mismatch", "review_ineligible")
LIVE_EVALUATION_SETS = ("all", "valid", "review_ineligible")
PRIMARY = {"model": "rbf_svm", "condition": "drop_invalid_segments", "evaluation_set": "valid",
           "non_inferiority_set": "review_ineligible"}
CRITERIA = {
    "minimum_mean_delta": 0.01, "maximum_per_class_drop": 0.05,
    "non_inferiority_mean_margin": 0.005, "non_inferiority_bootstrap_margin": 0.015,
    "bootstrap_replicates": 10000, "bootstrap_seed": 20260930,
}
SOURCES = ["experiments/label_review_retrain/run.py", "experiments/label_review_retrain/analysis.py",
           "experiments/label_review_retrain/encode_review.py", "experiments/despike_study/run.py",
           "experiments/despike_study/analysis.py", "experiments/runtime_vs_research/analysis.py",
           "experiments/runtime_vs_research/run.py", "experiments/submission_retrain/run.py",
           "src/signal_preprocessing.py", "src/model_comparison.py", "src/window_pipeline.py",
           "src/realtime.py", "src/posture_core.py", "src/training_methods.py", "src/app_services.py"]


def review_ineligible_subjects(subjects: list[str]) -> list[str]:
    """Subjects never in any 008 development validation split (review candidates came only from those)."""
    protocol = read(ROOT / ORIGIN_PROTOCOL)
    shown = set().union(*(set(p["validation"]) for p in protocol["partitions"]))
    return sorted(set(subjects) - shown)


def prepare(out: Path, stage: Path, decisions: list[dict]) -> dict:
    from experiments.despike_study.run import live_windows
    from src.model_comparison import _dataset
    from src.posture_core import CLASS_NAMES, read_sensor_csv
    from src.signal_preprocessing import SignalPreprocessing
    from src.window_pipeline import WindowSpec, extract_window_features
    tick = time.perf_counter()
    base = _dataset(stage, WINDOW_SECONDS, PARTITION_SEED, None, SignalPreprocessing())
    if base["fingerprint"] != EXPECTED_FINGERPRINT:
        raise ValueError("Cohort fingerprint differs from the despike-study cohort")
    print(f"[prepare] {len(base['y'])} windows ({time.perf_counter() - tick:.0f}s)", flush=True)
    records, meta = base["source_records"], base["meta"]
    by_sha = {}
    for record in records:
        by_sha.setdefault(record["sha256"], []).append(record)
    review_by_record, review_rows = {}, []
    for item in decisions:
        matches = by_sha.get(item["source_sha256"], [])
        if len(matches) != 1:
            raise ValueError(f"{item['review_id']}: source hash must match exactly one cohort record")
        record = matches[0]
        if (record["subject"], record["label"]) != (item["subject"], item["label"]):
            raise ValueError(f"{item['review_id']}: subject/label differ from the cohort record")
        review_by_record[record["record_id"]] = item
        review_rows.append({"review_id": item["review_id"], "cohort_record_id": record["record_id"],
                            "file": record["file"], "decision": item["decision"],
                            "window_count": record["window_count"]})
    record_ids = meta["record_id"].to_numpy(dtype=str)
    starts = meta["start_s"].to_numpy(dtype=float)
    groups = meta["subject"].to_numpy(dtype=str)
    masks = review_masks(record_ids, starts, review_by_record)
    fit = condition_fit_masks(masks)
    subjects = sorted(set(groups.tolist()))
    ineligible = review_ineligible_subjects(subjects)
    reviewed_subjects = sorted({item["subject"] for item in decisions})
    if len(ineligible) != 18 or set(ineligible) & set(reviewed_subjects):
        raise ValueError("Review-ineligible subject set is not the expected 18 unreviewed subjects")
    evaluation = evaluation_masks(masks)
    evaluation.pop("unreviewed")
    evaluation["review_ineligible"] = np.isin(groups, ineligible)
    y = base["y"]
    for name, mask in fit.items():
        for subject in subjects:
            train = mask & (groups != subject)
            if set(y[train].tolist()) != set(range(len(CLASS_NAMES))):
                raise ValueError(f"{name}: a LOSO fold lacks a class")

    spec = WindowSpec(window_seconds=float(WINDOW_SECONDS))
    live, live_meta = [], {"y": [], "groups": [], "records": [], "time": []}
    tick = time.perf_counter()
    for record in records:
        frame = read_sensor_csv(stage / record["file"])
        times, windows = live_windows(frame, SignalPreprocessing(), spec)
        live.extend(windows)
        live_meta["y"] += [CLASS_NAMES.index(record["label"])] * len(times)
        live_meta["groups"] += [record["subject"]] * len(times)
        live_meta["records"] += [record["record_id"]] * len(times)
        live_meta["time"] += times
    live_records, live_time = np.asarray(live_meta["records"], dtype=str), np.asarray(live_meta["time"])
    live_groups = np.asarray(live_meta["groups"], dtype=str)
    # A live prediction at time t covers the buffered [t - window, t].
    live_masks = review_masks(live_records, live_time - WINDOW_SECONDS, review_by_record)
    live_evaluation = {"all": np.ones(len(live_records), dtype=bool), "valid": ~live_masks["invalid_window"],
                       "review_ineligible": np.isin(live_groups, ineligible)}
    print(f"[prepare] live replay: {len(live_records)} predictions ({time.perf_counter() - tick:.0f}s)", flush=True)
    cache = {"X": base["X"], "y": y, "groups": groups, "records": record_ids, "start_s": starts,
             "live_X": extract_window_features(np.stack(live)), "live_y": np.asarray(live_meta["y"]),
             "live_groups": live_groups, "live_records": live_records, "live_time": live_time,
             **{f"fit__{k}": v for k, v in fit.items()}, **{f"eval__{k}": v for k, v in evaluation.items()},
             **{f"live_eval__{k}": v for k, v in live_evaluation.items()},
             **{f"mask__{k}": v for k, v in masks.items()}}
    cache_dir = out / "cache"
    cache_dir.mkdir()
    for key, value in cache.items():
        np.save(cache_dir / f"{key}.npy", value, allow_pickle=False)
    decision_of = {r: review_by_record[r]["decision"] if r in review_by_record else "unreviewed" for r in set(record_ids)}
    return {
        "dataset_fingerprint": base["fingerprint"], "source_records": records, "subjects": subjects,
        "records": len(records), "evaluation_windows": int(len(y)), "live_predictions": int(len(live_records)),
        "review_mapping": review_rows, "reviewed_subjects": reviewed_subjects, "review_ineligible_subjects": ineligible,
        "fit_windows": {k: int(v.sum()) for k, v in fit.items()},
        "fit_records": {k: len({r for r, keep in zip(record_ids, v) if keep}) for k, v in fit.items()},
        "evaluation_windows_by_set": {k: int(v.sum()) for k, v in evaluation.items()},
        "evaluation_subjects_by_set": {k: len(set(groups[v].tolist())) for k, v in evaluation.items()},
        "live_predictions_by_set": {k: int(v.sum()) for k, v in live_evaluation.items()},
        "windows_by_decision": {d: int(sum(decision_of[r] == d for r in record_ids))
                                for d in ("delete", "hold", "posture_mismatch", "unreviewed")},
        "removed_class_windows": {k: {label: int(np.sum((~v) & (y == i))) for i, label in enumerate(CLASS_NAMES)}
                                  for k, v in fit.items()},
        "class_window_counts": {label: int(np.sum(y == i)) for i, label in enumerate(CLASS_NAMES)},
        "cache_sha256": {key: sha(cache_dir / f"{key}.npy") for key in cache},
    }


_CACHE: dict | None = None


def _cache(directory: str) -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = {path.stem: np.load(path, mmap_mode="r", allow_pickle=False) for path in Path(directory).glob("*.npy")}
    return _CACHE


def fold_job(job: dict) -> dict:
    from src.posture_core import CLASS_NAMES
    from src.training_methods import build_method_model, fit_method_model
    from src.window_pipeline import aligned_predict_proba
    cache = _cache(job["cache"])
    condition, family, seed, held = job["condition"], job["family"], job["seed"], job["subject"]
    groups = np.asarray(cache["groups"])
    train = np.flatnonzero(np.asarray(cache[f"fit__{condition}"]) & (groups != held))
    evaluation = np.flatnonzero(groups == held)
    live = np.flatnonzero(np.asarray(cache["live_groups"]) == held)
    if held in set(groups[train].tolist()) or not len(evaluation) or not len(live):
        raise ValueError("Held-out subject leaked into training or has no rows")
    tick = time.perf_counter()
    # Same calls as src.model_comparison.loso_evaluation.
    model = build_method_model(family, MODELS[family], seed, "standard")
    fit_method_model(model, family, "standard", np.asarray(cache["X"][train]), np.asarray(cache["y"][train]),
                     groups=groups[train], record_ids=np.asarray(cache["records"][train]), validation_data=None)
    trained = time.perf_counter() - tick
    probability = aligned_predict_proba(model, np.asarray(cache["X"][evaluation]), len(CLASS_NAMES))
    live_probability = aligned_predict_proba(model, np.asarray(cache["live_X"][live]), len(CLASS_NAMES))
    for values in (probability, live_probability):
        if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1.0):
            raise ValueError("Invalid probability contract")
    return {"condition": condition, "family": family, "seed": seed, "subject": held,
            "evaluation_rows": evaluation.tolist(), "probability": probability.tolist(),
            "live_rows": live.tolist(), "live_probability": live_probability.tolist(),
            "training_windows": int(len(train)), "training_seconds": trained}


def metric_block(truth, prediction, groups) -> dict:
    from sklearn.metrics import confusion_matrix
    return {"macro_f1": macro_f1(truth, prediction), "accuracy": float((prediction == truth).mean()),
            "windows": int(len(truth)), "per_class_f1": per_class_f1(truth, prediction).tolist(),
            "confusion_matrix": confusion_matrix(truth, prediction, labels=np.arange(7)).tolist(),
            "subject_present_macro_f1": {s: present_macro_f1(truth[groups == s], prediction[groups == s])
                                         for s in sorted(set(groups.tolist()))}}


def analyse(cache: dict, pooled: dict, pooled_live: dict, families, seeds) -> dict:
    truth, groups, records = (np.asarray(cache[k]) for k in ("y", "groups", "records"))
    live_y, live_records, live_groups = (np.asarray(cache[k]) for k in ("live_y", "live_records", "live_groups"))
    sets = {name: np.asarray(cache[f"eval__{name}"]) for name in EVALUATION_SETS}
    live_sets = {name: np.asarray(cache[f"live_eval__{name}"]) for name in LIVE_EVALUATION_SETS}
    decision_masks = {"delete": np.asarray(cache["mask__delete_record"]),
                      "posture_mismatch": np.asarray(cache["mask__mismatch_record"]),
                      "hold": np.asarray(cache["mask__reviewed"]) & ~np.asarray(cache["mask__delete_record"])
                      & ~np.asarray(cache["mask__mismatch_record"]),
                      "unreviewed": ~np.asarray(cache["mask__reviewed"])}
    minutes = len(live_y) / 60.0
    metrics = {}
    for (family, condition, seed), probability in pooled.items():
        prediction = probability.argmax(axis=1)
        live_smoothed = smoothed_by_session(pooled_live[family, condition, seed], live_records)
        same = live_records[1:] == live_records[:-1]
        metrics[family, condition, seed] = {
            "sets": {name: metric_block(truth[m], prediction[m], groups[m]) for name, m in sets.items()},
            "window_accuracy_by_review_decision": {d: float((prediction[m] == truth[m]).mean())
                                                   for d, m in decision_masks.items()},
            "live": {**{f"smoothed_macro_f1_{name}": macro_f1(live_y[m], live_smoothed[m]) for name, m in live_sets.items()},
                     "smoothed_changes_per_minute": float(np.sum((live_smoothed[1:] != live_smoothed[:-1]) & same) / minutes)},
        }
    comparisons = {}
    for family in families:
        for condition in CONDITIONS[1:]:
            for name in EVALUATION_SETS:
                m = sets[name]
                base = [metrics[family, "baseline", s]["sets"][name] for s in seeds]
                arm = [metrics[family, condition, s]["sets"][name] for s in seeds]
                bootstrap = paired_subject_bootstrap(
                    truth[m], pooled[family, "baseline", seeds[0]][m].argmax(1),
                    pooled[family, condition, seeds[0]][m].argmax(1), groups[m],
                    replicates=CRITERIA["bootstrap_replicates"], seed=CRITERIA["bootstrap_seed"])
                decision = decide([b["macro_f1"] for b in base], [a["macro_f1"] for a in arm],
                                  [b["per_class_f1"] for b in base], [a["per_class_f1"] for a in arm],
                                  bootstrap["lower_2_5"], CRITERIA)
                people = sorted(base[0]["subject_present_macro_f1"])
                subject_delta = {p: float(np.mean([a["subject_present_macro_f1"][p] - b["subject_present_macro_f1"][p]
                                                   for a, b in zip(arm, base)])) for p in people}
                comparisons[f"{family}:{condition}:{name}"] = {
                    **decision, "bootstrap": bootstrap,
                    "subjects_improved": int(sum(v > 1e-12 for v in subject_delta.values())),
                    "subjects_worse": int(sum(v < -1e-12 for v in subject_delta.values())),
                    "subject_mean_delta": subject_delta,
                    "live_smoothed_macro_f1_delta_by_seed": {
                        live: [metrics[family, condition, s]["live"][f"smoothed_macro_f1_{live}"]
                               - metrics[family, "baseline", s]["live"][f"smoothed_macro_f1_{live}"] for s in seeds]
                        for live in LIVE_EVALUATION_SETS}}
    return {"metrics": {f"{f}:{c}:{s}": v for (f, c, s), v in metrics.items()}, "comparisons": comparisons}


def baseline_reproduction(cache: dict, pooled: dict, families, seeds) -> list[dict]:
    """Baseline arm vs the despike study's stored baseline LOSO probabilities (same cohort and recipe)."""
    path = ROOT / DESPIKE_RUN / "loso_predictions.npz"
    rows = []
    with np.load(path, allow_pickle=False) as stored:
        aligned = all(np.array_equal(stored[a], np.asarray(cache[b])) for a, b in
                      (("truth", "y"), ("groups", "groups"), ("record_ids", "records")))
        for family in families:
            for seed in seeds:
                key = f"{family}__baseline__{seed}"
                mine = pooled[family, "baseline", seed]
                diff = float(np.abs(stored[key] - mine).max()) if aligned else None
                rows.append({"key": key, "rows_aligned": aligned, "max_abs_probability_difference": diff,
                             "same_argmax": bool(aligned and np.array_equal(stored[key].argmax(1), mine.argmax(1)))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--smoke", action="store_true",
                        help="plumbing check only: Extra Trees, one seed, three subjects; output outside research/")
    args = parser.parse_args()
    out = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
    if out.exists():
        raise SystemExit(f"Output exists; use a new folder: {out}")
    if args.smoke == out.resolve().is_relative_to((ROOT / "research").resolve()):
        raise SystemExit("Smoke runs must write outside research/; real runs must write under research/results/")
    out.mkdir(parents=True)
    started = now()
    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    before = protected_state()
    write(out / "protected_before.json", before)
    for relative in SOURCES:
        target = out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    decisions_path = ROOT / REVIEW_DECISIONS
    if sha(decisions_path) != REVIEW_DECISIONS_SHA256:
        raise SystemExit("Review decision file differs from the encoded version")
    decisions = read(decisions_path)["records"]

    from experiments.submission_retrain.run import select_records
    from src.app_services import DatasetService
    service = DatasetService(ROOT, data_directory=ROOT / "data")
    eligible, selected, rule_excluded = select_records(service)
    stage_parent = Path(tempfile.mkdtemp(prefix="label_review_retrain_stage_"))
    stage = service.stage_for_training(selected, stage_parent / "training_data")
    (out / "staging").mkdir()
    for name in ("staged_sources.json", "staging_summary.json"):
        shutil.copy2(stage / name, out / "staging" / name)
    cohort = prepare(out, stage, decisions)
    cohort.update(eligible_records=len(eligible), selected_records=len(selected), rule_excluded=rule_excluded)
    write(out / "cohort.json", cohort)

    families = ["extra_trees"] if args.smoke else list(MODELS)
    seeds = SEEDS[:1] if args.smoke else SEEDS
    subjects = cohort["subjects"]
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    protocol = {
        "schema_version": 1, "started_at": started, "smoke": args.smoke,
        "question": ("Does removing the recordings/segments the user marked in the review of the 80 all-wrong "
                     "recordings improve 32-subject LOSO Macro F1?"),
        "review": {"decisions": REVIEW_DECISIONS, "sha256": REVIEW_DECISIONS_SHA256,
                   "counts": read(decisions_path)["counts"]},
        "conditions": {
            "baseline": "all 724 cohort records",
            "drop_delete": "fit rows exclude every window of the 32 'delete' recordings",
            "drop_invalid_segments": ("fit rows exclude windows judged invalid: 'delete' recordings except windows fully "
                                      "inside a reviewer-stated normal interval; any window overlapping a stated "
                                      "abnormal interval (primary)"),
            "drop_delete_mismatch": "fit rows exclude the 32 'delete' and 10 'posture_mismatch' recordings",
        },
        "evaluation_sets": {
            "all": "every held-out window with recorded labels",
            "valid": "windows not judged invalid by the review (same rows for every condition; primary)",
            "valid_no_mismatch": "valid minus posture_mismatch recordings",
            "review_ineligible": ("18 subjects never in any 008 development validation split, hence never review "
                                  "candidates; their evaluation windows are unaffected by review selection"),
        },
        "primary": PRIMARY, "criteria": CRITERIA, "models": MODELS, "window_seconds": WINDOW_SECONDS, "seeds": seeds,
        "decision_rule": ("Primary passes if, for RBF-SVM drop_invalid_segments vs baseline on the 'valid' set: seed-mean "
                          "pooled Macro F1 delta >= +1.0%p, every seed delta > 0, primary-seed paired subject bootstrap "
                          "2.5% bound > 0, no seed-mean per-class F1 drop > 5%p; AND on 'review_ineligible' it is "
                          "non-inferior (mean delta >= -0.5%p, bootstrap 2.5% bound >= -1.5%p, no class drop > 5%p). "
                          "All other model x condition x set comparisons are descriptive (no multiplicity correction)."),
        "adoption": ("A pass recommends recording the review in research/models/training_exclusions.json through the "
                     "existing workflow and retraining a candidate via TrainingJob/validation/approval; never an "
                     "automatic data or model change."),
        "evaluation": ("32-subject LOSO; each subject fitted on the other 31 with the condition's fit rows. Held-out "
                       "predictions cover every window; sets are masks over the same predictions. Live replay: "
                       "src.realtime.TimeWindowBuffer, 3 s, 1 s interval, 3-step smoothing restarted per recording; "
                       "a live row is invalid if [t-3, t] is invalid under the same review rule."),
        "known_biases": [
            "Review candidates were the 80 recordings the ET model got entirely wrong in 008 development validation; "
            "the reviewer saw model predictions (not blind).",
            "Excluding invalid windows from evaluation removes rows that models tend to get wrong, inflating absolute "
            "scores on 'valid'; condition deltas on the same rows remain comparable.",
            "Only 14 subjects could be reviewed; the review_ineligible set (18 subjects) is the unselected check.",
            "Hyperparameters were chosen earlier on data including these subjects; all conditions share this bias.",
        ],
        "cohort": {k: cohort[k] for k in ("dataset_fingerprint", "records", "evaluation_windows", "live_predictions",
                                           "fit_windows", "fit_records", "evaluation_windows_by_set",
                                           "review_ineligible_subjects", "reviewed_subjects", "cache_sha256")},
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
    fold_subjects = subjects[:3] if args.smoke else subjects
    jobs = [{"condition": c, "family": f, "seed": s, "subject": p, "cache": cache_dir}
            for f in families for s in seeds for c in CONDITIONS for p in fold_subjects]
    # Longest jobs first keeps the pool busy at the end.
    jobs.sort(key=lambda j: (j["family"] != "rbf_svm", j["condition"] != "baseline"))
    n_eval, n_live = len(cache["y"]), len(cache["live_y"])
    pooled = {(f, c, s): np.full((n_eval, 7), np.nan) for f in families for c in CONDITIONS for s in seeds}
    pooled_live = {(f, c, s): np.full((n_live, 7), np.nan) for f in families for c in CONDITIONS for s in seeds}
    folds = []
    tick = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fold_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            key = (row["family"], row["condition"], row["seed"])
            for target, rows, values in ((pooled[key], row["evaluation_rows"], row["probability"]),
                                         (pooled_live[key], row["live_rows"], row["live_probability"])):
                if not np.isnan(target[rows]).all():
                    raise ValueError("A row was predicted twice")
                target[rows] = np.asarray(values)
            folds.append({k: row[k] for k in ("family", "condition", "seed", "subject", "training_windows",
                                              "training_seconds")})
            if done % 16 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} folds done ({time.perf_counter() - tick:.0f}s)", flush=True)
    training_finished = now()
    write(out / "folds.json", folds)
    if args.smoke:
        held = np.isin(np.asarray(cache["groups"]), fold_subjects)
        live_held = np.isin(np.asarray(cache["live_groups"]), fold_subjects)
        for values in pooled.values():
            if np.isnan(values[held]).any() or not np.isnan(values[~held]).all():
                raise ValueError("Smoke fold rows are incomplete")
        for values in pooled_live.values():
            if np.isnan(values[live_held]).any():
                raise ValueError("Smoke live rows are incomplete")
        # Exercise the analysis path: fill the other subjects from the stored despike-study baseline.
        with np.load(ROOT / DESPIKE_RUN / "loso_predictions.npz", allow_pickle=False) as stored:
            for (f, c, s), values in pooled.items():
                values[~held] = stored[f"{f}__baseline__{s}"][~held]
            for (f, c, s), values in pooled_live.items():
                values[~live_held] = stored[f"live__{f}__baseline__{s}"][~live_held]
        smoke = {"analysis": analyse(cache, pooled, pooled_live, families, seeds),
                 "baseline_reproduction_held_subjects": baseline_reproduction(cache, pooled, families, seeds)}
        write(out / "summary_smoke.json", smoke)
        print({"smoke_ok": True, "folds": len(folds), "fit_windows": cohort["fit_windows"],
               "evaluation_windows_by_set": cohort["evaluation_windows_by_set"],
               "reproduction": smoke["baseline_reproduction_held_subjects"]}, flush=True)
    else:
        for key, values in [*pooled.items(), *pooled_live.items()]:
            if np.isnan(values).any():
                raise ValueError(f"Missing LOSO predictions for {key}")
        np.savez_compressed(out / "loso_predictions.npz", truth=np.asarray(cache["y"]),
                            groups=np.asarray(cache["groups"]), record_ids=np.asarray(cache["records"]),
                            start_s=np.asarray(cache["start_s"]), live_truth=np.asarray(cache["live_y"]),
                            live_groups=np.asarray(cache["live_groups"]),
                            live_record_ids=np.asarray(cache["live_records"]), live_time=np.asarray(cache["live_time"]),
                            **{f"eval__{n}": np.asarray(cache[f"eval__{n}"]) for n in EVALUATION_SETS},
                            **{f"fit__{c}": np.asarray(cache[f"fit__{c}"]) for c in CONDITIONS},
                            **{f"{f}__{c}__{s}": v for (f, c, s), v in pooled.items()},
                            **{f"live__{f}__{c}__{s}": v for (f, c, s), v in pooled_live.items()})
        summary = {"started_at": started, "training_finished_at": training_finished,
                   **analyse(cache, pooled, pooled_live, families, seeds)}
        summary["baseline_reproduction_vs_despike_study"] = baseline_reproduction(cache, pooled, families, seeds)
        primary = summary["comparisons"][f"{PRIMARY['model']}:{PRIMARY['condition']}:{PRIMARY['evaluation_set']}"]
        guard = summary["comparisons"][f"{PRIMARY['model']}:{PRIMARY['condition']}:{PRIMARY['non_inferiority_set']}"]
        summary["primary_decision"] = {**PRIMARY, "performance_pass": primary["performance_pass"],
                                       "non_inferior_on_review_ineligible": guard["non_inferior"],
                                       "pass": bool(primary["performance_pass"] and guard["non_inferior"])}
        summary["fold_training_seconds_median"] = {
            f"{f}:{c}": float(np.median([r["training_seconds"] for r in folds if r["family"] == f and r["condition"] == c]))
            for f in families for c in CONDITIONS}
    after = protected_state()
    write(out / "protected_after.json", after)
    result = {"protected_unchanged": before == after,
              "sources_unchanged": all(sha(ROOT / r) == protocol["source_hashes"][r] for r in SOURCES),
              "protocol_unchanged": sha(out / "protocol.json") == read(out / "protocol_lock.json")["sha256"],
              "review_decisions_unchanged": sha(decisions_path) == REVIEW_DECISIONS_SHA256}
    global _CACHE
    _CACHE = None
    shutil.rmtree(out / "cache")  # regenerable from data + source; hashes are in cohort.json
    shutil.rmtree(stage_parent)
    if not args.smoke:
        summary.update(result, finished_at=now())
        write(out / "summary.json", summary)
        print({"primary": summary["primary_decision"], **result,
               "reproduction": summary["baseline_reproduction_vs_despike_study"]}, flush=True)
    else:
        print(result, flush=True)


if __name__ == "__main__":
    main()
