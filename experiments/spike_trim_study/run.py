#!/usr/bin/env python3
"""Spike-repair and recording-start trim study (experiment only, no runtime change).

Conditions (fixed before any fit):
  baseline            submission pipeline as is
  despike             isolated one-sample repair, training AND every evaluation/live window
  median3             3-sample running median (optional arm, not run by default:
                      it alters ~42% of samples, i.e. smoothing, not spike repair)
  trim_0.5s/1.0s      drop the first N s of each *training* recording only (evaluation untrimmed)
  despike+trim_*      both

Protocol mirrors ``src.model_comparison`` at HEAD for the four default app model
families: seeded subject split (train/validation/final), parameters tuned on
validation Macro F1, LOSO over development subjects (train+validation), final
subjects scored once with the train-only model. In addition the live engine is
simulated (3 s window, 1 s prediction interval, mean of the last 3 probability
vectors) on held-out subjects to measure prediction flips and false alarms.

Runtime code is ``git archive HEAD src`` extracted to a temporary directory, so
the uncommitted working tree cannot change the results. ``data/`` is only read;
its size/mtime/SHA-256 are compared before and after.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import hashlib
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.spike_trim_study.signal_rules import (  # noqa: E402
    ISOLATED_SPIKE_PARAMETERS, apply_repair, review_spike_mask, start_trim_keep)

SUBMISSION_RUN = ROOT / "research/models/models/20260929_221617_db0a4f57"
SUBMISSION_EVALUATION = SUBMISSION_RUN / "artifacts/window_3s_rbf_svm_evaluation.json"
FAMILIES = ("rbf_svm", "logistic_regression", "random_forest", "extra_trees")
CONDITIONS = {
    "baseline": {"repair": None, "trim": 0.0},
    "despike": {"repair": "isolated", "trim": 0.0},
    "median3": {"repair": "median3", "trim": 0.0},
    "trim_0.5s": {"repair": None, "trim": 0.5},
    "trim_1.0s": {"repair": None, "trim": 1.0},
    "despike+trim_0.5s": {"repair": "isolated", "trim": 0.5},
    "despike+trim_1.0s": {"repair": "isolated", "trim": 1.0},
}
# median3 changes ~42% of all samples (smoke run), so it is a low-pass smoother
# rather than a spike repair; it stays available but is not a default arm.
DEFAULT_CONDITIONS = [name for name in CONDITIONS if name != "median3"]
# Longest tasks first so the process pool finishes evenly.
FAMILY_COST_ORDER = ("rbf_svm", "random_forest", "extra_trees", "logistic_regression")
WINDOW_SECONDS = 3.0
SAMPLING_RATE = 12.0
MIN_TAIL_FRACTION = 0.95          # submission bundle value
HOP_SECONDS = 1.0                 # LiveConfig.prediction_interval_seconds default
SMOOTHING = 3                     # LiveConfig.smoothing_history default
INITIAL_SECONDS = 0.5             # data_review "initial" spike region
SEED = 20260826
METHOD = "standard"


# --------------------------------------------------------------------------- utilities

def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def data_snapshot() -> dict[str, list]:
    data = ROOT / "data"
    return {path.relative_to(data).as_posix(): [path.stat().st_size, path.stat().st_mtime_ns, sha256_file(path)]
            for path in sorted(data.rglob("*")) if path.is_file()}


def write_json(path: Path, payload) -> None:
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=convert, allow_nan=False) + "\n",
                    encoding="utf-8")


def extract_head_source(destination: Path) -> str:
    revision = git("rev-parse", "HEAD")
    blob = subprocess.run(["git", "archive", revision, "src"], cwd=ROOT, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        archive.extractall(destination, filter="data")
    return revision


def activate_source(snapshot: Path) -> Path:
    snapshot = Path(snapshot).resolve()
    if str(snapshot) not in sys.path:
        sys.path.insert(0, str(snapshot))
    import src
    origin = Path(src.__file__).resolve()
    if not origin.is_relative_to(snapshot):
        raise RuntimeError(f"src imported from {origin}, expected HEAD snapshot {snapshot}")
    return origin


def paired(baseline: dict[str, float], other: dict[str, float]) -> dict:
    from scipy.stats import wilcoxon
    keys = sorted(set(baseline) & set(other))
    a = np.array([baseline[k] for k in keys], dtype=float)
    b = np.array([other[k] for k in keys], dtype=float)
    diff = b - a
    result = {"subjects": len(keys), "mean_diff": float(diff.mean()) if len(keys) else None,
              "median_diff": float(np.median(diff)) if len(keys) else None,
              "improved": int((diff > 1e-12).sum()), "worse": int((diff < -1e-12).sum()),
              "tied": int((np.abs(diff) <= 1e-12).sum()), "wilcoxon_p": None}
    if len(keys) and np.any(np.abs(diff) > 1e-12):
        result["wilcoxon_p"] = float(wilcoxon(b, a, zero_method="wilcox").pvalue)
    return result


# --------------------------------------------------------------------------- data

def load_records(data_root: Path) -> list[dict]:
    from src.posture_core import (LABEL_TO_ID, SENSOR_COLUMNS, normalize_text, read_sensor_csv, recording_label,
                                  training_csv_paths)
    records = []
    for number, path in enumerate(training_csv_paths(data_root), start=1):
        frame = read_sensor_csv(path)
        _, label = recording_label(frame)
        time_s = frame["elapsed_s"].to_numpy(dtype=np.float64)
        values = frame[SENSOR_COLUMNS].to_numpy(dtype=np.float64)
        spikes = review_spike_mask(values)
        records.append({"record_id": f"R{number:03d}", "path": path, "file": path.relative_to(data_root).as_posix(),
                        "subject": normalize_text(path.parent.name), "label": label, "class_id": LABEL_TO_ID[label],
                        "frame": frame, "time": time_s, "values": values,
                        "spike_times": (time_s - time_s[0])[spikes.any(axis=1)]})
    return records


def variant_frame(record: dict, repair: str | None, trim: float) -> pd.DataFrame:
    from src.posture_core import SENSOR_COLUMNS
    if repair is None and not trim:
        return record["frame"]
    frame = record["frame"].copy()
    if repair is not None:
        repaired, _ = apply_repair(repair, record["time"], record["values"])
        frame[SENSOR_COLUMNS] = repaired
    if trim:
        frame = frame.loc[start_trim_keep(record["time"], trim)].reset_index(drop=True)
    return frame


def _has_spike(spike_times: np.ndarray, start: float, end: float) -> bool:
    return bool(np.any((spike_times >= start - 1e-9) & (spike_times <= end + 1e-9)))


def window_set(records: list[dict], repair: str | None, trim: float) -> dict[str, np.ndarray]:
    """Non-overlapping windows exactly as the training pipeline cuts them."""
    from src.window_pipeline import WindowSpec, extract_window_features, segment_sensor_frame
    spec = WindowSpec(window_seconds=WINDOW_SECONDS, sampling_rate=SAMPLING_RATE, min_tail_fraction=MIN_TAIL_FRACTION)
    sequences, rows = [], []
    for record in records:
        frame = variant_frame(record, repair, trim)
        offset = float(frame["elapsed_s"].iloc[0] - record["time"][0])
        windows, meta = segment_sensor_frame(frame, spec)
        sequences.extend(windows)
        for row in meta.to_dict(orient="records"):
            start, end = offset + float(row["start_s"]), offset + float(row["nominal_end_s"])
            rows.append((record["record_id"], record["subject"], record["class_id"], int(row["window_index"]),
                         start, end, _has_spike(record["spike_times"], start, end)))
    sequences = np.stack(sequences)
    columns = list(zip(*rows))
    return {"X": extract_window_features(sequences), "record_id": np.array(columns[0], dtype=str),
            "subject": np.array(columns[1], dtype=str), "y": np.array(columns[2], dtype=int),
            "window_index": np.array(columns[3], dtype=int), "start_s": np.array(columns[4], dtype=float),
            "end_s": np.array(columns[5], dtype=float), "has_spike": np.array(columns[6], dtype=bool)}


def rolling_set(records: list[dict], repair: str | None) -> tuple[dict[str, np.ndarray], dict]:
    """Replay each recording through the live buffer and the sklearn adapter's window path.

    Mirrors RealtimePosturePredictor.push (HEAD): TimeWindowBuffer with the app's
    LiveConfig (window 3 s, minimum 3 x min_tail_fraction), one prediction per
    1 s; SklearnAdapter.predict_windows cuts the snapshot with segment_sensor_frame.
    The repair (if any) is applied to each live snapshot, as a runtime would.
    """
    from src.posture_core import SENSOR_COLUMNS, extract_feature_vector
    from src.realtime import LiveConfig, TimeWindowBuffer
    from src.window_pipeline import WindowSpec, segment_sensor_frame
    config = LiveConfig(window_seconds=WINDOW_SECONDS, minimum_window_seconds=WINDOW_SECONDS * MIN_TAIL_FRACTION,
                        prediction_interval_seconds=HOP_SECONDS, smoothing_history=SMOOTHING)
    spec = WindowSpec(window_seconds=WINDOW_SECONDS, sampling_rate=SAMPLING_RATE, min_tail_fraction=MIN_TAIL_FRACTION)
    features, rows = [], []
    repaired_snapshots = 0
    for record in records:
        buffer, last = TimeWindowBuffer(config), None
        t0 = record["time"][0]
        for timestamp, values in zip(record["time"], record["values"]):
            buffer.append(timestamp, values)
            if not buffer.is_ready() or (last is not None and timestamp - last < config.prediction_interval_seconds):
                continue
            snapshot = buffer.snapshot()
            if repair is not None:
                repaired, mask = apply_repair(repair, snapshot["elapsed_s"].to_numpy(dtype=np.float64),
                                              snapshot[SENSOR_COLUMNS].to_numpy(dtype=np.float64))
                if mask.any():
                    repaired_snapshots += 1
                    snapshot = snapshot.copy()
                    snapshot[SENSOR_COLUMNS] = repaired
            windows, _ = segment_sensor_frame(snapshot, spec)
            if len(windows) != 1:
                raise RuntimeError(f"live snapshot produced {len(windows)} windows")
            features.append(extract_feature_vector(windows[0]))
            end = float(timestamp - t0)
            start = max(0.0, end - WINDOW_SECONDS)
            rows.append((record["record_id"], record["subject"], record["class_id"], end,
                         _has_spike(record["spike_times"], start, end), start < INITIAL_SECONDS + 1e-9))
            last = timestamp
    columns = list(zip(*rows))
    return ({"X": np.stack(features), "record_id": np.array(columns[0], dtype=str),
             "subject": np.array(columns[1], dtype=str), "y": np.array(columns[2], dtype=int),
             "end_s": np.array(columns[3], dtype=float), "has_spike": np.array(columns[4], dtype=bool),
             "initial": np.array(columns[5], dtype=bool)},
            {"events": len(rows), "snapshots_changed_by_repair": repaired_snapshots})


# --------------------------------------------------------------------------- audit (label free)

def feature_group(name: str) -> str:
    prefix, rest = name.split("_", 1)
    channel = re.match(r"^(.+)_R\d\d$", rest)
    if channel:
        return f"{prefix}:{channel.group(1)}"
    spatial = re.match(r"^(row|column)_(.+)_\d$", rest)
    if spatial:
        return f"{prefix}:{spatial.group(1)}_{spatial.group(2)}"
    return f"{prefix}:derived_" + re.sub(r"_(global|left_right|front_back|diagonal|outer_row|outer_column)$", "", rest)


def spike_audit(records: list[dict], train_subjects: set[str]) -> dict:
    total_cells = total_rows = initial_cells = initial_rows = 0
    iso_cells = iso_rows = iso_initial_rows = med_cells = 0
    overlap_cells = review_cells_total = 0
    samples = initial_samples = 0
    duration = initial_duration = 0.0
    train_abs_diff = []
    per_record = []
    for record in records:
        t = record["time"] - record["time"][0]
        mask = review_spike_mask(record["values"])
        _, iso = apply_repair("isolated", record["time"], record["values"])
        _, med = apply_repair("median3", record["time"], record["values"])
        initial = t <= INITIAL_SECONDS
        rows = mask.any(axis=1)
        total_cells += int(mask.sum()); total_rows += int(rows.sum())
        initial_cells += int(mask[initial].sum()); initial_rows += int(rows[initial].sum())
        iso_cells += int(iso.sum()); iso_rows += int(iso.any(axis=1).sum())
        iso_initial_rows += int(iso.any(axis=1)[initial].sum())
        med_cells += int(med.sum())
        overlap_cells += int((mask & iso).sum()); review_cells_total += int(mask.sum())
        samples += len(t); initial_samples += int(initial.sum())
        duration += float(t[-1]); initial_duration += min(INITIAL_SECONDS, float(t[-1]))
        if record["subject"] in train_subjects:
            train_abs_diff.append(np.abs(np.diff(record["values"], axis=0)).ravel())
        per_record.append({"record_id": record["record_id"], "subject": record["subject"], "label": record["label"],
                           "review_spike_cells": int(mask.sum()), "review_spike_rows": int(rows.sum()),
                           "initial_review_spike_rows": int(rows[initial].sum()),
                           "isolated_repair_cells": int(iso.sum()), "median3_changed_cells": int(med.sum())})
    diffs = np.concatenate(train_abs_diff)
    frame = pd.DataFrame(per_record)
    later_duration = duration - initial_duration
    return {
        "records": len(records), "samples": samples, "channels": 16,
        "review_rule": "src.data_review transient_spike (record-adaptive thresholds)",
        "review_spike_cells": total_cells, "review_spike_rows": total_rows,
        "initial_region_seconds": INITIAL_SECONDS,
        "initial_review_spike_cells": initial_cells, "initial_review_spike_rows": initial_rows,
        "initial_share_of_review_spike_rows": initial_rows / total_rows if total_rows else None,
        "initial_share_of_review_spike_cells": initial_cells / total_cells if total_cells else None,
        "initial_share_of_samples": initial_samples / samples,
        "review_spike_rows_per_minute_initial": 60 * initial_rows / initial_duration,
        "review_spike_rows_per_minute_later": 60 * (total_rows - initial_rows) / later_duration,
        "records_with_review_spike": int((frame.review_spike_rows > 0).sum()),
        "records_with_initial_review_spike": int((frame.initial_review_spike_rows > 0).sum()),
        "isolated_rule": ISOLATED_SPIKE_PARAMETERS,
        "isolated_repair_cells": iso_cells, "isolated_repair_rows": iso_rows,
        "isolated_repair_initial_rows": iso_initial_rows,
        "isolated_repair_cell_fraction": iso_cells / (samples * 16),
        "median3_changed_cells": med_cells, "median3_changed_cell_fraction": med_cells / (samples * 16),
        "review_spike_cells_also_repaired_by_isolated": overlap_cells / review_cells_total if review_cells_total else None,
        "isolated_repairs_that_are_review_spikes": overlap_cells / iso_cells if iso_cells else None,
        "train_subject_abs_diff_percentiles": {f"p{q}": float(np.percentile(diffs, q)) for q in (90, 99, 99.5, 99.9)},
        "per_label_review_spike_rows": frame.groupby("label").review_spike_rows.sum().to_dict(),
        "per_record": per_record,
    }


def feature_sensitivity(base: dict, repaired: dict, train_mask: np.ndarray, names: list[str]) -> dict:
    scale = base["X"][train_mask].std(axis=0) + 1e-12
    delta = np.abs(repaired["X"] - base["X"]) / scale
    affected = np.any(np.abs(repaired["X"] - base["X"]) > 0, axis=1)
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(feature_group(name), []).append(index)
    rows = []
    for group, columns in groups.items():
        values = delta[affected][:, columns]
        rows.append({"group": group, "features": len(columns),
                     "mean_abs_change_sd": float(values.mean()) if affected.any() else 0.0,
                     "p95_abs_change_sd": float(np.percentile(values, 95)) if affected.any() else 0.0,
                     "share_windows_any_feature_over_0.5sd": float((values > 0.5).any(axis=1).mean())
                     if affected.any() else 0.0})
    rows.sort(key=lambda row: -row["mean_abs_change_sd"])
    return {"windows": int(len(affected)), "windows_changed_by_repair": int(affected.sum()),
            "scale": "baseline train-subject feature SD", "groups": rows}


# --------------------------------------------------------------------------- live metrics

def rolling_metrics(probability: np.ndarray, meta: dict[str, np.ndarray], rows: np.ndarray) -> tuple[dict, pd.DataFrame]:
    from src.model_comparison import scores
    record_ids = meta["record_id"][rows]
    end_s = meta["end_s"][rows]
    raw_pred = probability.argmax(axis=1)
    smooth = np.empty_like(probability)
    raw_changed = np.zeros(len(rows), dtype=bool)
    record_flips = np.zeros(len(rows), dtype=int)
    flips, span = {}, {}
    # Events of one record are contiguous and time ordered (rolling_set order).
    for record_id in dict.fromkeys(record_ids):
        index = np.flatnonzero(record_ids == record_id)
        if np.any(np.diff(index) != 1) or np.any(np.diff(end_s[index]) <= 0):
            raise ValueError(f"live events of {record_id} are not contiguous and time ordered")
        for position, row in enumerate(index):
            smooth[row] = probability[index[max(0, position - SMOOTHING + 1):position + 1]].mean(axis=0)
        predicted = smooth[index].argmax(axis=1)
        flips[record_id] = int(np.sum(predicted[1:] != predicted[:-1]))
        span[record_id] = float(end_s[index[-1]] - end_s[index[0]])
        raw_changed[index[1:]] = raw_pred[index[1:]] != raw_pred[index[:-1]]
        record_flips[index] = flips[record_id]
    frame = pd.DataFrame({"row": rows, "record_id": record_ids, "subject": meta["subject"][rows],
                          "y": meta["y"][rows], "end_s": end_s, "has_spike": meta["has_spike"][rows],
                          "initial": meta["initial"][rows], "raw_pred": raw_pred,
                          "smooth_pred": smooth.argmax(axis=1), "raw_changed": raw_changed,
                          "record_flips": record_flips})
    per_subject = {}
    upright = frame.y == 0
    for subject, group in frame.groupby("subject", sort=True):
        records = group.record_id.unique()
        minutes = sum(span[r] for r in records) / 60
        per_subject[subject] = {
            "smooth_accuracy": float((group.smooth_pred == group.y).mean()),
            "flips_per_minute": float(sum(flips[r] for r in records) / minutes) if minutes else 0.0,
            "upright_false_alarm": float((group.smooth_pred[group.y == 0] != 0).mean()) if (group.y == 0).any() else None,
        }
    minutes = sum(span.values()) / 60
    spike = frame.has_spike
    smooth_scores = scores(frame.y.to_numpy(), frame.smooth_pred.to_numpy())
    summary = {
        "events": int(len(frame)), "records": int(frame.record_id.nunique()), "monitored_minutes": minutes,
        "raw_accuracy": float((frame.raw_pred == frame.y).mean()),
        "smooth_accuracy": float((frame.smooth_pred == frame.y).mean()),
        "smooth_macro_f1": smooth_scores["macro_f1"], "smooth_worst_class_f1": smooth_scores["worst_class_f1"],
        "flips_per_minute": float(sum(flips.values()) / minutes) if minutes else 0.0,
        "records_without_flip": float(np.mean([count == 0 for count in flips.values()])),
        "upright_false_alarm": float((frame.smooth_pred[upright] != 0).mean()),
        "abnormal_reported_upright": float((frame.smooth_pred[~upright] == 0).mean()),
        "spike_events": int(spike.sum()),
        "raw_accuracy_spike_events": float((frame.raw_pred[spike] == frame.y[spike]).mean()) if spike.any() else None,
        "raw_accuracy_clean_events": float((frame.raw_pred[~spike] == frame.y[~spike]).mean()),
        "raw_change_rate_spike_events": float(frame.raw_changed[spike].mean()) if spike.any() else None,
        "raw_change_rate_clean_events": float(frame.raw_changed[~spike].mean()),
        "raw_accuracy_initial_events": float((frame.raw_pred[frame.initial] == frame.y[frame.initial]).mean()),
        "raw_accuracy_later_events": float((frame.raw_pred[~frame.initial] == frame.y[~frame.initial]).mean()),
    }
    return {"summary": summary, "per_subject": per_subject}, frame.assign(
        **{f"p{index}": probability[:, index] for index in range(probability.shape[1])})


def window_breakdown(y: np.ndarray, prediction: np.ndarray, window_index: np.ndarray, has_spike: np.ndarray) -> dict:
    first = window_index == 1
    return {"first_window_accuracy": float((prediction[first] == y[first]).mean()),
            "later_window_accuracy": float((prediction[~first] == y[~first]).mean()),
            "spike_window_accuracy": float((prediction[has_spike] == y[has_spike]).mean()) if has_spike.any() else None,
            "clean_window_accuracy": float((prediction[~has_spike] == y[~has_spike]).mean()),
            "spike_windows": int(has_spike.sum()), "first_windows": int(first.sum())}


# --------------------------------------------------------------------------- one (condition, family)

def run_task(task: dict) -> dict:
    activate_source(Path(task["snapshot"]))
    from src.model_comparison import scores
    from src.posture_core import CLASS_NAMES
    from src.training_methods import build_method_model, fit_method_model, method_candidates
    from src.window_pipeline import aligned_predict_proba
    started = time.perf_counter()
    cache = Path(task["cache"])
    condition, family, seed = task["condition"], task["family"], task["seed"]
    spec = CONDITIONS[condition]
    repair = spec["repair"] or "none"
    evaluation = dict(np.load(cache / f"eval_{repair}.npz"))
    fit = dict(np.load(cache / (f"fit_{repair}_{spec['trim']}.npz" if spec["trim"] else f"eval_{repair}.npz")))
    live = dict(np.load(cache / f"roll_{repair}.npz"))
    k = len(CLASS_NAMES)
    part = task["partitions"]
    mask = {name: np.isin(evaluation["subject"], part[name]) for name in part}
    fit_mask = {name: np.isin(fit["subject"], part[name]) for name in part}
    live_mask = {name: np.isin(live["subject"], part[name]) for name in part}
    X, y = evaluation["X"], evaluation["y"]

    def fit_model(params, rows, validation):
        model = build_method_model(family, params, seed, METHOD)
        fit_method_model(model, family, METHOD, fit["X"][rows], fit["y"][rows], groups=fit["subject"][rows],
                         record_ids=fit["record_id"][rows], validation_data=validation)
        return model

    candidates, best = [], None
    grid = [dict(task["fixed_params"])] if task.get("fixed_params") else method_candidates(family, METHOD)
    for params in grid:
        tic = time.perf_counter()
        model = fit_model(params, fit_mask["train"], (X[mask["validation"]], y[mask["validation"]]))
        validation = scores(y[mask["validation"]], aligned_predict_proba(model, X[mask["validation"]], k).argmax(axis=1))
        candidates.append({"params": dict(params), "validation_macro_f1": validation["macro_f1"],
                           "validation_worst_class_f1": validation["worst_class_f1"],
                           "fit_seconds": time.perf_counter() - tic})
        key = (validation["macro_f1"], validation["worst_class_f1"])
        if best is None or key > best[0]:
            best = (key, model, dict(params), validation)
    _, model, params, validation = best

    final_rows = mask["final_evaluation"]
    final_probability = aligned_predict_proba(model, X[final_rows], k)
    final_prediction = final_probability.argmax(axis=1)
    final = scores(y[final_rows], final_prediction)
    final["breakdown"] = window_breakdown(y[final_rows], final_prediction, evaluation["window_index"][final_rows],
                                          evaluation["has_spike"][final_rows])
    final_subjects = evaluation["subject"][final_rows]
    final["per_subject_macro_f1"] = {s: scores(y[final_rows][final_subjects == s],
                                               final_prediction[final_subjects == s])["macro_f1"]
                                     for s in sorted(set(final_subjects))}
    live_final_rows = np.flatnonzero(live_mask["final_evaluation"])
    live_final, live_final_frame = rolling_metrics(aligned_predict_proba(model, live["X"][live_final_rows], k),
                                                   live, live_final_rows)

    development = mask["train"] | mask["validation"]
    fit_development = fit_mask["train"] | fit_mask["validation"]
    people = sorted(set(evaluation["subject"][development]))
    if task.get("loso_limit"):
        people = people[:task["loso_limit"]]
    loso_probability = np.full((len(y), k), np.nan)
    live_rows = np.flatnonzero(np.isin(live["subject"], people))
    live_probability = np.full((len(live["y"]), k), np.nan)
    for person in people:
        fold = fit_model(params, fit_development & (fit["subject"] != person), None)
        held = development & (evaluation["subject"] == person)
        loso_probability[held] = aligned_predict_proba(fold, X[held], k)
        held_live = live["subject"] == person
        live_probability[held_live] = aligned_predict_proba(fold, live["X"][held_live], k)
    loso_rows = development & np.isin(evaluation["subject"], people)
    loso_prediction = loso_probability[loso_rows].argmax(axis=1)
    loso = scores(y[loso_rows], loso_prediction)
    loso_subjects = evaluation["subject"][loso_rows]
    loso["per_subject_macro_f1"] = {s: scores(y[loso_rows][loso_subjects == s],
                                              loso_prediction[loso_subjects == s])["macro_f1"] for s in people}
    loso["breakdown"] = window_breakdown(y[loso_rows], loso_prediction, evaluation["window_index"][loso_rows],
                                         evaluation["has_spike"][loso_rows])
    loso["subjects"] = len(people)
    loso["windows"] = int(loso_rows.sum())
    live_loso, live_loso_frame = rolling_metrics(live_probability[live_rows], live, live_rows)

    output = Path(task["output"])
    window_frame = pd.concat([
        pd.DataFrame({"scope": "final_window", "record_id": evaluation["record_id"][final_rows],
                      "subject": final_subjects, "y": y[final_rows],
                      "window_index": evaluation["window_index"][final_rows], "pred": final_prediction,
                      **{f"p{i}": final_probability[:, i] for i in range(k)}}),
        pd.DataFrame({"scope": "loso_window", "record_id": evaluation["record_id"][loso_rows],
                      "subject": loso_subjects, "y": y[loso_rows],
                      "window_index": evaluation["window_index"][loso_rows], "pred": loso_prediction,
                      **{f"p{i}": loso_probability[loso_rows][:, i] for i in range(k)}})])
    live_frame = pd.concat([live_final_frame.assign(scope="final_live"), live_loso_frame.assign(scope="loso_live")])
    stem = f"{condition}__{family}"
    window_frame.to_csv(output / "predictions" / f"{stem}__windows.csv.gz", index=False, float_format="%.6g")
    live_frame.drop(columns=["row"]).to_csv(output / "predictions" / f"{stem}__live.csv.gz", index=False,
                                            float_format="%.6g")
    result = {"condition": condition, "family": family, "condition_rule": spec, "selected_params": params,
              "candidates": candidates, "validation": validation, "loso": loso, "final": final,
              "live_final": live_final, "live_loso": live_loso,
              "fit_windows": {"train": int(fit_mask["train"].sum()), "development": int(fit_development.sum())},
              "evaluation_windows": {name: int(value.sum()) for name, value in mask.items()},
              "seconds": time.perf_counter() - started}
    write_json(output / "tasks" / f"{stem}.json", result)
    return result


# --------------------------------------------------------------------------- summary

SUMMARY_COLUMNS = [
    ("validation_macro_f1", lambda r: r["validation"]["macro_f1"]),
    ("loso_macro_f1", lambda r: r["loso"]["macro_f1"]),
    ("loso_worst_class_f1", lambda r: r["loso"]["worst_class_f1"]),
    ("final_macro_f1", lambda r: r["final"]["macro_f1"]),
    ("final_accuracy", lambda r: r["final"]["accuracy"]),
    ("final_worst_class_f1", lambda r: r["final"]["worst_class_f1"]),
    ("loso_live_smooth_accuracy", lambda r: r["live_loso"]["summary"]["smooth_accuracy"]),
    ("loso_live_flips_per_minute", lambda r: r["live_loso"]["summary"]["flips_per_minute"]),
    ("loso_live_upright_false_alarm", lambda r: r["live_loso"]["summary"]["upright_false_alarm"]),
    ("loso_live_abnormal_reported_upright", lambda r: r["live_loso"]["summary"]["abnormal_reported_upright"]),
    ("loso_live_raw_acc_spike", lambda r: r["live_loso"]["summary"]["raw_accuracy_spike_events"]),
    ("loso_live_raw_acc_clean", lambda r: r["live_loso"]["summary"]["raw_accuracy_clean_events"]),
    ("final_live_smooth_accuracy", lambda r: r["live_final"]["summary"]["smooth_accuracy"]),
    ("final_live_flips_per_minute", lambda r: r["live_final"]["summary"]["flips_per_minute"]),
    ("final_live_upright_false_alarm", lambda r: r["live_final"]["summary"]["upright_false_alarm"]),
    ("loso_first_window_accuracy", lambda r: r["loso"]["breakdown"]["first_window_accuracy"]),
    ("loso_later_window_accuracy", lambda r: r["loso"]["breakdown"]["later_window_accuracy"]),
]


def summarize(results: list[dict]) -> tuple[pd.DataFrame, list[dict], dict]:
    table = pd.DataFrame([{"condition": r["condition"], "family": r["family"],
                           "params": json.dumps(r["selected_params"], sort_keys=True),
                           **{name: getter(r) for name, getter in SUMMARY_COLUMNS}} for r in results])
    table["condition"] = pd.Categorical(table["condition"], [c for c in CONDITIONS if c in set(table.condition)])
    table = table.sort_values(["family", "condition"]).reset_index(drop=True)
    by_key = {(r["condition"], r["family"]): r for r in results}
    comparisons = []
    for (condition, family), result in by_key.items():
        base = by_key.get(("baseline", family))
        if condition == "baseline" or base is None:
            continue
        row = {"condition": condition, "family": family}
        row["loso_subject_macro_f1"] = paired(base["loso"]["per_subject_macro_f1"], result["loso"]["per_subject_macro_f1"])
        for metric in ("smooth_accuracy", "flips_per_minute", "upright_false_alarm"):
            a = {s: v[metric] for s, v in base["live_loso"]["per_subject"].items() if v[metric] is not None}
            b = {s: v[metric] for s, v in result["live_loso"]["per_subject"].items() if v[metric] is not None}
            row[f"loso_live_subject_{metric}"] = paired(a, b)
        comparisons.append(row)
    selected = {}
    for condition in CONDITIONS:
        rows = [r for r in results if r["condition"] == condition]
        if rows:
            winner = max(rows, key=lambda r: (r["loso"]["macro_f1"], r["loso"]["worst_class_f1"]))
            selected[condition] = {"family": winner["family"], "loso_macro_f1": winner["loso"]["macro_f1"],
                                   "final_macro_f1": winner["final"]["macro_f1"]}
    return table, comparisons, selected


def markdown_summary(table: pd.DataFrame, comparisons: list[dict], selected: dict, audit: dict, checks: dict) -> str:
    def pct(value):
        return "-" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{100 * value:.2f}"
    lines = ["# Spike repair / start trim study - generated tables", "",
             "Generated by `experiments/spike_trim_study/run.py`; see run.json for provenance.", "",
             f"Baseline reproduction of submission RBF-SVM: {checks.get('baseline_reproduction')}", "",
             "## Spike audit (label free, all 724 records)", "",
             f"- review-rule spike rows: {audit['review_spike_rows']}, of which in first {INITIAL_SECONDS}s: "
             f"{audit['initial_review_spike_rows']} ({pct(audit['initial_share_of_review_spike_rows'])}%); "
             f"first {INITIAL_SECONDS}s is {pct(audit['initial_share_of_samples'])}% of samples",
             f"- spike rows per minute: first {INITIAL_SECONDS}s {audit['review_spike_rows_per_minute_initial']:.2f}, "
             f"later {audit['review_spike_rows_per_minute_later']:.2f}",
             f"- isolated repair changes {audit['isolated_repair_cells']} cells "
             f"({100 * audit['isolated_repair_cell_fraction']:.3f}%), median3 {audit['median3_changed_cells']} "
             f"({100 * audit['median3_changed_cell_fraction']:.2f}%)", "",
             "## Metrics (%, flips per minute)", "",
             "| family | condition | val F1 | LOSO F1 | final F1 | final acc | live acc (LOSO) | flips/min (LOSO) "
             "| upright FA (LOSO) | abnormal->upright (LOSO) | final live acc | final flips/min | final upright FA |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in table.itertuples():
        lines.append(f"| {row.family} | {row.condition} | {pct(row.validation_macro_f1)} | {pct(row.loso_macro_f1)} | "
                     f"{pct(row.final_macro_f1)} | {pct(row.final_accuracy)} | {pct(row.loso_live_smooth_accuracy)} | "
                     f"{row.loso_live_flips_per_minute:.2f} | {pct(row.loso_live_upright_false_alarm)} | "
                     f"{pct(row.loso_live_abnormal_reported_upright)} | {pct(row.final_live_smooth_accuracy)} | "
                     f"{row.final_live_flips_per_minute:.2f} | {pct(row.final_live_upright_false_alarm)} |")
    lines += ["", "## Paired vs baseline over LOSO development subjects", "",
              "| family | condition | ΔF1 mean (pp) | +/-/= | p | Δlive acc (pp) | p | Δflips/min | p | Δupright FA (pp) | p |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    def p(value):
        return "-" if value is None else f"{value:.3f}"
    for row in comparisons:
        f1, acc = row["loso_subject_macro_f1"], row["loso_live_subject_smooth_accuracy"]
        flips, fa = row["loso_live_subject_flips_per_minute"], row["loso_live_subject_upright_false_alarm"]
        lines.append(f"| {row['family']} | {row['condition']} | {100 * f1['mean_diff']:+.2f} | "
                     f"{f1['improved']}/{f1['worse']}/{f1['tied']} | {p(f1['wilcoxon_p'])} | "
                     f"{100 * acc['mean_diff']:+.2f} | {p(acc['wilcoxon_p'])} | {flips['mean_diff']:+.2f} | "
                     f"{p(flips['wilcoxon_p'])} | {100 * fa['mean_diff']:+.2f} | {p(fa['wilcoxon_p'])} |")
    lines += ["", "## App selection rule per condition (max LOSO Macro F1)", ""]
    for condition, row in selected.items():
        lines.append(f"- {condition}: {row['family']} (LOSO {pct(row['loso_macro_f1'])}, final {pct(row['final_macro_f1'])})")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new empty run directory")
    parser.add_argument("--data", type=Path, default=SUBMISSION_RUN / "training_data",
                        help="staged submission training data (724 records, 32 subjects)")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=list(CONDITIONS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--loso-limit", type=int, default=0, help="smoke test only: LOSO over the first N subjects")
    parser.add_argument("--fixed-params", type=json.loads, default=None,
                        help="ablation: JSON parameters used instead of the validation grid (one family only)")
    args = parser.parse_args()
    if args.fixed_params is not None and (len(args.families) != 1 or not isinstance(args.fixed_params, dict)):
        raise SystemExit("--fixed-params requires exactly one --families entry and a JSON object")
    if "baseline" not in args.conditions:
        raise SystemExit("baseline is required for paired comparisons")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "tasks").mkdir()
    (output / "predictions").mkdir()
    started = now()
    before = data_snapshot()
    dirty = git("status", "--porcelain").splitlines()
    temporary = Path(tempfile.mkdtemp(prefix="spike_trim_study_"))
    try:
        snapshot = temporary / "head_source"
        revision = extract_head_source(snapshot)
        origin = activate_source(snapshot)
        from src.model_comparison import _dataset
        from src.posture_core import feature_names
        from src.window_pipeline import WindowSpec, load_window_dataset
        import sklearn, scipy

        data_root = args.data.resolve()
        print(f"[{now()}] loading {data_root}", flush=True)
        reference = json.loads(SUBMISSION_EVALUATION.read_text(encoding="utf-8"))
        staged = json.loads((data_root / "staged_sources.json").read_text(encoding="utf-8"))
        staged_mismatch = [row["staged_file"] for row in staged
                           if sha256_file(data_root / row["staged_file"]) != row["sha256"]
                           or sha256_file(Path(row["source_file"])) != row["sha256"]]
        head = _dataset(data_root, int(WINDOW_SECONDS), args.seed)
        partitions = {name: list(value["subjects"]) for name, value in head["partitions"].items()}
        checks = {
            "src_origin": str(origin), "staged_files": len(staged),
            "staged_or_source_hash_mismatch": staged_mismatch,
            "fingerprint_matches_submission": head["fingerprint"] == reference["dataset_fingerprint"],
            "partitions_match_submission": partitions == {n: v["subjects"] for n, v in reference["partitions"].items()},
        }
        records = load_records(data_root)
        base_eval = window_set(records, None, 0.0)
        reference_sequences, reference_meta, _ = load_window_dataset(data_root, WindowSpec(window_seconds=WINDOW_SECONDS))
        checks["baseline_windows_identical_to_training_pipeline"] = bool(
            np.array_equal(base_eval["record_id"], reference_meta["record_id"].to_numpy(dtype=str))
            and np.array_equal(base_eval["X"], head["X"]))
        if staged_mismatch or not all(checks[k] for k in ("fingerprint_matches_submission",
                                                          "partitions_match_submission",
                                                          "baseline_windows_identical_to_training_pipeline")):
            write_json(output / "checks.json", checks)
            raise SystemExit(f"input checks failed: {checks}")
        print(f"[{now()}] checks passed; {len(records)} records, {len(base_eval['y'])} windows", flush=True)

        train_subjects = set(partitions["train"])
        audit = spike_audit(records, train_subjects)
        write_json(output / "spike_audit.json", audit)

        cache = temporary / "cache"
        cache.mkdir()
        conditions = {name: CONDITIONS[name] for name in args.conditions}
        repairs = sorted({spec["repair"] or "none" for spec in conditions.values()})
        evaluation_sets = {"none": base_eval}
        live_info = {}
        for repair in repairs:
            tic = time.perf_counter()
            if repair not in evaluation_sets:
                evaluation_sets[repair] = window_set(records, repair, 0.0)
            np.savez(cache / f"eval_{repair}.npz", **evaluation_sets[repair])
            live, live_info[repair] = rolling_set(records, None if repair == "none" else repair)
            np.savez(cache / f"roll_{repair}.npz", **live)
            print(f"[{now()}] features {repair}: {time.perf_counter() - tic:.1f}s, live events {live_info[repair]}",
                  flush=True)
        fit_info = {}
        for spec in conditions.values():
            if spec["trim"]:
                repair = spec["repair"] or "none"
                fit_set = window_set(records, spec["repair"], spec["trim"])
                np.savez(cache / f"fit_{repair}_{spec['trim']}.npz", **fit_set)
                fit_info[f"{repair}_{spec['trim']}"] = {
                    "windows": int(len(fit_set["y"])),
                    "train_windows": int(np.isin(fit_set["subject"], partitions["train"]).sum())}
        train_mask = np.isin(base_eval["subject"], partitions["train"])
        sensitivity = {repair: feature_sensitivity(base_eval, evaluation_sets[repair], train_mask, feature_names())
                       for repair in repairs if repair != "none"}
        write_json(output / "feature_sensitivity.json", sensitivity)

        tasks = [{"condition": c, "family": f, "seed": args.seed, "partitions": partitions, "cache": str(cache),
                  "snapshot": str(snapshot), "output": str(output), "loso_limit": args.loso_limit,
                  "fixed_params": args.fixed_params}
                 for f in sorted(args.families, key=FAMILY_COST_ORDER.index) for c in conditions]
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_task, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"[{now()}] done {result['condition']} {result['family']}: "
                      f"val {result['validation']['macro_f1']:.4f} LOSO {result['loso']['macro_f1']:.4f} "
                      f"final {result['final']['macro_f1']:.4f} "
                      f"flips/min {result['live_loso']['summary']['flips_per_minute']:.2f} "
                      f"({result['seconds']:.0f}s)", flush=True)
        base_svm = next((r for r in results if r["condition"] == "baseline" and r["family"] == "rbf_svm"), None)
        if base_svm is not None:
            expected = {"validation": reference["validation_metrics"]["macro_f1"],
                        "final": reference["metrics"]["macro_f1"], "loso": reference["loso_metrics"]["macro_f1"]}
            observed = {"validation": base_svm["validation"]["macro_f1"], "final": base_svm["final"]["macro_f1"],
                        "loso": base_svm["loso"]["macro_f1"]}
            checks["baseline_reproduction"] = {
                "expected": expected, "observed": observed,
                "params_match": base_svm["selected_params"] == reference["selected_params"],
                "exact": all(abs(expected[k] - observed[k]) < 1e-12 for k in expected) and not args.loso_limit}
        table, comparisons, selected = summarize(results)
        table.to_csv(output / "summary.csv", index=False)
        write_json(output / "summary.json", {"table": table.astype({"condition": str}).to_dict(orient="records"),
                                             "paired_vs_baseline": comparisons, "app_selection_rule": selected})
        (output / "summary.md").write_text(markdown_summary(table, comparisons, selected, audit, checks),
                                           encoding="utf-8")
        after = data_snapshot()
        write_json(output / "run.json", {
            "purpose": "Experiment only: spike repair and training-start trim vs submission pipeline; no runtime change",
            "kind": "CSV replay / training / evaluation (no physical sensor)",
            "started_at": started, "finished_at": now(),
            "source_revision": revision, "source_used": "git archive HEAD src (working tree not imported)",
            "working_tree_dirty_files": dirty,
            "experiment_code": {path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
            "command": [sys.executable, *sys.argv], "seed": args.seed,
            "input": {"data_root": str(data_root.relative_to(ROOT)), "records": len(records),
                      "subjects": len({r["subject"] for r in records}), "dataset_fingerprint": head["fingerprint"],
                      "reference_evaluation": str(SUBMISSION_EVALUATION.relative_to(ROOT))},
            "split_rule": "seeded subject shuffle (HEAD src.model_comparison._dataset); final=n//5, validation=n//5, rest train",
            "partitions": partitions,
            "protocol": {"window_seconds": WINDOW_SECONDS, "sampling_rate": SAMPLING_RATE,
                         "min_tail_fraction": MIN_TAIL_FRACTION, "training_method": METHOD,
                         "parameter_selection": "validation Macro F1, tie worst-class F1 (HEAD)"
                         if args.fixed_params is None else f"ablation: fixed {json.dumps(args.fixed_params)}",
                         "loso": "development subjects (train+validation), fixed selected parameters",
                         "final": "train-subject model, scored once",
                         "trim_scope": "fitted rows only; validation, LOSO held-out, final and live rows untrimmed",
                         "repair_scope": "every row: training, evaluation and each live snapshot",
                         "live": {"window_seconds": WINDOW_SECONDS, "prediction_interval_seconds": HOP_SECONDS,
                                  "smoothing_history": SMOOTHING, "minimum_window_seconds": WINDOW_SECONDS * MIN_TAIL_FRACTION,
                                  "path": "TimeWindowBuffer.snapshot -> segment_sensor_frame -> features (HEAD)"},
                         "loso_limit": args.loso_limit},
            "conditions": conditions, "families": args.families, "isolated_rule": ISOLATED_SPIKE_PARAMETERS,
            "fit_sets": fit_info, "live_sets": live_info, "checks": checks,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                            "scipy": scipy.__version__, "workers": args.workers, "threads_per_worker": 1},
            "data_unchanged": before == after, "data_files_checked": len(before),
            "not_verified": ["physical Arduino sensor stream", "packaged macOS app", "runtime integration of the rules"],
        })
        print(json.dumps({"checks": checks, "app_selection_rule": selected, "data_unchanged": before == after},
                         ensure_ascii=False, indent=2, default=str), flush=True)
        if before != after:
            raise SystemExit("data/ changed during the run")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
