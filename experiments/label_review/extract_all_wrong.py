"""Extract all-wrong recordings from the adopted 008 baseline for human label review.

Read-only with respect to data/, the 008 run and every model: no fitting, no
inference, no relabeling and no exclusion. The output is a new review folder
under research/results/ with a review sheet that people fill in.

Similarity hints compare a recording with the same person's other recordings.
They only order the review; they never decide label truth.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, normalize_text, read_sensor_csv  # noqa: E402

VERSION = "label_review_all_wrong_v1"
TZ = ZoneInfo("Asia/Seoul")
ORIGIN = "research/results/2026-09-22_015526_iterative_008"
PROTOCOL_SHA256 = "50263a71143da8d0d0a2f384262099e0f788c34599ba9fbd3e67e049a93aacf4"
PRIOR_REVIEW = "research/results/2026-09-21_153100_current_data_outlier_review/audit.json"
EXCLUSIONS = "research/models/training_exclusions.json"
# Totals published in docs/reports/2026-09-22_adopted_baseline_error_audit.md.
EXPECTED = {"record_observations": 433, "all_wrong_observations": 91, "errors": 751,
            "errors_in_all_wrong": 594, "constant_wrong_observations": 80}
SIMILARITY_MARGIN = 0.05

# Fixed before viewing cross-model results: one plain-argmax representative per
# model family whose stored development predictions use exactly the 008 windows.
COMPARISON_MODELS = [
    {"name": "XGB", "family": "xgboost", "branch": "research/results/2026-09-22_070000_iterative_020/development/XGB_standard_3s",
     "data_policy": "ordinary_restored"},
    {"name": "CAT", "family": "catboost", "branch": "research/results/2026-09-22_010100_iterative_004/development/CAT_ordinary",
     "data_policy": "ordinary_restored"},
    {"name": "SVM", "family": "rbf_svm", "branch": "research/results/2026-09-23_150000_sensor_gain_2s3s/development/3s_raw_rbf_svm",
     "data_policy": "not_restored"},
    {"name": "RF", "family": "random_forest", "branch": "research/results/2026-09-23_150000_sensor_gain_2s3s/development/3s_raw_random_forest",
     "data_policy": "not_restored"},
    {"name": "ROCKET", "family": "minirocket", "branch": "research/results/2026-09-22_015526_iterative_008/development/Rocket_C01",
     "data_policy": "ordinary_restored"},
    {"name": "CNN", "family": "cnn1d", "branch": "research/results/2026-09-22_015526_iterative_008/development/CNN32",
     "data_policy": "ordinary_restored"},
    {"name": "QDA", "family": "shrinkage_qda", "branch": "research/results/2026-09-22_023800_iterative_011/development/QDA_reg_05",
     "data_policy": "ordinary_restored"},
]

HUMAN_COLUMNS = [
    "reviewer", "review_date", "verdict", "observed_posture", "evidence",
    "cross_check_source", "cross_check_result", "proposed_action", "notes",
]


def now() -> str:
    return datetime.now(TZ).isoformat()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_state(path: Path) -> dict:
    stat = path.stat()
    return {"sha256": sha256(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of 16-channel mean patterns after removing overall level."""
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def load_origin(root: Path):
    origin = root / ORIGIN
    protocol_path = origin / "protocol.json"
    if sha256(protocol_path) != PROTOCOL_SHA256 or read_json(origin / "protocol_lock.json")["sha256"] != PROTOCOL_SHA256:
        raise ValueError("008 protocol seal differs")
    protocol = read_json(protocol_path)
    parts, inputs = [], []
    for number, partition in enumerate(protocol["partitions"], 1):
        if partition["id"] != f"development_{number}":
            raise ValueError("Unexpected partition order")
        folder = origin / "development/baseline" / partition["id"]
        metrics_path, npz_path = folder / "metrics.json", folder / "predictions.npz"
        metrics = read_json(metrics_path)
        if read_json(folder / "metrics_seal.json")["sha256"] != sha256(metrics_path):
            raise ValueError("metrics seal differs: " + partition["id"])
        if metrics["predictions_sha256"] != sha256(npz_path) or metrics["protocol_sha256"] != PROTOCOL_SHA256:
            raise ValueError("prediction link differs: " + partition["id"])
        with np.load(npz_path, allow_pickle=False) as archive:
            part = {key: archive[key] for key in archive.files}
        if not np.array_equal(part["prediction"], part["probability"].argmax(1)):
            raise ValueError("prediction is not argmax")
        if set(part["groups"]) != set(partition["validation"]):
            raise ValueError("prediction subjects differ from validation subjects")
        part["record_ids"] = np.asarray([normalize_text(v) for v in part["record_ids"]])
        part["groups"] = np.asarray([normalize_text(v) for v in part["groups"]])
        part["fold"] = partition["id"]
        parts.append(part)
        inputs += [str(metrics_path.relative_to(root)), str(npz_path.relative_to(root))]
    return origin, protocol, parts, inputs


def observations(parts):
    rows, totals = [], defaultdict(int)
    for part in parts:
        for record in sorted(set(part["record_ids"].tolist())):
            take = np.flatnonzero(part["record_ids"] == record)
            take = take[np.argsort(part["indices"][take])]
            truth, prediction = part["truth"][take], part["prediction"][take]
            if len(set(truth.tolist())) != 1 or len(set(part["groups"][take].tolist())) != 1:
                raise ValueError("Recording is not pure: " + record)
            wrong = prediction != truth
            totals["record_observations"] += 1
            totals["errors"] += int(wrong.sum())
            if not wrong.all():
                continue
            switches = int(np.count_nonzero(prediction[1:] != prediction[:-1]))
            totals["all_wrong_observations"] += 1
            totals["errors_in_all_wrong"] += len(take)
            totals["constant_wrong_observations"] += int(switches == 0)
            probability = part["probability"][take]
            counts = np.bincount(prediction, minlength=7)
            rows.append({
                "fold": part["fold"], "record_id": record, "subject": str(part["groups"][take[0]]),
                "truth_id": int(truth[0]), "windows": len(take), "cache_indices": part["indices"][take].tolist(),
                "prediction_sequence": prediction.tolist(), "switches": switches,
                "majority_prediction_id": int(counts.argmax()),
                "mean_truth_probability": float(probability[:, truth[0]].mean()),
                "mean_top_probability": float(probability.max(1).mean()),
                "mean_class_probability": probability.mean(0).tolist(),
            })
    for key, value in EXPECTED.items():
        if totals[key] != value:
            raise ValueError(f"{key}={totals[key]} differs from published {value}")
    return rows, dict(totals)


def subject_patterns(cache):
    """Per-record 16-channel mean from the frozen 008 cache (resampled windows)."""
    records = cache["record_ids"]
    means, meta = {}, {}
    for record in sorted(set(records.tolist())):
        take = records == record
        means[record] = cache["seq"][take].reshape(-1, 16).astype(np.float64).mean(0)
        meta[record] = (str(cache["groups"][take][0]), int(cache["y"][take][0]))
    return means, meta


def similarity_hint(record, subject, truth_id, predicted_id, means, meta):
    own = defaultdict(list)
    for other, (other_subject, label) in meta.items():
        if other != record and other_subject == subject:
            own[label].append(means[other])
    by_class = {label: centered_cosine(means[record], np.mean(values, 0)) for label, values in own.items()}
    sim_truth, sim_pred = by_class.get(truth_id), by_class.get(predicted_id)
    nearest = max(by_class, key=by_class.get) if by_class else None
    if sim_truth is None or sim_pred is None:
        code = "no_comparison"
    elif sim_pred > sim_truth + SIMILARITY_MARGIN:
        code = "closer_to_own_predicted"
    elif sim_truth > sim_pred + SIMILARITY_MARGIN:
        code = "closer_to_own_truth"
    else:
        code = "ambiguous"
    return {"sim_own_truth": sim_truth, "sim_own_predicted": sim_pred, "nearest_own_class_id": nearest,
            "own_truth_records": len(own.get(truth_id, [])), "own_predicted_records": len(own.get(predicted_id, [])),
            "hint": code}


HINT_TEXT = {
    "closer_to_own_predicted": "같은 사람의 예측 자세 기록과 더 비슷함 → 라벨 확인 우선",
    "closer_to_own_truth": "같은 사람의 정답 자세 기록과 비슷함 → 사람 간 차이/센서 구분 한계 후보",
    "ambiguous": "정답·예측 자세 기록과 비슷한 정도가 같음 → 두 자세 경계 사례 후보",
    "no_comparison": "같은 사람의 비교 기록 없음 → 영상/관찰 기록 필요",
}


def fmt(value, digits=4):
    return "" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"


def plot_record(path, frame, record_obs, cache_start, record_mean, truth_mean, pred_mean, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    figure = plt.figure(figsize=(13, 8.5))
    grid = figure.add_gridspec(2 + len(record_obs), 3, height_ratios=[3] + [0.35] * len(record_obs) + [2.4])
    axis = figure.add_subplot(grid[0, :])
    elapsed = frame["elapsed_s"].to_numpy(float) - float(frame["elapsed_s"].iloc[0])
    colors = plt.get_cmap("tab20")
    for channel, name in enumerate(SENSOR_COLUMNS):
        axis.plot(elapsed, frame[name].to_numpy(float), lw=0.8, color=colors(channel), label=name)
    axis.set_xlabel("원본 CSV 경과 시간 (s)")
    axis.set_ylabel("센서값")
    axis.set_title(title, fontsize=10)
    axis.legend(ncol=8, fontsize=7, loc="upper right")
    palette = plt.get_cmap("Set1")
    for row, obs in enumerate(record_obs, 1):
        strip = figure.add_subplot(grid[row, :], sharex=axis)
        for index, predicted in zip(obs["cache_indices"], obs["prediction_sequence"]):
            start = float(cache_start[index])
            strip.axvspan(start, start + 3.0, color=palette(predicted), alpha=0.8)
            strip.text(start + 1.5, 0.5, CLASS_NAMES[predicted], ha="center", va="center", fontsize=7)
        strip.set_yticks([])
        strip.tick_params(labelbottom=False)
        strip.set_ylabel(obs["fold"].replace("development_", "dev"), rotation=0, ha="right", fontsize=8)
    figure.subplots_adjust(hspace=0.45)
    patterns = [(record_mean, "이 기록 평균"), truth_mean, pred_mean]
    present = [p for p, _ in patterns if p is not None]
    low, high = min(p.min() for p in present), max(p.max() for p in present)
    for column, (pattern, label) in enumerate(patterns):
        heat = figure.add_subplot(grid[-1, column])
        if pattern is None:
            heat.text(0.5, 0.5, "비교 기록 없음", ha="center", va="center")
            heat.set_axis_off()
            continue
        image = heat.imshow(pattern.reshape(4, 4), vmin=low, vmax=high, cmap="viridis")
        for i in range(4):
            for j in range(4):
                heat.text(j, i, f"R{i + 1}{j + 1}\n{pattern[i * 4 + j]:.2f}", ha="center", va="center", fontsize=6, color="white")
        heat.set_xticks([])
        heat.set_yticks([])
        heat.set_title(label, fontsize=9)
    figure.colorbar(image, ax=figure.axes[-3:], shrink=0.8)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def load_comparisons(root: Path, parts):
    """Load stored development predictions aligned window-for-window with 008."""
    from sklearn.metrics import f1_score
    loaded, evidence = {}, {}
    for model in COMPARISON_MODELS:
        branch, folds, files = root / model["branch"], [], {}
        for reference in parts:
            folder = branch / reference["fold"]
            metrics_path, npz_path = folder / "metrics.json", folder / "predictions.npz"
            seal = folder / "metrics_seal.json"
            if seal.is_file() and read_json(seal)["sha256"] != sha256(metrics_path):
                raise ValueError("metrics seal differs: " + str(folder))
            metrics = read_json(metrics_path)
            if "predictions_sha256" in metrics and metrics["predictions_sha256"] != sha256(npz_path):
                raise ValueError("prediction link differs: " + str(folder))
            with np.load(npz_path, allow_pickle=False) as archive:
                part = {key: archive[key] for key in archive.files}
            records = np.asarray([normalize_text(v) for v in part["record_ids"]])
            if not (np.array_equal(part["indices"], reference["indices"]) and np.array_equal(records, reference["record_ids"])
                    and np.array_equal(part["truth"], reference["truth"])):
                raise ValueError("Windows differ from 008: " + str(folder))
            if not np.array_equal(part["prediction"], part["probability"].argmax(1)):
                raise ValueError("Comparison prediction is not argmax: " + str(folder))
            stored = metrics.get("metrics", metrics.get("evaluation", {})).get("macro_f1")
            actual = f1_score(part["truth"], part["prediction"], labels=range(7), average="macro", zero_division=0)
            if stored is None or abs(stored - actual) > 1e-12:
                raise ValueError("Stored macro F1 does not reproduce: " + str(folder))
            folds.append(part["prediction"])
            files[str(npz_path.relative_to(root))] = sha256(npz_path)
            files[str(metrics_path.relative_to(root))] = sha256(metrics_path)
        loaded[model["name"]] = folds
        evidence[model["name"]] = {**model, "files": files,
                                   "split_macro_f1": [float(f1_score(p["truth"], f, labels=range(7), average="macro", zero_division=0))
                                                      for p, f in zip(parts, folds)]}
    return loaded, evidence


def cross_model(record, parts, reference_majority, loaded):
    """Per comparison model: record accuracy over every split that evaluated the record."""
    result, all_wrong, same_wrong = {}, 0, 0
    for name, folds in loaded.items():
        correct = total = 0
        counts = np.zeros(7, dtype=int)
        for part, prediction in zip(parts, folds):
            take = part["record_ids"] == record
            if take.any():
                correct += int((prediction[take] == part["truth"][take]).sum())
                total += int(take.sum())
                counts += np.bincount(prediction[take], minlength=7)
        majority = int(counts.argmax())
        result[name] = (correct / total, majority)
        all_wrong += int(correct == 0)
        same_wrong += int(correct == 0 and majority == reference_majority)
    count = len(loaded)
    if all_wrong == count:
        pattern = "all_models_wrong"
    elif all_wrong >= (count + 1) // 2:
        pattern = "most_models_wrong"
    elif all_wrong <= 1:
        pattern = "mostly_et_specific"
    else:
        pattern = "mixed"
    return result, all_wrong, same_wrong, pattern


CROSS_TEXT = {
    "all_models_wrong": "비교 모델 전부 전체 오분류 → 라벨 또는 센서 구분 한계 가능성 높음",
    "most_models_wrong": "비교 모델 과반 전체 오분류",
    "mixed": "모델마다 다름",
    "mostly_et_specific": "비교 모델 대부분이 일부 이상 맞힘 → ET 고유 오류 가능성",
}


def extract(root: Path, output: Path, make_plots: bool = True) -> dict:
    started = now()
    output = output.resolve()
    if not output.is_relative_to((root / "research/results").resolve()):
        raise ValueError("Output must be a new folder under research/results/")
    if output.exists():
        raise ValueError("Refuse to reuse an existing output folder")
    origin, protocol, parts, inputs = load_origin(root)
    cache_path, dataset_path = origin / "data.npz", origin / "dataset.json"
    if sha256(cache_path) != protocol["cache_sha256"] or sha256(dataset_path) != protocol["dataset_sha256"]:
        raise ValueError("008 cache/dataset link differs")
    with np.load(cache_path, allow_pickle=False) as archive:
        cache = {key: archive[key] for key in ("seq", "y", "groups", "record_ids", "start_s")}
    cache["record_ids"] = np.asarray([normalize_text(v) for v in cache["record_ids"]])
    cache["groups"] = np.asarray([normalize_text(v) for v in cache["groups"]])
    dataset = read_json(dataset_path)
    sources = {normalize_text(r["staged_file"]): r for r in dataset["records"]}

    rows, totals = observations(parts)
    for row in rows:
        indices = np.asarray(row["cache_indices"])
        if not (np.all(cache["record_ids"][indices] == row["record_id"]) and np.all(cache["y"][indices] == row["truth_id"])):
            raise ValueError("Prediction indices do not map to the cached recording: " + row["record_id"])
    loaded, comparison_evidence = load_comparisons(root, parts)

    data_root = root / "data"
    unique = sorted({row["record_id"] for row in rows})
    before = {}
    for record in unique:
        source = sources[record]
        path = data_root / source["source_relative_path"]
        before[record] = file_state(path)
        if before[record]["sha256"] != source["sha256"]:
            raise ValueError("Original CSV differs from 008 dataset hash: " + record)

    prior = {}
    prior_path = root / PRIOR_REVIEW
    if prior_path.is_file():
        for item in read_json(prior_path)["records"]:
            prior[item["sha256"]] = sorted({flag["code"] for flag in item.get("flags", [])})
    excluded = set()
    exclusion_path = root / EXCLUSIONS
    if exclusion_path.is_file():
        excluded = {e["source_sha256"] for e in read_json(exclusion_path)["exclusions"]}

    means, meta = subject_patterns(cache)
    by_record = defaultdict(list)
    for row in rows:
        by_record[row["record_id"]].append(row)
    folds_seen = defaultdict(set)
    for part in parts:
        for record in set(part["record_ids"].tolist()):
            folds_seen[record].add(part["fold"])

    output.mkdir(parents=True)
    plots = output / "plots"
    plots.mkdir()
    review_rows = []
    for number, record in enumerate(unique, 1):
        obs = by_record[record]
        first = obs[0]
        source = sources[record]
        majority = int(np.bincount([o["majority_prediction_id"] for o in obs], minlength=7).argmax())
        hint = similarity_hint(record, first["subject"], first["truth_id"], majority, means, meta)
        own = defaultdict(list)
        for other, (subject, label) in meta.items():
            if other != record and subject == first["subject"]:
                own[label].append(means[other])
        review_id = f"LR{number:03d}"
        plot_name = ""
        if make_plots:
            plot_name = f"plots/{review_id}.png"
            frame = read_sensor_csv(data_root / source["source_relative_path"])
            title = (f"{review_id}  {record}  라벨={CLASS_NAMES[first['truth_id']]}  "
                     f"주 예측={CLASS_NAMES[majority]}  ({HINT_TEXT[hint['hint']]})")
            def reference(label_id, role):
                values = own.get(label_id, [])
                title = f"같은 사람 {role}: {CLASS_NAMES[label_id]} ({len(values)}개 기록)"
                return (np.mean(values, 0) if values else None, title)
            plot_record(output / plot_name, frame, obs, cache["start_s"], means[record],
                        reference(first["truth_id"], "라벨 자세"), reference(majority, "예측 자세"), title)
        evaluated = sorted(folds_seen[record])
        always_wrong_repeat = len(evaluated) > 1 and len(obs) == len(evaluated)
        priority = (1 if always_wrong_repeat else 2 if hint["hint"] == "closer_to_own_predicted"
                    else 3 if all(o["switches"] == 0 for o in obs) else 4)
        per_model, models_wrong, models_same_wrong, pattern = cross_model(record, parts, majority, loaded)
        review_rows.append({
            "review_id": review_id, "review_priority": priority, "record_id": record,
            "source_file": "data/" + source["source_relative_path"], "source_sha256": source["sha256"],
            "subject": first["subject"], "label": CLASS_NAMES[first["truth_id"]],
            "majority_prediction": CLASS_NAMES[majority],
            "other_models_all_wrong": f"{models_wrong}/{len(loaded)}",
            "other_models_same_wrong_posture": f"{models_same_wrong}/{len(loaded)}",
            "cross_model": CROSS_TEXT[pattern],
            **{f"acc_{name}": fmt(value[0], 2) for name, value in per_model.items()},
            **{f"pred_{name}": CLASS_NAMES[value[1]] for name, value in per_model.items()},
            "all_wrong_folds": ";".join(o["fold"] for o in obs), "evaluated_folds": ";".join(evaluated),
            "always_wrong_when_repeated": always_wrong_repeat,
            "windows": first["windows"], "constant_wrong": all(o["switches"] == 0 for o in obs),
            "prediction_sequences": " | ".join(",".join(CLASS_NAMES[p] for p in o["prediction_sequence"]) for o in obs),
            "mean_label_probability": fmt(float(np.mean([o["mean_truth_probability"] for o in obs]))),
            "mean_top_probability": fmt(float(np.mean([o["mean_top_probability"] for o in obs]))),
            "sim_own_label": fmt(hint["sim_own_truth"]), "sim_own_prediction": fmt(hint["sim_own_predicted"]),
            "own_label_records": hint["own_truth_records"], "own_prediction_records": hint["own_predicted_records"],
            "nearest_own_class": "" if hint["nearest_own_class_id"] is None else CLASS_NAMES[hint["nearest_own_class_id"]],
            "hint": HINT_TEXT[hint["hint"]], "prior_review_flags": ";".join(prior.get(source["sha256"], [])),
            "already_training_excluded": source["sha256"] in excluded, "plot": plot_name,
            **{column: "" for column in HUMAN_COLUMNS},
        })
    review_rows.sort(key=lambda r: (r["review_priority"], -int(r["other_models_all_wrong"].split("/")[0]),
                                    r["label"], r["record_id"]))

    observation_rows = [{
        "fold": o["fold"], "record_id": o["record_id"], "subject": o["subject"], "label": CLASS_NAMES[o["truth_id"]],
        "windows": o["windows"], "switches": o["switches"], "majority_prediction": CLASS_NAMES[o["majority_prediction_id"]],
        "prediction_sequence": ",".join(CLASS_NAMES[p] for p in o["prediction_sequence"]),
        "mean_label_probability": fmt(o["mean_truth_probability"]), "mean_top_probability": fmt(o["mean_top_probability"]),
        **{f"mean_p_{name}": fmt(value) for name, value in zip(CLASS_NAMES, o["mean_class_probability"])},
    } for o in rows]

    def write_csv(path, items):
        with path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)

    write_csv(output / "observations.csv", observation_rows)
    write_csv(output / "review_sheet.csv", review_rows)
    (output / "observations.csv").chmod(0o444)

    for record in unique:
        after = file_state(data_root / sources[record]["source_relative_path"])
        if after != before[record]:
            raise ValueError("Original CSV changed during extraction: " + record)
    if sha256(origin / "protocol.json") != PROTOCOL_SHA256:
        raise ValueError("008 protocol changed during extraction")

    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True).stdout
    summary = {
        "schema_version": 1, "version": VERSION, "started_at": started, "finished_at": now(), "timezone": "Asia/Seoul",
        "command": [sys.executable, *sys.argv], "script_sha256": sha256(Path(__file__)),
        "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True).stdout.strip(),
        "dirty_state": status,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "origin_run": ORIGIN, "origin_protocol_sha256": PROTOCOL_SHA256,
        "inputs": {path: sha256(root / path) for path in inputs + [str(cache_path.relative_to(root)),
                   str(dataset_path.relative_to(root))] + [p for p in (PRIOR_REVIEW, EXCLUSIONS) if (root / p).is_file()]},
        "published_totals_verified": totals, "unique_records": len(unique),
        "unique_records_by_label": {name: sum(r["label"] == name for r in review_rows) for name in CLASS_NAMES},
        "hints": {code: sum(r["hint"] == text for r in review_rows) for code, text in HINT_TEXT.items()},
        "review_priority_counts": {str(p): sum(r["review_priority"] == p for r in review_rows) for p in (1, 2, 3, 4)},
        "comparison_models": comparison_evidence,
        "cross_model_counts": {code: sum(r["cross_model"] == text for r in review_rows) for code, text in CROSS_TEXT.items()},
        "other_models_all_wrong_distribution": {str(k): sum(int(r["other_models_all_wrong"].split("/")[0]) == k for r in review_rows)
                                                for k in range(len(COMPARISON_MODELS) + 1)},
        "original_csv_states": before, "original_csvs_unchanged": True, "raw_data_mutated": False,
        "models_loaded": False, "fitting_or_inference": False,
        "review_sheet_initial_sha256": sha256(output / "review_sheet.csv"),
        "observations_sha256": sha256(output / "observations.csv"),
        "similarity": {"pattern": "per-record mean of 008 cache seq (36 resampled samples x 16 channels per 3 s window)",
                       "measure": "cosine similarity after subtracting each pattern's cross-channel mean",
                       "reference": "mean of the same person's other cached recordings per label; the reviewed recording is excluded",
                       "margin": SIMILARITY_MARGIN, "scope": "Review ordering hint only; not a label verdict"},
        "limitations": [
            "91 counts split x recording observations; recordings evaluated in two development splits appear once per split in observations.csv and once in review_sheet.csv.",
            "Development predictions come from a model family already inspected during iterative selection; this extraction adds no new performance evidence.",
            "Similarity uses centered patterns and ignores absolute pressure level; person-internal comparison cannot detect a person whose every recording is mislabeled the same way.",
            "Cache seq window times follow the 008 cache start_s; window shading on plots assumes 3 s windows from original CSV elapsed time zero.",
            "Comparison models were fixed per family before viewing cross-model results; SVM and RF come from runs without ordinary correct-posture restoration. Their errors are correlated (shared features, labels and training people), so agreement is not independent confirmation.",
            "The 80 recordings were selected because ET failed on them; comparison counts describe only this ET-selected set.",
        ],
    }
    (output / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "run.json").chmod(0o444)
    (output / "README.md").write_text(GUIDE.format(
        started=started, unique=len(unique), **{k: totals[k] for k in EXPECTED},
        priority={int(k): v for k, v in summary["review_priority_counts"].items()}, hints=summary["hints"],
        cross=summary["cross_model_counts"], n_models=len(COMPARISON_MODELS),
        model_list=", ".join(f"{m['name']}({m['family']}, {'복원' if m['data_policy'] == 'ordinary_restored' else '미복원'})"
                             for m in COMPARISON_MODELS)), encoding="utf-8")
    return summary


GUIDE = """# 전체 오분류 기록 사람 검토 시트

생성: {started} (Asia/Seoul). 입력: 008 사용자 채택 기준선의 개발 예측 3개 분할
(`research/results/2026-09-22_015526_iterative_008`). 모델 로드·학습·추론은 하지 않았고
`data/` 원본은 읽기만 했다. 실행 전후 원본 CSV의 해시·크기·mtime이 같음을 `run.json`에 기록했다.

- 분할×녹화 관측 {record_observations}개 중 모든 창이 틀린 관측은 **{all_wrong_observations}개**로, 전체 오류 {errors}창 중 {errors_in_all_wrong}창을 차지한다.
- 두 분할에서 모두 전체 오분류된 녹화가 있어 **고유 녹화는 {unique}개**다. `observations.csv`에는 91개 관측을, `review_sheet.csv`에는 녹화당 한 행을 두었다.

## 검토 순서 (`review_priority`)

1. 두 분할에서 반복 평가됐고 매번 전체 오분류: {priority[1]}개
2. 같은 사람의 예측 자세 기록과 더 비슷함(라벨 의심): {priority[2]}개
3. 모든 창이 같은 잘못된 자세로 일관 분류: {priority[3]}개
4. 나머지: {priority[4]}개

`hint`는 같은 사람의 다른 녹화와 16채널 평균 패턴(전체 수준을 뺀 코사인 유사도)을 비교한 결과다.
차이가 0.05 이하면 경계 사례로 표시했다. 결과는 예측 자세 쪽이 더 비슷함 {hints[closer_to_own_predicted]}개,
라벨 자세 쪽이 더 비슷함 {hints[closer_to_own_truth]}개, 비슷한 정도가 같음 {hints[ambiguous]}개, 비교 기록 없음 {hints[no_comparison]}개다.
검토 순서를 정할 때만 참고하며, 이 값으로 라벨을 판정하지 않는다. 한 사람의 모든 기록이 같은 방식으로 잘못 라벨링됐다면 이 비교로는 찾을 수 없다.

## 다른 모델 교차 확인

같은 3개 개발 분할·같은 창의 저장된 예측만 읽었다(새 학습·추론 없음). 비교 모델 {n_models}개는 결과를 보기 전에
모델 계열별로 고정했다: {model_list}. '복원'은 008과 같은 일반 정자세 기록 복원 학습 정책이다.

- `other_models_all_wrong`: 이 녹화를 평가한 모든 분할에서 창을 하나도 맞히지 못한 비교 모델 수
- `other_models_same_wrong_posture`: 그중 ET와 같은 자세로 주로 오분류한 모델 수
- `acc_<모델>`: 녹화 창 정확도(반복 평가 분할 합산), `pred_<모델>`: 가장 많이 예측한 자세
- `cross_model`: 전부 오분류 {cross[all_models_wrong]}개, 과반 오분류 {cross[most_models_wrong]}개, 혼재 {cross[mixed]}개, ET 고유 오류 가능성 {cross[mostly_et_specific]}개

같은 우선순위 안에서는 `other_models_all_wrong`이 큰 녹화를 먼저 두었다. 모델들은 같은 특징·라벨·학습 참가자를 공유하므로
모두 틀렸다고 라벨 오류가 확정되지 않는다. 반대로 대부분의 모델이 맞힌 녹화는 `model_error` 후보로 먼저 검토할 수 있다.

## 그림 (`plots/LRxxx.png`)

위: 원본 CSV의 16채널 시계열. 가운데: 분할별 3초 창 예측(색 = 예측 자세). 아래: 4×4 센서 평균 배치
(이 녹화 / 같은 사람의 라벨 자세 녹화 / 같은 사람의 예측 자세 녹화, 같은 색 척도).

## 사람이 채우는 열

| 열 | 값 |
| --- | --- |
| `reviewer`, `review_date` | 검토자, 날짜(YYYY-MM-DD) |
| `verdict` | `label_error`(라벨 오류) / `sensor_indistinguishable`(자세는 맞지만 센서 패턴으로 구분 불가) / `model_error`(패턴은 라벨 자세와 뚜렷이 같은데 모델이 틀림) / `recording_issue`(착석 불량·센서 이상) / `unclear` |
| `observed_posture` | `label_error`일 때 실제 자세로 판단한 7개 자세 이름 |
| `evidence` | 판단 근거(그림에서 본 점, 측정 메모 등) |
| `cross_check_source` | `video`(동의받은 영상) / `observer`(관찰자 기록) / `none` |
| `cross_check_result` | `confirmed_label` / `confirmed_other:<자세>` / `inconclusive` |
| `proposed_action` | `keep` / `exclusion_candidate` / `relabel_candidate` / `recollect` |
| `notes` | 기타 |

## 규칙

- 이 시트는 제안이다. `data/`의 CSV·sidecar는 수정하거나 재라벨링하지 않는다.
- 실제 제외는 사용자가 명시한 뒤 기존 `research/models/training_exclusions.json` 워크플로(원본 해시 포함)로만 반영한다.
- 개발 예측을 여러 번 관찰한 결과이므로 검토 후 재학습 성능은 새로 고정한 평가로만 주장한다.
- `review_sheet.csv`의 생성 직후 해시는 `run.json`의 `review_sheet_initial_sha256`에 있다. 수정본은 날짜를 붙인 사본으로 저장하길 권장한다.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="new folder under research/results (default: timestamped label_review_all_wrong)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    output = args.output or Path("research/results") / (datetime.now(TZ).strftime("%Y-%m-%d_%H%M%S") + "_label_review_all_wrong")
    output = output if output.is_absolute() else ROOT / output
    summary = extract(ROOT, output, make_plots=not args.no_plots)
    print(json.dumps({"output": str(output), "unique_records": summary["unique_records"],
                      "totals": summary["published_totals_verified"], "hints": summary["hints"],
                      "priority": summary["review_priority_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
