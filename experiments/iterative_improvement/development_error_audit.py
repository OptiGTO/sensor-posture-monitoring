"""Describe sealed development predictions; never load training or test data."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

RUNS = ["2026-09-22_004500_iterative_001", "2026-09-22_004500_iterative_002",
        "2026-09-22_005300_iterative_003", "2026-09-22_010100_iterative_004"]
MAIN = {"ET_baseline": (0, "baseline"), "XGB_all": (1, "XGB_reference"),
        "CAT_all": (1, "CAT_all"), "SVM_reference": (0, "SVM_reference")}
LABELS = ["정자세", "앞으로 숙이기", "뒤로 기대기", "왼쪽 기울임", "오른쪽 기울임",
          "왼쪽 다리 꼬기", "오른쪽 다리 꼬기"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(truth, prediction):
    return {"accuracy": float(np.mean(truth == prediction)),
            "macro_f1": float(f1_score(truth, prediction, labels=range(7), average="macro", zero_division=0)),
            "per_class_f1": f1_score(truth, prediction, labels=range(7), average=None, zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(truth, prediction, labels=range(7)).tolist()}


def describe(parts):
    joined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    truth, pred, groups = (joined[k] for k in ("truth", "prediction", "groups"))
    records, transitions, switches, isolated, all_wrong_errors, constant_wrong_errors = [], 0, 0, 0, 0, 0
    by_position = {k: [0, 0] for k in ("first", "middle", "last")}
    for fold, part in enumerate(parts, 1):
        for rec in sorted(set(part["record_ids"])):
            take = np.flatnonzero(part["record_ids"] == rec)
            take = take[np.argsort(part["indices"][take])]
            assert len(set(part["truth"][take])) == 1
            assert len(set(part["groups"][take])) == 1
            assert np.all(np.diff(part["indices"][take]) == 1)
            y, p = part["truth"][take], part["prediction"][take]
            correct = p == y
            transition_count = int(np.count_nonzero(p[1:] != p[:-1]))
            transitions += max(0, len(p) - 1)
            switches += transition_count
            isolated += int(np.count_nonzero(~correct[1:-1] & correct[:-2] & correct[2:]))
            if not correct.any():
                all_wrong_errors += len(p)
                if transition_count == 0:
                    constant_wrong_errors += len(p)
            for label, mask in [("first", np.arange(len(p)) == 0),
                                ("middle", (np.arange(len(p)) > 0) & (np.arange(len(p)) < len(p) - 1)),
                                ("last", np.arange(len(p)) == len(p) - 1)]:
                by_position[label][0] += int(correct[mask].sum())
                by_position[label][1] += int(mask.sum())
            records.append({"fold": fold, "record_id": str(rec), "subject": str(part["groups"][take[0]]),
                            "truth": int(y[0]), "windows": len(p), "errors": int((~correct).sum()),
                            "prediction_switches": transition_count, "prediction_sequence": p.tolist(),
                            "accuracy": float(correct.mean())})
    error_count = int(np.count_nonzero(truth != pred))
    probability = joined["probability"]
    sorted_p = np.sort(probability, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    return {"pooled": stats(truth, pred), "split_macro_f1": [stats(p["truth"], p["prediction"])["macro_f1"] for p in parts],
            "mean_split_macro_f1": float(np.mean([stats(p["truth"], p["prediction"])["macro_f1"] for p in parts])),
            "subjects": {str(g): {"windows": int((groups == g).sum()), **stats(truth[groups == g], pred[groups == g])}
                         for g in sorted(set(groups))}, "record_details": records,
            "record_summary": {"observations": len(records), "unique_records": len(set(joined["record_ids"])),
                               "perfect": sum(r["errors"] == 0 for r in records),
                               "all_wrong": sum(r["errors"] == r["windows"] for r in records),
                               "mixed": sum(0 < r["errors"] < r["windows"] for r in records),
                               "constant_prediction": sum(r["prediction_switches"] == 0 for r in records),
                               "constant_wrong": sum(r["errors"] == r["windows"] and r["prediction_switches"] == 0 for r in records),
                               "mean_record_accuracy": float(np.mean([r["accuracy"] for r in records])),
                               "total_errors": error_count, "errors_in_all_wrong_records": all_wrong_errors,
                               "errors_in_constant_wrong_records": constant_wrong_errors,
                               "isolated_interior_error_windows": isolated,
                               "within_record_adjacent_pairs": transitions, "prediction_switches": switches,
                               "positions_correct_total": by_position},
            "confidence": {"mean_top_probability_correct": float(np.max(probability, axis=1)[pred == truth].mean()),
                           "mean_top_probability_wrong": float(np.max(probability, axis=1)[pred != truth].mean()),
                           "mean_margin_correct": float(margin[pred == truth].mean()),
                           "mean_margin_wrong": float(margin[pred != truth].mean())}}


def audit(root, output):
    started = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    output = output.resolve()
    if output.exists():
        raise ValueError("Refuse to overwrite audit evidence")
    inputs, models, arrays = [], {}, {}
    for number, name in enumerate(RUNS):
        for branch in sorted((root / "research/results" / name / "development").iterdir()):
            if not branch.is_dir():
                continue
            parts = []
            for fold in range(1, 4):
                folder = branch / f"development_{fold}"
                metrics_path, npz_path = folder / "metrics.json", folder / "predictions.npz"
                row = json.loads(metrics_path.read_text())
                seal = json.loads((folder / "metrics_seal.json").read_text())
                assert seal["sha256"] == sha(metrics_path)
                assert row["predictions_sha256"] == sha(npz_path)
                with np.load(npz_path, allow_pickle=False) as archive:
                    part = {k: archive[k] for k in archive.files}
                assert not set(row["training_subjects"]) & set(part["groups"])
                assert set(row["evaluation_subjects"]) == set(part["groups"])
                assert np.array_equal(part["prediction"], part["probability"].argmax(axis=1))
                recomputed = stats(part["truth"], part["prediction"])
                for key in ("macro_f1", "accuracy"):
                    assert abs(recomputed[key] - row["metrics"][key]) < 1e-12
                assert recomputed["confusion_matrix"] == row["metrics"]["confusion_matrix"]
                inputs.append({"metrics": str(metrics_path.relative_to(root)), "metrics_sha256": sha(metrics_path),
                               "predictions": str(npz_path.relative_to(root)), "predictions_sha256": sha(npz_path)})
                parts.append(part)
            key = f"{number + 1:03d}/{branch.name}"
            arrays[key] = parts
            models[key] = {"pipeline": row["pipeline"], "split_macro_f1": [stats(p["truth"], p["prediction"])["macro_f1"] for p in parts]}
            models[key]["mean_split_macro_f1"] = float(np.mean(models[key]["split_macro_f1"]))
    reference = arrays["001/baseline"]
    for key, parts in arrays.items():
        for original, current in zip(reference, parts):
            for field in ("indices", "truth", "groups", "record_ids"):
                assert np.array_equal(original[field], current[field]), (key, field)
            if key.endswith("/baseline"):
                for field in ("probability", "prediction"):
                    assert np.array_equal(original[field], current[field]), (key, field)
    main_arrays = {name: arrays[f"{run + 1:03d}/{branch}"] for name, (run, branch) in MAIN.items()}
    details = {name: describe(parts) for name, parts in main_arrays.items()}
    joined = {name: {field: np.concatenate([part[field] for part in parts]) for field in parts[0]}
              for name, parts in main_arrays.items()}
    complementarity = []
    for a, b in itertools.combinations(joined, 2):
        truth, pa, pb = joined[a]["truth"], joined[a]["prediction"], joined[b]["prediction"]
        ca, cb = pa == truth, pb == truth
        complementarity.append({"a": a, "b": b, "windows": len(truth),
                               "both_correct": int((ca & cb).sum()), "only_a_correct": int((ca & ~cb).sum()),
                               "only_b_correct": int((~ca & cb).sum()), "both_wrong": int((~ca & ~cb).sum()),
                               "both_wrong_same_prediction": int((~ca & ~cb & (pa == pb)).sum()),
                               "disagree": int((pa != pb).sum())})
    all_correct = np.column_stack([joined[name]["prediction"] == joined[name]["truth"] for name in joined])
    rows = reference
    payload = {"started_at": started, "finished_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
               "script_sha256": sha(Path(__file__)), "labels": LABELS, "inputs": inputs,
               "scope": "development_predictions_only_no_fitting_no_new_model_selection_no_test_opening",
               "baseline_identical_across_iterations": True,
               "fold_windows": [len(p["truth"]) for p in rows],
               "fold_record_observations": [len(set(p["record_ids"])) for p in rows],
               "unique_windows": len(set(np.concatenate([p["indices"] for p in rows]))),
               "unique_subjects": len(set(np.concatenate([p["groups"] for p in rows]))),
               "all_models": models, "main": details, "pairwise": complementarity,
               "all_four_wrong_windows": int((~all_correct.any(axis=1)).sum()),
               "all_four_correct_windows": int(all_correct.all(axis=1).sum())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[2], args.output)
    print(json.dumps({"unique_subjects": result["unique_subjects"], "unique_windows": result["unique_windows"],
                      "fold_windows": result["fold_windows"], "input_prediction_files": len(result["inputs"]),
                      "models": {k: {"mean_split_macro_f1": v["mean_split_macro_f1"],
                                     "record_summary": v["record_summary"]} for k, v in result["main"].items()}}, ensure_ascii=False))
