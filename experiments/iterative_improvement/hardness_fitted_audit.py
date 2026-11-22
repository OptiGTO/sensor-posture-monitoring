"""Independently compare saved forests with TRAIN rows and frozen OOF evidence."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.iterative_improvement.feature_tail_audit import selected_rows
from experiments.iterative_improvement.record_training import append_record_training
from src.hardness_pipeline_v1 import verify_fitted_hardness_identity, _array_digest, _json_digest
from src.posture_core import normalize_text


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    out, diagnostic = args.output.resolve(), args.diagnostic.resolve()
    started = datetime.now().astimezone().isoformat()
    protocol = read(out / "protocol.json")
    require(digest(out / "protocol.json") == read(out / "protocol_lock.json")["sha256"], "Protocol changed")
    for relative, expected in protocol["source_hashes"].items():
        require(digest(out / "source_snapshot" / relative) == expected, "Frozen source changed: " + relative)
    for relative in ("experiments/iterative_improvement/hardness_fitted_audit.py",
            "experiments/iterative_improvement/feature_tail_audit.py",
            "experiments/iterative_improvement/record_training.py",
            "experiments/iterative_improvement/supplemental.py", "src/hardness_pipeline_v1.py",
            "src/posture_core.py", "src/window_pipeline.py", "src/__init__.py"):
        require(digest(ROOT / relative) == protocol["source_hashes"][relative], "Current audit dependency changed")
    require(digest(out / "data.npz") == protocol["cache_sha256"], "Feature cache changed")
    require(digest(out / "supplemental_training.npz") == protocol["supplemental_training"]["cache_sha256"], "Ordinary cache changed")
    diagnostic_config = read(diagnostic / "diagnostic_config.json")
    require(digest(diagnostic / "diagnostic_config.json") == read(diagnostic / "diagnostic_lock.json")["config_sha256"], "OOF diagnostic config changed")
    require(diagnostic_config["outer_seed"] == protocol["seeds"][0], "OOF diagnostic seed differs")
    diagnostic_result = read(diagnostic / "oof_record_error_audit.json")
    diagnostic_digest = digest(diagnostic / "oof_record_error_audit.json")
    require(diagnostic_digest == read(diagnostic / "audit_seal.json")["sha256"], "OOF result seal changed")
    evidence = protocol["hypotheses"]["independent_oof_evidence"]
    require(diagnostic_digest in evidence and digest(diagnostic / "diagnostic_config.json") in evidence,
        "Independent OOF evidence was not pinned before outer evaluation")
    require(diagnostic_config["partitions"] == protocol["partitions"] and diagnostic_config["weights"]["coefficient"] == .5,
        "Independent OOF partitions or weighting rule differ")
    caches = {}
    for key, filename in (("base", "data.npz"), ("ordinary", "supplemental_training.npz")):
        with np.load(out / filename, allow_pickle=False) as cache:
            caches[key] = {"path": out / filename, "groups": cache["groups"], "record_ids": cache["record_ids"]}
    protected = read(out / "protected_before.json")
    def protect():
        for relative, expected in protected.items():
            path = ROOT / relative
            require(digest(path) == expected["sha256"] and path.stat().st_mtime_ns == expected["mtime_ns"], "Protected file changed: " + relative)
    protect()
    audited, baseline_reproduction = [], []
    for part in protocol["partitions"]:
        old_folder = ROOT / "research/results/2026-09-22_013650_adopt_et_ordinary/development/ET_ordinary" / part["id"]
        new_folder = out / "development/baseline" / part["id"]
        old_metrics, new_metrics = read(old_folder / "metrics.json"), read(new_folder / "metrics.json")
        for folder, metrics in ((old_folder, old_metrics), (new_folder, new_metrics)):
            require(digest(folder / "metrics.json") == read(folder / "metrics_seal.json")["sha256"], "Baseline metrics changed")
            require(digest(folder / "predictions.npz") == metrics["predictions_sha256"], "Baseline prediction evidence changed")
        require(old_metrics["metrics"] == new_metrics["metrics"] and old_metrics["per_subject"] == new_metrics["per_subject"],
            "Adopted baseline scores were not reproduced")
        with np.load(old_folder / "predictions.npz", allow_pickle=False) as old, np.load(new_folder / "predictions.npz", allow_pickle=False) as new:
            require(set(old.files) == set(new.files) and all(exact(old[key], new[key]) for key in old.files),
                "Adopted baseline predictions were not reproduced exactly")
        baseline_reproduction.append({"partition": part["id"], "array_dtype_shape_bytes_equal": True,
            "metrics_per_subject_equal": True, "new_predictions_sha256": new_metrics["predictions_sha256"],
            "reference_predictions_sha256": old_metrics["predictions_sha256"]})
        values = {}
        for key, cache in caches.items():
            take = np.flatnonzero(np.isin(cache["groups"], part["train"]))
            values[key] = {"X": selected_rows(cache["path"], "X", take, len(cache["groups"])),
                "y": selected_rows(cache["path"], "y", take, len(cache["groups"])),
                "groups": cache["groups"][take], "record_ids": cache["record_ids"][take]}
        X, y, groups, records, restoration = append_record_training(values["base"], values["ordinary"], np.arange(len(values["base"]["y"])))
        groups = np.asarray([normalize_text(value) for value in groups])
        records = np.asarray([normalize_text(value) for value in records])
        reference_scaler = StandardScaler().fit(X)
        diagnostic_folder = diagnostic / "outer_train" / part["id"]
        diagnostic_row = read(diagnostic_folder / "diagnostic.json")
        require(diagnostic_row == next(row for row in diagnostic_result["partitions"] if row["outer_partition"] == part["id"]),
            "Partition diagnostic differs from preregistered result")
        seal = read(diagnostic_folder / "diagnostic_seal.json")
        require(digest(diagnostic_folder / "diagnostic.json") == seal["sha256"], "OOF diagnostic changed")
        require(digest(diagnostic_folder / "oof_record_weights.npz") == seal["arrays_sha256"], "OOF arrays changed")
        with np.load(diagnostic_folder / "oof_record_weights.npz", allow_pickle=False) as saved:
            independent = dict(saved)
        for key, actual in (("truth", y), ("groups", groups), ("record_ids", records)):
            require(exact(independent[key], actual), "OOF diagnostic TRAIN rows differ: " + key)
        for branch in protocol["branches"].values():
            for name, recipe in branch.items():
                folder = out / "development" / name / part["id"]
                metrics = read(folder / "metrics.json")
                require(digest(folder / "metrics.json") == read(folder / "metrics_seal.json")["sha256"], "Fit metrics seal changed")
                require(metrics["protocol_sha256"] == digest(out / "protocol.json") and metrics["pipeline"] == recipe
                    and metrics["seed"] == protocol["seeds"][0], "Fit protocol, recipe or seed differs")
                require(digest(folder / "model.joblib") == metrics["model_sha256"], "Saved model changed")
                model = joblib.load(folder / "model.joblib")
                verify_fitted_hardness_identity(model, recipe, protocol["seeds"][0])
                require(metrics["training_details"] == model.training_details_, "Fit audit differs from saved model")
                for attribute, expected in (("training_input_sha256_", _array_digest(X)),
                        ("training_labels_sha256_", _array_digest(y.astype(np.int64))),
                        ("training_groups_sha256_", _json_digest(groups.tolist())),
                        ("training_records_sha256_", _json_digest(records.tolist()))):
                    require(getattr(model, attribute) == expected, "Actual TRAIN differs from model: " + attribute)
                for key in ("mean_", "var_", "scale_"):
                    require(exact(getattr(reference_scaler, key), getattr(model.scaler_, key)), "Final TRAIN scaler differs")
                require(metrics["supplemental_training_audit"] == restoration, "Restoration differs")
                checked_oof = recipe["training"]["hardness"]["coefficient"] > 0
                if checked_oof:
                    for attribute, key in (("oof_prediction_", "oof_prediction"), ("oof_fold_", "oof_fold_index"),
                            ("oof_probability_", "oof_probability"), ("sample_weight_", "sample_weight")):
                        require(exact(getattr(model, attribute), independent[key]), "Independent OOF differs: " + attribute)
                    for stored, independent_fold in zip(model.hardness_audit_["folds"], diagnostic_row["inner_fits"]):
                        require(stored["seed"] == independent_fold["seed"], "Inner teacher seed differs")
                        require(stored["train_indices"] == independent_fold["train_indices"] and
                            stored["heldout_indices"] == independent_fold["oof_indices"], "Inner teacher split differs")
                        require(stored["training_input_sha256"] == _array_digest(X[stored["train_indices"]]) and
                            stored["heldout_input_sha256"] == _array_digest(X[stored["heldout_indices"]]), "Actual inner TRAIN or OOF input differs")
                        require(digest(ROOT / independent_fold["model"]["path"]) == independent_fold["model"]["sha256"], "Independent teacher changed")
                        teacher = joblib.load(ROOT / independent_fold["model"]["path"])
                        for key in ("mean_", "var_", "scale_"):
                            require(exact(np.asarray(stored["scaler"][key]), getattr(teacher["scaler"], key)), "Independent inner scaler differs")
                    for record in model.hardness_audit_["weighting"]["records"]:
                        mask = records == record["record_id"]
                        require(np.all(independent["record_error"][mask] == record["error_rate"]), "Independent record error differs")
                        require(np.all(independent["raw_record_weight"][mask] == record["raw_factor"]), "Independent record factor differs")
                audited.append({"candidate": name, "partition": part["id"], "training_windows": len(y),
                    "training_records": len(np.unique(records)), "training_subjects": len(np.unique(groups)),
                    "model_sha256": metrics["model_sha256"], "full_state_verified": True,
                    "actual_train_row_hashes_equal": True, "fit_seals_linked": True,
                    "actual_train_scaler_bytes_equal": True, "independent_oof_bytes_equal": True if checked_oof else None,
                    "oof_teachers_checked": 4 if checked_oof else 0})
    protect()
    result = {"schema_version": 1, "started_at": started, "finished_at": datetime.now().astimezone().isoformat(),
        "protocol_sha256": digest(out / "protocol.json"), "script_sha256": digest(__file__),
        "diagnostic": str(diagnostic.relative_to(ROOT)), "diagnostic_config_sha256": digest(diagnostic / "diagnostic_config.json"),
        "models": audited, "baseline_reproduction": baseline_reproduction,
        "frozen_sources_verified": len(protocol["source_hashes"]), "protected_files_unchanged": len(protected),
        "new_model_fits": 0, "new_predictions": 0, "outer_evaluation_used": False,
        "scope": "Saved fitted-state and TRAIN-only numerical audit; no new performance or hardware evidence"}
    destination = out / "audits/hardness_fitted_identity.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    destination.chmod(0o444)
    print(json.dumps({"audit": str(destination.relative_to(ROOT)), "sha256": digest(destination), "models": len(audited)}))


if __name__ == "__main__":
    main()
