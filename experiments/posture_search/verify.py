"""Verify frozen artifacts and optionally reproduce selected fits without selection.

This command refuses to inspect prediction outcomes until the selection lock and
completed test summary both exist. It never changes configurations, checkpoints,
the selected model, or the original dataset.

Examples:
    .venv/bin/python -m experiments.posture_search.verify
    .venv/bin/python -m experiments.posture_search.verify --refit-ids B00 selected
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from .data import ROOT, OUT, digest, source_inventory, take, write_json


class VerificationError(RuntimeError):
    """Frozen evidence is incomplete, inconsistent, or has changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def read_json(path: Path):
    require(path.is_file(), f"Missing evidence file: {path}")
    return json.loads(path.read_text())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"Missing prediction/cache file: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def recompute_metrics(y, prediction, groups=None) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=np.arange(7), zero_division=0,
    )
    result = {
        "accuracy": float(accuracy_score(y, prediction)), "macro_f1": float(f1.mean()),
        "worst_class_f1": float(f1.min()), "precision": precision, "recall": recall,
        "f1": f1, "support": support,
        "confusion_matrix": confusion_matrix(y, prediction, labels=np.arange(7)),
    }
    if groups is not None:
        result["per_subject"] = {
            str(group): recompute_metrics(y[groups == group], prediction[groups == group])
            for group in sorted(set(groups))
        }
    return result


def compare_metrics(saved: dict, expected: dict, context: str) -> None:
    require(set(saved) == set(expected), f"{context}: metric keys differ")
    for key, value in expected.items():
        name = f"{context}/{key}"
        if isinstance(value, dict):
            compare_metrics(saved[key], value, name)
        else:
            actual = np.asarray(saved[key])
            reference = np.asarray(value)
            require(actual.shape == reference.shape, f"{name}: metric shape differs")
            if key in {"support", "confusion_matrix"}:
                require(np.array_equal(actual, reference), f"{name}: count metric differs")
            else:
                require(np.allclose(actual, reference, rtol=0, atol=1e-12), f"{name}: metric differs")


def validate_probability(probability: np.ndarray, count: int, context: str) -> None:
    require(probability.shape == (count, 7), f"{context}: expected {count} by 7 probabilities")
    require(np.isfinite(probability).all(), f"{context}: nonfinite probability")
    require(np.all(probability >= 0), f"{context}: negative probability")
    require(np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-5),
            f"{context}: probabilities do not sum to one")


def verify_prediction_rows(saved: dict, expected: dict, context: str) -> None:
    required = {"proba", "y", "groups", "record_ids", "start_s"}
    require(set(saved) == required, f"{context}: prediction fields differ")
    for key in required - {"proba"}:
        require(np.array_equal(saved[key], expected[key]), f"{context}: different or reordered {key}")
    validate_probability(saved["proba"], len(expected["y"]), context)


def verify_group_config(config, outer_train: set, context: str) -> int:
    """Validate every nested learner and stacking OOF audit against outer training groups."""
    if isinstance(config, list):
        return sum(verify_group_config(value, outer_train, context) for value in config)
    if not isinstance(config, dict):
        return 0
    current = outer_train
    if config.get("training_groups") is not None:
        current = set(config["training_groups"])
        require(bool(current) and current <= outer_train, f"{context}: learner contains non-training subject")
    fold_count = 0
    if "oof_folds" in config:
        held = Counter()
        for fold in config["oof_folds"]:
            train, validation = set(fold["train_subjects"]), set(fold["validation_subjects"])
            require(bool(train) and bool(validation), f"{context}: empty OOF group partition")
            require(not train & validation, f"{context}: subject leakage in OOF fold")
            require(train | validation == current, f"{context}: OOF groups differ from learner training groups")
            held.update(validation)
            fold_count += 1
        require(set(held) == current and all(count == 1 for count in held.values()),
                f"{context}: OOF validation subjects are not a partition")
        if "ensemble_members" in config:
            require(config["meta_input_features"] == 7 * len(config["ensemble_members"]),
                    f"{context}: stacking meta feature width is inconsistent")
    if "bootstrap_group_draws" in config:
        for draw in config["bootstrap_group_draws"]:
            require(bool(draw) and set(draw) <= current, f"{context}: bootstrap sampled non-training subject")
    for key, value in config.items():
        if key not in {"oof_folds", "bootstrap_group_draws", "training_groups"}:
            fold_count += verify_group_config(value, current, f"{context}/{key}")
    return fold_count


def verify_dataset(data: dict, manifest: dict) -> None:
    count = len(data["y"])
    require(all(len(value) == count for value in data.values()), "Cache arrays have different row counts")
    require(data["seq"].shape == (count, 36, 16), "Unexpected resampled window shape")
    for name in ("seq", "features", "personal", "start_s"):
        require(np.isfinite(data[name]).all(), f"Nonfinite cached {name}")
    rows = manifest["rows"]
    require(len(rows) == count, "Manifest window rows differ from cache")
    for index, row in enumerate(rows):
        require(row["relative_path"] == data["record_ids"][index], "Record provenance differs from cache")
        require(row["subject"] == data["groups"][index], "Subject provenance differs from cache")
        require(row["start_s"] == data["start_s"][index], "Window time differs from cache")
        require(row["label"] == manifest["class_names"][data["y"][index]], "Label provenance differs from cache")
    registrations = manifest["registrations"]
    records = {row["relative_path"]: row for row in rows}
    for record in list(records.values()) + registrations:
        source_name = f"data/{record['relative_path']}"
        require(record["sha256"] == manifest["source_inventory"].get(source_name),
                f"Recording checksum differs from source inventory: {source_name}")
    for key in ("relative_path", "sha256", "sensor_sha256"):
        require(not {r[key] for r in registrations} & {r[key] for r in records.values()},
                f"Registration/evaluation overlap by {key}")
    require(not manifest["audit"]["duplicate_raw_files"], "Duplicated raw recording")
    require(not manifest["audit"]["duplicate_sensor_sequences"], "Duplicated sensor sequence")
    registration_by_subject = {r["subject"]: r for r in registrations}
    require(len(registration_by_subject) == len(registrations), "Duplicate calibration subject")
    require(set(registration_by_subject) == set(data["groups"]), "Calibration and evaluated subjects differ")
    for split in manifest["partitions"]:
        assignments = [subject for name in ("train", "validation", "test") for subject in split[name]]
        require(len(assignments) == len(set(assignments)), f"{split['id']}: subject appears in two partitions")
        require(set(assignments) == set(data["groups"]), f"{split['id']}: incomplete subject partition")
        content = []
        for name in ("train", "validation", "test"):
            mask = np.isin(data["groups"], split[name])
            require(set(data["y"][mask]) == set(range(7)), f"{split['id']}/{name}: missing class")
            paths = set(data["record_ids"][mask])
            sources = [records[path] for path in paths] + [registration_by_subject[s] for s in split[name]]
            content.append({key: {r[key] for r in sources} for key in ("relative_path", "sha256", "sensor_sha256")})
        for left in range(3):
            for right in range(left + 1, 3):
                for key in content[left]:
                    require(not content[left][key] & content[right][key],
                            f"{split['id']}: cross-partition {key} leakage")


def includes_deep(spec: dict, specs: dict[str, dict], visited=None) -> bool:
    visited = set() if visited is None else visited
    require(spec["id"] not in visited, "Cycle in model graph")
    if spec["family"] == "deep":
        return True
    children = spec.get("members", []) if spec["family"] == "fusion" else (
        [spec.get("parent", "B00")] if spec["family"] == "temporal" else [])
    return any(includes_deep(specs[child], specs, visited | {spec["id"]}) for child in children)


def compare_probability(actual, saved, tolerance: float, context: str) -> float:
    require(actual.shape == saved.shape, f"{context}: probability shape differs")
    require(np.isfinite(actual).all(), f"{context}: nonfinite reproduced probability")
    difference = float(np.max(np.abs(actual - saved))) if actual.size else 0.0
    require(difference <= tolerance, f"{context}: probability difference {difference:.3g} exceeds {tolerance}")
    return difference


def verify(output=OUT, refit_ids=()) -> dict:
    output = Path(output)
    # Do not open any prediction outcomes before selection and the test stage finish.
    require((output / "selection_lock.json").is_file(), "Selection is not locked; outcome verification is disabled")
    require((output / "test_summary.json").is_file(), "Test stage is incomplete; outcome verification is disabled")
    started = time.perf_counter()
    lock = read_json(output / "selection_lock.json")
    require(lock["test_used_for_selection"] is False, "Test was used for model selection")
    manifest = read_json(output / "dataset.json")
    if "dataset_manifest_sha256" in lock:
        require(digest(output / "dataset.json") == lock["dataset_manifest_sha256"], "Locked dataset manifest changed")
    if "dataset_cache_sha256" in lock:
        require(digest(output / "data.npz") == lock["dataset_cache_sha256"], "Locked dataset cache changed")
    initial_inventory = source_inventory()
    require(initial_inventory == manifest["source_inventory"], "Dataset CSV/sidecar inventory has changed")
    require(digest(output / "data.npz") == manifest["cache_sha256"], "Dataset cache checksum mismatch")
    fingerprint = hashlib.sha256(json.dumps({"inventory": initial_inventory, "partitions": manifest["partitions"]},
                                           sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    require(fingerprint == manifest["fingerprint"] == lock["dataset_fingerprint"], "Dataset fingerprint mismatch")
    require(digest(output / "protocol.json") == lock["protocol_sha256"], "Protocol checksum mismatch")
    required_sources = {"src/posture_core.py", "src/window_pipeline.py", "src/app_services.py", "validate_personalization.py"}
    require(required_sources <= set(lock["source_hashes"]), "Lock omits a core preprocessing dependency")
    for relative, checksum in lock["source_hashes"].items():
        require(digest(ROOT / relative) == checksum, f"Locked source changed: {relative}")
    for relative, checksum in lock["checkpoints"].items():
        require(digest(output / relative) == checksum, f"Locked checkpoint changed: {relative}")
    require(digest(manifest["baseline"]["artifact"]) == manifest["baseline"]["sha256"], "Deployed baseline changed")
    data = load_npz(output / "data.npz")
    verify_dataset(data, manifest)
    specs = {spec["id"]: spec for spec in lock["all_specs"]}
    require(len(specs) == len(lock["all_specs"]), "Duplicate experiment IDs in lock")
    require(lock["selected_spec"] == specs[lock["selected_id"]], "Selected spec differs from registered experiment")
    partitions = {split["id"]: split for split in manifest["partitions"]}
    primary = partitions["primary"]
    test_data = take(data, np.isin(data["groups"], primary["test"]))
    summary = read_json(output / "test_summary.json")
    require(summary["selected_id"] == lock["selected_id"], "Test summary changed selected ID")
    require(summary["test_windows"] == len(test_data["y"]), "Wrong test window count")
    summary_rows = {row["spec"]["id"]: row for row in summary["results"]}
    require(set(summary_rows) == set(specs) and len(summary_rows) == len(summary["results"]), "Test summary experiment set differs")
    lock_sha = digest(output / "selection_lock.json")
    comparisons, oof_folds = [], 0
    for identifier, spec in specs.items():
        tolerance = 1e-6 if includes_deep(spec, specs) else 1e-8
        for split_id, split in partitions.items():
            folder = output / identifier / split_id
            relative = str((folder / "model.joblib").relative_to(output))
            require(relative in lock["checkpoints"], f"{relative}: checkpoint missing from lock")
            record = read_json(folder / "validation.json")
            require(record["spec"] == spec and record["split"] == split, f"{identifier}/{split_id}: config changed")
            require(record["dataset_fingerprint"] == fingerprint, f"{identifier}/{split_id}: different dataset")
            require(record["checkpoint_sha256"] == lock["checkpoints"][relative], f"{identifier}/{split_id}: checkpoint hash differs")
            expected = take(data, np.isin(data["groups"], split["validation"]))
            saved = load_npz(folder / "validation_predictions.npz")
            context = f"{identifier}/{split_id}/validation"
            verify_prediction_rows(saved, expected, context)
            compare_metrics(record["metrics"], recompute_metrics(saved["y"], saved["proba"].argmax(1), saved["groups"]), context)
            oof_folds += verify_group_config(record["complexity"], set(split["train"]), context)
            model = joblib.load(folder / "model.joblib")
            difference = compare_probability(model.predict_proba(expected), saved["proba"], tolerance, context)
            comparisons.append({"experiment": identifier, "split": split_id, "stage": "validation", "max_abs_error": difference})
            if split_id == "primary":
                row = read_json(folder / "test.json")
                require(row == summary_rows[identifier], f"{identifier}: test summary differs from per-arm evidence")
                require(row["spec"] == spec and row["lock_sha256"] == lock_sha, f"{identifier}: test selection lock differs")
                require(row["model_sha256"] == lock["checkpoints"][relative], f"{identifier}: test model hash differs")
                prediction = load_npz(folder / "test_predictions.npz")
                context = f"{identifier}/test"
                verify_prediction_rows(prediction, test_data, context)
                compare_metrics(row["metrics"], recompute_metrics(prediction["y"], prediction["proba"].argmax(1), prediction["groups"]), context)
                difference = compare_probability(model.predict_proba(test_data), prediction["proba"], tolerance, context)
                comparisons.append({"experiment": identifier, "split": split_id, "stage": "test", "max_abs_error": difference})
        print(f"Verified {identifier}: all validation splits and identical test rows", flush=True)
    bundle = joblib.load(manifest["baseline"]["artifact"])
    deployed = bundle.get("svm_model", bundle.get("model"))
    probability = deployed.predict_proba(test_data["features"])
    require(summary["stored_baseline"]["sha256"] == manifest["baseline"]["sha256"], "Stored baseline reference hash differs")
    compare_metrics(summary["stored_baseline"]["metrics"], recompute_metrics(test_data["y"], probability.argmax(1), test_data["groups"]), "stored_baseline/test")
    refits = []
    fitted = {}
    train = take(data, np.isin(data["groups"], primary["train"]))
    validation = take(data, np.isin(data["groups"], primary["validation"]))

    def refit(identifier):
        if identifier in fitted:
            return fitted[identifier]
        from .run import Arm, Fusion, Temporal
        spec = specs[identifier]
        if spec["family"] == "fusion":
            model = Fusion([refit(member) for member in spec["members"]], spec.get("weights"))
        elif spec["family"] == "temporal":
            model = Temporal(refit(spec.get("parent", "B00")), spec["kind"])
        else:
            model = Arm(spec, primary["seed"]).fit(train)
        fitted[identifier] = model
        return model

    requested = list(dict.fromkeys(lock["selected_id"] if identifier == "selected" else identifier for identifier in refit_ids))
    for identifier in requested:
        require(identifier in specs, f"Unknown refit experiment: {identifier}")
        before = time.perf_counter()
        model = refit(identifier)
        tolerance = 1e-6 if includes_deep(specs[identifier], specs) else 1e-8
        errors = {}
        for stage, expected in (("validation", validation), ("test", test_data)):
            saved = load_npz(output / identifier / "primary" / f"{stage}_predictions.npz")
            errors[stage] = compare_probability(model.predict_proba(expected), saved["proba"], tolerance, f"refit/{identifier}/{stage}")
        refits.append({"experiment": identifier, "split": "primary", "seed": primary["seed"],
                       "tolerance": tolerance, "max_abs_errors": errors, "seconds": time.perf_counter() - before})
        print(f"Reproduced primary fit {identifier} within tolerance {tolerance}", flush=True)
    require(source_inventory() == initial_inventory, "Dataset changed during verification")
    require(digest(output / "selection_lock.json") == lock_sha, "Selection lock changed during verification")
    require(digest(output / "protocol.json") == lock["protocol_sha256"], "Protocol changed during verification")
    for relative, checksum in lock["source_hashes"].items():
        require(digest(ROOT / relative) == checksum, f"Source changed during verification: {relative}")
    for relative, checksum in lock["checkpoints"].items():
        require(digest(output / relative) == checksum, f"Checkpoint changed during verification: {relative}")
    result = {
        "schema_version": 1, "status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
        "selection_lock_sha256": lock_sha, "dataset_fingerprint": fingerprint,
        "source_files_checked": len(initial_inventory), "source_dependencies_checked": len(lock["source_hashes"]),
        "checkpoints_checked": len(lock["checkpoints"]), "experiment_count": len(specs),
        "prediction_metric_checks": len(comparisons), "group_oof_folds_checked": oof_folds,
        "test_rows_identical_for_all_arms": True, "prediction_comparisons": comparisons,
        "refit_results": refits, "selection_modified": False, "elapsed_seconds": time.perf_counter() - started,
        "verifier_sha256": digest(Path(__file__)),
    }
    write_json(output / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--refit-ids", nargs="*", default=[], help="Refit primary experiment IDs; 'selected' resolves the locked winner")
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=1):
            result = verify(args.output, args.refit_ids)
    except VerificationError as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(f"Verification passed: {result['experiment_count']} experiments, {result['prediction_metric_checks']} metric checks")


if __name__ == "__main__":
    main()
