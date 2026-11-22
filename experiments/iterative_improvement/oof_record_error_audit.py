"""Frozen TRAIN-only GroupKFold recording-error diagnostic; no outer evaluation.

The fixed twelve fits estimate recording errors within three outer TRAIN sets.
The derived weights are audited, never used to fit a final model here.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from zoneinfo import ZoneInfo

ORIGIN = "research/results/2026-09-22_044600_iterative_017"
VERSION = "oof_record_error_audit_v1"


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference(path, root):
    stat = path.stat()
    return {"path": str(path.relative_to(root)), "sha256": digest(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    result = hashlib.sha256(json.dumps([array.dtype.str, list(array.shape)]).encode())
    result.update(array.tobytes())
    return result.hexdigest()


def environment():
    import joblib
    import numpy as np
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "joblib": joblib.__version__, "executable": sys.executable}


def freeze(root, output):
    origin = root / ORIGIN
    protocol = read(origin / "protocol.json")
    require(digest(origin / "protocol.json") == read(origin / "protocol_lock.json")["sha256"], "Origin protocol seal differs")
    require(not output.exists(), "Freeze requires a fresh diagnostic directory")
    require(digest(origin / "data.npz") == protocol["cache_sha256"]
        and digest(origin / "dataset.json") == protocol["dataset_sha256"], "Origin dataset/cache link differs")
    manifest = read(origin / "supplemental_manifest.json")
    require(digest(origin / "supplemental_manifest.json") == protocol["supplemental_training"]["manifest_sha256"]
        and digest(origin / "supplemental_training.npz") == protocol["supplemental_training"]["cache_sha256"]
        == manifest["cache_sha256"] and manifest["personal_reference_used"] is False, "Ordinary restoration link differs")
    adopted = read(root / "research/models/checkpoints/research_baseline.json")["current_research_baseline"]
    require(adopted["pipeline"] == protocol["baseline"], "Current adopted research baseline differs from017")
    sources = protocol["source_hashes"]
    for name, expected in sources.items():
        require(digest(origin / "source_snapshot" / name) == expected, "Frozen origin source mismatch: " + name)
    names = [origin / name for name in ("protocol.json", "protocol_lock.json", "data.npz", "dataset.json",
        "supplemental_training.npz", "supplemental_manifest.json", "protected_before.json", "audits/projection_fitted_identity.json")]
    names += [root / "research/models/checkpoints/research_baseline.json", root / "src/validated_checkpoints.json"]
    protected = read(origin / "protected_before.json")
    for name, expected in protected.items():
        current = reference(root / name, root)
        require({key: current[key] for key in ("sha256", "mtime_ns", "size")} == expected, "Protected original changed: " + name)
    config = {"schema_version": 1, "version": VERSION, "frozen_at": now(), "timezone": "Asia/Seoul",
        "origin_run": ORIGIN, "origin_protocol_sha256": digest(origin / "protocol.json"),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_state_present": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "script_sha256": digest(Path(__file__)), "origin_source_hashes": sources,
        "inputs": [reference(path, root) for path in names], "environment": environment(),
        "partitions": protocol["partitions"], "outer_seed": 20260921,
        "inner_split": {"class": "GroupKFold", "n_splits": 4, "shuffle": False,
            "groups": "NFC+strip normalized subject; same posture_core.normalize_text as record_training",
            "fold_index": "zero based 0..3", "seed_formula": "(outer_seed + 1000 + fold_index) % 2**32"},
        "model": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 8, "max_features": .5,
            "class_weight": "balanced", "n_jobs": 1, "bootstrap": False, "criterion": "gini"},
        "preprocessing": "raw canonical596 float32; fresh innerTRAIN-only StandardScaler before each ExtraTrees fit",
        "data_policy": "include_ordinary_reserved_training_records_v1", "fits_expected": 12,
        "weights": {"coefficient": .5, "record_error": "mean(OOF argmax != label) across windows of each normalized record",
            "raw_record_weight": "h_r = 1 + 0.5 * e_r", "sample_weight": "z_i = h_record(i) / mean(h_record(j) for all TRAIN windows j in class y_i)",
            "native_class_weight": "N / (7 * n_class)", "effective_weight": "z_i * native_class_weight[y_i]",
            "record_weight_is_constant_within_record": True, "class_normalizer_is_window_mean_not_record_mean": True},
        "descriptives": {"quantiles": [0, .25, .5, .75, .9, .95, .99, 1], "top_people": 3, "top_records": 10,
            "oof_metrics": ["window_error_rate", "macro_f1", "confusion_matrix", "per_class_window_errors", "record_error_distribution"]},
        "scope": {"outer_validation_or_test_predictions": False, "outer_validation_or_test_feature_labels": False,
            "final_reweighted_model_fit": False, "selection_or_coefficient_tuning": False,
            "oof_predictions_only_for_outer_train": True, "overlapping_outer_train_partitions_are_dependent": True}}
    require(protocol["baseline"]["model_params"] == {key: config["model"][key] for key in
        ("n_estimators", "max_depth", "min_samples_leaf", "max_features")}, "Diagnostic forest differs from adopted baseline")
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot/oof_record_error_audit.py"
    snapshot.parent.mkdir()
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    snapshot.chmod(0o444)
    write(output / "diagnostic_config.json", config)
    write(output / "diagnostic_lock.json", {"config_sha256": digest(output / "diagnostic_config.json"),
        "script_sha256": digest(snapshot), "frozen_at": config["frozen_at"], "fits_before_freeze": 0})
    print(json.dumps({"output": str(output), "config_sha256": digest(output / "diagnostic_config.json"),
        "script_sha256": digest(snapshot), "frozen_at": config["frozen_at"]}))


def quantiles(values, config):
    import numpy as np
    return {str(q): float(value) for q, value in zip(config["descriptives"]["quantiles"],
        np.quantile(values, config["descriptives"]["quantiles"], method="linear"))}


def weight_summary(labels, groups, records, raw, sample, effective, errors, config):
    import numpy as np
    baseline = len(labels) / (7 * np.bincount(labels, minlength=7)[labels])
    class_rows = []
    for class_id in range(7):
        take = labels == class_id
        require(np.allclose(sample[take].sum(), take.sum(), rtol=1e-13, atol=1e-12), "Class-normalized sample mass differs")
        require(np.allclose(effective[take].sum(), len(labels) / 7, rtol=1e-13, atol=1e-12), "Effective class mass differs")
        class_rows.append({"class_id": class_id, "windows": int(take.sum()), "records": len(np.unique(records[take])),
            "oof_errors": int(errors[take].sum()), "oof_window_error_rate": float(errors[take].mean()),
            "raw_weight_window_mean": float(raw[take].mean()), "sample_weight_sum": float(sample[take].sum()),
            "effective_weight_sum": float(effective[take].sum()), "effective_weight_share": float(effective[take].sum() / effective.sum()),
            "sample_weight_quantiles": quantiles(sample[take], config), "effective_weight_quantiles": quantiles(effective[take], config)})
    concentration = {}
    for kind, identities, top in (("people", groups, config["descriptives"]["top_people"]),
            ("records", records, config["descriptives"]["top_records"])):
        rows = []
        for identity in sorted(set(identities.tolist())):
            take = identities == identity
            rows.append({"identity": identity, "windows": int(take.sum()), "window_share": float(take.mean()),
                "baseline_class_balanced_share": float(baseline[take].sum() / baseline.sum()),
                "effective_weight_share": float(effective[take].sum() / effective.sum()),
                "effective_to_baseline_mass_ratio": float(effective[take].sum() / baseline[take].sum()),
                "mean_sample_weight": float(sample[take].mean()), "oof_window_error_rate": float(errors[take].mean())})
        ordered = sorted(rows, key=lambda row: (-row["effective_weight_share"], row["identity"]))
        concentration[kind] = {"count": len(rows), "rows": rows, "top_count": min(top, len(rows)),
            "largest_effective_share": ordered[0]["effective_weight_share"],
            "top_effective_share": sum(row["effective_weight_share"] for row in ordered[:top]),
            "same_top_baseline_share": sum(row["baseline_class_balanced_share"] for row in ordered[:top]),
            "maximum_mass_ratio": max(row["effective_to_baseline_mass_ratio"] for row in rows),
            "minimum_mass_ratio": min(row["effective_to_baseline_mass_ratio"] for row in rows)}
    return {"class_rows": class_rows, "raw_weight_quantiles": quantiles(raw, config),
        "sample_weight_quantiles": quantiles(sample, config), "effective_weight_quantiles": quantiles(effective, config),
        "concentration": concentration, "sample_weight_sum": float(sample.sum()), "effective_weight_sum": float(effective.sum())}


def execute(root, output):
    config, lock = read(output / "diagnostic_config.json"), read(output / "diagnostic_lock.json")
    snapshot = output / "source_snapshot/oof_record_error_audit.py"
    require(digest(output / "diagnostic_config.json") == lock["config_sha256"], "Diagnostic configuration changed")
    require(digest(snapshot) == lock["script_sha256"] == config["script_sha256"], "Frozen script changed")
    if Path(__file__).resolve() != snapshot.resolve():
        import os
        os.execv(sys.executable, [sys.executable, "-B", str(snapshot), "run", "--root", str(root), "--output", str(output)])
    require(environment() == config["environment"], "Numerical environment changed after freezing")
    require(not (output / "execution_start.json").exists(), "Diagnostic is non-resumable; preserve existing run evidence")
    origin = root / config["origin_run"]
    for row in config["inputs"]:
        require(reference(root / row["path"], root) == row, "Frozen input changed: " + row["path"])
    for name, expected in config["origin_source_hashes"].items():
        require(digest(origin / "source_snapshot" / name) == expected, "Origin snapshot changed: " + name)
    protected = read(origin / "protected_before.json")
    def check_protected():
        for name, expected in protected.items():
            row = reference(root / name, root)
            require({key: row[key] for key in ("sha256", "mtime_ns", "size")} == expected, "Protected original changed: " + name)
    check_protected()
    sys.path.insert(0, str(origin / "source_snapshot"))
    import joblib
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from src.posture_core import normalize_text
    from src.learned_projection_pipeline_v1 import _scaler_digest, _forest_digest
    from experiments.iterative_improvement.feature_tail_audit import selected_rows
    from experiments.iterative_improvement.record_training import append_record_training, validate_record_identities
    require(Path(selected_rows.__code__.co_filename).resolve() ==
        (origin / "source_snapshot/experiments/iterative_improvement/feature_tail_audit.py").resolve(), "TRAIN reader is not frozen")
    require(Path(append_record_training.__code__.co_filename).resolve() ==
        (origin / "source_snapshot/experiments/iterative_improvement/record_training.py").resolve(), "TRAIN restoration helper is not frozen")
    fitted = read(origin / "audits/projection_fitted_identity.json")
    require(fitted["passed"] and fitted["protocol_sha256"] == config["origin_protocol_sha256"], "017 fitted-input proof differs")
    expected_inputs = {row["fold"]: row for row in fitted["results"] if row["candidate"] == "PROJ_identity"}
    caches = {}
    for key, filename in (("base", "data.npz"), ("ordinary", "supplemental_training.npz")):
        with np.load(origin / filename, allow_pickle=False) as cache:
            groups, records = cache["groups"], cache["record_ids"]
        caches[key] = {"path": origin / filename, "groups": np.asarray([normalize_text(value) for value in groups]),
            "record_ids": np.asarray([normalize_text(value) for value in records])}
    require(not set(caches["base"]["record_ids"]) & set(caches["ordinary"]["record_ids"]), "Restored recording overlaps base cache")
    write(output / "execution_start.json", {"started_at": now(), "config_sha256": lock["config_sha256"],
        "script_sha256": lock["script_sha256"], "inputs_and_protected_verified_before_fit": True})
    results, total_fits, scoped_windows = [], 0, []
    for partition in config["partitions"]:
        started = now()
        name, people = partition["id"], sorted(normalize_text(value) for value in partition["train"])
        require(not set(people) & {normalize_text(value) for key in ("validation", "test") for value in partition[key]}, "Outer person overlap")
        selected = {}
        for key, cache in caches.items():
            take = np.flatnonzero(np.isin(cache["groups"], people))
            selected[key] = {"X": selected_rows(cache["path"], "X", take, len(cache["groups"])),
                "y": selected_rows(cache["path"], "y", take, len(cache["groups"])),
                "groups": cache["groups"][take], "record_ids": cache["record_ids"][take], "indices": take}
        base, ordinary = selected["base"], selected["ordinary"]
        X, y, groups, records, restoration = append_record_training(base, ordinary, np.arange(len(base["X"])))
        groups = np.asarray([normalize_text(value) for value in groups])
        records = np.asarray([normalize_text(value) for value in records])
        validate_record_identities(y, groups, records)
        require(X.dtype == np.dtype(np.float32) and X.shape == (len(y), 596), "Canonical raw596float32 contract differs")
        require(np.array_equal(np.unique(y), np.arange(7)) and set(groups) == set(people), "Outer TRAIN scope incomplete")
        scope = {"training_input_sha256": array_hash(X), "training_labels_sha256": array_hash(y),
            "training_groups_sha256": json_hash(groups.tolist())}
        require(all(value == expected_inputs[name][key] for key, value in scope.items()), "Actual TRAIN differs from017 proof")
        require(restoration == fitted["train_reconstruction"][[r["fold"] for r in fitted["train_reconstruction"]].index(name)]["restoration"], "Restoration order differs")
        scoped_windows.append({("base", int(i)) for i in base["indices"]} | {("ordinary", int(i)) for i in ordinary["indices"]})
        probability, assignment = np.full((len(y), 7), np.nan), np.full(len(y), -1, dtype=np.int64)
        folds = []
        for fold, (train, validation) in enumerate(GroupKFold(n_splits=4, shuffle=False).split(X, y, groups)):
            require(not set(groups[train]) & set(groups[validation]), "Inner subject overlap")
            require(not set(records[train]) & set(records[validation]), "Inner recording overlap")
            require(np.all(assignment[validation] == -1), "OOF row assigned twice")
            require(np.array_equal(np.unique(y[train]), np.arange(7)), "Inner TRAIN missing a class")
            require(len(train) + len(validation) == len(y) and not set(train) & set(validation), "Inner row overlap/coverage differs")
            seed = (config["outer_seed"] + 1000 + fold) % 2**32
            fold_start = now()
            with threadpool_limits(limits=1):
                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(X[train])
                validation_scaled = scaler.transform(X[validation])
                estimator = ExtraTreesClassifier(**config["model"], random_state=seed).fit(train_scaled, y[train])
                require(np.array_equal(estimator.classes_, np.arange(7)), "OOF forest class alignment differs")
                fold_probability = estimator.predict_proba(validation_scaled)
            require(fold_probability.shape == (len(validation), 7) and np.isfinite(fold_probability).all()
                and np.all(fold_probability >= 0) and np.allclose(fold_probability.sum(1), 1, rtol=0, atol=1e-12), "Invalid OOF probabilities")
            probability[validation], assignment[validation] = fold_probability, fold
            folder = output / "inner_fits" / name / f"fold_{fold}"
            folder.mkdir(parents=True)
            model_path, prediction_path = folder / "model.joblib", folder / "oof_predictions.npz"
            with model_path.open("xb") as stream:
                joblib.dump({"scaler": scaler, "estimator": estimator}, stream)
            model_path.chmod(0o444)
            with prediction_path.open("xb") as stream:
                np.savez_compressed(stream, train_indices=train, oof_indices=validation, truth=y[validation],
                    groups=groups[validation], record_ids=records[validation], probability=fold_probability,
                    prediction=fold_probability.argmax(1))
            prediction_path.chmod(0o444)
            row = {"fold_index": fold, "seed": seed, "started_at": fold_start, "finished_at": now(),
                "train_indices": train.tolist(), "oof_indices": validation.tolist(),
                "train_subjects": sorted(set(groups[train])), "oof_subjects": sorted(set(groups[validation])),
                "train_windows": len(train), "oof_windows": len(validation),
                "train_records": len(np.unique(records[train])), "oof_records": len(np.unique(records[validation])),
                "train_class_counts": np.bincount(y[train], minlength=7).tolist(),
                "oof_class_counts": np.bincount(y[validation], minlength=7).tolist(),
                "person_record_row_disjoint": True, "scaler_samples_seen": int(scaler.n_samples_seen_),
                "raw_train_sha256": array_hash(X[train]), "train_labels_sha256": array_hash(y[train]),
                "scaled_train_sha256": array_hash(train_scaled), "scaled_oof_sha256": array_hash(validation_scaled),
                "scaler_state_sha256": _scaler_digest(scaler), "forest_state_sha256": _forest_digest(estimator, 596),
                "oof_probability_sha256": array_hash(fold_probability), "full_estimator_params": estimator.get_params(deep=False),
                "model": reference(model_path, root), "oof_predictions": reference(prediction_path, root)}
            require(row["scaler_samples_seen"] == len(train), "Scaler consumed non-innerTRAIN rows")
            write(folder / "fit_audit.json", row)
            write(folder / "fit_seal.json", {"fit_audit_sha256": digest(folder / "fit_audit.json"),
                "config_sha256": lock["config_sha256"], "script_sha256": lock["script_sha256"]})
            folds.append(row)
            total_fits += 1
        require(np.isfinite(probability).all() and np.all(assignment >= 0), "Incomplete OOF coverage")
        prediction, errors = probability.argmax(1), probability.argmax(1) != y
        record_error, raw = np.empty(len(y), dtype=np.float64), np.empty(len(y), dtype=np.float64)
        record_rows = []
        for record in sorted(set(records.tolist())):
            take = records == record
            require(len(np.unique(assignment[take])) == 1, "One recording spans OOF folds")
            error = float(errors[take].mean())
            record_error[take], raw[take] = error, 1 + config["weights"]["coefficient"] * error
            record_rows.append({"record_id": record, "subject": groups[np.flatnonzero(take)[0]], "class_id": int(y[take][0]),
                "windows": int(take.sum()), "oof_errors": int(errors[take].sum()), "error_rate": error,
                "raw_weight": float(raw[take][0]), "oof_fold_index": int(assignment[take][0])})
        sample = np.empty(len(y), dtype=np.float64)
        for class_id in range(7):
            take = y == class_id
            sample[take] = raw[take] / raw[take].mean()
        require(all(np.unique(sample[records == record]).size == 1 for record in set(records.tolist())),
            "Normalized sample weight differs within a pure recording")
        effective = sample * (len(y) / (7 * np.bincount(y, minlength=7)[y]))
        require(np.isfinite(effective).all() and np.all(effective > 0), "Invalid fixed-coefficient weights")
        folder = output / "outer_train" / name
        folder.mkdir(parents=True)
        arrays = {"truth": y, "groups": groups, "record_ids": records, "oof_fold_index": assignment,
            "oof_probability": probability, "oof_prediction": prediction, "record_error": record_error,
            "raw_record_weight": raw, "sample_weight": sample, "effective_weight": effective}
        array_path = folder / "oof_record_weights.npz"
        with array_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        array_path.chmod(0o444)
        row = {"outer_partition": name, "started_at": started, "finished_at": now(),
            "windows": len(y), "records": len(record_rows), "subjects": len(people), **scope,
            "record_ids_sha256": array_hash(records), "base_train_indices": base["indices"].tolist(),
            "ordinary_train_indices": ordinary["indices"].tolist(), "restoration": restoration,
            "oof_coverage_exactly_once": True, "inner_fits": folds, "arrays": reference(array_path, root),
            "array_hashes": {key: array_hash(value) for key, value in arrays.items()},
            "oof_diagnostic": {"window_error_rate": float(errors.mean()), "macro_f1": float(f1_score(y, prediction,
                labels=np.arange(7), average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(y, prediction, labels=np.arange(7)).tolist(),
                "all_correct_records": sum(r["error_rate"] == 0 for r in record_rows),
                "all_wrong_records": sum(r["error_rate"] == 1 for r in record_rows),
                "record_error_quantiles": quantiles([r["error_rate"] for r in record_rows], config)},
            "record_errors": record_rows, "weights": weight_summary(y, groups, records, raw, sample, effective, errors, config)}
        write(folder / "diagnostic.json", row)
        write(folder / "diagnostic_seal.json", {"sha256": digest(folder / "diagnostic.json"), "arrays_sha256": digest(array_path),
            "config_sha256": lock["config_sha256"]})
        results.append(row)
        print(json.dumps({"completed_outer_train": name, "fits": total_fits, "windows": len(y)}), flush=True)
    require(total_fits == config["fits_expected"], "Unexpected fit count")
    check_protected()
    for row in config["inputs"]:
        require(reference(root / row["path"], root) == row, "Input changed during diagnostic: " + row["path"])
    for name, expected in config["origin_source_hashes"].items():
        require(digest(origin / "source_snapshot" / name) == expected, "Frozen source changed during diagnostic")
    summary = {"schema_version": 1, "version": VERSION, "passed": True,
        "started_at": read(output / "execution_start.json")["started_at"], "finished_at": now(),
        "timezone": "Asia/Seoul", "config_sha256": lock["config_sha256"], "script_sha256": lock["script_sha256"],
        "origin_protocol_sha256": config["origin_protocol_sha256"], "fits": total_fits, "final_reweighted_fits": 0,
        "protected_originals_count": len(protected), "protected_originals_unchanged": True, "inputs_unchanged": True,
        "split_times_window_observations": sum(len(rows) for rows in scoped_windows),
        "unique_source_windows": len(set.union(*scoped_windows)),
        "outer_train_windows_are_dependent": True, "outer_validation_test_feature_labels_decoded": False,
        "outer_validation_test_predictions_or_metrics": False, "selection_or_tuning": False,
        "environment": environment(), "partitions": results,
        "reproduction": "Freeze this exact script against unchanged017inputs, then run the frozen copy in a fresh output directory; no existing run is resumed/overwritten.",
        "limitations": ["OOF errors estimate transfer across the20 outerTRAIN people; they are not outerDEV performance.",
            "Repeated windows and overlapping outerTRAIN partitions are dependent observations.",
            "Large weights can emphasize ambiguous posture or sensor/person differences; diagnostic cannot establish causes or gains.",
            "Native class balancing multiplied by supplied weights preserves each class total mass by construction.",
            "Only TRAIN feature/label rows are decoded; all ID metadata and full file hashes are read. ZIP seeks may decompress skipped bytes."]}
    write(output / "oof_record_error_audit.json", summary)
    write(output / "audit_seal.json", {"sha256": digest(output / "oof_record_error_audit.json"),
        "config_sha256": lock["config_sha256"], "script_sha256": lock["script_sha256"], "sealed_at": now()})
    print(json.dumps({"output": str(output), "sha256": digest(output / "oof_record_error_audit.json"), "fits": total_fits}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output is not None else Path("research/results") / (
        datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S") + "_oof_record_error_audit")
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    require(output.is_relative_to(root / "research/results") and not output.is_relative_to(root / ORIGIN),
        "Use a new independent research/results output outside the origin run")
    if args.action == "freeze":
        freeze(root, output)
    else:
        require(args.output is not None, "Run requires an explicit frozen output path")
        execute(root, output)


if __name__ == "__main__":
    main()
