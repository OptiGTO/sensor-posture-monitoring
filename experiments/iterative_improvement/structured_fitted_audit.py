"""Audit iteration019 saved fitted state against exact TRAIN rows, without refits.

Numerical reconstruction uses only outer TRAIN plus eligible ordinary records.
Saved development predictions are compared only for baseline/control byte
reproduction; this script makes no predictions and computes no performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler


SCRIPT = "experiments/iterative_improvement/structured_fitted_audit.py"
ADOPTION = "research/results/2026-09-22_013650_adopt_et_ordinary"
CANDIDATES = {"STRUCT_identity", "OVO_pair32", "ET_subject_residual3"}
DEPENDENCIES = (SCRIPT, "experiments/iterative_improvement/feature_tail_audit.py",
    "experiments/iterative_improvement/record_training.py", "experiments/iterative_improvement/supplemental.py",
    "src/structured_forest_pipeline_v1.py", "src/subject_residual_v1.py", "src/pairwise_coupling_v1.py",
    "src/posture_core.py", "src/window_pipeline.py", "src/__init__.py")


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evidence(path):
    path = Path(path)
    state = path.stat()
    return {"sha256": digest(path), "size": state.st_size, "mtime_ns": state.st_mtime_ns}


def array_digest(value):
    array = np.ascontiguousarray(value)
    result = hashlib.sha256(json.dumps([array.dtype.str, list(array.shape)]).encode())
    result.update(array.tobytes())
    return result.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def exact(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")


def scaler_statistics(values):
    """Independent finite/dense unweighted first-batch corrected two-pass rule.

    No StandardScaler fitting or private statistics helper is called. The
    reduction order matches installed sklearn's dense first-batch semantics.
    """
    require(values.ndim == 2 and len(values) and values.dtype in (np.dtype("float32"), np.dtype("float64"))
        and np.isfinite(values).all(), "Invalid TRAIN scaler input")
    count = len(values)
    mean = np.sum(values, axis=0, dtype=np.float64) / count
    residual = values - mean
    correction = np.sum(residual, axis=0, dtype=np.float64)
    residual **= 2
    variance = (np.sum(residual, axis=0, dtype=np.float64) - correction ** 2 / count) / count
    epsilon = np.finfo(np.float64).eps
    constant = variance <= count * epsilon * variance + (count * mean * epsilon) ** 2
    scale = np.sqrt(variance)
    scale[constant] = 1.
    return {"mean_": mean, "var_": variance, "scale_": scale}


def standardized(values, statistics):
    result = np.array(values, order="C", copy=True)
    result -= statistics["mean_"].astype(values.dtype)
    result /= statistics["scale_"].astype(values.dtype)
    require(np.isfinite(result).all(), "Nonfinite independent TRAIN standardization")
    return result


def record_means(values, records, metadata):
    expected = []
    for row in metadata:
        indices = np.flatnonzero(records == row["record_id"])
        require(indices.tolist() == row["row_indices"], "Stored record rows differ from actual TRAIN")
        expected.append(np.sort(np.asarray(values[indices], dtype=np.float64), axis=0).mean(axis=0, dtype=np.float64))
    return np.ascontiguousarray(expected)


def residual_transform(values, components):
    numeric = np.array(values, dtype=np.float64, order="C", copy=True)
    coordinates = np.einsum("ni,ji->nj", numeric, components, optimize=False)
    return numeric - np.einsum("nj,ji->ni", coordinates, components, optimize=False)


def verify_arrays(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        require(set(a.files) == set(b.files) and all(exact(a[key], b[key]) for key in a.files),
            "Saved prediction array dtype/shape/bytes differ")
        return {key: array_digest(a[key]) for key in sorted(a.files)}


def audit(args):
    root, out = args.root.resolve(), args.output.resolve()
    require(out.is_relative_to(root / "research/results"), "Audit must target a project research run")
    destination = out / "audits/structured_fitted_identity.json"
    require(not destination.exists(), "Refusing to replace an existing fitted audit")
    protocol = read(out / "protocol.json")
    protocol_sha = digest(out / "protocol.json")
    require(protocol_sha == read(out / "protocol_lock.json")["sha256"], "Protocol seal differs")
    snapshot = out / "source_snapshot"
    for relative, expected in protocol["source_hashes"].items():
        require(digest(snapshot / relative) == expected, "Frozen source differs: " + relative)
    for relative in DEPENDENCIES:
        require(digest(root / relative) == protocol["source_hashes"][relative], "Current immutable audit dependency differs: " + relative)
    # Reset import state before loading joblib objects or a verifier. The executed
    # source and every imported project helper must be the protocol snapshot.
    if Path(__file__).resolve() != (snapshot / SCRIPT).resolve():
        env = os.environ.copy()
        env["PYTHONPATH"] = str(snapshot)
        os.execve(sys.executable, [sys.executable, str(snapshot / SCRIPT), "--root", str(root), "--output", str(out)], env)
    sys.path.insert(0, str(snapshot))
    from experiments.iterative_improvement.feature_tail_audit import selected_rows
    from experiments.iterative_improvement.record_training import append_record_training
    from src.posture_core import normalize_text
    from src import structured_forest_pipeline_v1 as structured
    for module_name in ("experiments.iterative_improvement.feature_tail_audit",
            "experiments.iterative_improvement.record_training", "experiments.iterative_improvement.supplemental",
            "src.structured_forest_pipeline_v1", "src.subject_residual_v1", "src.pairwise_coupling_v1", "src.posture_core"):
        module = sys.modules[module_name]
        relative = module_name.replace(".", "/") + ".py"
        require(Path(module.__file__).resolve() == (snapshot / relative).resolve(), "Audit imported a nonfrozen module: " + module_name)
    started = now()
    inputs = {}
    def bind(path):
        path = Path(path)
        inputs[str(path.relative_to(root))] = evidence(path)
        return inputs[str(path.relative_to(root))]
    for name in ("protocol.json", "protocol_lock.json", "dataset.json", "data.npz",
            "supplemental_manifest.json", "supplemental_training.npz", "protected_before.json"):
        bind(out / name)
    require(digest(out / "dataset.json") == protocol["dataset_sha256"] and digest(out / "data.npz") == protocol["cache_sha256"], "Base cache/manifest differs")
    supplemental = read(out / "supplemental_manifest.json")
    require(digest(out / "supplemental_manifest.json") == protocol["supplemental_training"]["manifest_sha256"]
        and digest(out / "supplemental_training.npz") == protocol["supplemental_training"]["cache_sha256"] == supplemental["cache_sha256"], "Ordinary restoration evidence differs")
    require(supplemental["personal_reference_used"] is False, "Personal reference was used")
    protected = read(out / "protected_before.json")
    def protect():
        require(all(evidence(root / relative) == expected for relative, expected in protected.items()), "Protected originals changed")
    protect()
    existing_audits = {str(path.relative_to(root)): evidence(path) for path in destination.parent.glob("*") if path.is_file()}
    recipes = {name: recipe for branch in protocol["branches"].values() for name, recipe in branch.items()}
    require(set(recipes) == CANDIDATES, "Unexpected iteration019 candidate set")
    partitions = {part["id"]: part for part in protocol["partitions"]}
    require(set(partitions) == {"development_1", "development_2", "development_3"}, "Unexpected partitions")
    expected_models = {out / "development" / name / fold / "model.joblib" for name in recipes for fold in partitions}
    actual_models = {path for path in (out / "development").glob("*/*/model.joblib") if path.parent.parent.name != "baseline"}
    require(actual_models == expected_models and len(actual_models) == 9, "Expected nine completed candidate models")
    seed = protocol["seeds"][0]
    caches = {}
    for key, filename in (("base", "data.npz"), ("ordinary", "supplemental_training.npz")):
        with np.load(out / filename, allow_pickle=False) as archive:
            caches[key] = {"path": out / filename, "groups": np.asarray([normalize_text(str(v)) for v in archive["groups"]]),
                "record_ids": np.asarray([normalize_text(str(v)) for v in archive["record_ids"]])}
    require(not set(caches["base"]["record_ids"]) & set(caches["ordinary"]["record_ids"]), "Base and ordinary record overlap")
    manifest_records = {normalize_text(row["record_id"]): row for row in supplemental["records"]}
    adoption = root / ADOPTION
    old_protocol = read(adoption / "protocol.json")
    require(bind(adoption / "protocol.json")["sha256"] == read(adoption / "protocol_lock.json")["sha256"], "Adoption protocol seal differs")
    bind(adoption / "protocol_lock.json")
    require(old_protocol["partitions"] == protocol["partitions"] and old_protocol["cache_sha256"] == protocol["cache_sha256"], "Adopted baseline evaluation scope differs")
    def fitted(folder, recipe, expected_protocol):
        metric_path, seal_path = folder / "metrics.json", folder / "metrics_seal.json"
        row = read(metric_path)
        require(bind(metric_path)["sha256"] == read(seal_path)["sha256"], "Fit metric seal differs")
        bind(seal_path)
        require(row["protocol_sha256"] == expected_protocol and row["pipeline"] == recipe and row["seed"] == seed, "Fit recipe/seed/protocol differs")
        require(bind(folder / "model.joblib")["sha256"] == row["model_sha256"] and
            bind(folder / "predictions.npz")["sha256"] == row["predictions_sha256"], "Fit artifact hash differs")
        return row
    models, train_scopes, baseline_reproduction = [], [], []
    for fold, part in sorted(partitions.items()):
        people = sorted(normalize_text(v) for v in part["train"])
        valid_people = sorted(normalize_text(v) for v in part["validation"])
        test_people = sorted(normalize_text(v) for v in part["test"])
        require(not (set(people) & set(valid_people) or set(people) & set(test_people) or set(valid_people) & set(test_people)), "Subject split overlap")
        selected = {}
        for key, cache in caches.items():
            indices = np.flatnonzero(np.isin(cache["groups"], people))
            selected[key] = {"X": selected_rows(cache["path"], "X", indices, len(cache["groups"])),
                "y": selected_rows(cache["path"], "y", indices, len(cache["groups"])),
                "groups": cache["groups"][indices], "record_ids": cache["record_ids"][indices], "indices": indices}
            require(set(selected[key]["groups"]) == set(people), "Incomplete TRAIN subject selection")
        base, extra = selected["base"], selected["ordinary"]
        expected = part["counts"]["train"]
        require(len(base["X"]) == expected["windows"] and len(np.unique(base["record_ids"])) == expected["records"]
            and np.bincount(base["y"], minlength=7).tolist() == expected["class_counts"], "Base TRAIN counts differ")
        for record in set(extra["record_ids"]):
            row = manifest_records[record]
            require(row["label"] == "정자세" and row["eligibility"]["explicit_registration"] is False
                and row["eligibility"]["globally_excluded"] is False and row["eligibility"]["training_role"] == "include"
                and row["windows"] == int(np.count_nonzero(extra["record_ids"] == record)), "Ineligible/incomplete ordinary TRAIN recording")
        X, y, groups, records, restoration = append_record_training(base, extra, np.arange(len(base["X"])))
        require(X.dtype == np.float32 and X.shape == (len(y), 596), "Frozen TRAIN precision/schema differs")
        require(set(groups) == set(people) and not set(records) & set(caches["base"]["record_ids"][~np.isin(caches["base"]["groups"], people)]), "Evaluation identity entered TRAIN")
        hashes = {"training_input_sha256": array_digest(X), "training_labels_sha256": array_digest(y.astype(np.int64)),
            "training_groups_sha256": json_digest(groups.tolist()), "training_records_sha256": json_digest(records.tolist())}
        stats = scaler_statistics(X)
        transformed = standardized(X, stats)
        scope = {"partition": fold, "base_train_indices": base["indices"].tolist(), "ordinary_train_indices": extra["indices"].tolist(),
            "windows": len(y), "subjects": len(people), "records": len(np.unique(records)), "class_counts": np.bincount(y, minlength=7).tolist(),
            **hashes, "standardized_input_sha256": array_digest(transformed),
            "independent_scaler_hashes": {key: array_digest(value) for key, value in stats.items()}, "restoration": restoration}
        train_scopes.append(scope)
        old_folder = adoption / "development/ET_ordinary" / fold
        new_folder = out / "development/baseline" / fold
        old = fitted(old_folder, protocol["baseline"], digest(adoption / "protocol.json"))
        new = fitted(new_folder, protocol["baseline"], protocol_sha)
        require(old["metrics"] == new["metrics"] and old["per_subject"] == new["per_subject"], "Stored adopted baseline metrics differ")
        require(new["training_subjects"] == people and new["evaluation_subjects"] == valid_people
            and new["supplemental_training_audit"] == restoration, "Baseline TRAIN/DEV scope differs")
        baseline_reproduction.append({"partition": fold, "reference": str(old_folder.relative_to(root)), "current": str(new_folder.relative_to(root)),
            "reference_predictions_sha256": old["predictions_sha256"], "current_predictions_sha256": new["predictions_sha256"],
            "prediction_array_hashes": verify_arrays(old_folder / "predictions.npz", new_folder / "predictions.npz"),
            "dtype_shape_bytes_equal": True, "stored_metrics_equal": True, "new_predictions": 0})
        for name, recipe in sorted(recipes.items()):
            require(recipe["training"]["seed"] == seed and recipe["training"]["data_policy"] == supplemental["policy"], "Candidate TRAIN policy differs")
            folder = out / "development" / name / fold
            metrics = fitted(folder, recipe, protocol_sha)
            require(metrics["training_subjects"] == people and metrics["evaluation_subjects"] == valid_people
                and metrics["supplemental_training_audit"] == restoration, "Candidate TRAIN/DEV scope differs")
            model = joblib.load(folder / "model.joblib")
            require(structured.verify_fitted_structured_identity(model, recipe, seed), "Frozen model verifier failed")
            require(metrics["training_details"] == model.training_details_, "Stored model/metric TRAIN audit differs")
            for key, value in hashes.items():
                require(getattr(model, key + "_") == value, "Actual TRAIN digest differs: " + key)
            require(model.training_details_["class_counts"] == scope["class_counts"], "Actual TRAIN class counts differ")
            for key, value in stats.items():
                require(exact(getattr(model.scaler_, key), value), "Independent actual TRAIN scaler bytes differ: " + key)
            require(exact(model.scaler_.transform(X), transformed), "Independent dtype-preserving standardization differs")
            require(model.training_details_["standardized_input_sha256"] == array_digest(transformed), "Standardized TRAIN digest differs")
            result = {"candidate": name, "partition": fold, "model_sha256": metrics["model_sha256"],
                "metrics_sha256": digest(folder / "metrics.json"), "predictions_sha256": metrics["predictions_sha256"],
                "full_frozen_state_verified": True, "actual_train_scope_hashes_equal": True,
                "independent_scaler_statistics_bytes_equal": True, "independent_standardization_bytes_equal": True,
                "fit_state_sha256": model.fit_state_sha256_, "training_windows": len(y), "class_counts": scope["class_counts"]}
            if recipe["family"] == structured.RESIDUAL:
                projector = model.projector_
                if recipe["feature_engineering"]["mode"] == "remove_subject3":
                    independent_means = record_means(transformed, records, projector.record_metadata_)
                    require(exact(independent_means, projector.record_means_), "Actual TRAIN record means differ")
                    processed = residual_transform(transformed, projector.components_)
                    result.update(actual_train_record_means_bytes_equal=True, record_means_sha256=array_digest(independent_means),
                        components_sha256=array_digest(projector.components_), components_c_contiguous=bool(projector.components_.flags.c_contiguous))
                else:
                    processed = transformed
                    require(projector.record_means_ is None and projector.components_ is None, "Identity acquired projection state")
                    result["identity_saved_prediction_bytes_equal"] = bool(verify_arrays(folder / "predictions.npz", new_folder / "predictions.npz"))
                require(exact(projector.transform(transformed), processed), "Independent stored projection transform differs")
                require(model.training_details_["transformed_input_sha256"] == array_digest(processed), "Forest TRAIN input scope differs")
                result.update(independent_projection_bytes_equal=True, transformed_input_sha256=array_digest(processed),
                    forest_native_float32_input_sha256=array_digest(np.asarray(processed, dtype=np.float32, order="C")),
                    projector_state_sha256=projector.state_sha256_, forest_state_sha256=model.training_details_["forest_state_sha256"])
            else:
                require(recipe["family"] == structured.PAIR, "Unexpected fitted candidate family")
                pair_input = np.asarray(transformed, dtype=np.float32, order="C")
                require(model.training_details_["pair_input_sha256"] == array_digest(pair_input), "Pair TRAIN input differs")
                pairs = []
                for index, (i, j) in enumerate(structured.PAIR_INDICES):
                    take = np.flatnonzero((y == i) | (y == j))
                    labels = (y[take] == j).astype(np.int64)
                    row = model.pair_audits_[index]
                    counts = np.bincount(labels, minlength=2)
                    expected = {"pair": [i, j], "pair_index": index, "seed": (seed + index) % 2**32,
                        "rows": len(take), "row_indices": take.tolist(), "binary_counts": counts.tolist(),
                        "subjects": sorted(set(groups[take].tolist())), "records": sorted(set(records[take].tolist())),
                        "binary_labels_sha256": array_digest(labels), "input_sha256": array_digest(pair_input[take]),
                        "binary_class_weights": (len(take) / (2 * counts)).tolist(), "class_weight_mass": [len(take)/2]*2,
                        "binary_label_rule": "original i->0, j->1; coupling uses probability column0 for i"}
                    require(all(row[key] == value for key, value in expected.items()), "Actual pair TRAIN scope/seed/labels differs")
                    pairs.append({**expected, "forest_state_sha256": row["forest_state_sha256"], "frozen_forest_verifier_passed": True})
                result.update(actual_train_pairs=pairs, pair_count=len(pairs), pair_training_row_exposures=sum(row["rows"] for row in pairs),
                    pair_input_sha256=array_digest(pair_input), coupling_rule_sha256=model.coupling_rule_["rule_sha256"])
            models.append(result)
    require(len(models) == 9, "Incomplete fitted-model audit")
    protect()
    for relative, old in {**inputs, **existing_audits}.items():
        require(evidence(root / relative) == old, "Audit input or previous audit changed: " + relative)
    for relative, expected in protocol["source_hashes"].items():
        require(digest(snapshot / relative) == expected, "Frozen source changed during audit")
    for relative in DEPENDENCIES:
        require(digest(root / relative) == protocol["source_hashes"][relative], "Current immutable dependency changed during audit")
    result = {"schema_version": 1, "started_at": started, "finished_at": now(), "protocol_sha256": protocol_sha,
        "script_sha256": digest(__file__), "executed_frozen_script": str(Path(__file__).relative_to(root)),
        "reproduction_command": f".venv/bin/python -m experiments.iterative_improvement.structured_fitted_audit --output {out.relative_to(root)}",
        "procedure_source": Path(__file__).read_text(encoding="utf-8"), "inputs": inputs,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "installed_scaler_transform_source_sha256": hashlib.sha256(inspect.getsource(StandardScaler.transform).encode()).hexdigest(),
        "train_scopes": train_scopes, "models": models, "baseline_reproduction": baseline_reproduction,
        "frozen_sources_verified": len(protocol["source_hashes"]), "immutable_dependency_hashes": {rel: protocol["source_hashes"][rel] for rel in DEPENDENCIES},
        "protected_files_unchanged": len(protected), "previous_audits_unchanged": len(existing_audits), "all_inputs_unchanged": True,
        "new_model_fits": 0, "new_scaler_or_projector_fits": 0, "new_predictions": 0, "new_performance_evaluations": 0,
        "heldout_predictions_loaded": False, "development_predictions_used": "stored adoption/baseline/identity byte comparison only; candidate files otherwise hashed",
        "scope": "Frozen model fitted state and actual TRAIN numerical reconstruction; no refit, new scoring or hardware evidence",
        "limitations": ["Stored residual SVD directions and forests are verified, not independently relearned.",
            "Exact state/input linkage does not establish causal benefit, calibration or new-cohort generalization.",
            "TRAIN partitions overlap; nine verifications are not nine independent cohorts."]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    destination.chmod(0o444)
    print(json.dumps({"audit": str(destination.relative_to(root)), "sha256": digest(destination), "models": len(models)}))


def self_test():
    """Small synthetic arithmetic checks only; never loads research artifacts."""
    rng = np.random.default_rng(19)
    for dtype in (np.float32, np.float64):
        for rows in (1, 91):
            X = rng.normal(size=(rows, 596)).astype(dtype)
            X[:, 0], X[:, 1] = 0., 1e10
            original = X.copy()
            # This synthetic-only comparison is outside the actual audit path.
            reference = StandardScaler().fit(X)
            stats = scaler_statistics(X)
            require(all(exact(getattr(reference, key), value) for key, value in stats.items()), "Synthetic scaler bytes differ")
            require(exact(reference.transform(X), standardized(X, stats)), "Synthetic transform bytes differ")
            require(exact(X, original), "Synthetic input mutated")
    X = rng.normal(size=(6, 596)).astype(np.float32)
    records = np.asarray(["b", "a", "b", "a", "c", "c"])
    metadata = [{"record_id": name, "row_indices": np.flatnonzero(records == name).tolist()} for name in sorted(set(records))]
    require(record_means(X, records, metadata).shape == (3, 596), "Synthetic record alignment differs")
    components = np.eye(596, dtype=np.float64)[:3].copy()
    transformed = residual_transform(X, components)
    require(np.all(transformed[:, :3] == 0) and exact(transformed[:, 3:], X.astype(np.float64)[:, 3:]), "Synthetic residual coordinates differ")
    print(json.dumps({"synthetic_self_test": "passed", "research_artifacts_loaded": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        require(args.output is None, "Self-test must not target a real run")
        self_test()
    else:
        require(args.output is not None, "--output is required for saved-state audit")
        audit(args)


if __name__ == "__main__":
    main()
