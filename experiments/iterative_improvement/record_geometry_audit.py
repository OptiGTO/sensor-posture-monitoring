"""Frozen TRAIN-only recording geometry diagnostic for posture classes0,1,2.

Only three StandardScaler fits are allowed; no classifier fit or prediction.
Within-person and equally weighted other-person comparisons are descriptive.
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

ORIGIN = "research/results/2026-09-22_051421_iterative_018"
VERSION = "record_geometry_audit_v1"


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
    require(not output.exists(), "Use a fresh independent diagnostic directory")
    require(digest(origin / "protocol.json") == read(origin / "protocol_lock.json")["sha256"], "018 protocol seal changed")
    require(digest(origin / "data.npz") == protocol["cache_sha256"] and
        digest(origin / "dataset.json") == protocol["dataset_sha256"], "018 dataset/cache link changed")
    manifest = read(origin / "supplemental_manifest.json")
    require(digest(origin / "supplemental_manifest.json") == protocol["supplemental_training"]["manifest_sha256"]
        and digest(origin / "supplemental_training.npz") == protocol["supplemental_training"]["cache_sha256"]
        == manifest["cache_sha256"] and manifest["personal_reference_used"] is False, "Ordinary restoration link changed")
    require(read(root / "research/models/checkpoints/research_baseline.json")["current_research_baseline"]["pipeline"]
        == protocol["baseline"], "Current research baseline differs from018")
    for name, expected in protocol["source_hashes"].items():
        require(digest(origin / "source_snapshot" / name) == expected, "018 source changed: " + name)
    protected = read(origin / "protected_before.json")
    for name, expected in protected.items():
        current = reference(root / name, root)
        require({key: current[key] for key in ("sha256", "size", "mtime_ns")} == expected, "Protected original changed: " + name)
    fitted = read(origin / "audits/hardness_fitted_identity.json")
    require(fitted["protocol_sha256"] == digest(origin / "protocol.json"), "Stored TRAIN proof protocol differs")
    names = [origin / name for name in ("protocol.json", "protocol_lock.json", "dataset.json", "data.npz",
        "supplemental_manifest.json", "supplemental_training.npz", "protected_before.json", "audits/hardness_fitted_identity.json")]
    names += [origin / "development/HARD_identity" / part["id"] / "model.joblib" for part in protocol["partitions"]]
    names += [root / "research/models/checkpoints/research_baseline.json", root / "src/validated_checkpoints.json"]
    for row in fitted["models"]:
        if row["candidate"] == "HARD_identity":
            require(digest(origin / "development/HARD_identity" / row["partition"] / "model.joblib") == row["model_sha256"],
                "Stored TRAIN identity model changed")
    config = {"schema_version": 1, "version": VERSION, "frozen_at": now(), "timezone": "Asia/Seoul",
        "origin_run": ORIGIN, "origin_protocol_sha256": digest(origin / "protocol.json"),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_state_present": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "script_sha256": digest(Path(__file__)), "origin_source_hashes": protocol["source_hashes"],
        "inputs": [reference(path, root) for path in names], "environment": environment(),
        "partitions": protocol["partitions"], "classes": [0, 1, 2],
        "class_names": ["정자세", "앞으로 숙이기", "뒤로 기대기"],
        "directed_contrasts": [[a, b] for a in range(3) for b in range(3) if a != b],
        "data_policy": "include_ordinary_reserved_training_records_v1",
        "preprocessing": "One unweighted StandardScaler per outerTRAIN; raw596float32 all7classes; same baseline numerical path",
        "prototype": "Float64 mean of standardized windows for each normalized single-person/single-class recording; only triplet records analyzed",
        "distance": "scipy.spatial.distance.cdist, float64 Euclidean over596 coordinates; no further metric learning or rescaling",
        "cell_sets": {"same_class": "Target person t, anchor class a, excluding anchor recording r",
            "other_class": "Target person t, comparison class b"},
        "distance_quantiles": [.1, .5, .9], "quantile_method": "linear",
        "raw_margin": "min(other_class distances) - min(same_class distances)",
        "normalized_margin": "raw_margin / (min(other_class distances) + min(same_class distances) + epsilon)",
        "epsilon": 1e-12, "epsilon_units": "fixed Euclidean-distance units; affects near-zero scale behavior",
        "missing": "Null margin when either set empty; no imputation; both-zero distances yield zero normalized margin",
        "other_people": "Compute per-target-person cells/margins first, then equal mean over available other people; never pool their nearest record",
        "primary_summary": "Paired available anchor records only; mean records within anchor-person and a->b; equal mean anchor people; equal mean six directed contrasts",
        "scope": {"classifier_fits": 0, "classifier_predictions": 0, "scaler_fits": 3,
            "outer_validation_test_feature_labels": False, "outer_validation_test_scores": False,
            "session_independence_known": False, "accuracy_or_model_selection": False}}
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot/record_geometry_audit.py"
    snapshot.parent.mkdir()
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    snapshot.chmod(0o444)
    write(output / "diagnostic_config.json", config)
    write(output / "diagnostic_lock.json", {"config_sha256": digest(output / "diagnostic_config.json"),
        "script_sha256": digest(snapshot), "frozen_at": config["frozen_at"], "statistics_before_freeze": False})
    print(json.dumps({"output": str(output), "config_sha256": digest(output / "diagnostic_config.json"),
        "script_sha256": digest(snapshot), "frozen_at": config["frozen_at"]}))


def cell(distances):
    import numpy as np
    count = len(distances)
    if not count:
        return {"count": 0, "minimum": None, "q10": None, "median": None, "q90": None}
    q = np.quantile(distances, [.1, .5, .9], method="linear")
    return {"count": count, "minimum": float(np.min(distances)), "q10": float(q[0]),
        "median": float(q[1]), "q90": float(q[2])}


def compare_cells(same, other, epsilon):
    if not same["count"] or not other["count"]:
        return {"available": False, "raw_margin": None, "normalized_margin": None}
    raw = other["minimum"] - same["minimum"]
    return {"available": True, "raw_margin": raw,
        "normalized_margin": raw / (other["minimum"] + same["minimum"] + epsilon)}


SUMMARY_KEYS = ("within_same_minimum", "within_other_minimum", "between_person_mean_same_minimum",
    "between_person_mean_other_minimum", "within_raw_margin", "between_person_mean_raw_margin",
    "within_normalized_margin", "between_person_mean_normalized_margin", "within_minus_between_normalized_margin")


def summarize_anchors(anchors, config):
    import numpy as np
    pairs = []
    for a, b in config["directed_contrasts"]:
        selected = [row for row in anchors if row["anchor_class"] == a and row["other_class"] == b]
        people = []
        for person in sorted({row["anchor_subject"] for row in selected}):
            raw = [row for row in selected if row["anchor_subject"] == person]
            valid = [row for row in raw if row["paired_available"]]
            means = None if not valid else {key: float(np.mean([row["paired"][key] for row in valid])) for key in SUMMARY_KEYS}
            people.append({"subject": person, "anchor_records": len(raw), "paired_records": len(valid),
                "missing_within": sum(not row["within"]["available"] for row in raw),
                "missing_between": sum(not row["between"]["available"] for row in raw), "paired_record_means": means})
        valid_people = [row for row in people if row["paired_record_means"] is not None]
        means = None if not valid_people else {key: float(np.mean([row["paired_record_means"][key] for row in valid_people])) for key in SUMMARY_KEYS}
        pairs.append({"anchor_class": a, "other_class": b, "anchor_records": len(selected),
            "paired_records": sum(row["paired_available"] for row in selected), "anchor_people": len(people),
            "paired_people": len(valid_people), "missing_within_records": sum(not row["within"]["available"] for row in selected),
            "missing_between_records": sum(not row["between"]["available"] for row in selected),
            "person_rows": people, "equal_person_means": means})
    available = [row for row in pairs if row["equal_person_means"] is not None]
    overall = None if not available else {key: float(np.mean([row["equal_person_means"][key] for row in available])) for key in SUMMARY_KEYS}
    return {"contrasts": pairs, "available_contrasts": len(available), "equal_contrast_equal_person_means": overall,
        "paired_anchor_contrast_observations": sum(row["paired_records"] for row in pairs),
        "total_anchor_contrast_observations": len(anchors)}


def execute(root, output):
    config, lock = read(output / "diagnostic_config.json"), read(output / "diagnostic_lock.json")
    snapshot = output / "source_snapshot/record_geometry_audit.py"
    require(digest(output / "diagnostic_config.json") == lock["config_sha256"], "Diagnostic configuration changed")
    require(digest(snapshot) == lock["script_sha256"] == config["script_sha256"], "Frozen script changed")
    if Path(__file__).resolve() != snapshot.resolve():
        import os
        os.execv(sys.executable, [sys.executable, "-B", str(snapshot), "run", "--root", str(root), "--output", str(output)])
    require(environment() == config["environment"], "Numerical environment changed")
    require(not (output / "execution_start.json").exists(), "Preserve existing execution; use a fresh diagnostic run")
    origin = root / config["origin_run"]
    protected = read(origin / "protected_before.json")
    def verify_inputs():
        for row in config["inputs"]:
            require(reference(root / row["path"], root) == row, "Frozen input changed: " + row["path"])
        for name, expected in config["origin_source_hashes"].items():
            require(digest(origin / "source_snapshot" / name) == expected, "Origin source changed: " + name)
        for name, expected in protected.items():
            row = reference(root / name, root)
            require({key: row[key] for key in ("sha256", "size", "mtime_ns")} == expected, "Protected original changed: " + name)
    verify_inputs()
    sys.path.insert(0, str(origin / "source_snapshot"))
    import joblib
    import numpy as np
    from scipy.spatial.distance import cdist
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from src.posture_core import CLASS_NAMES, normalize_text
    from experiments.iterative_improvement.feature_tail_audit import selected_rows
    from experiments.iterative_improvement.record_training import append_record_training, validate_record_identities
    require([CLASS_NAMES[index] for index in config["classes"]] == config["class_names"], "Class ID/name mismatch")
    require(Path(append_record_training.__code__.co_filename).resolve() ==
        (origin / "source_snapshot/experiments/iterative_improvement/record_training.py").resolve(), "Restoration helper is not frozen")
    caches = {}
    for key, filename in (("base", "data.npz"), ("ordinary", "supplemental_training.npz")):
        with np.load(origin / filename, allow_pickle=False) as archive:
            groups, records = archive["groups"], archive["record_ids"]
        caches[key] = {"path": origin / filename, "groups": np.asarray([normalize_text(value) for value in groups]),
            "record_ids": np.asarray([normalize_text(value) for value in records])}
    require(not set(caches["base"]["record_ids"]) & set(caches["ordinary"]["record_ids"]), "Base/restored record overlap")
    write(output / "execution_start.json", {"started_at": now(), "config_sha256": lock["config_sha256"],
        "script_sha256": lock["script_sha256"], "inputs_verified": True})
    results, scoped_windows = [], []
    for part in config["partitions"]:
        started = now()
        people = sorted(normalize_text(value) for value in part["train"])
        require(not set(people) & {normalize_text(value) for key in ("validation", "test") for value in part[key]}, "Outer subject overlap")
        selected = {}
        for key, cache in caches.items():
            take = np.flatnonzero(np.isin(cache["groups"], people))
            selected[key] = {"X": selected_rows(cache["path"], "X", take, len(cache["groups"])),
                "y": selected_rows(cache["path"], "y", take, len(cache["groups"])),
                "groups": cache["groups"][take], "record_ids": cache["record_ids"][take], "indices": take}
        base, ordinary = selected["base"], selected["ordinary"]
        X, y, groups, records, restoration = append_record_training(base, ordinary, np.arange(len(base["X"])))
        validate_record_identities(y, groups, records)
        require(X.dtype == np.dtype(np.float32) and X.shape == (len(y), 596), "Raw TRAIN shape/dtype differs")
        scoped_windows.append({("base", int(i)) for i in base["indices"]} | {("ordinary", int(i)) for i in ordinary["indices"]})
        identity = joblib.load(origin / "development/HARD_identity" / part["id"] / "model.joblib")
        scope = {"training_input_sha256": array_hash(X), "training_labels_sha256": array_hash(y),
            "training_groups_sha256": json_hash(groups.tolist()), "training_records_sha256": json_hash(records.tolist())}
        require(all(getattr(identity, key + "_") == value for key, value in scope.items()), "Actual TRAIN differs from018 stored input")
        with threadpool_limits(limits=1):
            scaler = StandardScaler()
            standardized = scaler.fit_transform(X)
        for key in ("mean_", "scale_", "var_"):
            require(getattr(scaler, key).tobytes() == getattr(identity.scaler_, key).tobytes(), "Baseline TRAIN scaler statistics differ")
        del identity
        metadata, prototypes = [], []
        for record in sorted(set(records.tolist())):
            take = np.flatnonzero(records == record)
            class_id = int(y[take[0]])
            if class_id not in config["classes"]:
                continue
            metadata.append({"record_id": record, "subject": groups[take[0]], "class_id": class_id,
                "windows": len(take), "row_indices": take.tolist()})
            prototypes.append(standardized[take].astype(np.float64).mean(axis=0))
        points = np.asarray(prototypes, dtype=np.float64, order="C")
        subjects = np.asarray([row["subject"] for row in metadata])
        labels = np.asarray([row["class_id"] for row in metadata], dtype=np.int64)
        distances = cdist(points, points, metric="euclidean")
        require(np.isfinite(distances).all() and np.all(distances >= 0), "Invalid TRAIN distances")
        anchors = []
        for i, record in enumerate(metadata):
            for other_class in config["classes"]:
                if other_class == record["class_id"]:
                    continue
                targets = []
                for person in people:
                    same = np.flatnonzero((subjects == person) & (labels == record["class_id"]) & (np.arange(len(points)) != i))
                    other = np.flatnonzero((subjects == person) & (labels == other_class))
                    same_cell, other_cell = cell(distances[i, same]), cell(distances[i, other])
                    targets.append({"target_subject": person, "same_class": same_cell, "other_class": other_cell,
                        **compare_cells(same_cell, other_cell, config["epsilon"])})
                within = next(row for row in targets if row["target_subject"] == record["subject"])
                others = [row for row in targets if row["target_subject"] != record["subject"]]
                valid_others = [row for row in others if row["available"]]
                between = {"available": bool(valid_others), "target_people": len(others), "available_people": len(valid_others),
                    "same_minimum_mean": None, "other_minimum_mean": None, "raw_margin_mean": None, "normalized_margin_mean": None}
                if valid_others:
                    between.update(same_minimum_mean=float(np.mean([row["same_class"]["minimum"] for row in valid_others])),
                        other_minimum_mean=float(np.mean([row["other_class"]["minimum"] for row in valid_others])),
                        raw_margin_mean=float(np.mean([row["raw_margin"] for row in valid_others])),
                        normalized_margin_mean=float(np.mean([row["normalized_margin"] for row in valid_others])))
                paired = None
                if within["available"] and between["available"]:
                    paired = {"within_same_minimum": within["same_class"]["minimum"], "within_other_minimum": within["other_class"]["minimum"],
                        "between_person_mean_same_minimum": between["same_minimum_mean"], "between_person_mean_other_minimum": between["other_minimum_mean"],
                        "within_raw_margin": within["raw_margin"], "between_person_mean_raw_margin": between["raw_margin_mean"],
                        "within_normalized_margin": within["normalized_margin"], "between_person_mean_normalized_margin": between["normalized_margin_mean"],
                        "within_minus_between_normalized_margin": within["normalized_margin"] - between["normalized_margin_mean"]}
                anchors.append({"anchor_record_id": record["record_id"], "anchor_subject": record["subject"],
                    "anchor_class": record["class_id"], "other_class": other_class, "within": within, "between": between,
                    "paired_available": paired is not None, "paired": paired, "target_rows": targets})
        folder = output / "train_geometry" / part["id"]
        folder.mkdir(parents=True)
        prototype_path, scaler_path = folder / "record_prototypes.npz", folder / "scaler.joblib"
        arrays = {"prototypes": points, "class_ids": labels, "subjects": subjects,
            "record_ids": np.asarray([row["record_id"] for row in metadata]),
            "windows_per_record": np.asarray([row["windows"] for row in metadata], dtype=np.int64)}
        with prototype_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        prototype_path.chmod(0o444)
        with scaler_path.open("xb") as stream:
            joblib.dump(scaler, stream)
        scaler_path.chmod(0o444)
        row = {"outer_partition": part["id"], "started_at": started, "finished_at": now(),
            "training_windows": len(y), "training_records": len(np.unique(records)), "training_people": len(people),
            "triplet_records": len(metadata), "triplet_windows": sum(record["windows"] for record in metadata),
            **scope, "standardized_train_sha256": array_hash(standardized), "baseline_scaler_statistics_bytes_equal": True,
            "base_train_indices": base["indices"].tolist(), "ordinary_train_indices": ordinary["indices"].tolist(),
            "restoration": restoration, "prototype_arrays": reference(prototype_path, root), "scaler": reference(scaler_path, root),
            "prototype_array_hashes": {key: array_hash(value) for key, value in arrays.items()},
            "record_metadata": metadata, "anchors": anchors, "summary": summarize_anchors(anchors, config)}
        write(folder / "record_geometry.json", row)
        write(folder / "geometry_seal.json", {"sha256": digest(folder / "record_geometry.json"),
            "prototype_sha256": digest(prototype_path), "scaler_sha256": digest(scaler_path), "config_sha256": lock["config_sha256"]})
        results.append({key: value for key, value in row.items() if key not in ("anchors", "record_metadata")})
        results[-1]["detail"] = reference(folder / "record_geometry.json", root)
        print(json.dumps({"completed_outer_train": part["id"], "triplet_records": len(metadata),
            "paired_anchor_contrasts": row["summary"]["paired_anchor_contrast_observations"]}), flush=True)
    verify_inputs()
    result = {"schema_version": 1, "version": VERSION, "passed": True,
        "started_at": read(output / "execution_start.json")["started_at"], "finished_at": now(), "timezone": "Asia/Seoul",
        "config_sha256": lock["config_sha256"], "script_sha256": lock["script_sha256"], "origin_protocol_sha256": config["origin_protocol_sha256"],
        "classifier_fits": 0, "classifier_predictions": 0, "scaler_fits": len(results),
        "outer_validation_test_features_labels_or_scores_used": False, "protected_originals_count": len(protected),
        "protected_originals_unchanged": True, "inputs_unchanged": True, "origin_sources_unchanged": True,
        "split_times_train_window_observations": sum(len(rows) for rows in scoped_windows),
        "unique_source_train_windows": len(set.union(*scoped_windows)), "overlapping_outer_train_partitions_are_dependent": True,
        "environment": environment(), "partitions": results,
        "limitations": ["TRAIN-only fixed geometry diagnostic; no classifier or accuracy comparison, including no013prototype reevaluation.",
            "Different recording IDs do not establish independent acquisition sessions; session provenance is unknown.",
            "Other people are weighted equally; their records are not pooled for a single global nearest neighbor.",
            "Per-person nearest distances still depend on available record counts; those counts and missing cells are preserved.",
            "Paired means omit anchors missing either side; missing counts are explicit and no values are imputed.",
            "Euclidean geometry depends on596-coordinate representation and redundant statistics; no physical distance or causal diagnosis.",
            "Normalized margin uses fixed epsilon1e-12 in distance units; exact scale invariance is not claimed nearzero.",
            "OuterTRAIN partitions overlap and windows/records may be dependent; no independent-sample inference."]}
    write(output / "record_geometry_audit.json", result)
    write(output / "audit_seal.json", {"sha256": digest(output / "record_geometry_audit.json"),
        "config_sha256": lock["config_sha256"], "script_sha256": lock["script_sha256"], "sealed_at": now()})
    print(json.dumps({"output": str(output), "sha256": digest(output / "record_geometry_audit.json"), "classifier_fits": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output is not None else Path("research/results") / (
        datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S") + "_record_geometry_audit")
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    require(output.is_relative_to(root / "research/results") and not output.is_relative_to(root / ORIGIN),
        "Use a new independent research/results run outside018")
    if args.action == "freeze":
        freeze(root, output)
    else:
        require(args.output is not None, "Specify the frozen run directory")
        execute(root, output)


if __name__ == "__main__":
    main()
