"""Freeze current eligible recordings without changing raw data or active models."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

from src.app_services import DatasetService
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, feature_names, normalize_text, read_sensor_csv, recording_label
from src.training_exclusions import is_training_excluded, load_training_exclusions
from src.window_pipeline import WindowSpec, extract_window_features, segment_sensor_frame

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260921


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def protected_snapshot(root=ROOT):
    """Record existing bytes and modification times; includes active identity files."""
    root = Path(root)
    result = {}
    for directory in (root / "data", root / "research/models"):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                result[path.relative_to(root).as_posix()] = {
                    "sha256": digest(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    "symlink_target": str(path.readlink()) if path.is_symlink() else None,
                }
    return result


def assert_protected(output, root=ROOT):
    if protected_snapshot(root) != read(Path(output) / "protected_before.json"):
        raise ValueError("Original data or existing model inventory/hash/mtime changed")


def sensor_digest(frame):
    """Canonical numeric channels ignore CSV encoding, labels and timestamp origin."""
    values = np.asarray(frame[SENSOR_COLUMNS], dtype="<f8").copy()
    values[values == 0] = 0.0  # Canonicalize IEEE negative zero.
    return hashlib.sha256(np.asarray(values.shape, dtype="<i8").tobytes() + values.tobytes()).hexdigest()


def reserve_registration(records, *, enabled=False):
    """Keep ordinary recordings unless historical reservation is explicitly requested."""
    if not enabled:
        return [], list(records), []
    grouped = defaultdict(list)
    for row in records:
        grouped[row["subject"]].append(row)
    registrations, retained, exclusions = [], [], []
    for subject, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: normalize_text(row["source_relative_path"]))
        correct = [row for row in rows if row["label"] == CLASS_NAMES[0]]
        if not correct:
            exclusions.extend({**row, "reason": "subject_has_no_eligible_registration_record"} for row in rows)
            continue
        registration = correct[0]
        registrations.append(registration)
        remaining = [row for row in rows if row is not registration]
        if not remaining:
            exclusions.append({**registration, "reason": "subject_has_only_reserved_registration_record"})
        retained.extend(remaining)
    return registrations, retained, exclusions


def subject_partitions(groups, y, record_ids):
    subjects = np.asarray(sorted(set(groups)))
    if len(subjects) < 13:
        raise ValueError(f"Six validation and six test subjects require at least 13 subjects; found {len(subjects)}")
    shuffled = np.random.default_rng(SEED).permutation(subjects)
    holdout = sorted(shuffled[:6].tolist())
    development = np.asarray(sorted(set(subjects) - set(holdout)))
    result = []
    for index, seed in enumerate((SEED, SEED + 1, SEED + 2)):
        order = np.random.default_rng(seed).permutation(development)
        split = {"id": f"development_{index + 1}", "seed": seed,
                 "train": sorted(order[6:].tolist()), "validation": sorted(order[:6].tolist()),
                 "test": holdout, "counts": {}}
        membership = [set(split[name]) for name in ("train", "validation", "test")]
        if any(membership[i] & membership[j] for i in range(3) for j in range(i)):
            raise ValueError("Subject overlap")
        for name in ("train", "validation", "test"):
            mask = np.isin(groups, split[name])
            if set(y[mask]) != set(range(len(CLASS_NAMES))):
                raise ValueError(f"Missing class in {split['id']} {name}; no score-based resampling allowed")
            split["counts"][name] = {"subjects": len(split[name]), "records": len(set(record_ids[mask])),
                "windows": int(mask.sum()), "class_counts": np.bincount(y[mask], minlength=7).tolist()}
        result.append(split)
    return result


def verify(output, root=ROOT, *, protected=True):
    output, root = Path(output), Path(root)
    manifest = read(output / "dataset.json")
    if digest(output / "dataset.json") != read(output / "dataset_lock.json")["sha256"]:
        raise ValueError("Frozen dataset manifest changed")
    if digest(output / "data.npz") != manifest["cache_sha256"]:
        raise ValueError("Frozen feature/sequence cache changed")
    for relative, expected in manifest["source_snapshot_hashes"].items():
        if digest(output / "source_snapshot" / relative) != expected:
            raise ValueError(f"Frozen data construction source changed: {relative}")
    for row in manifest["staged_sources"]:
        if digest(output / "training_data" / row["staged_file"]) != row["sha256"]:
            raise ValueError("Staged recording changed")
    if protected:
        assert_protected(output, root)
    return manifest


def prepare(output, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    if (output / "dataset.json").exists():
        return verify(output, root)
    output.mkdir(parents=True, exist_ok=False)
    started = now()
    protected = protected_snapshot(root)
    write_json(output / "protected_before.json", protected)
    service = DatasetService(root)
    indexed = service.records()
    exclusions = load_training_exclusions(root / "data")
    correction_path = root / "research/models/training_label_corrections.json"
    if correction_path.exists() and not hasattr(service, "correction_manifest"):
        raise ValueError("An active correction manifest exists but DatasetService cannot apply it")
    selected, excluded = [], []
    for record in indexed:
        metadata = read(record.manifest_path) if record.manifest_path else {}
        explicit_registration = any(str(metadata.get(key, "")).lower() in {
            "registration", "calibration", "personal_registration"}
            for key in ("mode", "source_type", "training_role", "purpose"))
        reason = ("user_training_exclusion" if is_training_excluded(record.csv_path, root / "data", exclusions)
                  else "explicit_registration" if explicit_registration
                  else "not_training_eligible" if not service.is_training_eligible(record) else None)
        if reason:
            excluded.append({"source_relative_path": record.csv_path.relative_to(root / "data").as_posix(),
                "subject": normalize_text(record.subject), "reason": reason, "label_status": record.label_status,
                "training_role": record.training_role, "qc_status": record.qc_status})
        else:
            selected.append(record)
    service.stage_for_training(selected, output / "training_data")
    staged = read(output / "training_data/staged_sources.json")
    original_by_path = {str(record.csv_path.resolve()): record for record in selected}
    records, frames = [], {}
    duplicates = {"raw": defaultdict(list), "sensor": defaultdict(list)}
    for row in staged:
        source = Path(row["source_file"])
        original = original_by_path[str(source)]
        frame = read_sensor_csv(output / "training_data" / row["staged_file"])
        _, label = recording_label(frame)
        sensor_hash = sensor_digest(frame)
        relative = source.relative_to(root / "data").as_posix()
        record = {"source_relative_path": relative, "subject": normalize_text(original.subject),
            "label": label, "staged_file": row["staged_file"], "sha256": digest(source),
            "staged_sha256": row["sha256"], "sensor_sha256": sensor_hash,
            "size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns,
            "rows": len(frame), "duration_s": float(frame.elapsed_s.iloc[-1] - frame.elapsed_s.iloc[0])}
        records.append(record)
        frames[relative] = frame
        duplicates["raw"][record["sha256"]].append(relative)
        duplicates["sensor"][sensor_hash].append(relative)
    duplicate_report = {key: [paths for paths in values.values() if len(paths) > 1] for key, values in duplicates.items()}
    write_json(output / "duplicate_audit.json", duplicate_report)
    if any(duplicate_report.values()):
        raise ValueError("Duplicate raw or normalized sensor content; resolve cohort before splitting")
    registrations, retained, registration_exclusions = reserve_registration(records)
    for row in registrations:
        frame = frames[row["source_relative_path"]]
        relative_time = frame.elapsed_s - frame.elapsed_s.iloc[0]
        row["reference_mean_first_10s"] = frame.loc[relative_time <= 10, SENSOR_COLUMNS].to_numpy(float).mean(0).tolist()
    seq, ys, groups, ids, starts = [], [], [], [], []
    for record in retained:
        windows, meta = segment_sensor_frame(frames[record["source_relative_path"]], WindowSpec(3))
        seq.extend(windows)
        ys.extend([CLASS_NAMES.index(record["label"])] * len(windows))
        groups.extend([record["subject"]] * len(windows))
        ids.extend([normalize_text(record["source_relative_path"])] * len(windows))
        starts.extend(meta.start_s.tolist())
        record["window_count"] = len(windows)
    data = {"seq": np.asarray(seq, dtype=np.float32), "y": np.asarray(ys, dtype=np.int64),
        "groups": np.asarray(groups), "record_ids": np.asarray(ids), "start_s": np.asarray(starts)}
    data["X"] = extract_window_features(data["seq"])
    if data["seq"].shape[1:] != (36, 16) or data["X"].shape != (len(ys), 596):
        raise ValueError("Unexpected sequence or feature shape")
    if not all(np.isfinite(data[key]).all() for key in ("seq", "X", "start_s")):
        raise ValueError("Nonfinite sequence or features")
    partitions = subject_partitions(data["groups"], data["y"], data["record_ids"])
    np.savez_compressed(output / "data.npz", **data)
    sources = [Path(__file__), Path(__file__).with_name("__init__.py"), root / "AGENTS.md",
        root / "src/app_services.py", root / "src/posture_core.py", root / "src/window_pipeline.py",
        root / "src/training_exclusions.py", root / "src/training_windows.py"]
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(root)
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[relative.as_posix()] = digest(source)
    manifests = {}
    for name in ("training_label_corrections.json", "training_exclusions.json"):
        path = root / "research/models" / name
        manifests[name] = {"exists": path.exists(), "sha256": digest(path) if path.exists() else None}
        if path.exists():
            shutil.copy2(path, output / name)
    active_identity = {}
    for name in ("active_model.json", "release_selection.json"):
        path = root / "research/models" / name
        active_identity[name] = {"sha256": digest(path), "payload": read(path)} if path.exists() else None
    manifest = {"schema_version": 1, "started_at": started, "finished_at": now(), "seed": SEED,
        "purpose": "Retrospective method rerun on reused historical subjects; no independent fresh validation",
        "historical_holdout": True, "test_previously_inspected": True, "physically_validated": False,
        "raw_data_mutated": False, "candidate_approval": False,
        "cohort": {"indexed_records": len(indexed), "eligible_before_registration": len(records),
            "subjects_before_registration": len({row["subject"] for row in records}),
            "subjects": len(set(groups)), "records": len(retained), "windows": len(ys),
            "reserved_registration_records": len(registrations), "class_window_counts": np.bincount(ys, minlength=7).tolist(),
            "class_record_counts": dict(Counter(row["label"] for row in retained))},
        "excluded_indexed_records": excluded, "registration_only_subject_exclusions": registration_exclusions,
        "registrations_excluded": registrations, "records": retained, "staged_sources": staged,
        "duplicate_audit": duplicate_report, "partitions": partitions,
        "input_originals": {key: value for key, value in protected.items() if key.startswith("data/")},
        "manifest_snapshots": manifests, "active_identity": active_identity,
        "label_correction_policy": "Use current DatasetService and active manifest only; absent correction manifest means no historical corrections restored",
        "training_exclusion_policy": "Hash-verified active training_exclusions applied explicitly in addition to DatasetService eligibility",
        "registration_policy": "Disabled: ordinary correct recordings remain eligible; no automatic registration reservation",
        "registration_scope": "No registration reference is constructed",
        "split_policy": "NFC-sorted subjects; numpy default_rng(20260921) permutation first6 holdout; remaining subjects independently permuted seeds20260921/22/23 first6 validation and all others train",
        "class_names": CLASS_NAMES, "feature_names": feature_names(), "feature_dimension": 596,
        "window_seconds": 3, "sampling_rate_hz": 12, "stride_seconds": 3, "min_tail_fraction": .95,
        "cache_sha256": digest(output / "data.npz"), "source_snapshot_hashes": source_hashes,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "command": [sys.executable, *sys.argv], "sensor_acquisition_time": "Not inferred from mtime or execution time",
        "environment": {"python": sys.version, "platform": platform.platform(),
            "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")}}}
    write_json(output / "dataset.json", manifest)
    write_json(output / "dataset_lock.json", {"sha256": digest(output / "dataset.json"), "locked_at": now()})
    assert_protected(output, root)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "cohort": result["cohort"],
        "cache_sha256": result["cache_sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
