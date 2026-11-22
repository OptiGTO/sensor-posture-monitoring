"""Freeze ordinary historical reservations for training-subject-only reuse.

The old retest reserved the first ordinary correct-posture recording per person.
This helper never includes true registration/calibration records, changes raw
files, constructs personal references, or changes any base evaluation window.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from src import posture_core, window_pipeline
from src.app_services import DatasetService
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, feature_names, normalize_text, read_sensor_csv, recording_label
from src.training_exclusions import is_training_excluded, load_training_exclusions
from src.window_pipeline import WindowSpec, extract_window_features, segment_sensor_frame


POLICY = "include_ordinary_reserved_training_records_v1"
CORE_SOURCES = {"src/posture_core.py": posture_core, "src/window_pipeline.py": window_pipeline}
EXPLICIT_REGISTRATION = frozenset({"registration", "calibration", "personal_registration"})


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sensor_digest(frame):
    values = np.asarray(frame[SENSOR_COLUMNS], dtype="<f8").copy()
    values[values == 0] = 0.
    return hashlib.sha256(np.asarray(values.shape, dtype="<i8").tobytes() + values.tobytes()).hexdigest()


def _reference(path, root):
    exists = path.is_file()
    return {"path": path.relative_to(root).as_posix(), "exists": exists,
        "sha256": _digest(path) if exists else None}


def _additive_sensor_identity_only(original_path, current_path):
    """Allow exactly the known identity helper, without feature-code changes.

    An additive function must not run decorators or default expressions during
    import. The removed import must be a standalone, unaliased hashlib import.
    Every remaining AST node, including feature signatures, stays identical.
    """
    original = ast.parse(Path(original_path).read_text(encoding="utf-8"))
    current = ast.parse(Path(current_path).read_text(encoding="utf-8"))

    def hash_import(node):
        return (isinstance(node, ast.Import) and len(node.names) == 1
            and node.names[0].name == "hashlib" and node.names[0].asname is None)

    def helper(node):
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "sensor_identity"

    if any(hash_import(node) or helper(node) for node in original.body):
        return False
    imports = [node for node in current.body if hash_import(node)]
    helpers = [node for node in current.body if helper(node)]
    if len(imports) != 1 or len(helpers) != 1:
        return False
    added = helpers[0]
    if (not isinstance(added, ast.FunctionDef) or added.decorator_list
            or added.args.defaults or any(value is not None for value in added.args.kw_defaults)
            or added.args.posonlyargs or added.args.kwonlyargs or added.args.vararg or added.args.kwarg
            or [argument.arg for argument in added.args.args] != ["frame"]):
        return False
    # Postponed annotations are required so the helper definition cannot add
    # annotation-expression execution to the original module's import.
    if not any(isinstance(node, ast.ImportFrom) and node.module == "__future__"
               and any(alias.name == "annotations" for alias in node.names) for node in original.body):
        return False
    current.body = [node for node in current.body if not hash_import(node) and not helper(node)]
    return ast.dump(original, include_attributes=False) == ast.dump(current, include_attributes=False)


def _feature_source_compatibility(root, origin, dataset):
    details = {}
    for relative, module in CORE_SOURCES.items():
        expected = dataset["source_snapshot_hashes"][relative]
        current_path = root / relative
        current_hash = _digest(current_path)
        if _digest(module.__file__) != current_hash:
            raise ValueError(f"Core feature source loaded by Python differs from project source: {relative}")
        policy = "exact_source_hash"
        if current_hash != expected:
            original_path = origin / "source_snapshot" / relative
            if not original_path.is_file() or _digest(original_path) != expected:
                raise ValueError(f"Verified original core source snapshot is unavailable: {relative}")
            if relative != "src/posture_core.py" or not _additive_sensor_identity_only(original_path, current_path):
                raise ValueError(f"Core feature source differs from frozen origin: {relative}")
            policy = "additive_sensor_identity_helper_only_ast_identical_features"
        details[relative] = {"original_sha256": expected, "current_sha256": current_hash,
            "loaded_sha256": current_hash, "compatibility": policy}
    return details


def _safe_source(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".csv":
        raise ValueError("Unsafe supplemental source path")
    path = root / "data" / relative
    if not path.resolve().is_relative_to((root / "data").resolve()):
        raise ValueError("Supplemental source escapes canonical data")
    return path


def _check_policy_manifests(root, dataset):
    result = {}
    for name in ("training_exclusions.json", "training_label_corrections.json"):
        path = root / "research/models" / name
        reference = _reference(path, root)
        expected = dataset["manifest_snapshots"][name]
        if reference["exists"] != expected["exists"] or reference["sha256"] != expected["sha256"]:
            raise ValueError(f"Eligibility manifest changed since origin freeze: {name}")
        result[name] = {**reference, "payload": _read(path) if reference["exists"] else None}
    if result["training_label_corrections.json"]["exists"]:
        raise ValueError("Supplemental restoration needs explicit support for active label corrections")
    return result


def _validate_arrays(data, *, supplemental=False):
    required = {"X", "y", "groups", "record_ids"}
    if not required.issubset(data):
        raise ValueError("Incomplete window arrays")
    X, y, groups, record_ids = (np.asarray(data[key]) for key in ("X", "y", "groups", "record_ids"))
    if X.ndim != 2 or X.shape[1] != 596 or not len(X) or not np.isfinite(X).all():
        raise ValueError("Expected finite nonempty 596-feature windows")
    if any(values.ndim != 1 or len(values) != len(X) for values in (y, groups, record_ids)):
        raise ValueError("Window identities and labels must align")
    if not np.issubdtype(y.dtype, np.integer) or not np.isin(y, np.arange(7)).all():
        raise ValueError("Invalid posture labels")
    if any(not str(value).strip() for values in (groups, record_ids) for value in values):
        raise ValueError("Empty subject or recording identity")
    if supplemental and np.any(y != CLASS_NAMES.index("정자세")):
        raise ValueError("Only ordinary correct-posture reservations may supplement training")
    ownership = {}
    for group, record in zip(groups, record_ids):
        identity = normalize_text(str(record))
        if identity in ownership and ownership[identity] != normalize_text(str(group)):
            raise ValueError("A recording cannot belong to multiple subjects")
        ownership[identity] = normalize_text(str(group))
    return X, y, groups, record_ids


def prepare_supplement(out, root, origin):
    """Write an audited feature cache and manifest, with no raw-data copies.

    ``out`` can be an existing run directory but both output filenames must be
    new. The returned manifest binds its feature cache to unchanged source
    bytes, current eligibility, and the original core feature implementation.
    """
    out, root, origin = map(lambda path: Path(path).resolve(), (out, root, origin))
    if out.is_relative_to((root / "data").resolve()) or out == origin:
        raise ValueError("Supplemental output must be isolated from data and the origin run")
    cache_path, manifest_path = out / "supplemental_training.npz", out / "supplemental_manifest.json"
    if cache_path.exists() or manifest_path.exists():
        raise ValueError("Supplemental outputs are immutable; use a new output directory")
    started = datetime.now().astimezone().isoformat()
    dataset_path, base_cache = origin / "dataset.json", origin / "data.npz"
    dataset = _read(dataset_path)
    dataset_hash = _digest(dataset_path)
    if _digest(base_cache) != dataset["cache_sha256"]:
        raise ValueError("Origin feature cache changed")
    expected_contract = (dataset["window_seconds"], dataset["sampling_rate_hz"],
        dataset["stride_seconds"], dataset["min_tail_fraction"], dataset["feature_dimension"])
    if expected_contract != (3, 12, 3, .95, 596) or dataset["class_names"] != CLASS_NAMES:
        raise ValueError("Supplement requires the original 3-second, 12-Hz, 596-feature contract")
    source_compatibility = _feature_source_compatibility(root, origin, dataset)
    source_hashes = {relative: row["current_sha256"] for relative, row in source_compatibility.items()}
    with np.load(base_cache, allow_pickle=False) as archive:
        base = {key: archive[key] for key in archive.files}
    _validate_arrays(base)
    subjects = {normalize_text(str(subject)) for subject in base["groups"]}
    if len(subjects) != dataset["cohort"]["subjects"]:
        raise ValueError("Frozen subject count differs from cache")
    policy_manifests = _check_policy_manifests(root, dataset)
    service = DatasetService(root)
    indexed = {record.csv_path.resolve(): record for record in service.records()}
    exclusions = load_training_exclusions(root / "data")
    base_paths = {normalize_text(str(record)) for record in base["record_ids"]}
    seen_paths = {normalize_text(row["source_relative_path"]) for row in dataset["records"]}
    seen_sensors = {row["sensor_sha256"] for row in dataset["records"]}
    seen_raw = {row["sha256"] for row in dataset["records"]}
    if seen_paths != base_paths:
        raise ValueError("Frozen recording identities differ from manifest")
    records, omitted, sequences, labels, groups, record_ids, starts = [], [], [], [], [], [], []
    for row in sorted(dataset["registrations_excluded"], key=lambda item: normalize_text(item["source_relative_path"])):
        subject = normalize_text(row["subject"])
        if subject not in subjects:
            omitted.append({"source_relative_path": row["source_relative_path"], "subject": subject,
                "reason": "not_in_frozen_subject_cohort"})
            continue
        path = _safe_source(root, row["source_relative_path"])
        identity = normalize_text(row["source_relative_path"])
        if _digest(path) != row["sha256"]:
            raise ValueError(f"Supplemental source hash changed: {identity}")
        before = path.stat()
        record = indexed.get(path.resolve())
        if record is None or not service.is_training_eligible(record) or is_training_excluded(path, root / "data", exclusions):
            raise ValueError(f"Supplemental recording is no longer eligible: {identity}")
        sidecar = path.with_suffix(".manifest.json")
        sidecar_ref = _reference(sidecar, root)
        original_sidecar = dataset["input_originals"].get(sidecar_ref["path"])
        if sidecar_ref["exists"] != (original_sidecar is not None) or original_sidecar and sidecar_ref["sha256"] != original_sidecar["sha256"]:
            raise ValueError(f"Supplemental sidecar changed: {identity}")
        metadata = _read(sidecar) if sidecar_ref["exists"] else {}
        if not isinstance(metadata, dict) or any(str(metadata.get(key, "")).strip().lower() in EXPLICIT_REGISTRATION
                for key in ("mode", "source_type", "training_role", "purpose")):
            raise ValueError(f"True registration or calibration must remain excluded: {identity}")
        if any(str(getattr(record, key, "")).strip().lower() in EXPLICIT_REGISTRATION
                for key in ("mode", "source_type", "training_role")):
            raise ValueError(f"True registration or calibration must remain excluded: {identity}")
        if normalize_text(record.subject) != subject or record.posture != "정자세" or row["label"] != "정자세":
            raise ValueError(f"Supplemental identity or posture differs from origin: {identity}")
        frame = read_sensor_csv(path)
        if recording_label(frame)[1] != "정자세":
            raise ValueError("Supplemental CSV does not contain ordinary correct-posture labels")
        sensor_hash = _sensor_digest(frame)
        if sensor_hash != row["sensor_sha256"]:
            raise ValueError(f"Supplemental sensor-content hash changed: {identity}")
        if identity in seen_paths or sensor_hash in seen_sensors or row["sha256"] in seen_raw:
            raise ValueError(f"Duplicate recording or sensor content across base and supplement: {identity}")
        seen_paths.add(identity); seen_sensors.add(sensor_hash); seen_raw.add(row["sha256"])
        seq, meta = segment_sensor_frame(frame, WindowSpec(3, 12, .95))
        if seq.shape[1:] != (36, 16) or not np.isfinite(seq).all():
            raise ValueError("Supplemental window shape or values violate the frozen input contract")
        sequences.extend(seq)
        labels.extend([CLASS_NAMES.index("정자세")] * len(seq))
        groups.extend([subject] * len(seq)); record_ids.extend([identity] * len(seq))
        starts.extend(meta.start_s.tolist())
        after = path.stat()
        if _digest(path) != row["sha256"] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Supplemental source changed while reading: {identity}")
        if _reference(sidecar, root) != sidecar_ref:
            raise ValueError(f"Supplemental sidecar changed while reading: {identity}")
        records.append({"source_relative_path": row["source_relative_path"], "record_id": identity,
            "subject": subject, "label": "정자세", "sha256": row["sha256"], "sensor_sha256": sensor_hash,
            "size": after.st_size, "mtime_ns": after.st_mtime_ns, "sidecar": sidecar_ref,
            "sidecar_metadata": metadata, "windows": len(seq), "window_start_s": meta.start_s.tolist(),
            "eligibility": {"source_type": record.source_type, "mode": record.mode,
                "label_status": record.label_status, "training_role": record.training_role,
                "qc_status": record.qc_status, "explicit_registration": False, "globally_excluded": False},
            "original_reservation": "First ordinary correct recording in historical NFC path order; not a personal reference input"})
    if set(groups) != subjects or len(records) != len(subjects):
        raise ValueError("Expected exactly one eligible ordinary reservation per frozen subject")
    arrays = {"seq": np.asarray(sequences, dtype=np.float32), "y": np.asarray(labels, dtype=np.int64),
        "groups": np.asarray(groups), "record_ids": np.asarray(record_ids), "start_s": np.asarray(starts)}
    arrays["X"] = extract_window_features(arrays["seq"])
    _validate_arrays(arrays, supplemental=True)
    if _check_policy_manifests(root, dataset) != policy_manifests or _digest(dataset_path) != dataset_hash or _digest(base_cache) != dataset["cache_sha256"]:
        raise ValueError("Frozen origin or eligibility policy changed during supplemental preparation")
    out.mkdir(parents=True, exist_ok=True)
    with cache_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest = {"schema_version": 1, "policy": POLICY, "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(), "origin": str(origin),
        "origin_dataset_sha256": dataset_hash, "origin_cache_sha256": dataset["cache_sha256"],
        "cache_sha256": _digest(cache_path), "feature_source_hashes": source_hashes,
        "feature_source_compatibility": source_compatibility,
        "eligibility_manifests": policy_manifests, "records": records, "omitted": omitted,
        "subjects": sorted(subjects), "record_count": len(records), "window_count": len(labels),
        "window_seconds": 3, "sampling_rate_hz": 12, "stride_seconds": 3, "min_tail_fraction": .95,
        "feature_names": feature_names(), "feature_dimension": 596, "class_names": CLASS_NAMES,
        "training_selection": "Append records only for subjects in the current complete training partition",
        "evaluation_policy": "Original validation/test rows and window identities remain unchanged",
        "weighting_policy": "Caller recomputes configured class or subject-class weights on augmented training rows",
        "personal_reference_used": False, "raw_files_copied": False, "raw_data_mutated": False}
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def append_training(data, supplement, train_indices):
    """Return (X, y, groups, audit) for one complete training-subject partition."""
    X, y, groups, record_ids = _validate_arrays(data)
    extra_X, extra_y, extra_groups, extra_records = _validate_arrays(supplement, supplemental=True)
    train = np.asarray(train_indices)
    if train.ndim != 1 or not np.issubdtype(train.dtype, np.integer) or not len(train) or len(set(train.tolist())) != len(train) or np.any(train < 0) or np.any(train >= len(X)):
        raise ValueError("Training indices must be unique valid integer rows")
    normalized_groups = np.asarray([normalize_text(str(group)) for group in groups])
    normalized_extra_groups = np.asarray([normalize_text(str(group)) for group in extra_groups])
    training_subjects = set(normalized_groups[train])
    if set(np.flatnonzero(np.isin(normalized_groups, list(training_subjects)))) != set(train.tolist()):
        raise ValueError("Training must contain complete subjects; partial-subject leakage is forbidden")
    if not set(normalized_extra_groups).issubset(set(normalized_groups)):
        raise ValueError("Supplement includes an unknown subject outside the frozen cohort")
    if {normalize_text(str(record)) for record in record_ids} & {normalize_text(str(record)) for record in extra_records}:
        raise ValueError("Supplement duplicates a base training or evaluation recording")
    take = np.flatnonzero(np.isin(normalized_extra_groups, list(training_subjects)))
    if set(normalized_extra_groups[take]) != training_subjects:
        raise ValueError("Supplement must cover every selected training subject")
    counts = Counter(str(record) for record in extra_records[take])
    audit = {"policy": POLICY, "training_subjects": sorted(training_subjects),
        "base_training_windows": len(train), "added_windows": len(take), "added_records": len(counts),
        "added_record_paths": sorted(counts), "added_windows_by_record": dict(sorted(counts.items())),
        "class_counts_before": np.bincount(y[train], minlength=7).tolist(),
        "class_counts_after": np.bincount(np.concatenate((y[train], extra_y[take])), minlength=7).tolist(),
        "evaluation_rows_unchanged": True, "personal_reference_used": False}
    return (np.concatenate((X[train], extra_X[take])), np.concatenate((y[train], extra_y[take])),
        np.concatenate((groups[train], extra_groups[take])), audit)
