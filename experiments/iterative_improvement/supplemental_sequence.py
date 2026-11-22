"""Validate sequence reuse in new, unsealed ordinary-restoration runs.

The original preparer already stores seq together with X. This extension keeps
those cache bytes and the original preparer unchanged, proves their alignment,
and adds sequence evidence only to the manifest it has just created.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from . import supplemental as base
from src.posture_core import SENSOR_COLUMNS, normalize_text
from src.window_pipeline import extract_window_features


SEALED_MARKERS = frozenset({"protocol.json", "protocol_seal.json", "selection.json", "selection_seal.json",
                            "evaluation.json", "evaluation_seal.json", "checkpoint.json"})


def _assert_unsealed(out):
    if any((out / marker).exists() for marker in SEALED_MARKERS):
        raise ValueError("Sequence supplementation requires a new unsealed run")


def _sequence_array(arrays, *, prove_features):
    if "seq" not in arrays:
        raise ValueError("Supplemental sequence cache is missing seq")
    sequence = np.asarray(arrays["seq"])
    features = np.asarray(arrays["X"])
    if (sequence.shape != (len(features), 36, 16) or sequence.dtype != np.dtype("float32")
            or not np.isfinite(sequence).all()):
        raise ValueError("Expected aligned finite float32 sequences with shape (N,36,16)")
    if prove_features:
        rebuilt = extract_window_features(sequence)
        if (rebuilt.dtype != features.dtype or not np.array_equal(rebuilt, features)
                or rebuilt.tobytes(order="C") != features.tobytes(order="C")):
            raise ValueError("Supplemental sequence/features fail bitwise parity")
    return sequence


def _validate_manifest_alignment(manifest, arrays):
    base._validate_arrays(arrays, supplemental=True)
    sequence = _sequence_array(arrays, prove_features=True)
    expected_groups, expected_records, expected_starts = [], [], []
    seen = set()
    for row in manifest["records"]:
        if row["record_id"] in seen:
            raise ValueError("Duplicate supplemental manifest record")
        seen.add(row["record_id"])
        count = row["windows"]
        if type(count) is not int or count < 1 or len(row["window_start_s"]) != count:
            raise ValueError("Supplemental manifest window counts differ")
        expected_groups.extend([row["subject"]] * count)
        expected_records.extend([row["record_id"]] * count)
        expected_starts.extend(row["window_start_s"])
    expected = {"groups": np.asarray(expected_groups), "record_ids": np.asarray(expected_records),
                "start_s": np.asarray(expected_starts, dtype=np.float64),
                "y": np.zeros(len(expected_groups), dtype=np.int64)}
    if (len(sequence) != manifest["window_count"] or len(seen) != manifest["record_count"]
            or set(expected_groups) != set(manifest["subjects"])):
        raise ValueError("Supplemental manifest sequence coverage differs")
    for key, values in expected.items():
        if key not in arrays or not np.array_equal(np.asarray(arrays[key]), values):
            raise ValueError(f"Supplemental sequence row identity differs from manifest: {key}")
    return sequence


def prepare_supplement_sequences(out, root, origin):
    """Create an eligibility-checked supplement and prove its sequence contract.

    Existing supplemental outputs or a frozen run are never amended. The base
    preparer verifies raw hashes, sidecars, exclusions and feature source parity.
    Its newly produced cache is retained byte-for-byte; only its new manifest
    receives additional evidence before the caller freezes the run protocol.
    """
    out = Path(out).resolve()
    _assert_unsealed(out)
    manifest = base.prepare_supplement(out, root, origin)
    cache_path = out / "supplemental_training.npz"
    manifest_path = out / "supplemental_manifest.json"
    original_manifest_hash = base._digest(manifest_path)
    cache_hash = base._digest(cache_path)
    if cache_hash != manifest["cache_sha256"] or base._read(manifest_path) != manifest:
        raise ValueError("Just-prepared supplement changed before sequence validation")
    with np.load(cache_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    sequence = _validate_manifest_alignment(manifest, arrays)
    manifest["supplemental_source_hashes"] = {
        "experiments/iterative_improvement/supplemental.py": base._digest(base.__file__),
        "experiments/iterative_improvement/supplemental_sequence.py": base._digest(__file__),
    }
    manifest["sequence_evidence"] = {
        "schema_version": 1, "shape": list(sequence.shape), "dtype": str(sequence.dtype),
        "axis_order": ["windows", "time", "sensors"], "sensor_order": SENSOR_COLUMNS,
        "aligned_keys": ["seq", "X", "y", "groups", "record_ids", "start_s"],
        "record_alignment": "All rows match manifest record order, subject, label and window_start_s",
        "feature_parity": {"method": "extract_window_features(seq); equal dtype, np.array_equal and C-order bytes",
                           "bitwise_equal": True, "checked_windows": len(sequence),
                           "feature_dimension": 596, "maximum_absolute_difference": 0.0},
        "sequence_array_sha256": hashlib.sha256(sequence.tobytes(order="C")).hexdigest(),
        "cache_bytes_reused_without_rewrite": True,
        "prepared_at": datetime.now().astimezone().isoformat(),
    }
    _assert_unsealed(out)
    if base._digest(cache_path) != cache_hash or base._digest(manifest_path) != original_manifest_hash:
        raise ValueError("Supplement changed during sequence validation")
    temporary = out / ".supplemental_manifest.sequence.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    _assert_unsealed(out)
    temporary.replace(manifest_path)
    return manifest


def append_sequence_training(data, supplement, train_indices):
    """Use the original augmentation guards/audit and return matching sequences."""
    # Preserve every identity, subject-completeness, label and overlap check in
    # the original feature helper, including overlap with base evaluation rows.
    unused_features, labels, groups, audit = base.append_training(data, supplement, train_indices)
    del unused_features
    sequence = _sequence_array(data, prove_features=False)
    extra_sequence = _sequence_array(supplement, prove_features=True)
    extra_groups = np.asarray([normalize_text(str(group)) for group in supplement["groups"]])
    take = np.flatnonzero(np.isin(extra_groups, audit["training_subjects"]))
    combined = np.concatenate((sequence[np.asarray(train_indices)], extra_sequence[take]))
    if len(combined) != len(labels):
        raise ValueError("Augmented sequence rows differ from feature training selection")
    return combined, labels, groups, audit
