"""Deterministic, record-constant training perturbations of frozen sensor windows.

This module never loads recordings or chooses training rows. Callers provide
only their selected training partition. Inference uses ``sequence_features``
directly, without applying the training perturbation.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Real

import numpy as np

from .posture_core import extract_feature_vector, normalize_text


GENERATOR_VERSION = "record_gain_v1"
MODES = frozenset({"identity", "common", "per_sensor"})
GENERATOR_RULE = (
    "Normalize record_id by Unicode NFC then strip. Encode the object with keys "
    "generator_version, seed, mode, record_id using json.dumps(ensure_ascii=False, "
    "sort_keys=True, separators=(',', ':')).encode('utf-8'). Use its SHA-256 digest "
    "as an unsigned big-endian integer seed for numpy.random.Generator(numpy.random.PCG64(seed)). "
    "For common draw one float64 uniform(1-strength, 1+strength) and repeat across 16 sensors; "
    "for per_sensor draw 16 float64 uniforms in canonical sensor order. Identity or zero strength "
    "uses 16 exact ones. Labels, subject, row order, window position and strength do not enter the seed."
)
GENERATOR_RULE_SHA256 = hashlib.sha256(GENERATOR_RULE.encode("utf-8")).hexdigest()
AUGMENTATION_RULE = (
    "One positive gain vector per canonical record, shared by all windows and times; "
    "multiply in float64 then round once to float32; zero strength copies bits unchanged; "
    "no clipping, offset, permutation, extra rows or in-place writes."
)


def validate_sequences(sequences):
    """Validate the fixed finite float32 (N,36,16) input without modifying it."""
    values = np.asarray(sequences)
    if values.ndim != 3 or values.shape[1:] != (36, 16) or not len(values):
        raise ValueError("Expected nonempty sensor sequences with shape (N, 36, 16)")
    if values.dtype != np.dtype(np.float32) or not np.isfinite(values).all():
        raise ValueError("Sensor sequences must be finite float32 values")
    return values


def _identities(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{name} must be a nonempty one-dimensional identity array")
    if any(not isinstance(value, str) or not normalize_text(value) for value in array.tolist()):
        raise ValueError(f"{name} must contain nonempty string identities")
    return np.asarray([normalize_text(value) for value in array.tolist()])


def validate_training_records(y, groups, record_ids):
    """Return canonical record IDs and sorted metadata after purity validation.

    Each normalized record must have exactly one normalized subject and one
    integer label in 0..6. The wrapper, rather than this helper, requires all
    seven labels. Original input arrays and identity strings remain untouched.
    """
    labels = np.asarray(y)
    people, records = _identities(groups, "groups"), _identities(record_ids, "record_ids")
    if labels.ndim != 1 or not len(labels) or len(people) != len(labels) or len(records) != len(labels):
        raise ValueError("Labels, groups and record identities must align")
    if labels.dtype.kind not in "iu" or not np.isin(labels, np.arange(7)).all():
        raise ValueError("Training labels must be integers in 0..6")
    metadata = []
    for record in sorted(set(records.tolist())):
        rows = np.flatnonzero(records == record)
        if len(set(people[rows])) != 1 or len(set(labels[rows])) != 1:
            raise ValueError("Each record must belong to one subject and one posture class")
        metadata.append({"record_id": record, "subject": str(people[rows[0]]),
                         "class_id": int(labels[rows[0]]), "windows": len(rows),
                         "row_indices": rows.tolist()})
    return records, metadata


def _configuration(mode, strength, seed):
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("Unknown record gain mode")
    if isinstance(strength, (bool, np.bool_)) or not isinstance(strength, Real):
        raise ValueError("Gain strength must be finite and in [0, 1)")
    strength = float(strength)
    if not np.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError("Gain strength must be finite and in [0, 1)")
    if mode == "identity" and strength != 0:
        raise ValueError("Identity mode requires zero gain strength")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Effective gain seed must be a nonnegative uint32 integer")
    return strength


def _generator_digest(record_id, mode, seed):
    payload = {"generator_version": GENERATOR_VERSION, "seed": seed,
               "mode": mode, "record_id": record_id}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def expected_record_gains(record_ids, *, mode, strength, seed):
    """Reconstruct exact sorted canonical-record -> 16 float64 gains, without data."""
    strength = _configuration(mode, strength, seed)
    records = _identities(record_ids, "record_ids")
    result = {}
    for record in sorted(set(records.tolist())):
        if strength == 0:
            gain = np.ones(16, dtype=np.float64)
        else:
            number = int.from_bytes(_generator_digest(record, mode, seed), byteorder="big", signed=False)
            generator = np.random.Generator(np.random.PCG64(number))
            gain = generator.uniform(1 - strength, 1 + strength, size=1 if mode == "common" else 16)
            if mode == "common":
                gain = np.repeat(gain, 16)
        if (not np.isfinite(gain).all() or (gain <= 0).any()
                or (gain < 1 - strength).any() or (gain > 1 + strength).any()):
            raise ValueError("Generated gains violate the finite positive configured bounds")
        result[record] = gain.tolist()
    return result


def augment_record_gains(sequences, y, groups, record_ids, *, mode="identity", strength=0.0, seed=20260921):
    """Return new sequences in the same row order and a complete deterministic audit."""
    values = validate_sequences(sequences)
    strength = _configuration(mode, strength, seed)
    normalized_records, metadata = validate_training_records(y, groups, record_ids)
    if len(normalized_records) != len(values):
        raise ValueError("Training recording metadata must align with all sequence rows")
    gains = expected_record_gains(normalized_records, mode=mode, strength=strength, seed=seed)
    result = values.copy()
    if strength != 0:
        for record in metadata:
            rows = record["row_indices"]
            gain = np.asarray(gains[record["record_id"]], dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                result[rows] = (values[rows].astype(np.float64) * gain[None, None, :]).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Gain augmentation produced nonfinite sensor values")
        if not np.array_equal(result == 0, values == 0) or not np.array_equal(np.signbit(result), np.signbit(values)):
            raise ValueError("Gain augmentation must preserve zero locations and pressure signs")
    records = [{**row, "generator_sha256": _generator_digest(row["record_id"], mode, seed).hex()}
               for row in metadata]
    audit = {"schema_version": 1, "generator_version": GENERATOR_VERSION,
        "generator_rule": GENERATOR_RULE, "generator_rule_sha256": GENERATOR_RULE_SHA256,
        "seed": seed, "mode": mode, "strength": strength,
        "record_id_normalization": "Unicode NFC then strip",
        "input_shape": list(values.shape), "output_shape": list(result.shape),
        "sequence_dtype": "float32", "gain_dtype": "float64",
        "record_count": len(records), "window_count": len(values),
        "records": records, "record_gains": gains, "augmentation_rule": AUGMENTATION_RULE,
        "row_count_unchanged": True, "row_order_unchanged": True,
        "labels_used_for_gain_generation": False}
    return result, audit


def sequence_features(sequences):
    """Extract all 596 canonical features without applying any gain augmentation."""
    values = validate_sequences(sequences)
    features = np.stack([extract_feature_vector(sequence) for sequence in values])
    if features.shape != (len(values), 596) or features.dtype != np.dtype(np.float32) or not np.isfinite(features).all():
        raise ValueError("Canonical extraction must produce finite float32 (N, 596) features")
    return features
