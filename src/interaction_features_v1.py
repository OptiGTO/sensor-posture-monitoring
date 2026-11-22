"""Fixed numerical feature interactions; no fitting or physical symmetry claim.

Raw windows use the immutable canonical extractor. Hadamard coordinates apply
only AFTER the caller's training-fitted StandardScaler; temporal correlations
are extracted separately inside each raw window BEFORE that scaler.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
import json

import numpy as np

from .posture_core import SENSOR_COLUMNS, feature_names
from .record_gain_features_v1 import sequence_features, validate_sequences


VERSION = "interaction_features_v1"
MODES = frozenset({"identity", "hadamard512", "temporal_corr120"})
STATISTICS = (
    "mean", "std", "median", "min", "max", "range", "iqr", "rms",
    "mean_abs_diff", "slope", "skew", "kurtosis", "dominant_frequency",
    "spectral_entropy", "spectral_centroid", "low_frequency_ratio",
)


def _mode(mode):
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"Unknown interaction feature mode: {mode!r}")
    return mode


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _schema():
    sensors = [f"R{i}{j}" for i in range(1, 5) for j in range(1, 5)]
    names = feature_names()
    if SENSOR_COLUMNS != sensors or len(names) != 596 or len(set(names)) != 596:
        raise ValueError("Interaction features require the canonical sensor and 596-feature schema")
    blocks = []
    for prefix, offset in (("abs", 0), ("rel", 298)):
        for number, statistic in enumerate(STATISTICS):
            start = offset + 16 * number
            if names[start:start + 16] != [f"{prefix}_{statistic}_{sensor}" for sensor in sensors]:
                raise ValueError("Canonical sensor-statistic feature ordering changed")
            blocks.append({"prefix": prefix, "statistic": statistic, "start": start, "stop": start + 16})
    return names, sensors, blocks


def _hadamard16():
    matrix = np.ones((1, 1), dtype=np.float64)
    for _ in range(4):
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / 4.0


def _names(mode, canonical, sensors, blocks):
    extraction = list(canonical)
    if mode == "temporal_corr120":
        extraction += [f"raw_temporal_corr_{sensors[i]}_vs_{sensors[j]}" for i, j in combinations(range(16), 2)]
    transformed = list(extraction)
    if mode == "hadamard512":
        for block in blocks:
            transformed[block["start"]:block["stop"]] = [
                f"{block['prefix']}_{block['statistic']}_hadamard_{coordinate:02d}" for coordinate in range(16)]
    return extraction, transformed


def interaction_feature_names(mode):
    """Names of final model coordinates, AFTER the standardized transform."""
    mode = _mode(mode)
    canonical, sensors, blocks = _schema()
    return np.asarray(_names(mode, canonical, sensors, blocks)[1], dtype=object)


def interaction_rule(mode):
    """Return a fresh complete JSON rule; its own digest excludes rule_sha256."""
    mode = _mode(mode)
    canonical, sensors, blocks = _schema()
    extraction, transformed = _names(mode, canonical, sensors, blocks)
    rule = {
        "schema_version": 1, "version": VERSION, "mode": mode,
        "sequence_shape": [None, 36, 16], "sequence_dtype": "float32",
        "feature_dtype": "float32", "sensor_order": sensors,
        "sensor_order_sha256": _json_digest(sensors),
        "canonical_feature_names_sha256": _json_digest(canonical),
        "extraction_feature_names": extraction,
        "transformed_feature_names": transformed,
        "extraction_feature_names_sha256": _json_digest(extraction),
        "transformed_feature_names_sha256": _json_digest(transformed),
        "extraction_feature_count": len(extraction), "transformed_feature_count": len(transformed),
        "canonical_extraction": "Immutable sequence_features: posture_core.extract_feature_vector per window, exact original 596 float32 values",
        "scaler_placement": "Caller fits global StandardScaler on extracted training features only, then applies standardized transform",
        "population_or_label_fitting": False, "evaluation_batch_adaptation": False,
        "physical_rotation_or_reflection_claim": False,
        "hadamard": None, "correlation": None,
        "standardized_transform": "Exact float32 copy; no arithmetic",
        "json_hash_encoding": "UTF-8 json.dumps(ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)",
    }
    if mode == "hadamard512":
        matrix = _hadamard16()
        transformed_indices = {index for block in blocks for index in range(block["start"], block["stop"])}
        rule["standardized_transform"] = "For each specified block, float64 einsum('ni,ji->nj', block, H16/4, optimize=False), then one float32 rounding; copy other coordinates bitwise"
        rule["hadamard"] = {
            "construction": "Sylvester H1=[1]; repeat H2n=[[Hn,Hn],[Hn,-Hn]] four times; divide H16 by 4",
            "matrix": matrix.tolist(), "matrix_shape": [16, 16],
            "matrix_sha256": hashlib.sha256(matrix.astype("<f8").tobytes(order="C")).hexdigest(),
            "matrix_hash_encoding": "little-endian float64 raw C-order bytes",
            "block_order": blocks, "block_order_sha256": _json_digest(blocks),
            "unchanged_indices": [index for index in range(596) if index not in transformed_indices],
            "input_coordinate_count": 596, "changed_coordinate_count": 512,
            "orthogonal_before_float32_rounding": True,
        }
    if mode == "temporal_corr120":
        pairs = [list(pair) for pair in combinations(range(16), 2)]
        rule["correlation"] = {
            "pair_indices": pairs, "pair_order_sha256": _json_digest(pairs),
            "pair_order": "Canonical sensor indices i<j in lexicographic order",
            "calculation": "Per raw window convert to float64; subtract each sensor's 36-timepoint mean; Gram=einsum('ti,tj->ij', centered, centered, optimize=False); divide Gram[i,j] by sqrt(Gram[i,i]*Gram[j,j])",
            "constant_channel_pair": 0.0,
            "clipping": "Clip computed correlation only to mathematical [-1,1] bounds for floating overshoot; never clip sensor inputs",
            "output_rounding": "One float32 rounding before concatenating after the unchanged original 596 features",
            "evaluation_record_aggregation": False,
        }
    rule["rule_sha256"] = _json_digest(rule)
    return rule


def extract_interaction_features(sequences, mode):
    """Extract features independently per raw window; no scaling is fitted."""
    mode = _mode(mode)
    _schema()
    values = validate_sequences(sequences)
    canonical = sequence_features(values)
    if mode != "temporal_corr120":
        return canonical
    pairs = np.asarray(list(combinations(range(16), 2)), dtype=np.int64)
    correlations = np.empty((len(values), 120), dtype=np.float32)
    for index, sequence in enumerate(values):
        raw = sequence.astype(np.float64)
        centered = raw - raw.mean(axis=0, keepdims=True)
        gram = np.einsum("ti,tj->ij", centered, centered, optimize=False)
        energy = np.diag(gram)
        denominator = np.sqrt(energy[pairs[:, 0]] * energy[pairs[:, 1]])
        correlation = np.zeros(120, dtype=np.float64)
        np.divide(gram[pairs[:, 0], pairs[:, 1]], denominator, out=correlation, where=denominator > 0)
        if not np.isfinite(correlation).all():
            raise ValueError("Temporal correlations must remain finite")
        correlations[index] = np.clip(correlation, -1.0, 1.0).astype(np.float32)
    result = np.concatenate((canonical, correlations), axis=1)
    if result.shape != (len(values), 716) or result.dtype != np.dtype(np.float32) or not np.isfinite(result).all():
        raise ValueError("Correlation extraction must produce finite float32 (N, 716) features")
    return result


def transform_standardized_features(values, mode):
    """Apply the fixed post-scaler transform while preserving input ownership."""
    mode = _mode(mode)
    canonical, sensors, blocks = _schema()
    names, _ = _names(mode, canonical, sensors, blocks)
    if hasattr(values, "columns") and list(values.columns) != names:
        raise ValueError("Standardized feature names or ordering differ from the extraction schema")
    array = np.asarray(values)
    if (array.ndim != 2 or array.shape[1] != len(names) or not len(array)
            or array.dtype != np.dtype(np.float32) or not np.isfinite(array).all()):
        raise ValueError(f"Expected nonempty finite float32 standardized features with shape (N, {len(names)})")
    result = array.copy()
    if mode == "hadamard512":
        matrix = _hadamard16()
        with np.errstate(over="ignore", invalid="ignore"):
            for block in blocks:
                take = slice(block["start"], block["stop"])
                rotated = np.einsum("ni,ji->nj", array[:, take].astype(np.float64), matrix, optimize=False)
                result[:, take] = rotated.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Hadamard coordinates exceed the finite float32 range")
    return result
