"""Train-only recording prototypes with independent 596-feature inference.

The caller owns feature scaling and the training split. This estimator never
aggregates evaluation windows, adapts to a query batch, or reads recordings.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from numbers import Real

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .record_gain_features_v1 import validate_training_records


BACKEND_VERSION = "record_prototype_v1"
FEATURE_COUNT = 596
DISTANCE_BATCH_SIZE = 128


def _parameters(n_neighbors, epsilon):
    if type(n_neighbors) is not int or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")
    if (isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real)
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("epsilon must be finite and strictly positive")
    return n_neighbors, float(epsilon)


def _features(X):
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] != FEATURE_COUNT or not len(values)
            or values.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or not np.isfinite(values).all()):
        raise ValueError("Expected nonempty finite float32/float64 features with shape (N, 596)")
    return values


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _metadata_digest(metadata):
    return hashlib.sha256(_json_bytes(metadata)).hexdigest()


def _input_digest(values):
    digest = hashlib.sha256(_json_bytes({"dtype": values.dtype.str, "shape": list(values.shape)}))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _state_digest(model):
    digest = hashlib.sha256(_json_bytes({
        "version": BACKEND_VERSION, "n_neighbors": model.n_neighbors,
        "epsilon": float(model.epsilon), "features": model.n_features_in_,
        "training_rows": model.n_training_samples_, "records": model.n_records_,
        "metadata_sha256": model.metadata_sha256_,
        "input_dtype": model.input_dtype_, "input_sha256": model.input_sha256_,
    }))
    for name in ("prototypes_", "prototype_labels_", "class_record_counts_", "class_counts_", "classes_"):
        values = getattr(model, name)
        digest.update(_json_bytes({"name": name, "dtype": values.dtype.str,
                                   "shape": list(values.shape)}))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _training_details(model):
    return {
        "backend_version": BACKEND_VERSION,
        "input_shape": [model.n_training_samples_, FEATURE_COUNT],
        "input_dtype": model.input_dtype_, "input_sha256": model.input_sha256_,
        "record_count": model.n_records_, "classes": model.classes_.tolist(),
        "class_counts": model.class_counts_.tolist(),
        "class_record_counts": model.class_record_counts_.tolist(),
        "training_subjects": sorted({row["subject"] for row in model.record_metadata_}),
        "record_metadata": deepcopy(model.record_metadata_),
        "n_neighbors": model.n_neighbors_fit_, "epsilon": model.epsilon_fit_,
        "prototype_aggregation": "Per-feature ascending float64 arithmetic mean within each canonical training record",
        "distance": "euclidean",
        "scoring": "1/(epsilon + mean k nearest distances within each class), normalized across seven classes",
        "record_id_normalization": "Unicode NFC then strip",
        "query_batch_size": DISTANCE_BATCH_SIZE,
        "evaluation_record_aggregation": False,
        "metadata_sha256": model.metadata_sha256_, "state_sha256": model.state_sha256_,
    }


class RecordPrototypeClassifier(ClassifierMixin, BaseEstimator):
    """Inverse mean distance to the k nearest training records in each class."""

    def __init__(self, n_neighbors=1, epsilon=1e-12):
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon

    def fit(self, X, y, *, groups, record_ids):
        k, epsilon = _parameters(self.n_neighbors, self.epsilon)
        values = _features(X)
        records, metadata = validate_training_records(y, groups, record_ids)
        if len(records) != len(values):
            raise ValueError("Training features and recording metadata must align")
        labels = np.asarray(y)
        if not np.array_equal(np.unique(labels), np.arange(7)):
            raise ValueError("Record prototypes require all seven training labels 0..6")
        prototype_labels = np.asarray([row["class_id"] for row in metadata], dtype=np.int64)
        counts = np.bincount(prototype_labels, minlength=7).astype(np.int64)
        if np.any(counts < k):
            raise ValueError("Every class must have at least n_neighbors distinct training records")
        # Sorting each feature's values gives the same arithmetic mean reduction
        # order after any permutation of training rows. No labels choose features.
        prototypes = []
        with np.errstate(over="ignore", invalid="ignore"):
            for row in metadata:
                observations = values[row["row_indices"]].astype(np.float64)
                prototypes.append(np.sort(observations, axis=0).mean(axis=0, dtype=np.float64))
        prototypes = np.asarray(prototypes, dtype=np.float64)
        if not np.isfinite(prototypes).all():
            raise ValueError("Recording means exceed the finite float64 range")
        # Commit fitted state only after the complete input has passed validation.
        self.prototypes_ = prototypes
        self.prototype_labels_ = prototype_labels
        self.class_record_counts_ = counts
        self.class_counts_ = np.bincount(labels.astype(np.int64), minlength=7).astype(np.int64)
        self.record_metadata_ = metadata
        self.classes_ = np.arange(7, dtype=np.int64)
        self.n_features_in_ = FEATURE_COUNT
        self.n_training_samples_ = len(values)
        self.n_records_ = len(metadata)
        self.n_neighbors_fit_ = k
        self.epsilon_fit_ = epsilon
        self.input_dtype_ = values.dtype.name
        self.input_sha256_ = _input_digest(values)
        self.metadata_sha256_ = _metadata_digest(metadata)
        self.state_sha256_ = _state_digest(self)
        self.training_details_ = _training_details(self)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "state_sha256_")
        verify_record_prototype_identity(self, self.n_neighbors, self.epsilon)
        values = _features(X)
        result = np.empty((len(values), 7), dtype=np.float64)
        for start in range(0, len(values), DISTANCE_BATCH_SIZE):
            query = values[start:start + DISTANCE_BATCH_SIZE]
            denominators = np.empty((len(query), 7), dtype=np.float64)
            for label in range(7):
                prototypes = self.prototypes_[self.prototype_labels_ == label]
                distances = cdist(query, prototypes, metric="euclidean")
                if not np.isfinite(distances).all():
                    raise ValueError("Prototype distances exceed the finite float64 range")
                nearest = np.partition(distances, self.n_neighbors_fit_ - 1, axis=1)[:, :self.n_neighbors_fit_]
                with np.errstate(over="ignore", invalid="ignore"):
                    denominators[:, label] = np.sort(nearest, axis=1).mean(axis=1) + self.epsilon_fit_
            if not np.isfinite(denominators).all() or np.any(denominators <= 0):
                raise ValueError("Prototype score denominators must remain finite and positive")
            # A common row scale cancels in normalization and avoids reciprocal
            # overflow for an exact match with a very small positive epsilon.
            scores = denominators.min(axis=1, keepdims=True) / denominators
            result[start:start + len(query)] = scores / scores.sum(axis=1, keepdims=True)
        if not np.isfinite(result).all():
            raise ValueError("Prototype probabilities must remain finite")
        return result

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def verify_record_prototype_identity(model, n_neighbors, epsilon):
    """Reject parameter, prototype or training-identity drift without refitting."""
    k, epsilon = _parameters(n_neighbors, epsilon)
    if type(model) is not RecordPrototypeClassifier:
        raise ValueError("Expected the exact record prototype backend class")
    try:
        actual_k, actual_epsilon = _parameters(model.n_neighbors, model.epsilon)
        if (actual_k != k or actual_epsilon != epsilon or model.n_neighbors_fit_ != k
                or model.epsilon_fit_ != epsilon or type(model.n_features_in_) is not int
                or model.n_features_in_ != FEATURE_COUNT or type(model.n_records_) is not int
                or model.n_records_ < 7 or type(model.n_training_samples_) is not int
                or model.n_training_samples_ < model.n_records_):
            raise ValueError("Record prototype fitted parameters or dimensions differ")
        for name, dtype, shape in (
                ("prototypes_", np.float64, (model.n_records_, FEATURE_COUNT)),
                ("prototype_labels_", np.int64, (model.n_records_,)),
                ("class_record_counts_", np.int64, (7,)),
                ("class_counts_", np.int64, (7,)),
                ("classes_", np.int64, (7,))):
            values = getattr(model, name)
            if (not isinstance(values, np.ndarray) or values.dtype != np.dtype(dtype)
                    or values.shape != shape or not np.isfinite(values).all()):
                raise ValueError(f"Invalid record prototype state: {name}")
        if (not np.array_equal(model.classes_, np.arange(7))
                or not np.isin(model.prototype_labels_, model.classes_).all()
                or not np.array_equal(model.class_record_counts_, np.bincount(model.prototype_labels_, minlength=7))
                or np.any(model.class_record_counts_ < k)):
            raise ValueError("Record prototype classes and record counts differ")
        if (model.input_dtype_ not in ("float32", "float64")
                or not isinstance(model.input_sha256_, str) or len(model.input_sha256_) != 64
                or any(character not in "0123456789abcdef" for character in model.input_sha256_)):
            raise ValueError("Invalid record prototype training input digest")
        metadata = model.record_metadata_
        if not isinstance(metadata, list) or len(metadata) != model.n_records_:
            raise ValueError("Record prototype metadata must cover every prototype")
        y = np.full(model.n_training_samples_, -1, dtype=np.int64)
        groups = np.full(model.n_training_samples_, "", dtype=object)
        records = np.full(model.n_training_samples_, "", dtype=object)
        seen = set()
        for index, row in enumerate(metadata):
            if (not isinstance(row, dict)
                    or set(row) != {"record_id", "subject", "class_id", "windows", "row_indices"}
                    or type(row["class_id"]) is not int or row["class_id"] != model.prototype_labels_[index]
                    or type(row["windows"]) is not int or row["windows"] < 1
                    or not isinstance(row["row_indices"], list) or row["windows"] != len(row["row_indices"])):
                raise ValueError("Invalid record prototype metadata entry")
            rows = row["row_indices"]
            if (any(type(item) is not int or not 0 <= item < model.n_training_samples_ for item in rows)
                    or rows != sorted(set(rows)) or seen.intersection(rows)):
                raise ValueError("Record prototype rows must be unique and aligned")
            seen.update(rows)
            y[rows], groups[rows], records[rows] = row["class_id"], row["subject"], row["record_id"]
        if len(seen) != model.n_training_samples_:
            raise ValueError("Record prototype metadata must cover every training row")
        _, canonical = validate_training_records(y, groups, records)
        if canonical != metadata or _metadata_digest(metadata) != model.metadata_sha256_:
            raise ValueError("Record prototype metadata is not canonical or its hash differs")
        if not np.array_equal(model.class_counts_, np.bincount(y, minlength=7)):
            raise ValueError("Record prototype training class row counts differ")
        if _state_digest(model) != model.state_sha256_:
            raise ValueError("Record prototype fitted state hash differs")
        if _training_details(model) != model.training_details_:
            raise ValueError("Record prototype training details differ from fitted state")
    except (AttributeError, KeyError, IndexError, TypeError) as error:
        raise ValueError("Record prototype fitted state is incomplete or invalid") from error
