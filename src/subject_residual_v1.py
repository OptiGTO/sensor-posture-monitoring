"""Remove three TRAIN between-person centroid directions, retaining 596 coordinates.

The caller owns StandardScaler. Person centroids equally weight postures and
recordings; inference needs only a window's features, never a person's identity.
These directions are not claimed to be pure nuisance or independent observations.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import unicodedata

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .posture_core import feature_names


VERSION = "subject_residual_v1"
MODES = frozenset({"identity", "remove_subject3"})
FEATURE_COUNT = 596
ORTHOGONALITY_ATOL = 1e-12
LEARNED_ARRAYS = (
    "record_means_", "person_class_means_", "person_centroids_", "centroid_mean_",
    "centered_centroids_", "singular_values_", "components_", "signs_",
)


def _mode(mode):
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("Unknown subject residual mode")
    return mode


def _input(X):
    if hasattr(X, "columns") and list(X.columns) != feature_names():
        raise ValueError("Subject residual input names differ from canonical596 feature order")
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] != FEATURE_COUNT or not len(values)
            or values.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or not np.isfinite(values).all()):
        raise ValueError("Expected finite nonempty float32/float64 input with shape (N,596)")
    return values


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _array_digest(array):
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256(json.dumps([values.dtype.str, list(values.shape)]).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _identities(values, rows, name):
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != rows:
        raise ValueError(f"Aligned training {name} are required")
    result = []
    for value in array:
        if not isinstance(value, (str, np.str_)):
            raise ValueError(f"Training {name} must be nonempty string identities")
        canonical = unicodedata.normalize("NFC", str(value)).strip()
        if not canonical:
            raise ValueError(f"Training {name} must be nonempty string identities")
        result.append(canonical)
    return np.asarray(result)


def _scope(y, groups, record_ids, rows):
    labels = np.asarray(y)
    if (labels.ndim != 1 or len(labels) != rows or labels.dtype.kind not in "iu"
            or not np.array_equal(np.unique(labels), np.arange(7))):
        raise ValueError("Aligned integer labels must contain all seven posture IDs 0..6")
    labels = labels.astype(np.int64, copy=True)
    people = _identities(groups, rows, "groups")
    records = _identities(record_ids, rows, "record IDs")
    subjects = sorted(set(people.tolist()))
    if len(subjects) < 4:
        raise ValueError("Subject residual fit requires at least four training people")
    metadata = []
    for record in sorted(set(records.tolist())):
        indices = np.flatnonzero(records == record)
        if len(np.unique(people[indices])) != 1 or len(np.unique(labels[indices])) != 1:
            raise ValueError("Each training record must have exactly one subject and one posture")
        metadata.append({"record_id": record, "subject": str(people[indices[0]]),
            "class_id": int(labels[indices[0]]), "windows": len(indices), "row_indices": indices.tolist()})
    for person in subjects:
        if not np.array_equal(np.unique(labels[people == person]), np.arange(7)):
            raise ValueError("Every training person must have all seven postures")
    return labels, people, records, subjects, metadata


def _mean(values):
    # A fixed per-coordinate reduction order makes row permutations and record
    # lengths irrelevant to the equal-record aggregation, apart from row means.
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.sort(np.asarray(values, dtype=np.float64), axis=0).mean(axis=0, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Training centroid arithmetic exceeded finite float64 range")
    return result


def _hierarchy(record_means, metadata, subjects):
    by_person_class = np.empty((len(subjects), 7, FEATURE_COUNT), dtype=np.float64)
    for person_index, person in enumerate(subjects):
        for label in range(7):
            chosen = [i for i, row in enumerate(metadata)
                      if row["subject"] == person and row["class_id"] == label]
            by_person_class[person_index, label] = _mean(record_means[chosen])
    centroids = np.ascontiguousarray([_mean(values) for values in by_person_class])
    center = _mean(centroids)
    return by_person_class, centroids, center, np.ascontiguousarray(centroids - center)


def subject_residual_rule(mode):
    mode = _mode(mode)
    rule = {"schema_version": 1, "version": VERSION, "mode": mode,
        "input_features": FEATURE_COUNT, "output_features": FEATURE_COUNT,
        "input_dtypes": ["float32", "float64"],
        "output_dtype": "input dtype unchanged" if mode == "identity" else "float64",
        "scaler_owner": "caller fits StandardScaler on original TRAIN rows; no scaler after this helper",
        "fit_scope": "supplied TRAIN rows only; aligned groups and record IDs required",
        "identity_normalization": "Unicode NFC then strip; nonempty strings only",
        "required_scope": "at least4 people, each with all7 postures; one subject and posture per record",
        "aggregation": "float64 per-feature ascending-value arithmetic means: windows within record, equal records within person/posture, equal7 postures within person, equal people to center B",
        "solver": None if mode == "identity" else "numpy.linalg.svd(B, full_matrices=False), float64, threadpool limit1",
        "removed_rank": 0 if mode == "identity" else 3,
        "rank_tolerance": "eps(float64)*max(number of training people,596)*largest singular value",
        "rank_and_gap": "require s[2]>tolerance and s[2]-s[3]>tolerance; missing fourth value is0; no fallback or dimension tuning",
        "axis_sign": "largest absolute loading positive; first coordinate breaks magnitude ties",
        "transform": "validated ndarray object passthrough" if mode == "identity" else
            "C-order float64 z; a=einsum('ni,ji->nj',z,V,optimize=False); z-einsum('nj,ji->ni',a,V,optimize=False)",
        "centering": "center person centroids only to learn B; no subtraction of centroid mean in inference",
        "component_layout": "C-contiguous float64 (3,596)",
        "orthogonality_atol": ORTHOGONALITY_ATOL,
        "posture_labels_used": "scope audit only" if mode == "identity" else "equal posture contributions within each training person",
        "external_validation_used": False, "evaluation_batch_statistics": False,
        "inference_subject_ids_required": False, "inference_record_ids_required": False,
        "interpretation": "between-person centroid directions; no claim of pure nuisance, complete information retention or causal/generalization benefit",
        "training_diagnostics": "descriptive only; never used to choose rank or tune a coefficient"}
    rule["rule_sha256"] = _json_digest(rule)
    return rule


def _diagnostics(model):
    if model.mode_ == "identity":
        return None
    V, singular = model.components_, model.singular_values_
    # Each person and each recording have equal influence in the class means.
    class_means = np.ascontiguousarray([_mean(model.person_class_means_[:, c]) for c in range(7)])
    contrast = class_means - _mean(class_means)
    contrast_removed = np.einsum("ni,ji->nj", contrast, V, optimize=False)
    energy = float(np.square(contrast).sum())
    pairs = []
    for left in range(7):
        for right in range(left + 1, 7):
            difference = class_means[left] - class_means[right]
            projection = np.einsum("i,ji->j", difference, V, optimize=False)
            after = difference - np.einsum("j,ji->i", projection, V, optimize=False)
            before_sq, after_sq = float(np.square(difference).sum()), float(np.square(after).sum())
            pairs.append({"classes": [left, right], "squared_distance_before": before_sq,
                "squared_distance_after": after_sq,
                "fraction_removed": None if before_sq == 0 else (before_sq - after_sq) / before_sq})
    result = {"subject_centroid_energy_fraction_removed": float(np.square(singular[:3]).sum() / np.square(singular).sum()),
        "class_centroid_contrast_energy": energy,
        "class_centroid_contrast_energy_fraction_removed": None if energy == 0 else float(np.square(contrast_removed).sum() / energy),
        "class_centroid_pair_distances": pairs, "third_singular_value": float(singular[2]),
        "fourth_singular_value": float(singular[3]) if len(singular) > 3 else 0.,
        "third_fourth_gap": model.singular_gap_, "rank_tolerance": model.rank_tolerance_,
        "not_used_for_selection": True}
    # Refuse non-finite diagnostics rather than hiding invalid learned state.
    _json_digest(result)
    return result


def _state(model):
    return {"version": VERSION, "mode": model.mode_, "rows": model.n_samples_seen_,
        "features": model.n_features_in_, "output_features": model.n_features_out_,
        "training_dtype": model.training_dtype_, "subjects": model.training_subjects_,
        "record_metadata": model.record_metadata_, "training_input_sha256": model.training_input_sha256_,
        "training_labels_sha256": model.training_labels_sha256_, "training_groups_sha256": model.training_groups_sha256_,
        "training_records_sha256": model.training_records_sha256_, "class_counts": _array_digest(model.class_counts_),
        "classes": _array_digest(model.classes_), "feature_names": model.feature_names_in_.tolist(),
        "rank_tolerance": model.rank_tolerance_, "singular_gap": model.singular_gap_,
        "rule": model.rule_, "learned_arrays": {name: None if getattr(model, name) is None else
            _array_digest(getattr(model, name)) for name in LEARNED_ARRAYS}}


def subject_residual_state_sha256(model):
    if type(model) is not SubjectResidualV1:
        raise ValueError("Expected exact SubjectResidualV1 type")
    try:
        return _json_digest(_state(model))
    except (AttributeError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Incomplete subject residual state") from error


def _audit(model):
    state = _state(model)
    return {"version": VERSION, "mode": model.mode_, "training_windows": model.n_samples_seen_,
        "training_dtype": model.training_dtype_, "input_features": FEATURE_COUNT, "output_features": FEATURE_COUNT,
        "training_input_sha256": model.training_input_sha256_, "training_labels_sha256": model.training_labels_sha256_,
        "training_groups_sha256": model.training_groups_sha256_, "training_records_sha256": model.training_records_sha256_,
        "training_subjects": list(model.training_subjects_), "training_records": len(model.record_metadata_),
        "record_metadata": deepcopy(model.record_metadata_), "classes": model.classes_.tolist(),
        "class_counts": model.class_counts_.tolist(), "feature_names": model.feature_names_in_.tolist(),
        "learned_array_hashes": state["learned_arrays"], "rank_tolerance": model.rank_tolerance_,
        "singular_gap": model.singular_gap_, "diagnostics": _diagnostics(model),
        "rule": deepcopy(model.rule_), "state_sha256": _json_digest(state),
        "external_validation_used": False, "evaluation_batch_statistics": False,
        "threads": 1, "numpy_version": np.__version__, "sklearn_version": sklearn.__version__}


class SubjectResidualV1(TransformerMixin, BaseEstimator):
    def __init__(self, mode="identity"):
        self.mode = mode

    def fit(self, X, y, *, groups, record_ids):
        mode, values = _mode(self.mode), _input(X)
        labels, people, records, subjects, metadata = _scope(y, groups, record_ids, len(values))
        learned = {name: None for name in LEARNED_ARRAYS}
        tolerance, gap = None, None
        if mode != "identity":
            record_means = np.ascontiguousarray([_mean(values[row["row_indices"]]) for row in metadata])
            person_class, centroids, center, B = _hierarchy(record_means, metadata, subjects)
            with threadpool_limits(limits=1):
                try:
                    _, singular, right = np.linalg.svd(B, full_matrices=False)
                except np.linalg.LinAlgError as error:
                    raise ValueError("TRAIN person-centroid SVD did not converge; no fallback") from error
            if not np.isfinite(singular).all() or not np.isfinite(right).all():
                raise ValueError("TRAIN person-centroid SVD produced non-finite state")
            tolerance = float(np.finfo(np.float64).eps * max(len(subjects), FEATURE_COUNT) * singular[0])
            gap = float(singular[2] - (singular[3] if len(singular) > 3 else 0.))
            if singular[2] <= tolerance:
                raise ValueError("TRAIN person-centroid numerical rank is below3; no fallback")
            if gap <= tolerance:
                raise ValueError("TRAIN third/fourth singular value boundary is tied; no fallback")
            raw = right[:3]
            signs = np.where(raw[np.arange(3), np.argmax(np.abs(raw), axis=1)] < 0., -1., 1.)
            learned = dict(zip(LEARNED_ARRAYS, (record_means, person_class, centroids, center, B,
                np.ascontiguousarray(singular), np.ascontiguousarray(raw * signs[:, None]), signs)))
        self.mode_, self.n_samples_seen_, self.n_features_in_, self.n_features_out_ = mode, len(values), FEATURE_COUNT, FEATURE_COUNT
        self.training_dtype_, self.training_subjects_, self.record_metadata_ = str(values.dtype), subjects, metadata
        self.training_input_sha256_, self.training_labels_sha256_ = _array_digest(values), _array_digest(labels)
        self.training_groups_sha256_, self.training_records_sha256_ = _json_digest(people.tolist()), _json_digest(records.tolist())
        self.classes_, self.class_counts_ = np.arange(7, dtype=np.int64), np.bincount(labels, minlength=7).astype(np.int64)
        self.feature_names_in_ = np.asarray(feature_names())
        self.feature_names_out_ = self.feature_names_in_.copy()
        self.rank_tolerance_, self.singular_gap_ = tolerance, gap
        for name, value in learned.items():
            setattr(self, name, value)
        self.rule_ = subject_residual_rule(mode)
        self.state_sha256_ = subject_residual_state_sha256(self)
        self.training_details_ = _audit(self)
        verify_subject_residual(self, mode)
        return self

    def transform(self, X):
        _inference_guard(self)
        values = _input(X)
        if self.mode_ == "identity":
            return values
        numeric = np.array(values, dtype=np.float64, order="C", copy=True)
        with np.errstate(over="ignore", invalid="ignore"):
            coordinates = np.einsum("ni,ji->nj", numeric, self.components_, optimize=False)
            result = numeric - np.einsum("nj,ji->ni", coordinates, self.components_, optimize=False)
        if not np.isfinite(result).all():
            raise ValueError("Subject residual transform exceeded finite float64 range")
        return result

    def get_feature_names_out(self, input_features=None):
        _inference_guard(self)
        if input_features is not None and not np.array_equal(np.asarray(input_features), self.feature_names_in_):
            raise ValueError("Input feature names differ from canonical596 order")
        return self.feature_names_out_.copy()


def _inference_guard(model):
    check_is_fitted(model, "mode_")
    try:
        if (type(model) is not SubjectResidualV1 or _mode(model.mode) != model.mode_
                or model.n_features_in_ != FEATURE_COUNT or model.n_features_out_ != FEATURE_COUNT
                or not np.array_equal(model.feature_names_in_, feature_names())
                or not np.array_equal(model.feature_names_out_, model.feature_names_in_)):
            raise ValueError("Subject residual fitted mode or feature schema changed")
        if model.mode_ == "identity":
            if model.components_ is not None:
                raise ValueError("Identity residual acquired learned directions")
        else:
            V = model.components_
            if (not isinstance(V, np.ndarray) or V.shape != (3, FEATURE_COUNT)
                    or V.dtype != np.dtype(np.float64) or not V.flags.c_contiguous or not np.isfinite(V).all()):
                raise ValueError("Subject residual components require finite C-order float64 (3,596)")
    except AttributeError as error:
        raise ValueError("Incomplete subject residual inference state") from error


def _reconstruct_scope(model):
    rows = model.n_samples_seen_
    y, people, records = np.full(rows, -1, dtype=np.int64), np.full(rows, "", dtype=object), np.full(rows, "", dtype=object)
    covered = np.zeros(rows, dtype=np.int64)
    if not isinstance(model.record_metadata_, list):
        raise ValueError("Invalid subject residual record metadata")
    for row in model.record_metadata_:
        if (not isinstance(row, dict) or set(row) != {"record_id", "subject", "class_id", "windows", "row_indices"}
                or type(row["class_id"]) is not int or not 0 <= row["class_id"] < 7
                or type(row["windows"]) is not int or row["windows"] < 1
                or not isinstance(row["row_indices"], list) or len(row["row_indices"]) != row["windows"]
                or any(type(index) is not int or not 0 <= index < rows for index in row["row_indices"])):
            raise ValueError("Invalid subject residual record metadata entry")
        indices = np.asarray(row["row_indices"], dtype=np.int64)
        np.add.at(covered, indices, 1)
        y[indices], people[indices], records[indices] = row["class_id"], row["subject"], row["record_id"]
    if not np.all(covered == 1):
        raise ValueError("Subject residual record metadata must cover every TRAIN row exactly once")
    labels, people, records, subjects, metadata = _scope(y, people, records, rows)
    if metadata != model.record_metadata_ or subjects != model.training_subjects_:
        raise ValueError("Subject residual canonical record/subject scope changed")
    if (not np.array_equal(np.bincount(labels, minlength=7), model.class_counts_)
            or _array_digest(labels) != model.training_labels_sha256_
            or _json_digest(people.tolist()) != model.training_groups_sha256_
            or _json_digest(records.tolist()) != model.training_records_sha256_):
        raise ValueError("Subject residual TRAIN label/group/record identity changed")


def verify_subject_residual(model, mode):
    """Validate fitted numerical state and its TRAIN audit without fitting a model."""
    mode = _mode(mode)
    if type(model) is not SubjectResidualV1:
        raise ValueError("Expected exact SubjectResidualV1 type")
    try:
        _inference_guard(model)
        if (model.mode_ != mode or type(model.n_samples_seen_) is not int or model.n_samples_seen_ < 28
                or model.training_dtype_ not in {"float32", "float64"}
                or model.rule_ != subject_residual_rule(mode)):
            raise ValueError("Subject residual rule or TRAIN dimensions changed")
        for name in ("classes_", "class_counts_"):
            array = getattr(model, name)
            if not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.int64) or array.shape != (7,):
                raise ValueError("Invalid subject residual class schema")
        if not np.array_equal(model.classes_, np.arange(7)):
            raise ValueError("Subject residual classes changed")
        for name in ("training_input_sha256_", "training_labels_sha256_", "training_groups_sha256_", "training_records_sha256_"):
            value = getattr(model, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid subject residual training identity digest")
        _reconstruct_scope(model)
        if mode == "identity":
            if any(getattr(model, name) is not None for name in LEARNED_ARRAYS) or model.rank_tolerance_ is not None or model.singular_gap_ is not None:
                raise ValueError("Identity subject residual must not learn numerical coordinates")
        else:
            subjects, records = len(model.training_subjects_), len(model.record_metadata_)
            shapes = ((records, FEATURE_COUNT), (subjects, 7, FEATURE_COUNT), (subjects, FEATURE_COUNT),
                (FEATURE_COUNT,), (subjects, FEATURE_COUNT), (min(subjects, FEATURE_COUNT),), (3, FEATURE_COUNT), (3,))
            for name, shape in zip(LEARNED_ARRAYS, shapes):
                value = getattr(model, name)
                if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float64)
                        or value.shape != shape or not value.flags.c_contiguous or not np.isfinite(value).all()):
                    raise ValueError(f"Invalid finite C-order learned state: {name}")
            hierarchy = _hierarchy(model.record_means_, model.record_metadata_, model.training_subjects_)
            for name, expected in zip(("person_class_means_", "person_centroids_", "centroid_mean_", "centered_centroids_"), hierarchy):
                if not np.array_equal(getattr(model, name), expected):
                    raise ValueError("Subject residual centroid aggregation changed")
            s, V = model.singular_values_, model.components_
            tolerance = float(np.finfo(np.float64).eps * max(subjects, FEATURE_COUNT) * s[0])
            gap = float(s[2] - (s[3] if len(s) > 3 else 0.))
            if (np.any(s < 0) or np.any(np.diff(s) > 0) or s[2] <= tolerance or gap <= tolerance
                    or model.rank_tolerance_ != tolerance or model.singular_gap_ != gap):
                raise ValueError("Subject residual numerical rank or boundary gap changed")
            gram = np.einsum("ji,ki->jk", V, V, optimize=False)
            if (not np.allclose(gram, np.eye(3), rtol=0, atol=ORTHOGONALITY_ATOL)
                    or np.any(V[np.arange(3), np.argmax(np.abs(V), axis=1)] < 0)
                    or not np.isin(model.signs_, (-1., 1.)).all()):
                raise ValueError("Subject residual component orthogonality or canonical signs changed")
            B = model.centered_centroids_
            projected = np.einsum("ni,ji->nj", B, V, optimize=False)
            eigen_action = np.einsum("nj,ni->ji", projected, B, optimize=False)
            if (not np.allclose(eigen_action, np.square(s[:3])[:, None] * V, rtol=1e-10,
                    atol=np.finfo(np.float64).eps * max(B.shape) * s[0] ** 2)
                    or not np.isclose(np.square(B).sum(), np.square(s).sum(), rtol=1e-12, atol=0)):
                raise ValueError("Subject residual singular coordinates do not match TRAIN centroids")
        if model.state_sha256_ != subject_residual_state_sha256(model) or model.training_details_ != _audit(model):
            raise ValueError("Subject residual full fitted state or audit changed")
    except (AttributeError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Incomplete subject residual fitted state") from error
    return True
