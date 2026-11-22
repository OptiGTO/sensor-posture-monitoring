"""Keep recording identities aligned with the immutable training augmentation.

This module does not prepare caches or change the selected training rows. It
extends the original helper's return value for estimators that need recording
boundaries while retaining its subject and evaluation-overlap guards.
"""
from __future__ import annotations

import numpy as np

from . import supplemental as base
from src.posture_core import normalize_text


def validate_record_identities(y, groups, record_ids):
    """Require each normalized recording identity to have one subject and label.

    Original identity strings and their order are left unchanged. The same
    validation can be applied to ordinary, nonaugmented training rows.
    """
    labels, subjects, records = map(np.asarray, (y, groups, record_ids))
    if (labels.ndim != 1 or not len(labels)
            or any(values.ndim != 1 or len(values) != len(labels)
                   for values in (subjects, records))):
        raise ValueError("Recording identities, subjects and labels must align")
    if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, np.arange(7)).all():
        raise ValueError("Recording identities require integer posture labels 0..6")
    ownership = {}
    for label, group, record in zip(labels, subjects, records):
        person, identity = normalize_text(str(group)), normalize_text(str(record))
        if not person or not identity:
            raise ValueError("Empty subject or recording identity")
        previous = ownership.setdefault(identity, (person, int(label)))
        if previous[0] != person:
            raise ValueError("A recording cannot belong to multiple subjects")
        if previous[1] != int(label):
            raise ValueError("A recording cannot contain multiple posture labels")


def append_record_training(data, supplement, train_indices):
    """Return (X, y, groups, record_ids, audit) for identical augmented rows.

    Base rows retain the caller's index order. Supplemental rows retain their
    cache order, including when subjects or recordings are interleaved. All
    arrays are returned only after the original augmentation and recording
    identity checks succeed; this function never writes files or input arrays.
    """
    features, labels, groups, audit = base.append_training(data, supplement, train_indices)
    train = np.asarray(train_indices)
    extra_groups = np.asarray([normalize_text(str(group)) for group in supplement["groups"]])
    take = np.flatnonzero(np.isin(extra_groups, audit["training_subjects"]))
    record_ids = np.concatenate((np.asarray(data["record_ids"])[train],
                                 np.asarray(supplement["record_ids"])[take]))
    if (len(record_ids) != len(features) or len(record_ids) != len(labels)
            or len(record_ids) != len(groups)
            or len(train) != audit["base_training_windows"]
            or len(take) != audit["added_windows"]):
        raise ValueError("Augmented recording rows differ from feature training selection")
    expected_labels = np.concatenate((np.asarray(data["y"])[train], np.asarray(supplement["y"])[take]))
    expected_groups = np.concatenate((np.asarray(data["groups"])[train], np.asarray(supplement["groups"])[take]))
    if not np.array_equal(labels, expected_labels) or not np.array_equal(groups, expected_groups):
        raise ValueError("Augmented recording order differs from feature training selection")
    validate_record_identities(labels, groups, record_ids)
    return features, labels, groups, record_ids, audit
