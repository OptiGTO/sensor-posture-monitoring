"""Keep restored raw sequences and recording IDs on the same training rows."""
from __future__ import annotations

import numpy as np

from .supplemental_sequence import append_sequence_training
from src.posture_core import normalize_text
from src.record_gain_features_v1 import validate_training_records


def append_record_sequence_training(data, supplement, train_indices):
    """Return (sequence, y, groups, record_ids, audit) without changing inputs.

    The frozen sequence helper retains the original eligibility, complete
    training-subject, evaluation overlap and supplemental feature-parity guards.
    Recording identities preserve exactly the returned row order.
    """
    sequence, labels, groups, audit = append_sequence_training(data, supplement, train_indices)
    train = np.asarray(train_indices)
    normalized_extra = np.asarray([normalize_text(str(value)) for value in supplement["groups"]])
    take = np.flatnonzero(np.isin(normalized_extra, audit["training_subjects"]))
    records = np.concatenate((np.asarray(data["record_ids"])[train], np.asarray(supplement["record_ids"])[take]))
    expected_labels = np.concatenate((np.asarray(data["y"])[train], np.asarray(supplement["y"])[take]))
    expected_groups = np.concatenate((np.asarray(data["groups"])[train], np.asarray(supplement["groups"])[take]))
    if (sequence.shape != (len(records), 36, 16) or len(labels) != len(records) or len(groups) != len(records)
            or len(train) != audit["base_training_windows"] or len(take) != audit["added_windows"]
            or not np.array_equal(labels, expected_labels) or not np.array_equal(groups, expected_groups)):
        raise ValueError("Restored sequence and recording rows differ from the training selection")
    validate_training_records(labels, groups, records)
    return sequence, labels, groups, records, audit
