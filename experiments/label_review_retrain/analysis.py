"""Pure helpers for the label-review retrain study (no I/O, no fitting).

Turns the structured human review into window masks, applies the preregistered
decision rule and computes a fast subject-cluster bootstrap. Everything here is
unit tested with known answers in ``test_analysis.py`` before the real run.
"""
from __future__ import annotations

import numpy as np

from experiments.runtime_vs_research.analysis import N_CLASSES, macro_f1, per_class_f1  # noqa: F401

WINDOW_SECONDS = 3.0
TOLERANCE = 1e-6
DECISIONS = ("delete", "hold", "posture_mismatch")

# Training conditions: which rows may be fitted. Evaluation rows are never changed
# by a condition; every held-out window is predicted in every condition.
CONDITIONS = ("baseline", "drop_delete", "drop_invalid_segments", "drop_delete_mismatch")


def _end(value) -> float:
    return np.inf if value is None else float(value)


def window_is_valid(start_s: float, decision: str, normal_intervals, abnormal_intervals,
                    window_seconds: float = WINDOW_SECONDS) -> bool:
    """Is a nominal window [start, start + window) trustworthy under the review?

    - If the reviewer stated normal intervals, the window must lie fully inside one.
    - Otherwise a deleted recording has no valid window; other decisions keep it.
    - A window overlapping any stated abnormal interval (half-open) is invalid.
    """
    if decision not in DECISIONS:
        raise ValueError(f"Unknown decision: {decision}")
    start, end = float(start_s), float(start_s) + window_seconds
    if normal_intervals:
        valid = any(start >= float(a) - TOLERANCE and end <= _end(b) + TOLERANCE for a, b in normal_intervals)
    else:
        valid = decision != "delete"
    overlaps = any(start < _end(b) - TOLERANCE and end > float(a) + TOLERANCE for a, b in abnormal_intervals)
    return bool(valid and not overlaps)


def review_masks(record_ids, start_s, decisions: dict) -> dict:
    """Window-level boolean masks. ``decisions`` maps record_id -> structured decision."""
    record_ids, start_s = np.asarray(record_ids), np.asarray(start_s, dtype=float)
    if record_ids.shape != start_s.shape:
        raise ValueError("record_ids and start_s must align")
    reviewed = np.array([r in decisions for r in record_ids])
    delete = np.array([decisions.get(r, {}).get("decision") == "delete" for r in record_ids])
    mismatch = np.array([decisions.get(r, {}).get("decision") == "posture_mismatch" for r in record_ids])
    invalid = np.array([r in decisions and not window_is_valid(s, decisions[r]["decision"],
                                                                decisions[r]["normal_intervals"],
                                                                decisions[r]["abnormal_intervals"])
                        for r, s in zip(record_ids, start_s)])
    return {"reviewed": reviewed, "delete_record": delete, "mismatch_record": mismatch, "invalid_window": invalid}


def condition_fit_masks(masks: dict) -> dict:
    return {
        "baseline": np.ones_like(masks["reviewed"]),
        "drop_delete": ~masks["delete_record"],
        "drop_invalid_segments": ~masks["invalid_window"],
        "drop_delete_mismatch": ~(masks["delete_record"] | masks["mismatch_record"]),
    }


def evaluation_masks(masks: dict) -> dict:
    return {
        "all": np.ones_like(masks["reviewed"]),
        "valid": ~masks["invalid_window"],
        "valid_no_mismatch": ~masks["invalid_window"] & ~masks["mismatch_record"],
        "unreviewed": ~masks["reviewed"],
    }


def confusion_by_group(truth, prediction, groups):
    """(subjects, per-subject 7x7 confusion counts) in sorted subject order."""
    truth, prediction, groups = np.asarray(truth), np.asarray(prediction), np.asarray(groups)
    subjects = np.unique(groups)
    matrices = np.zeros((len(subjects), N_CLASSES, N_CLASSES), dtype=np.int64)
    index = np.searchsorted(subjects, groups)
    np.add.at(matrices, (index, truth, prediction), 1)
    return subjects, matrices


def macro_f1_from_confusion(matrix) -> np.ndarray:
    """Macro F1 over all 7 labels (zero_division=0), for one (7,7) or a stack (..., 7, 7)."""
    matrix = np.asarray(matrix, dtype=float)
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return f1.mean(axis=-1)


def paired_subject_bootstrap(truth, pred_a, pred_b, groups, *, replicates: int, seed: int) -> dict:
    """Macro-F1 difference (b - a) with subjects resampled as clusters.

    Same resampling as experiments.runtime_vs_research.analysis.paired_subject_bootstrap
    (a subject drawn k times contributes its windows k times), computed from summed
    per-subject confusion matrices instead of re-scoring each replicate.
    """
    truth, pred_a, pred_b, groups = map(np.asarray, (truth, pred_a, pred_b, groups))
    if not (len(truth) == len(pred_a) == len(pred_b) == len(groups)) or not len(truth):
        raise ValueError("Bootstrap inputs must be aligned and nonempty")
    subjects, conf_a = confusion_by_group(truth, pred_a, groups)
    _, conf_b = confusion_by_group(truth, pred_b, groups)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(subjects), (replicates, len(subjects)))
    weights = np.stack([np.bincount(row, minlength=len(subjects)) for row in draws]).astype(float)
    sum_a = np.einsum("rs,sij->rij", weights, conf_a)
    sum_b = np.einsum("rs,sij->rij", weights, conf_b)
    deltas = macro_f1_from_confusion(sum_b) - macro_f1_from_confusion(sum_a)
    return {"observed": macro_f1(truth, pred_b) - macro_f1(truth, pred_a),
            "lower_2_5": float(np.percentile(deltas, 2.5)), "upper_97_5": float(np.percentile(deltas, 97.5)),
            "replicates": replicates, "seed": seed, "subjects": int(len(subjects))}


def present_macro_f1(truth, prediction) -> float:
    """Macro F1 over the labels present in ``truth`` (per-subject summaries)."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    labels = np.unique(truth)
    matrix = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(matrix, (truth, prediction), 1)
    tp = np.diag(matrix)[labels]
    denominator = matrix.sum(axis=1)[labels] + matrix.sum(axis=0)[labels]
    return float(np.mean(np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)))


def smoothed_by_session(probability, sessions, history: int = 3) -> np.ndarray:
    """App smoothing (mean of last ``history`` probabilities) restarted at every session."""
    from experiments.despike_study.analysis import smoothed_predictions
    probability, sessions = np.asarray(probability, float), np.asarray(sessions)
    if len(probability) != len(sessions) or not len(sessions):
        raise ValueError("probability and sessions must align and be nonempty")
    boundaries = np.flatnonzero(np.r_[True, sessions[1:] != sessions[:-1]])
    if len(boundaries) != len(np.unique(sessions)):
        raise ValueError("each session's rows must be contiguous")
    return np.concatenate([smoothed_predictions(block, history) for block in np.split(probability, boundaries[1:])])


def record_accuracy(probability, truth, records) -> dict:
    """Per-record window accuracy (argmax) for the given rows."""
    probability, truth, records = np.asarray(probability), np.asarray(truth), np.asarray(records)
    correct = probability.argmax(axis=1) == truth
    return {str(r): float(correct[records == r].mean()) for r in np.unique(records)}


def decide(base_f1, arm_f1, base_class, arm_class, bootstrap_lower, criteria) -> dict:
    """Preregistered performance rule and the unreviewed-set non-inferiority check.

    base_f1/arm_f1: pooled LOSO Macro F1 per seed; *_class: (seeds, 7) per-class F1.
    """
    base_f1, arm_f1 = np.asarray(base_f1, float), np.asarray(arm_f1, float)
    if base_f1.shape != arm_f1.shape or base_f1.ndim != 1 or not len(base_f1):
        raise ValueError("per-seed inputs must align")
    delta = arm_f1 - base_f1
    class_delta = (np.asarray(arm_class, float) - np.asarray(base_class, float)).mean(axis=0)
    no_class_drop = bool(class_delta.min() >= -criteria["maximum_per_class_drop"])
    performance = {
        "mean_delta": bool(delta.mean() >= criteria["minimum_mean_delta"]),
        "all_seed_deltas_positive": bool((delta > 0).all()),
        "bootstrap_lower_positive": bool(bootstrap_lower > 0),
        "per_class_drop": no_class_drop,
    }
    non_inferiority = {
        "mean_delta": bool(delta.mean() >= -criteria["non_inferiority_mean_margin"]),
        "bootstrap_lower": bool(bootstrap_lower >= -criteria["non_inferiority_bootstrap_margin"]),
        "per_class_drop": no_class_drop,
    }
    return {"performance_checks": performance, "performance_pass": all(performance.values()),
            "non_inferiority_checks": non_inferiority, "non_inferior": all(non_inferiority.values()),
            "seed_deltas": delta.tolist(), "mean_delta": float(delta.mean()),
            "mean_class_deltas": class_delta.tolist(), "bootstrap_lower": float(bootstrap_lower)}
