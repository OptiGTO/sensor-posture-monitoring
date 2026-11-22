"""Pure helpers for the despike / training-trim LOSO study (no I/O, no fitting).

Kept separate so the preregistered decision rules and the live-stability metric
can be unit tested with known answers before the real experiment runs.
"""
from __future__ import annotations

import numpy as np

from experiments.runtime_vs_research.analysis import macro_f1, paired_subject_bootstrap, per_class_f1  # noqa: F401

SMOOTHING_HISTORY = 3  # src.realtime.LiveConfig default used by the app


def smoothed_predictions(probability: np.ndarray, history: int = SMOOTHING_HISTORY) -> np.ndarray:
    """Argmax of the running mean over the last ``history`` rows (engine semantics)."""
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 2 or history < 1:
        raise ValueError("probability must be (N, classes) and history positive")
    cumulative = np.cumsum(np.vstack([np.zeros((1, probability.shape[1])), probability]), axis=0)
    ends = np.arange(1, len(probability) + 1)
    starts = np.maximum(0, ends - history)
    means = (cumulative[ends] - cumulative[starts]) / (ends - starts)[:, None]
    return means.argmax(axis=1)


def label_changes(predictions: np.ndarray, sessions: np.ndarray) -> int:
    """Count consecutive prediction changes inside each session (rows in time order)."""
    predictions, sessions = np.asarray(predictions), np.asarray(sessions)
    if predictions.shape != sessions.shape:
        raise ValueError("predictions and sessions must align")
    same_session = sessions[1:] == sessions[:-1]
    return int(np.sum((predictions[1:] != predictions[:-1]) & same_session))


def live_stability(probability: np.ndarray, truth: np.ndarray, sessions: np.ndarray,
                   interval_seconds: float = 1.0) -> dict:
    """Live-replay metrics; smoothing history restarts at every recording."""
    probability, truth, sessions = np.asarray(probability, float), np.asarray(truth), np.asarray(sessions)
    if not (len(probability) == len(truth) == len(sessions)) or not len(truth):
        raise ValueError("live inputs must align and be nonempty")
    boundaries = np.flatnonzero(np.r_[True, sessions[1:] != sessions[:-1]])
    if len(boundaries) != len(np.unique(sessions)):
        raise ValueError("each session's rows must be contiguous")
    raw = probability.argmax(axis=1)
    smoothed = np.concatenate([smoothed_predictions(block) for block in np.split(probability, boundaries[1:])])
    minutes = len(truth) * interval_seconds / 60.0
    raw_changes, smooth_changes = label_changes(raw, sessions), label_changes(smoothed, sessions)
    return {"predictions": int(len(truth)), "sessions": int(len(boundaries)), "prediction_minutes": minutes,
            "raw_macro_f1": macro_f1(truth, raw), "smoothed_macro_f1": macro_f1(truth, smoothed),
            "raw_changes": raw_changes, "smoothed_changes": smooth_changes,
            "raw_changes_per_minute": raw_changes / minutes, "smoothed_changes_per_minute": smooth_changes / minutes}


def decide(base_f1, arm_f1, base_class, arm_class, bootstrap_lower, base_changes, arm_changes, criteria) -> dict:
    """Apply both preregistered rules for one arm vs the baseline of one model.

    base_f1/arm_f1: pooled LOSO Macro F1 per seed. *_class: (seeds, 7) pooled per-class
    F1. bootstrap_lower: primary-seed paired subject bootstrap 2.5% bound of the
    Macro F1 difference. *_changes: smoothed live label changes per minute per seed.
    """
    base_f1, arm_f1 = np.asarray(base_f1, float), np.asarray(arm_f1, float)
    base_changes, arm_changes = np.asarray(base_changes, float), np.asarray(arm_changes, float)
    if base_f1.shape != arm_f1.shape or base_f1.ndim != 1 or base_changes.shape != base_f1.shape \
            or arm_changes.shape != base_f1.shape:
        raise ValueError("per-seed inputs must align")
    delta = arm_f1 - base_f1
    class_delta = (np.asarray(arm_class, float) - np.asarray(base_class, float)).mean(axis=0)
    no_class_drop = bool(class_delta.min() >= -criteria["maximum_per_class_drop"])
    relative_change_reduction = 1.0 - arm_changes / np.where(base_changes > 0, base_changes, np.nan)
    performance = {
        "mean_delta": bool(delta.mean() >= criteria["minimum_mean_delta"]),
        "all_seed_deltas_positive": bool((delta > 0).all()),
        "bootstrap_lower_positive": bool(bootstrap_lower > 0),
        "per_class_drop": no_class_drop,
    }
    stability = {
        "non_inferior_mean_delta": bool(delta.mean() >= -criteria["non_inferiority_mean_margin"]),
        "non_inferior_bootstrap_lower": bool(bootstrap_lower >= -criteria["non_inferiority_bootstrap_margin"]),
        "fewer_changes_every_seed": bool((arm_changes < base_changes).all()),
        "mean_relative_change_reduction": bool(np.nanmean(relative_change_reduction)
                                               >= criteria["minimum_relative_change_reduction"]),
        "per_class_drop": no_class_drop,
    }
    return {"performance_checks": performance, "performance_pass": all(performance.values()),
            "stability_checks": stability, "stability_pass": all(stability.values()),
            "seed_deltas": delta.tolist(), "mean_delta": float(delta.mean()),
            "mean_class_deltas": class_delta.tolist(), "bootstrap_lower": float(bootstrap_lower),
            "relative_change_reduction_by_seed": relative_change_reduction.tolist(),
            "mean_relative_change_reduction": float(np.nanmean(relative_change_reduction))}
