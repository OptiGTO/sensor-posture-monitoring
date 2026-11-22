"""Pure scoring and decision helpers for the finger-press calibration LOSO study.

No file I/O or model fitting, so the decision rule is unit tested with known
answers before any real result is seen.
"""
from __future__ import annotations

import numpy as np

N_CLASSES = 7


def confusion(truth, prediction, n_classes: int = N_CLASSES) -> np.ndarray:
    truth, prediction = np.asarray(truth, int), np.asarray(prediction, int)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction must be aligned")
    return np.bincount(truth * n_classes + prediction, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def class_f1_from_confusion(matrix: np.ndarray) -> np.ndarray:
    """F1 per class for (..., K, K) confusion matrices (rows = truth); 0 when undefined."""
    m = np.asarray(matrix, float)
    tp = np.diagonal(m, axis1=-2, axis2=-1)
    denominator = m.sum(-1) + m.sum(-2)
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def macro_f1_from_confusion(matrix: np.ndarray) -> np.ndarray:
    return class_f1_from_confusion(matrix).mean(-1)


def macro_f1(truth, prediction) -> float:
    return float(macro_f1_from_confusion(confusion(truth, prediction)))


def per_class_f1(truth, prediction) -> np.ndarray:
    return class_f1_from_confusion(confusion(truth, prediction))


def present_class_macro_f1(truth, prediction) -> float:
    matrix = confusion(truth, prediction)
    present = matrix.sum(1) > 0
    return float(class_f1_from_confusion(matrix)[present].mean())


def subject_confusions(truth, prediction, groups, subjects) -> np.ndarray:
    truth, prediction, groups = np.asarray(truth), np.asarray(prediction), np.asarray(groups)
    return np.stack([confusion(truth[groups == s], prediction[groups == s]) for s in subjects])


def seed_mean_bootstrap(truth, predictions_a, predictions_b, groups, *, replicates: int, seed: int,
                        comparisons: int = 1) -> dict:
    """Subject-cluster bootstrap of the seed-averaged pooled Macro-F1 difference (b - a).

    predictions_*: (seeds, windows) aligned to truth/groups. The same resampled
    subjects are used for every seed and both arms, so the interval reflects
    which people were evaluated, not model-refit noise. Bonferroni bounds use
    alpha = 0.05 / comparisons (two-sided).
    """
    a, b = np.atleast_2d(predictions_a), np.atleast_2d(predictions_b)
    truth, groups = np.asarray(truth), np.asarray(groups)
    if a.shape != b.shape or a.shape[1] != len(truth) or len(groups) != len(truth) or not len(truth):
        raise ValueError("Bootstrap inputs must be aligned and nonempty")
    if comparisons < 1:
        raise ValueError("comparisons must be >= 1")
    subjects = np.unique(groups)
    ca = np.stack([subject_confusions(truth, p, groups, subjects) for p in a]).reshape(len(a), len(subjects), -1)
    cb = np.stack([subject_confusions(truth, p, groups, subjects) for p in b]).reshape(len(b), len(subjects), -1)
    weights = np.random.default_rng(seed).multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), replicates)
    shape = (replicates, N_CLASSES, N_CLASSES)
    f1a = np.stack([macro_f1_from_confusion((weights @ ca[k]).reshape(shape)) for k in range(len(a))])
    f1b = np.stack([macro_f1_from_confusion((weights @ cb[k]).reshape(shape)) for k in range(len(b))])
    deltas = (f1b - f1a).mean(0)
    observed = float(np.mean([macro_f1(truth, pb) - macro_f1(truth, pa) for pa, pb in zip(a, b)]))
    tail = 100 * 0.05 / comparisons / 2
    return {"observed": observed,
            "lower_2_5": float(np.percentile(deltas, 2.5)), "upper_97_5": float(np.percentile(deltas, 97.5)),
            "bonferroni_lower": float(np.percentile(deltas, tail)),
            "bonferroni_upper": float(np.percentile(deltas, 100 - tail)),
            "bonferroni_comparisons": int(comparisons), "replicates": int(replicates), "seed": int(seed),
            "subjects": int(len(subjects)), "seeds": int(len(a))}


def decide(f1_none, f1_calibrated, class_f1_none, class_f1_calibrated, bootstrap: dict, criteria: dict) -> dict:
    """Preregistered rule for one (calibration mode, model family) comparison.

    f1_*: pooled LOSO Macro F1 per seed, shape (seeds,). class_f1_*: (seeds, 7).
    """
    a, b = np.asarray(f1_none, float), np.asarray(f1_calibrated, float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("Per-seed scores must be aligned 1-D arrays")
    deltas = b - a
    class_delta = (np.asarray(class_f1_calibrated, float) - np.asarray(class_f1_none, float)).mean(0)
    checks = {
        "mean_delta": bool(deltas.mean() >= criteria["minimum_mean_delta"]),
        "all_seeds_positive": bool((deltas > 0).all()),
        "bonferroni_lower_positive": bool(bootstrap["bonferroni_lower"] > 0),
        "per_class_drop": bool(class_delta.min() >= -criteria["maximum_per_class_drop"]),
    }
    return {"checks": checks, "significant": all(checks.values()), "deltas_by_seed": deltas.tolist(),
            "mean_delta": float(deltas.mean()), "class_deltas": class_delta.tolist(),
            "worst_class_delta": float(class_delta.min())}
