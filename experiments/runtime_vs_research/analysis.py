"""Pure helpers for the runtime RBF-SVM v5 vs research ET comparison.

No file I/O and no model fitting here, so the decision logic can be unit tested
with known answers before the real experiment is run.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

N_CLASSES = 7


def macro_f1(truth, prediction) -> float:
    return float(f1_score(truth, prediction, labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def per_class_f1(truth, prediction) -> np.ndarray:
    return f1_score(truth, prediction, labels=np.arange(N_CLASSES), average=None, zero_division=0)


def paired_subject_bootstrap(truth, pred_a, pred_b, groups, *, replicates: int, seed: int) -> dict:
    """Macro-F1 difference (b - a) with subjects resampled as clusters.

    Both predictions must be aligned to the same rows. A subject drawn k times
    contributes all of its windows k times. Descriptive interval only.
    """
    truth, pred_a, pred_b, groups = map(np.asarray, (truth, pred_a, pred_b, groups))
    if not (len(truth) == len(pred_a) == len(pred_b) == len(groups)) or not len(truth):
        raise ValueError("Bootstrap inputs must be aligned and nonempty")
    subjects = np.unique(groups)
    rows = [np.flatnonzero(groups == s) for s in subjects]
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates)
    for i in range(replicates):
        take = np.concatenate([rows[j] for j in rng.integers(0, len(subjects), len(subjects))])
        deltas[i] = macro_f1(truth[take], pred_b[take]) - macro_f1(truth[take], pred_a[take])
    return {"observed": macro_f1(truth, pred_b) - macro_f1(truth, pred_a),
            "lower_2_5": float(np.percentile(deltas, 2.5)), "upper_97_5": float(np.percentile(deltas, 97.5)),
            "replicates": replicates, "seed": seed, "subjects": int(len(subjects))}


def decide(dev_f1_a, dev_f1_b, dev_class_a, dev_class_b, holdout_f1_a, holdout_f1_b,
           holdout_class_a, holdout_class_b, bootstrap_lower, criteria) -> dict:
    """Apply the preregistered replacement rule: does B justify replacing A?

    dev_f1_*: shape (seeds, splits). dev_class_*: (seeds, splits, 7).
    holdout_f1_*: (seeds,). holdout_class_*: (seeds, 7). bootstrap_lower: primary seed.
    """
    dev_a, dev_b = np.asarray(dev_f1_a, float), np.asarray(dev_f1_b, float)
    if dev_a.shape != dev_b.shape or dev_a.ndim != 2:
        raise ValueError("Development scores must be aligned (seeds, splits)")
    dev_delta = dev_b - dev_a
    primary = dev_delta[0]
    seed_means = dev_delta.mean(1)
    dev_class_delta = (np.asarray(dev_class_b) - np.asarray(dev_class_a)).mean((0, 1))
    hold_delta = np.asarray(holdout_f1_b, float) - np.asarray(holdout_f1_a, float)
    hold_class_delta = (np.asarray(holdout_class_b) - np.asarray(holdout_class_a)).mean(0)
    checks = {
        "development_mean_delta": bool(primary.mean() >= criteria["minimum_development_mean_delta"]),
        "development_positive_splits": bool((primary > 0).sum() >= criteria["minimum_positive_splits"]),
        "development_all_seed_means_positive": bool((seed_means > 0).all()),
        "development_per_class_drop": bool(dev_class_delta.min() >= -criteria["maximum_per_class_drop"]),
        "holdout_mean_delta": bool(hold_delta.mean() >= criteria["minimum_holdout_delta"]),
        "holdout_bootstrap_lower_positive": bool(bootstrap_lower > 0),
        "holdout_per_class_drop": bool(hold_class_delta.min() >= -criteria["maximum_per_class_drop"]),
    }
    return {"checks": checks, "replace": all(checks.values()),
            "development_primary_deltas": primary.tolist(), "development_seed_mean_deltas": seed_means.tolist(),
            "development_mean_class_deltas": dev_class_delta.tolist(),
            "holdout_deltas": hold_delta.tolist(), "holdout_mean_delta": float(hold_delta.mean()),
            "holdout_mean_class_deltas": hold_class_delta.tolist()}


def factorial_effects(scores: dict) -> dict:
    """2x2 main effects on mean score. Keys: (model, policy) with model in {svm, et},
    policy in {standard, ordinary}."""
    s = {k: float(v) for k, v in scores.items()}
    model = ((s["et", "standard"] + s["et", "ordinary"]) - (s["svm", "standard"] + s["svm", "ordinary"])) / 2
    policy = ((s["svm", "ordinary"] + s["et", "ordinary"]) - (s["svm", "standard"] + s["et", "standard"])) / 2
    interaction = (s["et", "ordinary"] - s["et", "standard"]) - (s["svm", "ordinary"] - s["svm", "standard"])
    return {"model_effect": model, "policy_effect": policy, "interaction": interaction,
            "total_research_minus_runtime": s["et", "ordinary"] - s["svm", "standard"]}
