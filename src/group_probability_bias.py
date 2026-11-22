"""Bounded class-bias fitting from caller-supplied training OOF predictions.

This estimator cannot establish OOF provenance. The caller must generate the
probabilities using subject-held-out folds inside the training partition and
record those folds. Validation/test rows must never be supplied to ``fit``.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_is_fitted


CLASS_COUNT = 7
PROBABILITY_FLOOR = 1e-12
LOG_BIAS_GRID = (-float(np.log(1.5)), -float(np.log(1.2)), 0., float(np.log(1.2)), float(np.log(1.5)))
TIE_TOLERANCE = 1e-12


def _probability(values):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Probability input must be numeric") from error
    if array.ndim != 2 or array.shape[1] != CLASS_COUNT or len(array) == 0:
        raise ValueError("Probability input must be nonempty with exactly seven class columns")
    if (not np.isfinite(array).all() or (array < 0).any() or (array > 1).any()
            or not np.allclose(array.sum(1), 1., rtol=1e-7, atol=1e-8)):
        raise ValueError("Probabilities must be finite, nonnegative and normalized")
    return array


def _adjust(probability, log_bias):
    adjusted = np.clip(probability, PROBABILITY_FLOOR, 1.) * np.exp(log_bias)
    return adjusted / adjusted.sum(1, keepdims=True)


class GroupProbabilityBias(TransformerMixin, BaseEstimator):
    """Optimize OOF Macro F1 minus regularization * mean(log_bias ** 2).

    The seven classes are visited in fixed column order. Each coordinate may
    take only one of five preregistered values. Objective ties prefer smaller
    squared bias norm, then the fixed grid order; three sweeps is a hard limit.
    """
    def __init__(self, regularization=0.01, max_sweeps=3):
        self.regularization = regularization
        self.max_sweeps = max_sweeps

    def _parameters(self):
        if (isinstance(self.regularization, (bool, np.bool_))
                or not isinstance(self.regularization, (int, float, np.number))
                or not np.isfinite(self.regularization) or self.regularization < 0):
            raise ValueError("regularization must be finite and nonnegative")
        if type(self.max_sweeps) is not int or not 1 <= self.max_sweeps <= 3:
            raise ValueError("max_sweeps must be an integer from one to three")
        return float(self.regularization), self.max_sweeps

    def fit(self, probability, y, *, groups=None):
        regularization, max_sweeps = self._parameters()
        values = _probability(probability)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(values):
            raise ValueError("OOF labels must match the probability rows")
        if not np.isin(labels, np.arange(CLASS_COUNT)).all() or len(np.unique(labels)) != CLASS_COUNT:
            raise ValueError("OOF labels must contain all seven supported classes 0 through 6")
        labels = labels.astype(np.int64)
        subjects = np.asarray(groups)
        if groups is None or subjects.ndim != 1 or len(subjects) != len(labels):
            raise ValueError("OOF subject groups must match the probability rows")
        if any(value is None or not str(value).strip() or isinstance(value, (float, np.floating)) and not np.isfinite(value) for value in subjects):
            raise ValueError("OOF subject groups must be nonempty and finite")
        try:
            subject_count = len(np.unique(subjects))
        except TypeError as error:
            raise ValueError("OOF subject groups must have mutually comparable identities") from error
        if subject_count < 2:
            raise ValueError("OOF fitting requires at least two subject groups")

        def evaluate(bias):
            prediction = _adjust(values, bias).argmax(1)
            macro_f1 = float(f1_score(labels, prediction, labels=np.arange(CLASS_COUNT), average="macro", zero_division=0))
            norm = float(np.mean(np.square(bias)))
            return macro_f1 - regularization * norm, macro_f1, norm

        bias = np.zeros(CLASS_COUNT, dtype=np.float64)
        initial = evaluate(bias)
        history = []
        for sweep in range(max_sweeps):
            changed = False
            for column in range(CLASS_COUNT):
                current = bias.copy()
                best, best_result = current, evaluate(current)
                trials = []
                for value in LOG_BIAS_GRID:
                    candidate = current.copy()
                    candidate[column] = value
                    result = evaluate(candidate)
                    trials.append({"log_bias": value, "objective": result[0], "macro_f1": result[1], "mean_square_bias": result[2]})
                    if (result[0] > best_result[0] + TIE_TOLERANCE
                            or abs(result[0] - best_result[0]) <= TIE_TOLERANCE and result[2] < best_result[2] - TIE_TOLERANCE):
                        best, best_result = candidate, result
                coordinate_changed = not np.array_equal(best, current)
                changed = changed or coordinate_changed
                bias = best
                history.append({"sweep": sweep + 1, "class_index": column, "candidates": trials,
                    "selected_log_bias": float(bias[column]), "objective": best_result[0], "changed": coordinate_changed})
            if not changed:
                break
        final = evaluate(bias)
        self.log_bias_ = bias.copy()
        self.bias_ = np.exp(bias)
        self.initial_objective_, self.initial_macro_f1_ = initial[:2]
        self.objective_, self.macro_f1_ = final[:2]
        self.n_sweeps_ = sweep + 1
        self.n_features_in_ = CLASS_COUNT
        self.n_samples_seen_ = len(values)
        self.n_subjects_seen_ = subject_count
        self.fitted_parameters_ = (regularization, max_sweeps)
        self.training_details_ = {"algorithm": "bounded_coordinate_log_probability_bias_v1",
            "regularization": regularization, "maximum_sweeps": max_sweeps, "completed_sweeps": self.n_sweeps_,
            "log_bias_grid": list(LOG_BIAS_GRID), "class_order": list(range(CLASS_COUNT)),
            "tie_policy": "objective tolerance 1e-12; smaller mean squared bias; retain current on remaining tie; otherwise fixed grid order",
            "probability_floor": PROBABILITY_FLOOR, "log_bias": self.log_bias_.tolist(),
            "initial_macro_f1": self.initial_macro_f1_, "initial_objective": self.initial_objective_,
            "final_macro_f1": self.macro_f1_, "final_objective": self.objective_,
            "penalty": float(regularization * final[2]), "training_samples": len(values),
            "training_subjects": subject_count, "class_counts": np.bincount(labels, minlength=CLASS_COUNT).tolist(),
            "objective_audit": history,
            "provenance": "Caller must supply subject-held-out OOF predictions from training data only; groups validate identity count, not provenance"}
        return self

    def transform(self, probability):
        check_is_fitted(self, "log_bias_")
        if self._parameters() != self.fitted_parameters_:
            raise ValueError("Bias learner parameters changed after fitting; refit before transforming")
        return _adjust(_probability(probability), self.log_bias_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "log_bias_")
        return np.asarray([f"adjusted_probability_class_{index}" for index in range(CLASS_COUNT)], dtype=object)
