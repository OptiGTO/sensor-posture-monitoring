"""Training-only feature reduction for subject-disjoint posture experiments.

ANOVA F ratios are descriptive ranking scores here. Correlated recording
windows do not justify interpreting their nominal p values as evidence.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .posture_core import feature_names


FEATURE_COUNT = 596
F_SCORE_CAP = 1e12


def _input(X, y=None):
    if y is None:
        values = check_array(X)
        labels = None
    else:
        values, labels = check_X_y(X, y)
        classes = np.unique(labels)
        if len(classes) < 2 or len(classes) >= len(labels):
            raise ValueError("At least two classes and positive residual degrees of freedom are required")
    if values.shape[1] != FEATURE_COUNT or len(feature_names()) != FEATURE_COUNT:
        raise ValueError("Expected the original 596-feature input contract")
    return values, labels


def _finite_f(values, labels):
    # Constant/perfectly separating columns can emit warnings, NaN or infinity.
    # A fixed cap keeps their rank deterministic without infinite arithmetic.
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        warnings.simplefilter("ignore", UserWarning)
        raw, _ = f_classif(values, labels)
    return np.clip(np.nan_to_num(raw, nan=0., posinf=F_SCORE_CAP, neginf=0.), 0., F_SCORE_CAP)


class SubjectInvariantSelector(TransformerMixin, BaseEstimator):
    """Rank posture evidence after penalizing training-subject discrimination.

    score_j = F(posture)_j / (1 + strength * F(subject)_j).
    Ties prefer the original feature index. Selected columns retain their
    original order, preserving a stable serialized transformation contract.
    """
    def __init__(self, k=96, strength=1.0):
        self.k = k
        self.strength = strength

    def _parameters(self):
        if type(self.k) is not int or not 1 <= self.k <= FEATURE_COUNT:
            raise ValueError("k must be an integer between 1 and 596")
        if (isinstance(self.strength, (bool, np.bool_)) or not isinstance(self.strength, (int, float, np.number))
                or not np.isfinite(self.strength) or self.strength < 0):
            raise ValueError("strength must be nonnegative and finite")
        return self.k, float(self.strength)

    def fit(self, X, y, *, groups=None):
        k, strength = self._parameters()
        values, labels = _input(X, y)
        subjects = np.asarray(groups)
        if groups is None or subjects.ndim != 1 or len(subjects) != len(labels):
            raise ValueError("Training subject IDs must match training labels")
        if any(value is None or not str(value).strip() or isinstance(value, (float, np.floating)) and not np.isfinite(value) for value in subjects):
            raise ValueError("Training subject IDs must be nonempty and finite")
        try:
            identities, subject_counts = np.unique(subjects, return_counts=True)
        except TypeError as error:
            raise ValueError("Training subject IDs must have mutually comparable labels") from error
        if len(identities) < 2 or len(identities) >= len(subjects):
            raise ValueError("At least two training subjects and positive residual degrees of freedom are required")
        self.posture_f_scores_ = _finite_f(values, labels)
        self.subject_f_scores_ = _finite_f(values, subjects)
        with np.errstate(over="ignore"):
            self.scores_ = self.posture_f_scores_ / (1. + strength * self.subject_f_scores_)
        order = np.lexsort((np.arange(FEATURE_COUNT), -self.scores_))
        self.indices_ = np.sort(order[:k])
        self.feature_names_out_ = np.asarray(feature_names(), dtype=object)[self.indices_]
        self.n_features_in_ = FEATURE_COUNT
        self.n_samples_seen_ = len(labels)
        self.n_subjects_seen_ = len(identities)
        classes, class_counts = np.unique(labels, return_counts=True)
        self.n_classes_seen_ = len(classes)
        self.subject_counts_ = subject_counts
        self.class_counts_ = class_counts
        self.fitted_parameters_ = (k, strength)
        return self

    def _check_fitted_parameters(self):
        check_is_fitted(self, "indices_")
        if self._parameters() != self.fitted_parameters_:
            raise ValueError("Selector parameters changed after fitting; refit before transforming")

    def transform(self, X):
        self._check_fitted_parameters()
        values, _ = _input(X)
        return values[:, self.indices_]

    def get_feature_names_out(self, input_features=None):
        self._check_fitted_parameters()
        if input_features is not None and not np.array_equal(np.asarray(input_features), feature_names()):
            raise ValueError("Feature names differ from the original input contract")
        return self.feature_names_out_.copy()


class ShrinkageDiscriminantProjection(TransformerMixin, BaseEstimator):
    """Train-only standardized LDA eigen projection with automatic shrinkage.

    Constant training columns are removed before fitting. A singular within-
    class covariance fails explicitly; no unrecorded fallback is introduced.
    """
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        values, labels = _input(X, y)
        if self.n_components is not None and (type(self.n_components) is not int or self.n_components < 1):
            raise ValueError("n_components must be None or a positive integer")
        self.scaler_ = StandardScaler().fit(values)
        self.retained_indices_ = np.flatnonzero(self.scaler_.var_ > 0)
        maximum = min(len(np.unique(labels)) - 1, len(self.retained_indices_))
        if maximum < 1 or self.n_components is not None and self.n_components > maximum:
            raise ValueError("Projection dimension exceeds the retained features or classes minus one")
        count = maximum if self.n_components is None else self.n_components
        standardized = self.scaler_.transform(values)[:, self.retained_indices_]
        self.discriminant_ = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=count)
        try:
            self.discriminant_.fit(standardized, labels)
        except np.linalg.LinAlgError as error:
            raise ValueError("Training within-class covariance is singular; no projection was fitted") from error
        self.n_features_in_ = FEATURE_COUNT
        self.n_samples_seen_ = len(labels)
        self.n_components_ = count
        self.fitted_n_components_ = self.n_components
        self.feature_names_out_ = np.asarray([f"shrinkage_lda_axis_{index + 1}" for index in range(count)], dtype=object)
        return self

    def _check_parameters(self):
        check_is_fitted(self, "n_components_")
        if type(self.n_components) is not type(self.fitted_n_components_) or self.n_components != self.fitted_n_components_:
            raise ValueError("Projection parameters changed after fitting; refit before transforming")

    def transform(self, X):
        self._check_parameters()
        values, _ = _input(X)
        result = self.discriminant_.transform(self.scaler_.transform(values)[:, self.retained_indices_])
        if result.shape[1] != self.n_components_ or not np.isfinite(result).all():
            raise ValueError("Discriminant projection produced invalid output")
        return result

    def get_feature_names_out(self, input_features=None):
        self._check_parameters()
        if input_features is not None and not np.array_equal(np.asarray(input_features), feature_names()):
            raise ValueError("Feature names differ from the original input contract")
        return self.feature_names_out_.copy()
