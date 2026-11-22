"""Fixed spatial representations for the immutable 596-feature window schema.

No labels, population statistics or evaluation data determine the representation.
Pair features encode the order of two absolute sensor summaries, without
subtracting pressure values. Unnamed arrays must already use the canonical
feature order; named inputs are checked against that order explicitly.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .posture_core import SENSOR_COLUMNS, feature_names


MODES = frozenset({"all_plus_mean_order", "all_plus_mean_median_order", "abs_rel_sensor_means"})


def spatial_feature_names(mode):
    """Return names in exactly the order produced by the selected representation."""
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"Unknown spatial feature mode: {mode!r}")
    if mode == "abs_rel_sensor_means":
        return [f"{prefix}_mean_{sensor}" for prefix in ("abs", "rel") for sensor in SENSOR_COLUMNS]
    statistics = ("mean", "median") if mode == "all_plus_mean_median_order" else ("mean",)
    return feature_names() + [f"abs_{stat}_order_{first}_vs_{second}" for stat in statistics
        for first, second in combinations(SENSOR_COLUMNS, 2)]


class SpatialFeatureTransformer(TransformerMixin, BaseEstimator):
    """Append pressure-order features or select absolute/relative sensor means.

    For each sensor pair i < j in canonical row-major order, the result is +1
    when the first value is greater, -1 when less, and zero for a tie. Comparing
    directly avoids subtraction overflow for extreme finite input. Positive
    common gain and common offset preserve this order when the transformed
    floating-point values remain finite and do not collapse distinct values.

    Float32 and float64 input precision is preserved. Other real numeric input
    is converted to float64. Appended modes retain every original feature value
    and its order; no original input array is modified.
    """

    def __init__(self, mode="all_plus_mean_order"):
        self.mode = mode

    def _validate_input(self, X):
        names = feature_names()
        if len(names) != 596 or len(set(names)) != 596:
            raise ValueError("Spatial features require the canonical 596-feature schema")
        if hasattr(X, "columns") and list(X.columns) != names:
            raise ValueError("Input feature names or ordering differ from the canonical 596-feature schema")
        values = np.asarray(X)
        if values.ndim != 2 or values.shape[1] != 596 or not len(values):
            raise ValueError("Expected nonempty features with shape (N, 596)")
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("Spatial features require finite real numeric values")
        dtype = np.float32 if values.dtype == np.dtype(np.float32) else np.float64
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(values, dtype=dtype)
        if not np.isfinite(values).all():
            raise ValueError("Features exceed the finite supported numeric range")
        return values

    def fit(self, X, y=None):
        output_names = spatial_feature_names(self.mode)
        values = self._validate_input(X)
        if SENSOR_COLUMNS != [f"R{row}{column}" for row in range(1, 5) for column in range(1, 5)]:
            raise ValueError("Spatial features require the canonical row-major 4 by 4 sensor grid")
        names = feature_names()
        self.mode_ = self.mode
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.mean_indices_ = np.asarray([names.index(f"abs_mean_{sensor}") for sensor in SENSOR_COLUMNS], dtype=int)
        self.median_indices_ = np.asarray([names.index(f"abs_median_{sensor}") for sensor in SENSOR_COLUMNS], dtype=int)
        self.selected_indices_ = np.asarray([names.index(f"{prefix}_mean_{sensor}")
            for prefix in ("abs", "rel") for sensor in SENSOR_COLUMNS], dtype=int)
        self.pair_indices_ = np.asarray(list(combinations(range(16), 2)), dtype=int)
        self.feature_names_out_ = np.asarray(output_names, dtype=object)
        self.n_features_out_ = len(output_names)
        return self

    def _check_mode(self):
        check_is_fitted(self, ("mode_", "mean_indices_", "median_indices_", "selected_indices_",
                               "pair_indices_", "feature_names_out_"))
        if self.mode != self.mode_:
            raise ValueError("Spatial feature mode changed after fitting; refit the transformer")

    def transform(self, X):
        self._check_mode()
        values = self._validate_input(X)
        if self.mode_ == "abs_rel_sensor_means":
            return values[:, self.selected_indices_].copy()
        first, second = self.pair_indices_.T
        indices = [self.mean_indices_]
        if self.mode_ == "all_plus_mean_median_order":
            indices.append(self.median_indices_)
        blocks = [values]
        for selected in indices:
            summaries = values[:, selected]
            left, right = summaries[:, first], summaries[:, second]
            comparison = np.zeros(left.shape, dtype=values.dtype)
            comparison[left > right] = 1
            comparison[left < right] = -1
            blocks.append(comparison)
        return np.concatenate(blocks, axis=1)

    def get_feature_names_out(self, input_features=None):
        self._check_mode()
        if input_features is not None and not np.array_equal(np.asarray(input_features), self.feature_names_in_):
            raise ValueError("Input feature names differ from the fitted canonical schema")
        return self.feature_names_out_.copy()
