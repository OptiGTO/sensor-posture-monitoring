"""Stateless pressure-shape features for the existing 596-column contract.

Positive common gain and common additive pressure offset cancel from the shape
features. Temporal standard deviations scale with gain but not with offset.
Degenerate (constant-pressure) windows have zero contrast, mass, centre of
pressure and concentration. No population statistics or labels are learned.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .posture_core import SENSOR_COLUMNS, feature_names


MODES = frozenset({"contrast", "all_plus_contrast"})
EPSILON = 1e-12


def contrast_feature_names() -> list[str]:
    """Return the fixed 75 names in the transform's output order."""
    names = [f"contrast_{stat}_{sensor}"
        for stat in ("mean", "median", "pressure_fraction", "temporal_std")
        for sensor in SENSOR_COLUMNS]
    names += [f"contrast_{axis}_mass_{index}" for axis in ("row", "column")
        for index in range(1, 5)]
    return names + ["contrast_cop_x", "contrast_cop_y", "contrast_concentration"]


def _normalized_shape(values):
    # Scaling first avoids overflow in mean, norm and mass calculations. The
    # epsilon applies to this dimensionless norm; unresolved spatial contrast
    # (at most 1e-12 relative to the largest pressure) is treated as constant.
    # A consistent row-major layout keeps reduction order independent of batch
    # size (advanced feature indexing otherwise returns Fortran-style views).
    values = np.ascontiguousarray(values)
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    scaled = np.divide(values, scale, out=np.zeros_like(values), where=scale > 0)
    centered = scaled - scaled.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.square(centered).sum(axis=1, keepdims=True))
    present = norm > EPSILON
    normalized = np.divide(centered, norm, out=np.zeros_like(centered), where=present)
    return normalized, scaled, scale, norm, present


class PressureContrastFeatures(TransformerMixin, BaseEstimator):
    """Select stateless per-window pressure contrasts or append them to input.

    Input names, if supplied by a DataFrame or ``get_feature_names_out``, must
    exactly match ``posture_core.feature_names()``. The pressure grid is the
    row-major 4 by 4 ``SENSOR_COLUMNS`` grid. CoP x indexes columns and CoP y
    indexes rows, both on [-1, 1]. Concentration is the sum of squared masses.
    The appended mode retains all 596 original columns without modification.
    """

    def __init__(self, mode="contrast"):
        self.mode = mode

    def _validate_input(self, X):
        names = feature_names()
        values = check_array(X, dtype=np.float64)
        if values.shape[1] != len(names):
            raise ValueError("Expected original 596-feature runtime input")
        if hasattr(X, "columns") and list(X.columns) != names:
            raise ValueError("Input feature names or ordering differ from the 596-feature contract")
        return values

    def fit(self, X, y=None):
        if self.mode not in MODES:
            raise ValueError(f"Unknown pressure contrast mode: {self.mode}")
        values = self._validate_input(X)
        names = feature_names()
        if SENSOR_COLUMNS != [f"R{row}{column}" for row in range(1, 5) for column in range(1, 5)]:
            raise ValueError("Pressure contrast features require the row-major 4 by 4 sensor grid")
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.stat_indices_ = {stat: np.asarray([names.index(f"abs_{stat}_{sensor}")
            for sensor in SENSOR_COLUMNS], dtype=int) for stat in ("mean", "median", "std")}
        self.fitted_mode_ = self.mode
        output_names = ([] if self.mode == "contrast" else names) + contrast_feature_names()
        self.feature_names_out_ = np.asarray(output_names, dtype=object)
        self.n_features_out_ = len(output_names)
        return self

    def _check_mode(self):
        check_is_fitted(self, ("stat_indices_", "fitted_mode_"))
        if self.mode != self.fitted_mode_:
            raise ValueError("Pressure contrast mode changed after fit; refit the transformer")

    def transform(self, X):
        self._check_mode()
        values = self._validate_input(X)
        means = values[:, self.stat_indices_["mean"]]
        medians = values[:, self.stat_indices_["median"]]
        temporal_std = values[:, self.stat_indices_["std"]]
        mean_shape, scaled, scale, norm, present = _normalized_shape(means)
        median_shape = _normalized_shape(medians)[0]
        shifted = scaled - scaled.min(axis=1, keepdims=True)
        total = shifted.sum(axis=1, keepdims=True)
        fraction = np.divide(shifted, total, out=np.zeros_like(shifted),
            where=present & (total > EPSILON))
        # Division is split to avoid overflow in scale * norm. Saturation only
        # handles ratios beyond representable float64; ordinary finite sensor
        # windows are unchanged. Undefined constant-window ratios are zero.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            normalized_std = np.divide(temporal_std, scale, out=np.zeros_like(temporal_std),
                where=present)
            np.divide(normalized_std, norm, out=normalized_std, where=present)
        normalized_std = np.nan_to_num(normalized_std, copy=False)
        grid = fraction.reshape(-1, 4, 4)
        row_mass, column_mass = grid.sum(axis=2), grid.sum(axis=1)
        coordinates = np.linspace(-1., 1., 4)
        cop_x = (column_mass * coordinates).sum(axis=1)
        cop_y = (row_mass * coordinates).sum(axis=1)
        concentration = np.square(fraction).sum(axis=1)
        contrast = np.column_stack((mean_shape, median_shape, fraction, normalized_std,
            row_mass, column_mass, cop_x, cop_y, concentration))
        return contrast if self.mode == "contrast" else np.column_stack((values, contrast))

    def get_feature_names_out(self, input_features=None):
        self._check_mode()
        if input_features is not None and not np.array_equal(np.asarray(input_features), self.feature_names_in_):
            raise ValueError("Input feature names differ from the fitted input contract")
        return self.feature_names_out_.copy()
