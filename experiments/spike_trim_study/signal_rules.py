"""Experiment-only raw-signal rules for the spike-repair / start-trim study.

Nothing in the runtime (training wizard, replay, live inference, packaging)
imports this module. The study evaluates the rules without changing any model
contract; adoption after the preliminary round is documented separately.

All parameters are fixed here before any model is fitted.
"""
from __future__ import annotations

import numpy as np

# Isolated one-sample repair. Fixed values (sensor units: 0-5 V ADC scale).
ISOLATED_SPIKE_PARAMETERS = {
    "min_excursion": 0.10,
    "max_return_ratio": 0.5,
    "max_interval_seconds": 0.25,
}


def _check(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or time_s.shape != (len(values),):
        raise ValueError("time must be (N,) and values (N, channels)")
    if not (np.isfinite(time_s).all() and np.isfinite(values).all()):
        raise ValueError("time and values must be finite")
    if len(time_s) > 1 and np.any(np.diff(time_s) <= 0):
        raise ValueError("time must be strictly increasing")
    return time_s, values


def repair_isolated_spikes(time_s, values, *, min_excursion: float = 0.10, max_return_ratio: float = 0.5,
                           max_interval_seconds: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Replace one-sample up-and-back excursions by the mean of their neighbours.

    Sample ``i`` (never the first or last sample) of a channel is repaired when it
    lies on the same side of the previous (already repaired) sample and the next
    raw sample by at least ``min_excursion`` each, the neighbours agree within
    ``max_return_ratio`` of that excursion (so level steps and ramps are kept),
    and both adjacent sampling intervals are at most ``max_interval_seconds``.
    Returns (repaired values, boolean repair mask).
    """
    time_s, source = _check(time_s, values)
    repaired = source.copy()
    mask = np.zeros(source.shape, dtype=bool)
    if len(source) < 3:
        return repaired, mask
    close = np.diff(time_s) <= max_interval_seconds
    for index in range(1, len(source) - 1):
        if not (close[index - 1] and close[index]):
            continue
        previous, current, following = repaired[index - 1], source[index], source[index + 1]
        left, right = current - previous, current - following
        excursion = np.minimum(np.abs(left), np.abs(right))
        spike = ((left * right) > 0) & (excursion >= min_excursion) & (
            np.abs(following - previous) <= max_return_ratio * excursion)
        if spike.any():
            repaired[index, spike] = 0.5 * (previous[spike] + following[spike])
            mask[index, spike] = True
    return repaired, mask


def median3(time_s, values) -> tuple[np.ndarray, np.ndarray]:
    """Three-sample running median per channel; first and last samples unchanged."""
    _, source = _check(time_s, values)
    repaired = source.copy()
    if len(source) >= 3:
        repaired[1:-1] = np.median(np.stack([source[:-2], source[1:-1], source[2:]]), axis=0)
    return repaired, repaired != source


def start_trim_keep(time_s, seconds: float) -> np.ndarray:
    """Boolean mask keeping samples at least ``seconds`` after the first sample."""
    time_s = np.asarray(time_s, dtype=np.float64)
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("trim seconds must be a finite non-negative number")
    keep = (time_s - time_s[0]) >= seconds - 1e-9
    if keep.sum() < 2:
        raise ValueError("trim leaves fewer than two samples")
    return keep


def review_spike_mask(values) -> np.ndarray:
    """Transient-spike rule of ``src.data_review.inspect_record`` (HEAD 68a1937).

    Record-level adaptive thresholds, therefore used only for auditing and for
    marking analysis windows, never as a model input transform. The mask is
    aligned with ``values`` rows; the first and last rows are always False.
    """
    x = np.asarray(values, dtype=np.float64)
    mask = np.zeros(x.shape, dtype=bool)
    if len(x) < 3:
        return mask
    span = np.maximum(np.percentile(x, 95, axis=0) - np.percentile(x, 5, axis=0), 1e-6)
    spatial_floor = max(float(np.percentile(x, 95) - np.percentile(x, 5)) * .1, 1e-6)
    diff = np.diff(x, axis=0)
    mad = np.median(np.abs(diff - np.median(diff, axis=0)), axis=0) * 1.4826
    threshold = np.maximum(np.maximum(12 * mad, .25 * span), spatial_floor)
    mask[1:-1] = (np.abs(diff[:-1]) > threshold) & (np.abs(diff[1:]) > threshold) & (diff[:-1] * diff[1:] < 0)
    return mask


REPAIRS = {
    None: None,
    "isolated": lambda t, v: repair_isolated_spikes(t, v, **ISOLATED_SPIKE_PARAMETERS),
    "median3": median3,
}


def apply_repair(name: str | None, time_s, values) -> tuple[np.ndarray, np.ndarray]:
    if name not in REPAIRS:
        raise ValueError(f"unknown repair: {name!r}")
    if name is None:
        values = np.asarray(values, dtype=np.float64)
        return values.copy(), np.zeros(values.shape, dtype=bool)
    return REPAIRS[name](time_s, values)
