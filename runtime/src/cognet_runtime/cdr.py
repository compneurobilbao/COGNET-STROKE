"""Concept to Disconnection Representation (CDR) and lesion overlap.

Both functions work on parcel vectors (one value per parcel), exactly as the analysis workflow
does, and give the same numbers as the original implementation on valid input. Edge cases that
the original left to numpy warnings are defined explicitly here:

* A concept mask with no parcel inside (value 1) or none outside (value -1) has no defined CDR:
  the value is NaN, and so is the p-value.
* If the mean of the disconnection values outside the concept is exactly zero, the CDR is the
  mean inside (not a ratio), as in the original; ``cdr_flag`` reports this case.
* A null map whose CDR is NaN (empty inside or outside) never counts as a hit but stays in the
  denominator of the p-value.
* The p-value is the fraction of null maps with a strictly larger value, never below 1 / n_nulls.
"""

from __future__ import annotations

import numpy as np

EMPTY_MASK = "empty_mask"
ZERO_OUTSIDE_MEAN = "zero_outside_mean"


def _as_vector(values, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D parcel vector, got {arr.ndim} dimensions")
    return arr


def _inside_outside_means(concept_mask: np.ndarray, values: np.ndarray):
    inside = concept_mask == 1
    outside = concept_mask == -1
    if not inside.any() or not outside.any():
        return np.nan, np.nan
    # numpy scalars, not Python floats: the division then keeps the dtype of the input (float32
    # maps), which keeps results and tie handling identical to the original implementation
    return np.mean(values[inside]), np.mean(values[outside])


def _cdr(concept_mask: np.ndarray, values: np.ndarray):
    inside_mean, outside_mean = _inside_outside_means(concept_mask, values)
    if np.isnan(inside_mean):
        return np.nan
    if outside_mean == 0:
        return inside_mean
    return inside_mean / outside_mean


def cdr_flag(concept_mask, disconnection_map) -> str | None:
    """Name the degenerate case of a CDR, or None for a regular ratio."""
    mask = _as_vector(concept_mask, "concept_mask")
    values = _as_vector(disconnection_map, "disconnection_map")
    inside_mean, outside_mean = _inside_outside_means(mask, values)
    if np.isnan(inside_mean):
        return EMPTY_MASK
    if outside_mean == 0:
        return ZERO_OUTSIDE_MEAN
    return None


def _p_value(observed, null_values: np.ndarray) -> float:
    if null_values.size == 0:
        raise ValueError("nulls must contain at least one null map")
    if np.isnan(observed):
        return float("nan")
    hits = int(np.count_nonzero(null_values > observed))
    return max(hits, 1) / null_values.size


def _check_nulls(nulls, n_parcels: int) -> np.ndarray:
    arr = np.asarray(nulls)
    if arr.ndim != 2 or arr.shape[1] != n_parcels:
        raise ValueError(
            f"nulls must have shape (n_nulls, {n_parcels}), got {arr.shape}"
        )
    return arr


def compute_cdr(concept_mask, disconnection_map, nulls=None):
    """CDR of a concept for a disconnection map, with a permutation p-value if nulls are given.

    Parameters
    ----------
    concept_mask : array, shape (n_parcels,)
        1 for parcels inside the concept, -1 outside (other values are ignored).
    disconnection_map : array, shape (n_parcels,)
        Parcellated disconnection values.
    nulls : array, shape (n_nulls, n_parcels), optional
        Null concept masks in the same coding.

    Returns
    -------
    float, or (float, float) when ``nulls`` is given: the CDR and its p-value.
    """
    mask = _as_vector(concept_mask, "concept_mask")
    values = _as_vector(disconnection_map, "disconnection_map")
    if mask.shape != values.shape:
        raise ValueError("concept_mask and disconnection_map must have the same length")
    value = _cdr(mask, values)
    if nulls is None:
        return float(value)
    null_arr = _check_nulls(nulls, mask.size)
    null_values = np.array([_cdr(null, values) for null in null_arr])
    return float(value), _p_value(value, null_values)


def _overlap(concept_mask: np.ndarray, lesion: np.ndarray) -> float:
    total = float(np.sum(lesion))
    if total == 0:
        return float("nan")
    return float(np.sum((concept_mask == 1) * lesion) / total)


def compute_overlap(concept_mask, lesion_map, nulls=None):
    """Fraction of the (parcellated) lesion inside the concept, with an optional p-value.

    An empty lesion has no defined overlap: the value and the p-value are NaN.
    """
    mask = _as_vector(concept_mask, "concept_mask")
    lesion = _as_vector(lesion_map, "lesion_map")
    if mask.shape != lesion.shape:
        raise ValueError("concept_mask and lesion_map must have the same length")
    value = _overlap(mask, lesion)
    if nulls is None:
        return value
    null_arr = _check_nulls(nulls, mask.size)
    null_values = np.array([_overlap(null, lesion) for null in null_arr], dtype=float)
    return value, _p_value(value, null_values)
