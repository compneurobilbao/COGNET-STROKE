"""Voxel to parcel reduction (mean per parcel), on a parcellation image of the same grid."""

from __future__ import annotations

import numpy as np


def parcel_ids(labels: np.ndarray) -> np.ndarray:
    """Sorted non-zero integer labels of a parcellation array (0 is background)."""
    arr = np.asarray(labels)
    rounded = np.rint(arr)
    if not np.array_equal(arr, rounded):
        raise ValueError("the parcellation contains non-integer labels")
    ids = np.unique(rounded.astype(np.int64))
    return ids[ids > 0]


def parcellate_mean(data: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Mean of ``data`` over the voxels of every parcel in ``ids`` (same order).

    Parcels with no voxel (cannot happen for ids taken from ``labels``) give 0.
    """
    data = np.asarray(data, dtype=np.float64)
    lab = np.rint(np.asarray(labels)).astype(np.int64)
    if data.shape != lab.shape:
        raise ValueError("data and parcellation must have the same shape")
    flat_lab = lab.ravel()
    flat = data.ravel()
    inside = np.isin(flat_lab, ids)
    index = np.searchsorted(ids, flat_lab[inside])
    sums = np.bincount(index, weights=flat[inside], minlength=ids.size)
    counts = np.bincount(index, minlength=ids.size)
    return np.divide(sums, counts, out=np.zeros(ids.size), where=counts > 0)
