"""Tiny synthetic bundle on a 6 x 6 x 6 grid (2 x 2 x 2 voxel parcels, 27 parcels)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

SHAPE = (6, 6, 6)
N_PARCELS = 27


def parcel_labels():
    ii, jj, kk = np.indices(SHAPE)
    return (1 + ii // 2 + 3 * (jj // 2) + 9 * (kk // 2)).astype(np.int16)


def write_bundle(root: Path, concepts=None, n_nulls=8, seed=0):
    rng = np.random.default_rng(seed)
    concepts = concepts or {"alpha_one": "dom_a", "beta_two": "dom_a", "gamma": "dom_b"}
    (root / "concepts").mkdir(parents=True)
    nib.save(nib.Nifti1Image(parcel_labels(), np.eye(4)), str(root / "parcellation.nii.gz"))
    (root / "concepts.tsv").write_text(
        "concept\tdomain\n" + "".join(f"{c}\t{d}\n" for c, d in concepts.items()), encoding="utf-8"
    )
    for name in concepts:
        mask = np.where(rng.random(N_PARCELS) > 0.6, 1, -1).astype(np.int8)
        mask[0], mask[1] = 1, -1
        nulls = np.where(rng.random((n_nulls, N_PARCELS)) > 0.6, 1, -1).astype(np.int8)
        np.savez(root / "concepts" / f"{name}.npz", mask=mask, nulls=nulls)
    return root


@pytest.fixture
def bundle(tmp_path):
    return write_bundle(tmp_path / "bundle")
