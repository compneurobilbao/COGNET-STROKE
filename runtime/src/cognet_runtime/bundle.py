"""Reader of the ``concept-maps`` reference bundle.

Layout (every file is listed in ``bundle.yaml``)::

    parcellation.nii.gz     integer labels on the 2 mm MNI152NLin6Asym working grid, 0 = background
    concepts.tsv            columns concept and domain (snake_case); row order is the profile order
    concepts/<concept>.npz  mask: int8 (n_parcels,), 1 inside the concept and -1 outside;
                            nulls: int8 (n_nulls, n_parcels), null masks in the same coding

The parcels are the sorted non-zero labels of the parcellation; vectors follow that order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .parcellation import parcel_ids

NAME_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


class BundleLayoutError(ValueError):
    """The bundle does not follow the layout above."""


@dataclass(frozen=True)
class ConceptMaps:
    root: Path
    labels: np.ndarray  # parcellation array (integer valued)
    affine: np.ndarray
    ids: np.ndarray  # parcel labels, sorted
    concepts: tuple  # ((concept, domain), ...) in file order

    @property
    def n_parcels(self) -> int:
        return int(self.ids.size)


def load_concept_maps(root, grid_shape=None, grid_affine=None) -> ConceptMaps:
    """Read the parcellation and the concept table; check them against the working grid."""
    root = Path(root)
    parc_path = root / "parcellation.nii.gz"
    table_path = root / "concepts.tsv"
    for p in (parc_path, table_path):
        if not p.is_file():
            raise BundleLayoutError(f"{p.name} is missing from the bundle")
    img = nib.load(str(parc_path))
    if grid_shape is not None and tuple(img.shape) != tuple(grid_shape):
        raise BundleLayoutError("the parcellation is not on the working grid (shape)")
    if grid_affine is not None and not np.allclose(img.affine, grid_affine, rtol=0, atol=1e-3):
        raise BundleLayoutError("the parcellation is not on the working grid (affine)")
    labels = np.asanyarray(img.dataobj)
    try:
        ids = parcel_ids(labels)
    except ValueError as exc:
        raise BundleLayoutError(str(exc)) from exc
    if ids.size == 0:
        raise BundleLayoutError("the parcellation has no parcel")

    table = pd.read_csv(table_path, sep="\t", dtype=str, keep_default_na=False)
    if list(table.columns) != ["concept", "domain"]:
        raise BundleLayoutError("concepts.tsv must have the columns concept and domain")
    if table.empty:
        raise BundleLayoutError("concepts.tsv has no concept")
    if table["concept"].duplicated().any():
        raise BundleLayoutError("concepts.tsv lists a concept twice")
    for column in ("concept", "domain"):
        if not all(NAME_RE.match(v) for v in table[column]):
            raise BundleLayoutError(f"concepts.tsv: '{column}' values must be snake_case")
    concepts = tuple(zip(table["concept"], table["domain"]))
    return ConceptMaps(root, labels, img.affine, ids, concepts)


def load_concept(maps: ConceptMaps, concept: str):
    """Mask (n_parcels,) and nulls (n_nulls, n_parcels) of one concept, as int8."""
    path = maps.root / "concepts" / f"{concept}.npz"
    if not path.is_file():
        raise BundleLayoutError(f"concepts/{concept}.npz is missing from the bundle")
    with np.load(path, allow_pickle=False) as npz:
        if "mask" not in npz or "nulls" not in npz:
            raise BundleLayoutError(f"concepts/{concept}.npz needs the arrays mask and nulls")
        mask = npz["mask"]
        nulls = npz["nulls"]
    if mask.shape != (maps.n_parcels,):
        raise BundleLayoutError(f"concepts/{concept}.npz: mask length differs from the parcel count")
    if nulls.ndim != 2 or nulls.shape[1] != maps.n_parcels or nulls.shape[0] < 1:
        raise BundleLayoutError(f"concepts/{concept}.npz: nulls must be (n_nulls, n_parcels)")
    if not (np.isin(mask, (-1, 1)).all() and np.isin(nulls, (-1, 1)).all()):
        raise BundleLayoutError(f"concepts/{concept}.npz: values must be 1 or -1")
    return mask.astype(np.int8), nulls.astype(np.int8)
