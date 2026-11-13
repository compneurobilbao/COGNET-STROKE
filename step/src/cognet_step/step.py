"""Contract wrapper around the COGNET-STROKE runtime.

Flow: check inputs -> read the ``concept-maps`` bundle -> reduce the lesion and the positive and
negative parts of the disconnection map to parcels -> profile (CDR and lesion overlap per concept,
aggregated per domain) -> write the JSON profile and the figure.
"""

from __future__ import annotations

import importlib.metadata as md
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from jsonschema import Draft202012Validator

from cognet_runtime.bundle import BundleLayoutError, load_concept_maps
from cognet_runtime.figure import profile_figure
from cognet_runtime.parcellation import parcellate_mean
from cognet_runtime.profile import build_profile, split_disconnection
from stroke_step_sdk import (
    InputRejected,
    InternalFailure,
    QcFailed,
    StepContext,
    contracts,
)

BUNDLE = "concept_maps"
PROFILE_SCHEMA = "schemas/v1/cognet-profile.schema.json"


def working_grid():
    registry = yaml.safe_load(
        (contracts.contracts_dir() / "artefact-types.yaml").read_text(encoding="utf-8")
    )
    grid = registry["grids"]["MNI152NLin6Asym-res2"]
    return tuple(grid["shape"]), np.array(grid["affine"], dtype=float)


def read_inputs(ctx: StepContext):
    lesion_img = nib.load(str(ctx.input_path("lesion_mask")))
    disc_img = nib.load(str(ctx.input_path("fdisconnection_map")))
    lesion = np.asanyarray(lesion_img.dataobj) > 0
    if not lesion.any():
        raise InputRejected(
            "mask_empty", "The lesion mask has no voxels set.", port="lesion_mask"
        )
    disc = np.asanyarray(disc_img.dataobj).astype(np.float64)
    if not np.isfinite(disc).all():
        raise InputRejected(
            "disconnection_not_finite",
            "The disconnection map contains NaN or infinite values.",
            port="fdisconnection_map",
        )
    if not disc.any():
        raise QcFailed(
            "disconnection_map_empty",
            "The disconnection map is zero everywhere, so no profile can be computed.",
            port="fdisconnection_map",
        )
    return lesion_img, lesion, disc_img, disc


def validate_profile(profile: dict, ctx: StepContext) -> None:
    """Check the document against the contract schema shipped with the image."""
    path = contracts.contracts_dir() / PROFILE_SCHEMA
    if not path.is_file():
        ctx.log.warning("profile schema not found in the contracts directory; not checked")
        return
    schema = json.loads(path.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(profile))
    if errors:
        raise InternalFailure(
            "profile_schema_violation",
            "The computed profile does not satisfy its schema.",
            details={"first_error": errors[0].message[:200]},
        )


def run(ctx: StepContext) -> None:
    shape, affine = working_grid()
    lesion_img, lesion, disc_img, disc = read_inputs(ctx)

    bundle_dir = ctx.bundle_path(BUNDLE)
    bundle_doc = yaml.safe_load((bundle_dir / "bundle.yaml").read_text(encoding="utf-8"))
    try:
        maps = load_concept_maps(bundle_dir, shape, affine)
    except BundleLayoutError as exc:
        raise InternalFailure(
            "bundle_layout_invalid",
            "The concept-map bundle does not follow the layout the step expects.",
            details={"reason": str(exc)[:200]},
        ) from exc
    ctx.log.info(
        "bundle %s %s: %d parcels, %d concepts",
        bundle_doc["id"], bundle_doc["version"], maps.n_parcels, len(maps.concepts),
    )

    positive, negative = split_disconnection(disc)
    values = {
        "disconnection_positive": parcellate_mean(positive, maps.labels, maps.ids),
        "disconnection_negative": parcellate_mean(negative, maps.labels, maps.ids),
        # a parcel is touched by the lesion if any of its voxels is lesioned
        "lesion_overlap": (parcellate_mean(lesion, maps.labels, maps.ids) > 0).astype(float),
    }
    if not values["lesion_overlap"].any():
        raise QcFailed(
            "lesion_outside_parcellation",
            "The lesion does not overlap any parcel of the concept-map parcellation.",
            port="lesion_mask",
        )
    ctx.log.info("lesion touches %d of %d parcels", int(values["lesion_overlap"].sum()), maps.n_parcels)

    record = {
        "id": bundle_doc["id"],
        "version": bundle_doc["version"],
        "content_digest": bundle_doc["content_digest"],
    }
    if bundle_doc.get("synthetic"):
        record["synthetic"] = True
    try:
        profile = build_profile(maps, values, float(ctx.params["alpha"]), record)
    except (BundleLayoutError, ValueError) as exc:
        raise InternalFailure(
            "bundle_layout_invalid",
            "The concept-map bundle does not follow the layout the step expects.",
            details={"reason": str(exc)[:200]},
        ) from exc
    validate_profile(profile, ctx)

    ctx.output_path("profile").write_text(
        json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    profile_figure(
        ctx.output_path("profile_figure"),
        nib.Nifti1Image(disc.astype(np.float32), disc_img.affine),
        nib.Nifti1Image(lesion.astype(np.uint8), lesion_img.affine),
        profile,
    )

    for name in ("numpy", "nibabel", "nilearn", "pandas", "matplotlib"):
        ctx.add_software(name, md.version(name))
