# COGNET-STROKE step (step contract 1.0)

A container step that turns a lesion mask and a functional disconnection map (both on the 2 mm
MNI152NLin6Asym working grid) into a cognitive disconnection profile: a score and a permutation
p-value per concept and per domain. Manifest: [`step.yaml`](step.yaml).

## Two environments

| | Runtime (`runtime/`, package `cognet-runtime`) | Analysis (top-level `pyproject.toml`) |
|---|---|---|
| Purpose | CDR, parcellation, profile, figure; what the step image installs | generate concept maps (NiMARE, Neurosynth), null maps (brainsmash, neuromaps), reports for many subjects, notebooks |
| Dependencies | numpy, nibabel, nilearn, pandas (+ matplotlib for the figure, a plotting extra of nilearn) | the full pinned set |
| Image | yes (size gated below 1.5 GB in CI) | never |

`src/cdr.py` re-exports `compute_cdr` and `compute_overlap` from the runtime, so the analysis
workflows and notebooks have a single implementation to depend on.

## Inputs, outputs, bundle

| Port | Type | |
|---|---|---|
| `lesion_mask` (in) | `lesion_mask_mni` | binary, working grid |
| `fdisconnection_map` (in) | `fdisconnection_map` | float, positive = disconnection |
| `profile` (out) | `cognet_profile` | JSON, schema `contracts/schemas/v1/cognet-profile.schema.json` of the stroke-server repository |
| `profile_figure` (out) | `qc_figure` | PNG |
| `concept_maps` (bundle `concept-maps`, at `/ref/concept_maps`) | | layout below |

Bundle layout (documented in `runtime/src/cognet_runtime/bundle.py`):

```
parcellation.nii.gz     integer labels on the working grid, 0 = background (parcels = sorted non-zero labels)
concepts.tsv            columns concept, domain (snake_case); row order is the profile order
concepts/<concept>.npz  mask (n_parcels,) int8 in {1, -1};  nulls (n_nulls, n_parcels) int8 in {1, -1}
```

The step reduces the positive part and the magnitude of the negative part of the disconnection map,
and the lesion, to parcels (mean per parcel; a parcel counts as lesioned if any voxel is). For every
concept it computes the CDR (mean inside the concept divided by the mean outside) for both parts and
the overlap of the lesion with the concept, each against the bundle's null masks (p = fraction of
nulls with a strictly larger value, at least 1 / n_nulls). Concept p-values are Bonferroni-corrected
over all concepts per measure. A domain's score is the mean of its concepts' scores and its p-value
the smallest p-value among its concepts times their number, Bonferroni-corrected over domains.

Edge cases (tests in `runtime/tests/test_cdr.py`): an empty concept mask gives a null score and a
`flag: empty_mask`; a zero mean outside the concept gives the inside mean (not a ratio, as in the
original implementation) and `flag: zero_outside_mean`.

Producing a real bundle (concept maps from Neurosynth, nulls from brainsmash on the parcellation)
is done offline with the analysis environment (`generate_concept_maps`, and the null generation in
`src/generate_cdr_report`); that export is not part of this card. No real bundle is in this
repository or in CI.

## Tests and CI

- `runtime/tests`: `compute_cdr` and `compute_overlap` (hand-computed cases, ties, NaN nulls, zero
  outside mean, empty masks, equality with the original implementation), parcellation, bundle
  reader, profile, figure. `pip install ./runtime pytest && pytest runtime/tests`.
- `step/tests`: the wrapper on a synthetic dataset (`tests/make_reference.py`, fixed seed, generated,
  not committed) whose expected profile comes from an independent naive implementation.
- `.github/workflows/step.yml`: the above, then image build with a size gate (below 1.5 GB), the
  reference test under Docker and under Apptainer with the synthetic bundle mounted, validation with
  `contracts/validate.py` and against the profile schema. The SDK and the contract files are fetched
  at the commit in `ci/STROKE_SERVER_SHA`.
