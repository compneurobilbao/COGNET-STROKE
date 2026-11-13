"""Parcellation, bundle reader, profile and figure on a tiny synthetic bundle."""

import json

import nibabel as nib
import numpy as np
import pytest

from cognet_runtime.bundle import BundleLayoutError, load_concept, load_concept_maps
from cognet_runtime.parcellation import parcel_ids, parcellate_mean
from cognet_runtime.profile import MEASURES, bonferroni, build_profile, split_disconnection
from conftest import N_PARCELS, SHAPE, parcel_labels, write_bundle

RECORD = {"id": "toy", "version": "1.0.0", "content_digest": "sha256:" + "0" * 64}


def test_parcellate_mean_is_the_mean_per_parcel():
    labels = parcel_labels()
    data = np.arange(labels.size, dtype=float).reshape(SHAPE)
    ids = parcel_ids(labels)
    assert ids.tolist() == list(range(1, N_PARCELS + 1))
    got = parcellate_mean(data, labels, ids)
    expected = [data[labels == i].mean() for i in ids]
    assert np.allclose(got, expected)


def test_parcellate_mean_ignores_background_and_rejects_bad_input():
    labels = parcel_labels().copy()
    labels[:, :, 0] = 0
    ids = parcel_ids(labels)
    data = np.ones(SHAPE)
    data[:, :, 0] = 1000.0
    assert np.allclose(parcellate_mean(data, labels, ids), 1.0)
    with pytest.raises(ValueError):
        parcellate_mean(np.ones((2, 2, 2)), labels, ids)
    with pytest.raises(ValueError):
        parcel_ids(np.full(SHAPE, 1.5))


def test_split_disconnection():
    pos, neg = split_disconnection(np.array([-2.0, 0.0, 3.0]))
    assert pos.tolist() == [0.0, 0.0, 3.0]
    assert neg.tolist() == [2.0, 0.0, 0.0]


def test_bonferroni():
    assert bonferroni([0.01, None, 0.5], 3) == [0.03, None, 1.0]


def test_load_bundle(bundle):
    maps = load_concept_maps(bundle, SHAPE, np.eye(4))
    assert maps.n_parcels == N_PARCELS
    assert [c for c, _ in maps.concepts] == ["alpha_one", "beta_two", "gamma"]
    mask, nulls = load_concept(maps, "gamma")
    assert mask.shape == (N_PARCELS,) and nulls.shape == (8, N_PARCELS)


def test_bundle_layout_errors(tmp_path, bundle):
    with pytest.raises(BundleLayoutError):
        load_concept_maps(bundle, (7, 7, 7), np.eye(4))
    with pytest.raises(BundleLayoutError):
        load_concept_maps(bundle, SHAPE, np.diag([2.0, 2.0, 2.0, 1.0]))
    with pytest.raises(BundleLayoutError):
        load_concept_maps(tmp_path / "nothing")
    maps = load_concept_maps(bundle)
    (bundle / "concepts" / "gamma.npz").unlink()
    with pytest.raises(BundleLayoutError):
        load_concept(maps, "gamma")
    np.savez(
        bundle / "concepts" / "gamma.npz",
        mask=np.zeros(N_PARCELS, np.int8),
        nulls=np.ones((2, N_PARCELS), np.int8),
    )
    with pytest.raises(BundleLayoutError):  # the mask has zeros
        load_concept(maps, "gamma")
    bad = write_bundle(tmp_path / "bad", concepts={"Bad-Name": "dom"})
    with pytest.raises(BundleLayoutError):
        load_concept_maps(bad)


def _values(seed=3):
    rng = np.random.default_rng(seed)
    lesion = np.zeros(N_PARCELS)
    lesion[[0, 5]] = 1
    return {
        "disconnection_positive": rng.random(N_PARCELS),
        "disconnection_negative": rng.random(N_PARCELS),
        "lesion_overlap": lesion,
    }


def test_profile_structure_and_aggregation(bundle):
    maps = load_concept_maps(bundle, SHAPE, np.eye(4))
    prof = build_profile(maps, _values(), 0.05, RECORD)
    assert prof["method"]["n_nulls"] == 8 and prof["method"]["n_parcels"] == N_PARCELS
    assert [c["concept"] for c in prof["concepts"]] == ["alpha_one", "beta_two", "gamma"]
    assert [d["domain"] for d in prof["domains"]] == ["dom_a", "dom_b"]
    dom_a = prof["domains"][0]
    assert dom_a["n_concepts"] == 2
    for m in MEASURES:
        c1, c2 = prof["concepts"][0][m], prof["concepts"][1][m]
        assert dom_a[m]["score"] == pytest.approx((c1["score"] + c2["score"]) / 2)
        assert dom_a[m]["p_value"] == pytest.approx(min(1.0, 2 * min(c1["p_value"], c2["p_value"])))
        for c in prof["concepts"]:
            assert c[m]["p_corrected"] == pytest.approx(min(1.0, 3 * c[m]["p_value"]))
            assert c[m]["p_value"] >= 1 / 8
            assert c[m]["significant"] == (c[m]["p_corrected"] < 0.05)
        dom_b = prof["domains"][1][m]
        assert dom_b["p_corrected"] == pytest.approx(min(1.0, 2 * dom_b["p_value"]))
    json.dumps(prof, allow_nan=False)  # serialisable without NaN


def test_profile_with_degenerate_concept_has_null_score_and_flag(tmp_path):
    root = write_bundle(tmp_path / "b", concepts={"only": "dom"})
    np.savez(
        root / "concepts" / "only.npz",
        mask=np.ones(N_PARCELS, np.int8),
        nulls=np.ones((4, N_PARCELS), np.int8),
    )
    maps = load_concept_maps(root)
    prof = build_profile(maps, _values(), 0.05, RECORD)
    r = prof["concepts"][0]["disconnection_positive"]
    assert r["score"] is None and r["p_value"] is None and r["p_corrected"] is None
    assert r["flag"] == "empty_mask" and r["significant"] is False
    assert prof["domains"][0]["disconnection_positive"]["score"] is None
    json.dumps(prof, allow_nan=False)


def test_profile_rejects_unequal_null_counts(tmp_path):
    root = write_bundle(tmp_path / "b", concepts={"one": "d", "two": "d"})
    mask = np.where(np.arange(N_PARCELS) % 2 == 0, 1, -1).astype(np.int8)
    np.savez(root / "concepts" / "two.npz", mask=mask, nulls=np.tile(mask, (3, 1)))
    with pytest.raises(ValueError):
        build_profile(load_concept_maps(root), _values(), 0.05, RECORD)


def test_figure_is_written(bundle, tmp_path):
    from cognet_runtime.figure import profile_figure

    maps = load_concept_maps(bundle, SHAPE, np.eye(4))
    prof = build_profile(maps, _values(), 0.05, RECORD)
    rng = np.random.default_rng(0)
    disc = nib.Nifti1Image(rng.random(SHAPE).astype(np.float32), np.eye(4))
    les = np.zeros(SHAPE, np.uint8)
    les[2:4, 2:4, 2:4] = 1
    out = tmp_path / "fig.png"
    profile_figure(out, disc, nib.Nifti1Image(les, np.eye(4)), prof)
    assert out.read_bytes()[:4] == b"\x89PNG"
