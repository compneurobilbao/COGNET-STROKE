"""Tests of the step wrapper on the synthetic reference dataset (generated, not committed).

Everything runs in-process; the container paths (Docker, Apptainer) are exercised by
.github/workflows/step.yml with ``stroke-step-reftest``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import jsonschema
import nibabel as nib
import numpy as np
import pytest
import yaml

STEP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP_DIR / "src"))
sys.path.insert(0, str(STEP_DIR / "tests"))

import make_reference  # noqa: E402
from cognet_step import run as step_run  # noqa: E402

from stroke_step_sdk import (  # noqa: E402
    EXIT_INTERNAL,
    EXIT_OK,
    EXIT_REJECTED,
    contracts,
    run_step,
)
from stroke_step_sdk.reftest import compare_output  # noqa: E402

CONTRACTS = contracts.contracts_dir()
MANIFEST = STEP_DIR / "step.yaml"
SCHEMA = json.loads((CONTRACTS / "schemas/v1/cognet-profile.schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference_dir(tmp_path_factory) -> Path:
    dest = tmp_path_factory.mktemp("reference")
    make_reference.generate(dest, CONTRACTS)
    return dest


class Staged:
    def __init__(self, root: Path, reference: Path) -> None:
        self.reference = reference
        self.in_dir, self.out_dir = root / "in", root / "out"
        self.ref_dir = root / "ref"
        for d in (self.in_dir, self.out_dir, root / "tmp", self.ref_dir):
            d.mkdir()
        shutil.copytree(reference / "bundle", self.ref_dir / "concept_maps")
        shutil.copyfile(reference / "lesion_mask.nii.gz", self.in_dir / "lesion_mask.nii.gz")
        shutil.copyfile(
            reference / "fdisconnection_map.nii.gz", self.in_dir / "fdisconnection_map.nii.gz"
        )
        (self.in_dir / "params.json").write_text('{"alpha": 0.05}', encoding="utf-8")
        self.env = {"STEP_CPUS": "1", "TMPDIR": str(root / "tmp")}

    def write(self, port: str, values: np.ndarray) -> None:
        _, affine = make_reference.working_grid(CONTRACTS)
        nib.save(nib.Nifti1Image(values, affine), str(self.in_dir / f"{port}.nii.gz"))

    def read(self, port: str) -> np.ndarray:
        return np.asanyarray(nib.load(str(self.in_dir / f"{port}.nii.gz")).dataobj)

    def run(self) -> int:
        return run_step(
            step_run,
            self.in_dir,
            self.out_dir,
            manifest_path=MANIFEST,
            ref_dir=self.ref_dir,
            environ=self.env,
        )

    def json(self, name: str) -> dict:
        return json.loads((self.out_dir / name).read_text(encoding="utf-8"))

    def validate(self) -> list:
        docs = [MANIFEST, self.out_dir / "provenance.json"]
        if (self.out_dir / "error.json").exists():
            docs.append(self.out_dir / "error.json")
        issues = contracts.validator().validate_files(docs)
        return [i.message for i in issues if i.severity == "error"]


@pytest.fixture
def staged(tmp_path, reference_dir) -> Staged:
    return Staged(tmp_path, reference_dir)


def test_manifest_declares_standard_codes():
    from stroke_step_sdk.manifest import load_manifest

    assert load_manifest(MANIFEST).missing_sdk_codes() == []


def test_reference_case_matches_the_independent_expectation(staged):
    assert staged.run() == EXIT_OK
    spec = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["reference_test"]["expected"]["profile"]
    spec = dict(spec, file=str(staged.reference / "expected" / "cognet_profile.json"))
    assert compare_output(spec, staged.out_dir / "profile.json", STEP_DIR, "cognet_profile") is None
    assert (staged.out_dir / "profile_figure.png").read_bytes()[:4] == b"\x89PNG"
    assert staged.validate() == []


def test_profile_content_and_schema(staged):
    assert staged.run() == EXIT_OK
    profile = staged.json("profile.json")
    jsonschema.Draft202012Validator(SCHEMA).validate(profile)
    bundle_yaml = yaml.safe_load((staged.ref_dir / "concept_maps" / "bundle.yaml").read_text())
    assert profile["bundle"]["content_digest"] == bundle_yaml["content_digest"]
    assert profile["method"]["n_nulls"] == make_reference.N_NULLS
    by_name = {c["concept"]: c for c in profile["concepts"]}
    # the lesion sits inside motor_left: strongest disconnection, significant after correction
    top = by_name["motor_left"]["disconnection_positive"]
    assert top["significant"] is True
    assert top["score"] > 10 * by_name["visual_early"]["disconnection_positive"]["score"]
    assert by_name["motor_left"]["lesion_overlap"]["score"] > by_name["motor_right"]["lesion_overlap"]["score"]
    domains = {d["domain"]: d for d in profile["domains"]}
    assert domains["motor"]["disconnection_positive"]["significant"] is True
    assert domains["sensory"]["disconnection_positive"]["significant"] is False
    prov = staged.json("provenance.json")
    assert [b["id"] for b in prov["bundles"]] == ["synthetic-concept-maps"]


def test_output_is_deterministic(tmp_path, reference_dir):
    outputs = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        run = Staged(tmp_path / name, reference_dir)
        assert run.run() == EXIT_OK
        outputs.append((run.out_dir / "profile.json").read_bytes())
    assert outputs[0] == outputs[1]


def test_alpha_changes_significance_only(tmp_path, reference_dir):
    results = {}
    for name, alpha in (("strict", 0.001), ("loose", 0.05)):
        (tmp_path / name).mkdir()
        run = Staged(tmp_path / name, reference_dir)
        (run.in_dir / "params.json").write_text(json.dumps({"alpha": alpha}), encoding="utf-8")
        assert run.run() == EXIT_OK
        results[name] = run.json("profile.json")
    strict = results["strict"]["concepts"][0]["disconnection_positive"]
    loose = results["loose"]["concepts"][0]["disconnection_positive"]
    assert strict["p_value"] == loose["p_value"] and strict["score"] == loose["score"]
    assert (strict["significant"], loose["significant"]) == (False, True)


def test_empty_lesion_is_rejected(staged):
    staged.write("lesion_mask", np.zeros(staged.read("lesion_mask").shape, np.uint8))
    assert staged.run() == EXIT_REJECTED
    err = staged.json("error.json")
    assert (err["category"], err["code"], err["port"]) == ("input_rejected", "mask_empty", "lesion_mask")
    assert staged.validate() == []


def test_non_finite_disconnection_map_is_rejected(staged):
    values = staged.read("fdisconnection_map").astype(np.float32)
    values[40, 50, 40] = np.nan
    staged.write("fdisconnection_map", values)
    assert staged.run() == EXIT_REJECTED
    assert staged.json("error.json")["code"] == "disconnection_not_finite"
    assert staged.validate() == []


def test_zero_disconnection_map_fails_qc(staged):
    staged.write("fdisconnection_map", np.zeros(staged.read("lesion_mask").shape, np.float32))
    assert staged.run() == EXIT_REJECTED
    err = staged.json("error.json")
    assert (err["category"], err["code"]) == ("qc_failed", "disconnection_map_empty")
    assert staged.validate() == []


def test_lesion_outside_the_parcellation_fails_qc(staged):
    values = np.zeros(staged.read("lesion_mask").shape, np.uint8)
    values[0:3, 0:3, 0:3] = 1  # corner of the grid, outside the synthetic brain
    staged.write("lesion_mask", values)
    assert staged.run() == EXIT_REJECTED
    err = staged.json("error.json")
    assert (err["category"], err["code"]) == ("qc_failed", "lesion_outside_parcellation")
    assert staged.validate() == []


def test_invalid_bundle_is_an_internal_failure(staged):
    (staged.ref_dir / "concept_maps" / "concepts" / "attention.npz").unlink()
    assert staged.run() == EXIT_INTERNAL


def test_unknown_parameter_and_missing_input(staged):
    (staged.in_dir / "params.json").write_text('{"nope": 1}', encoding="utf-8")
    assert staged.run() == EXIT_REJECTED
    assert staged.json("error.json")["code"] == "invalid_parameters"
    (staged.in_dir / "params.json").write_text("{}", encoding="utf-8")
    (staged.in_dir / "fdisconnection_map.nii.gz").unlink()
    shutil.rmtree(staged.out_dir)
    staged.out_dir.mkdir()
    assert staged.run() == EXIT_REJECTED
    assert staged.json("error.json")["code"] == "input_missing"
