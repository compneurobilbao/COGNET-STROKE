"""Cognitive disconnection profile: score and p-value per concept and per domain.

Output format: ``contracts/schemas/v1/cognet-profile.schema.json`` of the stroke-server
repository (artefact type ``cognet_profile``).

Multiple comparisons: per measure, the p-value of every concept is Bonferroni-corrected over all
concepts (as the analysis workflow does). A domain's score is the mean of the finite scores of its
concepts; its p-value is the smallest p-value among its concepts times their number (at most 1),
Bonferroni-corrected again over all domains.
"""

from __future__ import annotations

import math

import numpy as np

from .bundle import ConceptMaps, load_concept
from .cdr import cdr_flag, compute_cdr, compute_overlap

PROFILE_VERSION = "1.0"
MEASURES = ("disconnection_positive", "disconnection_negative", "lesion_overlap")


def _num(value):
    value = float(value)
    return None if math.isnan(value) else value


def bonferroni(p_values, n_tests: int):
    """Bonferroni correction over ``n_tests`` tests; undefined p-values stay undefined."""
    return [None if p is None else min(1.0, p * n_tests) for p in p_values]


def split_disconnection(volume):
    """Positive part, and magnitude of the negative part, of a disconnection map."""
    volume = np.asarray(volume, dtype=np.float64)
    return np.where(volume > 0, volume, 0.0), np.where(volume < 0, -volume, 0.0)


def _concept_row(concept, domain, mask, nulls, parcel_values) -> dict:
    row: dict = {"concept": concept, "domain": domain}
    for measure in MEASURES:
        values = parcel_values[measure]
        if measure == "lesion_overlap":
            score, p = compute_overlap(mask, values, nulls)
            flag = None
        else:
            score, p = compute_cdr(mask, values, nulls)
            flag = cdr_flag(mask, values)
        result = {"score": _num(score), "p_value": _num(p)}
        if flag is not None:
            result["flag"] = flag
        row[measure] = result
    return row


def _domain_rows(concept_rows) -> list:
    order: list = []
    for row in concept_rows:
        if row["domain"] not in order:
            order.append(row["domain"])
    rows = []
    for domain in order:
        members = [r for r in concept_rows if r["domain"] == domain]
        out: dict = {"domain": domain, "n_concepts": len(members)}
        for measure in MEASURES:
            scores = [m[measure]["score"] for m in members if m[measure]["score"] is not None]
            ps = [m[measure]["p_value"] for m in members if m[measure]["p_value"] is not None]
            out[measure] = {
                "score": float(np.mean(scores)) if scores else None,
                "p_value": min(1.0, min(ps) * len(members)) if ps else None,
            }
        rows.append(out)
    return rows


def _finish(rows, alpha: float) -> None:
    """Add p_corrected (Bonferroni over the rows) and significant to every measure."""
    for measure in MEASURES:
        corrected = bonferroni([r[measure]["p_value"] for r in rows], len(rows))
        for r, pc in zip(rows, corrected):
            r[measure]["p_corrected"] = pc
            r[measure]["significant"] = pc is not None and pc < alpha


def build_profile(maps: ConceptMaps, parcel_values: dict, alpha: float, bundle_record: dict) -> dict:
    """Compute the profile.

    ``parcel_values`` maps each name in ``MEASURES`` to a parcel vector: the parcel means of the
    positive and of the negative part of the disconnection map, and the 0/1 indicator of the
    parcels touched by the lesion.
    """
    concept_rows = []
    n_nulls = None
    for concept, domain in maps.concepts:
        mask, nulls = load_concept(maps, concept)
        if n_nulls is None:
            n_nulls = int(nulls.shape[0])
        elif nulls.shape[0] != n_nulls:
            raise ValueError("all concepts must have the same number of null maps")
        concept_rows.append(_concept_row(concept, domain, mask, nulls, parcel_values))
    _finish(concept_rows, alpha)
    domain_rows = _domain_rows(concept_rows)
    _finish(domain_rows, alpha)
    return {
        "kind": "CognetProfile",
        "profile_version": PROFILE_VERSION,
        "method": {
            "name": "concept-to-disconnection-representation",
            "alpha": alpha,
            "correction": "bonferroni",
            "n_nulls": n_nulls,
            "n_parcels": maps.n_parcels,
            "measures": list(MEASURES),
        },
        "bundle": bundle_record,
        "concepts": concept_rows,
        "domains": domain_rows,
    }
