"""The single trusted schema-normal-form. Derives the control set, the lineage key,
and the magnitude gate as the strictest consistent with the schema. The lineage key
hashes the canonicalized CLAIM only (not the conditions), so a condition-only change
stays one lineage. Measures and dataset identities are alias-canonicalized. This is
the trust-concentration point and runs only in the trusted process."""
from __future__ import annotations

import re
from dataclasses import dataclass

from common.canonical import canonical_digest, sha256_hex

_MEASURE_ALIASES = {
    "acc": "accuracy",
    "top1 acc": "accuracy",
    "top-1 acc": "accuracy",
    "top1 accuracy": "accuracy",
    "top-1 accuracy": "accuracy",
    "accuracy": "accuracy",
}
_DATASET_ALIASES = {
    "ssv2": "something-something-v2",
    "something something v2": "something-something-v2",
    "something-something v2": "something-something-v2",
    "something-something-v2": "something-something-v2",
}
_MAGNITUDE_GATE_BY_TYPE = {
    "effect": "mie_at_power",
    "qualitative_phenomenon": "phenomenon_vs_null",
    "capability": "capability_separation",
    "law_shape": "law_shape_fit",
    "multi_benchmark_superiority": "sota_margin",
}
_MANDATORY_CONTROLS = ("arch_control", "control_of_the_control")


class ControlCatalogError(Exception):
    """The recomputed control catalog does not match the signed digest: the derivation
    machinery has drifted from what the config was signed over. HALT, do not score."""


def control_catalog_digest() -> str:
    """Canonical digest over the whole control-derivation catalog: the mandatory controls,
    the two floor mappings, and the magnitude-gate-by-type mapping. Any weakening edit to the
    derivation changes this digest."""
    catalog = {
        "mandatory_controls": list(_MANDATORY_CONTROLS),
        "floor_normal": "untrained_floor",
        "floor_prior": "prior_ablated_baseline",
        "magnitude_gate_by_type": _MAGNITUDE_GATE_BY_TYPE,
    }
    return canonical_digest(catalog)


def verify_control_catalog(expected_digest: str) -> None:
    actual = control_catalog_digest()
    if actual != expected_digest:
        raise ControlCatalogError(
            f"control catalog digest mismatch: expected {expected_digest}, got {actual}"
        )


@dataclass(frozen=True, slots=True)
class Schema:
    claim: str          # canonical claim text (the proposition; lineage hashes this)
    claim_type: str
    backbone: str
    dataset: str
    scale: str
    measure: str
    prior_claim: bool


@dataclass(frozen=True, slots=True)
class Derived:
    control_set: tuple[str, ...]
    lineage_key: str
    magnitude_gate: str


def _canon_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def normalize_schema(raw: dict) -> Schema:
    for field in ("claim", "claim_type", "backbone", "dataset", "scale", "measure", "prior_claim"):
        if field not in raw:
            raise ValueError(f"schema missing field: {field}")
    claim_type = _canon_text(str(raw["claim_type"]))
    if claim_type not in _MAGNITUDE_GATE_BY_TYPE:
        raise ValueError(f"unknown claim_type: {raw['claim_type']!r}")
    measure = _canon_text(str(raw["measure"]))
    dataset = _canon_text(str(raw["dataset"]))
    return Schema(
        claim=_canon_text(str(raw["claim"])),
        claim_type=claim_type,
        backbone=_canon_text(str(raw["backbone"])),
        dataset=_DATASET_ALIASES.get(dataset, dataset),
        scale=_canon_text(str(raw["scale"])),
        measure=_MEASURE_ALIASES.get(measure, measure),
        prior_claim=bool(raw["prior_claim"]),
    )


def lineage_key(schema: Schema) -> str:
    # CLAIM only -> condition-only changes (backbone/scale/measure/dataset) stay one lineage.
    return sha256_hex(schema.claim.encode("utf-8"))


def derive(schema: Schema) -> Derived:
    if schema.prior_claim:
        # the artifact-catcher for a prior claim is the prior-ablated baseline, not
        # the trained-minus-untrained FLOOR (works-at-init IS the contribution).
        floor = "prior_ablated_baseline"
    else:
        floor = "untrained_floor"
    control_set = (floor,) + _MANDATORY_CONTROLS
    return Derived(
        control_set=control_set,
        lineage_key=lineage_key(schema),
        magnitude_gate=_MAGNITUDE_GATE_BY_TYPE[schema.claim_type],
    )
