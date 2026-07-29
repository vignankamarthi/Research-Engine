"""Signed catalogs in the trusted process. The consequence-template catalog (claim-type ->
the pre-registered consequence a positive must discharge) and the per-task incumbent catalog
(task -> the strongest provenance-verified published held-out result) are both content-hashed
and signed into the config. Resolving from them verifies the digest first, so the consequence
and the incumbent are fixed BEFORE results, never authored at handoff to fit the finding."""
from __future__ import annotations

from common.canonical import canonical_digest


class CatalogError(Exception):
    """A catalog digest mismatch, or a lookup with no signed entry. Either way the run
    cannot proceed on an un-anchored consequence or incumbent."""


def catalog_digest(catalog: dict) -> str:
    return canonical_digest(catalog)


def verify_catalog(catalog: dict, expected_digest: str) -> None:
    if catalog_digest(catalog) != expected_digest:
        raise CatalogError("catalog digest mismatch: not the signed catalog")


def resolve_consequence_template(claim_type: str, catalog: dict, expected_digest: str) -> str:
    verify_catalog(catalog, expected_digest)
    if claim_type not in catalog:
        raise CatalogError(
            f"no pre-registered consequence template for claim_type {claim_type!r} "
            f"(anti-HARKing: a template cannot be authored at handoff)"
        )
    return catalog[claim_type]


def resolve_incumbent(task: str, catalog: dict, expected_digest: str) -> float:
    verify_catalog(catalog, expected_digest)
    if task not in catalog:
        raise CatalogError(f"no signed incumbent for task {task!r}")
    return float(catalog[task])


def incumbent_separated(claimed_value: float, incumbent_value: float, mie: float) -> bool:
    """The claimed held-out value must beat the incumbent's by at least the MIE."""
    return (claimed_value - incumbent_value) >= mie
