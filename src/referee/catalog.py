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
    entry = catalog[task]
    # The catalog carries a rich per-task record ({"value": v, "source": ..., "protocol": ...}); a
    # bare number is also accepted for a minimal catalog. Anything else is malformed.
    if isinstance(entry, dict):
        if "value" not in entry:
            raise CatalogError(f"signed incumbent for task {task!r} has no value")
        return float(entry["value"])
    if isinstance(entry, (int, float)) and not isinstance(entry, bool):
        return float(entry)
    raise CatalogError(f"signed incumbent for task {task!r} is malformed")


def resolve_mie(task: str, catalog: dict, expected_digest: str, *,
                fallback: float, mde: float) -> float:
    """Resolve the PER-TASK minimum interest effect (MIE) from the signed mie_distribution catalog.
    The per-task value is the field-anchored bar (that task's top-quartile of accepted effect sizes,
    read from `catalog[task]["mie_value"]`). A task with NO signed entry falls back to `fallback`
    (the config's mie_floor, the lowest ACCEPTABLE interest value, a backstop, not the per-task bar).
    The resolved MIE must sit strictly above the detectability floor (mde), else the interest bar is
    below what the design can even detect and the run cannot proceed. Verifies the digest first, so
    the per-task MIE is fixed BEFORE results, never retuned at handoff to fit the finding."""
    verify_catalog(catalog, expected_digest)
    entry = catalog.get(task)
    if entry is None:
        mie = float(fallback)
    elif not isinstance(entry, dict) or "mie_value" not in entry:
        raise CatalogError(f"signed mie entry for task {task!r} has no mie_value")
    else:
        mie = float(entry["mie_value"])
    if not (mie > mde):
        raise CatalogError(
            f"resolved MIE {mie} for task {task!r} is not strictly above mde {mde} "
            f"(the interest bar is below detectability)"
        )
    return mie


def incumbent_separated(claimed_value: float, incumbent_value: float, mie: float) -> bool:
    """The claimed held-out value must beat the incumbent's by at least the MIE."""
    return (claimed_value - incumbent_value) >= mie
