"""The real experiment callables the ExperimentSubstrate injects. Each MEASURES one gate input
from real evidence so the referee never gates on an agent-authored number. Built test-first on
the Mac (the model scoring and the live MCP audit swap in on the cluster). This module grows one
callable at a time: consequence first, then the G0 probe, the mechanism ablation, and the novelty
audit."""
from __future__ import annotations

from referee.catalog import incumbent_separated, resolve_consequence_template, resolve_incumbent


def resolve_consequence(claim_type: str, task: str, claimed_value: float, mie: float, *,
                        consequence_catalog: dict, consequence_digest: str,
                        incumbent_catalog: dict, incumbent_digest: str,
                        held_out_confirmed: bool) -> tuple[bool, bool]:
    """Produce (consequence_confirmed, incumbent_separated) from the SIGNED catalogs and the
    held-out consequence result. Resolving verifies each catalog's digest, so a tampered catalog
    raises (CatalogError) and a claim-type with no pre-registered template cannot get one at
    handoff. The incumbent separation is computed from the claimed vs signed-incumbent value at
    the MIE. `held_out_confirmed` is the outcome of the real held-out consequence experiment
    (injected: mocked on the Mac, run through the backend on the cluster)."""
    # Anti-HARKing: the consequence template must already exist in the signed catalog.
    resolve_consequence_template(claim_type, consequence_catalog, consequence_digest)
    incumbent_value = resolve_incumbent(task, incumbent_catalog, incumbent_digest)
    return bool(held_out_confirmed), incumbent_separated(claimed_value, incumbent_value, mie)
