"""The real experiment callables the ExperimentSubstrate injects. Each MEASURES one gate input
from real evidence so the referee never gates on an agent-authored number. Built test-first on
the Mac (the model scoring and the live MCP audit swap in on the cluster). This module grows one
callable at a time: consequence first, then the G0 probe, the mechanism ablation, and the novelty
audit."""
from __future__ import annotations

from gatelib import g0_detectable, mean_ci
from referee.catalog import incumbent_separated, resolve_consequence_template, resolve_incumbent


def real_g0(pipeline, *, mde: float, alpha: float, power_target: float = 0.8,
            n_trials: int = 200, rng) -> bool:
    """The G0 detectability probe. `pipeline(effect, rng) -> p_value` runs the real measurement
    path with a planted effect; G0 plants an MDE-sized effect and passes only if the empirical
    detection power clears the target. Returns g0_passed (False -> the referee reads INELIGIBLE)."""
    return g0_detectable(pipeline, mde, alpha, power_target, n_trials, rng).passed


def real_mechanism(*, score_full, score_ablated, specificity_ok: bool,
                   alpha: float) -> tuple[float, float, bool]:
    """The mechanism ablation. `score_full()` / `score_ablated()` return the per-item scores with
    and without the mechanism (through the backend on the cluster). Produces
    (mech_full_lo, mech_ablated_hi, specificity_ok): the full effect's LOWER CI and the ablated
    effect's UPPER CI, which mechanism_check requires to straddle the MIE."""
    full_lo, _ = mean_ci(score_full(), alpha)
    _, ablated_hi = mean_ci(score_ablated(), alpha)
    return full_lo, ablated_hi, bool(specificity_ok)


def real_novelty(schema, *, audit_fn, advance_argued: bool) -> tuple[bool, list, bool]:
    """The novelty audit. `audit_fn(schema) -> (collision, k_nearest)` queries the research MCPs
    (Semantic Scholar / arXiv / Scite on the cluster) for the mechanism in several phrasings.
    Produces the three novelty inputs the referee gates on."""
    collision, k_nearest = audit_fn(schema)
    return bool(collision), list(k_nearest), bool(advance_argued)


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
