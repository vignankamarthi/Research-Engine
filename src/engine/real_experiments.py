"""The real experiment callables the ExperimentSubstrate injects. Each MEASURES one gate input
from real evidence so the referee never gates on an agent-authored number. Built test-first on
the Mac (the model scoring and the live MCP audit swap in on the cluster). This module grows one
callable at a time: consequence first, then the G0 probe, the mechanism ablation, and the novelty
audit."""
from __future__ import annotations

import numpy as np

from gatelib import g0_detectable, mean_ci
from referee.catalog import incumbent_separated, resolve_consequence_template, resolve_incumbent


def real_g0(pipeline, *, mde: float, alpha: float, power_target: float = 0.8,
            n_trials: int = 200, rng) -> bool:
    """The G0 detectability probe. `pipeline(effect, rng) -> p_value` runs the real measurement
    path with a planted effect; G0 plants an MDE-sized effect and passes only if the empirical
    detection power clears the target. Returns g0_passed (False -> the referee reads INELIGIBLE)."""
    return g0_detectable(pipeline, mde, alpha, power_target, n_trials, rng).passed


def real_mechanism(*, score_full, score_ablated, specificity_ok: bool,
                   alpha: float) -> tuple[float, bool]:
    """The mechanism ablation. `score_full()` / `score_ablated()` return the PER-ITEM scores with
    and without the mechanism, over the SAME items in the same order (through the backend on the
    cluster). Produces (mech_contrast_lo, specificity_ok): the LOWER CI of the paired per-item
    (full minus ablated) CONTRAST, which mechanism_check tests against the MIE. Judging the paired
    contrast, not two absolute levels, keeps the gate scale-correct on a metric with a non-zero
    chance floor (an MCQ task cannot score below chance, so requiring the ablated level below the
    MIE was unsatisfiable)."""
    full = np.asarray(score_full(), dtype=float)
    ablated = np.asarray(score_ablated(), dtype=float)
    if full.shape != ablated.shape:
        raise ValueError(f"paired mechanism scores must align: {full.shape} vs {ablated.shape}")
    contrast_lo, _ = mean_ci(full - ablated, alpha)
    return contrast_lo, bool(specificity_ok)


def real_novelty(schema, *, audit_fn) -> tuple[bool, list, bool]:
    """The novelty audit. `audit_fn(schema)` queries the research MCPs (Semantic Scholar / arXiv /
    Scite on the cluster) for the mechanism in several phrasings and returns either
    (collision, k_nearest) or (collision, k_nearest, advance_argued). The positive-delta ADVANCE is
    determined by the audit PARTY (and the human at triage), never read from the agent's proposal,
    and is FAIL-CLOSED False when the audit does not assert one, so an agreeable proposal cannot
    self-certify its own novelty."""
    result = tuple(audit_fn(schema))
    if len(result) == 3:
        collision, k_nearest, advance = result
    else:
        collision, k_nearest = result
        advance = False
    return bool(collision), list(k_nearest), bool(advance)


def resolve_consequence(claim_type: str, task: str, measured_value: float, mie: float, *,
                        consequence_catalog: dict, consequence_digest: str,
                        incumbent_catalog: dict, incumbent_digest: str,
                        held_out_confirmed: bool) -> tuple[bool, bool]:
    """Produce (consequence_confirmed, incumbent_separated) from the SIGNED catalogs and the
    held-out consequence experiment. Resolving verifies each catalog's digest, so a tampered catalog
    raises (CatalogError) and a claim-type with no pre-registered template cannot get one at handoff.
    The incumbent is a CAPABILITY-only concept, so only a capability consequence resolves and separates
    over one (from the MEASURED held-out value versus the signed incumbent at the MIE, NEVER the agent's
    claimed value, so a proposal cannot discharge the separation by inflating a number it authored). An
    effect / phenomenon / law consequence never references an incumbent (its signed template does not),
    so its separation leg is trivially satisfied and a task with NO signed incumbent is fine, its tier
    decided by its own downstream consequence. The agent's claimed_value stays informational (its own
    belief), not a gate input. Both `measured_value` and `held_out_confirmed` are outcomes of the real
    held-out consequence experiment (injected: mocked on the Mac, run through the backend on the cluster)."""
    # Anti-HARKing: the consequence template must already exist in the signed catalog (every type).
    resolve_consequence_template(claim_type, consequence_catalog, consequence_digest)
    if claim_type == "capability":
        incumbent_value = resolve_incumbent(task, incumbent_catalog, incumbent_digest)
        separated = incumbent_separated(measured_value, incumbent_value, mie)
    else:
        separated = True  # non-capability consequences do not separate over an incumbent
    return bool(held_out_confirmed), separated
