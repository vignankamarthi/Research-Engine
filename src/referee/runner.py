"""The confirmatory runner: composes the frozen gates into a verdict in one atomic
exposure of the box. Order follows the spec (G0 -> controls/FLOOR -> backbone ->
magnitude -> mechanism -> consequence -> novelty -> score-once). The gate INPUTS that
are separate experiments (mechanism CIs, consequence flags, novelty audit, G0, OOD,
believed-claim) arrive in a bundle; the runner only DECIDES. Prior claims would swap
the FLOOR for the prior-ablated baseline (not yet wired; the MockBackend path is the
untrained FLOOR)."""
from __future__ import annotations

from dataclasses import dataclass

from gatelib import (
    EXCEEDS_MIE,
    INCONCLUSIVE,
    POWERED_NULL,
    backbone_check,
    classify_magnitude,
    consequence_check,
    floor_separation,
    magnitude_pvalue,
    mean_ci,
    mechanism_check,
    novelty_check,
    verify_gate_library,
)

from .lineage import verify_control_catalog


@dataclass(frozen=True, slots=True)
class Verdict:
    status: str            # CONFIRMED | STRONG | CONFIRMED_EFFECT | CONFIRMED_NEGATIVE
    reason: str = ""       # | INCONCLUSIVE | INELIGIBLE | FAILED
    backbone_label: str = ""
    pvalue: float | None = None  # magnitude p-value (None if the box was never scored)


@dataclass(frozen=True, slots=True)
class _Scored:
    """The shared, once-per-box scored context handed to a claim-type handler."""
    trained: object
    untrained: list
    floor: object
    bb: object            # BackboneResult
    pval: float
    mie: float
    alpha: float
    bundle: object


def _effect_handler(ctx: _Scored) -> Verdict:
    """The EFFECT (MIE-at-power) claim path: magnitude -> FLOOR -> mechanism -> novelty ->
    importance-consequence tier."""
    bb, pval, bundle = ctx.bb, ctx.pval, ctx.bundle
    lo, hi = mean_ci(ctx.trained, ctx.alpha)
    mag = classify_magnitude(lo, hi, ctx.mie)
    if mag == INCONCLUSIVE:
        return Verdict("INCONCLUSIVE", "ci_includes_mie", bb.label, pval)
    if mag == POWERED_NULL:
        if bundle.believed_claim:
            return Verdict("CONFIRMED_NEGATIVE", "powered_null_on_believed_claim", bb.label, pval)
        return Verdict("FAILED", "null_on_unbelieved_claim", bb.label, pval)

    assert mag == EXCEEDS_MIE  # the positive path must clear FLOOR, mechanism, novelty.
    if not ctx.floor.passed:
        return Verdict("FAILED", "floor_not_separated", bb.label, pval)
    if not mechanism_check(bundle.mech_full_lo, bundle.mech_ablated_hi, ctx.mie, bundle.specificity_ok):
        return Verdict("FAILED", "mechanism_unsupported", bb.label, pval)
    nov = novelty_check(bundle.novelty_collision, bundle.novelty_k_nearest, bundle.novelty_advance)
    if not nov.passed:
        return Verdict("FAILED", f"novelty_{nov.reason}", bb.label, pval)

    if not consequence_check(bundle.consequence_confirmed, bundle.incumbent_separated):
        return Verdict("CONFIRMED_EFFECT", "consequence_not_discharged", bb.label, pval)
    if bundle.ood_holds:
        return Verdict("STRONG", "generalizes", bb.label, pval)
    return Verdict("CONFIRMED", "all_gates_passed", bb.label, pval)


# The claim-type -> handler registry. Adding a claim-type is one entry here; an unwired type
# fails LOUD (ineligible) rather than silently drawing the EFFECT gauntlet. The non-EFFECT
# handlers (phenomenon / capability / law-shape) need their own gate INPUTS plumbed before they
# can be registered; the seam is here so that is an addition, not a rewrite.
_CLAIM_HANDLERS = {"effect": _effect_handler}


def precheck(schema, config, bundle) -> Verdict | None:
    """The box-INDEPENDENT gates, so a scarce box is never leased (or burned) for a failure that
    needs no scoring. Catalog drift RAISES (tamper HALT). G0 and an unwired claim-type return an
    INELIGIBLE verdict. `run_campaign` calls this BEFORE leasing; `confirm` repeats them as
    defense in depth. Returns None when it is safe to spend a box."""
    verify_control_catalog(config.control_catalog_hash)  # raises ControlCatalogError on drift
    verify_gate_library(config.gate_library_digest)       # raises GateLibraryError on gate-API drift
    if not bundle.g0_passed:
        return Verdict("INELIGIBLE", reason="g0_failed")
    if schema.claim_type not in _CLAIM_HANDLERS:
        return Verdict("INELIGIBLE", reason=f"claim_type_{schema.claim_type}_not_wired")
    return None


def confirm(backend, box, schema, config, bundle, k_untrained: int = 4) -> Verdict:
    mie = config.mie_floor
    alpha = config.alpha

    # Gate 0: the control catalog must still match what the config was signed over. A
    # mismatch means the derivation machinery drifted (a weakened control): HALT, do not score.
    verify_control_catalog(config.control_catalog_hash)
    verify_gate_library(config.gate_library_digest)  # the gate library's public shape is pinned too

    # Gate 1: G0 detectability is a precondition on any verdict.
    if not bundle.g0_passed:
        return Verdict("INELIGIBLE", reason="g0_failed")

    # One atomic exposure of the box: score trained + K untrained inits.
    trained = backend.score_box(box)
    untrained = [backend.score_box(box, untrained_init=k) for k in range(k_untrained)]
    pval = magnitude_pvalue(trained, mie)  # the selection family's p-value

    # Gate 2: the untrained-FLOOR separation (geometry-artifact catcher).
    floor = floor_separation(trained, untrained, mie, alpha)

    # Gate 3: backbone contamination (HARD on every positive).
    bb = backbone_check(box.origin_date, bundle.backbone_cutoff, bundle.membership_clean)
    if not bb.passed:
        return Verdict("INELIGIBLE", reason=f"backbone_{bb.label}", backbone_label=bb.label, pvalue=pval)

    # Dispatch the claim-type-specific gauntlet (magnitude / mechanism / consequence).
    handler = _CLAIM_HANDLERS.get(schema.claim_type)
    if handler is None:
        return Verdict("INELIGIBLE", reason=f"claim_type_{schema.claim_type}_not_wired",
                       backbone_label=bb.label, pvalue=pval)
    return handler(_Scored(trained, untrained, floor, bb, pval, mie, alpha, bundle))
