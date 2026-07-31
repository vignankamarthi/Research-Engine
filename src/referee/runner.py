"""The confirmatory runner: composes the frozen gates into a verdict in one atomic
exposure of the box. Order follows the spec (G0 -> controls/FLOOR -> backbone ->
magnitude -> mechanism -> consequence -> novelty -> score-once). The gate INPUTS that
are separate experiments (mechanism CIs, consequence flags, novelty audit, G0, OOD,
believed-claim) arrive in a bundle; the runner only DECIDES. Prior claims would swap
the FLOOR for the prior-ablated baseline (not yet wired; the MockBackend path is the
untrained FLOOR)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gatelib import (
    EXCEEDS_MIE,
    FAIL,
    INCONCLUSIVE,
    PASS,
    POWERED_NULL,
    backbone_check,
    consequence_check,
    floor_separation,
    magnitude_gate_for,
    magnitude_pvalue,
    mean_ci,
    mechanism_check,
    novelty_check,
    verify_gate_library,
)

from .lineage import derive, verify_control_catalog


@dataclass(frozen=True, slots=True)
class Verdict:
    status: str            # CONFIRMED | STRONG | CONFIRMED_EFFECT | CONFIRMED_NEGATIVE
    reason: str = ""       # | INCONCLUSIVE | INELIGIBLE | FAILED
    backbone_label: str = ""
    pvalue: float | None = None  # magnitude p-value (None if the box was never scored)


# The claim-types the runner actually confirms. An unwired type fails LOUD (INELIGIBLE) rather than
# silently borrowing another gauntlet. The magnitude GATE for each is chosen by the signed
# schema-normal-form (`lineage._MAGNITUDE_GATE_BY_TYPE`, covered by the control-catalog digest), so
# the per-type decision lives inside the ONE signed mapping, not a second unsigned dispatch table.
_WIRED_CLAIM_TYPES = frozenset({"effect", "capability", "qualitative_phenomenon", "law_shape"})


def _missing_threshold(gate_name, bundle):
    """The name of a substrate-measured threshold the gate needs but the bundle lacks, or None.
    Box-INDEPENDENT (bundle-only), so `precheck` can refuse an unmeasured bar BEFORE a box is leased.
    `mie_at_power` needs none here (its bar is the MIE, always present, and its magnitude is the
    trained-minus-untrained contrast formed at score time)."""
    if gate_name == "capability_separation" and bundle.incumbent_rate is None:
        return "incumbent_rate"
    if gate_name == "phenomenon_vs_null" and bundle.baseline_rate is None:
        return "baseline_rate"
    if gate_name == "law_shape_fit" and (
            bundle.law_predicted is None or bundle.law_observed is None or bundle.law_tol is None):
        return "law_shape_series"
    return None


def _magnitude_kwargs(gate_name, lo, hi, bundle):
    """The kwargs the signed magnitude gate needs, plus the threshold its p-value tests against
    (None for law_shape, a functional-form fit with no scalar p-value). Assumes the threshold is
    present -- callers guard with `_missing_threshold` first. This is the ONE per-gate switch (the
    p-value threshold comes from here too), so the kwargs and the statistic can never disagree."""
    if gate_name == "capability_separation":
        return {"ci_lo": lo, "ci_hi": hi, "incumbent_rate": bundle.incumbent_rate}, bundle.incumbent_rate
    if gate_name == "phenomenon_vs_null":
        return {"ci_lo": lo, "ci_hi": hi, "baseline_rate": bundle.baseline_rate}, bundle.baseline_rate
    if gate_name == "law_shape_fit":
        return {"predicted": bundle.law_predicted, "observed": bundle.law_observed,
                "tol": bundle.law_tol}, None
    raise ValueError(f"unsupported magnitude gate {gate_name!r}")


def precheck(schema, config, bundle) -> Verdict | None:
    """The box-INDEPENDENT gates, so a scarce box is never leased (or burned) for a failure that
    needs no scoring. Catalog drift RAISES (tamper HALT). G0, an unwired claim-type, a prior claim
    (the prior-ablated FLOOR is not yet executable), and a law_shape sweep over the single-GPU 24h
    envelope each return INELIGIBLE. `run_campaign` calls this BEFORE leasing; `confirm` repeats them
    as defense in depth. Returns None when it is safe to spend a box."""
    verify_control_catalog(config.control_catalog_hash)  # raises ControlCatalogError on drift
    verify_gate_library(config.gate_library_digest)       # raises GateLibraryError on gate-API drift
    if not bundle.g0_passed:
        return Verdict("INELIGIBLE", reason="g0_failed")
    if schema.claim_type not in _WIRED_CLAIM_TYPES:
        return Verdict("INELIGIBLE", reason=f"claim_type_{schema.claim_type}_not_wired")
    if schema.prior_claim:
        return Verdict("INELIGIBLE", reason="prior_ablated_floor_not_wired")
    # A magnitude threshold the substrate never measured is box-INDEPENDENT, so refuse it here rather
    # than after a box is leased. Checked BEFORE the envelope so an UNMEASURED law_shape reads
    # `magnitude_input_missing_law_shape_series`, not the mislabeled `over_24h_envelope`.
    missing = _missing_threshold(derive(schema).magnitude_gate, bundle)
    if missing is not None:
        return Verdict("INELIGIBLE", reason=f"magnitude_input_missing_{missing}")
    if schema.claim_type == "law_shape" and not bundle.law_within_envelope:
        return Verdict("INELIGIBLE", reason="law_shape_over_24h_envelope")
    return None


def confirm(backend, box, schema, config, bundle, k_untrained: int = 4) -> Verdict:
    # The operative interest bar is the PER-TASK MIE the substrate resolved from the signed
    # mie_distribution catalog; config.mie_floor is only the fallback for a task with no signed entry.
    mie = bundle.mie if bundle.mie is not None else config.mie_floor
    alpha = config.alpha

    # Gate 0: the signed derivation machinery + gate library are pinned. Drift -> HALT (do not score).
    verify_control_catalog(config.control_catalog_hash)
    verify_gate_library(config.gate_library_digest)

    # Gate 1: G0 detectability, the claim-type is wired, and (box-independent) the prior-claim refusal
    # -- defense in depth with precheck, so confirm is safe called directly.
    if not bundle.g0_passed:
        return Verdict("INELIGIBLE", reason="g0_failed")
    if schema.claim_type not in _WIRED_CLAIM_TYPES:
        return Verdict("INELIGIBLE", reason=f"claim_type_{schema.claim_type}_not_wired")
    if schema.prior_claim:
        return Verdict("INELIGIBLE", reason="prior_ablated_floor_not_wired")

    # The single trusted schema-normal-form runs at confirm time (derive() is LIVE, not test-only):
    # the control set and the magnitude-gate NAME are DERIVED here and drive the gates below, so no
    # phrasing draws a weaker gauntlet and the magnitude decision flows through the signed mapping.
    # (A prior claim, whose derived floor is the prior-ablated baseline, is already refused above.)
    derived = derive(schema)

    # One atomic exposure of the box: score trained + K untrained inits.
    trained = backend.score_box(box)
    untrained = [backend.score_box(box, untrained_init=k) for k in range(k_untrained)]

    # Gate 2: the untrained FLOOR (the geometry-artifact catcher) is MANDATORY for EVERY claim-type,
    # enforced HERE ahead of the type-specific magnitude. Computed first because the EFFECT magnitude
    # reuses its trained-minus-untrained residual.
    floor = floor_separation(trained, untrained, mie, alpha)

    # The magnitude quantity + p-value are gate-appropriate. For EFFECT (mie_at_power) the quantity is
    # the trained-minus-untrained CONTRAST (the FLOOR residual, a DELTA on the MIE's scale), so an
    # ABSOLUTE metric with a chance floor cannot clear a delta MIE tautologically and a null effect is
    # reachable as a powered null. Capability / phenomenon / law_shape compare the ABSOLUTE score to
    # their own signed threshold (unit-correct), so they keep the absolute CI.
    if derived.magnitude_gate == "mie_at_power":
        worst = np.asarray(untrained[floor.worst_init], dtype=float)
        residual = np.asarray(trained, dtype=float) - worst
        pval = magnitude_pvalue(residual, mie)
        kwargs, missing = {"ci_lo": floor.ci_lo, "ci_hi": floor.ci_hi, "mie": mie}, None
    else:
        missing = _missing_threshold(derived.magnitude_gate, bundle)  # box-independent, also in precheck
        if missing is not None:
            kwargs, pval = None, None
        else:
            lo, hi = mean_ci(trained, alpha)
            kwargs, threshold = _magnitude_kwargs(derived.magnitude_gate, lo, hi, bundle)
            pval = magnitude_pvalue(trained, threshold) if threshold is not None else None

    # Gate 3: backbone contamination (HARD on every positive).
    bb = backbone_check(box.origin_date, bundle.backbone_cutoff, bundle.membership_clean)
    if not bb.passed:
        return Verdict("INELIGIBLE", reason=f"backbone_{bb.label}", backbone_label=bb.label, pvalue=pval)

    # Gate 4: MAGNITUDE, dispatched by the DERIVED (signed) gate name. A threshold the substrate did
    # not measure fails closed (also caught box-independently in precheck).
    if missing is not None:
        return Verdict("INELIGIBLE", reason=f"magnitude_input_missing_{missing}",
                       backbone_label=bb.label, pvalue=pval)
    status = magnitude_gate_for(derived.magnitude_gate, **kwargs)
    if status == INCONCLUSIVE:
        return Verdict("INCONCLUSIVE", "ci_includes_mie", bb.label, pval)
    if status == POWERED_NULL:
        if bundle.believed_claim:
            return Verdict("CONFIRMED_NEGATIVE", "powered_null_on_believed_claim", bb.label, pval)
        return Verdict("FAILED", "null_on_unbelieved_claim", bb.label, pval)
    if status == FAIL:
        return Verdict("FAILED", "magnitude_not_separated", bb.label, pval)
    if status not in (EXCEEDS_MIE, PASS):
        # a disposer must not rely on `assert` (python -O strips it); an unknown status HALTs.
        raise RuntimeError(f"unexpected magnitude status {status!r} for gate {derived.magnitude_gate!r}")

    # Gate 5: FLOOR (mandatory), mechanism, novelty -- shared by every positive claim-type.
    if not floor.passed:
        return Verdict("FAILED", "floor_not_separated", bb.label, pval)
    if not mechanism_check(bundle.mech_contrast_lo, mie, bundle.specificity_ok):
        return Verdict("FAILED", "mechanism_unsupported", bb.label, pval)
    nov = novelty_check(bundle.novelty_collision, bundle.novelty_k_nearest, bundle.novelty_advance)
    if not nov.passed:
        return Verdict("FAILED", f"novelty_{nov.reason}", bb.label, pval)

    # Gate 6: the importance-consequence tier.
    if not consequence_check(bundle.consequence_confirmed, bundle.incumbent_separated):
        return Verdict("CONFIRMED_EFFECT", "consequence_not_discharged", bb.label, pval)
    if bundle.ood_holds:
        return Verdict("STRONG", "generalizes", bb.label, pval)
    return Verdict("CONFIRMED", "all_gates_passed", bb.label, pval)
