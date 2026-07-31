"""The real experiment callables behind the ExperimentSubstrate seam. Each MEASURES a gate input
so the referee never gates on an agent-authored number. These are built test-first on the Mac
with mock deps; the real model scoring and the live MCP audit swap in on the cluster.

First up: the consequence experiment. It resolves the pre-registered consequence template and the
incumbent from the SIGNED catalogs (verifying the digest, so a tampered catalog raises), computes
the incumbent separation at the MIE, and takes the held-out consequence result. This is the piece
that stops the anti-HARKing catalog from being a paper-only guarantee (red-audit #5)."""
import numpy as np
import pytest

from engine.real_experiments import (
    real_g0,
    real_mechanism,
    real_novelty,
    resolve_consequence,
)
from referee.catalog import CatalogError, catalog_digest


def test_real_g0_passes_when_the_pipeline_detects_the_planted_effect():
    pipeline = lambda effect, rng: 0.001  # always significant -> power 1.0
    assert real_g0(pipeline, mde=0.03, alpha=0.05, power_target=0.8, n_trials=50,
                   rng=np.random.default_rng(0)) is True


def test_real_g0_fails_when_the_pipeline_is_blind():
    pipeline = lambda effect, rng: 0.9  # never significant -> power 0
    assert real_g0(pipeline, mde=0.03, alpha=0.05, power_target=0.8, n_trials=50,
                   rng=np.random.default_rng(0)) is False


def test_real_mechanism_contrast_clears_mie_even_with_a_chance_floor():
    # The ablated model sits at the MCQ chance floor (~0.25, well ABOVE the 0.03 MIE), which the
    # old "ablated below the MIE" gate made unsatisfiable. The paired contrast (0.40 - 0.25) still
    # clears the MIE, so a real mechanism on a chance-floored task is now confirmable.
    rng = np.random.default_rng(0)
    full = (rng.random(400) < 0.40).astype(float)
    ablated = (rng.random(400) < 0.25).astype(float)
    contrast_lo, spec = real_mechanism(
        score_full=lambda: full, score_ablated=lambda: ablated,
        specificity_ok=True, alpha=0.05)
    assert contrast_lo > 0.03 and spec is True


def test_real_mechanism_carries_the_specificity_result():
    contrast_lo, spec = real_mechanism(
        score_full=lambda: np.full(10, 1.0), score_ablated=lambda: np.full(10, 0.0),
        specificity_ok=False, alpha=0.05)
    assert spec is False


def test_real_mechanism_rejects_unaligned_paired_scores():
    with pytest.raises(ValueError):
        real_mechanism(score_full=lambda: np.zeros(5), score_ablated=lambda: np.zeros(6),
                       specificity_ok=True, alpha=0.05)


def test_real_novelty_assembles_from_the_audit():
    # the ADVANCE comes from the audit party's return (3-tuple), never a passed-in agent flag
    collision, k_nearest, advance = real_novelty(
        {"claim": "c"}, audit_fn=lambda s: (False, ["Paper A", "Paper B"], True))
    assert collision is False and k_nearest == ["Paper A", "Paper B"] and advance is True


def test_real_novelty_fails_closed_when_the_audit_asserts_no_advance():
    # a 2-tuple audit (no advance asserted by the party) fails CLOSED to advance=False
    collision, _, advance = real_novelty({"claim": "c"}, audit_fn=lambda s: (False, ["Paper A"]))
    assert collision is False and advance is False


def test_real_novelty_flags_a_prior_art_collision():
    collision, _, _ = real_novelty({"claim": "c"}, audit_fn=lambda s: (True, ["Exact Match"], False))
    assert collision is True

CONS = {"effect": "downstream accuracy rises by >= MIE on held-out task T",
        "capability": "solves held-out instances the pre-registered incumbent fails, separated by >= MIE",
        "qualitative_phenomenon": "the phenomenon persists on a held-out split from a different source"}
INC = {"ssv2_recognition": 0.70}


def _digests():
    return dict(consequence_catalog=CONS, consequence_digest=catalog_digest(CONS),
               incumbent_catalog=INC, incumbent_digest=catalog_digest(INC))


def test_capability_consequence_separates_over_the_signed_incumbent():
    # separation uses the MEASURED held-out value, not a claim the agent authored. Only CAPABILITY.
    confirmed, separated = resolve_consequence(
        "capability", "ssv2_recognition", measured_value=0.80, mie=0.05,
        held_out_confirmed=True, **_digests())
    assert confirmed is True and separated is True   # 0.80 - 0.70 = 0.10 >= 0.05


def test_capability_not_separated_when_measured_margin_below_mie():
    confirmed, separated = resolve_consequence(
        "capability", "ssv2_recognition", measured_value=0.72, mie=0.05,
        held_out_confirmed=True, **_digests())
    assert separated is False  # measured 0.72 - incumbent 0.70 = 0.02 < 0.05


def test_effect_claim_needs_no_incumbent():
    # THE FIX: the incumbent is a capability-only concept. A non-capability consequence never resolves
    # an incumbent, so a task with NO signed incumbent does not raise and its separation leg is
    # trivially satisfied (an effect's tier is decided by its own downstream consequence, not a SOTA).
    confirmed, separated = resolve_consequence(
        "effect", "intphys2_physics_binary", measured_value=0.53, mie=0.03,
        held_out_confirmed=True, **_digests())          # intphys2 is NOT in INC
    assert confirmed is True and separated is True


def test_phenomenon_separation_is_trivially_true_regardless_of_value():
    # even a value below the incumbent does not fail a phenomenon: it never separates over one
    _, separated = resolve_consequence(
        "qualitative_phenomenon", "ssv2_recognition", measured_value=0.10, mie=0.05,
        held_out_confirmed=True, **_digests())
    assert separated is True


def test_held_out_failure_yields_unconfirmed_consequence():
    confirmed, _ = resolve_consequence(
        "effect", "ssv2_recognition", measured_value=0.80, mie=0.05,
        held_out_confirmed=False, **_digests())
    assert confirmed is False


def test_tampered_consequence_catalog_raises():
    d = _digests()
    d["consequence_digest"] = "sha256:tampered"
    with pytest.raises(CatalogError):
        resolve_consequence("effect", "ssv2_recognition", 0.80, 0.05, held_out_confirmed=True, **d)


def test_claim_type_with_no_pre_registered_template_raises():
    # anti-HARKing: a claim-type absent from the signed catalog cannot get a consequence at handoff.
    with pytest.raises(CatalogError):
        resolve_consequence("law_shape", "ssv2_recognition", 0.80, 0.05, held_out_confirmed=True, **_digests())
