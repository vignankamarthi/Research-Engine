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


def test_real_mechanism_full_clears_mie_and_ablated_falls_below():
    lo, hi, spec = real_mechanism(
        score_full=lambda: np.full(100, 0.10), score_ablated=lambda: np.full(100, 0.0),
        specificity_ok=True, alpha=0.05)
    assert lo > 0.03 and hi < 0.03 and spec is True


def test_real_mechanism_carries_the_specificity_result():
    _, _, spec = real_mechanism(
        score_full=lambda: np.full(10, 0.1), score_ablated=lambda: np.full(10, 0.0),
        specificity_ok=False, alpha=0.05)
    assert spec is False


def test_real_novelty_assembles_from_the_audit():
    collision, k_nearest, advance = real_novelty(
        {"claim": "c"}, audit_fn=lambda s: (False, ["Paper A", "Paper B"]), advance_argued=True)
    assert collision is False and k_nearest == ["Paper A", "Paper B"] and advance is True


def test_real_novelty_flags_a_prior_art_collision():
    collision, _, _ = real_novelty(
        {"claim": "c"}, audit_fn=lambda s: (True, ["Exact Match"]), advance_argued=True)
    assert collision is True

CONS = {"effect": "downstream accuracy rises by >= MIE on held-out task T"}
INC = {"ssv2_recognition": 0.70}


def _digests():
    return dict(consequence_catalog=CONS, consequence_digest=catalog_digest(CONS),
               incumbent_catalog=INC, incumbent_digest=catalog_digest(INC))


def test_resolve_consequence_from_signed_catalogs():
    confirmed, separated = resolve_consequence(
        "effect", "ssv2_recognition", claimed_value=0.80, mie=0.05,
        held_out_confirmed=True, **_digests())
    assert confirmed is True and separated is True


def test_incumbent_not_separated_when_margin_below_mie():
    confirmed, separated = resolve_consequence(
        "effect", "ssv2_recognition", claimed_value=0.72, mie=0.05,
        held_out_confirmed=True, **_digests())
    assert separated is False  # 0.72 - 0.70 = 0.02 < 0.05


def test_held_out_failure_yields_unconfirmed_consequence():
    confirmed, _ = resolve_consequence(
        "effect", "ssv2_recognition", claimed_value=0.80, mie=0.05,
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
