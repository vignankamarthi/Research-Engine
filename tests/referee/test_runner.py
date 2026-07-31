"""End-to-end confirmatory trial run on synthetic data. The runner composes the
ordered gauntlet (G0 -> FLOOR -> backbone -> magnitude -> mechanism -> consequence
-> novelty -> score-once) into a verdict, driven by a MockBackend. This is the
Milestone-3 plumbing check: every terminal state reached without touching the
cluster or a real checkpoint."""
from datetime import date

from backend import Box, MockBackend
from engine.agents import Bundle
from gateconfig import validate_config
from gatelib import library_digest
from referee import normalize_schema
from referee.lineage import control_catalog_digest
from referee.runner import Verdict, confirm


def cfg():
    return validate_config({
        "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.01, "mie_floor": 0.03,
        "claim_types": ["effect"], "gate_library_digest": library_digest(),
        "control_catalog_hash": control_catalog_digest(), "key_id": "test",
    })


def schema(**over):
    raw = {"claim": "x improves recognition", "claim_type": "effect", "backbone": "iv2",
           "dataset": "ssv2", "scale": "7b", "measure": "accuracy", "prior_claim": False}
    raw.update(over)
    return normalize_schema(raw)


POST_CUTOFF_BOX = Box(id="b", n=800, origin_date=date(2024, 6, 1))


def test_clean_genuine_effect_is_confirmed():
    be = MockBackend(trained_effect=0.25, untrained_effect=0.0, noise=0.1, seed=1)
    v = confirm(be, POST_CUTOFF_BOX, schema(), cfg(), Bundle.passing())
    assert v.status == "CONFIRMED"


def test_chance_floored_task_can_still_reach_confirmed():
    # The mechanism-ablated model sits at the MCQ chance floor (~0.25, well ABOVE the 0.03 MIE),
    # which the old "ablated below the MIE" gate made unsatisfiable, so CONFIRMED was unreachable on
    # any chance-floored benchmark. With the paired-contrast contract the run still reaches CONFIRMED.
    import numpy as np

    from engine.real_experiments import real_mechanism
    rng = np.random.default_rng(3)
    full = (rng.random(600) < 0.40).astype(float)
    ablated = (rng.random(600) < 0.25).astype(float)  # chance floor, NOT below the MIE
    contrast_lo, spec = real_mechanism(score_full=lambda: full, score_ablated=lambda: ablated,
                                       specificity_ok=True, alpha=0.05)
    assert contrast_lo > 0.03  # the ablated chance floor is now irrelevant to the mechanism gate
    be = MockBackend(trained_effect=0.25, untrained_effect=0.0, noise=0.1, seed=1)
    v = confirm(be, POST_CUTOFF_BOX, schema(), cfg(), Bundle.passing(mech_contrast_lo=contrast_lo))
    assert v.status == "CONFIRMED"


def test_backbone_fails_closed_when_cutoff_unset():
    # g0 populated from a real experiment, but the backbone fields left at defaults. The
    # fail-closed backbone_cutoff (date.max) must make the box read as contaminated, not pass.
    be = MockBackend(0.25, 0.0, 0.1, seed=1)
    v = confirm(be, POST_CUTOFF_BOX, schema(), cfg(), Bundle(g0_passed=True))
    assert v.status == "INELIGIBLE" and "backbone" in v.reason


def test_bare_bundle_fails_closed():
    # a bare Bundle() leaves every substrate gate unpopulated; the referee must NOT pass it.
    # Even with a clean, strong effect, an unproven G0 makes it ineligible.
    be = MockBackend(trained_effect=0.25, untrained_effect=0.0, noise=0.1, seed=1)
    from engine.agents import Bundle as _B
    v = confirm(be, POST_CUTOFF_BOX, schema(), cfg(), _B())
    assert v.status == "INELIGIBLE"


def test_tampered_control_catalog_halts_before_scoring():
    import pytest

    from referee.lineage import ControlCatalogError

    bad = validate_config({
        "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.01, "mie_floor": 0.03,
        "claim_types": ["effect"], "gate_library_digest": library_digest(),
        "control_catalog_hash": "sha256:" + "9" * 64, "key_id": "test",  # wrong digest
    })
    be = MockBackend(0.25, 0.0, 0.1, seed=1)
    with pytest.raises(ControlCatalogError):
        confirm(be, POST_CUTOFF_BOX, schema(), bad, Bundle.passing())


def test_tampered_gate_library_digest_halts():
    import pytest

    from gatelib import GateLibraryError

    bad = validate_config({
        "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.01, "mie_floor": 0.03,
        "claim_types": ["effect"], "gate_library_digest": "sha256:" + "9" * 64,  # wrong
        "control_catalog_hash": control_catalog_digest(), "key_id": "test",
    })
    be = MockBackend(0.25, 0.0, 0.1, seed=1)
    with pytest.raises(GateLibraryError):
        confirm(be, POST_CUTOFF_BOX, schema(), bad, Bundle.passing())


def test_ood_generalization_is_strong():
    be = MockBackend(0.25, 0.0, 0.1, seed=1)
    v = confirm(be, POST_CUTOFF_BOX, schema(), cfg(), Bundle.passing(ood_holds=True))
    assert v.status == "STRONG"


def test_consequence_not_discharged_is_confirmed_effect():
    be = MockBackend(0.25, 0.0, 0.1, seed=1)
    v = confirm(be, POST_CUTOFF_BOX, schema(), cfg(), Bundle.passing(consequence_confirmed=False))
    assert v.status == "CONFIRMED_EFFECT"


def test_geometry_artifact_effect_is_a_powered_null_not_a_false_positive():
    # a believed EFFECT fully reproduced by the untrained model -> the trained-minus-untrained
    # CONTRAST is ~0 -> a powered null on the effect (CONFIRMED_NEGATIVE), NEVER a CONFIRMED positive.
    # The Temporal-RoPE lesson, caught end to end via the contrast rather than the absolute level.
    be = MockBackend(trained_effect=0.25, untrained_effect=0.25, noise=0.1, seed=2)
    v = confirm(be, POST_CUTOFF_BOX, schema(), cfg(), Bundle.passing())
    assert v.status == "CONFIRMED_NEGATIVE"


def test_capability_geometry_artifact_fails_on_the_floor():
    # a CAPABILITY whose ABSOLUTE score clears the incumbent but is reproduced by the untrained model
    # -> magnitude passes, the FLOOR catches it -> FAILED/floor (the capability is geometry, not learned).
    be = MockBackend(trained_effect=0.85, untrained_effect=0.85, noise=0.05, seed=2)
    v = confirm(be, POST_CUTOFF_BOX, schema(claim_type="capability"), cfg(), Bundle.passing())
    assert v.status == "FAILED" and "floor" in v.reason


def test_contaminated_backbone_is_ineligible():
    be = MockBackend(0.25, 0.0, 0.1, seed=1)
    pre_cutoff = Box(id="old", n=800, origin_date=date(2022, 1, 1))
    v = confirm(be, pre_cutoff, schema(), cfg(), Bundle.passing())
    assert v.status == "INELIGIBLE" and "backbone" in v.reason


def test_underpowered_effect_is_inconclusive():
    # a small effect on a small box: the CI straddles the MIE
    be = MockBackend(trained_effect=0.03, untrained_effect=0.0, noise=0.2, seed=3)
    small_box = Box(id="s", n=60, origin_date=date(2024, 6, 1))
    v = confirm(be, small_box, schema(), cfg(), Bundle.passing())
    assert v.status == "INCONCLUSIVE"


def test_powered_null_on_a_believed_claim_is_confirmed_negative():
    be = MockBackend(trained_effect=0.0, untrained_effect=0.0, noise=0.1, seed=4)
    v = confirm(be, POST_CUTOFF_BOX, schema(), cfg(), Bundle.passing(believed_claim=True))
    assert v.status == "CONFIRMED_NEGATIVE"


def test_g0_failure_is_ineligible():
    be = MockBackend(0.25, 0.0, 0.1, seed=1)
    v = confirm(be, POST_CUTOFF_BOX, schema(), cfg(), Bundle.passing(g0_passed=False))
    assert v.status == "INELIGIBLE" and "g0" in v.reason


def test_novelty_collision_fails():
    be = MockBackend(0.25, 0.0, 0.1, seed=1)
    v = confirm(be, POST_CUTOFF_BOX, schema(), cfg(), Bundle.passing(novelty_collision=True))
    assert v.status == "FAILED" and "novelty" in v.reason


def test_unwired_claim_type_is_ineligible_not_silently_borrowed():
    # effect/capability/phenomenon/law_shape are now WIRED; multi_benchmark_superiority stays deferred
    # and must fail LOUD (INELIGIBLE), never silently borrow another gauntlet.
    be = MockBackend(0.25, 0.0, 0.1, seed=1)
    v = confirm(be, POST_CUTOFF_BOX, schema(claim_type="multi_benchmark_superiority"), cfg(),
                Bundle.passing())
    assert v.status == "INELIGIBLE" and "multi_benchmark_superiority_not_wired" in v.reason


def test_verdict_is_a_dataclass_with_status():
    assert Verdict(status="CONFIRMED").status == "CONFIRMED"


# --- step 53: the capability / phenomenon / law_shape gauntlets, dispatched by the signed derive() ---
def test_capability_claim_separates_above_the_incumbent_and_confirms():
    be = MockBackend(0.85, 0.0, 0.05, seed=1)  # trained ~0.85 accuracy, above the 0.70 incumbent
    v = confirm(be, POST_CUTOFF_BOX, schema(claim_type="capability"), cfg(), Bundle.passing())
    assert v.status == "CONFIRMED"


def test_capability_claim_below_the_incumbent_fails():
    be = MockBackend(0.50, 0.0, 0.05, seed=1)  # trained ~0.50, below the 0.70 incumbent
    v = confirm(be, POST_CUTOFF_BOX, schema(claim_type="capability"), cfg(), Bundle.passing())
    assert v.status == "FAILED" and "magnitude_not_separated" in v.reason


def test_capability_without_a_measured_incumbent_rate_is_ineligible():
    be = MockBackend(0.85, 0.0, 0.05, seed=1)
    v = confirm(be, POST_CUTOFF_BOX, schema(claim_type="capability"), cfg(),
                Bundle.passing(incumbent_rate=None))
    assert v.status == "INELIGIBLE" and "incumbent_rate" in v.reason


def test_phenomenon_claim_separates_above_the_null_and_confirms():
    be = MockBackend(0.60, 0.0, 0.05, seed=1)  # observed rate ~0.60, above the 0.25 null
    v = confirm(be, POST_CUTOFF_BOX, schema(claim_type="qualitative_phenomenon"), cfg(),
                Bundle.passing())
    assert v.status == "CONFIRMED"


def test_phenomenon_without_a_measured_baseline_rate_is_ineligible():
    be = MockBackend(0.60, 0.0, 0.05, seed=1)
    v = confirm(be, POST_CUTOFF_BOX, schema(claim_type="qualitative_phenomenon"), cfg(),
                Bundle.passing(baseline_rate=None))
    assert v.status == "INELIGIBLE" and "baseline_rate" in v.reason


def test_law_shape_fit_within_tolerance_confirms():
    be = MockBackend(0.85, 0.0, 0.05, seed=1)
    b = Bundle.passing(law_predicted=(0.1, 0.2, 0.3), law_observed=(0.1, 0.21, 0.29), law_tol=0.05)
    v = confirm(be, POST_CUTOFF_BOX, schema(claim_type="law_shape"), cfg(), b)
    assert v.status == "CONFIRMED"


def test_law_shape_over_the_24h_envelope_is_ineligible_at_precheck():
    from referee.runner import precheck
    # law_* series MEASURED but the sweep is over the envelope -> the envelope refusal
    b = Bundle.passing(law_predicted=(0.1, 0.2), law_observed=(0.1, 0.2), law_tol=0.05,
                       law_within_envelope=False)
    v = precheck(schema(claim_type="law_shape"), cfg(), b)
    assert v is not None and v.status == "INELIGIBLE" and "envelope" in v.reason


def test_law_shape_unmeasured_inputs_are_ineligible_at_precheck_before_a_box():
    # law_* series NOT measured -> the split reason `magnitude_input_missing_law_shape_series`,
    # NOT the mislabeled `over_24h_envelope`, and refused box-independently in precheck.
    from referee.runner import precheck
    v = precheck(schema(claim_type="law_shape"), cfg(), Bundle.passing(law_within_envelope=True))
    assert v is not None and v.status == "INELIGIBLE" and "law_shape_series" in v.reason


def test_capability_without_incumbent_rate_is_ineligible_at_precheck_no_box_burned():
    # the missing-threshold refusal is box-INDEPENDENT, so it fires in precheck before any lease
    from referee.runner import precheck
    v = precheck(schema(claim_type="capability"), cfg(), Bundle.passing(incumbent_rate=None))
    assert v is not None and v.status == "INELIGIBLE" and "incumbent_rate" in v.reason


def test_prior_claim_is_ineligible_until_the_prior_ablated_floor_is_wired():
    be = MockBackend(0.25, 0.0, 0.1, seed=1)
    v = confirm(be, POST_CUTOFF_BOX, schema(prior_claim=True), cfg(), Bundle.passing())
    assert v.status == "INELIGIBLE" and "prior_ablated_floor" in v.reason
