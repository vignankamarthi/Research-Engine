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


def test_geometry_artifact_fails_on_the_floor():
    # trained effect fully reproduced by the untrained model -> FLOOR fails even though
    # the magnitude clears the MIE. The Temporal-RoPE lesson, caught end to end.
    be = MockBackend(trained_effect=0.25, untrained_effect=0.25, noise=0.1, seed=2)
    v = confirm(be, POST_CUTOFF_BOX, schema(), cfg(), Bundle.passing())
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


def test_unwired_claim_type_is_ineligible_not_silently_effect():
    # a non-EFFECT claim must NOT silently draw the effect gauntlet; it fails loud until wired.
    be = MockBackend(0.25, 0.0, 0.1, seed=1)
    v = confirm(be, POST_CUTOFF_BOX, schema(claim_type="capability"), cfg(), Bundle.passing())
    assert v.status == "INELIGIBLE" and "capability_not_wired" in v.reason


def test_verdict_is_a_dataclass_with_status():
    assert Verdict(status="CONFIRMED").status == "CONFIRMED"
