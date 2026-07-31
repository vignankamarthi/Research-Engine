"""End-to-end campaign SMOKE test: the whole machine runs on the Mac with a mock
agent and a MockBackend. A scout proposes a hypothesis, it matures, the human-triage
touchpoint accepts, a box is leased, the confirmatory gauntlet scores it, the verdict
commits, and a narrative is drafted. This exercises the real pipeline (not a unit),
proving the plumbing before real agents and a real model swap in."""
from datetime import date

from backend import Box, MockBackend
from engine import MockAgent, run_campaign
from engine.handoff import TriageDecision, accept_as_proposed
from engine.substrate import MockSubstrate
from gateconfig import validate_config
from gatelib import library_digest
from referee.lease import LeaseStore
from referee.lineage import control_catalog_digest


def cfg():
    return validate_config({
        "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.01, "mie_floor": 0.03,
        "claim_types": ["effect"], "gate_library_digest": library_digest(),
        "control_catalog_hash": control_catalog_digest(), "key_id": "test",
    })


def box_factory(box_id):
    return Box(id=box_id, n=800, origin_date=date(2024, 6, 1))  # post backbone cutoff


def test_end_to_end_campaign_confirms(tmp_path):
    ls = LeaseStore(str(tmp_path / "campaign.db"))
    ls.add_boxes(["box0", "box1"])
    res = run_campaign(MockAgent(), MockBackend(0.25, 0.0, 0.1, seed=1), cfg(), ls, box_factory, substrate=MockSubstrate(), triage=accept_as_proposed)
    assert res.verdict is not None and res.verdict.status == "CONFIRMED"
    assert "Thesis" in res.narrative
    assert ls.box_status("box0") == "spent"          # the box was leased and committed once
    assert ls.bank_verdict(res.lineage)[2] == "spent"  # one-grant record written


def test_catalog_drift_does_not_burn_a_box(tmp_path):
    import pytest

    from referee.lineage import ControlCatalogError

    ls = LeaseStore(str(tmp_path / "campaign.db"))
    ls.add_boxes(["box0"])
    bad = validate_config({
        "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.01, "mie_floor": 0.03,
        "claim_types": ["effect"], "gate_library_digest": library_digest(),
        "control_catalog_hash": "sha256:" + "9" * 64, "key_id": "test",  # drifted
    })
    with pytest.raises(ControlCatalogError):
        run_campaign(MockAgent(), MockBackend(0.25, 0.0, 0.1, seed=1), bad, ls, box_factory, substrate=MockSubstrate(), triage=accept_as_proposed)
    assert ls.box_status("box0") == "live"  # precheck raised before any claim -- box intact


def test_triage_rejection_spends_no_box(tmp_path):
    ls = LeaseStore(str(tmp_path / "campaign.db"))
    ls.add_boxes(["box0"])
    res = run_campaign(MockAgent(), MockBackend(0.25, 0.0, 0.1, seed=1), cfg(), ls,
                       box_factory, substrate=MockSubstrate(),
                       triage=lambda dossier: TriageDecision(accept=False))
    assert res.verdict is None
    assert ls.box_status("box0") == "live"  # rejected at triage, no box touched


def test_geometry_artifact_campaign_is_not_confirmed_positive(tmp_path):
    ls = LeaseStore(str(tmp_path / "campaign.db"))
    ls.add_boxes(["box0"])
    # untrained reproduces the effect -> the trained-minus-untrained CONTRAST is a powered null, so a
    # believed effect claim is CONFIRMED_NEGATIVE, never a false CONFIRMED positive.
    res = run_campaign(MockAgent(), MockBackend(0.25, 0.25, 0.1, seed=2), cfg(), ls, box_factory, substrate=MockSubstrate(), triage=accept_as_proposed)
    assert res.verdict.status == "CONFIRMED_NEGATIVE"


def test_human_committed_claim_type_drives_the_schema(tmp_path):
    # the human PICKS the claim-type at triage; the loop uses the human's choice, not the proposal's
    ls = LeaseStore(str(tmp_path / "campaign.db"))
    ls.add_boxes(["box0"])
    res = run_campaign(
        MockAgent(), MockBackend(0.85, 0.0, 0.05, seed=1), cfg(), ls, box_factory,
        substrate=MockSubstrate(), triage=lambda d: TriageDecision(accept=True, claim_type="capability"))
    # MockAgent proposes an EFFECT claim; the human committed CAPABILITY, and the Schema follows.
    assert res.schema.claim_type == "capability"
    assert res.verdict is not None and res.verdict.status == "CONFIRMED"  # 0.85 clears the 0.70 incumbent


def test_dossier_carries_the_significance_case_to_the_human(tmp_path):
    # the human triages on a NEUTRAL dossier that carries the significance-adversary's strongest case,
    # not the framing agent's story alone (the hollow-dossier finding).
    from engine.discovery_roles import MockSignificanceAdversary
    captured = {}

    def capture(dossier):
        captured["case"] = dossier.significance_case
        return TriageDecision(accept=False)

    ls = LeaseStore(str(tmp_path / "d.db"))
    ls.add_boxes(["box0"])
    run_campaign(MockAgent(), MockBackend(0.25, 0.0, 0.1, seed=1), cfg(), ls, box_factory,
                 substrate=MockSubstrate(), triage=capture, significance=MockSignificanceAdversary())
    assert captured["case"] == "novel enough"
