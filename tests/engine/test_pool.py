"""Multi-hypothesis campaign close: the selection correction submits the real effects
and screens out the nulls, and reports the honest expected-fluke count."""
from datetime import date

from backend import Box, MockBackend
from engine import MockAgent, close_campaign, run_campaign
from engine.handoff import accept_as_proposed
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
    return Box(id=box_id, n=800, origin_date=date(2024, 6, 1))


def make_results(tmp_path):
    results = []
    c = cfg()
    for i, effect in enumerate([0.25, 0.0, 0.28, 0.0]):  # two real effects, two nulls
        ls = LeaseStore(str(tmp_path / f"c{i}.db"))
        ls.add_boxes(["b"])
        results.append(run_campaign(MockAgent(), MockBackend(effect, 0.0, 0.1, seed=i + 10),
                                    c, ls, box_factory, substrate=MockSubstrate(), triage=accept_as_proposed))
    return results


def test_selection_submits_real_effects_and_screens_nulls(tmp_path):
    report = close_campaign(make_results(tmp_path), alpha=0.05)
    assert report.n_scored == 4
    assert len(report.submitted) == 2                 # the two real effects
    assert all(r.verdict.status == "CONFIRMED" for r in report.submitted)
    assert abs(report.expected_false_family - 4 * 0.05) < 1e-9
    assert "SUBMIT" in report.narrative


def test_all_null_campaign_submits_nothing(tmp_path):
    results = []
    c = cfg()
    for i in range(3):
        ls = LeaseStore(str(tmp_path / f"n{i}.db"))
        ls.add_boxes(["b"])
        results.append(run_campaign(MockAgent(), MockBackend(0.0, 0.0, 0.1, seed=i + 50),
                                    c, ls, box_factory, substrate=MockSubstrate(), triage=accept_as_proposed))
    report = close_campaign(results, alpha=0.05)
    assert report.submitted == []
