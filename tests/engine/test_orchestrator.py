"""The assembled discovery loop (the TOP LAYER) run end to end on the Mac with mocks: the bandit +
steering pick veins, each tick runs one confirm cycle through the frozen referee, the negative bank
accumulates, and the supervisor halts at the maturation budget, after which the campaign closes with
the selection correction. Proves the whole two-tier loop is wired and terminates, before the real
substrate + real agent + real roles swap in on the cluster."""
from datetime import date

from backend import Box, MockBackend
from engine import MockAgent
from engine.discovery_roles import MockReviewerAdversary
from engine.handoff import accept_as_proposed
from engine.health import HealthGate
from engine.orchestrator import VEINS, run_loop
from engine.substrate import MockSubstrate
from engine.supervisor import BASE_CASE, Budget, HaltFlag
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


class _VaryingAgent:
    """A MockAgent that varies its claim each tick, so successive ticks hash to DISTINCT lineages
    (and therefore distinct boxes) instead of the same-claim one-grant barring every tick after one."""

    def __init__(self):
        self._base = MockAgent()
        self._n = 0

    def propose(self, context):
        self._n += 1
        c = dict(self._base.propose(context)[0])
        c["claim"] = f"{c['claim']} :: idea {self._n}"
        return [c]

    def mature(self, schema_raw):
        return self._base.mature(schema_raw)

    def frame(self, schema_raw, verdict):
        return self._base.frame(schema_raw, verdict)


def _box_factory(box_id):
    return Box(id=box_id, n=800, origin_date=date(2024, 6, 1))


def test_assembled_loop_runs_to_the_maturation_budget_and_closes(tmp_path):
    ls = LeaseStore(str(tmp_path / "loop.db"))
    ls.add_boxes([f"b{i}" for i in range(10)])
    reason, report, campaign = run_loop(
        agent=_VaryingAgent(), backend=MockBackend(0.25, 0.0, 0.1, seed=1), config=cfg(),
        lease_store=ls, box_factory=_box_factory, substrate=MockSubstrate(),
        triage=accept_as_proposed, reviewer=MockReviewerAdversary(),
        budget=Budget.default(max_gpu_hours=100, max_boxes=8, max_maturations=4),
        halt_flag=HaltFlag(str(tmp_path / "halt")), health_gate=HealthGate([]), seed=0)

    assert reason == BASE_CASE                         # halted at the maturation ceiling, not a bug
    assert campaign.results                            # multiple ideas were actually run
    assert report.n_scored >= 4                        # the selection family reached the budget
    assert any(r.verdict is not None and r.verdict.status == "CONFIRMED" for r in campaign.results)


def test_loop_vein_arm_space_is_the_full_diversity_axis():
    # nine veins (six derivative + three generative) are the bandit's arms, so the search spreads
    assert len(VEINS) == 9


def test_reviewer_rejection_blocks_maturation_and_spends_no_box(tmp_path):
    class _Reject(MockReviewerAdversary):
        def review(self, schema_raw, evidence):
            from engine.discovery_roles import ReviewerVerdict
            return ReviewerVerdict(survives=False, objections=["always objects"])

    ls = LeaseStore(str(tmp_path / "loop.db"))
    ls.add_boxes(["b0", "b1"])
    reason, report, campaign = run_loop(
        agent=_VaryingAgent(), backend=MockBackend(0.25, 0.0, 0.1, seed=1), config=cfg(),
        lease_store=ls, box_factory=_box_factory, substrate=MockSubstrate(),
        triage=accept_as_proposed, reviewer=_Reject(),
        budget=Budget.default(max_gpu_hours=12, max_boxes=8, max_maturations=4),
        halt_flag=HaltFlag(str(tmp_path / "halt")), health_gate=HealthGate([]), seed=0)
    # nothing matures past the reviewer, so no box is spent; the loop runs out its gpu-hour guard
    assert report.n_scored == 0
    assert ls.box_status("b0") == "live"


def test_loop_resumes_from_the_durable_bank_without_rescoring(tmp_path):
    # a crash-restart on the SAME lease db must continue from the durable maturation count and NEVER
    # re-score a box the first run already spent (the one-grant bars re-claiming a spent lineage).
    db = str(tmp_path / "resume.db")
    ls1 = LeaseStore(db)
    ls1.add_boxes([f"b{i}" for i in range(14)])
    run_loop(agent=_VaryingAgent(), backend=MockBackend(0.25, 0.0, 0.1, seed=1), config=cfg(),
             lease_store=ls1, box_factory=_box_factory, substrate=MockSubstrate(),
             triage=accept_as_proposed, reviewer=MockReviewerAdversary(),
             budget=Budget.default(max_gpu_hours=100, max_boxes=100, max_maturations=3),
             halt_flag=HaltFlag(str(tmp_path / "h1")), health_gate=HealthGate([]), seed=0)
    _, matur1 = ls1.counts()
    assert matur1 == 3
    spent = {b for b in [f"b{i}" for i in range(14)] if ls1.box_status(b) == "spent"}

    ls2 = LeaseStore(db)  # RESUME on the same durable store, higher ceiling
    run_loop(agent=_VaryingAgent(), backend=MockBackend(0.25, 0.0, 0.1, seed=2), config=cfg(),
             lease_store=ls2, box_factory=_box_factory, substrate=MockSubstrate(),
             triage=accept_as_proposed, reviewer=MockReviewerAdversary(),
             budget=Budget.default(max_gpu_hours=100, max_boxes=100, max_maturations=5),
             halt_flag=HaltFlag(str(tmp_path / "h2")), health_gate=HealthGate([]), seed=1)
    _, matur2 = ls2.counts()
    assert matur2 == 5                                   # continued 3 -> 5, did not restart at 0
    for b in spent:
        assert ls2.box_status(b) == "spent"             # first-run boxes never re-scored
