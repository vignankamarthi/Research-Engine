"""The box-budget CLOSURE validator. The physical live-box pool must be large enough to cover
the full signed demand BEFORE a campaign starts (per-maturation primary demand + a per-family
replication reserve + a correlated re-score-and-burn contingency + a backbone-cohort reserve),
and `Budget.max_boxes` is the reduced ceiling (live boxes minus the HELD-BACK reserves) so the
base case fires while those reserves still exist. An over-subscribed pool refuses to start, a
ceiling raise re-runs the same validator, and genuine box EXHAUSTION is a planned BASE_CASE, not
the bug-shaped stall BACKSTOP."""
import pytest

from engine.closure import (
    ClosureError,
    Reserves,
    closes,
    derive_max_boxes,
    validate_closure,
)
from engine.health import HealthGate
from engine.supervisor import (
    BoxExhausted,
    Budget,
    HaltFlag,
    SupervisorState,
    run_supervisor,
)
from referee.lease import LeaseStore


def _ok_gate():
    return HealthGate([])


# ------------------------------ the reserve accounting ------------------------------

def test_reserves_total_and_held_back_split():
    r = Reserves(primary_demand=5, replication=5, rescore=2, backbone=2)
    # the four categories sum to the full demand the pool must cover ...
    assert r.total_demand() == 14
    # ... but only the three reserves BEYOND the consumable primary allotment are held back,
    # so the reduced ceiling still lets the primary demand be spent.
    assert r.held_back() == 9


def test_for_campaign_sizes_from_maturations():
    r = Reserves.for_campaign(5)
    assert r.primary_demand == 5          # one primary box per planned maturation
    assert r.replication == 5             # one mandatory replication box per maturation
    assert r.rescore >= 1 and r.backbone >= 1
    assert r.total_demand() == 5 + 5 + r.rescore + r.backbone


# ------------------------------ closure accepts / refuses ------------------------------

def test_reserves_close_accepts_and_returns_reduced_ceiling():
    r = Reserves(primary_demand=5, replication=5, rescore=2, backbone=2)  # total 14
    live = 20
    assert closes(live, r)
    max_boxes = validate_closure(live, r)          # does not raise
    assert max_boxes == derive_max_boxes(live, r)  # live minus the held-back reserves
    assert max_boxes == 20 - 9


def test_reserves_oversubscribed_refuses_to_start():
    r = Reserves(primary_demand=10, replication=10, rescore=2, backbone=2)  # total 24
    live = 20  # the pool cannot cover the signed demand
    assert not closes(live, r)
    with pytest.raises(ClosureError):
        validate_closure(live, r)


def test_exactly_tight_closes_and_ceiling_equals_primary_demand():
    r = Reserves(primary_demand=6, replication=6, rescore=2, backbone=2)  # total 16
    max_boxes = validate_closure(16, r)  # exactly tight closes
    # with the pool tight, the reduced ceiling is exactly the primary allotment, and the
    # held-back reserves (replication + rescore + backbone) are what remains.
    assert max_boxes == r.primary_demand == 6


def test_max_boxes_holds_reserves_back():
    r = Reserves(primary_demand=5, replication=5, rescore=2, backbone=2)
    live = 20
    max_boxes = derive_max_boxes(live, r)
    # the base case fires at max_boxes; the difference to the physical pool is exactly the
    # reserve that survives to replicate / re-score / floor-control the findings.
    assert live - max_boxes == r.held_back()


# ------------------------------ ceiling raise re-validates ------------------------------

def test_ceiling_raise_reruns_validator_and_can_refuse():
    live = 20
    small = Reserves.for_campaign(5)
    validate_closure(live, small)  # the first campaign closes
    # a raise to a bigger maturation ceiling grows every reserve; re-running the SAME validator
    # against the unchanged pool now refuses, because the accounting no longer balances.
    bigger = Reserves.for_campaign(12)
    assert not closes(live, bigger)
    with pytest.raises(ClosureError):
        validate_closure(live, bigger)


def test_ceiling_raise_closes_when_the_pool_is_grown_too():
    small = Reserves.for_campaign(5)
    validate_closure(20, small)
    bigger = Reserves.for_campaign(12)
    # growing the physical pool alongside the ceiling restores closure.
    max_boxes = validate_closure(bigger.total_demand() + 3, bigger)
    assert max_boxes == derive_max_boxes(bigger.total_demand() + 3, bigger)


# ------------------------------ Budget.from_closure ------------------------------

def test_budget_from_closure_derives_max_boxes():
    r = Reserves(primary_demand=5, replication=5, rescore=2, backbone=2)
    b = Budget.from_closure(live_boxes=20, reserves=r, max_gpu_hours=200.0, max_maturations=5)
    assert b.max_boxes == derive_max_boxes(20, r)
    assert b.max_gpu_hours == 200.0
    assert b.max_maturations == 5


def test_budget_from_closure_refuses_oversubscribed_pool():
    r = Reserves(primary_demand=10, replication=10, rescore=2, backbone=2)
    with pytest.raises(ClosureError):
        Budget.from_closure(live_boxes=20, reserves=r, max_gpu_hours=200.0, max_maturations=10)


# ------------------------------ live-box count ground truth ------------------------------

def test_lease_live_count_reflects_pool_and_claims(tmp_path):
    s = LeaseStore(str(tmp_path / "lease.db"))
    s.add_boxes(["b0", "b1", "b2"])
    assert s.live_count() == 3
    s.claim(hypothesis="h1", lineage="L1")  # reserves one box -> no longer live
    assert s.live_count() == 2


# ------------------------------ box exhaustion -> BASE_CASE ------------------------------

def test_box_exhaustion_routes_to_base_case_not_backstop(tmp_path):
    budget = Budget(max_gpu_hours=1e9, max_boxes=1e9, max_maturations=1e9)  # no ceiling reached
    flag = HaltFlag(tmp_path / "halt")
    calls = {"n": 0}

    def step(s):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise BoxExhausted("no live box left")  # the pool is genuinely spent
        return SupervisorState(s.gpu_hours_spent, s.boxes_spent + 1, s.maturations)

    assert run_supervisor(step, budget, flag, _ok_gate()) == "BASE_CASE"
    # exhaustion is a planned terminal, not a bug, so it must NOT page via the halt flag
    # (contrast the stall backstop, which sets the flag).
    assert not flag.is_set()


def test_stall_still_backstops_when_not_exhaustion(tmp_path):
    budget = Budget(1e9, 5, 1e9)
    flag = HaltFlag(tmp_path / "halt")
    # a step that neither advances nor signals exhaustion is bug-shaped -> BACKSTOP + pages.
    assert run_supervisor(lambda s: s, budget, flag, _ok_gate()) == "BACKSTOP"
    assert flag.is_set()
