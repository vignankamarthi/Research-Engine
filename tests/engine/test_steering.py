"""The steering layer above the raw bandit. It guarantees the lead thesis a minimum number of
decisive experiments (depth floor) before breadth can dominate, guarantees a minimum breadth but
caps it so depth is never starved, keeps a high-patience breadth slot insulated from early-signal
pruning, reads the negative bank to skip dead ends, and gates the expensive gauntlet behind a
cheap four-check maturity screen."""
from engine.steering import (
    BREADTH,
    DEPTH,
    SteeringPolicy,
    choose_vein,
    is_dead_end,
    passes_cheap_gate,
    should_prune,
)

POL = SteeringPolicy(lead_thesis_floor=2, breadth_floor=1, breadth_cap=3, high_patience_slots=1)


def test_forces_depth_until_lead_thesis_floor_met():
    assert choose_vein("breadth", depth_count=0, breadth_count=0, policy=POL) == DEPTH
    assert choose_vein("breadth", depth_count=1, breadth_count=0, policy=POL) == DEPTH


def test_forces_breadth_until_breadth_floor_met():
    assert choose_vein("depth", depth_count=2, breadth_count=0, policy=POL) == BREADTH


def test_caps_breadth_to_protect_depth():
    assert choose_vein("breadth", depth_count=2, breadth_count=3, policy=POL) == DEPTH


def test_follows_the_bandit_within_the_floors_and_cap():
    assert choose_vein("breadth", depth_count=2, breadth_count=1, policy=POL) == BREADTH
    assert choose_vein("depth", depth_count=2, breadth_count=1, policy=POL) == DEPTH


def test_high_patience_slot_is_insulated_from_pruning():
    assert should_prune(0, early_signal_weak=True, policy=POL) is False   # protected slot
    assert should_prune(1, early_signal_weak=True, policy=POL) is True    # unprotected + weak -> prune


def test_dead_end_read_from_the_negative_bank():
    assert is_dead_end("k1", {"k1", "k2"}) is True
    assert is_dead_end("k3", {"k1"}) is False


def test_cheap_pre_gauntlet_gate_requires_all_four_checks():
    assert passes_cheap_gate(True, True, True, True) is True
    assert passes_cheap_gate(True, False, True, True) is False
    assert passes_cheap_gate(False, True, True, True) is False
