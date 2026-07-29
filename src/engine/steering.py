"""Steering above the raw Optuna bandit. The bandit supplies the acquisition; this layer
enforces the design's balance-on-the-margin: a lead-thesis depth floor, a capped breadth floor,
a high-patience breadth slot insulated from early pruning, a negative-bank dead-end read, and a
cheap pre-gauntlet maturity screen."""
from __future__ import annotations

from dataclasses import dataclass

DEPTH = "depth"
BREADTH = "breadth"


@dataclass(frozen=True)
class SteeringPolicy:
    lead_thesis_floor: int = 2   # min decisive experiments the lead thesis is guaranteed
    breadth_floor: int = 1       # min breadth explorations guaranteed
    breadth_cap: int = 5         # max breadth, so depth is never starved
    high_patience_slots: int = 1 # leading breadth slots insulated from early-signal pruning


def choose_vein(bandit_pick: str, depth_count: int, breadth_count: int,
                policy: SteeringPolicy) -> str:
    """Balance on the margin: honor the depth floor first, then the breadth floor, then the
    breadth cap, and only inside those bounds follow the bandit's pick."""
    if depth_count < policy.lead_thesis_floor:
        return DEPTH
    if breadth_count < policy.breadth_floor:
        return BREADTH
    if breadth_count >= policy.breadth_cap:
        return DEPTH
    return bandit_pick


def is_high_patience(breadth_slot_index: int, policy: SteeringPolicy) -> bool:
    return breadth_slot_index < policy.high_patience_slots


def should_prune(breadth_slot_index: int, early_signal_weak: bool, policy: SteeringPolicy) -> bool:
    """A high-patience slot is never pruned on early signal (that is the point of the slot).
    Any other slot is pruned when its early signal is weak."""
    if is_high_patience(breadth_slot_index, policy):
        return False
    return early_signal_weak


def is_dead_end(lineage_key: str, negative_bank) -> bool:
    """A lineage already in the negative bank is a dead end; discovery skips it before spending."""
    return lineage_key in negative_bank


def passes_cheap_gate(g0_ok: bool, floor_ok: bool, arch_ok: bool, executed_ok: bool) -> bool:
    """The standing cheap sanity screen on an exploratory run. All four must pass before the
    expensive confirmatory gauntlet is spent: G0 detectability, the untrained FLOOR, the arch
    control, and executed-not-fabricated."""
    return bool(g0_ok and floor_ok and arch_ok and executed_ok)
