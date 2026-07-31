"""The self-chaining supervisor. A deterministic state machine with a GUARANTEED base-case
halt: it stops the moment any one of GPU-hours, boxes, or maturations reaches its budget.
A stall backstop catches a step that stops consuming budget (a bug), a durable HALT flag lets
a human stop it out of band, and a health heartbeat halts on an integrity fault. Progress and
health are the two sides of the liveness probe."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .health import HaltError, HealthGate

_DEFAULT_STALL_LIMIT = 3

# Terminal reasons (named so callers compare a constant, not a magic literal).
BASE_CASE = "BASE_CASE"
HALTED = "HALTED"
HEALTH_HALT = "HEALTH_HALT"
BACKSTOP = "BACKSTOP"


class BoxExhausted(Exception):
    """The holdout box pool is genuinely spent (no live box left to claim). This is a PLANNED
    terminal, not a bug: the step_fn raises it so the supervisor routes to BASE_CASE, distinct
    from the bug-shaped stall BACKSTOP that pages via the halt flag. Exhaustion is a legitimate
    end to a campaign that spent its whole signed pool, so it must not look like a fault."""


# The scientific budget for a first campaign. Compute is not the limit here; this keeps the
# selection family's expected-false count under one (15 x 0.05 = 0.75) and human triage manageable.
DEFAULT_MAX_MATURATIONS = 15


@dataclass(frozen=True)
class Budget:
    max_gpu_hours: float
    max_boxes: float
    max_maturations: float

    @classmethod
    def default(cls, max_gpu_hours: float = 200.0, max_boxes: float = 120.0,
               max_maturations: float = DEFAULT_MAX_MATURATIONS) -> "Budget":
        """A sane starting budget. GPU-hours is a loose runaway guard (~3x the expected cost of the
        maturations), boxes covers the maturations plus replication/backbone reserves, and
        max_maturations is the real scientific ceiling."""
        return cls(max_gpu_hours=max_gpu_hours, max_boxes=max_boxes, max_maturations=max_maturations)

    @classmethod
    def from_closure(cls, *, live_boxes: int, reserves, max_gpu_hours: float,
                     max_maturations: float) -> "Budget":
        """Derive `max_boxes` from the live-box pool minus the signed reserves, REFUSING (raising
        `ClosureError`) if the reserves do not close against the pool. This is the campaign-start
        and ceiling-raise enforcement point: the box base case then fires while the held-back
        replication + rescore + backbone reserves still physically exist. See `engine.closure`."""
        from .closure import validate_closure
        max_boxes = validate_closure(live_boxes, reserves)
        return cls(max_gpu_hours=max_gpu_hours, max_boxes=max_boxes, max_maturations=max_maturations)


@dataclass(frozen=True)
class SupervisorState:
    gpu_hours_spent: float = 0.0
    boxes_spent: float = 0
    maturations: float = 0


class HaltFlag:
    """A durable, human-clearable stop signal. Presence of the file IS the halt."""

    def __init__(self, path):
        self._path = Path(path)

    def is_set(self) -> bool:
        return self._path.exists()

    def set(self, reason: str = "halt") -> None:
        self._path.write_text(reason or "halt")

    def reason(self):
        return self._path.read_text() if self._path.exists() else None

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()


def base_case_reached(state: SupervisorState, budget: Budget) -> bool:
    return (
        state.gpu_hours_spent >= budget.max_gpu_hours
        or state.boxes_spent >= budget.max_boxes
        or state.maturations >= budget.max_maturations
    )


def _made_progress(new: SupervisorState, old: SupervisorState) -> bool:
    return new != old


def run_supervisor(step_fn: "Callable[[SupervisorState], SupervisorState]", budget: Budget,
                   halt_flag: "HaltFlag", health_gate: HealthGate,
                   state: "SupervisorState | None" = None,
                   stall_limit: int = _DEFAULT_STALL_LIMIT) -> str:
    """Drive step_fn until a terminal condition. Returns the terminal reason:
    BASE_CASE, HALTED, HEALTH_HALT, or BACKSTOP."""
    if state is None:
        state = SupervisorState()

    if not _health_ok(health_gate, halt_flag, "preflight"):
        return HEALTH_HALT

    stalls = 0
    while True:
        if halt_flag.is_set():
            return HALTED
        if base_case_reached(state, budget):
            return BASE_CASE
        if not _health_ok(health_gate, halt_flag, "heartbeat"):
            return HEALTH_HALT

        try:
            new_state = step_fn(state)
        except BoxExhausted:
            # the pool is spent: a planned end, not a bug. Terminate BASE_CASE without paging the
            # halt flag (the stall backstop below is the bug-shaped path that does page).
            return BASE_CASE
        if _made_progress(new_state, state):
            stalls = 0
        else:
            stalls += 1
            if stalls >= stall_limit:
                halt_flag.set("stall backstop: step made no progress")
                return BACKSTOP
        state = new_state


def _health_ok(health_gate: HealthGate, halt_flag: HaltFlag, phase: str) -> bool:
    try:
        health_gate.run()
        return True
    except HaltError as e:
        halt_flag.set(f"{phase} health HALT: {e}")
        return False
