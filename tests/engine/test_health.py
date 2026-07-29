"""The tool-ledger four-state health gate: the deterministic fail-loud response.
HALT (disposer/integrity) stops immediately, needs human ack. RETRY / QUARANTINE
attempt a bounded self-heal then promote to HALT. DEGRADE (observability) warns and
continues. No silent fallback: a broken disposer either self-heals or HALTs."""
import pytest

from engine.health import DEGRADE, HALT, QUARANTINE, RETRY, HaltError, HealthGate, Probe


def ok():
    return True


def bad():
    return False


def test_all_healthy_reports_ok():
    gate = HealthGate([Probe("a", ok, RETRY), Probe("b", ok, HALT)])
    assert gate.run() == {"a": "ok", "b": "ok"}


def test_retry_self_heals_within_bound():
    state = {"n": 0}
    gate = HealthGate([Probe(
        "x", check=lambda: state["n"] >= 2, state=RETRY,
        self_heal=lambda: state.__setitem__("n", state["n"] + 1) or True, max_attempts=5,
    )])
    assert gate.run()["x"] == "self_healed"


def test_retry_that_never_heals_halts():
    gate = HealthGate([Probe("x", bad, RETRY, self_heal=lambda: True, max_attempts=3)])
    with pytest.raises(HaltError):
        gate.run()


def test_quarantine_promotes_to_halt_after_bound():
    gate = HealthGate([Probe("q", bad, QUARANTINE, self_heal=lambda: True, max_attempts=2)])
    with pytest.raises(HaltError):
        gate.run()


def test_degrade_warns_and_continues():
    gate = HealthGate([Probe("obs", bad, DEGRADE)])
    assert gate.run() == {"obs": "degraded"}


def test_retry_with_no_healer_halts_at_once():
    calls = {"check": 0}

    def check():
        calls["check"] += 1
        return False

    gate = HealthGate([Probe("x", check, RETRY, self_heal=None, max_attempts=5)])
    with pytest.raises(HaltError):
        gate.run()
    assert calls["check"] == 1  # did not spin: one check, then HALT (no healer to attempt)


def test_halt_is_immediate_and_never_self_heals():
    tried = {"heal": False}
    gate = HealthGate([Probe(
        "integrity", bad, HALT,
        self_heal=lambda: tried.__setitem__("heal", True) or True,
    )])
    with pytest.raises(HaltError):
        gate.run()
    assert tried["heal"] is False  # an integrity HALT is not self-healed
