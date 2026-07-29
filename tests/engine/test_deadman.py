"""External liveness. The dead-man's-switch detects a missed heartbeat against an injected
clock (the always-on host that runs it is infra). The escalation channel guarantees an alert
is DELIVERED and ACKED, retrying transient transport failures and distinguishing undelivered
from delivered-but-unacked. The clock anchor flags drift from an external reference."""
import pytest

from engine.deadman import DeadMansSwitch, EscalationChannel, MockTransport, clock_trustworthy


def test_escalation_does_not_mask_a_transport_bug():
    # a genuine bug (not a declared transient TransportError) must propagate, not be swallowed
    # as UNDELIVERED. No silent fallback on the one path that must not eat failures.
    class BuggyTransport:
        def send(self, message):
            raise ValueError("bug in transport wiring")

        def acked(self, message_id):
            return True

    with pytest.raises(ValueError):
        EscalationChannel(BuggyTransport(), max_attempts=3).escalate("halt")


def test_heartbeat_alive_within_threshold():
    dms = DeadMansSwitch(threshold_s=60)
    hb = dms.beat(now=100.0)
    assert dms.missed(hb, now=150.0) is False


def test_heartbeat_missed_after_threshold():
    dms = DeadMansSwitch(threshold_s=60)
    hb = dms.beat(now=100.0)
    assert dms.missed(hb, now=200.0) is True


def test_escalation_acked():
    ch = EscalationChannel(MockTransport(will_ack=True))
    assert ch.escalate("halt: campaign stalled") == "ACKED"


def test_escalation_unacked_when_never_acked():
    ch = EscalationChannel(MockTransport(will_ack=False), max_attempts=3)
    assert ch.escalate("halt") == "UNACKED"


def test_escalation_retries_transient_failure_then_acks():
    ch = EscalationChannel(MockTransport(will_ack=True, fail_first=2), max_attempts=5)
    assert ch.escalate("halt") == "ACKED"


def test_escalation_undelivered_when_transport_stays_down():
    ch = EscalationChannel(MockTransport(will_ack=True, fail_first=99), max_attempts=3)
    assert ch.escalate("halt") == "UNDELIVERED"


def test_clock_anchor_trusts_small_drift():
    assert clock_trustworthy(external_t=1000.0, local_t=1002.0, max_drift_s=5.0) is True


def test_clock_anchor_flags_large_drift():
    assert clock_trustworthy(external_t=1000.0, local_t=1200.0, max_drift_s=5.0) is False
