"""External liveness client logic. The dead-man's-switch and the escalation channel run on
an always-on host (SLURM scavenger or a small VM, never the laptop); that host is infra. The
logic here is the testable part: miss-detection against an injected clock, and an alert path
that is not trusted until it is both delivered and acknowledged."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class TransportError(Exception):
    """A declared, transient transport failure (outage, rate-limit) that escalation may retry.
    Anything else raised by a transport is a bug and must propagate, never be masked."""


@dataclass(frozen=True)
class Heartbeat:
    last_beat_t: float


class DeadMansSwitch:
    def __init__(self, threshold_s: float):
        self._threshold = threshold_s

    def beat(self, now: float) -> Heartbeat:
        return Heartbeat(now)

    def missed(self, heartbeat: Heartbeat, now: float) -> bool:
        return (now - heartbeat.last_beat_t) > self._threshold


@runtime_checkable
class Transport(Protocol):
    def send(self, message: str) -> str: ...   # returns a message id, raises if undelivered
    def acked(self, message_id: str) -> bool: ...


class EscalationChannel:
    """Sends until an alert is acknowledged. Distinguishes UNDELIVERED (transport never
    accepted the message) from UNACKED (delivered, but no human acknowledgement)."""

    def __init__(self, transport: Transport, max_attempts: int = 3):
        self._transport = transport
        self._max_attempts = max_attempts

    def escalate(self, message: str) -> str:
        delivered = False
        for _ in range(self._max_attempts):
            try:
                mid = self._transport.send(message)
            except TransportError:
                continue  # a declared transient failure: retry. A real bug propagates.
            delivered = True
            if self._transport.acked(mid):
                return "ACKED"
        return "UNACKED" if delivered else "UNDELIVERED"


def clock_trustworthy(external_t: float, local_t: float, max_drift_s: float) -> bool:
    """The clock anchor: the local clock is trusted only while its drift from an external
    reference (e.g. the timestamp of a pushed signed commit) stays within bound."""
    return abs(local_t - external_t) <= max_drift_s


class MockTransport:
    """Deterministic transport. fail_first send() calls raise (a transient outage); will_ack
    decides whether a delivered message is ever acknowledged."""

    def __init__(self, will_ack: bool = True, fail_first: int = 0):
        self._will_ack = will_ack
        self._fail_first = fail_first
        self._sends = 0

    def send(self, message: str) -> str:
        self._sends += 1
        if self._sends <= self._fail_first:
            raise TransportError("transport down")
        return f"mid-{self._sends}"

    def acked(self, message_id: str) -> bool:
        return self._will_ack
