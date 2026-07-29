"""The tool-ledger health gate. Every runtime dependency has a correctness-shaped
probe and one of four failure states. The response is DETERMINISTIC: a fault either
self-heals within a bound or HALTs, never a silent fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

HALT = "HALT"
QUARANTINE = "QUARANTINE"
RETRY = "RETRY"
DEGRADE = "DEGRADE"


class HaltError(Exception):
    """A disposer/integrity break, or a RETRY/QUARANTINE that did not clear within its
    bound. The campaign stops and pages Vignan."""


@dataclass
class Probe:
    name: str
    check: Callable[[], bool]            # True if healthy (correctness-shaped)
    state: str                           # failure state on unhealthy
    self_heal: Optional[Callable[[], bool]] = None
    max_attempts: int = 3


class HealthGate:
    def __init__(self, probes: list[Probe]):
        self.probes = probes

    def run(self) -> dict[str, str]:
        results: dict[str, str] = {}
        for p in self.probes:
            if p.check():
                results[p.name] = "ok"
                continue
            if p.state == DEGRADE:
                results[p.name] = "degraded"  # observability only: warn and continue
                continue
            if p.state == HALT:
                raise HaltError(f"{p.name}: integrity/disposer HALT (human ack required)")
            # RETRY (retry-in-place) and QUARANTINE (isolate + recompute) share one control
            # shape: bounded attempts on the caller's self_heal, else promote to HALT. They
            # differ in what self_heal DOES, not in the gate's flow. With no healer there is
            # nothing to attempt, so HALT at once rather than spin no-op iterations.
            if p.self_heal is None:
                raise HaltError(f"{p.name}: {p.state} with no self-heal -> HALT")
            for _ in range(p.max_attempts):
                if p.self_heal() and p.check():
                    results[p.name] = "self_healed"
                    break
            else:
                raise HaltError(
                    f"{p.name}: {p.state} did not clear in {p.max_attempts} attempts -> HALT"
                )
        return results
