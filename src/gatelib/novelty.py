"""The novelty re-audit: fail-closed on a prior-art collision, and a positive-delta
requirement (name the k nearest prior works and argue a human-checkable advance).
Absence of a collision is not certified novelty; it is only the absence of a match."""
from __future__ import annotations

from dataclasses import dataclass

from .verdicts import HALT_RETRY, PROCEED, REJECT


@dataclass(frozen=True, slots=True)
class NoveltyResult:
    passed: bool
    reason: str


def novelty_check(collision_found: bool, k_nearest, advance_argued: bool) -> NoveltyResult:
    if collision_found:
        return NoveltyResult(False, "prior_art_collision")
    if not k_nearest:
        return NoveltyResult(False, "no_nearest_priors_named")
    if not advance_argued:
        return NoveltyResult(False, "no_advance_argued")
    return NoveltyResult(True, "ok")


@dataclass(frozen=True, slots=True)
class CorpusStatus:
    reachable: bool     # False when the prior-art corpus is rate-limited or unreachable
    as_of_t: float      # timestamp of the corpus's last refresh


@dataclass(frozen=True, slots=True)
class NoveltyDecision:
    decision: str       # PROCEED | REJECT | HALT_RETRY
    reason: str
    checkpoint: str = ""


def novelty_gate(collision_found: bool, k_nearest, advance_argued: bool,
                 corpus: CorpusStatus, now: float, max_staleness_s: float,
                 checkpoint: str = "") -> NoveltyDecision:
    """The novelty audit at a checkpoint (pre_allocation or submit). An unreachable or stale
    corpus yields HALT_RETRY so the audit is retried rather than run against bad data, and a
    scored box is never burned on an untrustworthy novelty read."""
    if not corpus.reachable:
        return NoveltyDecision(HALT_RETRY, "corpus_unreachable", checkpoint)
    if (now - corpus.as_of_t) > max_staleness_s:
        return NoveltyDecision(HALT_RETRY, "corpus_stale", checkpoint)
    res = novelty_check(collision_found, k_nearest, advance_argued)
    if res.passed:
        return NoveltyDecision(PROCEED, "ok", checkpoint)
    return NoveltyDecision(REJECT, res.reason, checkpoint)
