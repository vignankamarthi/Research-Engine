"""The discovery-tier adversary and synthesis roles, each a Claude Code subagent behind a
protocol (Mock double for the green suite). Two adversaries with different force:

- reviewer-adversary: strongest CORRECTNESS objections. Surviving it is a maturity CONDITION.
- significance-adversary: strongest incremental/already-known case. ADVISORY only, it feeds
  the triage dossier and an importance rank, it never silently kills a hypothesis.

The synthesizer clusters matured findings under one thesis with a held-out joint prediction; an
arc ships as one paper only if that joint prediction is not already entailed by a single finding."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .claude_cli import ClaudeRoleBase


@dataclass
class ReviewerVerdict:
    survives: bool
    objections: list = field(default_factory=list)


@dataclass
class SignificanceVerdict:
    incremental: bool          # advisory: is the finding incremental / already known
    case: str = ""
    importance_penalty: float = 0.0


@dataclass
class Synthesis:
    thesis: str
    joint_prediction: str
    entailed_by_single: bool   # True -> not a real arc, ship the findings separately


@runtime_checkable
class ReviewerAdversary(Protocol):
    def review(self, schema_raw: dict, evidence: dict) -> ReviewerVerdict: ...


@runtime_checkable
class SignificanceAdversary(Protocol):
    def challenge(self, schema_raw: dict) -> SignificanceVerdict: ...


@runtime_checkable
class Synthesizer(Protocol):
    def synthesize(self, findings: list) -> Synthesis: ...


def is_mature(agent_matured: bool, reviewer: ReviewerVerdict) -> bool:
    """Maturity needs BOTH the agent's own judgment and survival of the reviewer-adversary.
    The significance-adversary is deliberately absent here: it is advisory, never a gate."""
    return bool(agent_matured and reviewer.survives)


def decide_arc(synthesis: Synthesis) -> bool:
    """Ship as one arc only if the joint prediction adds something no single finding entails."""
    return not synthesis.entailed_by_single


# --- deterministic mocks (keep the suite green and free) ---

class MockReviewerAdversary:
    def review(self, schema_raw: dict, evidence: dict) -> ReviewerVerdict:
        return ReviewerVerdict(survives=True, objections=[])


class MockSignificanceAdversary:
    def challenge(self, schema_raw: dict) -> SignificanceVerdict:
        return SignificanceVerdict(incremental=False, case="novel enough", importance_penalty=0.0)


class MockSynthesizer:
    def synthesize(self, findings: list) -> Synthesis:
        return Synthesis("unified thesis", "held-out joint prediction", entailed_by_single=False)


# --- real Claude Code subagent roles ---

_REVIEW = (
    "You are a reviewer-adversary. Raise the STRONGEST correctness objections to this hypothesis "
    "and its evidence. Reply ONLY as JSON: survives (bool, does it survive without a fatal flaw), "
    "objections (list of strings). Hypothesis: {schema}. Evidence: {evidence}"
)
_SIGNIF = (
    "You are a significance-adversary. Make the STRONGEST case that this is incremental or already "
    "known. This is ADVISORY, not a veto. Reply ONLY as JSON: incremental (bool), case (string), "
    "importance_penalty (0.0 to 1.0). Hypothesis: {schema}"
)
_SYNTH = (
    "Cluster these findings under ONE thesis with a single held-out-testable joint prediction. Reply "
    "ONLY as JSON: thesis (string), joint_prediction (string), entailed_by_single (bool, is the joint "
    "prediction already entailed by any single finding). Findings: {findings}"
)


class ClaudeReviewerAdversary(ClaudeRoleBase):
    def review(self, schema_raw: dict, evidence: dict) -> ReviewerVerdict:
        d = self._ask_json(_REVIEW.format(schema=json.dumps(schema_raw), evidence=json.dumps(evidence)))
        return ReviewerVerdict(survives=bool(d.get("survives", False)),
                               objections=list(d.get("objections", [])))


class ClaudeSignificanceAdversary(ClaudeRoleBase):
    def challenge(self, schema_raw: dict) -> SignificanceVerdict:
        d = self._ask_json(_SIGNIF.format(schema=json.dumps(schema_raw)))
        return SignificanceVerdict(incremental=bool(d.get("incremental", False)),
                                   case=str(d.get("case", "")),
                                   importance_penalty=float(d.get("importance_penalty", 0.0)))


class ClaudeSynthesizer(ClaudeRoleBase):
    def synthesize(self, findings: list) -> Synthesis:
        d = self._ask_json(_SYNTH.format(findings=json.dumps(findings)))
        return Synthesis(thesis=str(d.get("thesis", "")),
                         joint_prediction=str(d.get("joint_prediction", "")),
                         entailed_by_single=bool(d.get("entailed_by_single", True)))
