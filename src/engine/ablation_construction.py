"""The red/blue ablation-CONSTRUCTION loop (spec 3c, step 45), the idea-agnostic replacement for a
static `resolve_ablation`. A BLUE role proposes an ablation as a BOUNDED primitive composition (a
vetted primitive name + params, never free code), a RED panel attacks it on specificity,
anti-collusion, and confound, and blue revises against the refutations until the panel cannot
refute. If it cannot converge within K rounds the idea is marked NO CLEAN ABLATION and the mechanism
gate fails closed, it never falls back to an unverified ablation.

The loop is DETERMINISTIC Python. Blue and red are `claude -p` roles the loop spawns (the same
pattern as the discovery adversaries), mocked here so the suite is green and swapped for real roles
on the cluster. Because the spec is bounded to vetted primitives, the red team reasons about a small
object, not arbitrary code, and there is no code-execution trust hole."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from .ablation_primitives import vetted_primitives
from .claude_cli import ClaudeRoleBase


@dataclass(frozen=True)
class AblationSpec:
    """A bounded ablation: a vetted primitive name plus its parameters. Not free code."""
    primitive: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BlueProposal:
    spec: AblationSpec
    rationale: str = ""


@dataclass(frozen=True)
class RedVerdict:
    refuted: bool
    axis: str = ""      # specificity | anti_collusion | confound
    reason: str = ""


@dataclass(frozen=True)
class ConstructedAblation:
    spec: AblationSpec
    apply: Callable     # the frozen, vetted ablation the mechanism experiment runs
    rounds: int         # how many blue/red rounds it took to converge


class NoCleanAblation(Exception):
    """The red/blue loop could not converge on a clean ablation. The mechanism gate fails CLOSED for
    this idea, it never falls back to an unverified ablation."""


@runtime_checkable
class BlueBuilder(Protocol):
    def build(self, mechanism: str, task: str, feedback: list) -> BlueProposal: ...


@runtime_checkable
class RedAttacker(Protocol):
    def attack(self, mechanism: str, spec: AblationSpec) -> RedVerdict: ...


def resolve_spec(spec: AblationSpec, primitives: dict | None = None) -> Callable:
    """Turn a bounded spec into a callable via the VETTED primitive set only. A spec naming a
    primitive that is absent or failed its self-test raises (fail-closed), so only clean ops run."""
    pool = vetted_primitives() if primitives is None else primitives
    if spec.primitive not in pool:
        raise NoCleanAblation(
            f"primitive {spec.primitive!r} is not in the vetted set (absent or failed self-test)")
    prim = pool[spec.primitive]
    return lambda x: prim.apply(x, **spec.params)


def construct_ablation(mechanism: str, task: str, *, blue: BlueBuilder,
                       red_panel: list[RedAttacker], rounds: int = 4,
                       primitives: dict | None = None) -> ConstructedAblation:
    """Run the loop. Each round blue proposes a bounded spec (given the prior round's refutations),
    the red panel attacks it, and any refutation forces a revise. Converge when NO red attacker
    refutes; the converged ablation is frozen and returned. Exhaust `rounds` without convergence and
    the idea fails closed (NoCleanAblation)."""
    feedback: list[RedVerdict] = []
    for r in range(1, rounds + 1):
        proposal = blue.build(mechanism, task, feedback)
        resolve_spec(proposal.spec, primitives)  # the spec must resolve to a vetted primitive
        refutations = [v for v in (red.attack(mechanism, proposal.spec) for red in red_panel)
                       if v.refuted]
        if not refutations:  # the whole panel conceded -> converged
            return ConstructedAblation(
                spec=proposal.spec, apply=resolve_spec(proposal.spec, primitives), rounds=r)
        feedback = refutations
    raise NoCleanAblation(
        f"no clean ablation for mechanism {mechanism!r} on {task!r} within {rounds} rounds")


# --- deterministic mocks (the real roles are claude -p, added with the cluster wiring) ---

class MockBlueBuilder:
    """Proposes a fixed spec every round. A real blue would revise against the feedback."""

    def __init__(self, spec: AblationSpec):
        self._spec = spec

    def build(self, mechanism: str, task: str, feedback: list) -> BlueProposal:
        return BlueProposal(self._spec)


class MockRedAttacker:
    """Refutes for the first `refute_rounds` attacks, then concedes. Models a red that blue
    eventually satisfies (or, with a large count, one that never concedes)."""

    def __init__(self, refute_rounds: int = 0):
        self._refute_rounds = refute_rounds
        self._seen = 0

    def attack(self, mechanism: str, spec: AblationSpec) -> RedVerdict:
        self._seen += 1
        if self._seen <= self._refute_rounds:
            return RedVerdict(refuted=True, axis="specificity", reason="mock refutation")
        return RedVerdict(refuted=False)


# --- the real claude -p roles (same pattern as the discovery adversaries) ---

_BLUE_PROMPT = (
    "You are the BLUE builder in an adversarial ablation-construction loop. Propose an ablation "
    "that removes ONLY the named mechanism from a model's input, using EXACTLY ONE of the vetted "
    "primitives below. It must be SPECIFIC: it removes the mechanism and moves nothing else.\n\n"
    "Mechanism to remove: {mechanism}\nTask / data: {task}\n"
    "Vetted primitives (name: description): {primitives}\n"
    "Prior red-team refutations to address (may be empty): {feedback}\n\n"
    'Return ONLY JSON: {{"primitive": "<one vetted name>", "params": {{...}}, '
    '"rationale": "<why this isolates the mechanism>"}}'
)

_RED_PROMPT = (
    "You are a RED attacker in an adversarial ablation-construction loop. Try to REFUTE the "
    "proposed ablation on ONE axis: SPECIFICITY (removes more or less than the mechanism), "
    "ANTI-COLLUSION (rigged to trivially kill the effect regardless of the mechanism), or CONFOUND "
    "(introduces a gross artifact). Default to refuted=true if you find any real problem.\n\n"
    "Mechanism: {mechanism}\nProposed ablation: {spec}\n\n"
    'Return ONLY JSON: {{"refuted": <bool>, '
    '"axis": "specificity|anti_collusion|confound|none", "reason": "<objection, or why it holds>"}}'
)


class ClaudeBlueBuilder(ClaudeRoleBase):
    """The real blue builder, a `claude -p` role. Proposes a bounded primitive parameterization."""

    def build(self, mechanism: str, task: str, feedback: list) -> BlueProposal:
        primitives = ", ".join(f"{n}: {p.description}" for n, p in vetted_primitives().items())
        fb = json.dumps([{"axis": v.axis, "reason": v.reason} for v in feedback])
        d = self._ask_json(_BLUE_PROMPT.format(
            mechanism=mechanism, task=task, primitives=primitives, feedback=fb))
        return BlueProposal(
            spec=AblationSpec(primitive=str(d["primitive"]), params=dict(d.get("params", {}))),
            rationale=str(d.get("rationale", "")))


class ClaudeRedAttacker(ClaudeRoleBase):
    """The real red attacker, a `claude -p` role. Fail-closed: a missing/ambiguous verdict reads as
    refuted, so an unparseable attacker never lets a bad ablation through."""

    def attack(self, mechanism: str, spec: AblationSpec) -> RedVerdict:
        d = self._ask_json(_RED_PROMPT.format(
            mechanism=mechanism,
            spec=json.dumps({"primitive": spec.primitive, "params": spec.params})))
        return RedVerdict(refuted=bool(d.get("refuted", True)),
                          axis=str(d.get("axis", "")), reason=str(d.get("reason", "")))
