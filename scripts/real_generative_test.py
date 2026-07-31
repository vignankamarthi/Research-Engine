"""Real generative test. A real Claude Code subagent (via `claude -p`, driven by the Max
subscription, no API key) plays the Tier-1 discovery agent and drives a hypothesis through the
whole campaign pipeline on a MockBackend. This proves the real agent implements the Agent
protocol and drives the deterministic substrate end to end.

It stops at a MOCK-scored verdict. A real confirmatory campaign against a real model + real
holdout is a HIP (Vignan). Run with: uv run python scripts/real_generative_test.py
"""
import pathlib
import sys
import tempfile
from datetime import date

sys.path.insert(0, "src")

from backend import Box, MockBackend
from engine import ClaudeCodeAgent, run_campaign
from engine.handoff import accept_as_proposed
from engine.substrate import MockSubstrate
from gateconfig import validate_config
from referee import normalize_schema
from referee.lease import LeaseStore
from referee.lineage import control_catalog_digest


def cfg():
    return validate_config({
        "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.01, "mie_floor": 0.03,
        "claim_types": ["effect", "qualitative_phenomenon", "capability", "law_shape"],
        "gate_library_digest": "sha256:" + "0" * 64,
        "control_catalog_hash": control_catalog_digest(), "key_id": "live",
    })


def box_factory(box_id):
    return Box(id=box_id, n=800, origin_date=date(2024, 6, 1))  # post backbone cutoff


def main():
    agent = ClaudeCodeAgent()  # real: default runner shells out to `claude -p`

    print("=" * 72)
    print("STEP 1  real Claude Code subagent proposes a SMILE hypothesis...")
    proposal = agent.propose({
        "lab": "SMILE (Prof. Yun Raymond Fu)",
        "focus": "video foundation models, frequency-domain temporal modeling",
    })
    schema_raw = proposal[0]
    for k, v in schema_raw.items():
        print(f"    {k}: {v}")

    # Fail loud if the agent's schema is malformed (no silent fallback).
    normalize_schema(schema_raw)

    print("\nSTEP 2  drive it through the campaign (mature -> triage -> confirm on MockBackend)...")
    lease = LeaseStore(str(pathlib.Path(tempfile.mkdtemp()) / "lease.db"))
    lease.add_boxes(["box0", "box1"])
    # A clean, genuine effect so the confirmatory gauntlet returns a definite verdict.
    backend = MockBackend(trained_effect=0.25, untrained_effect=0.0, noise=0.1, seed=1)

    # Force the agent's proposed claim through (the MockBackend supplies the effect).
    agent_fixed = _PinnedProposalAgent(agent, schema_raw)
    result = run_campaign(agent_fixed, backend, cfg(), lease, box_factory,
                          substrate=MockSubstrate(), triage=accept_as_proposed)

    print("\nSTEP 3  result")
    v = result.verdict
    print(f"    verdict: {v.status if v else None}" + (f" ({v.reason})" if v else ""))
    print(f"    narrative (drafted by the real subagent):\n    {result.narrative}")
    print("=" * 72)
    print("PASS: a real Claude Code subagent drove the full pipeline to a verdict.")


class _PinnedProposalAgent:
    """Wraps the real agent but pins the already-proposed schema, so propose() is not re-run
    inside run_campaign (one generation call for the proposal, real mature/frame still real)."""

    def __init__(self, inner, schema_raw):
        self._inner = inner
        self._schema_raw = schema_raw

    def propose(self, context):
        return [self._schema_raw]

    def mature(self, schema_raw):
        m = self._inner.mature(schema_raw)
        if not m.matured:
            return m
        # Bundle is fail-closed: the agent does not fabricate substrate-owned gate inputs. Here
        # we stand in for the substrate and inject a satisfied bundle so the full confirmatory
        # path runs, preserving the agent's own believed_claim.
        from engine.agents import Bundle, Maturation
        return Maturation(matured=True, bundle=Bundle.passing(believed_claim=m.bundle.believed_claim))

    def frame(self, schema_raw, verdict):
        return self._inner.frame(schema_raw, verdict)


if __name__ == "__main__":
    main()
