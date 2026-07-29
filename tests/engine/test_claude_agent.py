"""The real Tier-1 agent, a Claude Code subagent driven by the Max subscription (no API key,
no per-token billing). It implements the same `Agent` protocol as MockAgent and swaps in at
runtime. The subprocess runner is injected, so the suite tests the parsing/plumbing against
canned output (deterministic, free); the real `claude -p` path is exercised only by the fenced
live test. The agent GENERATES hypotheses, JUDGES maturity, and NARRATES; it never fabricates
the statistical gate inputs (those come from the executed substrate)."""
from types import SimpleNamespace

import pytest

from engine.agents import Agent, Maturation
from engine.claude_agent import ClaudeAgentError, ClaudeCodeAgent, _extract_json

SCHEMA_JSON = ('{"claim":"c","claim_type":"effect","backbone":"iv2","dataset":"ssv2",'
               '"scale":"7b","measure":"acc","prior_claim":false}')


def test_extract_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced_json():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_amid_prose():
    assert _extract_json('Here is the schema:\n{"a": 1}\nThanks.') == {"a": 1}


def test_extract_json_array():
    assert _extract_json('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]


def test_extract_raises_when_no_json():
    with pytest.raises(ClaudeAgentError):
        _extract_json("there is no json here")


def test_satisfies_agent_protocol():
    assert isinstance(ClaudeCodeAgent(runner=lambda p: "{}"), Agent)


def test_propose_returns_list_of_schemas():
    agent = ClaudeCodeAgent(runner=lambda p: f"[{SCHEMA_JSON}]")
    out = agent.propose({"lab": "SMILE"})
    assert isinstance(out, list) and out[0]["claim"] == "c"


def test_propose_normalizes_single_object_to_list():
    agent = ClaudeCodeAgent(runner=lambda p: SCHEMA_JSON)
    out = agent.propose({})
    assert isinstance(out, list) and len(out) == 1


def test_mature_parses_matured_judgment():
    agent = ClaudeCodeAgent(runner=lambda p: '{"matured": true, "believed_claim": true}')
    m = agent.mature({"claim": "c"})
    assert isinstance(m, Maturation) and m.matured is True


def test_mature_can_return_not_matured():
    agent = ClaudeCodeAgent(runner=lambda p: '{"matured": false}')
    assert agent.mature({"claim": "c"}).matured is False


def test_frame_returns_narrative_text():
    agent = ClaudeCodeAgent(runner=lambda p: "Thesis: freq-domain temporal modeling helps.")
    v = SimpleNamespace(status="CONFIRMED")
    assert "Thesis" in agent.frame({"claim": "c"}, v)


def test_runner_failure_propagates():
    def boom(prompt):
        raise ClaudeAgentError("claude CLI failed: rc=1")

    with pytest.raises(ClaudeAgentError):
        ClaudeCodeAgent(runner=boom).propose({})
