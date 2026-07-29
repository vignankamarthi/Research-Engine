"""The real Tier-1 discovery agent: a Claude Code subagent driven by the Max subscription.
No API key, no per-token billing. It implements the same `Agent` protocol as MockAgent, so it
drops into `run_campaign` with zero change to the confirmatory substrate. The subprocess runner
is injected: the default shells out to `claude -p` (headless, same auth as an interactive
session), and the tests inject a canned runner so the plumbing is validated for free.

Separation of concerns: the agent GENERATES a hypothesis schema, JUDGES maturity, and NARRATES
a thesis. It never invents the statistical gate inputs (G0, effect CIs, p-values); those come
from the executed substrate and the confirmatory runner, which is what keeps the referee honest."""
from __future__ import annotations

import json

from .agents import Bundle, Maturation
from .claude_cli import ClaudeAgentError, ClaudeRoleBase, extract_json

# Back-compat re-exports: some call sites and tests import these from this module.
_extract_json = extract_json
__all__ = ["ClaudeCodeAgent", "ClaudeAgentError"]

_PROPOSE = (
    "You are a Tier-1 discovery agent for the SMILE lab (Prof. Yun Raymond Fu), video "
    "foundation models. Propose ONE concrete, falsifiable research hypothesis. Reply with ONLY "
    "a JSON object with keys: claim (a one-sentence proposition), claim_type (one of effect, "
    "qualitative_phenomenon, capability, law_shape), backbone, dataset, scale, measure, "
    "prior_claim (bool). Context: {context}"
)
_MATURE = (
    "You are judging whether a hypothesis is mature enough to spend a scarce confirmation box "
    "on. Reply with ONLY a JSON object: matured (bool), believed_claim (bool, do you expect it "
    "to hold), reason (short string). Do NOT invent any statistics. Hypothesis: {schema}"
)
_FRAME = (
    "Draft a two-sentence thesis for this hypothesis given its confirmatory verdict was "
    "{status}. Be plain and honest, no hype. Hypothesis: {schema}"
)


class ClaudeCodeAgent(ClaudeRoleBase):
    def propose(self, context: dict) -> list[dict]:
        result = self._ask_json(_PROPOSE.format(context=json.dumps(context)))
        return result if isinstance(result, list) else [result]

    def mature(self, schema_raw: dict) -> Maturation:
        d = self._ask_json(_MATURE.format(schema=json.dumps(schema_raw)))
        return Maturation(
            matured=bool(d.get("matured", False)),
            bundle=Bundle(believed_claim=bool(d.get("believed_claim", True))),
        )

    def frame(self, schema_raw: dict, verdict) -> str:
        status = verdict.status if verdict is not None else "pending"
        return self._runner(_FRAME.format(schema=json.dumps(schema_raw), status=status))
