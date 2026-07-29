"""The campaign loop that ties the two tiers together: discovery proposes and matures
a hypothesis, the human triages at the handoff (a callback, defaulting to accept for
tests; the real touchpoint is Vignan), the confirmatory tier leases a fresh box and
scores it once, the verdict commits through the lease, and a narrative is drafted.
This is the end-to-end spine; the bandit, pool, and selection layer on top."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from referee import lineage_key, normalize_schema
from referee.runner import confirm, precheck

from .substrate import MockSubstrate


@dataclass
class CampaignResult:
    verdict: object      # referee.runner.Verdict or None
    narrative: str
    schema: object       # referee.lineage.Schema or None
    lineage: str


def run_campaign(agent, backend, config, lease_store, box_factory,
                 triage: Callable[[str], bool] = lambda narrative: True,
                 substrate=None) -> CampaignResult:
    # --- Tier 1: discovery ---
    if substrate is None:
        substrate = MockSubstrate()
    candidates = agent.propose({})
    schema_raw = candidates[0]  # the bandit's pick (smoke: the first candidate)
    schema = normalize_schema(schema_raw)
    lk = lineage_key(schema)

    maturation = agent.mature(schema_raw)
    if not maturation.matured:
        return CampaignResult(None, "no maturation", schema, lk)

    # The SUBSTRATE measures the gate inputs (G0, mechanism, novelty, backbone, consequence); the
    # agent contributes only its own belief. The referee never gates on agent-authored numbers.
    bundle = substrate.produce(schema, backend, believed_claim=maturation.bundle.believed_claim)

    # Box-INDEPENDENT gates before leasing, so a scarce box is never burned on a free check
    # (catalog drift raises here, before any claim; G0 / unwired claim-type return ineligible).
    pre = precheck(schema, config, bundle)
    if pre is not None:
        return CampaignResult(pre, pre.reason, schema, lk)

    # --- Handoff: human triage of the drafted narrative (HIP) ---
    if not triage(agent.frame(schema_raw, None)):
        return CampaignResult(None, "triage rejected", schema, lk)

    # --- Tier 2: confirmation on a fresh leased box ---
    claim = lease_store.claim(hypothesis=schema_raw["claim"], lineage=lk)
    if claim is None:
        return CampaignResult(None, "no box available (exhausted or same-claim barred)", schema, lk)
    box = box_factory(claim.box_id)
    lease_store.mark_label_read(claim.box_id, claim.generation)
    verdict = confirm(backend, box, schema, config, bundle)
    lease_store.stage(claim.box_id, claim.generation, verdict=verdict.status, score=b"")
    lease_store.commit(claim.box_id, claim.generation)

    return CampaignResult(verdict, agent.frame(schema_raw, verdict), schema, lk)
