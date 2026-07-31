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

from .handoff import Dossier, TriageDecision, build_dossier


@dataclass
class CampaignResult:
    verdict: object      # referee.runner.Verdict or None
    narrative: str
    schema: object       # referee.lineage.Schema or None
    lineage: str


def run_campaign(agent, backend, config, lease_store, box_factory, *,
                 substrate, triage: Callable[[Dossier], TriageDecision],
                 context=None, reviewer=None, significance=None) -> CampaignResult:
    # `substrate` and `triage` are REQUIRED (no defaults). A silent MockSubstrate would make the
    # referee judge fabricated constants, and a default accept-as-proposed triage would let the
    # AGENT'S proposed type pick its own gauntlet. Both are fail-OPEN footguns, so a real caller
    # must pass an ExperimentSubstrate and a committed TriageDecision explicitly; tests pass a
    # MockSubstrate and `accept_as_proposed` deliberately.
    # `context` is the discovery-loop steering (the bandit-picked vein, mode, and negative-bank
    # exclusion) handed to the generator. `reviewer` is the correctness-adversary: when present,
    # maturity requires BOTH the agent's own judgment AND surviving the reviewer, so maturity is
    # never agent-self-judged (the three-party separation the loop rests on).
    # --- Tier 1: discovery ---
    candidates = agent.propose(context or {})
    schema_raw = candidates[0]  # the bandit's pick (smoke: the first candidate)

    maturation = agent.mature(schema_raw)
    matured = maturation.matured
    if reviewer is not None:
        from .discovery_roles import is_mature
        matured = is_mature(maturation.matured, reviewer.review(schema_raw, {}))
    if not matured:
        s = normalize_schema(schema_raw)
        return CampaignResult(None, "no maturation", s, lineage_key(s))

    # --- Handoff (HIP): the human triages a NEUTRAL dossier assembled by a party other than the
    # framing agent, and FREEZES the pre-registration. The human PICKS the claim-type, so the loop
    # never selects its own gauntlet; the agent's proposed type is advisory only.
    # the significance-adversary (advisory) supplies its strongest incremental case to the dossier;
    # the negative-bank neighbors (from the loop context) give the human the semantic collision view.
    sig_case = significance.challenge(schema_raw).case if significance is not None else ""
    neighbors = tuple((context or {}).get("negative_bank", ())[:5])
    dossier = build_dossier(schema_raw, maturation, significance_case=sig_case,
                            nearest_bank_neighbors=neighbors)
    decision = triage(dossier)
    if not decision.accept:
        s = normalize_schema(schema_raw)
        return CampaignResult(None, "triage rejected", s, lineage_key(s))

    # The COMMITTED claim-type AND prior-claim flag (the human's) drive the Schema and thus the
    # magnitude gauntlet + the control set; the consequence-template id and seeds ride along frozen.
    # None of these come from the agent's proposal.
    committed_raw = {**schema_raw, "claim_type": decision.claim_type,
                     "prior_claim": decision.prior_claim,
                     "consequence_template_id": decision.consequence_template_id,
                     "seeds": decision.seeds}
    schema = normalize_schema(committed_raw)
    lk = lineage_key(schema)

    # The SUBSTRATE measures the gate inputs for the COMMITTED type (G0, mechanism, novelty, backbone,
    # consequence). The agent contributes only its own belief; the referee never gates on
    # agent-authored numbers, and the type it measures for is the human's, not the proposal's.
    bundle = substrate.produce(committed_raw, backend, believed_claim=maturation.bundle.believed_claim)

    # Box-INDEPENDENT gates before leasing, so a scarce box is never burned on a free check
    # (catalog drift raises here, before any claim; G0 / unwired claim-type return ineligible).
    pre = precheck(schema, config, bundle)
    if pre is not None:
        return CampaignResult(pre, pre.reason, schema, lk)

    # --- Tier 2: confirmation on a fresh leased box ---
    claim = lease_store.claim(hypothesis=committed_raw["claim"], lineage=lk)
    if claim is None:
        return CampaignResult(None, "no box available (exhausted or same-claim barred)", schema, lk)
    box = box_factory(claim.box_id)
    lease_store.mark_label_read(claim.box_id, claim.generation)
    verdict = confirm(backend, box, schema, config, bundle)
    lease_store.stage(claim.box_id, claim.generation, verdict=verdict.status, score=b"")
    lease_store.commit(claim.box_id, claim.generation)

    return CampaignResult(verdict, agent.frame(committed_raw, verdict), schema, lk)
