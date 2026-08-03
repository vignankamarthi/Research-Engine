"""The assembled discovery loop, the TOP LAYER. Drives `run_supervisor` with a step_fn that, each
tick, uses the bandit + steering to pick a discovery VEIN, runs one confirm cycle (generate ->
mature-vs-reviewer -> human triage -> substrate MEASURES -> frozen referee JUDGES via run_campaign),
banks the result, rewards the bandit, and advances the budget. The supervisor guarantees the
base-case halt at the maturation budget; `close_campaign` then runs the selection correction over
the matured-and-scored family. This is what makes the two-tier loop run end to end, replacing the
single-shot spine. The GPU-hour meter is a per-tick placeholder on the Mac path (a real sacct meter
swaps in on the cluster); it also guarantees progress so a legitimately unproductive tick cannot
trip the stall backstop."""
from __future__ import annotations

from dataclasses import dataclass, field

from .bandit import Bandit
from .campaign import run_campaign
from .discovery_roles import decide_arc
from .generation import GENERATIVE_VEINS, VEINS, choose_mode, is_jepa_candidate
from .steering import BREADTH, DEPTH, SteeringPolicy
from .supervisor import SupervisorState, run_supervisor

# The discovery VEINS (imported from `generation`, the owning module) are the bandit's ARM SPACE
# (SPEC 3): six DERIVATIVE veins mined from the frontier literature + three GENERATIVE veins. This is
# the diversity axis, so the bandit steers ACROSS distinct idea-generation strategies rather than over
# anonymous integer arms, and a campaign spreads instead of collapsing onto one idea (the temporal
# monoculture the audit flagged). `VEINS` is re-exported for callers; GENERATIVE_VEINS drives the
# BREADTH/DEPTH bias below.

# statuses that spent a box (a look in the selection family)
_SCORED = ("CONFIRMED", "STRONG", "CONFIRMED_EFFECT", "CONFIRMED_NEGATIVE", "FAILED", "INCONCLUSIVE")
_NEGATIVE = ("FAILED", "INCONCLUSIVE", "CONFIRMED_NEGATIVE")
_POSITIVE = ("CONFIRMED", "STRONG", "CONFIRMED_EFFECT")  # a finding an arc can be built from


class _FixedClaimAgent:
    """Runs one PRE-FORMED claim (the synthesizer's joint prediction) through the normal confirm cycle.
    `propose` ignores the discovery context and returns the frozen schema; `mature`/`frame` delegate to
    the base agent, so the arc claim is matured, triaged, and scored on its own reserved box exactly
    like a scout candidate, and close reads a real verdict rather than an unscoreable post-hoc claim."""

    def __init__(self, base, schema: dict):
        self._base, self._schema = base, schema

    def propose(self, context):
        return [dict(self._schema)]

    def mature(self, schema_raw):
        return self._base.mature(schema_raw)

    def frame(self, schema_raw, verdict):
        return self._base.frame(schema_raw, verdict)


def _positive_cluster(results):
    """The findings an arc can unify: the matured-and-scored POSITIVE results. Returns the cluster once
    at least two exist (the synthesizer then judges whether they form a real arc), else None."""
    positive = [r for r in results if r.verdict is not None and r.verdict.status in _POSITIVE]
    return positive if len(positive) >= 2 else None


def _joint_schema(cluster, synth) -> dict:
    """Build the joint-prediction claim as its OWN confirmable schema. It inherits the shared conditions
    of the clustered findings (dataset / backbone / scale / measure / claim-type) and carries the
    synthesizer's joint prediction as the claim and the thesis as the mechanism the arc rests on."""
    s = cluster[0].schema
    return {"claim": synth.joint_prediction, "claim_type": s.claim_type, "backbone": s.backbone,
            "dataset": s.dataset, "scale": s.scale, "measure": s.measure, "prior_claim": False,
            "mechanism": synth.thesis or "joint_thesis", "believed_claim": True}


@dataclass
class Campaign:
    """The accumulating state of the assembled loop, closed at the base case."""
    results: list = field(default_factory=list)
    negative_bank: set = field(default_factory=set)  # lineage keys of dead ends, steered around
    depth: int = 0
    breadth: int = 0
    arc: object = None                                # the frozen mid-campaign Synthesis (once)
    lead_arc_confirmed: bool = False                  # did the arc's joint-prediction box confirm
    pending_arc: list = field(default_factory=list)   # joint-prediction schemas awaiting a reserved box
    jepa_matured: int = 0                             # running count of JEPA-tagged matured ideas (PLAN 78)


def _schema_view(schema) -> dict:
    """A dict view of a referee Schema for JEPA detection. The Schema drops `mechanism`, so this reads
    the surviving text fields; the primary signal is the agent's SELECTED candidate (which keeps the
    `jepa` stamp + mechanism), and this is the fallback for a pre-formed arc claim."""
    if schema is None:
        return {}
    return {k: getattr(schema, k, "") for k in ("claim", "measure", "dataset", "backbone")}


def _is_jepa_maturation(selected, result) -> bool:
    """Did the just-scored maturation sit in the JEPA DAG? Prefer the agent's SELECTED candidate (it
    carries the `jepa` stamp the wave computed, including the mechanism text the Schema loses); fall
    back to the result schema for a pre-formed arc claim with no wave selection."""
    if selected:
        if selected.get("jepa") or is_jepa_candidate(selected):
            return True
    return is_jepa_candidate(_schema_view(result.schema))


def _reward(verdict) -> float:
    if verdict is None:
        return 0.0
    if verdict.status in ("CONFIRMED", "STRONG"):
        return 1.0
    if verdict.status in ("CONFIRMED_EFFECT", "CONFIRMED_NEGATIVE"):
        return 0.5
    return 0.1  # a scored FAILED / INCONCLUSIVE still informed the search


def build_step_fn(*, agent, backend, config, lease_store, box_factory, substrate, triage,
                  reviewer, bandit, policy, campaign, veins=VEINS, significance=None,
                  synthesizer=None):
    def step_fn(state: SupervisorState) -> SupervisorState:
        if campaign.pending_arc:
            # A frozen ARC claim takes priority: it has a reserved slot, so its joint-prediction box is
            # scored rather than left an unscoreable post-hoc claim at close. It is not a bandit arm, so
            # it neither rewards the bandit nor counts as depth/breadth.
            result = run_campaign(
                _FixedClaimAgent(agent, campaign.pending_arc.pop(0)), backend, config, lease_store,
                box_factory, substrate=substrate, triage=triage, context=None, reviewer=reviewer,
                significance=significance)
            campaign.results.append(result)
            if result.verdict is not None and result.verdict.status in _POSITIVE:
                campaign.lead_arc_confirmed = True
            scored = result.verdict is not None and result.verdict.status in _SCORED
            if scored and _is_jepa_maturation(None, result):
                campaign.jepa_matured += 1  # the arc claim itself may sit in the JEPA DAG
        else:
            trial, arm = bandit.ask()
            vein = bandit.arm_label(arm)  # a NAMED vein, the bandit's arm space is the vein set now
            # generative veins bias toward BREADTH, derivative toward DEPTH; steering then honors the
            # depth-floor / breadth-cap on the margin.
            pick = BREADTH if vein in GENERATIVE_VEINS else DEPTH
            mode = choose_mode(pick, campaign.depth, campaign.breadth, policy)
            # The running JEPA count is injected so a reserve-aware agent (the WaveAgent) can enforce
            # the floor/cap on which candidate it surfaces first. A plain agent ignores it (harmless).
            context = {"vein": vein, "mode": mode, "negative_bank": sorted(campaign.negative_bank),
                       "jepa_matured": campaign.jepa_matured}

            result = run_campaign(agent, backend, config, lease_store, box_factory,
                                  substrate=substrate, triage=triage, context=context, reviewer=reviewer,
                                  significance=significance)
            campaign.results.append(result)
            bandit.tell(trial, _reward(result.verdict))

            scored = result.verdict is not None and result.verdict.status in _SCORED
            if scored and result.verdict.status in _NEGATIVE:
                campaign.negative_bank.add(result.lineage)
            if scored and _is_jepa_maturation(getattr(agent, "last_selected", None), result):
                campaign.jepa_matured += 1  # PLAN 78: count every JEPA-tagged maturation for the reserve
            if mode == DEPTH:
                campaign.depth += 1
            else:
                campaign.breadth += 1

        # MID-CAMPAIGN SYNTHESIS: once two positive findings cluster and no arc is frozen yet, the
        # synthesizer freezes a joint-prediction claim and enqueues it as its OWN confirmable claim with
        # a reserved slot, so close reads a verdict that already exists (SPEC 3, steps 33 / 58). It fires
        # mid-campaign, never at close, else the joint-prediction box could never be scored.
        if synthesizer is not None and campaign.arc is None:
            cluster = _positive_cluster(campaign.results)
            if cluster is not None:
                synth = synthesizer.synthesize([r.schema for r in cluster])
                if decide_arc(synth):
                    campaign.arc = synth
                    campaign.pending_arc.append(_joint_schema(cluster, synth))

        matured_scored = 1 if scored else 0
        return SupervisorState(
            gpu_hours_spent=state.gpu_hours_spent + 1.0,
            boxes_spent=state.boxes_spent + matured_scored,
            maturations=state.maturations + matured_scored)
    return step_fn


def run_loop(*, agent, backend, config, lease_store, box_factory, substrate, triage, reviewer,
             budget, halt_flag, health_gate, seed: int = 0, stall_limit: int = 40, veins=VEINS,
             significance=None, synthesizer=None):
    """Assemble + run the discovery loop to the budget, then close the campaign. Returns
    (terminal_reason, PoolReport, Campaign)."""
    from .pool import close_campaign
    from .supervisor import SupervisorState
    # DURABLE RESUME: reconcile any box orphaned by a prior crash, then rebuild the supervisor's
    # progress from the durable bank, so a restart neither re-scores a spent box (the (lineage,
    # purpose) one-grant bars it) nor exceeds the maturation budget across restarts. GPU-hours is not
    # durable and restarts at zero (a loose runaway guard, not the scientific ceiling).
    lease_store.resume()
    boxes_spent, maturations = lease_store.counts()
    state = SupervisorState(gpu_hours_spent=0.0, boxes_spent=boxes_spent, maturations=maturations)

    bandit = Bandit(arms=veins, seed=seed)  # the vein set IS the arm space (named arms)
    policy = SteeringPolicy()
    campaign = Campaign()
    step_fn = build_step_fn(agent=agent, backend=backend, config=config, lease_store=lease_store,
                            box_factory=box_factory, substrate=substrate, triage=triage,
                            reviewer=reviewer, bandit=bandit, policy=policy, campaign=campaign,
                            veins=veins, significance=significance, synthesizer=synthesizer)
    reason = run_supervisor(step_fn, budget, halt_flag, health_gate, state=state,
                            stall_limit=stall_limit)
    report = close_campaign(campaign.results)
    return reason, report, campaign
