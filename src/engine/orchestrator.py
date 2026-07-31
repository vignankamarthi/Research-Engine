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
from .generation import GENERATIVE_VEINS, VEINS, choose_mode
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


@dataclass
class Campaign:
    """The accumulating state of the assembled loop, closed at the base case."""
    results: list = field(default_factory=list)
    negative_bank: set = field(default_factory=set)  # lineage keys of dead ends, steered around
    depth: int = 0
    breadth: int = 0


def _reward(verdict) -> float:
    if verdict is None:
        return 0.0
    if verdict.status in ("CONFIRMED", "STRONG"):
        return 1.0
    if verdict.status in ("CONFIRMED_EFFECT", "CONFIRMED_NEGATIVE"):
        return 0.5
    return 0.1  # a scored FAILED / INCONCLUSIVE still informed the search


def build_step_fn(*, agent, backend, config, lease_store, box_factory, substrate, triage,
                  reviewer, bandit, policy, campaign, veins=VEINS, significance=None):
    def step_fn(state: SupervisorState) -> SupervisorState:
        trial, arm = bandit.ask()
        vein = bandit.arm_label(arm)  # a NAMED vein, the bandit's arm space is the vein set now
        # generative veins bias toward BREADTH, derivative toward DEPTH; steering then honors the
        # depth-floor / breadth-cap on the margin.
        pick = BREADTH if vein in GENERATIVE_VEINS else DEPTH
        mode = choose_mode(pick, campaign.depth, campaign.breadth, policy)
        context = {"vein": vein, "mode": mode, "negative_bank": sorted(campaign.negative_bank)}

        result = run_campaign(agent, backend, config, lease_store, box_factory,
                              substrate=substrate, triage=triage, context=context, reviewer=reviewer,
                              significance=significance)
        campaign.results.append(result)
        bandit.tell(trial, _reward(result.verdict))

        scored = result.verdict is not None and result.verdict.status in _SCORED
        if scored and result.verdict.status in _NEGATIVE:
            campaign.negative_bank.add(result.lineage)
        if mode == DEPTH:
            campaign.depth += 1
        else:
            campaign.breadth += 1

        matured_scored = 1 if scored else 0
        return SupervisorState(
            gpu_hours_spent=state.gpu_hours_spent + 1.0,
            boxes_spent=state.boxes_spent + matured_scored,
            maturations=state.maturations + matured_scored)
    return step_fn


def run_loop(*, agent, backend, config, lease_store, box_factory, substrate, triage, reviewer,
             budget, halt_flag, health_gate, seed: int = 0, stall_limit: int = 40, veins=VEINS,
             significance=None):
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
                            veins=veins, significance=significance)
    reason = run_supervisor(step_fn, budget, halt_flag, health_gate, state=state,
                            stall_limit=stall_limit)
    report = close_campaign(campaign.results)
    return reason, report, campaign
