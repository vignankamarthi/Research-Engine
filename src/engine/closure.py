"""The box-budget CLOSURE accounting for a campaign.

A campaign spends the touch-once holdout as physical BOXES, and a finding only becomes
submit-bound once its lineage can also draw a fresh REPLICATION box (see the lease's
`(lineage, purpose)` one-grant). So the physical pool must be provably large enough for the
FULL signed demand BEFORE the campaign starts, or a finding could reach maturity with no box
left to confirm it. The accounting has four signed categories:

  - primary_demand  -- one primary box per planned maturation (the CONSUMABLE allotment).
  - replication     -- a per-family reserve so every maturation can draw its mandatory second box.
  - rescore         -- a correlated re-score-and-burn contingency (a crash burns a box and the
                       one guarded re-score draws another; correlated crashes need slack).
  - backbone        -- a backbone-cohort reserve for the mandatory untrained/weights-randomized
                       FLOOR control that every claim-type carries.

`Budget.max_boxes` is NOT the raw pool size. It is the REDUCED ceiling, the pool minus the three
reserves HELD BACK beyond the primary allotment, so the supervisor's box base case fires while
those reserves still physically exist (the exact analog of step 6's GPU-hour rule, base case at
hard_cap minus the held-back reserve). Closure is then simply: the reduced ceiling still covers
the primary demand, equivalently the pool covers the total demand. The reserve MAGNITUDES are a
signed-config input; this module only does the arithmetic and the fail-closed refusal."""
from __future__ import annotations

from dataclasses import dataclass


class ClosureError(Exception):
    """The reserves do not CLOSE against the live-box pool: the accounting is over-subscribed,
    so the campaign is refused before it starts (fail closed, never a partial run that strands a
    matured finding with no box to confirm it)."""


@dataclass(frozen=True, slots=True)
class Reserves:
    primary_demand: int  # consumable: one primary box per planned maturation
    replication: int     # held back: per-family mandatory second-box reserve
    rescore: int         # held back: correlated re-score-and-burn contingency
    backbone: int        # held back: untrained/weights-randomized FLOOR cohort reserve

    def held_back(self) -> int:
        """The reserves kept BEYOND the primary allotment. Subtracting these from the pool gives
        the reduced ceiling, so the base case fires while they still exist."""
        return self.replication + self.rescore + self.backbone

    def total_demand(self) -> int:
        """Every box the signed accounting must cover. The pool CLOSES only if it is at least this."""
        return self.primary_demand + self.held_back()

    @classmethod
    def for_campaign(cls, max_maturations: int, *, replication_per_maturation: int = 1,
                     rescore_contingency: int = 2, backbone_cohort: int = 2) -> "Reserves":
        """Size the four categories off the maturation ceiling. Defaults are sane starting values;
        the REAL magnitudes come from the offline-signed gate config, never chosen by the loop."""
        return cls(
            primary_demand=max_maturations,
            replication=max_maturations * replication_per_maturation,
            rescore=rescore_contingency,
            backbone=backbone_cohort,
        )


def closes(live_boxes: int, reserves: Reserves) -> bool:
    """Does the live-box pool cover the full signed demand?"""
    return live_boxes >= reserves.total_demand()


def derive_max_boxes(live_boxes: int, reserves: Reserves) -> int:
    """The reduced box ceiling: the pool minus the HELD-BACK reserves. The supervisor's box base
    case fires here, leaving replication + rescore + backbone boxes physically available."""
    return live_boxes - reserves.held_back()


def validate_closure(live_boxes: int, reserves: Reserves) -> int:
    """Refuse to start unless the reserves close against the live boxes; otherwise return the
    derived `max_boxes` ceiling. This is the SINGLE enforcement point: call it at campaign start
    AND re-run it on any ceiling raise (a raise grows the reserves, so the accounting must be
    re-checked against the pool). Raises `ClosureError` fail-closed when over-subscribed."""
    if not closes(live_boxes, reserves):
        raise ClosureError(
            f"box reserves do not close: pool={live_boxes} < demand={reserves.total_demand()} "
            f"(primary={reserves.primary_demand}, replication={reserves.replication}, "
            f"rescore={reserves.rescore}, backbone={reserves.backbone})"
        )
    return derive_max_boxes(live_boxes, reserves)
