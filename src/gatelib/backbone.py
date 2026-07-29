"""The backbone contamination gate. Post-cutoff data ORIGIN date (not the engine's
capture date) AND a membership-verified clean split. Where membership is
unverifiable, the finding passes but is labeled 'origin-date-verified only' and a
stricter origin margin applies."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional


@dataclass(frozen=True, slots=True)
class BackboneResult:
    passed: bool
    label: str  # "clean" | "origin_date_verified_only" | "contaminated"


def backbone_check(box_origin: Optional[date], backbone_cutoff: date,
                   membership_clean: Optional[bool], margin_days: int = 0) -> BackboneResult:
    if box_origin is None:
        raise ValueError("box origin date is required to verify backbone cleanliness")
    if box_origin <= backbone_cutoff + timedelta(days=margin_days):
        return BackboneResult(False, "contaminated")
    if membership_clean is False:
        return BackboneResult(False, "contaminated")
    if membership_clean is True:
        return BackboneResult(True, "clean")
    return BackboneResult(True, "origin_date_verified_only")
