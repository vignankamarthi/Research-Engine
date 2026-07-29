"""The importance-consequence gate. The pre-registered templated consequence must be
confirmed on held-out data AND the incumbent's predicted value must be separated from
the claimed value at MIE. An effect that confirms without discharging its consequence
caps at CONFIRMED-EFFECT."""
from __future__ import annotations


def consequence_check(consequence_confirmed: bool, incumbent_separated_at_mie: bool) -> bool:
    return bool(consequence_confirmed and incumbent_separated_at_mie)
