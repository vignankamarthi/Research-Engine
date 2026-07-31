"""The mechanism gate. A causal ablation removes a meaningful chunk of the effect (the paired
full-minus-ablated CONTRAST exceeds the MIE), plus a specificity control. This certifies
ablation-and-specificity, not the drafted causal story.

The gate judges a CONTRAST, not two absolute levels. An earlier form required the ablated level
to fall below the MIE, which is unsatisfiable whenever the metric has a non-zero chance floor (an
MCQ task cannot score below chance), so a real mechanism could never be confirmed on such a task.
The magnitude gate separately establishes that the effect is present; the mechanism gate only has
to show that removing the named mechanism drops it by more than the interest floor."""
from __future__ import annotations


def mechanism_check(contrast_lo: float, mie: float, specificity_ok: bool) -> bool:
    """The paired (full minus ablated) contrast's LOWER CI must exceed the MIE (the ablation
    removes a drop larger than the interest floor), and the specificity control must hold."""
    return bool(contrast_lo > mie and specificity_ok)
