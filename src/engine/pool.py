"""Campaign close: the results pool and the selection correction. Score-time verdicts
are provisional; at close, BH over the matured-and-scored family (the selection family,
since the headline is a max over that many looks) N-adjusts the threshold, and the
expected false-positive count (N x alpha) is reported honestly. A submit-bound finding
is a CONFIRMED/STRONG that clears the corrected threshold; the second fresh-box
replication conjunction gate is the runtime step noted here."""
from __future__ import annotations

from dataclasses import dataclass

from gatelib import benjamini_hochberg

DELIVERABLE = "deliverable_arc_confirmed"
NO_ARC = "no_foundational_arc"
_POSITIVE = ("CONFIRMED", "STRONG")


@dataclass
class PoolReport:
    submitted: list        # CampaignResults that cleared selection and are positive
    n_scored: int          # the selection family size N
    expected_false_family: float
    narrative: str


@dataclass
class FamilyReport:
    families: dict                     # family key -> list of member lineage keys
    per_lineage_expected_false: dict   # lineage key -> alpha (post-correction bound)
    per_family_expected_false: dict    # family key -> family_size * alpha
    campaign_expected_false: float     # N * alpha


def depth_completion(lead_arc_joint_confirmed: bool) -> str:
    """Did the lead arc's joint-prediction claim (frozen at synthesis) confirm on its own box?
    If not, there is no foundational arc this campaign and the breadth findings are labeled
    not-the-deliverable rather than dressed up as one."""
    return DELIVERABLE if lead_arc_joint_confirmed else NO_ARC


def replication_conjunction(primary_status: str, replication_status: str,
                            replication_magnitude: str) -> bool:
    """The second fresh-box gate: BOTH the primary and the replication box must confirm, and
    the magnitude read from the UNBIASED replication box must still exceed the MIE."""
    return (
        primary_status in _POSITIVE
        and replication_status in _POSITIVE
        and replication_magnitude == "exceeds_mie"
    )


@dataclass
class CampaignClose:
    depth_status: str          # DELIVERABLE | NO_ARC
    submitted: list            # findings that cleared selection AND replicated on a second box
    family_report: "FamilyReport"
    dossier: str               # the human-facing summary the GO/NO-GO is made on


def finalize_campaign(results, replications: dict, arcs: dict, lead_arc_confirmed: bool,
                      alpha: float = 0.05) -> CampaignClose:
    """Compose the full close. `replications[lineage] = (status, magnitude)` from the second
    fresh box; `arcs[lineage] = arc-name or None`. The GO/NO-GO stays the human's."""
    pool = close_campaign(results, alpha)  # selection correction over the matured-and-scored family

    replicated = [
        r for r in pool.submitted
        if r.lineage in replications
        and replication_conjunction(r.verdict.status, *replications[r.lineage])
    ]
    fr = group_and_report([(r.lineage, arcs.get(r.lineage)) for r in replicated], alpha)
    depth = depth_completion(lead_arc_confirmed)
    return CampaignClose(depth, replicated, fr, _build_dossier(depth, pool, replicated, fr))


def _build_dossier(depth, pool, replicated, fr) -> str:
    lines = []
    if depth == DELIVERABLE:
        lines.append("Depth: the lead arc's joint-prediction claim CONFIRMED on its own box (foundational deliverable).")
    else:
        lines.append("Depth: NO foundational arc this campaign; the findings below are breadth-only, not the deliverable.")
    lines.append(
        f"Selection: {pool.n_scored} matured-and-scored, {len(pool.submitted)} cleared the N-adjusted "
        f"threshold, {len(replicated)} of those also replicated on a second fresh box."
    )
    lines.append(f"Expected false families campaign-wide: {fr.campaign_expected_false:.2f}.")
    for family_key, members in fr.families.items():
        label = "independent" if family_key.startswith("__independent__") else f"arc '{family_key}'"
        lines.append(f"- {label}: {len(members)} finding(s), expected-false {fr.per_family_expected_false[family_key]:.2f}.")
    lines.append("GO/NO-GO is Vignan's: a confirmed-but-not-worth-submitting exit is a valid, graceful outcome.")
    return "\n".join(lines)


def group_and_report(findings, alpha: float = 0.05) -> FamilyReport:
    """Overlap grouping: a finding with no arc is its own family (ships separately); findings
    sharing an arc merge into one family (ship as one paper under a family-wise count)."""
    families: dict = {}
    per_lineage: dict = {}
    for lk, arc in findings:
        family_key = arc if arc else f"__independent__:{lk}"
        families.setdefault(family_key, []).append(lk)
        per_lineage[lk] = alpha
    per_family = {k: len(v) * alpha for k, v in families.items()}
    campaign = len(findings) * alpha
    return FamilyReport(families, per_lineage, per_family, campaign)


_SCORED_STATUSES = ("CONFIRMED", "STRONG", "CONFIRMED_EFFECT", "CONFIRMED_NEGATIVE",
                    "FAILED", "INCONCLUSIVE")


def close_campaign(results, alpha: float = 0.05) -> PoolReport:
    # EVERY box-touching result is a look in the selection family (matured-and-scored), whether or not
    # it carries a scalar p-value. A law_shape fit has no one-sided p-value, so it is COUNTED in N
    # (keeping the others' N-adjusted threshold honest for the true number of looks) but is not itself
    # submitted through BH -- a proper permutation statistic for it is deferred.
    scored = [r for r in results if r.verdict is not None and r.verdict.status in _SCORED_STATUSES]
    if not scored:
        return PoolReport([], 0, 0.0, "Campaign closed: no matured-and-scored hypotheses.")

    n = len(scored)
    bh_scored = [r for r in scored if r.verdict.pvalue is not None]
    bh = benjamini_hochberg([r.verdict.pvalue for r in bh_scored], alpha, n_tests=n)
    submitted = [
        r for r, rejected in zip(bh_scored, bh.rejected)
        if rejected and r.verdict.status in ("CONFIRMED", "STRONG")
    ]
    expected_false = n * alpha

    lines = [
        f"Campaign closed. {n} matured-and-scored, {len(submitted)} submit-bound after the "
        f"N-adjusted selection correction.",
        f"Expected false positives across the family (N x alpha): {expected_false:.2f}.",
    ]
    for r in submitted:
        lines.append(f"- SUBMIT (pending a second fresh-box replication): {r.narrative}")
    return PoolReport(submitted, n, expected_false, "\n".join(lines))
