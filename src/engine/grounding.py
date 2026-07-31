"""GROUNDED-provenance (PLAN 57e). A blind scout must NAME the real work its vein came from, and
that provenance must resolve in the TRUSTED process (an arXiv/DOI lookup outside the generation
agent, injected here as `resolver`). An unresolvable id fails the CANDIDATE, never the campaign, and
each candidate is stamped grounded-vs-ungrounded so the drift claim ("grounded candidates drift
less") is measurable rather than asserted. This is the provenance half of the anti-drift fix; the
scout-isolation module is the other half."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# A DOI ("10.xxxx/...") and an arXiv id ("2401.12345", optional version). Kept deliberately narrow so
# a bare year or a section number is not mistaken for a real identifier.
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_ARXIV_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")


@dataclass(frozen=True)
class Provenance:
    dois: tuple
    arxiv_ids: tuple
    snippets: tuple
    raw: str


@dataclass(frozen=True)
class GroundingResult:
    grounded: bool
    reason: str
    provenance: Provenance
    resolved_ids: tuple = ()


@dataclass
class GroundingRecord:
    """The wave's grounded-vs-ungrounded ledger. `grounded` holds the stamped surviving candidates;
    `ungrounded` holds (stamped-candidate, reason) for the failed ones. `rate()` is the measurable
    grounded fraction the drift analysis reads."""
    grounded: list = field(default_factory=list)
    ungrounded: list = field(default_factory=list)

    def rate(self) -> float:
        total = len(self.grounded) + len(self.ungrounded)
        return len(self.grounded) / total if total else 0.0


def extract_provenance(candidate: dict) -> Provenance:
    """Pull DOIs, arXiv ids, and any retrieved snippets out of a candidate's `grounding` field."""
    raw = str(candidate.get("grounding", ""))
    dois = tuple(_DOI_RE.findall(raw))
    arxiv = tuple(m.group(0) for m in _ARXIV_RE.finditer(raw))
    snippets = tuple(candidate.get("snippets", []) or [])
    return Provenance(dois=dois, arxiv_ids=arxiv, snippets=snippets, raw=raw)


def ground_candidate(candidate: dict, resolver) -> GroundingResult:
    """Resolve a candidate's provenance in the trusted process. `resolver(id) -> bool` runs OUTSIDE
    the generation agent. No id, no resolution, or a resolver error all fail the CANDIDATE (no
    raise); the campaign is untouched."""
    prov = extract_provenance(candidate)
    ids = prov.dois + prov.arxiv_ids
    if not ids:
        return GroundingResult(False, "no DOI or arXiv id in grounding", prov)
    resolved = []
    for i in ids:
        try:
            ok = resolver(i)
        except Exception as e:  # a trusted-resolver failure fails THIS candidate, not the campaign
            return GroundingResult(False, f"resolver error for {i}: {e}", prov)
        if ok:
            resolved.append(i)
    if not resolved:
        return GroundingResult(False, "no id resolved in the trusted process", prov)
    return GroundingResult(True, "resolved", prov, tuple(resolved))


def ground_wave(candidates, resolver) -> GroundingRecord:
    """Ground every candidate, stamping each with `grounded` / `grounding_reason` / `resolved_ids`
    and splitting into grounded vs ungrounded. Never raises: an ungrounded candidate is dropped from
    the survivors and recorded, so drift stays measurable."""
    rec = GroundingRecord()
    for c in candidates:
        r = ground_candidate(c, resolver)
        stamped = {**c, "grounded": r.grounded, "grounding_reason": r.reason,
                   "resolved_ids": list(r.resolved_ids)}
        if r.grounded:
            rec.grounded.append(stamped)
        else:
            rec.ungrounded.append((stamped, r.reason))
    return rec
