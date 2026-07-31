"""Trusted-process provenance resolvers for the grounding stage. A resolver takes ONE citation id (a
DOI or an arXiv id) and returns True iff it resolves to a real record. It runs OUTSIDE the generation
agent, in the trusted process, so a scout cannot forge its own grounding by asserting a citation. An
unrecognized or unreachable id fails CLOSED (False), which fails the candidate (never the campaign).
The HTTP check is injectable so the resolver is testable without a network."""
from __future__ import annotations

import re
import urllib.request

_ARXIV_ID = re.compile(r"([0-9]{4}\.[0-9]{4,5})")
_DOI = re.compile(r"(10\.\d{4,9}/[^\s]+)")


def _http_ok(url: str, *, timeout: float = 15.0, opener=urllib.request.urlopen) -> bool:
    """A HEAD that resolves to a 2xx/3xx is a real record. Any error (404, timeout, DNS) fails closed."""
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "research-engine/1.0 (grounding resolver)"})
    try:
        with opener(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 400
    except Exception:
        return False


def arxiv_doi_resolver(citation_id: str, *, http_ok=_http_ok) -> bool:
    """True iff the id resolves. A DOI checks doi.org, an arXiv id checks the arXiv abstract page.
    A DOI is tried first (a DOI never looks like an arXiv id, but an arXiv id can appear inside a DOI's
    suffix, so the more specific pattern wins). Anything unrecognized or unreachable fails closed."""
    s = str(citation_id or "").strip()
    doi = _DOI.search(s)
    if doi:
        return http_ok(f"https://doi.org/{doi.group(1)}")
    ax = _ARXIV_ID.search(s)
    if ax:
        return http_ok(f"https://arxiv.org/abs/{ax.group(1)}")
    return False
