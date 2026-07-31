"""GROUNDED-provenance (PLAN 57e). Each candidate carries the DOIs/arXiv ids plus snippets its vein
came from. The ids are resolved in the TRUSTED process (the resolver is injected here). An
unresolvable provenance fails the CANDIDATE, never the campaign (no raise), and grounded-vs-ungrounded
is recorded per candidate so the drift claim is measurable."""
from engine.grounding import (
    GroundingResult,
    extract_provenance,
    ground_candidate,
    ground_wave,
)


def _cand(grounding, **kw):
    base = {"claim": "c", "claim_type": "effect", "grounding": grounding}
    base.update(kw)
    return base


def test_extract_provenance_finds_doi_and_arxiv():
    p = extract_provenance(_cand("see 10.1145/3581783 and arXiv:2401.12345 for the gap"))
    assert "10.1145/3581783" in p.dois
    assert "2401.12345" in p.arxiv_ids


def test_ground_candidate_fails_when_no_ids():
    r = ground_candidate(_cand("no identifier here"), resolver=lambda i: True)
    assert isinstance(r, GroundingResult) and r.grounded is False


def test_ground_candidate_fails_when_none_resolve():
    r = ground_candidate(_cand("10.1145/3581783"), resolver=lambda i: False)
    assert r.grounded is False and "resolve" in r.reason


def test_resolver_error_fails_candidate_not_campaign():
    def boom(i):
        raise RuntimeError("trusted resolver unreachable")

    # a resolver failure fails THIS candidate; it must not raise out (the campaign survives).
    r = ground_candidate(_cand("10.1145/3581783"), resolver=boom)
    assert r.grounded is False and "resolver error" in r.reason


def test_ground_candidate_grounded_when_resolves():
    r = ground_candidate(_cand("arXiv:2401.12345"), resolver=lambda i: True)
    assert r.grounded is True and "2401.12345" in r.resolved_ids


def test_ground_wave_records_grounded_and_ungrounded():
    cands = [
        _cand("10.1145/3581783"),      # resolves
        _cand("no id"),                # ungrounded: no id
        _cand("arXiv:2401.99999"),     # ungrounded: does not resolve
    ]

    def resolver(i):
        return i == "10.1145/3581783"

    rec = ground_wave(cands, resolver)
    assert len(rec.grounded) == 1 and len(rec.ungrounded) == 2
    # per-candidate grounded flag is stamped so drift is measurable
    assert rec.grounded[0]["grounded"] is True
    assert 0.0 < rec.rate() < 1.0
