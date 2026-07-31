"""The trusted-process provenance resolver. A DOI resolves via doi.org, an arXiv id via arxiv.org, and
anything unrecognized or unreachable fails CLOSED. The HTTP check is injected, so no network is used."""
from engine.resolvers import arxiv_doi_resolver


def _ok_for(substr):
    return lambda url, **kw: substr in url


def test_arxiv_id_resolves_against_the_abstract_page():
    assert arxiv_doi_resolver("arXiv:2401.12345", http_ok=_ok_for("arxiv.org/abs/2401.12345")) is True


def test_bare_arxiv_id_resolves():
    assert arxiv_doi_resolver("2401.12345", http_ok=_ok_for("arxiv.org/abs/2401.12345")) is True


def test_doi_resolves_against_doi_org():
    assert arxiv_doi_resolver("10.1007/978-3-031-73013-9_23",
                              http_ok=_ok_for("doi.org/10.1007")) is True


def test_doi_wins_over_an_embedded_arxiv_looking_suffix():
    # a DOI whose suffix contains 4.4 digits must resolve as a DOI, not be mis-parsed as an arXiv id
    called = {}

    def http_ok(url, **kw):
        called["url"] = url
        return True

    arxiv_doi_resolver("10.1234/2401.12345", http_ok=http_ok)
    assert called["url"].startswith("https://doi.org/")


def test_unrecognized_id_fails_closed():
    assert arxiv_doi_resolver("not-a-citation", http_ok=lambda url, **kw: True) is False


def test_unreachable_id_fails_closed():
    assert arxiv_doi_resolver("arXiv:2401.12345", http_ok=lambda url, **kw: False) is False


def test_empty_fails_closed():
    assert arxiv_doi_resolver("", http_ok=lambda url, **kw: True) is False
