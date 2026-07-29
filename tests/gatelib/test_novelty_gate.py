"""The novelty gate around the fail-closed collision check. It runs at two checkpoints
(before box allocation, and refreshed at submit), and a stale or unreachable corpus returns
HALT_RETRY, never REJECT or PROCEED, so a rate-limited corpus never burns a scored box. The
submit refresh is what catches prior art that appeared after allocation."""
from gatelib.novelty import CorpusStatus, novelty_gate

FRESH = CorpusStatus(reachable=True, as_of_t=1000.0)


def test_novel_claim_proceeds():
    d = novelty_gate(False, ["a"], True, FRESH, now=1005.0, max_staleness_s=3600, checkpoint="pre_allocation")
    assert d.decision == "PROCEED"


def test_collision_rejects():
    d = novelty_gate(True, ["a"], True, FRESH, now=1005.0, max_staleness_s=3600)
    assert d.decision == "REJECT"


def test_unreachable_corpus_halt_retries_and_never_burns_a_box():
    d = novelty_gate(False, ["a"], True, CorpusStatus(reachable=False, as_of_t=1000.0),
                     now=1005.0, max_staleness_s=3600)
    assert d.decision == "HALT_RETRY"
    assert d.reason == "corpus_unreachable"


def test_stale_corpus_halt_retries():
    d = novelty_gate(False, ["a"], True, CorpusStatus(reachable=True, as_of_t=1000.0),
                     now=99999.0, max_staleness_s=3600)
    assert d.decision == "HALT_RETRY"
    assert d.reason == "corpus_stale"


def test_submit_refresh_catches_new_prior_art():
    pre = novelty_gate(False, ["a"], True, FRESH, now=1005.0, max_staleness_s=3600,
                       checkpoint="pre_allocation")
    submit = novelty_gate(True, ["a", "new_paper"], True, CorpusStatus(True, 5000.0),
                          now=5005.0, max_staleness_s=3600, checkpoint="submit")
    assert pre.decision == "PROCEED"
    assert submit.decision == "REJECT"
    assert submit.checkpoint == "submit"
