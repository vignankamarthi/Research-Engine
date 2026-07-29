"""The holdout box lease: atomic box claim (the box, not the (hypothesis, box) pair,
is the serialization key), per-lineage one-grant, a two-marker (reserved -> label_read)
staged-score commit in one ACID store, and a lease-generation fence so a partitioned
orphan cannot read a reclaimed box. Resume is decidable: staged -> re-commit,
label_read-no-stage -> burn, reserved-only -> reclaim."""
import threading

import pytest

from referee.lease import FenceError, LeaseStore


def store(tmp_path, boxes=("box0",)):
    s = LeaseStore(str(tmp_path / "lease.db"))
    s.add_boxes(list(boxes))
    return s


def test_claim_returns_a_live_box_then_none_when_exhausted(tmp_path):
    s = store(tmp_path, boxes=("box0",))
    c = s.claim(hypothesis="h1", lineage="L1")
    assert c is not None and c.box_id == "box0"
    assert s.claim(hypothesis="h2", lineage="L2") is None  # no live box left


def test_two_different_lineages_cannot_share_one_live_box(tmp_path):
    s = store(tmp_path, boxes=("box0",))
    a = s.claim("hA", "LA")
    b = s.claim("hB", "LB")  # different lineage, but the box is already reserved
    assert a is not None and b is None


def test_same_lineage_retry_is_barred_after_a_spend(tmp_path):
    s = store(tmp_path, boxes=("box0", "box1"))
    c = s.claim("h1", "L1")
    s.mark_label_read(c.box_id, c.generation)
    s.stage(c.box_id, c.generation, verdict="confirmed", score=b"0.2")
    s.commit(c.box_id, c.generation)
    # a same-lineage reformulation is a relabel -> barred (one-grant)
    assert s.claim("h1b", "L1") is None


def test_generation_fence_rejects_a_stale_marker(tmp_path):
    s = store(tmp_path, boxes=("box0",))
    c = s.claim("h1", "L1")
    with pytest.raises(FenceError):
        s.mark_label_read(c.box_id, c.generation + 999)  # a partitioned orphan
    s.mark_label_read(c.box_id, c.generation)  # the real holder succeeds


def test_happy_path_commit(tmp_path):
    s = store(tmp_path)
    c = s.claim("h1", "L1")
    s.mark_label_read(c.box_id, c.generation)
    s.stage(c.box_id, c.generation, verdict="confirmed", score=b"0.2")
    s.commit(c.box_id, c.generation)
    assert s.box_status(c.box_id) == "spent"
    assert s.bank_verdict("L1") == ("box0", "confirmed", "spent")


def test_resume_staged_recommits_without_re_touch(tmp_path):
    s = store(tmp_path)
    c = s.claim("h1", "L1")
    s.mark_label_read(c.box_id, c.generation)
    s.stage(c.box_id, c.generation, verdict="confirmed", score=b"0.2")
    # crash before commit; a fresh store instance reconciles from the same db
    actions = LeaseStore(str(tmp_path / "lease.db")).resume()
    assert actions[c.box_id] == "recommitted"
    s2 = LeaseStore(str(tmp_path / "lease.db"))
    assert s2.box_status(c.box_id) == "spent"
    assert s2.bank_verdict("L1") == ("box0", "confirmed", "spent")  # staged verdict preserved


def test_resume_label_read_without_stage_burns(tmp_path):
    s = store(tmp_path)
    c = s.claim("h1", "L1")
    s.mark_label_read(c.box_id, c.generation)  # label may have been read; no staged score
    actions = LeaseStore(str(tmp_path / "lease.db")).resume()
    assert actions[c.box_id] == "burned"
    s2 = LeaseStore(str(tmp_path / "lease.db"))
    assert s2.box_status(c.box_id) == "burned"
    assert s2.bank_verdict("L1")[2] == "burned"  # durable one-grant record


def test_resume_reserved_only_reclaims(tmp_path):
    s = store(tmp_path)
    c = s.claim("h1", "L1")  # reserved, never read
    actions = LeaseStore(str(tmp_path / "lease.db")).resume()
    assert actions[c.box_id] == "reclaimed"
    assert LeaseStore(str(tmp_path / "lease.db")).box_status(c.box_id) == "live"


def test_atomic_claim_under_concurrency(tmp_path):
    # one live box, many threads racing to claim it -> exactly one winner
    db = str(tmp_path / "race.db")
    seed = LeaseStore(db)
    seed.add_boxes(["only"])
    winners = []
    lock = threading.Lock()

    def worker(i):
        got = LeaseStore(db).claim(f"h{i}", f"L{i}")
        if got is not None:
            with lock:
                winners.append(got.box_id)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert winners == ["only"]  # exactly one thread claimed the single box
