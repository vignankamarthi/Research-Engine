"""The environment-fingerprint recorder. Node-INVARIANT by construction: it records the
container digest, lib versions, GPU model + compute capability, and determinism flags, and
deliberately ignores hostname / GPU count (confirmatory scoring is single-GPU, so a node or
driver change must not strand a resume). Resume verification is byte-for-byte; a mismatch on
an UN-scored box PAGES rather than silently quarantining."""
from engine.fingerprint import Fingerprint, MockEnvProbe, record_fingerprint, resume_verify


def test_digest_is_deterministic_and_dict_order_independent():
    a = Fingerprint("sha256:c", {"torch": "2.11", "numpy": "2.0"}, "B200", "10.0", {"det": True})
    b = Fingerprint("sha256:c", {"numpy": "2.0", "torch": "2.11"}, "B200", "10.0", {"det": True})
    assert a.digest() == b.digest()


def test_record_is_node_invariant():
    a = MockEnvProbe("sha256:x", {"torch": "2.11"}, ("B200", "10.0"), {"det": True},
                     hostname="node01", gpu_count=1)
    b = MockEnvProbe("sha256:x", {"torch": "2.11"}, ("B200", "10.0"), {"det": True},
                     hostname="node47", gpu_count=8)
    assert record_fingerprint(a).digest() == record_fingerprint(b).digest()


def test_record_differs_on_lib_version():
    a = MockEnvProbe("c", {"torch": "2.11"}, ("B200", "10.0"), {})
    b = MockEnvProbe("c", {"torch": "2.10"}, ("B200", "10.0"), {})
    assert record_fingerprint(a).digest() != record_fingerprint(b).digest()


def test_record_differs_on_gpu_model():
    a = MockEnvProbe("c", {}, ("B200", "10.0"), {})
    b = MockEnvProbe("c", {}, ("RTX PRO 6000", "12.0"), {})
    assert record_fingerprint(a).digest() != record_fingerprint(b).digest()


def test_resume_match_when_identical():
    fp = record_fingerprint(MockEnvProbe("c", {}, ("g", "1"), {}))
    assert resume_verify(fp, fp, box_scored=False) == "MATCH"


def test_resume_mismatch_on_unscored_box_pages():
    a = record_fingerprint(MockEnvProbe("c", {"torch": "2.11"}, ("g", "1"), {}))
    b = record_fingerprint(MockEnvProbe("c", {"torch": "2.10"}, ("g", "1"), {}))
    assert resume_verify(a, b, box_scored=False) == "PAGE"


def test_resume_mismatch_on_scored_box_does_not_rescore():
    a = record_fingerprint(MockEnvProbe("c", {"torch": "2.11"}, ("g", "1"), {}))
    b = record_fingerprint(MockEnvProbe("c", {"torch": "2.10"}, ("g", "1"), {}))
    assert resume_verify(a, b, box_scored=True) == "SCORED_NO_RESCORE"
