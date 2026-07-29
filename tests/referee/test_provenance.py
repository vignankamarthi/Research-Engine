"""The executed-not-fabricated gate. A score artifact is trusted only when it links back to a
real execution: its checksum matches the checksum recorded at run time, an environment
fingerprint is present, and the seeds were declared. Fabricated numbers fail the checksum;
a run with no fingerprint or undeclared seeds is not a real run. This closes the adversarial
'fabricated numbers / undeclared seeds' vector."""
from referee.provenance import RunProvenance, artifact_checksum, executed_not_fabricated


def test_genuine_run_passes():
    data = b"per-item-scores-blob"
    prov = RunProvenance(artifact_checksum(data), fingerprint_digest="fp-abc", seeds=(1, 2))
    assert executed_not_fabricated(data, prov) is True


def test_fabricated_numbers_fail_the_checksum():
    prov = RunProvenance(artifact_checksum(b"the-real-scores"), "fp-abc", (1,))
    assert executed_not_fabricated(b"different-fabricated-scores", prov) is False


def test_run_with_no_fingerprint_is_not_a_real_run():
    data = b"x"
    prov = RunProvenance(artifact_checksum(data), fingerprint_digest="", seeds=(1,))
    assert executed_not_fabricated(data, prov) is False


def test_undeclared_seeds_fail():
    data = b"x"
    prov = RunProvenance(artifact_checksum(data), fingerprint_digest="fp-abc", seeds=())
    assert executed_not_fabricated(data, prov) is False


def test_checksum_is_deterministic():
    assert artifact_checksum(b"abc") == artifact_checksum(b"abc")
    assert artifact_checksum(b"abc") != artifact_checksum(b"abd")
