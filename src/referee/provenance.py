"""The executed-not-fabricated provenance gate. Scores are trusted only when they link back
to a real execution: the artifact's checksum matches what was recorded at run time, an
environment fingerprint is present, and the seeds were declared. This is a standing cheap
sanity check on every run, and the answer to the adversarial 'fabricated numbers / undeclared
seeds' attack."""
from __future__ import annotations

from dataclasses import dataclass

from common.canonical import sha256_hex


@dataclass(frozen=True)
class RunProvenance:
    artifact_checksum: str
    fingerprint_digest: str
    seeds: tuple


def artifact_checksum(data: bytes) -> str:
    return sha256_hex(data)


def executed_not_fabricated(scores_bytes: bytes, provenance: RunProvenance) -> bool:
    if not provenance.fingerprint_digest:
        return False  # no environment fingerprint: not a real executed run
    if not provenance.seeds:
        return False  # undeclared seeds
    return artifact_checksum(scores_bytes) == provenance.artifact_checksum
