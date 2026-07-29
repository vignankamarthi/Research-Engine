"""The environment-fingerprint recorder. Captures the node-INVARIANT environment that
determines whether a single-GPU score is reproducible: container digest, lib versions,
GPU model + compute capability, determinism flags. Hostname and GPU count are omitted on
purpose, so a node reschedule or a driver bump does not strand a resume. Resume checks are
byte-for-byte on the canonical form; a mismatch on an un-scored box PAGES."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from common.canonical import canonical_json_bytes, sha256_hex

# Resume-verify outcomes (named so callers compare a constant, not a magic literal).
MATCH = "MATCH"
PAGE = "PAGE"
SCORED_NO_RESCORE = "SCORED_NO_RESCORE"


@dataclass(frozen=True)
class Fingerprint:
    container_digest: str
    lib_versions: dict
    gpu_model: str
    compute_capability: str
    determinism_flags: dict

    def canonical_bytes(self) -> bytes:
        payload = {
            "container_digest": self.container_digest,
            "lib_versions": dict(sorted(self.lib_versions.items())),
            "gpu_model": self.gpu_model,
            "compute_capability": self.compute_capability,
            "determinism_flags": dict(sorted(self.determinism_flags.items())),
        }
        return canonical_json_bytes(payload)

    def digest(self) -> str:
        return sha256_hex(self.canonical_bytes())


@runtime_checkable
class EnvProbe(Protocol):
    def container_digest(self) -> str: ...
    def lib_versions(self) -> dict: ...
    def gpu(self) -> tuple: ...  # (model, compute_capability)
    def determinism_flags(self) -> dict: ...


def record_fingerprint(probe: EnvProbe) -> Fingerprint:
    model, cc = probe.gpu()
    return Fingerprint(
        container_digest=probe.container_digest(),
        lib_versions=dict(probe.lib_versions()),
        gpu_model=model,
        compute_capability=cc,
        determinism_flags=dict(probe.determinism_flags()),
    )


def resume_verify(recorded: Fingerprint, current: Fingerprint, box_scored: bool) -> str:
    """MATCH -> safe to proceed. PAGE -> un-scored box under a changed environment, escalate
    to a human (never a silent quarantine). SCORED_NO_RESCORE -> the box is already committed
    (touch-once), so the mismatch is noted but nothing is re-scored."""
    if recorded.digest() == current.digest():
        return MATCH
    if not box_scored:
        return PAGE
    return SCORED_NO_RESCORE


class MockEnvProbe:
    """Deterministic probe. hostname / gpu_count are accepted but ignored, which is exactly
    how node-invariance is enforced at the source."""

    def __init__(self, container, libs, gpu, flags, hostname="node00", gpu_count=1):
        self._container = container
        self._libs = libs
        self._gpu = gpu
        self._flags = flags

    def container_digest(self) -> str:
        return self._container

    def lib_versions(self) -> dict:
        return self._libs

    def gpu(self) -> tuple:
        return self._gpu

    def determinism_flags(self) -> dict:
        return self._flags
