"""A digest of the gate library's public shape, so the signed config can PIN which gates and
verdicts the library exposes. Mirrors referee.lineage's control-catalog lock: a declarative hash,
verified at use, so adding / removing / renaming a public gate symbol is caught rather than passing
silently. (It pins the API surface; a change to a gate's internal logic is out of scope here and is
what the two-sided validation suite guards.)"""
from __future__ import annotations

from common.canonical import canonical_digest


class GateLibraryError(Exception):
    """The gate library's public shape does not match the digest the config was signed over."""


def library_digest() -> str:
    import gatelib
    return canonical_digest(sorted(gatelib.__all__))


def verify_gate_library(expected_digest: str) -> None:
    actual = library_digest()
    if actual != expected_digest:
        raise GateLibraryError(
            f"gate library digest mismatch: expected {expected_digest}, got {actual}"
        )
