"""Shared canonical serialization + hashing. This is a TRUST primitive: every signed digest in
the system (gate-config signature, control-catalog lock, signed consequence/incumbent catalogs,
environment fingerprint) routes through here. Keeping one implementation means a signed artifact
can never fail to verify because a copied encoder drifted (a changed separator, or ensure_ascii
flipping non-ASCII handling, would silently diverge the digest)."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding: sorted keys, compact separators, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_digest(obj: Any) -> str:
    """SHA-256 hex of the canonical JSON encoding of obj."""
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def sha256_hex(data: bytes) -> str:
    """SHA-256 hex of raw bytes, for non-JSON artifacts (e.g. score blobs)."""
    return hashlib.sha256(data).hexdigest()
