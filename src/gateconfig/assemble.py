"""Assemble and sign the gate config. This is the seam between the HUMAN-owned trust inputs and
the CODE-derived ones. The acceptance constants (alpha, MIE floor, claim types) and the private
key are Vignan's, they never come from Claude or the running loop. This module only fills in the
machine digests (the gate library's code shape, the control catalog, the three signed catalogs)
and drives the existing signer. Signing the config transitively pins the catalogs, so once signed
the loop cannot swap in an easier incumbent or a lower MIE without breaking verification. The
private key is passed in by the caller (read from Vignan's Mac); it never lives or persists here."""
from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gatelib import library_digest
from referee.catalog import catalog_digest
from referee.lineage import control_catalog_digest

from .schema import validate_config
from .signing import sign_config

# The three signed catalogs, mapped to the GateConfig digest field each one pins.
_CATALOG_FILES = {
    "consequence_catalog_digest": "consequence_templates.json",
    "incumbent_catalog_digest": "incumbent_catalog.json",
    "mie_distribution_digest": "mie_distribution.json",
}


def generate_keypair() -> tuple[bytes, bytes]:
    """A fresh Ed25519 keypair as (private_raw, public_raw), 32 bytes each. Vignan runs this once on
    his Mac. The private bytes stay on the Mac; the public bytes get baked into the verifying
    process so the referee can check the signature."""
    priv = Ed25519PrivateKey.generate()
    private_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private_raw, public_raw


def build_signed_config(*, template: dict, catalogs_dir, private_key_bytes: bytes) -> bytes:
    """Merge the human's acceptance constants (template) with the code-derived digests, validate,
    and sign. Returns the envelope the referee's verify_config accepts. Keys beginning with `_`
    (human notes) are ignored, so a template comment never changes the signed bytes."""
    catalogs_dir = Path(catalogs_dir)
    data = {k: v for k, v in template.items() if not k.startswith("_")}
    data["gate_library_digest"] = library_digest()
    data["control_catalog_hash"] = control_catalog_digest()
    for field, filename in _CATALOG_FILES.items():
        catalog = json.loads((catalogs_dir / filename).read_text())
        data[field] = catalog_digest(catalog)
    cfg = validate_config(data)
    return sign_config(cfg, private_key_bytes)
