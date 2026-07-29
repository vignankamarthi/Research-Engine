"""At-use signature verification for the gate config. The signature is checked
INSIDE the trusted process at every use, not once at preflight, because arbitrary
code runs on the compute node and a preflight-only check is a TOCTOU window. A
valid signature is necessary but not sufficient: the schema is re-enforced at use
so a compromised signer cannot smuggle an out-of-range acceptance constant."""
from __future__ import annotations

import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .schema import GateConfig, canonical_bytes, validate_config


class SignatureError(Exception):
    """The signed gate config failed verification: tampered, wrong key, or malformed."""


def sign_config(cfg: GateConfig, private_key_bytes: bytes) -> bytes:
    """Offline signing (Vignan's Mac holds the real private key). Produces a self-describing
    envelope over the config's canonical bytes. The envelope's key_id IS the config's own key_id,
    so the advertised key can never disagree with the signed body."""
    body = canonical_bytes(cfg)
    sig = Ed25519PrivateKey.from_private_bytes(private_key_bytes).sign(body)
    envelope = {"body": body.decode("utf-8"), "sig": sig.hex(), "key_id": cfg.key_id}
    return json.dumps(envelope).encode("utf-8")


def verify_config(signed: bytes, public_key_bytes: bytes) -> GateConfig:
    """Verify the envelope signature against the baked-in public key, re-enforce the schema, and
    bind the envelope's advertised key_id to the signed body. Any envelope/parse/signature failure
    (or a key_id that disagrees with the body) raises SignatureError; a validly signed but
    schema-invalid config raises ConfigError."""
    try:
        envelope = json.loads(signed)
        body = envelope["body"]
        sig_hex = envelope["sig"]
        env_key_id = envelope["key_id"]
        if not isinstance(body, str) or not isinstance(sig_hex, str):
            raise ValueError("malformed envelope fields")
        body_bytes = body.encode("utf-8")
        sig = bytes.fromhex(sig_hex)
    except (ValueError, KeyError, TypeError) as e:
        raise SignatureError(f"malformed signed config: {e}") from e

    public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
    try:
        public_key.verify(sig, body_bytes)
    except InvalidSignature as e:
        raise SignatureError("signature verification failed") from e

    # Signature valid -> enforce the schema at use (belt and suspenders on the signer).
    config = validate_config(json.loads(body))
    # Bind the advertised key_id to the signed body: the envelope cannot claim a different key
    # than the one the config committed to.
    if env_key_id != config.key_id:
        raise SignatureError(
            f"envelope key_id {env_key_id!r} does not match signed config key_id {config.key_id!r}"
        )
    return config
