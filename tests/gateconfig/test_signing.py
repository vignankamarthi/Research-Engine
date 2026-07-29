"""At-use signature verification. The referee re-verifies the config's signature
INSIDE the trusted process at every use (a preflight-only check is a TOCTOU window
given arbitrary code runs on the node). The signing key lives only on Vignan's Mac;
these tests use an ephemeral test keypair, never the real key."""
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateconfig import (
    validate_config, sign_config, verify_config, canonical_bytes,
    ConfigError, SignatureError,
)


def valid_dict(**overrides):
    d = {
        "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.01, "mie_floor": 0.03,
        "claim_types": ["effect", "capability"],
        "gate_library_digest": "sha256:" + "0" * 64,
        "control_catalog_hash": "sha256:" + "1" * 64,
        "key_id": "test-key",
    }
    d.update(overrides)
    return d


def keypair():
    sk = Ed25519PrivateKey.generate()
    return sk.private_bytes_raw(), sk.public_key().public_bytes_raw()


def test_sign_then_verify_roundtrip():
    priv, pub = keypair()
    cfg = validate_config(valid_dict())
    signed = sign_config(cfg, priv)
    assert isinstance(signed, (bytes, bytearray))
    out = verify_config(signed, pub)
    assert canonical_bytes(out) == canonical_bytes(cfg)


def test_envelope_key_id_must_match_the_signed_body():
    import json
    priv, pub = keypair()
    cfg = validate_config(valid_dict())  # key_id "test-key"
    env = json.loads(sign_config(cfg, priv))
    env["key_id"] = "attacker-key"  # advertise a different key than the body committed to
    with pytest.raises(SignatureError):
        verify_config(json.dumps(env).encode(), pub)


def test_tampered_body_fails_verification():
    priv, pub = keypair()
    signed = bytearray(sign_config(validate_config(valid_dict()), priv))
    # flip a byte somewhere in the middle of the envelope
    signed[len(signed) // 2] ^= 0x01
    with pytest.raises(SignatureError):
        verify_config(bytes(signed), pub)


def test_wrong_public_key_fails_verification():
    priv, _ = keypair()
    _, other_pub = keypair()
    signed = sign_config(validate_config(valid_dict()), priv)
    with pytest.raises(SignatureError):
        verify_config(signed, other_pub)


def test_verify_also_enforces_the_schema_at_use():
    # A validly-signed but schema-invalid config must still be rejected at use,
    # so a compromised signer cannot smuggle an out-of-range acceptance constant.
    priv, pub = keypair()
    bad = valid_dict(alpha=1.5)  # out of range; bypass validate_config on purpose
    signed = sign_config.__wrapped__(bad, priv, "test-key") if hasattr(sign_config, "__wrapped__") \
        else _sign_raw_dict(bad, priv, "test-key")
    with pytest.raises(ConfigError):
        verify_config(signed, pub)


def _sign_raw_dict(d, priv, key_id):
    """Sign a raw (possibly-invalid) dict, mirroring sign_config's envelope so the
    at-use validation path can be exercised without going through validate_config."""
    import json
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    body = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    sig = Ed25519PrivateKey.from_private_bytes(priv).sign(body)
    return json.dumps({"body": body.decode(), "sig": sig.hex(), "key_id": key_id}).encode()
