"""The signing-assembly path. It merges the human's acceptance constants with the code-derived
digests (gate library, control catalog, the three signed catalogs), then signs with Vignan's key.
These tests prove the referee's verify_config accepts the result and that signing transitively PINS
the catalogs, so a post-signing catalog edit is detectable. The private key is generated here only
to drive the round-trip; in production it stays on Vignan's Mac."""
import json

import pytest

from gateconfig.assemble import build_signed_config, generate_keypair
from gateconfig.signing import SignatureError, verify_config
from referee.catalog import CatalogError, catalog_digest, verify_catalog


def _template():
    # mde deliberately below the lenient per-task MIE (0.01) so mie_floor stays strictly above mde.
    return {"version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.005, "mie_floor": 0.03,
            "claim_types": ["effect"], "key_id": "vignan-2026"}


def _write_catalogs(d):
    cons = {"effect": "downstream accuracy rises by >= MIE on the held-out task"}
    inc = {"ssv2_recognition_top1": {"value": 0.773}}
    mie = {"ssv2_recognition_top1": {"mie_value": 0.01}}
    (d / "consequence_templates.json").write_text(json.dumps(cons))
    (d / "incumbent_catalog.json").write_text(json.dumps(inc))
    (d / "mie_distribution.json").write_text(json.dumps(mie))
    return cons, inc, mie


def test_generate_keypair_yields_a_raw_ed25519_pair():
    priv, pub = generate_keypair()
    assert len(priv) == 32 and len(pub) == 32 and priv != pub


def test_build_signed_config_verifies_and_pins_the_catalogs(tmp_path):
    priv, pub = generate_keypair()
    cons, inc, mie = _write_catalogs(tmp_path)
    signed = build_signed_config(
        template=_template(), catalogs_dir=tmp_path, private_key_bytes=priv)
    cfg = verify_config(signed, pub)  # the referee's own at-use verifier accepts it
    assert cfg.consequence_catalog_digest == catalog_digest(cons)
    assert cfg.incumbent_catalog_digest == catalog_digest(inc)
    assert cfg.mie_distribution_digest == catalog_digest(mie)


def test_signing_pins_a_catalog_against_later_edits(tmp_path):
    priv, pub = generate_keypair()
    _, inc, _ = _write_catalogs(tmp_path)
    signed = build_signed_config(
        template=_template(), catalogs_dir=tmp_path, private_key_bytes=priv)
    cfg = verify_config(signed, pub)
    # someone edits the incumbent catalog AFTER signing to an easier bar -> the pinned digest breaks
    tampered = {"ssv2_recognition_top1": {"value": 0.60}}
    with pytest.raises(CatalogError):
        verify_catalog(tampered, cfg.incumbent_catalog_digest)


def test_a_wrong_public_key_is_rejected(tmp_path):
    priv, _ = generate_keypair()
    _, other_pub = generate_keypair()
    _write_catalogs(tmp_path)
    signed = build_signed_config(
        template=_template(), catalogs_dir=tmp_path, private_key_bytes=priv)
    with pytest.raises(SignatureError):
        verify_config(signed, other_pub)


def test_comment_keys_in_the_template_are_ignored(tmp_path):
    priv, pub = generate_keypair()
    _write_catalogs(tmp_path)
    template = {**_template(), "_comment": "human notes that must not affect the signed bytes"}
    signed = build_signed_config(template=template, catalogs_dir=tmp_path, private_key_bytes=priv)
    cfg = verify_config(signed, pub)
    assert cfg.key_id == "vignan-2026"
