"""Trust-root schema validation. The gate config holds every acceptance constant
the loop is forbidden to move, so a malformed or invariant-violating config must
be rejected before any gate can read it."""
import pytest

from gateconfig import GateConfig, validate_config, canonical_bytes, ConfigError


def valid_dict(**overrides):
    d = {
        "version": "1",
        "alpha": 0.05,
        "power": 0.8,
        "mde": 0.01,
        "mie_floor": 0.03,
        "claim_types": ["effect", "qualitative_phenomenon", "capability", "law_shape"],
        "gate_library_digest": "sha256:" + "0" * 64,
        "control_catalog_hash": "sha256:" + "1" * 64,
        "key_id": "vignan-mac-2026",
    }
    d.update(overrides)
    return d


def test_valid_config_validates():
    cfg = validate_config(valid_dict())
    assert isinstance(cfg, GateConfig)
    assert cfg.alpha == 0.05
    assert cfg.claim_types == ("effect", "qualitative_phenomenon", "capability", "law_shape")


def test_config_is_frozen():
    cfg = validate_config(valid_dict())
    with pytest.raises((AttributeError, TypeError)):
        cfg.alpha = 0.1  # acceptance constants are immutable at use


@pytest.mark.parametrize("field", [
    "version", "alpha", "power", "mde", "mie_floor", "claim_types",
    "gate_library_digest", "control_catalog_hash", "key_id",
])
def test_missing_field_rejected(field):
    d = valid_dict()
    del d[field]
    with pytest.raises(ConfigError):
        validate_config(d)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1])
def test_alpha_out_of_open_unit_interval_rejected(alpha):
    with pytest.raises(ConfigError):
        validate_config(valid_dict(alpha=alpha))


@pytest.mark.parametrize("power", [0.0, 1.0, 1.2, -0.5])
def test_power_out_of_open_unit_interval_rejected(power):
    with pytest.raises(ConfigError):
        validate_config(valid_dict(power=power))


def test_mde_must_be_positive():
    with pytest.raises(ConfigError):
        validate_config(valid_dict(mde=0.0))


def test_mie_floor_must_be_strictly_above_mde():
    # A real design invariant: MIE (interest) is strictly above MDE (detectability).
    with pytest.raises(ConfigError):
        validate_config(valid_dict(mde=0.03, mie_floor=0.03))
    with pytest.raises(ConfigError):
        validate_config(valid_dict(mde=0.03, mie_floor=0.01))


def test_empty_claim_types_rejected():
    with pytest.raises(ConfigError):
        validate_config(valid_dict(claim_types=[]))


def test_unknown_claim_type_rejected():
    with pytest.raises(ConfigError):
        validate_config(valid_dict(claim_types=["effect", "leaderboard_bump"]))


def test_multi_benchmark_is_a_known_type_but_deferred_ok_to_declare():
    # The type exists in the taxonomy even though it is deferred for campaign one.
    cfg = validate_config(valid_dict(claim_types=["effect", "multi_benchmark_superiority"]))
    assert "multi_benchmark_superiority" in cfg.claim_types


def test_canonical_bytes_is_deterministic_and_key_order_independent():
    a = validate_config(valid_dict())
    reordered = dict(reversed(list(valid_dict().items())))
    b = validate_config(reordered)
    assert canonical_bytes(a) == canonical_bytes(b)
    assert isinstance(canonical_bytes(a), bytes)
