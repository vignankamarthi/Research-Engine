"""The signed gate config: the immutable acceptance constants the loop may never
move. Validation enforces the design's hard invariants (open-unit alpha/power,
MIE strictly above MDE, a known claim-type taxonomy) before any gate reads it."""
from __future__ import annotations

from dataclasses import dataclass, asdict

from common.canonical import canonical_json_bytes

ALLOWED_CLAIM_TYPES = frozenset({
    "effect",
    "qualitative_phenomenon",
    "capability",
    "law_shape",
    "multi_benchmark_superiority",  # in the taxonomy, deferred for campaign one
})

_REQUIRED = (
    "version", "alpha", "power", "mde", "mie_floor", "claim_types",
    "gate_library_digest", "control_catalog_hash", "key_id",
)


class ConfigError(ValueError):
    """The gate config is missing a field, mistyped, or violates a hard invariant."""


@dataclass(frozen=True, slots=True)
class GateConfig:
    version: str
    alpha: float
    power: float
    mde: float
    mie_floor: float
    claim_types: tuple[str, ...]
    gate_library_digest: str
    control_catalog_hash: str
    key_id: str


def _number(data: dict, field: str) -> float:
    if field not in data:
        raise ConfigError(f"missing field: {field}")
    v = data[field]
    # bool is a subclass of int; an acceptance constant must be a real number.
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{field} must be a number, got {type(v).__name__}")
    return float(v)


def _nonempty_str(data: dict, field: str) -> str:
    if field not in data:
        raise ConfigError(f"missing field: {field}")
    v = data[field]
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{field} must be a non-empty string")
    return v


def validate_config(data: dict) -> GateConfig:
    if not isinstance(data, dict):
        raise ConfigError("config must be a mapping")
    for field in _REQUIRED:
        if field not in data:
            raise ConfigError(f"missing field: {field}")

    version = _nonempty_str(data, "version")
    alpha = _number(data, "alpha")
    power = _number(data, "power")
    mde = _number(data, "mde")
    mie_floor = _number(data, "mie_floor")

    if not (0.0 < alpha < 1.0):
        raise ConfigError(f"alpha must be in the open interval (0, 1), got {alpha}")
    if not (0.0 < power < 1.0):
        raise ConfigError(f"power must be in the open interval (0, 1), got {power}")
    if not (mde > 0.0):
        raise ConfigError(f"mde must be positive, got {mde}")
    if not (mie_floor > mde):
        raise ConfigError(
            f"mie_floor (interest) must be strictly above mde (detectability): "
            f"{mie_floor} !> {mde}"
        )

    raw_types = data["claim_types"]
    if not isinstance(raw_types, (list, tuple)) or len(raw_types) == 0:
        raise ConfigError("claim_types must be a non-empty list")
    for t in raw_types:
        if t not in ALLOWED_CLAIM_TYPES:
            raise ConfigError(f"unknown claim_type: {t!r}")

    gate_library_digest = _nonempty_str(data, "gate_library_digest")
    control_catalog_hash = _nonempty_str(data, "control_catalog_hash")
    key_id = _nonempty_str(data, "key_id")

    return GateConfig(
        version=version,
        alpha=alpha,
        power=power,
        mde=mde,
        mie_floor=mie_floor,
        claim_types=tuple(raw_types),
        gate_library_digest=gate_library_digest,
        control_catalog_hash=control_catalog_hash,
        key_id=key_id,
    )


def canonical_bytes(cfg: GateConfig) -> bytes:
    """Deterministic serialization for signing and verification. Key order and the
    caller's dict order must never change the signed bytes."""
    return canonical_json_bytes(asdict(cfg))
