"""gateconfig -- the offline-signed gate-config trust root.

The gate config holds every acceptance constant the loop is forbidden to move
(alpha, power, MDE, the MIE floor, the claim-type taxonomy, the frozen gate-library
digest, the control-catalog hash). It is signed offline on Vignan's Mac and
re-verified inside the trusted process at every use.
"""
from .schema import (
    ALLOWED_CLAIM_TYPES,
    ConfigError,
    GateConfig,
    canonical_bytes,
    validate_config,
)
from .signing import SignatureError, sign_config, verify_config

__all__ = [
    "GateConfig",
    "ConfigError",
    "SignatureError",
    "ALLOWED_CLAIM_TYPES",
    "validate_config",
    "canonical_bytes",
    "sign_config",
    "verify_config",
]
