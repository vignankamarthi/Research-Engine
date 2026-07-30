"""Generate Vignan's Ed25519 signing keypair. Run ONCE, on Vignan's Mac.

    uv run python scripts/keygen.py

The PRIVATE key is written outside the repo (~/.research-engine/signing_key, chmod 600) so it can
never be committed by accident, and it never leaves the Mac. The PUBLIC key is written into the repo
(keys/signing_pub.key); it is what the referee verifies signatures against, so it is safe to commit.
This refuses to overwrite an existing private key, losing it would orphan every prior signature."""
import sys
from pathlib import Path

sys.path.insert(0, "src")

from gateconfig.assemble import generate_keypair

PRIVATE_PATH = Path.home() / ".research-engine" / "signing_key"
PUBLIC_PATH = Path("keys") / "signing_pub.key"


def main() -> None:
    if PRIVATE_PATH.exists():
        print(f"refusing to overwrite existing private key at {PRIVATE_PATH}")
        print("delete it by hand first if you truly mean to rotate the key.")
        sys.exit(1)

    private_raw, public_raw = generate_keypair()

    PRIVATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_PATH.write_bytes(private_raw)
    PRIVATE_PATH.chmod(0o600)

    PUBLIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_PATH.write_bytes(public_raw)

    print(f"private key -> {PRIVATE_PATH} (chmod 600, stays on this Mac, never commit)")
    print(f"public key  -> {PUBLIC_PATH} (commit this; the referee verifies against it)")
    print("next: uv run python scripts/sign_config.py")


if __name__ == "__main__":
    main()
