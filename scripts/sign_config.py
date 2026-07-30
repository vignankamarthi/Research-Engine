"""Sign the gate config, pinning the three catalogs. Run on Vignan's Mac after keygen.

    uv run python scripts/sign_config.py

Reads the human-owned acceptance constants from config.template.json, fills in the code-derived
digests (gate library, control catalog, the three signed catalogs), signs with the private key from
~/.research-engine/signing_key, and writes signed_config.json. The private key is read here and
never stored. Editing any catalog afterward breaks the signature, which is the point."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from gateconfig.assemble import build_signed_config

PRIVATE_PATH = Path.home() / ".research-engine" / "signing_key"
TEMPLATE_PATH = Path("config.template.json")
CATALOGS_DIR = Path("catalogs")
OUT_PATH = Path("signed_config.json")


def main() -> None:
    if not PRIVATE_PATH.exists():
        print(f"no private key at {PRIVATE_PATH}. Run: uv run python scripts/keygen.py")
        sys.exit(1)
    private_key_bytes = PRIVATE_PATH.read_bytes()
    template = json.loads(TEMPLATE_PATH.read_text())

    signed = build_signed_config(
        template=template, catalogs_dir=CATALOGS_DIR, private_key_bytes=private_key_bytes)

    OUT_PATH.write_bytes(signed)
    print(f"signed config -> {OUT_PATH} (pins the three catalogs; re-run after any catalog edit)")


if __name__ == "__main__":
    main()
