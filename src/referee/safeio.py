"""Safe deserialization for the trusted referee. A checksum proves provenance, not
parse-safety, and a malicious pickle would execute code inside the referee, so
pickle is refused by extension, by magic byte, and by the numpy allow_pickle path.
Only JSON and non-object npy are permitted here (model weights use safetensors on
the cluster)."""
from __future__ import annotations

import json
import os

import numpy as np

_FORBIDDEN_EXT = {".pkl", ".pickle", ".pt", ".pth", ".bin"}
_ALLOWED_FORMATS = {"json", "npy"}


class SafeFormatError(Exception):
    """A refusal to deserialize an unsafe (pickle-family) or unsupported artifact."""


def safe_load(path: str, fmt: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in _FORBIDDEN_EXT:
        raise SafeFormatError(f"refusing pickle-family file extension: {ext}")
    with open(path, "rb") as f:
        if f.read(1) == b"\x80":
            raise SafeFormatError("refusing file with pickle magic byte 0x80")

    if fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if fmt == "npy":
        try:
            return np.load(path, allow_pickle=False)
        except ValueError as e:
            raise SafeFormatError(f"npy requires pickle (object array), refused: {e}") from e
    raise SafeFormatError(f"unsupported or forbidden format: {fmt!r}")
