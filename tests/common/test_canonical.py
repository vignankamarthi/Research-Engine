"""The shared canonical serialization + hashing primitive. Every signed digest in the system
routes through here, so a copy can never drift (a changed separator or ensure_ascii would
silently break verification). Behavior is pinned exactly to the encoding the callers already
used: sorted keys, compact separators, UTF-8."""
import hashlib

from common.canonical import canonical_digest, canonical_json_bytes, sha256_hex


def test_canonical_json_bytes_are_sorted_and_compact():
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_canonical_digest_is_key_order_independent():
    assert canonical_digest({"a": 1, "b": 2}) == canonical_digest({"b": 2, "a": 1})


def test_canonical_digest_changes_with_content():
    assert canonical_digest({"a": 1}) != canonical_digest({"a": 2})


def test_canonical_digest_matches_the_legacy_encoding():
    obj = {"x": [1, 2], "y": "z"}
    expected = hashlib.sha256(b'{"x":[1,2],"y":"z"}').hexdigest()
    assert canonical_digest(obj) == expected


def test_sha256_hex_of_raw_bytes():
    assert sha256_hex(b"abc") == hashlib.sha256(b"abc").hexdigest()
