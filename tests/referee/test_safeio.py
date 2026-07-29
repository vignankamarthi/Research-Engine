"""The safe-deserialization boundary. The referee parses checksummed artifacts with
SAFE formats only. A checksum proves provenance, not parse-safety, and a malicious
pickle would execute code inside the trusted process, so pickle is refused by
extension, by magic byte, and by the numpy allow_pickle path."""
import json

import numpy as np
import pytest

from referee import SafeFormatError, safe_load


def test_load_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"alpha": 0.05}))
    assert safe_load(str(p), "json") == {"alpha": 0.05}


def test_load_npy_array(tmp_path):
    p = tmp_path / "scores.npy"
    np.save(p, np.arange(5))
    assert np.array_equal(safe_load(str(p), "npy"), np.arange(5))


def test_object_npy_requiring_pickle_is_refused(tmp_path):
    p = tmp_path / "obj.npy"
    np.save(p, np.array([{"x": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(SafeFormatError):
        safe_load(str(p), "npy")


def test_pickle_magic_byte_refused(tmp_path):
    p = tmp_path / "sneaky.npy"
    p.write_bytes(b"\x80\x04payload")
    with pytest.raises(SafeFormatError):
        safe_load(str(p), "npy")


def test_pickle_extension_refused(tmp_path):
    p = tmp_path / "weights.pt"
    p.write_bytes(b"\x80\x04torch-pickle")
    with pytest.raises(SafeFormatError):
        safe_load(str(p), "npy")


def test_unknown_format_refused(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"whatever")
    with pytest.raises(SafeFormatError):
        safe_load(str(p), "pickle")
