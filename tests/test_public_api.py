"""Public-API parity guard. Each package's __all__ is hand-maintained; this turns any future
drift (a symbol listed in __all__ but no longer imported, or a missing __all__) into a test
failure instead of a silently-broken public API."""
import importlib

import pytest

PACKAGES = ["common", "gateconfig", "gatelib", "backend", "referee", "engine"]


@pytest.mark.parametrize("pkg_name", PACKAGES)
def test_package_declares_all(pkg_name):
    pkg = importlib.import_module(pkg_name)
    assert isinstance(getattr(pkg, "__all__", None), list), f"{pkg_name} must declare __all__"


@pytest.mark.parametrize("pkg_name", PACKAGES)
def test_every_exported_name_is_bound(pkg_name):
    pkg = importlib.import_module(pkg_name)
    missing = [name for name in pkg.__all__ if not hasattr(pkg, name)]
    assert not missing, f"{pkg_name}.__all__ lists names it does not bind: {missing}"
