"""Unit tests for tests/jax_batch_guard.py."""
import importlib.abc
import importlib.machinery
import sys
import types

import pytest

from jax_batch_guard import require_batch_jax


class _FailingJaxFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, exc):
        self._exc = exc

    def find_spec(self, fullname, path, target=None):
        if fullname == "jax":
            return importlib.machinery.ModuleSpec("jax", self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise self._exc


def test_require_batch_jax_skips_when_jax_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", None)
    with pytest.raises(pytest.skip.Exception) as exc_info:
        require_batch_jax()
    assert not exc_info.value.allow_module_level


def test_require_batch_jax_forwards_allow_module_level(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", None)
    with pytest.raises(pytest.skip.Exception) as exc_info:
        require_batch_jax(allow_module_level=True)
    assert exc_info.value.allow_module_level


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("broken jax install"),
        ModuleNotFoundError("No module named 'jaxlib'", name="jaxlib"),
    ],
)
def test_require_batch_jax_propagates_broken_import(monkeypatch, exc):
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_FailingJaxFinder(exc), *sys.meta_path])
    with pytest.raises(type(exc)) as exc_info:
        require_batch_jax(allow_module_level=True)
    assert exc_info.value is exc


@pytest.mark.parametrize(
    "attrs",
    [
        {"__version__": "0.4.34"},
        {"__version__": "0.4.34", "enable_x64": object()},
        {"__version__": "0.4.34", "shard_map": object()},
    ],
)
def test_require_batch_jax_skips_when_jax_lacks_capabilities(monkeypatch, attrs):
    monkeypatch.setitem(sys.modules, "jax", types.SimpleNamespace(**attrs))
    with pytest.raises(pytest.skip.Exception, match="enable_x64 and shard_map") as exc_info:
        require_batch_jax(allow_module_level=True)
    assert exc_info.value.allow_module_level


def test_require_batch_jax_returns_capable_module(monkeypatch):
    fake_jax = types.SimpleNamespace(
        __version__="0.12.0",
        enable_x64=object(),
        shard_map=object(),
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    assert require_batch_jax() is fake_jax
