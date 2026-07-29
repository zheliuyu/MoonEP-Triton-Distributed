"""Verify public API signatures match MoonEP."""

import inspect

import pytest

pytest.importorskip("moonep")
import moonep.api as orig_api
import moonep_td.api as td_api


def _params(cls, name):
    return inspect.signature(getattr(cls, name)).parameters.keys()


def test_buffer_init_signature():
    assert _params(orig_api.Buffer, "__init__") == _params(td_api.Buffer, "__init__")


def test_dispatch_signature():
    assert _params(orig_api.Buffer, "dispatch") == _params(td_api.Buffer, "dispatch")


def test_combine_signature():
    assert _params(orig_api.Buffer, "combine") == _params(td_api.Buffer, "combine")


def test_prefetch_signature():
    assert _params(orig_api.Buffer, "prefetch_weight") == _params(td_api.Buffer, "prefetch_weight")


def test_reduce_grad_signature():
    assert _params(orig_api.Buffer, "reduce_grad") == _params(td_api.Buffer, "reduce_grad")
