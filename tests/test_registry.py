"""Method registry and backend resolution."""

from __future__ import annotations

import pytest

from cuperiod.core.errors import BackendUnavailableError, UnknownMethodError
from cuperiod.methods.base import get_method, list_methods, method_names


def test_lookup_is_case_insensitive() -> None:
    assert get_method("gls") is get_method("GLS")


def test_unknown_method_raises() -> None:
    with pytest.raises(UnknownMethodError):
        get_method("not-a-method")


def test_registry_contains_phase1_methods() -> None:
    names = method_names()
    assert {"GLS", "BLS"} <= set(names)
    assert {m.name for m in list_methods()} == set(names)


def test_resolve_cpu_backend() -> None:
    assert get_method("GLS").resolve_backend("cpu") == "finufft"
    assert get_method("BLS").resolve_backend("cpu") == "astropy"


def test_resolve_gpu_without_device_raises() -> None:
    from cuperiod.core.backend import cuda_available

    if cuda_available():
        pytest.skip("a GPU is present")
    with pytest.raises(BackendUnavailableError):
        get_method("GLS").resolve_backend("gpu")


def test_resolve_unknown_backend_raises() -> None:
    with pytest.raises(BackendUnavailableError):
        get_method("GLS").resolve_backend("nonsense")
