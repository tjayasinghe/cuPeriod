"""Method registry and backend resolution."""

from __future__ import annotations

import pytest

from cuperiod.core.errors import BackendUnavailableError, UnknownMethodError
from cuperiod.methods.base import get_method, list_methods, method_names


def test_lookup_is_case_insensitive() -> None:
    assert get_method("gls") is get_method("GLS")


def test_lookup_ignores_separators() -> None:
    # The documented "String-Length" spelling (and underscores) must resolve.
    sl = get_method("STRINGLENGTH")
    assert get_method("String-Length") is sl
    assert get_method("StringLength") is sl
    assert get_method("string_length") is sl


def test_multiresult_getitem_normalizes() -> None:
    import numpy as np

    import cuperiod as cup

    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 80, 300))
    y = 12 + 0.3 * np.sin(2 * np.pi * t / 2.0) + 0.02 * rng.standard_normal(300)
    res = cup.periodogram((t, y, np.full(300, 0.02)), ["GLS", "String-Length"])
    assert res["String-Length"].method == "STRINGLENGTH"
    assert res["stringlength"] is res["String-Length"]


def test_unknown_method_raises() -> None:
    with pytest.raises(UnknownMethodError):
        get_method("not-a-method")


def test_registry_contains_phase1_methods() -> None:
    names = method_names()
    assert {"GLS", "BLS"} <= set(names)
    assert {m.name for m in list_methods()} == set(names)


def test_resolve_cpu_backend() -> None:
    from cuperiod.core.backend import available_backends

    assert get_method("GLS").resolve_backend("cpu") == "finufft"
    # BLS prefers the multicore numba box search on the CPU when installed,
    # falling back to astropy's compiled BoxLeastSquares otherwise.
    expected = "numba" if "numba" in available_backends() else "astropy"
    assert get_method("BLS").resolve_backend("cpu") == expected


def test_resolve_gpu_without_device_raises() -> None:
    from cuperiod.core.backend import cuda_available

    if cuda_available():
        pytest.skip("a GPU is present")
    with pytest.raises(BackendUnavailableError):
        get_method("GLS").resolve_backend("gpu")


def test_resolve_unknown_backend_raises() -> None:
    with pytest.raises(BackendUnavailableError):
        get_method("GLS").resolve_backend("nonsense")
