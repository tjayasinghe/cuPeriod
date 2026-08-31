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


def _fake_env(
    monkeypatch: pytest.MonkeyPatch, *, backends: set[str], cuda: bool, torch_gpu: bool
) -> None:
    """Pretend this machine has exactly ``backends`` and these device kinds."""
    import cuperiod.methods.base as base

    monkeypatch.setattr(base, "available_backends", lambda: set(backends))
    monkeypatch.setattr(base, "cuda_available", lambda: cuda)
    monkeypatch.setattr(base, "torch_gpu_available", lambda: torch_gpu)


def test_auto_skips_cuda_path_when_its_package_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # cupy sees a GPU, but GLS's fast path (cufinufft) is not installed: "auto"
    # must fall through to the CPU rather than pick a backend that will raise
    # ImportError once it actually runs.
    _fake_env(
        monkeypatch,
        backends={"numpy", "finufft", "astropy", "cupy"},
        cuda=True,
        torch_gpu=False,
    )
    assert get_method("GLS").resolve_backend("auto") == "finufft"


def test_auto_prefers_torch_gpu_over_cpu_when_cuda_path_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_env(
        monkeypatch,
        backends={"numpy", "finufft", "astropy", "cupy", "torch"},
        cuda=True,
        torch_gpu=True,
    )
    assert get_method("GLS").resolve_backend("auto") == "torch"


def test_auto_takes_cuda_path_when_fully_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_env(
        monkeypatch,
        backends={"numpy", "finufft", "astropy", "cupy", "cufinufft"},
        cuda=True,
        torch_gpu=False,
    )
    assert get_method("GLS").resolve_backend("auto") == "cufinufft"


def test_gpu_request_names_the_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_env(
        monkeypatch,
        backends={"numpy", "finufft", "astropy", "cupy"},
        cuda=True,
        torch_gpu=False,
    )
    with pytest.raises(BackendUnavailableError, match="cufinufft"):
        get_method("GLS").resolve_backend("gpu")
