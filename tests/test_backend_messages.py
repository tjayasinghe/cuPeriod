"""A named backend that cannot run says which extra installs it.

The environment is faked (the availability probes are patched on the methods
module), so these run the same on a machine with or without numba, cupy or a GPU.
"""

from __future__ import annotations

import importlib

import pytest

from cuperiod.core.errors import BackendUnavailableError
from cuperiod.methods.base import get_method

BASE = importlib.import_module("cuperiod.methods.base")


def test_missing_numba_names_the_fast_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BASE, "cpu_backends", lambda: {"numpy", "finufft", "astropy"})
    with pytest.raises(BackendUnavailableError, match=r"cuperiod\[fast\]"):
        get_method("PDM").resolve_backend("numba")


@pytest.mark.parametrize(("method", "backend"), [("PDM", "cupy"), ("GLS", "cufinufft")])
def test_gpu_backend_without_cuda_names_the_gpu_extra(
    method: str, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(BASE, "cuda_available", lambda: False)
    with pytest.raises(BackendUnavailableError, match=r"cuperiod\[gpu\]") as info:
        get_method(method).resolve_backend(backend)
    assert "CUDA device" in str(info.value)


def test_missing_cufinufft_names_the_gpu_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A CUDA device and cupy, but no cufinufft: GLS's fast path is not installed.
    monkeypatch.setattr(BASE, "cuda_available", lambda: True)
    monkeypatch.setattr(BASE, "available_backends", lambda: {"numpy", "cupy"})
    hint = r"not installed \(pip install 'cuperiod\[gpu\]'\)"
    with pytest.raises(BackendUnavailableError, match=hint):
        get_method("GLS").resolve_backend("cufinufft")
