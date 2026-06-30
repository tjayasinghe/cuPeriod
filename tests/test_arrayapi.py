"""Array-API shim, torch device detection, and vendor-aware backend resolution.

These are the new correctness-critical primitives behind the portable (torch) compute
paths. The pure-shim and numpy tests run everywhere; torch-specific tests skip without
the ``[torch]`` extra. No GPU is required (the torch path runs on the CPU device).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_torch
from cuperiod.core._arrayapi import (
    array_namespace,
    device_of,
    is_torch_array,
    resolve_precision,
    resolve_torch_device,
    scatter_add,
    to_host,
)
from cuperiod.core.backend import (
    available_backends,
    best_torch_device,
    torch_available,
    torch_devices,
)
from cuperiod.core.errors import BackendUnavailableError
from cuperiod.methods.bls import BLSMethod
from cuperiod.methods.gls import GLSMethod

# --- shim primitives ---------------------------------------------------------


def test_scatter_add_numpy_accumulates_repeats() -> None:
    target = np.zeros(4)
    scatter_add(target, np.array([0, 0, 1, 3]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert target.tolist() == [3.0, 3.0, 0.0, 4.0]


def test_to_host_from_numpy_is_float64() -> None:
    out = to_host(np.arange(3, dtype=np.float32))
    assert out.dtype == np.float64
    assert out.tolist() == [0.0, 1.0, 2.0]


def test_device_of_numpy_is_cpu() -> None:
    assert device_of(np.zeros(3)) == "cpu"
    assert not is_torch_array(np.zeros(3))


def test_resolve_precision_defaults() -> None:
    assert resolve_precision("auto", "cpu") == "float64"
    assert resolve_precision("auto", "cuda") == "float64"
    assert resolve_precision("auto", "xpu") == "float64"
    # MPS cannot do float64, so auto downgrades there (and only there).
    assert resolve_precision("auto", "mps") == "float32"
    assert resolve_precision("float32", "cpu") == "float32"
    assert resolve_precision("float64", "cuda") == "float64"


def test_resolve_precision_float64_on_mps_raises() -> None:
    with pytest.raises(BackendUnavailableError):
        resolve_precision("float64", "mps")


# --- torch detection ---------------------------------------------------------


def test_torch_available_returns_bool() -> None:
    assert isinstance(torch_available(), bool)


@requires_torch
def test_torch_registers_as_backend() -> None:
    assert "torch" in available_backends()


@requires_torch
def test_torch_devices_include_cpu() -> None:
    devices = torch_devices()
    assert "cpu" in devices
    assert best_torch_device() in devices


@requires_torch
def test_array_namespace_and_scatter_add_torch() -> None:
    import torch

    xp = array_namespace(torch.zeros(3))
    assert "torch" in xp.__name__

    target = torch.zeros(4, dtype=torch.float64)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    scatter_add(target, torch.tensor([0, 0, 1, 3]), values)
    assert target.tolist() == [3.0, 3.0, 0.0, 4.0]


@requires_torch
def test_device_of_and_to_host_torch() -> None:
    import torch

    t = torch.arange(3, dtype=torch.float32)
    assert device_of(t) == "cpu"
    assert is_torch_array(t)
    out = to_host(t)
    assert out.dtype == np.float64
    assert out.tolist() == [0.0, 1.0, 2.0]


# --- backend resolution ------------------------------------------------------


@requires_torch
def test_resolve_torch_device_explicit_and_default() -> None:
    assert resolve_torch_device("torch:cpu", "auto") == "cpu"
    assert resolve_torch_device("torch", "cpu") == "cpu"


@requires_torch
def test_gls_resolves_torch_backends() -> None:
    method = GLSMethod()
    assert method.resolve_backend("torch") == "torch"
    assert method.resolve_backend("torch:cpu") == "torch:cpu"
    assert method.is_gpu_backend("torch:cuda")
    assert not method.is_gpu_backend("torch:cpu")


@requires_torch
def test_bls_resolves_torch_backend() -> None:
    assert BLSMethod().resolve_backend("torch:cpu") == "torch:cpu"


@requires_torch
def test_resolve_unknown_torch_device_raises() -> None:
    with pytest.raises(BackendUnavailableError):
        GLSMethod().resolve_backend("torch:bogus")
