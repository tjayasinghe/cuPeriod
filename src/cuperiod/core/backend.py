"""Compute-backend discovery and CUDA setup.

cuPeriod methods each expose one or more named backends (``finufft``/``cufinufft``
for GLS; ``numpy``/``astropy``/``cupy`` for BLS; ``numba``/``cupy`` for the rest).
This module answers "what is importable here?" and handles the Windows CUDA-12 DLL
quirk so the GPU wheels load, and provides the numpy/cupy array-module dispatch the
device-agnostic assembly code uses.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from types import ModuleType
from typing import Any

from cuperiod.core.errors import BackendUnavailableError

_CUDA_DLL_READY = False

#: The default per-block dynamic shared-memory cap (bytes); larger needs an opt-in.
_DEFAULT_SHARED_MEM = 48 * 1024


def ensure_cuda_dll_path() -> None:
    """Put the ``nvidia-*-cu12`` wheel binary dirs on the Windows DLL search path.

    The cufinufft Windows wheel bundles ``cufinufft.dll`` but not its CUDA-12 runtime
    dependencies (cudart/cufft/nvrtc), and its loader silently swallows the resulting
    dependency-load error. Adding the nvidia wheel ``bin``/``lib`` dirs here (and
    importing cupy, which the GPU backends do) makes the dependencies resolvable.
    No-op off Windows and after the first call.
    """
    global _CUDA_DLL_READY
    if _CUDA_DLL_READY or sys.platform != "win32":
        _CUDA_DLL_READY = True
        return
    spec = importlib.util.find_spec("nvidia")
    if spec and spec.submodule_search_locations and hasattr(os, "add_dll_directory"):
        root = pathlib.Path(next(iter(spec.submodule_search_locations)))
        for sub in sorted(root.iterdir()):
            for name in ("bin", "lib"):
                d = sub / name
                if d.is_dir():
                    os.add_dll_directory(str(d))
    _CUDA_DLL_READY = True


def has_module(name: str) -> bool:
    """Whether ``name`` is importable, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def cuda_available() -> bool:
    """Whether cupy is installed *and* a CUDA device is actually present.

    Importing cupy succeeds without a GPU, so this also queries the device count;
    any failure (no driver, no device) reports ``False``.
    """
    if not has_module("cupy"):
        return False
    ensure_cuda_dll_path()
    try:
        import cupy

        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


def torch_available() -> bool:
    """Whether PyTorch is importable.

    A CPU device always exists, so once torch is installed the portable ``torch``
    backend is never *unavailable* — this is what lets users with no GPU (or an
    unsupported GPU) still run the accelerated code paths. Cheap: no import.
    """
    return has_module("torch")


def torch_devices() -> set[str]:
    """Torch device kinds usable here: ``{"cpu"}`` plus any of ``cuda``/``mps``/``xpu``.

    Imports torch to query the runtimes (so call only when a torch path is actually
    being taken). ROCm builds report AMD GPUs as ``"cuda"``, so AMD needs no separate
    name. Returns an empty set if torch is absent or fails to import.
    """
    if not has_module("torch"):
        return set()
    try:
        import torch
    except Exception:
        return set()
    out = {"cpu"}
    try:
        if torch.cuda.is_available():
            out.add("cuda")
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            out.add("mps")
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            out.add("xpu")
    except Exception:
        pass
    return out


def torch_gpu_available() -> bool:
    """Whether torch sees any non-CPU device (CUDA/ROCm, MPS, or XPU)."""
    return bool(torch_devices() - {"cpu"})


def best_torch_device() -> str:
    """Preferred torch device, in order ``cuda`` → ``xpu`` → ``mps`` → ``cpu``."""
    devices = torch_devices()
    for kind in ("cuda", "xpu", "mps"):
        if kind in devices:
            return kind
    return "cpu"


def available_backends() -> set[str]:
    """The set of backend names importable in this environment.

    Returns a union across methods: always includes ``numpy``; adds ``finufft``,
    ``astropy``, ``numba``, ``torch`` when importable, and the GPU names
    ``cufinufft``/``cupy`` only when a CUDA device is present.

    ``torch`` is added on a cheap import-spec check (no torch import here): a CPU device
    always exists, so an installed torch is always a usable backend. Which torch
    *devices* are present is answered separately by :func:`torch_devices`.
    """
    out: set[str] = {"numpy"}
    for name in ("finufft", "astropy", "numba"):
        if has_module(name):
            out.add(name)
    if torch_available():
        out.add("torch")
    if cuda_available():
        out.add("cupy")
        if has_module("cufinufft"):
            out.add("cufinufft")
    return out


def ensure_shared_memory(
    kernel: Any, dynamic_bytes: int, *, method: str, hint: str
) -> None:
    """Permit a cupy ``RawKernel`` to use ``dynamic_bytes`` of dynamic shared memory.

    The per-block default cap is 48 KB; a larger request needs an explicit opt-in and is
    still bounded by the device's ``MaxSharedMemoryPerBlockOptin``. When the request
    exceeds what the device can provide, raise a clear :class:`BackendUnavailableError`
    (pointing at the setting to reduce) instead of letting the kernel launch fail with a
    raw ``CUDADriverError``.

    Parameters
    ----------
    kernel : cupy.RawKernel
        The kernel about to be launched.
    dynamic_bytes : int
        The ``shared_mem=`` size the launch will request.
    method : str
        Method name, for the error message.
    hint : str
        The setting(s) the user should reduce, for the error message.
    """
    import cupy

    optin = int(
        cupy.cuda.Device().attributes.get(
            "MaxSharedMemoryPerBlockOptin", _DEFAULT_SHARED_MEM
        )
    )
    try:
        static = int(kernel.attributes.get("shared_size_bytes", 0))
    except Exception:
        static = 0
    needed = int(dynamic_bytes) + static
    if needed > optin:
        raise BackendUnavailableError(
            f"{method}: needs ~{needed // 1024} KB of GPU shared memory per block but "
            f"the device allows at most {optin // 1024} KB; reduce {hint}, or use "
            "backend='cpu'."
        )
    # The opt-in is needed whenever the *total* (dynamic + static) exceeds the
    # default cap — a launch with 48 KB dynamic still fails if the kernel also has
    # static shared arrays.
    if needed > _DEFAULT_SHARED_MEM:
        kernel.max_dynamic_shared_size_bytes = int(dynamic_bytes)


def array_module(a: object) -> ModuleType:
    """Return cupy for a device array, else numpy (for device-agnostic assembly)."""
    try:
        import cupy

        module: ModuleType = cupy.get_array_module(a)
        return module
    except Exception:
        import numpy

        return numpy


__all__ = [
    "array_module",
    "available_backends",
    "best_torch_device",
    "cuda_available",
    "ensure_cuda_dll_path",
    "ensure_shared_memory",
    "has_module",
    "torch_available",
    "torch_devices",
    "torch_gpu_available",
]
