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


def available_backends() -> set[str]:
    """The set of backend names importable in this environment.

    Returns a union across methods: always includes ``numpy``; adds ``finufft``,
    ``astropy``, ``numba`` when importable, and the GPU names ``cufinufft``/``cupy``
    only when a CUDA device is present.
    """
    out: set[str] = {"numpy"}
    for name in ("finufft", "astropy", "numba"):
        if has_module(name):
            out.add(name)
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
    if int(dynamic_bytes) > _DEFAULT_SHARED_MEM:
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
    "cuda_available",
    "ensure_cuda_dll_path",
    "ensure_shared_memory",
    "has_module",
]
