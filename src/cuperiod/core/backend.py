"""Compute-backend discovery and CUDA setup.

cuPeriod methods each expose one or more named backends (``finufft``/``cufinufft``
for GLS; ``numpy``/``astropy``/``cupy`` for BLS; ``numba``/``cupy`` for the rest).
This module answers "what is importable here?" and handles the Windows CUDA-12 DLL
quirk so the GPU wheels load, and provides the numpy/cupy array-module dispatch the
device-agnostic assembly code uses.

Discovery is memoised per process. ``importlib.util.find_spec`` walks ``sys.path`` for
a package that is installed but not yet imported (about half a millisecond each on
Windows), and the CUDA probe imports cupy (a few hundred milliseconds), so re-probing on
every resolution made backend selection cost more than a small periodogram. What is
installed cannot change under a running batch; a session that ``pip install``-s a
backend mid-way calls :func:`refresh_backends` to see it.
"""

from __future__ import annotations

import functools
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

#: Memoised result of the CUDA device probe (``None`` until the first probe).
_CUDA_PROBE: bool | None = None

#: Concrete backend names that only run on a CUDA device.
GPU_BACKEND_NAMES: frozenset[str] = frozenset({"cupy", "cufinufft"})

#: The generic selectors every method accepts besides its concrete backend names.
BACKEND_SELECTORS: tuple[str, ...] = ("auto", "cpu", "gpu")


def _nvidia_wheel_roots(locations: Any) -> list[pathlib.Path]:
    """The ``nvidia`` namespace portions that belong to installed ``nvidia-*`` wheels.

    ``nvidia`` is a namespace package, so its portions follow ``sys.path`` order and a
    stray folder named ``nvidia`` in the script or working directory comes first. A
    portion is trusted only when its parent directory also holds an installed nvidia
    wheel's ``nvidia_*.dist-info`` record — true for site-packages, user site and
    ``pip install --target`` trees, false for an unrelated project folder.
    """
    roots: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for location in locations or ():
        root = pathlib.Path(location)
        try:
            trusted = root.is_dir() and any(root.parent.glob("nvidia_*.dist-info"))
            resolved = root.resolve()
        except OSError:
            continue
        if trusted and resolved not in seen:
            seen.add(resolved)
            roots.append(root)
    return roots


def ensure_cuda_dll_path() -> None:
    """Put the ``nvidia-*-cu12`` wheel binary dirs on the Windows DLL search path.

    The cufinufft Windows wheel bundles ``cufinufft.dll`` but not its CUDA-12 runtime
    dependencies (cudart/cufft/nvrtc), and its loader silently swallows the resulting
    dependency-load error. Adding the nvidia wheel ``bin``/``lib`` dirs here (and
    importing cupy, which the GPU backends do) makes the dependencies resolvable.
    Only directories of installed nvidia wheels are added (see
    :func:`_nvidia_wheel_roots`), so a folder that happens to be called ``nvidia`` next
    to a script can neither hide the real runtime nor inject its own DLLs. No-op off
    Windows and after the first call.
    """
    global _CUDA_DLL_READY
    if _CUDA_DLL_READY or sys.platform != "win32":
        _CUDA_DLL_READY = True
        return
    spec = importlib.util.find_spec("nvidia")
    if spec and spec.submodule_search_locations and hasattr(os, "add_dll_directory"):
        for root in _nvidia_wheel_roots(spec.submodule_search_locations):
            try:
                subs = sorted(root.iterdir())
            except OSError:
                continue
            for sub in subs:
                for name in ("bin", "lib"):
                    d = sub / name
                    if d.is_dir():
                        os.add_dll_directory(str(d))
    _CUDA_DLL_READY = True


@functools.cache
def has_module(name: str) -> bool:
    """Whether ``name`` is importable, without importing it.

    Memoised per process (see the module docstring); :func:`refresh_backends` clears it.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _probe_cuda() -> bool:
    """Import cupy and ask the runtime for a device count (uncached)."""
    if not has_module("cupy"):
        return False
    ensure_cuda_dll_path()
    try:
        import cupy

        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


def cuda_available() -> bool:
    """Whether cupy is installed *and* a CUDA device is actually present.

    Importing cupy succeeds without a GPU, so this also queries the device count;
    any failure (no driver, no device) reports ``False``. The answer is memoised per
    process — the device count cannot change after the CUDA runtime has initialised —
    and :func:`refresh_backends` forgets it.
    """
    global _CUDA_PROBE
    if _CUDA_PROBE is None:
        _CUDA_PROBE = _probe_cuda()
    return _CUDA_PROBE


def refresh_backends() -> None:
    """Forget the memoised discovery so the next query re-probes the environment.

    Call after installing a backend package into a running session (a notebook
    ``pip install``), or in tests that fake the environment.
    """
    global _CUDA_PROBE
    has_module.cache_clear()
    _CUDA_PROBE = None


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


def cpu_backends() -> set[str]:
    """The CPU backend names importable here, without probing for a GPU.

    Always includes ``numpy``; adds ``finufft``, ``astropy``, ``numba`` and ``torch``
    when importable. Unlike :func:`available_backends` this never imports cupy or
    touches the CUDA driver, so a run that asked for the CPU stays off the GPU stack
    entirely. Returns a fresh set (callers may mutate it).
    """
    out: set[str] = {"numpy"}
    for name in ("finufft", "astropy", "numba"):
        if has_module(name):
            out.add(name)
    if torch_available():
        out.add("torch")
    return out


def available_backends() -> set[str]:
    """The set of backend names importable in this environment.

    Returns a union across methods: always includes ``numpy``; adds ``finufft``,
    ``astropy``, ``numba``, ``torch`` when importable, and the GPU names
    ``cufinufft``/``cupy`` only when a CUDA device is present.

    ``torch`` is added on a cheap import-spec check (no torch import here): a CPU device
    always exists, so an installed torch is always a usable backend. Which torch
    *devices* are present is answered separately by :func:`torch_devices`. The probes
    are memoised (see the module docstring) but the returned set is fresh each call.
    """
    out = cpu_backends()
    if cuda_available():
        out.add("cupy")
        if has_module("cufinufft"):
            out.add("cufinufft")
    return out


def normalize_backend(name: str) -> str:
    """Canonical spelling of a backend selector: stripped and lower-case.

    Method names already ignore case and separators; backend selectors follow suit for
    case and surrounding whitespace, so ``"CPU"``, ``" Auto "`` and ``"Torch:CUDA"``
    mean ``"cpu"``, ``"auto"`` and ``"torch:cuda"``.
    """
    return str(name).strip().lower()


def requested_backend(requested: str | None, settings: object | None = None) -> str:
    """The backend selector a run should resolve: the explicit request, else settings.

    Every settings model carries a ``backend`` field ("default backend for this
    method", also settable as ``CUPERIOD_<METHOD>_BACKEND``). An explicit, non-``auto``
    ``backend=`` argument wins; ``None`` or ``"auto"`` defers to ``settings.backend``,
    which itself defaults to ``"auto"``. A consequence: an explicit ``backend="auto"``
    cannot override a pinned ``settings.backend`` — pass ``"cpu"``/``"gpu"`` or a
    concrete name for that.

    Parameters
    ----------
    requested : str or None
        The caller's ``backend=`` argument.
    settings : settings model, optional
        The method's settings; only its ``backend`` attribute is read (if any).

    Returns
    -------
    str
        A normalised selector for :meth:`PeriodogramMethod.resolve_backend
        <cuperiod.methods.base.PeriodogramMethod.resolve_backend>`.
    """
    if requested is not None:
        explicit = normalize_backend(requested)
        if explicit != "auto":
            return explicit
    pinned = getattr(settings, "backend", None)
    if isinstance(pinned, str) and pinned.strip():
        return normalize_backend(pinned)
    return "auto"


def is_gpu_selector(selector: str) -> bool:
    """Whether a backend selector can only run on a GPU.

    True for ``"gpu"``, the CUDA backend names (``cupy``/``cufinufft``) and a torch
    request pinned to a non-CPU device (``"torch:cuda"`` ...). ``"auto"`` and bare
    ``"torch"`` are device-dependent and report ``False``.
    """
    sel = normalize_backend(selector)
    if sel == "gpu" or sel in GPU_BACKEND_NAMES:
        return True
    return sel.startswith("torch:") and sel != "torch:cpu"


def cpu_pool_backend(requested: str | None, settings: object | None = None) -> str:
    """The backend selector a CPU process-pool worker must resolve.

    A CPU batch pool (``device="cpu"``) spawns one process per core. Resolving the
    default ``"auto"`` inside each worker would put every one of them on the GPU
    whenever a CUDA device is present — dozens of CUDA contexts on one card. This maps
    the request (after the ``settings.backend`` fallback of :func:`requested_backend`)
    onto a CPU-only selector: ``"auto"`` becomes ``"cpu"``, bare ``"torch"`` becomes
    ``"torch:cpu"``, concrete CPU names pass through, and anything that can only run
    on a GPU raises.

    Parameters
    ----------
    requested : str or None
        The driver's ``backend=`` argument.
    settings : settings model, optional
        The method's settings (its ``backend`` field is the fallback).

    Returns
    -------
    str
        A selector that never resolves to a GPU backend.

    Raises
    ------
    ValueError
        If the request (or the settings pin) names a GPU-only backend.
    """
    sel = requested_backend(requested, settings)
    if sel == "auto":
        return "cpu"
    if sel == "torch":
        return "torch:cpu"
    if is_gpu_selector(sel):
        raise ValueError(
            f"backend {sel!r} runs on a GPU, but device='cpu' asks for a CPU process "
            "pool; use device='gpu', or a CPU backend ('auto', 'cpu', or a concrete "
            "CPU name)"
        )
    return sel


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
    """Return cupy for a device array, else numpy (for device-agnostic assembly).

    A cupy array cannot exist unless cupy has been imported, so when it is absent from
    ``sys.modules`` the answer is numpy without importing it: a CPU run never pays the
    cupy import (and its CUDA DLL loads) just to learn that its arrays are numpy.
    """
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            module: ModuleType = cupy.get_array_module(a)
            return module
        except Exception:
            pass
    import numpy

    return numpy


__all__ = [
    "BACKEND_SELECTORS",
    "GPU_BACKEND_NAMES",
    "array_module",
    "available_backends",
    "best_torch_device",
    "cpu_backends",
    "cpu_pool_backend",
    "cuda_available",
    "ensure_cuda_dll_path",
    "ensure_shared_memory",
    "has_module",
    "is_gpu_selector",
    "normalize_backend",
    "refresh_backends",
    "requested_backend",
    "torch_available",
    "torch_devices",
    "torch_gpu_available",
]
