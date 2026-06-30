"""Array-API dispatch and the few ops that live outside the standard.

cuPeriod's portable compute paths are written against the Python array API standard
(via :mod:`array_api_compat`) so a single code path runs on NumPy (CPU), CuPy
(CUDA/ROCm), and PyTorch (CUDA/ROCm/MPS/XPU/CPU). A handful of operations the kernels
need are not in the standard — scatter-add, host transfer, and device/precision
resolution — so they live here as thin, backend-aware shims and the method bodies stay
clean and namespace-generic.

The vendor-specific fast paths (cufinufft for GLS; the cupy ``RawKernel`` searches for
BLS/PDM/CE) do *not* go through here — they remain the NVIDIA accelerators. This module
is the portable tier that everything else rides on.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Literal

import numpy as np

from cuperiod.core.errors import BackendUnavailableError

#: User-facing precision selector.
Precision = Literal["auto", "float64", "float32"]

#: Torch device kinds cuPeriod understands (ROCm reports as ``"cuda"``).
TORCH_DEVICES: tuple[str, ...] = ("cpu", "cuda", "mps", "xpu")


def array_namespace(*xs: Any) -> ModuleType:
    """Return the array-API namespace for ``xs`` (an ``array_api_compat`` wrapper).

    Works for NumPy, CuPy, and PyTorch arrays. The returned namespace exposes the
    standard surface (``astype`` as a function, ``clip`` without ``out=``,
    ``remainder``, ``xp.float64`` dtype objects) regardless of the library version.
    """
    import array_api_compat

    return array_api_compat.array_namespace(*xs)  # type: ignore[no-any-return]


def is_torch_array(x: Any) -> bool:
    """Whether ``x`` is a torch tensor (``False`` if torch is not installed)."""
    try:
        import array_api_compat

        return bool(array_api_compat.is_torch_array(x))
    except Exception:
        return False


def is_cupy_array(x: Any) -> bool:
    """Whether ``x`` is a cupy ndarray (``False`` if cupy is not installed)."""
    try:
        import array_api_compat

        return bool(array_api_compat.is_cupy_array(x))
    except Exception:
        return False


def device_of(x: Any) -> str:
    """Device kind of an array: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"xpu"``.

    A cupy array is always on ``"cuda"`` (ROCm included); a numpy array on ``"cpu"``;
    a torch tensor reports its own ``device.type``.
    """
    if is_torch_array(x):
        return str(x.device.type)
    if is_cupy_array(x):
        return "cuda"
    return "cpu"


def scatter_add(target: Any, index: Any, values: Any) -> None:
    """In-place ``target[index] += values`` with repeated indices accumulated.

    ``target`` is 1-D and ``index``/``values`` are 1-D and aligned. ``numpy.add.at`` is
    not part of the array-API standard and PyTorch has no equivalent *function*, so this
    dispatches per backend: torch uses ``Tensor.index_add_``; numpy/cupy use ``add.at``.

    ``values.dtype`` must equal ``target.dtype``: torch ``index_add_`` rejects a
    mismatch (numpy/cupy would silently cast), so callers build both at the same
    working float dtype. ``index`` may be any integer dtype (coerced to int64).
    """
    if is_torch_array(target):
        target.index_add_(0, index.long(), values)
        return
    if is_cupy_array(target):
        import cupy

        cupy.add.at(target, index, values)
        return
    np.add.at(target, index, values)


def to_host(a: Any) -> np.ndarray:
    """Contiguous NumPy ``float64`` copy of a backend array (device → host).

    Preserves cuPeriod's output contract that every periodogram is returned as numpy
    float64, regardless of the device/precision it was computed in.
    """
    if is_torch_array(a):
        return np.ascontiguousarray(a.detach().to("cpu").numpy(), dtype=np.float64)
    if is_cupy_array(a):
        import cupy

        return np.ascontiguousarray(cupy.asnumpy(a), dtype=np.float64)
    return np.ascontiguousarray(np.asarray(a), dtype=np.float64)


def resolve_precision(precision: str, device: str) -> str:
    """Resolve ``precision`` to a concrete ``"float64"``/``"float32"`` for ``device``.

    float64 is the default everywhere it is supported. Apple MPS cannot represent
    float64 (a Metal limitation), so ``"auto"`` becomes float32 there; an explicit
    ``"float64"`` request on MPS raises rather than silently downgrading.
    """
    if precision == "float64":
        if device == "mps":
            raise BackendUnavailableError(
                "MPS cannot compute in float64; pass precision='float32' for the Apple "
                "GPU, or backend='cpu' (or backend='torch:cpu') for full float64."
            )
        return "float64"
    if precision == "float32":
        return "float32"
    return "float32" if device == "mps" else "float64"


def resolve_torch_device(backend: str, settings_device: str = "auto") -> str:
    """Concrete torch device for a (possibly device-qualified) ``backend`` string.

    ``"torch:mps"`` forces ``mps``; bare ``"torch"`` defers to ``settings_device``
    (``"auto"`` → :func:`~cuperiod.core.backend.best_torch_device`). Raises if the
    chosen device is not present here.
    """
    from cuperiod.core.backend import best_torch_device, torch_devices

    device = backend.split(":", 1)[1] if ":" in backend else settings_device
    if device == "auto":
        device = best_torch_device()
    if device not in torch_devices():
        raise BackendUnavailableError(
            f"torch device {device!r} is not available here "
            f"(present: {sorted(torch_devices()) or ['none']})"
        )
    return device


def float_dtype(xp: ModuleType, resolved_precision: str) -> Any:
    """The namespace float dtype object for a resolved precision name."""
    return xp.float32 if resolved_precision == "float32" else xp.float64


def int_dtype(xp: ModuleType) -> Any:
    """The namespace integer dtype used for index/bin arrays (always int64)."""
    return xp.int64


def to_device_array(host: np.ndarray, *, device: str, dtype: Any) -> Any:
    """Place a host numpy array onto a torch ``device`` as ``dtype``.

    Used by the portable torch paths. ``dtype`` is a torch dtype object. Returns a torch
    tensor; the caller obtains the matching array-API namespace via
    :func:`array_namespace`.
    """
    import torch

    return torch.as_tensor(
        np.ascontiguousarray(host), dtype=dtype, device=torch.device(device)
    )


__all__ = [
    "TORCH_DEVICES",
    "Precision",
    "array_namespace",
    "device_of",
    "float_dtype",
    "int_dtype",
    "is_cupy_array",
    "is_torch_array",
    "resolve_precision",
    "resolve_torch_device",
    "scatter_add",
    "to_device_array",
    "to_host",
]
