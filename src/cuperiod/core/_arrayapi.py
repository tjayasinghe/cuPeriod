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


def device_ref(x: Any) -> Any:
    """The array-API device object of ``x``, for device-matched array creation.

    Unlike :func:`device_of` (a coarse kind string), this returns the exact device a
    namespace's creation functions accept as ``device=``: a ``torch.device`` (with
    index), a cupy ``Device``, or numpy ``"cpu"``. Arrays a portable kernel builds with
    ``xp.zeros``/``arange`` then land on the input's device; without it they default to
    the host on torch (whose data may be on cuda), a cross-device error. numpy and cupy
    inherit the device from context, so it is a harmless no-op for them.
    """
    import array_api_compat

    return array_api_compat.device(x)


def scatter_add(target: Any, index: Any, values: Any) -> None:
    """In-place ``target[index] += values`` with repeated indices accumulated.

    ``target`` is 1-D and ``index``/``values`` are 1-D and aligned. ``numpy.add.at`` is
    not part of the array-API standard and PyTorch has no equivalent *function*, so this
    dispatches per backend: torch uses ``Tensor.index_add_``; cupy uses
    ``cupyx.scatter_add`` (its fast documented scatter); numpy uses ``bincount``, which
    computes the same float64 sums in one buffered pass — ``np.add.at`` is unbuffered
    and an order of magnitude slower on large inputs.

    ``values.dtype`` must equal ``target.dtype``: torch ``index_add_`` rejects a
    mismatch (numpy/cupy would silently cast), so callers build both at the same
    working float dtype. ``index`` may be any integer dtype (coerced to int64).
    """
    if is_torch_array(target):
        target.index_add_(0, index.long(), values)
        return
    if is_cupy_array(target):
        import cupyx

        cupyx.scatter_add(target, index, values)
        return
    binned = np.bincount(
        np.asarray(index), weights=np.asarray(values), minlength=target.size
    )
    target += binned.astype(target.dtype, copy=False)


def scatter_add_rows(target: Any, index: Any, values: Any) -> None:
    """In-place ``target[p, index[p, j]] += values[j]`` with repeats accumulated.

    The batch-kernel binning primitive: ``target`` is 2-D ``(P, W)``, ``index`` is
    ``(P, N)`` int64, and ``values`` is one shared row ``(N,)`` (the usual case — the
    same light curve binned at ``P`` trial periods) or a full ``(P, N)``. Compared to
    flattening and calling :func:`scatter_add`, this avoids materializing the broadcast
    ``(P, N)`` values *and* the flat ``(P, N)`` int64 index on torch — the scatter
    reads a stride-0 expanded view — and gives numpy one fused ``bincount`` pass.
    """
    if is_torch_array(target):
        src = values if values.ndim == 2 else values.unsqueeze(0).expand_as(index)
        target.scatter_add_(1, index, src)
        return
    if is_cupy_array(target):
        import cupy
        import cupyx

        rows = cupy.arange(target.shape[0])[:, None]
        cupyx.scatter_add(target, (rows, index), values)
        return
    n_rows, n_cols = target.shape
    flat = index + (np.arange(n_rows, dtype=np.int64) * n_cols)[:, None]
    weights = np.broadcast_to(values, index.shape)
    binned = np.bincount(
        flat.reshape(-1), weights=weights.reshape(-1), minlength=target.size
    )
    target += binned.reshape(target.shape).astype(target.dtype, copy=False)


def scatter_counts_rows(target: Any, index: Any) -> None:
    """In-place ``target[p, index[p, j]] += 1`` — a per-row histogram count.

    Same layout contract as :func:`scatter_add_rows` but without a values array:
    numpy uses weightless ``bincount`` and torch scatters a stride-0 view of a single
    one, so no ``(P, N)`` ones array is ever built. ``target`` is float (counts are
    accumulated in the working dtype for the entropy/variance math downstream).
    """
    if is_torch_array(target):
        import torch

        one = torch.ones((1, 1), dtype=target.dtype, device=target.device)
        target.scatter_add_(1, index, one.expand_as(index))
        return
    if is_cupy_array(target):
        import cupy
        import cupyx

        rows = cupy.arange(target.shape[0])[:, None]
        cupyx.scatter_add(target, (rows, index), target.dtype.type(1))
        return
    n_rows, n_cols = target.shape
    flat = index + (np.arange(n_rows, dtype=np.int64) * n_cols)[:, None]
    binned = np.bincount(flat.reshape(-1), minlength=target.size)
    target += binned.reshape(target.shape).astype(target.dtype, copy=False)


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
    "device_ref",
    "float_dtype",
    "int_dtype",
    "is_cupy_array",
    "is_torch_array",
    "resolve_precision",
    "resolve_torch_device",
    "scatter_add",
    "scatter_add_rows",
    "scatter_counts_rows",
    "to_device_array",
    "to_host",
]
