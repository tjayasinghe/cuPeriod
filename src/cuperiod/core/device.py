"""GPU device introspection, memory cleanup, and worker-count sizing.

Batch GPU runs are limited by device memory, not cores: each worker process holds its
own plans/kernels and transient per-light-curve buffers. :func:`suggest_gpu_workers`
turns a probed free-memory figure plus a per-worker footprint estimate into a safe
worker count, and :func:`free_gpu_memory` bounds cupy memory-pool growth across a
long-lived batch process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cuperiod.core.backend import cuda_available, ensure_cuda_dll_path, has_module

#: A conservative ceiling on GPU workers. Without NVIDIA MPS, processes time-slice a
#: single CUDA context, so many workers rarely help; this caps a runaway suggestion.
_SANE_MAX_GPU_WORKERS = 16


@dataclass(frozen=True)
class GpuInfo:
    """A snapshot of one CUDA device's identity and free memory."""

    device_id: int
    name: str
    total_bytes: int
    free_bytes: int
    mps_enabled: bool

    @property
    def total_gib(self) -> float:
        """Total device memory in GiB."""
        return self.total_bytes / 1024**3

    @property
    def free_gib(self) -> float:
        """Free device memory in GiB."""
        return self.free_bytes / 1024**3

    def __str__(self) -> str:
        mps = "on" if self.mps_enabled else "off"
        return (
            f"GPU {self.device_id}: {self.name} | "
            f"{self.free_gib:.1f}/{self.total_gib:.1f} GiB free | MPS {mps}"
        )


def gpu_info(device_id: int = 0) -> GpuInfo | None:
    """Return a :class:`GpuInfo` for ``device_id``, or ``None`` if no GPU is usable.

    Parameters
    ----------
    device_id : int, default 0
        CUDA device ordinal.

    Returns
    -------
    GpuInfo or None
    """
    if not cuda_available():
        return None
    ensure_cuda_dll_path()
    try:
        import cupy

        with cupy.cuda.Device(device_id):
            free, total = cupy.cuda.runtime.memGetInfo()
            props = cupy.cuda.runtime.getDeviceProperties(device_id)
        raw_name = props["name"]
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        return GpuInfo(
            device_id=device_id,
            name=name,
            total_bytes=int(total),
            free_bytes=int(free),
            mps_enabled="CUDA_MPS_PIPE_DIRECTORY" in os.environ,
        )
    except Exception:
        return None


def free_gpu_memory() -> None:
    """Return cupy's cached device + pinned blocks to the driver. No-op off GPU.

    A long-lived batch process allocates many variable-sized device arrays per light
    curve; cupy's default pool retains freed blocks, so across thousands of light
    curves the pool can grow and fragment until a routine allocation stalls. Calling
    this between batch shards bounds pool growth — the GPU analogue of a fresh worker
    teardown. The ``cudaFree`` of cached blocks costs ~milliseconds.
    """
    if not has_module("cupy"):
        return
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        return


def _estimate_worker_bytes(method: str, n_points: int) -> int:
    """Per-worker device footprint estimate (bytes) for ``method`` at ``n_points``.

    Defers to the method's own estimate when available, else uses a generic figure:
    a handful of complex128 buffers over the grid plus a fixed plan/context base.
    """
    try:
        from cuperiod.methods.base import get_method

        return int(get_method(method).estimate_device_bytes(n_points))
    except Exception:
        return 64 * 1024**2 + n_points * 16 * 6


def suggest_gpu_workers(
    method: str = "GLS",
    *,
    n_points_hint: int = 2000,
    grid_size: int | None = None,
    headroom: float = 0.20,
    device_id: int = 0,
) -> int:
    """Suggest a GPU worker count from probed free memory and a footprint estimate.

    Parameters
    ----------
    method : str, default "GLS"
        Method name, used to pick a per-worker footprint estimate.
    n_points_hint : int, default 2000
        Assumed grid size when ``grid_size`` is not given.
    grid_size : int, optional
        Actual grid size, if known.
    headroom : float, default 0.20
        Fraction of free memory to leave unused (driver/fragmentation slack).
    device_id : int, default 0
        CUDA device ordinal.

    Returns
    -------
    int
        A worker count ``>= 1``. Returns 1 when no GPU is available. Capped at a sane
        maximum; without MPS, 1-2 workers usually saturate a single device.
    """
    info = gpu_info(device_id)
    if info is None:
        return 1
    n_points = int(grid_size if grid_size is not None else n_points_hint)
    per_worker = max(1, _estimate_worker_bytes(method, n_points))
    usable = info.free_bytes * (1.0 - headroom)
    workers = int(usable // per_worker)
    return max(1, min(workers, _SANE_MAX_GPU_WORKERS))


__all__ = [
    "GpuInfo",
    "free_gpu_memory",
    "gpu_info",
    "suggest_gpu_workers",
]
