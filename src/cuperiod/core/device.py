"""GPU device introspection, memory cleanup, and worker-count sizing.

Batch GPU runs are limited by the device, not the host cores: each worker process holds
its own CUDA context, plans/kernels and transient per-light-curve buffers, and without
NVIDIA MPS the processes take turns on the GPU. :func:`suggest_gpu_workers` turns a
probed free-memory figure plus a per-worker footprint into a worker count, capped by
what time-slicing can use, and :func:`free_gpu_memory` bounds cupy memory-pool growth
across a long-lived batch process.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Final

from cuperiod.core.backend import cuda_available, ensure_cuda_dll_path, has_module

#: The ceiling on GPU workers with NVIDIA MPS, where kernels of several processes
#: run concurrently.
_SANE_MAX_GPU_WORKERS: Final = 16

#: The ceiling without MPS. The processes then time-slice the device, so extra
#: workers only overlap their host-side work. Measured on an RTX 5070 Ti with a
#: 16-core host: a GPU GLS batch of 1000-point curves ran 550, 860, 1150 and 1150
#: curves/s with 1, 2, 4 and 8 workers (pool start-up excluded), and a kernel-bound
#: BLS batch ran 16 curves/s with any of 1-8 -- while every extra worker costs a
#: CUDA context of device memory and seconds of start-up.
_NO_MPS_MAX_GPU_WORKERS: Final = 4

#: Device memory one worker process holds before it computes anything: its CUDA
#: context, the CUDA libraries' handles and the loaded kernel modules. Measured on an
#: RTX 5070 Ti (CUDA 12, Windows): 197 MiB for a bare cupy context, 200-235 MiB once
#: GLS, BLS and every other method had run (memory pools emptied); rounded up.
_CONTEXT_BYTES: Final = 256 * 1024**2

#: Grid size assumed when the caller knows none: a survey GLS grid (the default grid
#: holds ~12.5 samples per light-curve point, so this is an 8000-point curve).
_DEFAULT_GRID_HINT: Final = 100_000

#: Where an NVIDIA MPS control daemon listens unless ``CUDA_MPS_PIPE_DIRECTORY``
#: moves it (Linux only; there is no MPS on Windows or macOS).
_MPS_DEFAULT_PIPE: Final = "/tmp/nvidia-mps"


def _mps_active() -> bool:
    """Whether CUDA clients of this process would connect to an MPS daemon."""
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        return True
    if sys.platform.startswith("linux"):
        return os.path.exists(os.path.join(_MPS_DEFAULT_PIPE, "control"))
    return False


def _host_cores() -> int:
    """CPUs this process may run on (its affinity mask where the OS reports one)."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        try:
            return max(1, len(getter(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


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
            mps_enabled=_mps_active(),
        )
    except Exception:
        return None


def free_gpu_memory() -> None:
    """Return cupy's cached device + pinned blocks to the driver. No-op off GPU.

    A long-lived batch process allocates many variable-sized device arrays per light
    curve; cupy's default pool retains freed blocks, so across thousands of light
    curves the pool can grow and fragment until a routine allocation stalls. Calling
    this between batch shards bounds pool growth — the GPU analogue of a fresh worker
    teardown. The ``cudaFree`` of cached blocks costs ~milliseconds. The plans of the
    process-wide GLS engine (:func:`~cuperiod.methods.gls.shared_cufinufft_engine`)
    are destroyed too.
    """
    gls = sys.modules.get("cuperiod.methods.gls")
    if gls is not None:
        # Only when GLS is loaded: freeing memory must not import cufinufft.
        gls.release_shared_engines()
    if not has_module("cupy"):
        return
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        return


def _estimate_worker_bytes(method: str, n_points: int) -> int:
    """Per-worker device footprint (bytes) for ``method`` at ``n_points``.

    The worker's CUDA context (:data:`_CONTEXT_BYTES`, measured) plus the method's
    own working-set estimate, or a generic one -- a handful of complex128 buffers
    over the grid plus a plan base -- for a name no method answers to.
    """
    try:
        from cuperiod.methods.base import get_method

        working = int(get_method(method).estimate_device_bytes(n_points))
    except Exception:
        working = 64 * 1024**2 + n_points * 16 * 6
    return _CONTEXT_BYTES + working


def suggest_gpu_workers(
    method: str = "GLS",
    *,
    n_points_hint: int = _DEFAULT_GRID_HINT,
    grid_size: int | None = None,
    headroom: float = 0.20,
    device_id: int = 0,
) -> int:
    """Suggest a GPU worker count from probed free memory and a footprint estimate.

    Three limits, the smallest wins: the workers whose footprint fits in the free
    memory (each one's CUDA context -- measured at ~200-235 MiB -- plus the method's
    working set on the grid), what the device can use concurrently (16 with NVIDIA
    MPS; 4 without, where the processes take turns on the GPU and extra workers only
    overlap their host work -- measured, see :data:`_NO_MPS_MAX_GPU_WORKERS`), and
    the host CPUs this process may use (each worker drives the device from one).

    Parameters
    ----------
    method : str, default "GLS"
        Method name, used to pick a per-worker footprint estimate.
    n_points_hint : int, default 100000
        Assumed grid size when ``grid_size`` is not given (a survey GLS grid).
    grid_size : int, optional
        Actual grid size, if known.
    headroom : float, default 0.20
        Fraction of free memory to leave unused (driver/fragmentation slack), in
        ``[0, 1)``.
    device_id : int, default 0
        CUDA device ordinal.

    Returns
    -------
    int
        A worker count ``>= 1``; 1 when no GPU is available.

    Raises
    ------
    ValueError
        If ``headroom`` is not a finite fraction in ``[0, 1)``.
    """
    headroom = float(headroom)
    if not (math.isfinite(headroom) and 0.0 <= headroom < 1.0):
        raise ValueError(f"headroom must be in [0, 1), got {headroom}")
    info = gpu_info(device_id)
    if info is None:
        return 1
    n_points = int(grid_size if grid_size is not None else n_points_hint)
    per_worker = max(1, _estimate_worker_bytes(method, max(0, n_points)))
    usable = info.free_bytes * (1.0 - headroom)
    by_memory = int(usable // per_worker)
    by_device = _SANE_MAX_GPU_WORKERS if info.mps_enabled else _NO_MPS_MAX_GPU_WORKERS
    return max(1, min(by_memory, by_device, _host_cores()))


__all__ = [
    "GpuInfo",
    "free_gpu_memory",
    "gpu_info",
    "suggest_gpu_workers",
]
