"""Worker-count selection and thread pinning for batch runs.

CPU batches parallelize at the *process* level (one chunk per worker), so each worker's
numpy/FFT must stay single-threaded — otherwise ``N`` workers times ``M`` internal
threads oversubscribe the cores and throughput collapses as the worker count approaches
the core count. The GPU worker count comes from
:func:`cuperiod.core.device.suggest_gpu_workers`.

The recipe for a CPU process pool (``cpu_pool_backend`` lives in
:mod:`cuperiod.core.backend`)::

    n_workers = cpu_worker_count(workers)  # affinity/cgroup-aware, Windows-capped
    backend = cpu_pool_backend(backend, settings)  # never on the GPU
    with pinned_worker_threads():  # BLAS/OpenMP = 1 thread, restored afterwards
        ProcessPoolExecutor(
            n_workers, mp_context=spawn,
            initializer=init_worker_threads,
            initargs=(numba_threads_per_worker(n_workers),),
        )

BLAS/OpenMP are pinned to one thread through the environment the spawned children
inherit; the numba tier keeps a small thread pool of its own (``cores // workers``)
set inside each child, because a flat numba pin of 1 measured up to 4.5x slower for
small pools while unpinned numba oversubscribed large ones.
"""

from __future__ import annotations

import concurrent.futures.process
import contextlib
import math
import os
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

#: Environment variables that cap per-process math-library threading.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: ``ProcessPoolExecutor`` refuses more workers than this on Windows (the
#: ``WaitForMultipleObjects`` handle limit, less the executor's own two handles).
_WINDOWS_POOL_LIMIT = int(
    getattr(concurrent.futures.process, "_MAX_WINDOWS_WORKERS", 61)
)


def _affinity_cpu_count() -> int | None:
    """CPUs in this process's affinity mask, or ``None`` when it cannot be read.

    Linux (and other POSIX with ``sched_getaffinity``) reports the mask set by
    ``taskset``, cpusets, SLURM/PBS task binding or a container. On Windows the mask
    comes from ``GetProcessAffinityMask`` (``start /affinity``, job objects); a process
    spanning several processor groups reports no mask, and ``None`` is returned.
    """
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            return max(1, len(getaffinity(0)))
        except OSError:
            return None
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            # A private WinDLL with explicit prototypes: the default int restype
            # truncates the 64-bit process handle, and the shared ctypes.windll
            # prototypes are not ours to change.
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            current = kernel32.GetCurrentProcess
            current.restype = wintypes.HANDLE
            current.argtypes = []
            get_mask = kernel32.GetProcessAffinityMask
            get_mask.restype = wintypes.BOOL
            get_mask.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            if get_mask(
                current(), ctypes.byref(process_mask), ctypes.byref(system_mask)
            ) and process_mask.value:
                return max(1, bin(process_mask.value).count("1"))
        except Exception:
            return None
    return None


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _cgroup_cpu_limit() -> int | None:
    """The CPU quota of this process's cgroup (Linux), rounded up, or ``None``.

    Containers limited with ``--cpus`` (cgroup ``cpu.max`` / ``cfs_quota_us``) see every
    host CPU in their affinity mask; the quota is the real allocation.
    """
    if not sys.platform.startswith("linux"):
        return None
    # cgroup v2: "<quota> <period>" or "max <period>" in cpu.max, looked up first in
    # this process's own cgroup (from /proc/self/cgroup), then at the mount root
    # (what a container sees as its own cgroup).
    paths: list[str] = []
    for line in (_read_text("/proc/self/cgroup") or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            paths.append(f"/sys/fs/cgroup{parts[2].rstrip('/')}/cpu.max")
    paths.append("/sys/fs/cgroup/cpu.max")
    for path in paths:
        fields = (_read_text(path) or "").split()
        if len(fields) == 2:
            return _quota_cpus(fields[0], fields[1])
    # cgroup v1: a quota of -1 means unlimited.
    return _quota_cpus(
        _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") or "max",
        _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us") or "max",
    )


def _quota_cpus(quota_text: str, period_text: str) -> int | None:
    """``ceil(quota / period)`` for a positive numeric quota, else ``None``."""
    try:
        quota, period = float(quota_text), float(period_text)
    except ValueError:  # "max": no quota
        return None
    if quota > 0 and period > 0:
        return max(1, math.ceil(quota / period))
    return None


def usable_cpu_count() -> int:
    """The number of CPUs this process may actually use (``>= 1``).

    ``os.cpu_count()`` is the machine's logical CPU count; under a SLURM/PBS
    allocation, ``taskset``, a container or a Windows affinity mask the usable set is
    smaller, and sizing a pool from the machine count starts dozens of workers on a
    handful of cores. This takes the smallest of ``os.cpu_count()``,
    ``os.process_cpu_count()`` (Python 3.13+, which also honours ``-X cpu_count``),
    the affinity mask and the cgroup CPU quota.
    """
    counts = [os.cpu_count() or 1]
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        n = process_cpu_count()
        if n:
            counts.append(int(n))
    for probe in (_affinity_cpu_count, _cgroup_cpu_limit):
        n = probe()
        if n:
            counts.append(n)
    return max(1, min(counts))


def max_pool_workers() -> int | None:
    """The most worker processes a process pool accepts here (``None``: no cap).

    ``ProcessPoolExecutor`` raises above 61 workers on Windows; unlimited elsewhere.
    """
    return _WINDOWS_POOL_LIMIT if sys.platform == "win32" else None


def clamp_pool_workers(n_workers: int, *, warn: bool = True) -> int:
    """Clamp a process-pool size to :func:`max_pool_workers`.

    Parameters
    ----------
    n_workers : int
        The requested pool size.
    warn : bool, default True
        Emit a ``RuntimeWarning`` when the request is reduced (an explicit user
        request); pass ``False`` for a computed default.

    Returns
    -------
    int
        ``n_workers``, or the platform cap when it is lower.
    """
    cap = max_pool_workers()
    if cap is not None and n_workers > cap:
        if warn:
            warnings.warn(
                f"{n_workers} workers exceeds the {cap}-process limit of a process "
                f"pool on this platform; using {cap}.",
                RuntimeWarning,
                stacklevel=3,
            )
        return cap
    return n_workers


def cpu_worker_count(requested: int | None) -> int:
    """Resolve a CPU worker count.

    Parameters
    ----------
    requested : int or None
        Explicit count, ``None``/``-1`` for "all but one usable core", or ``0`` for
        inline (single-process) execution.

    Returns
    -------
    int
        Worker count ``>= 1``. The default counts the CPUs this process may use
        (:func:`usable_cpu_count`), not the machine's; every count is capped at the
        platform's process-pool limit (61 on Windows), with a ``RuntimeWarning`` when an
        explicit request is reduced.
    """
    if requested is None or requested < 0:
        return clamp_pool_workers(max(1, usable_cpu_count() - 1), warn=False)
    if requested == 0:
        return 1
    return clamp_pool_workers(max(1, requested))


def numba_threads_per_worker(n_workers: int) -> int:
    """numba threads each of ``n_workers`` pool processes should run (``>= 1``).

    ``usable_cpu_count() // n_workers``: the full default pool runs one thread per
    process, a small pool lets each worker's ``prange`` kernels use its share of the
    cores. Pass the number of processes that will actually run.
    """
    return max(1, usable_cpu_count() // max(1, n_workers))


@contextlib.contextmanager
def pinned_worker_threads() -> Iterator[None]:
    """Pin BLAS/OpenMP to one thread for processes spawned inside the block.

    Sets the thread-count environment variables to ``1`` where unset (a user's own
    setting is kept), yields, and restores the previous environment on exit — also
    when the block raises. Spawned children (the start method every batch pool uses)
    inherit the environment at spawn and import numpy fresh, so creating and draining
    the pool inside the block pins them without leaving the parent session
    single-threaded afterwards.
    """
    saved = {var: os.environ.get(var) for var in _THREAD_ENV_VARS}
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, "1")
    try:
        yield
    finally:
        for var, old in saved.items():
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old


def pin_worker_threads() -> None:
    """Force one math-library thread per process (call before spawning the pool).

    Sets the BLAS/OpenMP thread-count environment variables to ``1`` if unset. The
    change is permanent for this process; prefer :func:`pinned_worker_threads`, which
    restores the environment once the pool is done.
    """
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, "1")


def init_worker_threads(
    numba_threads: int,
    initializer: Callable[..., object] | None = None,
    initargs: tuple[Any, ...] = (),
) -> None:
    """Process-pool initializer: cap this worker's numba threads, then chain on.

    Runs inside each spawned worker, so the parent is never touched. numba ignores
    ``OMP_NUM_THREADS``; left alone, every worker's ``prange`` kernels start one thread
    per core (workers x cores threads). If numba is not imported yet,
    ``NUMBA_NUM_THREADS`` is set so its pool starts at the right size; if it already
    is, the supported ``numba.set_num_threads`` is used instead (rewriting the variable
    after numba has read it makes its next configuration reload raise). A user's own
    ``NUMBA_NUM_THREADS`` is respected.

    Parameters
    ----------
    numba_threads : int
        Threads for this worker (see :func:`numba_threads_per_worker`).
    initializer : callable, optional
        A further initializer to run afterwards (e.g. a GPU pool's engine builder).
    initargs : tuple
        Arguments for ``initializer``.
    """
    n = max(1, int(numba_threads))
    if "NUMBA_NUM_THREADS" not in os.environ:
        numba = sys.modules.get("numba")
        if numba is None:
            os.environ["NUMBA_NUM_THREADS"] = str(n)
        else:
            # A thread cap must never kill a worker.
            with contextlib.suppress(Exception):
                numba.set_num_threads(min(n, int(numba.config.NUMBA_NUM_THREADS)))
    if initializer is not None:
        initializer(*initargs)


__all__ = [
    "clamp_pool_workers",
    "cpu_worker_count",
    "init_worker_threads",
    "max_pool_workers",
    "numba_threads_per_worker",
    "pin_worker_threads",
    "pinned_worker_threads",
    "usable_cpu_count",
]
