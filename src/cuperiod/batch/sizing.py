"""Worker-count selection and thread pinning for batch runs.

CPU batches parallelize at the *process* level (one chunk per worker), so each worker's
numpy/FFT must stay single-threaded — otherwise ``N`` workers times ``M`` internal
threads oversubscribe the cores and throughput collapses as the worker count approaches
the core count. :func:`pin_worker_threads` enforces one math thread per process; the GPU
worker count comes from :func:`cuperiod.core.device.suggest_gpu_workers`.
"""

from __future__ import annotations

import os

#: Environment variables that cap per-process math-library threading.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def pin_worker_threads() -> None:
    """Force one math-library thread per process (call before spawning the pool).

    Sets the BLAS/OpenMP thread-count environment variables to ``1`` if unset. On the
    ``spawn`` start method (Windows/macOS default) children inherit this environment and
    import numpy fresh, so setting it here pins them too.
    """
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, "1")


def cpu_worker_count(requested: int | None) -> int:
    """Resolve a CPU worker count.

    Parameters
    ----------
    requested : int or None
        Explicit count, ``None``/``-1`` for "all but one core", or ``0`` for inline
        (single-process) execution.

    Returns
    -------
    int
        Worker count ``>= 1``.
    """
    cores = os.cpu_count() or 1
    if requested is None or requested < 0:
        return max(1, cores - 1)
    if requested == 0:
        return 1
    return max(1, requested)


__all__ = ["cpu_worker_count", "pin_worker_threads"]
