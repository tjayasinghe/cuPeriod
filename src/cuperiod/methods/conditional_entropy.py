"""Conditional-entropy period search (Graham et al. 2013).

For a trial period the folded data are binned into a 2-D phase-magnitude histogram,
and the Shannon conditional entropy ``H(m | phase)`` of that distribution is computed.
At the true period magnitude is well predicted by phase, so the distribution
concentrates and the conditional entropy drops — this is a *minimization* method. CE is
notably robust to the sparse, aliased sampling of wide-field surveys.

One vectorized kernel runs on numpy (CPU) and cupy (GPU), so the two backends agree
exactly.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, ClassVar, Final, Literal

import numpy as np

from cuperiod.core._arrayapi import (
    array_namespace,
    device_ref,
    resolve_precision,
    resolve_torch_device,
    scatter_counts_rows,
    to_device_array,
    to_host,
)
from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.core.backend import ensure_cuda_dll_path
from cuperiod.core.config import CESettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import (
    GridSpec,
    pseudo_nyquist_frequency,
    uniform_frequency_grid,
)
from cuperiod.core.lightcurve import LightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, register

CEBackend = Literal["numpy", "cupy"]

#: Trial periods per vectorized batch.
DEFAULT_BATCH: Final = 1024


def _entropy_batch(
    xp: ModuleType,
    tau: Any,
    mag_bin: Any,
    periods: Any,
    *,
    n_phase: int,
    n_mag: int,
    batch: int,
) -> Any:
    """Conditional entropy H(m|phase) for each trial period, vectorized over periods.

    Array-API generic (numpy/cupy/torch via array_api_compat). The float dtype follows
    ``periods`` (float64, or float32 on a float32 device); index/bin arrays are int64.
    The cupy ``RawKernel`` fast path (NVIDIA) is separate and not this code.
    """
    fdtype = periods.dtype
    idtype = xp.int64
    dev = device_ref(periods)
    n_points = int(tau.shape[0])
    n_periods = int(periods.shape[0])
    n_cells = n_phase * n_mag
    entropy = xp.empty(n_periods, dtype=fdtype, device=dev)
    for start in range(0, n_periods, batch):
        stop = min(start + batch, n_periods)
        pb = periods[start:stop]
        n_p = int(pb.shape[0])
        phase = xp.remainder(tau[None, :] / pb[:, None], 1.0)
        phase_bin = xp.clip(xp.astype(phase * n_phase, idtype), 0, n_phase - 1)
        cell = phase_bin * n_mag + mag_bin[None, :]  # (P, N) in [0, n_cells)
        count = xp.zeros((n_p, n_cells), dtype=fdtype, device=dev)
        scatter_counts_rows(count, cell)
        count = xp.reshape(count, (n_p, n_phase, n_mag))

        phase_total = xp.sum(count, axis=2, keepdims=True)  # (P, n_phase, 1)
        mask = count > 0.0
        safe_count = xp.where(mask, count, 1.0)
        safe_total = xp.where(phase_total > 0.0, phase_total, 1.0)
        term = xp.where(
            mask, count * (xp.log(safe_total) - xp.log(safe_count)), 0.0
        )
        entropy[start:stop] = xp.sum(term, axis=(1, 2)) / n_points
    return entropy


#: CUDA threads per block (one block per trial period).
CUDA_BLOCK: Final = 128

#: One-block-per-period CE kernel: a block folds its points into a shared-memory 2-D
#: phase-magnitude histogram (magnitude bins precomputed on the host), then one thread
#: reduces it to the Shannon conditional entropy H(m | phase).
_CE_CUDA_SRC: Final = r"""
extern "C" __global__ void ce_block(
    const double* __restrict__ tau, const int* __restrict__ mag_bin,
    const double* __restrict__ periods,
    const int n_points, const int n_periods, const int n_phase, const int n_mag,
    double* o_entropy)
{
    const int pidx = blockIdx.x;
    if (pidx >= n_periods) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const double period = periods[pidx];
    const int M = n_phase * n_mag;

    extern __shared__ double cnt[];  // (M) point count per (phase, mag) cell
    for (int i = tid; i < M; i += nth) cnt[i] = 0.0;
    __syncthreads();

    for (int j = tid; j < n_points; j += nth) {
        double x = tau[j];
        double ph = (x - period * floor(x / period)) / period;  // mod(tau/period, 1)
        int pb = (int)(ph * n_phase);
        if (pb >= n_phase) pb = n_phase - 1;
        if (pb < 0) pb = 0;
        atomicAdd(&cnt[pb * n_mag + mag_bin[j]], 1.0);
    }
    __syncthreads();

    if (tid == 0) {
        double h = 0.0;
        for (int p = 0; p < n_phase; ++p) {
            double ci = 0.0;
            for (int m = 0; m < n_mag; ++m) ci += cnt[p * n_mag + m];
            if (ci > 0.0) {
                double lci = log(ci);
                for (int m = 0; m < n_mag; ++m) {
                    double c = cnt[p * n_mag + m];
                    if (c > 0.0) h += c * (lci - log(c));
                }
            }
        }
        o_entropy[pidx] = h / (double)n_points;
    }
}
"""

_ce_kernel_cache: dict[int, Any] = {}


def _ce_kernel(block: int) -> Any:
    """Compile (once) and cache the CE RawKernel for a given block size."""
    kernel = _ce_kernel_cache.get(block)
    if kernel is None:
        import cupy

        kernel = cupy.RawKernel(_CE_CUDA_SRC, "ce_block")
        _ce_kernel_cache[block] = kernel
    return kernel


def _ce_cuda(
    tau: FloatArray,
    mag_bin: IntArray,
    periods: FloatArray,
    *,
    n_phase: int,
    n_mag: int,
    block: int = CUDA_BLOCK,
) -> FloatArray:
    """One-block-per-period CUDA conditional entropy; returns a host float64 array."""
    import cupy as cp

    tau_d = cp.asarray(np.ascontiguousarray(tau, dtype=np.float64))
    mag_d = cp.asarray(np.ascontiguousarray(mag_bin, dtype=np.int32))
    per_d = cp.asarray(np.ascontiguousarray(periods, dtype=np.float64))
    n_periods = int(per_d.size)
    if n_periods == 0:
        return np.zeros(0, dtype=np.float64)
    out = cp.empty(n_periods, dtype=cp.float64)
    from cuperiod.core.backend import ensure_shared_memory

    smem = n_phase * n_mag * 8
    kernel = _ce_kernel(block)
    ensure_shared_memory(kernel, smem, method="CE", hint="n_phase_bins / n_mag_bins")
    kernel(
        (n_periods,),
        (block,),
        (
            tau_d, mag_d, per_d,
            np.int32(tau_d.size), np.int32(n_periods),
            np.int32(n_phase), np.int32(n_mag), out,
        ),
        shared_mem=smem,
    )
    return np.asarray(cp.asnumpy(out), dtype=np.float64)


def conditional_entropy(
    t: FloatArray,
    y: FloatArray,
    periods: FloatArray,
    *,
    n_phase_bins: int = 10,
    n_mag_bins: int = 10,
    backend: str = "numpy",
    batch: int = DEFAULT_BATCH,
    precision: str = "auto",
) -> FloatArray:
    """Conditional entropy for each trial period (minimized at the true period).

    Parameters
    ----------
    t, y : numpy.ndarray
        Finite times (days) and values of one band.
    periods : numpy.ndarray
        Trial periods (days).
    n_phase_bins, n_mag_bins : int, default 10
        Histogram resolution in phase and magnitude.
    backend : {"numpy", "cupy"}, default "numpy"
        CPU or GPU.
    batch : int, default 1024
        Trial periods per vectorized batch.

    Returns
    -------
    numpy.ndarray
        Conditional entropy per period.
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    periods_host = np.ascontiguousarray(periods, dtype=np.float64)
    if periods_host.size == 0:
        return np.zeros(0, dtype=np.float64)
    tau = t - t.min()
    span = float(y.max() - y.min())
    if span <= 0.0:
        return np.zeros(periods_host.size, dtype=np.float64)
    mag_bin = np.clip(
        ((y - y.min()) / span * n_mag_bins).astype(np.int64), 0, n_mag_bins - 1
    )

    if backend == "cupy":
        ensure_cuda_dll_path()
        return _ce_cuda(
            tau, mag_bin, periods_host, n_phase=n_phase_bins, n_mag=n_mag_bins
        )
    if backend == "torch" or backend.startswith("torch:"):
        import torch

        device = backend.split(":", 1)[1] if ":" in backend else "cpu"
        fdt = (torch.float32
               if resolve_precision(precision, device) == "float32" else torch.float64)
        tau_d = to_device_array(tau, device=device, dtype=fdt)
        mag_d = to_device_array(mag_bin, device=device, dtype=torch.int64)
        per_d = to_device_array(periods_host, device=device, dtype=fdt)
        return to_host(_entropy_batch(
            array_namespace(per_d), tau_d, mag_d, per_d,
            n_phase=n_phase_bins, n_mag=n_mag_bins, batch=batch,
        ))
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    return to_host(_entropy_batch(
        array_namespace(periods_host), tau, mag_bin, periods_host,
        n_phase=n_phase_bins, n_mag=n_mag_bins, batch=batch,
    ))


class ConditionalEntropyMethod(PeriodogramMethod):
    """Conditional-entropy period search (numpy CPU, cupy GPU)."""

    name: ClassVar[str] = "CE"
    objective_sense: ClassVar[Literal["max", "min"]] = "min"
    supports_multiband: ClassVar[bool] = False
    settings_cls: ClassVar[type] = CESettings
    cpu_backend: ClassVar[str] = "numpy"
    gpu_backend: ClassVar[str | None] = "cupy"
    portable_gpu_backend: ClassVar[str | None] = "torch"
    all_backends: ClassVar[tuple[str, ...]] = ("numpy", "cupy", "torch")

    def default_grid(self, lc: LightCurve, settings: CESettings) -> GridSpec:  # type: ignore[override]
        finite = lc.finite()
        if finite.baseline <= 0.0:
            raise InsufficientDataError("CE: no usable time baseline")
        minimum = settings.minimum_frequency or 1.0 / finite.baseline
        maximum = settings.maximum_frequency or pseudo_nyquist_frequency(
            finite.time, settings.nyquist_factor
        )
        return uniform_frequency_grid(
            finite.baseline,
            maximum_frequency=maximum,
            minimum_frequency=minimum,
            samples_per_peak=settings.samples_per_peak,
        )

    def power(  # type: ignore[override]
        self,
        grid: GridSpec,
        lc: LightCurve,
        settings: CESettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        finite = lc.finite()
        n = finite.n
        if n < settings.min_detections:
            raise InsufficientDataError(
                f"CE: {n} finite points < min_detections {settings.min_detections}"
            )
        if finite.baseline <= 0.0:
            raise InsufficientDataError("CE: no usable time baseline")
        periods = grid.period
        if backend == "torch" or backend.startswith("torch:"):
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"
        entropy = conditional_entropy(
            finite.time, finite.value, periods,
            n_phase_bins=settings.n_phase_bins, n_mag_bins=settings.n_mag_bins,
            backend=backend, batch=settings.batch_periods,
            precision=settings.precision,
        )
        return Periodogram.from_spectrum(
            method="CE",
            backend=backend,
            frequency=1.0 / periods,
            power=entropy,
            objective_sense="min",
            n_samples=n,
            baseline=finite.baseline,
            meta=finite.meta,
        )

    def estimate_device_bytes(self, n_points: int) -> int:
        return 128 * 1024**2 + n_points * 8 * 8


register(ConditionalEntropyMethod())

__all__ = [
    "CEBackend",
    "ConditionalEntropyMethod",
    "DEFAULT_BATCH",
    "conditional_entropy",
]
