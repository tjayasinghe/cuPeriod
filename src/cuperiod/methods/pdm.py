"""Phase Dispersion Minimization (Stellingwerf 1978).

PDM is a non-parametric period search well suited to non-sinusoidal variables (eclipsing
binaries, RR Lyrae, sawtooth pulsators) where Fourier methods need many harmonics. For a
trial period the data are phase-folded and binned; the statistic

    Theta = s^2 / sigma^2

compares the pooled within-bin variance ``s^2`` to the total variance ``sigma^2``.
At the true period the folded scatter collapses and ``Theta`` drops well below 1, so
PDM is a *minimization* method. Overlapping bin sets ("covers", Stellingwerf) reduce
sensitivity to the bin phase.

Backends share one vectorized kernel: ``numpy`` on the CPU and ``cupy`` on the GPU (the
same array code runs on either), so the GPU result is identical to the CPU result.
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
    scatter_add_rows,
    scatter_counts_rows,
    to_device_array,
    to_host,
)
from cuperiod.core._typing import FloatArray
from cuperiod.core.backend import ensure_cuda_dll_path
from cuperiod.core.config import PDMSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import (
    GridSpec,
    pseudo_nyquist_frequency,
    uniform_frequency_grid,
)
from cuperiod.core.lightcurve import LightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, register

PDMBackend = Literal["numpy", "cupy"]

#: Trial periods processed per vectorized batch (bounds the (P, N) phase arrays).
DEFAULT_BATCH: Final = 2048


def _cover_hists(xp: ModuleType, fine: Any, n_bins: int, n_covers: int) -> Any:
    """All covers' bin histograms from one fine histogram, laid side by side.

    ``fine`` is ``(P, n_bins*n_covers)``; cover ``c``'s ``n_bins`` histogram is an
    exact integer regroup of the fine bins — roll right by ``c`` and sum groups of
    ``n_covers`` — so the folded points are binned **once** rather than per cover.
    Returns ``(P, n_bins*n_covers)`` with cover ``c`` at columns ``[c*n_bins, ...)``.
    """
    n_p = int(fine.shape[0])
    parts = [
        xp.sum(
            xp.reshape(xp.roll(fine, cover, axis=1), (n_p, n_bins, n_covers)), axis=2
        )
        for cover in range(n_covers)
    ]
    return xp.concat(parts, axis=1)


def _theta_batch(
    xp: ModuleType,
    tau: Any,
    y: Any,
    y2: Any,
    periods: Any,
    *,
    n_bins: int,
    n_covers: int,
    sigma2: float,
    batch: int,
) -> Any:
    """Stellingwerf Theta for each trial period, vectorized over ``periods``.

    ``s^2 = sum_j SSD_j / (N*n_covers - n_nonempty)`` with ``SSD_j`` the within-bin sum
    of squared deviations across all covers, and ``Theta = s^2 / sigma^2``. The points
    are binned once into ``n_bins*n_covers`` fine bins; every cover's bin statistics
    are exact regroups of that histogram (see :func:`_cover_hists`), which cuts the
    scatter work by ``n_covers``.
    """
    fdtype = periods.dtype
    idtype = xp.int64
    dev = device_ref(periods)
    inf = float("inf")
    n_points = int(tau.shape[0])
    n_periods = int(periods.shape[0])
    n_global = n_bins * n_covers
    theta = xp.empty(n_periods, dtype=fdtype, device=dev)

    for start in range(0, n_periods, batch):
        stop = min(start + batch, n_periods)
        pb = periods[start:stop]
        n_p = int(pb.shape[0])
        phase = xp.remainder(tau[None, :] / pb[:, None], 1.0)  # (P, N) in [0, 1)
        fine = xp.clip(
            xp.astype(phase * float(n_global), idtype), 0, n_global - 1
        )
        f_count = xp.zeros((n_p, n_global), dtype=fdtype, device=dev)
        f_ysum = xp.zeros((n_p, n_global), dtype=fdtype, device=dev)
        f_ysq = xp.zeros((n_p, n_global), dtype=fdtype, device=dev)
        scatter_counts_rows(f_count, fine)
        scatter_add_rows(f_ysum, fine, y)
        scatter_add_rows(f_ysq, fine, y2)

        count = _cover_hists(xp, f_count, n_bins, n_covers)
        ysum = _cover_hists(xp, f_ysum, n_bins, n_covers)
        ysq = _cover_hists(xp, f_ysq, n_bins, n_covers)
        mask = count > 0.0
        safe = xp.where(mask, count, 1.0)
        ssd = xp.where(mask, ysq - ysum * ysum / safe, 0.0)
        nonempty = xp.sum(xp.astype(mask, fdtype), axis=1)
        den = float(n_points * n_covers) - nonempty
        safe_den = xp.where(den > 0.0, den, 1.0)
        s2 = xp.where(den > 0.0, xp.sum(ssd, axis=1) / safe_den, inf)
        theta[start:stop] = s2 / sigma2
    return theta


#: CUDA threads per block (one block per trial period).
CUDA_BLOCK: Final = 128

#: One-block-per-period PDM kernel: a block bins its folded points **once** into
#: ``n_bins*n_covers`` fine shared-memory bins (sum, sum of squares, count); every
#: cover's bin statistics are exact regroups of ``n_covers`` consecutive fine bins
#: (rolled by the cover index), assembled in the one-thread reduction. Cuts the
#: dominant atomic work per point from ``3*n_covers`` to 3.
_PDM_CUDA_SRC: Final = r"""
extern "C" __global__ void pdm_block(
    const double* __restrict__ tau, const double* __restrict__ y,
    const double* __restrict__ periods,
    const int n_points, const int n_periods, const int n_bins, const int n_covers,
    const double sigma2, double* o_theta)
{
    const int pidx = blockIdx.x;
    if (pidx >= n_periods) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const double period = periods[pidx];
    const int M = n_bins * n_covers;

    extern __shared__ double sh[];
    double* s_sum = sh;          // (M) sum of y per fine bin
    double* s_sq = sh + M;       // (M) sum of y^2 per fine bin
    double* s_cnt = sh + 2 * M;  // (M) point count per fine bin
    for (int i = tid; i < M; i += nth) {
        s_sum[i] = 0.0; s_sq[i] = 0.0; s_cnt[i] = 0.0;
    }
    __syncthreads();

    for (int j = tid; j < n_points; j += nth) {
        double x = tau[j];
        double ph = (x - period * floor(x / period)) / period;  // mod(tau/period, 1)
        double yj = y[j];
        int f = (int)(ph * (double)M);
        if (f >= M) f = M - 1;
        if (f < 0) f = 0;
        atomicAdd(&s_sum[f], yj);
        atomicAdd(&s_sq[f], yj * yj);
        atomicAdd(&s_cnt[f], 1.0);
    }
    __syncthreads();

    if (tid == 0) {
        double ssd = 0.0;
        int nonempty = 0;
        for (int c = 0; c < n_covers; ++c) {
            for (int b = 0; b < n_bins; ++b) {
                // cover-c bin b = fine bins (b*n_covers - c .. +n_covers-1) mod M
                double cnt = 0.0, sum = 0.0, sq = 0.0;
                int f0 = b * n_covers - c;
                if (f0 < 0) f0 += M;
                for (int k = 0; k < n_covers; ++k) {
                    int f = f0 + k;
                    if (f >= M) f -= M;
                    cnt += s_cnt[f]; sum += s_sum[f]; sq += s_sq[f];
                }
                if (cnt > 0.0) {
                    ssd += sq - sum * sum / cnt;
                    nonempty += 1;
                }
            }
        }
        double den = (double)(n_points * n_covers - nonempty);
        o_theta[pidx] = (den > 0.0) ? (ssd / den) / sigma2 : 1e30;
    }
}
"""

_pdm_kernel_cache: dict[int, Any] = {}


def _pdm_kernel(block: int) -> Any:
    """Compile (once) and cache the PDM RawKernel for a given block size."""
    kernel = _pdm_kernel_cache.get(block)
    if kernel is None:
        import cupy

        kernel = cupy.RawKernel(_PDM_CUDA_SRC, "pdm_block")
        _pdm_kernel_cache[block] = kernel
    return kernel


def _pdm_cuda(
    tau: FloatArray,
    y: FloatArray,
    periods: FloatArray,
    *,
    n_bins: int,
    n_covers: int,
    sigma2: float,
    block: int = CUDA_BLOCK,
) -> FloatArray:
    """One-block-per-period CUDA PDM Theta; returns a host float64 array."""
    import cupy as cp

    tau_d = cp.asarray(np.ascontiguousarray(tau, dtype=np.float64))
    y_d = cp.asarray(np.ascontiguousarray(y, dtype=np.float64))
    per_d = cp.asarray(np.ascontiguousarray(periods, dtype=np.float64))
    n_periods = int(per_d.size)
    if n_periods == 0:
        return np.zeros(0, dtype=np.float64)
    out = cp.empty(n_periods, dtype=cp.float64)
    from cuperiod.core.backend import ensure_shared_memory

    smem = 3 * n_bins * n_covers * 8
    kernel = _pdm_kernel(block)
    ensure_shared_memory(kernel, smem, method="PDM", hint="n_bins / n_covers")
    kernel(
        (n_periods,),
        (block,),
        (
            tau_d, y_d, per_d,
            np.int32(tau_d.size), np.int32(n_periods),
            np.int32(n_bins), np.int32(n_covers), np.float64(sigma2), out,
        ),
        shared_mem=smem,
    )
    return np.asarray(cp.asnumpy(out), dtype=np.float64)


# --- CPU fast path: numba-parallel, one loop-iteration per trial period -------

_NUMBA_PDM_KERNEL: Any = None


def _numba_pdm_kernel() -> Any:
    """Lazily compile (once) and cache the numba PDM kernel.

    The same fine-bin search as the CUDA kernel — bin each folded point once into
    ``n_bins*n_covers`` fine bins, regroup covers exactly in the reduction — JIT-run
    across all cores with ``prange`` (each trial period is independent). The phase is
    ``q - floor(q)``, the formula of the vectorized numpy path, so the two CPU
    backends bin identically.
    """
    global _NUMBA_PDM_KERNEL
    if _NUMBA_PDM_KERNEL is not None:
        return _NUMBA_PDM_KERNEL
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=False)  # pragma: no cover - njit
    def _kernel(tau, y, periods, n_bins, n_covers, sigma2):  # type: ignore[no-untyped-def]
        n_periods = periods.shape[0]
        n_points = tau.shape[0]
        m_fine = n_bins * n_covers
        out = np.empty(n_periods)
        for pidx in prange(n_periods):
            period = periods[pidx]
            s_sum = np.zeros(m_fine)
            s_sq = np.zeros(m_fine)
            s_cnt = np.zeros(m_fine)
            for j in range(n_points):
                q = tau[j] / period
                ph = q - np.floor(q)
                f = int(ph * m_fine)
                if f > m_fine - 1:
                    f = m_fine - 1
                elif f < 0:
                    f = 0
                yj = y[j]
                s_sum[f] += yj
                s_sq[f] += yj * yj
                s_cnt[f] += 1.0
            ssd = 0.0
            nonempty = 0
            for c in range(n_covers):
                for b in range(n_bins):
                    cnt = 0.0
                    su = 0.0
                    sq = 0.0
                    f0 = b * n_covers - c
                    if f0 < 0:
                        f0 += m_fine
                    for k in range(n_covers):
                        f = f0 + k
                        if f >= m_fine:
                            f -= m_fine
                        cnt += s_cnt[f]
                        su += s_sum[f]
                        sq += s_sq[f]
                    if cnt > 0.0:
                        ssd += sq - su * su / cnt
                        nonempty += 1
            den = float(n_points * n_covers - nonempty)
            out[pidx] = (ssd / den) / sigma2 if den > 0.0 else np.inf
        return out

    _NUMBA_PDM_KERNEL = _kernel
    return _kernel


def pdm_theta(
    t: FloatArray,
    y: FloatArray,
    periods: FloatArray,
    *,
    n_bins: int = 10,
    n_covers: int = 3,
    backend: str = "numpy",
    batch: int = DEFAULT_BATCH,
    precision: str = "auto",
) -> FloatArray:
    """PDM Theta statistic for each trial period.

    Parameters
    ----------
    t, y : numpy.ndarray
        Finite times (days) and values of one band.
    periods : numpy.ndarray
        Trial periods (days).
    n_bins : int, default 10
        Phase bins per cover.
    n_covers : int, default 3
        Overlapping bin sets, offset by ``1/(n_bins*n_covers)`` in phase.
    backend : str, default "numpy"
        ``"numpy"`` (vectorized CPU), ``"numba"`` (multicore CPU), ``"cupy"``
        (NVIDIA RawKernel), or ``"torch"`` / ``"torch:<device>"`` (portable).
    batch : int, default 2048
        Trial periods per vectorized batch.

    Returns
    -------
    numpy.ndarray
        ``Theta`` for each period (minimized at the true period). Returns all-ones for a
        constant signal (no variance to reduce).
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    periods_host = np.ascontiguousarray(periods, dtype=np.float64)
    if periods_host.size == 0:
        return np.zeros(0, dtype=np.float64)
    n = t.size
    tau = t - t.min()
    y_mean = float(y.mean())
    sigma2 = float(((y - y_mean) ** 2).sum() / (n - 1)) if n > 1 else 0.0
    if sigma2 <= 0.0:
        return np.ones(periods_host.size, dtype=np.float64)

    if backend == "cupy":
        ensure_cuda_dll_path()
        return _pdm_cuda(
            tau, y, periods_host, n_bins=n_bins, n_covers=n_covers, sigma2=sigma2
        )
    if backend == "numba":
        kernel = _numba_pdm_kernel()
        return np.asarray(
            kernel(tau, y, periods_host, n_bins, n_covers, sigma2), dtype=np.float64
        )
    if backend == "torch" or backend.startswith("torch:"):
        import torch

        device = backend.split(":", 1)[1] if ":" in backend else "cpu"
        fdt = (torch.float32
               if resolve_precision(precision, device) == "float32" else torch.float64)
        tau_d = to_device_array(tau, device=device, dtype=fdt)
        y_d = to_device_array(y, device=device, dtype=fdt)
        y2_d = to_device_array(y * y, device=device, dtype=fdt)
        per_d = to_device_array(periods_host, device=device, dtype=fdt)
        return to_host(_theta_batch(
            array_namespace(per_d), tau_d, y_d, y2_d, per_d,
            n_bins=n_bins, n_covers=n_covers, sigma2=sigma2, batch=batch,
        ))
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    theta = _theta_batch(
        array_namespace(periods_host), tau, y, y * y, periods_host,
        n_bins=n_bins, n_covers=n_covers, sigma2=sigma2, batch=batch,
    )
    return np.asarray(theta, dtype=np.float64)


class PDMMethod(PeriodogramMethod):
    """Phase Dispersion Minimization (numba/numpy CPU, cupy GPU)."""

    name: ClassVar[str] = "PDM"
    objective_sense: ClassVar[Literal["max", "min"]] = "min"
    supports_multiband: ClassVar[bool] = False
    settings_cls: ClassVar[type] = PDMSettings
    cpu_backend: ClassVar[str] = "numpy"
    fast_cpu_backend: ClassVar[str | None] = "numba"
    gpu_backend: ClassVar[str | None] = "cupy"
    portable_gpu_backend: ClassVar[str | None] = "torch"
    all_backends: ClassVar[tuple[str, ...]] = ("numba", "numpy", "cupy", "torch")

    def default_grid(self, lc: LightCurve, settings: PDMSettings) -> GridSpec:  # type: ignore[override]
        finite = lc.finite()
        if finite.baseline <= 0.0:
            raise InsufficientDataError("PDM: no usable time baseline")
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
        settings: PDMSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        finite = lc.finite()
        n = finite.n
        if n < settings.min_detections:
            raise InsufficientDataError(
                f"PDM: {n} finite points < min_detections {settings.min_detections}"
            )
        if finite.baseline <= 0.0:
            raise InsufficientDataError("PDM: no usable time baseline")
        periods = grid.period
        if backend == "torch" or backend.startswith("torch:"):
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"
        theta = pdm_theta(
            finite.time,
            finite.value,
            periods,
            n_bins=settings.n_bins,
            n_covers=settings.n_covers,
            backend=backend,
            batch=settings.batch_periods,
            precision=settings.precision,
        )
        return Periodogram.from_spectrum(
            method="PDM",
            backend=backend,
            frequency=1.0 / periods,
            power=theta,
            objective_sense="min",
            n_samples=n,
            baseline=finite.baseline,
            meta=finite.meta,
        )

    def estimate_device_bytes(self, n_points: int) -> int:
        return 128 * 1024**2 + n_points * 8 * 8


register(PDMMethod())

__all__ = ["DEFAULT_BATCH", "PDMBackend", "PDMMethod", "pdm_theta"]
