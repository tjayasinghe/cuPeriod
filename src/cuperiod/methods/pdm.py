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
    of squared deviations across all covers, and ``Theta = s^2 / sigma^2``.
    """
    n_points = int(tau.shape[0])
    n_periods = int(periods.shape[0])
    n_global = n_bins * n_covers
    cover_step = 1.0 / (n_bins * n_covers)
    theta = xp.empty(n_periods, dtype=np.float64)

    for start in range(0, n_periods, batch):
        stop = min(start + batch, n_periods)
        pb = periods[start:stop]
        n_p = int(pb.shape[0])
        rows = xp.arange(n_p)
        phase = xp.mod(tau[None, :] / pb[:, None], 1.0)  # (P, N) in [0, 1)

        count = xp.zeros(n_p * n_global, dtype=np.float64)
        ysum = xp.zeros(n_p * n_global, dtype=np.float64)
        ysq = xp.zeros(n_p * n_global, dtype=np.float64)
        ones = xp.broadcast_to(xp.ones(1), (n_p, n_points)).ravel()
        y_b = xp.broadcast_to(y, (n_p, n_points)).ravel()
        y2_b = xp.broadcast_to(y2, (n_p, n_points)).ravel()
        for cover in range(n_covers):
            offset = cover * cover_step
            b = (xp.mod(phase + offset, 1.0) * n_bins).astype(np.int64)
            xp.clip(b, 0, n_bins - 1, out=b)
            flat = (rows[:, None] * n_global + (b + cover * n_bins)).ravel()
            xp.add.at(count, flat, ones)
            xp.add.at(ysum, flat, y_b)
            xp.add.at(ysq, flat, y2_b)

        count = count.reshape(n_p, n_global)
        ysum = ysum.reshape(n_p, n_global)
        ysq = ysq.reshape(n_p, n_global)
        safe = xp.where(count > 0.0, count, 1.0)
        ssd = xp.where(count > 0.0, ysq - ysum * ysum / safe, 0.0)
        nonempty = (count > 0.0).sum(axis=1)
        den = float(n_points * n_covers) - nonempty
        safe_den = xp.where(den > 0.0, den, 1.0)
        s2 = xp.where(den > 0.0, ssd.sum(axis=1) / safe_den, xp.inf)
        theta[start:stop] = s2 / sigma2
    return theta


#: CUDA threads per block (one block per trial period).
CUDA_BLOCK: Final = 128

#: One-block-per-period PDM kernel: a block bins its folded points in shared memory
#: (sum, sum of squares, count per cover) and reduces them to Theta on one thread.
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
    double* s_sum = sh;          // (M) sum of y per bin
    double* s_sq = sh + M;       // (M) sum of y^2 per bin
    double* s_cnt = sh + 2 * M;  // (M) point count per bin
    for (int i = tid; i < M; i += nth) {
        s_sum[i] = 0.0; s_sq[i] = 0.0; s_cnt[i] = 0.0;
    }
    __syncthreads();

    const double cover_step = 1.0 / ((double)n_bins * (double)n_covers);
    for (int j = tid; j < n_points; j += nth) {
        double x = tau[j];
        double ph = (x - period * floor(x / period)) / period;  // mod(tau/period, 1)
        double yj = y[j];
        for (int c = 0; c < n_covers; ++c) {
            double pc = ph + (double)c * cover_step;
            pc -= floor(pc);
            int b = (int)(pc * n_bins);
            if (b >= n_bins) b = n_bins - 1;
            if (b < 0) b = 0;
            int gid = c * n_bins + b;
            atomicAdd(&s_sum[gid], yj);
            atomicAdd(&s_sq[gid], yj * yj);
            atomicAdd(&s_cnt[gid], 1.0);
        }
    }
    __syncthreads();

    if (tid == 0) {
        double ssd = 0.0;
        int nonempty = 0;
        for (int i = 0; i < M; ++i) {
            double cnt = s_cnt[i];
            if (cnt > 0.0) {
                ssd += s_sq[i] - s_sum[i] * s_sum[i] / cnt;
                nonempty += 1;
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
    _pdm_kernel(block)(
        (n_periods,),
        (block,),
        (
            tau_d, y_d, per_d,
            np.int32(tau_d.size), np.int32(n_periods),
            np.int32(n_bins), np.int32(n_covers), np.float64(sigma2), out,
        ),
        shared_mem=3 * n_bins * n_covers * 8,
    )
    return np.asarray(cp.asnumpy(out), dtype=np.float64)


def pdm_theta(
    t: FloatArray,
    y: FloatArray,
    periods: FloatArray,
    *,
    n_bins: int = 10,
    n_covers: int = 3,
    backend: PDMBackend = "numpy",
    batch: int = DEFAULT_BATCH,
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
    backend : {"numpy", "cupy"}, default "numpy"
        CPU or GPU.
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
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    theta = _theta_batch(
        np, tau, y, y * y, periods_host,
        n_bins=n_bins, n_covers=n_covers, sigma2=sigma2, batch=batch,
    )
    return np.asarray(theta, dtype=np.float64)


class PDMMethod(PeriodogramMethod):
    """Phase Dispersion Minimization (numpy CPU, cupy GPU)."""

    name: ClassVar[str] = "PDM"
    objective_sense: ClassVar[Literal["max", "min"]] = "min"
    supports_multiband: ClassVar[bool] = False
    settings_cls: ClassVar[type] = PDMSettings
    cpu_backend: ClassVar[str] = "numpy"
    gpu_backend: ClassVar[str | None] = "cupy"
    all_backends: ClassVar[tuple[str, ...]] = ("numpy", "cupy")

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
        theta = pdm_theta(
            finite.time,
            finite.value,
            periods,
            n_bins=settings.n_bins,
            n_covers=settings.n_covers,
            backend=backend,  # type: ignore[arg-type]
            batch=settings.batch_periods,
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
