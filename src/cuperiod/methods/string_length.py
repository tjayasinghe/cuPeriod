"""String-length period search (Lafler-Kinman 1965 / Dworetsky 1983).

The cheapest useful period statistic: fold the data, sort by phase, and measure the
total length of the "string" joining consecutive points (with wrap-around). At the true
period a smooth folded light curve gives a short string, so this is a *minimization*
method — excellent for eclipsing and eccentric shapes and nearly free to evaluate.

Magnitudes are scaled to a span of 0.5 (Dworetsky's normalization) so the magnitude
and phase axes contribute comparably to the Euclidean step length. One vectorized
kernel runs on numpy (CPU) and cupy (GPU).
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, ClassVar, Final, Literal

import numpy as np

from cuperiod.core._arrayapi import (
    array_namespace,
    device_ref,
    is_torch_array,
    resolve_precision,
    resolve_torch_device,
    to_device_array,
    to_host,
)
from cuperiod.core._typing import FloatArray
from cuperiod.core.backend import ensure_cuda_dll_path
from cuperiod.core.config import StringLengthSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec, default_frequency_grid
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve, raise_if_unusable
from cuperiod.core.result import Periodogram
from cuperiod.methods._fold import (
    check_backend,
    evaluate_trials,
    fold_batch,
    fold_phase,
    foldable_periods,
    usable_curve,
)
from cuperiod.methods.base import PeriodogramMethod, register

SLBackend = Literal["numpy", "cupy"]

#: Trial periods per vectorized batch (bounds the (P, N) sort/gather arrays), lowered
#: on long curves to a transient budget (see
#: :func:`~cuperiod.methods._fold.fold_batch`).
DEFAULT_BATCH: Final = 1024

#: ``(batch, N)`` arrays alive at the peak of :func:`_length_batch` (phase, sort order,
#: the two gathered rows, their differences and squares).
_WORKSPACES: Final = 10

#: Fullest phase bucket the numba kernel's O(N) stable sort accepts; a fold that
#: clumps more points into one 1/N-wide bucket (a trial period far longer than a dense
#: run of points) would make its insertion pass quadratic, so it takes the comparison
#: sort instead.
_SL_BUCKET_FILL: Final = 16


def _length_batch(
    xp: ModuleType, tau: Any, m_scaled: Any, periods: Any, *, batch: int
) -> Any:
    """Total folded string length for each trial period, vectorized over periods.

    Array-API generic (numpy/cupy/torch). Uses only fancy indexing and slicing for the
    per-row sort/gather and the consecutive differences, avoiding ``take_along_axis`` /
    ``diff`` (not in every namespace); float dtype follows ``periods``.

    The phase sort must be **stable** so equal phases keep a backend-independent
    order: the string length depends on neighbour pairing, and an unstable sort breaks
    ties differently across numpy/torch (even across platforms), drifting the result.
    All backends therefore run through the array-API namespace, whose ``argsort``
    defaults to ``stable=True`` — never raw ``numpy``/``cupy`` (quicksort, unstable).
    """
    idtype = xp.int64
    dev = device_ref(periods)
    n_periods = int(periods.shape[0])
    length = xp.empty(n_periods, dtype=periods.dtype, device=dev)
    torch_input = is_torch_array(periods)
    for start in range(0, n_periods, batch):
        stop = min(start + batch, n_periods)
        pb = periods[start:stop]
        n_p = int(pb.shape[0])
        phase = fold_phase(xp, tau, pb)  # (P, N)
        if torch_input:
            # torch.sort returns sorted values and the (stable) order in one kernel,
            # replacing the argsort + row-gather pair.
            import torch

            ph, order = torch.sort(phase, dim=1, stable=True)
        else:
            order = xp.argsort(phase, axis=1)
            rows = xp.arange(n_p, dtype=idtype, device=dev)[:, None]
            ph = phase[rows, order]        # phase sorted per row
        mm = m_scaled[order]               # magnitudes gathered in the same order
        dphi = ph[:, 1:] - ph[:, :-1]
        dmag = mm[:, 1:] - mm[:, :-1]
        total = xp.sum(xp.sqrt(dphi * dphi + dmag * dmag), axis=1)
        wrap_phi = (ph[:, 0] + 1.0) - ph[:, -1]
        wrap_mag = mm[:, 0] - mm[:, -1]
        total = total + xp.sqrt(wrap_phi * wrap_phi + wrap_mag * wrap_mag)
        length[start:stop] = total
    return length


# --- GPU fast path: one CUDA block per trial period ---------------------------

#: CUDA threads per block for the sort kernel (one block per trial period).
CUDA_BLOCK: Final = 256

#: Shared-memory bytes per point in the sort kernel: a float64 phase + int32 index.
_SL_BYTES_PER_POINT: Final = 12

#: One-block-per-period string-length kernel: a block folds its points, bitonic-sorts
#: the (phase, original index) pairs in shared memory — comparing the index on equal
#: phases makes the order *stable*, matching the array-API paths' stable argsort — and
#: accumulates the string length from the sorted neighbours. No (P, N) intermediates
#: and no global sort scratch. The sort pads to a power of two, so it fits curves up to
#: the largest power of two whose padding fits in shared memory (2048 points at 48 KB,
#: 8192 at the 99 KB opt-in of recent GPUs); longer curves take the vectorized path.
_SL_CUDA_SRC: Final = r"""
extern "C" __global__ void sl_block(
    const double* __restrict__ tau, const double* __restrict__ mag,
    const double* __restrict__ periods,
    const int n_points, const int n_pad, const int n_periods,
    double* o_length)
{
    const int pidx = blockIdx.x;
    if (pidx >= n_periods) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const double period = periods[pidx];

    extern __shared__ double sh[];
    double* ph = sh;                    // (n_pad) folded phases
    int* idx = (int*)(sh + n_pad);      // (n_pad) original indices

    for (int i = tid; i < n_pad; i += nth) {
        if (i < n_points) {
            double q = tau[i] / period;
            ph[i] = q - floor(q);       // mod(tau/period, 1), as the CPU paths
            idx[i] = i;
        } else {
            ph[i] = 2.0;                // pad above any phase; sorts to the end
            idx[i] = 0x7fffffff;
        }
    }
    __syncthreads();

    for (int k = 2; k <= n_pad; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < n_pad; i += nth) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);
                    double pa = ph[i], pb = ph[ixj];
                    int ia = idx[i], ib = idx[ixj];
                    bool greater = (pa > pb) || (pa == pb && ia > ib);
                    if (greater == up) {
                        ph[i] = pb; ph[ixj] = pa;
                        idx[i] = ib; idx[ixj] = ia;
                    }
                }
            }
            __syncthreads();
        }
    }

    double total = 0.0;
    for (int i = tid; i < n_points - 1; i += nth) {
        double dphi = ph[i + 1] - ph[i];
        double dmag = mag[idx[i + 1]] - mag[idx[i]];
        total += sqrt(dphi * dphi + dmag * dmag);
    }
    if (tid == 0) {
        double dphi = (ph[0] + 1.0) - ph[n_points - 1];
        double dmag = mag[idx[0]] - mag[idx[n_points - 1]];
        total += sqrt(dphi * dphi + dmag * dmag);
    }
    __shared__ double r_t[CUDA_BLOCK];
    r_t[tid] = total;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) r_t[tid] += r_t[tid + s];
        __syncthreads();
    }
    if (tid == 0) o_length[pidx] = r_t[0];
}
"""

_sl_kernel_cache: dict[int, Any] = {}


def _sl_kernel(block: int) -> Any:
    """Compile (once) and cache the string-length RawKernel for a block size."""
    kernel = _sl_kernel_cache.get(block)
    if kernel is None:
        import cupy

        src = _SL_CUDA_SRC.replace("CUDA_BLOCK", str(block))
        kernel = cupy.RawKernel(src, "sl_block")
        _sl_kernel_cache[block] = kernel
    return kernel


def _sl_kernel_capacity(optin: int, block: int = CUDA_BLOCK) -> int:
    """Longest light curve the sort kernel fits in ``optin`` bytes of shared memory.

    The kernel sorts ``n_pad`` (the next power of two >= ``n``) slots of
    ``_SL_BYTES_PER_POINT`` bytes, beside its static ``block``-double reduction buffer.
    So the capacity is the largest *power of two* whose padding fits: comparing the
    raw ``n`` with ``(optin - static) // 12`` let every length between that power of
    two and the raw quotient (8193..8277 points at 99 KB) through to a launch that
    could not get its shared memory, instead of to the vectorized fallback.
    """
    raw = (int(optin) - block * 8) // _SL_BYTES_PER_POINT
    return 1 << (raw.bit_length() - 1) if raw > 0 else 0


def _sl_cuda_capacity() -> int:
    """Longest light curve the sort kernel can hold in this device's shared memory."""
    import cupy

    optin = int(
        cupy.cuda.Device().attributes.get("MaxSharedMemoryPerBlockOptin", 48 * 1024)
    )
    return _sl_kernel_capacity(optin)


def _sl_cuda(
    tau: FloatArray,
    m_scaled: FloatArray,
    periods: FloatArray,
    *,
    block: int = CUDA_BLOCK,
) -> FloatArray:
    """One-block-per-period CUDA string length; returns a host float64 array."""
    import cupy as cp

    n = int(tau.size)
    n_pad = 1
    while n_pad < n:
        n_pad <<= 1
    tau_d = cp.asarray(np.ascontiguousarray(tau, dtype=np.float64))
    mag_d = cp.asarray(np.ascontiguousarray(m_scaled, dtype=np.float64))
    per_d = cp.asarray(np.ascontiguousarray(periods, dtype=np.float64))
    n_periods = int(per_d.size)
    out = cp.empty(n_periods, dtype=cp.float64)
    from cuperiod.core.backend import ensure_shared_memory

    smem = n_pad * _SL_BYTES_PER_POINT
    kernel = _sl_kernel(block)
    ensure_shared_memory(kernel, smem, method="STRINGLENGTH", hint="n (light curve)")
    kernel(
        (n_periods,),
        (block,),
        (tau_d, mag_d, per_d, np.int32(n), np.int32(n_pad), np.int32(n_periods), out),
        shared_mem=smem,
    )
    return np.asarray(cp.asnumpy(out), dtype=np.float64)


# --- CPU fast path: numba-parallel, one loop-iteration per trial period -------

_NUMBA_SL_KERNEL: Any = None


def _numba_sl_kernel() -> Any:
    """Lazily compile (once) and cache the numba string-length kernel.

    Per period: fold (``q - floor(q)``, the vectorized path's formula), sort the
    phases **stably** (matching the array-API paths' stable sort, so equal phases
    keep the same backend-independent neighbour pairing), and accumulate the string.
    ``prange`` over periods; no ``(P, N)`` transients at all.

    The sort is a counting sort into ``N`` equal phase buckets, filled in original
    index order, followed by one insertion pass that shifts only on a strictly
    greater phase. Folded phases are close to uniform, so that is expected ``O(N)``
    -- the comparison sort was ~90% of the kernel (2.7-7x faster overall). The bucket
    index is monotone in phase and both passes keep equal phases in index order, so
    the permutation is *the* stable order: identical to ``argsort(kind="mergesort")``,
    which still handles a fold with a NaN phase or a bucket fuller than
    :data:`_SL_BUCKET_FILL` (clumped phases would make the insertion pass quadratic).
    The sort is written inline: an ``njit`` helper defined in this factory would be
    recompiled in every process instead of loaded from numba's cache.
    """
    global _NUMBA_SL_KERNEL
    if _NUMBA_SL_KERNEL is not None:
        return _NUMBA_SL_KERNEL
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=False)  # pragma: no cover - njit
    def _kernel(tau, m_scaled, periods):  # type: ignore[no-untyped-def]
        n_periods = periods.shape[0]
        n_points = tau.shape[0]
        out = np.empty(n_periods)
        for pidx in prange(n_periods):
            period = periods[pidx]
            phase = np.empty(n_points)
            for j in range(n_points):
                q = tau[j] / period
                phase[j] = q - np.floor(q)
            # Stable bucket sort (see the factory docstring): count per bucket ...
            order = np.empty(n_points, dtype=np.int64)
            fill = np.zeros(n_points + 1, dtype=np.int64)
            bucketed = True
            for j in range(n_points):
                ph = phase[j]
                if not (ph >= 0.0 and ph <= 1.0):  # NaN: no bucket to put it in
                    bucketed = False
                    break
                b = int(ph * n_points)
                if b > n_points - 1:
                    b = n_points - 1
                fill[b + 1] += 1
            if bucketed:
                fullest = 0
                for b in range(n_points):
                    if fill[b + 1] > fullest:
                        fullest = fill[b + 1]
                    fill[b + 1] += fill[b]  # -> start offset of each bucket
                if fullest > _SL_BUCKET_FILL:
                    bucketed = False
            if bucketed:
                # ... place in ascending original index (stable within a bucket) ...
                for j in range(n_points):
                    b = int(phase[j] * n_points)
                    if b > n_points - 1:
                        b = n_points - 1
                    order[fill[b]] = j
                    fill[b] += 1
                # ... and order each bucket, shifting only past strictly greater.
                for i in range(1, n_points):
                    key = order[i]
                    ph = phase[key]
                    k = i - 1
                    while k >= 0 and phase[order[k]] > ph:
                        order[k + 1] = order[k]
                        k -= 1
                    order[k + 1] = key
            else:
                order[:] = np.argsort(phase, kind="mergesort")
            first = order[0]
            prev_ph = phase[first]
            prev_m = m_scaled[first]
            total = 0.0
            for j in range(1, n_points):
                idx = order[j]
                ph = phase[idx]
                mm = m_scaled[idx]
                dphi = ph - prev_ph
                dmag = mm - prev_m
                total += np.sqrt(dphi * dphi + dmag * dmag)
                prev_ph = ph
                prev_m = mm
            wrap_phi = (phase[first] + 1.0) - prev_ph
            wrap_mag = m_scaled[first] - prev_m
            out[pidx] = total + np.sqrt(wrap_phi * wrap_phi + wrap_mag * wrap_mag)
        return out

    _NUMBA_SL_KERNEL = _kernel
    return _kernel


def string_length(
    t: FloatArray,
    y: FloatArray,
    periods: FloatArray,
    *,
    backend: str = "numpy",
    batch: int = DEFAULT_BATCH,
    precision: str = "auto",
) -> FloatArray:
    """String length for each trial period (minimized at the true period).

    Parameters
    ----------
    t, y : numpy.ndarray
        Finite times (days) and values of one band.
    periods : numpy.ndarray
        Trial periods (days).
    backend : {"numpy", "cupy"}, default "numpy"
        CPU or GPU.
    batch : int, default 1024
        Trial periods per vectorized batch (numpy/torch, and the cupy fallback for
        long curves), lowered on long curves so the batch's transient arrays stay
        within ~128 MiB (512 MiB on a GPU); it never changes the result.

    Returns
    -------
    numpy.ndarray
        String length per period. A trial period that cannot be folded (0, NaN, or
        so small that ``tau / period`` overflows) gives NaN -- on the GPU it no longer
        reaches the bitonic sort, where NaN phases let pad slots into the string.
    """
    check_backend(backend)
    t = np.ascontiguousarray(t, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    periods_host = np.ascontiguousarray(periods, dtype=np.float64)
    if periods_host.size == 0:
        return np.zeros(0, dtype=np.float64)
    tau = t - t.min()
    span = float(y.max() - y.min())
    m_scaled = (y - y.min()) / span * 0.5 - 0.25 if span > 0.0 else np.zeros_like(y)

    def length_for(trials: FloatArray) -> FloatArray:
        return _sl_dispatch(
            tau, m_scaled, trials, backend=backend, batch=batch, precision=precision
        )

    ok = foldable_periods(float(tau.max()), periods_host)
    return evaluate_trials(length_for, periods_host, ok)


def _sl_dispatch(
    tau: FloatArray,
    m_scaled: FloatArray,
    periods_host: FloatArray,
    *,
    backend: str,
    batch: int,
    precision: str,
) -> FloatArray:
    """Run :func:`string_length` on one backend (valid periods only)."""
    n = int(tau.size)
    if backend == "cupy":
        ensure_cuda_dll_path()
        # The in-block sort kernel needs the whole (power-of-two padded) curve in
        # shared memory; longer curves fall back to the vectorized sort-based path.
        if n <= _sl_cuda_capacity():
            return _sl_cuda(tau, m_scaled, periods_host)
        import cupy as cp

        per_cp = cp.asarray(periods_host)
        length = _length_batch(
            array_namespace(per_cp), cp.asarray(tau), cp.asarray(m_scaled), per_cp,
            batch=fold_batch(
                batch, n, workspaces=_WORKSPACES, on_device=True,
                default=DEFAULT_BATCH,
            ),
        )
        return np.asarray(cp.asnumpy(length), dtype=np.float64)
    if backend == "numba":
        kernel = _numba_sl_kernel()
        return np.asarray(kernel(tau, m_scaled, periods_host), dtype=np.float64)
    if backend == "torch" or backend.startswith("torch:"):
        import torch

        device = backend.split(":", 1)[1] if ":" in backend else "cpu"
        fdt = (torch.float32
               if resolve_precision(precision, device) == "float32" else torch.float64)
        tau_d = to_device_array(tau, device=device, dtype=fdt)
        m_d = to_device_array(m_scaled, device=device, dtype=fdt)
        per_d = to_device_array(periods_host, device=device, dtype=fdt)
        chunk = fold_batch(
            batch, n, workspaces=_WORKSPACES, on_device=device != "cpu",
            default=DEFAULT_BATCH,
        )
        return to_host(
            _length_batch(array_namespace(per_d), tau_d, m_d, per_d, batch=chunk)
        )
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    chunk = fold_batch(
        batch, n, workspaces=_WORKSPACES, on_device=False, default=DEFAULT_BATCH
    )
    return np.asarray(
        _length_batch(
            array_namespace(periods_host), tau, m_scaled, periods_host, batch=chunk
        ),
        dtype=np.float64,
    )


class StringLengthMethod(PeriodogramMethod):
    """String-length period search (numba/numpy CPU, cupy GPU)."""

    name: ClassVar[str] = "STRINGLENGTH"
    objective_sense: ClassVar[Literal["max", "min"]] = "min"
    supports_multiband: ClassVar[bool] = True
    settings_cls: ClassVar[type] = StringLengthSettings
    cpu_backend: ClassVar[str] = "numpy"
    fast_cpu_backend: ClassVar[str | None] = "numba"
    gpu_backend: ClassVar[str | None] = "cupy"
    portable_gpu_backend: ClassVar[str | None] = "torch"
    all_backends: ClassVar[tuple[str, ...]] = ("numba", "numpy", "cupy", "torch")

    def default_grid(self, lc: LightCurve, settings: StringLengthSettings) -> GridSpec:  # type: ignore[override]
        finite = usable_curve(lc, "string-length")
        if finite.baseline <= 0.0:
            raise InsufficientDataError("string-length: no usable time baseline")
        # The shared fold-method recipe, size-checked before allocation (see PDM).
        return default_frequency_grid(
            finite.time,
            minimum_frequency=settings.minimum_frequency or None,
            maximum_frequency=settings.maximum_frequency or None,
            samples_per_peak=settings.samples_per_peak,
            nyquist_factor=settings.nyquist_factor,
            context="STRINGLENGTH default grid",
        )

    def power(  # type: ignore[override]
        self,
        grid: GridSpec,
        lc: LightCurve,
        settings: StringLengthSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        finite = usable_curve(lc, "string-length")
        n = finite.n
        if n < settings.min_detections:
            raise InsufficientDataError(
                f"string-length: {n} finite points < min_detections "
                f"{settings.min_detections}"
            )
        if finite.baseline <= 0.0:
            raise InsufficientDataError("string-length: no usable time baseline")
        periods = grid.period
        if backend == "torch" or backend.startswith("torch:"):
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"
        length = string_length(
            finite.time, finite.value, periods,
            backend=backend, batch=settings.batch_periods,
            precision=settings.precision,
        )
        return Periodogram.from_spectrum(
            method="STRINGLENGTH",
            backend=backend,
            frequency=1.0 / periods,
            power=length,
            objective_sense="min",
            n_samples=n,
            baseline=finite.baseline,
            meta=finite.meta,
        )

    def multiband_power(  # type: ignore[override]
        self,
        grid: GridSpec,
        mblc: MultiBandLightCurve,
        settings: StringLengthSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        from cuperiod.multiband.string_length_mb import string_length_multiband

        raise_if_unusable(mblc, "string-length")
        if backend == "torch" or backend.startswith("torch:"):
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"
        return string_length_multiband(grid, mblc, settings, backend)

    def estimate_device_bytes(self, n_points: int) -> int:
        return 128 * 1024**2 + n_points * 8 * 8


register(StringLengthMethod())

__all__ = ["DEFAULT_BATCH", "SLBackend", "StringLengthMethod", "string_length"]
