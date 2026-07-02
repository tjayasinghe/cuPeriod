"""Transit Least Squares (limb-darkened matched-filter transit search).

A from-scratch search in the spirit of TLS (Hippke & Heller 2019): instead of BLS's
rectangular box it correlates the folded light curve with a realistic *limb-darkened*
transit template, which matches real transit/eclipse shapes far better and so improves
the signal detection efficiency (SDE) for shallow signals.

For each trial period the (flux) light curve is phase-folded and binned; the template —
a quadratic limb-darkening profile for a central transit, resampled to each trial
duration — is slid across phase, and at every position the matched-filter depth and its
signal residue ``SR = (sum w g y')^2 / sum w g^2`` are evaluated in one batched
correlation. The per-period maximum SR is normalized across the grid into the SDE, which
is *maximized* at the transit period. Inputs are flux (a transit is a dip); magnitudes
are converted automatically.

This implementation is CPU (numpy) and vectorizes over the period grid; a GPU kernel and
the Ofir (2014) optimal-frequency grid are planned refinements.
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
    to_device_array,
    to_host,
)
from cuperiod.core._typing import FloatArray
from cuperiod.core.backend import ensure_cuda_dll_path
from cuperiod.core.columns import Domain
from cuperiod.core.config import TLSSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, register

TLSBackend = Literal["numpy", "cupy"]

#: Smallest admissible per-bin weight (an empty side is skipped, not divided by).
_W_EPS: Final = 1e-300


def limb_darkened_template(n: int, u1: float, u2: float) -> FloatArray:
    """Normalized central-transit flux-deficit profile over ``n`` samples.

    Quadratic limb darkening ``I(mu) = 1 - u1(1-mu) - u2(1-mu)^2`` sampled along the
    central chord; the profile is 1 at mid-transit and tapers toward the limb. Used as
    the matched-filter template (peak-normalized to 1).

    Parameters
    ----------
    n : int
        Number of template samples (transit duration in phase bins).
    u1, u2 : float
        Quadratic limb-darkening coefficients.

    Returns
    -------
    numpy.ndarray
        The deficit profile of length ``n``.
    """
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    x = (np.arange(n, dtype=np.float64) + 0.5) / n - 0.5  # bin centers in [-0.5, 0.5)
    z = np.clip(2.0 * np.abs(x), 0.0, 1.0)  # central-transit projected separation
    mu = np.sqrt(1.0 - z * z)
    g = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    g = np.where(z < 1.0, g, 0.0)
    peak = float(g.max())
    return g / peak if peak > 0.0 else g


def _duration_bins(settings: TLSSettings) -> list[int]:
    """Transit widths in phase bins (deduped, ascending), clamped to ``[1, n_bins-1]``.

    Clamping (rather than filtering) guarantees at least one valid width even when the
    duration fractions are too small or too large for ``n_phase_bins`` — otherwise the
    search would silently return an all-zero spectrum.
    """
    n_bins = settings.n_phase_bins
    fracs = np.geomspace(
        settings.duration_min_frac, settings.duration_max_frac, settings.n_durations
    )
    widths = {min(max(int(round(f * n_bins)), 1), n_bins - 1) for f in fracs}
    return sorted(widths)


def _period_grid(baseline: float, settings: TLSSettings) -> FloatArray:
    """Ascending trial periods from a frequency grid capped by the baseline."""
    max_period = min(settings.max_period_days, baseline / settings.min_transits)
    if max_period <= settings.min_period_days:
        return np.zeros(0, dtype=np.float64)
    # Spacing tied to the transit width so TLS's narrow peaks are well sampled.
    df = settings.grid_duration_frac / (settings.oversample * baseline)
    freq = np.arange(1.0 / max_period, 1.0 / settings.min_period_days, df)
    if freq.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.ascontiguousarray(1.0 / freq[::-1], dtype=np.float64)


def _matched_filter(
    xp: ModuleType,
    tau: Any,
    yw: Any,
    w: Any,
    periods: Any,
    *,
    n_bins: int,
    dur_bins: list[int],
    templates: dict[int, FloatArray],
    period_batch: int,
) -> dict[str, Any]:
    """Per-period best matched-filter signal residue and transit parameters.

    Array-API generic: ``xp`` is an array_api_compat namespace (numpy on CPU, torch on
    any device). The phase correlation is a short width-loop of shifted slices (the
    template width is small), which needs no ``sliding_window_view``. Templates stay on
    the host; their per-bin scalars broadcast against the device arrays. The cupy
    ``RawKernel`` fast path (NVIDIA) is separate. Float dtype follows ``periods``.
    """
    fdtype = periods.dtype
    idtype = xp.int64
    dev = device_ref(periods)
    n_periods = int(periods.shape[0])
    out = {
        "sr": xp.zeros(n_periods, dtype=fdtype, device=dev),
        "depth": xp.zeros(n_periods, dtype=fdtype, device=dev),
        "duration": xp.zeros(n_periods, dtype=fdtype, device=dev),
        "t0": xp.zeros(n_periods, dtype=fdtype, device=dev),
    }
    for start in range(0, n_periods, period_batch):
        stop = min(start + period_batch, n_periods)
        pb = periods[start:stop]
        n_p = int(pb.shape[0])
        rows_p = xp.arange(n_p, dtype=idtype, device=dev)
        phase = xp.remainder(tau[None, :] / pb[:, None], 1.0)
        bin_idx = xp.clip(xp.astype(phase * n_bins, idtype), 0, n_bins - 1)
        a = xp.zeros((n_p, n_bins), dtype=fdtype, device=dev)  # sum w*y' per bin
        b = xp.zeros((n_p, n_bins), dtype=fdtype, device=dev)  # sum w per bin
        scatter_add_rows(a, bin_idx, yw)
        scatter_add_rows(b, bin_idx, w)
        # One circular pad sized for the widest template serves every width below.
        max_pad = max(dur_bins) - 1
        a_ext = xp.concat([a, a[:, :max_pad]], axis=1) if max_pad else a
        b_ext = xp.concat([b, b[:, :max_pad]], axis=1) if max_pad else b

        best_sr = xp.zeros(n_p, dtype=fdtype, device=dev)
        best_depth = xp.zeros(n_p, dtype=fdtype, device=dev)
        best_start = xp.zeros(n_p, dtype=idtype, device=dev)
        best_width = xp.zeros(n_p, dtype=idtype, device=dev)
        for width in dur_bins:
            g = templates[width]
            num = float(g[0]) * a_ext[:, :n_bins]
            den = float(g[0] * g[0]) * b_ext[:, :n_bins]
            for k in range(1, width):  # correlate the folded data with the template
                gk = float(g[k])
                num += gk * a_ext[:, k : k + n_bins]
                den += (gk * gk) * b_ext[:, k : k + n_bins]
            safe_den = xp.where(den > _W_EPS, den, 1.0)
            # a dip means the in-transit weighted residual is negative -> num < 0
            sr = xp.where((den > _W_EPS) & (num < 0.0), num * num / safe_den, 0.0)
            s_best = xp.argmax(sr, axis=1)
            sr_best = sr[rows_p, s_best]
            improve = sr_best > best_sr
            best_sr = xp.where(improve, sr_best, best_sr)
            den_best = den[rows_p, s_best]
            depth = -num[rows_p, s_best] / xp.where(den_best > _W_EPS, den_best, 1.0)
            best_depth = xp.where(improve, depth, best_depth)
            best_start = xp.where(improve, s_best, best_start)
            best_width = xp.where(improve, width, best_width)

        sl = slice(start, stop)
        out["sr"][sl] = best_sr
        out["depth"][sl] = best_depth
        out["duration"][sl] = xp.astype(best_width, fdtype) / n_bins * pb
        centre = (
            xp.astype(best_start, fdtype) + xp.astype(best_width, fdtype) / 2.0
        ) / n_bins
        out["t0"][sl] = xp.remainder(centre, 1.0) * pb
    return out


#: CUDA threads per block (one block per trial period).
CUDA_BLOCK: Final = 128

#: One-block-per-period TLS kernel: a block folds its points into shared-memory phase
#: bins (weighted residual ``a`` and weight ``b``), then its threads sweep every
#: (duration, phase) box, correlating the limb-darkened template against the bins, and
#: block-reduce to the best signal residue. No (period, point) global intermediates.
_TLS_CUDA_SRC: Final = r"""
extern "C" __global__ void tls_block(
    const REAL* __restrict__ tau, const REAL* __restrict__ yw,
    const REAL* __restrict__ wv, const REAL* __restrict__ periods,
    const int* __restrict__ dur_bins, const int* __restrict__ tmpl_off,
    const REAL* __restrict__ tmpl, const int n_dur, const int tmpl_len,
    const int n_points, const int n_periods, const int n_bins, const REAL W_EPS,
    double* o_sr, double* o_depth, double* o_duration, double* o_t0)
{
    const int pidx = blockIdx.x;
    if (pidx >= n_periods) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const REAL period = periods[pidx];

    extern __shared__ REAL sh[];
    REAL* a = sh;                        // (n_bins) sum w*y' per bin
    REAL* b = sh + n_bins;               // (n_bins) sum w per bin
    REAL* s_tmpl = sh + 2 * n_bins;      // (tmpl_len) all templates, concatenated
    for (int i = tid; i < n_bins; i += nth) { a[i] = 0.0; b[i] = 0.0; }
    for (int i = tid; i < tmpl_len; i += nth) s_tmpl[i] = tmpl[i];
    __syncthreads();

    for (int j = tid; j < n_points; j += nth) {
        REAL x = tau[j];
        REAL ph = (x - period * floor(x / period)) / period;
        int bb = (int)(ph * (REAL)n_bins);
        if (bb >= n_bins) bb = n_bins - 1;
        if (bb < 0) bb = 0;
        atomicAdd(&a[bb], yw[j]);
        atomicAdd(&b[bb], wv[j]);
    }
    __syncthreads();

    REAL loc_sr = 0.0, loc_depth = 0.0;
    int loc_start = 0, loc_width = 0;
    for (int idx = tid; idx < n_dur * n_bins; idx += nth) {
        const int di = idx / n_bins;
        const int start = idx % n_bins;
        const int width = dur_bins[di];
        const int off = tmpl_off[di];
        REAL num = 0.0, den = 0.0;
        for (int k = 0; k < width; ++k) {
            int bb = start + k;
            if (bb >= n_bins) bb -= n_bins;  // circular wrap
            REAL gk = s_tmpl[off + k];
            num += gk * a[bb];
            den += gk * gk * b[bb];
        }
        if (den > W_EPS && num < 0.0) {
            REAL sr = num * num / den;
            if (sr > loc_sr) {
                loc_sr = sr; loc_start = start; loc_width = width;
                loc_depth = -num / den;
            }
        }
    }

    __shared__ REAL r_sr[CUDA_BLOCK];
    __shared__ REAL r_depth[CUDA_BLOCK];
    __shared__ int r_start[CUDA_BLOCK];
    __shared__ int r_width[CUDA_BLOCK];
    r_sr[tid] = loc_sr; r_depth[tid] = loc_depth;
    r_start[tid] = loc_start; r_width[tid] = loc_width;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && r_sr[tid + s] > r_sr[tid]) {
            r_sr[tid] = r_sr[tid + s]; r_depth[tid] = r_depth[tid + s];
            r_start[tid] = r_start[tid + s]; r_width[tid] = r_width[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        o_sr[pidx] = r_sr[0];
        o_depth[pidx] = r_depth[0];
        o_duration[pidx] = (double)r_width[0] / (double)n_bins * period;
        double half = (double)r_width[0] / 2.0;
        double centre = ((double)r_start[0] + half) / (double)n_bins;
        o_t0[pidx] = (centre - floor(centre)) * period;
    }
}
"""

_tls_kernel_cache: dict[tuple[int, str], Any] = {}


def _tls_kernel(block: int, real: str) -> Any:
    """Compile (once) and cache the TLS RawKernel for a block size and precision."""
    kernel = _tls_kernel_cache.get((block, real))
    if kernel is None:
        import cupy

        src = _TLS_CUDA_SRC.replace("CUDA_BLOCK", str(block)).replace(
            "REAL", "float" if real == "float32" else "double"
        )
        kernel = cupy.RawKernel(src, "tls_block")
        _tls_kernel_cache[(block, real)] = kernel
    return kernel


def _tls_cuda(
    tau: FloatArray,
    yw: FloatArray,
    w: FloatArray,
    periods: FloatArray,
    *,
    n_bins: int,
    dur_bins: list[int],
    templates: dict[int, FloatArray],
    precision: str = "float64",
    block: int = CUDA_BLOCK,
) -> dict[str, FloatArray]:
    """One-block-per-period CUDA matched filter; returns host arrays per output."""
    import cupy as cp

    rdtype = np.float32 if precision == "float32" else np.float64
    real_size = 4 if precision == "float32" else 8
    flat = np.concatenate([templates[wd] for wd in dur_bins]).astype(rdtype)
    widths = [len(templates[wd]) for wd in dur_bins]
    offsets = np.zeros(len(dur_bins), dtype=np.int32)
    if len(widths) > 1:
        offsets[1:] = np.cumsum(widths[:-1])
    tau_d = cp.asarray(np.ascontiguousarray(tau, dtype=rdtype))
    yw_d = cp.asarray(np.ascontiguousarray(yw, dtype=rdtype))
    w_d = cp.asarray(np.ascontiguousarray(w, dtype=rdtype))
    per_d = cp.asarray(np.ascontiguousarray(periods, dtype=rdtype))
    dur_d = cp.asarray(np.asarray(dur_bins, dtype=np.int32))
    off_d = cp.asarray(offsets)
    tmpl_d = cp.asarray(flat)
    n_periods = int(per_d.size)
    names = ("sr", "depth", "duration", "t0")
    out = {name: cp.empty(n_periods, dtype=cp.float64) for name in names}
    if n_periods == 0:
        return {name: np.zeros(0, dtype=np.float64) for name in names}
    from cuperiod.core.backend import ensure_shared_memory

    # bins (a, b) plus the concatenated templates all live in shared memory.
    smem = (2 * n_bins + int(flat.size)) * real_size
    kernel = _tls_kernel(block, precision)
    ensure_shared_memory(kernel, smem, method="TLS", hint="n_phase_bins")
    kernel(
        (n_periods,),
        (block,),
        (
            tau_d, yw_d, w_d, per_d, dur_d, off_d, tmpl_d,
            np.int32(len(dur_bins)), np.int32(flat.size),
            np.int32(tau_d.size), np.int32(n_periods),
            np.int32(n_bins),
            # _W_EPS (1e-300) underflows to 0 in float32; use a float32-scale floor.
            rdtype(1e-30) if precision == "float32" else np.float64(_W_EPS),
            out["sr"], out["depth"], out["duration"], out["t0"],
        ),
        shared_mem=smem,
    )
    return {name: np.asarray(cp.asnumpy(out[name]), dtype=np.float64) for name in names}


# --- CPU fast path: numba-parallel, one loop-iteration per trial period -------

_NUMBA_TLS_KERNEL: Any = None


def _numba_tls_kernel() -> Any:
    """Lazily compile (once) and cache the numba TLS kernel.

    The same fold-bin-correlate sweep as the CUDA kernel — bin the weighted residuals
    into phase bins, slide every limb-darkened template over every start phase, keep
    the best signal residue — JIT-run across all cores with ``prange``. The duration
    then start iteration order and strict ``>`` improvement reproduce the vectorized
    path's tie-breaking exactly.
    """
    global _NUMBA_TLS_KERNEL
    if _NUMBA_TLS_KERNEL is not None:
        return _NUMBA_TLS_KERNEL
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=False)  # pragma: no cover - njit
    def _kernel(tau, yw, wv, periods, dur_bins, tmpl_off, tmpl, n_bins, w_eps):  # type: ignore[no-untyped-def]
        n_periods = periods.shape[0]
        n_points = tau.shape[0]
        n_dur = dur_bins.shape[0]
        o_sr = np.zeros(n_periods)
        o_depth = np.zeros(n_periods)
        o_duration = np.zeros(n_periods)
        o_t0 = np.zeros(n_periods)
        for pidx in prange(n_periods):
            period = periods[pidx]
            a = np.zeros(n_bins)
            b = np.zeros(n_bins)
            for j in range(n_points):
                q = tau[j] / period
                ph = q - np.floor(q)
                bb = int(ph * n_bins)
                if bb > n_bins - 1:
                    bb = n_bins - 1
                elif bb < 0:
                    bb = 0
                a[bb] += yw[j]
                b[bb] += wv[j]
            loc_sr = 0.0
            loc_depth = 0.0
            loc_start = 0
            loc_width = 0
            for di in range(n_dur):
                width = dur_bins[di]
                off = tmpl_off[di]
                for start in range(n_bins):
                    num = 0.0
                    den = 0.0
                    for k in range(width):
                        bb = start + k
                        if bb >= n_bins:
                            bb -= n_bins  # circular wrap
                        gk = tmpl[off + k]
                        num += gk * a[bb]
                        den += gk * gk * b[bb]
                    if den > w_eps and num < 0.0:
                        sr = num * num / den
                        if sr > loc_sr:
                            loc_sr = sr
                            loc_start = start
                            loc_width = width
                            loc_depth = -num / den
            o_sr[pidx] = loc_sr
            o_depth[pidx] = loc_depth
            o_duration[pidx] = loc_width / n_bins * period
            centre = (loc_start + loc_width / 2.0) / n_bins
            o_t0[pidx] = (centre - np.floor(centre)) * period
        return o_sr, o_depth, o_duration, o_t0

    _NUMBA_TLS_KERNEL = _kernel
    return _kernel


def _tls_numba(
    tau: FloatArray,
    yw: FloatArray,
    w: FloatArray,
    periods: FloatArray,
    *,
    n_bins: int,
    dur_bins: list[int],
    templates: dict[int, FloatArray],
) -> dict[str, FloatArray]:
    """Numba CPU matched filter; same output dict as :func:`_matched_filter`."""
    kernel = _numba_tls_kernel()
    tmpl = np.concatenate([templates[wd] for wd in dur_bins]).astype(np.float64)
    widths = np.asarray([len(templates[wd]) for wd in dur_bins], dtype=np.int64)
    offsets = np.zeros(len(dur_bins), dtype=np.int64)
    if len(dur_bins) > 1:
        offsets[1:] = np.cumsum(widths[:-1])
    sr, depth, duration, t0 = kernel(
        np.ascontiguousarray(tau), np.ascontiguousarray(yw), np.ascontiguousarray(w),
        np.ascontiguousarray(periods),
        np.asarray(dur_bins, dtype=np.int64), offsets, tmpl,
        int(n_bins), float(_W_EPS),
    )
    return {"sr": sr, "depth": depth, "duration": duration, "t0": t0}


def tls_power(
    t: FloatArray,
    y: FloatArray,
    dy: FloatArray | None,
    periods: FloatArray,
    *,
    settings: TLSSettings,
    backend: str = "numpy",
) -> dict[str, FloatArray]:
    """TLS matched-filter search over ``periods`` (flux input; a transit is a dip).

    Parameters
    ----------
    t, y : numpy.ndarray
        Finite times (days) and flux of one band.
    dy : numpy.ndarray or None
        Flux errors (inverse-variance weights); ``None`` for uniform weights.
    periods : numpy.ndarray
        Trial periods (days).
    settings : TLSSettings
        Search configuration.
    backend : {"numpy", "cupy"}, default "numpy"
        CPU or GPU; both run the identical vectorized matched filter.

    Returns
    -------
    dict of numpy.ndarray
        ``sde`` (per-period detection efficiency) plus ``sr``/``depth``/``duration``/
        ``t0`` aligned with ``periods``.
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    periods_host = np.ascontiguousarray(periods, dtype=np.float64)
    n_periods = int(periods_host.size)
    if n_periods == 0:
        empty = np.zeros(0)
        return {k: empty for k in ("sde", "sr", "depth", "duration", "t0")}

    w = (
        np.ones_like(y)
        if dy is None
        else 1.0 / np.ascontiguousarray(dy, dtype=np.float64) ** 2
    )
    y_mean = float(np.dot(w, y) / w.sum())
    yw = w * (y - y_mean)  # weighted, mean-subtracted
    tau = t - t.min()

    dur_bins = _duration_bins(settings)
    u1, u2 = settings.limb_dark_u1, settings.limb_dark_u2
    templates = {width: limb_darkened_template(width, u1, u2) for width in dur_bins}
    if not dur_bins:
        z = np.zeros(n_periods)
        return {"sde": z, "sr": z, "depth": z, "duration": z, "t0": z}

    n_bins = settings.n_phase_bins
    if backend == "cupy":
        ensure_cuda_dll_path()
        res = _tls_cuda(
            tau, yw, w, periods_host,
            n_bins=n_bins, dur_bins=dur_bins, templates=templates,
            precision=resolve_precision(settings.precision, "cuda"),
        )
    elif backend == "numba":
        res = _tls_numba(
            tau, yw, w, periods_host,
            n_bins=n_bins, dur_bins=dur_bins, templates=templates,
        )
    elif backend == "torch" or backend.startswith("torch:"):
        import torch

        device = backend.split(":", 1)[1] if ":" in backend else "cpu"
        fdt = (torch.float32
               if resolve_precision(settings.precision, device) == "float32"
               else torch.float64)
        per_d = to_device_array(periods_host, device=device, dtype=fdt)
        dev_res = _matched_filter(
            array_namespace(per_d),
            to_device_array(tau, device=device, dtype=fdt),
            to_device_array(yw, device=device, dtype=fdt),
            to_device_array(w, device=device, dtype=fdt),
            per_d,
            n_bins=n_bins, dur_bins=dur_bins, templates=templates,
            period_batch=settings.period_batch,
        )
        res = {k: to_host(v) for k, v in dev_res.items()}
    elif backend == "numpy":
        res = _matched_filter(
            array_namespace(periods_host), tau, yw, w, periods_host,
            n_bins=n_bins, dur_bins=dur_bins, templates=templates,
            period_batch=settings.period_batch,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")

    sr = res["sr"]
    mean, std = float(sr.mean()), float(sr.std())
    res["sde"] = (sr - mean) / std if std > 0.0 else np.zeros_like(sr)
    return res


class TLSMethod(PeriodogramMethod):
    """Transit Least Squares — limb-darkened matched filter (numba CPU, cupy GPU)."""

    name: ClassVar[str] = "TLS"
    objective_sense: ClassVar[Literal["max", "min"]] = "max"
    supports_multiband: ClassVar[bool] = False
    natural_domain: ClassVar[Domain] = Domain.FLUX
    settings_cls: ClassVar[type] = TLSSettings
    cpu_backend: ClassVar[str] = "numpy"
    fast_cpu_backend: ClassVar[str | None] = "numba"
    gpu_backend: ClassVar[str | None] = "cupy"
    portable_gpu_backend: ClassVar[str | None] = "torch"
    all_backends: ClassVar[tuple[str, ...]] = ("numba", "numpy", "cupy", "torch")

    def default_grid(self, lc: LightCurve, settings: TLSSettings) -> GridSpec:  # type: ignore[override]
        finite = lc.finite()
        periods = _period_grid(finite.baseline, settings)
        if periods.size == 0:
            raise InsufficientDataError(
                "TLS: baseline too short for the configured period range"
            )
        return GridSpec(kind="period", values=periods, uniform=False)

    def power(  # type: ignore[override]
        self,
        grid: GridSpec,
        lc: LightCurve,
        settings: TLSSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        finite = lc.finite()
        n = finite.n
        if n < settings.min_detections:
            raise InsufficientDataError(
                f"TLS: {n} finite points < min_detections {settings.min_detections}"
            )
        if finite.baseline <= 0.0:
            raise InsufficientDataError("TLS: no usable time baseline")
        periods = grid.period
        if backend == "torch" or backend.startswith("torch:"):
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"
        res = tls_power(
            finite.time, finite.value, finite.error, periods,
            settings=settings, backend=backend,
        )
        extras = {
            "sr": res["sr"],
            "depth": res["depth"],
            "duration": res["duration"],
            "t0": res["t0"],
        }
        return Periodogram.from_spectrum(
            method="TLS",
            backend=backend,
            frequency=1.0 / periods,
            power=res["sde"],
            objective_sense="max",
            n_samples=n,
            baseline=finite.baseline,
            extras=extras,
            meta=finite.meta,
        )

    def estimate_device_bytes(self, n_points: int) -> int:
        return 128 * 1024**2 + n_points * 8 * 4


register(TLSMethod())

__all__ = ["TLSBackend", "TLSMethod", "limb_darkened_template", "tls_power"]
