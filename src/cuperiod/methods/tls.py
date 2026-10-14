"""Transit Least Squares (limb-darkened matched-filter transit search).

A from-scratch search in the spirit of TLS (Hippke & Heller 2019): instead of BLS's
rectangular box it correlates the folded light curve with a realistic *limb-darkened*
transit template, which matches real transit/eclipse shapes far better and so improves
the signal detection efficiency (SDE) for shallow signals.

For each trial period the (flux) light curve is phase-folded and binned; the template —
a quadratic limb-darkening profile for a central transit, resampled to each trial
duration — is slid across phase, and at every position the out-of-transit level and
the transit depth are fitted jointly by weighted least squares, in one batched
correlation. With ``y'`` the flux minus its weighted mean, ``g`` the template (1 at
mid-transit, 0 outside its window) and ``W = sum w``, the fitted depth is
``-(sum w g y') / D`` and the signal residue — the chi-square reduction of the fit —
is ``SR = (sum w g y')^2 / D``, with ``D = sum w g^2 - (sum w g)^2 / W``. (Fitting the
depth alone, ``D = sum w g^2``, would bias it low by about the duty cycle: the mean
that ``y'`` is taken about already contains the transit.) The per-period maximum SR is
normalized across the grid into the SDE, which is *maximized* at the transit period.
Each period also reports the best template's ``duration`` (days) and ``t0``, the
mid-transit time of the first transit at or after the first observation — an absolute
time on the input's time axis, like BLS's — or the first observation where no dip is
found. Inputs are flux (a transit is a dip); magnitudes are converted automatically.

Backends: the array-API matched filter (``numpy`` on the CPU, ``torch`` on any torch
device), a numba-parallel CPU tier and a one-block-per-period cupy ``RawKernel``. The
Ofir (2014) optimal-frequency grid is a planned refinement.
"""

from __future__ import annotations

import math
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
from cuperiod.core.grid import GridSpec, check_grid_size
from cuperiod.core.lightcurve import LightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, register

TLSBackend = Literal["numpy", "cupy"]

#: Smallest admissible per-bin weight (an empty side is skipped, not divided by).
_W_EPS: Final = 1e-300

#: Relative floor on the level-corrected template energy ``D`` (against
#: ``sum w g^2``). Below it the template window holds essentially all the weight at a
#: single template value, so level and depth are degenerate and the position is
#: skipped. The float32 floor sits well above that dtype's cancellation error.
_DEN_REL_EPS: Final = 1e-9
_DEN_REL_EPS_F32: Final = 1e-5

#: A flux series whose full range is within this many ulps of its largest magnitude
#: is constant (see :func:`_is_constant`).
_CONSTANT_ULPS: Final = 64.0

#: Transient byte budgets of one trial-period batch of the array-API path, which
#: materializes about five ``(batch, N)`` 8-byte arrays (phase, bin index, and each
#: scatter's flat index and broadcast weights). ``period_batch`` is capped by them,
#: so the working set no longer grows with the light-curve length. Device backends
#: get a larger budget: fewer, larger batches cut per-batch dispatch overhead.
_BATCH_BYTES: Final = 1 << 27
_DEVICE_BATCH_BYTES: Final = 1 << 28

#: The per-period output arrays of every backend.
_OUTPUTS: Final = ("sde", "sr", "depth", "duration", "t0")


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


def _grid_duration_frac(settings: TLSSettings) -> float:
    """Phase-drift budget of one period-grid step (before ``oversample``).

    ``grid_duration_frac``, but never more than twice the shortest searched duration:
    between two grid periods a transit must not drift across more than about its own
    width over the baseline, or the narrowest transits the duration grid templates
    lose most of their signal residue (57-66% retained midway between samples with
    the former 0.05 step against the default 0.01 shortest duration; 0.90-0.98 now).
    """
    return min(settings.grid_duration_frac, 2.0 * settings.duration_min_frac)


def _period_grid(
    baseline: float, settings: TLSSettings, *, time: FloatArray | None = None
) -> FloatArray:
    """Ascending trial periods from a frequency grid capped by the baseline.

    The size is checked before allocation (:func:`~cuperiod.core.grid.check_grid_size`):
    the step scales with ``1/baseline``, so one mis-converted timestamp can ask for
    billions of trial periods. ``time`` lets the check diagnose such an outlier.
    """
    max_period = min(settings.max_period_days, baseline / settings.min_transits)
    if max_period <= settings.min_period_days:
        return np.zeros(0, dtype=np.float64)
    # Spacing tied to the transit width so TLS's narrow peaks are well sampled.
    df = _grid_duration_frac(settings) / (settings.oversample * baseline)
    f_lo, f_hi = 1.0 / max_period, 1.0 / settings.min_period_days
    check_grid_size(
        math.ceil((f_hi - f_lo) / df),
        automatic=True,
        time=time,
        baseline=baseline,
        maximum_frequency=f_hi,
        context=(
            "TLS default period grid (min_period_days, grid_duration_frac and "
            "oversample set its size)"
        ),
    )
    freq = np.arange(f_lo, f_hi, df)
    if freq.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.ascontiguousarray(1.0 / freq[::-1], dtype=np.float64)


def _resolve_period_batch(
    period_batch: int, n_points: int, n_bins: int, max_width: int, on_device: bool
) -> int:
    """Trial periods per array-API batch: ``period_batch``, capped by a byte budget.

    Besides the ~five ``(batch, N)`` point workspaces, a batch holds about a dozen
    ``(batch, n_bins + max_width)`` bin arrays. The batch does not affect the result
    (every row is computed independently).
    """
    per_row = 8 * (5 * max(n_points, 1) + 12 * (n_bins + max_width))
    budget = _DEVICE_BATCH_BYTES if on_device else _BATCH_BYTES
    return max(1, min(int(period_batch), budget // per_row))


def _is_constant(y: FloatArray) -> bool:
    """Whether the flux is constant to within :data:`_CONSTANT_ULPS` ulps.

    Decided on the data itself, not on ``y - mean``: the weighted mean of a constant
    series is an ulp or two off, so every window looked like a uniform ~1e-15 'dip'
    whose signal residue the SDE normalization blew up to an order-unity spectrum.
    """
    if y.size == 0:
        return True
    scale = float(np.max(np.abs(y)))
    eps = float(np.finfo(np.float64).eps)
    return float(np.ptp(y)) <= _CONSTANT_ULPS * eps * scale


def _flux_scale(y: FloatArray, dy: FloatArray | None) -> float:
    """A power of two near the flux noise scale (``median(dy)``, else ``std(y)``).

    The flux and errors are divided by it before the weights ``1/dy^2`` are formed,
    so fluxes in physical units (1e-15 erg/s/cm^2/A, or 1e12 counts) keep the
    float32 weights and their bin sums in range. A power of two divides exactly, so
    float64 results are unchanged by it.
    """
    s = float(np.std(y)) if dy is None else float(np.median(np.abs(dy)))
    if not (math.isfinite(s) and s > 0.0):
        return 1.0
    return 2.0 ** min(max(round(math.log2(s)), -1000), 1000)


def _no_transit(n_periods: int, t_ref: float) -> dict[str, FloatArray]:
    """The output of a search that finds no dip at any period."""
    out = {key: np.zeros(n_periods, dtype=np.float64) for key in _OUTPUTS}
    out["t0"] = np.full(n_periods, t_ref, dtype=np.float64)
    return out


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
    w_total: float,
    den_rel: float,
    period_batch: int,
) -> dict[str, Any]:
    """Per-period best matched-filter signal residue and transit parameters.

    Array-API generic: ``xp`` is an array_api_compat namespace (numpy on CPU, torch on
    any device). The phase correlation is a short width-loop of shifted slices (the
    template width is small), which needs no ``sliding_window_view``. Templates stay on
    the host; their per-bin scalars broadcast against the device arrays. The cupy
    ``RawKernel`` fast path (NVIDIA) is separate. Float dtype follows ``periods``.
    ``w_total`` is the total weight ``W`` of the level-corrected denominator ``D``;
    ``t0`` is returned relative to ``tau = 0``.
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
            g0 = float(g[0])
            num = g0 * a_ext[:, :n_bins]  # sum w g y'
            cwg = g0 * b_ext[:, :n_bins]  # sum w g: template overlap with the level
            den = float(g[0] * g[0]) * b_ext[:, :n_bins]  # sum w g^2
            for k in range(1, width):  # correlate the folded data with the template
                gk = float(g[k])
                num += gk * a_ext[:, k : k + n_bins]
                cwg += gk * b_ext[:, k : k + n_bins]
                den += (gk * gk) * b_ext[:, k : k + n_bins]
            # Joint level + depth fit: the level absorbs the template's overlap with
            # the mean, D = sum w g^2 - (sum w g)^2 / W.
            den_eff = den - cwg * cwg / w_total
            # a dip means the in-transit weighted residual is negative -> num < 0
            ok = (num < 0.0) & (den > _W_EPS) & (den_eff > den_rel * den)
            safe_den = xp.where(ok, den_eff, 1.0)
            sr = xp.where(ok, num * num / safe_den, 0.0)
            s_best = xp.argmax(sr, axis=1)
            sr_best = sr[rows_p, s_best]
            improve = sr_best > best_sr
            best_sr = xp.where(improve, sr_best, best_sr)
            depth = -num[rows_p, s_best] / safe_den[rows_p, s_best]
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
#: (duration, phase) box, correlating the limb-darkened template against the bins
#: (joint level + depth fit, as in :func:`_matched_filter`), and block-reduce to the
#: best signal residue. No (period, point) global intermediates.
_TLS_CUDA_SRC: Final = r"""
extern "C" __global__ void tls_block(
    const REAL* __restrict__ tau, const REAL* __restrict__ yw,
    const REAL* __restrict__ wv, const REAL* __restrict__ periods,
    const int* __restrict__ dur_bins, const int* __restrict__ tmpl_off,
    const REAL* __restrict__ tmpl, const int n_dur, const int tmpl_len,
    const int n_points, const int n_periods, const int n_bins, const REAL W_EPS,
    const REAL W_TOTAL, const REAL DEN_REL,
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
        REAL num = 0.0, cwg = 0.0, den = 0.0;
        for (int k = 0; k < width; ++k) {
            int bb = start + k;
            if (bb >= n_bins) bb -= n_bins;  // circular wrap
            REAL gk = s_tmpl[off + k];
            num += gk * a[bb];
            cwg += gk * b[bb];
            den += gk * gk * b[bb];
        }
        if (num < 0.0 && den > W_EPS) {
            REAL den_eff = den - cwg * cwg / W_TOTAL;  // joint level + depth fit
            if (den_eff > DEN_REL * den) {
                REAL sr = num * num / den_eff;
                if (sr > loc_sr) {
                    loc_sr = sr; loc_start = start; loc_width = width;
                    loc_depth = -num / den_eff;
                }
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
    w_total: float,
    den_rel: float,
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
            rdtype(w_total), rdtype(den_rel),
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

    The correlation loops run template sample outermost over a circularly padded bin
    array, so the innermost loop is a branch-free elementwise update over all start
    phases that LLVM vectorizes (2-5x faster than one scalar reduction per start). Each
    start still accumulates ``k = 0, 1, ...`` in order, exactly as the array-API path
    does; with ``fastmath=False`` the results are bit-identical to it. The kernel must
    not call other jitted helpers defined in this factory: a closure over dispatchers
    defeats numba's on-disk cache.
    """
    global _NUMBA_TLS_KERNEL
    if _NUMBA_TLS_KERNEL is not None:
        return _NUMBA_TLS_KERNEL
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=False)  # pragma: no cover - njit
    def _kernel(  # type: ignore[no-untyped-def]
        tau, yw, wv, periods, dur_bins, tmpl_off, tmpl, n_bins, w_eps, w_total,
        den_rel,
    ):
        n_periods = periods.shape[0]
        n_points = tau.shape[0]
        n_dur = dur_bins.shape[0]
        max_width = 1
        for di in range(n_dur):
            if dur_bins[di] > max_width:
                max_width = dur_bins[di]
        n_ext = n_bins + max_width - 1  # one circular pad serves every width
        o_sr = np.zeros(n_periods)
        o_depth = np.zeros(n_periods)
        o_duration = np.zeros(n_periods)
        o_t0 = np.zeros(n_periods)
        for pidx in prange(n_periods):
            period = periods[pidx]
            a = np.zeros(n_ext)  # sum w*y' per bin
            b = np.zeros(n_ext)  # sum w per bin
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
            for j in range(max_width - 1):
                a[n_bins + j] = a[j]
                b[n_bins + j] = b[j]
            num = np.empty(n_bins)  # sum w g y'
            cwg = np.empty(n_bins)  # sum w g
            den = np.empty(n_bins)  # sum w g^2
            loc_sr = 0.0
            loc_depth = 0.0
            loc_start = 0
            loc_width = 0
            for di in range(n_dur):
                width = dur_bins[di]
                off = tmpl_off[di]
                g0 = tmpl[off]
                g00 = g0 * g0
                for s in range(n_bins):
                    num[s] = g0 * a[s]
                    cwg[s] = g0 * b[s]
                    den[s] = g00 * b[s]
                for k in range(1, width):
                    gk = tmpl[off + k]
                    gkk = gk * gk
                    for s in range(n_bins):
                        num[s] += gk * a[k + s]
                        cwg[s] += gk * b[k + s]
                        den[s] += gkk * b[k + s]
                for s in range(n_bins):
                    nm = num[s]
                    dn = den[s]
                    if nm < 0.0 and dn > w_eps:
                        # Joint level + depth fit: D = sum w g^2 - (sum w g)^2 / W.
                        de = dn - cwg[s] * cwg[s] / w_total
                        if de > den_rel * dn:
                            sr = nm * nm / de
                            if sr > loc_sr:
                                loc_sr = sr
                                loc_start = s
                                loc_width = width
                                loc_depth = -nm / de
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
    w_total: float,
    den_rel: float,
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
        int(n_bins), float(_W_EPS), float(w_total), float(den_rel),
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
    backend : str, default "numpy"
        ``"numpy"``, ``"numba"``, ``"cupy"``, or ``"torch"`` / ``"torch:<device>"``;
        all run the same matched filter.

    Returns
    -------
    dict of numpy.ndarray
        float64 arrays aligned with ``periods``: ``sde`` (signal detection
        efficiency), ``sr`` (signal residue: the chi-square reduction of the best
        level + transit fit; in flux units squared when ``dy`` is None), ``depth``
        (mid-transit depth in flux units), ``duration`` (days) and ``t0``, the
        mid-transit time of the first transit at or after ``t.min()`` -- an absolute
        time on ``t``'s axis, in ``[t.min(), t.min() + period)``. Where no dip is
        found (``sr == 0``) ``depth`` and ``duration`` are 0 and ``t0`` is
        ``t.min()``; a constant flux finds none at any period.
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    periods_host = np.ascontiguousarray(periods, dtype=np.float64)
    n_periods = int(periods_host.size)
    if n_periods == 0:
        return {key: np.zeros(0, dtype=np.float64) for key in _OUTPUTS}
    t_ref = float(t.min())
    if _is_constant(y):
        return _no_transit(n_periods, t_ref)

    # Work in units of a power of two near the flux noise: exact in float64, and it
    # keeps float32 weights 1/dy^2 in range for fluxes in physical units.
    scale = _flux_scale(y, dy)
    y = y / scale
    w = (
        np.ones_like(y)
        if dy is None
        else 1.0 / (np.ascontiguousarray(dy, dtype=np.float64) / scale) ** 2
    )
    w_total = float(w.sum())
    y_mean = float(np.dot(w, y) / w_total)
    yw = w * (y - y_mean)  # weighted, mean-subtracted
    tau = t - t_ref

    dur_bins = _duration_bins(settings)
    u1, u2 = settings.limb_dark_u1, settings.limb_dark_u2
    templates = {width: limb_darkened_template(width, u1, u2) for width in dur_bins}
    if not dur_bins:
        return _no_transit(n_periods, t_ref)

    n_bins = settings.n_phase_bins
    max_width = max(dur_bins)
    if backend == "cupy":
        ensure_cuda_dll_path()
        precision = resolve_precision(settings.precision, "cuda")
        res = _tls_cuda(
            tau, yw, w, periods_host,
            n_bins=n_bins, dur_bins=dur_bins, templates=templates,
            w_total=w_total,
            den_rel=_DEN_REL_EPS_F32 if precision == "float32" else _DEN_REL_EPS,
            precision=precision,
        )
    elif backend == "numba":
        res = _tls_numba(
            tau, yw, w, periods_host,
            n_bins=n_bins, dur_bins=dur_bins, templates=templates,
            w_total=w_total, den_rel=_DEN_REL_EPS,
        )
    elif backend == "torch" or backend.startswith("torch:"):
        import torch

        device = backend.split(":", 1)[1] if ":" in backend else "cpu"
        single = resolve_precision(settings.precision, device) == "float32"
        fdt = torch.float32 if single else torch.float64
        per_d = to_device_array(periods_host, device=device, dtype=fdt)
        dev_res = _matched_filter(
            array_namespace(per_d),
            to_device_array(tau, device=device, dtype=fdt),
            to_device_array(yw, device=device, dtype=fdt),
            to_device_array(w, device=device, dtype=fdt),
            per_d,
            n_bins=n_bins, dur_bins=dur_bins, templates=templates,
            w_total=w_total,
            den_rel=_DEN_REL_EPS_F32 if single else _DEN_REL_EPS,
            period_batch=_resolve_period_batch(
                settings.period_batch, tau.size, n_bins, max_width,
                on_device=device != "cpu",
            ),
        )
        res = {k: to_host(v) for k, v in dev_res.items()}
    elif backend == "numpy":
        res = _matched_filter(
            array_namespace(periods_host), tau, yw, w, periods_host,
            n_bins=n_bins, dur_bins=dur_bins, templates=templates,
            w_total=w_total, den_rel=_DEN_REL_EPS,
            period_batch=_resolve_period_batch(
                settings.period_batch, tau.size, n_bins, max_width, on_device=False
            ),
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")

    # float64 on the host from here on: a JD-scale epoch does not fit float32.
    res = {key: np.asarray(value, dtype=np.float64) for key, value in res.items()}
    sr = res["sr"]
    mean, std = float(sr.mean()), float(sr.std())
    res["sde"] = (sr - mean) / std if std > 0.0 else np.zeros_like(sr)
    # Back to the caller's flux units (the SDE is scale-free); SR carries the flux
    # unit squared only without errors, where the weights do not scale with it.
    res["depth"] = res["depth"] * scale
    if dy is None:
        res["sr"] = sr * (scale * scale)
    # The kernels fold tau = t - t.min(); report the epoch on the input time axis.
    res["t0"] = res["t0"] + t_ref
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
        periods = _period_grid(finite.baseline, settings, time=finite.time)
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
        # The torch path's batch workspaces are capped at _DEVICE_BATCH_BYTES.
        return _DEVICE_BATCH_BYTES + 64 * 1024**2 + n_points * 8 * 4


register(TLSMethod())

__all__ = ["TLSBackend", "TLSMethod", "limb_darkened_template", "tls_power"]
