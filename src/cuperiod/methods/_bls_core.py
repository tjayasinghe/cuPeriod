"""Box Least Squares core: one vectorized search, numpy (CPU) or cupy (GPU).

Computes the same binned box search as astropy ``BoxLeastSquares.power`` — a faithful,
fully vectorized port of astropy's C ``run_bls``: identical phase binning (the ``+1``
offset and the wrap-around pad of ``oversample`` bins), identical cumulative-sum box
sweep, identical ``snr``/``likelihood`` objectives, and identical first-wins tie-break
over (duration, phase). Because the algorithm is the same, results match astropy to
floating-point round-off — accurate by construction, not an approximation.

The win is parallelism: every trial period runs independently, so a single GPU sweeps
hundreds of thousands of periods at once.

* ``numpy`` — CPU; a vectorized reference searching the period grid in batches.
* ``cupy``  — GPU; a one-CUDA-block-per-period kernel that bins into shared memory and
  block-reduces to the best box with no global intermediates. Compiled at runtime by
  cupy/NVRTC, so no host C++ compiler is needed.

Like astropy, the flux is centred on its median before the search (every box statistic
is invariant to a constant offset, and centring keeps the cumulative sums from
carrying the flux level), and an empty or all-swallowing box is recognised by a
*relative* floor on its inverse-variance sum (see :func:`_ivar_floor`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final, Literal

import numpy as np

from cuperiod.core._arrayapi import (
    array_namespace,
    device_ref,
    resolve_precision,
    scatter_add_rows,
    to_device_array,
    to_host,
)
from cuperiod.core._typing import FloatArray

#: Core single-array-library backends (the torch backend is device-qualified, e.g.
#: ``"torch:cpu"``, so backend strings are typed as ``str`` where torch is accepted).
BLSBackend = Literal["numpy", "cupy", "numba"]

_F64_EPS: Final = float(np.finfo(np.float64).eps)
_F32_EPS: Final = float(np.finfo(np.float32).eps)

#: Kept for backward compatibility; the searches use the relative :func:`_ivar_floor`.
_IVAR_EPS: Final = _F64_EPS

#: Default trial periods per vectorized batch; bounds the transient (batch, n_bins)
#: work arrays while amortizing GPU launch overhead.
DEFAULT_BATCH: Final = 2048

#: Byte budgets for one batch of the array-API search's transient work arrays (about
#: ``4*N + 14*width`` float64 values per trial period, see :func:`_resolve_batch`).
#: The host budget keeps the default 2048-period batch for ordinary curves.
_HOST_BATCH_BYTES: Final = 1 << 29
_DEVICE_BATCH_BYTES: Final = 1 << 30

#: CUDA threads per block (one block per trial period).
CUDA_BLOCK: Final = 128

#: Screen slack of the numba sweep, in units of eps * sum|y*ivar| (see the kernel).
_SCREEN_SLACK_EPS: Final = 256.0

#: The numba screen runs only while sum|y*ivar| and sum(ivar) lie in this range, so
#: every intermediate of both passes is a normal float; outside it every box is
#: evaluated exactly, as without the screen.
_SCREEN_SCALE_RANGE: Final = (1e-150, 1e150)


def _ivar_floor(n_points: int, sum_ivar: float, *, float32: bool = False) -> float:
    """Smallest admissible inverse-variance sum on either side of a box.

    A box is skipped when its in-box or out-of-box inverse-variance sum falls below
    this. Both sums come from cumulative sums, so a side holding *no* points is not
    exactly zero but rounding noise of order ``N * eps * sum(ivar)``; an absolute
    floor (astropy uses ``DBL_EPSILON``) admits such a box whenever ``sum(ivar)``
    is large -- a box swallowing every point then reports a fabricated depth of order
    the flux level, a likelihood of ~1e6 on pure noise -- and rejects every box when
    the errors are large (``sum(ivar) < 2e-16``). ``4 * N * eps * sum(ivar)`` bounds
    that round-off and is scale-invariant; the cumulative sums of a float32 device
    pass also need ``8 * eps32 * sum(ivar)``. Never below the smallest normal float,
    so an all-zero weight vector admits nothing.
    """
    floor = 4.0 * n_points * _F64_EPS * sum_ivar
    if float32:
        floor = max(floor, 8.0 * _F32_EPS * sum_ivar)
    if not floor > 0.0:  # also NaN
        return float(np.finfo(np.float64).tiny)
    return floor


def _resolve_batch(
    batch: int, n_points: int, width: int, *, on_device: bool
) -> int:
    """Trial periods per array-API batch: ``batch``, capped by the byte budget.

    One batch materialises the phase-binning temporaries (``(batch, N)`` phases,
    bin indices and weights, about ``4*N`` values per period) and the
    ``(batch, width)`` bin sums, cumulative sums and box-objective arrays (about
    ``14*width``). Without a cap, long light curves or wide period ranges need
    gigabytes per batch. The result does not depend on the batch length.
    """
    per_period = 8 * (4 * max(1, n_points) + 14 * max(1, width))
    budget = _DEVICE_BATCH_BYTES if on_device else _HOST_BATCH_BYTES
    return max(1, min(int(batch), budget // per_period))


@dataclass(frozen=True, slots=True)
class BLSPower:
    """Per-period BLS maxima — the fields astropy's ``power()`` result exposes.

    All arrays are float64 and aligned with the input ``periods`` order. ``power`` is
    the chosen objective (``depth_snr`` for ``"snr"``, ``log_likelihood`` for
    ``"likelihood"``); ``transit_time`` is the mid-transit time in ``[t_min, t_min +
    period)``. Periods with no admissible box come back with ``power = -inf`` and
    zeroed parameters.
    """

    period: FloatArray
    power: FloatArray
    depth: FloatArray
    depth_err: FloatArray
    depth_snr: FloatArray
    duration: FloatArray
    transit_time: FloatArray
    log_likelihood: FloatArray


def _duration_bins(durations: FloatArray, bin_duration: float) -> list[int]:
    """Box widths in bins, deduped in first-seen order (astropy's loop order).

    Rounded half away from zero, like the C ``round()`` of astropy's ``run_bls``
    (Python's ``round`` goes half to even: 2.5 bins became 2, so a trial duration
    could vanish into its neighbour and the search differ from astropy's). The
    ratio is at least ``oversample >= 1``, so ``floor(x + 0.5)`` is exact here.
    """
    seen: set[int] = set()
    out: list[int] = []
    for d in np.asarray(durations, dtype=np.float64).tolist():
        kd = math.floor(d / bin_duration + 0.5)
        if kd >= 1 and kd not in seen:
            seen.add(kd)
            out.append(kd)
    return out


def _objective(
    xp: ModuleType,
    y_in: Any,
    ivar_in: Any,
    sum_y: float,
    ivar_out: Any,
    obj_flag: int,
    valid: Any,
) -> Any:
    """Box objective over all (period, start) cells; invalid cells → ``-inf``.

    ``obj_flag`` 0 = depth signal-to-noise, 1 = log-likelihood. Only dips
    (``y_out >= y_in``) are admissible.
    """
    safe_in = xp.where(valid, ivar_in, 1.0)
    safe_out = xp.where(valid, ivar_out, 1.0)
    y_in_n = y_in / safe_in
    y_out_n = (sum_y - y_in) / safe_out
    depth = y_out_n - y_in_n
    if obj_flag == 0:
        obj = depth / xp.sqrt(1.0 / safe_in + 1.0 / safe_out)
    else:
        obj = 0.5 * ivar_in * depth * depth
    admissible = valid & (depth >= 0.0)
    return xp.where(admissible, obj, -np.inf)


def _bls_search(
    xp: ModuleType,
    t: Any,
    y: Any,
    ivar: Any,
    periods: Any,
    *,
    bin_duration: float,
    dur_bins: list[int],
    oversample: int,
    width: int,
    obj_flag: int,
    batch: int,
    ivar_floor: float | None = None,
) -> dict[str, Any]:
    """Vectorized port of astropy ``run_bls`` for one duration-homogeneous call.

    Array-API generic: ``xp`` is an :mod:`array_api_compat` namespace, so the identical
    body runs on numpy (CPU) and torch (CPU/CUDA/ROCm/MPS/XPU). The working float dtype
    follows ``periods`` (float64, or float32 on a float32 device); index/bin arrays are
    int64. The cupy ``RawKernel`` fast path (NVIDIA) is separate and not this code.
    ``ivar_floor`` defaults to :func:`_ivar_floor` for the working dtype.
    """
    fdtype = periods.dtype
    idtype = xp.int64
    dev = device_ref(periods)
    yw = y * ivar
    sum_y = float(xp.sum(yw))
    sum_ivar = float(xp.sum(ivar))
    if ivar_floor is None:
        ivar_floor = _ivar_floor(
            int(t.shape[0]), sum_ivar, float32=fdtype == xp.float32
        )
    t_min = float(xp.min(t))
    tau = t - t_min

    n_periods = int(periods.shape[0])
    out = {
        "power": xp.full(n_periods, -np.inf, dtype=fdtype, device=dev),
        "depth": xp.zeros(n_periods, dtype=fdtype, device=dev),
        "depth_err": xp.zeros(n_periods, dtype=fdtype, device=dev),
        "depth_snr": xp.zeros(n_periods, dtype=fdtype, device=dev),
        "duration": xp.zeros(n_periods, dtype=fdtype, device=dev),
        "transit_time": xp.zeros(n_periods, dtype=fdtype, device=dev),
        "log_likelihood": xp.zeros(n_periods, dtype=fdtype, device=dev),
    }
    if n_periods == 0 or not dur_bins:
        return out
    cols_full = xp.arange(width, dtype=idtype, device=dev)

    for start in range(0, n_periods, batch):
        stop = min(start + batch, n_periods)
        pb = periods[start:stop]
        n_p = int(pb.shape[0])
        rows = xp.arange(n_p, dtype=idtype, device=dev)
        n_bins = xp.astype(xp.ceil(pb / bin_duration), idtype) + oversample

        phase_t = xp.remainder(tau[None, :], pb[:, None])
        ind = xp.astype(phase_t / bin_duration, idtype) + 1
        ind = xp.clip(ind, 0, width - 1)
        mean_y = xp.zeros((n_p, width), dtype=fdtype, device=dev)
        mean_ivar = xp.zeros((n_p, width), dtype=fdtype, device=dev)
        scatter_add_rows(mean_y, ind, yw)
        scatter_add_rows(mean_ivar, ind, ivar)

        for j in range(oversample):
            dst = xp.clip(n_bins - oversample + j, 0, width - 1)
            mean_y[rows, dst] = mean_y[rows, 1 + j]
            mean_ivar[rows, dst] = mean_ivar[rows, 1 + j]

        cy = xp.cumsum(mean_y, axis=1)
        cw = xp.cumsum(mean_ivar, axis=1)
        del mean_y, mean_ivar

        best_obj = xp.full(n_p, -np.inf, dtype=fdtype, device=dev)
        best_n = xp.zeros(n_p, dtype=idtype, device=dev)
        best_d = xp.zeros(n_p, dtype=idtype, device=dev)
        for kd in dur_bins:
            if kd >= width:
                continue
            y_in = cy[:, kd:] - cy[:, :-kd]
            ivar_in = cw[:, kd:] - cw[:, :-kd]
            ivar_out = sum_ivar - ivar_in
            cols = cols_full[: width - kd]
            valid = (
                (cols[None, :] <= (n_bins[:, None] - kd))
                & (ivar_in >= ivar_floor)
                & (ivar_out >= ivar_floor)
            )
            obj = _objective(xp, y_in, ivar_in, sum_y, ivar_out, obj_flag, valid)
            cand_n = xp.argmax(obj, axis=1)
            cand_obj = obj[rows, cand_n]
            improve = cand_obj > best_obj
            best_obj = xp.where(improve, cand_obj, best_obj)
            best_n = xp.where(improve, xp.astype(cand_n, idtype), best_n)
            best_d = xp.where(improve, kd, best_d)

        finite = xp.isfinite(best_obj)
        y_in = cy[rows, best_n + best_d] - cy[rows, best_n]
        ivar_in = cw[rows, best_n + best_d] - cw[rows, best_n]
        ivar_out = sum_ivar - ivar_in
        safe_in = xp.where(finite & (ivar_in >= ivar_floor), ivar_in, 1.0)
        safe_out = xp.where(finite & (ivar_out >= ivar_floor), ivar_out, 1.0)
        y_in_n = y_in / safe_in
        y_out_n = (sum_y - y_in) / safe_out
        depth = y_out_n - y_in_n
        depth_err = xp.sqrt(1.0 / safe_in + 1.0 / safe_out)
        depth_snr = depth / depth_err
        log_like = 0.5 * ivar_in * depth * depth
        duration = xp.astype(best_d, fdtype) * bin_duration
        transit_time = (
            xp.remainder(xp.astype(best_n, fdtype) * bin_duration + 0.5 * duration, pb)
            + t_min
        )
        power = depth_snr if obj_flag == 0 else log_like

        sl = slice(start, stop)
        zero = xp.zeros(n_p, dtype=fdtype, device=dev)
        out["power"][sl] = xp.where(finite, power, -np.inf)
        out["depth"][sl] = xp.where(finite, depth, zero)
        out["depth_err"][sl] = xp.where(finite, depth_err, zero)
        out["depth_snr"][sl] = xp.where(finite, depth_snr, zero)
        out["duration"][sl] = xp.where(finite, duration, zero)
        out["transit_time"][sl] = xp.where(finite, transit_time, zero)
        out["log_likelihood"][sl] = xp.where(finite, log_like, zero)
    return out


# --- CPU fast path: numba-parallel, one loop-iteration per trial period -------
# The same per-period binned box search as the CUDA kernel below, JIT-compiled
# and run across all CPU cores with numba ``prange`` — each trial period is
# independent, so the search parallelizes perfectly. numba is imported lazily and
# the kernel cached, so this is pure opt-in: nothing imports numba unless the
# ``numba`` backend actually runs. Matches astropy ``run_bls`` to round-off, like
# the numpy and cupy backends, but is the fast CPU product path.
#
# The box sweep is screened. A sparse light curve bins a few hundred points into
# thousands of phase bins, so nearly all the time goes into evaluating boxes (five
# divisions and a square root each) that cannot beat the best box found so far. With
# t = y_in - ybar*iv_in (ybar = sum_y/sum_ivar), the box is a dip iff t <= 0 and
#     snr^2 = sum_ivar * t^2 / (iv_in*iv_out),
#     2*loglike = sum_ivar^2 * t^2 / (iv_in*iv_out^2),
# so a first, division-free pass over each duration stores an optimistic bound
# (|t| + slack)^2 / den on every box, and the second pass evaluates the original
# expressions only where that bound can reach the incumbent's pessimistic bound
# (|t| - slack)^2 / den. The slack (256 eps of sum|y*ivar|, ~30x the worst rounding of
# t and of the exact objective) makes every skip a proof: the chosen box, its
# statistics and the first-wins tie-break are exactly those of the unscreened sweep.
# t and the denominators are normalised by sum|y*ivar| and sum_ivar so neither
# product can over- or underflow; for data so extreme that the normalisation itself
# is not exact (sums outside 1e+-150) the slack is infinite and nothing is skipped.

_NUMBA_BLS_KERNEL: Any = None


def _numba_bls_kernel() -> Any:
    """Lazily compile (once) and cache the numba BLS kernel."""
    global _NUMBA_BLS_KERNEL
    if _NUMBA_BLS_KERNEL is not None:
        return _NUMBA_BLS_KERNEL
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=False)  # pragma: no cover - njit
    def _kernel(tau, yw, wv, periods, dur_bins, bin_duration, oversample, width,  # type: ignore[no-untyped-def]
                sum_y, sum_ivar, t_min, obj_flag, ivar_floor, inv_a, inv_s, slack):
        n_periods = periods.shape[0]
        n_points = tau.shape[0]
        n_dur = dur_bins.shape[0]
        neg = -np.inf
        ybar = sum_y / sum_ivar if sum_ivar > 0.0 else 0.0
        o_power = np.full(n_periods, neg)
        o_depth = np.zeros(n_periods)
        o_depth_err = np.zeros(n_periods)
        o_depth_snr = np.zeros(n_periods)
        o_duration = np.zeros(n_periods)
        o_tt = np.zeros(n_periods)
        o_ll = np.zeros(n_periods)
        for pidx in prange(n_periods):
            period = periods[pidx]
            n_bins = int(np.ceil(period / bin_duration)) + oversample
            if n_bins > width - 1:
                n_bins = width - 1
            # Only bins [0, n_bins] are ever written or read for this period, so
            # zero just that prefix — width is sized for the *longest* period in
            # the batch and can be ~2x larger than n_bins for the shortest.
            my = np.empty(width)
            mi = np.empty(width)
            for i in range(n_bins + 1):
                my[i] = 0.0
                mi[i] = 0.0
            for j in range(n_points):
                x = tau[j]
                w = x - period * np.floor(x / period)
                ind = int(w / bin_duration) + 1
                if ind < 0:
                    ind = 0
                elif ind > width - 1:
                    ind = width - 1
                my[ind] += yw[j]
                mi[ind] += wv[j]
            for k in range(oversample):
                dst = n_bins - oversample + k
                if 0 <= dst < width:
                    my[dst] = my[1 + k]
                    mi[dst] = mi[1 + k]
            ay = 0.0
            ai = 0.0
            for i in range(n_bins + 1):
                ay += my[i]
                my[i] = ay
                ai += mi[i]
                mi[i] = ai
            num = np.empty(width)
            den = np.empty(width)
            loc = neg
            ln = 0
            ld = 0
            best_t2 = 0.0
            best_den = 1.0
            for di in range(n_dur):
                d = dur_bins[di]
                if d < 1 or d >= n_bins:
                    continue
                m = n_bins - d + 1
                # Pass 1: division-free optimistic bound on every box (-1 = never).
                for n in range(m):
                    y_in = my[n + d] - my[n]
                    iv_in = mi[n + d] - mi[n]
                    iv_out = sum_ivar - iv_in
                    t = (y_in - ybar * iv_in) * inv_a
                    t_hi = abs(t) + slack
                    dn = (iv_in * inv_s) * (iv_out * inv_s)
                    if obj_flag != 0:
                        dn = dn * (iv_out * inv_s)
                    ok = (iv_in >= ivar_floor) & (iv_out >= ivar_floor) & (t <= slack)
                    num[n] = t_hi * t_hi if ok else -1.0
                    den[n] = dn if ok else 1.0
                # Pass 2: the original box evaluation, where the bound can win.
                for n in range(m):
                    if num[n] * best_den < best_t2 * den[n]:
                        continue
                    y_in = my[n + d] - my[n]
                    iv_in = mi[n + d] - mi[n]
                    iv_out = sum_ivar - iv_in
                    if iv_in < ivar_floor or iv_out < ivar_floor:
                        continue
                    yin = y_in / iv_in
                    yout = (sum_y - y_in) / iv_out
                    if yout < yin:
                        continue
                    depth = yout - yin
                    if obj_flag == 0:
                        obj = depth / np.sqrt(1.0 / iv_in + 1.0 / iv_out)
                    else:
                        obj = 0.5 * iv_in * depth * depth
                    if obj > loc:
                        loc = obj
                        ln = n
                        ld = d
                        t_lo = abs((y_in - ybar * iv_in) * inv_a) - slack
                        if not (t_lo > 0.0):
                            t_lo = 0.0
                        best_t2 = t_lo * t_lo
                        best_den = den[n]
            if loc <= neg:
                continue
            y_in = my[ln + ld] - my[ln]
            iv_in = mi[ln + ld] - mi[ln]
            iv_out = sum_ivar - iv_in
            yin = y_in / iv_in
            yout = (sum_y - y_in) / iv_out
            depth = yout - yin
            derr = np.sqrt(1.0 / iv_in + 1.0 / iv_out)
            dsnr = depth / derr
            ll = 0.5 * iv_in * depth * depth
            dur = ld * bin_duration
            o_power[pidx] = dsnr if obj_flag == 0 else ll
            o_depth[pidx] = depth
            o_depth_err[pidx] = derr
            o_depth_snr[pidx] = dsnr
            o_duration[pidx] = dur
            o_tt[pidx] = (ln * bin_duration + 0.5 * dur) % period + t_min
            o_ll[pidx] = ll
        return o_power, o_depth, o_depth_err, o_depth_snr, o_duration, o_tt, o_ll

    _NUMBA_BLS_KERNEL = _kernel
    return _kernel


def _bls_search_numba(
    t: Any, y: Any, ivar: Any, periods: Any, *,
    bin_duration: float, dur_bins: list[int], oversample: int, width: int,
    obj_flag: int, screen: bool = True,
) -> dict[str, Any]:
    """Numba CPU box search; same output dict as :func:`_bls_search`.

    ``screen=False`` evaluates every box exactly (the reference the screened sweep
    must reproduce bit for bit; for tests and diagnostics).
    """
    kernel = _numba_bls_kernel()
    yw = np.ascontiguousarray(y * ivar, dtype=np.float64)
    t_min = float(t.min())
    tau = np.ascontiguousarray(t - t_min, dtype=np.float64)
    db = np.ascontiguousarray(dur_bins, dtype=np.int64)
    sum_y = float(yw.sum())
    sum_ivar = float(ivar.sum())
    sum_abs = float(np.abs(yw).sum())
    lo, hi = _SCREEN_SCALE_RANGE
    if screen and lo <= sum_abs <= hi and lo <= sum_ivar <= hi:
        inv_a, inv_s = 1.0 / sum_abs, 1.0 / sum_ivar
        slack = _SCREEN_SLACK_EPS * _F64_EPS
    else:
        inv_a, inv_s, slack = 0.0, 0.0, math.inf
    pw, dep, de, ds, du, tt, ll = kernel(
        tau, yw, np.ascontiguousarray(ivar, dtype=np.float64),
        np.ascontiguousarray(periods, dtype=np.float64), db,
        float(bin_duration), int(oversample), int(width),
        sum_y, sum_ivar, t_min, int(obj_flag),
        _ivar_floor(int(tau.shape[0]), sum_ivar), inv_a, inv_s, slack,
    )
    return {"power": pw, "depth": dep, "depth_err": de, "depth_snr": ds,
            "duration": du, "transit_time": tt, "log_likelihood": ll}


# --- GPU fast path: one CUDA block per trial period --------------------------

_BLS_CUDA_SRC: Final = r"""
extern "C" __global__ void bls_block(
    const REAL* __restrict__ tau,        // (N) times - t_min
    const REAL* __restrict__ yw,         // (N) y * ivar
    const REAL* __restrict__ wv,         // (N) ivar
    const REAL* __restrict__ periods,    // (P)
    const int* __restrict__ dur_bins,    // (D) box widths in bins
    const int n_points, const int n_periods, const int n_dur,
    const REAL bin_duration, const int oversample, const int width,
    const REAL sum_y, const REAL sum_ivar, const double t_min,
    const int obj_flag, const REAL NEG_INF, const REAL IVAR_EPS,
    double* o_power, double* o_depth, double* o_depth_err, double* o_depth_snr,
    double* o_duration, double* o_transit_time, double* o_loglike)
{
    const int pidx = blockIdx.x;
    if (pidx >= n_periods) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const REAL period = periods[pidx];

    int n_bins = (int)ceil(period / bin_duration) + oversample;
    if (n_bins > width - 1) n_bins = width - 1;

    extern __shared__ REAL sh[];
    REAL* my = sh;          // weighted-y per bin
    REAL* mi = sh + width;  // ivar per bin

    for (int i = tid; i < width; i += nth) { my[i] = 0.0; mi[i] = 0.0; }
    __syncthreads();

    for (int j = tid; j < n_points; j += nth) {
        REAL x = tau[j];
        REAL w = x - period * floor(x / period);
        int ind = (int)(w / bin_duration) + 1;
        if (ind < 0) ind = 0;
        if (ind > width - 1) ind = width - 1;
        atomicAdd(&my[ind], yw[j]);
        atomicAdd(&mi[ind], wv[j]);
    }
    __syncthreads();

    if (tid == 0) {
        for (int k = 0; k < oversample; ++k) {
            int dst = n_bins - oversample + k;
            if (dst >= 0 && dst < width) { my[dst] = my[1 + k]; mi[dst] = mi[1 + k]; }
        }
    }
    __syncthreads();

    __shared__ REAL cof_y[CUDA_BLOCK];
    __shared__ REAL cof_i[CUDA_BLOCK];
    {
        int span = n_bins + 1;
        int chunk = (span + nth - 1) / nth;
        int lo = tid * chunk;
        int hi = lo + chunk; if (hi > span) hi = span;
        REAL ay = 0.0, ai = 0.0;
        for (int i = lo; i < hi; ++i) {
            ay += my[i]; my[i] = ay; ai += mi[i]; mi[i] = ai;
        }
        cof_y[tid] = ay; cof_i[tid] = ai;
        __syncthreads();
        if (tid == 0) {
            REAL sy = 0.0, si = 0.0;
            for (int k = 0; k < nth; ++k) {
                REAL ty = cof_y[k], ti = cof_i[k];
                cof_y[k] = sy; cof_i[k] = si; sy += ty; si += ti;
            }
        }
        __syncthreads();
        REAL oy = cof_y[tid], oi = cof_i[tid];
        for (int i = lo; i < hi; ++i) { my[i] += oy; mi[i] += oi; }
    }
    __syncthreads();

    REAL loc_obj = NEG_INF;
    int loc_n = 0, loc_d = 0, loc_di = 0;
    for (int di = 0; di < n_dur; ++di) {
        const int d = dur_bins[di];
        if (d < 1 || d >= n_bins) continue;
        const int n_max = n_bins - d;
        for (int n = tid; n <= n_max; n += nth) {
            REAL y_in = my[n + d] - my[n];
            REAL iv_in = mi[n + d] - mi[n];
            REAL iv_out = sum_ivar - iv_in;
            if (iv_in < IVAR_EPS) continue;
            if (iv_out < IVAR_EPS) continue;
            REAL yin = y_in / iv_in;
            REAL yout = (sum_y - y_in) / iv_out;
            if (yout < yin) continue;
            REAL depth = yout - yin;
            REAL obj = (obj_flag == 0)
                ? depth / sqrt((REAL)1.0 / iv_in + (REAL)1.0 / iv_out)
                : (REAL)0.5 * iv_in * depth * depth;
            if (obj > loc_obj) { loc_obj = obj; loc_n = n; loc_d = d; loc_di = di; }
        }
    }

    // An exact tie (boxes that differ only by empty bins hold the same points, so
    // sparse curves have many) goes to the earliest (duration, start) of the sweep
    // order: astropy's first-wins rule, which the host searches follow too.
    __shared__ REAL r_obj[CUDA_BLOCK];
    __shared__ int r_n[CUDA_BLOCK];
    __shared__ int r_d[CUDA_BLOCK];
    __shared__ int r_di[CUDA_BLOCK];
    r_obj[tid] = loc_obj; r_n[tid] = loc_n; r_d[tid] = loc_d; r_di[tid] = loc_di;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            const REAL other = r_obj[tid + s], mine = r_obj[tid];
            const bool earlier = r_di[tid + s] < r_di[tid]
                || (r_di[tid + s] == r_di[tid] && r_n[tid + s] < r_n[tid]);
            if (other > mine || (other == mine && earlier)) {
                r_obj[tid] = other;
                r_n[tid] = r_n[tid + s];
                r_d[tid] = r_d[tid + s];
                r_di[tid] = r_di[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        REAL best = r_obj[0];
        if (best <= NEG_INF) {
            o_power[pidx] = NEG_INF; o_depth[pidx] = 0.0; o_depth_err[pidx] = 0.0;
            o_depth_snr[pidx] = 0.0; o_duration[pidx] = 0.0;
            o_transit_time[pidx] = 0.0; o_loglike[pidx] = 0.0;
            return;
        }
        int bn = r_n[0], bd = r_d[0];
        double y_in = (double)my[bn + bd] - (double)my[bn];
        double iv_in = (double)mi[bn + bd] - (double)mi[bn];
        double iv_out = (double)sum_ivar - iv_in;
        double yin = y_in / iv_in;
        double yout = ((double)sum_y - y_in) / iv_out;
        double depth = yout - yin;
        double derr = sqrt(1.0 / iv_in + 1.0 / iv_out);
        double dsnr = depth / derr;
        double loglike = 0.5 * iv_in * depth * depth;
        double dur = (double)bd * (double)bin_duration;
        o_power[pidx] = (obj_flag == 0) ? dsnr : loglike;
        o_depth[pidx] = depth;
        o_depth_err[pidx] = derr;
        o_depth_snr[pidx] = dsnr;
        o_duration[pidx] = dur;
        o_transit_time[pidx] =
            fmod((double)bn * (double)bin_duration + 0.5 * dur, (double)period) + t_min;
        o_loglike[pidx] = loglike;
    }
}
"""

_cuda_kernel_cache: dict[tuple[int, str], Any] = {}


def _cuda_kernel(block: int, real: str) -> Any:
    """Compile (once) and cache the BLS RawKernel for a block size and precision."""
    kernel = _cuda_kernel_cache.get((block, real))
    if kernel is None:
        import cupy

        src = _BLS_CUDA_SRC.replace("CUDA_BLOCK", str(block)).replace(
            "REAL", "float" if real == "float32" else "double"
        )
        kernel = cupy.RawKernel(src, "bls_block")
        _cuda_kernel_cache[(block, real)] = kernel
    return kernel


#: The per-period output arrays of every box search, in kernel-argument order.
_OUTPUT_NAMES: Final = (
    "power",
    "depth",
    "depth_err",
    "depth_snr",
    "duration",
    "transit_time",
    "log_likelihood",
)


@dataclass(frozen=True, slots=True)
class _SearchSpec:
    """One duration-homogeneous box search: a period block and its binning."""

    periods: FloatArray
    bin_duration: float
    dur_bins: list[int]
    width: int


def _search_spec(
    periods: FloatArray, durations: FloatArray, oversample: int
) -> _SearchSpec:
    """Binning of one segment: bin width, box widths in bins and bins per period."""
    periods_host = np.ascontiguousarray(periods, dtype=np.float64)
    durations_host = np.ascontiguousarray(durations, dtype=np.float64)
    bin_duration = float(durations_host.min()) / oversample
    dur_bins = _duration_bins(durations_host, bin_duration)
    max_n_bins = int(np.ceil(float(periods_host.max()) / bin_duration)) + oversample
    return _SearchSpec(periods_host, bin_duration, dur_bins, max_n_bins + 1)


def _cuda_curve(
    cp: Any,
    t: FloatArray,
    y: FloatArray,
    ivar: FloatArray,
    rdtype: Any,
    device_cache: dict[str, Any] | None,
) -> dict[str, Any]:
    """The light curve's device arrays and float64 sums, cached per light curve."""
    data: dict[str, Any] | None = None
    if device_cache is not None:
        data = device_cache.get("cuda")
        if data is not None and (data["n"] != t.shape[0] or data["dtype"] != rdtype):
            data = None
    if data is None:
        t_min = float(t.min())
        yw_h = y * ivar
        data = {
            "n": int(t.shape[0]),
            "dtype": rdtype,
            "t_min": t_min,
            "tau": cp.asarray(np.ascontiguousarray(t - t_min, dtype=rdtype)),
            "yw": cp.asarray(np.ascontiguousarray(yw_h, dtype=rdtype)),
            "ivar": cp.asarray(np.ascontiguousarray(ivar, dtype=rdtype)),
            "sum_y": float(yw_h.sum()),
            "sum_ivar": float(ivar.sum()),
        }
        if device_cache is not None:
            device_cache["cuda"] = data
    return data


def _bls_search_cuda_specs(
    t: FloatArray,
    y: FloatArray,
    ivar: FloatArray,
    specs: list[_SearchSpec],
    *,
    oversample: int,
    obj_flag: int,
    precision: str = "float64",
    device_cache: dict[str, Any] | None = None,
    block: int = CUDA_BLOCK,
) -> list[dict[str, Any]]:
    """The CUDA box search of several segments with one upload and one download.

    Every segment's periods and box widths go up in one transfer and every kernel
    writes into its slice of one ``(7, n_periods)`` output that comes back in one
    transfer, so the launches queue back to back. Issued per segment, the transfers
    each waited for the previous kernel (a copy from pageable host memory
    synchronises its stream) and left the device idle while the host prepared the
    next launch. Each segment's results are exactly those of a separate search.
    """
    import cupy as cp

    from cuperiod.core.backend import ensure_shared_memory

    rdtype = np.float32 if precision == "float32" else np.float64
    data = _cuda_curve(cp, t, y, ivar, rdtype, device_cache)
    sizes = [int(spec.periods.shape[0]) for spec in specs]
    total = sum(sizes)
    if total == 0:
        return [_empty_output(0) for _ in specs]
    periods_d = cp.asarray(
        np.concatenate([spec.periods for spec in specs]).astype(rdtype, copy=False)
    )
    dur_all = [np.asarray(spec.dur_bins, dtype=np.int32) for spec in specs]
    dur_d = cp.asarray(np.concatenate(dur_all))
    out = cp.empty((len(_OUTPUT_NAMES), total), dtype=cp.float64)
    real_size = 4 if precision == "float32" else 8
    kernel = _cuda_kernel(block, precision)
    # The relative floor of _ivar_floor: an empty side of a box (or the outside of a
    # box holding every point) is cumulative-sum *noise* of order N*eps*sum_ivar
    # (sum_ivar*eps_f32 in float32), never exactly 0.
    floor = rdtype(
        max(
            _ivar_floor(data["n"], data["sum_ivar"], float32=precision == "float32"),
            float(np.finfo(rdtype).tiny),
        )
    )
    empty: list[int] = []
    p_off = d_off = 0
    for k, (spec, n_periods) in enumerate(zip(specs, sizes, strict=True)):
        n_dur = int(dur_all[k].shape[0])
        if n_periods == 0 or n_dur == 0:
            empty.append(k)
        else:
            smem = 2 * spec.width * real_size
            ensure_shared_memory(
                kernel, smem, method="BLS", hint="the period range (max_period_days)"
            )
            sl = slice(p_off, p_off + n_periods)
            kernel(
                (n_periods,),
                (block,),
                (
                    data["tau"],
                    data["yw"],
                    data["ivar"],
                    periods_d[sl],
                    dur_d[d_off : d_off + n_dur],
                    np.int32(data["n"]),
                    np.int32(n_periods),
                    np.int32(n_dur),
                    rdtype(spec.bin_duration),
                    np.int32(oversample),
                    np.int32(spec.width),
                    rdtype(data["sum_y"]),
                    rdtype(data["sum_ivar"]),
                    np.float64(data["t_min"]),
                    np.int32(obj_flag),
                    rdtype(-np.inf),
                    floor,
                    *(out[i, sl] for i in range(len(_OUTPUT_NAMES))),
                ),
                shared_mem=smem,
            )
        p_off += n_periods
        d_off += n_dur
    host = out.get()
    results: list[dict[str, Any]] = []
    p_off = 0
    for k, n_periods in enumerate(sizes):
        if k in empty:
            results.append(_empty_output(n_periods))
        else:
            sl = slice(p_off, p_off + n_periods)
            results.append(
                {name: host[i, sl] for i, name in enumerate(_OUTPUT_NAMES)}
            )
        p_off += n_periods
    return results


def _empty_output(n_periods: int) -> dict[str, Any]:
    """The output of a search with no admissible box: power -inf, zeros elsewhere."""
    out = {name: np.zeros(n_periods, dtype=np.float64) for name in _OUTPUT_NAMES}
    out["power"][...] = -np.inf
    return out


def _bls_search_cuda(
    t: FloatArray,
    y: FloatArray,
    ivar: FloatArray,
    periods: FloatArray,
    *,
    bin_duration: float,
    dur_bins: list[int],
    oversample: int,
    width: int,
    obj_flag: int,
    precision: str = "float64",
    device_cache: dict[str, Any] | None = None,
    block: int = CUDA_BLOCK,
) -> dict[str, Any]:
    """One-block-per-period CUDA box search; returns host float64 arrays per output.

    ``device_cache`` (caller-owned, one per light curve) keeps the uploaded
    ``tau``/``yw``/``ivar`` device arrays and their float64 sums across calls, so the
    segmented search uploads each light curve once instead of once per segment. The
    time origin is subtracted on the host in float64 before any (possibly float32)
    device cast; the kernel adds ``t_min`` back into ``transit_time`` in double.
    """
    spec = _SearchSpec(
        np.ascontiguousarray(periods, dtype=np.float64), bin_duration, dur_bins, width
    )
    return _bls_search_cuda_specs(
        t, y, ivar, [spec], oversample=oversample, obj_flag=obj_flag,
        precision=precision, device_cache=device_cache, block=block,
    )[0]


def bls_power(
    t: FloatArray,
    y: FloatArray,
    dy: FloatArray,
    periods: FloatArray,
    durations: FloatArray,
    oversample: int,
    *,
    objective: str = "snr",
    backend: str = "numpy",
    batch: int = DEFAULT_BATCH,
    precision: str = "auto",
    device_cache: dict[str, Any] | None = None,
) -> BLSPower:
    """BLS box search over ``periods`` via numpy/torch (portable), cupy, or numba.

    Equivalent to ``BoxLeastSquares(t, y, dy).power(periods, durations,
    objective=objective, oversample=oversample)`` — same binning, objective, median
    centring of ``y`` and half-away-from-zero rounding of the durations to bins. ``y``
    must be in the transit-as-dip convention (a flux decrease). Inverse variances are
    ``1/dy**2``; ``durations`` are absolute (same units as ``t``); ``oversample`` is the
    number of bins per the shortest duration. One deliberate difference: a box whose
    inside or outside holds less than a rounding-level share of the total inverse
    variance (:func:`_ivar_floor`) is never admitted, where astropy's absolute floor
    lets a box that swallows every point report a fabricated depth.

    Parameters
    ----------
    t, y, dy : numpy.ndarray
        Finite times, flux, and 1-sigma errors of one band.
    periods, durations : numpy.ndarray
        Trial periods and box durations (days).
    oversample : int
        Phase bins per shortest duration.
    objective : {"snr", "likelihood"}, default "snr"
        Box objective.
    backend : str, default "numpy"
        ``"numpy"`` (array-API CPU reference), ``"torch"`` / ``"torch:<device>"``
        (portable array-API path, any torch device), ``"cupy"`` (NVIDIA RawKernel), or
        ``"numba"`` (multicore CPU).
    batch : int, default 2048
        Trial periods per vectorized batch (numpy/torch backends), capped so one
        batch's work arrays stay within a byte budget (512 MiB on the host, 1 GiB on
        a device; see :func:`_resolve_batch`). Does not affect the result.
    precision : {"auto", "float64", "float32"}, default "auto"
        Device-side compute precision for the torch and cupy backends (float64
        except on MPS; ``"float32"`` is an opt-in speedup on consumer GPUs).
    device_cache : dict or None
        Caller-owned scratch dict reused across calls that share the *same*
        ``t``/``y``/``dy`` (the segmented search): device uploads happen once per
        light curve instead of once per call. Pass a fresh ``{}`` per light curve.

    Returns
    -------
    BLSPower
        Per-period maxima aligned with ``periods``.
    """
    obj_flag = _objective_flag(objective)
    is_torch = backend == "torch" or backend.startswith("torch:")
    if not is_torch and backend not in ("numpy", "cupy", "numba"):
        raise ValueError(f"unknown backend {backend!r}")

    periods_host = np.ascontiguousarray(periods, dtype=np.float64)
    if periods_host.size == 0:
        return _empty_power()
    spec = _search_spec(periods_host, durations, oversample)
    bin_duration, dur_bins, width = spec.bin_duration, spec.dur_bins, spec.width
    t_host, y_host, ivar_host = _host_inputs(t, y, dy)

    # Time origin subtracted before any (possibly float32) device cast; restored
    # into the absolute ``transit_time`` on the host below. Stays 0.0 for the float64
    # host paths (numpy/numba/cupy), which build the absolute time directly. See the
    # torch branch.
    t_ref = 0.0

    if backend == "cupy":
        from cuperiod.core.backend import ensure_cuda_dll_path

        ensure_cuda_dll_path()
        out = _bls_search_cuda(
            t_host, y_host, ivar_host, periods_host,
            bin_duration=bin_duration, dur_bins=dur_bins,
            oversample=oversample, width=width, obj_flag=obj_flag,
            precision=resolve_precision(precision, "cuda"),
            device_cache=device_cache,
        )
    elif backend == "numba":
        out = _bls_search_numba(
            t_host, y_host, ivar_host, periods_host,
            bin_duration=bin_duration, dur_bins=dur_bins,
            oversample=oversample, width=width, obj_flag=obj_flag,
        )
    elif is_torch:
        import torch

        device = backend.split(":", 1)[1] if ":" in backend else "cpu"
        tdtype = (
            torch.float32
            if resolve_precision(precision, device) == "float32"
            else torch.float64
        )
        # Subtract the time origin in float64 *before* the device cast. Absolute BJDs
        # (~2.458e6) lose all sub-0.25-day timing when cast to float32 — the default
        # precision on Apple MPS — so a raw cast would silently corrupt the phase fold.
        # The small t-min-relative tau survives float32; ``transit_time`` is shifted
        # back to absolute on the host (float64) at the return. Mirrors ``gls._prep``.
        data: dict[str, Any] | None = None
        if device_cache is not None:
            data = device_cache.get("torch")
            if data is not None and (
                data["n"] != t_host.shape[0]
                or data["dtype"] != tdtype
                or data["device"] != device
            ):
                data = None
        if data is None:
            t_ref_val = float(np.min(t_host))
            data = {
                "n": int(t_host.shape[0]),
                "dtype": tdtype,
                "device": device,
                "t_ref": t_ref_val,
                "t": to_device_array(t_host - t_ref_val, device=device, dtype=tdtype),
                "y": to_device_array(y_host, device=device, dtype=tdtype),
                "ivar": to_device_array(ivar_host, device=device, dtype=tdtype),
            }
            if device_cache is not None:
                device_cache["torch"] = data
        t_ref = data["t_ref"]
        periods_d = to_device_array(periods_host, device=device, dtype=tdtype)
        out = _bls_search(
            array_namespace(periods_d), data["t"], data["y"], data["ivar"], periods_d,
            bin_duration=bin_duration, dur_bins=dur_bins,
            oversample=oversample, width=width, obj_flag=obj_flag,
            batch=_resolve_batch(
                batch, int(t_host.shape[0]), width, on_device=device != "cpu"
            ),
        )
    else:  # numpy, through the array-API compat namespace
        out = _bls_search(
            array_namespace(periods_host), t_host, y_host, ivar_host, periods_host,
            bin_duration=bin_duration, dur_bins=dur_bins,
            oversample=oversample, width=width, obj_flag=obj_flag,
            batch=_resolve_batch(batch, int(t_host.shape[0]), width, on_device=False),
        )

    return _to_power(periods_host, out, t_ref)


def _objective_flag(objective: str) -> int:
    """Kernel flag of a box objective: 0 = depth SNR, 1 = log-likelihood."""
    if objective not in ("snr", "likelihood"):
        raise ValueError("objective must be 'snr' or 'likelihood'")
    return 0 if objective == "snr" else 1


def _empty_power() -> BLSPower:
    """The :class:`BLSPower` of an empty period grid."""
    return BLSPower(*(np.empty(0, dtype=np.float64) for _ in range(8)))


def _host_inputs(
    t: FloatArray, y: FloatArray, dy: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(t, y - median(y), 1/dy**2)`` as contiguous float64 host arrays.

    Centred on the median in float64, before any device cast, as astropy does: every
    box statistic is offset-invariant, but uncentred cumulative sums of y*ivar carry
    the whole flux level -- in float32 (forced on Apple MPS) its 6e-8 rounding swamps
    ppm-level transits, and even in float64 it costs parity digits.
    """
    t_host = np.ascontiguousarray(t, dtype=np.float64)
    y_host = np.ascontiguousarray(y, dtype=np.float64)
    if y_host.size:
        y_host = y_host - float(np.median(y_host))
    ivar_host = 1.0 / (np.ascontiguousarray(dy, dtype=np.float64) ** 2)
    return t_host, y_host, ivar_host


def _to_power(periods: FloatArray, out: dict[str, Any], t_ref: float) -> BLSPower:
    """A search's output dict as host float64 :class:`BLSPower` arrays."""
    return BLSPower(
        period=periods,
        power=to_host(out["power"]),
        depth=to_host(out["depth"]),
        depth_err=to_host(out["depth_err"]),
        depth_snr=to_host(out["depth_snr"]),
        duration=to_host(out["duration"]),
        transit_time=to_host(out["transit_time"]) + t_ref,
        log_likelihood=to_host(out["log_likelihood"]),
    )


def bls_power_segments(
    t: FloatArray,
    y: FloatArray,
    dy: FloatArray,
    segments: list[tuple[FloatArray, FloatArray]],
    oversample: int,
    *,
    objective: str = "snr",
    backend: str = "numpy",
    batch: int = DEFAULT_BATCH,
    precision: str = "auto",
    device_cache: dict[str, Any] | None = None,
) -> list[BLSPower]:
    """:func:`bls_power` over the ``(periods, durations)`` segments of one curve.

    Equal to calling :func:`bls_power` once per segment (the same arguments, the
    same results). On ``cupy`` the segments share one upload of their period grids
    and one download of every result, so the kernels run back to back instead of
    each waiting on a transfer; the other backends loop over :func:`bls_power`.
    """
    if backend != "cupy":
        return [
            bls_power(
                t, y, dy, periods, durations, oversample, objective=objective,
                backend=backend, batch=batch, precision=precision,
                device_cache=device_cache,
            )
            for periods, durations in segments
        ]
    from cuperiod.core.backend import ensure_cuda_dll_path

    obj_flag = _objective_flag(objective)
    ensure_cuda_dll_path()
    t_host, y_host, ivar_host = _host_inputs(t, y, dy)
    specs = [
        _search_spec(periods, durations, oversample)
        for periods, durations in segments
        if np.asarray(periods).size
    ]
    outs = iter(
        _bls_search_cuda_specs(
            t_host, y_host, ivar_host, specs, oversample=oversample,
            obj_flag=obj_flag, precision=resolve_precision(precision, "cuda"),
            device_cache=device_cache,
        )
        if specs
        else []
    )
    spec_iter = iter(specs)
    results: list[BLSPower] = []
    for periods, _ in segments:
        if np.asarray(periods).size == 0:
            results.append(_empty_power())
        else:
            results.append(_to_power(next(spec_iter).periods, next(outs), 0.0))
    return results


__all__ = [
    "BLSBackend",
    "BLSPower",
    "DEFAULT_BATCH",
    "bls_power",
    "bls_power_segments",
]
