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
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final, Literal

import numpy as np

from cuperiod.core._typing import FloatArray

BLSBackend = Literal["numpy", "cupy", "numba"]

#: Inverse-variance floor: an empty in/out side (ivar sum 0) is skipped, not divided by.
_IVAR_EPS: Final = float(np.finfo(np.float64).eps)

#: Default trial periods per vectorized batch; bounds the transient (batch, n_bins)
#: work arrays while amortizing GPU launch overhead.
DEFAULT_BATCH: Final = 2048

#: CUDA threads per block (one block per trial period).
CUDA_BLOCK: Final = 128


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


def _scatter_add(xp: ModuleType, target: Any, idx: Any, values: Any) -> None:
    """``target[idx] += values`` with index repeats accumulated, on numpy/cupy."""
    xp.add.at(target, idx, values)


def _duration_bins(durations: FloatArray, bin_duration: float) -> list[int]:
    """Box widths in bins, deduped in first-seen order (astropy's loop order)."""
    seen: set[int] = set()
    out: list[int] = []
    for d in np.asarray(durations, dtype=np.float64).tolist():
        kd = round(d / bin_duration)
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
) -> dict[str, Any]:
    """Vectorized port of astropy ``run_bls`` for one duration-homogeneous call."""
    n_points = int(t.shape[0])
    yw = y * ivar
    sum_y = float(yw.sum())
    sum_ivar = float(ivar.sum())
    t_min = float(t.min())
    tau = t - t_min

    n_periods = int(periods.shape[0])
    out = {
        "power": xp.full(n_periods, -np.inf, dtype=np.float64),
        "depth": xp.zeros(n_periods, dtype=np.float64),
        "depth_err": xp.zeros(n_periods, dtype=np.float64),
        "depth_snr": xp.zeros(n_periods, dtype=np.float64),
        "duration": xp.zeros(n_periods, dtype=np.float64),
        "transit_time": xp.zeros(n_periods, dtype=np.float64),
        "log_likelihood": xp.zeros(n_periods, dtype=np.float64),
    }
    if n_periods == 0 or not dur_bins:
        return out

    for start in range(0, n_periods, batch):
        stop = min(start + batch, n_periods)
        pb = periods[start:stop]
        n_p = int(pb.shape[0])
        rows = xp.arange(n_p)
        n_bins = xp.ceil(pb / bin_duration).astype(np.int64) + oversample

        phase_t = xp.mod(tau[None, :], pb[:, None])
        ind = (phase_t / bin_duration).astype(np.int64) + 1
        xp.clip(ind, 0, width - 1, out=ind)
        flat = (rows[:, None] * width + ind).ravel()
        mean_y = xp.zeros(n_p * width, dtype=np.float64)
        mean_ivar = xp.zeros(n_p * width, dtype=np.float64)
        yw_b = xp.broadcast_to(yw, (n_p, n_points)).ravel()
        ivar_b = xp.broadcast_to(ivar, (n_p, n_points)).ravel()
        _scatter_add(xp, mean_y, flat, yw_b)
        _scatter_add(xp, mean_ivar, flat, ivar_b)
        mean_y = mean_y.reshape(n_p, width)
        mean_ivar = mean_ivar.reshape(n_p, width)

        for j in range(oversample):
            dst = xp.clip(n_bins - oversample + j, 0, width - 1)
            mean_y[rows, dst] = mean_y[rows, 1 + j]
            mean_ivar[rows, dst] = mean_ivar[rows, 1 + j]

        cy = xp.cumsum(mean_y, axis=1)
        cw = xp.cumsum(mean_ivar, axis=1)
        del mean_y, mean_ivar

        best_obj = xp.full(n_p, -np.inf, dtype=np.float64)
        best_n = xp.zeros(n_p, dtype=np.int64)
        best_d = xp.zeros(n_p, dtype=np.int64)
        for kd in dur_bins:
            if kd >= width:
                continue
            y_in = cy[:, kd:] - cy[:, :-kd]
            ivar_in = cw[:, kd:] - cw[:, :-kd]
            ivar_out = sum_ivar - ivar_in
            cols = xp.arange(y_in.shape[1])
            valid = (
                (cols[None, :] <= (n_bins[:, None] - kd))
                & (ivar_in >= _IVAR_EPS)
                & (ivar_out >= _IVAR_EPS)
            )
            obj = _objective(xp, y_in, ivar_in, sum_y, ivar_out, obj_flag, valid)
            cand_n = xp.argmax(obj, axis=1)
            cand_obj = obj[rows, cand_n]
            improve = cand_obj > best_obj
            best_obj = xp.where(improve, cand_obj, best_obj)
            best_n = xp.where(improve, cand_n.astype(np.int64), best_n)
            best_d = xp.where(improve, np.int64(kd), best_d)

        finite = xp.isfinite(best_obj)
        y_in = cy[rows, best_n + best_d] - cy[rows, best_n]
        ivar_in = cw[rows, best_n + best_d] - cw[rows, best_n]
        ivar_out = sum_ivar - ivar_in
        safe_in = xp.where(finite & (ivar_in >= _IVAR_EPS), ivar_in, 1.0)
        safe_out = xp.where(finite & (ivar_out >= _IVAR_EPS), ivar_out, 1.0)
        y_in_n = y_in / safe_in
        y_out_n = (sum_y - y_in) / safe_out
        depth = y_out_n - y_in_n
        depth_err = xp.sqrt(1.0 / safe_in + 1.0 / safe_out)
        depth_snr = depth / depth_err
        log_like = 0.5 * ivar_in * depth * depth
        duration = best_d.astype(np.float64) * bin_duration
        transit_time = (
            xp.mod(best_n.astype(np.float64) * bin_duration + 0.5 * duration, pb)
            + t_min
        )
        power = depth_snr if obj_flag == 0 else log_like

        sl = slice(start, stop)
        zero = xp.zeros(n_p, dtype=np.float64)
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

_NUMBA_BLS_KERNEL: Any = None


def _numba_bls_kernel() -> Any:
    """Lazily compile (once) and cache the numba BLS kernel."""
    global _NUMBA_BLS_KERNEL
    if _NUMBA_BLS_KERNEL is not None:
        return _NUMBA_BLS_KERNEL
    from numba import njit, prange

    eps = _IVAR_EPS

    @njit(parallel=True, cache=True, fastmath=False)  # pragma: no cover - njit
    def _kernel(tau, yw, wv, periods, dur_bins, bin_duration, oversample, width,  # type: ignore[no-untyped-def]
                sum_y, sum_ivar, t_min, obj_flag):
        n_periods = periods.shape[0]
        n_points = tau.shape[0]
        n_dur = dur_bins.shape[0]
        neg = -np.inf
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
            my = np.zeros(width)
            mi = np.zeros(width)
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
            loc = neg
            ln = 0
            ld = 0
            for di in range(n_dur):
                d = dur_bins[di]
                if d < 1 or d >= n_bins:
                    continue
                for n in range(n_bins - d + 1):
                    y_in = my[n + d] - my[n]
                    iv_in = mi[n + d] - mi[n]
                    iv_out = sum_ivar - iv_in
                    if iv_in < eps or iv_out < eps:
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
    obj_flag: int,
) -> dict[str, Any]:
    """Numba CPU box search; same output dict as :func:`_bls_search`."""
    kernel = _numba_bls_kernel()
    yw = np.ascontiguousarray(y * ivar, dtype=np.float64)
    t_min = float(t.min())
    tau = np.ascontiguousarray(t - t_min, dtype=np.float64)
    db = np.ascontiguousarray(dur_bins, dtype=np.int64)
    pw, dep, de, ds, du, tt, ll = kernel(
        tau, yw, np.ascontiguousarray(ivar, dtype=np.float64),
        np.ascontiguousarray(periods, dtype=np.float64), db,
        float(bin_duration), int(oversample), int(width),
        float(yw.sum()), float(ivar.sum()), t_min, int(obj_flag),
    )
    return {"power": pw, "depth": dep, "depth_err": de, "depth_snr": ds,
            "duration": du, "transit_time": tt, "log_likelihood": ll}


# --- GPU fast path: one CUDA block per trial period --------------------------

_BLS_CUDA_SRC: Final = r"""
extern "C" __global__ void bls_block(
    const double* __restrict__ tau,      // (N) times - t_min
    const double* __restrict__ yw,       // (N) y * ivar
    const double* __restrict__ wv,       // (N) ivar
    const double* __restrict__ periods,  // (P)
    const int* __restrict__ dur_bins,    // (D) box widths in bins
    const int n_points, const int n_periods, const int n_dur,
    const double bin_duration, const int oversample, const int width,
    const double sum_y, const double sum_ivar, const double t_min,
    const int obj_flag, const double NEG_INF,
    double* o_power, double* o_depth, double* o_depth_err, double* o_depth_snr,
    double* o_duration, double* o_transit_time, double* o_loglike)
{
    const int pidx = blockIdx.x;
    if (pidx >= n_periods) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const double period = periods[pidx];

    int n_bins = (int)ceil(period / bin_duration) + oversample;
    if (n_bins > width - 1) n_bins = width - 1;

    extern __shared__ double sh[];
    double* my = sh;          // weighted-y per bin
    double* mi = sh + width;  // ivar per bin

    for (int i = tid; i < width; i += nth) { my[i] = 0.0; mi[i] = 0.0; }
    __syncthreads();

    for (int j = tid; j < n_points; j += nth) {
        double x = tau[j];
        double w = x - period * floor(x / period);
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

    __shared__ double cof_y[CUDA_BLOCK];
    __shared__ double cof_i[CUDA_BLOCK];
    {
        int span = n_bins + 1;
        int chunk = (span + nth - 1) / nth;
        int lo = tid * chunk;
        int hi = lo + chunk; if (hi > span) hi = span;
        double ay = 0.0, ai = 0.0;
        for (int i = lo; i < hi; ++i) {
            ay += my[i]; my[i] = ay; ai += mi[i]; mi[i] = ai;
        }
        cof_y[tid] = ay; cof_i[tid] = ai;
        __syncthreads();
        if (tid == 0) {
            double sy = 0.0, si = 0.0;
            for (int k = 0; k < nth; ++k) {
                double ty = cof_y[k], ti = cof_i[k];
                cof_y[k] = sy; cof_i[k] = si; sy += ty; si += ti;
            }
        }
        __syncthreads();
        double oy = cof_y[tid], oi = cof_i[tid];
        for (int i = lo; i < hi; ++i) { my[i] += oy; mi[i] += oi; }
    }
    __syncthreads();

    double loc_obj = NEG_INF;
    int loc_n = 0, loc_d = 0;
    for (int di = 0; di < n_dur; ++di) {
        const int d = dur_bins[di];
        if (d < 1 || d >= n_bins) continue;
        const int n_max = n_bins - d;
        for (int n = tid; n <= n_max; n += nth) {
            double y_in = my[n + d] - my[n];
            double iv_in = mi[n + d] - mi[n];
            double iv_out = sum_ivar - iv_in;
            if (iv_in < 2.2204460492503131e-16) continue;
            if (iv_out < 2.2204460492503131e-16) continue;
            double yin = y_in / iv_in;
            double yout = (sum_y - y_in) / iv_out;
            if (yout < yin) continue;
            double depth = yout - yin;
            double obj = (obj_flag == 0)
                ? depth / sqrt(1.0 / iv_in + 1.0 / iv_out)
                : 0.5 * iv_in * depth * depth;
            if (obj > loc_obj) { loc_obj = obj; loc_n = n; loc_d = d; }
        }
    }

    __shared__ double r_obj[CUDA_BLOCK];
    __shared__ int r_n[CUDA_BLOCK];
    __shared__ int r_d[CUDA_BLOCK];
    r_obj[tid] = loc_obj; r_n[tid] = loc_n; r_d[tid] = loc_d;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && r_obj[tid + s] > r_obj[tid]) {
            r_obj[tid] = r_obj[tid + s];
            r_n[tid] = r_n[tid + s];
            r_d[tid] = r_d[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double best = r_obj[0];
        if (best <= NEG_INF) {
            o_power[pidx] = NEG_INF; o_depth[pidx] = 0.0; o_depth_err[pidx] = 0.0;
            o_depth_snr[pidx] = 0.0; o_duration[pidx] = 0.0;
            o_transit_time[pidx] = 0.0; o_loglike[pidx] = 0.0;
            return;
        }
        int bn = r_n[0], bd = r_d[0];
        double y_in = my[bn + bd] - my[bn];
        double iv_in = mi[bn + bd] - mi[bn];
        double iv_out = sum_ivar - iv_in;
        double yin = y_in / iv_in;
        double yout = (sum_y - y_in) / iv_out;
        double depth = yout - yin;
        double derr = sqrt(1.0 / iv_in + 1.0 / iv_out);
        double dsnr = depth / derr;
        double loglike = 0.5 * iv_in * depth * depth;
        double dur = (double)bd * bin_duration;
        o_power[pidx] = (obj_flag == 0) ? dsnr : loglike;
        o_depth[pidx] = depth;
        o_depth_err[pidx] = derr;
        o_depth_snr[pidx] = dsnr;
        o_duration[pidx] = dur;
        o_transit_time[pidx] =
            fmod((double)bn * bin_duration + 0.5 * dur, period) + t_min;
        o_loglike[pidx] = loglike;
    }
}
"""

_cuda_kernel_cache: dict[int, Any] = {}


def _cuda_kernel(block: int) -> Any:
    """Compile (once) and cache the BLS RawKernel for a given block size."""
    kernel = _cuda_kernel_cache.get(block)
    if kernel is None:
        import cupy

        src = _BLS_CUDA_SRC.replace("CUDA_BLOCK", str(block))
        kernel = cupy.RawKernel(src, "bls_block")
        _cuda_kernel_cache[block] = kernel
    return kernel


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
    block: int = CUDA_BLOCK,
) -> dict[str, Any]:
    """One-block-per-period CUDA box search; returns cupy arrays per output."""
    import cupy as cp

    t_d = cp.asarray(np.ascontiguousarray(t, dtype=np.float64))
    ivar_d = cp.asarray(np.ascontiguousarray(ivar, dtype=np.float64))
    y_d = cp.asarray(np.ascontiguousarray(y, dtype=np.float64))
    t_min = float(t_d.min())
    tau = t_d - t_min
    yw = y_d * ivar_d
    sum_y = float(yw.sum())
    sum_ivar = float(ivar_d.sum())
    periods_d = cp.asarray(np.ascontiguousarray(periods, dtype=np.float64))
    n_periods = int(periods_d.shape[0])
    dur_d = cp.asarray(np.asarray(dur_bins, dtype=np.int32))

    names = (
        "power",
        "depth",
        "depth_err",
        "depth_snr",
        "duration",
        "transit_time",
        "log_likelihood",
    )
    out = {name: cp.empty(n_periods, dtype=cp.float64) for name in names}
    if n_periods == 0 or len(dur_bins) == 0:
        out["power"][...] = -np.inf
        return out

    _cuda_kernel(block)(
        (n_periods,),
        (block,),
        (
            tau,
            yw,
            ivar_d,
            periods_d,
            dur_d,
            np.int32(t_d.shape[0]),
            np.int32(n_periods),
            np.int32(dur_d.shape[0]),
            np.float64(bin_duration),
            np.int32(oversample),
            np.int32(width),
            np.float64(sum_y),
            np.float64(sum_ivar),
            np.float64(t_min),
            np.int32(obj_flag),
            np.float64(-np.inf),
            out["power"],
            out["depth"],
            out["depth_err"],
            out["depth_snr"],
            out["duration"],
            out["transit_time"],
            out["log_likelihood"],
        ),
        shared_mem=2 * width * 8,
    )
    return out


def bls_power(
    t: FloatArray,
    y: FloatArray,
    dy: FloatArray,
    periods: FloatArray,
    durations: FloatArray,
    oversample: int,
    *,
    objective: str = "snr",
    backend: BLSBackend = "numpy",
    batch: int = DEFAULT_BATCH,
) -> BLSPower:
    """BLS box search over ``periods`` via numpy (CPU) or cupy (GPU).

    Equivalent to ``BoxLeastSquares(t, y, dy).power(periods, durations,
    objective=objective, oversample=oversample)`` — same binning and objective. ``y``
    must be in the transit-as-dip convention (a flux decrease). Inverse variances are
    ``1/dy**2``; ``durations`` are absolute (same units as ``t``); ``oversample`` is the
    number of bins per the shortest duration.

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
    backend : {"numpy", "cupy"}, default "numpy"
        CPU reference or GPU kernel.
    batch : int, default 2048
        Trial periods per vectorized batch (numpy backend).

    Returns
    -------
    BLSPower
        Per-period maxima aligned with ``periods``.
    """
    if objective not in ("snr", "likelihood"):
        raise ValueError("objective must be 'snr' or 'likelihood'")
    obj_flag = 0 if objective == "snr" else 1
    if backend not in ("numpy", "cupy", "numba"):
        raise ValueError(f"unknown backend {backend!r}")

    periods_host = np.ascontiguousarray(periods, dtype=np.float64)
    durations_host = np.ascontiguousarray(durations, dtype=np.float64)
    if periods_host.size == 0:
        empty = tuple(np.empty(0, dtype=np.float64) for _ in range(8))
        return BLSPower(*empty)
    bin_duration = float(durations_host.min()) / oversample
    dur_bins = _duration_bins(durations_host, bin_duration)
    max_n_bins = int(np.ceil(float(periods_host.max()) / bin_duration)) + oversample
    width = max_n_bins + 1

    ivar_host = 1.0 / (np.ascontiguousarray(dy, dtype=np.float64) ** 2)
    if backend == "cupy":
        from cuperiod.core.backend import ensure_cuda_dll_path

        ensure_cuda_dll_path()
        out = _bls_search_cuda(
            np.ascontiguousarray(t, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            ivar_host,
            periods_host,
            bin_duration=bin_duration,
            dur_bins=dur_bins,
            oversample=oversample,
            width=width,
            obj_flag=obj_flag,
        )
        import cupy

        def host(a: Any) -> FloatArray:
            return np.asarray(cupy.asnumpy(a), dtype=np.float64)
    elif backend == "numba":
        out = _bls_search_numba(
            np.ascontiguousarray(t, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            ivar_host,
            periods_host,
            bin_duration=bin_duration,
            dur_bins=dur_bins,
            oversample=oversample,
            width=width,
            obj_flag=obj_flag,
        )

        def host(a: Any) -> FloatArray:
            return np.asarray(a, dtype=np.float64)
    else:
        out = _bls_search(
            np,
            np.ascontiguousarray(t, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            ivar_host,
            periods_host,
            bin_duration=bin_duration,
            dur_bins=dur_bins,
            oversample=oversample,
            width=width,
            obj_flag=obj_flag,
            batch=batch,
        )

        def host(a: Any) -> FloatArray:
            return np.asarray(a, dtype=np.float64)

    return BLSPower(
        period=periods_host,
        power=host(out["power"]),
        depth=host(out["depth"]),
        depth_err=host(out["depth_err"]),
        depth_snr=host(out["depth_snr"]),
        duration=host(out["duration"]),
        transit_time=host(out["transit_time"]),
        log_likelihood=host(out["log_likelihood"]),
    )


__all__ = [
    "BLSBackend",
    "BLSPower",
    "DEFAULT_BATCH",
    "bls_power",
]
