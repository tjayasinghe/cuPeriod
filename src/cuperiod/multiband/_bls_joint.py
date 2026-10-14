"""Joint-ephemeris multi-band box search: one box shared by every band.

A transit is achromatic: every band dips at the same mid-transit time for the same
duration, only the depth changes with wavelength. The joint statistic therefore
places *one* box -- one ``(duration, phase)`` on a phase grid common to all bands,
binned from one time origin -- and lets each band fit its own depth inside it::

    J(box) = sum_k max(depth_k, 0)^2 / (1/ivar_in_k + 1/ivar_out_k)

the summed squared depth-SNRs of the bands that dip in that box (a band with no
point in the box, or brightening there, adds nothing). ``J`` is maximised over the
boxes of each trial period, so -- unlike the independent per-band stack -- noise
maxima of different bands at unrelated phases no longer add up. With
``objective="likelihood"`` the box is chosen on the summed per-band
log-likelihoods ``0.5*ivar_in_k*depth_k^2`` instead. Either way the reported power
is the quadrature stack ``sum_k max(SNR_k, 0)^2`` at the chosen box, and the
geometry (duration, mid-transit time) is that shared box; the reported depth is the
band with the largest SNR there.

Three exact implementations of the same sweep, mirroring the single-band search of
:mod:`cuperiod.methods._bls_core`: an array-API one (numpy, torch), a multicore numba
kernel that reproduces it bit for bit (same binning order, same cumulative sums, the
same box arithmetic and first-wins tie-break), and a CUDA kernel (one block per trial
period; each band is binned and scanned in shared memory in turn and its cumulative
sums parked in a global scratch, so the shared-memory footprint is that of a
single-band search however many bands there are).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final

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
from cuperiod.methods._bls_core import (
    CUDA_BLOCK,
    _duration_bins,
    _ivar_floor,
)

#: Fields of one joint search, aligned with the trial periods.
JOINT_FIELDS: Final = (
    "power", "depth", "depth_snr", "duration", "transit_time", "band_index",
)

#: Transient bytes one array-API batch may hold (host / device).
_HOST_BATCH_BYTES: Final = 1 << 29
_DEVICE_BATCH_BYTES: Final = 1 << 30

#: Global cumulative-sum scratch of one CUDA launch (both arrays together).
_CUDA_SCRATCH_BYTES: Final = 1 << 28


@dataclass(frozen=True)
class JointBands:
    """Every contributing band, ready for the joint sweep (host float64).

    ``tau`` holds each band's times minus the *common* origin ``t_min`` (so every band
    is phase-binned against the same clock), ``yw``/``ivar`` the median-centred
    ``y*ivar`` and ``1/dy**2``. ``starts`` delimits the bands in the concatenated
    arrays; ``sum_y``/``sum_ivar``/``floor`` are per band.
    """

    tau: FloatArray
    yw: FloatArray
    ivar: FloatArray
    starts: np.ndarray
    sum_y: FloatArray
    sum_ivar: FloatArray
    floor: FloatArray
    t_min: float

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        return int(self.starts.size - 1)

    def band(self, k: int) -> slice:
        """The slice of band ``k`` in the concatenated arrays."""
        return slice(int(self.starts[k]), int(self.starts[k + 1]))


def joint_bands(
    times: list[FloatArray], values: list[FloatArray], errors: list[FloatArray]
) -> JointBands:
    """Prepare the bands of one star for :func:`joint_search` (any backend).

    Each band is centred on its own median (as the single-band search does: every box
    statistic is offset-invariant, and centring keeps the cumulative sums from
    carrying the flux level); the times of all bands share one origin.
    """
    t_min = float(min(float(np.min(t)) for t in times))
    taus, yws, ivars = [], [], []
    sum_y, sum_ivar, floors = [], [], []
    for t, y, dy in zip(times, values, errors, strict=True):
        y_c = np.asarray(y, dtype=np.float64)
        y_c = y_c - float(np.median(y_c))
        ivar = 1.0 / (np.asarray(dy, dtype=np.float64) ** 2)
        yw = y_c * ivar
        taus.append(np.asarray(t, dtype=np.float64) - t_min)
        yws.append(yw)
        ivars.append(ivar)
        sum_y.append(float(yw.sum()))
        sum_ivar.append(float(ivar.sum()))
        floors.append(_ivar_floor(int(yw.size), sum_ivar[-1]))
    starts = np.concatenate([[0], np.cumsum([t.size for t in taus])]).astype(np.int64)
    return JointBands(
        tau=np.ascontiguousarray(np.concatenate(taus)),
        yw=np.ascontiguousarray(np.concatenate(yws)),
        ivar=np.ascontiguousarray(np.concatenate(ivars)),
        starts=starts,
        sum_y=np.asarray(sum_y, dtype=np.float64),
        sum_ivar=np.asarray(sum_ivar, dtype=np.float64),
        floor=np.asarray(floors, dtype=np.float64),
        t_min=t_min,
    )


@dataclass(frozen=True)
class _Segment:
    periods: FloatArray
    bin_duration: float
    dur_bins: list[int]
    width: int


def _segment(periods: FloatArray, durations: FloatArray, oversample: int) -> _Segment:
    periods = np.ascontiguousarray(periods, dtype=np.float64)
    durations = np.ascontiguousarray(durations, dtype=np.float64)
    bin_duration = float(durations.min()) / oversample
    dur_bins = _duration_bins(durations, bin_duration)
    width = int(np.ceil(float(periods.max()) / bin_duration)) + oversample + 1
    return _Segment(periods, bin_duration, dur_bins, width)


def _empty(n: int) -> dict[str, FloatArray]:
    return {name: np.zeros(n, dtype=np.float64) for name in JOINT_FIELDS}


def joint_search(
    bands: JointBands,
    segments: list[tuple[FloatArray, FloatArray]],
    oversample: int,
    *,
    objective: str,
    backend: str,
    batch: int = 2048,
    precision: str = "auto",
) -> list[dict[str, FloatArray]]:
    """The joint-ephemeris box search over every ``(periods, durations)`` segment.

    Parameters
    ----------
    bands : JointBands
        From :func:`joint_bands`.
    segments : list of (periods, durations)
        The period grid in duration-homogeneous segments (as the single-band search).
    oversample : int
        Phase bins per shortest duration.
    objective : {"snr", "likelihood"}
        What the shared box maximises (see the module docstring).
    backend : str
        ``"numba"``, ``"numpy"``, ``"cupy"`` or ``"torch:<device>"``.
    batch : int, default 2048
        Trial periods per array-API batch (capped by a memory budget; no effect on
        the result).
    precision : {"auto", "float64", "float32"}
        Device precision of the cupy and torch searches.

    Returns
    -------
    list of dict
        Per segment, :data:`JOINT_FIELDS` arrays aligned with its periods (host
        float64). A period without any admissible box has power 0 and zero geometry.
    """
    if objective not in ("snr", "likelihood"):
        raise ValueError("objective must be 'snr' or 'likelihood'")
    obj_flag = 0 if objective == "snr" else 1
    specs: list[_Segment | None] = [
        _segment(p, d, oversample) if np.asarray(p).size else None
        for p, d in segments
    ]
    live = [s for s in specs if s is not None]
    if backend == "cupy":
        found = iter(
            _joint_cuda(bands, live, oversample, obj_flag, precision) if live else []
        )
    else:
        found = iter(
            [_joint_one(bands, s, oversample, obj_flag, backend, batch, precision)
             for s in live]
        )
    return [_empty(0) if s is None else next(found) for s in specs]


def _joint_one(
    bands: JointBands,
    seg: _Segment,
    oversample: int,
    obj_flag: int,
    backend: str,
    batch: int,
    precision: str,
) -> dict[str, FloatArray]:
    if backend == "numba":
        return _joint_numba(bands, seg, oversample, obj_flag)
    is_torch = backend == "torch" or backend.startswith("torch:")
    if backend == "numpy":
        xp = array_namespace(seg.periods)
        return _joint_array_api(
            xp, bands, bands.tau, bands.yw, bands.ivar, seg.periods, seg,
            oversample, obj_flag, batch, on_device=False,
        )
    if is_torch:
        import torch

        device = backend.split(":", 1)[1] if ":" in backend else "cpu"
        tdtype = (
            torch.float32
            if resolve_precision(precision, device) == "float32"
            else torch.float64
        )

        def put(a: FloatArray) -> Any:
            return to_device_array(a, device=device, dtype=tdtype)

        periods = put(seg.periods)
        return _joint_array_api(
            array_namespace(periods), bands, put(bands.tau), put(bands.yw),
            put(bands.ivar), periods, seg, oversample, obj_flag, batch,
            on_device=device != "cpu",
        )
    raise ValueError(f"unknown backend {backend!r}")


# --- array-API (numpy / torch) ------------------------------------------------


def _joint_array_api(
    xp: ModuleType,
    bands: JointBands,
    tau: Any,
    yw: Any,
    ivar: Any,
    periods: Any,
    seg: _Segment,
    oversample: int,
    obj_flag: int,
    batch: int,
    *,
    on_device: bool,
) -> dict[str, FloatArray]:
    """The joint sweep, vectorised over a batch of periods (array-API generic)."""
    fdtype = periods.dtype
    idtype = xp.int64
    dev = device_ref(periods)
    n_bands, width = bands.n_bands, seg.width
    n_points = int(bands.tau.size)
    float32 = fdtype == xp.float32
    floors = [
        _ivar_floor(bands.band(k).stop - bands.band(k).start, float(bands.sum_ivar[k]),
                    float32=float32)
        for k in range(n_bands)
    ]
    per_period = 8 * (4 * max(1, n_points) + (2 * n_bands + 12) * max(1, width))
    budget = _DEVICE_BATCH_BYTES if on_device else _HOST_BATCH_BYTES
    step = max(1, min(int(batch), budget // per_period))
    n_periods = int(periods.shape[0])
    out = _empty(n_periods)
    cols_full = xp.arange(width, dtype=idtype, device=dev)
    taus = [tau[bands.band(k)] for k in range(n_bands)]
    yws = [yw[bands.band(k)] for k in range(n_bands)]
    ivs = [ivar[bands.band(k)] for k in range(n_bands)]
    for start in range(0, n_periods, step):
        stop = min(start + step, n_periods)
        pb = periods[start:stop]
        n_p = int(pb.shape[0])
        rows = xp.arange(n_p, dtype=idtype, device=dev)
        n_bins = xp.astype(xp.ceil(pb / seg.bin_duration), idtype) + oversample
        n_bins = xp.clip(n_bins, 0, width - 1)
        cy, cw = [], []
        for k in range(n_bands):
            phase = xp.remainder(taus[k][None, :], pb[:, None])
            ind = xp.astype(phase / seg.bin_duration, idtype) + 1
            ind = xp.clip(ind, 0, width - 1)
            my = xp.zeros((n_p, width), dtype=fdtype, device=dev)
            mi = xp.zeros((n_p, width), dtype=fdtype, device=dev)
            scatter_add_rows(my, ind, yws[k])
            scatter_add_rows(mi, ind, ivs[k])
            for j in range(oversample):
                dst = xp.clip(n_bins - oversample + j, 0, width - 1)
                my[rows, dst] = my[rows, 1 + j]
                mi[rows, dst] = mi[rows, 1 + j]
            cy.append(xp.cumsum(my, axis=1))
            cw.append(xp.cumsum(mi, axis=1))
        best = xp.full(n_p, -np.inf, dtype=fdtype, device=dev)
        best_n = xp.zeros(n_p, dtype=idtype, device=dev)
        best_d = xp.zeros(n_p, dtype=idtype, device=dev)
        for kd in seg.dur_bins:
            if kd >= width:
                continue
            cols = cols_full[: width - kd]
            valid = (cols[None, :] <= (n_bins[:, None] - kd)) & (
                n_bins[:, None] > kd
            )
            total = xp.zeros((n_p, width - kd), dtype=fdtype, device=dev)
            hit = xp.zeros((n_p, width - kd), dtype=xp.bool, device=dev)
            for k in range(n_bands):
                y_in = cy[k][:, kd:] - cy[k][:, :-kd]
                iv_in = cw[k][:, kd:] - cw[k][:, :-kd]
                iv_out = float(bands.sum_ivar[k]) - iv_in
                ok = (iv_in >= floors[k]) & (iv_out >= floors[k])
                safe_in = xp.where(ok, iv_in, 1.0)
                safe_out = xp.where(ok, iv_out, 1.0)
                yin = y_in / safe_in
                yout = (float(bands.sum_y[k]) - y_in) / safe_out
                dip = ok & (yout >= yin)
                depth = yout - yin
                if obj_flag == 0:
                    term = depth * depth / (1.0 / safe_in + 1.0 / safe_out)
                else:
                    term = 0.5 * safe_in * depth * depth
                total = total + xp.where(dip, term, 0.0)
                hit = hit | dip
            obj = xp.where(valid & hit, total, -np.inf)
            cand = xp.argmax(obj, axis=1)
            cand_obj = obj[rows, cand]
            improve = cand_obj > best
            best = xp.where(improve, cand_obj, best)
            best_n = xp.where(improve, xp.astype(cand, idtype), best_n)
            best_d = xp.where(improve, kd, best_d)
        found = xp.isfinite(best)
        # The shared box's per-band statistics (float64 on the host).
        bn = to_host(best_n).astype(np.int64)
        bd = to_host(best_d).astype(np.int64)
        ok_rows = to_host(found).astype(bool)
        yins = [to_host(cy[k][rows, best_n + best_d] - cy[k][rows, best_n])
                for k in range(n_bands)]
        ivins = [to_host(cw[k][rows, best_n + best_d] - cw[k][rows, best_n])
                 for k in range(n_bands)]
        _finish(
            out, slice(start, stop), bands, yins, ivins, bn, bd, ok_rows,
            to_host(pb), seg.bin_duration, floors,
        )
    return out


def _finish(
    out: dict[str, FloatArray],
    sl: slice,
    bands: JointBands,
    y_in: list[FloatArray],
    iv_in: list[FloatArray],
    best_n: np.ndarray,
    best_d: np.ndarray,
    found: np.ndarray,
    periods: FloatArray,
    bin_duration: float,
    floors: list[float],
) -> None:
    """Per-band depths and SNRs at the chosen boxes -> the reported fields."""
    n_bands = bands.n_bands
    n_p = int(periods.size)
    snr = np.zeros((n_bands, n_p))
    depth = np.zeros((n_bands, n_p))
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(n_bands):
            iv_out = float(bands.sum_ivar[k]) - iv_in[k]
            ok = (iv_in[k] >= floors[k]) & (iv_out >= floors[k])
            safe_in = np.where(ok, iv_in[k], 1.0)
            safe_out = np.where(ok, iv_out, 1.0)
            yin = y_in[k] / safe_in
            yout = (float(bands.sum_y[k]) - y_in[k]) / safe_out
            dip = ok & (yout >= yin)
            d = yout - yin
            depth[k] = np.where(dip, d, 0.0)
            snr[k] = np.where(dip, d / np.sqrt(1.0 / safe_in + 1.0 / safe_out), 0.0)
    power = np.zeros(n_p)
    for k in range(n_bands):
        power = power + snr[k] * snr[k]
    band = np.argmax(snr, axis=0)
    duration = best_d.astype(np.float64) * bin_duration
    t0 = (
        np.remainder(best_n.astype(np.float64) * bin_duration + 0.5 * duration, periods)
        + bands.t_min
    )
    cols = np.arange(n_p)
    out["power"][sl] = np.where(found, power, 0.0)
    out["depth"][sl] = np.where(found, depth[band, cols], 0.0)
    out["depth_snr"][sl] = np.where(found, np.sqrt(power), 0.0)
    out["duration"][sl] = np.where(found, duration, 0.0)
    out["transit_time"][sl] = np.where(found, t0, 0.0)
    out["band_index"][sl] = np.where(found, band.astype(np.float64), 0.0)


# --- numba (multicore CPU) -------------------------------------------------------

_NUMBA_JOINT: Any = None


def _numba_joint_kernel() -> Any:
    """Lazily compile (once) and cache the numba joint sweep."""
    global _NUMBA_JOINT
    if _NUMBA_JOINT is not None:
        return _NUMBA_JOINT
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=False)  # pragma: no cover - njit
    def _kernel(tau, yw, wv, starts, sum_y, sum_ivar, floors, periods, dur_bins,  # type: ignore[no-untyped-def]
                bin_duration, oversample, width, obj_flag):
        n_periods = periods.shape[0]
        n_bands = starts.shape[0] - 1
        n_dur = dur_bins.shape[0]
        neg = -np.inf
        o_n = np.zeros(n_periods, dtype=np.int64)
        o_d = np.zeros(n_periods, dtype=np.int64)
        o_found = np.zeros(n_periods, dtype=np.bool_)
        o_yin = np.zeros((n_bands, n_periods))
        o_ivin = np.zeros((n_bands, n_periods))
        for pidx in prange(n_periods):
            period = periods[pidx]
            n_bins = int(np.ceil(period / bin_duration)) + oversample
            if n_bins > width - 1:
                n_bins = width - 1
            my = np.empty((n_bands, width))
            mi = np.empty((n_bands, width))
            for k in range(n_bands):
                for i in range(width):
                    my[k, i] = 0.0
                    mi[k, i] = 0.0
                for j in range(starts[k], starts[k + 1]):
                    x = tau[j]
                    w = x - period * np.floor(x / period)
                    ind = int(w / bin_duration) + 1
                    if ind < 0:
                        ind = 0
                    elif ind > width - 1:
                        ind = width - 1
                    my[k, ind] += yw[j]
                    mi[k, ind] += wv[j]
                for o in range(oversample):
                    dst = n_bins - oversample + o
                    if dst < 0:
                        dst = 0
                    elif dst > width - 1:
                        dst = width - 1
                    my[k, dst] = my[k, 1 + o]
                    mi[k, dst] = mi[k, 1 + o]
                ay = 0.0
                ai = 0.0
                for i in range(width):
                    ay += my[k, i]
                    my[k, i] = ay
                    ai += mi[k, i]
                    mi[k, i] = ai
            best = neg
            bn = 0
            bd = 0
            for di in range(n_dur):
                d = dur_bins[di]
                if d >= width or d >= n_bins:
                    continue
                for n in range(n_bins - d + 1):
                    total = 0.0
                    hit = False
                    for k in range(n_bands):
                        y_in = my[k, n + d] - my[k, n]
                        iv_in = mi[k, n + d] - mi[k, n]
                        iv_out = sum_ivar[k] - iv_in
                        if iv_in < floors[k] or iv_out < floors[k]:
                            continue
                        yin = y_in / iv_in
                        yout = (sum_y[k] - y_in) / iv_out
                        if not (yout >= yin):
                            continue
                        depth = yout - yin
                        hit = True
                        if obj_flag == 0:
                            total += depth * depth / (1.0 / iv_in + 1.0 / iv_out)
                        else:
                            total += 0.5 * iv_in * depth * depth
                    if hit and total > best:
                        best = total
                        bn = n
                        bd = d
            if best > neg:
                o_found[pidx] = True
                o_n[pidx] = bn
                o_d[pidx] = bd
                for k in range(n_bands):
                    o_yin[k, pidx] = my[k, bn + bd] - my[k, bn]
                    o_ivin[k, pidx] = mi[k, bn + bd] - mi[k, bn]
        return o_n, o_d, o_found, o_yin, o_ivin

    _NUMBA_JOINT = _kernel
    return _kernel


def _joint_numba(
    bands: JointBands, seg: _Segment, oversample: int, obj_flag: int
) -> dict[str, FloatArray]:
    kernel = _numba_joint_kernel()
    floors = [float(f) for f in bands.floor]
    bn, bd, found, yin, ivin = kernel(
        bands.tau, bands.yw, bands.ivar, bands.starts, bands.sum_y, bands.sum_ivar,
        np.asarray(floors), seg.periods,
        np.ascontiguousarray(seg.dur_bins, dtype=np.int64),
        float(seg.bin_duration), int(oversample), int(seg.width), int(obj_flag),
    )
    out = _empty(int(seg.periods.size))
    _finish(
        out, slice(None), bands, list(yin), list(ivin), bn, bd, found, seg.periods,
        seg.bin_duration, floors,
    )
    return out


# --- CUDA ----------------------------------------------------------------------------

_JOINT_CUDA_SRC: Final = r"""
extern "C" __global__ void bls_joint_block(
    const REAL* __restrict__ tau,        // (N) every band, common origin removed
    const REAL* __restrict__ yw,         // (N) y * ivar, median-centred per band
    const REAL* __restrict__ wv,         // (N) ivar
    const int* __restrict__ starts,      // (K+1) band offsets
    const REAL* __restrict__ sum_y,      // (K)
    const REAL* __restrict__ sum_ivar,   // (K)
    const REAL* __restrict__ floors,     // (K) per-band ivar floors
    const REAL* __restrict__ periods,    // (all periods of the run)
    const int* __restrict__ dur_bins,    // (D)
    const int n_bands, const int n_periods, const int n_dur,
    const REAL bin_duration, const int oversample, const int width,
    const int obj_flag, const REAL NEG_INF,
    const int p0, const int stride,      // this launch's first period; outputs' row
    REAL* __restrict__ cum_y,            // (n_periods, K, width) scratch
    REAL* __restrict__ cum_i,            // (n_periods, K, width) scratch
    long long* o_n, long long* o_d, double* o_yin, double* o_ivin, int* o_found)
{
    const int pidx = blockIdx.x;
    if (pidx >= n_periods) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const int gidx = p0 + pidx;          // index into the run's periods and outputs
    const REAL period = periods[gidx];
    int n_bins = (int)ceil(period / bin_duration) + oversample;
    if (n_bins > width - 1) n_bins = width - 1;

    extern __shared__ REAL sh[];
    REAL* my = sh;
    REAL* mi = sh + width;
    __shared__ REAL cof_y[CUDA_BLOCK];
    __shared__ REAL cof_i[CUDA_BLOCK];
    REAL* gy = cum_y + (size_t)pidx * n_bands * width;
    REAL* gi = cum_i + (size_t)pidx * n_bands * width;

    for (int k = 0; k < n_bands; ++k) {
        for (int i = tid; i < width; i += nth) { my[i] = 0.0; mi[i] = 0.0; }
        __syncthreads();
        for (int j = starts[k] + tid; j < starts[k + 1]; j += nth) {
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
            for (int o = 0; o < oversample; ++o) {
                int dst = n_bins - oversample + o;
                if (dst < 0) dst = 0;
                if (dst > width - 1) dst = width - 1;
                my[dst] = my[1 + o]; mi[dst] = mi[1 + o];
            }
        }
        __syncthreads();
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
                for (int q = 0; q < nth; ++q) {
                    REAL ty = cof_y[q], ti = cof_i[q];
                    cof_y[q] = sy; cof_i[q] = si; sy += ty; si += ti;
                }
            }
            __syncthreads();
            REAL oy = cof_y[tid], oi = cof_i[tid];
            for (int i = lo; i < hi; ++i) { my[i] += oy; mi[i] += oi; }
        }
        __syncthreads();
        for (int i = tid; i <= n_bins; i += nth) {
            gy[(size_t)k * width + i] = my[i];
            gi[(size_t)k * width + i] = mi[i];
        }
        __syncthreads();
    }

    REAL loc = NEG_INF;
    int loc_n = 0, loc_d = 0, loc_di = 0;
    for (int di = 0; di < n_dur; ++di) {
        const int d = dur_bins[di];
        if (d >= width || d >= n_bins) continue;
        for (int n = tid; n <= n_bins - d; n += nth) {
            REAL total = 0.0;
            bool hit = false;
            for (int k = 0; k < n_bands; ++k) {
                const REAL* cy = gy + (size_t)k * width;
                const REAL* cw = gi + (size_t)k * width;
                REAL y_in = cy[n + d] - cy[n];
                REAL iv_in = cw[n + d] - cw[n];
                REAL iv_out = sum_ivar[k] - iv_in;
                if (iv_in < floors[k] || iv_out < floors[k]) continue;
                REAL yin = y_in / iv_in;
                REAL yout = (sum_y[k] - y_in) / iv_out;
                if (!(yout >= yin)) continue;
                REAL depth = yout - yin;
                hit = true;
                total += (obj_flag == 0)
                    ? depth * depth / ((REAL)1.0 / iv_in + (REAL)1.0 / iv_out)
                    : (REAL)0.5 * iv_in * depth * depth;
            }
            if (hit && total > loc) { loc = total; loc_n = n; loc_d = d; loc_di = di; }
        }
    }

    // Reduce to the best box; an exact tie (common: boxes that differ only by
    // empty bins hold the same points) goes to the earliest (duration, start) in
    // the sweep order, the first-wins rule of the host searches.
    __shared__ REAL r_obj[CUDA_BLOCK];
    __shared__ int r_n[CUDA_BLOCK];
    __shared__ int r_d[CUDA_BLOCK];
    __shared__ int r_di[CUDA_BLOCK];
    r_obj[tid] = loc; r_n[tid] = loc_n; r_d[tid] = loc_d; r_di[tid] = loc_di;
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
    // The chosen box's per-band sums go back in double for the host to finish.
    const int bn = r_n[0], bd = r_d[0];
    const bool found = r_obj[0] > NEG_INF;
    for (int k = tid; k < n_bands; k += nth) {
        const REAL* cy = gy + (size_t)k * width;
        const REAL* cw = gi + (size_t)k * width;
        o_yin[(size_t)k * stride + gidx] =
            found ? (double)cy[bn + bd] - (double)cy[bn] : 0.0;
        o_ivin[(size_t)k * stride + gidx] =
            found ? (double)cw[bn + bd] - (double)cw[bn] : 0.0;
    }
    if (tid == 0) {
        o_n[gidx] = bn; o_d[gidx] = bd; o_found[gidx] = found ? 1 : 0;
    }
}
"""

_joint_kernel_cache: dict[tuple[int, str], Any] = {}


def _joint_cuda_kernel(block: int, real: str) -> Any:
    kernel = _joint_kernel_cache.get((block, real))
    if kernel is None:
        import cupy

        src = _JOINT_CUDA_SRC.replace("CUDA_BLOCK", str(block)).replace(
            "REAL", "float" if real == "float32" else "double"
        )
        kernel = cupy.RawKernel(src, "bls_joint_block")
        _joint_kernel_cache[(block, real)] = kernel
    return kernel


def _joint_cuda(
    bands: JointBands,
    segments: list[_Segment],
    oversample: int,
    obj_flag: int,
    precision: str,
    block: int = CUDA_BLOCK,
) -> list[dict[str, FloatArray]]:
    """The CUDA joint sweep: one upload, launches back to back, one download."""
    from cuperiod.core.backend import ensure_cuda_dll_path, ensure_shared_memory

    ensure_cuda_dll_path()
    import cupy as cp

    real = resolve_precision(precision, "cuda")
    rdtype = np.float32 if real == "float32" else np.float64
    real_size = 4 if real == "float32" else 8
    n_bands = bands.n_bands
    floors = np.asarray(
        [_ivar_floor(bands.band(k).stop - bands.band(k).start,
                     float(bands.sum_ivar[k]), float32=real == "float32")
         for k in range(n_bands)],
        dtype=np.float64,
    )
    floors = np.maximum(floors, float(np.finfo(rdtype).tiny))

    def up(a: Any, dtype: Any) -> Any:
        return cp.asarray(np.ascontiguousarray(a, dtype=dtype))

    tau_d, yw_d, iv_d = (up(a, rdtype) for a in (bands.tau, bands.yw, bands.ivar))
    starts_d = up(bands.starts, np.int32)
    sum_y_d, sum_iv_d, floor_d = (
        up(a, rdtype) for a in (bands.sum_y, bands.sum_ivar, floors)
    )
    sizes = [int(s.periods.size) for s in segments]
    total = sum(sizes)
    periods_d = up(np.concatenate([s.periods for s in segments]), rdtype)
    durs = [np.asarray(s.dur_bins, dtype=np.int32) for s in segments]
    dur_d = up(np.concatenate(durs), np.int32)
    o_n = cp.zeros(total, dtype=cp.int64)
    o_d = cp.zeros(total, dtype=cp.int64)
    o_found = cp.zeros(total, dtype=cp.int32)
    o_yin = cp.zeros((n_bands, total), dtype=cp.float64)
    o_ivin = cp.zeros((n_bands, total), dtype=cp.float64)
    kernel = _joint_cuda_kernel(block, real)
    per_period = 2 * n_bands * max(s.width for s in segments) * real_size
    chunk = max(1, min(max(sizes), _CUDA_SCRATCH_BYTES // per_period))
    width_max = max(s.width for s in segments)
    cum_y = cp.empty(chunk * n_bands * width_max, dtype=rdtype)
    cum_i = cp.empty(chunk * n_bands * width_max, dtype=rdtype)
    p_off = d_off = 0
    for seg, n_seg, dur in zip(segments, sizes, durs, strict=True):
        smem = 2 * seg.width * real_size
        ensure_shared_memory(
            kernel, smem, method="BLS", hint="the period range (max_period_days)"
        )
        for lo in range(0, n_seg, chunk):
            hi = min(lo + chunk, n_seg)
            kernel(
                (hi - lo,),
                (block,),
                (
                    tau_d, yw_d, iv_d, starts_d, sum_y_d, sum_iv_d, floor_d,
                    periods_d, dur_d[d_off : d_off + dur.size],
                    np.int32(n_bands), np.int32(hi - lo), np.int32(dur.size),
                    rdtype(seg.bin_duration), np.int32(oversample),
                    np.int32(seg.width), np.int32(obj_flag), rdtype(-np.inf),
                    np.int32(p_off + lo), np.int32(total),
                    cum_y, cum_i, o_n, o_d, o_yin, o_ivin, o_found,
                ),
                shared_mem=smem,
            )
        p_off += n_seg
        d_off += int(dur.size)
    host_n, host_d, host_found = (a.get() for a in (o_n, o_d, o_found))
    host_yin, host_ivin = o_yin.get(), o_ivin.get()
    outs: list[dict[str, FloatArray]] = []
    p_off = 0
    for seg, n_seg in zip(segments, sizes, strict=True):
        sl = slice(p_off, p_off + n_seg)
        out = _empty(n_seg)
        _finish(
            out, slice(None), bands,
            [host_yin[k, sl] for k in range(n_bands)],
            [host_ivin[k, sl] for k in range(n_bands)],
            host_n[sl], host_d[sl], host_found[sl].astype(bool), seg.periods,
            seg.bin_duration, [float(f) for f in floors],
        )
        outs.append(out)
        p_off += n_seg
    return outs


__all__ = ["JOINT_FIELDS", "JointBands", "joint_bands", "joint_search"]
