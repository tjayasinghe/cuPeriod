"""The Weighted Wavelet Z-transform of Foster (1996): :func:`wwz`.

The WWZ is a *frequency-adaptive* time-frequency transform for unevenly sampled
light curves. At each cell of a ``(tau, frequency)`` grid it fits the three
trial functions

    phi_1 = 1,   phi_2 = cos(omega*(t - tau)),   phi_3 = sin(omega*(t - tau))

to the data by weighted least squares, under the Gaussian statistical weights

    w_a = exp(-c * omega^2 * (t_a - tau)^2),     omega = 2*pi*f,

and reports the F-like ratio of the fitted variation to the residual variation,

    WWZ = (N_eff - 3) * V_y / (2 * (V_x - V_y)),

together with the fitted semi-amplitude ``WWA = sqrt(y_2^2 + y_3^2)`` and the
effective number of points ``N_eff = (sum w)^2 / sum(w^2)`` the cell actually
used (Foster 1996, AJ 112, 1709). The constant trial function is what makes the
projection *date-compensated* (Foster's DCDFT lineage): without it, the mean of
a lopsided or gap-riddled window leaks straight into the fitted amplitude.

**Why the window scales with frequency.** :func:`cuperiod.sliding_periodogram`
slides a *fixed* window ``W`` and evaluates every trial frequency inside it: at
long periods the window holds barely one cycle (nothing to fit), at short
periods it holds thousands (over-resolved, and blind to fast wander). The WWZ
window instead has

    sigma_t = P / (2*pi*sqrt(2c)),     sigma_f / f = sqrt(2c),

so it is a *constant number of periods* wide — and has a constant *fractional*
frequency resolution — everywhere on the grid. That is exactly the property the
wander problem needs, and the reason WWZ became the standard tool for the AAVSO
long-period variables.

Decay constant ``c`` (the one knob that matters; it sets the time/frequency
trade-off and nothing else):

- ``c = 1/(8*pi^2) = 0.012665`` — the default here and in VARTOOLS. The window
  is *exactly one period* wide at every trial frequency; ``sigma_f/f = 15.9%``.
- ``c = 0.0125`` — Foster's own value, functionally equivalent to the default
  (``sigma_t = 1.007 P``, ``sigma_f/f = 15.8%``).
- ``c = 0.001`` — the long-period-variable convention, used by Templeton,
  Mattei & Willson (2005, AJ 130, 776) for the period wander of 547 Miras:
  ``sigma_t = 3.56 P``, ``sigma_f/f = 4.5%``. A much sharper frequency ridge,
  bought with a much coarser time resolution.

Typical uses: Mira and semi-regular period wander (the wander is a few percent
per cycle, which a static periodogram smears into one broad blob); mode
switches and glitches; exploratory hunting for quasi-periodic oscillations in
AGN light curves, where the transient, drifting nature of the signal is the
whole question.

**Significance: do not read a p-value off the WWZ.** Foster is explicit about
this, and it is the single most abused property of the statistic. ``N_eff`` is
not an integer and the weights are not independent, so the nominal
F-distribution with ``(2, N_eff - 3)`` degrees of freedom that the algebra
suggests does *not* hold; the effective number of independent trials on a
``(tau, f)`` map is unknown as well. A WWZ value is a *relative* measure — a
ridge that stands out from its own background. Quantitative significance needs
surrogates (shuffled/bootstrapped or red-noise light curves put through the
same transform), which are out of scope for this function.

**Gaps.** The Gaussian window is centred, not clipped: a ``tau`` sitting in a
data gap borrows whatever lies on either side, and the ``N_eff`` map is the
only honest record of that. Trying to quantify a period or an amplitude inside
a gap longer than the period in question is, in Foster's word, "folly"
(Foster 1996, section 6); the ``min_neff`` mask exists to keep such cells out
of the map rather than to rescue them.

Implementation notes. float64 throughout, on one of two CPU tiers: a
multicore numba kernel (one parallel iteration per ``(tau, frequency)`` cell,
the ``[fast]`` extra) and a vectorized NumPy fallback that evaluates blocks of
frequencies at once. Both make the same two simplifications, exact in exact
arithmetic. The values are centred on their mean first: the constant trial
function makes the statistic invariant to an offset, and the sums otherwise
carry the squared mean, which ``<x|x> - y.b`` then has to cancel. And each
cell only visits the points under its own Gaussian: times are sorted once, and
a cell keeps the points whose weight is within ``exp(-45)`` of its *nearest*
datum's, so a ``tau`` inside a gap (where every weight is tiny but the
statistic is still defined) keeps the same mask and, to round-off, the same
values and ``N_eff`` as the full sum. The per-point measurement errors are
*not* used: Foster's weights are the wavelet window alone, and mixing the two
would change the statistic's meaning. A GPU/array-API path (the arithmetic is
a handful of ``(nf, n)`` reductions, so it maps well onto a device) is future
work.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, Final

import numpy as np

from cuperiod.core._coerce import as_float_array
from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.core.backend import has_module, normalize_backend
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.errors import BackendUnavailableError, InsufficientDataError
from cuperiod.core.grid import pseudo_nyquist_frequency
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.pdot.scan import _parabolic_offset

#: Foster's decay constant in the VARTOOLS convention: the Gaussian window is
#: exactly one period wide (``sigma_t = P``) at every trial frequency.
_DEFAULT_C: Final[float] = 1.0 / (8.0 * math.pi * math.pi)

#: Fewest finite points before a WWZ map is attempted at all.
_MIN_POINTS: Final[int] = 20

#: Largest automatic frequency grid; beyond it the geometric step is coarsened
#: to span the same range (recorded as ``grid_capped`` in ``meta``).
_MAX_FREQUENCIES: Final[int] = 2000

#: Elements per transient ``(block, window)`` matrix of the NumPy path (512 KB
#: float64). A block is a run of frequencies sharing one point window, sized so
#: its half-dozen live matrices stay cache-resident (measured fastest of 2^16,
#: 2^18 and 2^20 on 1e3-2e4 points); a single frequency whose window holds more
#: points than this is evaluated on its own.
_BLOCK_ELEMS: Final[int] = 1 << 16

#: Dynamic range of the Gaussian weights a cell keeps, as ``-ln`` of the weight
#: ratio to the cell's *nearest* datum: points beyond
#: ``sqrt(d_min^2 + _WINDOW_CUT/(c*omega^2))`` of ``tau`` carry less than
#: ``exp(-45) = 2.9e-20`` of that datum's weight, below float64 resolution of
#: every sum they would join. Anchoring on the nearest datum rather than on a
#: weight of 1 is what keeps a ``tau`` inside a gap exact: there all weights are
#: tiny, but the statistic is ratio-invariant and still defined. ``inf`` sums
#: every point (the tests compare the two).
_WINDOW_CUT: Final[float] = 45.0

#: Smallest usable determinant of the 3x3 normal matrix. The matrix has a unit
#: leading entry and ``S_22 + S_33 = 1``, so its determinant lives in
#: ``[0, 0.25]`` and a floor this far below 1 only catches genuinely collapsed
#: cells (one point under the window, or a window holding less than a cycle).
_DET_FLOOR: Final[float] = 1e-12

#: Backend selectors :func:`wwz` accepts (``"cpu"`` and ``"auto"`` are the same:
#: there is no GPU path).
_BACKENDS: Final[tuple[str, ...]] = ("auto", "cpu", "numba", "numpy")


def _wwz_block(
    dt: FloatArray,
    value: FloatArray,
    omega: FloatArray,
    c: float,
    min_neff: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """WWZ, WWA and N_eff for one ``tau`` over a block of angular frequencies.

    ``dt`` is ``t - tau`` (days) and ``value`` the mean-centred data of the
    points in the block's window, ``omega`` a block of angular frequencies;
    every quantity below is one of Foster's, evaluated for all frequencies of
    the block at once. Returns ``(wwz, wwa, n_eff)``, each of shape
    ``omega.shape``, with ``wwz``/``wwa`` set to NaN on degenerate cells and
    ``n_eff`` always finite (it is the diagnostic that explains the mask).
    """
    arg = np.multiply.outer(omega, dt)
    cos_a = np.cos(arg)
    sin_a = np.sin(arg)
    # The weights reuse the angle buffer: one (block, window) matrix fewer.
    np.multiply(arg, arg, out=arg)
    arg *= -c
    w = np.exp(arg, out=arg)
    wc = w * cos_a
    ws = w * sin_a

    # Every value-weighted sum in one matrix product per weight matrix.
    rhs = np.empty((dt.size, 3), dtype=np.float64)
    rhs[:, 0] = 1.0
    rhs[:, 1] = value
    rhs[:, 2] = value * value
    w_sums = w @ rhs  # sum w, sum w x, sum w x^2
    c_sums = wc @ rhs[:, :2]  # sum w cos, sum w cos x
    s_sums = ws @ rhs[:, :2]  # sum w sin, sum w sin x

    sum_w = w_sums[:, 0]
    sum_w2 = np.einsum("ij,ij->i", w, w)
    # Degenerate cells divide by zero otherwise; they are masked out below.
    safe_w = np.where(sum_w > 0.0, sum_w, 1.0)
    safe_w2 = np.where(sum_w2 > 0.0, sum_w2, 1.0)
    n_eff = np.where(sum_w2 > 0.0, sum_w * sum_w / safe_w2, 0.0)

    # The 3x3 Gram matrix S_ab = <phi_a|phi_b> (S_11 = <1|1> = 1 identically).
    s12 = c_sums[:, 0] / safe_w
    s13 = s_sums[:, 0] / safe_w
    s22 = np.einsum("ij,ij->i", wc, cos_a) / safe_w
    s23 = np.einsum("ij,ij->i", wc, sin_a) / safe_w
    s33 = np.einsum("ij,ij->i", ws, sin_a) / safe_w

    # The right-hand side <phi_a|x>, and <x|x> for the data variation.
    b1 = w_sums[:, 1] / safe_w
    b2 = c_sums[:, 1] / safe_w
    b3 = s_sums[:, 1] / safe_w
    xx = w_sums[:, 2] / safe_w

    # Closed-form symmetric 3x3 solve (adjugate / determinant): no per-cell
    # linear-algebra call, and the determinant doubles as the singularity test.
    a11 = s22 * s33 - s23 * s23
    a12 = s13 * s23 - s12 * s33
    a13 = s12 * s23 - s13 * s22
    a22 = s33 - s13 * s13
    a23 = s12 * s13 - s23
    a33 = s22 - s12 * s12
    det = a11 + s12 * a12 + s13 * a13
    safe_det = np.where(det > _DET_FLOOR, det, 1.0)
    y1 = (a11 * b1 + a12 * b2 + a13 * b3) / safe_det
    y2 = (a12 * b1 + a22 * b2 + a23 * b3) / safe_det
    y3 = (a13 * b1 + a23 * b2 + a33 * b3) / safe_det

    # <y|y> = y.S.y = y.b (the normal equations), and <1|y> = (S y)_1 = <1|x>,
    # so V_y = y.b - <1|x>^2 and V_x - V_y = <x|x> - y.b. Both are differences
    # of terms of the size of the squared *mean*, which is why the values come
    # in centred: on raw data far from zero (a flux of 1e6, a magnitude offset
    # to 1e5) the cancellation decided the mask and the sign of the statistic.
    # V_y is a weighted variance, so a negative value is round-off: clamp it.
    ydotb = y1 * b1 + y2 * b2 + y3 * b3
    v_y = np.maximum(ydotb - b1 * b1, 0.0)
    v_resid = xx - ydotb

    good = (n_eff > min_neff) & (det > _DET_FLOOR) & (v_resid > 0.0)
    safe_resid = np.where(v_resid > 0.0, v_resid, 1.0)
    wwz_block = np.where(good, (n_eff - 3.0) * v_y / (2.0 * safe_resid), np.nan)
    wwa_block = np.where(good, np.hypot(y2, y3), np.nan)
    return (
        np.asarray(wwz_block, dtype=np.float64),
        np.asarray(wwa_block, dtype=np.float64),
        np.asarray(n_eff, dtype=np.float64),
    )


def _nearest_distance(times: FloatArray, tau: FloatArray) -> FloatArray:
    """Distance from each ``tau`` to the closest of the (sorted) ``times``."""
    n = int(times.size)
    idx = np.searchsorted(times, tau)
    right = times[np.minimum(idx, n - 1)]
    left = times[np.maximum(idx - 1, 0)]
    return np.asarray(
        np.minimum(np.abs(right - tau), np.abs(tau - left)), dtype=np.float64
    )


def _numpy_maps(
    times: FloatArray,
    value: FloatArray,
    tau: FloatArray,
    d_min: FloatArray,
    omega: FloatArray,
    c: float,
    min_neff: float,
    cut: float,
    out: tuple[FloatArray, FloatArray, FloatArray],
) -> None:
    """Fill the three maps with the vectorized NumPy path.

    For each ``tau`` the frequencies are visited in ascending order, so their
    point windows (all centred on ``tau``, radius shrinking with ``omega``) are
    nested: a block of consecutive frequencies is evaluated together on the
    window of its lowest one, and grows until it reaches :data:`_BLOCK_ELEMS`.
    ``times`` must be sorted.
    """
    wwz_map, wwa_map, neff_map = out
    nf = int(omega.size)
    order = np.argsort(omega, kind="stable")
    reach2 = cut / (c * omega[order] * omega[order])
    for i in range(int(tau.size)):
        tau_i = float(tau[i])
        radius = np.sqrt(float(d_min[i]) ** 2 + reach2)
        lo = np.searchsorted(times, tau_i - radius, side="left")
        hi = np.searchsorted(times, tau_i + radius, side="right")
        p = 0
        while p < nf:
            a, b = int(lo[p]), int(hi[p])
            k = max(1, min(nf - p, _BLOCK_ELEMS // max(1, b - a)))
            cols = order[p : p + k]
            z, amp, ne = _wwz_block(
                times[a:b] - tau_i, value[a:b], omega[cols], c, min_neff
            )
            wwz_map[i, cols] = z
            wwa_map[i, cols] = amp
            neff_map[i, cols] = ne
            p += k


_NUMBA_WWZ_KERNEL: Any = None


def _numba_wwz_kernel() -> Any:
    """Lazily compile (once) and cache the numba WWZ kernel.

    One ``prange`` iteration per ``(tau, frequency)`` cell: a binary search for
    the cell's point window (the same nearest-datum-anchored radius as the
    NumPy path), then a single pass accumulating the eleven weighted sums, then
    Foster's closed-form 3x3 solve exactly as :func:`_wwz_block` does it. The
    two CPU tiers therefore differ only in summation order (round-off).
    """
    global _NUMBA_WWZ_KERNEL
    if _NUMBA_WWZ_KERNEL is not None:
        return _NUMBA_WWZ_KERNEL
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=False)  # pragma: no cover - njit
    def _kernel(  # type: ignore[no-untyped-def]
        times,
        value,
        tau,
        d_min,
        omega,
        c,
        min_neff,
        det_floor,
        cut,
        wwz_map,
        wwa_map,
        neff_map,
    ):
        n = times.shape[0]
        nt = tau.shape[0]
        nf = omega.shape[0]
        for cell in prange(nt * nf):
            i = cell // nf
            j = cell - i * nf
            om = omega[j]
            tau_i = tau[i]
            radius = math.sqrt(d_min[i] * d_min[i] + cut / (c * om * om))
            # First index with times >= tau - radius ...
            edge = tau_i - radius
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if times[mid] < edge:
                    lo = mid + 1
                else:
                    hi = mid
            start = lo
            # ... and first index with times > tau + radius.
            edge = tau_i + radius
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if times[mid] <= edge:
                    lo = mid + 1
                else:
                    hi = mid
            stop = lo

            sw = 0.0
            sw2 = 0.0
            swc = 0.0
            sws = 0.0
            swcc = 0.0
            swcs = 0.0
            swss = 0.0
            swx = 0.0
            swcx = 0.0
            swsx = 0.0
            swxx = 0.0
            for a in range(start, stop):
                arg = om * (times[a] - tau_i)
                w = math.exp(-c * arg * arg)
                ca = math.cos(arg)
                sa = math.sin(arg)
                x = value[a]
                wc = w * ca
                ws = w * sa
                wx = w * x
                sw += w
                sw2 += w * w
                swc += wc
                sws += ws
                swcc += wc * ca
                swcs += wc * sa
                swss += ws * sa
                swx += wx
                swcx += wc * x
                swsx += ws * x
                swxx += wx * x

            safe_w = sw if sw > 0.0 else 1.0
            n_eff = sw * sw / sw2 if sw2 > 0.0 else 0.0
            s12 = swc / safe_w
            s13 = sws / safe_w
            s22 = swcc / safe_w
            s23 = swcs / safe_w
            s33 = swss / safe_w
            b1 = swx / safe_w
            b2 = swcx / safe_w
            b3 = swsx / safe_w
            xx = swxx / safe_w
            a11 = s22 * s33 - s23 * s23
            a12 = s13 * s23 - s12 * s33
            a13 = s12 * s23 - s13 * s22
            a22 = s33 - s13 * s13
            a23 = s12 * s13 - s23
            a33 = s22 - s12 * s12
            det = a11 + s12 * a12 + s13 * a13
            safe_det = det if det > det_floor else 1.0
            y1 = (a11 * b1 + a12 * b2 + a13 * b3) / safe_det
            y2 = (a12 * b1 + a22 * b2 + a23 * b3) / safe_det
            y3 = (a13 * b1 + a23 * b2 + a33 * b3) / safe_det
            ydotb = y1 * b1 + y2 * b2 + y3 * b3
            v_y = ydotb - b1 * b1
            if v_y < 0.0:
                v_y = 0.0
            v_resid = xx - ydotb
            neff_map[i, j] = n_eff
            if n_eff > min_neff and det > det_floor and v_resid > 0.0:
                wwz_map[i, j] = (n_eff - 3.0) * v_y / (2.0 * v_resid)
                wwa_map[i, j] = math.hypot(y2, y3)
            else:
                wwz_map[i, j] = np.nan
                wwa_map[i, j] = np.nan

    _NUMBA_WWZ_KERNEL = _kernel
    return _kernel


def _numba_available() -> bool:
    """Whether the multicore tier can run here (the ``[fast]`` extra)."""
    return has_module("numba")


def _resolve_backend(backend: str) -> str:
    """``"auto"``/``"cpu"`` -> the fastest CPU tier; ``"numba"``/``"numpy"`` as is."""
    name = normalize_backend(backend)
    if name in ("auto", "cpu"):
        return "numba" if _numba_available() else "numpy"
    if name == "numpy":
        return "numpy"
    if name == "numba":
        if not _numba_available():
            raise BackendUnavailableError(
                "wwz: backend 'numba' needs numba (pip install 'cuperiod[fast]')"
            )
        return "numba"
    raise BackendUnavailableError(
        f"wwz: unknown backend {backend!r}; choose from {_BACKENDS} (WWZ runs on "
        "the CPU only)"
    )


def _auto_frequency(
    time: FloatArray, baseline: float, c: float
) -> tuple[FloatArray, bool]:
    """A geometric trial-frequency grid at a quarter of the WWZ resolution.

    The window's fractional frequency resolution is ``sqrt(2c)`` *everywhere*,
    so the natural grid is geometric with ratio ``1 + sqrt(2c)/4`` — four
    samples across the local resolution element at every frequency, which a
    uniform grid can only deliver at one. Runs from two Rayleigh widths
    (``2/baseline``: a WWZ cell cannot say anything about a period longer than
    half the run) to the pseudo-Nyquist frequency at factor 1 — WWZ is for slow
    wander, not for short-period hunting. Returns ``(values, capped)``.
    """
    f_min = 2.0 / baseline
    f_max = pseudo_nyquist_frequency(time, 1)
    if f_max <= f_min:
        raise InsufficientDataError(
            f"wwz: the automatic frequency range collapsed (2/baseline = {f_min} "
            f">= pseudo-Nyquist {f_max} cycles/day); pass frequency= explicitly"
        )
    ratio = 1.0 + math.sqrt(2.0 * c) / 4.0
    n = int(math.floor(math.log(f_max / f_min) / math.log(ratio))) + 1
    capped = n > _MAX_FREQUENCIES
    if capped:
        values = np.geomspace(f_min, f_max, _MAX_FREQUENCIES)
    else:
        values = f_min * ratio ** np.arange(n, dtype=np.float64)
    return np.asarray(values, dtype=np.float64), capped


@dataclass(frozen=True)
class WWZResult:
    """A Weighted Wavelet Z-transform map plus its per-``tau`` ridge.

    Attributes
    ----------
    frequency : numpy.ndarray
        Trial frequency axis (cycles/day), shape ``(nf,)``.
    tau : numpy.ndarray
        Window centres (same time system as the input), shape ``(nt,)``.
    wwz : numpy.ndarray
        The Z-statistic, shape ``(nt, nf)``; NaN on cells the ``min_neff`` /
        degeneracy guards masked out.
    wwa : numpy.ndarray
        Fitted semi-amplitude ``sqrt(y_2^2 + y_3^2)`` in the units of the input
        values, same shape and same mask as ``wwz``.
    n_eff : numpy.ndarray
        Effective number of points under each cell's window,
        ``(sum w)^2 / sum(w^2)``. Always finite, *including* on masked cells —
        it is the record of why a cell was dropped.
    c : float
        The decay constant the map was computed with.
    n_samples : int
        Number of finite light-curve points used.
    baseline : float
        Time baseline in days.
    meta : Mapping
        Run provenance under the ``"wwz"`` key (``c``, ``min_neff``, ``n_tau``,
        the CPU ``backend`` that ran, whether the automatic grid was capped,
        whether the axes were supplied) plus metadata carried from the light
        curve.

    Notes
    -----
    Large ``wwz`` means "a coherent sinusoid of this frequency was present near
    this epoch", but *only* relative to the rest of the map — see the module
    docstring on why the nominal F-distribution does not apply.

    The ridge is spelled as on :class:`cuperiod.SlidingPeriodogram`:
    :attr:`peak_frequency`, :attr:`peak_period` (and :attr:`peak_wwa`) are
    arrays, and ``to_dict(include_map=True)`` adds the 2-D maps, so code
    written against one time-frequency map works on the other.
    """

    frequency: FloatArray
    tau: FloatArray
    wwz: FloatArray
    wwa: FloatArray
    n_eff: FloatArray
    c: float
    n_samples: int
    baseline: float
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """Map shape ``(n_tau, n_frequency)``."""
        return (int(self.tau.size), int(self.frequency.size))

    @property
    def period(self) -> FloatArray:
        """The frequency axis as periods (days), aligned with ``frequency``."""
        with np.errstate(divide="ignore"):
            return np.asarray(1.0 / self.frequency, dtype=np.float64)

    @property
    def sigma_t_over_p(self) -> float:
        """Gaussian window width in periods, ``1/(2*pi*sqrt(2c))``.

        1.0 at the default ``c`` (the window is exactly one period wide at
        every trial frequency); 3.56 at the ``c = 0.001`` LPV convention.
        """
        return 1.0 / (2.0 * math.pi * math.sqrt(2.0 * self.c))

    @property
    def sigma_f_over_f(self) -> float:
        """Fractional frequency resolution ``sqrt(2c)`` (15.9% by default)."""
        return math.sqrt(2.0 * self.c)

    def _peak_indices(self) -> IntArray:
        """Column of the largest WWZ per ``tau``; ``-1`` for a fully masked row."""
        finite = np.isfinite(self.wwz)
        score = np.where(finite, self.wwz, -np.inf)
        idx = np.asarray(np.argmax(score, axis=1), dtype=np.int64)
        idx[~finite.any(axis=1)] = -1
        return idx

    @cached_property
    def peak_frequency(self) -> FloatArray:
        """Per-``tau`` ridge frequency in cycles/day (NaN on a masked row).

        The largest WWZ cell of each row, parabolically refined against its two
        neighbours when the peak is interior. The refinement uses the *local*
        grid spacing, which is what the geometric default frequency axis needs,
        and steps toward the neighbour the parabola leans to, so an axis given
        in descending order (``frequency=1/periods``) refines the same way as
        its ascending copy. Computed once and cached read-only.
        """
        nf = int(self.frequency.size)
        out = np.full(int(self.tau.size), np.nan, dtype=np.float64)
        for i, j in enumerate(self._peak_indices().tolist()):
            if j < 0:
                continue
            f_best = float(self.frequency[j])
            if 0 < j < nf - 1:
                offset = _parabolic_offset(
                    float(self.wwz[i, j - 1]),
                    float(self.wwz[i, j]),
                    float(self.wwz[i, j + 1]),
                )
                # ``offset`` counts index steps; the signed spacing to the
                # neighbour it leans to converts it to cycles/day on either
                # axis direction (an unsigned one moved a descending axis's
                # ridge the wrong way).
                neighbor = j + 1 if offset >= 0.0 else j - 1
                step = float(self.frequency[neighbor]) - float(self.frequency[j])
                f_best += abs(offset) * step
            out[i] = f_best
        out.setflags(write=False)
        return out

    @property
    def peak_period(self) -> FloatArray:
        """Per-``tau`` ridge period in days (``1/peak_frequency``)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(1.0 / self.peak_frequency, dtype=np.float64)

    @cached_property
    def peak_wwa(self) -> FloatArray:
        """Semi-amplitude at each ``tau``'s ridge cell (NaN on a masked row).

        Read at the peak *cell*, not at the parabolically refined frequency —
        the amplitude varies far more slowly across the window's resolution
        element than the statistic does. Computed once and cached read-only.
        """
        out = np.full(int(self.tau.size), np.nan, dtype=np.float64)
        for i, j in enumerate(self._peak_indices().tolist()):
            if j >= 0:
                out[i] = float(self.wwa[i, j])
        out.setflags(write=False)
        return out

    def period_trend(self) -> tuple[float, float]:
        """Rough fractional period drift from the ridge, ``((dP/dt)/P, error)``.

        Fits an unweighted straight line to :attr:`peak_frequency` against
        ``tau - mean(tau)`` over the rows that produced a ridge at all. A slope
        ``s`` (cycles/day per day) is a fractional frequency drift ``s/f_bar``,
        so the fractional *period* rate is ``(dP/dt)/P = -s/f_bar`` in 1/day
        (``f_bar`` = mean ridge frequency). Multiply by the period for dP/dt in
        day/day.

        Returns
        -------
        pdot_over_p : float
            Fractional period-change rate (dP/dt)/P in 1/day.
        pdot_over_p_err : float
            Its standard error, from the scatter of the ridge about the line
            with the degrees of freedom counted in *independent windows*
            rather than in taus (see Notes); ``inf`` when the ridge spans too
            few windows to constrain a slope at all.

        Raises
        ------
        InsufficientDataError
            If fewer than three ``tau`` rows produced a ridge.

        Notes
        -----
        A rough *exploratory diagnostic*, exactly as
        :meth:`cuperiod.SlidingPeriodogram.period_trend` is. WWZ windows at
        neighbouring ``tau`` overlap unless the spacing exceeds a few
        ``sigma_t``, so consecutive ridge points are strongly correlated: at
        the ``c = 0.001`` convention (``sigma_t = 3.56 P``) the default 50 taus
        over a few-thousand-day run hold only a handful of independent windows.
        Treating them as independent made the error fall as ``1/sqrt(n_tau)``
        while the real scatter stayed put (about 4x too small on constant-period
        light curves). The error therefore uses

            n_ind = min(n, 1 + span / (2*sqrt(pi)*sigma_t)),
            var(slope) = RSS / ((n_ind - 2) * sum(x^2)),

        the integral scale of the overlap of two Gaussian windows
        (``2*sqrt(pi)*sigma_t``, with ``sigma_t`` evaluated at the mean ridge
        period) counting the independent windows along the ``span`` of the
        taus. It is identical to the ordinary formula when the windows do not
        overlap, and independent of ``n_tau`` when they do. A star that
        *wanders* has no meaningful linear drift to fit in the first place —
        which is usually the very thing the map was made to show. For a number
        to publish, localize with :func:`cuperiod.pdot_search` and measure
        with :func:`cuperiod.phase_track`.
        """
        freq = self.peak_frequency
        ok = np.isfinite(freq)
        n = int(np.count_nonzero(ok))
        if n < 3:
            raise InsufficientDataError(
                f"period_trend: need at least 3 valid taus, got {n}"
            )
        x = np.asarray(self.tau[ok], dtype=np.float64)
        x = x - float(np.mean(x))
        y = np.asarray(freq[ok], dtype=np.float64)
        f_bar = float(np.mean(y))
        sxx = float(np.dot(x, x))
        if sxx <= 0.0 or f_bar <= 0.0:
            raise InsufficientDataError(
                "period_trend: degenerate tau grid or ridge frequencies"
            )
        slope = float(np.dot(x, y - f_bar)) / sxx
        resid = y - (f_bar + slope * x)
        sigma_t = self.sigma_t_over_p / f_bar
        span = float(np.max(x) - np.min(x))
        n_ind = min(float(n), 1.0 + span / (2.0 * math.sqrt(math.pi) * sigma_t))
        if n_ind <= 2.0:
            return -slope / f_bar, math.inf
        variance = float(np.dot(resid, resid)) / (n_ind - 2.0)
        slope_err = math.sqrt(variance / sxx)
        return -slope / f_bar, slope_err / f_bar

    def to_dict(self, *, include_map: bool = False) -> dict[str, Any]:
        """Serialize to a plain dict (the ridge always; the 2-D maps optional).

        ``include_map=True`` adds ``frequency`` and the ``wwz``/``wwa``/``n_eff``
        maps, the same keyword as :meth:`cuperiod.SlidingPeriodogram.to_dict`.
        """
        nt, nf = self.shape
        out: dict[str, Any] = {
            "c": self.c,
            "sigma_t_over_p": self.sigma_t_over_p,
            "sigma_f_over_f": self.sigma_f_over_f,
            "n_tau": nt,
            "n_frequency": nf,
            "n_samples": self.n_samples,
            "baseline": self.baseline,
            "tau": self.tau.tolist(),
            "peak_frequency": self.peak_frequency.tolist(),
            "peak_period": self.peak_period.tolist(),
            "peak_wwa": self.peak_wwa.tolist(),
        }
        if include_map:
            out["frequency"] = self.frequency.tolist()
            out["wwz"] = self.wwz.tolist()
            out["wwa"] = self.wwa.tolist()
            out["n_eff"] = self.n_eff.tolist()
        return out


def wwz(
    data: Any,
    *,
    frequency: Any | None = None,
    tau: Any | None = None,
    n_tau: int = 50,
    c: float = _DEFAULT_C,
    min_neff: float = 3.0,
    columns: ColumnMap | None = None,
    domain: Domain | str | None = None,
    backend: str = "auto",
) -> WWZResult:
    """Weighted Wavelet Z-transform of a single-band light curve (Foster 1996).

    Fits ``{1, cos(omega*(t-tau)), sin(omega*(t-tau))}`` under Gaussian weights
    ``exp(-c*omega^2*(t-tau)^2)`` at every cell of a ``(tau, frequency)`` grid
    and returns the Z-statistic map, the fitted semi-amplitude map, and the
    effective point count of each cell. The window is a fixed number of
    *periods* wide at every frequency (``sigma_t = P/(2*pi*sqrt(2c))``), which
    is what distinguishes WWZ from the fixed-window
    :func:`cuperiod.sliding_periodogram`. See the module docstring for the
    conventions, the ``c`` table, and Foster's significance warning.

    Parameters
    ----------
    data : various
        The light curve, in any form :func:`cuperiod.to_input` accepts.
        Single-band only. Per-point errors are ignored (the weights are the
        wavelet window), so a row whose error is missing, zero or negative —
        the visual estimates of AAVSO-style data — is *kept*, unlike in the
        error-weighted periodograms; only rows with a non-finite time or value
        are dropped.
    frequency : array-like, optional
        Trial frequencies in cycles/day, used verbatim (copied) and in the
        given order; an ascending or a descending axis (``1/periods``) both
        work, while an unordered one leaves the ridge's parabolic refinement
        meaningless. ``None`` builds a geometric grid from ``2/baseline`` to
        the pseudo-Nyquist frequency, stepped at a quarter of the local
        resolution ``sqrt(2c)*f`` and capped at 2000 samples (recorded as
        ``grid_capped`` in ``meta``).
    tau : array-like, optional
        Window centres, in the input's time system, used verbatim (copied).
        ``None`` places ``n_tau`` of them uniformly over ``[t_min, t_max]``.
    n_tau : int, default 50
        Number of automatic window centres (the "time divisions" of VStar and
        PERANSO). Ignored when ``tau`` is given.
    c : float, default ``1/(8*pi^2)``
        Decay constant of the Gaussian window. The default makes the window
        exactly one period wide; ``0.001`` is the long-period-variable
        convention (3.56 periods, a 4.5% frequency resolution).
    min_neff : float, default 3.0
        Cells whose ``N_eff`` does not exceed this are masked to NaN. 3.0 is
        the algebraic floor (the fit has three free parameters and ``N_eff-3``
        appears in the numerator, so a lower value would let negative Z values
        through), not a sane analysis floor: 10-20 is a much better practical
        choice, and the ``n_eff`` map is always returned so the threshold can
        be revisited without recomputing.
    columns, domain
        Passed to :func:`cuperiod.to_input` for table inputs.
    backend : {"auto", "cpu", "numba", "numpy"}, default "auto"
        CPU tier. ``"auto"``/``"cpu"`` take the multicore numba kernel when the
        ``[fast]`` extra is installed and the vectorized NumPy path otherwise;
        the two agree to round-off. The tier that ran is recorded as
        ``meta["wwz"]["backend"]``.

    Returns
    -------
    WWZResult
        The ``(n_tau, n_frequency)`` maps plus the per-``tau`` ridge.

    Raises
    ------
    ValueError
        For a multi-band input, a non-positive or non-finite ``c``, a
        ``min_neff`` below 3, a non-positive ``n_tau``, or non-finite /
        non-positive explicit axes.
    BackendUnavailableError
        For an unknown ``backend``, or ``"numba"`` without numba installed.
    InsufficientDataError
        Fewer than 20 finite points, no usable baseline, or an automatic
        frequency range that collapsed.

    Examples
    --------
    >>> res = cup.wwz((t, mag, err), c=0.001)          # doctest: +SKIP
    >>> res.shape                                      # doctest: +SKIP
    >>> res.peak_period[:3]                            # doctest: +SKIP
    >>> res.period_trend()   # rough (dP/dt)/P, 1/day  # doctest: +SKIP
    """
    from cuperiod.api import to_input

    c = float(c)
    if not (math.isfinite(c) and c > 0.0):
        raise ValueError(f"c must be positive and finite, got {c}")
    min_neff = float(min_neff)
    # Written so that NaN fails too; an infinite floor would mask every cell.
    if not (min_neff >= 3.0 and math.isfinite(min_neff)):
        raise ValueError(
            f"min_neff must be finite and >= 3 (the algebraic floor: the fit has "
            f"three free parameters), got {min_neff}"
        )
    resolved = _resolve_backend(backend)

    lc = to_input(data, columns=columns, domain=domain)
    if isinstance(lc, MultiBandLightCurve):
        raise ValueError(
            "wwz is single-band; pool or pick a band first (stack the bands "
            "into one LightCurve, or index lc.bands)"
        )
    # The errors are never used, so a missing or non-positive one must not cost
    # its point: finite() on an error-free copy only drops non-finite time/value.
    work = replace(lc, error=None).finite()
    n_samples = work.n
    if n_samples < _MIN_POINTS:
        raise InsufficientDataError(
            f"wwz: {n_samples} finite points < the minimum of {_MIN_POINTS}"
        )
    baseline = work.baseline
    if baseline <= 0.0:
        raise InsufficientDataError("wwz: no usable time baseline")
    time = work.time
    t_min, t_max = float(time.min()), float(time.max())

    # -- axes ---------------------------------------------------------------------
    grid_capped = False
    if frequency is None:
        freq_values, grid_capped = _auto_frequency(time, baseline, c)
    else:
        # A private copy: the frozen result must not alias the caller's array.
        freq_values = as_float_array(frequency, name="wwz frequency").ravel().copy()
        if freq_values.size == 0:
            raise ValueError("wwz: frequency must hold at least one trial frequency")
        if not bool(np.all(np.isfinite(freq_values))) or bool(
            np.any(freq_values <= 0.0)
        ):
            raise ValueError("wwz: frequency values must be finite and positive")
    if tau is None:
        # `not (... >= 1)` so a NaN fails here rather than in int() below.
        if not (math.isfinite(n_tau) and float(n_tau).is_integer() and n_tau >= 1):
            raise ValueError(f"n_tau must be an integer >= 1, got {n_tau}")
        tau_values = np.asarray(
            np.linspace(t_min, t_max, int(n_tau)), dtype=np.float64
        )
    else:
        tau_values = as_float_array(tau, name="wwz tau", time=True).ravel().copy()
        if tau_values.size == 0:
            raise ValueError("wwz: tau must hold at least one window centre")
        if not bool(np.all(np.isfinite(tau_values))):
            raise ValueError("wwz: tau values must be finite")

    # -- the map ------------------------------------------------------------------
    # Times are referenced to t_min before differencing: the subtraction is
    # exact either way, but this keeps the angle arguments free of the
    # large-absolute-JD magnitudes. Sorting once lets every cell find its own
    # point window by binary search.
    order = np.argsort(time, kind="stable")
    times = np.ascontiguousarray(time[order] - t_min)
    tau_shift = np.ascontiguousarray(tau_values - t_min)
    # Centred: WWZ/WWA are exactly invariant to a constant offset (the constant
    # trial function absorbs it), and the sums would otherwise carry the
    # squared mean into the <x|x> - y.b cancellation. A constant series is made
    # exactly zero, so round-off in its mean cannot fake a variation.
    raw = work.value[order]
    if float(np.ptp(raw)) == 0.0:
        value = np.zeros_like(raw)
    else:
        value = np.ascontiguousarray(raw - float(np.mean(raw)))
    omega = np.ascontiguousarray(2.0 * np.pi * freq_values)
    d_min = _nearest_distance(times, tau_shift)
    cut = float(_WINDOW_CUT)
    nt, nf = int(tau_values.size), int(freq_values.size)
    wwz_map = np.empty((nt, nf), dtype=np.float64)
    wwa_map = np.empty((nt, nf), dtype=np.float64)
    neff_map = np.empty((nt, nf), dtype=np.float64)
    maps = (wwz_map, wwa_map, neff_map)
    if resolved == "numba":
        kernel = _numba_wwz_kernel()
        kernel(
            times, value, tau_shift, d_min, omega, c, min_neff, _DET_FLOOR, cut, *maps
        )
    else:
        _numpy_maps(times, value, tau_shift, d_min, omega, c, min_neff, cut, maps)

    meta = {
        "wwz": {
            "c": c,
            "min_neff": min_neff,
            "n_tau": nt,
            "backend": resolved,
            "grid_capped": grid_capped,
            "explicit_frequency": frequency is not None,
            "explicit_tau": tau is not None,
            "f_min": float(freq_values.min()),
            "f_max": float(freq_values.max()),
        },
        **dict(work.meta),
    }
    return WWZResult(
        frequency=freq_values,
        tau=tau_values,
        wwz=wwz_map,
        wwa=wwa_map,
        n_eff=neff_map,
        c=c,
        n_samples=n_samples,
        baseline=baseline,
        meta=meta,
    )


__all__ = ["WWZResult", "wwz"]
