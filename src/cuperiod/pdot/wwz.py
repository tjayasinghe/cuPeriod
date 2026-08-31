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

Implementation notes. Pure NumPy float64, vectorized over frequency for each
``tau`` and chunked so the transient ``(chunk, n)`` matrices stay bounded. The
per-point measurement errors are *not* used: Foster's weights are the wavelet
window alone, and mixing the two would change the statistic's meaning. A
GPU/array-API path (the arithmetic is a handful of ``(nf, n)`` reductions, so
it maps well onto a device) is future work; correctness first.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.errors import InsufficientDataError
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

#: Elements per transient ``(chunk, n)`` matrix (64 MB float64), mirroring the
#: direct GLS path: the frequency axis is chunked so a long light curve cannot
#: blow up host memory.
_CHUNK_ELEMS: Final[int] = 1 << 23

#: Smallest usable determinant of the 3x3 normal matrix. The matrix has a unit
#: leading entry and ``S_22 + S_33 = 1``, so its determinant lives in
#: ``[0, 0.25]`` and a floor this far below 1 only catches genuinely collapsed
#: cells (one point under the window, or a window holding less than a cycle).
_DET_FLOOR: Final[float] = 1e-12


def _wwz_block(
    dt: FloatArray,
    value: FloatArray,
    omega: FloatArray,
    c: float,
    min_neff: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """WWZ, WWA and N_eff for one ``tau`` over a block of angular frequencies.

    ``dt`` is ``t - tau`` (days) and ``omega`` a block of angular frequencies;
    every quantity below is one of Foster's, evaluated for all frequencies of
    the block at once. Returns ``(wwz, wwa, n_eff)``, each of shape
    ``omega.shape``, with ``wwz``/``wwa`` set to NaN on degenerate cells and
    ``n_eff`` always finite (it is the diagnostic that explains the mask).
    """
    arg = omega[:, None] * dt[None, :]
    w = np.exp(-c * arg * arg)
    cos_a = np.cos(arg)
    sin_a = np.sin(arg)
    del arg
    wc = w * cos_a
    ws = w * sin_a

    sum_w = w.sum(axis=1)
    sum_w2 = (w * w).sum(axis=1)
    # Degenerate cells divide by zero otherwise; they are masked out below.
    safe_w = np.where(sum_w > 0.0, sum_w, 1.0)
    safe_w2 = np.where(sum_w2 > 0.0, sum_w2, 1.0)
    n_eff = np.where(sum_w2 > 0.0, sum_w * sum_w / safe_w2, 0.0)

    # The 3x3 Gram matrix S_ab = <phi_a|phi_b> (S_11 = <1|1> = 1 identically).
    s12 = wc.sum(axis=1) / safe_w
    s13 = ws.sum(axis=1) / safe_w
    s22 = (wc * cos_a).sum(axis=1) / safe_w
    s23 = (wc * sin_a).sum(axis=1) / safe_w
    s33 = (ws * sin_a).sum(axis=1) / safe_w

    # The right-hand side <phi_a|x>, and <x|x> for the data variation.
    b1 = (w @ value) / safe_w
    b2 = (wc @ value) / safe_w
    b3 = (ws @ value) / safe_w
    xx = (w @ (value * value)) / safe_w

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
    # so V_y = y.b - <1|x>^2 and V_x - V_y = <x|x> - y.b with no cancellation
    # of the (large) squared mean in the residual variation.
    ydotb = y1 * b1 + y2 * b2 + y3 * b3
    v_y = ydotb - b1 * b1
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
        whether the automatic grid was capped, whether the axes were supplied)
        plus metadata carried from the light curve.

    Notes
    -----
    Large ``wwz`` means "a coherent sinusoid of this frequency was present near
    this epoch", but *only* relative to the rest of the map — see the module
    docstring on why the nominal F-distribution does not apply.
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

    def _peak_indices(self) -> list[int]:
        """Column of the largest WWZ per ``tau``; ``-1`` for a fully masked row."""
        out: list[int] = []
        for row in self.wwz:
            finite = np.isfinite(row)
            out.append(int(np.nanargmax(row)) if finite.any() else -1)
        return out

    def peak_frequency(self) -> FloatArray:
        """Per-``tau`` ridge frequency in cycles/day (NaN on a masked row).

        The largest WWZ cell of each row, parabolically refined against its two
        neighbours when the peak is interior. The refinement uses the *local*
        grid spacing, which is what the geometric default frequency axis needs.
        """
        nf = int(self.frequency.size)
        out = np.full(int(self.tau.size), np.nan, dtype=np.float64)
        for i, j in enumerate(self._peak_indices()):
            if j < 0:
                continue
            f_best = float(self.frequency[j])
            if 0 < j < nf - 1:
                offset = _parabolic_offset(
                    float(self.wwz[i, j - 1]),
                    float(self.wwz[i, j]),
                    float(self.wwz[i, j + 1]),
                )
                neighbor = j + 1 if offset >= 0.0 else j - 1
                step = abs(float(self.frequency[neighbor]) - float(self.frequency[j]))
                f_best += offset * step
            out[i] = f_best
        return out

    def peak_period(self) -> FloatArray:
        """Per-``tau`` ridge period in days (``1/peak_frequency()``)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(1.0 / self.peak_frequency(), dtype=np.float64)

    def peak_wwa(self) -> FloatArray:
        """Semi-amplitude at each ``tau``'s ridge cell (NaN on a masked row).

        Read at the peak *cell*, not at the parabolically refined frequency —
        the amplitude varies far more slowly across the window's resolution
        element than the statistic does.
        """
        out = np.full(int(self.tau.size), np.nan, dtype=np.float64)
        for i, j in enumerate(self._peak_indices()):
            if j >= 0:
                out[i] = float(self.wwa[i, j])
        return out

    def period_trend(self) -> tuple[float, float]:
        """Rough fractional period drift from the ridge, ``((dP/dt)/P, error)``.

        Fits an unweighted straight line to :meth:`peak_frequency` against
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
            Its standard error, from the scatter of the ridge about the line.

        Raises
        ------
        InsufficientDataError
            If fewer than three ``tau`` rows produced a ridge.

        Notes
        -----
        A rough *exploratory diagnostic*, exactly as
        :meth:`cuperiod.SlidingPeriodogram.period_trend` is, and with the same
        caveats sharpened: WWZ windows at neighbouring ``tau`` overlap unless
        the spacing exceeds a few ``sigma_t``, so consecutive ridge points are
        positively correlated and the quoted error is optimistic; and a star
        that *wanders* has no meaningful linear drift to fit in the first place
        — which is usually the very thing the map was made to show. For a
        number to publish, localize with :func:`cuperiod.pdot_search` and
        measure with :func:`cuperiod.phase_track`.
        """
        freq = self.peak_frequency()
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
        variance = float(np.dot(resid, resid)) / (n - 2)
        slope_err = math.sqrt(variance / sxx)
        return -slope / f_bar, slope_err / f_bar

    def to_dict(self, *, include_maps: bool = False) -> dict[str, Any]:
        """Serialize to a plain dict (the ridge always; the 2-D maps optional)."""
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
            "peak_frequency": self.peak_frequency().tolist(),
            "peak_period": self.peak_period().tolist(),
            "peak_wwa": self.peak_wwa().tolist(),
        }
        if include_maps:
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
    domain: Domain | None = None,
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
        wavelet window).
    frequency : array-like, optional
        Trial frequencies in cycles/day, used verbatim. ``None`` builds a
        geometric grid from ``2/baseline`` to the pseudo-Nyquist frequency,
        stepped at a quarter of the local resolution ``sqrt(2c)*f`` and capped
        at 2000 samples (recorded as ``grid_capped`` in ``meta``).
    tau : array-like, optional
        Window centres, in the input's time system, used verbatim. ``None``
        places ``n_tau`` of them uniformly over ``[t_min, t_max]``.
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
        appears in the numerator), not a sane analysis floor: 10-20 is a much
        better practical choice, and the ``n_eff`` map is always returned so
        the threshold can be revisited without recomputing.
    columns, domain
        Passed to :func:`cuperiod.to_input` for table inputs.

    Returns
    -------
    WWZResult
        The ``(n_tau, n_frequency)`` maps plus the per-``tau`` ridge.

    Raises
    ------
    ValueError
        For a multi-band input, a non-positive ``c``/``n_tau``, or non-finite /
        non-positive explicit axes.
    InsufficientDataError
        Fewer than 20 finite points, no usable baseline, or an automatic
        frequency range that collapsed.

    Examples
    --------
    >>> res = cup.wwz((t, mag, err), c=0.001)          # doctest: +SKIP
    >>> res.shape                                      # doctest: +SKIP
    >>> res.peak_period()[:3]                          # doctest: +SKIP
    >>> res.period_trend()   # rough (dP/dt)/P, 1/day  # doctest: +SKIP
    """
    from cuperiod.api import to_input

    if c <= 0.0:
        raise ValueError(f"c must be positive, got {c}")

    lc = to_input(data, columns=columns, domain=domain)
    if isinstance(lc, MultiBandLightCurve):
        raise ValueError(
            "wwz is single-band; pool or pick a band first (stack the bands "
            "into one LightCurve, or index lc.bands)"
        )
    work = lc.finite()
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
        freq_values = np.asarray(frequency, dtype=np.float64).ravel()
        if freq_values.size == 0:
            raise ValueError("wwz: frequency must hold at least one trial frequency")
        if not bool(np.all(np.isfinite(freq_values))) or bool(
            np.any(freq_values <= 0.0)
        ):
            raise ValueError("wwz: frequency values must be finite and positive")
    if tau is None:
        if n_tau < 1:
            raise ValueError(f"n_tau must be >= 1, got {n_tau}")
        tau_values = np.asarray(
            np.linspace(t_min, t_max, int(n_tau)), dtype=np.float64
        )
    else:
        tau_values = np.asarray(tau, dtype=np.float64).ravel()
        if tau_values.size == 0:
            raise ValueError("wwz: tau must hold at least one window centre")
        if not bool(np.all(np.isfinite(tau_values))):
            raise ValueError("wwz: tau values must be finite")

    # -- the map ------------------------------------------------------------------
    # Times are referenced to t_min before differencing: the subtraction is
    # exact either way, but this keeps the (nf, n) angle matrix free of the
    # large-absolute-JD magnitudes.
    dt_all = time - t_min
    tau_shift = tau_values - t_min
    value = work.value
    omega = 2.0 * np.pi * freq_values
    nt, nf = int(tau_values.size), int(freq_values.size)
    wwz_map = np.empty((nt, nf), dtype=np.float64)
    wwa_map = np.empty((nt, nf), dtype=np.float64)
    neff_map = np.empty((nt, nf), dtype=np.float64)
    chunk = max(1, min(nf, _CHUNK_ELEMS // max(1, n_samples)))
    for i in range(nt):
        dt = dt_all - float(tau_shift[i])
        for start in range(0, nf, chunk):
            stop = min(start + chunk, nf)
            z, a, ne = _wwz_block(dt, value, omega[start:stop], c, min_neff)
            wwz_map[i, start:stop] = z
            wwa_map[i, start:stop] = a
            neff_map[i, start:stop] = ne

    meta = {
        "wwz": {
            "c": c,
            "min_neff": min_neff,
            "n_tau": nt,
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
        c=float(c),
        n_samples=n_samples,
        baseline=baseline,
        meta=meta,
    )


__all__ = ["WWZResult", "wwz"]
