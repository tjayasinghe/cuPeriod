"""Sliding-window (dynamic) periodograms: :func:`sliding_periodogram`.

A *dynamic* periodogram is the ordinary periodogram evaluated on a window that
slides along the light curve: one row of the method's statistic per window,
stacked into a ``(n_windows, n_frequency)`` map whose vertical axis is time. It
answers the question a single spectrum cannot — *when* was the star doing that?

Typical uses:

- **Mira / semi-regular period wander.** These stars do not have one period;
  they have a period that meanders by a few percent from cycle to cycle. A
  static periodogram averages the wander into one broad blob, while the sliding
  map shows the ridge moving.
- **Glitches and mode transitions.** An RR Lyrae that jumps period, a pulsar-like
  glitch, a mode switch: a discontinuous step in the ridge is unmistakable in
  the map and invisible in the integrated spectrum.
- **Sanity-checking a drift detection.** :func:`cuperiod.pdot_search` reports a
  linear dP/dt; the map shows whether the ridge really slopes steadily (a true
  drift) or wanders, jumps, or is driven by one noisy stretch of data.

**The window trade-off is unavoidable.** A window of length ``W`` resolves
frequencies to about ``1/W`` and time to about ``W``: halving the window doubles
the time resolution and doubles the frequency blur. There is no setting that
gives both — pick the window from the science (how fast can the period plausibly
change?), not from the plot. The default, ``baseline/12``, is a compromise that
keeps roughly a dozen windows across the run; overlapping them (the default
``step = window/2``) smooths the map without adding independent information.

Every window is evaluated on the *same* frequency grid, so the result is a
rectangular map, and that grid is built at the *window's* resolution
(``1/(5*window)``), not the baseline's — a baseline-resolution grid would
oversample each window's much broader peak by the ratio of the two spans.

:func:`sliding_periodogram` is an exploratory tool. Its :meth:`SlidingPeriodogram.
period_trend` is a coarse drift diagnostic; for a *measurement* of dP/dt use
:func:`cuperiod.pdot_search` to localize and :func:`cuperiod.phase_track` to
measure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic_settings import BaseSettings

from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import (
    LightCurve,
    MultiBandLightCurve,
    raise_if_unusable,
)
from cuperiod.core.result import ObjectiveSense, Periodogram
from cuperiod.methods.base import get_method
from cuperiod.pdot.scan import (
    _UNSUPPORTED,
    _default_grid,
    _parabolic_offset,
    _row_power,
    _row_settings,
    weighted_epoch,
)

#: Windows across the baseline when ``window`` is not given (window = T/12).
_WINDOWS_PER_BASELINE: float = 12.0

#: Frequency oversampling of the shared grid: step ``1/(_SAMPLES_PER_PEAK*window)``.
_SAMPLES_PER_PEAK: float = 5.0

#: Oversampling of the narrow band built around an explicit ``period``.
_BAND_SAMPLES_PER_PEAK: float = 10.0

#: Half-width of that band, in window-Rayleigh widths (``1/window``).
_BAND_RAYLEIGH: float = 6.0

#: Largest shared grid the default-bounds path builds before coarsening the step.
_MAX_GRID_POINTS: int = 200_000

#: Guard against a pathologically small ``step`` producing millions of windows.
_MAX_WINDOWS: int = 10_000


def _window_starts(
    t_min: float, t_max: float, window: float, step: float
) -> FloatArray:
    """Window start times: uniform stepping plus a flush-right final window.

    Windows start at ``t_min + k*step`` for as long as they fit inside the
    baseline. If that leaves more than half a step of the tail uncovered, one
    final window ending exactly at ``t_max`` is appended, so at most half a
    step of the last epochs is left out of the map (none when the windows tile
    the baseline exactly, as the default window and step do; see
    :func:`_window_end` for the rounding of that last edge).
    """
    span = t_max - t_min
    tiny = 1e-9 * max(window, 1.0)
    count = int(math.floor((span - window + tiny) / step)) + 1
    if count < 1:
        raise InsufficientDataError(
            f"sliding_periodogram: a {window} d window does not fit in the "
            f"{span} d baseline; shorten window="
        )
    if count > _MAX_WINDOWS:
        raise ValueError(
            f"sliding_periodogram: step={step} d gives {count} windows over a "
            f"{span} d baseline (limit {_MAX_WINDOWS}); raise step="
        )
    starts = [t_min + step * k for k in range(count)]
    if t_max - (starts[-1] + window) > 0.5 * step:
        starts.append(t_max - window)
    return np.asarray(starts, dtype=np.float64)


def _band_grid(f_center: float, window: float) -> GridSpec:
    """A narrow uniform band around ``f_center``, at the window's resolution.

    Mirrors the window construction of :func:`cuperiod.pdot_search` with the
    *window* (not the baseline) as the resolution scale: half-width
    ``6/window`` cycles/day, step ``1/(10*window)``. A period longer than the
    window (under one cycle per window) is refused: nothing can be fitted in
    such a window, and past ``20*window`` the clipped band would not even
    contain ``f_center`` any more.
    """
    if not f_center * window >= 1.0 - 1e-12:  # period == window rounds either way
        raise ValueError(
            f"sliding_periodogram: period={1.0 / f_center} d is too long for this "
            f"window ({window} d holds under one cycle); lengthen window= or drop "
            "period="
        )
    df = 1.0 / (_BAND_SAMPLES_PER_PEAK * window)
    k = max(1, int(math.ceil(_BAND_RAYLEIGH / window / df)))
    values = f_center + df * np.arange(-k, k + 1, dtype=np.float64)
    values = values[values > 0.5 * df]
    if values.size < 3:
        raise InsufficientDataError(
            f"sliding_periodogram: frequency band around {f_center} cycles/day "
            "collapsed; the period is too long for this window"
        )
    return GridSpec(kind="frequency", values=values, uniform=True)


def _shared_grid(f_lo: float, f_hi: float, window: float) -> tuple[GridSpec, bool]:
    """Uniform grid over ``[f_lo, f_hi]`` at the window's own resolution.

    Returns ``(grid, coarsened)``; ``coarsened`` is True when the step had to be
    widened to keep the map under :data:`_MAX_GRID_POINTS` columns.
    """
    if not 0.0 < f_lo < f_hi:
        raise InsufficientDataError(
            f"sliding_periodogram: unusable frequency bounds ({f_lo}, {f_hi}); "
            "pass period= or grid="
        )
    df = 1.0 / (_SAMPLES_PER_PEAK * window)
    nf = int(math.ceil((f_hi - f_lo) / df)) + 1
    coarsened = nf > _MAX_GRID_POINTS
    if coarsened:
        nf = _MAX_GRID_POINTS
        df = (f_hi - f_lo) / (nf - 1)
    values = f_lo + df * np.arange(nf, dtype=np.float64)
    return GridSpec(kind="frequency", values=values, uniform=True), coarsened


def _window_end(lo: float, window: float, t_max: float, last: bool) -> float:
    """Right edge of the window starting at ``lo``.

    ``lo + window``, except that the final window, which is meant to reach
    ``t_max`` (it is closed on the right for exactly that reason), is snapped
    onto ``t_max`` when the sum falls short of it by rounding alone: at JD
    magnitudes ``t_min + k*step + window`` lands one ulp below ``t_max`` in
    about 2% of light curves, which used to drop the final observation. A
    genuine shortfall (the tail of at most half a step that the placement
    leaves uncovered by design) is not stretched over.
    """
    hi = lo + window
    if last and hi < t_max:
        tol = max(1e-9 * max(window, 1.0), 4.0 * float(np.spacing(abs(t_max))))
        if t_max - hi <= tol:
            return t_max
    return hi


def _cut(lc: LightCurve, lo: float, hi: float, closed: bool) -> LightCurve:
    """The points of ``lc`` in ``[lo, hi)`` (``[lo, hi]`` when ``closed``)."""
    mask = (lc.time >= lo) & ((lc.time <= hi) if closed else (lc.time < hi))
    return LightCurve(
        time=lc.time[mask],
        value=lc.value[mask],
        error=None if lc.error is None else lc.error[mask],
        domain=lc.domain,
        meta=lc.meta,
    )


def _slice_window(
    work: LightCurve | MultiBandLightCurve, lo: float, hi: float, closed: bool
) -> tuple[LightCurve | MultiBandLightCurve | None, int]:
    """One window's light curve and its pooled point count.

    Multi-band inputs are sliced band by band on the same time range; bands with
    no points in the window are dropped. Returns ``(None, 0)`` for a window that
    holds no points at all.
    """
    if isinstance(work, MultiBandLightCurve):
        bands: dict[str, LightCurve] = {}
        total = 0
        for name, band in work.bands.items():
            cut = _cut(band, lo, hi, closed)
            if cut.n == 0:
                continue
            bands[name] = cut
            total += cut.n
        if not bands:
            return None, 0
        return MultiBandLightCurve.from_light_curves(bands, meta=work.meta), total
    cut = _cut(work, lo, hi, closed)
    if cut.n == 0:
        return None, 0
    return cut, cut.n


@dataclass(frozen=True)
class SlidingPeriodogram:
    """A time-resolved (dynamic) periodogram: one spectrum per time window.

    Attributes
    ----------
    method, backend : str
        The wrapped periodogram method and the concrete backend that ran it.
    frequency : numpy.ndarray
        Frequency axis (cycles/day), ascending, shared by every window — this is
        what makes the map rectangular.
    t_mid : numpy.ndarray
        Per-window inverse-variance-weighted mean epoch (same time system as the
        input). The natural time coordinate of each row.
    t_start : numpy.ndarray
        Per-window start time; the window covers ``[t_start, t_start + window)``.
    power : numpy.ndarray
        The method's statistic, shape ``(t_mid.size, frequency.size)``.
    peak_frequency, peak_power : numpy.ndarray
        Per-window best frequency (parabolically refined when the best cell is
        interior) and the raw statistic at that cell.
    objective_sense : {"max", "min"}
        Whether significance means large or small ``power``.
    window, step : float
        Window length and window-start spacing, in days.
    n_samples : int
        Number of finite light-curve points in the whole input.
    baseline : float
        Full time baseline in days (not the window length).
    meta : Mapping
        Run provenance under the ``"sliding_periodogram"`` key (window/step, the
        skipped-window count, whether the grid was coarsened) plus metadata
        carried from the light curve.
    n_window_points : numpy.ndarray or None
        Points in each row's window (pooled across bands), aligned with
        ``t_mid``. ``None`` only on a result built by hand without it.
    """

    method: str
    backend: str
    frequency: FloatArray
    t_mid: FloatArray
    t_start: FloatArray
    power: FloatArray
    peak_frequency: FloatArray
    peak_power: FloatArray
    objective_sense: ObjectiveSense
    window: float
    step: float
    n_samples: int
    baseline: float
    meta: Mapping[str, Any] = field(default_factory=dict)
    n_window_points: IntArray | None = None

    @property
    def n_windows(self) -> int:
        """Number of usable windows (rows of the map)."""
        return int(self.t_mid.size)

    @property
    def shape(self) -> tuple[int, int]:
        """Map shape ``(n_windows, n_frequency)``."""
        return (int(self.t_mid.size), int(self.frequency.size))

    @property
    def period(self) -> FloatArray:
        """The frequency axis as periods (days), aligned with ``frequency``."""
        with np.errstate(divide="ignore"):
            return 1.0 / self.frequency

    @property
    def peak_period(self) -> FloatArray:
        """Per-window best period in days (``1/peak_frequency``)."""
        with np.errstate(divide="ignore"):
            return 1.0 / self.peak_frequency

    def window_periodogram(self, i: int) -> Periodogram:
        """Row ``i`` of the map as an ordinary :class:`Periodogram`.

        Parameters
        ----------
        i : int
            Window index (negative indices count from the end).

        Returns
        -------
        Periodogram
            The window's spectrum, so the usual peak finding, plotting and
            serialization apply. Its ``baseline`` is the *window* length and its
            ``n_samples`` the window's point count: the Rayleigh resolution of
            a window row is ``1/window``, and peak separation in
            :meth:`~cuperiod.Periodogram.best_periods` is measured in it (with
            the run's baseline, 12x longer by default, the main peak's own
            sidelobes passed as independent periods). The window's identity is
            in ``meta`` (``window_index``, ``t_start``, ``window``), with the
            run's values as ``run_baseline`` and ``run_n_samples``.
        """
        meta = {
            **dict(self.meta),
            "window_index": int(i),
            "t_start": float(self.t_start[i]),
            "window": self.window,
            "run_baseline": self.baseline,
            "run_n_samples": self.n_samples,
        }
        n_points = (
            self.n_samples
            if self.n_window_points is None
            else int(self.n_window_points[i])
        )
        return Periodogram.from_spectrum(
            method=self.method,
            backend=self.backend,
            frequency=self.frequency,
            power=self.power[i],
            objective_sense=self.objective_sense,
            n_samples=n_points,
            baseline=self.window,
            meta=meta,
        )

    def period_trend(self) -> tuple[float, float]:
        """Rough fractional period drift from the per-window peaks.

        Fits an unweighted straight line to ``peak_frequency`` against
        ``t_mid - mean(t_mid)``. A slope ``s`` (cycles/day per day) is a
        fractional frequency drift ``s / f_bar``, so the fractional *period*
        rate is ``(dP/dt)/P = -s / f_bar`` in 1/day (``f_bar`` = mean peak
        frequency). Multiply by the period to get dP/dt in day/day.

        Returns
        -------
        pdot_over_p : float
            Fractional period-change rate (dP/dt)/P in 1/day.
        pdot_over_p_err : float
            Its standard error, from the scatter of the peaks about the line.

        Raises
        ------
        InsufficientDataError
            If fewer than three windows survived.

        Notes
        -----
        This is a rough *exploratory diagnostic*, not a measurement. It ignores
        the per-window peak uncertainties; it is only as good as the window
        peaks (a wandering or multi-periodic star has no meaningful linear
        drift at all); and when ``step < window`` the windows overlap, so
        consecutive points share data, are positively correlated, and the
        quoted error is optimistic. For a number to publish, localize with
        :func:`cuperiod.pdot_search` and measure with
        :func:`cuperiod.phase_track`.
        """
        n = int(self.t_mid.size)
        if n < 3:
            raise InsufficientDataError(
                f"period_trend: need at least 3 windows, got {n}"
            )
        x = self.t_mid - float(np.mean(self.t_mid))
        y = self.peak_frequency
        f_bar = float(np.mean(y))
        sxx = float(np.dot(x, x))
        if sxx <= 0.0 or f_bar <= 0.0:
            raise InsufficientDataError(
                "period_trend: degenerate window epochs or peak frequencies"
            )
        slope = float(np.dot(x, y - np.mean(y))) / sxx
        resid = y - (float(np.mean(y)) + slope * x)
        variance = float(np.dot(resid, resid)) / (n - 2)
        slope_err = math.sqrt(variance / sxx)
        return -slope / f_bar, slope_err / f_bar

    def to_dict(self, *, include_map: bool = False) -> dict[str, Any]:
        """Serialize to a plain dict (per-window peaks always; the map optional)."""
        out: dict[str, Any] = {
            "method": self.method,
            "backend": self.backend,
            "objective_sense": self.objective_sense,
            "window": self.window,
            "step": self.step,
            "n_windows": self.n_windows,
            "n_samples": self.n_samples,
            "baseline": self.baseline,
            "t_mid": self.t_mid.tolist(),
            "t_start": self.t_start.tolist(),
            "peak_frequency": self.peak_frequency.tolist(),
            "peak_power": self.peak_power.tolist(),
        }
        if self.n_window_points is not None:
            out["n_window_points"] = self.n_window_points.tolist()
        if include_map:
            out["frequency"] = self.frequency.tolist()
            out["power"] = self.power.tolist()
        return out


def sliding_periodogram(
    data: Any,
    *,
    method: str = "GLS",
    window: float | None = None,
    step: float | None = None,
    period: float | None = None,
    grid: GridSpec | None = None,
    method_settings: BaseSettings | None = None,
    backend: str = "auto",
    min_window_points: int = 20,
    columns: ColumnMap | None = None,
    domain: Domain | str | None = None,
) -> SlidingPeriodogram:
    """Evaluate a periodogram on a window that slides along the light curve.

    Every window is run on one shared frequency grid, so the rows stack into a
    rectangular ``(n_windows, n_frequency)`` map of the method's statistic
    against time — the dynamic periodogram. Any registered method works (GLS,
    PDM, CE, String-Length, MHAOV, SuperSmoother; single- or multi-band); the
    method's batch engine and resolved backend are built once and reused for
    every window. See the module docstring for the window/resolution trade-off.

    Parameters
    ----------
    data : various
        The light curve, in any form :func:`cuperiod.to_input` accepts.
    method : str, default "GLS"
        Wrapped periodogram method. BLS and TLS are not supported (their trial
        grids carry box/transit structure a shared frequency axis cannot
        represent); use :func:`cuperiod.phase_track` /
        :func:`cuperiod.fit_ephemeris` for transit-timing style work.
    window : float, optional
        Window length in days (positive and finite). ``None`` uses
        ``baseline/12``.
    step : float, optional
        Spacing of window *starts* in days (positive and finite). ``None`` uses
        ``window/2`` (50% overlap). ``step >= window`` gives disjoint windows.
    period : float, optional
        Center a narrow frequency band on ``1/period`` instead of scanning the
        method's full default range: half-width ``6/window`` cycles/day, step
        ``1/(10*window)``. Much cheaper, and the right choice when you already
        know roughly where the signal is. Must be positive, finite and no
        longer than the window (a window has to hold at least one cycle).
    grid : GridSpec, optional
        Explicit shared frequency grid, overriding both constructions. A uniform
        frequency grid keeps the fast NUFFT path for GLS. The map's axis is the
        grid's frequencies in ascending order, which is the order every row is
        computed in (so a period grid's columns run from its longest period to
        its shortest).
    method_settings : pydantic settings, optional
        The wrapped method's own settings (e.g. ``PDMSettings(n_bins=15)``).
        Per-window false-alarm probabilities are switched off.
    backend : str, default "auto"
        Backend request, resolved once and reused for every window. ``"auto"``
        defers to ``method_settings.backend`` (and its
        ``CUPERIOD_<METHOD>_BACKEND`` override), as in
        :func:`cuperiod.periodogram`.
    min_window_points : int, default 20
        Windows holding fewer points (pooled across bands) are skipped; so are
        windows the method itself rejects as too sparse. The count lands in
        ``meta["sliding_periodogram"]["n_windows_skipped"]``.
    columns, domain
        Passed to :func:`cuperiod.to_input` for table inputs.

    Returns
    -------
    SlidingPeriodogram
        The time-frequency map plus the per-window refined peaks.

    Raises
    ------
    ValueError
        For an unsupported method, a multi-band input to a single-band method,
        a non-positive or non-finite ``window``/``step``/``period``, or a
        ``period`` longer than the window.
    InsufficientDataError
        If the input has no usable baseline, the window does not fit in it, or
        fewer than two windows survive the point-count cuts.

    Examples
    --------
    >>> res = sliding_periodogram((t, mag, err), window=120.0)   # doctest: +SKIP
    >>> res.shape                                                # doctest: +SKIP
    >>> res.peak_period[:3]                                      # doctest: +SKIP
    >>> res.period_trend()          # rough (dP/dt)/P and its error  # doctest: +SKIP
    >>> res.window_periodogram(0).best_period()                  # doctest: +SKIP
    """
    from cuperiod.api import to_input

    m = get_method(method)
    if m.name in _UNSUPPORTED:
        raise ValueError(
            f"sliding_periodogram does not support {m.name} (its trial grid "
            "carries box/transit structure that a shared frequency axis cannot "
            "represent); slide GLS/PDM/CE, or measure a transit-timing dP/dt "
            "with phase_track / fit_ephemeris"
        )
    msettings = m.coerce_settings(method_settings)
    resolved = m.resolve_backend_for(backend, msettings)

    lc = to_input(data, columns=columns, domain=domain)
    # A placeholder error column (all 0 or NaN) would otherwise empty the curve
    # in finite() below and be reported as 'no finite points'.
    raise_if_unusable(lc, "sliding_periodogram")
    work: LightCurve | MultiBandLightCurve
    if isinstance(lc, MultiBandLightCurve):
        if not m.supports_multiband:
            raise ValueError(f"{m.name} does not support multi-band input")
        work = lc.finite()
        time, _, _, _ = work.stacked()
    else:
        single = lc.in_domain(m.natural_domain) if m.natural_domain else lc
        work = single.finite()
        time = work.time
    n_samples = int(time.size)
    if n_samples == 0:
        raise InsufficientDataError("sliding_periodogram: no finite points")
    t_min, t_max = float(time.min()), float(time.max())
    baseline = t_max - t_min
    if baseline <= 0.0:
        raise InsufficientDataError("sliding_periodogram: no usable time baseline")

    # A bare `x <= 0` lets NaN through and inf on to a division or a floor.
    win = baseline / _WINDOWS_PER_BASELINE if window is None else float(window)
    if not (math.isfinite(win) and win > 0.0):
        raise ValueError(f"window must be positive and finite, got {win}")
    stride = 0.5 * win if step is None else float(step)
    if not (math.isfinite(stride) and stride > 0.0):
        raise ValueError(f"step must be positive and finite, got {stride}")
    # A NaN floor skipped no window (every comparison is False) and an infinite one
    # every window, reported as "too few usable windows".
    if not (
        math.isfinite(min_window_points)
        and float(min_window_points).is_integer()
        and min_window_points >= 0
    ):
        raise ValueError(
            f"min_window_points must be a non-negative integer, got "
            f"{min_window_points}"
        )

    row_settings = _row_settings(msettings)

    # -- the shared frequency axis --------------------------------------------------
    coarsened = False
    if grid is not None:
        fgrid = grid
    elif period is not None:
        p = float(period)
        if not (math.isfinite(p) and p > 0.0):
            raise ValueError(f"period must be positive and finite, got {p}")
        fgrid = _band_grid(1.0 / p, win)
    else:
        bounds = _default_grid(m, work, row_settings).frequency
        if bounds.size == 0:
            raise InsufficientDataError(
                f"sliding_periodogram: {m.name} produced an empty default grid"
            )
        fgrid, coarsened = _shared_grid(float(bounds[0]), float(bounds[-1]), win)
    # Every row comes back from Periodogram.from_spectrum sorted by ascending
    # frequency, so that is the axis the rows live on. Taking the grid in its
    # stored order instead mirrored every row of a descending grid onto its axis
    # (GridSpec now refuses descending grids; sorting keeps the invariant local).
    frequency = np.sort(np.asarray(fgrid.frequency, dtype=np.float64))
    if frequency.size < 2:
        raise InsufficientDataError(
            f"sliding_periodogram: the shared grid has {frequency.size} samples"
        )

    # -- slide ----------------------------------------------------------------------
    starts = _window_starts(t_min, t_max, win, stride)
    engine = m.make_engine(resolved, row_settings)
    rows: list[FloatArray] = []
    t_mids: list[float] = []
    used_starts: list[float] = []
    used_points: list[int] = []
    n_skipped = 0
    backend_actual = resolved
    last = int(starts.size) - 1
    for k in range(starts.size):
        lo = float(starts[k])
        hi = _window_end(lo, win, t_max, k == last)
        sliced, n_points = _slice_window(work, lo, hi, k == last)
        if sliced is None or n_points < min_window_points:
            n_skipped += 1
            continue
        try:
            pg = _row_power(m, fgrid, sliced, row_settings, resolved, engine)
        except InsufficientDataError:
            # The method's own floor (min_detections, a collapsed fold, ...):
            # a sparse window is a skipped window, not a failed run.
            n_skipped += 1
            continue
        if pg.power.size != frequency.size:
            raise RuntimeError(
                f"sliding_periodogram: {m.name} returned {pg.power.size} samples "
                f"for a {frequency.size}-point grid"
            )
        if not rows:
            backend_actual = pg.backend
        rows.append(np.asarray(pg.power, dtype=np.float64))
        t_mids.append(weighted_epoch(sliced))
        used_starts.append(lo)
        used_points.append(n_points)
    if len(rows) < 2:
        raise InsufficientDataError(
            f"sliding_periodogram: only {len(rows)} usable window(s) "
            f"({n_skipped} skipped) with window={win} d, step={stride} d, "
            f"min_window_points={min_window_points}; shorten the window, "
            "shorten the step, or lower min_window_points"
        )

    # -- per-window peaks, parabolically refined ------------------------------------
    power = np.asarray(rows, dtype=np.float64)
    score = power if m.objective_sense == "max" else -power
    n_windows, nf = power.shape
    peak_frequency = np.empty(n_windows, dtype=np.float64)
    peak_power = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        j = int(np.nanargmax(score[i]))
        f_best = float(frequency[j])
        if 0 < j < nf - 1:
            offset = _parabolic_offset(
                float(score[i, j - 1]), float(score[i, j]), float(score[i, j + 1])
            )
            # Local step: exact on a uniform grid, sane on a user-supplied one.
            # Signed toward the neighbour the parabola leans to (``offset``
            # counts index steps), so it stays right whatever the axis order.
            neighbor = j + 1 if offset >= 0.0 else j - 1
            step_f = float(frequency[neighbor]) - float(frequency[j])
            f_best += abs(offset) * step_f
        peak_frequency[i] = f_best
        peak_power[i] = float(power[i, j])

    meta = {
        "sliding_periodogram": {
            "window": win,
            "step": stride,
            "min_window_points": min_window_points,
            "n_windows_skipped": n_skipped,
            "n_windows_placed": int(starts.size),
            "grid_coarsened": coarsened,
            "period": period,
            "explicit_grid": grid is not None,
        },
        **dict(work.meta),
    }
    return SlidingPeriodogram(
        method=m.name,
        backend=backend_actual,
        frequency=frequency,
        t_mid=np.asarray(t_mids, dtype=np.float64),
        t_start=np.asarray(used_starts, dtype=np.float64),
        power=power,
        peak_frequency=peak_frequency,
        peak_power=peak_power,
        objective_sense=m.objective_sense,
        window=win,
        step=stride,
        n_samples=n_samples,
        baseline=baseline,
        meta=meta,
        n_window_points=np.asarray(used_points, dtype=np.int64),
    )


__all__ = ["SlidingPeriodogram", "sliding_periodogram"]
