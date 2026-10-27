"""Diagnostic figures for the period-change (dP/dt) tools.

A period-change claim is never believable from one number. The rate that
:func:`cuperiod.pdot_search` localizes, the O-C parabola
:func:`cuperiod.phase_track` fits, the model-free ``P(t)`` track
:func:`cuperiod.period_profile` measures and the law
:func:`cuperiod.fit_period_law` puts through it are four partly independent
views of the same drift, and the failure modes that matter — a degenerate
search ridge, a slipped cycle count, a period pinned at the edge of its search
window, a rate that is itself accelerating — are visible in the pictures long
before they show up in a summary table. :func:`pdot_figure` puts whichever of
those views the caller has produced side by side on one page.

Every panel is also exposed on its own, taking an existing ``ax``, so the
figure can be recomposed rather than reproduced. The panel helpers assume
nothing about how the axes were made: they draw with the artist API only and
never touch pyplot's global state, which is also why :func:`pdot_figure` builds
its :class:`~matplotlib.figure.Figure` directly and attaches an Agg canvas —
the whole module is safe to call from a worker process and never opens a
window.

matplotlib is an optional dependency (``pip install 'cuperiod[plot]'``); it is
imported lazily, so importing cuPeriod does not require it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from cuperiod.core._coerce import as_float_array
from cuperiod.core._typing import FloatArray
from cuperiod.pdot.laws import PeriodLawFit
from cuperiod.pdot.ltte import LTTEFit
from cuperiod.pdot.models import get_model
from cuperiod.pdot.profile import PeriodProfile
from cuperiod.pdot.result import SECONDS_PER_YEAR, PdotSearchResult
from cuperiod.pdot.track import PhaseTrackResult

if TYPE_CHECKING:  # matplotlib stays optional: types only, never imported here.
    from matplotlib.figure import Figure

#: Conventional epoch offset for JD/HJD/BJD axes, so tick labels stay legible.
DEFAULT_T_OFFSET: Final[float] = 2450000.0

#: Hours per day, for O-C axes (the unit eclipse timings are published in).
_HOURS_PER_DAY: Final[float] = 24.0

#: Above this many points, per-point error bars stop being readable and start
#: dominating the render time of a survey light curve; plain markers instead.
_MAX_ERRORBAR_POINTS: Final[int] = 3000

#: Points on the dense curves overlaid on measured points.
_CURVE_SAMPLES: Final[int] = 400


# --- small shared helpers -----------------------------------------------------


def _floats(values: Any, *, time: bool = False) -> FloatArray:
    """``values`` as a flat float64 array, masked/missing entries as NaN.

    Goes through the library's one input coercion, so a numpy or astropy mask
    is honoured (a bare ``np.asarray`` returns the placeholder *under* the mask,
    which every panel would then draw as data) and pandas/pyarrow nulls become
    NaN; ``time=True`` also reads calendar times as the light-curve loaders do.
    The result may share memory with the caller's array; nothing here writes
    to it.
    """
    return as_float_array(values, name="plot values", time=time).ravel()


def _time_offset(time: FloatArray) -> float:
    """Epoch offset for a time axis: the JD convention, or none.

    Subtracting 2450000 from times that are not JD-like (MJD, TESS BTJD, or a
    synthetic series starting at 0) produces meaningless negative labels, so the
    offset is only applied when the data actually live past it.
    """
    if time.size == 0:
        return 0.0
    lo = float(np.nanmin(time))
    return DEFAULT_T_OFFSET if np.isfinite(lo) and lo >= DEFAULT_T_OFFSET else 0.0


def _time_label(t_offset: float) -> str:
    """Axis label for a time axis carrying ``t_offset``."""
    return "time (d)" if t_offset == 0.0 else f"time - {t_offset:.0f} (d)"


def _empty(ax: Any, message: str) -> None:
    """Turn ``ax`` into a labelled blank rather than raising.

    A diagnostic figure is usually built in a batch loop over many stars, where
    one degenerate light curve must not take the run down with it.
    """
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
        color="0.45",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _usable_errors(error: Any, n: int) -> FloatArray | None:
    """Error bars ready for ``errorbar``, or ``None`` when there are none.

    Non-finite and non-positive entries become zero-length bars instead of
    poisoning the axis limits; an array with nothing usable in it is dropped
    entirely so the points are still drawn.
    """
    if error is None:
        return None
    err = _floats(error)
    if err.size != n:
        return None
    good = np.isfinite(err) & (err > 0.0)
    if not bool(good.any()):
        return None
    return np.where(good, np.abs(err), 0.0)


def _dense_grid(lo: float, hi: float) -> FloatArray | None:
    """A plotting grid spanning ``[lo, hi]``, or ``None`` if that is degenerate."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    return np.linspace(lo, hi, _CURVE_SAMPLES, dtype=np.float64)


def _as_arrays(
    light_curve: Any,
) -> tuple[FloatArray, FloatArray, FloatArray | None, bool] | None:
    """``(time, value, error, invert)`` from anything cuPeriod accepts.

    Recognizes a ``(time, value[, error])`` tuple and a
    :class:`~cuperiod.core.lightcurve.LightCurve` directly, and falls back to
    :func:`cuperiod.to_input` for tables and files. The trailing flag says
    whether the y axis should be flipped, taken from the light curve's declared
    brightness domain when there is one and assumed to be magnitudes otherwise.
    Non-finite and masked samples are dropped here so no panel has to repeat
    the check.
    """
    if light_curve is None:
        return None
    invert = True
    obj = light_curve
    if not (hasattr(obj, "time") and hasattr(obj, "value")):
        if isinstance(obj, tuple | list) and 2 <= len(obj) <= 3:
            time, value = _floats(obj[0], time=True), _floats(obj[1])
            error = _floats(obj[2]) if len(obj) == 3 and obj[2] is not None else None
            return _finite_curve(time, value, error, invert)
        from cuperiod.api import to_input

        obj = to_input(obj)
    if not (hasattr(obj, "time") and hasattr(obj, "value")):
        raise TypeError(
            "light_curve must be a (time, value, error) tuple, a LightCurve, or "
            "anything cuperiod.to_input accepts (a single band, not multi-band)"
        )
    domain = getattr(obj, "domain", None)
    if domain is not None:
        invert = str(domain) != "flux"
    error_attr = getattr(obj, "error", None)
    return _finite_curve(
        _floats(obj.time, time=True),
        _floats(obj.value),
        None if error_attr is None else _floats(error_attr),
        invert,
    )


def _finite_curve(
    time: FloatArray, value: FloatArray, error: FloatArray | None, invert: bool
) -> tuple[FloatArray, FloatArray, FloatArray | None, bool]:
    """Drop samples that cannot be plotted, keeping the three arrays aligned."""
    n = min(time.size, value.size)
    time, value = time[:n], value[:n]
    if error is not None and error.size != n:
        error = None
    keep = np.isfinite(time) & np.isfinite(value)
    return time[keep], value[keep], None if error is None else error[keep], invert


# --- individual panels --------------------------------------------------------


def plot_light_curve(
    ax: Any,
    time: Any,
    value: Any,
    error: Any = None,
    *,
    t_offset: float = DEFAULT_T_OFFSET,
    invert: bool = True,
) -> None:
    """Draw the raw light curve.

    The first thing to check on any period-change result: a secular brightness
    trend, a gap that splits the baseline, or a handful of outliers explain more
    spurious dP/dt detections than any subtlety of the fitting.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    time, value : array-like
        Observation epochs (days) and brightness.
    error : array-like, optional
        1-sigma errors. Drawn as bars only while they are still legible (up to a
        few thousand points); beyond that the markers alone are clearer.
    t_offset : float, default 2450000.0
        Subtracted from the epochs before plotting. Pass ``0.0`` for times that
        are not JD-like.
    invert : bool, default True
        Flip the y axis, the magnitude convention (brighter upwards). Pass
        ``False`` for fluxes.
    """
    t, y = _floats(time, time=True), _floats(value)
    n = min(t.size, y.size)
    t, y = t[:n], y[:n]
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    if t.size == 0:
        _empty(ax, "no finite light-curve points")
        return
    err = _usable_errors(error, keep.size)
    if err is not None:
        err = err[keep]
    if err is not None and t.size <= _MAX_ERRORBAR_POINTS:
        ax.errorbar(
            t - t_offset,
            y,
            yerr=err,
            fmt=".",
            ms=3,
            color="0.35",
            ecolor="0.75",
            elinewidth=0.6,
            capsize=0,
        )
    else:
        ax.plot(t - t_offset, y, ".", ms=2, color="0.35")
    if invert:
        ax.invert_yaxis()
    ax.set_xlabel(_time_label(t_offset))
    ax.set_ylabel("magnitude" if invert else "flux")
    ax.set_title(f"light curve ({t.size} points)")


def plot_phase_fold(
    ax: Any,
    time: Any,
    value: Any,
    *,
    period: float,
    pdot: float = 0.0,
    t_ref: float | None = None,
    model: Any = None,
    error: Any = None,
    ylabel: str = "magnitude",
) -> None:
    """Draw the phase fold on the *drifting* phase, coloured by epoch.

    Folding on a constant period smears a drifting light curve; folding on the
    warped time of the fitted period law un-smears it, which is the sharpest
    visual check that the measured rate is right. Points are coloured by epoch
    so that a *residual* drift — the part the rate did not remove — appears as a
    colour gradient sliding across the fold, which a monochrome fold hides
    completely.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    time, value : array-like
        Observation epochs (days) and brightness.
    period : float
        Period at ``t_ref`` (days).
    pdot : float, default 0.0
        Period derivative dP/dt (day/day). Zero gives the ordinary fold.
    t_ref : float, optional
        Epoch the period and rate refer to; defaults to the mean epoch. Use the
        ``t_ref`` of whatever produced ``pdot`` — the rate is only defined
        together with its reference epoch.
    model : str or PeriodModel, optional
        Period-evolution law used to warp the times
        (:mod:`cuperiod.pdot.models`); defaults to ``"constant-pdot"``.
    error : array-like, optional
        1-sigma errors, drawn faintly beneath the points.
    ylabel : str, default "magnitude"
        Label of the brightness axis (:func:`pdot_figure` passes ``"flux"`` for
        a flux light curve).

    Raises
    ------
    ValueError
        If ``model`` names no known period-evolution law.

    Notes
    -----
    The y axis is left as it is; call ``ax.invert_yaxis()`` for magnitudes (as
    :func:`pdot_figure` does). A ``constant-pdot`` warp is undefined once the
    period would pass through zero inside the span; if the given rate does that,
    the panel falls back to an unwarped fold rather than raising, and its title
    says so instead of quoting a rate it did not apply.
    """
    # Resolved before anything else, so a misspelt law is always an error
    # rather than a silently unwarped fold.
    law = get_model(model)
    t, y = _floats(time, time=True), _floats(value)
    n = min(t.size, y.size)
    t, y = t[:n], y[:n]
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    if t.size == 0 or not (np.isfinite(period) and period > 0.0):
        _empty(ax, "no fold (need finite points and a positive period)")
        return
    epoch = float(np.mean(t)) if t_ref is None else float(t_ref)
    dt = t - epoch
    rate = pdot / period if np.isfinite(pdot) else 0.0
    try:
        warped = law.warp(dt, rate)
        applied = True
    except ValueError:
        # The period would pass through zero inside the span (singular warp).
        warped = dt
        applied = False
    phase = np.mod(warped / period, 1.0)

    err = _usable_errors(error, keep.size)
    if err is not None:
        err = err[keep]
    if err is not None and t.size <= _MAX_ERRORBAR_POINTS:
        ax.errorbar(
            phase, y, yerr=err, fmt="none", ecolor="0.8", elinewidth=0.6, zorder=1
        )
    t0 = float(np.min(t))
    points = ax.scatter(
        phase, y, c=t - t0, s=7, cmap="viridis", linewidths=0, zorder=2
    )
    figure = ax.get_figure()
    if figure is not None:
        figure.colorbar(points, ax=ax, label=f"t - {t0:.0f} (d)")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("phase")
    ax.set_ylabel(ylabel)
    drift = "" if rate == 0.0 else f", dP/dt = {pdot * SECONDS_PER_YEAR:+.3g} s/yr"
    if drift and not applied:
        drift += " outside the model's domain (unwarped)"
    ax.set_title(f"fold at P = {period:.6g} d{drift}")


def plot_search_map(ax: Any, search: PdotSearchResult) -> None:
    """Draw the 2-D (dP/dt, frequency) statistic map with its best cell marked.

    The map is the only place a *degenerate* search shows itself: a peak drawn
    out into a diagonal ridge means the frequency and the rate trade off against
    each other, so the joint uncertainty is much wider than either marginal
    error bar suggests. The panel title carries the same thing as a number,
    :attr:`~cuperiod.PdotBest.ridge_correlation`, whenever the curvature at the
    peak allows it to be measured.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    search : PdotSearchResult
        The result of :func:`cuperiod.pdot_search`.

    Notes
    -----
    The rate axis is dimensionful dP/dt in seconds per year, obtained by scaling
    the search's fractional ``pdot_over_p`` coordinate by the best period; it is
    therefore exact along the best column and a good approximation across the
    narrow frequency span of a typical map.
    """
    best = search.best
    power = np.asarray(search.power, dtype=np.float64)
    if power.ndim != 2 or min(power.shape) < 2:
        _empty(ax, "search grid too small to map")
        return
    rate = search.pdot_over_p * SECONDS_PER_YEAR * best.period
    mesh = ax.pcolormesh(
        search.frequency, rate, power, shading="auto", cmap="viridis"
    )
    ax.plot(best.frequency, best.pdot_s_per_yr, "+", color="red", ms=12, mew=1.5)
    figure = ax.get_figure()
    if figure is not None:
        sense = "power" if search.objective_sense == "max" else "statistic"
        figure.colorbar(mesh, ax=ax, label=f"{search.method} {sense}")
    ax.set_xlabel("frequency (c/d)")
    ax.set_ylabel("dP/dt (s/yr)")
    title = f"pdot_search ({search.method})"
    if np.isfinite(best.ridge_correlation):
        title += f", ridge r = {best.ridge_correlation:+.2f}"
    ax.set_title(title)


def plot_oc(
    ax: Any,
    track: PhaseTrackResult,
    *,
    ltte: LTTEFit | None = None,
    t_offset: float = DEFAULT_T_OFFSET,
    degree_overlays: Sequence[int] = (2,),
) -> None:
    """Draw the O-C diagram and its fitted quadratic ephemeris.

    A constant dP/dt makes the O-C a parabola, so this panel *is* the
    measurement: the curvature is the rate and the scatter about the curve is
    the honest error on it. Two things to read off it — a sawtooth, which means
    the cycle count slipped (:attr:`~cuperiod.PhaseTrackResult.wrap_risk`, which
    is annotated when set), and residual structure left after the parabola,
    which means the rate is not constant.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    track : PhaseTrackResult
        The result of :func:`cuperiod.phase_track`.
    ltte : LTTEFit, optional
        The best fit of :func:`cuperiod.ltte_search` over the same delays.
        Overlaid as the joint polynomial + light-time curve, with the outer
        period in the label — a cycle shorter than the baseline is what makes
        the O-C non-monotonic, and the parabola alone then averages over it.
        Drawn as given: pass it only for a detected cycle (``batch_pdot`` uses
        ``delta_bic > 10`` and not ``at_grid_edge``).
    t_offset : float, default 2450000.0
        Subtracted from the epochs before plotting.
    degree_overlays : sequence of int, default ``(2,)``
        Extra weighted :func:`numpy.polyfit` curves to overlay for comparison.
        The point of them is to see whether a higher-order term is still
        shedding residual — a light curve whose O-C keeps improving past degree
        2 has a rate that is itself evolving. The fitted degree (2) is skipped:
        it is already drawn from ``track.quadratic_fit``, and the two are the
        same weighted least-squares curve.

    Notes
    -----
    The plotted error bars are the per-segment timing errors inflated by
    ``sqrt(chi2_reduced)`` of the quadratic fit, following
    :attr:`~cuperiod.EphemerisFit.pdot_err_scaled`: on real data the light-curve
    shape evolves and the formal errors are optimistic. As there, the factor is
    floored at 1 so a fit that beats its errors never gets *shrunken* bars.
    """
    fit = track.quadratic_fit
    t = _floats(track.t_mid)
    oc = _floats(track.delay) * _HOURS_PER_DAY
    if t.size == 0 or t.size != oc.size:
        _empty(ax, "no O-C points")
        return
    inflate = float(np.sqrt(max(1.0, fit.chi2_reduced)))
    err = _usable_errors(track.delay_err, t.size)
    yerr = None if err is None else err * _HOURS_PER_DAY * inflate
    ax.errorbar(t - t_offset, oc, yerr=yerr, fmt="o", ms=4, color="C0", zorder=3)

    grid = _dense_grid(float(np.min(t)), float(np.max(t)))
    if grid is not None:
        ax.plot(
            grid - t_offset,
            _floats(fit.predict(grid)) * _HOURS_PER_DAY,
            "-",
            color="C3",
            label="quadratic (constant dP/dt)",
        )
        weights = None if err is None else 1.0 / np.where(err > 0.0, err, np.inf)
        for degree in degree_overlays:
            if degree == fit.degree or degree < 1 or t.size <= degree + 1:
                continue
            coef = _poly_overlay(t - fit.t_ref, _floats(track.delay), weights, degree)
            if coef is None:
                continue
            ax.plot(
                grid - t_offset,
                np.polyval(coef, grid - fit.t_ref) * _HOURS_PER_DAY,
                "--",
                lw=1.2,
                label=f"degree {degree}",
            )
        if ltte is not None:
            ax.plot(
                grid - t_offset,
                _floats(ltte.predict(grid)) * _HOURS_PER_DAY,
                "-",
                color="C2",
                lw=1.4,
                label=(
                    f"+ light-time orbit, P2 = {ltte.outer_period_yr:.2f} yr, "
                    f"A = {ltte.amplitude_s:.0f} s"
                ),
            )
        ax.legend(fontsize=8)
    if track.wrap_risk:
        ax.text(
            0.02,
            0.95,
            "wrap risk: cycle count may have slipped",
            transform=ax.transAxes,
            fontsize=8,
            color="C3",
            va="top",
        )
    ax.set_xlabel(_time_label(t_offset))
    ax.set_ylabel("O-C (hours)")
    sigma = fit.pdot_err_scaled * SECONDS_PER_YEAR
    title = f"O-C: dP/dt = {fit.pdot_s_per_yr:+.3g} +/- {sigma:.2g} s/yr"
    if np.isfinite(fit.f_test_p):
        title += f" (p = {fit.f_test_p:.1e})"
    ax.set_title(title, fontsize=10)


def _poly_overlay(
    dt: FloatArray, oc: FloatArray, weights: FloatArray | None, degree: int
) -> FloatArray | None:
    """Weighted polynomial coefficients for an overlay, or ``None`` if unfittable."""
    try:
        coef = np.polyfit(dt, oc, degree, w=weights)
    except (np.linalg.LinAlgError, ValueError, TypeError):
        return None
    out = np.asarray(coef, dtype=np.float64)
    return out if bool(np.all(np.isfinite(out))) else None


def plot_period_profile(
    ax: Any,
    profile: PeriodProfile,
    *,
    law: PeriodLawFit | None = None,
    t_offset: float = DEFAULT_T_OFFSET,
) -> None:
    """Draw the measured ``P(t)`` track, optionally with a fitted law over it.

    This is the model-free view: every point was measured in its own interval,
    so the *shape* of the track is data rather than an assumption, and a law
    drawn through it can be judged by eye instead of by BIC alone. Intervals
    whose period ran into the edge of its search window are marked separately —
    those are bounds, not measurements, and a track that looks flat only because
    it is clipped is the standard way to miss a real period change.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    profile : PeriodProfile
        The result of :func:`cuperiod.period_profile`.
    law : PeriodLawFit, optional
        A fitted law (:func:`cuperiod.fit_period_law`), overlaid on a dense grid.
    t_offset : float, default 2450000.0
        Subtracted from the epochs before plotting.
    """
    t = _floats(profile.t_mid)
    periods = _floats(profile.period)
    if t.size == 0:
        _empty(ax, "no period-profile points")
        return
    err = _usable_errors(profile.period_err, t.size)
    ax.errorbar(
        t - t_offset,
        periods,
        yerr=err,
        fmt="o",
        ms=4,
        color="C0",
        zorder=3,
        label="measured",
    )
    edge = np.array([p.at_window_edge for p in profile.points], dtype=bool)
    if bool(edge.any()) and edge.size == t.size:
        ax.plot(
            t[edge] - t_offset,
            periods[edge],
            "s",
            mfc="none",
            mec="C3",
            ms=9,
            zorder=4,
            label="at window edge",
        )
    title = f"period profile ({profile.n_intervals} intervals)"
    if law is not None:
        grid = _dense_grid(float(np.min(t)), float(np.max(t)))
        if grid is not None:
            model = _floats(law.period_at(grid))
            good = np.isfinite(model)
            if bool(good.any()):
                ax.plot(
                    grid[good] - t_offset,
                    model[good],
                    "-",
                    color="C3",
                    label=f"{law.law} law",
                )
        title += f", {law.law} fit (chi2_r = {law.chi2_reduced:.2f})"
    ax.legend(fontsize=8)
    ax.set_xlabel(_time_label(t_offset))
    ax.set_ylabel("P (d)")
    ax.set_title(title, fontsize=10)


def plot_timescale(
    ax: Any,
    law: PeriodLawFit,
    time_range: Any,
    *,
    t_offset: float = DEFAULT_T_OFFSET,
) -> None:
    """Draw the decay timescale ``P/|dP/dt|`` in years against time.

    The panel that shows an inspiral accelerating. A constant-rate system has a
    timescale that barely moves over a baseline; a system heading for a
    finite-time singularity has one that collapses — for the ``singular`` law
    exactly as ``(t - t0)**2 / b`` — and on a logarithmic axis the difference
    between the two is impossible to miss. It is also the quantity the
    period-change literature quotes (Tylenda et al. 2011 report V1309 Sco's
    falling from ~1000 yr to ~170 yr over six years), so this panel is what a
    published number is compared against.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    law : PeriodLawFit
        The fitted law (:func:`cuperiod.fit_period_law`).
    time_range : array-like
        The epochs to span. Only the extremes are used, so either a
        ``(start, end)`` pair or a full time array works.
    t_offset : float, default 2450000.0
        Subtracted from the epochs before plotting.
    """
    span = _floats(time_range, time=True)
    span = span[np.isfinite(span)]
    grid = (
        None
        if span.size == 0
        else _dense_grid(float(np.min(span)), float(np.max(span)))
    )
    if grid is None:
        # A PeriodLawFit carries no epochs of its own, so pdot_figure lands here
        # when a law is the *only* input it was given.
        _empty(ax, "no epoch span for the timescale")
        return
    with np.errstate(divide="ignore", invalid="ignore"):
        years = _floats(law.timescale_yr_at(grid))
    good = np.isfinite(years) & (years > 0.0)
    if not bool(good.any()):
        _empty(ax, f"{law.law} law has no finite timescale")
        return
    ends = years[good]
    ax.plot(
        grid[good] - t_offset,
        ends,
        "-",
        color="C0",
        lw=2.0,
        label=f"{ends[0]:.0f} -> {ends[-1]:.0f} yr",
    )
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_xlabel(_time_label(t_offset))
    ax.set_ylabel("P / |dP/dt| (yr)")
    # The singular epoch goes in the title rather than in the axes: the curve
    # may run either way across the panel, so no corner is reliably free.
    title = f"decay timescale ({law.law} law)"
    t0 = law.singular_epoch
    if np.isfinite(t0):
        title += f", t0 = {t0 - t_offset:.0f}"
    ax.set_title(title, fontsize=10)


# --- the assembled figure -----------------------------------------------------


def _grid_shape(n_panels: int) -> tuple[int, int]:
    """Rows and columns for ``n_panels``: one column, else two."""
    if n_panels <= 1:
        return (1, 1)
    return ((n_panels + 1) // 2, 2)


def _fold_parameters(
    search: PdotSearchResult | None,
    track: PhaseTrackResult | None,
    profile: PeriodProfile | None,
    law: PeriodLawFit | None,
    time: FloatArray,
) -> tuple[float, float, float, Any] | None:
    """``(period, pdot, t_ref, model)`` for the fold, from the best source given.

    Preference order is the order in which the tools sharpen a fold: the
    tracker's O-C measurement first, then the search's best cell, then a fitted
    law linearized at its reference epoch (its local rate is exactly the
    ``exponential`` model's parameter), and finally the profile's seed period
    with no drift at all.

    The tracker's ephemeris is its *fitted* one, not its input: ``track.period``
    is the reference period the delays were measured against, and any error in
    it is absorbed into the O-C's linear term ``c1`` (``P_true ~ P*(1 + c1)``,
    expanded about the fit's own ``t_ref``). Folding at the input period with
    only the quadratic rate leaves that linear drift in the panel, which then
    looks smeared exactly when the measurement is fine.
    """
    if track is not None:
        fit = track.quadratic_fit
        coefficients = np.asarray(fit.coefficients, dtype=np.float64)
        if coefficients.size >= 2:
            period = float(track.period) * (1.0 + float(coefficients[1]))
            if np.isfinite(period) and period > 0.0 and np.isfinite(fit.pdot):
                return (period, float(fit.pdot), float(fit.t_ref), track.model)
        return (track.period, track.pdot, track.t_ref, track.model)
    if search is not None:
        best = search.best
        return (best.period, best.pdot, best.t_ref, best.model)
    if law is not None:
        epoch = law.t_ref
        if not np.isfinite(epoch) and time.size:
            epoch = float(np.mean(time))
        period = float(np.asarray(law.period_at(epoch)).ravel()[0])
        pdot = float(np.asarray(law.pdot_at(epoch)).ravel()[0])
        if np.isfinite(period) and period > 0.0 and np.isfinite(pdot):
            return (period, pdot, float(epoch), "exponential")
        return None
    if profile is not None:
        epoch = float(np.mean(time)) if time.size else 0.0
        return (profile.reference_period, 0.0, epoch, None)
    return None


def _timescale_range(
    time: FloatArray,
    search: PdotSearchResult | None,
    track: PhaseTrackResult | None,
    profile: PeriodProfile | None,
) -> FloatArray:
    """The widest epoch span any of the inputs can vouch for."""
    if time.size:
        return time
    if profile is not None and profile.n_intervals:
        return _floats(profile.t_mid)
    if track is not None and track.n_segments:
        return _floats(track.t_mid)
    if search is not None:
        half = 0.5 * search.baseline
        return np.array([search.t_ref - half, search.t_ref + half], dtype=np.float64)
    return np.zeros(0, dtype=np.float64)


def _figure_classes() -> tuple[type[Figure], Any]:
    """Import matplotlib lazily, or explain which extra provides it.

    Returns the figure and Agg-canvas classes rather than pyplot: a figure that
    was never registered with pyplot needs no closing, carries no backend, and
    can be built from a worker process.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure as FigureClass
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "cuperiod.pdot.plots needs matplotlib, which is an optional "
            "dependency. Install it with:\n"
            "    pip install 'cuperiod[plot]'"
        ) from exc
    return FigureClass, FigureCanvasAgg


def pdot_figure(
    *,
    light_curve: Any = None,
    search: PdotSearchResult | None = None,
    track: PhaseTrackResult | None = None,
    profile: PeriodProfile | None = None,
    law: PeriodLawFit | None = None,
    ltte: LTTEFit | None = None,
    title: str = "",
    path: str | Path | None = None,
    dpi: int = 140,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Build the multi-panel period-change diagnostic figure for one star.

    Draws only the panels whose inputs were supplied, so the same call works
    after a bare :func:`cuperiod.pdot_search` and after the full chain. In
    order: the raw light curve, the drift-corrected phase fold, the search's
    ``(dP/dt, frequency)`` map, the O-C diagram, the measured ``P(t)`` track,
    and the decay timescale. Panels are laid out two to a row, with a lone
    trailing panel spanning its row.

    Parameters
    ----------
    light_curve : various, optional
        The light curve, as a ``(time, value, error)`` tuple, a
        :class:`~cuperiod.core.lightcurve.LightCurve`, or anything
        :func:`cuperiod.to_input` accepts. Enables the light-curve panel, and
        the fold panel whenever one of the other inputs supplies a period.
    search : PdotSearchResult, optional
        From :func:`cuperiod.pdot_search`.
    track : PhaseTrackResult, optional
        From :func:`cuperiod.phase_track`.
    profile : PeriodProfile, optional
        From :func:`cuperiod.period_profile`.
    law : PeriodLawFit, optional
        From :func:`cuperiod.fit_period_law`. Overlaid on the profile panel, and
        the source of the decay-timescale panel.
    ltte : LTTEFit, optional
        The best fit of :func:`cuperiod.ltte_search`, overlaid on the O-C panel as
        given (pass it only for a detected cycle; see :func:`plot_oc`).
    title : str, default ""
        Figure suptitle (typically the star's name).
    path : str or pathlib.Path, optional
        Where to save the figure. Parent directories are created. The figure is
        returned either way and no window is ever opened.
    dpi : int, default 140
        Resolution of the figure and of the saved file.
    figsize : tuple of float, optional
        Figure size in inches; by default it grows with the panel count.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, with an Agg canvas attached. It is not registered with
        pyplot, so nothing has to be closed and the call is safe inside a worker
        process; ``fig.savefig(...)`` and notebook display both work as usual.

    Raises
    ------
    ImportError
        If matplotlib is not installed (``pip install 'cuperiod[plot]'``).

    Examples
    --------
    >>> res = cup.pdot_search(lc, period=1.43)                  # doctest: +SKIP
    >>> trk = cup.phase_track(lc, period=res.best.period)       # doctest: +SKIP
    >>> fig = pdot_figure(light_curve=lc, search=res, track=trk,
    ...                   title="V1309 Sco", path="v1309.png")  # doctest: +SKIP
    """
    figure_cls, canvas_cls = _figure_classes()
    curve = _as_arrays(light_curve)
    time = curve[0] if curve is not None else np.zeros(0, dtype=np.float64)
    t_offset = _time_offset(
        time if time.size else _timescale_range(time, search, track, profile)
    )

    panels: list[Callable[[Any], None]] = []
    if curve is not None:
        lc_time, lc_value, lc_error, invert = curve
        panels.append(
            lambda ax: plot_light_curve(
                ax, lc_time, lc_value, lc_error, t_offset=t_offset, invert=invert
            )
        )
        fold = _fold_parameters(search, track, profile, law, lc_time)
        if fold is not None:
            period, pdot, t_ref, model = fold
            panels.append(
                lambda ax: _fold_panel(
                    ax, lc_time, lc_value, lc_error, period, pdot, t_ref, model, invert
                )
            )
    if search is not None:
        panels.append(lambda ax: plot_search_map(ax, search))
    if track is not None:
        panels.append(
            lambda ax: plot_oc(ax, track, ltte=ltte, t_offset=t_offset)
        )
    if profile is not None:
        panels.append(
            lambda ax: plot_period_profile(ax, profile, law=law, t_offset=t_offset)
        )
    if law is not None:
        span = _timescale_range(time, search, track, profile)
        panels.append(lambda ax: plot_timescale(ax, law, span, t_offset=t_offset))

    n_panels = max(1, len(panels))
    nrows, ncols = _grid_shape(n_panels)
    if figsize is None:
        figsize = (7.0, 4.4) if n_panels == 1 else (6.0 * ncols, 3.7 * nrows)
    # Built without pyplot: no global figure registry, no backend switching, no
    # window — just a figure with an Agg canvas, which savefig and the notebook
    # display hook both understand. constrained layout (rather than
    # tight_layout) is what keeps the colorbars from overlapping their axes.
    fig = figure_cls(figsize=figsize, dpi=dpi, layout="constrained")
    canvas_cls(fig)
    spec = fig.add_gridspec(nrows, ncols)

    if not panels:
        _empty(fig.add_subplot(spec[0, 0]), "nothing to plot: no results given")
    for index, draw in enumerate(panels):
        row, col = divmod(index, ncols)
        lone_last = index == len(panels) - 1 and len(panels) % ncols == 1
        cell = spec[row, :] if lone_last and ncols > 1 else spec[row, col]
        draw(fig.add_subplot(cell))
    if title:
        fig.suptitle(title)
    if path is not None:
        target = Path(path)
        if target.parent != Path():
            target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=dpi)
    return fig


def _fold_panel(
    ax: Any,
    time: FloatArray,
    value: FloatArray,
    error: FloatArray | None,
    period: float,
    pdot: float,
    t_ref: float,
    model: Any,
    invert: bool,
) -> None:
    """:func:`plot_phase_fold` with the light-curve panel's axis label and flip.

    Both panels follow the light curve's brightness domain: magnitudes are
    labelled as such and flipped (brighter upwards), fluxes are neither.
    """
    plot_phase_fold(
        ax,
        time,
        value,
        period=period,
        pdot=pdot,
        t_ref=t_ref,
        model=model,
        error=error,
        ylabel="magnitude" if invert else "flux",
    )
    if invert and ax.has_data():
        ax.invert_yaxis()


__all__ = [
    "DEFAULT_T_OFFSET",
    "pdot_figure",
    "plot_light_curve",
    "plot_oc",
    "plot_period_profile",
    "plot_phase_fold",
    "plot_search_map",
    "plot_timescale",
]
