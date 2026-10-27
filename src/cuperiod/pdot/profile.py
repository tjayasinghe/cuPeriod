"""Period measured in consecutive time intervals: :func:`period_profile`.

:func:`cuperiod.pdot_search` and :func:`cuperiod.phase_track` both assume a
*model* for how the period evolves — a single rate over the whole baseline —
and return one number for it. When the rate itself changes, that number is a
baseline-weighted average of a moving quantity, and neither its value nor its
error bar means what it appears to.

The model-free alternative, and the one the period-change literature actually
publishes, is to chop the light curve into intervals short enough that the
period is constant within each, measure it independently in every one, and
treat the resulting ``P(t)`` track as the data product. Nothing about the shape
of the evolution is assumed until a law is fitted to that track with
:func:`cuperiod.fit_period_law`.

Each interval's period comes from minimizing the chi-squared of a fixed-order
harmonic model over frequency — multi-harmonic AOV, the same statistic
Schwarzenberg-Czerny (1996) built for non-sinusoidal light curves — with the
uncertainty read off the curvature of that chi-squared at its minimum. The
harmonic order is chosen once — by a scale-free BIC, a criterion that does not
trust the quoted photometric errors, as the median of the orders picked on up to
five intervals spread across the baseline — then held fixed, so every point on
the track is measured the same way and the intervals are comparable to each
other, which is what a subsequent law fit needs.

Interval periods are searched only inside a narrow window around the reference
period (``period_window``, a fractional half-width). That is a deliberate
constraint rather than a shortcut: an interval holding a few tens of cycles has
coarse frequency resolution and a rich alias comb, and an unconstrained search
in one would happily return a 1 c/d alias. The window's price is that a period
change larger than it will be clipped — :attr:`PeriodPoint.at_window_edge`
flags exactly that, and it is worth checking before believing a flat track.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import PdotSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.pdot.laws import PeriodLawFit, compare_period_laws, fit_period_law
from cuperiod.pdot.track import _design


@dataclass(frozen=True)
class PeriodPoint:
    """One interval's period measurement.

    Attributes
    ----------
    index : int
        Interval number, ascending in time from 0.
    t_mid : float
        Weighted mean epoch of the interval's points (the epoch the period is
        attributed to).
    t_start, t_end : float
        First and last observation epoch in the interval.
    n_points : int
        Points used.
    period, period_err : float
        Measured period (days) and its 1-sigma error from the curvature of the
        harmonic chi-squared, inflated by ``sqrt(chi2_reduced)`` when the model
        does not fit within the stated photometric errors.

        This is the *formal precision of this interval's measurement*, and on
        real stars the track will scatter about any smooth law by considerably
        more. The excess is physical, not a broken error bar: the light-curve
        shape evolves, spots migrate, and many period-changing systems really do
        wander on timescales shorter than the baseline. It is also correlated
        between neighbouring intervals, which a per-interval error cannot
        express. Believe these errors for *relative* weighting between
        intervals, and let :func:`cuperiod.fit_period_law` rescale them by
        ``sqrt(chi2_reduced)`` (its default) before quoting a law parameter.
    frequency, frequency_err : float
        The same measurement in cycles/day.
    chi2_reduced : float
        Per-degree-of-freedom chi-squared of the harmonic fit at the best
        frequency. Large values mean the fixed template does not describe this
        interval — an evolving light-curve shape, or a bad period.
    amplitude : float
        Semi-amplitude of the fundamental harmonic, in the input's units.
    power : float
        Fraction of the interval's weighted variance the harmonic model
        explains: 0 for a useless fit, 1 for a perfect one.
    at_window_edge : bool
        The best frequency sits at the edge of the search window, so the true
        period may lie outside it. Treat the value as a bound, not a
        measurement, and widen ``period_window``.
    pdot, pdot_err : float
        Interval-local ``dP/dt`` (day/day), NaN unless ``measure_pdot=True``.
    """

    index: int
    t_mid: float
    t_start: float
    t_end: float
    n_points: int
    period: float
    period_err: float
    frequency: float
    frequency_err: float
    chi2_reduced: float
    amplitude: float
    power: float
    at_window_edge: bool
    pdot: float = float("nan")
    pdot_err: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dict (one row of the profile table)."""
        return {
            "index": self.index,
            "t_mid": self.t_mid,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "n_points": self.n_points,
            "period": self.period,
            "period_err": self.period_err,
            "frequency": self.frequency,
            "frequency_err": self.frequency_err,
            "chi2_reduced": self.chi2_reduced,
            "amplitude": self.amplitude,
            "power": self.power,
            "at_window_edge": self.at_window_edge,
            "pdot": self.pdot,
            "pdot_err": self.pdot_err,
        }


@dataclass(frozen=True)
class PeriodProfile:
    """A measured ``P(t)`` track: the periods of consecutive intervals.

    Attributes
    ----------
    points : tuple of PeriodPoint
        The per-interval measurements, ascending in time.
    reference_period : float
        The seed period the search windows were centred on (days).
    n_harmonics : int
        Harmonic order of the shared template.
    n_samples : int
        Finite points in the whole light curve.
    baseline : float
        Full time baseline (days).
    period_window : float
        Fractional half-width of each interval's frequency search window.
    meta : Mapping
        Provenance plus light-curve metadata.
    """

    points: tuple[PeriodPoint, ...]
    reference_period: float
    n_harmonics: int
    n_samples: int
    baseline: float
    period_window: float
    meta: Mapping[str, Any] = field(default_factory=dict)

    # -- array views ---------------------------------------------------------------

    def _column(self, name: str) -> FloatArray:
        return np.array([getattr(p, name) for p in self.points], dtype=np.float64)

    @property
    def t_mid(self) -> FloatArray:
        """Interval epochs (days)."""
        return self._column("t_mid")

    @property
    def period(self) -> FloatArray:
        """Measured periods (days)."""
        return self._column("period")

    @property
    def period_err(self) -> FloatArray:
        """1-sigma period errors (days)."""
        return self._column("period_err")

    @property
    def n_intervals(self) -> int:
        """Number of usable intervals."""
        return len(self.points)

    @property
    def any_at_window_edge(self) -> bool:
        """Whether any interval's period ran into the edge of its search window."""
        return any(p.at_window_edge for p in self.points)

    # -- summaries -----------------------------------------------------------------

    def total_fractional_change(self) -> float:
        """``P_last/P_first - 1`` straight from the endpoint measurements.

        The assumption-free companion to
        :meth:`PeriodLawFit.fractional_change`: no law, no fit, just the two end
        intervals — and correspondingly noisier, since it throws away every
        point in between.
        """
        if self.n_intervals < 2:
            return float("nan")
        first, last = self.points[0].period, self.points[-1].period
        return float(last / first - 1.0) if first else float("nan")

    def fit(self, law: str = "auto", **kwargs: Any) -> PeriodLawFit:
        """Fit a period law to this track (see :func:`cuperiod.fit_period_law`)."""
        return fit_period_law(
            self.t_mid, self.period, self.period_err, law=law, **kwargs
        )

    def compare(
        self, laws: Sequence[str] | None = None, **kwargs: Any
    ) -> list[PeriodLawFit]:
        """Rank several period laws on this track by BIC."""
        if laws is None:
            return compare_period_laws(
                self.t_mid, self.period, self.period_err, **kwargs
            )
        return compare_period_laws(
            self.t_mid, self.period, self.period_err, laws=laws, **kwargs
        )

    def to_table(self) -> list[dict[str, Any]]:
        """The track as a list of row dicts, ready for a DataFrame."""
        return [p.to_dict() for p in self.points]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the summary scalars plus the track."""
        return {
            "reference_period": self.reference_period,
            "n_harmonics": self.n_harmonics,
            "n_samples": self.n_samples,
            "baseline": self.baseline,
            "period_window": self.period_window,
            "n_intervals": self.n_intervals,
            "any_at_window_edge": self.any_at_window_edge,
            "total_fractional_change": self.total_fractional_change(),
            "points": self.to_table(),
        }


# --- the per-interval measurement ---------------------------------------------


def _harmonic_chi2(
    dt: FloatArray, y: FloatArray, w: FloatArray, f: float, n_harmonics: int
) -> float:
    """Weighted chi-squared of the best ``n_harmonics`` model at frequency ``f``."""
    design = _design(dt, f, n_harmonics)
    xtw = design.T * w[None, :]
    try:
        coef = np.linalg.solve(xtw @ design, xtw @ y)
    except np.linalg.LinAlgError:
        return float("inf")
    resid = y - design @ coef
    return float(np.dot(w, resid * resid))


#: Elements of one stacked (trials x points x coefficients) design in
#: :func:`_harmonic_chi2_many`: cache-sized, so batching never loses to the
#: per-trial loop even on long intervals.
_STACK_BUDGET: int = 1 << 16


def _harmonic_chi2_many(
    dt: FloatArray, y: FloatArray, w: FloatArray, freqs: FloatArray, n_harmonics: int
) -> FloatArray:
    """:func:`_harmonic_chi2` at every frequency in ``freqs``, batched.

    One stacked design, one batched solve and one residual pass per chunk of
    trials, instead of a Python call per trial, whose overhead dominated the
    whole profile on the usual ~50-point intervals. The design is built in the
    same operation order as :func:`cuperiod.pdot.track._design`; a chunk with
    a singular trial falls back to the scalar function, which keeps its ``inf``.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    n = int(dt.size)
    k = 2 * n_harmonics + 1
    chunk = max(1, _STACK_BUDGET // max(1, n * k))
    m = np.arange(1, n_harmonics + 1, dtype=np.float64)
    out = np.empty(freqs.size, dtype=np.float64)
    for start in range(0, freqs.size, chunk):
        f = freqs[start : start + chunk]
        ang = (2.0 * np.pi * f)[:, None, None] * dt[None, :, None] * m[None, None, :]
        design = np.empty((f.size, n, k), dtype=np.float64)
        design[:, :, 0] = 1.0
        design[:, :, 1 : 1 + n_harmonics] = np.cos(ang)
        design[:, :, 1 + n_harmonics :] = np.sin(ang)
        xtw = np.swapaxes(design, 1, 2) * w[None, None, :]
        try:
            coef = np.linalg.solve(xtw @ design, (xtw @ y)[..., None])
        except np.linalg.LinAlgError:
            out[start : start + f.size] = [
                _harmonic_chi2(dt, y, w, float(v), n_harmonics) for v in f
            ]
            continue
        resid = y[None, :] - (design @ coef)[..., 0]
        out[start : start + f.size] = (resid * resid) @ w
    return out


def _refine_frequency(
    dt: FloatArray,
    y: FloatArray,
    w: FloatArray,
    f_lo: float,
    f_hi: float,
    n_harmonics: int,
    oversample: int,
    *,
    unit_weights: bool = False,
) -> tuple[float, float, float, bool]:
    """Best frequency in ``[f_lo, f_hi]`` and its curvature error.

    A dense scan finds the global minimum inside the window (the harmonic
    chi-squared is multi-modal on the alias comb), a bounded minimization
    converges on it inside the scan's bracket, and two parabolic passes read
    the curvature there. Returns ``(frequency, sigma_f, chi2_min, at_edge)``.

    The scan steps at ``1/(oversample * span * n_harmonics)``: the ``m``-th
    harmonic narrows the chi-squared's structure to ``1/(m * span)``, and a
    stencil scaled for the fundamental alone can bracket the wrong sub-basin
    of a many-harmonic template. And the minimum is *converged*, not read off
    the scan stencil's parabolic vertex: at stencil scale the chi-squared of
    irregular sampling is asymmetric, the vertex misses the minimum by more
    than the curvature sigma whenever that sigma is small (many harmonics,
    high signal-to-noise), and the period then scatters several times beyond
    its own error bar.

    With ``unit_weights`` (no photometric errors) the curvature sigma is in
    the units of the values until ``chi2_reduced`` restores it, so the second
    curvature pass sizes its stencil from the restored, physical sigma; the
    errors are then the same whatever the units of the light curve.
    """
    from scipy.optimize import minimize_scalar

    span = max(float(dt.max() - dt.min()), 1e-9)
    step = 1.0 / (oversample * span * max(1, n_harmonics))
    n = max(9, int(math.ceil((f_hi - f_lo) / step)) + 1)
    grid = np.linspace(f_lo, f_hi, n)
    chi2 = _harmonic_chi2_many(dt, y, w, grid, n_harmonics)
    k = int(np.argmin(chi2))
    at_edge = k in (0, n - 1)
    f_best = float(grid[k])
    chi2_min = float(chi2[k])
    if not at_edge and np.isfinite(chi2_min):
        res = minimize_scalar(
            lambda f: _harmonic_chi2(dt, y, w, float(f), n_harmonics),
            bounds=(float(grid[k - 1]), float(grid[k + 1])),
            method="bounded",
            options={"xatol": 1e-6 * float(grid[1] - grid[0])},
        )
        if np.isfinite(res.fun) and float(res.fun) <= chi2_min:
            f_best, chi2_min = float(res.x), float(res.fun)

    # Curvature at the minimum. The first pass uses a scan-scale step, the
    # second one scaled to the sigma it implied, so the parabola is evaluated
    # where the chi-squared is actually parabolic.
    dof = max(1, int(dt.size) - (2 * n_harmonics + 1))
    unit = math.sqrt(max(chi2_min, 0.0) / dof) if unit_weights else 1.0
    h = 0.25 * step
    sigma = float("nan")
    for _ in range(2):
        lo, hi = _harmonic_chi2_many(
            dt, y, w, np.array([f_best - h, f_best + h]), n_harmonics
        )
        curv = (lo - 2.0 * chi2_min + hi) / (h * h)
        if not np.isfinite(curv) or curv <= 0.0:
            sigma = float("nan")
            break
        sigma = math.sqrt(2.0 / curv)
        h = max(min(sigma * unit, 4.0 * step), 1e-4 * step)
    return f_best, sigma, chi2_min, at_edge


def _analytic_sigma_f(
    n_points: int, span: float, residual_rms: float, amplitude: float
) -> float:
    """Montgomery & O'Donoghue (1999) frequency error, as a last resort.

    ``sigma_f = sqrt(6/N) * sigma / (pi * T * A)``, the same closed form the
    pre-whitening module uses. It assumes an isolated sinusoid and so is only a
    stand-in for the curvature error, which makes no such assumption.
    """
    if n_points <= 0 or span <= 0.0 or amplitude <= 0.0:
        return float("nan")
    if not np.isfinite(residual_rms):
        return float("nan")
    return math.sqrt(6.0 / n_points) * residual_rms / (math.pi * span * amplitude)


def _select_harmonics(
    time: FloatArray,
    y: FloatArray,
    w: FloatArray,
    f_ref: float,
    slices: Sequence[slice],
    *,
    pinned: int | None,
    window: float,
    oversample: int,
    max_harmonics: int,
) -> int:
    """Harmonic order for the shared template: a vote over several intervals.

    Two things make this harder than an ordinary template-order choice.

    The reference period is wrong inside any given interval by up to roughly
    half the total period change, which over a few tens of cycles is enough
    phase drift to smear the fold. The order is therefore chosen against a
    frequency refined *in that interval*, at a deliberately low order —
    otherwise the selection buys harmonics to describe the smear and runs
    straight to ``max_harmonics``.

    And the criterion must be the *scale-free* BIC ``n*ln(chi2/n) + k*ln(n)``,
    which depends on the weights only up to an overall factor. The ordinary
    weighted form ``chi2 + k*ln(n)`` is right when the quoted errors are right
    and catastrophic when they are not: survey photometry routinely understates
    its errors several-fold, every harmonic then looks like a bargain, and the
    selection again runs to the cap. That matters beyond tidiness — an
    over-fitted template has a jagged chi-squared in frequency and locks onto
    sampling aliases in the sparser intervals, turning a clean ``P(t)`` track
    into a scattered one.

    Even so, one interval's answer is noisy enough to swing by several orders
    depending on which stretch it lands in, so the order is the median over a
    handful of intervals spread across the baseline.
    """
    if pinned is not None:
        return int(pinned)
    picks: list[int] = []
    for sl in _selection_intervals(slices, _ORDER_VOTES):
        seg_t, seg_y, seg_w = time[sl], y[sl], w[sl]
        n = int(seg_t.size)
        if n < 4 * 2 + 3:  # too short to choose an order from
            continue
        dt = seg_t - float(np.mean(seg_t))
        f_mid, _, _, _ = _refine_frequency(
            dt, seg_y, seg_w, f_ref / (1.0 + window), f_ref / (1.0 - window),
            2, oversample,
        )
        if not (np.isfinite(f_mid) and f_mid > 0.0):
            f_mid = f_ref
        # An interval must still afford 2*H+1 coefficients with room to spare.
        top = max(1, min(max_harmonics, (n - 3) // 4))
        log_n = math.log(n)
        best, best_bic = 1, math.inf
        for order in range(1, top + 1):
            chi2 = _harmonic_chi2(dt, seg_y, seg_w, f_mid, order)
            if not np.isfinite(chi2) or chi2 <= 0.0:
                continue
            bic = n * math.log(chi2 / n) + (2 * order + 1) * log_n
            if bic < best_bic - 1e-12:
                best, best_bic = order, bic
        picks.append(best)
    if not picks:
        return 2
    return int(np.median(picks))


#: Intervals polled for the harmonic-order vote. Odd, so the median is one of
#: the votes rather than an average of two.
_ORDER_VOTES: int = 5


def _selection_intervals(slices: Sequence[slice], count: int) -> list[slice]:
    """``count`` intervals spread evenly across the baseline (or all of them)."""
    if len(slices) <= count:
        return list(slices)
    idx = np.linspace(0, len(slices) - 1, count).round().astype(int)
    return [slices[int(i)] for i in dict.fromkeys(idx.tolist())]


def _interval_slices(n: int, per_interval: int, n_intervals: int | None) -> list[slice]:
    """Contiguous, points-balanced intervals over ``n`` time-ordered samples."""
    if n_intervals is not None:
        count = n_intervals
    else:
        count = max(1, n // max(1, per_interval))
    count = max(1, min(count, n))
    edges = np.linspace(0, n, count + 1).astype(np.int64)
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def period_profile(
    data: Any,
    *,
    period: float | None = None,
    points_per_interval: int | None = None,
    n_intervals: int | None = None,
    period_window: float | None = None,
    harmonics: int | None = None,
    measure_pdot: bool = False,
    settings: PdotSettings | None = None,
    backend: str = "auto",
    columns: ColumnMap | None = None,
    domain: Domain | str | None = None,
) -> PeriodProfile:
    """Measure the period independently in consecutive time intervals.

    Produces the ``P(t)`` track that :func:`cuperiod.fit_period_law` turns into
    a period-change law. Assumes nothing about *how* the period evolves — only
    that it is effectively constant within one interval — which makes it the
    right tool when a single ``dP/dt`` is suspect and the reference method for
    reproducing published period-change measurements.

    Parameters
    ----------
    data : various
        The light curve (single-band), in any form :func:`cuperiod.to_input`
        accepts.
    period : float, optional
        Reference period in days, the centre of every interval's search window.
        ``None`` seeds it from a :func:`cuperiod.pdot_search` over the whole
        light curve, which tolerates a period that drifts across the baseline.
    points_per_interval : int, optional
        Target points per interval (default 50, the convention of the published
        V1309 Sco analysis). Ignored when ``n_intervals`` is given.
    n_intervals : int, optional
        Fixed number of intervals instead of a target size.
    period_window : float, optional
        Fractional half-width of the per-interval search window (default 0.05,
        i.e. +-5% in period). Must contain the true total period change.
    harmonics : int, optional
        Harmonic order of the shared template. ``None`` chooses it once, by a
        scale-free BIC (``n*ln(chi2/n) + k*ln(n)``, blind to the overall error
        scale), as the median of the orders picked on up to five intervals spread
        across the baseline, and then holds it fixed across every interval.
    measure_pdot : bool, default False
        Also run a small joint (P, dP/dt) search inside each interval. Rarely
        worth it — an interval short enough to have a constant period is usually
        too short to measure its derivative — but it does expose a rate that
        varies faster than the interval spacing.
    settings : PdotSettings, optional
        Supplies the defaults for the arguments above and the seeding search.
    backend : str, default "auto"
        Backend for the seeding search (the interval fits are direct linear
        algebra and always run on the CPU).
    columns, domain
        Passed to :func:`cuperiod.to_input` for table inputs.

    Returns
    -------
    PeriodProfile

    Examples
    --------
    >>> prof = period_profile(lc, period=1.4456)             # doctest: +SKIP
    >>> fit = prof.fit(law="singular")                       # doctest: +SKIP
    >>> fit.fractional_change(t[0], t[-1])                   # doctest: +SKIP
    """
    from cuperiod.api import to_input
    from cuperiod.pdot.scan import pdot_search

    psettings = settings or PdotSettings()
    per_interval = (
        points_per_interval
        if points_per_interval is not None
        else psettings.profile_points_per_interval
    )
    window = (
        period_window if period_window is not None else psettings.profile_period_window
    )
    # `not (a < x < b)` rather than `x <= a or x >= b`: NaN fails every
    # comparison and must be refused here, not deep inside the scan.
    if not 0.0 < window < 1.0:
        raise ValueError(f"period_window must be in (0, 1), got {window}")
    # Counts: `not (... >= 1)` so a NaN fails here, and an infinity too, rather than
    # as an int() conversion error or an empty profile further down.
    for name, count in (
        ("harmonics", harmonics),
        ("points_per_interval", per_interval),
        ("n_intervals", n_intervals),
    ):
        if count is not None and not (
            math.isfinite(count) and float(count).is_integer() and count >= 1
        ):
            raise ValueError(f"{name} must be an integer >= 1, got {count}")
    if period is not None and not (math.isfinite(period) and period > 0.0):
        raise ValueError(f"period must be finite and positive, got {period}")

    lc = to_input(data, columns=columns, domain=domain)
    if isinstance(lc, MultiBandLightCurve):
        raise ValueError(
            "period_profile works on a single band; pass one band of the "
            "MultiBandLightCurve"
        )
    finite = lc.finite()
    if finite.n < psettings.min_detections:
        raise InsufficientDataError(
            f"period_profile: {finite.n} finite points < min_detections "
            f"{psettings.min_detections}"
        )
    order = np.argsort(finite.time, kind="stable")
    time = finite.time[order]
    y = finite.value[order]
    if finite.error is not None:
        err = finite.error[order]
        w = 1.0 / (err * err)
        weighted = True
    else:
        w = np.ones(time.size, dtype=np.float64)
        weighted = False

    if period is None:
        period = pdot_search(
            finite, settings=psettings, backend=backend
        ).best.period
    if not (math.isfinite(period) and period > 0.0):
        raise ValueError(f"period must be finite and positive, got {period}")
    f_ref = 1.0 / period

    slices = _interval_slices(time.size, per_interval, n_intervals)
    # One template order for the whole run, so intervals stay comparable — but
    # selected on a *single* interval, not the whole light curve. A drifting
    # period smears a fixed-period fold over the full baseline, and the order
    # selection then spends harmonics describing the smear rather than the
    # light-curve shape (on drifting data it simply runs to max_harmonics).
    n_harm = _select_harmonics(
        time,
        y,
        w,
        f_ref,
        slices,
        pinned=harmonics if harmonics is not None else psettings.harmonics,
        window=window,
        oversample=psettings.samples_per_peak,
        max_harmonics=psettings.max_harmonics,
    )

    points: list[PeriodPoint] = []
    for idx, sl in enumerate(slices):
        seg_t, seg_y, seg_w = time[sl], y[sl], w[sl]
        n_seg = int(seg_t.size)
        # A harmonic model needs 2*H+1 coefficients; demand a real margin over
        # that and at least a couple of cycles, or the "period" is a fit artefact.
        span = float(seg_t[-1] - seg_t[0]) if n_seg else 0.0
        if n_seg < max(2 * n_harm + 3, psettings.profile_min_points):
            continue
        if span < 2.0 * period:
            continue
        if float(np.ptp(seg_y)) == 0.0:
            # Nothing varies: any "period" would be the scan's argmin of a flat
            # chi-squared. (Not `total > 0` below -- on constant data the
            # weighted mean's rounding leaves `total` at ~1e-24, not 0.)
            continue
        dt = seg_t - float(np.mean(seg_t))
        f_best, sigma_f, chi2_min, at_edge = _refine_frequency(
            dt,
            seg_y,
            seg_w,
            f_ref / (1.0 + window),
            f_ref / (1.0 - window),
            n_harm,
            psettings.samples_per_peak,
            unit_weights=not weighted,
        )
        if not (np.isfinite(f_best) and f_best > 0.0):
            continue
        dof = max(1, n_seg - (2 * n_harm + 1))
        chi2_red = chi2_min / dof
        # Without real photometric errors the weights carry no scale, so the
        # curvature sigma is in arbitrary units until chi2_reduced restores it;
        # with real ones, inflating by a poor chi2 is the usual honesty tax.
        scale = math.sqrt(max(1.0, chi2_red)) if weighted else math.sqrt(chi2_red)
        sigma_f = sigma_f * scale
        design = _design(dt, f_best, n_harm)
        xtw = design.T * seg_w[None, :]
        try:
            coef = np.linalg.solve(xtw @ design, xtw @ seg_y)
        except np.linalg.LinAlgError:
            continue  # weights that underflow, or a degenerate interval
        # Columns are [1, cos(1..H), sin(1..H)]: the fundamental's sine
        # coefficient is coef[1 + H], not coef[2] (the second cosine).
        amplitude = float(math.hypot(float(coef[1]), float(coef[1 + n_harm])))
        mean = float(np.dot(seg_w, seg_y) / seg_w.sum())
        total = float(np.dot(seg_w, (seg_y - mean) ** 2))
        if not np.isfinite(sigma_f):
            # A chi-squared minimum too flat or too jagged for a local parabola
            # (a gappy interval, or one where the template barely fits). Fall
            # back to the analytic sinusoid error rather than emitting a point
            # with no weight at all.
            resid = seg_y - design @ coef
            rms = float(np.sqrt(np.dot(seg_w, resid * resid) / seg_w.sum()))
            sigma_f = _analytic_sigma_f(n_seg, span, rms, amplitude)
        pdot_val, pdot_err = float("nan"), float("nan")
        if measure_pdot:
            pdot_val, pdot_err = _interval_pdot(
                seg_t, seg_y, seg_w, weighted, f_best, psettings, backend
            )
        points.append(
            PeriodPoint(
                index=idx,
                t_mid=float(np.dot(seg_w, seg_t) / seg_w.sum()),
                t_start=float(seg_t[0]),
                t_end=float(seg_t[-1]),
                n_points=n_seg,
                period=1.0 / f_best,
                period_err=sigma_f / (f_best * f_best),
                frequency=f_best,
                frequency_err=sigma_f,
                chi2_reduced=chi2_red,
                amplitude=amplitude,
                power=float(1.0 - chi2_min / total) if total > 0.0 else float("nan"),
                at_window_edge=at_edge,
                pdot=pdot_val,
                pdot_err=pdot_err,
            )
        )

    if len(points) < 2:
        raise InsufficientDataError(
            f"period_profile: only {len(points)} usable intervals from "
            f"{time.size} points; lower points_per_interval, or the baseline is "
            "too short for the period"
        )
    return PeriodProfile(
        points=tuple(points),
        reference_period=float(period),
        n_harmonics=n_harm,
        n_samples=int(time.size),
        baseline=float(time[-1] - time[0]),
        period_window=float(window),
        meta={
            "period_profile": {
                "points_per_interval": per_interval,
                "n_intervals": n_intervals,
                "measure_pdot": measure_pdot,
                "weighted": weighted,
            },
            **dict(finite.meta),
        },
    )


def _interval_pdot(
    seg_t: FloatArray,
    seg_y: FloatArray,
    seg_w: FloatArray,
    weighted: bool,
    f_best: float,
    psettings: PdotSettings,
    backend: str,
) -> tuple[float, float]:
    """Interval-local dP/dt from a small joint search; NaN when it cannot run."""
    from cuperiod.pdot.scan import pdot_search

    err = 1.0 / np.sqrt(seg_w) if weighted else None
    try:
        res = pdot_search(
            LightCurve.from_arrays(seg_t, seg_y, err),
            period=1.0 / f_best,
            settings=psettings,
            backend=backend,
        )
    except (InsufficientDataError, ValueError):
        return float("nan"), float("nan")
    # One grid step of the rate axis is the resolution of this short interval.
    axis = res.pdot_over_p
    step = float(axis[1] - axis[0]) if axis.size > 1 else float("nan")
    return res.best.pdot, abs(step / f_best)


__all__ = ["PeriodPoint", "PeriodProfile", "period_profile"]
