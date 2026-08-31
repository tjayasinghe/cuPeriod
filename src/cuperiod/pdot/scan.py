"""Joint (period, dP/dt) grid search via time warping: :func:`pdot_search`.

A signal whose period drifts linearly, ``P(t) = P0 + Pdot*(t - t_ref)``, has the
quadratic phase (in cycles)

    phi(t) = f*dt + (1/2)*fdot*dt**2,        dt = t - t_ref,  fdot = -Pdot*f**2.

Writing ``a = Pdot/P`` (the fractional rate, 1/day), the phase factorizes as
``phi = f * tau`` with the *warped time*

    tau = dt * (1 - (a/2)*dt),

which does not depend on ``f``. So for each trial ``a`` the whole frequency axis
of the quadratic-phase model is exactly an *ordinary* periodogram of the warped
light curve — any registered method, any backend, unchanged. :func:`pdot_search`
steps a uniform ``a`` grid, re-evaluates the method on the warped times (reusing
its batch engine, e.g. the cached cufinufft plans), and stacks the rows into a
``(n_pdot, n_frequency)`` map whose refined best cell is the joint (P, dP/dt)
estimate.

Resolution and range follow the same phase-coherence logic as the frequency
grid: a frequency step ``1/(s*T)`` mis-phases the edges of the baseline by
``1/(2s)`` cycles, and an ``a`` step of ``4/(s*f*T**2)`` does the same through
the quadratic term, so both axes are sampled to equal phase smear. The default
extent of the ``a`` axis is set in accumulated drift cycles (``max_cycle_drift``)
rather than physical units, which keeps the row count baseline-independent
(``4 * samples_per_drift * max_cycle_drift + 1``) while reaching every drift
large enough to defocus a constant-period search.

The reference epoch is the weighted mean epoch of the finite points, where the
frequency and drift estimates decouple; the reported frequency is the
instantaneous frequency at that epoch.

``pdot_search`` localizes a drifting signal; for a *measurement* with honest
uncertainties, follow with :func:`cuperiod.phase_track` at the found period.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from pydantic_settings import BaseSettings

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import PdotSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, get_method
from cuperiod.pdot.result import PdotBest, PdotSearchResult

#: Methods whose trial grids carry method-specific structure (box durations /
#: transit templates) that a plain frequency window cannot represent. Their
#: science case — transit/eclipse timing — is served by :func:`phase_track` and
#: :func:`fit_ephemeris` instead.
_UNSUPPORTED: frozenset[str] = frozenset({"BLS", "TLS"})


def weighted_epoch(lc: LightCurve | MultiBandLightCurve) -> float:
    """Inverse-variance-weighted mean observation epoch of the finite points.

    At this epoch a fitted phase is uncorrelated with the fitted frequency, so
    it is the natural expansion point for a period-drift model (and the epoch
    the pre-whitening module references phases to). Bands of a multi-band input
    are pooled.
    """
    if isinstance(lc, MultiBandLightCurve):
        time, _, error, _ = lc.finite().stacked()
    else:
        finite = lc.finite()
        time, error = finite.time, finite.error
    if time.size == 0:
        raise InsufficientDataError("weighted_epoch: no finite points")
    if error is None:
        return float(np.mean(time))
    w = 1.0 / (error * error)
    return float(np.dot(w, time) / w.sum())


def _pdot_axis(
    settings: PdotSettings, f_center: float, baseline: float
) -> FloatArray:
    """The (dP/dt)/P axis: uniform, and containing an exact 0 by default."""
    step = 4.0 / (settings.samples_per_drift * f_center * baseline * baseline)
    if settings.pdot_over_p_range is not None:
        lo, hi = settings.pdot_over_p_range
        n = int(math.floor((hi - lo) / step)) + 1
        return lo + step * np.arange(n, dtype=np.float64)
    a_max = 8.0 * settings.max_cycle_drift / (f_center * baseline * baseline)
    # Keep the warp monotonic over the span: |a| * (T/2) must stay below 1.
    a_max = min(a_max, 1.0 / baseline)
    n_side = max(1, int(math.ceil(a_max / step)))
    return step * np.arange(-n_side, n_side + 1, dtype=np.float64)


def _frequency_window(
    f_center: float, baseline: float, settings: PdotSettings
) -> GridSpec:
    """A uniform frequency window centered on ``f_center`` (grid hits it exactly)."""
    df = 1.0 / (settings.samples_per_peak * baseline)
    k = max(1, int(math.ceil(settings.window_rayleigh / baseline / df)))
    values = f_center + df * np.arange(-k, k + 1, dtype=np.float64)
    values = values[values > 0.5 * df]
    if values.size < 3:
        raise InsufficientDataError(
            f"pdot_search: frequency window around {f_center} cycles/day collapsed; "
            "the period is too long for this baseline"
        )
    return GridSpec(kind="frequency", values=values, uniform=True)


def _row_settings(method_settings: BaseSettings) -> BaseSettings:
    """Per-row settings: identical physics, per-row extras (FAP) switched off."""
    if "fap_method" in type(method_settings).model_fields:
        return method_settings.model_copy(update={"fap_method": "none"})
    return method_settings


def _seed_frequency(
    m: PeriodogramMethod,
    work: LightCurve | MultiBandLightCurve,
    settings: BaseSettings,
    backend: str,
) -> float:
    """Best frequency from a standard (constant-period) run with default grid."""
    pg = _row_power(m, _default_grid(m, work, settings), work, settings, backend, None)
    peaks = pg.best_periods(1)
    if not peaks:
        raise InsufficientDataError(
            "pdot_search: seeding run found no usable peak; pass period=..."
        )
    return peaks[0].frequency


def _default_grid(
    m: PeriodogramMethod,
    work: LightCurve | MultiBandLightCurve,
    settings: BaseSettings,
) -> GridSpec:
    if isinstance(work, MultiBandLightCurve):
        time, value, error, _ = work.stacked()
        synthetic = LightCurve.from_arrays(time, value, error)
        return m.default_grid(synthetic, settings)
    return m.default_grid(work, settings)


def _row_power(
    m: PeriodogramMethod,
    grid: GridSpec,
    work: LightCurve | MultiBandLightCurve,
    settings: BaseSettings,
    backend: str,
    engine: object | None,
) -> Periodogram:
    if isinstance(work, MultiBandLightCurve):
        return m.multiband_power(grid, work, settings, backend, engine=engine)
    return m.power(grid, work, settings, backend, engine=engine)


def _warp(
    work: LightCurve | MultiBandLightCurve, t_ref: float, a: float
) -> LightCurve | MultiBandLightCurve:
    """The light curve on warped times ``tau = dt*(1 - (a/2)*dt)``, origin t_ref."""
    if isinstance(work, MultiBandLightCurve):
        bands = {
            name: replace(b, time=_warp_times(b.time, t_ref, a))
            for name, b in work.bands.items()
        }
        return MultiBandLightCurve.from_light_curves(bands, meta=work.meta)
    return replace(work, time=_warp_times(work.time, t_ref, a))


def _warp_times(time: FloatArray, t_ref: float, a: float) -> FloatArray:
    dt = time - t_ref
    return dt * (1.0 - (0.5 * a) * dt)


def _parabolic_offset(sm: float, s0: float, sp: float) -> float:
    """Sub-cell vertex offset (in steps) of a 3-point parabola around a maximum."""
    denom = sm - 2.0 * s0 + sp
    if not np.isfinite(denom) or denom >= 0.0:
        return 0.0
    off = 0.5 * (sm - sp) / denom
    return float(np.clip(off, -0.5, 0.5))


def pdot_search(
    data: Any,
    *,
    period: float | None = None,
    method: str = "GLS",
    settings: PdotSettings | None = None,
    method_settings: BaseSettings | None = None,
    grid: GridSpec | None = None,
    backend: str = "auto",
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
) -> PdotSearchResult:
    """Search a light curve jointly over period and period derivative.

    Evaluates ``method`` on quadratically time-warped copies of the light curve
    — one warp per trial fractional rate (dP/dt)/P — so every registered
    periodogram (GLS, PDM, CE, String-Length, MHAOV, SuperSmoother; single- or
    multi-band) becomes sensitive to a linearly drifting period without any
    change to the method itself. See the module docstring for the model and the
    grid conventions.

    Parameters
    ----------
    data : various
        The light curve, in any form :func:`cuperiod.to_input` accepts.
    period : float, optional
        Center of the frequency window (days). ``None`` runs the method once on
        its default grid first and centers on the best peak.
    method : str, default "GLS"
        Wrapped periodogram method. BLS and TLS are not supported (their grids
        carry box/transit structure a plain window cannot represent); use
        :func:`cuperiod.phase_track` / :func:`cuperiod.fit_ephemeris` for
        transit-timing style dP/dt work.
    settings : PdotSettings, optional
        Search-axis settings (window width, oversampling, drift range).
    method_settings : pydantic settings, optional
        The wrapped method's own settings (e.g. ``PDMSettings(n_bins=15)``).
    grid : GridSpec, optional
        Explicit frequency grid, overriding the window construction. A uniform
        frequency grid keeps the fast NUFFT path for GLS.
    backend : str, default "auto"
        Backend request, resolved once and reused for every row.
    columns, domain
        Passed to :func:`cuperiod.to_input` for table inputs.

    Returns
    -------
    PdotSearchResult
        The 2-D statistic map plus the refined best (period, dP/dt).

    Examples
    --------
    >>> res = pdot_search((t, mag, err), period=0.5213)   # doctest: +SKIP
    >>> res.best.pdot_s_per_yr                            # doctest: +SKIP
    """
    from cuperiod.api import to_input

    m = get_method(method)
    if m.name in _UNSUPPORTED:
        raise ValueError(
            f"pdot_search does not support {m.name} (its trial grid carries "
            "box/transit structure); localize with GLS/PDM/CE, or measure a "
            "transit-timing dP/dt with phase_track / fit_ephemeris"
        )
    psettings = settings or PdotSettings()
    msettings = m.coerce_settings(method_settings)
    resolved = m.resolve_backend(backend)

    lc = to_input(data, columns=columns, domain=domain)
    work: LightCurve | MultiBandLightCurve
    if isinstance(lc, MultiBandLightCurve):
        if not m.supports_multiband:
            raise ValueError(f"{m.name} does not support multi-band input")
        work = lc.finite()
        time, _, _, _ = work.stacked()
        n_samples = int(time.size)
        baseline = float(time.max() - time.min()) if n_samples else 0.0
    else:
        single = lc.in_domain(m.natural_domain) if m.natural_domain else lc
        work = single.finite()
        n_samples = work.n
        baseline = work.baseline
    if n_samples < psettings.min_detections:
        raise InsufficientDataError(
            f"pdot_search: {n_samples} finite points < min_detections "
            f"{psettings.min_detections}"
        )
    if baseline <= 0.0:
        raise InsufficientDataError("pdot_search: no usable time baseline")

    row_settings = _row_settings(msettings)

    # -- axes ---------------------------------------------------------------------
    if period is not None:
        if period <= 0.0:
            raise ValueError(f"period must be positive, got {period}")
        f_center = 1.0 / period
    elif grid is not None:
        freq = grid.frequency
        f_center = float(0.5 * (freq[0] + freq[-1]))
    else:
        f_center = _seed_frequency(m, work, row_settings, resolved)
    fgrid = grid if grid is not None else _frequency_window(
        f_center, baseline, psettings
    )
    frequency = fgrid.frequency
    axis = _pdot_axis(psettings, f_center, baseline)
    n_cells = int(axis.size) * int(frequency.size)
    if n_cells > psettings.max_grid_cells:
        raise ValueError(
            f"pdot_search: {axis.size} x {frequency.size} = {n_cells} grid cells "
            f"exceeds max_grid_cells={psettings.max_grid_cells}; narrow "
            "pdot_over_p_range / window_rayleigh or raise the guard"
        )

    # -- scan ---------------------------------------------------------------------
    t_ref = weighted_epoch(work)
    engine = m.make_engine(resolved, row_settings)
    power = np.empty((axis.size, frequency.size), dtype=np.float64)
    backend_actual = resolved
    for j, a in enumerate(axis):
        pg = _row_power(
            m, fgrid, _warp(work, t_ref, float(a)), row_settings, resolved, engine
        )
        if pg.power.size != frequency.size:
            raise RuntimeError(
                f"pdot_search: {m.name} returned {pg.power.size} samples for a "
                f"{frequency.size}-point grid"
            )
        power[j] = pg.power
        if j == 0:
            backend_actual = pg.backend

    # -- best cell + separable parabolic refinement --------------------------------
    score = power if m.objective_sense == "max" else -power
    flat = int(np.nanargmax(score))
    j0_raw, i0_raw = np.unravel_index(flat, score.shape)
    j0, i0 = int(j0_raw), int(i0_raw)
    df = float(frequency[1] - frequency[0]) if frequency.size > 1 else 0.0
    da = float(axis[1] - axis[0]) if axis.size > 1 else 0.0
    f_best = float(frequency[i0])
    if 0 < i0 < frequency.size - 1:
        f_best += df * _parabolic_offset(
            float(score[j0, i0 - 1]), float(score[j0, i0]), float(score[j0, i0 + 1])
        )
    a_best = float(axis[j0])
    if 0 < j0 < axis.size - 1:
        a_best += da * _parabolic_offset(
            float(score[j0 - 1, i0]), float(score[j0, i0]), float(score[j0 + 1, i0])
        )
    j_zero = int(np.argmin(np.abs(axis)))
    zero_power = float(power[j_zero, int(np.nanargmax(score[j_zero]))])
    best = PdotBest(
        period=1.0 / f_best,
        pdot=a_best / f_best,
        frequency=f_best,
        fdot=-a_best * f_best,
        pdot_over_p=a_best,
        power=float(power[j0, i0]),
        zero_pdot_power=zero_power,
        t_ref=t_ref,
    )
    meta = {
        "pdot_search": {
            "f_center": f_center,
            "window_rayleigh": psettings.window_rayleigh,
            "samples_per_peak": psettings.samples_per_peak,
            "max_cycle_drift": psettings.max_cycle_drift,
            "samples_per_drift": psettings.samples_per_drift,
            "seeded": period is None and grid is None,
        },
        **dict(work.meta),
    }
    return PdotSearchResult(
        method=m.name,
        backend=backend_actual,
        frequency=np.asarray(frequency, dtype=np.float64),
        pdot_over_p=axis,
        power=power,
        objective_sense=m.objective_sense,
        best=best,
        n_samples=n_samples,
        baseline=baseline,
        t_ref=t_ref,
        meta=meta,
    )


__all__ = ["pdot_search", "weighted_epoch"]
