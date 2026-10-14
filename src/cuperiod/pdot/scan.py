"""Joint (period, dP/dt) grid search via time warping: :func:`pdot_search`.

A constant period derivative means ``P`` grows linearly in *time*; the phase
(in cycles) is then exactly

    phi(t) = f * tau,     tau = ln(1 + a*dt) / a,     a = Pdot/P,

with ``dt = t - t_ref`` and ``a`` the fractional rate in 1/day (Mikulasek et
al. 2012; the same logarithmic phase PHOEBE integrates for its ``dpdt``). The
warp ``tau`` does not depend on ``f``, so for each trial ``a`` the whole
frequency axis of the drifting-period model is exactly an *ordinary*
periodogram of the warped light curve — any registered method, any backend,
unchanged. To second order ``tau ~ dt*(1 - (a/2)*dt)``, the familiar quadratic
ephemeris; the exact form matters at the edges of the search range for light
curves spanning few hundred cycles, where the truncated warp mis-phases by
whole cycles. :func:`pdot_search` steps a uniform ``a`` grid, re-evaluates the
method on the warped times (reusing its batch engine, e.g. the cached
cufinufft plans), and stacks the rows into a ``(n_pdot, n_frequency)`` map
whose refined best cell is the joint (P, dP/dt) estimate.

The frequency-independence of ``tau`` is a property of the *family* of period
laws, not of this one law, so ``model=`` selects which is searched:
``"constant-pdot"`` above, or ``"exponential"`` (``P = P0*exp(a*dt)``, warp
``tau = (1 - exp(-a*dt))/a``), the accelerating decay of an inspiralling or
mass-transferring binary. Both are parameterized by the same ``a`` at
``t_ref``, so their ``period`` and ``pdot`` are directly comparable and only
their behaviour away from the reference epoch differs. See
:mod:`cuperiod.pdot.models`.

Resolution and range follow the same phase-coherence logic as the frequency
grid: a frequency step ``1/(s*T)`` mis-phases the edges of the baseline by
``1/(2s)`` cycles, and an ``a`` step of ``4/(s*f*T**2)`` does the same through
the quadratic term, so both axes are sampled to equal phase smear (the
template-bank mismatch of the default ``s = 5`` is ~3.5%; ``s = 2`` matches
the pulsar-search convention at ~22% for survey speed). Methods that resolve
structure *sharper* than a sinusoid need a proportionally finer step — a phase
feature of width ``1/h`` cycles smears ``h`` times sooner — so the step is
divided by a per-method harmonic scale (MHAOV: its harmonic count; PDM and
conditional entropy: half the phase-bin count; SuperSmoother: half the inverse
of its smallest span). The default extent is set in accumulated drift cycles
(``max_cycle_drift``) rather than physical units, which keeps the row count
baseline-independent (``4 * samples_per_drift * max_cycle_drift *
harmonic_scale + 1``) while reaching every drift large enough to defocus a
constant-period search. Only a baseline under ``8 * max_cycle_drift`` cycles
(80 by default) has fewer rows: there the range is trimmed so the period stays
positive across the span (the constant-dP/dt model's domain).

The reference epoch is the weighted mean epoch of the finite points, and the
reported frequency is the instantaneous frequency there. That choice zeroes the
first moment of the times, which decouples the *phase* from the frequency and
removes the leading frequency-drift correlation. It does not by itself make the
two axes orthogonal: the residual correlation is set by the third central moment
of the sampling, so it vanishes only when the observations are symmetric about
the reference epoch. Sampling that thins or densifies strongly across the run
leaves a tilted ridge in the map — it widens the joint uncertainty but does not
bias the best cell.

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
from cuperiod.core.lightcurve import (
    LightCurve,
    MultiBandLightCurve,
    raise_if_unusable,
)
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, get_method
from cuperiod.pdot.models import PeriodModel, get_model
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


def _harmonic_scale(m: PeriodogramMethod, settings: BaseSettings) -> float:
    """How much sharper than a sinusoid the method's phase structure is.

    A feature of width ``1/h`` cycles loses coherence ``h`` times sooner under
    a drift-step error, so the (dP/dt)/P step must shrink by this factor
    (the criterion of Katz et al. 2021 for their conditional-entropy dP/dt
    search, generalized per method).
    """
    key = "".join(ch for ch in m.name if ch.isalnum()).upper()
    if key == "MHAOV":
        return float(max(1, int(getattr(settings, "n_harmonics", 3))))
    if key in {"PDM", "CE"}:
        bins = getattr(settings, "n_bins", None)
        if bins is None:
            bins = getattr(settings, "n_phase_bins", 10)
        return max(1.0, 0.5 * float(bins if bins is not None else 10))
    if key == "STRINGLENGTH":
        return 4.0
    if key == "SUPERSMOOTHER":
        spans = getattr(settings, "primary_spans", None)
        if spans:
            return max(1.0, 0.5 / float(min(spans)))
        return 10.0
    return 1.0


def _pdot_axis(
    settings: PdotSettings,
    f_center: float,
    baseline: float,
    h_scale: float,
    model: PeriodModel | None = None,
) -> FloatArray:
    """The (dP/dt)/P axis: uniform, and containing an exact 0 whenever it can.

    The default axis is symmetric with ``4 * samples_per_drift *
    max_cycle_drift * scale + 1`` rows, trimmed (never extended) to stay inside
    the period model's domain over the baseline. An explicit
    ``pdot_over_p_range`` that spans zero is laid on a zero-anchored lattice
    whose step divides the wider bound, so the ``dP/dt = 0`` row — the
    constant-period comparison every caller reads — exists and that bound is
    sampled exactly; a range excluding zero is sampled end to end. Either way
    the step is never coarser than the derived one.
    """
    scale = h_scale if settings.harmonic_scale is None else settings.harmonic_scale
    scale = max(1.0, float(scale))
    step = 4.0 / (
        settings.samples_per_drift * f_center * baseline * baseline * scale
    )
    if settings.pdot_over_p_range is not None:
        lo, hi = (float(v) for v in settings.pdot_over_p_range)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(
                f"pdot_over_p_range must be finite, got ({lo}, {hi})"
            )
        if lo <= 0.0 <= hi:
            reach = max(-lo, hi)
            n_side = max(1, math.ceil(reach / step - 1e-9))
            step = reach / n_side
            k_lo = math.ceil(lo / step - 1e-9)
            k_hi = math.floor(hi / step + 1e-9)
            return step * np.arange(k_lo, k_hi + 1, dtype=np.float64)
        n_steps = max(1, math.ceil((hi - lo) / step - 1e-9))
        return np.linspace(lo, hi, n_steps + 1, dtype=np.float64)
    # a_max / step is 2 * max_cycle_drift * samples_per_drift * scale exactly;
    # computing it as that float ratio rounds above the integer for some (f, T)
    # and added a row on each side.
    n_side = max(
        1,
        math.ceil(
            2.0 * settings.max_cycle_drift * settings.samples_per_drift * scale
            - 1e-9
        ),
    )
    # Stay inside the model's domain — for the singular constant-dP/dt warp
    # that means the period must not reach zero within the span. The cap is
    # taken over the whole baseline, which bounds the reach on either side of
    # any reference epoch inside it; a ceil() here overshot it by up to a step.
    pmodel = get_model(model)
    limit = pmodel.max_abs_rate(baseline)
    n_fit = n_side
    if math.isfinite(limit):
        n_fit = math.floor(limit * (1.0 - 1e-9) / step)
    if n_fit < 1:
        raise InsufficientDataError(
            f"pdot_search: the baseline spans only {f_center * baseline:.2f} "
            f"cycles, so even one (dP/dt)/P step ({step:.3e} 1/day) leaves the "
            f"{pmodel.name} model's domain (|a| < {limit:.3e} 1/day over "
            f"{baseline:.1f} d); too few cycles to measure a period change"
        )
    n_side = min(n_side, n_fit)
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


#: Warped-time values per batched block of rows (~64 MB of float64).
_ROW_BLOCK_VALUES = 1 << 23


def _batched_rows(
    m: PeriodogramMethod,
    grid: GridSpec,
    work: LightCurve | MultiBandLightCurve,
    t_ref: float,
    axis: FloatArray,
    model: PeriodModel,
    settings: BaseSettings,
    backend: str,
    engine: object | None,
    power: FloatArray,
) -> int:
    """Fill leading rows of ``power`` through the method's batched-rows path.

    A method with a device path (GLS on cufinufft, PDM on cupy; see
    :meth:`~cuperiod.methods.base.PeriodogramMethod.power_rows`) takes the warped
    times of a block of rows at once and returns their rows in one transfer, where
    the per-row loop uploaded the curve and waited on the device once per row. The
    rows are the ones the loop would produce. Returns how many leading rows were
    filled: 0 when the method has no batched path, fewer than all when it declined
    a later block (the caller loops over the rest).
    """
    if isinstance(work, MultiBandLightCurve) or not m.supports_power_rows(
        grid, backend
    ):
        return 0
    dt = work.time - t_ref
    rows_per_block = max(1, _ROW_BLOCK_VALUES // max(1, dt.size))
    for start in range(0, int(axis.size), rows_per_block):
        stop = min(start + rows_per_block, int(axis.size))
        times = np.stack([model.warp(dt, float(a)) for a in axis[start:stop]])
        block = m.power_rows(grid, work, times, settings, backend, engine)
        if block is None:
            return start
        if block.shape != (stop - start, power.shape[1]):
            raise RuntimeError(
                f"pdot_search: {m.name} returned a {block.shape} block for "
                f"{stop - start} rows of a {power.shape[1]}-point grid"
            )
        power[start:stop] = block
    return int(axis.size)


def _warp(
    work: LightCurve | MultiBandLightCurve,
    t_ref: float,
    a: float,
    model: PeriodModel,
) -> LightCurve | MultiBandLightCurve:
    """The light curve on ``model``-warped times, origin ``t_ref``."""
    if isinstance(work, MultiBandLightCurve):
        bands = {
            name: replace(b, time=model.warp(b.time - t_ref, a))
            for name, b in work.bands.items()
        }
        return MultiBandLightCurve.from_light_curves(bands, meta=work.meta)
    return replace(work, time=model.warp(work.time - t_ref, a))


def warp_dt(
    dt: FloatArray, a: float, model: str | PeriodModel | None = None
) -> FloatArray:
    """Exact period-drift time warp for ``dt`` measured from the reference epoch.

    The phase of a drifting period is ``f * tau`` exactly, whatever the model
    (see :mod:`cuperiod.pdot.models`). Defaults to ``"constant-pdot"``, whose
    warp is ``tau = ln(1 + a*dt)/a``; ``log1p`` keeps precision as ``a*dt -> 0``,
    and the caller must keep ``1 + a*dt > 0`` (the period stays positive).
    """
    return get_model(model).warp(np.asarray(dt, dtype=np.float64), a)


def _parabolic_offset(sm: float, s0: float, sp: float) -> float:
    """Sub-cell vertex offset (in steps) of a 3-point parabola around a maximum."""
    denom = sm - 2.0 * s0 + sp
    if not np.isfinite(denom) or denom >= 0.0:
        return 0.0
    off = 0.5 * (sm - sp) / denom
    return float(np.clip(off, -0.5, 0.5))


def _ridge_correlation(score: FloatArray, j0: int, i0: int) -> float:
    """Frequency/rate correlation at the peak from the local curvature.

    Fits the 3x3 stencil's second derivatives and reads the correlation of the
    implied covariance ``-H^-1``. Scale-free, so the grid steps cancel: the
    value is the same in physical units. See :attr:`PdotBest.ridge_correlation`.
    """
    ny, nx = score.shape
    if not (0 < j0 < ny - 1 and 0 < i0 < nx - 1):
        return float("nan")
    block = np.asarray(score[j0 - 1 : j0 + 2, i0 - 1 : i0 + 2], dtype=np.float64)
    if not np.all(np.isfinite(block)):
        return float("nan")
    s_ff = block[1, 2] - 2.0 * block[1, 1] + block[1, 0]
    s_aa = block[2, 1] - 2.0 * block[1, 1] + block[0, 1]
    s_fa = 0.25 * (block[2, 2] - block[2, 0] - block[0, 2] + block[0, 0])
    if s_ff >= 0.0 or s_aa >= 0.0:  # not a peak along both axes
        return float("nan")
    if s_ff * s_aa - s_fa * s_fa <= 0.0:  # degenerate/saddle: no finite ellipse
        return float("nan")
    return float(np.clip(s_fa / math.sqrt(s_ff * s_aa), -1.0, 1.0))


def pdot_search(
    data: Any,
    *,
    period: float | None = None,
    method: str = "GLS",
    model: str | PeriodModel | None = None,
    settings: PdotSettings | None = None,
    method_settings: BaseSettings | None = None,
    grid: GridSpec | None = None,
    backend: str = "auto",
    columns: ColumnMap | None = None,
    domain: Domain | str | None = None,
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
        Center of the frequency window (days; positive and finite). ``None``
        runs the method once on its default grid first and centers on the best
        peak.
    method : str, default "GLS"
        Wrapped periodogram method. BLS and TLS are not supported (their grids
        carry box/transit structure a plain window cannot represent); use
        :func:`cuperiod.phase_track` / :func:`cuperiod.fit_ephemeris` for
        transit-timing style dP/dt work.
    model : str, optional
        Period-evolution law: ``"constant-pdot"`` (default; ``dP/dt`` constant,
        the classical quadratic ephemeris) or ``"exponential"``
        (``d(ln P)/dt`` constant, the accelerating decay of a mass-transferring
        or inspiralling binary). Both are parameterized by the same
        ``a = (dP/dt)/P`` at ``t_ref`` and so return directly comparable
        ``best.period`` / ``best.pdot``; they differ in how the rate evolves
        away from that epoch. See :mod:`cuperiod.pdot.models`.
    settings : PdotSettings, optional
        Search-axis settings (window width, oversampling, drift range).
    method_settings : pydantic settings, optional
        The wrapped method's own settings (e.g. ``PDMSettings(n_bins=15)``).
    grid : GridSpec, optional
        Explicit frequency grid, overriding the window construction. A uniform
        frequency grid keeps the fast NUFFT path for GLS.
    backend : str, default "auto"
        Backend request, resolved once and reused for every row. ``"auto"``
        defers to ``method_settings.backend`` (and so to its
        ``CUPERIOD_<METHOD>_BACKEND`` override), as :func:`cuperiod.periodogram`
        does.
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
    pmodel = get_model(model)
    if period is not None and not (math.isfinite(period) and period > 0.0):
        raise ValueError(f"period must be positive and finite, got {period}")
    msettings = m.coerce_settings(method_settings)
    resolved = m.resolve_backend_for(backend, msettings)

    lc = to_input(data, columns=columns, domain=domain)
    # A placeholder error column (all 0, -1 or NaN) empties the curve; name that
    # rather than the "no usable time baseline" the empty curve would produce.
    raise_if_unusable(lc, "pdot_search")
    work: LightCurve | MultiBandLightCurve
    if isinstance(lc, MultiBandLightCurve):
        if not m.supports_multiband:
            raise ValueError(f"{m.name} does not support multi-band input")
        work = lc.finite()
        time, _, _, _ = work.stacked()
        n_samples = int(time.size)
        baseline = float(time.max() - time.min()) if n_samples else 0.0
        t_lo = float(time.min()) if n_samples else 0.0
        t_hi = float(time.max()) if n_samples else 0.0
    else:
        single = lc.in_domain(m.natural_domain) if m.natural_domain else lc
        work = single.finite()
        n_samples = work.n
        baseline = work.baseline
        t_lo = float(work.time.min()) if n_samples else 0.0
        t_hi = float(work.time.max()) if n_samples else 0.0
    if n_samples < psettings.min_detections:
        raise InsufficientDataError(
            f"pdot_search: {n_samples} finite points < min_detections "
            f"{psettings.min_detections}"
        )
    if baseline <= 0.0:
        raise InsufficientDataError("pdot_search: no usable time baseline")
    t_ref = weighted_epoch(work)

    row_settings = _row_settings(msettings)

    # -- axes ---------------------------------------------------------------------
    if period is not None:
        f_center = 1.0 / period
    elif grid is not None:
        freq = grid.frequency
        f_center = float(0.5 * (freq[0] + freq[-1]))
    else:
        f_center = _seed_frequency(m, work, row_settings, resolved)
    fgrid = grid if grid is not None else _frequency_window(
        f_center, baseline, psettings
    )
    # Every row comes back from Periodogram.from_spectrum sorted by ascending
    # frequency, so that is the axis the rows live on (GridSpec refuses
    # descending grids; sorting keeps the invariant local, as sliding does).
    frequency = np.sort(np.asarray(fgrid.frequency, dtype=np.float64))
    axis = _pdot_axis(
        psettings, f_center, baseline, _harmonic_scale(m, msettings), pmodel
    )
    n_cells = int(axis.size) * int(frequency.size)
    if n_cells > psettings.max_grid_cells:
        raise ValueError(
            f"pdot_search: {axis.size} x {frequency.size} = {n_cells} grid cells "
            f"exceeds max_grid_cells={psettings.max_grid_cells}; narrow "
            "pdot_over_p_range / window_rayleigh or raise the guard"
        )
    dt_extent = max(t_hi - t_ref, t_ref - t_lo)
    a_worst = float(max(abs(axis[0]), abs(axis[-1])))
    if a_worst >= pmodel.max_abs_rate(dt_extent):
        raise ValueError(
            f"pdot_search: the (dP/dt)/P range is outside the {pmodel.name} "
            f"model's domain at the span edge (|a| = {a_worst:.3e} 1/day over "
            f"{dt_extent:.1f} d)"
            + (
                " — the period would pass through zero; narrow "
                "pdot_over_p_range or use model='exponential', which has no "
                "singular epoch"
                if pmodel.singular
                else "; narrow pdot_over_p_range"
            )
        )

    # -- scan ---------------------------------------------------------------------
    engine = m.make_engine(resolved, row_settings)
    power = np.empty((axis.size, frequency.size), dtype=np.float64)
    backend_actual = resolved
    done = _batched_rows(
        m, fgrid, work, t_ref, axis, pmodel, row_settings, resolved, engine, power
    )
    for j in range(done, int(axis.size)):
        pg = _row_power(
            m,
            fgrid,
            _warp(work, t_ref, float(axis[j]), pmodel),
            row_settings,
            resolved,
            engine,
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
    # Local spacing: exact on the uniform default grid, and still sensible on
    # a non-uniform user-supplied grid (e.g. a log period grid).
    if frequency.size > 1:
        lo, hi = max(i0 - 1, 0), min(i0 + 1, frequency.size - 1)
        df = float(frequency[hi] - frequency[lo]) / (hi - lo)
    else:
        df = 0.0
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
        model=pmodel.name,
        ridge_correlation=_ridge_correlation(score, j0, i0),
    )
    meta = {
        "pdot_search": {
            "f_center": f_center,
            "model": pmodel.name,
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


__all__ = ["pdot_search", "warp_dt", "weighted_epoch"]
