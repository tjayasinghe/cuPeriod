"""Uncertainties on extracted frequencies, amplitudes and phases.

Three estimators, in increasing order of cost and decreasing order of assumption:

``"covariance"`` (the default)
    The linearised least-squares covariance ``s^2 (J^T W J)^{-1}`` of the *joint* fit,
    computed by :mod:`cuperiod.prewhiten.fit`. It is the only one of the three that
    accounts for correlations *between* components, which matters as soon as two
    frequencies sit within a few Rayleigh widths of each other — exactly the situation
    in a dense δ Scuti or g-mode spectrum.
``"analytic"``
    The closed-form expressions of Montgomery & O'Donoghue (1999) — the numbers most
    pulsation papers quote. They assume an isolated sinusoid in white noise and ignore
    the correlations between components, so for close frequencies they are a lower
    bound; useful for comparison with the literature. For a weighted fit the noise
    level is the weighted residual rms, so unequal error bars do not inflate them.
``"bootstrap"``
    Perturb the residuals, re-fit, and take the scatter. Makes no linearity assumption
    and needs no error bars, at the price of ``n_resamples`` extra fits.

The covariance and analytic errors can be inflated by the Schwarzenberg-Czerny (1991)
correlation factor: real photometry has residuals that are correlated point-to-point
(instrumental drifts, unresolved modes), so the *effective* number of independent
samples is smaller than ``N`` and the formal errors are optimistic. The factor is read
off the residuals in time order, whatever order the rows came in. The bootstrap is
never inflated — it already uses the residuals as they are.

Every estimator also reports the error of the constant offset, inflated alike.

A constrained fit (one carrying :attr:`~cuperiod.prewhiten.fit.MultiSineFit.detail`)
is characterised under its constraints: a fixed frequency, amplitude or phase has an
error of exactly zero, a tied frequency inherits its parents' (their covariance, the
scatter of the tied replicate, or ``sqrt(sum n_j^2 sigma_j^2)`` analytically), and the
per-subset amplitudes, phases and zero points get errors of their own
(:attr:`Uncertainties.subset_amplitude` and friends).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.prewhiten.fit import MultiSineFit, Refinement, fit_multisine

if TYPE_CHECKING:
    from cuperiod.prewhiten.constraints import FitConstraints

#: How component uncertainties are estimated.
UncertaintyMethod = Literal["covariance", "analytic", "bootstrap"]

#: Largest component count whose bootstrap replicates get a full simultaneous fit when
#: the caller's policy pins frequencies (mirrors the default of
#: ``PreWhitenSettings.max_simultaneous``); above it the replicates are swept
#: cyclically, at least :data:`_MIN_CYCLIC_SWEEPS` times.
_PROMOTED_MAX_SIMULTANEOUS: Final = 60

#: Fewest cyclic sweeps a promoted bootstrap replicate gets. One block-coordinate sweep
#: started at the best fit stops short of a replicate's joint optimum when components
#: are correlated, and the frequency scatter of close pairs came out 30-40% low.
_MIN_CYCLIC_SWEEPS: Final = 3


# Equality is identity (eq=False): the generated __eq__ would compare the arrays and
# raise on their ambiguous truth value, and frozen+eq would make hash() fail on them.
@dataclass(frozen=True, eq=False)
class Uncertainties:
    """1-sigma uncertainties for every component of a multi-sine solution.

    Attributes
    ----------
    frequency, amplitude, phase : numpy.ndarray
        Per-component 1-sigma errors (cycles/day, input units, radians).
    method : str
        Which estimator produced them.
    correlation_factor : float
        The Schwarzenberg-Czerny ``D``; errors were multiplied by ``sqrt(D)``
        (1.0 when the correction is off or the estimator is the bootstrap).
    offset : float
        1-sigma error of the constant offset by the same estimator and inflation (NaN
        when the fit had no free offset).
    subset_amplitude, subset_phase : numpy.ndarray or None
        ``(K, S_v)`` errors of the per-subset amplitudes and phases of a constrained
        fit with variation subsets (NaN where the quantity does not vary); ``None``
        otherwise.
    subset_offset : numpy.ndarray or None
        ``(S_o,)`` errors of the per-subset zero points; ``None`` without them.
    """

    frequency: FloatArray
    amplitude: FloatArray
    phase: FloatArray
    method: str
    correlation_factor: float
    offset: float = float("nan")
    subset_amplitude: FloatArray | None = None
    subset_phase: FloatArray | None = None
    subset_offset: FloatArray | None = None


def correlation_factor(residuals: FloatArray) -> float:
    """Schwarzenberg-Czerny (1991) correlation factor ``D`` from residual sign runs.

    Counts the mean length of a run of same-sign residuals. Independent residuals give a
    mean run length of exactly 2, so the return value is normalised by that: white noise
    yields ``D = 1`` and correlated residuals yield ``D > 1``. Formal uncertainties are
    then multiplied by ``sqrt(D)``, which is the standard prescription for reporting
    realistic frequency errors from ground- and space-based photometry.

    Parameters
    ----------
    residuals : numpy.ndarray
        Fit residuals in *time* order: the runs are only meaningful between
        consecutive observations. :func:`component_uncertainties` orders them by time
        itself, so rows may arrive in any order there.

    Returns
    -------
    float
        ``D >= 1``. Returns 1.0 for fewer than three usable residuals.
    """
    r = np.asarray(residuals, dtype=np.float64)
    signs = np.sign(r[np.isfinite(r) & (r != 0.0)])
    if signs.size < 3:
        return 1.0
    n_runs = 1 + int(np.count_nonzero(np.diff(signs) != 0.0))
    mean_run = float(signs.size) / float(n_runs)
    return max(mean_run / 2.0, 1.0)


def analytic_uncertainties(
    n_samples: int,
    baseline: float,
    residual_sigma: float,
    amplitude: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Montgomery & O'Donoghue (1999) analytic errors for isolated sinusoids.

    .. math::

        \\sigma_A = \\sqrt{2/N}\\,\\sigma, \\qquad
        \\sigma_\\phi = \\sqrt{2/N}\\,\\sigma/A, \\qquad
        \\sigma_f = \\sqrt{6/N}\\,\\frac{\\sigma}{\\pi T A}

    Parameters
    ----------
    n_samples : int
        Number of data points.
    baseline : float
        Total time span ``T`` in days.
    residual_sigma : float
        Standard deviation of the residuals after the fit.
    amplitude : numpy.ndarray
        Component amplitudes.

    Returns
    -------
    tuple of numpy.ndarray
        ``(sigma_f, sigma_A, sigma_phase)`` in cycles/day, input units, and radians.
    """
    amp = np.asarray(amplitude, dtype=np.float64)
    if n_samples <= 0 or not np.isfinite(residual_sigma):
        nan = np.full(amp.shape, np.nan, dtype=np.float64)
        return nan, nan.copy(), nan.copy()
    sigma_a = np.full(amp.shape, np.sqrt(2.0 / n_samples) * residual_sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_amp = np.where(amp > 0.0, amp, np.nan)
        sigma_phase = sigma_a / safe_amp
        sigma_f = (
            np.sqrt(6.0 / n_samples)
            * residual_sigma
            / (np.pi * baseline * safe_amp)
            if baseline > 0.0
            else np.full(amp.shape, np.nan)
        )
    return np.asarray(sigma_f), sigma_a, np.asarray(sigma_phase)


def bootstrap_uncertainties(
    time: FloatArray,
    error: FloatArray | None,
    fit: MultiSineFit,
    *,
    n_resamples: int = 200,
    seed: int = 0,
    refine: Refinement = "simultaneous",
    sweeps: int = 1,
    frequency_bounds: tuple[Any, Any] | None = None,
    max_nfev: int = 200,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Residual (wild) bootstrap errors for a fitted solution.

    Each replicate adds the fit residuals, with random signs, back onto the best-fit
    model and re-fits from the current solution; the scatter of the replicate
    parameters is the uncertainty. This is the *wild* bootstrap (Wu 1986; Mammen
    1993): every residual stays on its own observation, so heteroscedastic noise is
    reproduced point by point. Pooling the residuals and drawing them with
    replacement instead put the residuals of noisy points onto precise, heavily
    weighted ones and inflated the errors of weighted fits several-fold. The
    residuals are scaled by ``sqrt(N / (N - M))`` for the ``M`` fitted parameters,
    which the fit has absorbed. This makes no linearity assumption and needs no error
    bars, but it does treat the residuals as independent — if they are strongly
    correlated, prefer the correlation-factor-inflated covariance errors.

    Phases are combined on the unit circle so the wrap at ``2*pi`` cannot inflate the
    scatter of a phase that happens to sit near zero.

    Parameters
    ----------
    time : numpy.ndarray
        Observation times (days).
    error : numpy.ndarray, optional
        1-sigma uncertainties, used to weight the replicate fits.
    fit : MultiSineFit
        The solution to bootstrap around.
    n_resamples : int, default 200
        Number of replicates.
    seed : int, default 0
        Seed for the resampling RNG (results are reproducible).
    refine, sweeps
        Refinement policy for the replicate fits; ``"simultaneous"`` by default, like
        the final polish of :func:`~cuperiod.prewhiten.prewhiten`, so correlated
        components are re-optimised jointly. ``"none"`` and ``"last"`` are
        extraction-loop policies that pin some or all frequencies, which would report
        exactly zero frequency scatter for the pinned components — they are promoted
        to ``"simultaneous"`` (or, above 60 components, to at least three
        ``"cyclic"`` sweeps).
    frequency_bounds : (lower, upper), optional
        Per-frequency refinement boxes for the replicate fits (see
        :func:`~cuperiod.prewhiten.fit.fit_multisine`). Keeps a replicate from sliding
        onto a close neighbour, exactly as in the fit being characterised.
    max_nfev : int, default 200
        Optimiser evaluation cap per replicate fit.
    should_stop : callable, optional
        Polled before every replicate; once it returns True the remaining replicates
        are skipped and the errors come from those already fitted (NaN when fewer
        than two). The caller knows it asked to stop.

    Returns
    -------
    tuple of numpy.ndarray
        ``(sigma_f, sigma_A, sigma_phase)``.

    Notes
    -----
    A constrained fit is re-fitted under the same constraints, starting its non-linear
    phases from the fitted values, around the exact (per-subset) model. A fixed
    quantity's error is exactly zero; a tied frequency's is the scatter of its tied
    replicate value.
    """
    result = _bootstrap(
        time, error, fit,
        n_resamples=n_resamples, seed=seed, refine=refine, sweeps=sweeps,
        frequency_bounds=frequency_bounds, max_nfev=max_nfev, should_stop=should_stop,
    )
    return result.frequency, result.amplitude, result.phase


@dataclass(frozen=True, eq=False)
class _BootstrapScatter:
    frequency: FloatArray
    amplitude: FloatArray
    phase: FloatArray
    offset: float
    subset_amplitude: FloatArray | None = None
    subset_phase: FloatArray | None = None
    subset_offset: FloatArray | None = None


def _circular_sigma(phasors: Any, axis: int = 0) -> FloatArray:
    """Circular standard deviation ``sqrt(-2 ln R)`` of unit phasors."""
    resultant = np.clip(np.abs(np.mean(phasors, axis=axis)), 1e-12, 1.0)
    out: FloatArray = np.sqrt(-2.0 * np.log(resultant))
    return out


def _replicate_constraints(fit: MultiSineFit) -> FitConstraints | None:
    """The fit's constraints, with its solution as the replicates' starting phases."""
    detail = fit.detail
    if detail is None:
        return None
    from cuperiod.prewhiten.constraints import FitStart

    start = FitStart(
        amplitude=np.asarray(fit.amplitude, dtype=np.float64),
        phase=np.asarray(fit.phase, dtype=np.float64),
        subset_amplitude=detail.subset_amplitude,
        subset_phase=detail.subset_phase,
    )
    return replace(detail.constraints, start=start)


def _zero_fixed(fit: MultiSineFit, scatter: _BootstrapScatter) -> _BootstrapScatter:
    """Exact zeros for every fixed quantity (a fixed phase's phasors only agree to
    rounding, which the circular deviation would turn into ~1e-8 rad)."""
    detail = fit.detail
    if detail is None:
        return scatter
    k = fit.n_components
    constraints = detail.constraints
    resolved = constraints.resolve(fit.frequency)
    frequency = scatter.frequency.copy()
    amplitude = scatter.amplitude.copy()
    phase = scatter.phase.copy()
    for i in range(k):
        comp = constraints.component(i)
        if not np.any(resolved.matrix[i] != 0.0):
            frequency[i] = 0.0
        if comp.amplitude is not None:
            amplitude[i] = 0.0
        if comp.phase is not None:
            phase[i] = 0.0
    return replace(scatter, frequency=frequency, amplitude=amplitude, phase=phase)


def _bootstrap(
    time: FloatArray,
    error: FloatArray | None,
    fit: MultiSineFit,
    *,
    n_resamples: int,
    seed: int,
    refine: Refinement,
    sweeps: int,
    frequency_bounds: tuple[Any, Any] | None,
    max_nfev: int,
    should_stop: Callable[[], bool] | None = None,
) -> _BootstrapScatter:
    """:func:`bootstrap_uncertainties` plus the replicate scatter of the offset (and,
    for a constrained fit, of the per-subset values)."""
    k = fit.n_components
    if k == 0 or n_resamples < 2:
        empty = np.full(k, np.nan, dtype=np.float64)
        return _BootstrapScatter(empty, empty.copy(), empty.copy(), float("nan"))
    policy: Refinement = refine
    if refine in {"none", "last"}:
        policy = "simultaneous" if k <= _PROMOTED_MAX_SIMULTANEOUS else "cyclic"
        sweeps = max(sweeps, _MIN_CYCLIC_SWEEPS)
    t = np.asarray(time, dtype=np.float64)
    detail = fit.detail
    constraints = _replicate_constraints(fit)
    if constraints is None:
        model = fit.model(t)
    else:
        # The exact model: per-subset zero points, amplitudes and phases.
        model = fit.model(
            t, offset_groups=constraints.offset_groups,
            variation_groups=constraints.variation_groups,
        )
    residuals = np.asarray(fit.residuals, dtype=np.float64)
    n = int(residuals.size)
    dof = n - fit.n_parameters
    perturbation = residuals * (np.sqrt(n / dof) if dof > 0 else 1.0)
    rng = np.random.default_rng(seed)
    signs = np.array([-1.0, 1.0])
    freqs = np.empty((n_resamples, k), dtype=np.float64)
    amps = np.empty((n_resamples, k), dtype=np.float64)
    phasors = np.empty((n_resamples, k), dtype=np.complex128)
    offsets = np.empty(n_resamples, dtype=np.float64)
    sub_amp = sub_phase = sub_off = None
    if detail is not None and detail.subset_amplitude is not None:
        sub_amp = np.empty((n_resamples, *detail.subset_amplitude.shape))
        sub_phase = np.empty((n_resamples, *detail.subset_amplitude.shape),
                             dtype=np.complex128)
    if detail is not None and detail.subset_offset is not None:
        sub_off = np.empty((n_resamples, detail.subset_offset.size))
    done = 0
    for i in range(n_resamples):
        if should_stop is not None and should_stop():
            break
        sample = model + perturbation * rng.choice(signs, size=n)
        replicate = fit_multisine(
            t,
            sample,
            error,
            fit.frequency,
            fit_mean=fit.fit_mean,
            t_ref=fit.t_ref,
            refine=policy,
            sweeps=sweeps,
            frequency_bounds=frequency_bounds,
            max_nfev=max_nfev,
            covariance=False,
            constraints=constraints,
        )
        freqs[i] = replicate.frequency
        amps[i] = replicate.amplitude
        phasors[i] = np.exp(1j * replicate.phase)
        offsets[i] = replicate.offset
        rd = replicate.detail
        if sub_amp is not None and sub_phase is not None and rd is not None:
            assert rd.subset_amplitude is not None and rd.subset_phase is not None
            sub_amp[i] = rd.subset_amplitude
            sub_phase[i] = np.exp(1j * rd.subset_phase)
        if sub_off is not None and rd is not None and rd.subset_offset is not None:
            sub_off[i] = rd.subset_offset
        done = i + 1
    if done < 2:
        empty = np.full(k, np.nan, dtype=np.float64)
        return _BootstrapScatter(empty, empty.copy(), empty.copy(), float("nan"))
    if done < n_resamples:
        freqs, amps, phasors, offsets = (
            freqs[:done], amps[:done], phasors[:done], offsets[:done]
        )
    sigma_f = np.std(freqs, axis=0, ddof=1)
    sigma_a = np.std(amps, axis=0, ddof=1)
    # Circular standard deviation: sqrt(-2 ln R) with R the mean resultant length.
    resultant = np.clip(np.abs(np.mean(phasors, axis=0)), 1e-12, 1.0)
    sigma_phase = np.sqrt(-2.0 * np.log(resultant))
    sigma_offset = float(np.std(offsets, ddof=1)) if fit.fit_mean else float("nan")
    scatter = _BootstrapScatter(sigma_f, sigma_a, sigma_phase, sigma_offset)
    if detail is None:
        return scatter
    if sub_amp is not None and sub_phase is not None:
        varies_a = np.array([detail.constraints.component(i).varies_amplitude
                             for i in range(k)])
        varies_p = np.array([detail.constraints.component(i).varies_phase
                             for i in range(k)])
        sigma_sub_a = np.std(sub_amp[:done], axis=0, ddof=1)
        sigma_sub_p = _circular_sigma(sub_phase[:done])
        sigma_sub_a[~varies_a] = np.nan
        sigma_sub_p[~varies_p] = np.nan
        scatter = replace(scatter, subset_amplitude=sigma_sub_a,
                          subset_phase=sigma_sub_p)
    if sub_off is not None:
        scatter = replace(scatter,
                          subset_offset=np.std(sub_off[:done], axis=0, ddof=1))
    return _zero_fixed(fit, scatter)


def _residual_sigma(fit: MultiSineFit, error: FloatArray | None) -> float:
    """The per-point noise level the analytic formulae take.

    For a weighted fit, the weighted residual rms ``sqrt(sum(w r^2) / sum(w))``: the
    weighted estimator's variance is ``2/sum(w)``, which the plain rms — dominated by
    the noisy points — overstates by ``sqrt(mean(s^2) mean(1/s^2))``. With equal error
    bars the two agree. Relative point weights of a constrained fit enter ``w``
    (normalised to mean one, as in the fit).
    """
    if fit.residuals.size == 0:
        return float("nan")
    detail = fit.detail
    if detail is not None and detail.constraints.point_weights is not None:
        omega = np.asarray(detail.constraints.point_weights, dtype=np.float64)
        w = omega / float(np.mean(omega))
        if fit.weighted and error is not None:
            dy = np.asarray(error, dtype=np.float64)
            w = w / (dy * dy)
        total = float(np.sum(w))
        return float(np.sqrt(fit.rss / total)) if total > 0.0 else float("nan")
    if fit.weighted and error is not None:
        dy = np.asarray(error, dtype=np.float64)
        total = float(np.sum(1.0 / (dy * dy)))
        return float(np.sqrt(fit.rss / total)) if total > 0.0 else float("nan")
    return float(np.std(fit.residuals))


def _time_ordered(time: FloatArray, residuals: FloatArray) -> FloatArray:
    """``residuals`` in time order (a stable sort; no copy when already sorted)."""
    t = np.asarray(time, dtype=np.float64)
    if t.shape != residuals.shape or t.size < 2 or not np.any(np.diff(t) < 0.0):
        return residuals
    return residuals[np.argsort(t, kind="stable")]


def component_uncertainties(
    time: FloatArray,
    error: FloatArray | None,
    fit: MultiSineFit,
    *,
    method: UncertaintyMethod = "covariance",
    correlation_correction: bool = True,
    n_resamples: int = 200,
    seed: int = 0,
    refine: Refinement = "simultaneous",
    sweeps: int = 1,
    frequency_bounds: tuple[Any, Any] | None = None,
    max_nfev: int = 200,
    should_stop: Callable[[], bool] | None = None,
) -> Uncertainties:
    """Per-component 1-sigma errors by the requested estimator.

    Parameters
    ----------
    time : numpy.ndarray
        Observation times (days), aligned with ``fit.residuals``; any row order.
    error : numpy.ndarray, optional
        1-sigma uncertainties.
    fit : MultiSineFit
        The solution to characterise.
    method : {"covariance", "analytic", "bootstrap"}, default "covariance"
        Estimator (see the module docstring).
    correlation_correction : bool, default True
        Multiply the errors by ``sqrt(D)`` with ``D`` from :func:`correlation_factor`
        of the residuals in time order. Never applied to the bootstrap, which already
        uses the residuals as they are.
    n_resamples, seed, refine, sweeps, frequency_bounds, max_nfev, should_stop
        Bootstrap controls (see :func:`bootstrap_uncertainties`).

    Returns
    -------
    Uncertainties
        For a constrained fit, fixed quantities have error 0.0 and the per-subset
        fields are filled (see the module docstring).

    Raises
    ------
    ValueError
        For an unknown ``method`` (rather than silently returning covariance errors
        under another name).
    """
    if method not in get_args(UncertaintyMethod):
        raise ValueError(
            f"unknown uncertainty method {method!r}; "
            f"choose from {get_args(UncertaintyMethod)}"
        )
    subsets: tuple[FloatArray | None, FloatArray | None, FloatArray | None] = (
        None, None, None,
    )
    if method == "analytic":
        residual_sigma = _residual_sigma(fit, error)
        # |A|: under a fixed phase a constrained amplitude is signed, and the
        # formulae divide by the amplitude.
        sigma_f, sigma_a, sigma_phase = analytic_uncertainties(
            fit.n_samples,
            float(np.ptp(time)) if np.size(time) else 0.0,
            residual_sigma,
            np.abs(fit.amplitude),
        )
        sigma_offset = (
            residual_sigma / float(np.sqrt(fit.n_samples))
            if fit.fit_mean and fit.n_samples > 0
            else float("nan")
        )
        if fit.detail is not None:
            sigma_f, sigma_a, sigma_phase, subsets = _analytic_constrained(
                fit, residual_sigma, sigma_f, sigma_a, sigma_phase
            )
    elif method == "bootstrap":
        scatter = _bootstrap(
            time, error, fit,
            n_resamples=n_resamples, seed=seed, refine=refine, sweeps=sweeps,
            frequency_bounds=frequency_bounds, max_nfev=max_nfev,
            should_stop=should_stop,
        )
        sigma_f, sigma_a, sigma_phase = (
            scatter.frequency, scatter.amplitude, scatter.phase
        )
        sigma_offset = scatter.offset
        subsets = (scatter.subset_amplitude, scatter.subset_phase,
                   scatter.subset_offset)
        if fit.n_components == 0 or n_resamples < 2:
            sigma_offset = fit.offset_error
    else:
        sigma_f = fit.frequency_error
        sigma_a = fit.amplitude_error
        sigma_phase = fit.phase_error
        sigma_offset = fit.offset_error
        if fit.detail is not None:
            subsets = (fit.detail.subset_amplitude_error,
                       fit.detail.subset_phase_error, fit.detail.subset_offset_error)

    if method != "covariance" and fit.detail is not None:
        sigma_f, sigma_a, sigma_phase, subsets = _undetermined_as_nan(
            fit, (sigma_f, sigma_a, sigma_phase), subsets
        )

    factor = 1.0
    if correlation_correction and method != "bootstrap":
        ordered = _time_ordered(time, np.asarray(fit.residuals, dtype=np.float64))
        factor = float(np.sqrt(correlation_factor(ordered)))
    sub_a, sub_p, sub_o = (
        None if values is None else np.asarray(values, dtype=np.float64) * factor
        for values in subsets
    )
    return Uncertainties(
        frequency=np.asarray(sigma_f, dtype=np.float64) * factor,
        amplitude=np.asarray(sigma_a, dtype=np.float64) * factor,
        phase=np.asarray(sigma_phase, dtype=np.float64) * factor,
        method=method,
        correlation_factor=factor**2,
        offset=float(sigma_offset) * factor,
        subset_amplitude=sub_a,
        subset_phase=sub_p,
        subset_offset=sub_o,
    )


def _undetermined_as_nan(
    fit: MultiSineFit,
    headline: tuple[FloatArray, FloatArray, FloatArray],
    subsets: tuple[FloatArray | None, FloatArray | None, FloatArray | None],
) -> tuple[
    FloatArray, FloatArray, FloatArray,
    tuple[FloatArray | None, FloatArray | None, FloatArray | None],
]:
    """NaN wherever the constrained fit found a parameter unidentifiable.

    The covariance reports a parameter the data do not determine -- the phase of an
    amplitude at round-off, two collinear components -- as NaN with the component
    flagged ``degenerate`` (spec 1.8). The analytic formulae divide by the
    amplitude and the bootstrap measures the scatter of replicate fits; neither
    notices, and both reported a finite number for such a phase. Every estimator now
    reports it the same way. Only flagged components are touched, so a fit whose
    covariance is NaN for another reason (no degrees of freedom) keeps its analytic
    and bootstrap errors.
    """
    detail = fit.detail
    assert detail is not None
    flagged = np.array(["degenerate" in flags for flags in detail.flags], dtype=bool)
    if not flagged.any():
        return (*headline, subsets)
    reference = (fit.frequency_error, fit.amplitude_error, fit.phase_error)
    out: list[FloatArray] = []
    for sigma, covariance in zip(headline, reference, strict=True):
        values = np.array(sigma, dtype=np.float64, copy=True)
        values[flagged & np.isnan(covariance)] = np.nan
        out.append(values)
    per_subset: list[FloatArray | None] = []
    for subset_sigma, subset_covariance in zip(
        subsets,
        (detail.subset_amplitude_error, detail.subset_phase_error, None),
        strict=True,
    ):
        if subset_sigma is None or subset_covariance is None:
            per_subset.append(subset_sigma)
            continue
        table = np.array(subset_sigma, dtype=np.float64, copy=True)
        table[flagged[:, None] & np.isnan(subset_covariance)] = np.nan
        per_subset.append(table)
    return out[0], out[1], out[2], (per_subset[0], per_subset[1], per_subset[2])


def _analytic_constrained(
    fit: MultiSineFit,
    residual_sigma: float,
    sigma_f: FloatArray,
    sigma_a: FloatArray,
    sigma_phase: FloatArray,
) -> tuple[
    FloatArray, FloatArray, FloatArray,
    tuple[FloatArray | None, FloatArray | None, FloatArray | None],
]:
    """Montgomery & O'Donoghue under constraints.

    Fixed quantities get 0; a tied frequency ``sqrt(sum_j n_j^2 sigma_j^2)`` over its
    free roots; a per-subset amplitude or phase the formula with ``N -> n_s`` (and its
    own ``|A_s|``); a per-subset zero point ``sigma / sqrt(n_s)``.
    """
    detail = fit.detail
    assert detail is not None
    constraints = detail.constraints
    k = fit.n_components
    resolved = constraints.resolve(fit.frequency)
    root_sigma = np.asarray(sigma_f, dtype=np.float64)[resolved.free_index]
    out_f = np.array(sigma_f, dtype=np.float64, copy=True)
    out_a = np.array(sigma_a, dtype=np.float64, copy=True)
    out_p = np.array(sigma_phase, dtype=np.float64, copy=True)
    for i in range(k):
        comp = constraints.component(i)
        status = resolved.status[i]
        if status == "fixed":
            out_f[i] = 0.0
        elif status == "tied":
            row = resolved.matrix[i]
            out_f[i] = float(np.sqrt(np.sum(row * row * root_sigma * root_sigma)))
        if comp.amplitude is not None:
            out_a[i] = 0.0
        if comp.phase is not None:
            out_p[i] = 0.0
    sub_a = sub_p = sub_o = None
    if constraints.variation_groups is not None and detail.subset_amplitude is not None:
        counts = np.bincount(np.asarray(constraints.variation_groups).astype(np.int64))
        base = np.sqrt(2.0 / counts) * residual_sigma
        sub_a = np.full(detail.subset_amplitude.shape, np.nan)
        sub_p = np.full(detail.subset_amplitude.shape, np.nan)
        for i in range(k):
            comp = constraints.component(i)
            if comp.varies_amplitude:
                sub_a[i] = base
            if comp.varies_phase:
                amp = np.abs(detail.subset_amplitude[i])
                safe = np.where(amp > 0.0, amp, 1.0)
                sub_p[i] = np.where(amp > 0.0, base / safe, np.nan)
    if constraints.offset_groups is not None:
        counts = np.bincount(np.asarray(constraints.offset_groups).astype(np.int64))
        sub_o = residual_sigma / np.sqrt(counts)
    return out_f, out_a, out_p, (sub_a, sub_p, sub_o)


__all__ = [
    "UncertaintyMethod",
    "Uncertainties",
    "analytic_uncertainties",
    "bootstrap_uncertainties",
    "component_uncertainties",
    "correlation_factor",
]
