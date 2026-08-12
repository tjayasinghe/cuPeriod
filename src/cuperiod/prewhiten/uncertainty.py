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
    pulsation papers quote. They assume an isolated sinusoid in white noise, so they are
    a lower bound; useful for comparison with the literature.
``"bootstrap"``
    Resample the residuals, re-fit, and take the scatter. Makes no linearity assumption
    and needs no error bars, at the price of ``n_resamples`` extra fits.

All three can be inflated by the Schwarzenberg-Czerny (1991) correlation factor: real
photometry has residuals that are correlated point-to-point (instrumental drifts,
unresolved modes), so the *effective* number of independent samples is smaller than
``N`` and the formal errors are optimistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.prewhiten.fit import MultiSineFit, Refinement, fit_multisine

#: How component uncertainties are estimated.
UncertaintyMethod = Literal["covariance", "analytic", "bootstrap"]


@dataclass(frozen=True)
class Uncertainties:
    """1-sigma uncertainties for every component of a multi-sine solution.

    Attributes
    ----------
    frequency, amplitude, phase : numpy.ndarray
        Per-component 1-sigma errors (cycles/day, input units, radians).
    method : str
        Which estimator produced them.
    correlation_factor : float
        The inflation factor that was applied (1.0 when the correction is off).
    """

    frequency: FloatArray
    amplitude: FloatArray
    phase: FloatArray
    method: str
    correlation_factor: float


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
        Fit residuals in observation order (they must not be re-sorted).

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
    refine: Refinement = "cyclic",
    sweeps: int = 1,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Residual-resampling bootstrap errors for a fitted solution.

    Each replicate adds a resampling (with replacement) of the fit residuals back onto
    the best-fit model and re-fits from the current solution. The scatter of the
    replicate parameters is the uncertainty. This makes no linearity assumption, and
    needs no error bars, but it does assume the residuals are exchangeable — if they are
    strongly correlated, prefer the correlation-factor-inflated covariance errors.

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
        Refinement policy for the replicate fits.

    Returns
    -------
    tuple of numpy.ndarray
        ``(sigma_f, sigma_A, sigma_phase)``.
    """
    k = fit.n_components
    if k == 0 or n_resamples < 2:
        empty = np.full(k, np.nan, dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    t = np.asarray(time, dtype=np.float64)
    model = fit.model(t)
    residuals = np.asarray(fit.residuals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    freqs = np.empty((n_resamples, k), dtype=np.float64)
    amps = np.empty((n_resamples, k), dtype=np.float64)
    phasors = np.empty((n_resamples, k), dtype=np.complex128)
    for i in range(n_resamples):
        sample = model + rng.choice(residuals, size=residuals.size, replace=True)
        replicate = fit_multisine(
            t,
            sample,
            error,
            fit.frequency,
            fit_mean=fit.fit_mean,
            t_ref=fit.t_ref,
            refine=refine,
            sweeps=sweeps,
            covariance=False,
        )
        freqs[i] = replicate.frequency
        amps[i] = replicate.amplitude
        phasors[i] = np.exp(1j * replicate.phase)
    sigma_f = np.std(freqs, axis=0, ddof=1)
    sigma_a = np.std(amps, axis=0, ddof=1)
    # Circular standard deviation: sqrt(-2 ln R) with R the mean resultant length.
    resultant = np.clip(np.abs(np.mean(phasors, axis=0)), 1e-12, 1.0)
    sigma_phase = np.sqrt(-2.0 * np.log(resultant))
    return sigma_f, sigma_a, sigma_phase


def component_uncertainties(
    time: FloatArray,
    error: FloatArray | None,
    fit: MultiSineFit,
    *,
    method: UncertaintyMethod = "covariance",
    correlation_correction: bool = True,
    n_resamples: int = 200,
    seed: int = 0,
    refine: Refinement = "cyclic",
    sweeps: int = 1,
) -> Uncertainties:
    """Per-component 1-sigma errors by the requested estimator.

    Parameters
    ----------
    time : numpy.ndarray
        Observation times (days).
    error : numpy.ndarray, optional
        1-sigma uncertainties.
    fit : MultiSineFit
        The solution to characterise.
    method : {"covariance", "analytic", "bootstrap"}, default "covariance"
        Estimator (see the module docstring).
    correlation_correction : bool, default True
        Multiply the errors by ``sqrt(D)`` with ``D`` from :func:`correlation_factor`.
        Never applied to the bootstrap, which already samples the residuals as they are.
    n_resamples, seed, refine, sweeps
        Bootstrap controls.

    Returns
    -------
    Uncertainties
    """
    if method == "analytic":
        sigma_f, sigma_a, sigma_phase = analytic_uncertainties(
            fit.n_samples,
            float(np.ptp(time)) if np.size(time) else 0.0,
            float(np.std(fit.residuals)) if fit.residuals.size else float("nan"),
            fit.amplitude,
        )
    elif method == "bootstrap":
        sigma_f, sigma_a, sigma_phase = bootstrap_uncertainties(
            time, error, fit,
            n_resamples=n_resamples, seed=seed, refine=refine, sweeps=sweeps,
        )
    else:
        sigma_f = fit.frequency_error
        sigma_a = fit.amplitude_error
        sigma_phase = fit.phase_error

    factor = 1.0
    if correlation_correction and method != "bootstrap":
        factor = float(np.sqrt(correlation_factor(fit.residuals)))
    return Uncertainties(
        frequency=np.asarray(sigma_f, dtype=np.float64) * factor,
        amplitude=np.asarray(sigma_a, dtype=np.float64) * factor,
        phase=np.asarray(sigma_phase, dtype=np.float64) * factor,
        method=method,
        correlation_factor=factor**2,
    )


__all__ = [
    "UncertaintyMethod",
    "Uncertainties",
    "analytic_uncertainties",
    "bootstrap_uncertainties",
    "component_uncertainties",
    "correlation_factor",
]
