"""Uncertainty estimators: time order, weights, the bootstrap scheme, and the offset."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.prewhiten import prewhiten
from cuperiod.prewhiten.fit import fit_multisine
from cuperiod.prewhiten.uncertainty import (
    Uncertainties,
    bootstrap_uncertainties,
    component_uncertainties,
    correlation_factor,
)
from synth import synthetic_pulsator

_TWO_PI = 2.0 * np.pi


def _settings(**overrides) -> cup.PreWhitenSettings:
    defaults = {"backend": "finufft", "max_frequencies": 6}
    return cup.PreWhitenSettings(**{**defaults, **overrides})


def _red_pulsator(n: int = 800, span: float = 20.0, phi: float = 0.9, seed: int = 5):
    """Two planted modes on AR(1) red noise, so the residuals are correlated."""
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0.0, span, n)) + 2458000.0
    noise = np.empty(n)
    noise[0] = rng.normal(0.0, 0.002)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(0.0, 0.002 * np.sqrt(1 - phi**2))
    value = (
        10.0
        + 0.012 * np.sin(_TWO_PI * 6.3 * (time - time[0]) + 0.5)
        + 0.007 * np.sin(_TWO_PI * 9.81 * (time - time[0]) + 2.0)
        + noise
    )
    return time, value, np.full(n, 0.002)


def _heteroscedastic(low: float = 0.01, high: float = 0.1, n: int = 400, seed: int = 0):
    """One sinusoid; half the points have error bars ``high``, half ``low``."""
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0.0, 30.0, n))
    error = np.where(rng.uniform(size=n) < 0.5, low, high)
    value = 5.0 + 0.05 * np.sin(_TWO_PI * 3.3 * time + 0.4) + rng.normal(0.0, error)
    return time, value, error


# --- PW-B-2 / PW-A-6: the correlation factor reads the residuals in time order --------


@pytest.mark.parametrize("method", ["covariance", "analytic"])
def test_row_order_does_not_change_the_reported_uncertainties(method: str) -> None:
    time, value, error = _red_pulsator()
    order = np.random.default_rng(1).permutation(time.size)
    settings = _settings(uncertainty=method)
    sorted_run = prewhiten((time, value, error), settings=settings)
    shuffled = prewhiten((time[order], value[order], error[order]), settings=settings)
    assert sorted_run.correlation_factor > 1.2  # the effect the test is about
    assert shuffled.n_components == sorted_run.n_components
    assert shuffled.correlation_factor == pytest.approx(
        sorted_run.correlation_factor, rel=1e-9
    )
    assert np.allclose(
        shuffled.frequency_error, sorted_run.frequency_error, rtol=1e-6
    )
    assert np.allclose(
        shuffled.amplitude_error, sorted_run.amplitude_error, rtol=1e-6
    )
    # The result keeps the caller's row order.
    assert np.array_equal(shuffled.time, time[order])


def test_correlation_factor_is_taken_in_time_order_by_the_estimator() -> None:
    time, value, error = _red_pulsator(n=600)
    fit = fit_multisine(time, value, error, [6.3, 9.81], refine="simultaneous")
    reference = component_uncertainties(time, error, fit)
    order = np.random.default_rng(2).permutation(time.size)
    shuffled_fit = fit_multisine(
        time[order], value[order], error[order], fit.frequency, refine="none",
        t_ref=fit.t_ref,
    )
    shuffled = component_uncertainties(time[order], error[order], shuffled_fit)
    assert shuffled.correlation_factor == pytest.approx(
        reference.correlation_factor, rel=1e-12
    )
    assert reference.correlation_factor == pytest.approx(
        correlation_factor(fit.residuals)
    )


# --- PW-B-3: the bootstrap on heteroscedastic data ------------------------------------


def test_bootstrap_is_right_for_heteroscedastic_error_bars() -> None:
    # Pooled raw residuals put the noisy points' residuals on precise, heavily
    # weighted points: sigma_f came out 6-9x too large for bars of 0.01 and 0.1.
    time, value, error = _heteroscedastic()
    fit = fit_multisine(time, value, error, [3.3], refine="simultaneous")
    covariance = component_uncertainties(
        time, error, fit, correlation_correction=False
    )
    boot = component_uncertainties(
        time, error, fit, method="bootstrap", n_resamples=150, seed=4,
        correlation_correction=False,
    )
    assert boot.frequency[0] == pytest.approx(covariance.frequency[0], rel=0.35)
    assert boot.amplitude[0] == pytest.approx(covariance.amplitude[0], rel=0.35)


def test_bootstrap_inflates_for_the_fitted_degrees_of_freedom() -> None:
    # Small N, many parameters: raw residuals are shrunk by sqrt((N - M) / N).
    rng = np.random.default_rng(6)
    n = 60
    time = np.sort(rng.uniform(0.0, 20.0, n))
    freqs = np.array([1.1, 2.3, 3.7, 4.9, 6.2])
    value = rng.normal(0.0, 0.01, n)
    for j, f in enumerate(freqs):
        value = value + 0.05 * np.sin(_TWO_PI * f * time + j)
    ratios = []
    for seed in range(4):
        y = value + np.random.default_rng(100 + seed).normal(0.0, 0.01, n)
        fit = fit_multisine(time, y, None, freqs, refine="simultaneous")
        boot = bootstrap_uncertainties(
            time, None, fit, n_resamples=120, seed=seed, refine="simultaneous"
        )
        ratios.append(np.mean(boot[1] / fit.amplitude_error))
    assert np.mean(ratios) > 0.9


# --- PW-B-4: the bootstrap's replicate fits mirror the final polish -------------------


def test_bootstrap_of_a_close_pair_is_not_underestimated() -> None:
    rng = np.random.default_rng(3)
    n, span = 500, 30.0
    time = np.sort(rng.uniform(0.0, span, n))
    pair = np.array([5.0, 5.0 + 1.5 / span])
    value = (
        0.05 * np.sin(_TWO_PI * pair[0] * time + 0.3)
        + 0.03 * np.sin(_TWO_PI * pair[1] * time + 1.9)
        + rng.normal(0.0, 0.02, n)
    )
    fit = fit_multisine(time, value, None, pair, refine="simultaneous")
    covariance = component_uncertainties(time, None, fit, correlation_correction=False)
    # "last" is the engine's per-iteration policy; the bootstrap promotes it.
    boot = component_uncertainties(
        time, None, fit, method="bootstrap", n_resamples=120, seed=11, refine="last"
    )
    assert np.all(boot.frequency / covariance.frequency > 0.8)


# --- PW-B-5: analytic errors honour the weights ---------------------------------------


def test_analytic_errors_use_the_weighted_noise_of_a_weighted_fit() -> None:
    time, value, error = _heteroscedastic(low=0.01, high=0.03)
    fit = fit_multisine(time, value, error, [3.3], refine="simultaneous")
    covariance = component_uncertainties(
        time, error, fit, correlation_correction=False
    )
    analytic = component_uncertainties(
        time, error, fit, method="analytic", correlation_correction=False
    )
    assert analytic.frequency[0] == pytest.approx(covariance.frequency[0], rel=0.15)
    assert analytic.amplitude[0] == pytest.approx(covariance.amplitude[0], rel=0.15)


def test_analytic_errors_are_unchanged_for_equal_error_bars() -> None:
    from cuperiod.prewhiten.uncertainty import analytic_uncertainties

    time, value, error = synthetic_pulsator(n=600)
    fit = fit_multisine(time, value, error, [12.34, 17.81, 24.68], refine="none")
    analytic = component_uncertainties(
        time, error, fit, method="analytic", correlation_correction=False
    )
    expected = analytic_uncertainties(
        fit.n_samples, float(np.ptp(time)), float(np.std(fit.residuals)), fit.amplitude
    )
    assert np.allclose(analytic.frequency, expected[0], rtol=1e-10)
    assert np.allclose(analytic.amplitude, expected[1], rtol=1e-10)


# --- PW-B-9: the offset error goes through the same estimator -------------------------


def test_offset_error_carries_the_correlation_factor() -> None:
    time, value, error = _red_pulsator()
    on = prewhiten((time, value, error), settings=_settings())
    off = prewhiten(
        (time, value, error), settings=_settings(correlation_correction=False)
    )
    assert on.correlation_factor > 1.2
    assert on.offset_error / off.offset_error == pytest.approx(
        np.sqrt(on.correlation_factor), rel=1e-6
    )


def test_offset_error_follows_the_selected_estimator() -> None:
    time, value, error = _heteroscedastic()
    fit = fit_multisine(time, value, error, [3.3], refine="simultaneous")
    covariance = component_uncertainties(time, error, fit, correlation_correction=False)
    boot = component_uncertainties(
        time, error, fit, method="bootstrap", n_resamples=60, seed=1,
        correlation_correction=False,
    )
    analytic = component_uncertainties(
        time, error, fit, method="analytic", correlation_correction=False
    )
    assert covariance.offset == pytest.approx(fit.offset_error)
    assert boot.offset > 0.0 and boot.offset != covariance.offset
    assert analytic.offset > 0.0 and analytic.offset != covariance.offset
    centred = fit_multisine(time, value - 5.0, error, [3.3], fit_mean=False)
    assert np.isnan(component_uncertainties(time, error, centred).offset)


# --- PW-B-11 / PW-B-10 ----------------------------------------------------------------


@pytest.mark.parametrize("method", ["montecarlo", "Bootstrap", "bootstrap ", ""])
def test_an_unknown_method_raises(method: str) -> None:
    time, value, error = synthetic_pulsator(n=300)
    fit = fit_multisine(time, value, error, [12.34], refine="none")
    with pytest.raises(ValueError, match="uncertainty method"):
        component_uncertainties(time, error, fit, method=method)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["covariance", "analytic", "bootstrap"])
def test_valid_methods_are_echoed(method: str) -> None:
    time, value, error = synthetic_pulsator(n=300)
    fit = fit_multisine(time, value, error, [12.34], refine="none")
    result = component_uncertainties(time, error, fit, method=method, n_resamples=4)
    assert result.method == method


def test_uncertainties_compare_by_identity_without_raising() -> None:
    time, value, error = synthetic_pulsator(n=300)
    fit = fit_multisine(time, value, error, [12.34], refine="none")
    first = component_uncertainties(time, error, fit)
    second = component_uncertainties(time, error, fit)
    assert isinstance(first, Uncertainties)
    assert first == first and (first == second) is False
    assert len({first, second}) == 2
