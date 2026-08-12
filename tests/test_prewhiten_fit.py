"""Multi-sinusoid fitting: recovery, refinement policies, and calibrated covariance."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.prewhiten.fit import fit_multisine
from cuperiod.prewhiten.uncertainty import (
    analytic_uncertainties,
    bootstrap_uncertainties,
    component_uncertainties,
    correlation_factor,
)
from synth import synthetic_pulsator

_TRUTH = ((12.34, 0.012, 0.5), (17.81, 0.007, 2.0))


def _two_mode(n: int = 1200, span: float = 30.0, noise: float = 0.002, seed: int = 0):
    return synthetic_pulsator(
        n=n,
        span=span,
        frequencies=tuple(f for f, _, _ in _TRUTH),
        amplitudes=tuple(a for _, a, _ in _TRUTH),
        phases=tuple(p for _, _, p in _TRUTH),
        noise=noise,
        seed=seed,
    )


def test_recovers_planted_frequencies_and_amplitudes() -> None:
    time, value, error = _two_mode()
    # Started within a fraction of a Rayleigh width, as the extraction loop always does.
    fit = fit_multisine(
        time, value, error, np.array([12.335, 17.818]), refine="simultaneous"
    )
    assert fit.frequency[0] == pytest.approx(12.34, abs=5e-4)
    assert fit.frequency[1] == pytest.approx(17.81, abs=5e-4)
    assert fit.amplitude[0] == pytest.approx(0.012, rel=0.05)
    assert fit.amplitude[1] == pytest.approx(0.007, rel=0.05)
    assert fit.offset == pytest.approx(10.0, abs=1e-3)
    assert fit.n_parameters == 3 * 2 + 1


def test_model_reproduces_the_data_to_the_noise_level() -> None:
    time, value, error = _two_mode(noise=0.001)
    fit = fit_multisine(
        time, value, error, np.array([12.34, 17.81]), refine="simultaneous"
    )
    assert np.allclose(fit.residuals, value - fit.model(time), atol=1e-12)
    assert fit.rms == pytest.approx(0.001, rel=0.15)
    assert fit.reduced_chi2 == pytest.approx(1.0, rel=0.15)


def test_refinement_policies_reach_the_same_solution() -> None:
    time, value, error = _two_mode()
    start = np.array([12.33, 17.82])
    solutions = {
        policy: fit_multisine(time, value, error, start, refine=policy, sweeps=3)
        for policy in ("cyclic", "simultaneous")
    }
    assert np.allclose(
        solutions["cyclic"].frequency, solutions["simultaneous"].frequency, atol=1e-6
    )
    # ...and both improve on doing nothing at all.
    none_fit = fit_multisine(time, value, error, start, refine="none")
    assert solutions["simultaneous"].rss < none_fit.rss


def test_last_policy_only_moves_the_newest_frequency() -> None:
    time, value, error = _two_mode()
    start = np.array([12.30, 17.82])
    fit = fit_multisine(time, value, error, start, refine="last")
    assert fit.frequency[0] == start[0]  # untouched
    assert fit.frequency[1] != start[1]  # refined


def test_refinement_respects_per_frequency_bounds() -> None:
    time, value, error = _two_mode()
    lower = np.array([12.30, 17.75])
    upper = np.array([12.32, 17.78])
    fit = fit_multisine(
        time, value, error, np.array([12.31, 17.76]),
        refine="simultaneous", frequency_bounds=(lower, upper),
    )
    assert np.all(fit.frequency >= lower - 1e-12)
    assert np.all(fit.frequency <= upper + 1e-12)


def test_offset_only_fit_is_the_weighted_mean() -> None:
    time, value, error = _two_mode()
    fit = fit_multisine(time, value, error, np.zeros(0))
    weight = 1.0 / error**2
    assert fit.n_components == 0
    assert fit.offset == pytest.approx(float(np.dot(weight, value) / weight.sum()))


def test_adding_a_real_component_lowers_the_bic() -> None:
    time, value, error = _two_mode()
    base = fit_multisine(time, value, error, np.zeros(0))
    one = fit_multisine(time, value, error, np.array([12.34]))
    two = fit_multisine(time, value, error, np.array([12.34, 17.81]))
    assert one.bic < base.bic - 10.0
    assert two.bic < one.bic - 10.0
    assert two.aic < one.aic


def test_unweighted_fit_uses_the_profiled_likelihood_bic() -> None:
    time, value, _ = _two_mode()
    fit = fit_multisine(time, value, None, np.array([12.34]))
    assert not fit.weighted
    expected = fit.n_samples * np.log(fit.rss / fit.n_samples) + fit.n_parameters * (
        np.log(fit.n_samples)
    )
    assert fit.bic == pytest.approx(expected)


def test_covariance_errors_are_calibrated_against_monte_carlo() -> None:
    # 60 noise realisations of the same planted signal: the reported 1-sigma errors
    # should match the realised scatter to within tens of percent.
    time, clean, error = _two_mode(noise=0.002)
    truth = np.array([12.34, 17.81])
    dt = time - time.mean()
    signal = np.full(time.size, 10.0)
    for freq, amp, phase in _TRUTH:
        signal += amp * np.sin(2 * np.pi * freq * (time - time.min()) + phase)
    frequencies, reported = [], []
    rng = np.random.default_rng(12)
    for _ in range(60):
        noisy = signal + rng.normal(0.0, 0.002, time.size)
        fit = fit_multisine(time, noisy, error, truth, refine="simultaneous")
        frequencies.append(fit.frequency)
        reported.append(fit.frequency_error)
    empirical = np.std(np.asarray(frequencies), axis=0, ddof=1)
    predicted = np.mean(np.asarray(reported), axis=0)
    assert np.all(predicted / empirical > 0.6)
    assert np.all(predicted / empirical < 1.6)
    assert dt.size == time.size  # (sanity: the epoch shift did not resize anything)


def test_correlation_factor_is_one_for_white_noise() -> None:
    rng = np.random.default_rng(3)
    assert correlation_factor(rng.normal(size=5000)) == pytest.approx(1.0, abs=0.05)


def test_correlation_factor_rises_for_correlated_residuals() -> None:
    # A slow sinusoid has long same-sign runs — what the factor exists to catch.
    correlated = np.sin(np.linspace(0.0, 6 * np.pi, 3000))
    assert correlation_factor(correlated) > 100.0
    assert correlation_factor(np.array([1.0, -1.0])) == 1.0  # too few points


def test_analytic_errors_match_the_covariance_for_isolated_modes() -> None:
    time, value, error = _two_mode(noise=0.002)
    fit = fit_multisine(
        time, value, error, np.array([12.34, 17.81]), refine="simultaneous"
    )
    sigma_f, sigma_a, sigma_phase = analytic_uncertainties(
        fit.n_samples, float(np.ptp(time)), float(np.std(fit.residuals)), fit.amplitude
    )
    assert np.allclose(sigma_f, fit.frequency_error, rtol=0.25)
    assert np.allclose(sigma_a, fit.amplitude_error, rtol=0.25)
    assert np.allclose(sigma_phase, fit.phase_error, rtol=0.25)


def test_bootstrap_errors_agree_with_the_covariance() -> None:
    time, value, error = _two_mode(n=600, noise=0.002)
    fit = fit_multisine(
        time, value, error, np.array([12.34, 17.81]), refine="simultaneous"
    )
    sigma_f, sigma_a, sigma_phase = bootstrap_uncertainties(
        time, error, fit, n_resamples=40, seed=1
    )
    assert np.allclose(sigma_f, fit.frequency_error, rtol=0.5)
    assert np.allclose(sigma_a, fit.amplitude_error, rtol=0.5)
    assert np.all(sigma_phase > 0.0)


def test_bootstrap_promotes_frequency_pinning_policies_to_a_full_sweep() -> None:
    # Regression: "none" pins every frequency and "last" pins all but the newest, so a
    # replicate fit under either policy returns the start frequencies verbatim and the
    # bootstrap scatter collapses to exactly zero for the pinned components.
    time, value, error = _two_mode(n=600, noise=0.002)
    fit = fit_multisine(
        time, value, error, np.array([12.34, 17.81]), refine="simultaneous"
    )
    for policy in ("none", "last"):
        sigma_f, _, _ = bootstrap_uncertainties(
            time, error, fit, n_resamples=24, seed=3, refine=policy
        )
        assert np.all(sigma_f > 0.1 * fit.frequency_error), policy


def test_component_uncertainties_apply_the_correlation_correction() -> None:
    time, value, error = _two_mode()
    fit = fit_multisine(time, value, error, np.array([12.34, 17.81]))
    plain = component_uncertainties(
        time, error, fit, method="covariance", correlation_correction=False
    )
    corrected = component_uncertainties(
        time, error, fit, method="covariance", correlation_correction=True
    )
    assert corrected.correlation_factor >= 1.0
    assert np.all(corrected.frequency >= plain.frequency - 1e-18)


def test_fit_rejects_mismatched_inputs() -> None:
    time, value, error = _two_mode(n=100)
    with pytest.raises(ValueError, match="same length"):
        fit_multisine(time, value[:-1], error[:-1], np.array([12.0]))
    with pytest.raises(ValueError, match="at least one frequency"):
        fit_multisine(time, value, error, np.zeros(0), fit_mean=False)
