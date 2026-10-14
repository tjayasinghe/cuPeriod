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


# --- regressions ---------------------------------------------------------------

_TWO_PI = 2.0 * np.pi


def _svd_reference_sigma(time, error, fit) -> np.ndarray:
    """Independent covariance: explicit Jacobian, column-scaled, SVD inverse."""
    k = fit.n_components
    dt = time - fit.t_ref
    sw = np.ones(time.size) if error is None else 1.0 / error
    columns = []
    for block in range(3):
        for j in range(k):
            theta = _TWO_PI * fit.frequency[j] * dt + fit.phase[j]
            if block == 0:
                columns.append(fit.amplitude[j] * np.cos(theta) * _TWO_PI * dt)
            elif block == 1:
                columns.append(np.sin(theta))
            else:
                columns.append(fit.amplitude[j] * np.cos(theta))
    columns.append(np.ones(time.size))
    jac = np.stack(columns, axis=1) * sw[:, None]
    norms = np.linalg.norm(jac, axis=0)
    _, s, vt = np.linalg.svd(jac / norms, full_matrices=False)
    cov = ((vt.T / s**2) @ vt) / np.outer(norms, norms)
    return np.sqrt(np.diag(cov) * fit.rss / (fit.n_samples - fit.n_parameters))


def _ogle_like(seed: int = 1):
    """A 20-year, 3000-point curve with a 0.3 mag and a 3 mmag mode."""
    rng = np.random.default_rng(seed)
    n, span, sigma = 3000, 7300.0, 0.005
    time = np.sort(rng.uniform(0.0, span, n)) + 2452000.0
    frequency = np.array([0.531234, 0.817345])
    amplitude = np.array([0.3, 0.003])
    phase = np.array([0.7, 2.1])
    value = 15.0 + rng.normal(0.0, sigma, n)
    for f, a, p in zip(frequency, amplitude, phase, strict=True):
        value += a * np.sin(_TWO_PI * f * (time - time.mean()) + p)
    return time, value, np.full(n, sigma), frequency, amplitude, sigma


def test_covariance_survives_widely_different_parameter_scales() -> None:
    # PW-A-2: the pseudo-inverse truncated on the *raw* eigenvalues of [f, A, phi, c0]
    # whose columns differ in scale by (2 pi A T)^2, so the weak mode's phase error
    # collapsed to ~1e-8 rad on a long baseline (truth 0.043 rad).
    time, value, error, frequency, amplitude, sigma = _ogle_like()
    fit = fit_multisine(time, value, error, frequency, refine="simultaneous")
    reference = _svd_reference_sigma(time, error, fit)
    k = fit.n_components
    assert np.allclose(fit.frequency_error, reference[:k], rtol=1e-6)
    assert np.allclose(fit.amplitude_error, reference[k : 2 * k], rtol=1e-6)
    assert np.allclose(fit.phase_error, reference[2 * k : 3 * k], rtol=1e-6)
    assert fit.offset_error == pytest.approx(reference[-1], rel=1e-6)
    expected_phase = np.sqrt(2.0 / time.size) * sigma / amplitude
    assert fit.phase_error[1] == pytest.approx(expected_phase[1], rel=0.05)


def test_covariance_errors_scale_with_the_data_units() -> None:
    # PW-A-2: multiplying a 27-d curve by 1e7 (flux units) gave amplitude errors of
    # 1e-9 instead of 1e7 times the magnitude-scale value.
    time, value, error = _two_mode(n=800, span=27.0)
    start = np.array([12.34, 17.81])
    base = fit_multisine(time, value, error, start, refine="simultaneous")
    scaled = fit_multisine(
        time, value * 1e7, error * 1e7, start, refine="simultaneous"
    )
    assert np.allclose(scaled.amplitude_error, 1e7 * base.amplitude_error, rtol=1e-6)
    assert scaled.offset_error == pytest.approx(1e7 * base.offset_error, rel=1e-6)
    assert np.allclose(scaled.frequency_error, base.frequency_error, rtol=1e-6)
    assert np.allclose(scaled.phase_error, base.phase_error, rtol=1e-6)


def test_frequency_and_amplitude_errors_do_not_depend_on_the_reference_epoch() -> None:
    # PW-A-2: an explicit t_ref far from the data (t_ref=0 on Julian dates) made the
    # normal matrix singular to working precision and sigma_f collapsed by 1e3-1e7.
    rng = np.random.default_rng(3)
    time = np.sort(rng.uniform(0.0, 5.0, 500)) + 2458000.0
    value = (
        10.0
        + 0.01 * np.sin(_TWO_PI * 12.34 * (time - time.mean()) + 0.4)
        + 0.004 * np.sin(_TWO_PI * 12.71 * (time - time.mean()) + 1.4)
        + rng.normal(0.0, 0.002, time.size)
    )
    error = np.full(time.size, 0.002)
    near = fit_multisine(time, value, error, [12.34, 12.71], refine="simultaneous")
    for t_ref in (2458001.0, 2450000.0, 0.0):
        far = fit_multisine(
            time, value, error, near.frequency, refine="none", t_ref=t_ref
        )
        assert np.allclose(far.frequency_error, near.frequency_error, rtol=1e-6)
        assert np.allclose(far.amplitude_error, near.amplitude_error, rtol=1e-6)
        # The phase at a distant epoch inherits the frequency error, 2*pi*D*sigma_f.
        lever = _TWO_PI * (near.t_ref - t_ref) * near.frequency_error
        assert np.all(far.phase_error >= 0.99 * lever)


def test_an_undetermined_parameter_reports_nan_not_zero() -> None:
    # A component of exactly zero amplitude constrains neither its frequency nor its
    # phase: their variance is undefined (it used to be reported as exactly 0), and
    # that must not leak into the well-determined parameters, even at a far epoch.
    from cuperiod.prewhiten.fit import _parameter_covariance

    time, _, error = _two_mode(n=300)
    for shift in (0.0, 1234.5):
        dt = time - time.mean() + shift
        w = 1.0 / error**2
        cov = _parameter_covariance(
            dt, w, np.sqrt(w), np.array([12.34, 17.81]), np.array([0.0, 0.01]),
            np.array([0.3, 1.1]), True, rss=300.0, n_samples=300, n_parameters=7,
        )
        sigma = np.sqrt(np.diag(cov))
        # Order: f0, f1, A0, A1, phi0, phi1, c0.
        assert np.isnan(sigma[0]) and np.isnan(sigma[4])
        assert np.all(np.isfinite(sigma[[1, 2, 3, 5, 6]]))
        assert np.all(np.isfinite(cov[np.ix_([1, 2, 3, 5, 6], [1, 2, 3, 5, 6])]))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda t, y, e: (t, y, np.where(np.arange(t.size) == 3, 0.0, e)), "error"),
        (lambda t, y, e: (t, y, np.where(np.arange(t.size) == 3, np.nan, e)), "error"),
        (lambda t, y, e: (t, y, np.where(np.arange(t.size) == 3, -1.0, e)), "error"),
        (lambda t, y, e: (t, np.where(np.arange(t.size) == 3, np.nan, y), e), "finite"),
        (lambda t, y, e: (np.where(np.arange(t.size) == 3, np.inf, t), y, e), "finite"),
        (lambda t, y, e: (t, y, e[:-1]), "same length"),
        (lambda t, y, e: (t[:0], y[:0], e[:0]), "at least one point"),
    ],
)
def test_fit_rejects_unusable_inputs(mutate, match) -> None:
    # PW-A-8: a zero/NaN error or NaN value ended in 'LinAlgError: Eigenvalues did not
    # converge', a short error array in 'shapes not aligned', empty input in a NaN fit.
    time, value, error = _two_mode(n=300)
    t, y, e = mutate(time, value, error)
    with pytest.raises(ValueError, match=match):
        fit_multisine(t, y, e, np.array([12.34]))


def test_fit_rejects_unusable_frequencies_and_policies() -> None:
    time, value, error = _two_mode(n=300)
    with pytest.raises(ValueError, match="positive"):
        fit_multisine(time, value, error, np.array([-12.34]))  # was clipped to 1e-12
    with pytest.raises(ValueError, match="finite"):
        fit_multisine(time, value, error, np.array([np.nan]))
    with pytest.raises(ValueError, match="refine"):
        fit_multisine(time, value, error, np.array([12.34]), refine="bogus")  # type: ignore[arg-type]
    # A negative frequency is a valid *fixed* sinusoid (sin is odd): no refinement.
    fixed = fit_multisine(time, value, error, np.array([-12.34]), refine="none")
    positive = fit_multisine(time, value, error, np.array([12.34]), refine="none")
    assert fixed.amplitude[0] == pytest.approx(positive.amplitude[0], rel=1e-12)


def test_wide_designs_solve_like_the_pivoted_qr() -> None:
    # PERF-PREWHITEN-01: designs wider than 12 columns go through a condition-guarded
    # Cholesky now; on separated modes it must agree with gelsy to round-off, and a
    # (near-)duplicated frequency must still fall back to the rank-revealing solve.
    from scipy.linalg import lstsq

    from cuperiod.prewhiten.fit import _design, _solve

    rng = np.random.default_rng(8)
    dt = np.sort(rng.uniform(-15.0, 15.0, 3000))
    rhs = rng.normal(size=dt.size)
    separated = 2.0 + 0.37 * np.arange(9)
    design = _design(dt, separated, True)
    reference = lstsq(design, rhs, cond=1e-12, lapack_driver="gelsy")[0]
    assert np.max(np.abs(_solve(design, rhs) - reference)) < 1e-10 * np.max(
        np.abs(reference)
    )
    for gap in (0.0, 1e-6 / 30.0):
        merged = np.append(separated, separated[3] + gap)
        design = _design(dt, merged, True)
        reference = lstsq(design, rhs, cond=1e-12, lapack_driver="gelsy")[0]
        assert np.array_equal(_solve(design, rhs), reference)


def test_simultaneous_fit_of_many_components_is_unchanged_by_the_fast_solver() -> None:
    from cuperiod.prewhiten import fit as fit_module

    rng = np.random.default_rng(2)
    n, span = 2000, 40.0
    time = np.sort(rng.uniform(0.0, span, n))
    truth = 1.5 + 0.9 * np.arange(8)
    value = 5.0 + rng.normal(0.0, 0.002, n)
    for j, f in enumerate(truth):
        value += (0.02 / (1 + j)) * np.sin(_TWO_PI * f * time + j)
    error = np.full(n, 0.002)
    fast = fit_multisine(time, value, error, truth + 1e-3, refine="simultaneous")

    def gelsy_only(design, rhs):
        from scipy.linalg import lstsq

        return lstsq(design, rhs, cond=1e-12, lapack_driver="gelsy")[0]

    original = fit_module._solve
    try:
        fit_module._solve = gelsy_only  # the shipped wide-design behaviour
        slow = fit_multisine(time, value, error, truth + 1e-3, refine="simultaneous")
    finally:
        fit_module._solve = original
    assert np.all(np.abs(fast.frequency - slow.frequency) < 1e-3 * slow.frequency_error)
    assert np.all(np.abs(fast.amplitude - slow.amplitude) < 1e-3 * slow.amplitude_error)


def test_design_is_column_major_and_reused_after_a_cyclic_sweep() -> None:
    # PERF-PREWHITEN-03: _cyclic_refine hands back the design it already updated, and
    # the design is column-major so every column operation is contiguous.
    from cuperiod.prewhiten.fit import _cyclic_refine, _design, _normalize_bounds

    time, value, error = _two_mode(n=600)
    dt = time - time.mean()
    design = _design(dt, np.array([12.3, 17.8]), True)
    assert design.flags.f_contiguous
    by_rows = np.empty_like(design, order="C")
    for j, f in enumerate((12.3, 17.8)):
        by_rows[:, 2 * j] = np.cos((_TWO_PI * f) * dt)
        by_rows[:, 2 * j + 1] = np.sin((_TWO_PI * f) * dt)
    by_rows[:, -1] = 1.0
    assert np.array_equal(design, by_rows)
    sw = 1.0 / error
    freqs, reused = _cyclic_refine(
        dt, value, value * sw, sw, np.array([12.33, 17.82]),
        fit_mean=True, bounds=_normalize_bounds((12.0, 18.0), 2),
        scale=1.0 / 30.0, max_nfev=50, sweeps=2,
    )
    assert np.array_equal(reused, _design(dt, freqs, True))


def test_cyclic_sweeps_build_the_full_design_once(monkeypatch) -> None:
    # PERF-PREWHITEN-03: every refined column is rewritten in place, so neither later
    # sweeps nor the final joint solve need a fresh N x (2K + 1) design.
    from cuperiod.prewhiten import fit as fit_module

    time, value, error = _two_mode(n=600)
    widths: list[int] = []
    original = fit_module._design

    def counting(dt, freqs, fit_mean):  # type: ignore[no-untyped-def]
        widths.append(int(np.size(freqs)))
        return original(dt, freqs, fit_mean)

    monkeypatch.setattr(fit_module, "_design", counting)
    reference = fit_multisine(
        time, value, error, [12.33, 17.82], refine="cyclic", sweeps=3
    )
    assert widths.count(2) == 1
    monkeypatch.setattr(fit_module, "_design", original)
    again = fit_multisine(time, value, error, [12.33, 17.82], refine="cyclic", sweeps=3)
    assert np.array_equal(reference.frequency, again.frequency)
    assert np.array_equal(reference.residuals, again.residuals)


def test_fit_results_compare_by_identity_without_raising() -> None:
    # PW-B-10
    time, value, error = _two_mode(n=300)
    first = fit_multisine(time, value, error, [12.34], refine="none")
    second = fit_multisine(time, value, error, [12.34], refine="none")
    assert first == first and (first == second) is False
    assert len({first, second}) == 2
