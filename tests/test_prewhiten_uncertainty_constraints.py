"""Uncertainty estimators under fit constraints (E4, spec 1.6).

Fixed quantities report exactly zero, tied frequencies inherit their parents'
uncertainty, per-subset amplitudes, phases and zero points get errors of their own,
a negative amplitude under a fixed phase still gives finite analytic errors, and the
bootstrap can be stopped between replicates. Calls on unconstrained fits are
unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cuperiod.prewhiten import uncertainty as unc
from cuperiod.prewhiten.fit import MultiSineFit, fit_multisine
from cuperiod.prewhiten.uncertainty import (
    bootstrap_uncertainties,
    component_uncertainties,
)
from test_prewhiten_constrained_fit import SCENARIOS, Built, build


def _scenario(name: str) -> Any:
    return next(s for s in SCENARIOS if s.name == name)


def _fit(built: Built) -> MultiSineFit:
    return fit_multisine(
        built.t, built.y, built.error, built.start, fit_mean=built.fit_mean,
        t_ref=built.t_ref, refine="simultaneous", constraints=built.constraints,
    )


@pytest.fixture(scope="module")
def sink() -> tuple[Built, MultiSineFit]:
    built = build(_scenario("kitchen-sink"), "short")
    return built, _fit(built)


# --- covariance -----------------------------------------------------------------------


def test_covariance_estimator_reports_subset_errors(sink) -> None:
    built, fit = sink
    detail = fit.detail
    errors = component_uncertainties(built.t, built.error, fit,
                                     correlation_correction=False)
    assert np.array_equal(errors.frequency, fit.frequency_error)
    assert errors.subset_amplitude is not None and errors.subset_phase is not None
    assert errors.subset_offset is not None
    assert np.array_equal(errors.subset_amplitude, detail.subset_amplitude_error,
                          equal_nan=True)
    assert np.array_equal(errors.subset_offset, detail.subset_offset_error)
    inflated = component_uncertainties(built.t, built.error, fit)
    factor = np.sqrt(inflated.correlation_factor)
    assert np.allclose(inflated.subset_offset, errors.subset_offset * factor)
    # The fixed phase of the last component reports exactly zero.
    assert errors.phase[4] == 0.0


# --- analytic -------------------------------------------------------------------------


def test_analytic_reads_the_magnitude_of_a_signed_amplitude() -> None:
    built = build(_scenario("fixed-phase-pi"), "short")
    fit = _fit(built)
    assert fit.amplitude[2] < 0.0
    errors = component_uncertainties(built.t, built.error, fit, method="analytic",
                                     correlation_correction=False)
    assert np.all(np.isfinite(errors.frequency))
    assert np.all(np.isfinite(errors.amplitude))
    assert errors.phase[2] == 0.0  # fixed
    n, span = fit.n_samples, float(np.ptp(built.t))
    sigma = float(np.sqrt(fit.rss / np.sum(1.0 / built.error**2)))
    expected_f = np.sqrt(6.0 / n) * sigma / (np.pi * span * abs(fit.amplitude[2]))
    assert errors.frequency[2] == pytest.approx(expected_f, rel=1e-12)


def test_analytic_fixed_tied_and_per_subset(sink) -> None:
    built, fit = sink
    errors = component_uncertainties(built.t, built.error, fit, method="analytic",
                                     correlation_correction=False)
    # f3 = f1 + f2: sqrt(sigma_1^2 + sigma_2^2) of the roots' own analytic errors.
    plain = component_uncertainties(built.t, built.error, _plain_copy(fit),
                                    method="analytic", correlation_correction=False)
    assert errors.frequency[2] == pytest.approx(
        np.hypot(plain.frequency[0], plain.frequency[1]), rel=1e-12)
    assert errors.phase[4] == 0.0
    vcodes = built.constraints.variation_groups
    counts = np.bincount(vcodes)
    sigma = unc._residual_sigma(fit, built.error)
    assert errors.subset_amplitude is not None and errors.subset_phase is not None
    # Component 0 varies in amplitude only; component 3 in both.
    assert np.allclose(errors.subset_amplitude[0], np.sqrt(2.0 / counts) * sigma)
    assert np.all(np.isnan(errors.subset_phase[0]))
    amp3 = np.abs(fit.detail.subset_amplitude[3])
    assert np.allclose(errors.subset_phase[3], np.sqrt(2.0 / counts) * sigma / amp3)
    assert np.all(np.isnan(errors.subset_amplitude[1]))
    ocounts = np.bincount(built.constraints.offset_groups)
    assert np.allclose(errors.subset_offset, sigma / np.sqrt(ocounts))


def _plain_copy(fit: MultiSineFit) -> MultiSineFit:
    from dataclasses import replace

    return replace(fit, detail=None, amplitude=np.abs(fit.amplitude))


def test_analytic_noise_level_includes_the_point_weights(sink) -> None:
    built, fit = sink
    omega = built.constraints.point_weights
    w = omega / omega.mean() / built.error**2
    assert unc._residual_sigma(fit, built.error) == pytest.approx(
        np.sqrt(fit.rss / w.sum()), rel=1e-14)


# --- bootstrap ------------------------------------------------------------------------


def test_bootstrap_under_constraints() -> None:
    built = build(_scenario("kitchen-sink"), "long")
    fit = _fit(built)
    errors = component_uncertainties(
        built.t, built.error, fit, method="bootstrap", n_resamples=40, seed=1,
    )
    # fixed -> exactly zero (the last component's phase).
    assert errors.phase[4] == 0.0
    assert np.all(errors.frequency > 0.0)
    # tied scatter consistent with the propagated covariance.
    ratio = errors.frequency / fit.frequency_error
    assert 0.6 < ratio[2] < 1.6
    assert errors.subset_amplitude is not None and errors.subset_offset is not None
    assert errors.subset_phase is not None
    assert np.all(np.isnan(errors.subset_amplitude[1]))  # does not vary
    assert np.all(np.isfinite(errors.subset_amplitude[0]))
    assert np.all(np.isfinite(errors.subset_phase[3]))
    detail = fit.detail
    assert np.all(errors.subset_offset / detail.subset_offset_error > 0.5)
    assert np.all(errors.subset_offset / detail.subset_offset_error < 2.0)
    assert errors.correlation_factor == 1.0


def test_bootstrap_of_fixed_quantities_is_exactly_zero() -> None:
    built = build(_scenario("psi-only"), "short")
    fit = _fit(built)
    sigma_f, sigma_a, sigma_p = bootstrap_uncertainties(
        built.t, built.error, fit, n_resamples=10, seed=2,
    )
    assert np.all(sigma_f == 0.0)
    assert sigma_a[0] == 0.0 and sigma_a[2] == 0.0
    assert sigma_a[1] > 0.0 and sigma_p[0] > 0.0


def test_bootstrap_replicates_keep_the_constraints(monkeypatch, sink) -> None:
    built, fit = sink
    seen: list[Any] = []
    original = unc.fit_multisine

    def spy(*args: Any, **kwargs: Any) -> MultiSineFit:
        seen.append(kwargs.get("constraints"))
        return original(*args, **kwargs)

    monkeypatch.setattr(unc, "fit_multisine", spy)
    bootstrap_uncertainties(built.t, built.error, fit, n_resamples=3, seed=0)
    assert len(seen) == 3
    for constraints in seen:
        assert constraints is not None
        assert constraints.components == fit.detail.constraints.components
        assert constraints.start is not None
        assert np.array_equal(constraints.start.phase, fit.phase)


def test_should_stop_is_honoured(monkeypatch, sink) -> None:
    built, fit = sink
    calls = {"fits": 0, "polls": 0}
    original = unc.fit_multisine

    def counting(*args: Any, **kwargs: Any) -> MultiSineFit:
        calls["fits"] += 1
        return original(*args, **kwargs)

    def stop() -> bool:
        calls["polls"] += 1
        return calls["polls"] > 4

    monkeypatch.setattr(unc, "fit_multisine", counting)
    errors = component_uncertainties(built.t, built.error, fit, method="bootstrap",
                                     n_resamples=50, should_stop=stop)
    assert calls["fits"] == 4
    assert np.all(np.isfinite(errors.frequency))
    calls.update(fits=0, polls=0)
    early = bootstrap_uncertainties(built.t, built.error, fit, n_resamples=50,
                                    should_stop=lambda: True)
    assert calls["fits"] == 0
    assert all(np.all(np.isnan(e)) for e in early)


def test_should_stop_never_called_back_runs_everything(sink) -> None:
    built, fit = sink
    a = bootstrap_uncertainties(built.t, built.error, fit, n_resamples=4, seed=5)
    b = bootstrap_uncertainties(built.t, built.error, fit, n_resamples=4, seed=5,
                                should_stop=lambda: False)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


# --- legacy calls ---------------------------------------------------------------------


def test_unconstrained_fits_are_unchanged() -> None:
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 20, 300))
    y = 0.02 * np.sin(2 * np.pi * 3.3 * t + 0.4) + rng.normal(0, 0.002, t.size)
    fit = fit_multisine(t, y, None, [3.3], refine="simultaneous")
    assert fit.detail is None
    for method in ("covariance", "analytic", "bootstrap"):
        errors = component_uncertainties(t, None, fit, method=method, n_resamples=8)
        assert errors.subset_amplitude is None
        assert errors.subset_phase is None
        assert errors.subset_offset is None
    direct = bootstrap_uncertainties(t, None, fit, n_resamples=8, seed=4)
    wrapped = component_uncertainties(t, None, fit, method="bootstrap",
                                      n_resamples=8, seed=4)
    assert np.array_equal(direct[0], wrapped.frequency)
    assert np.array_equal(direct[1], wrapped.amplitude)
    assert np.array_equal(direct[2], wrapped.phase)
