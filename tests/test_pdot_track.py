"""Tests for phase tracking and ephemeris fitting (`cuperiod.phase_track`).

The physics of record: a constant dP/dt makes the O-C delay a parabola
``(pdot / 2P) * dt**2`` (Sterken 2005). `synthetic_period_change` plants that
exact model, so the delays themselves are checked against the closed form on
noiseless data before any statistics are trusted. The sigma calibration
follows the house Monte-Carlo pattern of test_prewhiten_fit.py.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from pydantic import ValidationError

import cuperiod as cup
from cuperiod.core.errors import InsufficientDataError
from cuperiod.pdot import track as track_module
from cuperiod.pdot.track import (
    PhaseTemplate,
    _ephemeris_from_delays,
    _fit_template_order,
    _ladder_chi2,
    _shift_scan_chi2,
    fit_template,
)
from synth import synthetic_period_change, synthetic_sine

# --- delays against the closed form ------------------------------------------


def test_delays_match_analytic_parabola():
    t, mag, err, truth = synthetic_period_change(noise=1e-6, seed=0)
    settings = cup.PdotSettings(harmonics=2)
    res = cup.phase_track((t, mag, err), period=truth["period"], settings=settings)
    dt = res.t_mid - truth["t_ref"]
    expected = 0.5 * (truth["pdot"] / truth["period"]) * dt * dt
    # Common-mode offsets are free (c0/c1); compare after removing them.
    coef = np.polyfit(dt, res.delay - expected, 1)
    resid = res.delay - expected - np.polyval(coef, dt)
    assert float(np.max(np.abs(resid))) < 2.5e-4


def test_positive_pdot_gives_upward_parabola():
    t, mag, err, truth = synthetic_period_change(seed=1)
    res = cup.phase_track((t, mag, err), period=truth["period"])
    dt = res.t_mid - res.quadratic_fit.t_ref
    edges = res.delay[np.abs(dt) > 0.7 * np.max(np.abs(dt))]
    center = res.delay[np.abs(dt) < 0.3 * np.max(np.abs(dt))]
    assert edges.mean() > center.mean()
    assert res.quadratic_fit.coefficients[2] > 0.0


# --- recovery -----------------------------------------------------------------


def test_recovers_planted_pdot():
    t, mag, err, truth = synthetic_period_change(seed=1)
    res = cup.phase_track((t, mag, err), period=truth["period"])
    assert res.pdot == pytest.approx(truth["pdot"], abs=5 * res.pdot_err)
    assert res.f_test_p < 1e-8
    assert not res.wrap_risk
    assert res.pdot_s_per_yr == pytest.approx(res.pdot * 86400 * 365.25)


def test_recovers_negative_pdot():
    t, mag, err, truth = synthetic_period_change(pdot=-8e-8, seed=6)
    res = cup.phase_track((t, mag, err), period=truth["period"])
    assert res.pdot < 0.0
    assert res.pdot == pytest.approx(truth["pdot"], abs=5 * res.pdot_err)


def test_null_has_no_significant_quadratic():
    t, mag, err, truth = synthetic_period_change(pdot=0.0, seed=7)
    res = cup.phase_track((t, mag, err), period=truth["period"])
    assert abs(res.pdot) < 5 * res.pdot_err
    assert res.f_test_p > 1e-3


def test_seeding_from_search_agrees_with_iteration():
    t, mag, err, truth = synthetic_period_change(pdot=2e-7 * 0.5, seed=9)
    search = cup.pdot_search((t, mag, err), period=truth["period"], backend="cpu")
    seeded = cup.phase_track(
        (t, mag, err), period=truth["period"], pdot=search.best.pdot
    )
    plain = cup.phase_track((t, mag, err), period=truth["period"])
    assert seeded.pdot == pytest.approx(plain.pdot, abs=3 * seeded.pdot_err)
    assert seeded.pdot == pytest.approx(truth["pdot"], abs=5 * seeded.pdot_err)


# --- sigma calibration (Monte Carlo, house band 0.6-1.6) ----------------------


def test_pdot_sigma_calibration():
    pulls = []
    for seed in range(25):
        t, mag, err, truth = synthetic_period_change(
            n=1200, span=1200.0, pdot=5e-8, seed=200 + seed
        )
        res = cup.phase_track((t, mag, err), period=truth["period"])
        pulls.append((res.pdot - truth["pdot"]) / res.pdot_err)
    pulls = np.asarray(pulls)
    scatter = float(pulls.std(ddof=1))
    assert abs(float(pulls.mean())) < 0.6
    assert 1.0 / 1.6 < scatter < 1.6


# --- template -----------------------------------------------------------------


def test_bic_selects_harmonic_order():
    t, mag, err = synthetic_sine(n=1500, span=400.0, seed=3)
    lc = cup.LightCurve.from_arrays(t, mag, err)
    finite = lc.finite()
    w = 1.0 / (finite.error**2)
    one = fit_template(
        finite.time - finite.time.mean(), finite.value, w, 1.0 / 0.6234
    )
    assert one.n_harmonics == 1
    t2, mag2, err2, truth = synthetic_period_change(pdot=0.0, seed=4)
    w2 = np.full(t2.size, 1.0 / 0.05**2)
    two = fit_template(t2 - t2.mean(), mag2, w2, 1.0 / truth["period"])
    assert two.n_harmonics == 2


def test_template_evaluation_matches_model():
    rng = np.random.default_rng(0)
    dt = np.sort(rng.uniform(-10, 10, 500))
    tpl = fit_template(
        dt,
        1.0 + 0.3 * np.cos(2 * np.pi * 2.0 * dt) + 0.1 * np.sin(2 * np.pi * 4.0 * dt),
        np.full(500, 1.0),
        2.0,
        n_harmonics=2,
    )
    manual = (
        tpl.offset
        + tpl.cos_coeffs[0] * np.cos(2 * np.pi * 2.0 * dt)
        + tpl.sin_coeffs[0] * np.sin(2 * np.pi * 2.0 * dt)
        + tpl.cos_coeffs[1] * np.cos(2 * np.pi * 4.0 * dt)
        + tpl.sin_coeffs[1] * np.sin(2 * np.pi * 4.0 * dt)
    )
    assert np.allclose(tpl(dt), manual, atol=1e-12)
    assert tpl.amplitudes[0] == pytest.approx(0.3, abs=1e-6)


def test_segment_offsets_absorb_calibration_drift():
    t, mag, err, truth = synthetic_period_change(seed=5)
    thirds = np.searchsorted(t, np.quantile(t, [1 / 3, 2 / 3]))
    mag = mag.copy()
    mag[: thirds[0]] += 0.02
    mag[thirds[1] :] -= 0.02
    res = cup.phase_track((t, mag, err), period=truth["period"])
    assert res.pdot == pytest.approx(truth["pdot"], abs=6 * res.pdot_err)


# --- unweighted input ---------------------------------------------------------


def test_unweighted_input_calibrated():
    pulls = []
    for seed in range(12):
        t, mag, _, truth = synthetic_period_change(
            n=1200, span=1200.0, pdot=5e-8, seed=300 + seed
        )
        res = cup.phase_track((t, mag), period=truth["period"])
        pulls.append((res.pdot - truth["pdot"]) / res.pdot_err)
    scatter = float(np.std(pulls, ddof=1))
    assert 0.4 < scatter < 2.2


# --- fit_ephemeris ------------------------------------------------------------


def test_fit_ephemeris_recovers_quadratic():
    p0, pdot = 2.5, 4.0e-8
    cycles = np.arange(0.0, 4000.0, 7.0)
    times = 2450000.0 + p0 * cycles + 0.5 * p0 * pdot * cycles**2
    times = times + np.random.default_rng(3).normal(0.0, 0.001, times.size)
    fit = cup.fit_ephemeris(times, period=p0, errors=np.full(times.size, 0.001))
    assert fit.pdot == pytest.approx(pdot, abs=4 * fit.pdot_err)
    assert fit.f_test_p < 1e-10
    assert 0.5 < fit.chi2_reduced < 2.0
    assert fit.degree == 2
    predicted = fit.predict(times)
    assert predicted.shape == times.shape


def test_pddot_err_scaled_inflates_like_pdot_err_scaled():
    p0, pdot = 2.5, 4.0e-8
    cycles = np.arange(0.0, 4000.0, 7.0)
    times = 2450000.0 + p0 * cycles + 0.5 * p0 * pdot * cycles**2
    times = times + np.random.default_rng(3).normal(0.0, 0.001, times.size)
    fit = cup.fit_ephemeris(times, period=p0, errors=np.full(times.size, 0.001))
    # A quadratic fits no pddot, so there is nothing to scale.
    assert np.isnan(fit.pddot_err_scaled)
    assert np.isnan(fit.to_dict()["pddot_err_scaled"])
    # The cubic's error is inflated by sqrt(chi2_reduced) when > 1, never deflated.
    loose = dataclasses.replace(fit, pddot=1e-12, pddot_err=2e-13, chi2_reduced=4.0)
    assert loose.pddot_err_scaled == pytest.approx(4e-13, rel=1e-12)
    assert loose.to_dict()["pddot_err_scaled"] == loose.pddot_err_scaled
    tight = dataclasses.replace(loose, chi2_reduced=0.5)
    assert tight.pddot_err_scaled == tight.pddot_err


def test_pdot_equals_2q_over_p():
    # The classical quadratic ephemeris T(E) = T0 + P*E + Q*E^2 has
    # dP/dt = 2Q/P (Sterken 2005; OCFit's QuadTerm) — the conversion a
    # refereed AJ paper once inverted.
    p0, q = 2.5, 6.25e-8
    cycles = np.arange(0.0, 3000.0, 5.0)
    times = 2450000.0 + p0 * cycles + q * cycles**2
    times = times + np.random.default_rng(5).normal(0.0, 5e-4, times.size)
    fit = cup.fit_ephemeris(times, period=p0, errors=np.full(times.size, 5e-4))
    assert fit.pdot == pytest.approx(2.0 * q / p0, rel=0.05)


def test_information_criteria_and_cycle_residual():
    t, mag, err, truth = synthetic_period_change(seed=1)
    res = cup.phase_track((t, mag, err), period=truth["period"])
    # Positive linear-minus-quadratic BIC difference favors the dP/dt model.
    assert res.linear_fit.bic - res.quadratic_fit.bic > 10.0
    assert np.isnan(res.quadratic_fit.worst_phase_residual)
    p0 = 1.7
    cycles = np.arange(0.0, 900.0, 3.0)
    times = 2450000.0 + p0 * cycles + np.random.default_rng(1).normal(
        0.0, 0.002, cycles.size
    )
    fit = cup.fit_ephemeris(times, period=p0)
    assert fit.worst_phase_residual < 0.05


def test_fit_ephemeris_without_errors():
    p0 = 1.7
    cycles = np.arange(0.0, 900.0, 3.0)
    times = 2450000.0 + p0 * cycles + np.random.default_rng(1).normal(
        0.0, 0.002, cycles.size
    )
    fit = cup.fit_ephemeris(times, period=p0)
    assert abs(fit.pdot) < 5 * fit.pdot_err


def test_fit_ephemeris_guards():
    with pytest.raises(InsufficientDataError):
        cup.fit_ephemeris([1.0, 2.0, 3.0], period=1.0)
    with pytest.raises(ValueError, match="positive"):
        cup.fit_ephemeris([1.0, 2.0, 3.0, 4.0], period=-1.0)
    with pytest.raises(ValueError, match="length"):
        cup.fit_ephemeris([1.0, 2.0, 3.0, 4.0], period=1.0, errors=[0.1, 0.1])


# --- guards and plumbing ------------------------------------------------------


def test_multiband_rejected():
    bands = {
        "g": cup.LightCurve.from_arrays([1.0, 2.0], [1.0, 1.0]),
        "r": cup.LightCurve.from_arrays([1.5, 2.5], [1.0, 1.0]),
    }
    with pytest.raises(ValueError, match="single band"):
        cup.phase_track(
            cup.MultiBandLightCurve.from_light_curves(bands), period=1.0
        )


def test_insufficient_segments_raise():
    t, mag, err = synthetic_sine(n=60, span=10.0, seed=2)
    with pytest.raises(InsufficientDataError):
        cup.phase_track(
            (t, mag, err),
            period=0.6234,
            settings=cup.PdotSettings(min_segment_points=40),
        )


def test_wrap_risk_flagged_for_extreme_drift():
    t, mag, err, truth = synthetic_period_change(
        n=1500, span=1500.0, period=0.2, pdot=0.2 * 4e-6, noise=0.02, seed=13
    )
    settings = cup.PdotSettings(template_iterations=0, harmonics=2)
    res = cup.phase_track((t, mag, err), period=truth["period"], settings=settings)
    assert res.wrap_risk


def test_to_dict_json_serializable():
    t, mag, err, truth = synthetic_period_change(n=900, span=900.0, seed=14)
    res = cup.phase_track((t, mag, err), period=truth["period"])
    payload = json.dumps(res.to_dict())
    assert "quadratic_fit" in payload
    assert res.n_segments == res.t_mid.size


# --- regressions: fit_ephemeris -----------------------------------------------


def _timing_list(
    seed: int = 0, *, n: int = 300, p0: float = 0.75, pdot: float = 3e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Event times whose old half is 20x noisier than the new half."""
    rng = np.random.default_rng(seed)
    cycles = np.sort(rng.choice(np.arange(4000), n, replace=False)).astype(float)
    sig = np.where(cycles < cycles[n // 2], 0.01, 5e-4)
    times = 2450000.0 + p0 * cycles + 0.5 * p0 * pdot * cycles**2
    return times + rng.normal(0.0, sig), sig


def test_fit_ephemeris_keeps_errors_paired_with_their_events():
    # The errors used to stay in the caller's order while the times were sorted,
    # so a shuffled (or source-grouped) list attached each error to another event.
    times, sig = _timing_list()
    ref = cup.fit_ephemeris(times, period=0.75, errors=sig)
    perm = np.random.default_rng(1).permutation(times.size)
    shuffled = cup.fit_ephemeris(times[perm], period=0.75, errors=sig[perm])
    assert shuffled.pdot == pytest.approx(ref.pdot, rel=1e-12)
    assert shuffled.pdot_err == pytest.approx(ref.pdot_err, rel=1e-12)
    assert shuffled.chi2 == pytest.approx(ref.chi2, rel=1e-12)
    assert 0.6 < ref.chi2_reduced < 1.6


def test_fit_ephemeris_drops_a_non_finite_time_with_its_error():
    times, sig = _timing_list()
    holed = times.copy()
    holed[7] = np.nan
    fit = cup.fit_ephemeris(holed, period=0.75, errors=sig)
    ref = cup.fit_ephemeris(
        np.delete(times, 7), period=0.75, errors=np.delete(sig, 7)
    )
    assert fit.n_used == times.size - 1
    assert fit.pdot == pytest.approx(ref.pdot, rel=1e-12)
    assert fit.chi2 == pytest.approx(ref.chi2, rel=1e-12)


def _quadratic_timings(seed: int, *, scatter: float = 3e-4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cycles = np.sort(rng.choice(np.arange(8000), 200, replace=False)).astype(float)
    times = 2450000.0 + 0.5 * cycles + 0.5 * 0.5 * 2e-9 * cycles**2
    return times + rng.normal(0.0, scatter, cycles.size)


def test_fit_ephemeris_without_errors_scales_from_the_quadratic_residuals():
    # errors=None used the raw O-C spread as the timing error: it contains the
    # trial-period slope and the parabola itself, so pdot_err measured the trend
    # (tens of times too large, and dependent on the trial period).
    times = _quadratic_timings(0)
    fit = cup.fit_ephemeris(times, period=0.5)
    honest = cup.fit_ephemeris(times, period=0.5, errors=np.full(times.size, 3e-4))
    assert fit.chi2_reduced == pytest.approx(1.0, rel=1e-9)
    assert fit.pdot == pytest.approx(honest.pdot, rel=1e-9)
    assert fit.pdot_err == pytest.approx(honest.pdot_err, rel=0.15)
    shifted = cup.fit_ephemeris(times, period=0.5 + 1e-5)
    assert shifted.pdot_err == pytest.approx(fit.pdot_err, rel=1e-3)


def test_fit_ephemeris_without_errors_is_calibrated():
    pulls = []
    for seed in range(60):
        fit = cup.fit_ephemeris(_quadratic_timings(100 + seed), period=0.5 + 1e-5)
        truth = 2e-9 * (1.0 + 2.0 * 1e-5 / 0.5)  # pdot against the trial period
        pulls.append((fit.pdot - truth) / fit.pdot_err)
    scatter = float(np.std(pulls, ddof=1))
    assert 1.0 / 1.6 < scatter < 1.6
    assert abs(float(np.mean(pulls))) < 0.6


def test_f_test_is_nan_when_there_is_nothing_to_test():
    # A noise-free linear ephemeris has chi2 = 0 for both models: the quadratic
    # term explains nothing, which used to come out as p = 0 ("infinitely
    # significant").
    assert np.isnan(cup.fit_ephemeris(np.arange(12.0), period=1.0).f_test_p)
    exact = 2450000.0 + 2.5 * np.arange(20.0)
    assert np.isnan(cup.fit_ephemeris(exact, period=2.5).f_test_p)
    t = np.arange(10.0)
    oc = 1e-3 * np.sin(t)
    oc[3] = np.nan
    _, quad = _ephemeris_from_delays(t, oc, np.full(t.size, 1e-3), 1.0)
    assert np.isnan(quad.f_test_p)
    # A genuine parabola stays significant.
    times, sig = _timing_list()
    assert cup.fit_ephemeris(times, period=0.75, errors=sig).f_test_p < 1e-6


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_arguments_rejected(bad: float):
    times = 2450000.0 + 0.5 * np.arange(20.0)
    with pytest.raises(ValueError, match="finite"):
        cup.fit_ephemeris(times, period=bad)
    with pytest.raises(ValueError, match="finite"):
        cup.fit_ephemeris(times, period=0.5, t0=bad)
    t, mag, err, truth = synthetic_period_change(n=400, span=400.0, seed=2)
    with pytest.raises(ValueError, match="finite"):
        cup.phase_track((t, mag, err), period=bad)
    with pytest.raises(ValueError, match="finite"):
        cup.phase_track((t, mag, err), period=truth["period"], pdot=bad)


# --- regressions: segmentation ------------------------------------------------


def test_three_segments_reported_as_a_segmentation_problem():
    # The quadratic ephemeris needs four O-C points; three segments used to pass
    # the segmentation checks and die later in the ephemeris fit.
    t, mag, err = synthetic_sine(n=70, span=60.0, seed=2)
    with pytest.raises(InsufficientDataError, match="4 usable segments"):
        cup.phase_track((t, mag, err), period=0.6234)
    # An explicit 2 or 3 can never succeed, so the settings refuse it up front.
    with pytest.raises(ValidationError, match="greater than or equal to 4"):
        cup.PdotSettings(n_segments=3)
    t, mag, err, truth = synthetic_period_change(n=1200, span=1200.0, seed=1)
    res = cup.phase_track(
        (t, mag, err), period=truth["period"], settings=cup.PdotSettings(n_segments=4)
    )
    assert res.n_segments == 4


# --- regressions: template order ----------------------------------------------


def test_template_order_ignores_a_global_error_rescale():
    # chi2 + k*ln(N) buys a harmonic whenever the quoted errors are optimistic;
    # the scale-free BIC does not care about an overall weight factor.
    t, mag, err = synthetic_sine(n=1500, span=400.0, seed=3)
    dt = t - t.mean()
    w = 1.0 / err**2
    orders = {
        fac: fit_template(dt, mag, w * fac, 1.0 / 0.6234).n_harmonics
        for fac in (1.0, 25.0, 0.04)
    }
    assert orders == {1.0: 1, 25.0: 1, 0.04: 1}


def test_phase_track_unchanged_by_optimistic_errors():
    t, mag, err, truth = synthetic_period_change(n=1200, span=1200.0, seed=21)
    honest = cup.phase_track((t, mag, err), period=truth["period"])
    optimistic = cup.phase_track((t, mag, 0.2 * err), period=truth["period"])
    assert optimistic.template.n_harmonics == honest.template.n_harmonics == 2
    assert optimistic.pdot == pytest.approx(honest.pdot, rel=1e-6)
    # error_scale = sqrt(chi2_reduced) absorbs the understated errors.
    assert optimistic.pdot_err == pytest.approx(honest.pdot_err, rel=0.1)


@pytest.mark.parametrize("per_order", [False, True])
def test_noise_free_template_order_is_not_chosen_on_rounding(monkeypatch, per_order):
    # On exact data every order >= 2 leaves chi2 at rounding level, whose
    # logarithm is noise; without a floor the per-order fits picked order 6.
    if per_order:
        monkeypatch.setattr(track_module, "_ladder_chi2", lambda *a: None)
    rng = np.random.default_rng(0)
    dt = np.sort(rng.uniform(-200.0, 200.0, 800))
    y = 13.0 + 0.3 * np.sin(2 * np.pi * dt / 0.5) + 0.1 * np.cos(4 * np.pi * dt / 0.5)
    tpl = fit_template(dt, y, np.full(dt.size, 1.0 / 0.01**2), 2.0)
    assert tpl.n_harmonics == 2


# --- regressions: Gram-form shift scan and QR order ladder ----------------------


def _direct_scan(dt, y, w, template, shifts, free_offset):
    """The v1.3.0 coarse scan, kept verbatim as the reference."""
    shape = np.empty((shifts.size, dt.size), dtype=np.float64)
    for k, s in enumerate(shifts):
        shape[k] = template._shape(dt - s)
    resid0 = y[None, :] - template.offset - shape
    if free_offset:
        c = (resid0 @ w) / float(w.sum())
        resid0 = resid0 - c[:, None]
    return (resid0 * resid0) @ w


@pytest.mark.parametrize("harmonics", [1, 3, 10])
@pytest.mark.parametrize("free_offset", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("noise", [0.0, 0.03])
def test_gram_scan_matches_direct_evaluation(harmonics, free_offset, weighted, noise):
    rng = np.random.default_rng(harmonics + 10 * free_offset + 100 * weighted)
    f0 = 1.0 / 0.7
    dt = np.sort(rng.uniform(-30.0, 30.0, 400))
    tpl = PhaseTemplate(
        frequency=f0,
        offset=13.0,
        cos_coeffs=rng.normal(0.0, 0.2, harmonics) / np.arange(1, harmonics + 1),
        sin_coeffs=rng.normal(0.0, 0.2, harmonics) / np.arange(1, harmonics + 1),
        chi2_reduced=1.0,
    )
    y = tpl(dt, 0.13) + 0.02 + rng.normal(0.0, 1.0, dt.size) * noise
    w = (
        1.0 / rng.uniform(0.01, 0.05, dt.size) ** 2
        if weighted
        else np.full(dt.size, 1.0 / dt.size)
    )
    n_scan = max(128, 32 * harmonics)
    shifts = (np.arange(n_scan, dtype=np.float64) / n_scan - 0.5) / f0
    ref = _direct_scan(dt, y, w, tpl, shifts, free_offset)
    got = _shift_scan_chi2(dt, y, w, tpl, shifts, free_offset=free_offset)
    span = float(ref.max() - ref.min())
    assert np.max(np.abs(got - ref)) <= 1e-10 * span
    assert int(np.argmin(got)) == int(np.argmin(ref))


def _direct_segment_shift(dt, y, w, template, *, free_offset, error_scale):
    """The v1.3.0 _fit_segment_shift (direct scan + polish), the parity reference."""
    period = 1.0 / template.frequency
    n_scan = max(128, 32 * template.n_harmonics)
    shifts = (np.arange(n_scan, dtype=np.float64) / n_scan - 0.5) * period
    sum_w = float(w.sum())
    chi2_grid = _direct_scan(dt, y, w, template, shifts, free_offset)
    k0 = int(np.argmin(chi2_grid))
    km, kp = (k0 - 1) % n_scan, (k0 + 1) % n_scan
    sm, s0, sp = chi2_grid[km], chi2_grid[k0], chi2_grid[kp]
    denom = sm - 2.0 * s0 + sp
    step = period / n_scan
    off = 0.5 * (sm - sp) / denom if denom > 0 else 0.0
    delay = shifts[k0] + step * float(np.clip(off, -0.5, 0.5))
    for _ in range(6):
        x = dt - delay
        slope = -template._shape_deriv(x)
        resid = y - template.offset - template._shape(x)
        if free_offset:
            resid = resid - float(np.dot(w, resid)) / sum_w
            slope = slope - float(np.dot(w, slope)) / sum_w
        grad = -2.0 * float(np.dot(w, resid * slope))
        curv = 2.0 * float(np.dot(w, slope * slope))
        if curv <= 0.0:
            break
        delay += float(np.clip(-grad / curv, -step, step))
    delay = (delay + 0.5 * period) % period - 0.5 * period
    x = dt - delay
    slope = -template._shape_deriv(x)
    resid = y - template.offset - template._shape(x)
    n_free = 1
    if free_offset:
        resid = resid - float(np.dot(w, resid)) / sum_w
        slope = slope - float(np.dot(w, slope)) / sum_w
        n_free = 2
    chi2 = float(np.dot(w, resid * resid))
    fisher = float(np.dot(w, slope * slope))
    if fisher <= 0.0:
        return delay, np.inf, chi2 / max(1, dt.size - n_free)
    return delay, error_scale / np.sqrt(fisher), chi2 / max(1, dt.size - n_free)


@pytest.mark.parametrize(
    "settings",
    [
        cup.PdotSettings(),
        cup.PdotSettings(harmonics=10),
        cup.PdotSettings(segment_offsets=False),
    ],
)
@pytest.mark.parametrize("weighted", [True, False])
def test_phase_track_matches_direct_scan(monkeypatch, settings, weighted):
    t, mag, err, truth = synthetic_period_change(n=900, span=900.0, seed=17)
    data = (t, mag, err) if weighted else (t, mag)
    fast = cup.phase_track(data, period=truth["period"], settings=settings)
    monkeypatch.setattr(track_module, "_fit_segment_shift", _direct_segment_shift)
    slow = cup.phase_track(data, period=truth["period"], settings=settings)
    assert fast.template.n_harmonics == slow.template.n_harmonics
    assert np.all(np.abs(fast.delay - slow.delay) < 1e-6 * slow.delay_err)
    assert np.allclose(fast.delay_err, slow.delay_err, rtol=1e-9)
    assert fast.pdot == pytest.approx(slow.pdot, rel=1e-9)


@pytest.mark.parametrize("weighted", [True, False])
def test_qr_ladder_matches_per_order_fits(weighted):
    rng = np.random.default_rng(4)
    dt = np.sort(rng.uniform(-100.0, 100.0, 600))
    y = 13.0 + 0.3 * np.sin(2 * np.pi * dt / 0.37) + rng.normal(0.0, 0.05, dt.size)
    w = (
        1.0 / rng.uniform(0.03, 0.08, dt.size) ** 2
        if weighted
        else np.full(dt.size, 1.0 / dt.size)
    )
    got = _ladder_chi2(dt, y, w, 1.0 / 0.37, 10)
    ref = [_fit_template_order(dt, y, w, 1.0 / 0.37, k)[1] for k in range(1, 11)]
    assert got is not None
    assert np.allclose(got, ref, rtol=1e-9, atol=0.0)


def test_qr_ladder_falls_back_on_commensurate_sampling():
    # Sampling locked to a whole number of points per cycle makes the high
    # harmonics degenerate (rank-deficient design): the cheap nested QR is then
    # not trustworthy, and the order must come from the per-order fits.
    dt = np.arange(400.0) * 0.25  # 4 points per cycle of a 1 c/d template
    y = 1.0 + np.sin(2 * np.pi * dt) + np.random.default_rng(0).normal(0, 0.01, 400)
    w = np.full(dt.size, 1.0 / 0.01**2)
    assert _ladder_chi2(dt, y, w, 1.0, 10) is None
    tpl = fit_template(dt, y, w, 1.0)
    assert tpl.n_harmonics == 1
