"""Tests for phase tracking and ephemeris fitting (`cuperiod.phase_track`).

The physics of record: a constant dP/dt makes the O-C delay a parabola
``(pdot / 2P) * dt**2`` (Sterken 2005). `synthetic_period_change` plants that
exact model, so the delays themselves are checked against the closed form on
noiseless data before any statistics are trusted. The sigma calibration
follows the house Monte-Carlo pattern of test_prewhiten_fit.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.core.errors import InsufficientDataError
from cuperiod.pdot.track import fit_template
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
