"""Tests for the cyclic O-C / light-travel-time search (`cuperiod.ltte_search`).

The model of record is Irwin's (1952) light-time orbit; `synth.py` carries an
independent Kepler solver and delay function, so the planted cycles never share
code with the implementation. Parameters that are degenerate for a circular
orbit (phase vs periastron epoch) are compared through the predicted delay
curve, not through the parameters themselves.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.core.errors import InsufficientDataError
from cuperiod.pdot.batch import pdot_to_rows
from cuperiod.pdot.ltte import ltte_delay
from synth import ltte_delay_reference, synthetic_ltte

DAY = 86400.0


def _oc_points(
    *,
    seed: int = 1,
    n: int = 40,
    span: float = 5000.0,
    period: float = 0.4,
    pdot: float = 3e-8,
    outer_period: float = 1300.0,
    amplitude: float = 0.003,
    omega: float = 0.4,
    eccentricity: float = 0.0,
    t_peri: float = 0.0,
    sigma: float = 5e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Noisy O-C points: parabola + light-time orbit; returns (t, oc, err, truth)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n)) + 2450000.0
    dt = t - t.mean()
    truth = 0.5 * (pdot / period) * dt**2 + ltte_delay_reference(
        dt, outer_period, amplitude, omega, eccentricity, t_peri
    )
    oc = truth + rng.normal(0.0, sigma, n)
    return t, oc, np.full(n, sigma), truth


# --- the delay itself matches the independent reference ----------------------


def test_delay_matches_independent_irwin_reference():
    dt = np.linspace(-1000.0, 1000.0, 501)
    for e in (0.0, 0.3, 0.8):
        ours = ltte_delay(dt, 700.0, 0.01, 1.1, e, 123.0)
        ref = ltte_delay_reference(dt, 700.0, 0.01, 1.1, e, 123.0)
        assert np.max(np.abs(ours - ref)) < 1e-9
    # Circular: a plain sinusoid, positive = late.
    assert ltte_delay(0.0, 100.0, 2.0, np.pi / 2) == pytest.approx(2.0)


# --- recovery on O-C points ---------------------------------------------------


def test_recovers_circular_orbit_and_secular_rate():
    t, oc, err, _ = _oc_points()
    res = cup.ltte_search(t, oc=oc, errors=err, period=0.4)
    best = res.best
    assert best.outer_period == pytest.approx(1300.0, abs=4 * best.outer_period_err)
    assert best.amplitude == pytest.approx(0.003, abs=4 * best.amplitude_err)
    assert best.pdot == pytest.approx(3e-8, abs=4 * best.pdot_err)
    assert res.delta_bic > 100.0
    assert res.f_test_p < 1e-10
    assert not best.at_grid_edge
    assert best.n_cycles == pytest.approx(res.baseline / best.outer_period)
    assert 0.3 < best.chi2_reduced < 2.0
    # Units and derived quantities are consistent with each other.
    assert best.amplitude_s == pytest.approx(best.amplitude * DAY)
    assert best.outer_period_yr == pytest.approx(best.outer_period / 365.25)
    assert best.pdot_s_per_yr == pytest.approx(best.pdot * DAY * 365.25)
    assert best.pdot_d_per_yr == pytest.approx(best.pdot * 365.25)
    assert best.degree == 2 and res.degree == 2
    # The spectrum peaks where the fit landed.
    j = int(np.argmax(res.delta_chi2))
    assert res.outer_period[j] == pytest.approx(best.outer_period, rel=0.05)


def test_recovers_eccentric_elements():
    t, oc, err, truth = _oc_points(
        seed=2, eccentricity=0.5, omega=1.0, t_peri=200.0, sigma=3e-4
    )
    res = cup.ltte_search(t, oc=oc, errors=err, period=0.4, eccentric=True)
    best = res.best
    assert best.eccentric
    assert best.eccentricity == pytest.approx(0.5, abs=4 * best.eccentricity_err)
    assert best.omega == pytest.approx(1.0, abs=4 * best.omega_err)
    assert best.outer_period == pytest.approx(1300.0, abs=4 * best.outer_period_err)
    assert best.t_periastron - best.t_ref == pytest.approx(
        200.0, abs=4 * best.t_periastron_err
    )
    # The curve, not the parameters, is the invariant: close everywhere.
    assert np.max(np.abs(best.predict(t) - truth)) < 3 * 3e-4
    assert best.n_params == 8
    # The circular model cannot follow it as well.
    circ = cup.ltte_search(t, oc=oc, errors=err, period=0.4)
    assert circ.best.chi2 > best.chi2
    assert not circ.best.eccentric
    assert np.isnan(circ.best.eccentricity_err)


def test_null_parabola_is_not_promoted():
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0.0, 4000.0, 30)) + 2450000.0
    dt = t - t.mean()
    oc = 0.5 * (2e-8 / 0.5) * dt**2 + rng.normal(0.0, 5e-4, t.size)
    res = cup.ltte_search(t, oc=oc, errors=np.full(t.size, 5e-4), period=0.5)
    assert res.delta_bic < 5.0
    assert res.best.amplitude < 4 * res.best.amplitude_err + 5e-4
    # The parabola-only rate is untouched by a cycle that is not there.
    assert res.null.pdot == pytest.approx(2e-8, abs=4 * res.null.pdot_err)
    assert abs(res.pdot_change) < 4 * res.best.pdot_err


def test_event_times_input_assigns_cycles():
    rng = np.random.default_rng(3)
    period = 0.4
    cycles = np.arange(0.0, 12000.0, 37.0)
    mid = period * cycles.mean()
    times = 2450000.0 + period * cycles + ltte_delay_reference(
        period * cycles - mid, 900.0, 0.002, 0.3
    )
    times = times + rng.normal(0.0, 3e-4, cycles.size)
    res = cup.ltte_search(times, period=period, errors=np.full(times.size, 3e-4))
    assert res.meta["input"] == "times"
    assert res.best.outer_period == pytest.approx(900.0, rel=0.02)
    assert res.best.amplitude == pytest.approx(0.002, rel=0.1)
    # Without errors an RMS scale is used and the answer does not move.
    res2 = cup.ltte_search(times, period=period)
    assert res2.best.outer_period == pytest.approx(res.best.outer_period, rel=0.01)


def test_fixed_outer_period_fits_amplitude_only():
    t, oc, err, _ = _oc_points(seed=4)
    res = cup.ltte_search(t, oc=oc, errors=err, period=0.4, outer_period=1300.0)
    assert res.best.outer_period == 1300.0
    assert np.isnan(res.best.outer_period_err)
    assert res.meta["fixed_outer_period"] is True
    assert res.outer_period.size == 1
    assert res.best.amplitude == pytest.approx(0.003, abs=4 * res.best.amplitude_err)
    assert res.best.n_params == 5  # amplitude, phase, three polynomial terms


def test_degree_one_has_no_secular_rate():
    t, oc, err, _ = _oc_points(seed=6, pdot=0.0)
    res = cup.ltte_search(t, oc=oc, errors=err, period=0.4, degree=1)
    assert res.best.pdot == 0.0 and res.best.pdot_err == 0.0
    assert res.best.degree == 1
    assert res.best.outer_period == pytest.approx(1300.0, rel=0.03)


def test_mass_function_and_minimum_third_mass():
    # a sin i = 1 au (a delay semi-amplitude of one light-au) and P2 = 1 yr
    # give f(m) = 1 Msun by construction of the solar-unit Kepler law.
    t, oc, err, _ = _oc_points(seed=8)
    res = cup.ltte_search(t, oc=oc, errors=err, period=0.4)
    best = res.best
    scaled = best.__class__(
        **{
            **best.__dict__,
            "amplitude": 499.004783836 / DAY,
            "outer_period": 365.25,
        }
    )
    assert scaled.a_sini_au == pytest.approx(1.0)
    assert scaled.mass_function == pytest.approx(1.0)
    m3 = scaled.minimum_third_mass(2.0)
    assert m3**3 / (2.0 + m3) ** 2 == pytest.approx(1.0, rel=1e-8)
    assert np.isnan(scaled.minimum_third_mass(-1.0))


def test_period_at_wobbles_about_the_reference():
    t, oc, err, _ = _oc_points(seed=9, pdot=0.0)
    res = cup.ltte_search(t, oc=oc, errors=err, period=0.4)
    grid = np.linspace(t.min(), t.max(), 400)
    p_app = res.best.period_at(grid)
    # Peak-to-peak apparent period change = 2 * (2 pi A / P2) * P.
    expected = 2.0 * (2.0 * np.pi * 0.003 / 1300.0) * 0.4
    assert p_app.max() - p_app.min() == pytest.approx(expected, rel=0.15)
    assert np.mean(p_app) == pytest.approx(0.4, rel=1e-3)


# --- through phase_track --------------------------------------------------------


def test_track_ltte_removes_the_spurious_drift():
    t, mag, err, truth = synthetic_ltte(seed=3, outer_period=800.0, amplitude=0.004)
    settings = cup.PdotSettings(n_segments=30)
    track = cup.phase_track((t, mag, err), period=truth["period"], settings=settings)
    res = track.ltte_search()
    assert res.meta["input"] == "phase_track"
    best = res.best
    assert best.outer_period == pytest.approx(800.0, abs=4 * best.outer_period_err)
    assert best.amplitude == pytest.approx(0.004, abs=4 * best.amplitude_err)
    assert res.delta_bic > 50.0
    # No secular rate was planted: the joint fit says so, whatever the parabola
    # alone made of the cycle it was averaging over.
    assert abs(best.pdot) < 4 * best.pdot_err_scaled
    # The recovered cycle matches the planted delay curve, not just its numbers.
    dt = track.t_mid - truth["t_ref"]
    planted = ltte_delay_reference(
        dt, 800.0, 0.004, truth["omega"], 0.0, truth["t_periastron"] - truth["t_ref"]
    )
    resid = best.ltte_part(track.t_mid) - planted
    assert np.std(resid) < 0.35 * 0.004


def test_track_convenience_rejects_overrides():
    t, mag, err, truth = synthetic_ltte(seed=4, n=1500, span=2000.0)
    track = cup.phase_track((t, mag, err), period=truth["period"])
    with pytest.raises(ValueError, match="taken from the PhaseTrackResult"):
        cup.ltte_search(track, oc=track.delay)


# --- guards and plumbing ------------------------------------------------------


def test_guards():
    t, oc, err, _ = _oc_points(n=6)
    with pytest.raises(InsufficientDataError, match="at least"):
        cup.ltte_search(t, oc=oc, errors=err, period=0.4)
    t, oc, err, _ = _oc_points()
    with pytest.raises(ValueError, match="period="):
        cup.ltte_search(t, oc=oc, errors=err)
    with pytest.raises(ValueError, match="length"):
        cup.ltte_search(t, oc=oc[:-1], errors=err, period=0.4)
    with pytest.raises(ValueError, match="degree"):
        cup.ltte_search(t, oc=oc, errors=err, period=0.4, degree=5)
    with pytest.raises(ValueError, match="outer_period"):
        cup.ltte_search(t, oc=oc, errors=err, period=0.4, outer_period=-3.0)
    with pytest.raises(InsufficientDataError, match="resolvable"):
        cup.ltte_search(
            t, oc=oc, errors=err, period=0.4,
            min_outer_period=6000.0, max_outer_period=5000.0,
        )


def _kepler_bisection_delay(dt, outer_period, amplitude, omega, e):
    """Irwin's delay from Kepler's equation solved by plain bisection.

    Slow and unconditionally convergent for ``e < 1``, so it is a reference the
    Newton solver can be checked against right up to the parabolic limit.
    """
    m = np.mod(2.0 * np.pi * np.asarray(dt, dtype=np.float64) / outer_period, 2 * np.pi)
    lo, hi = np.zeros_like(m), np.full_like(m, 2.0 * np.pi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = mid - e * np.sin(mid) - m < 0.0
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    ecc = 0.5 * (lo + hi)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(0.5 * ecc), np.sqrt(1.0 - e) * np.cos(0.5 * ecc)
    )
    return amplitude * (
        (1.0 - e * e) / (1.0 + e * np.cos(nu)) * np.sin(nu + omega) + e * np.sin(omega)
    )


@pytest.mark.parametrize("e", [0.5, 0.99, 0.994, 0.999, 0.999999])
def test_kepler_solver_converges_up_to_the_parabolic_limit(e):
    dt = np.linspace(0.0, 100.0, 20001)
    ours = ltte_delay(dt, 100.0, 1.0, 0.7, e, 0.0)
    ref = _kepler_bisection_delay(dt, 100.0, 1.0, 0.7, e)
    assert np.max(np.abs(ours - ref)) < 1e-6


def test_delay_rejects_unbound_orbits_and_handles_empty_input():
    with pytest.raises(ValueError, match="eccentricity"):
        ltte_delay(np.linspace(0, 10, 5), 100.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="eccentricity"):
        ltte_delay(np.linspace(0, 10, 5), 100.0, 1.0, 0.0, -0.1)
    with pytest.raises(ValueError, match="outer_period"):
        ltte_delay(np.linspace(0, 10, 5), 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="outer_period"):
        ltte_delay(np.linspace(0, 10, 5), np.nan, 1.0, 0.0, 0.3)
    assert ltte_delay(np.array([]), 700.0, 0.01, 1.1, 0.3).shape == (0,)
    assert ltte_delay(np.array([]), 700.0, 0.01, 1.1).shape == (0,)
    t, oc, err, _ = _oc_points(
        seed=2, eccentricity=0.5, omega=1.0, t_peri=200.0, sigma=3e-4
    )
    fit = cup.ltte_search(t, oc=oc, errors=err, period=0.4, eccentric=True).best
    assert fit.predict(np.array([])).shape == (0,)
    assert fit.ltte_part(np.array([])).shape == (0,)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"period": np.nan}, ValueError, "period="),
        ({"period": np.inf}, ValueError, "period="),
        ({"period": 0.4, "outer_period": np.nan}, ValueError, "outer_period"),
        ({"period": 0.4, "outer_period": np.inf}, ValueError, "outer_period"),
        ({"period": 0.4, "max_outer_period": np.inf}, InsufficientDataError, "resol"),
        ({"period": 0.4, "min_outer_period": np.nan}, InsufficientDataError, "resol"),
    ],
)
def test_non_finite_scalars_are_refused(kwargs, error, match):
    t, oc, err, _ = _oc_points()
    with pytest.raises(error, match=match):
        cup.ltte_search(t, oc=oc, errors=err, **kwargs)


def test_empty_or_all_nan_event_times_are_insufficient():
    with pytest.raises(InsufficientDataError):
        cup.ltte_search(np.array([]), period=0.4)
    with pytest.raises(InsufficientDataError):
        cup.ltte_search(np.full(12, np.nan), period=0.4)


def test_minimum_third_mass_edges():
    t, oc, err, _ = _oc_points(seed=8)
    best = cup.ltte_search(t, oc=oc, errors=err, period=0.4).best
    f = best.mass_function
    # m^3/(0 + m)^2 = f: a massless inner binary needs a companion of exactly f.
    assert best.minimum_third_mass(0.0) == pytest.approx(f, rel=1e-9)
    assert np.isnan(best.minimum_third_mass(np.nan))
    m3 = best.minimum_third_mass(1.5)
    assert m3**3 / (1.5 + m3) ** 2 == pytest.approx(f, rel=1e-8)


def test_masked_timings_are_dropped_not_fitted():
    t, oc, err, _ = _oc_points()
    bad = oc.copy()
    bad[17] += 0.5  # a wild timing that the mask says to ignore
    masked = np.ma.masked_array(bad, mask=np.zeros(t.size, dtype=bool))
    masked[17] = np.ma.masked
    res = cup.ltte_search(t, oc=masked, errors=err, period=0.4)
    keep = np.arange(t.size) != 17
    ref = cup.ltte_search(t[keep], oc=oc[keep], errors=err[keep], period=0.4)
    assert res.n_used == t.size - 1
    assert res.best.outer_period == pytest.approx(ref.best.outer_period, rel=1e-9)
    assert res.best.amplitude == pytest.approx(ref.best.amplitude, rel=1e-9)
    # A masked error drops its point too.
    err_m = np.ma.masked_array(err, mask=~keep)
    assert cup.ltte_search(t, oc=bad, errors=err_m, period=0.4).n_used == t.size - 1


def test_multi_column_input_is_refused():
    t, oc, err, _ = _oc_points()
    with pytest.raises(ValueError, match="1-D"):
        cup.ltte_search((t, oc, err), period=0.4)
    with pytest.raises(ValueError, match="1-D"):
        cup.ltte_search(np.column_stack([t, oc, err]), period=0.4)
    # A single column is still a column.
    res = cup.ltte_search(t[:, None], oc=oc[:, None], errors=err, period=0.4)
    assert res.n_used == t.size


def test_repeated_epochs_do_not_collapse_the_scan_floor():
    """The same eclipse timed in three filters is one epoch, not three."""
    rng = np.random.default_rng(21)
    cycles = np.round(np.arange(16) * 365.25 / 0.4 + rng.uniform(-50, 50, 16))
    once = 2450000.0 + 0.4 * cycles
    oc_once = rng.normal(0.0, 3e-4, once.size)
    thrice = np.repeat(once, 3)
    oc_thrice = np.repeat(oc_once, 3) + rng.normal(0.0, 1e-5, thrice.size)
    err = np.full(thrice.size, 3e-4)
    r1 = cup.ltte_search(once, oc=oc_once, errors=err[: once.size], period=0.4)
    r3 = cup.ltte_search(thrice, oc=oc_thrice, errors=err, period=0.4)
    assert r3.meta["min_outer_period"] == pytest.approx(r1.meta["min_outer_period"])
    assert r3.meta["n_trials"] == r1.meta["n_trials"]
    assert r3.meta["n_epochs"] == once.size and r3.n_used == thrice.size


def test_seasonal_clusters_keep_the_floor_at_the_season_spacing():
    """Three eclipses a season are not three independent samples of the cycle."""
    rng = np.random.default_rng(17000)
    centres = 2450000.0 + 365.25 * np.arange(16) + rng.uniform(-40, 40, 16)
    raw = (centres[:, None] + np.sort(rng.uniform(0, 10, (16, 3)), axis=1)).ravel()
    t = np.sort(2450000.0 + 0.4 * np.round((raw - 2450000.0) / 0.4))
    oc = 0.001 * np.sin(2 * np.pi * (t - t.mean()) / 2100.0 + 0.7)
    oc = oc + rng.normal(0.0, np.hypot(3e-4, 6e-4), t.size)
    res = cup.ltte_search(t, oc=oc, errors=np.full(t.size, 3e-4), period=0.4)
    # The within-season spacing (days) used to set a 16-day floor, and the
    # scan then returned a 167-day junk cycle on this very seed.
    assert res.meta["min_outer_period"] > 300.0
    assert res.best.outer_period == pytest.approx(2100.0, rel=0.1)


def test_two_distinct_epochs_are_insufficient():
    t = np.repeat([2450000.0, 2452000.0], 6) + np.tile(np.arange(6) * 1e-3, 2)
    oc = np.random.default_rng(0).normal(0.0, 1e-4, t.size)
    with pytest.raises(InsufficientDataError, match="distinct"):
        cup.ltte_search(t, oc=oc, errors=np.full(t.size, 1e-4), period=0.4)


def test_estimated_errors_do_not_cap_the_detection():
    """Without errors=, a 150-sigma cycle must still be a strong detection.

    The scale used to come from the polynomial-only residuals, which contain the
    cycle itself, so the null chi-squared was n by construction and delta_bic
    could never exceed n - 3 ln n (6.9 for 15 points), whatever the signal.
    """
    rng = np.random.default_rng(7)
    n = 15
    t = np.sort(rng.uniform(0.0, 5000.0, n)) + 2450000.0
    dt = t - t.mean()
    oc = 0.003 * np.sin(2 * np.pi * dt / 1300.0 + 0.4) + rng.normal(0.0, 2e-5, n)
    est = cup.ltte_search(t, oc=oc, period=0.4)
    given = cup.ltte_search(t, oc=oc, errors=np.full(n, 2e-5), period=0.4)
    assert est.meta["errors"] == "estimated" and given.meta["errors"] == "given"
    assert est.delta_bic > 10.0
    assert est.delta_bic == pytest.approx(est.null.bic - est.best.bic)
    assert est.best.chi2_reduced == pytest.approx(1.0)
    assert est.best.outer_period == pytest.approx(given.best.outer_period, rel=1e-6)
    ratio = est.best.amplitude_err / given.best.amplitude_err
    assert 1.0 / 3.0 < ratio < 3.0
    assert 1.0 / 3.0 < est.best.pdot_err / given.best.pdot_err < 3.0
    # The periodogram is on the same (estimated) scale as the fits.
    j = int(np.argmax(est.delta_chi2))
    assert est.delta_chi2[j] <= est.null.chi2 + 1e-9


def test_estimated_errors_keep_white_noise_quiet():
    hits = 0
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        t = np.sort(rng.uniform(0.0, 5000.0, 25)) + 2450000.0
        res = cup.ltte_search(t, oc=rng.normal(0.0, 3e-4, t.size), period=0.4)
        hits += res.delta_bic > 10.0
    assert hits <= 2


def _scan_reference(dt, scaled, oc, w, degree, freqs):
    """Per-trial weighted lstsq of polynomial + sinusoid (the scan's definition)."""
    base = np.column_stack([scaled**j for j in range(degree + 1)])
    sw = np.sqrt(w)
    chi2 = np.empty(freqs.size)
    coefs = np.empty((freqs.size, degree + 3))
    for i, f in enumerate(freqs):
        arg = 2.0 * np.pi * f * dt
        x = np.column_stack([base, np.cos(arg), np.sin(arg)])
        coef = np.linalg.lstsq(x * sw[:, None], oc * sw, rcond=None)[0]
        resid = oc - x @ coef
        chi2[i] = np.dot(w, resid * resid)
        coefs[i] = coef
    return chi2, coefs


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_scan_matches_per_trial_least_squares(degree):
    from cuperiod.pdot.ltte import _FREQ_CHUNK, _scan

    rng = np.random.default_rng(degree)
    t = np.sort(rng.uniform(0.0, 4000.0, 60))
    dt = t - t.mean()
    scaled = dt / (0.5 * (t.max() - t.min()))
    oc = 0.002 * np.sin(2 * np.pi * dt / 900.0) + rng.normal(0, 3e-4, t.size)
    w = 1.0 / rng.uniform(2e-4, 6e-4, t.size) ** 2
    freqs = np.linspace(1.0 / 4000.0, 1.0 / 200.0, _FREQ_CHUNK + 37)
    chi2, cos_sin, poly = _scan(dt, scaled, oc, w, degree, freqs)
    ref_chi2, ref_coef = _scan_reference(dt, scaled, oc, w, degree, freqs)
    assert np.allclose(chi2, ref_chi2, rtol=1e-9)
    assert np.allclose(cos_sin, ref_coef[:, -2:], rtol=1e-6, atol=1e-12)
    assert np.allclose(poly, ref_coef[:, : degree + 1], rtol=1e-6, atol=1e-12)
    # A zero-frequency trial is degenerate with the constant: it falls back to
    # least squares and returns the polynomial-only chi-squared.
    zero = np.array([0.0, 1.0 / 900.0])
    chi2_0, _, _ = _scan(dt, scaled, oc, w, degree, zero)
    base = np.column_stack([scaled**j for j in range(degree + 1)])
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(base * sw[:, None], oc * sw, rcond=None)[0]
    poly_chi2 = float(np.dot(w, (oc - base @ coef) ** 2))
    assert chi2_0[0] == pytest.approx(poly_chi2, rel=1e-9)


def test_grid_edge_is_flagged():
    t, oc, err, _ = _oc_points(seed=11, outer_period=1300.0)
    # A scan that stops just short of the true cycle piles up at its long end.
    res = cup.ltte_search(
        t, oc=oc, errors=err, period=0.4, min_outer_period=700.0,
        max_outer_period=1200.0,
    )
    assert res.best.at_grid_edge
    assert res.best.outer_period == pytest.approx(1200.0, rel=0.02)


def test_settings_drive_the_defaults():
    t, oc, err, _ = _oc_points(seed=12)
    settings = cup.PdotSettings(ltte_degree=1, ltte_samples_per_peak=4)
    res = cup.ltte_search(t, oc=oc, errors=err, period=0.4, settings=settings)
    assert res.degree == 1
    assert res.meta["samples_per_peak"] == 4
    assert cup.PdotSettings().ltte_min_points == 8


def test_to_dict_and_batch_row():
    t, oc, err, _ = _oc_points(seed=13)
    res = cup.ltte_search(t, oc=oc, errors=err, period=0.4)
    payload = json.dumps(res.to_dict(include_spectrum=True))
    assert "delta_chi2" in payload and "mass_function" in payload
    assert "delta_chi2" not in json.dumps(res.to_dict())
    rows = pdot_to_rows("star", ltte=res)
    row = rows["summary"][0]
    assert row["ltte_outer_period_yr"] == pytest.approx(res.best.outer_period_yr)
    assert row["ltte_delta_bic"] == pytest.approx(res.delta_bic)
    assert row["ltte_pdot_change"] == pytest.approx(res.pdot_change)
    assert row["ltte_at_grid_edge"] is False
    empty = pdot_to_rows("bare")["summary"][0]
    assert np.isnan(empty["ltte_outer_period_yr"])
    assert set(empty) == set(row)
