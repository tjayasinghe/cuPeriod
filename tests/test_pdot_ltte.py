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
