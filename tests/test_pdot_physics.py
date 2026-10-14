"""Tests for the physical-interpretation helpers (`cuperiod.pdot.physics`).

The relations of record are the Jeans-mode mass-loss law ``(dP/dt)/P = 2 mdot/M``
and Kwee's (1958) conservative-transfer law ``(dP/dt)/P = 3 mdot (1/M_d - 1/M_a)``;
the sensitivity floor is pinned against a Monte Carlo of the quadratic
ephemeris fit it describes and against its uniform-sampling closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.pdot.physics import pdot_units

# --- unit conversions ---------------------------------------------------------


def test_survey_unit_matches_the_feedback_scaling():
    # 1e-6 d/yr is ~3e-9 s/s: the typical eclipsing-binary survey detection.
    assert cup.convert_pdot(1e-6, "d/yr", "s/s") == pytest.approx(2.738e-9, rel=1e-3)
    # WASP-12b's -29.35 ms/yr is -9.3e-10 (the fixture the tracker tests pin).
    assert cup.convert_pdot(-29.35, "ms/yr", "d/d") == pytest.approx(-9.3e-10, rel=1e-3)


def test_round_trips_and_per_period_units():
    for unit in pdot_units():
        back = cup.convert_pdot(
            cup.convert_pdot(3.3e-9, "s/s", unit, period=0.7), unit, "s/s", period=0.7
        )
        assert back == pytest.approx(3.3e-9, rel=1e-12)
    # Per-cycle: dP/dE = P * dP/dt; fractional: (dP/dt)/P per year.
    assert cup.convert_pdot(1e-8, "d/d", "d/cycle", period=0.5) == pytest.approx(5e-9)
    assert cup.convert_pdot(1e-8, "d/d", "1/yr", period=0.5) == pytest.approx(
        1e-8 / 0.5 * 365.25
    )
    with pytest.raises(ValueError, match="needs period"):
        cup.convert_pdot(1.0, "d/d", "d/cycle")
    with pytest.raises(ValueError, match="unknown"):
        cup.convert_pdot(1.0, "d/d", "furlongs")
    arr = cup.convert_pdot(np.array([1.0, 2.0]), "s/yr", "ms/yr")
    assert np.allclose(arr, [1000.0, 2000.0])


# --- mass loss and transfer ---------------------------------------------------


def test_mass_loss_reproduces_the_feedback_scaling():
    # dP/dt = 2e-6 (P/yr)(mdot/1e-6 Msun/yr)(Msun/M)
    pdot = cup.pdot_from_mass_loss(1e-6, 365.25, 1.0)
    assert pdot == pytest.approx(2e-6)
    assert cup.mass_loss_rate(pdot, 365.25, 1.0) == pytest.approx(1e-6)
    # Linear in each factor.
    assert cup.pdot_from_mass_loss(2e-6, 365.25, 1.0) == pytest.approx(4e-6)
    assert cup.pdot_from_mass_loss(1e-6, 365.25 / 2, 1.0) == pytest.approx(1e-6)
    assert cup.pdot_from_mass_loss(1e-6, 365.25, 2.0) == pytest.approx(1e-6)


def test_mass_loss_sign_is_physical():
    # A shortening period cannot come from an isotropic wind: negative "loss".
    assert cup.mass_loss_rate(-1e-9, 0.5, 1.5) < 0.0
    assert cup.mass_loss_rate(1e-9, 0.5, 1.5) > 0.0


def test_conservative_transfer_signs_and_limits():
    p, md, ma = 0.4, 1.2, 0.6
    # Transfer from the more massive star shortens the period ...
    assert cup.pdot_from_mass_transfer(1e-7, p, md, ma) < 0.0
    # ... from the less massive one lengthens it ...
    assert cup.pdot_from_mass_transfer(1e-7, p, ma, md) > 0.0
    # ... and equal masses leave it unchanged (no rate implied).
    assert cup.pdot_from_mass_transfer(1e-7, p, 1.0, 1.0) == 0.0
    assert np.isnan(cup.mass_transfer_rate(1e-8, p, 1.0, 1.0))
    # Round trip and the Kwee form 3 (mdot/M_d)(1 - q).
    pdot = cup.pdot_from_mass_transfer(2e-7, p, md, ma)
    assert cup.mass_transfer_rate(pdot, p, md, ma) == pytest.approx(2e-7)
    q = md / ma
    expected = 3.0 * 2e-7 / md * (1.0 - q) * p / 365.25
    assert pdot == pytest.approx(expected)
    # A wrong-sign rate for the assumed donor comes back negative.
    assert cup.mass_transfer_rate(1e-8, p, md, ma) < 0.0


def test_dominant_mass_limit_matches_mass_loss_within_factor():
    # When the accretor dominates, transfer looks like loss at 3/2 the rate.
    p = 1.0
    loss = cup.pdot_from_mass_loss(1e-7, p, 10.0)
    transfer = cup.pdot_from_mass_transfer(1e-7, p, 1e-3, 10.0)
    # (3/2) (M / M_d) (1 - q) for a donor of 1e-3 Msun and a 10 Msun accretor.
    expected = 1.5 * (10.0 / 1e-3) * (1.0 - 1e-4)
    assert transfer / loss == pytest.approx(expected, rel=1e-6)


def test_mass_relations_reject_bad_masses():
    with pytest.raises(ValueError, match="positive"):
        cup.mass_loss_rate(1e-9, 0.5, -1.0)
    with pytest.raises(ValueError, match="positive"):
        cup.mass_transfer_rate(1e-9, 0.0, 1.0, 1.0)


# --- sensitivity ---------------------------------------------------------------


def test_floor_matches_uniform_closed_form():
    n, baseline, period, sigma_t = 400, 5000.0, 0.5, 0.002
    s = cup.detectable_pdot(period, sigma_t, n_timings=n, baseline=baseline)
    closed = 4.0 * np.sqrt(45.0) * period * sigma_t / (baseline**2 * np.sqrt(n))
    assert s.sigma_pdot == pytest.approx(closed, rel=0.02)
    assert s.n_timings == n
    assert s.baseline == pytest.approx(baseline)
    assert s.threshold(3.0) == pytest.approx(3.0 * s.sigma_pdot)
    assert s.sigma_pdot_d_per_yr == pytest.approx(s.sigma_pdot * 365.25)
    assert s.sigma_pdot_s_per_yr == pytest.approx(s.sigma_pdot * 86400 * 365.25)


def test_floor_is_the_monte_carlo_scatter_of_the_quadratic_fit():
    rng = np.random.default_rng(7)
    times = np.sort(rng.uniform(0.0, 3000.0, 60))
    period, sigma_t = 0.4, 0.001
    s = cup.detectable_pdot(period, sigma_t, times=times)
    dt = times - times.mean()
    rates = []
    for _ in range(400):
        oc = rng.normal(0.0, sigma_t, times.size)
        c2 = np.polyfit(dt, oc, 2)[0]
        rates.append(2.0 * period * c2)
    assert np.std(rates) == pytest.approx(s.sigma_pdot, rel=0.15)


def test_amplitude_floor_grows_toward_the_baseline():
    s = cup.detectable_pdot(0.4, 0.001, n_timings=15, baseline=15 * 365.25)
    short, long = s.sigma_amplitude([3 * 365.25, 14 * 365.25])
    # A cycle comparable to the baseline is partly absorbed by the parabola.
    assert long > short
    # A well-sampled short cycle approaches sigma_t * sqrt(2/n).
    assert short == pytest.approx(0.001 * np.sqrt(2.0 / 15), rel=0.25)
    assert np.isnan(s.sigma_amplitude(-1.0))


def test_amplitude_floor_is_infinite_where_the_cycle_is_unresolvable():
    """A commensurate or absurdly long outer period has no finite floor.

    Fifteen annual timings cannot see a one-year (or half-year) cycle at all:
    its cosine and sine are constant over the sampling. The honest floor there
    is infinite, not a LinAlgError that aborts a whole sweep -- and an outer
    period far beyond the baseline, where the inverse used to come back as
    garbage, NaN or 0.0, is unresolvable too.
    """
    s = cup.detectable_pdot(0.4, 0.001, n_timings=15, baseline=14 * 365.25)
    got = s.sigma_amplitude([365.25, 1000.0])
    assert np.isinf(got[0]) and np.isfinite(got[1]) and got[1] > 0.0
    sweep = s.sigma_amplitude(np.arange(1, 15) * 365.25)  # must not raise
    assert np.isinf(sweep[0]) and np.all(np.isfinite(sweep[2:]))
    assert np.isinf(s.sigma_amplitude(1e12))
    s3 = cup.detectable_pdot(0.4, 0.001, n_timings=15, baseline=14 * 365.25, degree=3)
    assert np.isinf(s3.sigma_amplitude(1e12))
    assert np.isnan(s.sigma_amplitude(np.nan))
    # Ordinary values are untouched by the reconditioning (pinned from the
    # previous unscaled inverse, and the user-guide example).
    assert s.sigma_amplitude(6 * 365.25) == pytest.approx(3.87631e-4, rel=1e-5)
    s15 = cup.detectable_pdot(0.4, 0.001, n_timings=15, baseline=15 * 365.25)
    assert s15.sigma_amplitude(6 * 365.25) * 86400.0 == pytest.approx(32.753, abs=1e-3)


def test_too_few_distinct_epochs_raise():
    """Repeated timings of two eclipses cannot constrain a parabola."""
    with pytest.raises(ValueError, match="distinct epochs"):
        cup.detectable_pdot(0.4, 1e-3, times=np.repeat([0.0, 100.0], 4))
    with pytest.raises(ValueError, match="distinct epochs"):
        cup.detectable_pdot(0.4, 1e-3, times=np.repeat([2450000.3, 2455000.7], 4))
    # Two distinct epochs are enough for a straight line.
    s = cup.detectable_pdot(0.4, 1e-3, times=np.repeat([0.0, 100.0], 4), degree=1)
    assert np.isfinite(s.sigma_period)


def test_nan_epoch_drops_its_own_timing_error():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 3000.0, 20)) + 2450000.0
    sig = rng.uniform(5e-4, 2e-3, 20)
    t_nan = t.copy()
    t_nan[3] = np.nan
    s = cup.detectable_pdot(0.4, sig, times=t_nan)
    keep = np.isfinite(t_nan)
    ref = cup.detectable_pdot(0.4, sig[keep], times=t[keep])
    assert s.n_timings == 19
    assert s.sigma_pdot == pytest.approx(ref.sigma_pdot, rel=1e-12)
    # A NaN error drops its epoch the same way.
    sig_nan = sig.copy()
    sig_nan[3] = np.nan
    assert cup.detectable_pdot(0.4, sig_nan, times=t).n_timings == 19
    # 2-D times with matching 2-D errors are read point by point.
    s2 = cup.detectable_pdot(0.4, sig.reshape(4, 5), times=t.reshape(4, 5))
    assert s2.n_timings == 20
    with pytest.raises(ValueError, match="timing_error"):
        cup.detectable_pdot(0.4, sig[:-1], times=t)
    with pytest.raises(ValueError, match="positive"):
        cup.detectable_pdot(0.4, np.nan, times=t)


def test_sensitivity_guards_and_dict():
    with pytest.raises(ValueError, match="times= or both"):
        cup.detectable_pdot(0.4, 0.001)
    with pytest.raises(ValueError, match="at least"):
        cup.detectable_pdot(0.4, 0.001, n_timings=3, baseline=100.0)
    with pytest.raises(ValueError, match="positive"):
        cup.detectable_pdot(0.4, -0.001, n_timings=10, baseline=100.0)
    with pytest.raises(ValueError, match="degree"):
        cup.detectable_pdot(0.4, 0.001, n_timings=10, baseline=100.0, degree=7)
    s = cup.detectable_pdot(0.4, 0.001, n_timings=10, baseline=100.0, degree=4)
    d = s.to_dict()
    assert d["degree"] == 4 and d["sigma_pdot"] > 0.0
    # A free quartic (even, so correlated with the parabola) costs rate
    # sensitivity; a cubic would not on symmetric sampling.
    s2 = cup.detectable_pdot(0.4, 0.001, n_timings=10, baseline=100.0)
    assert s.sigma_pdot > s2.sigma_pdot
