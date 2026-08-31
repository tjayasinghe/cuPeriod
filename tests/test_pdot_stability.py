"""Tests for the Eddington-Plakidis fluctuation test (`cuperiod.pdot.stability`).

The planted model is the 1929 one exactly: per-cycle period steps drawn from
N(0, sigma_eps) accumulate a random walk in the times of maximum, on top of
independent N(0, sigma_alpha) timing errors. The estimator works in the
classical mean-absolute-error convention, so the recovered ``epsilon_sigma`` /
``alpha_sigma`` properties are what compare against the planted Gaussian
sigmas.
"""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.core.errors import InsufficientDataError
from cuperiod.pdot.stability import eddington_plakidis


def _random_walk_times(
    *,
    n: int = 400,
    period: float = 0.5,
    sigma_eps: float = 0.002,
    sigma_alpha: float = 0.001,
    pdot: float = 0.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    periods = period + rng.normal(0.0, sigma_eps, n)
    times = 2450000.0 + np.concatenate([[0.0], np.cumsum(periods)])
    if pdot != 0.0:
        e = np.arange(times.size, dtype=np.float64)
        times = times + 0.5 * pdot * period * e * e
    return times + rng.normal(0.0, sigma_alpha, times.size)


# --- recovery ----------------------------------------------------------------


def test_recovers_planted_random_walk():
    times = _random_walk_times(seed=1)
    ep = eddington_plakidis(times, period=0.5)
    assert ep.slope_significance > 5.0
    assert ep.epsilon_sigma == pytest.approx(0.002, rel=0.35)
    assert ep.alpha_sigma == pytest.approx(0.001, rel=0.5)
    assert ep.lags.size >= 3
    assert np.all(ep.n_pairs >= 5)


def test_clean_ephemeris_shows_no_wander():
    rng = np.random.default_rng(2)
    e = np.arange(400.0)
    times = 2450000.0 + 0.5 * e + rng.normal(0.0, 0.001, e.size)
    ep = eddington_plakidis(times, period=0.5)
    assert ep.slope_significance < 3.0
    assert ep.alpha_sigma == pytest.approx(0.001, rel=0.4)


def test_secular_pdot_is_not_mistaken_for_wander():
    # A real quadratic drift, detrended away by default (detrend=2).
    times = _random_walk_times(sigma_eps=0.0, pdot=2e-6, seed=3)
    ep = eddington_plakidis(times, period=0.5)
    assert ep.slope_significance < 3.0
    # Without the quadratic detrend the drift leaks into the lag profile.
    ep_raw = eddington_plakidis(times, period=0.5, detrend=1)
    assert ep_raw.slope_significance > ep.slope_significance


def test_gappy_cycle_coverage():
    rng = np.random.default_rng(4)
    steps = rng.integers(1, 4, 250)
    cycles = np.concatenate([[0], np.cumsum(steps)]).astype(np.float64)
    periods_walk = rng.normal(0.0, 0.002, cycles.size)
    walk = np.cumsum(periods_walk)
    times = 2450000.0 + 0.5 * cycles + walk + rng.normal(0.0, 0.0005, cycles.size)
    ep = eddington_plakidis(times, period=0.5)
    assert ep.slope_significance > 3.0
    assert ep.epsilon > 0.0


# --- guards and plumbing ------------------------------------------------------


def test_guards():
    with pytest.raises(InsufficientDataError):
        eddington_plakidis(np.arange(5.0), period=1.0)
    with pytest.raises(ValueError, match="positive"):
        eddington_plakidis(np.arange(20.0), period=-1.0)
    with pytest.raises(ValueError, match="detrend"):
        eddington_plakidis(np.arange(20.0), period=1.0, detrend=7)


def test_max_lag_and_to_dict():
    times = _random_walk_times(n=200, seed=5)
    ep = eddington_plakidis(times, period=0.5, max_lag=20)
    assert ep.lags.max() <= 20
    d = ep.to_dict()
    assert d["epsilon"] == pytest.approx(ep.epsilon)
    assert len(d["lags"]) == ep.lags.size
