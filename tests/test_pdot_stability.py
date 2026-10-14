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


@pytest.mark.parametrize("bad", [{"period": np.nan}, {"period": np.inf}])
def test_non_finite_period_is_refused(bad):
    with pytest.raises(ValueError, match="positive"):
        eddington_plakidis(np.arange(20.0), **bad)


def test_non_finite_epoch_is_refused():
    with pytest.raises(ValueError, match="t0"):
        eddington_plakidis(np.arange(20.0), period=1.0, t0=np.nan)


# --- an exactly periodic series ----------------------------------------------


@pytest.mark.parametrize(
    ("times", "period"),
    [
        (2450000.0 + 0.5 * np.arange(400.0), 0.5),
        (2440000.0 + 331.0 * np.arange(60.0), 331.0),
    ],
)
def test_exact_ephemeris_reports_no_wander(times, period):
    """Every O-C exactly zero: no wander, not infinitely significant wander."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ep = eddington_plakidis(times, period=period)
    assert ep.alpha == 0.0 and ep.epsilon == 0.0
    assert ep.slope_significance == 0.0
    assert np.isnan(ep.alpha_err) and np.isnan(ep.epsilon_err)


def test_results_compare_by_value():
    times = _random_walk_times(n=200, seed=5)
    e1 = eddington_plakidis(times, period=0.5)
    e2 = eddington_plakidis(times, period=0.5)
    assert e1 == e1
    assert e1 == e2 and e2 in [e1]
    assert e1 != eddington_plakidis(times, period=0.5, detrend=1)


# --- the exact expectation without dense n x n matrices ------------------------


def _dense_reference(times, period, detrend=2, max_lag=None):
    """The estimator's point values from dense projector algebra (n^2 memory)."""
    t = np.sort(np.asarray(times, dtype=np.float64))
    cycles = np.round((t - t[0]) / period)
    oc = t - t[0] - period * cycles
    n = cycles.size
    span = int(cycles[-1] - cycles[0])
    top = max(3, min(span // 3, 100)) if max_lag is None else min(int(max_lag), span)
    basis = np.vander(cycles - cycles.mean(), detrend + 1, increasing=True)
    q, _ = np.linalg.qr(basis)
    a_mat = np.eye(n) - q @ q.T
    resid = a_mat @ oc
    e_rel = cycles - cycles[0]
    b_walk = a_mat @ np.minimum.outer(e_rel, e_rel) @ a_mat.T
    ii, jj = np.triu_indices(n, k=1)
    lag = (cycles[jj] - cycles[ii]).astype(np.int64)
    keep = (lag >= 1) & (lag <= top)
    ii, jj, lag = ii[keep], jj[keep], lag[keep]
    dw, dn = np.diag(b_walk), np.diag(a_mat)
    g_walk = dw[ii] + dw[jj] - 2.0 * b_walk[ii, jj]
    g_noise = dn[ii] + dn[jj] - 2.0 * a_mat[ii, jj]
    u2 = (resid[jj] - resid[ii]) ** 2
    counts = np.bincount(lag, minlength=top + 1)
    good = np.flatnonzero(counts >= 5)
    good = good[good >= 1]
    m = counts[good].astype(np.float64)
    y = np.bincount(lag, weights=u2, minlength=top + 1)[good] / m
    x_mat = np.stack(
        [
            np.bincount(lag, weights=g_noise, minlength=top + 1)[good] / m,
            np.bincount(lag, weights=g_walk, minlength=top + 1)[good] / m,
        ],
        axis=1,
    )
    weights = m.copy()
    coef = np.zeros(2)
    for _ in range(2):
        xtw = x_mat.T * weights[None, :]
        coef = np.linalg.solve(xtw @ x_mat, xtw @ y)
        floor = 0.05 * float(np.median(y))
        predicted = np.maximum(x_mat @ coef, max(floor, np.finfo(float).tiny))
        weights = m / (2.0 * predicted * predicted)
    return good, np.sqrt(y), m, coef


@pytest.mark.parametrize(
    ("make", "kwargs"),
    [
        ({"seed": 1}, {}),
        ({"seed": 6, "n": 150}, {"detrend": 0}),
        ({"seed": 7, "n": 150}, {"detrend": 4}),
        ({"seed": 8, "n": 300}, {"max_lag": 40}),
    ],
)
def test_matrix_free_estimator_matches_the_dense_one(make, kwargs):
    times = _random_walk_times(**make)
    ep = eddington_plakidis(times, period=0.5, **kwargs)
    lags, mean_u, n_pairs, coef = _dense_reference(times, 0.5, **kwargs)
    assert np.array_equal(ep.lags, lags) and np.array_equal(ep.n_pairs, n_pairs)
    assert np.allclose(ep.mean_u, mean_u, rtol=1e-9)
    if coef[1] > 0.0:
        assert ep.epsilon_sigma == pytest.approx(np.sqrt(coef[1]), rel=1e-9)
    if coef[0] > 0.0:
        assert ep.alpha_sigma == pytest.approx(np.sqrt(coef[0]), rel=1e-9)


def test_gappy_and_duplicated_cycles_match_the_dense_estimator():
    rng = np.random.default_rng(4)
    cycles = np.concatenate([[0], np.cumsum(rng.integers(1, 4, 250))]).astype(float)
    times = 2450000.0 + 0.5 * cycles + np.cumsum(rng.normal(0.0, 0.002, cycles.size))
    times = times + rng.normal(0.0, 0.0005, cycles.size)
    ep = eddington_plakidis(times, period=0.5)
    lags, mean_u, _, coef = _dense_reference(times, 0.5)
    assert np.array_equal(ep.lags, lags)
    assert np.allclose(ep.mean_u, mean_u, rtol=1e-9)
    assert ep.epsilon_sigma == pytest.approx(np.sqrt(coef[1]), rel=1e-9)


def test_memory_stays_far_below_dense_matrices():
    """2000 events used to hold ~6.5 dense n x n arrays (~200 MB)."""
    import tracemalloc

    times = _random_walk_times(n=2999, seed=9)
    tracemalloc.start()
    try:
        eddington_plakidis(times, period=0.5)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100e6


# --- calibrated errors and significance -----------------------------------------


@pytest.mark.parametrize("n_events", [12, 150])
def test_significance_is_calibrated_under_the_null(n_events):
    """White timing noise only: the slope significance is a z-score near 0.

    The fit's own residual scatter said almost nothing about the estimator's
    variance (every lag bin is a quadratic form in the same residuals), and
    slope_significance > 3 came up 5-15% of the time on pure timing noise.
    """
    over = 0
    values = []
    for seed in range(200):
        rng = np.random.default_rng(10_000 + seed)
        times = 2450000.0 + 0.5 * np.arange(n_events) + rng.normal(0, 1e-3, n_events)
        sig = eddington_plakidis(times, period=0.5).slope_significance
        values.append(sig)
        over += sig > 3.0
    assert over <= 4
    assert 0.7 < float(np.std(values)) < 1.3


def test_walk_variance_errors_cover_the_truth():
    """The reported error on the walk variance must be a real 1-sigma."""
    inside = 0
    trials = 60
    truth = 0.002**2
    for seed in range(trials):
        times = _random_walk_times(n=150, seed=500 + seed)
        ep = eddington_plakidis(times, period=0.5)
        var = ep.epsilon_sigma**2
        var_err = 2.0 * ep.epsilon_sigma * ep.epsilon_err / np.sqrt(2.0 / np.pi)
        inside += abs(var - truth) < var_err
    assert 0.45 * trials <= inside <= 0.92 * trials
