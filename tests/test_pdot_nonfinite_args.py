"""Non-finite and non-integral arguments of the period-change tools (PDOT-A-12).

Every public entry point refuses a NaN or infinite period, rate, window, frequency or
count up front with a ValueError that names the argument. Before, they surfaced far
from the cause: an SVD that did not converge (``fit_template``), an int() conversion
error (``wwz(n_tau=nan)``, ``period_profile(harmonics=inf)``), an empty profile, a
silent global-median trend (``running_median(min_points=nan)``), NaN warped times
(``warp_dt(a=nan)``) or a "period passes through zero" for an infinite rate.
"""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.pdot.models import get_model
from cuperiod.pdot.scan import warp_dt
from cuperiod.pdot.track import fit_template
from synth import synthetic_period_change

BAD = [float("nan"), float("inf"), float("-inf")]

T, MAG, ERR, TRUTH = synthetic_period_change(n=400, seed=1)
P = TRUTH["period"]
DATA = (T, MAG, ERR)
DT = T - T.mean()
W = 1.0 / ERR**2
EVENTS = 2458000.0 + P * np.arange(30)


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize(
    ("label", "call", "match"),
    [
        ("pdot_search period", lambda v: cup.pdot_search(DATA, period=v), "period"),
        ("phase_track period", lambda v: cup.phase_track(DATA, period=v), "period"),
        ("phase_track pdot", lambda v: cup.phase_track(DATA, period=P, pdot=v), "pdot"),
        ("fit_ephemeris period", lambda v: cup.fit_ephemeris(EVENTS, period=v),
         "period"),
        ("fit_ephemeris t0", lambda v: cup.fit_ephemeris(EVENTS, period=P, t0=v),
         "t0"),
        ("fit_template f0", lambda v: fit_template(DT, MAG, W, v), "f0"),
        ("fit_template n_harmonics",
         lambda v: fit_template(DT, MAG, W, 1.0 / P, n_harmonics=v), "n_harmonics"),
        ("fit_template max_harmonics",
         lambda v: fit_template(DT, MAG, W, 1.0 / P, max_harmonics=v),
         "max_harmonics"),
        ("sliding window", lambda v: cup.sliding_periodogram(DATA, window=v),
         "window"),
        ("sliding step", lambda v: cup.sliding_periodogram(DATA, window=200.0, step=v),
         "step"),
        ("sliding period",
         lambda v: cup.sliding_periodogram(DATA, window=200.0, period=v), "period"),
        ("sliding min_window_points",
         lambda v: cup.sliding_periodogram(DATA, window=200.0, min_window_points=v),
         "min_window_points"),
        ("wwz c", lambda v: cup.wwz(DATA, c=v), "c must be"),
        ("wwz min_neff", lambda v: cup.wwz(DATA, min_neff=v), "min_neff"),
        ("wwz n_tau", lambda v: cup.wwz(DATA, n_tau=v), "n_tau"),
        ("profile period", lambda v: cup.period_profile(DATA, period=v), "period"),
        ("profile period_window",
         lambda v: cup.period_profile(DATA, period=P, period_window=v),
         "period_window"),
        ("profile points_per_interval",
         lambda v: cup.period_profile(DATA, period=P, points_per_interval=v),
         "points_per_interval"),
        ("profile n_intervals",
         lambda v: cup.period_profile(DATA, period=P, n_intervals=v), "n_intervals"),
        ("profile harmonics",
         lambda v: cup.period_profile(DATA, period=P, harmonics=v), "harmonics"),
        ("detrend width", lambda v: cup.detrend(T, MAG, v), "width"),
        ("running_median min_points",
         lambda v: cup.running_median(T, MAG, 50.0, min_points=v), "min_points"),
        ("warp_dt a", lambda v: warp_dt(DT, v), "rate"),
        ("warp_dt a exponential", lambda v: warp_dt(DT, v, "exponential"), "rate"),
        ("model delay a", lambda v: get_model(None).delay(DT, v), "rate"),
    ],
)
def test_non_finite_arguments_are_refused(label, call, match, bad) -> None:
    with pytest.raises(ValueError, match=match):
        call(bad)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: cup.wwz(DATA, n_tau=2.5), "n_tau"),
        (lambda: cup.period_profile(DATA, period=P, harmonics=1.5), "harmonics"),
        (lambda: cup.sliding_periodogram(DATA, window=200.0, min_window_points=-1),
         "min_window_points"),
        (lambda: cup.running_median(T, MAG, 50.0, min_points=0), "min_points"),
        (lambda: fit_template(DT, MAG, W, 0.0), "f0"),
    ],
)
def test_non_integral_or_out_of_range_counts_are_refused(call, match) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_valid_integral_values_still_work() -> None:
    # Integral floats are accepted for the counts (they used to be).
    template = fit_template(DT, MAG, W, 1.0 / P, n_harmonics=2.0)  # type: ignore[arg-type]
    assert template.n_harmonics == 2
    trend = cup.running_median(T, MAG, 50.0, min_points=3.0)  # type: ignore[arg-type]
    assert np.all(np.isfinite(trend))
    assert warp_dt(DT, 0.0) is not None
