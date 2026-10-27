"""Every period-change result carries the derivative conventions the guide lists.

The guide's table promises ``pdot`` (with ``pdot_s_per_yr``, ``pdot_ms_per_yr`` and
``pdot_d_per_myr``), ``pdot_over_p`` and ``dlnp_dt_per_yr``. They are checked here on
each object a user reads a rate from, against :func:`cuperiod.convert_pdot`, so the
accessors and the converter can never disagree.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import cuperiod as cup
from synth import synthetic_period_change

#: Accessor -> the convert_pdot unit it must equal (from the dimensionless rate).
UNITS = {
    "pdot_s_per_yr": "s/yr",
    "pdot_ms_per_yr": "ms/yr",
    "pdot_d_per_yr": "d/yr",
    "pdot_d_per_myr": "d/Myr",
    "dlnp_dt_per_yr": "1/yr",
}


@pytest.fixture(scope="module")
def track() -> cup.PhaseTrackResult:
    time, mag, err, truth = synthetic_period_change(
        n=1200, span=1500.0, pdot=5.0e-7, seed=3
    )
    return cup.phase_track(
        (time, mag, err), period=truth["period"], pdot=truth["pdot"]
    )


def _check(obj: Any, period: float) -> None:
    for name, unit in UNITS.items():
        expected = cup.convert_pdot(obj.pdot, "d/d", unit, period=period)
        assert getattr(obj, name) == pytest.approx(expected, rel=1e-12), name
    assert obj.pdot_over_p == pytest.approx(obj.pdot / period, rel=1e-15)
    assert obj.to_dict()["pdot_over_p"] == obj.pdot_over_p
    assert obj.to_dict()["pdot_d_per_myr"] == obj.pdot_d_per_myr


def test_phase_track_result_has_every_convention(
    track: cup.PhaseTrackResult,
) -> None:
    assert track.pdot != 0.0
    _check(track, track.period)
    assert track.to_dict()["dlnp_dt_per_yr"] == track.dlnp_dt_per_yr
    assert track.to_dict()["pdot_ms_per_yr"] == track.pdot_ms_per_yr


def test_ephemeris_fit_has_every_convention(track: cup.PhaseTrackResult) -> None:
    fit = track.quadratic_fit
    _check(fit, fit.period_ref)
    # The fit's fractional rate is the one its track reports.
    assert fit.pdot_over_p == track.pdot_over_p


def test_ltte_fit_has_every_convention() -> None:
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 5000.0, 40)) + 2450000.0
    dt = t - t.mean()
    oc = (
        0.5 * (3e-8 / 0.4) * dt**2
        + 0.003 * np.sin(2.0 * np.pi * dt / 1300.0 + 0.4)
        + rng.normal(0.0, 5e-4, t.size)
    )
    best = cup.ltte_search(t, oc=oc, errors=np.full(t.size, 5e-4), period=0.4).best
    assert best.pdot != 0.0
    _check(best, best.period_ref)
    assert best.to_dict()["pdot_ms_per_yr"] == best.pdot_ms_per_yr
