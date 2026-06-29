"""The periodogram() dispatcher: input forms, multi-method, settings."""

from __future__ import annotations

import pytest

import cuperiod as cup
from cuperiod.core.result import MultiResult, Periodogram
from synth import synthetic_sine


def test_single_method_returns_periodogram() -> None:
    t, mag, err = synthetic_sine()
    pg = cup.periodogram((t, mag, err), "GLS")
    assert isinstance(pg, Periodogram)
    assert pg.method == "GLS"


def test_multi_method_returns_multiresult() -> None:
    t, mag, err = synthetic_sine()
    res = cup.periodogram((t, mag, err), ["gls", "bls"], domain="flux")
    assert isinstance(res, MultiResult)
    assert set(res.keys()) == {"GLS", "BLS"}


def test_dict_of_arrays_input() -> None:
    t, mag, err = synthetic_sine()
    pg = cup.periodogram({"jd": t, "mag": mag, "mag_err": err}, "GLS")
    assert pg.best_period() == pytest.approx(0.6234, rel=1e-3)


def test_mapping_of_lightcurves_is_multiband() -> None:
    t, mag, err = synthetic_sine(n=200)
    lcs = {
        "g": cup.LightCurve.from_arrays(t, mag, err),
        "r": cup.LightCurve.from_arrays(t, mag + 1.0, err),
    }
    pg = cup.periodogram(lcs, "GLS")
    assert "bands" in pg.meta


def test_settings_override_changes_grid() -> None:
    t, mag, err = synthetic_sine()
    coarse = cup.periodogram(
        (t, mag, err), "GLS", settings=cup.GLSSettings(samples_per_peak=2)
    )
    fine = cup.periodogram(
        (t, mag, err), "GLS", settings=cup.GLSSettings(samples_per_peak=10)
    )
    assert fine.size > coarse.size


def test_best_periods_convenience() -> None:
    t, mag, err = synthetic_sine()
    pg = cup.periodogram((t, mag, err), "GLS")
    assert len(cup.best_periods(pg, 3)) == 3


def test_unknown_method_raises() -> None:
    t, mag, err = synthetic_sine()
    with pytest.raises(cup.UnknownMethodError):
        cup.periodogram((t, mag, err), "NOPE")
