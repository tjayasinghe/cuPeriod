"""core.lightcurve.stack_bands: the GUI controller's band merge, moved verbatim."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.core.lightcurve import stack_bands


def _mblc(with_errors: bool = True) -> cup.MultiBandLightCurve:
    rng = np.random.default_rng(3)
    bands = {}
    for name, zero in (("V", 12.0), ("B", 12.6), ("R", 11.7)):
        t = np.sort(rng.uniform(0.0, 30.0, 80))
        y = zero + 0.1 * np.sin(2 * np.pi * 3.1 * t) + rng.normal(0, 0.01, t.size)
        e = np.full(t.size, 0.01) if with_errors else None
        y[5] = np.nan  # a non-finite point per band, dropped by the merge
        bands[name] = cup.LightCurve.from_arrays(t, y, e)
    return cup.MultiBandLightCurve.from_light_curves(bands)


@pytest.mark.parametrize("with_errors", [True, False])
def test_uncentred_merge(with_errors: bool) -> None:
    mblc = _mblc(with_errors)
    lc, band = stack_bands(mblc, centre=False)
    time, value, error, labels = mblc.finite().stacked()
    np.testing.assert_array_equal(lc.time, time)
    np.testing.assert_array_equal(lc.value, value)
    if with_errors:
        np.testing.assert_array_equal(lc.error, error)
    else:
        assert lc.error is None
    assert band.dtype == object and band.tolist() == labels.tolist()
    assert np.all(np.diff(lc.time) >= 0.0)


def test_centred_merge_removes_each_band_mean() -> None:
    mblc = _mblc()
    lc, band = stack_bands(mblc, centre=True)
    assert lc.n == 3 * 79 and band.size == lc.n
    for name in ("V", "B", "R"):
        assert abs(float(np.mean(lc.value[band == name]))) < 1e-12
    assert set(band.tolist()) == {"V", "B", "R"}


def test_matches_the_gui_controller() -> None:
    pytest.importorskip("PySide6")
    from cuperiod.gui.state import AppController

    mblc = _mblc()
    for centre, legacy in (
        (False, AppController._stack(mblc)),
        (True, AppController._stack_centred(mblc)),
    ):
        lc, _ = stack_bands(mblc, centre=centre)
        np.testing.assert_array_equal(lc.time, legacy.time)
        np.testing.assert_array_equal(lc.value, legacy.value)
        np.testing.assert_array_equal(lc.error, legacy.error)
        assert lc.domain == legacy.domain
