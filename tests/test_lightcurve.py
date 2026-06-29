"""LightCurve / MultiBandLightCurve construction and transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve


def test_from_arrays_and_properties() -> None:
    lc = LightCurve.from_arrays([1.0, 2.0, 4.0], [10.0, 11.0, 12.0], [0.1, 0.1, 0.1])
    assert lc.n == 3
    assert lc.baseline == pytest.approx(3.0)
    assert lc.time.dtype == np.float64


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        LightCurve.from_arrays([1.0, 2.0], [1.0])


def test_finite_drops_nan_and_bad_error() -> None:
    t = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([1.0, np.nan, 3.0, 4.0])
    e = np.array([0.1, 0.1, -1.0, 0.1])
    lc = LightCurve.from_arrays(t, v, e).finite()
    assert lc.n == 2
    assert np.array_equal(lc.time, [1.0, 4.0])


def test_flux_magnitude_roundtrip() -> None:
    lc = LightCurve.from_arrays([1.0, 2.0], [15.0, 16.0], [0.05, 0.05])
    back = lc.as_flux().as_magnitude()
    assert np.allclose(back.value, lc.value)
    assert back.error is not None
    assert np.allclose(back.error, lc.error, rtol=1e-6)
    assert back.domain is Domain.MAGNITUDE


def test_from_dataframe() -> None:
    df = pd.DataFrame({"HJD": [1.0, 2.0], "Vmag": [12.0, 12.1], "e": [0.1, 0.1]})
    lc = LightCurve.from_dataframe(df, columns=ColumnMap(value="Vmag", error="e"))
    assert lc.n == 2
    assert lc.value[0] == pytest.approx(12.0)


def test_multiband_stacked_sorted() -> None:
    g = LightCurve.from_arrays([3.0, 1.0], [1.0, 2.0], [0.1, 0.1])
    r = LightCurve.from_arrays([2.0], [3.0], [0.1])
    mb = MultiBandLightCurve.from_light_curves({"g": g, "r": r})
    time, value, error, band = mb.stacked()
    assert np.array_equal(time, [1.0, 2.0, 3.0])
    assert list(band) == ["g", "r", "g"]
    assert error is not None


def test_multiband_requires_band() -> None:
    with pytest.raises(ValueError, match="at least one band"):
        MultiBandLightCurve.from_light_curves({})
