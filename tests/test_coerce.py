"""The shared float64 coercion used by every input path (cuperiod.core._coerce)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from astropy import units as u
from astropy.table import Column, MaskedColumn
from astropy.time import Time, TimeDelta
from astropy.utils.masked import Masked

from cuperiod.core._coerce import as_float_array

_MASK = np.array([False, True, False])
_EXPECTED = np.array([1.0, np.nan, 3.0])

_CONTAINERS: dict[str, Any] = {
    "np.ma float": np.ma.MaskedArray([1.0, 99.99, 3.0], mask=_MASK),
    "np.ma int64": np.ma.MaskedArray([1, 99, 3], mask=_MASK),
    "np.ma uint16": np.ma.MaskedArray(np.array([1, 99, 3], np.uint16), mask=_MASK),
    "np.ma object": np.ma.MaskedArray(np.array([1.0, None, 3.0], object), mask=_MASK),
    "MaskedColumn float": MaskedColumn([1.0, 0.0, 3.0], mask=_MASK),
    "MaskedColumn int": MaskedColumn([1, 0, 3], mask=_MASK),
    "astropy Masked float": Masked(np.array([1.0, 0.0, 3.0]), mask=_MASK),
    "astropy Masked int": Masked(np.array([1, 0, 3]), mask=_MASK),
    "Masked Quantity": Masked(np.array([1.0, 0.0, 3.0]) * u.Jy, mask=_MASK),
    "pandas Float64": pd.array([1.0, None, 3.0], dtype="Float64"),
    "pandas Int64 series": pd.Series([1, None, 3], dtype="Int64"),
    "pandas object pd.NA": pd.Series([1.0, pd.NA, 3.0], dtype=object),
    "pandas arrow int": pd.Series([1, None, 3], dtype="int64[pyarrow]"),
    "pyarrow array": pa.array([1.0, None, 3.0]),
    "pyarrow chunked int": pa.chunked_array([[1, None], [3]]),
    # A dictionary column's to_numpy reads a null's index slot: decode first.
    "pyarrow dictionary": pa.chunked_array(
        [pa.array([1.0, None, 3.0]).dictionary_encode()]
    ),
    "list with None": [1.0, None, 3.0],
}


@pytest.mark.parametrize("key", list(_CONTAINERS))
def test_missing_entries_become_nan(key: str) -> None:
    out = as_float_array(_CONTAINERS[key])
    assert out.dtype == np.float64
    assert type(out) is np.ndarray
    assert np.array_equal(out, _EXPECTED, equal_nan=True)


_STRING_WRAPPERS: dict[str, Any] = {
    "np.ma": lambda raw: np.ma.MaskedArray(raw, mask=_MASK),
    "MaskedColumn": lambda raw: MaskedColumn(raw, mask=_MASK),
    "astropy Masked": lambda raw: Masked(raw, mask=_MASK),
}


@pytest.mark.parametrize("wrapper", list(_STRING_WRAPPERS))
@pytest.mark.parametrize("dtype", ["U1", "U2", "U3", "S1", "S2"])
def test_masked_short_strings_become_nan(wrapper: str, dtype: str) -> None:
    # Regression: filled(nan) on a U1/U2/S1/S2 column stores 'nan' cut to the column
    # width ('n', 'na'), which then failed to parse -- only U3+ happened to work.
    raw = np.array(["1", "x", "3"]).astype(dtype)
    out = as_float_array(_STRING_WRAPPERS[wrapper](raw))
    assert np.array_equal(out, _EXPECTED, equal_nan=True)


def test_plain_inputs_pass_through() -> None:
    a = np.arange(5.0)
    assert as_float_array(a) is a  # float64 C-contiguous ndarray: no copy
    assert np.array_equal(as_float_array([1, 2, 3]), [1.0, 2.0, 3.0])
    assert np.array_equal(as_float_array(Column([1.0, 2.0])), [1.0, 2.0])
    assert np.array_equal(as_float_array(np.array([True, False])), [1.0, 0.0])
    assert as_float_array(np.arange(6.0).reshape(2, 3)[:, 1]).flags.c_contiguous


def test_datetimes_refused_unless_time() -> None:
    stamps = np.array(["1858-11-18", "1858-11-19T12:00"], dtype="datetime64[s]")
    with pytest.raises(TypeError, match="flux .*datetime64"):
        as_float_array(stamps, name="flux")
    with pytest.raises(TypeError, match="timedelta64"):
        as_float_array(np.array([1, 2], dtype="timedelta64[h]"))
    with pytest.raises(TypeError, match="astropy Time"):
        as_float_array(Time([2459000.5], format="jd"))
    assert np.array_equal(as_float_array(stamps, time=True), [1.0, 2.5])


def test_time_conversions() -> None:
    nat = np.array(["1858-11-18", "NaT"], dtype="datetime64[ns]")
    assert np.array_equal(as_float_array(nat, time=True), [1.0, np.nan], equal_nan=True)
    hours = np.array([12, 36], dtype="timedelta64[h]")
    assert np.array_equal(as_float_array(hours, time=True), [0.5, 1.5])
    assert np.array_equal(
        as_float_array(pd.Series(nat), time=True), [1.0, np.nan], equal_nan=True
    )
    jd = Time([2459000.5, 2459001.5], format="jd")
    assert np.array_equal(as_float_array(jd, time=True), [2459000.5, 2459001.5])
    assert np.allclose(
        as_float_array(Time(jd, format="isot"), time=True),
        [59000.0, 59001.0],
        rtol=0.0,
        atol=1e-8,
    )
    span = TimeDelta([0.5, 1.0], format="jd")
    assert np.array_equal(as_float_array(span, time=True), [0.5, 1.0])
    masked = jd.copy()
    masked[1] = np.ma.masked
    assert np.array_equal(
        as_float_array(masked, time=True), [2459000.5, np.nan], equal_nan=True
    )


@pytest.mark.parametrize("time", [False, True])
def test_unit_is_ignored_alike_for_quantity_and_column(time: bool) -> None:
    # A time unit is not applied: a Quantity (QTable) and a plain Column with a unit
    # (what FITS/ECSV reads give) must agree, or one file would load 86400x apart
    # depending on the container it arrives in.
    secs = [0.0, 86400.0]
    for container in (secs * u.s, Column(secs, unit="s"), Masked(secs * u.s)):
        assert np.array_equal(as_float_array(container, time=time), secs)


def test_ndim_check_and_named_errors() -> None:
    with pytest.raises(ValueError, match="epochs must be 1-D, got shape"):
        as_float_array(np.zeros((2, 2)), name="epochs", ndim=1)
    with pytest.raises(ValueError, match="value column 'mag' .*could not convert"):
        as_float_array(np.array(["g", "r"]), name="value column 'mag'")
