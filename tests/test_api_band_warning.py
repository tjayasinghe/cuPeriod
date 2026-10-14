"""A long-format table with a band column is not pooled silently (CORE-A-6).

``periodogram()`` / ``to_input()`` read a table as one light curve unless a band
column is named (the documented default). When the table carries a recognisable band
column (``band``, ``filter``, ``filtercode``, ``passband``, ``fid``, ...) holding
several labels, that pooling mixes the filters' zero points into every period
search, so it warns, naming the column and the exact argument that asks for a
multi-band analysis. Naming the band, or a column with one label, stays silent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import cuperiod as cup
from cuperiod.api import to_input


def _ztf_like(column: str = "filtercode", constant: bool = False) -> pd.DataFrame:
    """Two filters on alternate nights, 0.6 mag apart, one 0.5731-d sinusoid."""
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 60.0, 120)) + 58000.0
    band = np.where(np.floor(t).astype(int) % 2 == 0, "zg", "zr")
    if constant:
        band = np.full(t.size, "zg")
    mag = np.where(band == "zg", 16.9, 16.3) + 0.3 * np.sin(2 * np.pi * t / 0.5731)
    return pd.DataFrame({"mjd": t, "mag": mag, "magerr": 0.02, column: band})


def test_periodogram_of_a_multi_filter_table_warns_at_the_caller() -> None:
    frame = _ztf_like()
    with pytest.warns(UserWarning, match="band column 'filtercode'") as rec:
        pg = cup.periodogram(frame, "GLS", backend="cpu")
    assert pg.n_samples == 120  # still the documented single-band read
    message = str(rec[0].message)
    assert "2 labels ('zg', 'zr')" in message
    assert "columns=ColumnMap(band='filtercode')" in message
    assert "band_column='filtercode'" in message
    assert "MultiBandLightCurve.from_dataframe" in message
    # Attributed to the user's call, not to a line inside the library.
    assert rec[0].filename == __file__


@pytest.mark.parametrize(
    "column", ["band", "filter", "filtercode", "FilterID", "passband", "fid"]
)
def test_every_recognised_band_column_warns(column: str) -> None:
    with pytest.warns(UserWarning, match=f"band column '{column}'") as rec:
        lc = to_input(_ztf_like(column))
    assert isinstance(lc, cup.LightCurve) and lc.n == 120
    assert rec[0].filename == __file__


@pytest.mark.parametrize("kind", ["mapping", "pyarrow", "astropy"])
def test_other_table_kinds_warn_too(kind: str) -> None:
    frame = _ztf_like()
    if kind == "mapping":
        table: object = {c: frame[c].to_numpy() for c in frame.columns}
    elif kind == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        table = pa.Table.from_pandas(frame)
    else:
        astropy_table = pytest.importorskip("astropy.table")
        table = astropy_table.Table.from_pandas(frame)
    with pytest.warns(UserWarning, match="ColumnMap\\(band='filtercode'\\)"):
        to_input(table)


def test_numeric_filter_ids_warn() -> None:
    frame = _ztf_like("fid")
    frame["fid"] = np.where(frame["fid"] == "zg", 1, 2)
    with pytest.warns(UserWarning, match=r"'fid' holds 2 labels \('1', '2'\)"):
        to_input(frame)


def test_naming_the_band_column_splits_without_a_warning() -> None:
    frame = _ztf_like()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = to_input(frame, columns=cup.ColumnMap(band="filtercode"))
        assert isinstance(data, cup.MultiBandLightCurve)
        assert set(data.band_names) == {"zg", "zr"}
        cup.periodogram(
            frame, "GLS", backend="cpu", columns=cup.ColumnMap(band="filtercode")
        )


def test_a_single_label_band_column_is_silent() -> None:
    frame = _ztf_like(constant=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert to_input(frame).n == 120
        # Rows without a label do not make a second label either.
        frame.loc[::3, "filtercode"] = None
        assert to_input(frame).n == 120
