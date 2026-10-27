"""LightCurve / MultiBandLightCurve construction and transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cuperiod as cup
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.lightcurve import (
    LightCurve,
    MultiBandLightCurve,
    raise_if_unusable,
)
from cuperiod.core.result import Periodogram
from synth import synthetic_sine


def test_as_magnitude_handles_nonpositive_flux() -> None:
    # Negative/zero flux has no magnitude: it must become non-finite *without* a bare
    # NumPy RuntimeWarning (which crashes under -W error) and then be dropped by finite.
    import warnings

    flux = np.array([1.0, 2.0, -0.5, 0.0, 3.0])
    lc = LightCurve.from_arrays(
        np.arange(5.0), flux, np.full(5, 0.1), domain=Domain.FLUX
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mlc = lc.as_magnitude()  # must not raise
    assert mlc.finite().n == 3  # the two non-positive-flux points are removed


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


def _gls(data: Any, **kwargs: Any) -> Periodogram:
    pg = cup.periodogram(data, "GLS", backend="finufft", **kwargs)
    assert isinstance(pg, Periodogram)
    return pg


# --- a string domain converts the right way -----------------------------------------


def _zero_mean_flux(n: int = 300) -> LightCurve:
    t, mag, err = synthetic_sine(n=n, noise=0.01)
    flux = (mag - mag.mean()) * 100.0  # a difference flux: half the points negative
    return LightCurve.from_arrays(t, flux, err * 100.0, domain=Domain.FLUX)


def test_in_domain_accepts_strings() -> None:
    lc_flux = _zero_mean_flux()
    assert lc_flux.in_domain("flux") is lc_flux  # type: ignore[arg-type]
    assert lc_flux.in_domain("FLUX") is lc_flux  # type: ignore[arg-type]
    lc_mag = LightCurve.from_arrays([1.0, 2.0], [15.0, 16.0], [0.05, 0.05])
    assert lc_mag.in_domain("flux").domain is Domain.FLUX  # type: ignore[arg-type]
    assert lc_mag.in_domain("magnitude") is lc_mag  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not a valid Domain"):
        lc_mag.in_domain("jansky")  # type: ignore[arg-type]


def test_periodogram_with_string_flux_domain_keeps_every_point() -> None:
    # A zero-mean flux curve with domain="flux" used to be converted to magnitudes,
    # dropping every negative flux and losing most of the signal.
    lc = _zero_mean_flux()
    pg = _gls(lc, domain="flux")
    assert pg.n_samples == lc.n
    assert cup.to_input(lc, domain="flux") is lc


# --- masked input: the tuple path agrees with the table path ------------------------


def _masked_sine() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, mag, err = synthetic_sine(n=300, period=0.6234, noise=0.02)
    bad = np.zeros(t.size, dtype=bool)
    bad[::10] = True
    sentinel = mag.copy()
    sentinel[bad] = 99.99  # a survey sentinel hidden behind the mask
    return t, sentinel, err, bad


def test_masked_value_in_tuple_is_dropped() -> None:
    t, mag, err, bad = _masked_sine()
    ym = np.ma.MaskedArray(mag, mask=bad)
    lc = LightCurve.from_arrays(t, ym, err)
    assert np.isnan(lc.value[bad]).all()
    assert lc.finite().n == int((~bad).sum())
    tup = _gls((t, ym, err))
    tab = _gls({"jd": t, "mag": ym, "mag_err": err})
    assert tup.n_samples == tab.n_samples == int((~bad).sum())
    assert tup.best_period() == pytest.approx(0.6234, rel=1e-3)
    assert tup.best_period() == pytest.approx(tab.best_period(), rel=1e-12)


@pytest.mark.parametrize("role", ["time", "value", "error"])
def test_masked_entries_become_nan_for_every_role(role: str) -> None:
    t, mag, err, bad = _masked_sine()
    arrays = {"time": t, "value": mag, "error": err}
    arrays[role] = np.ma.MaskedArray(arrays[role], mask=bad)
    lc = LightCurve(**arrays)  # type: ignore[arg-type]
    assert lc.finite().n == int((~bad).sum())


def test_astropy_masked_columns_in_tuple() -> None:
    from astropy.table import MaskedColumn
    from astropy.utils.masked import Masked

    t, mag, err, bad = _masked_sine()
    for wrapped in (MaskedColumn(mag, mask=bad), Masked(mag, mask=bad)):
        lc = LightCurve.from_arrays(t, wrapped, err)
        assert lc.finite().n == int((~bad).sum())


def test_masked_integer_columns_do_not_crash() -> None:
    from astropy.table import MaskedColumn, Table

    t, mag, err, bad = _masked_sine()
    counts = np.round(1000.0 * 10 ** (-0.4 * (mag - 12.0))).astype(np.int64)
    n_good = int((~bad).sum())
    table = Table({"jd": t, "flux": MaskedColumn(counts, mask=bad), "flux_err": err})
    assert LightCurve.from_dataframe(table).finite().n == n_good
    as_dict = {"jd": t, "flux": np.ma.MaskedArray(counts, mask=bad)}
    assert cup.to_input(as_dict).finite().n == n_good  # type: ignore[union-attr]
    u16 = MaskedColumn(counts.astype(np.uint16), mask=bad)
    assert LightCurve.from_arrays(t, u16).finite().n == n_good


def test_csv_with_blank_integer_cells(tmp_path: Path) -> None:
    # astropy reads an integer column with blank cells as a masked int64 column,
    # whose filled(nan) used to raise TypeError.
    t, mag, _, bad = _masked_sine()
    counts = np.round(1000.0 * 10 ** (-0.4 * (mag - 12.0))).astype(np.int64)
    lines = ["jd,flux"] + [
        f"{float(ti)!r},{'' if b else int(c)}"
        for ti, c, b in zip(t, counts, bad, strict=True)
    ]
    path = tmp_path / "counts.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    lc = LightCurve.from_file(path)
    assert lc.domain is Domain.FLUX
    assert lc.finite().n == int((~bad).sum())


def test_masked_time_integer_column(tmp_path: Path) -> None:
    from astropy.table import MaskedColumn, Table

    t = np.arange(20, dtype=np.int64)
    mask = np.zeros(20, dtype=bool)
    mask[3] = True
    table = Table({"t": MaskedColumn(t, mask=mask), "mag": np.ones(20)})
    lc = LightCurve.from_dataframe(table)
    assert np.isnan(lc.time[3])
    assert lc.finite().n == 19


def test_masked_bool_and_string_columns() -> None:
    mask = np.array([False, True, False])
    flags = LightCurve.from_arrays(
        [1.0, 2.0, 3.0], np.ma.MaskedArray([True, False, True], mask=mask)
    )
    assert np.array_equal(flags.value, [1.0, np.nan, 1.0], equal_nan=True)
    text = LightCurve.from_arrays(
        [1.0, 2.0, 3.0], np.ma.MaskedArray(["1.5", "N/A", "2.5"], mask=mask)
    )
    assert np.array_equal(text.value, [1.5, np.nan, 2.5], equal_nan=True)


@pytest.mark.parametrize("dtype", ["U1", "U2", "S1", "S2"])
def test_masked_narrow_string_column_in_table(dtype: str) -> None:
    # A masked text column narrower than 'nan' used to fail to parse: filled(nan)
    # stored 'n' or 'na'. "N/A" above passed only because it makes the column U3.
    from astropy.table import MaskedColumn, Table

    mask = np.array([False, True, False])
    mag = MaskedColumn(np.array(["1", "xx", "3"]).astype(dtype), mask=mask)
    lc = LightCurve.from_dataframe(Table({"jd": [1.0, 2.0, 3.0], "mag": mag}))
    assert np.array_equal(lc.value, [1.0, np.nan, 3.0], equal_nan=True)
    assert lc.finite().n == 2


def test_pandas_nullable_columns_still_work() -> None:
    df = pd.DataFrame(
        {
            "jd": pd.array([1.0, 2.0, 3.0, 4.0], dtype="Float64"),
            "mag": pd.array([10, None, 12, 13], dtype="Int64"),
            "mag_err": pd.Series([0.1, 0.1, pd.NA, 0.1], dtype=object),
        }
    )
    lc = LightCurve.from_dataframe(df)
    assert np.array_equal(lc.value, [10.0, np.nan, 12.0, 13.0], equal_nan=True)
    assert lc.finite().n == 2
    flags = LightCurve.from_arrays([1.0, 2.0], pd.Series([True, None], dtype="boolean"))
    assert np.array_equal(flags.value, [1.0, np.nan], equal_nan=True)


# --- calendar times become days -----------------------------------------------------

_MJD0 = np.datetime64("1858-11-17T00:00:00")


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_datetime_time_column_becomes_mjd(unit: str) -> None:
    t, mag, err = synthetic_sine(n=300, period=0.6234, base_jd=0.0)
    mjd = 59000.0 + t
    stamps = (_MJD0 + np.round(mjd * 86400e3).astype("timedelta64[ms]")).astype(
        f"datetime64[{unit}]"
    )
    df = pd.DataFrame({"date": stamps, "mag": mag, "mag_err": err})
    lc = LightCurve.from_dataframe(df)
    assert np.allclose(lc.time, mjd, rtol=0.0, atol=1.0 / 86400.0)
    assert lc.baseline == pytest.approx(np.ptp(mjd), abs=1e-5)
    ref = _gls((mjd, mag, err)).best_period()
    got = _gls(df).best_period()
    assert got == pytest.approx(ref, rel=1e-6)


def test_datetime_tuple_index_and_timedelta() -> None:
    days = np.array([0.0, 0.25, 1.5, 3.0])
    stamps = pd.DatetimeIndex(_MJD0 + (days * 86400).astype("timedelta64[s]"))
    lc = LightCurve.from_arrays(stamps, np.ones(4))
    assert np.allclose(lc.time, days)
    delta = (days * 24).astype("timedelta64[h]")
    assert np.allclose(LightCurve.from_arrays(delta, np.ones(4)).time, days)
    eastern = pd.Series(stamps).dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    assert np.allclose(LightCurve.from_arrays(eastern, np.ones(4)).time, days)


def test_pyarrow_timestamp_and_nat() -> None:
    import pyarrow as pa

    stamps = np.array(
        ["1858-11-18T00:00:00", "NaT", "1858-11-20T12:00:00"], dtype="datetime64[us]"
    )
    table = pa.table({"time": pa.array(stamps), "mag": [1.0, 2.0, 3.0]})
    lc = LightCurve.from_dataframe(table)  # type: ignore[arg-type]
    assert np.array_equal(lc.time, [1.0, np.nan, 3.5], equal_nan=True)
    assert lc.finite().n == 2  # NaT is missing, not a finite -9.2e18


def test_time_unit_does_not_depend_on_the_container(tmp_path: Path) -> None:
    # The same seconds-valued time column must load the same whether it arrives as a
    # QTable Quantity, a Table Column with a unit (what FITS/ECSV reads give), or a
    # tuple Quantity. Units are not applied, as before: times are taken as days.
    from astropy import units as u
    from astropy.table import Column, QTable, Table

    secs = np.array([0.0, 86400.0, 172800.0])
    mag = np.array([10.0, 10.5, 10.2])
    qtable = LightCurve.from_dataframe(QTable({"time": secs * u.s, "mag": mag}))
    table = Table({"time": Column(secs, unit="s"), "mag": mag})
    path = tmp_path / "secs.ecsv"
    table.write(path, format="ascii.ecsv")
    for lc in (
        qtable,
        LightCurve.from_dataframe(table),
        LightCurve.from_file(path),
        LightCurve.from_arrays(secs * u.s, mag),
    ):
        assert np.array_equal(lc.time, secs)


def test_datetime_value_column_is_refused() -> None:
    stamps = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    with pytest.raises(TypeError, match="value.*datetime"):
        LightCurve.from_arrays([1.0, 2.0], stamps)


# --- astropy Time columns -----------------------------------------------------------


def test_astropy_time_columns() -> None:
    from astropy.table import QTable
    from astropy.time import Time
    from astropy.timeseries import TimeSeries

    t, flux, err = synthetic_sine(n=300, period=0.6234)
    ts = TimeSeries(time=Time(t, format="jd"), data={"flux": flux, "flux_err": err})
    lc = LightCurve.from_dataframe(ts)
    assert np.array_equal(lc.time, t)
    assert lc.domain is Domain.FLUX
    qt = QTable({"time": Time(t, format="jd"), "mag": flux, "mag_err": err})
    assert np.array_equal(LightCurve.from_dataframe(qt).time, t)
    tup = LightCurve.from_arrays(ts.time, ts["flux"], ts["flux_err"])
    assert np.array_equal(tup.time, t)
    ref = _gls((t, flux, err)).best_period()
    assert _gls(ts).best_period() == ref


def test_astropy_time_formats() -> None:
    from astropy.time import Time, TimeDelta

    jd = np.array([2459000.5, 2459001.0, 2459003.25])
    # Day-valued formats keep the user's frame; others become MJD.
    assert np.array_equal(LightCurve.from_arrays(Time(jd, format="jd"), jd).time, jd)
    mjd = jd - 2400000.5
    for fmt in ("isot", "unix", "decimalyear"):
        lc = LightCurve.from_arrays(Time(Time(jd, format="jd"), format=fmt), jd)
        assert np.allclose(lc.time, mjd, rtol=0.0, atol=1e-6)
    secs = TimeDelta(np.array([0.0, 43200.0, 86400.0]), format="sec")
    assert np.allclose(LightCurve.from_arrays(secs, jd).time, [0.0, 0.5, 1.0])
    with pytest.raises(TypeError, match="value"):
        LightCurve.from_arrays(jd, Time(jd, format="jd"))


def test_ecsv_round_trip_with_time_column(tmp_path: Path) -> None:
    from astropy.time import Time
    from astropy.timeseries import TimeSeries

    t, flux, err = synthetic_sine(n=50)
    ts = TimeSeries(time=Time(t, format="jd"), data={"flux": flux, "flux_err": err})
    path = tmp_path / "ts.ecsv"
    ts.write(path, format="ascii.ecsv")
    lc = LightCurve.from_file(path)
    assert np.allclose(lc.time, t, rtol=0.0, atol=1e-9)
    assert lc.n == 50


# --- an unusable error column is named as the cause ---------------------------------


@pytest.mark.parametrize("bad", [0.0, np.nan, -1.0])
def test_unusable_reason_names_the_errors(bad: float) -> None:
    t, mag, _ = synthetic_sine(n=100)
    lc = LightCurve.from_arrays(t, mag, np.full(t.size, bad))
    reason = lc.unusable_reason()
    assert reason is not None
    assert "error" in reason and "100" in reason
    with pytest.raises(InsufficientDataError, match="GLS: .*error"):
        raise_if_unusable(lc, "GLS")
    mb = MultiBandLightCurve.from_light_curves({"g": lc, "r": lc})
    assert mb.unusable_reason() is not None
    with pytest.raises(InsufficientDataError, match="band 'g'"):
        raise_if_unusable(mb, "GLS multiband")


def test_unusable_reason_is_none_for_usable_curves() -> None:
    t, mag, err = synthetic_sine(n=100)
    assert LightCurve.from_arrays(t, mag, err).unusable_reason() is None
    assert LightCurve.from_arrays(t, mag).unusable_reason() is None
    partly = err.copy()
    partly[::10] = 0.0  # a few bad weights are dropped silently, as documented
    lc = LightCurve.from_arrays(t, mag, partly)
    assert lc.unusable_reason() is None
    raise_if_unusable(lc, "GLS")  # no raise
    # A band wiped out by its errors does not condemn the others.
    dead = LightCurve.from_arrays(t, mag, np.zeros(t.size))
    good = LightCurve.from_arrays(t, mag, err)
    assert (
        MultiBandLightCurve.from_light_curves({"g": dead, "r": good}).unusable_reason()
        is None
    )
    # No finite values at all is a different problem, left to the methods.
    empty = LightCurve.from_arrays(t, np.full(t.size, np.nan), np.zeros(t.size))
    assert empty.unusable_reason() is None


# --- compressed FITS ----------------------------------------------------------------


def test_from_file_reads_compressed_fits(tmp_path: Path) -> None:
    import bz2
    import gzip

    from astropy.table import Table

    t, mag, err = synthetic_sine(n=40)
    plain = tmp_path / "lc.fits"
    Table({"jd": t, "mag": mag, "mag_err": err}).write(plain, format="fits")
    raw = plain.read_bytes()
    (tmp_path / "lc.fits.gz").write_bytes(gzip.compress(raw))
    (tmp_path / "lc.fts.bz2").write_bytes(bz2.compress(raw))
    ref = LightCurve.from_file(plain)
    for name in ("lc.fits.gz", "lc.fts.bz2"):
        lc = LightCurve.from_file(tmp_path / name)
        assert np.array_equal(lc.time, ref.time)
        assert np.array_equal(lc.value, ref.value)
    csv = tmp_path / "lc.csv"
    Table({"jd": t, "mag": mag}).write(csv, format="ascii.csv")
    (tmp_path / "lc.csv.gz").write_bytes(gzip.compress(csv.read_bytes()))
    assert np.array_equal(LightCurve.from_file(tmp_path / "lc.csv.gz").time, t)


# --- a UTF-8 byte-order mark (Excel's "CSV UTF-8") ----------------------------------


@pytest.mark.parametrize(
    ("name", "sep"),
    [("lc.csv", ","), ("lc.tsv", "\t"), ("lc.dat", " "), ("lc.csv.gz", ",")],
)
def test_from_file_reads_a_file_with_a_byte_order_mark(
    tmp_path: Path, name: str, sep: str
) -> None:
    import gzip

    t, mag, err = synthetic_sine(n=40)
    lines = [sep.join(("time", "mag", "mag_err"))]
    lines += [
        sep.join(f"{v:.10g}" for v in row) for row in zip(t, mag, err, strict=True)
    ]
    raw = ("\n".join(lines) + "\n").encode("utf-8-sig")
    assert raw.startswith(b"\xef\xbb\xbf")
    path = tmp_path / name
    path.write_bytes(gzip.compress(raw) if name.endswith(".gz") else raw)
    lc = LightCurve.from_file(path)
    np.testing.assert_allclose(lc.time, t, rtol=1e-9)
    assert lc.domain is Domain.MAGNITUDE and lc.error is not None
    # A multiband file too (it loaded in 1.2 through pandas, which drops the mark).
    banded = path.with_name("bands" + name[2:])
    rows = [sep.join(("time", "mag", "mag_err", "band"))]
    rows += [
        sep.join((f"{a:.10g}", f"{b:.10g}", "0.01", "gr"[i % 2]))
        for i, (a, b) in enumerate(zip(t, mag, strict=True))
    ]
    raw = ("\n".join(rows) + "\n").encode("utf-8-sig")
    banded.write_bytes(gzip.compress(raw) if name.endswith(".gz") else raw)
    multi = MultiBandLightCurve.from_file(banded)
    assert sorted(multi.bands) == ["g", "r"]


# --- non-string column labels -------------------------------------------------------


def test_integer_labelled_frame_with_string_column_map() -> None:
    t, mag, err = synthetic_sine(n=60)
    df = pd.DataFrame({0: t, 1: mag, 2: err})
    cmap = ColumnMap(time="0", value="1", error="2")
    lc = LightCurve.from_dataframe(df, columns=cmap)
    assert np.array_equal(lc.time, t) and np.array_equal(lc.value, mag)
    assert lc.error is not None and np.array_equal(lc.error, err)
    as_ints = LightCurve.from_dataframe(
        df,
        columns=ColumnMap(time=0, value=1),  # type: ignore[arg-type]
    )
    assert np.array_equal(as_ints.value, mag)
    from_dict = cup.to_input({0: t, 1: mag}, columns=ColumnMap(time="0", value="1"))
    assert isinstance(from_dict, LightCurve)
    assert np.array_equal(from_dict.value, mag)
