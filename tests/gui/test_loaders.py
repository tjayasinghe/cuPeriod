"""Tests for source enumeration and the demo/multiband loaders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cuperiod.core.lightcurve as core_lightcurve
from cuperiod.core.columns import ColumnMap
from cuperiod.core.errors import ColumnResolutionError
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.gui import loaders
from cuperiod.gui.loaders import (
    LOAD_NOTES_KEY,
    _read_dataframe,
    asassn_sources,
    demo_sources,
    enumerate_sources,
    example_data_dir,
    load_path,
    make_demo_multiband,
    preview_file,
)


def _write_single(path: Path, period: float, n: int = 200) -> None:
    time = np.linspace(0.0, 50.0, n)
    frame = pd.DataFrame(
        {
            "time": time,
            "mag": np.sin(2 * np.pi * time / period),
            "mag_err": np.full(n, 0.01),
        }
    )
    frame.to_csv(path, index=False)


def test_make_demo_multiband_bands() -> None:
    mblc = make_demo_multiband()
    assert isinstance(mblc, MultiBandLightCurve)
    assert set(mblc.band_names) == {"g", "r", "i"}


def test_enumerate_sources_directory(tmp_path: Path) -> None:
    for k, period in enumerate([1.5, 2.0, 3.0]):
        _write_single(tmp_path / f"lc_{k}.csv", period)
    items = enumerate_sources(str(tmp_path))
    assert len(items) == 3
    lc = items[0].loader()
    assert isinstance(lc, LightCurve)
    assert lc.n == 200


def test_enumerate_sources_dataframe_multiband() -> None:
    time = np.linspace(0.0, 50.0, 300)
    frame = pd.DataFrame(
        {
            "id": np.repeat(["A", "B"], 150),
            "band": np.tile(np.repeat(["g", "r"], 75), 2),
            "time": time,
            "mag": np.sin(time),
            "mag_err": np.full(300, 0.01),
        }
    )
    items = enumerate_sources((frame, "id"), band_column="band")
    assert len(items) == 2
    assert isinstance(items[0].loader(), MultiBandLightCurve)


def test_load_path_single_vs_multiband(tmp_path: Path) -> None:
    _write_single(tmp_path / "single.csv", 2.0)
    assert isinstance(load_path(tmp_path / "single.csv"), LightCurve)

    time = np.linspace(0.0, 50.0, 200)
    multi = pd.DataFrame(
        {
            "band": np.repeat(["g", "r"], 100),
            "time": time,
            "mag": np.sin(time),
            "mag_err": np.full(200, 0.01),
        }
    )
    multi.to_csv(tmp_path / "multi.csv", index=False)
    assert isinstance(load_path(tmp_path / "multi.csv"), MultiBandLightCurve)


def test_preview_file_detects_mapping(tmp_path: Path) -> None:
    _write_single(tmp_path / "p.csv", 2.0, n=50)
    preview = preview_file(tmp_path / "p.csv")
    assert preview is not None
    assert preview.mapping["time"] == "time"
    assert preview.mapping["value"] == "mag"
    assert preview.mapping["error"] == "mag_err"
    assert preview.n_rows == 50
    assert 0 < len(preview.rows) <= 12


def test_preview_file_only_reads_head_rows_for_csv(tmp_path: Path) -> None:
    path = tmp_path / "big.csv"
    _write_single(path, 2.0, n=5000)
    preview = preview_file(path, max_rows=12)
    assert preview is not None
    # the true row count is still reported accurately...
    assert preview.n_rows == 5000
    # ...even though only a small head was actually parsed
    assert len(preview.rows) == 12


def test_read_dataframe_nrows_limits_csv_parse(tmp_path: Path) -> None:
    path = tmp_path / "p.csv"
    _write_single(path, 2.0, n=200)
    frame = _read_dataframe(path, nrows=10)
    assert frame is not None
    assert len(frame) == 10
    full = _read_dataframe(path)
    assert len(full) == 200


def test_read_dataframe_logs_parse_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bad = tmp_path / "bad.csv"
    # a ragged row count reliably makes pandas' C parser raise ParserError
    bad.write_text("a,b,c\n1,2\n3,4,5,6,7\n", encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger="cuperiod.gui.loaders"):
        result = _read_dataframe(bad)
    assert result is None
    assert "bad.csv" in caplog.text


def test_asassn_adapter_when_available() -> None:
    data_dir = example_data_dir()
    parquet = None if data_dir is None else data_dir / "asassn_examples.parquet"
    if parquet is None or not parquet.is_file():
        pytest.skip("ASAS-SN demo data not present")
    items = asassn_sources(parquet)
    assert len(items) == 6
    lc = items[0].loader()
    assert isinstance(lc, LightCurve)
    assert lc.n > 0


# -- GUI-A-5: a band column splits the curve in every supported format --------------


def _band_table(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    band = np.where(rng.random(n) < 0.45, "g", "r")
    time = np.sort(rng.uniform(0.0, 100.0, n))
    mag = 15.0 + np.where(band == "g", 0.8, 0.0) + 0.2 * np.sin(time)
    return pd.DataFrame(
        {"time": time, "mag": mag, "mag_err": np.full(n, 0.01), "filter": band}
    )


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    from astropy.table import Table

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".tsv", ".tab"}:
        frame.to_csv(path, index=False, sep="\t")
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    elif suffix == ".ecsv":
        Table.from_pandas(frame).write(path, format="ascii.ecsv")
    elif suffix in {".fits", ".fit"}:
        Table.from_pandas(frame).write(path, format="fits")
    else:  # .dat / .txt
        Table.from_pandas(frame).write(path, format="ascii.basic")


_ALL_SUFFIXES = [
    ".csv",
    ".tsv",
    ".tab",
    ".parquet",
    ".pq",
    ".ecsv",
    ".fits",
    ".fit",
    ".dat",
    ".txt",
]


@pytest.mark.parametrize("suffix", _ALL_SUFFIXES)
def test_band_column_is_honoured_for_every_format(tmp_path: Path, suffix: str) -> None:
    frame = _band_table()
    path = tmp_path / f"mb{suffix}"
    _write_table(frame, path)
    lc = load_path(path)
    assert isinstance(lc, MultiBandLightCurve), suffix
    counts = frame["filter"].value_counts().to_dict()
    assert {name: band.n for name, band in lc.bands.items()} == counts
    assert lc.meta["source"] == str(path)


def test_space_after_comma_csv_is_still_multiband(tmp_path: Path) -> None:
    frame = _band_table(60)
    path = tmp_path / "spaced.csv"
    lines = [", ".join(frame.columns)]
    lines += [", ".join(str(v) for v in row) for row in frame.itertuples(index=False)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    lc = load_path(path)
    assert isinstance(lc, MultiBandLightCurve)
    assert set(lc.band_names) == {"g", "r"}
    preview = preview_file(path)
    assert preview is not None and preview.mapping.get("band") == "filter"


# -- GUI-A-6: blank, empty and padded band labels ------------------------------------


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_an_empty_band_column_loads_single_band(tmp_path: Path, suffix: str) -> None:
    frame = _band_table(100)
    frame["filter"] = None
    path = tmp_path / f"empty{suffix}"
    _write_table(frame, path)
    lc = load_path(path)
    assert isinstance(lc, LightCurve)
    assert lc.n == 100


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_unlabelled_rows_are_reported(tmp_path: Path, suffix: str) -> None:
    frame = _band_table(100)
    frame.loc[:39, "filter"] = None
    path = tmp_path / f"partial{suffix}"
    _write_table(frame, path)
    lc = load_path(path)
    assert isinstance(lc, MultiBandLightCurve)
    assert sum(band.n for band in lc.bands.values()) == 60
    notes = lc.meta[LOAD_NOTES_KEY]
    assert any("40" in note for note in notes)


def test_only_data_warnings_become_load_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A library's DeprecationWarning says nothing about the file: it must reach the
    # normal warning machinery, not the user's load notes.
    real = loaders._build_curve

    def noisy(*args: Any) -> Any:
        import warnings

        warnings.warn("old pandas API", DeprecationWarning, stacklevel=1)
        return real(*args)

    monkeypatch.setattr(loaders, "_build_curve", noisy)
    path = tmp_path / "plain.csv"
    _band_table(50).drop(columns="filter").to_csv(path, index=False)
    with pytest.warns(DeprecationWarning, match="old pandas API"):
        lc = load_path(path)
    assert LOAD_NOTES_KEY not in lc.meta


def test_padded_band_labels_merge(tmp_path: Path) -> None:
    frame = _band_table(100)
    frame.loc[::2, "filter"] = frame.loc[::2, "filter"] + " "
    path = tmp_path / "padded.csv"
    frame.to_csv(path, index=False)
    lc = load_path(path)
    assert isinstance(lc, MultiBandLightCurve)
    assert set(lc.band_names) == {"g", "r"}


# -- SEC-12: an id-like "band" column does not explode into thousands of bands ------


def test_too_many_band_labels_load_single_band(tmp_path: Path) -> None:
    frame = _band_table(400)
    frame["filter"] = [f"exp{i:05d}" for i in range(400)]  # an exposure id, per row
    path = tmp_path / "ids.csv"
    frame.to_csv(path, index=False)
    lc = load_path(path)
    assert isinstance(lc, LightCurve)
    assert lc.n == 400
    assert any("400" in note for note in lc.meta[LOAD_NOTES_KEY])
    # an explicitly chosen band column is the user's call and is kept
    pinned = load_path(path, columns=ColumnMap(band="filter"))
    assert isinstance(pinned, MultiBandLightCurve)
    assert pinned.n_bands == 400


# -- GUI-A-21: one full parse per file -------------------------------------------------


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_single_band_file_is_read_once(
    tmp_path: Path, suffix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _band_table(200).drop(columns="filter")
    path = tmp_path / f"single{suffix}"
    _write_table(frame, path)
    reference = LightCurve.from_file(path)

    reads: list[str] = []
    real = core_lightcurve._read_table_object
    real_csv, real_parquet = pd.read_csv, pd.read_parquet

    def counting(p: Path) -> Any:
        reads.append(f"table:{Path(p).name}")
        return real(p)

    def pandas_csv(*args: Any, **kwargs: Any) -> Any:
        reads.append("pandas.read_csv")
        return real_csv(*args, **kwargs)

    def pandas_parquet(*args: Any, **kwargs: Any) -> Any:
        reads.append("pandas.read_parquet")
        return real_parquet(*args, **kwargs)

    monkeypatch.setattr(core_lightcurve, "_read_table_object", counting)
    monkeypatch.setattr(pd, "read_csv", pandas_csv)
    monkeypatch.setattr(pd, "read_parquet", pandas_parquet)
    lc = load_path(path)
    assert reads == [f"table:{path.name}"]  # one parse, no pandas probe first
    assert isinstance(lc, LightCurve)
    np.testing.assert_array_equal(lc.time, reference.time)
    np.testing.assert_array_equal(lc.value, reference.value)
    assert lc.error is not None and reference.error is not None
    np.testing.assert_array_equal(lc.error, reference.error)


# -- GUI-B-16: wide (one row per object) tables get a clear message -----------------


def _wide_parquet(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "jd": [np.arange(5.0) + 2458000.0, np.arange(7.0) + 2458000.0],
            "mag": [np.ones(5), np.ones(7)],
            "mag_err": [np.full(5, 0.01), np.full(7, 0.01)],
        }
    )
    frame.to_parquet(path, index=False)


def test_wide_table_load_raises_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "wide.parquet"
    _wide_parquet(path)
    with pytest.raises(ColumnResolutionError, match="nested|wide"):
        load_path(path)


def test_wide_table_preview_is_not_loadable(tmp_path: Path) -> None:
    path = tmp_path / "wide.parquet"
    _wide_parquet(path)
    preview = preview_file(path)
    assert preview is not None
    assert not preview.loadable
    assert "wide" in preview.note or "nested" in preview.note
    assert preview.rows[0][1] == "[5 values]"


# -- GUI-A-22 / GUI-B-16 / UX-21: the preview shows what is in the file -------------


def test_preview_shows_full_precision_times_and_exact_ids(tmp_path: Path) -> None:
    path = tmp_path / "jd.csv"
    path.write_text(
        "source_id,hjd,mag,mag_err\n"
        "4567890123456789012,2458000.123456,12.3451,0.01\n"
        "4567890123456789012,2458000.987654,12.3452,0.01\n"
        "4567890123456789012,2458003.5,12.3453,0.01\n",
        encoding="utf-8",
    )
    preview = preview_file(path)
    assert preview is not None
    times = [row[1] for row in preview.rows]
    assert times == ["2458000.123456", "2458000.987654", "2458003.5"]
    assert preview.rows[0][0] == "4567890123456789012"
    assert preview.rows[0][2] == "12.3451"


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_preview_row_count_ignores_blank_lines(tmp_path: Path, newline: str) -> None:
    path = tmp_path / "blank.csv"
    path.write_bytes(
        newline.join(["time,mag", "1,2", "", "3,4", "", "", ""]).encode("ascii")
    )
    preview = preview_file(path)
    assert preview is not None
    assert preview.n_rows == 2
    assert preview.loadable


def test_header_only_file_preview_is_not_loadable(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("time,mag,mag_err\n", encoding="utf-8")
    preview = preview_file(path)
    assert preview is not None
    assert preview.n_rows == 0
    assert not preview.loadable
    assert "no data rows" in preview.note


# -- GUI-A-7: folders whose name contains glob characters -----------------------------


def test_enumerate_sources_in_a_bracketed_directory(tmp_path: Path) -> None:
    folder = tmp_path / "TESS [sector 5]"
    folder.mkdir()
    for k, period in enumerate([1.5, 2.0]):
        _write_single(folder / f"lc_{k}.csv", period)
    assert len(enumerate_sources(str(folder))) == 2
    assert len(enumerate_sources(folder)) == 2
    single = enumerate_sources(str(folder / "lc_0.csv"))
    assert len(single) == 1 and isinstance(single[0].loader(), LightCurve)
    # a real glob still works
    assert len(enumerate_sources(str(tmp_path / "TESS*" / "*.csv"))) == 2


def test_enumerate_sources_disambiguates_equal_stems(tmp_path: Path) -> None:
    _write_single(tmp_path / "tgt.csv", 2.0)
    frame = pd.read_csv(tmp_path / "tgt.csv")
    frame.to_csv(tmp_path / "tgt.tsv", index=False, sep="\t")
    items = enumerate_sources(str(tmp_path))
    assert len(items) == 2
    assert len({item.key for item in items}) == 2
    assert sorted(item.label for item in items) == ["tgt.csv", "tgt.tsv"]


# -- SEC-07: demo data is looked up in the checkout only, and never blocks start-up -


def test_example_data_dir_does_not_walk_up_from_an_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUPERIOD_EXAMPLE_DATA", raising=False)
    (tmp_path / "examples" / "data").mkdir(parents=True)  # planted in an ancestor
    install = tmp_path / "env" / "Lib" / "site-packages" / "cuperiod" / "gui"
    install.mkdir(parents=True)
    monkeypatch.setattr(loaders, "__file__", str(install / "loaders.py"))
    assert example_data_dir() is None


def test_example_data_dir_finds_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUPERIOD_EXAMPLE_DATA", raising=False)
    gui_dir = tmp_path / "src" / "cuperiod" / "gui"
    gui_dir.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'cuperiod'\n")
    (tmp_path / "examples" / "data").mkdir(parents=True)
    monkeypatch.setattr(loaders, "__file__", str(gui_dir / "loaders.py"))
    found = example_data_dir()
    assert found is not None
    assert found.resolve() == (tmp_path / "examples" / "data").resolve()


def test_a_corrupt_demo_file_does_not_break_the_demo_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "asassn_examples.parquet").write_bytes(b"not a parquet file")
    monkeypatch.setenv("CUPERIOD_EXAMPLE_DATA", str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="cuperiod.gui.loaders"):
        items = demo_sources()
    assert [item.key for item in items] == ["demo_multiband"]
    assert "asassn_examples.parquet" in caplog.text
