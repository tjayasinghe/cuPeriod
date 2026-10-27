"""Batch input resolution, result rows and file sinks (cuperiod.batch.io)."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import cuperiod as cup
from cuperiod.batch import io as bio
from cuperiod.batch.io import resolve_inputs
from synth import synthetic_eclipser, synthetic_multiband_sine, synthetic_sine


def _write_csv(path: Path, seed: int = 0, period: float = 0.6, n: int = 120) -> Path:
    t, mag, err = synthetic_sine(n=n, period=period, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"jd": t, "mag": mag, "mag_err": err}).to_csv(path, index=False)
    return path


# --- literal paths before globs (BATCH-CLI-5, SEC-15) -------------------------------


def test_paths_with_brackets_are_taken_literally(tmp_path: Path) -> None:
    folder = tmp_path / "survey [v2]"
    for i in range(2):
        _write_csv(folder / f"star{i}.csv", seed=i)
    for spec in (str(folder), folder):
        assert [k for k, _ in resolve_inputs(spec)] == ["star0", "star1"]
    assert [k for k, _ in resolve_inputs(folder / "star1.csv")] == ["star1"]
    # A pattern below the bracketed folder, and a genuine character class.
    assert len(resolve_inputs(str(folder / "*.csv"))) == 2
    assert len(resolve_inputs(str(folder / "star[01].csv"))) == 2
    summary = cup.batch_periodograms(str(folder), "GLS", workers=0)
    assert summary.n_done == 2


def test_a_directory_scan_takes_every_format_the_reader_takes(tmp_path: Path) -> None:
    """A folder scan skipped .tab / .fts / .fz files and compressed light curves,
    which LightCurve.from_file reads (an Open folder of them found nothing)."""
    import bz2
    import gzip

    from astropy.table import Table

    t, mag, err = synthetic_sine(n=60, seed=3)
    folder = tmp_path / "mixed"
    csv = _write_csv(folder / "a.csv", seed=1)
    (folder / "b.csv.gz").write_bytes(gzip.compress(csv.read_bytes()))
    table = Table({"jd": t, "mag": mag, "mag_err": err})
    table.write(folder / "c.tab", format="ascii.tab")
    table.write(tmp_path / "d.fits", format="fits")
    raw = (tmp_path / "d.fits").read_bytes()
    (folder / "d.fts").write_bytes(raw)
    (folder / "e.fits.gz").write_bytes(gzip.compress(raw))
    (folder / "f.dat.bz2").write_bytes(
        bz2.compress(b"jd mag mag_err\n1.0 12.0 0.1\n2.0 12.1 0.1\n")
    )
    (folder / "notes.md").write_text("not a light curve", encoding="utf-8")
    (folder / "archive.gz").write_bytes(gzip.compress(b"x"))
    # pyarrow cannot read a compressed Parquet file: left out, not a failed curve.
    (folder / "g.parquet.gz").write_bytes(gzip.compress(b"PAR1"))
    names = [p.name for _, p in resolve_inputs(folder)]
    assert names == ["a.csv", "b.csv.gz", "c.tab", "d.fts", "e.fits.gz", "f.dat.bz2"]
    for _, path in resolve_inputs(folder):
        assert isinstance(path, Path)
        assert cup.LightCurve.from_file(path).n >= 2


def test_a_glob_matching_nothing_raises_and_leaves_the_sink_alone(
    tmp_path: Path,
) -> None:
    _write_csv(tmp_path / "lcs" / "a.csv")
    out = tmp_path / "many.parquet"
    cup.batch_periodograms(str(tmp_path / "lcs" / "*.csv"), "GLS", workers=0, sink=out)
    before = out.read_bytes()
    with pytest.raises(FileNotFoundError, match="matched no files"):
        cup.batch_periodograms(
            str(tmp_path / "lcs" / "*.parqet"),
            "GLS",
            workers=0,
            sink=out,
            resume=False,
        )
    assert out.read_bytes() == before
    with pytest.raises(FileNotFoundError):
        resolve_inputs(tmp_path / "missing.csv")


def test_zero_inputs_leave_an_existing_sink_untouched(tmp_path: Path) -> None:
    out = tmp_path / "r.csv"
    cup.batch_periodograms(
        [("a", cup.LightCurve.from_arrays(*synthetic_sine(n=120)))],
        "GLS",
        workers=0,
        sink=out,
    )
    before = out.read_bytes()
    (tmp_path / "empty").mkdir()
    with pytest.warns(UserWarning, match="no inputs"):
        summary = cup.batch_periodograms(
            str(tmp_path / "empty"), "GLS", workers=0, sink=out, resume=False
        )
    assert summary.n_inputs == 0
    assert out.read_bytes() == before


# --- keys (BATCH-CLI-7, SEC-05, BATCH-CLI-11) ----------------------------------------


def test_colliding_file_stems_get_path_qualified_keys(tmp_path: Path) -> None:
    for field, seed in (("fieldA", 0), ("fieldB", 10)):
        for i in range(3):
            _write_csv(tmp_path / field / f"lc_{i:04d}.csv", seed=seed + i)
    _write_csv(tmp_path / "fieldA" / "solo.csv", seed=99)
    with pytest.warns(UserWarning, match="share a file stem"):
        items = resolve_inputs(str(tmp_path / "field*" / "*.csv"))
    keys = [k for k, _ in items]
    assert len(set(keys)) == len(keys) == 7
    assert "fieldA/lc_0000" in keys and "fieldB/lc_0000" in keys
    assert "solo" in keys  # a unique stem stays bare
    # Same folder, different suffix: the file name tells them apart.
    _write_csv(tmp_path / "mixed" / "a.csv")
    pd.read_csv(tmp_path / "mixed" / "a.csv").to_parquet(
        tmp_path / "mixed" / "a.parquet"
    )
    with pytest.warns(UserWarning):
        keys = [k for k, _ in resolve_inputs(tmp_path / "mixed")]
    assert sorted(keys) == ["a.csv", "a.parquet"]
    # Unique stems are untouched and raise no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert [k for k, _ in resolve_inputs(tmp_path / "fieldB")] == [
            "lc_0000",
            "lc_0001",
            "lc_0002",
        ]


def test_rows_record_their_source_and_a_clashing_resume_is_reported(
    tmp_path: Path,
) -> None:
    for field, period in (("fieldA", 0.6), ("fieldB", 1.3)):
        for i in range(2):
            _write_csv(tmp_path / field / f"lc_{i}.csv", seed=i, period=period)
    out = tmp_path / "E.parquet"
    cup.batch_periodograms(
        str(tmp_path / "fieldA" / "*.csv"), "GLS", workers=0, sink=out
    )
    rows = pq.read_table(out).to_pylist()
    assert {Path(r["source"]).parent.name for r in rows} == {"fieldA"}
    with pytest.warns(UserWarning, match="different file"):
        second = cup.batch_periodograms(
            str(tmp_path / "fieldB" / "*.csv"), "GLS", workers=0, sink=out
        )
    assert (second.n_done, second.n_skipped) == (0, 2)  # reported, not a silent done=2


def test_numpy_integer_keys_resolve_like_python_ints() -> None:
    lcs = [cup.LightCurve.from_arrays(*synthetic_sine(n=120, seed=i)) for i in range(3)]
    expected = ["101", "102", "103"]
    for dtype in (np.int64, np.int32, np.uint64):
        ids = np.array([101, 102, 103], dtype=dtype)
        assert [
            k for k, _ in resolve_inputs(list(zip(ids, lcs, strict=True)))
        ] == expected
    names = np.array(["a", "b", "c"])
    assert [k for k, _ in resolve_inputs(list(zip(names, lcs, strict=True)))] == list(
        "abc"
    )
    with pytest.raises(TypeError, match="float64"):
        resolve_inputs([(np.float64(1.5), lcs[0])])
    summary = cup.batch_periodograms(
        list(zip(np.array([101, 102, 103]), lcs, strict=True)), "GLS", workers=0
    )
    assert summary.rows is not None
    assert [r["key"] for r in summary.rows] == expected


def test_repeated_explicit_keys_warn() -> None:
    lc = cup.LightCurve.from_arrays(*synthetic_sine(n=120))
    with pytest.warns(UserWarning, match="more than once"):
        resolve_inputs([("x", lc), ("x", lc)])


# --- (DataFrame, group) honours ColumnMap(band=...) (BATCH-CLI-4) ---------------------


def _long_table() -> pd.DataFrame:
    frames = []
    for star in range(2):
        bands = synthetic_multiband_sine(
            band_points=(80, 80),
            amplitudes=(0.3, 0.3),
            offsets=(15.0, 15.8),
            seed=10 * star,
        )
        for band, (t, mag, err) in bands.items():
            frames.append(
                pd.DataFrame(
                    {"id": star, "band": band, "jd": t, "mag": mag, "mag_err": err}
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_dataframe_groups_honour_a_band_in_the_column_map(tmp_path: Path) -> None:
    df = _long_table()
    by_map = resolve_inputs((df, "id"), columns=cup.ColumnMap(band="band"))
    assert all(isinstance(src, cup.MultiBandLightCurve) for _, src in by_map)
    plain = resolve_inputs((df, "id"), columns=cup.ColumnMap(time="jd"))
    assert all(isinstance(src, cup.LightCurve) for _, src in plain)

    for star, sub in df.groupby("id"):
        sub.to_csv(tmp_path / f"{star}.csv", index=False)
    calls = {
        "map": dict(inputs=(df, "id"), columns=cup.ColumnMap(band="band")),
        "band_column": dict(inputs=(df, "id"), band_column="band"),
        "files": dict(
            inputs=str(tmp_path / "*.csv"), columns=cup.ColumnMap(band="band")
        ),
    }
    results = {}
    for name, kw in calls.items():
        summary = cup.batch_periodograms(method="GLS", workers=0, **kw)  # type: ignore[arg-type]
        assert summary.rows is not None and summary.n_failed == 0, name
        results[name] = [
            (r["key"], r["best_period"], r["power_1"]) for r in summary.rows
        ]
    assert results["map"] == results["band_column"] == results["files"]


# --- result rows (BATCH-CLI-6, PERF-METHODS-11) ---------------------------------------


def _small_flux_curve(n: int = 300) -> cup.LightCurve:
    t, flux, err = synthetic_eclipser(n=n, span=30.0)
    return cup.LightCurve.from_arrays(t, flux, err, domain="flux")


@pytest.mark.parametrize("method", ["GLS", "BLS", "TLS", "PDM"])
def test_declared_peak_extras_match_what_each_method_reports(method: str) -> None:
    pg = cup.periodogram(_small_flux_curve(), method, backend="cpu")
    assert tuple(pg.extras) == bio.PEAK_EXTRA_COLUMNS.get(method, ())


def test_rows_keep_their_columns_when_peaks_are_missing() -> None:
    pg = cup.periodogram(_small_flux_curve(), "BLS", backend="cpu")
    tiny = cup.Periodogram.from_spectrum(
        method="BLS",
        backend="numba",
        frequency=pg.frequency[:3],
        power=pg.power[:3],
        objective_sense="max",
        n_samples=pg.n_samples,
        baseline=pg.baseline,
        extras={k: v[:3] for k, v in pg.extras.items()},
    )
    keys = bio.peak_extra_columns(["GLS", "BLS"])
    full = bio.periodogram_to_row("a", pg, n_best=10, store_raw=False, extra_keys=keys)
    short = bio.periodogram_to_row(
        "b", tiny, n_best=10, store_raw=False, extra_keys=keys
    )
    assert list(full) == list(short)
    assert np.isnan(short["depth_10"]) and np.isnan(short["fap_1"])


# --- CSV formula injection (SEC-09) -------------------------------------------


def test_csv_safe_escapes_formulas_but_not_numbers() -> None:
    for cell in ("=1+1", "@SUM(1+1)", "-1+1", "+A1", "\t=x"):
        assert bio.csv_safe(cell) == "'" + cell
    for cell in ("-5", "+7", "star", "", "1e3", "'=quoted"):
        assert bio.csv_safe(cell) == cell
    assert bio.csv_safe(-5.0) == -5.0 and bio.csv_safe(None) is None
    assert bio.csv_safe(bio.csv_safe("=x")) == "'=x"  # idempotent


def test_csv_sinks_escape_keys_and_resume_without_growing(tmp_path: Path) -> None:
    lcs = [cup.LightCurve.from_arrays(*synthetic_sine(n=120, seed=i)) for i in range(3)]
    inputs = list(zip(["=1+1", "@SUM(1+1)", "-5"], lcs, strict=True))
    out = tmp_path / "r.csv"
    cup.batch_periodograms(inputs, "GLS", workers=0, sink=out)
    keys = pd.read_csv(out, dtype={"key": str})["key"].tolist()
    assert keys == ["'=1+1", "'@SUM(1+1)", "-5"]
    again = cup.batch_periodograms(inputs, "GLS", workers=0, sink=out)
    assert (again.n_done, again.n_skipped) == (0, 3)
    assert len(pd.read_csv(out)) == 3
    parquet = tmp_path / "r.parquet"
    cup.batch_periodograms(inputs, "GLS", workers=0, sink=parquet)
    assert pq.read_table(parquet).column("key").to_pylist() == [
        "=1+1",
        "@SUM(1+1)",
        "-5",
    ]


# --- unreadable sinks and the temporary file (SEC-13) -------------------------


def test_an_unreadable_csv_sink_is_moved_aside_not_overwritten(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    old = "key,method,best_period\nsté,GLS,1.0\n".encode("cp1252")
    out.write_bytes(old)
    lcs = [("nova0", cup.LightCurve.from_arrays(*synthetic_sine(n=120)))]
    with pytest.warns(UserWarning, match="moved to"):
        cup.batch_periodograms(lcs, "GLS", workers=0, sink=out)
    moved = list(tmp_path.glob("results.csv.corrupt-*"))
    assert len(moved) == 1 and moved[0].read_bytes() == old
    assert pd.read_csv(out)["key"].tolist() == ["nova0"]


def test_a_truncated_parquet_sink_is_moved_aside(tmp_path: Path) -> None:
    out = tmp_path / "results.parquet"
    out.write_bytes(b"PAR1 truncated")
    lcs = [("nova0", cup.LightCurve.from_arrays(*synthetic_sine(n=120)))]
    with pytest.warns(UserWarning, match="moved to"):
        cup.batch_periodograms(lcs, "GLS", workers=0, sink=out)
    assert len(list(tmp_path.glob("results.parquet.corrupt-*"))) == 1
    assert pq.read_table(out).column("key").to_pylist() == ["nova0"]


def test_an_io_error_reading_the_sink_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cuperiod.batch import runner

    out = tmp_path / "results.parquet"
    lcs = [
        (f"s{i}", cup.LightCurve.from_arrays(*synthetic_sine(n=120, seed=i)))
        for i in range(2)
    ]
    cup.batch_periodograms(lcs[:1], "GLS", workers=0, sink=out)
    before = out.read_bytes()

    def flaky(*args: object, **kwargs: object) -> None:
        raise OSError("transient network error")

    monkeypatch.setattr(runner, "_read_key_table", flaky)
    with pytest.raises(OSError, match="transient"):
        cup.batch_periodograms(lcs, "GLS", workers=0, sink=out)
    monkeypatch.undo()
    import pyarrow.parquet as pq_mod

    monkeypatch.setattr(pq_mod, "read_table", flaky)
    with pytest.raises(OSError, match="transient"):
        runner._finalize_file([{"key": "x", "method": "GLS"}], out, True)
    assert out.read_bytes() == before


def test_writes_never_reuse_a_fixed_temporary_name(tmp_path: Path) -> None:
    out = tmp_path / "r.parquet"
    planted = tmp_path / "r.parquet.tmp"
    planted.write_text("someone else's file")
    bio.write_rows([{"key": "a", "value": 1.0}], out)
    assert planted.read_text() == "someone else's file"
    assert pq.read_table(out).num_rows == 1
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_planted_symlink_is_not_written_through(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    out = tmp_path / "r.csv"
    (tmp_path / "r.csv.tmp").symlink_to(victim)
    bio.write_rows([{"key": "a"}], out)
    assert victim.read_text() == "precious"
    assert not out.is_symlink()
