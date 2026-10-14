"""Multi-band file loading: MultiBandLightCurve.from_file, CLI --band, batch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import cuperiod as cup
from cuperiod.cli.app import app
from cuperiod.core.errors import ColumnResolutionError
from synth import synthetic_multiband_sine

runner = CliRunner()
PERIOD = 0.7365


def write_long_csv(
    path: Path, band_points: tuple[int, ...] = (50, 40)
) -> dict[str, int]:
    """A long-format two-band CSV; returns the per-band point counts."""
    bands = synthetic_multiband_sine(
        band_points=band_points, amplitudes=(0.3, 0.2), offsets=(15.0, 14.2),
        period=PERIOD,
    )
    lines = ["jd,mag,mag_err,band"]
    counts = {}
    for name, (t, m, e) in bands.items():
        counts[name] = t.size
        lines += [
            f"{ti},{mi},{ei},{name}"
            for ti, mi, ei in zip(t, m, e, strict=True)
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return counts


def test_from_file_splits_bands(tmp_path: Path) -> None:
    csv = tmp_path / "star.csv"
    counts = write_long_csv(csv)
    mb = cup.MultiBandLightCurve.from_file(csv, band_column="band")
    assert mb.band_names == ("b0", "b1")
    for name, lc in mb.bands.items():
        assert lc.n == counts[name]
        assert lc.error is not None
        assert lc.meta["band"] == name
    assert mb.meta["source"] == str(csv)


def test_from_file_autodetects_band_column(tmp_path: Path) -> None:
    csv = tmp_path / "star.csv"
    write_long_csv(csv)
    mb = cup.MultiBandLightCurve.from_file(csv)
    assert mb.band_names == ("b0", "b1")


def test_from_file_parquet_roundtrip(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    bands = synthetic_multiband_sine(band_points=(30, 20), amplitudes=(0.3, 0.2),
                                     offsets=(15.0, 14.2))
    t = np.concatenate([bands["b0"][0], bands["b1"][0]])
    m = np.concatenate([bands["b0"][1], bands["b1"][1]])
    e = np.concatenate([bands["b0"][2], bands["b1"][2]])
    label = np.array(["g"] * 30 + ["r"] * 20)
    table = pa.table({"mjd": t, "mag": m, "magerr": e, "filter": label})
    path = tmp_path / "star.parquet"
    pq.write_table(table, path)
    mb = cup.MultiBandLightCurve.from_file(path, band_column="filter")
    assert mb.band_names == ("g", "r")
    assert mb.bands["g"].n == 30 and mb.bands["r"].n == 20


def test_from_dataframe_band_column_with_explicit_columns() -> None:
    # band_column must survive an explicit ColumnMap that pins only time/value:
    # the caller has no way to spell the band otherwise than the keyword.
    import pandas as pd

    bands = synthetic_multiband_sine(band_points=(30, 20), amplitudes=(0.3, 0.2),
                                     offsets=(15.0, 14.2))
    df = pd.DataFrame(
        {
            "t_obs": np.concatenate([bands["b0"][0], bands["b1"][0]]),
            "brightness": np.concatenate([bands["b0"][1], bands["b1"][1]]),
            "survey_filter": np.array(["g"] * 30 + ["r"] * 20),
        }
    )
    mb = cup.MultiBandLightCurve.from_dataframe(
        df,
        band_column="survey_filter",
        columns=cup.ColumnMap(time="t_obs", value="brightness"),
    )
    assert mb.band_names == ("g", "r")
    assert mb.bands["g"].n == 30 and mb.bands["r"].n == 20


def _long_frame(labels: Sequence[object]) -> pd.DataFrame:
    n = len(labels)
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "hjd": np.sort(rng.uniform(0.0, 100.0, n)),
            "mag": 15.0 + rng.normal(0.0, 0.05, n),
            "mag_err": np.full(n, 0.05),
            "band": labels,
        }
    )


def _band_sizes(mb: cup.MultiBandLightCurve) -> dict[str, int]:
    return {name: lc.n for name, lc in mb.bands.items()}


def test_mixed_type_labels_merge_instead_of_overwriting() -> None:
    # 1 and "1" print the same: they used to share a dict key, so the later group
    # silently overwrote the earlier one and 300 rows vanished.
    df = _long_frame([1] * 300 + ["1"] * 300 + [2] * 300)
    mb = cup.MultiBandLightCurve.from_dataframe(df, band_column="band")
    assert _band_sizes(mb) == {"1": 600, "2": 300}
    assert sum(_band_sizes(mb).values()) == 900


def test_dirty_labels_agree_across_constructors(tmp_path: Path) -> None:
    # Padded labels merge with their clean spelling and blank cells are dropped with
    # a warning -- identically whether the table arrives as a frame, a CSV or a
    # Parquet file (from_file used to invent a phantom band 'N' or 'None').
    labels: list[object] = ["g"] * 300 + ["g "] * 300 + [None] * 300
    df = _long_frame(labels)
    csv = tmp_path / "labels.csv"
    df.to_csv(csv, index=False)
    parquet = tmp_path / "labels.parquet"
    df.to_parquet(parquet)
    expected = {"g": 600}
    with pytest.warns(UserWarning, match="300 of 900 rows"):
        assert _band_sizes(
            cup.MultiBandLightCurve.from_dataframe(df, band_column="band")
        ) == expected
    for path in (csv, parquet):
        with pytest.warns(UserWarning, match="300 of 900 rows"):
            assert _band_sizes(
                cup.MultiBandLightCurve.from_file(path, band_column="band")
            ) == expected


def test_categorical_parquet_nulls_are_not_assigned_a_band(tmp_path: Path) -> None:
    # pandas categoricals become dictionary-encoded Parquet columns, whose
    # to_numpy() hands a null the value of whatever category its index slot holds.
    df = _long_frame(["g", None, "r", "g"] * 30)
    df["band"] = df["band"].astype("category")
    path = tmp_path / "cat.parquet"
    df.to_parquet(path)
    for load in (
        lambda: cup.MultiBandLightCurve.from_file(path),
        lambda: cup.MultiBandLightCurve.from_dataframe(df),
    ):
        with pytest.warns(UserWarning, match="30 of 120 rows"):
            assert _band_sizes(load()) == {"g": 60, "r": 30}


def test_blank_cells_do_not_join_a_real_band_named_n(tmp_path: Path) -> None:
    df = _long_frame(["N"] * 300 + ["V"] * 300 + [None] * 300)
    csv = tmp_path / "labels.csv"
    df.to_csv(csv, index=False)
    with pytest.warns(UserWarning, match="dropped"):
        mb = cup.MultiBandLightCurve.from_file(csv, band_column="band")
    assert _band_sizes(mb) == {"N": 300, "V": 300}


def test_integer_ids_with_blanks_agree(tmp_path: Path) -> None:
    # An integer filter-id column with blank cells is float in pandas (1.0, 2.0) and
    # masked int in astropy: both must name the bands '1' and '2'.
    df = _long_frame([1] * 300 + [2] * 300 + [None] * 300)
    csv = tmp_path / "fid.csv"
    df.to_csv(csv, index=False)
    with pytest.warns(UserWarning, match="dropped"):
        from_frame = cup.MultiBandLightCurve.from_dataframe(
            pd.read_csv(csv), band_column="band"
        )
    with pytest.warns(UserWarning, match="dropped"):
        from_file = cup.MultiBandLightCurve.from_file(csv, band_column="band")
    assert _band_sizes(from_frame) == _band_sizes(from_file) == {"1": 300, "2": 300}


def test_from_dataframe_integer_column_labels() -> None:
    df = _long_frame(["g"] * 30 + ["r"] * 20)
    df.columns = pd.Index([0, 1, 2, 3])
    mb = cup.MultiBandLightCurve.from_dataframe(
        df, band_column="3", columns=cup.ColumnMap(time="0", value="1", error="2")
    )
    assert _band_sizes(mb) == {"g": 30, "r": 20}


def test_from_dataframe_accepts_any_table() -> None:
    from astropy.table import Table

    df = _long_frame(["g"] * 30 + ["r"] * 20)
    mb = cup.MultiBandLightCurve.from_dataframe(Table.from_pandas(df))  # type: ignore[arg-type]
    assert _band_sizes(mb) == {"g": 30, "r": 20}


def test_from_dataframe_explicit_magnitude_reads_mag_column() -> None:
    df = _long_frame(["g"] * 30 + ["r"] * 20)
    df["flux"] = 10 ** (-0.4 * (df["mag"] - 25.0))
    df["flux_err"] = 0.05 * df["flux"]
    mb = cup.MultiBandLightCurve.from_dataframe(df, domain=cup.Domain.MAGNITUDE)
    for name, lc in mb.bands.items():
        assert lc.domain is cup.Domain.MAGNITUDE
        assert np.array_equal(lc.value, df.loc[df["band"] == name, "mag"].to_numpy())
    flux = cup.MultiBandLightCurve.from_dataframe(df)
    assert all(lc.domain is cup.Domain.FLUX for lc in flux.bands.values())


def test_from_file_reads_compressed_fits(tmp_path: Path) -> None:
    import gzip

    from astropy.table import Table

    df = _long_frame(["g"] * 30 + ["r"] * 20)
    plain = tmp_path / "star.fits"
    Table.from_pandas(df).write(plain, format="fits")
    packed = tmp_path / "star.fits.gz"
    packed.write_bytes(gzip.compress(plain.read_bytes()))
    mb = cup.MultiBandLightCurve.from_file(packed)
    assert _band_sizes(mb) == {"g": 30, "r": 20}


def test_from_file_without_band_column_raises(tmp_path: Path) -> None:
    csv = tmp_path / "noband.csv"
    csv.write_text("jd,mag\n1.0,12.0\n2.0,12.1\n", encoding="utf-8")
    with pytest.raises(ColumnResolutionError):
        cup.MultiBandLightCurve.from_file(csv)


def test_cli_run_with_band_is_multiband(tmp_path: Path) -> None:
    csv = tmp_path / "star.csv"
    write_long_csv(csv)
    result = runner.invoke(app, ["run", str(csv), "--band", "band", "-m", "GLS"])
    assert result.exit_code == 0, result.output
    # 90 stacked points prove both bands entered the fit.
    assert "n=90" in result.output


def test_batch_file_inputs_go_multiband(tmp_path: Path) -> None:
    # Dense enough that the default pseudo-Nyquist grid reaches the true
    # frequency (1.36 c/d needs > 90 points over the 180-day span).
    for star in range(2):
        write_long_csv(tmp_path / f"star{star}.csv", band_points=(90, 70))
    summary = cup.batch_periodograms(
        str(tmp_path / "*.csv"),
        method="GLS",
        backend="cpu",
        band_column="band",
    )
    assert summary.n_done == 2 and summary.n_failed == 0
    assert summary.rows is not None
    for row in summary.rows:
        assert row["n_samples"] == 160
        assert row["best_period"] == pytest.approx(PERIOD, rel=2e-3)


# --- a band column read as one curve is flagged (CORE-A-6) --------------------------


def _alternating_filters() -> pd.DataFrame:
    t = np.linspace(0.0, 30.0, 60)
    return pd.DataFrame(
        {
            "mjd": t,
            "mag": 15.0 + 0.1 * np.sin(t),
            "magerr": np.full(t.size, 0.02),
            "filter": np.where(np.arange(t.size) % 2, "g", "r"),
        }
    )


def test_single_band_read_of_a_multi_label_band_column_warns(tmp_path: Path) -> None:
    # Pooling g and r into one curve corrupts every period search (two zero points);
    # it used to happen silently.
    frame = _alternating_filters()
    path = tmp_path / "long.csv"
    frame.to_csv(path, index=False)
    for read in (
        lambda: cup.LightCurve.from_dataframe(frame),
        lambda: cup.LightCurve.from_file(path),
        lambda: cup.to_input(frame.to_dict(orient="list")),
    ):
        with pytest.warns(UserWarning, match=r"'filter'.*MultiBandLightCurve"):
            lc = read()
        assert lc.n == 60  # still read as asked: one pooled curve


def test_multiband_split_and_one_label_tables_do_not_warn() -> None:
    import warnings

    frame = _alternating_filters()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mblc = cup.MultiBandLightCurve.from_dataframe(frame)
        assert mblc.band_names == ("r", "g")
        one = cup.LightCurve.from_dataframe(frame.assign(filter="g"))
        assert one.n == 60


# --- MultiBandLightCurve validates its bands (CORE-A-18) ----------------------------


def test_multiband_rejects_a_non_lightcurve_band() -> None:
    t = np.linspace(0.0, 10.0, 20)
    bad: dict[str, object] = {"g": (t, np.sin(t), None)}
    with pytest.raises(TypeError, match="'g'"):
        cup.MultiBandLightCurve(bands=bad)  # type: ignore[arg-type]


def test_multiband_rejects_mixed_brightness_domains() -> None:
    t = np.linspace(0.0, 10.0, 20)
    mag = cup.LightCurve.from_arrays(t, 15.0 + np.sin(t))
    with pytest.raises(ValueError, match=r"g: magnitude, r: flux.*domain="):
        cup.MultiBandLightCurve.from_light_curves({"g": mag, "r": mag.as_flux()})
    same = cup.MultiBandLightCurve.from_light_curves(
        {"g": mag.as_flux(), "r": mag.as_flux()}
    )
    assert same.n_bands == 2
