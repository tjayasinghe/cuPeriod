"""Multi-band file loading: MultiBandLightCurve.from_file, CLI --band, batch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
