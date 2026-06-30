"""Batch processing: in-memory, file/dir sinks, resume, pool, and globbing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

import cuperiod as cup
from synth import synthetic_sine


def _light_curves(count: int) -> list[cup.LightCurve]:
    out = []
    for i in range(count):
        t, mag, err = synthetic_sine(n=120, period=0.5 + 0.1 * i, seed=i)
        out.append(cup.LightCurve.from_arrays(t, mag, err, meta={"id": f"star{i}"}))
    return out


def test_batch_in_memory() -> None:
    summary = cup.batch_periodograms(_light_curves(3), "GLS", workers=1)
    assert summary.n_done == 3
    assert summary.n_failed == 0
    assert summary.rows is not None
    assert all("period_1" in row for row in summary.rows)


def test_batch_parquet_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "results.parquet"
    summary = cup.batch_periodograms(_light_curves(3), "GLS", workers=1, sink=out)
    assert out.exists()
    table = pq.read_table(out)
    assert table.num_rows == 3
    assert "best_period" in table.column_names
    assert summary.n_done == 3


def test_batch_directory_resume(tmp_path: Path) -> None:
    lcs = _light_curves(4)
    first = cup.batch_periodograms(lcs, "GLS", workers=1, sink=tmp_path, chunk_size=2)
    assert first.n_done == 4
    assert (tmp_path / "part-00000.parquet").exists()
    second = cup.batch_periodograms(lcs, "GLS", workers=1, sink=tmp_path, chunk_size=2)
    assert second.n_done == 0
    assert second.n_skipped == 4


def test_batch_process_pool() -> None:
    summary = cup.batch_periodograms(_light_curves(4), "GLS", workers=2, chunk_size=2)
    assert summary.n_done == 4
    assert summary.n_failed == 0


def test_batch_glob_csv(tmp_path: Path) -> None:
    for i in range(2):
        t, mag, err = synthetic_sine(n=120, seed=10 + i)
        pd.DataFrame({"jd": t, "mag": mag, "mag_err": err}).to_csv(
            tmp_path / f"star{i}.csv", index=False
        )
    summary = cup.batch_periodograms(str(tmp_path / "*.csv"), "GLS", workers=1)
    assert summary.n_done == 2
    assert summary.rows is not None


def test_batch_records_failures() -> None:
    bad = cup.LightCurve.from_arrays([1.0, 2.0], [1.0, 2.0], [0.1, 0.1])  # too few pts
    summary = cup.batch_periodograms([bad], "GLS", workers=1)
    assert summary.n_failed == 1
    assert summary.n_done == 0


def test_batch_store_raw(tmp_path: Path) -> None:
    out = tmp_path / "raw.parquet"
    cup.batch_periodograms(_light_curves(2), "GLS", workers=1, sink=out, store_raw=True)
    table = pq.read_table(out)
    assert "pgram_power" in table.column_names
    first = table.column("pgram_power")[0].as_py()
    assert isinstance(first, list) and len(first) > 0


def test_batch_file_sink_keeps_second_method(tmp_path: Path) -> None:
    # Regression (M3): adding a second method to an existing file sink must not drop it.
    lcs = _light_curves(2)
    out = tmp_path / "r.parquet"
    cup.batch_periodograms(lcs, "GLS", workers=1, sink=out)
    cup.batch_periodograms(lcs, "BLS", workers=1, sink=out)
    table = pq.read_table(out)
    assert set(table.column("method").to_pylist()) == {"GLS", "BLS"}
    assert table.num_rows == 4


def test_batch_dir_resume_chunksize_mismatch_raises(tmp_path: Path) -> None:
    # Regression (M4): a resumed dir sink with a different chunk_size would realign part
    # indices and silently drop/duplicate rows; refuse it.
    lcs = _light_curves(4)
    cup.batch_periodograms(lcs, "GLS", workers=1, sink=tmp_path, chunk_size=2)
    with pytest.raises(ValueError, match="chunk_size"):
        cup.batch_periodograms(lcs, "GLS", workers=1, sink=tmp_path, chunk_size=3)


def test_batch_csv_store_raw_rejected(tmp_path: Path) -> None:
    # Regression (M5): CSV can't hold raw spectrum arrays; must fail loudly.
    with pytest.raises(ValueError, match="CSV"):
        cup.batch_periodograms(
            _light_curves(1), "GLS", workers=1, sink=tmp_path / "r.csv", store_raw=True
        )


def test_batch_empty_methods_raises() -> None:
    with pytest.raises(ValueError, match="no methods"):
        cup.batch_periodograms(_light_curves(1), [], workers=1)
