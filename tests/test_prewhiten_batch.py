"""Batch pre-whitening: row layout, sinks, resume, and per-curve error isolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cuperiod as cup
from cuperiod.prewhiten.batch import batch_prewhiten, prewhiten_to_rows
from cuperiod.prewhiten.engine import prewhiten
from synth import synthetic_pulsator


def _settings(**overrides) -> cup.PreWhitenSettings:
    defaults = {"backend": "finufft", "max_frequencies": 3, "store_spectra": False}
    return cup.PreWhitenSettings(**{**defaults, **overrides})


def _curves(n: int = 3) -> list[tuple[str, cup.LightCurve]]:
    out = []
    for index in range(n):
        time, value, error = synthetic_pulsator(n=500, span=20.0, seed=index)
        out.append((f"star{index}", cup.LightCurve.from_arrays(time, value, error)))
    return out


def test_rows_are_one_per_component_with_a_repeated_summary() -> None:
    time, value, error = synthetic_pulsator(n=600)
    solution = prewhiten((time, value, error), settings=_settings())
    rows = prewhiten_to_rows("star", solution)
    assert len(rows) == solution.n_components
    assert {row["key"] for row in rows} == {"star"}
    assert all(row["n_components"] == solution.n_components for row in rows)
    assert rows[0]["label"] == "F1"
    assert rows[0]["frequency"] == pytest.approx(solution.components[0].frequency)


def test_a_curve_with_no_components_still_produces_a_row() -> None:
    rng = np.random.default_rng(7)
    time = np.sort(rng.uniform(0.0, 27.0, 800)) + 2458000.0
    noise = 10.0 + rng.normal(0.0, 0.002, 800)
    solution = prewhiten(
        (time, noise, np.full(800, 0.002)), settings=_settings(snr_threshold=6.0)
    )
    assert solution.n_components == 0
    (row,) = prewhiten_to_rows("quiet", solution)
    assert row["key"] == "quiet" and row["n_components"] == 0
    # Missing cells are NaN / "" rather than None, so a Parquet part written from an
    # all-quiet chunk infers the same column types as any other chunk.
    assert np.isnan(row["frequency"]) and row["combination"] == ""


def test_max_components_truncates_the_rows() -> None:
    time, value, error = synthetic_pulsator(n=600)
    solution = prewhiten((time, value, error), settings=_settings())
    rows = prewhiten_to_rows("star", solution, max_components=1)
    assert len(rows) == 1 and rows[0]["label"] == "F1"


def test_in_memory_batch_covers_every_input() -> None:
    summary = batch_prewhiten(_curves(), settings=_settings(), workers=1)
    assert summary.n_inputs == 3
    assert summary.n_failed == 0
    assert summary.methods == ("PREWHITEN",)
    assert summary.rows is not None
    assert {row["key"] for row in summary.rows} == {"star0", "star1", "star2"}


def test_parquet_sink_round_trips(tmp_path: Path) -> None:
    sink = tmp_path / "modes.parquet"
    summary = batch_prewhiten(_curves(2), settings=_settings(), workers=1, sink=sink)
    assert sink.exists() and summary.n_done > 0
    frame = pd.read_parquet(sink)
    assert set(frame["key"]) == {"star0", "star1"}
    assert {"frequency", "frequency_error", "snr", "stop_reason"} <= set(frame.columns)


def test_directory_sink_is_resumable(tmp_path: Path) -> None:
    sink = tmp_path / "parts"
    first = batch_prewhiten(
        _curves(2), settings=_settings(), workers=1, sink=sink, chunk_size=1
    )
    assert first.n_skipped == 0
    assert len(list(sink.glob("part-*.parquet"))) == 2
    second = batch_prewhiten(
        _curves(2), settings=_settings(), workers=1, sink=sink, chunk_size=1
    )
    assert second.n_skipped == 2 and second.n_done == 0


def test_a_bad_light_curve_is_reported_not_raised() -> None:
    good = _curves(1)
    broken = cup.LightCurve.from_arrays(
        np.full(40, 2458000.0), np.ones(40), np.full(40, 0.01)
    )
    summary = batch_prewhiten(
        [*good, ("broken", broken)], settings=_settings(), workers=1
    )
    assert summary.n_failed == 1
    assert summary.errors[0][0] == "broken"
    assert summary.rows is not None and len(summary.rows) > 0  # the good one survived


def test_multiband_input_is_reported_as_an_error() -> None:
    time, value, error = synthetic_pulsator(n=300)
    mblc = cup.MultiBandLightCurve.from_light_curves(
        {"V": cup.LightCurve.from_arrays(time, value, error)}
    )
    summary = batch_prewhiten([("mb", mblc)], settings=_settings(), workers=1)
    assert summary.n_failed == 1
    assert "single-band" in summary.errors[0][1]


def test_an_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="cpu"):
        batch_prewhiten(_curves(1), device="tpu")


def test_top_level_batch_entry_point_exists() -> None:
    summary = cup.batch_prewhiten(_curves(1), settings=_settings(), workers=1)
    assert summary.n_inputs == 1


def test_directory_sink_parts_share_one_schema(tmp_path: Path) -> None:
    # Regression: component fields were emitted as Python None, so a chunk in which no
    # star had an identified combination typed that column `null` while a later chunk
    # typed it `string` — and a pyarrow dataset takes its schema from the first
    # fragment, making the whole directory unreadable.
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    rng = np.random.default_rng(3)
    quiet_time = np.sort(rng.uniform(0.0, 27.0, 600)) + 2458000.0
    quiet = cup.LightCurve.from_arrays(
        quiet_time, 10.0 + rng.normal(0.0, 0.002, 600), np.full(600, 0.002)
    )
    time, value, error = synthetic_pulsator(n=600, span=20.0)
    loud = cup.LightCurve.from_arrays(time, value, error)

    sink = tmp_path / "parts"
    # Chunk 0 yields no components at all; chunk 1 yields several with a combination.
    summary = batch_prewhiten(
        [("quiet", quiet), ("loud", loud)],
        settings=_settings(snr_threshold=8.0),
        workers=1,
        sink=sink,
        chunk_size=1,
    )
    assert summary.n_failed == 0
    parts = sorted(sink.glob("part-*.parquet"))
    assert len(parts) == 2
    schemas = [pq.read_schema(p) for p in parts]
    assert schemas[0] == schemas[1]
    assert pa.types.is_string(schemas[0].field("combination").type)  # never null-typed
    assert pa.types.is_floating(schemas[0].field("frequency").type)
    table = ds.dataset(sink, format="parquet").to_table()
    assert set(table.column("key").to_pylist()) == {"quiet", "loud"}


def test_component_cells_are_stably_typed() -> None:
    time, value, error = synthetic_pulsator(n=600)
    solution = prewhiten((time, value, error), settings=_settings())
    (filled,) = prewhiten_to_rows("star", solution, max_components=1)
    rng = np.random.default_rng(7)
    quiet_time = np.sort(rng.uniform(0.0, 27.0, 600)) + 2458000.0
    empty_solution = prewhiten(
        (quiet_time, 10.0 + rng.normal(0.0, 0.002, 600), np.full(600, 0.002)),
        settings=_settings(snr_threshold=8.0),
    )
    (empty,) = prewhiten_to_rows("quiet", empty_solution)
    assert set(filled) == set(empty)
    for name in filled:
        assert type(filled[name]) is type(empty[name]), name
    assert isinstance(filled["rank"], float)  # float, so NaN can fill it
    assert filled["combination"] == "" or isinstance(filled["combination"], str)
    assert empty["label"] == "" and np.isnan(empty["frequency"])
