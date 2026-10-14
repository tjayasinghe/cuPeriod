"""Batch pre-whitening: row layout, sinks, resume, and per-curve error isolation."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cuperiod as cup
from cuperiod.prewhiten.batch import batch_prewhiten, prewhiten_to_rows
from cuperiod.prewhiten.engine import prewhiten
from synth import synthetic_pulsator

#: The driver module itself: ``cuperiod.prewhiten`` as an attribute path is the
#: top-level ``cup.prewhiten`` function, so string targets cannot reach it.
PW_BATCH = importlib.import_module("cuperiod.prewhiten.batch")


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


def test_a_misspelt_domain_is_refused_before_any_work() -> None:
    # In-memory LightCurve inputs never read the domain, so a typo used to run the
    # whole catalogue and only bite on the first file or table input.
    with pytest.raises(ValueError, match="bogus"):
        batch_prewhiten(_curves(1), settings=_settings(), domain="bogus", workers=1)


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


class _Spy:
    """Stands in for ``prewhiten`` inside the batch driver: records, then fails fast."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, lc: object, *, settings: object, backend: object) -> None:
        self.calls.append({"settings": settings, "backend": backend})
        raise RuntimeError("spy: not computed")


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    recorder = _Spy()
    monkeypatch.setattr(PW_BATCH, "prewhiten", recorder)
    return recorder


# --- backend selection (BATCH-B-5, BATCH-B-6, PERF-PREWHITEN-04) ---------------


def test_settings_backend_is_used_when_backend_is_omitted() -> None:
    """PreWhitenSettings.backend was overridden by the batch default 'auto'."""
    summary = batch_prewhiten(
        _curves(1), settings=_settings(backend="numpy", max_frequencies=1), workers=1
    )
    assert summary.rows is not None and summary.rows
    assert {row["backend"] for row in summary.rows} == {"numpy"}


def test_explicit_backend_beats_the_settings_backend(spy: _Spy) -> None:
    batch_prewhiten(
        _curves(1), settings=_settings(backend="numpy"), backend="finufft", workers=1
    )
    assert spy.calls[0]["backend"] == "finufft"


def test_env_backend_is_honoured(monkeypatch: pytest.MonkeyPatch, spy: _Spy) -> None:
    monkeypatch.setenv("CUPERIOD_PREWHITEN_BACKEND", "numpy")
    batch_prewhiten(_curves(1), workers=1)
    assert spy.calls[0]["backend"] == "numpy"


def test_cpu_device_never_hands_auto_to_the_workers(
    monkeypatch: pytest.MonkeyPatch, spy: _Spy
) -> None:
    """On a CUDA box 'auto' resolves to cufinufft: a CPU pool must not ask for it."""
    from cuperiod.prewhiten import spectrum

    monkeypatch.setattr(
        spectrum, "available_backends", lambda: {"numpy", "finufft", "cufinufft"}
    )
    assert spectrum.resolve_spectrum_backend("auto") == "cufinufft"  # the fake works
    batch_prewhiten(_curves(1), workers=1)
    (call,) = spy.calls
    assert call["backend"] == "cpu"
    assert spectrum.resolve_spectrum_backend(str(call["backend"])) == "finufft"


def test_gpu_only_backend_on_a_cpu_pool_is_refused(spy: _Spy) -> None:
    with pytest.raises(ValueError, match="device='cpu'"):
        batch_prewhiten(_curves(1), backend="cufinufft", workers=1)
    with pytest.raises(ValueError, match="device='cpu'"):
        batch_prewhiten(_curves(1), settings=_settings(backend="gpu"), workers=1)
    assert spy.calls == []  # refused before any light curve was touched


def test_gpu_device_still_requests_the_gpu(
    monkeypatch: pytest.MonkeyPatch, spy: _Spy
) -> None:
    monkeypatch.setattr(PW_BATCH, "free_gpu_memory", lambda: None)
    batch_prewhiten(
        _curves(1), settings=_settings(backend="numpy"), device="gpu", workers=1
    )
    assert spy.calls[0]["backend"] == "gpu"


# --- file-sink resume (BATCH-B-7) ---------------------------------------------


def test_file_sink_resume_skips_curves_already_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = tmp_path / "modes.parquet"
    first = batch_prewhiten(_curves(2), settings=_settings(), workers=1, sink=sink)
    assert first.n_done > 0 and first.n_skipped == 0
    before = sink.read_bytes()

    recorder = _Spy()
    monkeypatch.setattr(PW_BATCH, "prewhiten", recorder)
    again = batch_prewhiten(_curves(2), settings=_settings(), workers=1, sink=sink)
    assert recorder.calls == []  # nothing recomputed
    assert (again.n_done, again.n_skipped, again.n_failed) == (0, 2, 0)
    assert sink.read_bytes() == before  # and the file is left alone


def test_file_sink_resume_computes_only_new_curves(tmp_path: Path) -> None:
    sink = tmp_path / "modes.csv"
    batch_prewhiten(_curves(2), settings=_settings(), workers=1, sink=sink)
    rows_before = len(pd.read_csv(sink))
    more = batch_prewhiten(_curves(3), settings=_settings(), workers=1, sink=sink)
    assert more.n_skipped == 2
    frame = pd.read_csv(sink)
    assert set(frame["key"]) == {"star0", "star1", "star2"}
    assert len(frame) == rows_before + more.n_done  # n_done counts rows written


def test_csv_resume_matches_formula_escaped_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CSV sink stores '=x' as "'=x"; resume must still recognise the curve."""
    _fake_prewhiten(monkeypatch)
    t = np.linspace(0.0, 10.0, 30)
    curves = [
        (key, cup.LightCurve.from_arrays(t, np.sin(t), meta={"id": i}))
        for i, key in enumerate(("=x", "-y", "plain"))
    ]
    sink = tmp_path / "modes.csv"
    batch_prewhiten(curves, workers=1, sink=sink)
    recorder = _Spy()
    monkeypatch.setattr(PW_BATCH, "prewhiten", recorder)
    again = batch_prewhiten(curves, workers=1, sink=sink)
    assert recorder.calls == [] and again.n_skipped == 3
    assert len(pd.read_csv(sink)) == 3


def test_file_sink_without_resume_recomputes(tmp_path: Path) -> None:
    sink = tmp_path / "modes.parquet"
    batch_prewhiten(_curves(1), settings=_settings(), workers=1, sink=sink)
    again = batch_prewhiten(
        _curves(1), settings=_settings(), workers=1, sink=sink, resume=False
    )
    assert again.n_skipped == 0 and again.n_done > 0
    assert len(pd.read_parquet(sink)) == again.n_done


# --- pool chunking, row order, thread pinning (PERF-PDOT-04) ------------------


class _InlinePool:
    """Replaces ``_run_pool``: runs chunks in-process, in *reverse* completion order."""

    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.max_workers = 0
        self.initargs: tuple[Any, ...] = ()
        self.pinned_env: str | None = None

    def __call__(
        self,
        chunks: list[list[Any]],
        pending: list[int],
        cfg: Any,
        absorb: Callable[..., None],
        max_workers: int,
        *,
        initializer: Any = None,
        initargs: tuple[Any, ...] = (),
        worker: Callable[..., tuple[Any, ...]],
    ) -> None:
        self.max_workers = max_workers
        self.initargs = initargs
        self.pinned_env = os.environ.get("OMP_NUM_THREADS")
        self.sizes = [len(chunks[idx]) for idx in pending]
        for idx in reversed(pending):
            absorb(idx, *worker(chunks[idx], cfg))


def _fake_prewhiten(monkeypatch: pytest.MonkeyPatch) -> None:
    """One cheap row per curve, so ordering is visible without any extraction."""
    monkeypatch.setattr(
        PW_BATCH, "prewhiten", lambda lc, *, settings, backend: lc.meta.get("id")
    )
    monkeypatch.setattr(
        PW_BATCH,
        "prewhiten_to_rows",
        lambda key, result, max_components=None: [{"key": key, "id": result}],
    )


def _tagged_curves(n: int) -> list[tuple[str, cup.LightCurve]]:
    t = np.linspace(0.0, 10.0, 30)
    return [
        (f"k{i:02d}", cup.LightCurve.from_arrays(t, np.sin(t), meta={"id": i}))
        for i in range(n)
    ]


def test_small_batches_are_split_across_every_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_prewhiten(monkeypatch)
    pool = _InlinePool()
    monkeypatch.setattr(PW_BATCH, "_run_pool", pool)
    monkeypatch.setattr(PW_BATCH, "cpu_worker_count", lambda w: 4)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    summary = batch_prewhiten(_tagged_curves(8), workers=4)
    assert len(pool.sizes) >= 4  # one 64-curve chunk used to feed a single worker
    assert sum(pool.sizes) == 8
    assert pool.max_workers == 4
    # Rows come back in input order although the chunks completed in reverse.
    assert summary.rows is not None
    assert [row["key"] for row in summary.rows] == [f"k{i:02d}" for i in range(8)]
    # BLAS pinned while the pool runs, numba capped per worker, parent restored.
    assert pool.pinned_env == "1"
    assert "OMP_NUM_THREADS" not in os.environ
    assert len(pool.initargs) == 1 and int(pool.initargs[0]) >= 1


def test_errors_come_back_in_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(lc: cup.LightCurve, *, settings: object, backend: object) -> None:
        raise RuntimeError(str(lc.meta.get("id")))

    monkeypatch.setattr(PW_BATCH, "prewhiten", fail)
    monkeypatch.setattr(PW_BATCH, "_run_pool", _InlinePool())
    monkeypatch.setattr(PW_BATCH, "cpu_worker_count", lambda w: 3)
    summary = batch_prewhiten(_tagged_curves(6), workers=3)
    assert [key for key, _ in summary.errors] == [f"k{i:02d}" for i in range(6)]


def test_directory_sink_keeps_the_requested_chunk_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Part files are named by chunk index; their layout must not follow workers."""
    _fake_prewhiten(monkeypatch)
    pool = _InlinePool()
    monkeypatch.setattr(PW_BATCH, "_run_pool", pool)
    monkeypatch.setattr(PW_BATCH, "cpu_worker_count", lambda w: 4)
    batch_prewhiten(_tagged_curves(8), workers=4, sink=tmp_path / "parts")
    assert pool.sizes == [8]


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


def test_directory_sink_resumes_with_another_chunk_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resume is by key, so a new chunk_size neither refuses nor duplicates."""
    import pyarrow.parquet as pq

    _fake_prewhiten(monkeypatch)
    sink = tmp_path / "parts"
    first = batch_prewhiten(_tagged_curves(4), workers=1, sink=sink, chunk_size=2)
    assert first.n_done == 4
    second = batch_prewhiten(_tagged_curves(7), workers=1, sink=sink, chunk_size=3)
    assert second.n_skipped == 4 and second.n_done == 3
    keys = sorted(
        row["key"]
        for part in sink.glob("part-*.parquet")
        for row in pq.read_table(part).to_pylist()
    )
    assert keys == [f"k{i:02d}" for i in range(7)]
