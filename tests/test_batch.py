"""Batch processing: in-memory, file/dir sinks, resume, pool, and globbing."""

from __future__ import annotations

import json
import multiprocessing
from collections import Counter
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import cuperiod as cup
from conftest import requires_gpu
from cuperiod.batch import runner
from synth import synthetic_eclipser, synthetic_sine


def _light_curves(count: int) -> list[cup.LightCurve]:
    out = []
    for i in range(count):
        t, mag, err = synthetic_sine(n=120, period=0.5 + 0.1 * i, seed=i)
        out.append(cup.LightCurve.from_arrays(t, mag, err, meta={"id": f"star{i}"}))
    return out


def _write_csv(path: Path, seed: int = 0, n: int = 120) -> Path:
    t, mag, err = synthetic_sine(n=n, period=0.5 + 0.05 * seed, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"jd": t, "mag": mag, "mag_err": err}).to_csv(path, index=False)
    return path


def _dir_pairs(sink: Path) -> list[tuple[str, str]]:
    """(key, method) of every row in a directory sink's part files."""
    pairs: list[tuple[str, str]] = []
    for part in sorted(sink.glob("part-*.parquet")):
        table = pq.read_table(part, columns=["key", "method"])
        pairs.extend(
            zip(table.column("key").to_pylist(), table.column("method").to_pylist(),
                strict=True)
        )
    return pairs


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


def test_batch_dir_resume_with_another_chunk_size_neither_drops_nor_duplicates(
    tmp_path: Path,
) -> None:
    # Regression (M4), restated: resuming used to trust part files by chunk index, so a
    # different chunk_size realigned the indices and dropped/duplicated rows, and was
    # refused. Resume now reads the keys already written, so a new chunk_size is safe.
    lcs = _light_curves(5)
    cup.batch_periodograms(lcs[:4], "GLS", workers=1, sink=tmp_path, chunk_size=2)
    second = cup.batch_periodograms(lcs, "GLS", workers=1, sink=tmp_path, chunk_size=3)
    assert (second.n_done, second.n_skipped) == (1, 4)
    keys = [k for k, _ in _dir_pairs(tmp_path)]
    assert sorted(keys) == [f"star{i}" for i in range(5)]


def test_batch_csv_store_raw_rejected(tmp_path: Path) -> None:
    # Regression (M5): CSV can't hold raw spectrum arrays; must fail loudly.
    with pytest.raises(ValueError, match="CSV"):
        cup.batch_periodograms(
            _light_curves(1), "GLS", workers=1, sink=tmp_path / "r.csv", store_raw=True
        )


def test_batch_empty_methods_raises() -> None:
    with pytest.raises(ValueError, match="no methods"):
        cup.batch_periodograms(_light_curves(1), [], workers=1)


# --- directory-sink resume is key based (BATCH-CLI-1, -15, -16) -------------------


def test_dir_resume_after_inputs_change_computes_exactly_the_new_ones(
    tmp_path: Path,
) -> None:
    src, sink = tmp_path / "src", tmp_path / "out"
    for seed, name in enumerate("ace"):
        _write_csv(src / f"{name}.csv", seed)
    cup.batch_periodograms(
        str(src / "*.csv"), "GLS", workers=0, sink=sink, chunk_size=2
    )
    # New files that sort between the old ones shift every chunk's membership.
    for seed, name in enumerate("bd", start=5):
        _write_csv(src / f"{name}.csv", seed)
    second = cup.batch_periodograms(
        str(src / "*.csv"), "GLS", workers=0, sink=sink, chunk_size=2
    )
    assert (second.n_done, second.n_skipped, second.n_failed) == (2, 3, 0)
    # An append onto the old partial last chunk, then a removed input.
    _write_csv(src / "f.csv", 9)
    (src / "a.csv").unlink()
    third = cup.batch_periodograms(
        str(src / "*.csv"), "GLS", workers=0, sink=sink, chunk_size=2
    )
    assert (third.n_done, third.n_skipped) == (1, 4)
    pairs = _dir_pairs(sink)
    assert Counter(pairs) == Counter((k, "GLS") for k in "abcdef")


def test_dir_resume_adds_a_new_method(tmp_path: Path) -> None:
    lcs = _light_curves(4)
    cup.batch_periodograms(lcs, "GLS", workers=0, sink=tmp_path, chunk_size=2)
    second = cup.batch_periodograms(
        lcs, ["GLS", "PDM"], workers=0, sink=tmp_path, chunk_size=2
    )
    assert (second.n_done, second.n_skipped) == (4, 0)  # four PDM rows, no GLS rerun
    pairs = _dir_pairs(tmp_path)
    assert Counter(pairs) == Counter(
        (f"star{i}", m) for i in range(4) for m in ("GLS", "PDM")
    )
    third = cup.batch_periodograms(lcs, "PDM", workers=0, sink=tmp_path)
    assert (third.n_done, third.n_skipped) == (0, 4)


@pytest.mark.parametrize(
    "change",
    [
        {"settings": cup.GLSSettings(samples_per_peak=7)},
        {"n_best": 3},
        {"store_raw": True},
    ],
    ids=["settings", "n_best", "store_raw"],
)
def test_dir_resume_refuses_a_request_that_changes_the_rows(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    lcs = _light_curves(2)
    cup.batch_periodograms(lcs, "GLS", workers=0, sink=tmp_path)
    with pytest.raises(ValueError, match="different request"):
        cup.batch_periodograms(lcs, "GLS", workers=0, sink=tmp_path, **change)
    # resume=False is the explicit way to start over.
    summary = cup.batch_periodograms(
        lcs, "GLS", workers=0, sink=tmp_path, resume=False, **change
    )
    assert summary.n_done == 2


def test_dir_resume_after_an_all_failed_run_accepts_a_corrected_request(
    tmp_path: Path,
) -> None:
    # Every curve failing (a wrong column name, say) leaves no part file, so there are
    # no rows to keep consistent: the corrected re-run must not be refused as a
    # "different request" against the record of the run that failed.
    for i in range(2):
        t, y, err = synthetic_sine(n=150, seed=i)
        pd.DataFrame({"jd": t, "mag": y, "err": err}).to_csv(
            tmp_path / f"s{i}.csv", index=False
        )
    out = tmp_path / "parts"
    pattern = str(tmp_path / "*.csv")
    failed = cup.batch_periodograms(
        pattern, "GLS", workers=0, sink=out, columns=cup.ColumnMap(time="wrong")
    )
    assert (failed.n_done, failed.n_failed) == (0, 2)
    fixed = cup.batch_periodograms(
        pattern, "GLS", workers=0, sink=out, columns=cup.ColumnMap(time="jd")
    )
    assert (fixed.n_done, fixed.n_failed) == (2, 0)
    # Once rows exist, the record protects them again.
    with pytest.raises(ValueError, match="different request"):
        cup.batch_periodograms(
            pattern, "GLS", workers=0, sink=out, columns=cup.ColumnMap(time="t")
        )


def test_dir_resume_keeps_one_schema_across_runs(tmp_path: Path) -> None:
    lcs = _light_curves(3)
    cup.batch_periodograms(lcs[:2], ["GLS", "BLS"], workers=0, sink=tmp_path)
    # Fewer methods later: the new part still carries the BLS columns (NaN).
    cup.batch_periodograms(lcs, "GLS", workers=0, sink=tmp_path)
    schemas = [pq.read_schema(p) for p in sorted(tmp_path.glob("part-*.parquet"))]
    assert len(schemas) == 2 and schemas[0].equals(schemas[1])
    # A method whose columns the existing parts lack is refused, not silently lost.
    gls_only = tmp_path / "gls"
    cup.batch_periodograms(lcs[:2], "GLS", workers=0, sink=gls_only)
    with pytest.raises(ValueError, match="new per-peak columns"):
        cup.batch_periodograms(lcs[:2], "BLS", workers=0, sink=gls_only)


def test_dir_resume_accepts_a_manifest_without_a_fingerprint(tmp_path: Path) -> None:
    lcs = _light_curves(3)
    cup.batch_periodograms(lcs[:2], "GLS", workers=0, sink=tmp_path, chunk_size=2)
    (tmp_path / "_manifest.json").write_text(json.dumps({"chunk_size": 256}))
    summary = cup.batch_periodograms(lcs, "GLS", workers=0, sink=tmp_path)
    assert (summary.n_done, summary.n_skipped) == (1, 2)
    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["config"]["n_best"] == 10


def test_dir_resume_moves_a_corrupt_part_aside_and_recomputes_it(
    tmp_path: Path,
) -> None:
    lcs = _light_curves(4)
    cup.batch_periodograms(lcs, "GLS", workers=0, sink=tmp_path, chunk_size=2)
    (tmp_path / "part-00001.parquet").write_bytes(b"PAR1 torn")
    (tmp_path / "part-00007.parquet.tmp").write_bytes(b"old interrupted write")
    with pytest.warns(UserWarning, match="unreadable"):
        again = cup.batch_periodograms(lcs, "GLS", workers=0, sink=tmp_path)
    assert (again.n_done, again.n_skipped) == (2, 2)
    assert not (tmp_path / "part-00007.parquet.tmp").exists()
    assert len(list(tmp_path.glob(".part-00001.parquet.corrupt-*"))) == 1
    assert sorted(k for k, _ in _dir_pairs(tmp_path)) == [f"star{i}" for i in range(4)]
    assert pq.read_table(tmp_path).num_rows == 4  # the dataset reads cleanly


def test_dir_no_resume_replaces_the_previous_parts(tmp_path: Path) -> None:
    cup.batch_periodograms(
        _light_curves(6), "GLS", workers=0, sink=tmp_path, chunk_size=2
    )
    (tmp_path / "notes.txt").write_text("keep me")
    (tmp_path / "other.parquet").write_bytes(b"not a part")
    fresh = [
        cup.LightCurve.from_arrays(
            *synthetic_sine(n=120, seed=40 + i), meta={"id": f"new{i}"}
        )
        for i in range(2)
    ]
    summary = cup.batch_periodograms(
        fresh, "GLS", workers=0, sink=tmp_path, chunk_size=6, resume=False
    )
    assert summary.n_done == 2
    parts = sorted(p.name for p in tmp_path.glob("part-*.parquet"))
    assert parts == ["part-00000.parquet"]
    assert sorted(k for k, _ in _dir_pairs(tmp_path)) == ["new0", "new1"]
    assert (tmp_path / "notes.txt").read_text() == "keep me"
    assert (tmp_path / "other.parquet").exists()


def test_dir_no_resume_with_only_failures_leaves_no_stale_part(tmp_path: Path) -> None:
    cup.batch_periodograms(_light_curves(2), "GLS", workers=0, sink=tmp_path)
    bad = cup.LightCurve.from_arrays([1.0, 2.0], [1.0, 2.0], [0.1, 0.1])
    summary = cup.batch_periodograms(
        [bad], "GLS", workers=0, sink=tmp_path, resume=False
    )
    assert summary.n_failed == 1
    assert not list(tmp_path.glob("part-*.parquet"))


def test_legacy_pending_chunks_is_key_based(tmp_path: Path) -> None:
    # The chunk-index helper the pre-whitening and period-change drivers call: it now
    # re-plans the chunks in place, so changed inputs never drop or duplicate a curve.
    lcs = _light_curves(5)
    items = [(f"star{i}", lc) for i, lc in enumerate(lcs)]
    chunks = runner._chunked([items[0], items[2], items[4]], 2)
    pending = runner._pending_chunks(chunks, "dir", tmp_path, True)
    for idx in pending:
        rows, _ = runner._process_chunk(chunks[idx], _cfg(("GLS",)))
        runner.write_rows(rows, runner._part_path(tmp_path, idx))
    chunks = runner._chunked(items, 2)  # star1 and star3 now sort in between
    pending = runner._pending_chunks(chunks, "dir", tmp_path, True)
    todo = [key for idx in pending for key, _ in chunks[idx]]
    assert todo == ["star1", "star3"]
    assert min(pending) == 2  # past the existing part-00000/part-00001


def _cfg(methods: tuple[str, ...]) -> runner._ChunkConfig:
    from cuperiod.methods.base import get_method

    return runner._ChunkConfig(
        methods=methods,
        backends={m: "cpu" for m in methods},
        settings_map={m: get_method(m).coerce_settings(None) for m in methods},
        columns=None,
        domain=None,
        band_column=None,
        n_best=3,
        store_raw=False,
    )


# --- file-sink resume skips finished work (BATCH-CLI-9) ---------------------------


def test_file_resume_skips_finished_light_curves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lcs = _light_curves(3)
    out = tmp_path / "r.parquet"
    cup.batch_periodograms(lcs, "GLS", workers=0, sink=out)
    before = out.read_bytes()

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("nothing should be computed")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "_process_chunk", boom)
        again = cup.batch_periodograms(lcs, "GLS", workers=0, sink=out)
    assert (again.n_done, again.n_skipped, again.n_failed) == (0, 3, 0)
    assert out.read_bytes() == before

    calls: list[int] = []
    real = runner._process_chunk

    def counting(items: Any, cfg: Any) -> Any:
        calls.append(len(items))
        return real(items, cfg)

    monkeypatch.setattr(runner, "_process_chunk", counting)
    more = cup.batch_periodograms(
        [*lcs, *_light_curves(4)[3:]], "GLS", workers=0, sink=out
    )
    assert (more.n_done, more.n_skipped) == (1, 3)
    assert sum(calls) == 1
    assert pq.read_table(out).num_rows == 4


# --- input order (BATCH-CLI-14) ----------------------------------------------------


def _reversed_pool(
    chunks: Any, pending: Any, cfg: Any, absorb: Any, max_workers: int, **kw: Any
) -> None:
    """A stand-in pool that finishes the chunks in reverse order."""
    configs = kw.get("configs")
    for idx in reversed(list(pending)):
        chunk_cfg = configs[idx] if configs else cfg
        absorb(idx, *runner._process_chunk(chunks[idx], chunk_cfg))


@pytest.mark.parametrize("sink_name", [None, "r.parquet"])
def test_pool_rows_and_errors_come_back_in_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sink_name: str | None
) -> None:
    monkeypatch.setattr(runner, "_run_pool", _reversed_pool)
    bad = cup.LightCurve.from_arrays([1.0, 2.0], [1.0, 2.0], [0.1, 0.1])
    inputs = [(f"k{i}", lc) for i, lc in enumerate(_light_curves(4))]
    inputs.insert(1, ("bad_a", bad))
    inputs.append(("bad_b", bad))
    sink = tmp_path / sink_name if sink_name else None
    summary = cup.batch_periodograms(inputs, "GLS", workers=2, chunk_size=1, sink=sink)
    rows = summary.rows if sink is None else pq.read_table(sink).to_pylist()
    assert rows is not None
    assert [r["key"] for r in rows] == ["k0", "k1", "k2", "k3"]
    assert [k for k, _ in summary.errors] == ["bad_a", "bad_b"]


# --- worker sizing (BATCH-CLI-10) --------------------------------------------------


def test_auto_chunk_size_gives_every_pool_worker_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def record(
        chunks: Any, pending: Any, cfg: Any, absorb: Any, n: int, **kw: Any
    ) -> None:
        seen["tasks"], seen["workers"] = len(pending), n
        seen["sizes"] = [len(chunks[i]) for i in pending]

    monkeypatch.setattr(runner, "_run_pool", record)
    lcs = [(f"s{i}", lc) for i, lc in enumerate(_light_curves(1) * 48)]
    cup.batch_periodograms(lcs, "GLS", workers=8)
    assert seen["tasks"] >= 8 and seen["workers"] == 8
    assert sum(seen["sizes"]) == 48
    cup.batch_periodograms(lcs, "GLS", workers=8, chunk_size=48)  # explicit wins
    assert (seen["tasks"], seen["workers"]) == (1, 1)


# --- backend follows the device (BATCH-CLI-2, CORE-B-3) ----------------------------


def test_cpu_device_keeps_auto_off_the_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import cuperiod.methods.base as base

    # Pretend a CUDA device and the CUDA backends are present.
    monkeypatch.setattr(
        base, "available_backends",
        lambda: {"numpy", "finufft", "numba", "astropy", "cupy", "cufinufft"},
    )
    monkeypatch.setattr(base, "cuda_available", lambda: True)
    monkeypatch.setattr(base, "torch_gpu_available", lambda: False)
    assert base.get_method("GLS").resolve_backend("auto") == "cufinufft"  # the trap
    seen: list[str] = []
    gls = base.get_method("GLS")
    real_power = gls.power

    def spy(grid: Any, lc: Any, settings: Any, backend: str, engine: Any = None) -> Any:
        seen.append(backend)
        assert backend not in {"cufinufft", "cupy"}, "device='cpu' ran on the GPU"
        return real_power(grid, lc, settings, backend, engine=engine)

    monkeypatch.setattr(gls, "power", spy)
    summary = cup.batch_periodograms(_light_curves(2), "GLS", workers=0)
    assert summary.n_failed == 0 and seen == ["finufft", "finufft"]
    with pytest.raises(ValueError, match="device='gpu'"):
        cup.batch_periodograms(_light_curves(1), "GLS", workers=0, backend="cufinufft")


def test_worker_backend_selector_rules() -> None:
    settings = cup.GLSSettings()
    assert runner._worker_backend("cpu", "auto", settings) == "cpu"
    assert runner._worker_backend("cpu", None, settings) == "cpu"
    assert runner._worker_backend("cpu", "torch", settings) == "torch:cpu"
    assert runner._worker_backend("cpu", "finufft", settings) == "finufft"
    assert runner._worker_backend("gpu", "auto", settings) == "gpu"
    assert runner._worker_backend("gpu", "torch", settings) == "torch"
    assert runner._worker_backend("gpu", "torch:cuda", settings) == "torch:cuda"
    assert runner._worker_backend("gpu", "cufinufft", settings) == "cufinufft"
    with pytest.raises(ValueError, match="runs on the CPU"):
        runner._worker_backend("gpu", "cpu", settings)
    with pytest.raises(ValueError, match="GPU"):
        runner._worker_backend("cpu", "gpu", settings)
    # A settings pin is honoured under either device.
    pinned = cup.GLSSettings(backend="astropy")
    assert runner._worker_backend("cpu", "auto", pinned) == "astropy"
    assert runner._worker_backend("gpu", "auto", pinned) == "astropy"


@requires_gpu
def test_gpu_device_runs_on_the_gpu_and_matches_the_cpu() -> None:
    lcs = _light_curves(3)
    gpu = cup.batch_periodograms(lcs, ["GLS", "PDM"], device="gpu", workers=1)
    cpu = cup.batch_periodograms(lcs, ["GLS", "PDM"], device="cpu", workers=0)
    assert gpu.rows is not None and cpu.rows is not None and gpu.n_failed == 0
    assert {r["backend"] for r in gpu.rows} == {"cufinufft", "cupy"}
    for g, c in zip(gpu.rows, cpu.rows, strict=True):
        assert (g["key"], g["method"]) == (c["key"], c["method"])
        assert g["best_period"] == pytest.approx(c["best_period"], rel=1e-4)


def test_an_unavailable_backend_fails_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_process_chunk", lambda *a: pytest.fail("ran"))
    with pytest.raises(cup.BackendUnavailableError):
        cup.batch_periodograms(_light_curves(1), "GLS", workers=0, backend="bogus")


# --- custom grid (BATCH-CLI-3) -----------------------------------------------------


def test_batch_honours_a_custom_grid() -> None:
    lc = _light_curves(1)[0]
    grid = cup.GridSpec("frequency", np.linspace(1.0, 2.0, 501), uniform=True)
    summary = cup.batch_periodograms([lc], "GLS", workers=0, grid=grid, store_raw=True)
    assert summary.rows is not None
    row = summary.rows[0]
    freq = np.asarray(row["pgram_frequency"])
    assert freq.min() >= 1.0 - 1e-6 and freq.max() <= 2.0 + 1e-6
    expected = cup.periodogram(lc, "GLS", backend="cpu", grid=grid).best_period()
    assert row["best_period"] == pytest.approx(expected, rel=1e-9)
    with pytest.raises(TypeError, match="GridSpec"):
        cup.batch_periodograms([lc], "GLS", workers=0, grid="not a grid")  # type: ignore[arg-type]


# --- settings selection (BATCH-CLI-17) ----------------------------------------------


def test_settings_mapping_keys_match_like_method_names() -> None:
    lc = _light_curves(1)[0]
    tuned = cup.StringLengthSettings(minimum_frequency=1.0, maximum_frequency=2.0)
    for key in ("String-Length", "string_length", "STRINGLENGTH"):
        summary = cup.batch_periodograms(
            [lc], "String-Length", workers=0, settings={key: tuned}, store_raw=True
        )
        assert summary.rows is not None
        assert max(summary.rows[0]["pgram_frequency"]) < 2.1, key


def test_mismatched_settings_raise_instead_of_falling_back() -> None:
    lc = _light_curves(1)
    with pytest.raises(TypeError, match="GLSSettings"):
        cup.batch_periodograms(lc, "GLS", workers=0, settings=cup.PDMSettings())
    with pytest.raises(cup.UnknownMethodError):
        cup.batch_periodograms(
            lc, "GLS", workers=0, settings={"GSL": cup.GLSSettings()}
        )
    # A bare object for one of several methods still tunes that method only.
    summary = cup.batch_periodograms(
        lc, ["GLS", "PDM"], workers=0, settings=cup.GLSSettings(samples_per_peak=7)
    )
    assert summary.n_done == 2


def test_settings_peak_fields_reach_the_rows() -> None:
    summary = cup.batch_periodograms(
        _light_curves(1), "GLS", workers=0, store_raw=True,
        settings=cup.GLSSettings(downsample_points=100),
    )
    assert summary.rows is not None
    assert len(summary.rows[0]["pgram_power"]) == 100


# --- sink checks before any work (BATCH-CLI-21) -------------------------------------


@pytest.mark.parametrize("name", ["dataset.parquet", "table.csv"])
def test_a_directory_with_a_file_suffix_is_refused_up_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    (tmp_path / name).mkdir()
    monkeypatch.setattr(runner, "_process_chunk", lambda *a: pytest.fail("ran"))
    with pytest.raises(ValueError, match="existing directory"):
        cup.batch_periodograms(_light_curves(1), "GLS", workers=0, sink=tmp_path / name)


def test_a_failed_final_rename_names_the_kept_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cuperiod.batch.io as bio

    def refuse(src: Any, dst: Any) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(bio.os, "replace", refuse)
    out = tmp_path / "deep" / "r.parquet"
    with pytest.raises(PermissionError, match=r"\.tmp") as info:
        cup.batch_periodograms(_light_curves(2), "GLS", workers=0, sink=out)
    kept = Path(str(info.value).rsplit("kept in ", 1)[1])
    assert kept.exists() and pq.read_table(kept).num_rows == 2


# --- spawned pools and unguarded scripts (BATCH-CLI-13) -----------------------------


def test_a_batch_started_while_a_worker_bootstraps_explains_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        multiprocessing.current_process(), "_inheriting", True, raising=False
    )
    with pytest.raises(RuntimeError, match="another batch.*workers=1"):
        cup.batch_periodograms(_light_curves(2), "GLS", workers=2, chunk_size=1)


def test_a_broken_pool_explains_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    class DeadPool:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> DeadPool:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def submit(self, fn: Any, *args: Any) -> Future[Any]:
            future: Future[Any] = Future()
            future.set_exception(BrokenProcessPool("a child process terminated"))
            return future

    monkeypatch.setattr(runner, "ProcessPoolExecutor", DeadPool)
    with pytest.raises(BrokenProcessPool, match="__main__.*workers=1") as info:
        cup.batch_periodograms(_light_curves(2), "GLS", workers=2, chunk_size=1)
    assert "terminated" in str(info.value)


# --- error column named, not the baseline it empties (CORE-A-14) ---------------------


def test_an_unusable_error_column_is_named() -> None:
    t, mag, _ = synthetic_sine(n=120)
    lc = cup.LightCurve.from_arrays(t, mag, np.zeros_like(t))
    summary = cup.batch_periodograms([lc], "GLS", workers=0)
    assert summary.n_failed == 1
    assert "has a usable error" in summary.errors[0][1]


# --- one row schema for every part (BATCH-CLI-6) --------------------------------------


def test_directory_parts_share_one_schema(tmp_path: Path) -> None:
    t, flux, err = synthetic_eclipser(n=600, span=60.0)
    transit = cup.LightCurve.from_arrays(t, flux, err, domain="flux")
    rng = np.random.default_rng(1)
    short_t = np.sort(rng.uniform(0.0, 3.0, 40)) + 2458000.0
    short = cup.LightCurve.from_arrays(
        short_t, 1.0 + 0.001 * rng.standard_normal(40), np.full(40, 0.001),
        domain="flux",
    )
    sink = tmp_path / "parts"
    summary = cup.batch_periodograms(
        [("short", short), ("transit", transit)], ["GLS", "BLS"], workers=0,
        sink=sink, chunk_size=1, backend="cpu",
    )
    assert summary.n_failed == 0
    parts = sorted(sink.glob("part-*.parquet"))
    schemas = [pq.read_schema(p) for p in parts]
    assert len(parts) == 2 and schemas[0].equals(schemas[1])
    table = pq.read_table(sink)
    assert {"depth_10", "sde_10", "fap_10", "source"} <= set(table.column_names)
    by_key = {(r["key"], r["method"]): r for r in table.to_pylist()}
    assert np.isfinite(by_key[("transit", "BLS")]["depth_2"])
    assert np.isnan(by_key[("transit", "GLS")]["depth_1"])  # GLS has no depth


def test_bls_best_period_is_the_first_listed_peak() -> None:
    t, flux, err = synthetic_eclipser(n=600, span=60.0)
    lc = cup.LightCurve.from_arrays(t, flux, err, domain="flux")
    summary = cup.batch_periodograms([lc], "BLS", workers=0, backend="cpu", n_best=5)
    assert summary.rows is not None
    row = summary.rows[0]
    assert row["best_period"] == row["period_1"]
    pg = cup.periodogram(lc, "BLS", backend="cpu")
    expected = [p.period for p in pg.best_periods(5, alias_diverse=True)]
    assert [row[f"period_{k}"] for k in range(1, 6)] == pytest.approx(expected)
