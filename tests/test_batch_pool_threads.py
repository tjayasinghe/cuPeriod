"""Every CPU batch pool caps its workers' threads and leaves the parent alone.

BATCH-CLI-19 / PERF-PDOT-06: a CPU pool parallelises over processes, so each
worker's math libraries must not open a thread per core as well (``workers x cores``
threads). BLAS/OpenMP are pinned to one thread through the environment the spawned
workers inherit, numba to the worker's share of the cores (``cores // workers``) by
the pool initializer; and the pin is scoped to the pool, where it used to stay in
the parent session for good (later subprocesses and lazily imported libraries ran
single-threaded).

Each driver runs a real spawned pool here. Only the submitted chunk function is
swapped (``ProcessPoolExecutor.submit``) for a probe that reports the worker's own
thread configuration as result rows; the driver's pool, pin and initializer run
unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pytest

import cuperiod as cup
import pool_probe
from cuperiod.batch import runner
from cuperiod.batch.sizing import numba_threads_per_worker
from synth import synthetic_sine

pytest.importorskip("numba")

WORKERS = 2


def _probe_pool(probe: Callable[..., Any]) -> type[ProcessPoolExecutor]:
    """A real spawned pool whose every task runs ``probe`` instead."""

    class ProbePool(ProcessPoolExecutor):
        def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
            return super().submit(probe, *args, **kwargs)

    return ProbePool


def _environment() -> dict[str, str]:
    """This process's environment, less the variable pytest rewrites per phase."""
    return {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}


@pytest.fixture
def unpinned(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """No BLAS pin in this process; returns the environment to compare against."""
    for var in pool_probe.BLAS_VARS:
        monkeypatch.delenv(var, raising=False)
    return _environment()


def _curves(n: int = 4) -> list[tuple[str, cup.LightCurve]]:
    out = []
    for i in range(n):
        t, mag, err = synthetic_sine(n=60, period=0.5 + 0.1 * i, seed=i)
        out.append((f"star{i}", cup.LightCurve.from_arrays(t, mag, err)))
    return out


def _expected_numba_threads() -> int:
    # A user's own NUMBA_NUM_THREADS is respected by the initializer.
    preset = os.environ.get("NUMBA_NUM_THREADS")
    return int(preset) if preset else numba_threads_per_worker(WORKERS)


def _check_workers(rows: list[dict[str, Any]] | None) -> None:
    assert rows is not None and len(rows) == 4
    for row in rows:
        assert row["pid"] != os.getpid()  # really ran in a pool worker
        assert row["numba_threads"] == _expected_numba_threads(), row
        for var in pool_probe.BLAS_VARS:
            assert row[var] == "1", (var, row)


RUNS: dict[str, Callable[[], Any]] = {
    "periodograms": lambda: cup.batch_periodograms(
        _curves(), "GLS", backend="cpu", workers=WORKERS, chunk_size=1
    ),
    "prewhiten": lambda: cup.batch_prewhiten(
        _curves(), workers=WORKERS, chunk_size=1
    ),
    "pdot": lambda: cup.batch_pdot(_curves(), workers=WORKERS, chunk_size=1),
}


@pytest.mark.parametrize("driver", list(RUNS))
def test_pool_workers_report_capped_threads_and_the_parent_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, unpinned: dict[str, str], driver: str
) -> None:
    probe = pool_probe.tables_chunk if driver == "pdot" else pool_probe.rows_chunk
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _probe_pool(probe))
    summary = RUNS[driver]()
    _check_workers(summary.rows)
    assert _environment() == unpinned  # the pin did not outlive the pool


def test_a_failing_pool_still_restores_the_parent(
    monkeypatch: pytest.MonkeyPatch, unpinned: dict[str, str]
) -> None:
    monkeypatch.setattr(
        runner, "ProcessPoolExecutor", _probe_pool(pool_probe.failing_chunk)
    )
    with pytest.raises(RuntimeError, match="pool probe failure"):
        RUNS["periodograms"]()
    assert _environment() == unpinned


def test_a_user_blas_setting_reaches_the_workers_unchanged(
    monkeypatch: pytest.MonkeyPatch, unpinned: dict[str, str]
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    before = _environment()
    monkeypatch.setattr(
        runner, "ProcessPoolExecutor", _probe_pool(pool_probe.rows_chunk)
    )
    summary = RUNS["periodograms"]()
    assert summary.rows is not None
    assert {row["OMP_NUM_THREADS"] for row in summary.rows} == {"3"}
    assert {row["MKL_NUM_THREADS"] for row in summary.rows} == {"1"}
    assert _environment() == before
