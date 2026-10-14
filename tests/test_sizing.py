"""Batch worker sizing and thread pinning (cuperiod.batch.sizing)."""

from __future__ import annotations

import os
import sys
import types
import warnings

import pytest

from cuperiod.batch import sizing


def _cores(monkeypatch: pytest.MonkeyPatch, n: int, *, platform: str) -> None:
    monkeypatch.setattr(sizing, "usable_cpu_count", lambda: n)
    monkeypatch.setattr(sys, "platform", platform)


def test_default_workers_respect_the_windows_pool_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ProcessPoolExecutor raises "max_workers must be <= 61" on Windows; a 64+-thread
    # host used to fail the default batch before any work.
    _cores(monkeypatch, 128, platform="win32")
    assert sizing.cpu_worker_count(None) == 61
    assert sizing.cpu_worker_count(-1) == 61
    with pytest.warns(RuntimeWarning, match="61"):
        assert sizing.cpu_worker_count(200) == 61
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sizing.cpu_worker_count(8) == 8
        assert sizing.cpu_worker_count(0) == 1


def test_no_pool_limit_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    _cores(monkeypatch, 128, platform="linux")
    assert sizing.max_pool_workers() is None
    assert sizing.cpu_worker_count(None) == 127
    assert sizing.cpu_worker_count(200) == 200


def test_clamp_pool_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert sizing.clamp_pool_workers(64, warn=False) == 61
    assert sizing.clamp_pool_workers(4) == 4
    monkeypatch.setattr(sys, "platform", "linux")
    assert sizing.clamp_pool_workers(64) == 64


def test_default_workers_follow_the_usable_cpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A job granted 2 of 32 logical CPUs (SLURM cgroup, taskset, Windows affinity)
    # used to start 31 workers; the default is now "usable CPUs minus one".
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setattr(sizing, "_affinity_cpu_count", lambda: 2)
    monkeypatch.setattr(sizing, "_cgroup_cpu_limit", lambda: None)
    assert sizing.usable_cpu_count() == 2
    assert sizing.cpu_worker_count(None) == 1
    # A CPU quota tighter than the affinity mask wins too (docker --cpus=3).
    monkeypatch.setattr(sizing, "_affinity_cpu_count", lambda: 16)
    monkeypatch.setattr(sizing, "_cgroup_cpu_limit", lambda: 3)
    assert sizing.usable_cpu_count() == 3
    # No information beyond cpu_count -> the old behaviour.
    monkeypatch.setattr(sizing, "_affinity_cpu_count", lambda: None)
    monkeypatch.setattr(sizing, "_cgroup_cpu_limit", lambda: None)
    assert sizing.usable_cpu_count() == 32


def test_usable_cpu_count_on_this_machine() -> None:
    n = sizing.usable_cpu_count()
    assert 1 <= n <= (os.cpu_count() or 1)


def test_numba_threads_per_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sizing, "usable_cpu_count", lambda: 32)
    assert sizing.numba_threads_per_worker(4) == 8
    assert sizing.numba_threads_per_worker(31) == 1
    assert sizing.numba_threads_per_worker(64) == 1
    assert sizing.numba_threads_per_worker(0) == 32


_PIN_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _clear_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _PIN_VARS:
        monkeypatch.setenv(var, "x")  # records the original for teardown
        monkeypatch.delenv(var)


def test_pinned_worker_threads_restores_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pin used to be applied with setdefault and never undone, so after one
    # pooled batch every later subprocess and lazily imported library in the
    # session was single-threaded.
    _clear_pins(monkeypatch)
    monkeypatch.setenv("MKL_NUM_THREADS", "4")  # a user preset must survive
    before = dict(os.environ)
    with sizing.pinned_worker_threads():
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "4"
    assert dict(os.environ) == before
    with pytest.raises(RuntimeError), sizing.pinned_worker_threads():
        raise RuntimeError("pool failed")
    assert dict(os.environ) == before


def test_init_worker_threads_pins_numba_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # numba ignores OMP_NUM_THREADS: each pool worker used to run its prange kernels
    # on every core (workers x cores threads). The initializer runs in the child.
    monkeypatch.setenv("NUMBA_NUM_THREADS", "x")
    monkeypatch.delenv("NUMBA_NUM_THREADS")
    monkeypatch.delitem(sys.modules, "numba", raising=False)
    ran: list[int] = []
    sizing.init_worker_threads(8, ran.append, (7,))
    assert os.environ["NUMBA_NUM_THREADS"] == "8"
    assert ran == [7]


def test_init_worker_threads_uses_the_runtime_api_once_numba_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With numba already imported, rewriting NUMBA_NUM_THREADS would make its next
    # config reload raise; the supported runtime call is used instead.
    calls: list[int] = []
    fake = types.SimpleNamespace(
        set_num_threads=calls.append,
        config=types.SimpleNamespace(NUMBA_NUM_THREADS=6),
    )
    monkeypatch.setitem(sys.modules, "numba", fake)
    monkeypatch.setenv("NUMBA_NUM_THREADS", "x")
    monkeypatch.delenv("NUMBA_NUM_THREADS")
    sizing.init_worker_threads(8)
    assert calls == [6]
    assert "NUMBA_NUM_THREADS" not in os.environ


def test_init_worker_threads_respects_a_user_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    fake = types.SimpleNamespace(
        set_num_threads=calls.append,
        config=types.SimpleNamespace(NUMBA_NUM_THREADS=3),
    )
    monkeypatch.setitem(sys.modules, "numba", fake)
    monkeypatch.setenv("NUMBA_NUM_THREADS", "3")
    sizing.init_worker_threads(8)
    assert calls == []
    assert os.environ["NUMBA_NUM_THREADS"] == "3"
