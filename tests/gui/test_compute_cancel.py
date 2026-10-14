"""Stop for one-shot runs, and session jobs that never read the generation counter."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

import cuperiod
from cuperiod.core.config import GLSSettings
from cuperiod.core.lightcurve import LightCurve
from cuperiod.gui.compute import ComputeManager, SessionJob, compute_active
from cuperiod.gui.models import ResultKey


def _curve() -> LightCurve:
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 30.0, 200))
    return LightCurve.from_arrays(t, np.sin(2 * np.pi * t / 1.7), np.full(200, 0.01))


def _key(tag: str) -> ResultKey:
    return ResultKey(tag, "GLS", "h", "cpu")


class _Gate:
    """A ``cup.periodogram`` stand-in that blocks until released."""

    def __init__(self, real: Any) -> None:
        self.real = real
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.entered.set()
        assert self.release.wait(10.0), "test gate never released"
        return self.real(*args, **kwargs)


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Gate]:
    g = _Gate(cuperiod.periodogram)
    monkeypatch.setattr(cuperiod, "periodogram", g)
    yield g
    g.release.set()


@pytest.fixture
def manager(qtbot: QtBot) -> Iterator[ComputeManager]:
    m = ComputeManager()
    yield m
    m.wait_for_done(10_000)


def test_stop_releases_at_once_and_the_worker_finishes_in_background(
    manager: ComputeManager, gate: _Gate, qtbot: QtBot
) -> None:
    finished: list[ResultKey] = []
    manager.signals.finished.connect(lambda key, pg, ms: finished.append(key))
    manager.submit(_key("a"), _curve(), "GLS", GLSSettings(), "cpu")
    assert gate.entered.wait(10.0)
    qtbot.waitUntil(lambda: manager.running_key == _key("a"))
    assert not manager.finishing
    # A run queued behind the executing one ...
    manager.submit(_key("b"), _curve(), "GLS", GLSSettings(), "cpu")
    with qtbot.waitSignal(manager.signals.cancelled, timeout=1000) as blocker:
        key = manager.cancel_running()  # returns while the worker is still busy
    assert key == _key("a") and blocker.args == [_key("a")]
    assert manager.active and manager.finishing
    gate.release.set()
    qtbot.waitUntil(lambda: finished == [_key("a")], timeout=10_000)
    assert manager.wait_for_done(10_000)
    # ... was superseded by the Stop and never ran; the stopped one was delivered
    # under its own key (for the cache), and the worker is free again.
    assert gate.calls == 1 and finished == [_key("a")]
    assert not manager.finishing and manager.running_key is None


def test_stop_when_idle(manager: ComputeManager, qtbot: QtBot) -> None:
    with qtbot.waitSignal(manager.signals.cancelled) as blocker:
        assert manager.cancel_running() is None
    assert blocker.args == [None]
    assert not manager.finishing


def _job(log: list[str], name: str, **kw: Any) -> SessionJob:
    def work(job: SessionJob) -> str:
        log.append(name)
        return name

    return SessionJob(work, key="s", label=name, **kw)


def test_session_jobs_ignore_generation(
    manager: ComputeManager, gate: _Gate, qtbot: QtBot
) -> None:
    log: list[str] = []
    done: list[object] = []
    manager.session_signals.done.connect(lambda job, outcome: done.append(outcome))
    manager.submit(_key("a"), _curve(), "GLS", GLSSettings(), "cpu")
    assert gate.entered.wait(10.0)
    manager.start_session_job(_job(log, "edit-1"))
    manager.start_session_job(_job(log, "edit-2"))
    assert manager.session_jobs == 2
    # Everything a one-shot run does to supersede work leaves session jobs alone.
    manager.cancel_all()
    manager.cancel_running()
    manager.submit(_key("b"), _curve(), "GLS", GLSSettings(), "cpu")
    gate.release.set()
    qtbot.waitUntil(lambda: len(done) == 2, timeout=10_000)
    assert log == ["edit-1", "edit-2"] and done == ["edit-1", "edit-2"]
    qtbot.waitUntil(lambda: manager.session_jobs == 0)
    assert manager.wait_for_done(10_000)


def test_session_job_progress_cancel_and_failure(
    manager: ComputeManager, qtbot: QtBot
) -> None:
    proceed = threading.Event()
    progress: list[object] = []
    manager.session_signals.progress.connect(lambda job, p: progress.append(p))

    def long_work(job: SessionJob) -> int:
        steps = 0
        job.report("started")
        proceed.wait(10.0)
        while not job.should_stop() and steps < 10_000:
            steps += 1
            if steps <= 2:
                job.report(steps)
            threading.Event().wait(0.001)
        return steps

    job = SessionJob(long_work, key="k", kind="mutation", long=True, label="Auto")
    assert not job.cancelled and repr(job).startswith("SessionJob('mutation'")
    with qtbot.waitSignal(manager.session_signals.done, timeout=10_000) as blocker:
        manager.start_session_job(job)
        qtbot.waitUntil(lambda: job.running)
        assert compute_active()
        proceed.set()
        qtbot.waitUntil(lambda: 2 in progress)
        job.cancel()
    finished_job, steps = blocker.args
    assert finished_job is job and job.cancelled and job.finished
    assert 2 <= steps < 10_000 and job.elapsed_ms > 0.0
    assert progress[:3] == ["started", 1, 2]

    def broken(job: SessionJob) -> None:
        raise RuntimeError("boom")

    with qtbot.waitSignal(manager.session_signals.failed, timeout=10_000) as failed:
        manager.start_session_job(SessionJob(broken, key="k"))
    assert failed.args[1] == "RuntimeError: boom"
    qtbot.waitUntil(lambda: manager.session_jobs == 0)
