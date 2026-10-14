"""Off-thread periodogram and pre-whitening computation.

:func:`cuperiod.periodogram` is CPU/GPU-bound and can take seconds, and
:func:`cuperiod.prewhiten` runs a whole iterative extraction, so neither may run on the
GUI thread. :class:`PeriodogramTask` and :class:`PreWhitenTask` run them on a worker and
report back via plain signals carrying only the immutable result object; no Qt object is
built off the GUI thread.

Superseding uses *generation gating*: the manager bumps a counter per submit and a task
no-ops if the counter changed before it ran. A single-worker pool gives cheap
"latest wins"; the controller also re-checks the returned key before applying it. Both
task kinds share the counter, so switching analysis mode mid-run supersedes cleanly.

A task that has *started* cannot be interrupted (the library calls are not
cooperative), so :attr:`ComputeManager.active` and :func:`compute_active` report a
running worker: closing the window must not block on it, and the console entry point
uses it to end the process instead of waiting for an orphaned computation.
:meth:`ComputeManager.cancel_running` is the UI's Stop for a one-shot run: it
supersedes the executing task at once, so the user is released immediately, and lets
the worker finish in the background; :attr:`ComputeManager.finishing` reports that.

Interactive pre-whitening sessions run as :class:`SessionJob` units on the same pool.
They **never read the generation counter**: a session mutation is an edit the user made
and must not be dropped by a periodogram submit or ``cancel_all``. Ordering and
superseding of session work belong to the session bridge (one FIFO per session); each
job carries its own cancel token instead.
"""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from pydantic_settings import BaseSettings

from cuperiod.core.config import PreWhitenSettings
from cuperiod.core.result import Periodogram
from cuperiod.gui.models import LoadedCurve, ResultKey
from cuperiod.gui.qt import QObject, QtCore, Signal


class _TaskSignals(QObject):
    """Signals emitted by the compute tasks (owned by the manager)."""

    started = Signal(object)  # ResultKey
    finished = Signal(object, object, float)  # ResultKey, Periodogram, elapsed_ms
    solution_ready = Signal(object, object, float)  # key, PreWhitenResult, elapsed_ms
    failed = Signal(object, str)  # ResultKey, message
    cancelled = Signal(object)  # key of the superseded executing task, or None


class _SessionSignals(QObject):
    """Signals of :class:`SessionJob` (owned by the manager; delivered queued)."""

    started = Signal(object)  # SessionJob
    progress = Signal(object, object)  # SessionJob, payload
    done = Signal(object, object)  # SessionJob, outcome
    failed = Signal(object, str)  # SessionJob, message


class PeriodogramTask(QtCore.QRunnable):
    """Compute one periodogram on a worker thread and emit the result or an error."""

    def __init__(
        self,
        key: ResultKey,
        lc: LoadedCurve,
        method: str,
        settings: BaseSettings,
        backend: str,
        generation: int,
        manager: ComputeManager,
        signals: _TaskSignals,
    ) -> None:
        super().__init__()
        self._key = key
        self._lc = lc
        self._method = method
        self._settings = settings
        self._backend = backend
        self._generation = generation
        self._manager = manager
        self._signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        if self._generation != self._manager.generation:
            return  # superseded before we started — skip the heavy compute
        with self._manager.executing(self._key, self._generation):
            self._run()

    def _run(self) -> None:
        self._signals.started.emit(self._key)
        start = time.perf_counter()
        try:
            import cuperiod as cup

            result = cup.periodogram(
                self._lc,
                self._method,
                backend=self._backend,
                settings=self._settings,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a failure
            self._signals.failed.emit(self._key, f"{type(exc).__name__}: {exc}")
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not isinstance(result, Periodogram):  # pragma: no cover - single method only
            self._signals.failed.emit(self._key, "expected a single Periodogram")
            return
        self._signals.finished.emit(self._key, result, elapsed_ms)


class PreWhitenTask(QtCore.QRunnable):
    """Run one pre-whitening extraction on a worker thread and emit the solution."""

    def __init__(
        self,
        key: ResultKey,
        lc: LoadedCurve,
        settings: PreWhitenSettings,
        backend: str,
        generation: int,
        manager: ComputeManager,
        signals: _TaskSignals,
    ) -> None:
        super().__init__()
        self._key = key
        self._lc = lc
        self._settings = settings
        self._backend = backend
        self._generation = generation
        self._manager = manager
        self._signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        if self._generation != self._manager.generation:
            return  # superseded before we started — skip the heavy compute
        with self._manager.executing(self._key, self._generation):
            self._run()

    def _run(self) -> None:
        self._signals.started.emit(self._key)
        start = time.perf_counter()
        try:
            from cuperiod.prewhiten import prewhiten

            result: Any = prewhiten(
                self._lc, settings=self._settings, backend=self._backend
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a failure
            self._signals.failed.emit(self._key, f"{type(exc).__name__}: {exc}")
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._signals.solution_ready.emit(self._key, result, elapsed_ms)


class SessionJob(QtCore.QRunnable):
    """One unit of interactive-session work, run on the compute worker.

    Unlike the one-shot tasks it ignores :attr:`ComputeManager.generation` entirely:
    whether it runs, and in which order, is decided by whoever submits it (the
    session bridge keeps one FIFO per session). It carries its own cancel token,
    which ``work`` polls between steps (:meth:`should_stop`).

    Parameters
    ----------
    work : callable
        ``work(job) -> outcome``, executed on the worker thread. It may call
        :meth:`report` to send progress to the GUI thread. An exception is reported
        through ``failed`` as ``"Type: message"``.
    key : object, optional
        The session it belongs to (opaque to the manager).
    kind : str
        ``"mutation"``, ``"view"``, ``"query"`` or ``"open"`` (informational).
    long : bool
        A long job (busy state, Stop enabled); informational for the manager.
    label : str
        What the info bar may say while it runs.
    tag : object, optional
        Free payload for the submitter (e.g. a query generation).
    """

    def __init__(
        self,
        work: Callable[[SessionJob], Any],
        *,
        key: object = None,
        kind: str = "mutation",
        long: bool = False,
        label: str = "",
        tag: object = None,
    ) -> None:
        super().__init__()
        # Python keeps the job (the manager holds it until it reports back), so the
        # pool must not delete the C++ side behind the queued done/failed signals.
        self.setAutoDelete(False)
        self.work = work
        self.key = key
        self.kind = kind
        self.long = long
        self.label = label
        self.tag = tag
        self.token = threading.Event()
        self.elapsed_ms = 0.0
        self.running = False
        self.finished = False
        self._signals: _SessionSignals | None = None

    def cancel(self) -> None:
        """Ask ``work`` to stop at its next check (what it committed stands)."""
        self.token.set()

    @property
    def cancelled(self) -> bool:
        return self.token.is_set()

    def should_stop(self) -> bool:
        """The cancel check ``work`` passes to the library (``should_stop=``)."""
        return self.token.is_set()

    def report(self, payload: object) -> None:
        """Send ``payload`` to the GUI thread (``progress`` signal); thread-safe."""
        if self._signals is not None:
            self._signals.progress.emit(self, payload)

    def run(self) -> None:
        signals = self._signals
        if signals is None:  # pragma: no cover - only reachable through a bare pool
            return
        self.running = True
        signals.started.emit(self)
        start = time.perf_counter()
        try:
            outcome = self.work(self)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a failure
            self._end(start)
            signals.failed.emit(self, f"{type(exc).__name__}: {exc}")
            return
        self._end(start)
        signals.done.emit(self, outcome)

    def _end(self, start: float) -> None:
        self.elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.running = False
        self.finished = True

    def __repr__(self) -> str:
        return (
            f"SessionJob({self.kind!r}, key={self.key!r}, long={self.long}, "
            f"label={self.label!r})"
        )


#: ``runner(key, lc, settings, backend)``: what :meth:`ComputeManager.submit_prewhiten`
#: delegates to when a session runner is attached.
PreWhitenRunner = Callable[[ResultKey, LoadedCurve, PreWhitenSettings, str], None]

#: Every live manager, so :func:`compute_active` can answer for the whole process.
_MANAGERS: weakref.WeakSet[ComputeManager] = weakref.WeakSet()


class ComputeManager(QObject):
    """Worker pool + generation counter; delivers results onto the GUI thread."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)  # serialize; compute is internally parallel
        self._generation = 0
        self.signals = _TaskSignals(self)
        self.session_signals = _SessionSignals(self)
        # Session jobs stay referenced until they report back (autoDelete is off).
        self._session_jobs: set[SessionJob] = set()
        self.session_signals.done.connect(self._release_job)
        self.session_signals.failed.connect(self._release_job)
        # The one-shot task a worker is executing, as (key, generation), or None.
        self._lock = threading.Lock()
        self._executing_task: tuple[ResultKey, int] | None = None
        self._session_runner: PreWhitenRunner | None = None
        _MANAGERS.add(self)

    def set_session_runner(self, runner: PreWhitenRunner | None) -> None:
        """Run :meth:`submit_prewhiten` as an interactive session's Auto instead.

        The GUI controller attaches one: a fresh pre-whitening run is then exactly
        the one-shot run it replaces (same input, same settings, same numbers), but
        it runs as a session job, so it keeps its session for editing and caching.
        ``None`` restores the one-shot :class:`PreWhitenTask`.
        """
        self._session_runner = runner

    @property
    def generation(self) -> int:
        """Current submit generation; older-generation tasks no-op."""
        return self._generation

    @property
    def active(self) -> bool:
        """Whether a worker is executing a task right now (it cannot be stopped)."""
        try:
            return self._pool.activeThreadCount() > 0
        except RuntimeError:  # the C++ pool is already gone (teardown)
            return False

    def submit(
        self,
        key: ResultKey,
        lc: LoadedCurve,
        method: str,
        settings: BaseSettings,
        backend: str,
    ) -> None:
        """Queue a compute, superseding any pending/in-flight one."""
        self._generation += 1
        task = PeriodogramTask(
            key, lc, method, settings, backend, self._generation, self, self.signals
        )
        self._pool.start(task)

    def submit_prewhiten(
        self,
        key: ResultKey,
        lc: LoadedCurve,
        settings: PreWhitenSettings,
        backend: str,
    ) -> None:
        """Queue a pre-whitening run, superseding any pending/in-flight compute.

        With a session runner attached (:meth:`set_session_runner`) the run is handed
        to it; the one-shot tasks are superseded all the same.
        """
        self._generation += 1
        if self._session_runner is not None:
            self._session_runner(key, lc, settings, backend)
            return
        task = PreWhitenTask(
            key, lc, settings, backend, self._generation, self, self.signals
        )
        self._pool.start(task)

    def cancel_all(self) -> None:
        """Supersede all pending/in-flight tasks (they no-op on completion).

        Session jobs are not affected: they never read the generation.
        """
        self._generation += 1

    # -- Stop for one-shot runs -------------------------------------------------------
    @contextmanager
    def executing(self, key: ResultKey, generation: int) -> Iterator[None]:
        """Mark the one-shot task a worker runs (for Stop and :attr:`finishing`)."""
        with self._lock:
            self._executing_task = (key, generation)
        try:
            yield
        finally:
            with self._lock:
                self._executing_task = None

    @property
    def running_key(self) -> ResultKey | None:
        """The key of the one-shot task a worker is executing right now, if any."""
        with self._lock:
            task = self._executing_task
        return None if task is None else task[0]

    @property
    def finishing(self) -> bool:
        """A superseded one-shot task still occupies the worker ("finishing…").

        New work queues behind it; nobody waits for its result.
        """
        with self._lock:
            task = self._executing_task
        return task is not None and task[1] != self._generation

    def cancel_running(self) -> ResultKey | None:
        """Stop: supersede every one-shot task at once, the executing one included.

        The library calls are not cooperative, so an executing task keeps the worker
        until it returns; its result is still delivered under its own key (a
        controller caches it, as it caches any superseded result) but nobody applies
        it once the controller has dropped its pending key. ``signals.cancelled`` is
        emitted synchronously with that task's key (``None`` when no one-shot task
        was executing), so the busy state can be released immediately.
        """
        self._generation += 1
        key = self.running_key
        self.signals.cancelled.emit(key)
        return key

    # -- session jobs -----------------------------------------------------------------
    def start_session_job(self, job: SessionJob) -> None:
        """Queue ``job`` on the worker pool (FIFO with every other task).

        Ordering between the jobs of one session is the caller's: submit a session's
        next job only after its previous one reported ``done`` or ``failed``.
        """
        job._signals = self.session_signals
        self._session_jobs.add(job)
        self._pool.start(job)

    def _release_job(self, job: SessionJob, _: object = None) -> None:
        self._session_jobs.discard(job)

    @property
    def session_jobs(self) -> int:
        """Session jobs queued or running."""
        return len(self._session_jobs)

    def wait_for_done(self, timeout_ms: int = -1) -> bool:
        """Block until the pool drains (test helper)."""
        return self._pool.waitForDone(timeout_ms)


def compute_active() -> bool:
    """Whether any :class:`ComputeManager` in this process is running a task."""
    return any(manager.active for manager in list(_MANAGERS))


__all__ = [
    "ComputeManager",
    "PeriodogramTask",
    "PreWhitenRunner",
    "PreWhitenTask",
    "SessionJob",
    "compute_active",
]
