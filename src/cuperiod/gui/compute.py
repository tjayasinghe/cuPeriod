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
"""

from __future__ import annotations

import time
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


class ComputeManager(QObject):
    """Worker pool + generation counter; delivers results onto the GUI thread."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)  # serialize; compute is internally parallel
        self._generation = 0
        self.signals = _TaskSignals(self)

    @property
    def generation(self) -> int:
        """Current submit generation; older-generation tasks no-op."""
        return self._generation

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
        """Queue a pre-whitening run, superseding any pending/in-flight compute."""
        self._generation += 1
        task = PreWhitenTask(
            key, lc, settings, backend, self._generation, self, self.signals
        )
        self._pool.start(task)

    def cancel_all(self) -> None:
        """Supersede all pending/in-flight tasks (they no-op on completion)."""
        self._generation += 1

    def wait_for_done(self, timeout_ms: int = -1) -> bool:
        """Block until the pool drains (test helper)."""
        return self._pool.waitForDone(timeout_ms)


__all__ = ["ComputeManager", "PeriodogramTask", "PreWhitenTask"]
