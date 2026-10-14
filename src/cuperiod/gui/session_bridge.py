"""The bridge between the GUI thread and the worker-confined pre-whitening sessions.

:class:`SessionBridge` owns every live :class:`~cuperiod.prewhiten.PreWhitenSession`
of the application (one per :class:`~cuperiod.gui.viewmodels.SessionKey`) and is the
only object that touches them; it is private to the controller. Its rules (design
section 4.2, decision D15):

* **Mutations are never discarded.** Every op, undo/redo/goto, Auto, Step, open and
  restore goes into its session's FIFO and runs, in order, on the compute worker — not
  dropped by a periodogram submit, ``cancel_all``, a source or analysis switch, nor by
  a newer submit. A session's next job is started only when its previous job has
  reported back, so one session is touched by one thread at a time.
* **Views and read queries are latest-wins.** A
  :class:`~cuperiod.gui.viewmodels.SessionView` is built only when the session's
  queue is empty after a job (five queued toggles → one view) and is emitted only
  while the session is displayed (else kept as its latest view). Each query kind has
  its own counter: a stale query is skipped at the head of the queue and dropped on
  arrival.
* **Per-job cancel tokens.** :meth:`cancel` stops the in-flight *long* job only; what it
  committed stands and the session stays valid; queued edits still run.
* **Id epochs.** Every committed ``prune`` / ``renumber`` / ``finalize`` and every
  history move bumps the session's id epoch; an id-addressed op sent with a stale
  ``expect_id_epoch`` is refused with a message, so a stale "F6" never hits another
  component.
* **Dirty sessions are never evicted.** Clean sessions live in an LRU of
  :data:`MAX_CLEAN_SESSIONS`.
* **Refusals carry their remedy.** A refused action is reported as a
  :class:`~cuperiod.gui.refusals.Refusal`: the message, the library's remedy code
  (``StepReport.hint``) and, for an op, a way to queue it again with the remedy's
  arguments. Such a retry runs only if nothing changed the session in between.

Session jobs never read :attr:`ComputeManager.generation` (see
:class:`~cuperiod.gui.compute.SessionJob`).
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from cuperiod.core._typing import FloatArray
from cuperiod.gui.compute import ComputeManager, SessionJob
from cuperiod.gui.qt import QObject, QtCore, Signal
from cuperiod.gui.refusals import Refusal, report_hint
from cuperiod.gui.viewmodels import (
    SessionKey,
    SessionView,
    SolutionBundle,
    build_session_view,
)

if TYPE_CHECKING:
    from cuperiod.core.config import PreWhitenSettings
    from cuperiod.prewhiten.result import PreWhitenResult
    from cuperiod.prewhiten.session.session import PreWhitenSession
    from cuperiod.prewhiten.session.state import SessionSnapshot, StepReport

#: Clean (saved or pristine) sessions kept alive besides the displayed one.
MAX_CLEAN_SESSIONS = 8

#: A quick job running longer than this shows the non-blocking busy indicator.
QUICK_BUSY_MS = 300

#: Ops that run long (busy state, Stop enabled, mutating commands disabled).
LONG_OPS: frozenset[str] = frozenset(
    {"step", "add_peak", "improve", "polish", "prune", "renumber", "finalize"}
)

#: Committed ops that may remap component ids.
_ID_REMAP_OPS: frozenset[str] = frozenset({"prune", "renumber", "finalize"})

#: Spectrum backends that hold no device memory worth releasing after a job.
_CPU_BACKENDS: frozenset[str] = frozenset({"numpy", "finufft"})

STALE_IDS_MESSAGE = (
    "The component list changed before this action ran; nothing was changed."
)

#: A retry offered for a refusal arrived after the session changed.
STALE_RETRY_MESSAGE = (
    "The session changed before your answer arrived; nothing was changed. Try the "
    "action again."
)

RequestKind = Literal["mutation", "view", "query", "open"]


@dataclass
class _Outcome:
    """What a job reports back to the GUI thread (built on the worker)."""

    reports: tuple[StepReport, ...] = ()
    committed: bool = False
    id_changed: bool = False
    replaced: bool = False
    result: PreWhitenResult | None = None
    bundle: SolutionBundle | None = None
    view: SessionView | None = None
    payload: Any = None
    dirty: bool | None = None
    pristine: bool | None = None
    backend: str = ""
    saved_path: str | None = None
    messages: tuple[tuple[str, str], ...] = ()
    cancelled: bool = False
    skipped: bool = False
    # The backend a session re-created by a factory was asked for (D17 re-open).
    requested_backend: str | None = None
    # A project load: what ``describe(session)`` returned (``loaded`` signal).
    loaded: Any = None
    # The submitted ops that were refused (their indices in the request), in order.
    refused_ops: tuple[int, ...] = ()


@dataclass
class _Request:
    """One queued unit of work for one session (GUI-thread data)."""

    kind: RequestKind
    label: str
    make: Callable[[], Callable[[SessionJob], _Outcome]]
    long: bool = False
    expect_id_epoch: int | None = None
    query: str = ""
    query_gen: int = 0
    preview: bool = False
    needs_session: bool = True
    # May leave the session non-pristine (a manual op, a history move, a configure):
    # while one is queued, the controller must not restart the session (D17).
    manual: bool = False
    # A project load: its ``loaded`` payload is announced on completion.
    load: bool = False
    # The ops of a submitted request, as given (a refusal's retry repeats them).
    ops: tuple[tuple[str, dict[str, Any]], ...] = ()
    # Run only if the session's change count is still this (a refusal's retry).
    expect_changes: int | None = None


@dataclass(eq=False)
class _Entry:
    """Everything the bridge knows about one session key."""

    key: SessionKey
    session: PreWhitenSession | None = None  # touched on the worker only
    queue: deque[_Request] = field(default_factory=deque)
    running: SessionJob | None = None
    running_request: _Request | None = None
    id_epoch: int = 0
    dirty: bool = False
    pristine: bool = True
    latest_view: SessionView | None = None
    need_view: bool = False
    pending_result: PreWhitenResult | None = None
    last_frequencies: FloatArray | None = None
    project_path: str | None = None
    requested_backend: str = ""
    warnings: list[str] = field(default_factory=list)
    query_gen: dict[str, int] = field(default_factory=dict)
    discarded: bool = False
    opened: bool = False  # an open/restore has been queued
    # Jobs that changed (or replaced) the session so far, in completion order.
    changes: int = 0

    @property
    def idle(self) -> bool:
        return self.running is None and not self.queue


def _settings_differ(a: PreWhitenSettings, b: PreWhitenSettings) -> bool:
    """Requested settings compared without ``backend`` / ``device`` (D17)."""
    exclude = {"backend", "device"}
    return a.model_dump(exclude=exclude) != b.model_dump(exclude=exclude)


def step_notice(report: StepReport, stop_reason: str = "") -> str:
    """What to tell the user about a Step that added no component ("" otherwise).

    Only a refusal used to reach the status bar, so a Step at the stopping rule did
    nothing at all, silently, and the button looked dead; a rejected or blocked
    candidate likewise left no trace outside the History dock.
    """
    kind = str(report.kind)
    if kind == "noop":
        why = f" ({stop_reason})" if stop_reason else ""
        return (
            f"Step: {report.message}{why}. Raise “max frequencies” or relax the "
            "stopping rule in Options and Step again, or add a peak yourself: Add "
            "peak (F6), or A at the spectrum cursor."
        )
    if kind in ("rejected", "blocked", "stopped"):
        return f"Step {kind}: {report.message}"
    return ""


def _state_of(session: PreWhitenSession) -> tuple[bool, bool]:
    """``(pristine, dirty)``: dirty = not pristine and changed since the last save."""
    pristine = bool(session.is_pristine)
    return pristine, (not pristine) and bool(session.is_dirty)


class SessionBridge(QObject):
    """Owns the sessions, their FIFOs, id epochs, views and the dirty set.

    Parameters
    ----------
    compute : ComputeManager
        The worker pool shared with the one-shot periodogram tasks.
    max_clean : int
        Clean sessions kept besides the displayed one (LRU).
    progress_interval : float
        Minimum seconds between two Auto progress views (a ``progress`` report is
        sent after every step regardless); ``0`` builds one after every step.

    Signals
    -------
    view_ready(key, SessionView)
        A view of the displayed session.
    progress(key, StepReport)
        An Auto or Step attempt committed (during Auto, after every step).
    message(severity, text)
        Warnings and notices (``"info"``, ``"warning"``, ``"error"``).
    refused(key, Refusal)
        An action of a finished job was refused: the first refusal of the job (the
        rest follow from it, like the tie of a combination whose add was refused),
        with its remedy code and, for submitted ops, a retry.
    failed(key, message)
        A job raised (not a refusal); the controller surfaces it as a failure.
    preview_failed(key, reason)
        A lazy preview open failed: placeholder text, never a failure banner.
    busy(bool)
        The displayed session has a long job queued or running.
    quick_busy(bool)
        A quick job of the displayed session has run longer than 300 ms.
    auto_finished(key, SolutionBundle)
        An Auto run completed (result and snapshot, for the controller's cache).
    finalized(key, PreWhitenResult)
        A ``finalize`` submitted as an op completed (full result).
    query_ready(kind, payload)
        ``"variations"`` → VariationResult; ``"errors"`` → PreWhitenResult;
        ``"entry_script"`` → ``(seq, text)``.
    evicted(key)
        A clean session was dropped from the LRU.
    reported(key, reports)
        The :class:`StepReport` s of a finished mutation (tuple; refusals included).
    saved(key, path)
        A save completed; the session is clean.
    loaded(key, payload)
        A project load completed; ``payload`` is what its ``describe`` returned.
    settled(key)
        ``key``'s queue ran dry (its last job, view included, has reported back).
    """

    view_ready = Signal(object, object)
    progress = Signal(object, object)
    message = Signal(str, str)
    refused = Signal(object, object)
    failed = Signal(object, str)
    preview_failed = Signal(object, str)
    busy = Signal(bool)
    quick_busy = Signal(bool)
    auto_finished = Signal(object, object)
    finalized = Signal(object, object)
    query_ready = Signal(str, object)
    evicted = Signal(object)
    reported = Signal(object, object)
    saved = Signal(object, str)
    loaded = Signal(object, object)
    settled = Signal(object)

    def __init__(
        self,
        compute: ComputeManager,
        *,
        max_clean: int = MAX_CLEAN_SESSIONS,
        progress_interval: float = 0.25,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._compute = compute
        self._max_clean = max(0, int(max_clean))
        self._progress_interval = max(0.0, float(progress_interval))
        self._entries: OrderedDict[SessionKey, _Entry] = OrderedDict()
        self._displayed: SessionKey | None = None
        self._busy = False
        self._quick_busy = False
        self._quick_timer = QtCore.QTimer(self)
        self._quick_timer.setSingleShot(True)
        self._quick_timer.setInterval(QUICK_BUSY_MS)
        self._quick_timer.timeout.connect(self._on_quick_timeout)
        # Jobs view builds requested by tests / diagnostics.
        self.view_builds = 0
        signals = compute.session_signals
        signals.done.connect(self._on_done)
        signals.failed.connect(self._on_failed)
        signals.progress.connect(self._on_progress)

    # -- introspection -------------------------------------------------------------
    @property
    def displayed(self) -> SessionKey | None:
        return self._displayed

    def keys(self) -> tuple[SessionKey, ...]:
        """Every live session key, least recently used first."""
        return tuple(self._entries)

    def has_session(self, key: SessionKey) -> bool:
        return key in self._entries

    def id_epoch(self, key: SessionKey) -> int:
        entry = self._entries.get(key)
        return 0 if entry is None else entry.id_epoch

    def latest_view(self, key: SessionKey) -> SessionView | None:
        entry = self._entries.get(key)
        return None if entry is None else entry.latest_view

    def is_dirty(self, key: SessionKey) -> bool:
        """Not pristine and changed since the last save (or never saved)."""
        entry = self._entries.get(key)
        return entry is not None and entry.dirty

    def dirty_keys(self) -> tuple[SessionKey, ...]:
        return tuple(k for k, e in self._entries.items() if e.dirty)

    def is_pristine(self, key: SessionKey) -> bool:
        """The session's log holds only automatic entries, and nothing queued or
        running could change that (a manual op, a history move, a configure).

        ``False`` for a key without a session. The pristine flag itself is the
        worker's answer after the last job, so a request still in the queue counts
        as already applied: the controller never restarts a session an edit is
        waiting for (D17).
        """
        entry = self._entries.get(key)
        if entry is None or not entry.opened:
            return False
        if entry.running_request is not None and entry.running_request.manual:
            return False
        return entry.pristine and not any(r.manual for r in entry.queue)

    def is_open(self, key: SessionKey) -> bool:
        """``key`` has a session, or an open / restore / load of one is queued."""
        entry = self._entries.get(key)
        return entry is not None and entry.opened

    def has_live_session(self, key: SessionKey) -> bool:
        """A session object exists for ``key`` (an open that failed leaves none)."""
        entry = self._entries.get(key)
        return entry is not None and entry.session is not None

    def project_path(self, key: SessionKey) -> str | None:
        """The project file ``key``'s session was last saved to or loaded from."""
        entry = self._entries.get(key)
        return None if entry is None else entry.project_path

    def set_project_path(self, key: SessionKey, path: str | None) -> None:
        """Record the project file of ``key``'s session (shown in later views)."""
        entry = self._entries.get(key)
        if entry is not None:
            entry.project_path = path

    def is_idle(self, key: SessionKey | None = None) -> bool:
        """No job queued or running (for ``key``, or for any session)."""
        entries = (
            [self._entries[key]]
            if key is not None and key in self._entries
            else [] if key is not None else list(self._entries.values())
        )
        return all(e.idle for e in entries)

    # -- sessions --------------------------------------------------------------------
    def open(
        self,
        key: SessionKey,
        factory: Callable[[], PreWhitenSession],
        *,
        preview: bool = False,
        backend: str | None = None,
    ) -> None:
        """Queue the creation of ``key``'s session by ``factory()`` (on the worker).

        A non-preview open replaces the live session of ``key`` once the jobs queued
        before it have run. A ``preview`` open (the lazy open of D22) is skipped when
        the key already has a session or is no longer displayed when it reaches the
        head of the queue, and a failure is reported through ``preview_failed``.
        ``backend`` is the requested backend the session was created for (compared
        by :meth:`auto` / :meth:`step`); defaults to the session's own.
        """
        entry = self._entry(key)
        if preview and entry.opened:
            return
        entry.opened = True

        def make() -> Callable[[SessionJob], _Outcome]:
            def work(job: SessionJob) -> _Outcome:
                session = factory()
                entry.session = session
                return self._after(session, _Outcome(replaced=True, id_changed=True))

            return work

        self._enqueue(
            entry,
            _Request(
                kind="open",
                label="Opening session",
                make=make,
                preview=preview,
                needs_session=False,
            ),
            requested_backend=backend,
        )

    def restore(
        self,
        key: SessionKey,
        snapshot: SessionSnapshot,
        data: Any,
        *,
        backend: str | None = None,
        labels: Mapping[str, Any] | None = None,
        requested_backend: str | None = None,
    ) -> None:
        """Queue ``PreWhitenSession.restore(snapshot, data)``: exact, no re-run.

        ``labels`` are the per-point label columns the session was created with (the
        ``band`` column of a stacked multiband curve). ``backend=None`` uses the
        recorded backend when this machine has it; ``requested_backend`` is what the
        user asked for originally (``"auto"``), compared by :meth:`auto` / :meth:`step`.
        """
        entry = self._entry(key)
        entry.opened = True

        def make() -> Callable[[SessionJob], _Outcome]:
            def work(job: SessionJob) -> _Outcome:
                from cuperiod.prewhiten.session.session import PreWhitenSession

                session = PreWhitenSession.restore(
                    snapshot, data, backend=backend, labels=labels
                )
                entry.session = session
                return self._after(session, _Outcome(replaced=True, id_changed=True))

            return work

        self._enqueue(
            entry,
            _Request(
                kind="open", label="Restoring session", make=make, needs_session=False
            ),
            requested_backend=requested_backend or backend,
        )

    def load(
        self,
        key: SessionKey,
        loader: Callable[[], PreWhitenSession],
        *,
        project_path: str,
        describe: Callable[[PreWhitenSession], Any] | None = None,
        backend: str | None = None,
    ) -> None:
        """Queue a project load (long): ``loader()`` becomes ``key``'s session.

        On completion ``loaded(key, describe(session))`` is emitted before the view
        is built, so the controller can show the project's curve first. A failure is
        reported through ``failed``; the key then holds no session.
        """
        entry = self._entry(key)
        entry.opened = True
        entry.project_path = project_path

        def make() -> Callable[[SessionJob], _Outcome]:
            def work(job: SessionJob) -> _Outcome:
                session = loader()
                entry.session = session
                payload = describe(session) if describe is not None else None
                out = _Outcome(
                    replaced=True,
                    id_changed=True,
                    loaded=payload,
                    requested_backend=backend or session.backend,
                )
                return self._after(session, out)

            return work

        self._enqueue(
            entry,
            _Request(
                kind="open",
                label="Opening project",
                make=make,
                long=True,
                needs_session=False,
                load=True,
            ),
            requested_backend=backend,
        )

    def discard(self, key: SessionKey) -> None:
        """Drop ``key``'s session and everything queued for it (user decision)."""
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        entry.discarded = True
        entry.queue.clear()
        if entry.running is not None:
            entry.running.cancel()
        if key == self._displayed:
            self._update_busy()

    def display(self, key: SessionKey | None) -> None:
        """Make ``key`` the displayed session; its latest view is emitted at once."""
        self._displayed = key
        entry = self._entries.get(key) if key is not None else None
        if entry is not None:
            self._entries.move_to_end(key)  # type: ignore[arg-type]
            if entry.latest_view is not None:
                self.view_ready.emit(key, entry.latest_view)
        self._update_busy()
        self._evict()

    # -- mutations -------------------------------------------------------------------
    def submit(
        self,
        key: SessionKey,
        op: str,
        args: Mapping[str, Any] | None = None,
        *,
        expect_id_epoch: int | None = None,
    ) -> None:
        """Queue ``session.do(op, **args)``; ``"finalize"`` is polish + prune.

        ``("add", {"frequency": f, "snap": True, ...})`` — an add with ``snap`` among
        its arguments — goes through :meth:`PreWhitenSession.add`, which snaps ``f``
        to the residual peak on the worker (only the worker has the spectrum).
        """
        self.submit_group(
            key, [(op, dict(args or {}))], expect_id_epoch=expect_id_epoch
        )

    def submit_group(
        self,
        key: SessionKey,
        actions: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        expect_id_epoch: int | None = None,
        expect_changes: int | None = None,
    ) -> None:
        """Queue several ops that undo as one group.

        ``expect_changes``: run only if the session's change count is still that
        (a refusal's retry); otherwise the request is dropped with a message.
        """
        entry = self._live_entry(key)
        if entry is None:
            return
        from cuperiod.prewhiten.session.actions import PRISTINE_OPS

        ops = [(op, dict(args)) for op, args in actions]
        long = any(op in LONG_OPS for op, _ in ops)
        label = ", ".join(op for op, _ in ops)
        manual = any(op not in PRISTINE_OPS and op != "finalize" for op, _ in ops)

        def make() -> Callable[[SessionJob], _Outcome]:
            def work(job: SessionJob) -> _Outcome:
                session = self._session(entry)
                return self._after(session, self._run_ops(session, ops))

            return work

        self._enqueue(
            entry,
            _Request(
                kind="mutation",
                label=label,
                make=make,
                long=long,
                expect_id_epoch=expect_id_epoch,
                manual=manual,
                ops=tuple(ops),
                expect_changes=expect_changes,
            ),
        )

    def submit_call(
        self,
        key: SessionKey,
        label: str,
        call: Callable[[PreWhitenSession], StepReport | Sequence[StepReport] | None],
        *,
        long: bool = False,
        expect_id_epoch: int | None = None,
        manual: bool = True,
        notice: str = "",
    ) -> None:
        """Queue ``call(session)`` as one mutation (one undo group) in the FIFO.

        For the edits whose arguments only the worker can compute: a partition
        resolved against the session's own time stamps, a point selection turned into
        index runs, a frequency snapped to the residual peak. ``call`` returns the
        :class:`StepReport` (s) of what it did (refusals become messages);
        ``SessionError`` / ``ValueError`` raised by it are refusals too (D21).
        """
        entry = self._live_entry(key)
        if entry is None:
            return

        def make() -> Callable[[SessionJob], _Outcome]:
            def work(job: SessionJob) -> _Outcome:
                from cuperiod.prewhiten.session.state import SessionError

                session = self._session(entry)
                out = _Outcome()
                try:
                    with session.group():
                        got = call(session)
                except (SessionError, ValueError) as exc:
                    out.messages = (("warning", str(exc)),)
                    return self._after(session, out)
                reports = (
                    ()
                    if got is None
                    else (got,)
                    if not isinstance(got, Sequence)
                    else tuple(got)
                )
                out.reports = reports
                out.committed = any(r.committed for r in reports)
                out.id_changed = any(
                    r.committed and "ids" in r.changed for r in reports
                )
                if notice:
                    out.messages = (("info", notice),)
                return self._after(session, out)

            return work

        self._enqueue(
            entry,
            _Request(
                kind="mutation",
                label=label,
                make=make,
                long=long,
                expect_id_epoch=expect_id_epoch,
                manual=manual,
            ),
        )

    def _run_ops(
        self, session: PreWhitenSession, ops: Sequence[tuple[str, dict[str, Any]]]
    ) -> _Outcome:
        reports: list[StepReport] = []
        refused: list[int] = []
        out = _Outcome()
        with session.group():
            for index, (op, args) in enumerate(ops):
                if op == "finalize":
                    out.result = session.finalize()
                    out.committed = out.id_changed = True
                    continue
                report = self._do(session, op, args)
                reports.append(report)
                if report.kind == "refused":
                    refused.append(index)
                if report.committed:
                    out.committed = True
                    out.id_changed = out.id_changed or op in _ID_REMAP_OPS
        out.reports = tuple(reports)
        out.refused_ops = tuple(refused)
        return out

    @staticmethod
    def _do(
        session: PreWhitenSession, op: str, args: Mapping[str, Any]
    ) -> StepReport:
        """``session.do(op, **args)``; an ``add`` with ``snap`` among its arguments
        goes through :meth:`PreWhitenSession.add` (it snaps on the worker)."""
        if op == "add" and "snap" in args:
            from cuperiod.prewhiten.session.state import SessionError, StepReport

            options = dict(args)
            frequency = options.pop("frequency", None)
            try:
                return session.add(frequency, **options)
            except (SessionError, ValueError) as exc:
                # As do() would: a refusal, not a failed job (D21).
                return StepReport(kind="refused", message=str(exc))
        return session.do(op, **args)

    def undo(self, key: SessionKey) -> None:
        self._history_move(key, "undo")

    def redo(self, key: SessionKey) -> None:
        self._history_move(key, "redo")

    def goto(self, key: SessionKey, seq: int) -> None:
        self._history_move(key, "goto", seq)

    def _history_move(
        self, key: SessionKey, move: Literal["undo", "redo", "goto"], seq: int = 0
    ) -> None:
        entry = self._live_entry(key)
        if entry is None:
            return

        def make() -> Callable[[SessionJob], _Outcome]:
            def work(job: SessionJob) -> _Outcome:
                session = self._session(entry)
                messages: tuple[tuple[str, str], ...] = ()
                if move == "goto":
                    report = session.goto(seq)
                    moved = report.committed
                    if not moved:
                        messages = (("warning", report.message),)
                else:
                    moved = session.undo() if move == "undo" else session.redo()
                    if not moved:
                        messages = (("info", f"Nothing to {move}."),)
                out = _Outcome(committed=moved, id_changed=moved, messages=messages)
                return self._after(session, out)

            return work

        self._enqueue(
            entry, _Request(kind="mutation", label=move, make=make, manual=True)
        )

    def auto(
        self,
        key: SessionKey,
        settings: PreWhitenSettings,
        backend: str | None,
        *,
        factory: Callable[[], PreWhitenSession] | None = None,
        restart: bool = False,
        tag: object = None,
    ) -> None:
        """Queue Auto: ``run()`` to the stopping rule, then ``finalize()`` (long).

        The live session is continued (case 1 of 4.2): a different requested backend
        is refused, settings that differ from the head's are applied by
        ``configure`` first (``backend`` / ``device`` kept). With ``factory``, a
        session whose log is empty is simply re-created instead (D17); with
        ``restart`` as well, the session is re-created whatever its log holds (case
        3: the controller restarts only a pristine session). Emits ``progress``
        after every step, light views on the way (at most every
        ``progress_interval`` s), ``auto_finished`` (whose bundle carries ``tag``)
        and a full view at the end.
        """
        self._attempt(
            key, settings, backend, factory, "auto", restart=restart, tag=tag
        )

    def step(
        self,
        key: SessionKey,
        settings: PreWhitenSettings,
        backend: str | None,
        *,
        factory: Callable[[], PreWhitenSession] | None = None,
        manual: bool = False,
    ) -> None:
        """Queue one Step (D17: never restarts; configures when the form differs).

        ``manual`` tells :meth:`is_pristine` that the step will configure the
        session (the caller knows the form differs from the session's settings).
        """
        self._attempt(key, settings, backend, factory, "step", manual=manual)

    def improve(
        self,
        key: SessionKey,
        settings: PreWhitenSettings,
        backend: str | None,
        *,
        ids: Sequence[int] | None = None,
        factory: Callable[[], PreWhitenSession] | None = None,
        expect_id_epoch: int | None = None,
    ) -> None:
        """Queue Improve (all, or ``ids``) after a ``configure`` to the form (D17).

        Like :meth:`step`: never restarts, re-opens a session whose log is empty,
        refuses a backend change. The configure and the improve undo as one group.
        """
        chosen = None if ids is None else [int(i) for i in ids]
        self._attempt(
            key,
            settings,
            backend,
            factory,
            "improve",
            manual=True,
            ids=chosen,
            expect_id_epoch=expect_id_epoch,
        )

    def _attempt(
        self,
        key: SessionKey,
        settings: PreWhitenSettings,
        backend: str | None,
        factory: Callable[[], PreWhitenSession] | None,
        what: Literal["auto", "step", "improve"],
        *,
        restart: bool = False,
        tag: object = None,
        manual: bool = False,
        ids: list[int] | None = None,
        expect_id_epoch: int | None = None,
    ) -> None:
        entry = self._entries.get(key)
        if entry is None and factory is not None:
            entry = self._entry(key)
            entry.opened = True
            if backend is not None:
                entry.requested_backend = backend
        elif entry is None:
            self._no_session()
            return
        if factory is not None:
            entry.opened = True

        def make() -> Callable[[SessionJob], _Outcome]:
            epoch = entry.id_epoch
            requested = entry.requested_backend
            interval = self._progress_interval

            def work(job: SessionJob) -> _Outcome:
                out = _Outcome()
                session = entry.session
                if factory is not None and (
                    restart or session is None or not session.history
                ):
                    session = factory()
                    entry.session = session
                    out.replaced = out.id_changed = True
                    if backend is not None:
                        out.requested_backend = backend
                if session is None:
                    out.messages = (("warning", "No pre-whitening session is open."),)
                    return out
                if (
                    not out.replaced
                    and backend is not None
                    and backend not in (requested, session.backend)
                ):
                    out.messages = (
                        (
                            "warning",
                            f"This session runs on {session.backend!r}; the backend "
                            "choice applies to new sessions.",
                        ),
                    )
                    return self._after(session, out)
                if what == "improve":
                    # One undo group: the configure the form asked for and the
                    # improve it was asked for.
                    with session.group():
                        self._configure(session, settings, out)
                        if out.messages:
                            return self._after(session, out)
                        report = session.do("improve", ids=ids)
                    out.reports = (report,)
                    out.committed = out.committed or report.committed
                    return self._after(session, out)
                self._configure(session, settings, out)
                if out.messages:
                    return self._after(session, out)
                if what == "step":
                    report = session.step()
                    job.report(("report", report))
                    out.reports = (report,)  # a refusal becomes a message on arrival
                    out.committed = out.committed or report.committed
                    notice = step_notice(report, session.state.stop_reason)
                    if notice:
                        out.messages = (("info", notice),)
                    return self._after(session, out)
                return self._after(
                    session,
                    self._auto(entry, session, job, out, epoch, interval, tag),
                )

            return work

        labels = {"auto": "Auto", "step": "Step", "improve": "Improve"}
        self._enqueue(
            entry,
            _Request(
                kind="mutation",
                label=labels[what],
                make=make,
                long=True,
                needs_session=factory is None,
                manual=manual,
                expect_id_epoch=expect_id_epoch,
            ),
        )

    @staticmethod
    def _configure(
        session: PreWhitenSession, settings: PreWhitenSettings, out: _Outcome
    ) -> None:
        """Apply the form's settings when they differ (``backend``/``device`` kept).

        A refusal is left in ``out.messages`` (the caller stops there).
        """
        current = session.settings
        if not _settings_differ(settings, current):
            return
        wanted = settings.model_copy(
            update={"backend": current.backend, "device": current.device}
        )
        report = session.configure(wanted)
        if not report.committed and report.kind == "refused":
            out.messages = (("warning", report.message),)
            return
        out.committed = out.committed or report.committed

    def _auto(
        self,
        entry: _Entry,
        session: PreWhitenSession,
        job: SessionJob,
        out: _Outcome,
        epoch: int,
        interval: float,
        tag: object = None,
    ) -> _Outcome:
        """``run()`` with progress and a cancel check per step, then ``finalize()``.

        The loop is :meth:`PreWhitenSession.run` (same group, same exit rules) spelled
        out so a light view can be built between steps, outside the session's guard.
        """
        reports: list[StepReport] = []
        last_view = -float("inf")  # the first committed step always shows
        with session.group():
            while session.status != "stopped":
                if job.should_stop():
                    out.cancelled = True
                    break
                report = session.step()
                reports.append(report)
                job.report(("report", report))
                if report.committed:
                    out.committed = True
                    now = time.perf_counter()
                    if now - last_view >= interval:
                        last_view = now
                        job.report(
                            (
                                "view",
                                build_session_view(
                                    session,
                                    key=entry.key,
                                    id_epoch=epoch,
                                    progress=True,
                                    dirty=_state_of(session)[1],
                                    project_path=entry.project_path,
                                    requested_backend=entry.requested_backend,
                                    status=f"Auto — {report.message}",
                                ),
                            )
                        )
                if report.kind in ("refused", "noop"):
                    break
        out.reports = tuple(reports)
        if out.cancelled:
            out.messages = (
                ("info", f"Auto stopped after {len(reports)} step(s); kept."),
            )
            return out
        result = session.finalize()
        out.result = result
        out.committed = out.id_changed = True
        out.bundle = SolutionBundle(result, session.snapshot(with_states=True), tag)
        return out

    def cancel(self, key: SessionKey | None = None) -> bool:
        """Stop the in-flight *long* job of ``key`` (default: the displayed one).

        Returns whether a job was asked to stop. Queued jobs are untouched.
        """
        key = self._displayed if key is None else key
        entry = self._entries.get(key) if key is not None else None
        if entry is None or entry.running is None or not entry.running.long:
            return False
        entry.running.cancel()
        return True

    # -- read queries ----------------------------------------------------------------
    def query_variations(
        self,
        key: SessionKey,
        ids: Sequence[int] | None,
        mode: str,
        by: str | None,
        improve_frequencies: bool = False,
    ) -> None:
        """Per-subset amplitude/phase (``query_ready("variations", result)``)."""
        chosen = None if ids is None else [int(i) for i in ids]

        def compute(session: PreWhitenSession, job: SessionJob) -> Any:
            return session.variations(
                chosen,
                mode=mode,  # type: ignore[arg-type]
                by=by,
                improve_frequencies=improve_frequencies,
                should_stop=job.should_stop,
            )

        self._query(key, "variations", compute, long=improve_frequencies)

    def compute_errors(self, key: SessionKey) -> None:
        """The full result with the configured estimator, and a full view (long)."""

        def compute(session: PreWhitenSession, job: SessionJob) -> Any:
            return session.result(should_stop=job.should_stop)

        self._query(key, "errors", compute, long=True, full_view=True)

    def entry_script(self, key: SessionKey, seq: int) -> None:
        """One log entry as Python: ``query_ready("entry_script", (seq, text))``."""

        def compute(session: PreWhitenSession, job: SessionJob) -> Any:
            from cuperiod.prewhiten.session import script

            for e in session.log_entries:  # the redo tail included
                if e.seq == seq:
                    return (seq, script.render_entry(e))
            raise KeyError(f"no log entry {seq}")

        self._query(key, "entry_script", compute)

    def _query(
        self,
        key: SessionKey,
        kind: str,
        compute: Callable[[PreWhitenSession, SessionJob], Any],
        *,
        long: bool = False,
        full_view: bool = False,
    ) -> None:
        entry = self._live_entry(key)
        if entry is None:
            return
        gen = entry.query_gen.get(kind, 0) + 1
        entry.query_gen[kind] = gen

        def make() -> Callable[[SessionJob], _Outcome]:
            epoch = entry.id_epoch
            project_path = entry.project_path
            requested = entry.requested_backend

            def work(job: SessionJob) -> _Outcome:
                if entry.query_gen.get(kind) != gen:
                    # Superseded while it waited in the pool: skip the computation.
                    return _Outcome(skipped=True)
                session = self._session(entry)
                payload = compute(session, job)
                out = _Outcome(payload=payload)
                if full_view and not job.should_stop():
                    pristine, dirty = _state_of(session)
                    out.view = build_session_view(
                        session,
                        key=entry.key,
                        id_epoch=epoch,
                        result=payload,
                        dirty=dirty,
                        project_path=project_path,
                        requested_backend=requested,
                    )
                return self._after(session, out)

            return work

        self._enqueue(
            entry,
            _Request(
                kind="query",
                label=kind,
                make=make,
                long=long,
                query=kind,
                query_gen=gen,
            ),
        )

    # -- persistence -----------------------------------------------------------------
    def save(self, key: SessionKey, path: str) -> None:
        """Queue ``session.save(path)`` in order with the edits (clears dirty)."""
        entry = self._live_entry(key)
        if entry is None:
            return

        def make() -> Callable[[SessionJob], _Outcome]:
            def work(job: SessionJob) -> _Outcome:
                session = self._session(entry)
                session.save(path)
                out = _Outcome(
                    saved_path=str(path), messages=(("info", f"Saved {path}"),)
                )
                return self._after(session, out)

            return work

        self._enqueue(entry, _Request(kind="mutation", label="Save", make=make))

    # -- internals: queueing ---------------------------------------------------------
    def _entry(self, key: SessionKey) -> _Entry:
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry(key)
            self._entries[key] = entry
        return entry

    def _live_entry(self, key: SessionKey) -> _Entry | None:
        entry = self._entries.get(key)
        if entry is None or not entry.opened:
            self._no_session()
            return None
        return entry

    def _no_session(self) -> None:
        self.message.emit("warning", "No pre-whitening session is open.")

    @staticmethod
    def _session(entry: _Entry) -> PreWhitenSession:
        session = entry.session
        if session is None:  # pragma: no cover - guarded by _pump
            raise RuntimeError("the session is not open")
        return session

    def _after(self, session: PreWhitenSession, out: _Outcome) -> _Outcome:
        """Worker-side epilogue of every job: state flags, device release."""
        out.pristine, out.dirty = _state_of(session)
        out.backend = session.backend
        if session.backend.split(":")[0] not in _CPU_BACKENDS:
            session.release()  # no device memory held between jobs (D22)
        return out

    def _enqueue(
        self,
        entry: _Entry,
        request: _Request,
        *,
        requested_backend: str | None = None,
    ) -> None:
        if requested_backend is not None:
            entry.requested_backend = requested_backend
        entry.queue.append(request)
        self._entries.move_to_end(entry.key)
        self._pump(entry)
        self._update_busy()

    def _pump(self, entry: _Entry) -> None:
        """Start the next runnable request of ``entry`` (one at a time)."""
        if entry.running is not None or entry.discarded:
            return
        while entry.queue:
            req = entry.queue.popleft()
            if req.kind == "view" and entry.queue:
                entry.need_view = True  # superseded: a later view follows
                continue
            if req.kind == "query" and req.query_gen != entry.query_gen.get(req.query):
                continue  # stale at the head of the queue
            if req.preview and (
                entry.key != self._displayed or entry.session is not None
            ):
                continue  # a preview of a key no longer shown holds no user work
            if req.needs_session and entry.session is None:
                if req.kind == "mutation":
                    self.message.emit("warning", "No pre-whitening session is open.")
                continue
            epoch = req.expect_id_epoch
            if epoch is not None and epoch != entry.id_epoch:
                self.message.emit("warning", STALE_IDS_MESSAGE)
                continue
            if req.expect_changes is not None and req.expect_changes != entry.changes:
                # The user answered a question about a session that has moved on.
                self.message.emit("warning", STALE_RETRY_MESSAGE)
                continue
            job = SessionJob(
                req.make(),
                key=entry.key,
                kind=req.kind,
                long=req.long,
                label=req.label,
                tag=req,
            )
            entry.running = job
            entry.running_request = req
            if not req.long and not req.preview and entry.key == self._displayed:
                self._quick_timer.start()
            self._compute.start_session_job(job)
            return
        # Nothing left to run (a finished job or skipped requests): the queue ran dry.
        self.settled.emit(entry.key)

    def _request_view(self, entry: _Entry) -> None:
        if not entry.need_view or entry.session is None:
            return
        entry.need_view = False

        def make() -> Callable[[SessionJob], _Outcome]:
            epoch = entry.id_epoch
            result = entry.pending_result
            entry.pending_result = None
            previous = entry.last_frequencies
            project_path = entry.project_path
            requested = entry.requested_backend
            warnings = tuple(entry.warnings)
            entry.warnings.clear()

            def work(job: SessionJob) -> _Outcome:
                if entry.queue:
                    # Work arrived behind this view while it waited in the pool: a
                    # later view will show its effect, so skip the build.
                    return _Outcome(skipped=True, result=result, payload=warnings)
                session = self._session(entry)
                pristine, dirty = _state_of(session)
                view = build_session_view(
                    session,
                    key=entry.key,
                    id_epoch=epoch,
                    result=result,
                    previous_frequencies=previous,
                    dirty=dirty,
                    project_path=project_path,
                    warnings=warnings,
                    requested_backend=requested,
                )
                return self._after(session, _Outcome(view=view))

            return work

        self.view_builds += 1
        entry.queue.append(_Request(kind="view", label="view", make=make))
        self._pump(entry)

    # -- internals: completion -------------------------------------------------------
    def _finish_job(self, job: SessionJob) -> tuple[_Entry, _Request] | None:
        key = job.key
        entry = self._entries.get(key) if isinstance(key, SessionKey) else None
        if entry is None or entry.running is not job:
            return None  # discarded meanwhile
        req = entry.running_request
        entry.running = None
        entry.running_request = None
        if self._quick_timer.isActive():
            self._quick_timer.stop()
        if self._quick_busy:
            self._quick_busy = False
            self.quick_busy.emit(False)
        assert req is not None
        return entry, req

    def _on_done(self, job: SessionJob, outcome: object) -> None:
        found = self._finish_job(job)
        if found is None:
            return
        entry, req = found
        assert isinstance(outcome, _Outcome)
        key = entry.key
        if outcome.committed or outcome.replaced:
            entry.changes += 1
        for severity, text in outcome.messages:
            self.message.emit(severity, text)
        refused = [r for r in outcome.reports if r.kind == "refused"]
        refusal = (
            self._refusal(entry, req, outcome, refused[0]) if refused else None
        )
        for report in outcome.reports:
            if report.kind != "refused" and report.warning:
                entry.warnings.append(report.warning)
        # An action that changed nothing ("nothing to improve", "ids are already
        # compact") said so nowhere, so its button looked dead.
        if (
            req.kind == "mutation"
            and not outcome.messages
            and outcome.result is None
            and len(outcome.reports) == 1
            and outcome.reports[0].kind == "noop"
            and outcome.reports[0].message
        ):
            self.message.emit("info", f"Nothing changed: {outcome.reports[0].message}")
        if outcome.pristine is not None:
            entry.pristine = outcome.pristine
        if outcome.dirty is not None:
            entry.dirty = outcome.dirty
        if outcome.requested_backend:
            entry.requested_backend = outcome.requested_backend
        if outcome.backend and not entry.requested_backend:
            entry.requested_backend = outcome.backend
        if outcome.saved_path is not None:
            entry.project_path = outcome.saved_path
        if outcome.id_changed:
            entry.id_epoch += 1
        if outcome.replaced:
            entry.last_frequencies = None
            entry.pending_result = None
        if req.kind == "query":
            if (
                not outcome.skipped
                and req.query_gen == entry.query_gen.get(req.query)
                and not job.cancelled
            ):
                self.query_ready.emit(req.query, outcome.payload)
                if outcome.view is not None:
                    self._deliver(entry, outcome.view)
        elif req.kind == "view":
            if outcome.skipped:
                if entry.pending_result is None:
                    entry.pending_result = outcome.result
                entry.warnings[:0] = list(outcome.payload or ())
                entry.need_view = True
            elif outcome.view is not None:
                self._deliver(entry, outcome.view)
        else:
            if outcome.result is not None:
                entry.pending_result = outcome.result
            elif outcome.committed:
                entry.pending_result = None  # a later edit made it stale
            if outcome.committed or outcome.replaced or outcome.cancelled:
                entry.need_view = True
            if outcome.saved_path is not None and entry.latest_view is not None:
                entry.latest_view = replace(
                    entry.latest_view,
                    dirty=entry.dirty,
                    project_path=entry.project_path,
                )
                if key == self._displayed:
                    self.view_ready.emit(key, entry.latest_view)
        if req.kind == "mutation" and outcome.reports:
            self.reported.emit(key, outcome.reports)
        if outcome.saved_path is not None:
            self.saved.emit(key, outcome.saved_path)
        if req.load:
            self.loaded.emit(key, outcome.loaded)
        if outcome.bundle is not None:
            self.auto_finished.emit(key, outcome.bundle)
        elif req.kind == "mutation" and outcome.result is not None:
            self.finalized.emit(key, outcome.result)
        self._continue(entry)
        if refusal is not None:
            # Last, once this job is fully accounted for: a retry the user accepts
            # (at once, for a scripted answer) queues behind what is queued already.
            self.refused.emit(key, refusal)

    def _refusal(
        self, entry: _Entry, req: _Request, outcome: _Outcome, report: StepReport
    ) -> Refusal:
        """The :class:`Refusal` of a job's first refused ``report``.

        For submitted ops the refused ones (the first and those that followed it,
        such as the tie of a combination whose add was refused) can be queued again,
        with the arguments of a remedy the user accepts.
        """
        op = ""
        args: Mapping[str, Any] = {}
        retry: Callable[[Mapping[str, Any]], None] | None = None
        if req.ops and outcome.refused_ops:
            op, args = req.ops[outcome.refused_ops[0]]
            again = tuple(req.ops[i] for i in outcome.refused_ops)
            retry = partial(self._retry, entry.key, again, entry.changes)
        return Refusal(
            key=entry.key,
            text=report.message,
            hint=report_hint(report),
            op=op,
            args=dict(args),
            retry=retry,
        )

    def _retry(
        self,
        key: SessionKey,
        ops: Sequence[tuple[str, Mapping[str, Any]]],
        changes: int,
        extra: Mapping[str, Any],
    ) -> None:
        """Queue refused ``ops`` again, each with those of the ``extra`` arguments its
        operation takes, unless the session changed since they were refused."""
        from cuperiod.prewhiten.session.actions import ARGS

        patched = [
            (
                op,
                {
                    **args,
                    **{
                        name: value
                        for name, value in extra.items()
                        if op in ARGS and name in ARGS[op].model_fields
                    },
                },
            )
            for op, args in ops
        ]
        self.submit_group(key, patched, expect_changes=changes)

    def _on_failed(self, job: SessionJob, text: str) -> None:
        found = self._finish_job(job)
        if found is None:
            return
        entry, req = found
        if req.kind == "mutation":
            entry.changes += 1  # it may have committed some of its actions
        if req.preview:
            entry.opened = False
            self.preview_failed.emit(entry.key, text)
        else:
            if entry.session is None:
                entry.opened = False  # an open that failed: nothing to act on
            self.failed.emit(entry.key, text)
            if entry.session is not None and req.kind == "mutation":
                entry.need_view = True  # show whatever the session holds now
        self._continue(entry)

    def _continue(self, entry: _Entry) -> None:
        if not entry.queue:
            self._request_view(entry)
        self._pump(entry)
        self._update_busy()
        self._evict()

    def _on_progress(self, job: SessionJob, payload: object) -> None:
        key = job.key
        entry = self._entries.get(key) if isinstance(key, SessionKey) else None
        if entry is None or entry.running is not job:
            return
        if not isinstance(payload, tuple) or len(payload) != 2:
            return
        what, value = payload
        if what == "report":
            self.progress.emit(key, value)
        elif what == "view" and isinstance(value, SessionView):
            self._deliver(entry, value, progress=True)

    def _deliver(
        self, entry: _Entry, view: SessionView, *, progress: bool = False
    ) -> None:
        entry.latest_view = view
        if not progress:
            # The baseline for "did the active frequencies move" (light results);
            # Auto progress views never carry one, so they do not move it.
            entry.last_frequencies = view.active_frequencies
        if entry.key == self._displayed:
            self.view_ready.emit(entry.key, view)

    # -- internals: busy and eviction ------------------------------------------------
    def _update_busy(self) -> None:
        entry = self._entries.get(self._displayed) if self._displayed else None
        busy = entry is not None and (
            (entry.running is not None and entry.running.long)
            or any(r.long for r in entry.queue)
        )
        if busy != self._busy:
            self._busy = busy
            self.busy.emit(busy)

    @property
    def is_busy(self) -> bool:
        """The displayed session has a long job queued or running."""
        return self._busy

    def _on_quick_timeout(self) -> None:
        entry = self._entries.get(self._displayed) if self._displayed else None
        if entry is not None and entry.running is not None and not self._quick_busy:
            self._quick_busy = True
            self.quick_busy.emit(True)

    def _evict(self) -> None:
        clean = [
            k
            for k, e in self._entries.items()
            if not e.dirty and k != self._displayed
        ]
        excess = len(clean) - self._max_clean
        for k in clean:
            if excess <= 0:
                break
            entry = self._entries[k]
            if not entry.idle:
                continue
            del self._entries[k]
            entry.discarded = True
            entry.session = None
            excess -= 1
            self.evicted.emit(k)


__all__ = [
    "LONG_OPS",
    "MAX_CLEAN_SESSIONS",
    "QUICK_BUSY_MS",
    "STALE_IDS_MESSAGE",
    "STALE_RETRY_MESSAGE",
    "Refusal",
    "SessionBridge",
    "SessionKey",
]
