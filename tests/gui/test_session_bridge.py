"""The session bridge: FIFO mutations, coalesced views, queries, cancel, id epochs.

Sessions run on ``backend="numpy"`` (no CUDA context). Only automatic ops (step,
polish, prune, renumber, finalize, undo/redo/goto) run real reducers; the one manual
edit the dirty/epoch tests need (``set_note``) is a test-local reducer.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

import cuperiod as cup
from cuperiod.core.config import GLSSettings, PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve
from cuperiod.gui.compute import ComputeManager, SessionJob
from cuperiod.gui.models import ResultKey
from cuperiod.gui.session_bridge import (
    STALE_IDS_MESSAGE,
    SessionBridge,
    SessionKey,
)
from cuperiod.gui.viewmodels import SessionView, SolutionBundle
from cuperiod.prewhiten.session import actions
from cuperiod.prewhiten.session.state import StepReport

KEY = SessionKey("star", "")
OTHER = SessionKey("other", "")


def _data(n: int = 300, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 40.0, n))
    y = (
        0.3 * np.sin(2 * np.pi * 1.3 * t)
        + 0.12 * np.sin(2 * np.pi * 2.71 * t + 1.0)
        + 0.05 * np.sin(2 * np.pi * 0.61 * t + 2.0)
        + rng.normal(0.0, 0.01, n)
    )
    return t, y, np.full(n, 0.01)


DATA = _data()


def _factory(
    record: list[str] | None = None, data: Any = DATA
) -> Callable[[], cup.PreWhitenSession]:
    """A session factory; ``record`` collects every op the session executes."""

    def make() -> cup.PreWhitenSession:
        session = cup.PreWhitenSession(data, backend="numpy")
        if record is not None:
            real_do = session.do

            def do(op: str, **args: Any) -> StepReport:
                record.append(op)
                return real_do(op, **args)

            session.do = do  # type: ignore[method-assign]
        return session

    return make


@pytest.fixture
def rig(qtbot: QtBot) -> Iterator[tuple[ComputeManager, SessionBridge]]:
    compute = ComputeManager()
    bridge = SessionBridge(compute, progress_interval=0.0)
    yield compute, bridge
    compute.wait_for_done(20_000)


@pytest.fixture
def note_reducer() -> Iterator[None]:
    """``set_note`` as a test-local neutral reducer (a manual, logged edit)."""
    from cuperiod.prewhiten.session.session import _load_ops

    _load_ops()
    saved = actions.reducer("set_note")

    def reducer(kernel: Any, state: Any, derived: Any, args: Any) -> Any:
        return state, None, StepReport(kind="edited", message=f"note {args.note!r}")

    actions.register("set_note")(reducer)
    yield
    actions.unregister("set_note")
    if saved is not None:
        actions.register("set_note")(saved)


def _idle(qtbot: QtBot, bridge: SessionBridge, compute: ComputeManager) -> None:
    qtbot.waitUntil(
        lambda: bridge.is_idle() and compute.session_jobs == 0, timeout=30_000
    )


def _views(bridge: SessionBridge) -> list[SessionView]:
    seen: list[SessionView] = []
    bridge.view_ready.connect(lambda key, view: seen.append(view))
    return seen


def _block(compute: ComputeManager) -> threading.Event:
    """Occupy the single worker until the returned gate is set."""
    gate = threading.Event()
    entered = threading.Event()

    def work(job: SessionJob) -> None:
        entered.set()
        gate.wait(20.0)

    compute.start_session_job(SessionJob(work, key="blocker"))
    assert entered.wait(10.0)
    return gate


def _open(
    qtbot: QtBot,
    rig: tuple[ComputeManager, SessionBridge],
    key: SessionKey = KEY,
    record: list[str] | None = None,
) -> None:
    compute, bridge = rig
    bridge.display(key)
    bridge.open(key, _factory(record), backend="numpy")
    _idle(qtbot, bridge, compute)


# -- ordering and never-dropped mutations ------------------------------------------
def test_open_emits_a_light_view_without_a_result(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    views = _views(bridge)
    _open(qtbot, rig)
    assert len(views) == 1
    view = views[0]
    assert view.key == KEY and view.light and view.result is None
    assert view.rows == () and view.id_epoch == bridge.id_epoch(KEY) == 1
    assert view.requested_backend == "numpy" and not bridge.is_dirty(KEY)


def test_fifo_order_per_session(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    record: list[str] = []
    _open(qtbot, rig, record=record)
    gate = _block(compute)
    for op in ("step", "step", "polish", "step", "renumber"):
        bridge.submit(KEY, op)
    bridge.undo(KEY)
    bridge.submit(KEY, "step")
    gate.set()
    _idle(qtbot, bridge, compute)
    assert record == ["step", "step", "polish", "step", "renumber", "step"]


def test_a_mutation_is_never_dropped(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    views = _views(bridge)
    _open(qtbot, rig)
    bridge.open(OTHER, _factory(), backend="numpy")
    _idle(qtbot, bridge, compute)
    gate = _block(compute)
    bridge.submit(KEY, "step")
    bridge.submit(KEY, "step")
    # Everything that supersedes one-shot work, and switching what is displayed.
    lc = LightCurve.from_arrays(*DATA)
    compute.submit(ResultKey("x", "GLS", "h", "cpu"), lc, "GLS", GLSSettings(), "cpu")
    compute.cancel_all()
    compute.cancel_running()
    bridge.display(OTHER)  # source switch
    bridge.display(None)  # analysis switch
    views.clear()
    gate.set()
    _idle(qtbot, bridge, compute)
    assert views == []  # nothing displayed meanwhile
    bridge.display(KEY)
    assert len(views) == 1 and len(views[0].rows) == 2  # both edits landed
    assert [h.op for h in views[0].history] == ["step", "step"]


def test_views_coalesce(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    _open(qtbot, rig)
    bridge.submit(KEY, "step")
    bridge.submit(KEY, "step")
    _idle(qtbot, bridge, compute)
    views = _views(bridge)
    builds = bridge.view_builds
    gate = _block(compute)
    for _ in range(5):  # five toggles of the last step
        bridge.undo(KEY)
        bridge.redo(KEY)
    bridge.undo(KEY)
    gate.set()
    _idle(qtbot, bridge, compute)
    assert bridge.view_builds == builds + 1
    assert len(views) == 1 and len(views[0].rows) == 1
    assert views[0].can_redo and views[0].cursor == 1


def test_light_result_only_when_active_frequencies_move(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    views = _views(bridge)
    _open(qtbot, rig)
    assert views[-1].result is None  # never for the bare opening
    bridge.submit(KEY, "step")
    _idle(qtbot, bridge, compute)
    assert views[-1].light and views[-1].result is not None
    assert len(views[-1].result.components) == 1
    bridge.submit(KEY, "renumber")  # nothing to renumber: no change, no view
    _idle(qtbot, bridge, compute)
    bridge.undo(KEY)
    _idle(qtbot, bridge, compute)
    assert views[-1].result is not None and not views[-1].result.components


# -- Auto, cancel, failures ----------------------------------------------------------
def test_auto_progress_views_are_light_and_completion_is_full(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    views = _views(bridge)
    progress: list[StepReport] = []
    bundles: list[SolutionBundle] = []
    busy: list[bool] = []
    bridge.progress.connect(lambda key, report: progress.append(report))
    bridge.auto_finished.connect(lambda key, bundle: bundles.append(bundle))
    bridge.busy.connect(busy.append)
    _open(qtbot, rig)
    views.clear()
    bridge.auto(KEY, PreWhitenSettings(), "numpy")
    assert bridge.is_busy
    _idle(qtbot, bridge, compute)
    assert busy == [True, False]
    assert len(progress) >= 3 and progress[0].kind == "accepted"
    incremental, final = views[:-1], views[-1]
    assert incremental and all(v.light and v.result is None for v in incremental)
    assert all(v.candidate is None for v in incremental)
    assert not final.light and final.result is not None and final.stop_reason
    (bundle,) = bundles
    assert bundle.result is final.result
    assert bundle.snapshot.head.serial == final.serial
    # Auto through the bridge is cup.prewhiten, bitwise.
    reference = cup.prewhiten(DATA, backend="numpy")
    got = [(c.frequency, c.amplitude, c.phase) for c in final.result.components]
    want = [(c.frequency, c.amplitude, c.phase) for c in reference.components]
    assert got == want and final.id_epoch == bridge.id_epoch(KEY)


def test_stop_cancels_auto_only_and_leaves_a_valid_session(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    resume = threading.Event()
    calls: list[int] = []

    def factory() -> cup.PreWhitenSession:
        session = cup.PreWhitenSession(DATA, backend="numpy")
        real_step = session.step

        def step(**kw: Any) -> StepReport:
            calls.append(1)
            report = real_step(**kw)
            if len(calls) == 1:
                resume.wait(20.0)  # hold Auto after its first step
            return report

        session.step = step  # type: ignore[method-assign]
        return session

    messages: list[tuple[str, str]] = []
    bundles: list[object] = []
    bridge.message.connect(lambda sev, text: messages.append((sev, text)))
    bridge.auto_finished.connect(lambda key, b: bundles.append(b))
    views = _views(bridge)
    bridge.display(KEY)
    bridge.open(KEY, factory, backend="numpy")
    bridge.auto(KEY, PreWhitenSettings(), "numpy")
    qtbot.waitUntil(lambda: len(calls) == 1)
    bridge.submit(KEY, "step")  # a queued edit behind the long job
    assert not bridge.cancel(OTHER)
    assert bridge.cancel(KEY)
    resume.set()
    _idle(qtbot, bridge, compute)
    assert bundles == []  # no completion
    assert any("stopped" in text for _, text in messages)
    final = views[-1]
    assert len(final.rows) == 2  # Auto's committed step + the queued edit
    assert [h.op for h in final.history] == ["step", "step"]
    assert final.history[0].group != final.history[1].group
    # The session is valid: Auto again runs to completion.
    bridge.auto(KEY, PreWhitenSettings(), "numpy")
    _idle(qtbot, bridge, compute)
    assert len(bundles) == 1 and views[-1].result is not None


def test_backend_change_on_a_stepped_session_is_refused(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    messages: list[str] = []
    bridge.message.connect(lambda sev, text: messages.append(text))
    _open(qtbot, rig)
    bridge.step(KEY, PreWhitenSettings(), "numpy")
    _idle(qtbot, bridge, compute)
    bridge.auto(KEY, PreWhitenSettings(), "finufft")
    _idle(qtbot, bridge, compute)
    assert any("backend choice applies to new sessions" in m for m in messages)
    assert len(bridge.latest_view(KEY).rows) == 1  # type: ignore[union-attr]


def test_step_on_an_empty_log_reopens_with_the_factory(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    views = _views(bridge)
    bridge.display(KEY)
    bridge.step(KEY, PreWhitenSettings(), "numpy", factory=_factory())
    _idle(qtbot, bridge, compute)
    assert len(views[-1].rows) == 1 and bridge.id_epoch(KEY) == 1


def test_failures_and_preview_failures(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    failed: list[tuple[object, str]] = []
    preview: list[tuple[object, str]] = []
    busy: list[bool] = []
    bridge.failed.connect(lambda key, text: failed.append((key, text)))
    bridge.preview_failed.connect(lambda key, text: preview.append((key, text)))
    bridge.busy.connect(busy.append)

    def broken() -> cup.PreWhitenSession:
        raise ValueError("not enough points")

    bridge.display(KEY)
    bridge.open(KEY, broken, preview=True)
    _idle(qtbot, bridge, compute)
    assert preview == [(KEY, "ValueError: not enough points")] and failed == []
    bridge.open(OTHER, broken)
    _idle(qtbot, bridge, compute)
    assert failed == [(OTHER, "ValueError: not enough points")]
    assert busy == []
    # A preview of a key that is no longer displayed is skipped.
    gate = _block(compute)
    record: list[str] = []
    bridge.open(SessionKey("late"), _factory(record), preview=True)
    bridge.display(KEY)
    gate.set()
    _idle(qtbot, bridge, compute)
    assert not bridge.latest_view(SessionKey("late"))


# -- queries -----------------------------------------------------------------------
def test_stale_variation_queries_are_skipped_and_dropped(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    asked: list[object] = []
    hold = threading.Event()

    def factory() -> cup.PreWhitenSession:
        session = cup.PreWhitenSession(DATA, backend="numpy")

        def variations(ids: Any = None, **kw: Any) -> Any:
            asked.append(ids)
            if ids == [99]:
                hold.wait(20.0)
            return ("result", ids)

        session.variations = variations  # type: ignore[method-assign]
        return session

    ready: list[tuple[str, object]] = []
    bridge.query_ready.connect(lambda kind, payload: ready.append((kind, payload)))
    bridge.display(KEY)
    bridge.open(KEY, factory)
    _idle(qtbot, bridge, compute)
    gate = _block(compute)
    bridge.query_variations(KEY, [1], "both", None)
    bridge.query_variations(KEY, [2], "both", None)  # supersedes [1] at the head
    gate.set()
    _idle(qtbot, bridge, compute)
    assert asked == [[2]] and ready == [("variations", ("result", [2]))]
    # Dropped on arrival: a newer query was asked while the old one ran.
    bridge.query_variations(KEY, [99], "both", None)
    qtbot.waitUntil(lambda: asked[-1] == [99])
    bridge.query_variations(KEY, [3], "both", None)
    hold.set()
    _idle(qtbot, bridge, compute)
    assert ready[1:] == [("variations", ("result", [3]))]


def test_compute_errors_gives_a_full_view(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    views = _views(bridge)
    ready: list[tuple[str, object]] = []
    bridge.query_ready.connect(lambda kind, payload: ready.append((kind, payload)))
    _open(qtbot, rig)
    bridge.submit(KEY, "step")
    _idle(qtbot, bridge, compute)
    assert np.isnan(views[-1].rows[0].frequency_error)
    bridge.compute_errors(KEY)
    _idle(qtbot, bridge, compute)
    (kind, result), = ready
    assert kind == "errors" and not views[-1].light and views[-1].result is result
    assert np.isfinite(result.components[0].frequency_error)  # type: ignore[attr-defined]


def test_entry_script_query(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    ready: list[tuple[str, object]] = []
    bridge.query_ready.connect(lambda kind, payload: ready.append((kind, payload)))
    _open(qtbot, rig)
    bridge.submit(KEY, "step")
    bridge.entry_script(KEY, 1)
    _idle(qtbot, bridge, compute)
    ((kind, (seq, text)),) = ready  # type: ignore[misc]
    assert kind == "entry_script" and seq == 1 and "step" in text


# -- id epochs ---------------------------------------------------------------------
def test_id_epoch_bumps_and_stale_ops_are_refused(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot, note_reducer: None
) -> None:
    compute, bridge = rig
    messages: list[str] = []
    bridge.message.connect(lambda sev, text: messages.append(text))
    _open(qtbot, rig)
    epoch = bridge.id_epoch(KEY)
    for op in ("step", "step", "step"):
        bridge.submit(KEY, op)
    _idle(qtbot, bridge, compute)
    assert bridge.id_epoch(KEY) == epoch  # adding components remaps nothing
    for move, expected in (
        (lambda: bridge.undo(KEY), epoch + 1),
        (lambda: bridge.redo(KEY), epoch + 2),
        (lambda: bridge.goto(KEY, 1), epoch + 3),
        (lambda: bridge.goto(KEY, 3), epoch + 4),
        (lambda: bridge.submit(KEY, "finalize"), epoch + 5),
    ):
        move()
        _idle(qtbot, bridge, compute)
        assert bridge.id_epoch(KEY) == expected
    assert bridge.latest_view(KEY).id_epoch == epoch + 5  # type: ignore[union-attr]
    # A note sent against the old numbering is refused; the current one lands.
    bridge.submit(KEY, "set_note", {"id": 1, "note": "x"}, expect_id_epoch=epoch)
    _idle(qtbot, bridge, compute)
    assert messages[-1] == STALE_IDS_MESSAGE
    assert bridge.latest_view(KEY).history[-1].op != "set_note"  # type: ignore[union-attr]
    bridge.submit(KEY, "set_note", {"id": 1, "note": "x"}, expect_id_epoch=epoch + 5)
    _idle(qtbot, bridge, compute)
    assert bridge.latest_view(KEY).history[-1].op == "set_note"  # type: ignore[union-attr]
    # Nothing to undo is a message, not an epoch bump.
    bridge.redo(KEY)
    _idle(qtbot, bridge, compute)
    assert messages[-1] == "Nothing to redo." and bridge.id_epoch(KEY) == epoch + 5


def test_renumber_and_prune_bump_only_when_committed(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    record: list[str] = []
    _open(qtbot, rig, record=record)
    bridge.submit(KEY, "step")
    bridge.submit(KEY, "step")
    _idle(qtbot, bridge, compute)
    before = bridge.id_epoch(KEY)
    entries = len(bridge.latest_view(KEY).history)  # type: ignore[union-attr]
    bridge.submit(KEY, "prune")
    bridge.submit(KEY, "renumber")
    _idle(qtbot, bridge, compute)
    after = len(bridge.latest_view(KEY).history)  # type: ignore[union-attr]
    assert bridge.id_epoch(KEY) == before + (after - entries)


# -- dirty set, LRU, persistence ---------------------------------------------------
def test_dirty_sessions_are_never_evicted(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot, note_reducer: None
) -> None:
    compute, _ = rig
    bridge = SessionBridge(compute, max_clean=1, progress_interval=0.0)
    evicted: list[object] = []
    bridge.evicted.connect(evicted.append)
    keys = [SessionKey(f"s{i}") for i in range(4)]
    for k in keys:
        bridge.display(k)
        bridge.open(k, _factory(), backend="numpy")
        bridge.submit(k, "step")
    bridge.submit(keys[0], "set_note", {"id": 1, "note": "mine"})  # dirty
    _idle(qtbot, bridge, compute)
    assert bridge.dirty_keys() == (keys[0],) and bridge.is_dirty(keys[0])
    bridge.display(None)
    live = bridge.keys()
    assert keys[0] in live  # dirty: kept
    clean = [k for k in live if k != keys[0]]
    assert len(clean) == 1 and set(evicted) == set(keys[1:]) - set(clean)
    # Undo back to a pristine state clears the dirty flag.
    bridge.undo(keys[0])
    _idle(qtbot, bridge, compute)
    assert not bridge.is_dirty(keys[0])


def test_save_clears_dirty_and_sets_the_project_path(
    rig: tuple[ComputeManager, SessionBridge],
    qtbot: QtBot,
    note_reducer: None,
    tmp_path: Any,
) -> None:
    compute, bridge = rig
    views = _views(bridge)
    _open(qtbot, rig)
    bridge.submit(KEY, "step")
    bridge.submit(KEY, "set_note", {"id": 1, "note": "saved"})
    _idle(qtbot, bridge, compute)
    assert bridge.is_dirty(KEY) and views[-1].dirty
    path = str(tmp_path / "star.cupw.json")
    bridge.save(KEY, path)
    _idle(qtbot, bridge, compute)
    assert not bridge.is_dirty(KEY)
    assert views[-1].project_path == path and not views[-1].dirty


def test_restore_from_a_bundle_is_exact(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    bundles: list[SolutionBundle] = []
    bridge.auto_finished.connect(lambda key, b: bundles.append(b))
    _open(qtbot, rig)
    bridge.auto(KEY, PreWhitenSettings(), "numpy")
    _idle(qtbot, bridge, compute)
    (bundle,) = bundles
    bridge.discard(KEY)
    assert not bridge.has_session(KEY)
    views = _views(bridge)
    bridge.restore(KEY, bundle.snapshot, DATA)
    _idle(qtbot, bridge, compute)
    rows = views[-1].rows
    assert [r.id for r in rows] == [int(c.label[1:]) for c in bundle.result.components]
    for r, c in zip(rows, bundle.result.components, strict=True):
        assert (r.frequency, r.amplitude) == (c.frequency, c.amplitude)
        assert r.phase == c.phase
    assert views[-1].can_undo  # the undo states came back with the snapshot


def test_ops_without_a_session_are_reported(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    _, bridge = rig
    messages: list[str] = []
    bridge.message.connect(lambda sev, text: messages.append(text))
    bridge.submit(KEY, "step")
    bridge.undo(KEY)
    bridge.compute_errors(KEY)
    bridge.step(KEY, PreWhitenSettings(), "numpy")
    assert len(messages) == 4 and not bridge.has_session(KEY)


def test_slow_quick_job_shows_the_non_blocking_indicator(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig

    def factory() -> cup.PreWhitenSession:
        session = cup.PreWhitenSession(DATA, backend="numpy")
        real_undo = session.undo

        def undo() -> bool:
            threading.Event().wait(0.6)  # longer than QUICK_BUSY_MS
            return real_undo()

        session.undo = undo  # type: ignore[method-assign]
        return session

    quick: list[bool] = []
    busy: list[bool] = []
    bridge.quick_busy.connect(quick.append)
    bridge.busy.connect(busy.append)
    bridge.display(KEY)
    bridge.open(KEY, factory)
    bridge.submit(KEY, "step")
    _idle(qtbot, bridge, compute)
    busy.clear()
    quick.clear()
    bridge.undo(KEY)  # quick class: no busy state, the indicator after 300 ms
    _idle(qtbot, bridge, compute)
    assert quick == [True, False] and busy == []
