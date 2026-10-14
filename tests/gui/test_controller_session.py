"""The controller's interactive pre-whitening (design 4.2, stage G8a).

Sessions run on ``backend="numpy"`` (``"finufft"`` where a test says so); the lazy
preview resolves to a CPU backend, so nothing here creates a CUDA context.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

import cuperiod as cup
from cuperiod.core.config import GLSSettings, PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.gui.dialogs import FakeDialogProvider
from cuperiod.gui.models import SourceItem
from cuperiod.gui.state import AppController
from cuperiod.gui.viewmodels import SessionKey, SessionView
from cuperiod.prewhiten.result import PreWhitenResult
from cuperiod.prewhiten.subsets import Partition

SETTINGS = PreWhitenSettings(backend="numpy", max_frequencies=3, samples_per_peak=5)


def _curve(n: int = 300, seed: int = 1, span: float = 40.0) -> LightCurve:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n))
    y = (
        0.3 * np.sin(2 * np.pi * 1.3 * t)
        + 0.12 * np.sin(2 * np.pi * 2.71 * t + 1.0)
        + 0.05 * np.sin(2 * np.pi * 0.61 * t + 2.0)
        + rng.normal(0.0, 0.01, n)
    )
    return LightCurve.from_arrays(t, y, np.full(n, 0.01))


@pytest.fixture
def ctl(qtbot: QtBot) -> Iterator[AppController]:
    controller = AppController(dialogs=FakeDialogProvider())
    controller.set_analysis("prewhiten")
    yield controller
    controller.shutdown()
    controller._compute.wait_for_done(30_000)


def _idle(qtbot: QtBot, ctl: AppController) -> None:
    """Every session job, view builds included, has reported back."""
    bridge = ctl.session_bridge
    qtbot.waitUntil(
        lambda: bridge.is_idle() and ctl._compute.session_jobs == 0, timeout=60_000
    )
    qtbot.wait(20)


def _auto(qtbot: QtBot, ctl: AppController, backend: str = "numpy") -> PreWhitenResult:
    with qtbot.waitSignal(ctl.solution_ready, timeout=60_000) as blocker:
        ctl.run_prewhiten(backend, SETTINGS)
    _idle(qtbot, ctl)
    (result,) = blocker.args
    assert isinstance(result, PreWhitenResult)
    return result


def _view(ctl: AppController) -> SessionView:
    view = ctl.state.session_view
    assert view is not None
    return view


def _row_numbers(view: SessionView) -> list[tuple[int, float, float, float]]:
    return [(r.id, r.frequency, r.amplitude, r.phase) for r in view.rows if r.active]


def _result_numbers(result: PreWhitenResult) -> list[tuple[int, float, float, float]]:
    return [
        (int(c.label.lstrip("F")), c.frequency, c.amplitude, c.phase)
        for c in result.components
    ]


# -- Auto ≡ cup.prewhiten, and the three cases of run_prewhiten ----------------------
def test_auto_through_the_controller_equals_cup_prewhiten(
    ctl: AppController, qtbot: QtBot
) -> None:
    lc = _curve()
    ctl.set_light_curve(lc, "star")
    result = _auto(qtbot, ctl)
    view = _view(ctl)
    assert view.settings is not None and not view.light
    reference = cup.prewhiten(lc, settings=view.settings, backend="numpy")
    assert _result_numbers(result) == _result_numbers(reference)
    assert ctl.state.current_solution is result
    assert not ctl.state.last_from_cache
    # The full view's rows are the solution, and the first component is selected.
    assert _row_numbers(view) == _result_numbers(result)
    assert ctl.state.selected_components == (view.rows[0].id,)
    assert ctl.state.selected_period == pytest.approx(result.components[0].period)


def test_a_repeat_auto_is_a_cache_hit_on_the_same_session(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(), "star")
    first = _auto(qtbot, ctl)
    serial = _view(ctl).serial
    builds = ctl.session_bridge.view_builds
    second = _auto(qtbot, ctl)
    assert second is first
    assert ctl.state.last_from_cache and ctl.state.last_compute_ms == 0.0
    # The live session was the cached one: shown again, not restored or re-run.
    assert _view(ctl).serial == serial
    assert ctl.session_bridge.view_builds == builds


@pytest.mark.parametrize("backend", ["numpy", "finufft"])
def test_case_2_after_eviction_restores_the_exact_session(
    ctl: AppController, qtbot: QtBot, backend: str
) -> None:
    star, other = _curve(seed=1), _curve(seed=2)
    ctl.set_light_curve(star, "star")
    result = _auto(qtbot, ctl, backend)
    key = ctl.displayed_session
    assert key is not None
    ctl.set_light_curve(other, "other")
    ctl.session_bridge.discard(key)  # evicted from the clean LRU
    assert not ctl.session_bridge.has_session(key)
    ctl.set_light_curve(star, "star")
    with qtbot.waitSignal(ctl.solution_ready, timeout=5_000) as blocker:
        ctl.run_prewhiten(backend, SETTINGS)
    assert blocker.args[0] is result and ctl.state.last_from_cache
    _idle(qtbot, ctl)
    view = _view(ctl)
    # Restored from the snapshot, bitwise: no re-run that could re-map the F-ids.
    assert _row_numbers(view) == _result_numbers(result)
    assert view.can_undo  # the undo states came back too
    # The first edit addresses the same component.
    first = view.rows[0]
    ctl.session_do(
        "set_note", id=first.id, note="checked", expect_id_epoch=view.id_epoch
    )
    _idle(qtbot, ctl)
    edited = _view(ctl).row(first.id)
    assert edited is not None and edited.note == "checked"
    assert edited.frequency == first.frequency


def test_a_finished_auto_is_cached_even_when_superseded(
    ctl: AppController, qtbot: QtBot
) -> None:
    star, other = _curve(seed=1), _curve(seed=2)
    applied: list[object] = []
    ctl.solution_ready.connect(applied.append)
    ctl.set_light_curve(star, "star")
    ctl.run_prewhiten("numpy", SETTINGS)
    ctl.set_light_curve(other, "other")  # superseded while it runs
    _idle(qtbot, ctl)
    assert applied == []  # never lands on the other curve
    assert ctl.state.current_solution is None
    ctl.set_light_curve(star, "star")
    with qtbot.waitSignal(ctl.solution_ready, timeout=5_000):
        ctl.run_prewhiten("numpy", SETTINGS)
    assert ctl.state.last_from_cache


def test_case_1_continues_an_edited_session_and_refuses_a_backend_change(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(), "star")
    _auto(qtbot, ctl)
    view = _view(ctl)
    epoch = view.id_epoch
    ctl.session_do("set_note", id=view.rows[0].id, note="mine", expect_id_epoch=epoch)
    _idle(qtbot, ctl)
    assert not _view(ctl).pristine
    messages: list[tuple[str, str]] = []
    ctl.session_message.connect(lambda s, t: messages.append((s, t)))
    ctl.run_prewhiten("finufft", SETTINGS)
    _idle(qtbot, ctl)
    assert any("backend choice applies to new sessions" in t for _, t in messages)
    # Same backend: continued (the note is still in the log), never cached.
    ctl.run_prewhiten("numpy", SETTINGS)
    _idle(qtbot, ctl)
    after = _view(ctl)
    assert any(h.op == "set_note" for h in after.history)
    assert after.rows[0].note == "mine"


def test_case_3_restarts_a_pristine_session_whose_settings_changed(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(), "star")
    ctl.session_step("numpy", SETTINGS)
    ctl.session_step("numpy", SETTINGS)
    _idle(qtbot, ctl)
    assert sum(h.op == "step" for h in _view(ctl).history) == 2
    changed = SETTINGS.model_copy(update={"max_frequencies": 2})
    with qtbot.waitSignal(ctl.solution_ready, timeout=60_000) as blocker:
        ctl.run_prewhiten("numpy", changed)
    _idle(qtbot, ctl)
    view = _view(ctl)
    assert view.settings is not None and view.settings.max_frequencies == 2
    assert all(h.op != "configure" for h in view.history)  # a new session
    assert blocker.args[0].n_components <= 2


def test_step_on_a_stepped_session_configures_instead_of_restarting(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(), "star")
    ctl.session_step("numpy", SETTINGS)
    _idle(qtbot, ctl)
    changed = SETTINGS.model_copy(update={"snr_threshold": 3.5})
    ctl.session_step("numpy", changed)
    _idle(qtbot, ctl)
    ops = [h.op for h in _view(ctl).history]
    assert ops[0] == "step" and "configure" in ops and ops[-1] == "step"
    view = _view(ctl)
    assert view.settings is not None and view.settings.snr_threshold == 3.5


def test_a_stacked_multiband_session_has_a_band_partition(
    ctl: AppController, qtbot: QtBot
) -> None:
    a, b = _curve(seed=1), _curve(seed=2)
    mblc = MultiBandLightCurve.from_light_curves(
        {
            "g": LightCurve.from_arrays(a.time, a.value + 15.0, a.error),
            "r": LightCurve.from_arrays(b.time, b.value + 14.2, b.error),
        }
    )
    ctl.set_light_curve(mblc, "mb")
    with qtbot.waitSignal(ctl.solution_ready, timeout=60_000) as blocker:
        ctl.run_prewhiten("numpy", SETTINGS, band="stacked")
    _idle(qtbot, ctl)
    assert blocker.args[0].components[0].frequency == pytest.approx(1.3, abs=0.01)
    view = _view(ctl)
    assert "band" in view.partitions and "band" in view.label_columns


# -- solution_ready, selection and the pick cursor (4.2, 4.3) --------------------------
def test_solution_ready_only_when_the_active_frequencies_change(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(), "star")
    _auto(qtbot, ctl)
    view = _view(ctl)
    solutions: list[object] = []
    changes: list[SessionView] = []
    ctl.solution_ready.connect(solutions.append)
    ctl.session_changed.connect(changes.append)
    first = view.rows[0]
    ctl.session_do("set_note", id=first.id, note="n", expect_id_epoch=view.id_epoch)
    ctl.session_do("fix", ids=[first.id], amplitude=True, expect_id_epoch=view.id_epoch)
    _idle(qtbot, ctl)
    assert changes and solutions == []
    view = _view(ctl)
    epoch = view.id_epoch
    ctl.session_do("set_active", ids=[first.id], active=False, expect_id_epoch=epoch)
    _idle(qtbot, ctl)
    assert len(solutions) == 1 and _view(ctl).result is solutions[0]


def test_selection_and_cursor_follow_step_add_remove_and_undo(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(), "star")
    cursors: list[float] = []
    ctl.cursor_changed.connect(cursors.append)
    # A fresh, displayed session: nothing selected, the cursor hidden.
    ctl.session_step("numpy", SETTINGS)
    _idle(qtbot, ctl)
    view = _view(ctl)
    # Step accepted: the new component is selected, the cursor on the next candidate.
    assert ctl.state.selected_components == (view.rows[-1].id,)
    assert view.candidate is not None
    assert ctl.state.cursor_frequency == view.candidate.frequency
    assert cursors[-1] == view.candidate.frequency
    assert ctl.state.selected_period == pytest.approx(view.rows[-1].period)
    # Add: the added component.
    ctl.session_do("add", frequency=2.71)
    _idle(qtbot, ctl)
    view = _view(ctl)
    added = max(r.id for r in view.rows)
    assert ctl.state.selected_components == (added,)
    # Remove the selected one: the first remaining active component; cursor kept.
    cursor = ctl.state.cursor_frequency
    ctl.session_do("remove", ids=[added], expect_id_epoch=view.id_epoch)
    _idle(qtbot, ctl)
    view = _view(ctl)
    assert ctl.state.selected_components == (view.rows[0].id,)
    assert ctl.state.cursor_frequency == cursor
    # Undo brings it back; the selection stays on the same, unchanged component.
    selected = ctl.state.selected_components
    ctl.session_undo()
    _idle(qtbot, ctl)
    assert ctl.state.selected_components == selected
    assert len(_view(ctl).rows) == 2


def test_auto_completion_selects_the_first_component_and_moves_the_cursor(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(), "star")
    _auto(qtbot, ctl)
    view = _view(ctl)
    assert ctl.state.selected_components == (view.rows[0].id,)
    if view.candidate is not None:
        assert ctl.state.cursor_frequency == view.candidate.frequency


def test_select_components_moves_the_fold_not_the_cursor(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(), "star")
    _auto(qtbot, ctl)
    view = _view(ctl)
    second = view.rows[1]
    cursor = ctl.state.cursor_frequency
    picked: list[object] = []
    ctl.components_selected.connect(picked.append)
    with qtbot.waitSignal(ctl.fold_changed, timeout=2_000):
        ctl.select_components([second.id, 999])
    assert picked == [(second.id,)]
    assert ctl.state.selected_period == second.period
    assert ctl.state.cursor_frequency == cursor
    ctl.set_cursor(4.2)
    assert ctl.state.cursor_frequency == 4.2


def test_a_stale_id_addressed_op_is_refused(ctl: AppController, qtbot: QtBot) -> None:
    ctl.set_light_curve(_curve(), "star")
    _auto(qtbot, ctl)
    stale = _view(ctl).id_epoch
    ctl.session_undo()
    _idle(qtbot, ctl)
    messages: list[str] = []
    ctl.session_message.connect(lambda _s, t: messages.append(t))
    ctl.session_do("set_note", id=1, note="x", expect_id_epoch=stale)
    _idle(qtbot, ctl)
    assert any("component list changed" in m for m in messages)


# -- the worker-side edits -------------------------------------------------------------
def test_point_edits_partitions_and_exact_adds(
    ctl: AppController, qtbot: QtBot
) -> None:
    lc = _curve()
    ctl.set_light_curve(lc, "star")
    ctl.session_step("numpy", SETTINGS)
    _idle(qtbot, ctl)
    ctl.delete_points([0, 1, 2])
    _idle(qtbot, ctl)
    view = _view(ctl)
    assert view.deleted[:3].all() and not view.deleted[3:].any()
    ctl.restore_points(None)
    _idle(qtbot, ctl)
    assert not _view(ctl).deleted.any()

    class Halves:
        def resolve(self, time: Any, *, partitions: Any, labels: Any = None) -> Any:
            codes = np.where(np.asarray(time) < np.median(time), "early", "late")
            return Partition.from_labels(codes, name="half")

    ctl.define_partition(Halves())
    _idle(qtbot, ctl)
    view = _view(ctl)
    assert "half" in view.partitions and "half" in view.partition_codes
    subsets = [s for s in view.subsets if s.partition == "half"]
    assert [s.label for s in subsets] == ["early", "late"]
    assert all(not s.excluded and s.n_deleted == 0 for s in subsets)

    ctl.add_frequency(2.7, exact=True)
    _idle(qtbot, ctl)
    view = _view(ctl)
    assert any(r.frequency == 2.7 for r in view.rows)
    assert view.next_id == max(r.id for r in view.rows) + 1


def test_cancel_stops_auto_and_keeps_a_valid_session(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(), "star")
    statuses: list[str] = []
    ctl.status_message.connect(statuses.append)
    busy: list[bool] = []
    ctl.busy_changed.connect(busy.append)
    ctl.run_prewhiten("numpy", SETTINGS)
    ctl.cancel_compute()  # the job's own token: it stops before its first step
    _idle(qtbot, ctl)
    assert any("Stopping after the current step" in s for s in statuses)
    assert busy[-1] is False
    view = _view(ctl)
    assert ctl.state.current_solution is None  # nothing finished, nothing cached
    ctl.session_step("numpy", SETTINGS)  # the session is still usable
    _idle(qtbot, ctl)
    assert len(_view(ctl).rows) == len(view.rows) + 1


# -- lazy open (D22) -------------------------------------------------------------------
def test_lazy_open_waits_for_the_debounce_and_sets_no_busy(
    ctl: AppController, qtbot: QtBot
) -> None:
    busy: list[bool] = []
    ctl.set_light_curve(_curve(), "star")
    ctl.busy_changed.connect(busy.append)
    ctl.schedule_preview("auto", SETTINGS)
    key = ctl.displayed_session
    assert key is not None and not ctl.session_bridge.is_open(key)
    with qtbot.waitSignal(ctl.session_changed, timeout=10_000) as blocker:
        pass
    view = blocker.args[0]
    assert isinstance(view, SessionView) and view.rows == () and view.result is None
    assert view.backend in ("finufft", "numpy")  # a CPU preview
    assert True not in busy


def test_lazy_open_never_runs_in_batch_mode(ctl: AppController, qtbot: QtBot) -> None:
    ctl.schedule_preview("numpy", SETTINGS)
    ctl.load_sources([SourceItem("a", "a", _curve), SourceItem("b", "b", _curve)])
    ctl.select_source(0)
    qtbot.wait(700)
    key = ctl.displayed_session
    assert key is not None and not ctl.session_bridge.is_open(key)
    assert ctl._compute.session_jobs == 0


def test_a_failing_preview_shows_placeholder_text_not_a_failure(
    ctl: AppController, qtbot: QtBot
) -> None:
    flat = LightCurve.from_arrays(np.full(40, 5.0), np.ones(40), np.full(40, 0.01))
    failures: list[str] = []
    ctl.compute_failed.connect(failures.append)
    ctl.set_light_curve(flat, "flat")
    with qtbot.waitSignal(ctl.preview_failed, timeout=10_000) as blocker:
        ctl.schedule_preview("numpy", SETTINGS)
    assert blocker.args[0].startswith("Spectrum unavailable:")
    _idle(qtbot, ctl)
    assert failures == []


def test_an_empty_batch_source_takes_the_failure_path(
    ctl: AppController, qtbot: QtBot
) -> None:
    empty = SourceItem(
        "empty", "empty.csv", lambda: LightCurve.from_arrays([], [], [])
    )
    ctl.load_sources([SourceItem("ok", "ok", _curve), empty])
    ctl.select_source(0)
    failed: list[int] = []
    messages: list[str] = []
    ctl.source_failed.connect(failed.append)
    ctl.compute_failed.connect(messages.append)
    ctl.select_source(1)
    assert failed == [1] and ctl.state.current_source_index == 0
    assert messages == ["empty.csv contains no data rows."]


def test_switching_to_periodogram_hides_the_session(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(), "star")
    _auto(qtbot, ctl)
    key = ctl.displayed_session
    ctl.set_analysis("periodogram")
    assert ctl.displayed_session is None and ctl.state.session_view is None
    with qtbot.waitSignal(ctl.periodogram_ready, timeout=60_000):
        ctl.run("GLS", "cpu", GLSSettings(), 5)
    with qtbot.waitSignal(ctl.session_changed, timeout=2_000):
        ctl.set_analysis("prewhiten")  # reattached: its kept view comes back
    assert ctl.displayed_session == key
    assert isinstance(key, SessionKey) and ctl.state.selected_components


def test_run_command_covers_the_widget_free_commands(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(), "star")
    _auto(qtbot, ctl)
    assert ctl.run_command("edit.undo")
    _idle(qtbot, ctl)
    assert _view(ctl).can_redo
    assert ctl.run_command("edit.redo")
    _idle(qtbot, ctl)
    assert not ctl.run_command("view.focus_spectrum")
    cursor = ctl.state.cursor_frequency
    assert cursor is None or math.isfinite(cursor)
