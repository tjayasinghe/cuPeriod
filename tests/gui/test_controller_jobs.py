"""The controller's dirty flow, projects, prompts and Stop (design 4.2 / 4.8, G8a).

Every prompt goes through a scripted :class:`FakeDialogProvider`; sessions run on
``backend="numpy"``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

import cuperiod
from cuperiod.core.config import GLSSettings, PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve
from cuperiod.gui.dialogs import DiscardChoice, FakeDialogProvider
from cuperiod.gui.models import SourceItem
from cuperiod.gui.state import PROJECT_PREFIX, AppController
from cuperiod.gui.viewmodels import SessionView
from cuperiod.prewhiten.session.session import PreWhitenSession
from cuperiod.prewhiten.session.state import ProjectError

SETTINGS = PreWhitenSettings(backend="numpy", max_frequencies=2, samples_per_peak=5)


def _curve(seed: int = 1, n: int = 300) -> LightCurve:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 40.0, n))
    y = (
        0.3 * np.sin(2 * np.pi * 1.3 * t)
        + 0.12 * np.sin(2 * np.pi * 2.71 * t + 1.0)
        + rng.normal(0.0, 0.01, n)
    )
    return LightCurve.from_arrays(t, y, np.full(n, 0.01))


class _Deferred(FakeDialogProvider):
    """Keeps the discard prompt open: the test answers it later."""

    def __init__(self, answers: Sequence[Any] = ()) -> None:
        super().__init__(answers)
        self.pending: list[Callable[[DiscardChoice], None]] = []

    def confirm_discard(
        self, names: Sequence[str], done: Callable[[DiscardChoice], None]
    ) -> None:
        self.calls.append(("confirm_discard", (tuple(names),)))
        self.pending.append(done)


@pytest.fixture
def ctl(qtbot: QtBot) -> Iterator[AppController]:
    controller = AppController(dialogs=FakeDialogProvider())
    controller.set_analysis("prewhiten")
    yield controller
    controller.shutdown()
    controller._compute.wait_for_done(30_000)


def _idle(qtbot: QtBot, ctl: AppController) -> None:
    bridge = ctl.session_bridge
    qtbot.waitUntil(
        lambda: bridge.is_idle() and ctl._compute.session_jobs == 0, timeout=60_000
    )
    qtbot.wait(20)


def _view(ctl: AppController) -> SessionView:
    view = ctl.state.session_view
    assert view is not None
    return view


def _stepped_and_noted(qtbot: QtBot, ctl: AppController, note: str = "x") -> None:
    """A dirty session: one step, then a note (a logged manual edit)."""
    ctl.session_step("numpy", SETTINGS)
    _idle(qtbot, ctl)
    view = _view(ctl)
    ctl.session_do(
        "set_note", id=view.rows[0].id, note=note, expect_id_epoch=view.id_epoch
    )
    _idle(qtbot, ctl)


def _provider(ctl: AppController, answers: Sequence[Any] = ()) -> FakeDialogProvider:
    provider = FakeDialogProvider(answers)
    ctl.set_dialog_provider(provider)
    return provider


# -- dirty definition and the title-bar mark -------------------------------------------
def test_dirty_follows_the_saved_serial(
    ctl: AppController, qtbot: QtBot, tmp_path: Path
) -> None:
    marks: list[tuple[str, bool]] = []
    ctl.project_changed.connect(lambda p, d: marks.append((p, d)))
    ctl.set_light_curve(_curve(), "star")
    ctl.session_step("numpy", SETTINGS)
    _idle(qtbot, ctl)
    assert not ctl.is_session_dirty()  # pristine work is never "unsaved"
    _stepped_and_noted(qtbot, ctl)
    assert ctl.is_session_dirty() and marks[-1] == ("", True)
    target = tmp_path / "star"
    ctl.save_project(str(target))  # the suffix is added
    _idle(qtbot, ctl)
    saved = str(target) + ".cupw.json"
    assert Path(saved).is_file()
    assert marks[-1] == (saved, False) and ctl.state.project_path == saved
    view = _view(ctl)
    ctl.session_do(
        "set_note", id=view.rows[0].id, note="later", expect_id_epoch=view.id_epoch
    )
    _idle(qtbot, ctl)
    assert marks[-1] == (saved, True)
    ctl.session_undo()  # back to the saved serial: clean again
    _idle(qtbot, ctl)
    assert marks[-1] == (saved, False) and not ctl.is_session_dirty()


def test_save_without_a_path_asks_for_one(
    ctl: AppController, qtbot: QtBot, tmp_path: Path
) -> None:
    target = tmp_path / "picked.cupw.json"
    provider = _provider(ctl, [str(target)])
    ctl.set_light_curve(_curve(), "star")
    _stepped_and_noted(qtbot, ctl)
    ctl.save_project()
    _idle(qtbot, ctl)
    assert provider.calls[0][0] == "save_file"
    assert provider.calls[0][1][2] == "star.cupw.json"  # suggested name
    assert target.is_file() and not ctl.is_session_dirty()


# -- dirty sessions survive switches ---------------------------------------------------
def test_switching_source_keeps_a_dirty_session_and_reattaches_it(
    ctl: AppController, qtbot: QtBot
) -> None:
    star, other = _curve(1), _curve(2)
    ctl.set_light_curve(star, "star")
    _stepped_and_noted(qtbot, ctl, "keep me")
    key = ctl.displayed_session
    assert key is not None
    ctl.set_light_curve(other, "other")
    assert ctl.displayed_session != key
    assert key in ctl.session_bridge.dirty_keys()
    assert [name for _, name in ctl.dirty_sessions()] == ["star"]
    with qtbot.waitSignal(ctl.session_changed, timeout=5_000) as blocker:
        ctl.set_light_curve(star, "star")  # reattached, view and all
    view = blocker.args[0]
    assert view.rows[0].note == "keep me"
    assert ctl.displayed_session == key
    assert ctl.state.selected_components == (view.rows[0].id,)


def test_a_queued_edit_lands_although_the_source_switched(
    ctl: AppController, qtbot: QtBot
) -> None:
    star, other = _curve(1), _curve(2)
    ctl.set_light_curve(star, "star")
    ctl.session_step("numpy", SETTINGS)
    _idle(qtbot, ctl)
    view = _view(ctl)
    ctl.session_do(
        "set_note", id=view.rows[0].id, note="queued", expect_id_epoch=view.id_epoch
    )
    ctl.set_light_curve(other, "other")  # before the edit has run
    _idle(qtbot, ctl)
    with qtbot.waitSignal(ctl.session_changed, timeout=5_000) as blocker:
        ctl.set_light_curve(star, "star")
    assert blocker.args[0].rows[0].note == "queued"


# -- prompts over dirty sessions -------------------------------------------------------
def test_open_light_curve_over_dirty_sessions_asks(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(1), "star")
    _stepped_and_noted(qtbot, ctl)
    provider = _provider(ctl, ["cancel"])
    opened: list[str] = []
    ran = ctl.confirm_then(lambda: opened.append("b"))
    assert not ran and opened == []  # cancelled: nothing discarded, nothing opened
    assert provider.calls == [("confirm_discard", (("star",),))]
    provider.answers.append("discard")
    assert ctl.confirm_then(lambda: opened.append("b"))
    assert opened == ["b"] and not ctl.session_bridge.dirty_keys()


def test_prompts_name_a_source_by_its_label_not_its_key(
    ctl: AppController, qtbot: QtBot
) -> None:
    label = "ASAS-SN V0521 Her — RRab (P=0.567 d)"
    ctl.set_light_curve(_curve(1), "asassn_446677590359", label=label)
    _stepped_and_noted(qtbot, ctl)
    assert [name for _, name in ctl.dirty_sessions()] == [label]
    provider = _provider(ctl, ["cancel"])
    assert not ctl.confirm_then(lambda: None)
    assert provider.calls == [("confirm_discard", ((label,),))]
    # The label is remembered for the source after another curve is shown, and the
    # suggested project file keeps the safe source id.
    ctl.set_light_curve(_curve(2), "other")
    assert [name for _, name in ctl.dirty_sessions()] == [label]
    key = ctl.session_bridge.dirty_keys()[0]
    assert ctl._suggested_project(key) == "asassn_446677590359.cupw.json"


def test_a_batch_source_is_named_by_its_label(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.load_sources([SourceItem("k1", "Star one", lambda: _curve(1))])
    ctl.select_source(0)
    key = ctl._session_key("")
    assert ctl.session_name(key) == "Star one"


def test_save_in_the_prompt_saves_then_proceeds(
    ctl: AppController, qtbot: QtBot, tmp_path: Path
) -> None:
    ctl.set_light_curve(_curve(1), "star")
    _stepped_and_noted(qtbot, ctl)
    target = tmp_path / "star.cupw.json"
    _provider(ctl, ["save", str(target)])
    opened: list[str] = []
    ran = ctl.confirm_then(lambda: opened.append("b"))
    assert not ran  # waits for the save job
    _idle(qtbot, ctl)
    assert opened == ["b"] and target.is_file()
    assert not ctl.session_bridge.dirty_keys()


def test_new_session_over_a_dirty_session_asks(
    ctl: AppController, qtbot: QtBot
) -> None:
    ctl.set_light_curve(_curve(1), "star")
    _stepped_and_noted(qtbot, ctl)
    provider = _provider(ctl, ["cancel"])
    ctl.new_session()
    _idle(qtbot, ctl)
    assert provider.calls[0][0] == "confirm_discard"
    assert _view(ctl).rows  # kept
    provider.answers.append("discard")
    ctl.new_session(backend="numpy", settings=SETTINGS)
    _idle(qtbot, ctl)
    view = _view(ctl)
    assert view.rows == () and view.history == () and not view.dirty


def test_close_asks_and_approves_once_resolved(
    ctl: AppController, qtbot: QtBot
) -> None:
    assert ctl.confirm_close()  # nothing unsaved
    ctl.set_light_curve(_curve(1), "star")
    _stepped_and_noted(qtbot, ctl)
    provider = _Deferred()
    ctl.set_dialog_provider(provider)
    approved: list[bool] = []
    ctl.close_approved.connect(lambda: approved.append(True))
    assert not ctl.confirm_close()
    assert provider.calls[0][0] == "confirm_discard" and approved == []
    provider.pending.pop()("cancel")
    assert approved == [] and not ctl.confirm_close()
    provider.pending.pop()("discard")
    assert approved == [True] and ctl.confirm_close()


# -- projects --------------------------------------------------------------------------
def _saved_project(qtbot: QtBot, ctl: AppController, path: Path) -> SessionView:
    ctl.set_light_curve(_curve(1), "star")
    _stepped_and_noted(qtbot, ctl, "from the file")
    ctl.save_project(str(path))
    _idle(qtbot, ctl)
    return _view(ctl)


def test_open_project_becomes_the_current_source(
    ctl: AppController, qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "p.cupw.json"
    saved = _saved_project(qtbot, ctl, path)
    other = AppController(dialogs=FakeDialogProvider())
    try:
        other.load_sources([SourceItem("a", "a", lambda: _curve(3))])
        restored: list[object] = []
        other.settings_restored.connect(restored.append)
        with qtbot.waitSignal(other.session_changed, timeout=30_000) as blocker:
            other.open_project(path)
        _idle(qtbot, other)
        assert other.state.mode == "single" and other.state.analysis == "prewhiten"
        assert other.state.source_id == PROJECT_PREFIX + str(path)
        assert other.state.project_path == str(path)
        view = blocker.args[0]
        assert [(r.id, r.frequency, r.note) for r in view.rows] == [
            (r.id, r.frequency, r.note) for r in saved.rows
        ]
        assert restored and isinstance(restored[0], PreWhitenSettings)
        assert not other.is_session_dirty()
        np.testing.assert_array_equal(other.current_time(), view.time)
    finally:
        other.shutdown()
        other._compute.wait_for_done(30_000)


def test_reopening_a_dirty_project_asks(
    ctl: AppController, qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "p.cupw.json"
    _saved_project(qtbot, ctl, path)
    ctl.open_project(path)
    _idle(qtbot, ctl)
    view = _view(ctl)
    ctl.session_do(
        "set_note", id=view.rows[0].id, note="edited", expect_id_epoch=view.id_epoch
    )
    _idle(qtbot, ctl)
    provider = _provider(ctl, ["cancel"])
    ctl.open_project(path)
    assert provider.calls[0][0] == "confirm_discard"
    assert _view(ctl).rows[0].note == "edited"


def test_a_project_whose_data_is_missing_asks_for_the_file(
    ctl: AppController, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "p.cupw.json"
    _saved_project(qtbot, ctl, path)
    real_load = PreWhitenSession.load
    seen: list[Any] = []

    def load(cls: Any, target: Any, **kwargs: Any) -> PreWhitenSession:
        seen.append(kwargs.get("data"))
        if kwargs.get("data") is None:
            raise ProjectError("data file 'star.csv' not found next to the project")
        return real_load(target, data=None)

    monkeypatch.setattr(PreWhitenSession, "load", classmethod(load))
    provider = _provider(ctl, ["elsewhere/star.csv"])
    with qtbot.waitSignal(ctl.settings_restored, timeout=30_000):
        ctl.open_project(path)
    _idle(qtbot, ctl)
    assert provider.calls[0][0] == "open_file"
    assert seen == [None, "elsewhere/star.csv"]


def test_export_script_and_copy_entry(
    ctl: AppController, qtbot: QtBot, tmp_path: Path
) -> None:
    ctl.set_light_curve(_curve(1), "star")
    _stepped_and_noted(qtbot, ctl)
    script = tmp_path / "run.py"
    messages: list[str] = []
    ctl.session_message.connect(lambda _s, t: messages.append(t))
    ctl.export_script(script)
    _idle(qtbot, ctl)
    assert script.is_file() and "PreWhitenSession" in script.read_text("utf-8")
    assert any("Exported the script" in m for m in messages)
    with qtbot.waitSignal(ctl.entry_script_ready, timeout=10_000) as blocker:
        ctl.copy_entry_script(_view(ctl).history[-1].seq)
    seq, text = blocker.args
    assert seq == _view(ctl).history[-1].seq and "set_note" in text


# -- Stop ------------------------------------------------------------------------------
def test_stop_releases_a_one_shot_periodogram_at_once(
    ctl: AppController, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctl.set_analysis("periodogram")
    gate = threading.Event()
    entered = threading.Event()
    real = cuperiod.periodogram

    def slow(lc: Any, method: str, **kwargs: Any) -> Any:
        entered.set()
        gate.wait(30)
        return real(lc, method, **kwargs)

    monkeypatch.setattr(cuperiod, "periodogram", slow)
    ctl.set_light_curve(_curve(1), "star")
    applied: list[object] = []
    ctl.periodogram_ready.connect(applied.append)
    busy: list[bool] = []
    ctl.busy_changed.connect(busy.append)
    statuses: list[str] = []
    ctl.status_message.connect(statuses.append)
    ctl.run("GLS", "cpu", GLSSettings(), 5)
    assert entered.wait(10)
    assert ctl._compute.running_key is not None
    ctl.cancel_compute()
    assert busy[-1] is False and statuses  # released at once
    assert ctl.finishing
    gate.set()
    ctl._compute.wait_for_done(30_000)
    qtbot.wait(100)
    assert applied == []  # finished in the background, cached, not shown
    with qtbot.waitSignal(ctl.periodogram_ready, timeout=2_000):
        ctl.run("GLS", "cpu", GLSSettings(), 5)
    assert ctl.state.last_from_cache
