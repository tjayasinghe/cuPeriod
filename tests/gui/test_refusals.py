"""Refused edits: the library's remedy (``StepReport.hint``) in the window's terms.

A refusal reaches the status bar worded for a GUI user (from its hint, else by
rewriting the library's message), and a remedy the window can carry out is offered
through the DialogProvider — a non-blocking question — and done on a yes: dissolve
a tie parent's ties and remove it, add or tie a frequency anyway, add one exactly.
The bridge, the controller and the window are tested with scripted refusals, so the
tests do not depend on which refusals the library hints; sessions run on ``numpy``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

import cuperiod as cup
from cuperiod.gui.compute import ComputeManager
from cuperiod.gui.dialogs import DialogProvider, FakeDialogProvider, QtDialogProvider
from cuperiod.gui.main_window import MainWindow
from cuperiod.gui.qt import QtCore, QtWidgets
from cuperiod.gui.refusals import (
    HINT_ARGUMENTS,
    Refusal,
    gui_wording,
    refusal_offer,
    refusal_text,
    report_hint,
)
from cuperiod.gui.session_bridge import (
    STALE_RETRY_MESSAGE,
    SessionBridge,
    SessionKey,
)
from cuperiod.gui.state import AppController

KEY = SessionKey("star", "")

CASCADE = (
    "F1 is a tie parent of F5 (F5 = F1 + F2); remove with cascade=True to dissolve "
    "the tie"
)
CLOSE_ADD = (
    "12.3456 c/d is within 0.25 Rayleigh widths of F3 (12.3461 c/d); pass "
    "allow_close=True to add it anyway"
)
CLOSE_TIE = (
    "tie puts F5 at 3 c/d, within 0.25 Rayleigh widths of F2; pass allow_close=True "
    "to tie it anyway"
)
OUTSIDE = (
    "2000 c/d is outside the frequency grid [0.02, 980.2]; add it with refine='none'"
)
MOVED = (
    "the refinement moved 12.34 c/d onto F2 (12.3748547 c/d); add it with "
    "refine='none' or allow_close=True"
)
ZERO_POINTS = "zero points per subset need fit_mean=True (configure it first)"


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path: Path) -> None:
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        str(tmp_path / "settings"),
    )


@dataclass(frozen=True)
class _Refused:
    """A refused report, with or without the library's hint."""

    message: str
    hint: str | None = None
    kind: str = "refused"
    warning: str = ""
    changed: frozenset[str] = frozenset()
    statistics: Mapping[str, float] = field(default_factory=dict)
    component_id: int | None = None

    @property
    def committed(self) -> bool:
        return False


@dataclass
class _Row:
    id: int
    tie: str | None = None

    @property
    def label(self) -> str:
        return f"F{self.id}"


class _Retry:
    """A refusal's retry that records what it was asked to add."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, extra: Mapping[str, Any]) -> None:
        self.calls.append(dict(extra))


# -- wording ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("message", "hint", "said", "unsaid"),
    [
        (CASCADE, "cascade", "F1 is a tie parent of F5 (F5 = F1 + F2): untie it", "="),
        (CLOSE_ADD, "allow_close", "(12.3461 c/d): too close to tell apart", "pass"),
        (CLOSE_TIE, "allow_close", "within 0.25 Rayleigh widths of F2: too close", "_"),
        (OUTSIDE, "exact", "[0.02, 980.2]: add it exactly instead", "refine="),
        (MOVED, "exact", "(12.3748547 c/d): add it exactly instead", "refine="),
        (ZERO_POINTS, "fit_mean", "need a floating mean: tick", "fit_mean"),
    ],
)
def test_a_hint_words_the_remedy_in_the_windows_terms(
    message: str, hint: str, said: str, unsaid: str
) -> None:
    text = refusal_text(Refusal(KEY, message, hint))
    assert said in text
    assert unsaid not in text.replace("F5 = F1 + F2", "")  # no keyword argument


def test_without_a_hint_the_known_messages_are_rewritten() -> None:
    # A library from before StepReport.hint: the message itself is reworded.
    assert refusal_text(Refusal(KEY, MOVED)) == gui_wording(MOVED)
    assert "Add exactly here" in refusal_text(Refusal(KEY, MOVED))
    assert refusal_text(Refusal(KEY, "nothing to improve")) == "nothing to improve"
    # A hint this version does not know is no hint.
    assert refusal_text(Refusal(KEY, MOVED, "teleport")) == gui_wording(MOVED)


def test_the_hint_is_read_with_getattr() -> None:
    assert report_hint(_Refused(CASCADE, "cascade")) == "cascade"
    assert report_hint(_Refused(CASCADE, "teleport")) is None
    assert report_hint(object()) is None  # a report from before the field existed
    assert set(HINT_ARGUMENTS) == {"allow_close", "cascade", "exact"}


# -- offers ----------------------------------------------------------------------
def test_offers_name_what_will_happen() -> None:
    retry = _Retry()
    rows = [_Row(1), _Row(2), _Row(5, "F1 + F2"), _Row(7, "2F1 - F3"), _Row(8, "F2")]
    offer = refusal_offer(
        Refusal(KEY, CASCADE, "cascade", "remove", {"ids": [1]}, retry), rows
    )
    assert offer is not None and offer.accept == "Dissolve ties and remove"
    assert offer.text == (
        "F1 has tied children (F5, F7). Dissolve their ties and remove F1?"
    )
    single = refusal_offer(
        Refusal(KEY, CASCADE, "cascade", "remove", {"ids": [2, 3]}, retry), rows
    )
    assert single is not None and single.text == (
        "F2 and F3 have tied children (F5, F7, F8). Dissolve their ties and remove "
        "F2 and F3?"
    )
    one = refusal_offer(
        Refusal(KEY, CASCADE, "cascade", "remove", {"ids": ["F7"]}, retry),
        [_Row(7), _Row(9, "F7 + F1")],
    )
    assert one is not None and one.text == (
        "F7 has a tied child (F9). Dissolve its tie and remove F7?"
    )
    close = refusal_offer(
        Refusal(KEY, CLOSE_ADD, "allow_close", "add", {"frequency": 12.3}, retry)
    )
    assert close is not None and close.accept == "Add anyway"
    assert "within the collision guard of F3" in close.text
    assert close.text.endswith("Add it anyway?")
    tie = refusal_offer(
        Refusal(KEY, CLOSE_TIE, "allow_close", "tie", {"id": 5}, retry)
    )
    assert tie is not None and tie.accept == "Tie anyway"
    assert tie.text.startswith("This tie puts F5 within the collision guard of F2")
    exact = refusal_offer(Refusal(KEY, MOVED, "exact", "add", {}, retry))
    assert exact is not None and exact.accept == "Add exactly"
    assert exact.text == (
        "The refinement moved 12.34 c/d onto F2 (12.3748547 c/d). Add it exactly at "
        "12.34 c/d (no refinement)?"
    )


def test_nothing_is_offered_that_the_window_cannot_do() -> None:
    retry = _Retry()
    ids = {"ids": [1]}
    zero = Refusal(KEY, ZERO_POINTS, "fit_mean", "use_offsets", {}, retry)
    assert refusal_offer(zero) is None  # the user ticks "floating mean"
    assert refusal_offer(Refusal(KEY, CASCADE, None, "remove", ids, retry)) is None
    # Not repeatable (a Step, another session's edit): reported only.
    assert refusal_offer(Refusal(KEY, CASCADE, "cascade", "remove", ids)) is None


# -- the dialog ------------------------------------------------------------------
def test_the_question_is_asked_without_a_nested_event_loop(qtbot: QtBot) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    provider = QtDialogProvider(parent, native_files=False)
    assert isinstance(provider, DialogProvider)
    got: list[bool] = []
    provider.confirm("Remove and untie?", "F1 has … remove F1?", "Remove", got.append)
    (box,) = provider.live_dialogs  # open() returned at once
    assert isinstance(box, QtWidgets.QMessageBox) and box.isVisible()
    assert box.text() == "F1 has … remove F1?"
    assert box.textFormat() == QtCore.Qt.TextFormat.PlainText
    accept = next(b for b in box.buttons() if b.text() == "Remove")
    assert box.defaultButton() is accept
    accept.click()
    qtbot.waitUntil(lambda: got == [True])
    provider.confirm("Add anyway?", "Add it anyway?", "Add anyway", got.append)
    (box,) = provider.live_dialogs
    box.reject()  # Esc, Cancel or the title bar
    qtbot.waitUntil(lambda: got == [True, False])
    fake = FakeDialogProvider([True, "a path"])
    fake.confirm("t", "q", "Yes", got.append)
    fake.confirm("t", "q", "Yes", got.append)  # only True accepts
    fake.confirm("t", "q", "Yes", got.append)  # exhausted: declined
    assert got[2:] == [True, False, False]
    assert fake.calls[0] == ("confirm", ("t", "q", "Yes"))


# -- the bridge ------------------------------------------------------------------
def _data(n: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 40.0, n))
    y = (
        0.3 * np.sin(2 * np.pi * 1.3 * t)
        + 0.12 * np.sin(2 * np.pi * 2.71 * t + 1.0)
        + rng.normal(0.0, 0.01, n)
    )
    return t, y, np.full(n, 0.01)


DATA = _data()


def _scripted(
    calls: list[tuple[str, dict[str, Any]]],
    refuse: Callable[[str, dict[str, Any]], _Refused | None],
) -> Callable[[], cup.PreWhitenSession]:
    """A session whose ``do`` records every op and refuses as ``refuse`` says."""

    def make() -> cup.PreWhitenSession:
        session = cup.PreWhitenSession(DATA, backend="numpy")
        real_do = session.do

        def do(op: str, **args: Any) -> Any:
            calls.append((op, dict(args)))
            refused = refuse(op, args)
            return refused if refused is not None else real_do(op, **args)

        session.do = do  # type: ignore[method-assign]
        return session

    return make


@pytest.fixture
def bridge(qtbot: QtBot) -> Iterator[tuple[ComputeManager, SessionBridge]]:
    compute = ComputeManager()
    yield compute, SessionBridge(compute, progress_interval=0.0)
    compute.wait_for_done(20_000)


def _idle(qtbot: QtBot, rig: tuple[ComputeManager, SessionBridge]) -> None:
    compute, bridge = rig
    qtbot.waitUntil(
        lambda: bridge.is_idle() and compute.session_jobs == 0, timeout=30_000
    )


def test_a_refusal_carries_its_hint_and_retries_with_the_remedy(
    bridge: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    _, b = bridge
    calls: list[tuple[str, dict[str, Any]]] = []

    def refuse(op: str, args: dict[str, Any]) -> _Refused | None:
        if op == "remove" and not args.get("cascade"):
            return _Refused(CASCADE, "cascade")
        return None

    refusals: list[tuple[object, Refusal]] = []
    messages: list[str] = []
    b.refused.connect(lambda key, r: refusals.append((key, r)))
    b.message.connect(lambda _s, text: messages.append(text))
    b.display(KEY)
    b.open(KEY, _scripted(calls, refuse), backend="numpy")
    b.submit(KEY, "step")
    b.submit(KEY, "step")
    _idle(qtbot, bridge)
    b.submit(KEY, "remove", {"ids": [1]})
    _idle(qtbot, bridge)
    ((key, refusal),) = refusals
    assert key == KEY and refusal.key == KEY
    assert (refusal.hint, refusal.op, dict(refusal.args)) == (
        "cascade", "remove", {"ids": [1]},
    )
    assert refusal.text == CASCADE and refusal.retry is not None
    assert messages == []  # a refusal is reported once, as a refusal
    # Only the remedy's arguments the op takes are added.
    refusal.retry({"cascade": True, "allow_close": True})
    _idle(qtbot, bridge)
    assert calls[-1] == ("remove", {"ids": [1], "cascade": True})
    view = b.latest_view(KEY)
    assert view is not None and [r.id for r in view.rows] == [2]


def test_a_retry_after_the_session_changed_is_dropped(
    bridge: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    _, b = bridge
    calls: list[tuple[str, dict[str, Any]]] = []

    def refuse(op: str, args: dict[str, Any]) -> _Refused | None:
        if op == "remove" and not args.get("cascade"):
            return _Refused(CASCADE, "cascade")
        return None

    refusals: list[Refusal] = []
    messages: list[str] = []
    b.refused.connect(lambda _key, r: refusals.append(r))
    b.message.connect(lambda _s, text: messages.append(text))
    b.display(KEY)
    b.open(KEY, _scripted(calls, refuse), backend="numpy")
    b.submit(KEY, "step")
    b.submit(KEY, "remove", {"ids": [1]})
    _idle(qtbot, bridge)
    b.submit(KEY, "step")  # the session moves on while the user is asked
    _idle(qtbot, bridge)
    (refusal,) = refusals
    assert refusal.retry is not None
    refusal.retry({"cascade": True})
    _idle(qtbot, bridge)
    assert messages[-1] == STALE_RETRY_MESSAGE
    assert ("remove", {"ids": [1], "cascade": True}) not in calls
    view = b.latest_view(KEY)
    assert view is not None and len(view.rows) == 2


def test_a_refused_group_retries_what_was_refused(
    bridge: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    # A typed combination: add then tie, as one undo group. When the add is
    # refused, the tie of the component it would have made is refused with it,
    # and the retry repeats both, each with the remedy's arguments it takes.
    _, b = bridge
    calls: list[tuple[str, dict[str, Any]]] = []

    def refuse(op: str, args: dict[str, Any]) -> _Refused | None:
        if args.get("allow_close"):
            return None
        if op == "add":
            return _Refused(CLOSE_ADD, "allow_close")
        if op == "tie":
            return _Refused("no component F3")  # the refused add made none
        return None

    refusals: list[Refusal] = []
    b.refused.connect(lambda _key, r: refusals.append(r))
    b.display(KEY)
    b.open(KEY, _scripted(calls, refuse), backend="numpy")
    b.submit(KEY, "step")
    b.submit(KEY, "step")
    _idle(qtbot, bridge)
    group = [
        ("add", {"frequency": 4.01, "refine": "none"}),
        ("tie", {"id": 3, "expression": "F1 + F2"}),
    ]
    b.submit_group(KEY, group)
    _idle(qtbot, bridge)
    (refusal,) = refusals  # the first refusal of the job speaks for it
    assert (refusal.op, refusal.hint, refusal.text) == ("add", "allow_close", CLOSE_ADD)
    assert refusal.retry is not None
    del calls[:]
    refusal.retry({"allow_close": True})
    _idle(qtbot, bridge)
    assert calls == [
        ("add", {"frequency": 4.01, "refine": "none", "allow_close": True}),
        ("tie", {"id": 3, "expression": "F1 + F2", "allow_close": True}),
    ]
    view = b.latest_view(KEY)
    assert view is not None and view.rows[-1].tie == "F1 + F2"


def test_only_submitted_ops_can_be_retried(
    bridge: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    _, b = bridge
    refusals: list[Refusal] = []
    b.refused.connect(lambda _key, r: refusals.append(r))
    b.display(KEY)
    b.open(KEY, _scripted([], lambda op, args: None), backend="numpy")

    def call(session: cup.PreWhitenSession) -> Any:
        return _Refused(CASCADE, "cascade")

    b.submit_call(KEY, "custom", call)
    _idle(qtbot, bridge)
    (refusal,) = refusals
    assert refusal.hint == "cascade" and refusal.op == "" and refusal.retry is None


def test_an_add_that_snaps_goes_through_the_session_facade(
    bridge: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    # The spectrum's click adds at the residual peak near the click; only the
    # worker has that spectrum, so the bridge snaps there (PreWhitenSession.add).
    _, b = bridge
    calls: list[tuple[str, dict[str, Any]]] = []
    b.display(KEY)
    b.open(KEY, _scripted(calls, lambda op, args: None), backend="numpy")
    b.submit(KEY, "step")
    _idle(qtbot, bridge)
    b.submit(KEY, "add", {"frequency": 2.705, "snap": True})
    _idle(qtbot, bridge)
    view = b.latest_view(KEY)
    assert view is not None and len(view.rows) == 2
    snapped = view.rows[-1].frequency
    assert snapped != 2.705 and abs(snapped - 2.71) < 0.01
    assert view.history[-1].op == "add"
    assert not [args for op, args in calls if "snap" in args]  # never do("add", snap=)


# -- the controller --------------------------------------------------------------
def test_the_controller_carries_out_an_accepted_remedy(qtbot: QtBot) -> None:
    ctl = AppController(dialogs=FakeDialogProvider())
    retry = _Retry()
    for hint, done in (("cascade", True), ("allow_close", True), ("exact", True)):
        assert ctl.retry_refused(Refusal(KEY, "x", hint, "op", {}, retry)) is done
    assert retry.calls == [{"cascade": True}, {"allow_close": True}, {"refine": "none"}]
    assert not ctl.retry_refused(Refusal(KEY, ZERO_POINTS, "fit_mean", "", {}, retry))
    assert not ctl.retry_refused(Refusal(KEY, CASCADE, "cascade"))  # nothing to retry
    assert len(retry.calls) == 3
    # A refusal of a session that is not on show is reported, never retried.
    seen: list[Refusal] = []
    ctl.session_refused.connect(seen.append)
    other = SessionKey("other")
    ctl.session_bridge.refused.emit(
        other, Refusal(other, CASCADE, "cascade", "remove", {"ids": [1]}, retry)
    )
    (shown,) = seen
    assert shown.hint == "cascade" and shown.retry is None
    ctl.shutdown()


# -- the window ------------------------------------------------------------------
def _window(qtbot: QtBot, provider: FakeDialogProvider) -> MainWindow:
    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    window = MainWindow(app)
    qtbot.addWidget(window)
    window.set_dialog_provider(provider)
    window.resize(1366, 768)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_the_window_offers_the_remedy_and_does_it_on_a_yes(qtbot: QtBot) -> None:
    provider = FakeDialogProvider([True])
    window = _window(qtbot, provider)
    status = window.statusBar()
    assert status is not None
    retry = _Retry()
    refusal = Refusal(KEY, CASCADE, "cascade", "remove", {"ids": [1]}, retry)
    window.controller.session_refused.emit(refusal)
    assert status.currentMessage().startswith("F1 is a tie parent of F5")
    assert "untie it first" in status.currentMessage()
    (call,) = [c for c in provider.calls if c[0] == "confirm"]
    title, text, accept = call[1]
    assert (title, accept) == ("Remove and untie?", "Dissolve ties and remove")
    assert "remove F1?" in text
    assert retry.calls == [{"cascade": True}]
    # No: nothing happens, and the status bar says so.
    window.controller.session_refused.emit(
        Refusal(KEY, CLOSE_ADD, "allow_close", "add", {"frequency": 12.3}, retry)
    )
    assert provider.calls[-1][1][2] == "Add anyway"
    assert retry.calls == [{"cascade": True}]
    assert status.currentMessage() == "Nothing was changed."
    window.controller.shutdown()


def test_a_remedy_the_window_cannot_do_is_only_explained(qtbot: QtBot) -> None:
    provider = FakeDialogProvider([True])
    window = _window(qtbot, provider)
    status = window.statusBar()
    assert status is not None
    retry = _Retry()
    window.controller.session_refused.emit(
        Refusal(KEY, ZERO_POINTS, "fit_mean", "use_offsets", {"name": "night"}, retry)
    )
    assert "tick “floating mean” in the Options" in status.currentMessage()
    # Without a hint (an older library) the message is reworded, nothing offered.
    window.controller.session_refused.emit(Refusal(KEY, MOVED, None, "add", {}, retry))
    assert "Add exactly here" in status.currentMessage()
    assert not [c for c in provider.calls if c[0] == "confirm"] and not retry.calls
    window.controller.shutdown()
