"""The History dock (design 4.8, stage G7)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PySide6 import QtTest
from pytestqt.qtbot import QtBot

import cuperiod as cup
from cuperiod.gui.commands import CommandRegistry
from cuperiod.gui.qt import Qt, QtWidgets
from cuperiod.gui.viewmodels import (
    HistoryItem,
    SessionKey,
    SessionView,
    build_session_view,
)
from cuperiod.gui.widgets.history_panel import SEQ_ROLE, UNREACHABLE_TIP, HistoryPanel


def _view(history: list[HistoryItem], cursor: int, **kw: Any) -> SessionView:
    n = 10
    base: dict[str, Any] = dict(
        key=SessionKey("s"), serial=1, id_epoch=0, light=True, rows=(),
        summary_html="", result=None, data_spectrum=None, residual_spectrum=None,
        window=None, threshold=None, candidate=None, time=np.arange(n, dtype=float),
        value=np.zeros(n), model=np.zeros(n), residuals=np.zeros(n),
        deleted=np.zeros(n, dtype=bool), point_codes=None, partitions=(),
        subsets=(), offsets_by=None, variation_by=None, weights_by=None,
        warnings=(), history=tuple(history), cursor=cursor, can_undo=cursor > 0,
        can_redo=cursor < len(history), pristine=False, dirty=True, status="",
        stop_reason="", project_path=None, backend="numpy",
    )
    base.update(kw)
    return SessionView(**base)


def _log(*, reachable: bool = True) -> list[HistoryItem]:
    return [
        HistoryItem(1, 1, "step", "Step: accepted F1 = 1.3 c/d", "accepted", reachable),
        HistoryItem(2, 1, "step", "Step: accepted F2 = 2.7 c/d", "accepted", reachable),
        HistoryItem(3, 1, "step", "Step: S/N 3.1 < 4, stopped", "stopped", reachable),
        HistoryItem(4, 2, "fix", "Fixed F1 frequency", "edited", True),
        HistoryItem(5, 3, "set_note", "Note on F1", "edited", True),
    ]


@pytest.fixture
def panel(qtbot: QtBot) -> HistoryPanel:
    widget = HistoryPanel()
    qtbot.addWidget(widget)
    return widget


def _top(panel: HistoryPanel) -> list[QtWidgets.QTreeWidgetItem]:
    tree = panel.tree
    return [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]


def test_empty_panel(panel: HistoryPanel) -> None:
    assert [i.text(1) for i in _top(panel)] == ["No session yet"]
    assert not panel.export_button.isEnabled()
    assert panel.tree.accessibleName() == "Session history"
    assert panel.export_button.accessibleName() == "Export script"


def test_groups_head_and_redo_tail(panel: HistoryPanel) -> None:
    panel.set_view(_view(_log(), cursor=4))
    top = _top(panel)
    assert [i.text(1) for i in top] == [
        "Session opened",
        "Auto: 3 steps, 2 accepted",
        "▶ Fixed F1 frequency",
        "Note on F1  (undone)",
    ]
    group = top[1]
    assert group.childCount() == 3 and group.text(0) == "1–3"
    assert [group.child(i).data(0, SEQ_ROLE) for i in range(3)] == [1, 2, 3]
    head, undone = top[2], top[3]
    assert head.font(1).bold() and not head.font(1).italic()
    assert "current state" in head.toolTip(1)
    assert undone.font(1).italic() and "Redo" in undone.toolTip(1)
    assert panel.tree.currentItem() is head
    assert panel.export_button.isEnabled()


def test_head_inside_a_group_expands_it(panel: HistoryPanel) -> None:
    panel.set_view(_view(_log(), cursor=2))
    group = _top(panel)[1]
    assert group.isExpanded()
    assert group.child(1).text(1).startswith("▶ ")
    assert group.font(1).bold()
    assert group.child(2).text(1).endswith("(undone)")


def test_unreachable_entries_are_greyed_and_say_why(panel: HistoryPanel) -> None:
    panel.set_view(_view(_log(reachable=False), cursor=5))
    group = _top(panel)[1]
    step = group.child(0)
    assert step.text(1).endswith("(not in memory)")
    assert UNREACHABLE_TIP in step.toolTip(1)
    fix = _top(panel)[2]
    assert step.foreground(1).color() != fix.foreground(1).color()
    assert "not in memory" in str(step.data(1, Qt.ItemDataRole.AccessibleTextRole))


def test_activation_requests_goto(panel: HistoryPanel) -> None:
    panel.set_view(_view(_log(), cursor=5))
    seen: list[int] = []
    panel.goto_requested.connect(seen.append)
    top = _top(panel)
    for item in (top[0], top[1], top[1].child(0), top[3]):
        panel.tree.itemActivated.emit(item, 1)
    assert seen == [0, 3, 1, 5]


def test_double_click_and_enter_go_to_a_state(
    qtbot: QtBot, panel: HistoryPanel
) -> None:
    panel.set_view(_view(_log(), cursor=5))
    panel.show()
    seen: list[int] = []
    panel.goto_requested.connect(seen.append)
    tree = panel.tree
    fix = _top(panel)[2]
    rect = tree.visualItemRect(fix)
    viewport = tree.viewport()
    # A double click lands on the item a press selected first (as a real one does).
    QtTest.QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=rect.center())
    assert seen == []  # a single click only selects
    QtTest.QTest.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=rect.center())
    assert seen == [4]
    tree.setFocus()
    tree.setCurrentItem(_top(panel)[0])
    QtTest.QTest.keyClick(tree, Qt.Key.Key_Return)
    assert seen[-1] == 0


def test_copy_as_python(qtbot: QtBot, panel: HistoryPanel) -> None:
    panel.set_view(_view(_log(), cursor=5))
    seen: list[int] = []
    panel.copy_script_requested.connect(seen.append)
    fix = _top(panel)[2]
    menu = panel.context_menu(fix)
    texts = [a.text().split("\t")[0] for a in menu.actions() if a.text()]
    assert texts == ["Go to this state", "Copy as Python", "Export script…"]
    next(a for a in menu.actions() if a.text().startswith("Copy")).trigger()
    assert seen == [4]
    root_menu = panel.context_menu(_top(panel)[0])
    copy = next(a for a in root_menu.actions() if a.text().startswith("Copy"))
    assert not copy.isEnabled()  # "Session opened" is not an action
    panel.show()
    panel.tree.setFocus()
    qtbot.waitUntil(panel.tree.hasFocus, timeout=2000)
    panel.tree.setCurrentItem(_top(panel)[1].child(1))
    QtTest.QTest.keyClick(panel.tree, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert seen == [4, 2]
    assert menu.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)


def test_export_script(qtbot: QtBot) -> None:
    panel = HistoryPanel()
    qtbot.addWidget(panel)
    panel.set_view(_view(_log(), cursor=5))
    fired: list[bool] = []
    panel.export_script_requested.connect(lambda: fired.append(True))
    panel.export_button.click()
    assert fired == [True]
    window = QtWidgets.QWidget()
    qtbot.addWidget(window)
    registry = CommandRegistry(window)
    registry.set_state(session=True)
    panel.bind_commands(registry)
    triggered: list[str] = []
    registry.triggered.connect(triggered.append)
    panel.export_button.click()
    assert triggered == ["file.export_script"] and fired == [True]
    assert panel.export_button.text() == "Export script…"


def test_rebuilt_only_when_the_log_moves(panel: HistoryPanel) -> None:
    panel.set_view(_view(_log(), cursor=5))
    first = _top(panel)[1]
    panel.set_view(_view(_log(), cursor=5, serial=2))
    assert _top(panel)[1] is first
    first.setExpanded(True)  # the user opens the Auto group
    panel.set_view(_view(_log(), cursor=4))
    assert _top(panel)[1] is not first
    assert _top(panel)[1].isExpanded()  # kept across the rebuild
    _top(panel)[1].setExpanded(False)
    panel.set_view(_view(_log(), cursor=5))
    assert not _top(panel)[1].isExpanded()


def test_real_session_log(panel: HistoryPanel) -> None:
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0.0, 30.0, 400))
    y = 0.3 * np.sin(2 * np.pi * 1.3 * t) + rng.normal(0.0, 0.01, t.size)
    session = cup.PreWhitenSession((t, y, np.full(t.size, 0.01)), backend="numpy")
    session.run()
    session.set_note(1, "main")
    session.undo()
    view = build_session_view(session, key=SessionKey("r"), id_epoch=0)
    panel.set_view(view)
    top = _top(panel)
    assert top[0].text(1).startswith("▶") is False
    texts = [i.text(1) for i in top]
    assert texts[-1].endswith("(undone)")
    run = top[1]
    assert run.childCount() >= 2  # the Auto run is one group
    assert any(t.startswith("▶ ") for t in [run.text(1)] + [
        run.child(i).text(1) for i in range(run.childCount())
    ])
