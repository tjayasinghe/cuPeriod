"""The session bar (design 4.4, stage G7) with fabricated view models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PySide6 import QtTest
from pytestqt.qtbot import QtBot

from cuperiod.gui.commands import CommandRegistry, command
from cuperiod.gui.qt import Qt, QtWidgets
from cuperiod.gui.viewmodels import (
    CandidateView,
    ComponentView,
    SessionKey,
    SessionView,
)
from cuperiod.gui.widgets.frequency_table import split_epoch, split_group
from cuperiod.gui.widgets.session_bar import (
    BUTTONS,
    SessionBar,
    candidate_text,
    near_combination,
    next_id,
)
from cuperiod.prewhiten.session import actions


def _row(i: int, frequency: float, **kw: Any) -> ComponentView:
    base: dict[str, Any] = dict(
        id=i, label=f"F{i}", rank=i, frequency=frequency, frequency_error=1e-4,
        period=1.0 / frequency, amplitude=0.1, amplitude_error=1e-3, phase=0.5,
        phase_error=0.01, amplitude_rms=None, snr=12.0, fap=1e-6, delta_bic=-40.0,
        spectrum_amplitude=0.1, amplitude_ratio=1.0, blended=False, active=True,
        fix_frequency=False, fix_amplitude=False, fix_phase=False, tie=None,
        variation="none", combination=None, origin="step", flags=frozenset(),
        note="", passes=True,
    )
    base.update(kw)
    return ComponentView(**base)


def _view(rows: list[ComponentView], **kw: Any) -> SessionView:
    n = 40
    base: dict[str, Any] = dict(
        key=SessionKey("s"), serial=1, id_epoch=5, light=True, rows=tuple(rows),
        summary_html="", result=None, data_spectrum=None, residual_spectrum=None,
        window=None, threshold=None, candidate=None,
        time=np.linspace(0.0, 100.0, n), value=np.zeros(n), model=np.zeros(n),
        residuals=np.zeros(n), deleted=np.zeros(n, dtype=bool), point_codes=None,
        partitions=(), subsets=(), offsets_by=None, variation_by=None,
        weights_by=None, warnings=(), history=(), cursor=0, can_undo=True,
        can_redo=False, pristine=False, dirty=True, status="", stop_reason="",
        project_path=None, backend="numpy",
    )
    base.update(kw)
    return SessionView(**base)


@pytest.fixture
def bar(qtbot: QtBot) -> SessionBar:
    widget = SessionBar()
    qtbot.addWidget(widget)
    return widget


def test_every_button_is_named_and_explained(bar: SessionBar) -> None:
    assert [cid for cid, _ in BUTTONS] == [
        "session.auto", "session.step", "session.add_peak", "session.calculate",
        "session.improve_all", "session.improve_selected", "edit.delete",
        "edit.undo", "edit.redo", "session.stop",
    ]
    for cid, text in BUTTONS:
        button = bar.button(cid)
        assert button.text() == text
        assert button.accessibleName() == command(cid).accessible_name
        assert button.toolTip() == command(cid).tooltip  # ends with its keys
        assert button.defaultAction().statusTip()
    assert bar.field.accessibleName() == "Add frequency"
    assert bar.field.toolTip() and bar.chip.accessibleName() == "Next candidate"


def test_buttons_emit_their_command(bar: SessionBar) -> None:
    pressed: list[str] = []
    bar.command_requested.connect(pressed.append)
    for cid in ("session.step", "session.improve_selected", "edit.delete"):
        bar.button(cid).click()
    assert pressed == ["session.step", "session.improve_selected", "edit.delete"]


def test_enabled_state_follows_view_and_busy(bar: SessionBar) -> None:
    assert not bar.button("session.stop").isEnabled()
    assert not bar.button("edit.undo").isEnabled()  # no session yet
    bar.set_view(_view([], can_undo=True, can_redo=False))
    assert bar.button("edit.undo").isEnabled()
    assert not bar.button("edit.redo").isEnabled()
    bar.set_busy(True)
    assert bar.button("session.stop").isEnabled()
    assert not any(
        bar.button(cid).isEnabled() for cid, _ in BUTTONS if cid != "session.stop"
    )
    assert not bar.field.isEnabled()
    bar.set_busy(False)
    assert not bar.button("session.stop").isEnabled()
    assert bar.button("session.auto").isEnabled()


def test_add_field_adds_a_frequency_or_a_period(bar: SessionBar) -> None:
    edits: list[tuple[str, dict[str, Any]]] = []
    bar.edit_requested.connect(lambda op, args: edits.append((op, args)))
    bar.set_view(_view([_row(1, 2.0)]))
    for text in ("12.34", "0.081 d"):
        bar.field.setText(text)
        assert bar.submit_entry()
        assert bar.field.text() == ""
    assert [op for op, _ in edits] == ["add", "add"]
    for op, args in edits:
        assert "expect_id_epoch" in args
        clean, _ = split_epoch(args)
        actions.parse(op, clean)
    assert edits[0][1]["frequency"] == 12.34
    assert edits[1][1]["frequency"] == pytest.approx(1 / 0.081)


@pytest.mark.parametrize(("text", "frequency"), [("=F1+F2", 5.0), ("2f1", 4.0)])
def test_add_field_adds_and_ties_a_combination(
    bar: SessionBar, text: str, frequency: float
) -> None:
    groups: list[Any] = []
    bar.group_requested.connect(groups.append)
    bar.set_view(_view([_row(1, 2.0), _row(2, 3.0)], id_epoch=9))
    bar.field.setText(text)
    QtTest.QTest.keyClick(bar.field, Qt.Key.Key_Return)
    assert len(groups) == 1
    clean, epoch = split_group(groups[0])
    assert epoch == 9
    assert all(args["expect_id_epoch"] == 9 for _, args in groups[0])
    expression = "F1 + F2" if text == "=F1+F2" else "2F1"
    assert clean == [
        ("add", {"frequency": pytest.approx(frequency), "refine": "none"}),
        ("tie", {"id": 3, "expression": expression}),
    ]
    for op, args in clean:
        actions.parse(op, args)


def test_combination_naming_a_missing_component_is_refused(bar: SessionBar) -> None:
    messages: list[str] = []
    groups: list[Any] = []
    bar.status_message.connect(messages.append)
    bar.group_requested.connect(groups.append)
    bar.set_view(_view([_row(1, 2.0), _row(4, 3.0)]))
    bar.field.setText("=F1+F2")
    assert not bar.submit_entry()
    assert groups == []
    assert "F2" in messages[0]
    assert bar.field.text() == "=F1+F2"  # kept for fixing
    assert messages[0] in bar.field.toolTip()
    bar.field.setText("=F1+F4")
    assert bar.submit_entry()
    clean, _ = split_group(groups[0])
    assert clean[0][1]["frequency"] == pytest.approx(5.0)
    assert clean[1][1] == {"id": 5, "expression": "F1 + F4"}


def test_bad_entries_report_and_keep_the_text(bar: SessionBar) -> None:
    messages: list[str] = []
    bar.status_message.connect(messages.append)
    for text in ("", "fast", "-2", "3 lightyears"):
        bar.field.setText(text)
        assert not bar.submit_entry()
    assert len(messages) == 4
    assert all(m.startswith("Add frequency:") for m in messages)


def test_next_id_prefers_the_views_own(bar: SessionBar) -> None:
    view = _view([_row(2, 1.0), _row(7, 2.0)])
    assert next_id(view) == 8
    assert next_id(_view([])) == 1

    class _WithNext:
        next_id = 12
        rows = view.rows

    assert next_id(_WithNext()) == 12  # type: ignore[arg-type]


def test_candidate_chip(bar: SessionBar) -> None:
    rows = [_row(1, 2.0), _row(2, 3.1)]
    candidate = CandidateView(5.1001, 1 / 5.1001, 0.0039, 8.24, "Next peak; Step (F7)")
    bar.set_view(_view(rows, candidate=candidate))
    text = bar.chip.text()
    assert text.startswith("Next: 5.1001 c/d")
    assert "P 0.1960746 d" in text and "A 0.0039" in text and "S/N ≈ 8.2" in text
    assert text.endswith("near F1 + F2")
    assert bar.chip.toolTip() == "Next peak; Step (F7)"
    bar.set_view(_view(rows, candidate=None))
    assert bar.chip.text() == "Next peak: –"
    bar.clear()
    assert bar.chip.text() == "Next peak: –"


def test_near_combination_picks_the_lowest_order() -> None:
    rows = [_row(1, 2.0), _row(2, 3.1), _row(3, 0.5, active=False)]
    assert near_combination(4.0005, rows, 0.01) == "2F1"
    assert near_combination(1.1, rows, 0.01) == "F2 - F1"
    assert near_combination(7.1, rows, 0.01) == "2F1 + F2"
    assert near_combination(2.5, rows, 0.01) is None  # F3 is not in use
    assert near_combination(9.9, rows, 0.01) is None
    candidate = CandidateView(4.0, 0.25, 0.1, float("nan"), "")
    assert "S/N ≈ –" in candidate_text(candidate, rows, 0.01)


def test_bound_to_the_registry(qtbot: QtBot) -> None:
    window = QtWidgets.QWidget()
    qtbot.addWidget(window)
    layout = QtWidgets.QVBoxLayout(window)
    bar = SessionBar()
    layout.addWidget(bar)
    registry = CommandRegistry(window)
    bar.bind_commands(registry)
    registry.set_state(session=True, busy=False)
    window.show()
    triggered: list[str] = []
    local: list[str] = []
    registry.triggered.connect(triggered.append)
    bar.command_requested.connect(local.append)
    step = bar.button("session.step")
    assert step.defaultAction() is registry.action("session.step")
    assert step.text() == "Step"  # short text, while the menu keeps the long one
    assert registry.action("session.auto").text().startswith("Auto —")
    step.click()
    assert triggered == ["session.step"] and local == []
    registry.set_state(busy=True)
    assert bar.button("session.stop").isEnabled()
    assert not step.isEnabled()
    registry.set_state(busy=False)
    registry.action("session.add_frequency").trigger()
    qtbot.waitUntil(bar.field.hasFocus, timeout=2000)


def test_letters_typed_in_the_field_trigger_nothing(qtbot: QtBot) -> None:
    window = QtWidgets.QWidget()
    qtbot.addWidget(window)
    layout = QtWidgets.QVBoxLayout(window)
    bar = SessionBar()
    table = QtWidgets.QTableWidget()
    layout.addWidget(bar)
    layout.addWidget(table)
    registry = CommandRegistry(window)
    registry.attach_context("table", table)
    registry.set_state(session=True, busy=False)
    window.show()
    window.activateWindow()
    qtbot.waitActive(window)
    triggered: list[str] = []
    registry.triggered.connect(triggered.append)
    bar.focus_add_field()
    qtbot.waitUntil(bar.field.hasFocus, timeout=2000)
    QtTest.QTest.keyClicks(bar.field, "2f1 sci")
    assert bar.field.text() == "2f1 sci"
    assert triggered == []
