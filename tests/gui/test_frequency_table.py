"""The editable frequency table (design 4.4, stage G3), driven offscreen."""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from PySide6 import QtTest
from pytestqt.qtbot import QtBot

import cuperiod as cup
from cuperiod.gui.commands import CommandRegistry
from cuperiod.gui.qt import Qt, QtWidgets
from cuperiod.gui.viewmodels import (
    ComponentView,
    SessionKey,
    SessionView,
    build_session_view,
)
from cuperiod.gui.widgets.frequency_table import (
    COLUMN_KEYS,
    COLUMNS,
    EMPTY_HINT,
    FrequencyTable,
    parse_entry,
    split_epoch,
    split_group,
    tie_frequency,
)
from cuperiod.gui.widgets.solution_panel import SESSION_CSV_FIELDS, SolutionPanel
from cuperiod.prewhiten.session import actions

_READ_ONLY_HEADERS = [
    "ID", "frequency", "± f", "amplitude", "± A", "S/N", "combination",
    "period (d)", "A/Asp", "phase", "± ph", "FAP",
]


def _row(i: int, frequency: float | None = None, **kw: Any) -> ComponentView:
    f = float(i) * 1.1 if frequency is None else frequency
    base: dict[str, Any] = dict(
        id=i, label=f"F{i}", rank=i, frequency=f, frequency_error=1e-4,
        period=1.0 / f, amplitude=0.1 / i, amplitude_error=1e-3, phase=0.5,
        phase_error=0.01, amplitude_rms=None, snr=12.0, fap=1e-6, delta_bic=-40.0,
        spectrum_amplitude=0.1 / i, amplitude_ratio=1.0, blended=False, active=True,
        fix_frequency=False, fix_amplitude=False, fix_phase=False, tie=None,
        variation="none", combination=None, origin="step", flags=frozenset(),
        note="", passes=True,
    )
    base.update(kw)
    return ComponentView(**base)


def _view(rows: list[ComponentView], *, epoch: int = 7, **kw: Any) -> SessionView:
    n = 50
    time = np.linspace(0.0, 20.0, n)
    base: dict[str, Any] = dict(
        key=SessionKey("s"), serial=1, id_epoch=epoch, light=True, rows=tuple(rows),
        summary_html="<b>2</b> active of 2 components", result=None,
        data_spectrum=None, residual_spectrum=None, window=None, threshold=None,
        candidate=None, time=time, value=np.zeros(n), model=np.zeros(n),
        residuals=np.zeros(n), deleted=np.zeros(n, dtype=bool), point_codes=None,
        partitions=(), subsets=(), offsets_by=None, variation_by=None,
        weights_by=None, warnings=(), history=(), cursor=0, can_undo=False,
        can_redo=False, pristine=False, dirty=True, status="", stop_reason="",
        project_path=None, backend="numpy",
    )
    base.update(kw)
    return SessionView(**base)


class _Recorder:
    """Every op the table emits, epoch split off and validated by the registry."""

    def __init__(self, table: FrequencyTable | SolutionPanel) -> None:
        self.ops: list[tuple[str, dict[str, Any]]] = []
        self.groups: list[list[tuple[str, dict[str, Any]]]] = []
        self.epochs: list[int | None] = []
        self.messages: list[str] = []
        table.edit_requested.connect(self._edit)
        table.group_requested.connect(self._group)
        table.status_message.connect(self.messages.append)

    def _edit(self, op: str, args: dict[str, Any]) -> None:
        assert "expect_id_epoch" in args
        clean, epoch = split_epoch(args)
        actions.parse(op, clean)  # the shape the session accepts
        self.ops.append((op, clean))
        self.epochs.append(epoch)

    def _group(self, group: tuple[tuple[str, dict[str, Any]], ...]) -> None:
        for _, args in group:
            assert "expect_id_epoch" in args
        clean, epoch = split_group(group)
        for op, args in clean:
            actions.parse(op, args)
        self.groups.append(clean)
        self.epochs.append(epoch)


def _data(n: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 30.0, n))
    y = (0.3 * np.sin(2 * np.pi * 1.3 * t) + 0.1 * np.sin(2 * np.pi * 2.71 * t + 1)
         + rng.normal(0.0, 0.01, t.size))
    return t, y, np.full(t.size, 0.01)


def _result() -> cup.PreWhitenResult:
    session = cup.PreWhitenSession(_data(), backend="numpy")
    session.run()
    return session.finalize()


def _col(key: str) -> int:
    return COLUMN_KEYS.index(key)


def _table(qtbot: QtBot, view: SessionView) -> tuple[SolutionPanel, FrequencyTable]:
    panel = SolutionPanel()
    qtbot.addWidget(panel)
    panel.set_view(view)
    return panel, panel.table


def _headers(table: QtWidgets.QTableWidget) -> list[str]:
    return [
        table.horizontalHeaderItem(i).text() for i in range(table.columnCount())
    ]


def _edit_text(table: FrequencyTable, ident: int, key: str, text: str) -> None:
    """What committing a cell editor does: set the item's text."""
    row = table.row_of(ident)
    assert row is not None
    table.item(row, _col(key)).setText(text)


def _click_check(table: FrequencyTable, ident: int, key: str) -> None:
    row = table.row_of(ident)
    assert row is not None
    item = table.item(row, _col(key))
    checked = item.checkState() == Qt.CheckState.Checked
    item.setCheckState(Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked)


# -- parsing --


@pytest.mark.parametrize(
    ("text", "frequency"),
    [
        ("12.34", 12.34),
        ("12.34 c/d", 12.34),
        ("0.081 d", 1 / 0.081),
        ("0.081d", 1 / 0.081),
        ("2 h", 12.0),
        ("100 µHz", 8.64),
        ("1e1", 10.0),
    ],
)
def test_entry_reads_frequencies_and_periods(text: str, frequency: float) -> None:
    entry = parse_entry(text)
    assert entry.kind == "frequency"
    assert entry.frequency == pytest.approx(frequency)


@pytest.mark.parametrize(
    ("text", "expression"),
    [("=F1+F2", "F1 + F2"), ("2f1", "2F1"), ("F3 = f1 - F2", "F1 - F2")],
)
def test_entry_reads_ties(text: str, expression: str) -> None:
    entry = parse_entry(text)
    assert entry.kind == "tie" and entry.expression == expression
    rows = [_row(1, 3.0), _row(2, 1.0)]
    if expression == "F1 + F2":
        assert tie_frequency(entry, rows) == pytest.approx(4.0)


@pytest.mark.parametrize("text", ["", "abc", "-3", "0", "12 parsecs", "=F1+", "f1-f1"])
def test_entry_rejects_with_a_message(text: str) -> None:
    with pytest.raises(ValueError, match=r"\w"):
        parse_entry(text)


def test_tie_frequency_needs_existing_positive_parents() -> None:
    rows = [_row(1, 3.0), _row(2, 1.0)]
    with pytest.raises(ValueError, match="F9"):
        tie_frequency(parse_entry("F1+F9"), rows)
    with pytest.raises(ValueError, match="not a frequency"):
        tie_frequency(parse_entry("F2-F1"), rows)


# -- layout --


def test_editable_mode_keeps_the_read_only_headers_and_adds_the_edit_columns(
    qtbot: QtBot,
) -> None:
    panel, table = _table(qtbot, _view([_row(1), _row(2)]))
    assert panel.editable
    headers = _headers(table)
    assert headers[0] == "ID"
    assert set(_READ_ONLY_HEADERS) <= set(headers)
    for extra in ("fix f", "fix A", "fix ph", "tie", "per subset", "note"):
        assert extra in headers
    # Most useful first (UX-18): S/N and the combination before the phase.
    assert headers.index("S/N") < headers.index("phase")
    assert headers.index("combination") < headers.index("phase")
    assert [table.item(r, 0).text() for r in range(2)] == ["F1", "F2"]
    assert all(table.horizontalHeaderItem(i).toolTip() for i in range(len(COLUMNS)))
    assert table.accessibleName() == "Frequency solution"


def test_read_only_mode_is_unchanged_after_a_session(qtbot: QtBot) -> None:
    panel, table = _table(qtbot, _view([_row(1), _row(2)]))
    result = _result()
    panel.set_solution(result)
    assert not panel.editable
    assert _headers(table) == _READ_ONLY_HEADERS
    assert table.item(0, 0).text() == "F1"
    assert table.item(0, 0).data(Qt.ItemDataRole.CheckStateRole) is None
    with qtbot.waitSignal(panel.component_selected, timeout=2000) as blocker:
        table.selectRow(1)
    assert blocker.args[0].rank == 2
    assert not table.editTriggers()


def test_empty_session_shows_the_hint(qtbot: QtBot) -> None:
    panel, table = _table(qtbot, _view([]))
    assert panel._hint.isVisibleTo(panel)
    assert panel._hint.text() == EMPTY_HINT
    assert table.rowCount() == 0
    panel.set_view(_view([_row(1)]))
    assert not panel._hint.isVisibleTo(panel)


# -- check boxes --


def test_use_checkbox_sends_set_active(qtbot: QtBot) -> None:
    _, table = _table(qtbot, _view([_row(1), _row(2)], epoch=11))
    rec = _Recorder(table)
    _click_check(table, 2, "label")
    assert rec.ops == [("set_active", {"ids": [2], "active": False})]
    assert rec.epochs == [11]


def test_fix_checkboxes_send_fix(qtbot: QtBot) -> None:
    _, table = _table(qtbot, _view([_row(1), _row(2, fix_phase=True)]))
    rec = _Recorder(table)
    _click_check(table, 1, "fix_frequency")
    _click_check(table, 1, "fix_amplitude")
    _click_check(table, 2, "fix_phase")
    assert rec.ops == [
        ("fix", {"ids": [1], "frequency": True}),
        ("fix", {"ids": [1], "amplitude": True}),
        ("fix", {"ids": [2], "phase": False}),
    ]


def test_tied_row_reads_tied_and_cannot_fix_its_frequency(qtbot: QtBot) -> None:
    rows = [_row(1), _row(2), _row(3, 3.3, tie="F1 + F2", flags=frozenset({"tied"}))]
    _, table = _table(qtbot, _view(rows))
    r = table.row_of(3)
    fix_f = table.item(r, _col("fix_frequency"))
    assert fix_f.text() == "tied"
    assert not fix_f.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert fix_f.data(Qt.ItemDataRole.CheckStateRole) is None
    assert table.item(r, _col("frequency")).font().bold()
    assert table.item(r, _col("frequency_error")).text() == "tied"
    assert table.item(r, _col("tie")).text() == "F1 + F2"


# -- text edits --


def test_frequency_cell_takes_a_number_a_period_or_a_tie(qtbot: QtBot) -> None:
    _, table = _table(qtbot, _view([_row(1, 3.0), _row(2, 1.0), _row(3, 4.1)]))
    rec = _Recorder(table)
    _edit_text(table, 3, "frequency", "4.05")
    _edit_text(table, 3, "frequency", "=F1+F2")
    _edit_text(table, 3, "frequency", "0.25 d")
    assert rec.ops == [
        ("set_values", {"id": 3, "frequency": 4.05}),
        ("tie", {"id": 3, "expression": "F1 + F2"}),
        ("set_values", {"id": 3, "frequency": 4.0}),
    ]
    # The cell shows the view's value until the next view arrives.
    assert table.item(table.row_of(3), _col("frequency")).text() == "4.1"


def test_invalid_text_reverts_and_reports(qtbot: QtBot) -> None:
    _, table = _table(qtbot, _view([_row(1, 3.0), _row(2, 1.0)]))
    rec = _Recorder(table)
    _edit_text(table, 2, "frequency", "F2+F1")  # a self reference
    _edit_text(table, 2, "tie", "F1 +")
    _edit_text(table, 2, "amplitude", "big")
    assert rec.ops == [] and rec.groups == []
    assert len(rec.messages) == 3 and all(m.startswith("F2:") for m in rec.messages)
    r = table.row_of(2)
    assert table.item(r, _col("frequency")).text() == "1"
    assert table.item(r, _col("tie")).text() == ""
    assert "F1 +" in table.item(r, _col("tie")).toolTip()  # the parser's message
    assert table.item(r, _col("amplitude")).toolTip() == rec.messages[-1]


def test_unchanged_editor_text_sends_nothing(qtbot: QtBot) -> None:
    # A frequency shown with 9 digits must not be written back rounded.
    _, table = _table(qtbot, _view([_row(1, 1.23456789012345)]))
    rec = _Recorder(table)
    item = table.item(0, _col("frequency"))
    item.setText(item.text())
    assert rec.ops == []


def test_signed_amplitude_and_value_edits_hold_the_value(qtbot: QtBot) -> None:
    rows = [_row(1, amplitude=-0.0123, fix_phase=True,
                 flags=frozenset({"fixed_phi", "negative_amplitude"}))]
    _, table = _table(qtbot, _view(rows))
    cell = table.item(0, _col("amplitude"))
    assert cell.text() == "−0.0123"
    assert "phase + π" in cell.toolTip()
    assert table.item(0, 0).text() == "⚠ F1"  # negative_amplitude is a warning
    assert "negative_amplitude" in table.item(0, 0).toolTip()
    rec = _Recorder(table)
    _edit_text(table, 1, "amplitude", "−0.02")
    _edit_text(table, 1, "phase", "1.5")
    assert rec.groups == [
        [("set_values", {"id": 1, "amplitude": -0.02}),
         ("fix", {"ids": [1], "amplitude": True})],
        [("set_values", {"id": 1, "phase": 1.5}), ("fix", {"ids": [1], "phase": True})],
    ]


def test_tie_note_and_per_subset_cells(qtbot: QtBot) -> None:
    rows = [_row(1, 3.0), _row(2, 1.0), _row(3, 4.0, tie="F1 + F2")]
    _, table = _table(qtbot, _view(rows))
    rec = _Recorder(table)
    _edit_text(table, 2, "tie", "=f1 - 2*F1 + f1 + f1")  # merged: 1F1
    _edit_text(table, 3, "tie", "")
    _edit_text(table, 1, "note", "  dominant mode  ")
    _edit_text(table, 1, "variation", "A+φ")
    _edit_text(table, 1, "variation", "B")
    assert rec.ops == [
        ("tie", {"id": 2, "expression": "F1"}),
        ("tie", {"id": 3, "expression": None}),
        ("set_note", {"id": 1, "note": "dominant mode"}),
        ("set_variation", {"ids": [1], "mode": "both"}),
    ]
    assert rec.messages and "per subset" in rec.messages[-1]


def test_variation_editor_is_a_combo(qtbot: QtBot) -> None:
    panel, table = _table(qtbot, _view([_row(1)]))
    panel.show()
    rec = _Recorder(table)
    item = table.item(0, _col("variation"))
    assert item.text() == "–"
    table.editItem(item)
    combo = table.findChild(QtWidgets.QComboBox)
    assert combo is not None
    assert [combo.itemText(i) for i in range(combo.count())] == ["–", "A", "φ", "A+φ"]
    combo.setCurrentIndex(2)
    combo.activated.emit(2)
    assert rec.ops == [("set_variation", {"ids": [1], "mode": "phase"})]


def test_double_click_on_a_combination_ties_it(qtbot: QtBot) -> None:
    rows = [_row(1, 3.0), _row(2, 1.0), _row(3, 4.0001, combination="F3 = F1 + F2")]
    _, table = _table(qtbot, _view(rows))
    rec = _Recorder(table)
    item = table.item(table.row_of(3), _col("combination"))
    assert item.font().italic()  # identified, not tied yet
    table.itemDoubleClicked.emit(item)
    assert rec.groups == [] and rec.ops == [
        ("tie", {"id": 3, "expression": "F1 + F2"})
    ]


# -- commands and multi-row selection --


def _select(table: FrequencyTable, *ids: int) -> None:
    table.select_ids(ids)


def test_row_commands_act_on_the_whole_selection(qtbot: QtBot) -> None:
    rows = [_row(1), _row(2, active=False, rank=None), _row(3, tie="F1 + F2"), _row(4)]
    _, table = _table(qtbot, _view(rows))
    rec = _Recorder(table)
    _select(table, 4, 2)
    assert table.selected_ids() == (4, 2)  # current row first
    assert table.handle_command("session.toggle_use")  # mixed: use all
    _select(table, 1, 3)
    assert table.handle_command("session.fix_frequency")  # the tied row is skipped
    assert table.handle_command("session.fix_all")
    assert table.handle_command("session.improve_selected")
    assert table.handle_command("edit.delete")
    assert rec.ops == [
        ("set_active", {"ids": [4, 2], "active": True}),
        ("fix", {"ids": [1], "frequency": True}),
        ("improve", {"ids": [1, 3]}),
        ("remove", {"ids": [1, 3]}),
    ]
    assert rec.groups == [
        [("fix", {"ids": [1], "frequency": True, "amplitude": True, "phase": True}),
         ("fix", {"ids": [3], "amplitude": True, "phase": True})],
    ]
    table.clearSelection()
    assert not table.handle_command("edit.delete")
    assert "Select" in rec.messages[-1]


def test_table_letters_without_a_registry(qtbot: QtBot) -> None:
    panel, table = _table(qtbot, _view([_row(1), _row(2)]))
    panel.show()
    rec = _Recorder(table)
    _select(table, 1, 2)
    table.setFocus()
    QtTest.QTest.keyClick(table, Qt.Key.Key_Space)
    QtTest.QTest.keyClick(table, Qt.Key.Key_L)
    QtTest.QTest.keyClick(table, Qt.Key.Key_I, Qt.KeyboardModifier.ShiftModifier)
    QtTest.QTest.keyClick(table, Qt.Key.Key_Delete)
    assert [op for op, _ in rec.ops] == ["set_active", "fix", "improve", "remove"]
    assert rec.ops[0][1] == {"ids": [1, 2], "active": False}


def test_registry_letters_route_to_the_table(qtbot: QtBot) -> None:
    window = QtWidgets.QWidget()
    qtbot.addWidget(window)
    layout = QtWidgets.QVBoxLayout(window)
    panel = SolutionPanel()
    other = QtWidgets.QLineEdit()
    layout.addWidget(panel)
    layout.addWidget(other)
    registry = CommandRegistry(window)
    panel.attach_commands(registry)
    registry.set_state(session=True, busy=False)
    panel.set_view(_view([_row(1), _row(2)]))
    window.show()
    window.activateWindow()
    qtbot.waitActive(window)
    table = panel.table
    hosted = {sc.objectName() for sc in registry.shortcuts("table")}
    for cid in ("toggle_use", "fix_frequency", "fix_all", "tie", "improve_selected"):
        assert f"shortcut_table_session_{cid}" in hosted
    assert "shortcut_table_edit_delete" in hosted
    assert registry.conflicts() == ()
    rec = _Recorder(table)
    _select(table, 2)
    table.setFocus()
    qtbot.waitUntil(table.hasFocus, timeout=2000)
    QtTest.QTest.keyClick(table, Qt.Key.Key_Space)
    QtTest.QTest.keyClick(table, Qt.Key.Key_L, Qt.KeyboardModifier.ShiftModifier)
    assert rec.ops == [
        ("set_active", {"ids": [2], "active": False}),
        ("fix", {"ids": [2], "frequency": True, "amplitude": True, "phase": True}),
    ]
    # Delete belongs to whichever pane has the focus: not the table here.
    other.setFocus()
    qtbot.waitUntil(other.hasFocus, timeout=2000)
    registry.action("edit.delete").trigger()
    assert len(rec.ops) == 2
    # A menu trigger of a row command acts on the selection wherever the focus is.
    registry.action("session.improve_selected").trigger()
    assert rec.ops[-1] == ("improve", {"ids": [2]})
    table.setFocus()
    qtbot.waitUntil(table.hasFocus, timeout=2000)
    QtTest.QTest.keyClick(table, Qt.Key.Key_Delete)
    assert rec.ops[-1] == ("remove", {"ids": [2]})
    # Busy: the letters are disabled and the table does not act on its own.
    registry.set_state(busy=True)
    QtTest.QTest.keyClick(table, Qt.Key.Key_Space)
    assert len(rec.ops) == 4


def test_tie_letter_opens_the_tie_editor(qtbot: QtBot) -> None:
    panel, table = _table(qtbot, _view([_row(1), _row(2)]))
    panel.show()
    _select(table, 2)
    assert table.handle_command("session.tie")
    assert table.state() == QtWidgets.QAbstractItemView.State.EditingState
    assert table.currentColumn() == _col("tie")
    assert table.id_at(table.currentRow()) == 2


def test_context_menu_offers_the_row_actions(qtbot: QtBot) -> None:
    panel, table = _table(qtbot, _view([_row(1), _row(2, tie="F1 + F1")]))
    rec = _Recorder(table)
    _select(table, 1, 2)
    menu = panel.context_menu()
    texts = [a.text().split("\t")[0] for a in menu.actions() if a.text()]
    for expected in ("Improve selected", "Fix", "Tie…", "Untie", "Per subset", "Use",
                     "Don't use", "Delete", "Copy selected rows", "Export CSV…"):
        assert expected in texts
    menu_by_text = {a.text().split("\t")[0]: a for a in menu.actions()}
    assert not menu_by_text["Use"].isEnabled()  # both are in use already
    menu_by_text["Don't use"].trigger()
    menu_by_text["Untie"].trigger()
    per_subset = menu_by_text["Per subset"].menu()
    next(a for a in per_subset.actions() if a.text() == "A").trigger()
    assert rec.ops == [
        ("set_active", {"ids": [1, 2], "active": False}),
        ("tie", {"id": 2, "expression": None}),
        ("set_variation", {"ids": [1, 2], "mode": "amplitude"}),
    ]
    assert menu.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)


# -- addressing, in-place updates, selection --


def test_every_emitted_args_mapping_carries_the_epoch(qtbot: QtBot) -> None:
    raw: list[Any] = []
    rows = [_row(1, 3.0), _row(2, 1.0), _row(3, 4.0, combination="F3 = F1 + F2")]
    panel, table = _table(qtbot, _view(rows, epoch=42))
    panel.edit_requested.connect(lambda op, args: raw.append(args))
    panel.group_requested.connect(lambda group: raw.extend(a for _, a in group))
    _click_check(table, 1, "label")
    _click_check(table, 1, "fix_amplitude")
    _edit_text(table, 2, "frequency", "1.01")
    _edit_text(table, 2, "amplitude", "0.5")
    _edit_text(table, 2, "note", "x")
    table.itemDoubleClicked.emit(table.item(table.row_of(3), _col("combination")))
    _select(table, 1, 2)
    for command in ("session.toggle_use", "session.fix_all", "edit.delete",
                    "session.improve_selected"):
        table.handle_command(command)
    assert len(raw) >= 10
    assert all(args["expect_id_epoch"] == 42 for args in raw)


def test_sorting_is_display_only(qtbot: QtBot) -> None:
    _, table = _table(qtbot, _view([_row(1, 1.0), _row(2, 5.0), _row(3, 3.0)]))
    rec = _Recorder(table)
    table.sortItems(_col("frequency"), Qt.SortOrder.DescendingOrder)
    assert [table.id_at(r) for r in range(3)] == [2, 3, 1]
    table.item(0, _col("note")).setText("tallest")
    _click_check(table, 1, "fix_frequency")
    assert rec.ops == [
        ("set_note", {"id": 2, "note": "tallest"}),
        ("fix", {"ids": [1], "frequency": True}),
    ]
    # A view with the same ids keeps the user's sort and updates in place.
    table.show_view(_view([_row(1, 1.0), _row(2, 5.0), _row(3, 6.0)]))
    assert [table.id_at(r) for r in range(3)] == [3, 2, 1]


def test_in_place_updates_keep_items_selection_and_scroll(qtbot: QtBot) -> None:
    rows = [_row(i, 0.5 * i) for i in range(1, 41)]
    panel, table = _table(qtbot, _view(rows))
    panel.resize(500, 300)
    panel.show()
    _select(table, 30, 31)
    table.scrollToItem(table.item(table.row_of(30), 0))
    bar = table.verticalScrollBar()
    scroll = bar.value()
    assert scroll > 0
    item = table.item(table.row_of(30), _col("amplitude"))
    emitted: list[object] = []
    table.components_selected.connect(emitted.append)
    changed = list(rows)
    changed[29] = replace(rows[29], amplitude=0.75, fix_amplitude=True)
    table.show_view(_view(changed, serial=2))
    assert table.item(table.row_of(30), _col("amplitude")) is item  # in place
    assert item.text() == "0.75"
    assert table.selected_ids() == (30, 31)
    assert bar.value() == scroll
    assert emitted == []
    # A structural change (F31 removed) refills but keeps what survives.
    table.show_view(_view([r for r in changed if r.id != 31], serial=3))
    assert table.rowCount() == 39
    assert table.selected_ids() == (30,)


def test_check_box_click_survives_until_the_next_view(qtbot: QtBot) -> None:
    view = _view([_row(1), _row(2)])
    _, table = _table(qtbot, view)
    _click_check(table, 1, "label")
    item = table.item(table.row_of(1), 0)
    assert item.checkState() == Qt.CheckState.Unchecked  # as clicked
    table.resync()  # the edit was refused: back to the view
    assert item.checkState() == Qt.CheckState.Checked


def test_user_selection_emits_ids(qtbot: QtBot) -> None:
    _, table = _table(qtbot, _view([_row(1), _row(2), _row(3)]))
    with qtbot.waitSignal(table.components_selected, timeout=2000) as blocker:
        table.selectRow(table.row_of(3))
    assert blocker.args == [(3,)]


def test_controller_echo_selects_without_emitting(qtbot: QtBot) -> None:
    panel, table = _table(qtbot, _view([_row(1, 2.0), _row(2, 4.0)]))
    emitted: list[object] = []
    panel.components_selected.connect(emitted.append)
    panel.select_components([2])
    assert panel.selected_components() == (2,)
    panel.select_period(0.5)  # F1's period
    assert panel.selected_components() == (1,)
    assert emitted == []


# -- display --


def test_snr_failing_a_stopping_rule_is_marked_beyond_colour(qtbot: QtBot) -> None:
    rows = [_row(1), _row(2, snr=3.1, passes=False),
            _row(3, amplitude_rms=0.2, snr=7.0)]
    panel, table = _table(qtbot, _view(rows))
    failing = table.item(table.row_of(2), _col("snr"))
    assert failing.text().endswith("!")
    assert "fails" in failing.toolTip()
    normal = table.item(table.row_of(1), _col("snr"))
    assert failing.foreground().color() != normal.foreground().color()
    assert "RMS" in table.item(table.row_of(3), _col("snr")).toolTip()
    from cuperiod.gui.theme import palette

    panel.apply_theme(palette("light"))
    assert table.item(table.row_of(2), _col("snr")).foreground().color().name() == (
        "#b45309"
    )


def test_missing_errors_say_how_to_get_them(qtbot: QtBot) -> None:
    rows = [_row(1, frequency_error=math.nan, amplitude_error=math.nan,
                 phase_error=math.nan, fix_phase=True)]
    _, table = _table(qtbot, _view(rows))
    err = table.item(0, _col("frequency_error"))
    # The explicit Calculate (and Improve) solve with the covariance; only the light
    # re-calculation after an edit leaves the errors out.
    assert err.text() == "–" and "Calculate (F8)" in err.toolTip()
    assert "Improve" in err.toolTip()
    assert table.item(0, _col("phase_error")).text() == "fixed"


def test_inactive_rows_are_unchecked_not_only_grey(qtbot: QtBot) -> None:
    _, table = _table(qtbot, _view([_row(1), _row(2, active=False, rank=None)]))
    item = table.item(table.row_of(2), 0)
    assert item.checkState() == Qt.CheckState.Unchecked
    assert "not in use" in item.toolTip()


# -- busy, copy, export --


def test_busy_table_takes_no_edits(qtbot: QtBot) -> None:
    panel, table = _table(qtbot, _view([_row(1)]))
    rec = _Recorder(table)
    panel.set_busy(True)
    assert not table.editTriggers()
    _click_check(table, 1, "label")
    _select(table, 1)
    assert not table.handle_command("edit.delete")
    assert rec.ops == []
    assert table.item(0, 0).checkState() == Qt.CheckState.Checked
    assert "Busy" in rec.messages[0]
    panel.set_busy(False)
    assert table.editTriggers()


def test_copy_and_export_in_a_session(qtbot: QtBot, tmp_path: Any) -> None:
    rows = [_row(1, note="a, b"), _row(2, active=False, rank=None, tie="F1 + F1")]
    panel, table = _table(qtbot, _view(rows))
    _select(table, 1)
    panel._copy_selected()
    lines = QtWidgets.QApplication.clipboard().text().splitlines()
    assert lines[0].split("\t") == [c.header for c in COLUMNS]
    first = lines[1].split("\t")
    assert first[0] == "F1" and first[_col("fix_frequency")] == "0"
    out = tmp_path / "session.csv"
    panel.export_csv(str(out))
    with open(out, encoding="utf-8") as fh:
        written = list(csv.DictReader(fh))
    assert len(written) == 2  # inactive rows included
    assert set(SESSION_CSV_FIELDS) <= set(written[0])
    assert written[1]["active"] == "False" and written[1]["tie"] == "F1 + F1"
    assert written[0]["note"] == "a, b"


# -- a real session --


def test_real_session_round_trip(qtbot: QtBot) -> None:
    session = cup.PreWhitenSession(_data(), backend="numpy")
    session.step()
    session.step()
    view = build_session_view(session, key=SessionKey("real"), id_epoch=0)
    panel, table = _table(qtbot, view)
    assert table.rowCount() == len(view.rows) == 2
    applied: list[str] = []

    def apply(op: str, args: dict[str, Any]) -> None:
        clean, _ = split_epoch(args)
        report = session.do(op, **clean)
        applied.append(report.kind)
        panel.set_view(build_session_view(session, key=SessionKey("real"),
                                          id_epoch=0))

    def apply_group(group: Any) -> None:
        clean, _ = split_group(group)
        with session.group():
            for op, args in clean:
                applied.append(session.do(op, **args).kind)
        panel.set_view(build_session_view(session, key=SessionKey("real"),
                                          id_epoch=0))

    panel.edit_requested.connect(apply)
    panel.group_requested.connect(apply_group)
    item = table.item(table.row_of(1), 0)
    _click_check(table, 2, "fix_frequency")
    _edit_text(table, 1, "note", "main mode")
    _edit_text(table, 2, "phase", "0.3")
    assert "refused" not in applied, applied
    assert session.state.component(2).fix_frequency
    assert session.state.component(2).fix_phase
    assert session.state.component(1).note == "main mode"
    assert table.item(table.row_of(1), 0) is item  # updated in place throughout
    assert table.item(table.row_of(1), _col("note")).text() == "main mode"
    assert table.item(table.row_of(2), _col("fix_phase")).checkState() == (
        Qt.CheckState.Checked
    )


def test_units_line_names_what_the_headers_do_not(qtbot: QtBot) -> None:
    panel, _ = _table(qtbot, _view([_row(1)]))
    assert panel._units.text() == "frequency c/d · period d · phase rad"
    panel.set_amplitude_unit("mag")
    assert panel._units.text().endswith("· amplitude mag")
    panel.set_amplitude_unit("<b>flux</b>")  # plain text: never markup
    assert panel._units.textFormat() == Qt.TextFormat.PlainText
    panel.set_amplitude_unit(None)
    assert "amplitude" not in panel._units.text()
