"""The command registry: action metadata, menus, the shortcut invariant, letters."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pytestqt.qtbot import QtBot

from cuperiod.gui.commands import (
    COMMAND_IDS,
    COMMANDS,
    MENUS,
    CommandRegistry,
    command,
    shortcut_sheet,
    shortcut_sheet_html,
)
from cuperiod.gui.dialogs import DialogProvider, FakeDialogProvider, QtDialogProvider
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets


class _Window(QtWidgets.QMainWindow):
    """A window with the three hosts and an Options-like form outside them."""

    def __init__(self) -> None:
        super().__init__()
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        self.spectrum = QtWidgets.QWidget()
        self.spectrum.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.table = QtWidgets.QTableWidget(3, 2)
        self.lightcurve = QtWidgets.QWidget()
        self.lightcurve.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        form = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout(form)
        self.line = QtWidgets.QLineEdit()
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["Lomb", "Scargle", "Ice", "Cat"])
        self.check = QtWidgets.QCheckBox("Clip")
        form_layout.addRow("Text", self.line)
        form_layout.addRow("Combo", self.combo)
        form_layout.addRow("Check", self.check)
        for w in (self.spectrum, self.table, self.lightcurve, form):
            layout.addWidget(w)
        self.setCentralWidget(central)
        self.registry = CommandRegistry(self)
        self.registry.attach_context("spectrum", self.spectrum)
        self.registry.attach_context("table", self.table)
        self.registry.attach_context("lightcurve", self.lightcurve)


@pytest.fixture
def window(qtbot: QtBot) -> Iterator[_Window]:
    win = _Window()
    qtbot.addWidget(win)
    win.resize(800, 400)
    win.show()
    qtbot.waitExposed(win)
    win.activateWindow()
    QtWidgets.QApplication.setActiveWindow(win)
    yield win


def _fired(window: _Window) -> list[str]:
    seen: list[str] = []
    window.registry.triggered.connect(seen.append)
    return seen


def test_every_command_is_described() -> None:
    assert len(COMMAND_IDS) == len(COMMANDS)  # ids are unique
    for c in COMMANDS:
        assert c.text and c.tip and c.menu in MENUS
        assert c.accessible_name and "&" not in c.text
    assert command("session.step").keys == ("F7",)
    with pytest.raises(KeyError):
        command("nope")
    keys = {k for c in COMMANDS for k in c.keys}
    keys |= {k for c in COMMANDS for _, k in c.context}
    assert "F10" not in keys


def test_actions_carry_metadata(window: _Window) -> None:
    registry = window.registry
    names = set()
    for c in COMMANDS:
        action = registry.action(c.id)
        assert action.text() == c.text
        assert action.statusTip() == c.tip
        assert action.property("accessibleName") == c.accessible_name
        assert action.objectName() == c.object_name
        names.add(action.objectName())
        assert action.shortcutContext() == Qt.ShortcutContext.WindowShortcut
        if c.keys:
            first = QtGui.QKeySequence(c.keys[0]).toString(
                QtGui.QKeySequence.SequenceFormat.NativeText
            )
            assert first in action.toolTip() and action.toolTip().endswith(")")
        assert action in window.actions()
    assert len(names) == len(COMMANDS)
    step = registry.action("session.step")
    assert step.toolTip().startswith("Step  (F7, S (spectrum), S (frequency table)")
    button = registry.tool_button("session.step", window)
    assert button.accessibleName() == "Step" and button.defaultAction() is step


@pytest.mark.parametrize("session", [False, True])
@pytest.mark.parametrize("busy", [False, True])
def test_no_duplicate_enabled_binding(
    window: _Window, session: bool, busy: bool
) -> None:
    registry = window.registry
    registry.set_state(session=session, busy=busy)
    assert registry.conflicts() == ()
    # Independently of conflicts(): over QActions and QShortcuts themselves.
    seen: dict[tuple[str, str], str] = {}
    for cid, action in registry.actions().items():
        if action.isEnabled():
            for seq in action.shortcuts():
                slot = ("window", seq.toString())
                assert slot not in seen, (slot, seen.get(slot), cid)
                seen[slot] = cid
    for host in ("spectrum", "table", "lightcurve"):
        for sc in registry.shortcuts(host):
            assert sc.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
            if sc.isEnabled():
                slot = (host, sc.key().toString())
                assert slot not in seen
                assert ("window", sc.key().toString()) not in seen
                seen[slot] = sc.objectName()
    stop = registry.action("session.stop").isEnabled()
    clear = registry.action("data.clear_selection").isEnabled()
    assert not (stop and clear)
    assert stop == busy


def test_state_table(window: _Window) -> None:
    r = window.registry
    r.set_state(session=False, busy=False)
    assert r.action("file.open_curve").isEnabled()
    assert not r.action("session.step").isEnabled()
    assert not r.action("session.stop").isEnabled()
    r.set_state(session=True)
    assert r.action("session.step").isEnabled()
    assert r.action("data.clear_selection").isEnabled()
    r.set_state(busy=True)
    assert not r.action("session.step").isEnabled()
    assert not r.action("edit.undo").isEnabled()
    assert not r.action("file.open_curve").isEnabled()
    assert r.action("session.stop").isEnabled()
    assert r.action("view.focus_spectrum").isEnabled()
    assert r.action("data.tool_lasso").isEnabled()  # view toggles stay usable
    assert not r.action("data.clear_selection").isEnabled()
    # Context shortcuts follow their action.
    letters = [sc for sc in r.shortcuts("table") if sc.key().toString() == "S"]
    assert letters and not letters[0].isEnabled()
    r.set_state(busy=False)
    assert letters[0].isEnabled()


def test_menus_list_every_command(window: _Window) -> None:
    menus = window.registry.populate_menus(window.menuBar())
    assert list(menus) == list(MENUS)
    listed = {a.objectName() for m in menus.values() for a in m.actions()}
    assert {c.object_name for c in COMMANDS} <= listed
    # Re-populating an existing bar reuses the menus.
    again = window.registry.populate_menus(window.menuBar())
    assert again["File"] is menus["File"]


def test_tool_actions_are_exclusive(window: _Window) -> None:
    r = window.registry
    r.set_state(session=True)
    assert r.action("data.tool_pan").isChecked()
    r.action("data.tool_lasso").trigger()
    assert r.action("data.tool_lasso").isChecked()
    assert not r.action("data.tool_pan").isChecked()


def test_sheet_is_generated_from_the_registry() -> None:
    rows = shortcut_sheet()
    texts = {text for _, text, _ in rows}
    for c in COMMANDS:
        assert (c.text in texts) == bool(c.keys or c.context)
    sheet = shortcut_sheet_html()
    assert "F7" in sheet and "<table" in sheet and "Pre-whitening" in sheet


def test_letters_fire_in_their_host_only(window: _Window, qtbot: QtBot) -> None:
    r = window.registry
    r.set_state(session=True)
    fired = _fired(window)
    window.table.setFocus()
    qtbot.waitUntil(window.table.hasFocus)
    qtbot.keyClick(window.table, QtCore.Qt.Key.Key_S)
    shift = QtCore.Qt.KeyboardModifier.ShiftModifier
    qtbot.keyClick(window.table, QtCore.Qt.Key.Key_I, shift)
    assert fired == ["session.step", "session.improve_selected"]
    fired.clear()
    window.lightcurve.setFocus()
    qtbot.waitUntil(window.lightcurve.hasFocus)
    qtbot.keyClick(window.lightcurve, QtCore.Qt.Key.Key_G)
    qtbot.keyClick(window.lightcurve, QtCore.Qt.Key.Key_S)  # not a light-curve letter
    qtbot.keyClick(window.lightcurve, QtCore.Qt.Key.Key_Escape)
    assert fired == ["data.tool_lasso", "data.clear_selection"]
    fired.clear()
    r.set_state(session=False)
    window.table.setFocus()
    qtbot.keyClick(window.table, QtCore.Qt.Key.Key_S)
    assert fired == []


def test_letters_never_fire_from_the_options_form(
    window: _Window, qtbot: QtBot
) -> None:
    window.registry.set_state(session=True)
    fired = _fired(window)
    letters = [
        QtCore.Qt.Key.Key_S, QtCore.Qt.Key.Key_C, QtCore.Qt.Key.Key_I,
        QtCore.Qt.Key.Key_A, QtCore.Qt.Key.Key_L, QtCore.Qt.Key.Key_T,
        QtCore.Qt.Key.Key_U, QtCore.Qt.Key.Key_V, QtCore.Qt.Key.Key_X,
        QtCore.Qt.Key.Key_G, QtCore.Qt.Key.Key_Space, QtCore.Qt.Key.Key_Delete,
        QtCore.Qt.Key.Key_Insert,
    ]
    for widget in (window.line, window.combo, window.check):
        widget.setFocus()
        qtbot.waitUntil(widget.hasFocus)
        for key in letters:
            qtbot.keyClick(widget, key)
    assert fired == []
    assert window.line.text().upper().startswith("SCIALTUVXG")


def test_esc_is_stop_while_busy(window: _Window, qtbot: QtBot) -> None:
    r = window.registry
    r.set_state(session=True, busy=True)
    fired = _fired(window)
    window.lightcurve.setFocus()
    qtbot.waitUntil(window.lightcurve.hasFocus)
    qtbot.keyClick(window.lightcurve, QtCore.Qt.Key.Key_Escape)
    window.line.setFocus()
    qtbot.keyClick(window.line, QtCore.Qt.Key.Key_F7)  # disabled while busy
    assert fired == ["session.stop"]


def test_attach_twice_is_refused(window: _Window) -> None:
    with pytest.raises(ValueError):
        window.registry.attach_context("table", window.table)
    with pytest.raises(ValueError):
        window.registry.attach_context("nowhere", window.table)  # type: ignore[arg-type]


# -- dialogs (non-blocking provider) ----------------------------------------------
def test_fake_dialog_provider_scripts_answers() -> None:
    fake = FakeDialogProvider(["save", "/tmp/x.cupw.json"])
    assert isinstance(fake, DialogProvider)
    got: list[object] = []
    fake.confirm_discard(["star A"], got.append)
    fake.save_file("Save", "*.json", "a.json", got.append)
    fake.ask_text("Tie", "F3 =", "", got.append)  # script exhausted: cancel
    fake.confirm_discard(["star B"], got.append)
    fake.inform("Note", "hello", "warning")
    assert got == ["save", "/tmp/x.cupw.json", None, "cancel"]
    assert [c[0] for c in fake.calls] == [
        "confirm_discard", "save_file", "ask_text", "confirm_discard", "inform"
    ]


def test_qt_dialogs_open_without_exec(window: _Window, qtbot: QtBot) -> None:
    provider = QtDialogProvider(window, native_files=False)
    assert isinstance(provider, DialogProvider)
    got: list[object] = []
    provider.confirm_discard(["star A", "star B"], got.append)  # returns at once
    (box,) = provider.live_dialogs
    assert isinstance(box, QtWidgets.QMessageBox) and box.isVisible()
    assert "star B" in box.informativeText()
    box.button(QtWidgets.QMessageBox.StandardButton.Discard).click()
    qtbot.waitUntil(lambda: got == ["discard"])
    qtbot.waitUntil(lambda: not provider.live_dialogs)

    provider.confirm_discard(["star A"], got.append)
    (box,) = provider.live_dialogs
    box.reject()  # Esc / title-bar close
    qtbot.waitUntil(lambda: got == ["discard", "cancel"])

    provider.ask_float("Add frequency", "f [c/d]", 1.5, got.append)
    (dialog,) = provider.live_dialogs
    assert isinstance(dialog, QtWidgets.QInputDialog)
    dialog.setDoubleValue(2.25)
    dialog.accept()
    qtbot.waitUntil(lambda: got[-1] == 2.25)

    provider.ask_text("Tie", "F3 =", "F1+F2", got.append)
    (dialog,) = provider.live_dialogs
    dialog.reject()
    qtbot.waitUntil(lambda: got[-1] is None and len(got) == 4)

    provider.save_file(
        "Save project", "Projects (*.cupw.json)", "a.cupw.json", got.append
    )
    (files,) = provider.live_dialogs
    assert isinstance(files, QtWidgets.QFileDialog)
    files.reject()
    qtbot.waitUntil(lambda: len(got) == 5 and got[-1] is None)

    provider.show_html("Keyboard shortcuts", window.registry.sheet_html())
    (sheet,) = provider.live_dialogs
    assert sheet.isVisible()
    sheet.reject()
    qtbot.waitUntil(lambda: not provider.live_dialogs)
