"""The main window's menu bar, mnemonics, Help, recent files and command states (G8b).

UX-07 / A11Y-06 / GUI-B-12: every command is in the menu bar with an Alt mnemonic;
UX-08: closed panes come back from View; the F1 sheet, documentation and About live
under Help; File ▸ Open recent remembers files.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.lightcurve import LightCurve
from cuperiod.gui.commands import COMMAND_IDS, COMMANDS
from cuperiod.gui.dialogs import FakeDialogProvider
from cuperiod.gui.loaders import ImportChoice, ReadOptions
from cuperiod.gui.main_window import (
    DOCS_URL,
    MainWindow,
    assign_mnemonics,
    import_choice_from_json,
    import_choice_to_json,
)
from cuperiod.gui.qt import QtCore, QtGui, QtWidgets

MENU_TITLES = ["File", "Edit", "Pre-whitening", "Data", "View", "Help"]


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path: Path) -> None:
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        str(tmp_path),
    )


def _window(qtbot: QtBot, provider: FakeDialogProvider | None = None) -> MainWindow:
    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    window = MainWindow(app)
    qtbot.addWidget(window)
    if provider is not None:
        window.set_dialog_provider(provider)
    window.show()
    return window


def _curve() -> LightCurve:
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 30.0, 200))
    y = 12.0 + 0.2 * np.sin(2 * np.pi * 1.7 * t) + rng.normal(0.0, 0.01, 200)
    return LightCurve.from_arrays(t, y, np.full(200, 0.01))


def _menus(window: MainWindow) -> dict[str, QtWidgets.QMenu]:
    # Not QAction.menu(): PySide6 hands its result to Python ownership, and the
    # first discarded wrapper deletes the C++ menu.
    bar = window.menuBar()
    assert bar is not None
    return {
        m.title().replace("&", ""): m
        for m in bar.findChildren(
            QtWidgets.QMenu, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly
        )
    }


def _mnemonic(text: str) -> str | None:
    match = re.search(r"&([^&\s])", text)
    return match.group(1).lower() if match else None


# -- the menu bar --------------------------------------------------------------------
def test_menu_bar_has_the_six_menus_with_mnemonics(qtbot: QtBot) -> None:
    window = _window(qtbot)
    bar = window.menuBar()
    assert bar is not None
    titles = [a.text() for a in bar.actions()]
    assert [t.replace("&", "") for t in titles] == MENU_TITLES
    letters = [_mnemonic(t) for t in titles]
    assert None not in letters and len(set(letters)) == len(letters)


def test_every_menu_item_has_a_unique_mnemonic(qtbot: QtBot) -> None:
    window = _window(qtbot)
    for title, menu in _menus(window).items():
        items = [a for a in menu.actions() if not a.isSeparator() and a.text()]
        letters = [_mnemonic(a.text()) for a in items]
        assert None not in letters, (title, [a.text() for a in items])
        assert len(set(letters)) == len(letters), (title, letters)


def test_every_registry_command_is_in_a_menu(qtbot: QtBot) -> None:
    window = _window(qtbot)
    in_menus = {
        a.objectName() for menu in _menus(window).values() for a in menu.actions()
    }
    for command in COMMANDS:
        assert command.object_name in in_menus, command.id


def test_window_commands_are_keyboard_reachable(qtbot: QtBot) -> None:
    window = _window(qtbot)
    file_menu = _menus(window)["File"]
    names = {a.objectName(): a for a in file_menu.actions()}
    assert not names["action_file_open_folder"].shortcut().isEmpty()
    assert not names["action_file_quit"].shortcut().isEmpty()
    titles = [a.text().replace("&", "") for a in file_menu.actions()]
    assert {"Open recent", "Load demo", "Unsaved sessions"} <= set(titles)
    view = _menus(window)["View"]
    theme = [a for a in view.actions() if a.objectName() == "action_view_theme"]
    assert theme and not theme[0].shortcut().isEmpty()
    # Every interactive thing of the toolbar is also a menu item with a key.
    assert window._folder_action in window.actions()


def test_view_lists_every_pane_and_brings_a_closed_one_back(qtbot: QtBot) -> None:
    window = _window(qtbot)
    panes = window.findChild(QtWidgets.QMenu, "menu_panes")
    assert isinstance(panes, QtWidgets.QMenu)
    docks = [*window._result_docks(), window._sources_dock]
    toggles = panes.actions()
    for dock in docks:
        assert dock.toggleViewAction() in toggles
    window._peaks_dock.close()
    assert not window._peaks_dock.isVisible()
    window._peaks_dock.toggleViewAction().trigger()
    assert window._peaks_dock.isVisible()


def test_the_toolbar_cannot_be_hidden_for_good(qtbot: QtBot) -> None:
    window = _window(qtbot)
    assert not window._toolbar.toggleViewAction().isVisible()
    window._toolbar.setVisible(False)
    window.close()
    again = _window(qtbot)
    assert again._toolbar.isVisible()


def test_dock_title_buttons_are_24_px_and_named(qtbot: QtBot) -> None:
    window = _window(qtbot)
    for bar in window._title_bars:
        for button in bar.buttons:
            assert button.width() >= 24 and button.height() >= 24
            assert button.accessibleName()
            assert button.toolTip()


def test_analysis_radio_items_switch_the_analysis(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._analysis_actions["prewhiten"].trigger()
    assert window._controls.analysis == "prewhiten"
    assert window._solution_dock.isVisible()
    window._analysis_actions["periodogram"].trigger()
    assert window._controls.analysis == "periodogram"
    assert window._analysis_actions["periodogram"].isChecked()


# -- Help ----------------------------------------------------------------------------
def test_help_menu_offers_the_sheet_whats_this_docs_and_about(qtbot: QtBot) -> None:
    provider = FakeDialogProvider()
    window = _window(qtbot, provider)
    help_menu = _menus(window)["Help"]
    names = [a.objectName() for a in help_menu.actions() if not a.isSeparator()]
    assert "action_view_shortcuts" in names
    assert "action_help_documentation" in names and "action_help_about" in names
    assert window._whats_this_action in help_menu.actions()
    shortcuts = window.commands.action("view.shortcuts")
    assert QtGui.QKeySequence("F1") in shortcuts.shortcuts()
    shortcuts.trigger()
    kind, (title, html) = provider.calls[-1]
    assert kind == "show_html" and title == "Keyboard shortcuts"
    assert "Step" in html and "F7" in html
    window._about_action.trigger()
    kind, (title, html) = provider.calls[-1]
    assert kind == "show_html" and "cuPeriod" in title and DOCS_URL in html
    assert DOCS_URL in window._docs_action.statusTip()


def test_the_f1_sheet_lists_every_shortcut_of_the_window(qtbot: QtBot) -> None:
    """F1 listed only the registry's commands: Open folder, Quit, the theme, zoom,
    What's This?, the raw light curve, Compute periodogram and the plots' own keys
    were missing from a sheet that says it lists every shortcut."""
    import html as html_module

    from cuperiod.gui.widgets.rawlc_view import LIGHT_CURVE_KEYS
    from cuperiod.gui.widgets.spectrum_view import SPECTRUM_KEYS

    provider = FakeDialogProvider()
    window = _window(qtbot, provider)
    window.commands.action("view.shortcuts").trigger()
    kind, (_title, sheet) = provider.calls[-1]
    assert kind == "show_html"
    native = QtGui.QKeySequence.SequenceFormat.NativeText

    def listed(text: str, keys: str) -> bool:
        cells = f"<td>{html_module.escape(text)}</td><td><b>{html_module.escape(keys)}"
        return cells in sheet

    registry = {id(window.commands.action(c)) for c in COMMAND_IDS}
    own = [
        a
        for a in window.findChildren(QtGui.QAction)
        if a.shortcuts() and id(a) not in registry
    ]
    assert len(own) >= 9
    for action in own:
        text = action.text().replace("&", "")
        keys = ", ".join(s.toString(native) for s in action.shortcuts())
        assert listed(text, keys), (text, keys)
    for key in ("Ctrl+Q", "Ctrl+T", "Ctrl+Shift+F", "Shift+F1", "Ctrl+0", "Ctrl+R"):
        assert QtGui.QKeySequence(key).toString(native) in sheet, key
    # Every key a pane binds itself (the context letters, a table's copy) is there.
    for shortcut in window.findChildren(QtGui.QShortcut):
        assert html_module.escape(shortcut.key().toString(native)) in sheet
    # And the keys a focused plot handles on its own, in sections of their own.
    assert "Spectrum (focused)" in sheet and "Light curve (focused)" in sheet
    for what, _keys in (*SPECTRUM_KEYS, *LIGHT_CURVE_KEYS):
        assert html_module.escape(what) in sheet
    window.controller.shutdown()


# -- command states ------------------------------------------------------------------
def test_no_key_is_bound_twice_in_any_state(qtbot: QtBot) -> None:
    window = _window(qtbot)
    registry = window.commands

    def check() -> None:
        assert registry.conflicts() == ()
        stop = registry.action("session.stop").isEnabled()
        clear = registry.action("data.clear_selection").isEnabled()
        assert not (stop and clear)
        # The window's Compute and the registry's Auto share F5 / Ctrl+Return.
        assert not (
            window._compute_action.isEnabled()
            and registry.action("session.auto").isEnabled()
        )

    check()
    window.controller.set_light_curve(_curve(), "star")
    check()
    assert window._compute_action.isEnabled()
    window._controls.set_analysis("prewhiten")
    check()
    assert registry.action("session.auto").isEnabled()
    assert not window._compute_action.isEnabled()
    window._on_busy_changed(True)
    check()
    assert registry.action("session.stop").isEnabled()
    assert not registry.action("session.step").isEnabled()
    window._on_busy_changed(False)
    check()
    window.controller.shutdown()


def test_undo_and_redo_follow_the_history(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window.controller.set_light_curve(_curve(), "star")
    window._controls.set_analysis("prewhiten")
    registry = window.commands
    # No session view yet: nothing to undo.
    assert not registry.action("edit.undo").isEnabled()
    assert not registry.action("edit.redo").isEnabled()
    window.controller.shutdown()


def test_esc_stops_a_running_periodogram(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    calls: list[bool] = []
    monkeypatch.setattr(window.controller, "cancel_compute", lambda: calls.append(True))
    window._on_busy_changed(True)
    assert window._controls._stop_btn.isEnabled()
    window.commands.action("session.stop").trigger()
    window._controls._stop_btn.click()
    assert calls == [True, True]
    window._on_busy_changed(False)
    assert not window._controls._stop_btn.isEnabled()


# -- recent files --------------------------------------------------------------------
def _write_csv(path: Path) -> None:
    rng = np.random.default_rng(1)
    time = np.sort(rng.uniform(0.0, 40.0, 100))
    mag = 14.0 + 0.3 * np.sin(2 * np.pi * time / 3.0)
    lines = ["time,mag,mag_err"]
    lines += [f"{t:.6f},{m:.6f},0.02" for t, m in zip(time, mag, strict=True)]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_open_recent_lists_and_reopens_files(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("cuperiod.gui.main_window.preview_file", lambda _path: None)
    window = _window(qtbot)
    csv = tmp_path / "star.csv"
    _write_csv(csv)
    window._load_file(csv)
    assert window._recent()[0] == str(csv)
    window._recent_menu.aboutToShow.emit()
    entries = [a for a in window._recent_menu.actions() if not a.isSeparator()]
    assert entries[0].text().replace("&", "").endswith("star.csv")
    assert entries[0].statusTip() == str(csv)
    loads = window._load_count
    entries[0].trigger()
    assert window._load_count == loads + 1
    # A file that moved away is dropped from the list, with a banner saying so.
    csv.unlink()
    window._open_recent(str(csv))
    assert window._banner_container.isVisible()
    assert str(csv) not in window._recent()


def test_recent_list_is_capped_and_most_recent_first(qtbot: QtBot) -> None:
    window = _window(qtbot)
    for i in range(12):
        window._add_recent(f"C:/data/f{i}.csv")
    recent = window._recent()
    assert len(recent) == 8
    assert recent[0].endswith("f11.csv")


# -- helpers -------------------------------------------------------------------------
def test_assign_mnemonics_prefers_initials_and_keeps_existing(qtbot: QtBot) -> None:
    menu = QtWidgets.QMenu()
    qtbot.addWidget(menu)
    for text in ("Open", "&Save", "Open folder", "Quit", "Options"):
        menu.addAction(text)
    assign_mnemonics(menu)
    texts = [a.text() for a in menu.actions()]
    assert texts[0] == "&Open"
    assert texts[1] == "&Save"
    letters = [_mnemonic(t) for t in texts]
    assert len(set(letters)) == len(letters)


def test_import_choice_round_trips_through_settings_text() -> None:
    choice = ImportChoice(
        columns=ColumnMap(time="HJD", value="V", error="e_V"),
        domain=Domain.MAGNITUDE,
        read=ReadOptions(delimiter=";", header=False),
    )
    again = import_choice_from_json(import_choice_to_json(choice))
    assert again == choice
    assert import_choice_from_json("not json") is None
    assert import_choice_from_json(None) is None
    assert import_choice_from_json('{"columns": 5}') is None


def test_no_pane_mnemonic_takes_a_menu_bar_letter(qtbot: QtBot) -> None:
    # Alt+F must open File: a label or button "Add &frequency" in a visible pane made
    # the key ambiguous, and Qt then fires neither.
    window = _window(qtbot)
    window._controls.set_analysis("prewhiten")
    bar = window.menuBar()
    assert bar is not None
    taken = {_mnemonic(a.text()) for a in bar.actions()}
    clashes = []
    for widget in window.findChildren(QtWidgets.QWidget):
        if isinstance(widget, QtWidgets.QMenu | QtWidgets.QMenuBar):
            continue
        if isinstance(widget, QtWidgets.QLabel | QtWidgets.QAbstractButton):
            letter = _mnemonic(widget.text())
            if letter in taken:
                clashes.append(widget.text())
    assert clashes == []
