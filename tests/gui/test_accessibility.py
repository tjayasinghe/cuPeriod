"""Keyboard and screen-reader access to the whole main window (design 4.9, H2).

Every visible interactive widget has an accessible name (what a screen reader
announces) and a tooltip or status tip; Tab cycles through the main controls of every
state without a trap and without stopping on anything the user cannot see (tabs
behind other tabs, tab bars QMainWindow parks off-screen); every command is in the
menu bar; no key sequence is bound twice in one shortcut context, and no Alt
mnemonic is shared by two widgets the user can reach at once. Colour contrast has its
own tests (test_theme.py).

Sessions run on ``numpy`` on a small curve, the periodogram on ``finufft``: nothing
here creates a CUDA context.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtTest
from pytestqt.qtbot import QtBot

from cuperiod.core.config import PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve
from cuperiod.gui.commands import COMMANDS
from cuperiod.gui.dialogs import FakeDialogProvider
from cuperiod.gui.main_window import MainWindow, window_minimum
from cuperiod.gui.models import SourceItem
from cuperiod.gui.qt import QtCore, QtGui, QtWidgets
from cuperiod.gui.viewmodels import SessionView
from cuperiod.gui.widgets.info_bar import _BADGE_HPAD, _BadgeLabel

Qt = QtCore.Qt
SETTINGS = PreWhitenSettings(max_frequencies=2, samples_per_peak=5)

#: Widgets a keyboard or screen-reader user operates.
INTERACTIVE = (
    QtWidgets.QAbstractButton,
    QtWidgets.QLineEdit,
    QtWidgets.QComboBox,
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QAbstractItemView,
    QtWidgets.QAbstractSlider,
    QtWidgets.QTabBar,
    QtWidgets.QTextEdit,
    QtWidgets.QPlainTextEdit,
)


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path: Path) -> None:
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        str(tmp_path / "settings"),
    )


def _curve(n: int = 300) -> LightCurve:
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0.0, 40.0, n))
    y = (
        12.0
        + 0.3 * np.sin(2 * np.pi * 1.3 * t)
        + 0.1 * np.sin(2 * np.pi * 2.71 * t + 1.0)
        + rng.normal(0.0, 0.01, n)
    )
    return LightCurve.from_arrays(t, y, np.full(n, 0.01))


def _window(qtbot: QtBot) -> MainWindow:
    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    window = MainWindow(app)
    qtbot.addWidget(window)
    window.set_dialog_provider(FakeDialogProvider())
    window.resize(1366, 768)
    window.show()
    window.activateWindow()
    qtbot.waitExposed(window)
    qtbot.wait(30)
    return window


def _settle(qtbot: QtBot, window: MainWindow) -> SessionView:
    ctl = window.controller
    bridge = ctl.session_bridge
    qtbot.waitUntil(
        lambda: bridge.is_idle() and ctl._compute.session_jobs == 0, timeout=60_000
    )
    qtbot.wait(40)
    view = ctl.state.session_view
    assert isinstance(view, SessionView)
    return view


def _load(window: MainWindow) -> None:
    window._load_demo(SourceItem("star", "Test star", _curve))


def _periodogram(qtbot: QtBot, window: MainWindow) -> None:
    _load(window)
    window._controls._backend_combo.setCurrentText("finufft")
    with qtbot.waitSignal(window.controller.periodogram_ready, timeout=60_000):
        window._controls.request_compute()
    qtbot.wait(40)


def _session(qtbot: QtBot, window: MainWindow) -> SessionView:
    _load(window)
    window._analysis_actions["prewhiten"].trigger()
    window._controls._backend_combo.setCurrentText("numpy")
    window._controls.set_prewhiten_settings(SETTINGS)
    with qtbot.waitSignal(window.controller.solution_ready, timeout=60_000):
        window._controls.request_compute()
    return _settle(qtbot, window)


def accessible_name(widget: QtWidgets.QWidget) -> str:
    """What a screen reader announces for ``widget`` (its own name, its text or
    its buddy label, as Qt's accessibility bridge resolves it)."""
    iface = QtGui.QAccessible.queryAccessibleInterface(widget)
    return "" if iface is None else (iface.text(QtGui.QAccessible.Text.Name) or "")


def _on_screen(window: QtWidgets.QWidget, widget: QtWidgets.QWidget) -> bool:
    """Visible and inside the window (QMainWindow parks unused tab bars and the
    tabs behind the current one outside it, still "visible")."""
    if not widget.isVisible():
        return False
    top_left = widget.mapTo(window, QtCore.QPoint(0, 0))
    return window.rect().intersects(QtCore.QRect(top_left, widget.size()))


def _reachable(window: QtWidgets.QWidget) -> Iterator[QtWidgets.QWidget]:
    """Every interactive widget a user can see and reach in ``window``."""
    for widget in window.findChildren(QtWidgets.QWidget):
        if not isinstance(widget, INTERACTIVE) or isinstance(
            widget, QtWidgets.QScrollBar
        ):
            continue
        if not widget.focusPolicy() & Qt.FocusPolicy.TabFocus:
            continue  # header views, combo pop-ups, mouse-only scroll arrows
        if isinstance(widget, QtWidgets.QLineEdit) and isinstance(
            widget.parent(), QtWidgets.QComboBox | QtWidgets.QAbstractSpinBox
        ):
            continue  # the editor inside a combo or spin box speaks for it
        if _on_screen(window, widget):
            yield widget


def _describe(widget: QtWidgets.QWidget | None) -> str:
    if widget is None:
        return "<none>"
    chain: list[str] = []
    parent: QtWidgets.QWidget | None = widget
    while parent is not None and len(chain) < 4:
        chain.append(f"{type(parent).__name__}({parent.objectName()!r})")
        parent = parent.parentWidget()
    return " < ".join(chain)


def _check_named(window: MainWindow) -> None:
    unnamed = [
        _describe(w) for w in _reachable(window) if not accessible_name(w).strip()
    ]
    untipped = [
        _describe(w)
        for w in _reachable(window)
        if not (w.toolTip().strip() or w.statusTip().strip())
        # an item view's rows carry their own tooltips; the view is named
        and not isinstance(w, QtWidgets.QAbstractItemView)
    ]
    assert unnamed == []
    assert untipped == []


# -- every visible control is named ----------------------------------------------------
def test_every_control_is_named_in_every_state(qtbot: QtBot) -> None:
    window = _window(qtbot)
    _check_named(window)  # nothing loaded
    _periodogram(qtbot, window)
    _check_named(window)
    window._analysis_actions["prewhiten"].trigger()
    window._controls._backend_combo.setCurrentText("numpy")
    window._controls.set_prewhiten_settings(SETTINGS)
    with qtbot.waitSignal(window.controller.solution_ready, timeout=60_000):
        window._controls.request_compute()
    _settle(qtbot, window)
    _check_named(window)
    window.commands.action("data.tool_lasso").trigger()  # shows the raw view tools
    qtbot.wait(40)
    _check_named(window)
    for dock in (
        window._subsets_dock,
        window._variations_dock,
        window._history_dock,
        window._spacing_dock,
    ):
        dock.raise_()
        qtbot.wait(40)
        _check_named(window)
    window.controller.shutdown()


def test_batch_browsing_controls_are_named(qtbot: QtBot, tmp_path: Path) -> None:
    for i in range(2):
        rng = np.random.default_rng(i)
        t = np.sort(rng.uniform(0, 30, 200))
        y = np.sin(2 * np.pi * (1.0 + i) * t) + rng.normal(0, 0.05, t.size)
        np.savetxt(
            tmp_path / f"star{i}.csv",
            np.c_[t, y, np.full(t.size, 0.05)],
            delimiter=",",
            header="time,mag,err",
            comments="",
        )
    window = _window(qtbot)
    window._controls._backend_combo.setCurrentText("finufft")
    window._load_folder(tmp_path)
    qtbot.waitUntil(lambda: window._sources_dock.isVisible(), timeout=10_000)
    qtbot.wait(40)
    _check_named(window)
    window.controller.shutdown()


# -- Tab order -------------------------------------------------------------------------
def _tab_walk(
    qtbot: QtBot, window: MainWindow, start: QtWidgets.QWidget, *, back: bool = False
) -> list[QtWidgets.QWidget]:
    """Press Tab (Shift+Tab) from ``start`` until the focus comes back to it.

    Fails on a trap (the cycle never returns) or a stop on a widget the user cannot
    see.
    """
    start.setFocus(Qt.FocusReason.TabFocusReason)
    qtbot.waitUntil(start.hasFocus, timeout=2_000)
    key = Qt.Key.Key_Backtab if back else Qt.Key.Key_Tab
    modifiers = (
        Qt.KeyboardModifier.ShiftModifier if back else Qt.KeyboardModifier.NoModifier
    )
    seen: list[QtWidgets.QWidget] = []
    for _ in range(400):
        app = QtWidgets.QApplication.instance()
        assert isinstance(app, QtWidgets.QApplication)
        current = app.focusWidget()
        assert current is not None
        QtTest.QTest.keyClick(current, key, modifiers)
        app.processEvents()
        focused = app.focusWidget()
        assert focused is not None, f"Tab from {_describe(current)} lost the focus"
        if focused is start:
            return seen
        assert window.isAncestorOf(focused), _describe(focused)
        assert _on_screen(window, focused), (
            f"Tab stopped on a widget the user cannot see: {_describe(focused)}"
        )
        seen.append(focused)
    raise AssertionError(
        "Tab never came back to the start (a trap):\n"
        + "\n".join(_describe(w) for w in seen[:60])
    )


def test_tab_cycles_through_the_periodogram_window(qtbot: QtBot) -> None:
    window = _window(qtbot)
    _periodogram(qtbot, window)
    controls = window._controls
    seen = _tab_walk(qtbot, window, controls._analysis_combo)
    for wanted in (
        controls._method_combo,
        controls._compute_btn,
        window._spectrum._plot,
        window._peaks_table._table,
    ):
        assert wanted in seen, _describe(wanted)
    # Shift+Tab walks the same cycle backwards.
    back = _tab_walk(qtbot, window, controls._analysis_combo, back=True)
    assert set(back) == set(seen)
    window.controller.shutdown()


def test_tab_cycles_through_a_session(qtbot: QtBot) -> None:
    window = _window(qtbot)
    _session(qtbot, window)
    controls = window._controls
    seen = _tab_walk(qtbot, window, controls._analysis_combo)
    for wanted in (
        controls._compute_btn,
        window._spectrum._plot,
        window._session_bar.field,
        window._session_bar.button("session.step"),
        window._solution_panel.table,
    ):
        assert wanted in seen, _describe(wanted)
    # Nothing of a pane behind another tab is in the cycle (A11Y-15).
    for dock in (window._subsets_dock, window._history_dock):
        content = dock.widget()
        assert content is not None
        assert not any(content.isAncestorOf(w) for w in seen), dock.windowTitle()
    # The History pane, brought forward, joins it.
    window._history_dock.raise_()
    qtbot.wait(40)
    seen = _tab_walk(qtbot, window, controls._analysis_combo)
    assert window._history_panel.tree in seen
    window.controller.shutdown()


def test_parked_tab_bars_never_take_the_focus(qtbot: QtBot) -> None:
    # QMainWindow keeps tab bars it is not using "visible" but off-screen; Tab
    # landed on an invisible "Period spacing" tab and its scroll arrows.
    window = _window(qtbot)
    for analysis in ("prewhiten", "periodogram", "prewhiten"):
        window._analysis_actions[analysis].trigger()
        qtbot.wait(40)
        for bar in window.findChildren(QtWidgets.QTabBar):
            for arrow in bar.findChildren(QtWidgets.QToolButton):
                assert arrow.focusPolicy() == Qt.FocusPolicy.NoFocus
            if bar.parent() is window and not _on_screen(window, bar):
                assert bar.focusPolicy() == Qt.FocusPolicy.NoFocus, [
                    bar.tabText(i) for i in range(bar.count())
                ]
    window.controller.shutdown()


# -- commands and shortcuts ------------------------------------------------------------
def _menu_actions(menu: QtWidgets.QMenu) -> Iterator[QtGui.QAction]:
    for action in menu.actions():
        yield action
        sub = action.menu()
        if isinstance(sub, QtWidgets.QMenu):
            yield from _menu_actions(sub)


def test_every_command_is_reachable_from_the_menu_bar(qtbot: QtBot) -> None:
    window = _window(qtbot)
    bar = window.menuBar()
    assert bar is not None
    in_menus: set[int] = set()
    for top in bar.actions():
        menu = top.menu()
        if isinstance(menu, QtWidgets.QMenu):
            in_menus.update(id(a) for a in _menu_actions(menu))
    for command in COMMANDS:
        assert id(window.commands.action(command.id)) in in_menus, command.id
    # Every window action with a key is also a menu item (keyboard users can find
    # it without knowing the key, mouse users learn the key there).
    missing = [
        a.text()
        for a in window.findChildren(QtGui.QAction)
        if not a.shortcut().isEmpty()
        and a.shortcutContext() != Qt.ShortcutContext.WidgetWithChildrenShortcut
        and a.shortcutContext() != Qt.ShortcutContext.WidgetShortcut
        and id(a) not in in_menus
        and a.parent() is window
    ]
    assert missing == []
    window.controller.shutdown()


def _binding_clashes(window: MainWindow) -> list[str]:
    """Enabled key sequences bound twice in one context (the window, or one widget
    with its children)."""
    by_context: dict[object, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for action in window.findChildren(QtGui.QAction):
        if not action.isEnabled():
            continue
        context = action.shortcutContext()
        host: object = window
        if context in (
            Qt.ShortcutContext.WidgetShortcut,
            Qt.ShortcutContext.WidgetWithChildrenShortcut,
        ):
            host = action.parent()
        for key in action.shortcuts():
            if not key.isEmpty():
                text = key.toString()
                by_context[id(host)][text].append(action.text())
    for shortcut in window.findChildren(QtGui.QShortcut):
        if not shortcut.isEnabled() or shortcut.key().isEmpty():
            continue
        host = shortcut.parent()
        if shortcut.context() in (
            Qt.ShortcutContext.WindowShortcut,
            Qt.ShortcutContext.ApplicationShortcut,
        ):
            host = window
        by_context[id(host)][shortcut.key().toString()].append(
            f"QShortcut on {type(host).__name__}"
        )
    return [
        f"{key}: {names}"
        for keys in by_context.values()
        for key, names in keys.items()
        if len(names) > 1
    ]


def test_no_key_is_bound_twice_in_one_context(qtbot: QtBot) -> None:
    window = _window(qtbot)
    assert _binding_clashes(window) == []
    _periodogram(qtbot, window)
    assert _binding_clashes(window) == []
    _session(qtbot, window)
    assert _binding_clashes(window) == []
    window._on_busy_changed(True)  # Stop (Esc) enabled, Clear selection not
    assert _binding_clashes(window) == []
    window._on_busy_changed(False)
    window.controller.shutdown()


def _mnemonic(text: str) -> str | None:
    i = 0
    while i < len(text) - 1:
        if text[i] == "&":
            if text[i + 1] != "&":
                return text[i + 1].lower()
            i += 1
        i += 1
    return None


def _mnemonic_clashes(window: MainWindow) -> list[str]:
    """Alt letters shared by two widgets the user can reach at the same time
    (Qt then only moves the focus between them and clicks neither)."""
    owners: dict[str, list[str]] = defaultdict(list)
    for widget in window.findChildren(QtWidgets.QWidget):
        if isinstance(widget, QtWidgets.QMenu | QtWidgets.QMenuBar):
            continue
        if not isinstance(widget, QtWidgets.QLabel | QtWidgets.QAbstractButton):
            continue
        if not (_on_screen(window, widget) and widget.isEnabled()):
            continue
        if isinstance(widget, QtWidgets.QLabel) and widget.buddy() is None:
            continue  # a label without a buddy shows the "&" and binds nothing
        letter = _mnemonic(widget.text())
        if letter is not None:
            owners[letter].append(widget.text())
    return [f"Alt+{k}: {v}" for k, v in owners.items() if len(v) > 1]


def test_alt_letters_are_unique_on_screen(qtbot: QtBot) -> None:
    window = _window(qtbot)
    _session(qtbot, window)
    for dock in (
        window._solution_dock,
        window._subsets_dock,
        window._variations_dock,
        window._history_dock,
        window._spacing_dock,
    ):
        dock.raise_()
        qtbot.wait(40)
        assert _mnemonic_clashes(window) == [], dock.windowTitle()
    window.controller.shutdown()


# -- the minimum size ------------------------------------------------------------------
def test_window_minimum_covers_the_panes_when_the_screen_allows() -> None:
    size = QtCore.QSize
    # The panes need 976 px: a 1366 px screen gets that minimum, not 900 (at which
    # the plots were drawn over the Controls).
    assert window_minimum(size(976, 456), size(1366, 768)) == (976, 600)
    assert window_minimum(size(700, 400), size(1366, 768)) == (900, 600)
    # Never more than the screen shows.
    assert window_minimum(size(1200, 700), size(1024, 640)) == (1024, 640)


def test_controls_grow_with_the_zoom(qtbot: QtBot) -> None:
    from cuperiod.gui.theme import apply_theme, font_scale

    window = _window(qtbot)
    _load(window)
    window._analysis_actions["prewhiten"].trigger()
    try:
        for _ in range(3):
            window._zoom(1)
        qtbot.wait(80)
        assert font_scale() > 1.2
        controls = window._controls
        assert controls.minimumWidth() >= controls.minimumSizeHint().width()
        button = controls._compute_btn
        assert button.width() >= button.minimumSizeHint().width()
        # The frequency table keeps room for its ID check box.
        table = window._solution_panel.table
        assert table.columnWidth(0) >= table.fontMetrics().horizontalAdvance("F99")
        # The status badges were measured once and clipped ("ackend: finuff").
        def clipped() -> list[str]:
            return [
                badge.text()
                for badge in window._info.findChildren(_BadgeLabel)
                if badge.isVisible()
                and badge.width()
                < badge.fontMetrics().horizontalAdvance(badge.text()) + _BADGE_HPAD
            ]

        qtbot.waitUntil(lambda: clipped() == [], timeout=3_000)
    finally:
        app = QtWidgets.QApplication.instance()
        assert isinstance(app, QtWidgets.QApplication)
        apply_theme(app, "dark", font_scale=1.0)  # never leak a zoom into other tests
        window.controller.shutdown()


# -- text that is cut off or says the wrong thing --------------------------------------
def test_wide_panes_wrap_instead_of_scrolling_sideways(qtbot: QtBot) -> None:
    # As single rows the Subsets, Variations and Period spacing panes were 757, 664
    # and 632 px wide: in a 400 px dock Apply, the partition picker and "Find period
    # spacing" sat off to the right of a horizontal scroll bar.
    window = _window(qtbot)
    for pane in (
        window._subsets_panel,
        window._variation_panel,
        window._spacing_panel,
    ):
        assert pane.minimumSizeHint().width() <= 300, type(pane).__name__
    window.controller.shutdown()


def test_readouts_and_badges_never_show_half_a_text(qtbot: QtBot) -> None:
    window = _window(qtbot)
    _session(qtbot, window)
    # No empty period pill before a fold, no "0 active of 0 components".
    phased = window._phased
    phased.clear()
    assert not phased._period_label.isVisible()
    window.controller.shutdown()
    # The spectrum readout keeps its start (the frequency) when it does not fit.
    readout = type(window._spectrum._readout)("—")
    qtbot.addWidget(readout)
    full = "cursor  f = 15.4 c/d    P = 0.0649353 d    amplitude = 0.003671 mag"
    readout.setText(full)
    readout.resize(160, 24)
    readout.show()
    QtWidgets.QApplication.processEvents()
    shown = QtWidgets.QLabel.text(readout)
    assert readout.text() == full
    assert shown.startswith("cursor") and shown.endswith("…")  # elided at its end
    assert readout.accessibleDescription() == full


def test_the_import_dialog_does_not_claim_a_header_it_did_not_find(
    qtbot: QtBot, tmp_path: Path
) -> None:
    from cuperiod.gui.loaders import preview_file
    from cuperiod.gui.widgets.preview_dialog import PreviewDialog

    path = tmp_path / "v1234_her.dat"
    lines = ["# V1234 Her, two-site campaign", "# HJD-2450000  Vmag  eV  flag"]
    lines += [f"{8000 + i / 100:.4f}  {10 + i / 1000:.4f}  0.003  0" for i in range(40)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    preview = preview_file(path)
    assert preview is not None
    dialog = PreviewDialog(path.name, preview, None, path=path)
    qtbot.addWidget(dialog)
    # The reader found no names under the commented header (col1, col2, ...): the
    # box saying "First row holds the column names" stayed ticked regardless.
    assert not dialog._header.isChecked()
    options = dialog.read_options()
    assert options is not None and not options.header


def test_history_lines_do_not_say_everything_twice() -> None:
    from cuperiod.gui.viewmodels import HistoryItem
    from cuperiod.gui.widgets.history_panel import _group_title, compact_entry_text

    assert compact_entry_text("Restore 22 point(s): restored 22 points") == (
        "Restore 22 point(s)"
    )
    assert compact_entry_text("Improve all: improved all 4 frequencies") == (
        "Improved all 4 frequencies"
    )
    kept = "Add 15.4 c/d: F4 added at 15.3966 c/d"  # the outcome adds the name
    assert compact_entry_text(kept) == kept
    group = [
        HistoryItem(7, 3, "use_variation", "Variation per 'chunk'", "edited", True),
        HistoryItem(8, 3, "set_variation", "Vary F1: both", "edited", True),
    ]
    title = _group_title(group)
    assert "_" not in title and "vary per subset" in title
