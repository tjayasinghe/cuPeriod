"""The full Period04 loop through the main window, end to end (design 5.5, H2).

A two-site delta Scuti campaign with a zero-point offset per night and a few
outliers goes through everything a user does in one sitting: Auto, then taking over
(Step at the stopping rule, Shift+click in the spectrum, a tie typed in the session
bar, fix and toggle in the table, Calculate, Improve all and Improve selected), a
lasso delete and its restore, nights with a zero point each, amplitude and phase per
chunk of the campaign (Variations: calculate, apply), undo / redo, a jump in the
History pane, Save project, Export script, running that script in a fresh Python,
and opening the project again. Each step is checked on the controller's state, the
Frequencies table and the plots (markers, deleted points, readouts).

Sessions run on ``numpy`` with a 30 c/d search range; prompts go to a
FakeDialogProvider; nothing creates a CUDA context.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6 import QtTest
from pytestqt.qtbot import QtBot

from cuperiod.core.config import PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve
from cuperiod.gui.dialogs import FakeDialogProvider
from cuperiod.gui.main_window import MainWindow
from cuperiod.gui.models import SourceItem
from cuperiod.gui.qt import QtCore, QtGui, QtWidgets
from cuperiod.gui.viewmodels import SessionView
from cuperiod.gui.widgets.frequency_table import COLUMNS, ID_ROLE, FrequencyTable

Qt = QtCore.Qt
SETTINGS = PreWhitenSettings(
    max_frequencies=3, samples_per_peak=5, maximum_frequency=30.0
)
#: The modes of the synthetic star (c/d) and their amplitudes (mag).
MODES = ((9.6555, 0.030), (12.7162, 0.012), (10.4515, 0.007), (15.40, 0.005))


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path: Path) -> None:
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        str(tmp_path / "settings"),
    )


def campaign() -> LightCurve:
    """Two sites half a day apart, 12 nights each (some lost to weather), one
    point per 10 minutes; each night has its own zero point; six outliers."""
    rng = np.random.default_rng(42)
    times, offsets = [], []
    for site_start in (0.0, 0.5):
        for night in range(12):
            if night > 1 and rng.uniform() < 0.15:
                continue
            start = night + site_start + rng.uniform(0.0, 0.03)
            n = int(rng.uniform(0.12, 0.16) * 144)
            times.append(start + np.sort(rng.uniform(0.0, n / 144, n)))
            offsets.append(np.full(n, rng.normal(0.0, 0.004)))
    order = np.argsort(np.concatenate(times))
    t = np.concatenate(times)[order]
    zero_points = np.concatenate(offsets)[order]
    y = 10.0 + zero_points + rng.normal(0.0, 0.002, t.size)
    for k, (f, a) in enumerate(MODES):
        y += a * np.sin(2 * np.pi * f * t + 0.7 * k)
    y[rng.choice(t.size, 6, replace=False)] += 0.08
    return LightCurve.from_arrays(t, y, np.full(t.size, 0.002))


def _window(qtbot: QtBot, provider: FakeDialogProvider) -> MainWindow:
    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    window = MainWindow(app)
    qtbot.addWidget(window)
    window.set_dialog_provider(provider)
    window.resize(1366, 768)
    window.show()
    window.activateWindow()
    qtbot.waitExposed(window)
    return window


def _settle(qtbot: QtBot, window: MainWindow) -> SessionView:
    """Every queued session job (and its view) has come back; the shown view."""
    ctl = window.controller
    bridge = ctl.session_bridge
    qtbot.waitUntil(
        lambda: bridge.is_idle() and ctl._compute.session_jobs == 0, timeout=90_000
    )
    qtbot.wait(40)
    view = ctl.state.session_view
    assert isinstance(view, SessionView)
    return view


def _key(widget: QtWidgets.QWidget, key: Qt.Key, mods: Any = None) -> None:
    modifiers = Qt.KeyboardModifier.NoModifier if mods is None else mods
    QtTest.QTest.keyClick(widget, key, modifiers)
    QtWidgets.QApplication.processEvents()


def _column(key: str) -> int:
    return next(i for i, c in enumerate(COLUMNS) if c.key == key)


def _cell(table: FrequencyTable, ident: int, key: str) -> QtWidgets.QTableWidgetItem:
    """The table cell of component ``ident`` in column ``key`` (rows by id)."""
    for row in range(table.rowCount()):
        first = table.item(row, 0)
        if first is not None and first.data(ID_ROLE) == ident:
            item = table.item(row, _column(key))
            assert item is not None
            return item
    raise AssertionError(f"F{ident} is not in the table")


def _marker_symbols(window: MainWindow) -> dict[int, str]:
    """Component id -> marker symbol drawn in the spectrum."""
    out: dict[int, str] = {}
    for spot in window._spectrum._markers.data:
        payload = spot["data"]
        if payload is not None:
            out[int(payload[1])] = str(spot["symbol"])
    return out


def _shift_click(window: MainWindow, frequency: float) -> None:
    """A real Shift+click on the spectrum at ``frequency`` (pick mode: add there)."""
    view = window._spectrum
    plot = view._plot
    box = plot.getPlotItem().vb
    (y_low, y_high) = box.viewRange()[1]
    scene = box.mapViewToScene(
        QtCore.QPointF(view._to_plot_x(frequency), 0.5 * (y_low + y_high))
    )
    pos = QtCore.QPointF(plot.mapFromScene(scene))
    viewport = plot.viewport()
    assert viewport is not None
    left, shift = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier
    QtCore.QThread.msleep(12)  # pyqtgraph rate-limits moves
    for kind, button, buttons, mods in (
        (QtCore.QEvent.Type.MouseMove, Qt.MouseButton.NoButton,
         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier),
        (QtCore.QEvent.Type.MouseButtonPress, left, left, shift),
        (QtCore.QEvent.Type.MouseButtonRelease, left, Qt.MouseButton.NoButton, shift),
    ):
        event = QtGui.QMouseEvent(
            kind, pos, viewport.mapToGlobal(pos.toPoint()).toPointF(),
            button, buttons, mods,
        )
        QtWidgets.QApplication.sendEvent(viewport, event)
        QtWidgets.QApplication.processEvents()


def _history_items(
    tree: QtWidgets.QTreeWidget,
) -> list[QtWidgets.QTreeWidgetItem]:
    """Every entry row of the History pane (group nodes left out)."""
    found = []
    walker = QtWidgets.QTreeWidgetItemIterator(tree)
    while (item := walker.value()) is not None:
        if item.data(0, Qt.ItemDataRole.UserRole):  # a log entry's sequence number
            found.append(item)
        walker += 1
    return found


def test_the_full_period04_loop(qtbot: QtBot, tmp_path: Path) -> None:
    provider = FakeDialogProvider()
    window = _window(qtbot, provider)
    ctl = window.controller
    controls = window._controls
    table = window._solution_panel.table
    status: list[str] = []
    bar = window.statusBar()
    assert bar is not None
    bar.messageChanged.connect(status.append)

    # -- load, pre-whitening, Auto -----------------------------------------------------
    window._load_demo(SourceItem("campaign", "Two-site campaign", campaign))
    window._analysis_actions["prewhiten"].trigger()
    controls._backend_combo.setCurrentText("numpy")
    controls.set_prewhiten_settings(SETTINGS)
    with qtbot.waitSignal(ctl.solution_ready, timeout=90_000):
        controls._compute_btn.click()  # "Run pre-whitening" = Auto
    view = _settle(qtbot, window)
    found = [r.frequency for r in view.rows]
    assert found == pytest.approx([f for f, _ in MODES[:3]], abs=0.02)
    assert view.status.startswith("stopped")
    assert table.rowCount() == 3 and window._solution_panel.editable
    assert _cell(table, 1, "label").text().endswith("F1")
    assert set(_marker_symbols(window).values()) == {"o"}  # every component free
    assert not window.windowTitle().endswith("*")

    # -- Step at the stopping rule says why nothing happened ---------------------------
    window.commands.action("session.step").trigger()
    view = _settle(qtbot, window)
    assert len(view.rows) == 3
    assert any("Step: the search has stopped" in s for s in status), status[-3:]

    # -- take over: Shift+click on the fourth mode adds it -----------------------------
    window._spectrum._plot.setFocus(Qt.FocusReason.OtherFocusReason)
    _shift_click(window, 15.40)
    view = _settle(qtbot, window)
    assert len(view.rows) == 4
    added = view.rows[-1]
    assert added.frequency == pytest.approx(15.40, abs=0.02)
    assert added.origin == "manual"
    assert table.rowCount() == 4
    assert window.windowTitle().endswith("*")  # unsaved work
    assert window._spectrum.cursor_frequency() is not None

    # -- a combination typed in the session bar, tied ----------------------------------
    field = window._session_bar.field
    field.setText("=F1+F2")
    _key(field, Qt.Key.Key_Return)
    view = _settle(qtbot, window)
    tied = view.rows[-1]
    assert tied.tie == "F1 + F2"
    assert tied.frequency == pytest.approx(
        view.rows[0].frequency + view.rows[1].frequency, rel=1e-12
    )
    assert _cell(table, tied.id, "tie").text() == "F1 + F2"
    assert _cell(table, tied.id, "frequency").font().bold()  # tied: bold
    assert _marker_symbols(window)[tied.id] == "d"  # diamond = tied

    # -- fix F2's frequency (L), switch F3 off (Space) ---------------------------------
    table.select_ids([2])
    table.setFocus(Qt.FocusReason.OtherFocusReason)
    _key(table, Qt.Key.Key_L)
    view = _settle(qtbot, window)
    row = view.row(2)
    assert row is not None and row.fix_frequency
    assert _cell(table, 2, "fix_frequency").checkState() == Qt.CheckState.Checked
    assert _marker_symbols(window)[2] == "s"  # square = fixed frequency
    table.select_ids([3])
    _key(table, Qt.Key.Key_Space)
    view = _settle(qtbot, window)
    row = view.row(3)
    assert row is not None and not row.active
    assert _cell(table, 3, "label").checkState() == Qt.CheckState.Unchecked

    # -- Calculate (F8), Improve all (errors), Improve selected ------------------------
    amplitudes = {r.id: r.amplitude for r in view.rows}
    window.commands.action("session.calculate").trigger()
    view = _settle(qtbot, window)
    assert {r.id for r in view.rows} == set(amplitudes)  # frequencies held
    # The explicit Calculate solves with the covariance: errors at once.
    row = view.row(1)
    assert row is not None and np.isfinite(row.frequency_error)
    assert _cell(table, 1, "frequency_error").text() not in ("", "–")
    assert _cell(table, 1, "amplitude_error").text() not in ("", "–")
    before = {r.id: r.frequency for r in view.rows}
    window.commands.action("session.improve_all").trigger()
    view = _settle(qtbot, window)
    row = view.row(2)
    assert row is not None and row.frequency == before[2]  # fixed: not moved
    row = view.row(1)
    assert row is not None and np.isfinite(row.frequency_error)
    assert _cell(table, 1, "frequency_error").text() not in ("", "–")
    assert _cell(table, 2, "frequency_error").text() == "fixed"
    table.select_ids([1])
    window.commands.action("session.improve_selected").trigger()
    view = _settle(qtbot, window)
    assert not window.commands.action("session.stop").isEnabled()  # idle again

    # -- lasso the outlier-rich second night, delete, restore --------------------------
    window.commands.action("data.tool_lasso").trigger()
    raw = window._raw
    assert raw.isVisible() and raw.tool() == "lasso"
    t0 = float(np.min(view.time))
    first_night = np.array(
        [[0.9, -1e3], [1.4, -1e3], [1.4, 1e3], [0.9, 1e3]]
    )  # times are plotted from the first sample
    raw._box.region_selected.emit(first_night, False)
    chosen = raw.selection_count()
    assert chosen > 0
    raw.focus_plot()
    _key(raw._plot, Qt.Key.Key_Delete)
    view = _settle(qtbot, window)
    assert int(np.count_nonzero(view.deleted)) == chosen
    assert f"{chosen:,} deleted" in raw._title.text()
    assert t0 == pytest.approx(float(np.min(view.time)))
    raw._box.region_selected.emit(first_night, False)
    _key(raw._plot, Qt.Key.Key_U)
    view = _settle(qtbot, window)
    assert not view.deleted.any()
    assert "0 deleted" in raw._title.text()

    # -- nights, a zero point each -----------------------------------------------------
    window.commands.action("view.focus_subsets").trigger()
    subsets = window._subsets_panel
    qtbot.waitUntil(lambda: subsets._hint_button.isVisibleTo(subsets))
    subsets._hint_button.click()
    view = _settle(qtbot, window)
    nights = view.partitions[-1]
    subsets._offsets.click()
    view = _settle(qtbot, window)
    assert view.offsets_by == nights
    shown = [s for s in view.subsets if s.partition == nights]
    assert len(shown) >= 15
    assert all(np.isfinite(s.zero_point) for s in shown)
    assert subsets._table.rowCount() == len(shown)

    # -- amplitude and phase per chunk of the campaign ---------------------------------
    subsets._preset.setCurrentIndex(subsets._preset.findData("chunks"))
    subsets._param.setValue(4.0)
    subsets._name.setText("chunk")
    subsets._apply.click()
    view = _settle(qtbot, window)
    assert "chunk" in view.partitions
    window._variations_dock.raise_()
    qtbot.wait(40)
    variations = window._variation_panel
    variations._components.setText("F1")
    variations._by.setCurrentIndex(variations._by.findText("chunk"))
    with qtbot.waitSignal(ctl.variations_ready, timeout=90_000):
        variations._calculate.click()
    qtbot.waitUntil(lambda: variations.result is not None)
    assert variations._tests.rowCount() == 1
    assert variations._apply.isEnabled()
    variations._apply.click()
    view = _settle(qtbot, window)
    row = view.row(1)
    assert row is not None and row.variation != "none"
    assert view.variation_by == "chunk"
    serial_full = view.serial

    # -- undo ×3, redo -----------------------------------------------------------------
    undo = window.commands.action("edit.undo")
    for _ in range(3):
        undo.trigger()
        _settle(qtbot, window)
    view = _settle(qtbot, window)
    assert view.can_redo and view.serial != serial_full
    assert view.variation_by is None
    window.commands.action("edit.redo").trigger()
    view = _settle(qtbot, window)
    assert view.can_redo  # one undone action left to redo

    # -- History: jump to the entry where F4 was added (keyboard: Enter) ---------------
    window.commands.action("view.focus_history").trigger()
    tree = window._history_panel.tree
    assert tree.hasFocus() or QtWidgets.QApplication.focusWidget() is tree
    entries = _history_items(tree)
    adds = [i for i in entries if "F4 added" in i.text(1)]
    assert len(adds) == 1, " / ".join(i.text(1) for i in entries)
    target = adds[0]
    tree.setCurrentItem(target)
    _key(tree, Qt.Key.Key_Return)
    view = _settle(qtbot, window)
    assert [r.id for r in view.rows] == [1, 2, 3, 4]
    assert table.rowCount() == 4
    for _ in range(30):  # back to the end of the history
        if not view.can_redo:
            break
        window.commands.action("edit.redo").trigger()
        view = _settle(qtbot, window)
    assert view.variation_by == "chunk"
    final_rows = [(r.id, r.frequency, r.amplitude) for r in view.rows]

    # -- save the project, export the script and run it --------------------------------
    project = tmp_path / "campaign.cupw.json"
    provider.answers.append(str(project))
    with qtbot.waitSignal(ctl.project_changed, timeout=60_000):
        window.commands.action("file.save_project").trigger()
    qtbot.waitUntil(lambda: project.is_file() and not ctl.is_session_dirty())
    qtbot.waitUntil(lambda: not window.windowTitle().endswith("*"))
    script = tmp_path / "campaign_session.py"
    provider.answers.append(str(script))
    window.commands.action("file.export_script").trigger()
    qtbot.waitUntil(script.is_file, timeout=60_000)
    run = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    assert run.returncode == 0, run.stderr[-2000:]

    # -- open the project again --------------------------------------------------------
    provider.answers.append(str(project))  # File ▸ Open project…
    window.commands.action("file.open_project").trigger()
    qtbot.waitUntil(lambda: ctl.state.source_id.startswith("project:"), timeout=90_000)
    view = _settle(qtbot, window)
    reopened = [(r.id, r.frequency, r.amplitude) for r in view.rows]
    assert [r[0] for r in reopened] == [r[0] for r in final_rows]
    assert np.allclose([r[1:] for r in reopened], [r[1:] for r in final_rows])
    assert view.offsets_by == nights and view.variation_by == "chunk"
    assert table.rowCount() == len(final_rows)
    assert "campaign.cupw.json" in window.windowTitle()
    ctl.shutdown()


def test_every_action_answers_even_when_it_changes_nothing(qtbot: QtBot) -> None:
    # A Step at the stopping rule, a Renumber of compact ids or a Restore with nothing
    # deleted did nothing and said nothing, so their buttons looked dead; refusals
    # told a GUI user to "pass allow_close=True" or "remove with cascade=True".
    provider = FakeDialogProvider()
    window = _window(qtbot, provider)
    ctl = window.controller
    status: list[str] = []
    bar = window.statusBar()
    assert bar is not None
    bar.messageChanged.connect(status.append)
    window._load_demo(SourceItem("campaign", "Two-site campaign", campaign))
    window._analysis_actions["prewhiten"].trigger()
    window._controls._backend_combo.setCurrentText("numpy")
    window._controls.set_prewhiten_settings(SETTINGS)
    with qtbot.waitSignal(ctl.solution_ready, timeout=90_000):
        window._controls.request_compute()
    _settle(qtbot, window)
    for command, said in (
        ("session.renumber", "Nothing changed: ids are already compact"),
        ("data.restore_all", "Nothing changed: "),
    ):
        status.clear()
        window.commands.action(command).trigger()
        _settle(qtbot, window)
        assert any(s.startswith(said) for s in status), (command, status)
    # A tie, then deleting its parent: refused, in the window's own terms.
    field = window._session_bar.field
    field.setText("=F1+F2")
    _key(field, Qt.Key.Key_Return)
    _settle(qtbot, window)
    table = window._solution_panel.table
    table.select_ids([1])
    table.setFocus(Qt.FocusReason.OtherFocusReason)
    status.clear()
    _key(table, Qt.Key.Key_Delete)
    view = _settle(qtbot, window)
    assert view.row(1) is not None  # refused: F1 is still there
    assert any("untie it first" in s for s in status), status
    assert not any("cascade=True" in s for s in status)
    ctl.shutdown()


def test_library_advice_is_put_in_gui_terms() -> None:
    from cuperiod.gui.main_window import gui_wording

    moved = (
        "the refinement moved 12.34 c/d onto F2 (12.3748547 c/d); add it with "
        "refine='none' or allow_close=True"
    )
    assert "Add exactly here" in gui_wording(moved)
    assert "refine=" not in gui_wording(moved)
    close = "tie puts F5 at 3 c/d, within 1.5 Rayleigh widths of F2; pass "
    close += "allow_close=True to tie it anyway"
    assert "too close to tell apart" in gui_wording(close)
    assert gui_wording("nothing to improve") == "nothing to improve"


def test_exports_ask_through_the_window_dialogs(qtbot: QtBot, tmp_path: Path) -> None:
    # The views' Export… buttons used the static file dialog (a nested event loop
    # the GUI otherwise never runs, and that a test cannot answer).
    provider = FakeDialogProvider()
    window = _window(qtbot, provider)
    ctl = window.controller
    window._load_demo(SourceItem("campaign", "Two-site campaign", campaign))
    window._analysis_actions["prewhiten"].trigger()
    window._controls._backend_combo.setCurrentText("numpy")
    window._controls.set_prewhiten_settings(SETTINGS)
    with qtbot.waitSignal(ctl.solution_ready, timeout=90_000):
        window._controls.request_compute()
    _settle(qtbot, window)
    exports = (
        (window._spectrum._export_csv, "spectrum.csv"),
        (window._spectrum._export_png, "spectrum.png"),
        (window._phased._export_png, "phased.png"),
        (window._solution_panel._export_csv, "frequencies.csv"),
    )
    for slot, name in exports:
        target = tmp_path / name
        provider.answers.append(str(target))
        slot()
        assert provider.calls[-1][0] == "save_file"
        assert target.is_file() and target.stat().st_size > 0, name
    provider.answers.clear()
    window._spectrum._export_csv()  # cancelled: nothing written, nothing raised
    assert provider.calls[-1][0] == "save_file"
    ctl.shutdown()
