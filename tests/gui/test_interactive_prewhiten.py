"""Interactive pre-whitening through the main window, end to end (design 5.5, G8b).

Load → periodogram → select peaks by click and by keyboard → switch to pre-whitening
→ Auto → take over (add a peak at the cursor, a tied harmonic, fix a frequency,
lasso-delete points, split into nights with zero points, Improve all) → undo ×3 →
redo → save the project → export the script → reopen the project.

Everything goes through the window's widgets and registry actions, as a user would;
sessions run on ``numpy`` and the periodogram on ``finufft`` (CPU), so nothing here
creates a CUDA context. Prompts are answered by a FakeDialogProvider.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtTest
from pytestqt.qtbot import QtBot

from cuperiod.core.config import PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve
from cuperiod.gui.dialogs import FakeDialogProvider
from cuperiod.gui.main_window import MainWindow
from cuperiod.gui.models import SourceItem
from cuperiod.gui.qt import QtCore, QtWidgets
from cuperiod.gui.viewmodels import SessionView

Qt = QtCore.Qt
SETTINGS = PreWhitenSettings(max_frequencies=2, samples_per_peak=5)


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path: Path) -> None:
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        str(tmp_path / "settings"),
    )


def nightly_curve() -> LightCurve:
    """Three modes sampled in 24 nights of 0.25 d: nights to split, a harmonic."""
    rng = np.random.default_rng(7)
    starts = np.arange(24) * 1.0 + rng.uniform(0.0, 0.05, 24)
    time = np.concatenate([s + np.sort(rng.uniform(0.0, 0.25, 30)) for s in starts])
    value = (
        12.0
        + 0.20 * np.sin(2 * np.pi * 7.3 * time)
        + 0.06 * np.sin(2 * np.pi * 14.6 * time + 0.4)
        + 0.05 * np.sin(2 * np.pi * 9.85 * time + 1.3)
        + rng.normal(0.0, 0.004, time.size)
    )
    return LightCurve.from_arrays(time, value, np.full(time.size, 0.004))


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
    qtbot.wait(30)
    view = ctl.state.session_view
    assert isinstance(view, SessionView)
    return view


def _key(widget: QtWidgets.QWidget, key: Qt.Key, mods: object = None) -> None:
    modifiers = Qt.KeyboardModifier.NoModifier if mods is None else mods
    QtTest.QTest.keyClick(widget, key, modifiers)  # type: ignore[arg-type]
    QtWidgets.QApplication.processEvents()


def test_interactive_prewhitening_end_to_end(qtbot: QtBot, tmp_path: Path) -> None:
    provider = FakeDialogProvider()
    window = _window(qtbot, provider)
    ctl = window.controller
    controls = window._controls

    # -- load and periodogram ---------------------------------------------------------
    window._load_demo(SourceItem("nightly", "Nightly pulsator", nightly_curve))
    assert "Nightly pulsator" in window.windowTitle()
    controls._backend_combo.setCurrentText("finufft")
    with qtbot.waitSignal(ctl.periodogram_ready, timeout=60_000):
        controls._compute_btn.click()
    peaks = ctl._peaks
    assert len(peaks) >= 3
    assert ctl.state.selected_period == pytest.approx(peaks[0].period)

    # A click on the second row of the Peaks table folds at that peak.
    table = window._peaks_table._table
    rect = table.visualRect(table.model().index(1, 0))
    QtTest.QTest.mouseClick(
        table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center()
    )
    qtbot.waitUntil(lambda: ctl.state.selected_period == pytest.approx(peaks[1].period))
    # The keyboard: Ctrl+1 focuses the spectrum, Home goes to the tallest peak,
    # Ctrl+Right steps to the next marked peak and Enter folds there.
    window.commands.action("view.focus_spectrum").trigger()
    plot = window._spectrum._plot
    assert plot.hasFocus() or QtWidgets.QApplication.focusWidget() is plot
    _key(plot, Qt.Key.Key_Home)
    _key(plot, Qt.Key.Key_Return)
    qtbot.waitUntil(
        lambda: ctl.state.selected_period == pytest.approx(peaks[0].period, rel=1e-3)
    )
    _key(plot, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    _key(plot, Qt.Key.Key_Return)
    periods = [p.period for p in peaks]
    qtbot.waitUntil(
        lambda: (
            ctl.state.selected_period is not None
            and ctl.state.selected_period != pytest.approx(peaks[0].period, rel=1e-3)
            and min(abs(ctl.state.selected_period - p) for p in periods) < 1e-6
        )
    )

    # -- pre-whitening: Auto ---------------------------------------------------------
    window._analysis_actions["prewhiten"].trigger()
    assert window._solution_dock.isVisible() and not window._peaks_dock.isVisible()
    controls._backend_combo.setCurrentText("numpy")
    controls.set_prewhiten_settings(SETTINGS)
    with qtbot.waitSignal(ctl.solution_ready, timeout=90_000):
        controls._compute_btn.click()  # "Run pre-whitening" = Auto
    view = _settle(qtbot, window)
    assert window._solution_panel.editable
    assert len([r for r in view.rows if r.active]) == 2
    assert window._session_bar.isVisible()
    assert not window.isWindowModified()  # an untouched Auto is clean
    first = view.rows[0].frequency
    assert first == pytest.approx(7.3, abs=0.01)

    # -- take over: N puts the cursor on the candidate, A adds the peak there --------
    window.commands.action("view.focus_spectrum").trigger()
    _key(plot, Qt.Key.Key_N)
    assert window._spectrum.cursor_frequency() is not None
    _key(plot, Qt.Key.Key_A)
    view = _settle(qtbot, window)
    assert len(view.rows) == 3
    assert window.isWindowModified()  # edited: unsaved work
    assert window.commands.action("edit.undo").isEnabled()

    # A combination typed in the session bar: "=F1+F2" adds F1 + F2 and ties it.
    f1, f2 = view.rows[0], view.rows[1]
    field = window._session_bar.field
    field.setText(f"={f1.label}+{f2.label}")
    _key(field, Qt.Key.Key_Return)
    view = _settle(qtbot, window)
    tied = [r for r in view.rows if r.tie]
    assert tied, ctl.state
    assert tied[0].frequency == pytest.approx(f1.frequency + f2.frequency, rel=1e-9)

    # L in the table fixes the selected frequency.
    freq_table = window._solution_panel.table
    target = view.rows[1].id
    freq_table.select_ids([target])
    freq_table.setFocus(Qt.FocusReason.OtherFocusReason)
    _key(freq_table, Qt.Key.Key_L)
    view = _settle(qtbot, window)
    row = view.row(target)
    assert row is not None and row.fix_frequency

    # Lasso a stretch of the light curve and delete those points.
    window.commands.action("data.tool_lasso").trigger()
    raw = window._raw
    assert raw.isVisible() and raw.tool() == "lasso"
    t0 = float(np.min(view.time))
    polygon = np.array(
        [[0.9, -100.0], [1.4, -100.0], [1.4, 100.0], [0.9, 100.0]]
    )  # times are plotted from the first sample
    raw._box.region_selected.emit(polygon, False)
    selected = raw.selection_count()
    assert selected > 0
    raw.focus_plot()
    _key(raw._plot, Qt.Key.Key_Delete)
    view = _settle(qtbot, window)
    assert int(np.count_nonzero(view.deleted)) == selected
    assert t0 == pytest.approx(float(np.min(view.time)))

    # Split into nights (the Subsets dock suggests it) with a zero point per night.
    window.commands.action("view.focus_subsets").trigger()
    subsets = window._subsets_panel
    qtbot.waitUntil(lambda: subsets._hint_button.isVisibleTo(subsets))
    subsets._hint_button.click()
    view = _settle(qtbot, window)
    assert view.partitions
    partition = view.partitions[-1]
    subsets._offsets.click()
    view = _settle(qtbot, window)
    assert view.offsets_by == partition
    assert len([s for s in view.subsets if s.partition == partition]) >= 20

    # Improve all (F9) is a long job: Stop is offered while it runs.
    with qtbot.waitSignal(ctl.busy_changed, timeout=60_000):
        window.commands.action("session.improve_all").trigger()
    view = _settle(qtbot, window)
    assert not window.commands.action("session.stop").isEnabled()
    serial_after_improve = view.serial

    # -- undo x3, redo -----------------------------------------------------------------
    undo = window.commands.action("edit.undo")
    for _ in range(3):
        undo.trigger()
        _settle(qtbot, window)
    view = _settle(qtbot, window)
    assert view.can_redo and view.serial != serial_after_improve
    assert view.offsets_by is None  # undone: improve, zero points, the partition
    window.commands.action("edit.redo").trigger()
    view = _settle(qtbot, window)
    assert view.partitions  # the split is back

    # -- save the project, export the script, reopen -----------------------------------
    project = tmp_path / "nightly.cupw.json"
    provider.answers.append(str(project))
    with qtbot.waitSignal(ctl.project_changed, timeout=60_000):
        window.commands.action("file.save_project").trigger()
    qtbot.waitUntil(lambda: project.is_file() and not ctl.is_session_dirty())
    qtbot.waitUntil(lambda: "nightly.cupw.json" in window.windowTitle())
    assert not window.isWindowModified()
    saved_rows = [(r.id, r.frequency, r.amplitude) for r in view.rows]

    script = tmp_path / "nightly_session.py"
    provider.answers.append(str(script))
    window.commands.action("file.export_script").trigger()
    qtbot.waitUntil(script.is_file, timeout=60_000)
    text = script.read_text(encoding="utf-8")
    assert "cuperiod" in text and "PreWhitenSession" in text

    ctl.open_project(project)
    qtbot.waitUntil(lambda: ctl.state.source_id.startswith("project:"), timeout=60_000)
    view = _settle(qtbot, window)
    reopened = [(r.id, r.frequency, r.amplitude) for r in view.rows]
    assert [r[0] for r in reopened] == [r[0] for r in saved_rows]
    assert np.allclose([r[1:] for r in reopened], [r[1:] for r in saved_rows])
    assert controls.analysis == "prewhiten"
    assert "nightly.cupw.json" in window.windowTitle()
    assert window._recent()[0] == str(project)
    ctl.shutdown()
