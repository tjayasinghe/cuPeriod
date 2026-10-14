"""The spectrum is keyboard operable and describes itself (A11Y-05, GUI-C-15).

A focused spectrum moves its cursor with the arrow keys (local maxima; Ctrl: marked
peaks; Shift: one grid sample), folds there with Enter, zooms with +/-/0, jumps to the
candidate with N and opens its action menu with the Menu key. Its accessible name and
description tell a screen reader what the plot shows.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.qt import QtCore, QtGui, QtWidgets
from cuperiod.gui.viewmodels import CandidateView, MarkerState
from cuperiod.gui.widgets.spectrum_view import PREWHITEN_METHOD, SpectrumView

Qt = QtCore.Qt
QTest = pytest.importorskip("PySide6.QtTest").QTest

CENTRES = (3.0, 7.0, 12.0)


def _spectrum(n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.linspace(0.05, 20.0, n)
    power = np.full(n, 0.01)
    for i, centre in enumerate(CENTRES):
        power += (1.0 - 0.2 * i) * np.exp(-(((frequency - centre) / 0.05) ** 2))
    return frequency, power


def _periodogram(method: str = "GLS", sense: str = "max") -> Periodogram:
    frequency, power = _spectrum()
    if sense == "min":
        power = 1.0 - power
    return Periodogram.from_spectrum(
        method=method, backend="numpy", frequency=frequency, power=power,
        objective_sense=sense,  # type: ignore[arg-type]
        n_samples=300, baseline=50.0,
    )


def _peaks() -> list[Peak]:
    return [
        Peak(period=1 / f, frequency=f, power=1.0 - 0.2 * i, rank=i + 1)
        for i, f in enumerate(CENTRES)
    ]


def _view(qtbot: QtBot, *, method: str = "GLS", sense: str = "max") -> SpectrumView:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.resize(900, 460)
    view.show()
    qtbot.waitExposed(view)
    view.set_periodogram(_periodogram(method, sense))
    view.set_peaks(_peaks())
    view._plot.setFocus()
    QtWidgets.QApplication.processEvents()
    return view


def _key(
    view: SpectrumView,
    key: Qt.Key,
    mods: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    QTest.keyClick(view._plot, key, mods)
    QtWidgets.QApplication.processEvents()


def test_plot_is_a_named_focus_stop_with_a_summary(qtbot: QtBot) -> None:
    view = _view(qtbot)
    plot = view._plot
    assert plot.focusPolicy() & Qt.FocusPolicy.TabFocus
    assert plot.accessibleName() == "Spectrum plot"
    description = plot.accessibleDescription()
    assert "3 peaks marked" in description
    assert "best P = 0.333333 d" in description
    assert "GLS power versus frequency" in description
    assert "Enter" in plot.statusTip() and plot.whatsThis() == plot.statusTip()
    view.set_selected_period(1 / 7.0)
    assert "selected P = 0.142857 d" in plot.accessibleDescription()
    view.set_pick_enabled(True)
    assert "Shift+click" in plot.statusTip()


def test_right_steps_the_cursor_and_enter_folds(qtbot: QtBot) -> None:
    view = _view(qtbot)
    moved: list[float] = []
    periods: list[float] = []
    view.cursor_changed.connect(moved.append)
    view.period_selected.connect(periods.append)
    view.set_selected_period(1 / 3.0)  # from the table: the cursor starts here
    _key(view, Qt.Key.Key_Right)
    assert len(moved) == 1 and moved[0] > 3.0  # the next local maximum to the right
    assert periods == []  # moving the cursor does not fold
    while view.cursor_frequency() is not None and view.cursor_frequency() < 6.5:  # type: ignore[operator]
        _key(view, Qt.Key.Key_Right)
    assert view.cursor_frequency() == pytest.approx(7.0, abs=0.01)
    assert view._cursor_line.isVisible()  # away from the selection, so it is drawn
    readout = view._readout.text()  # the keyboard user's numbers
    assert readout.startswith("cursor") and "P = 0.1428" in readout
    _key(view, Qt.Key.Key_Return)
    assert len(periods) == 1 and 1 / periods[0] == pytest.approx(7.0, abs=0.01)
    assert view._sel_period == pytest.approx(periods[0])
    assert not view._cursor_line.isVisible()  # back on the selection
    _key(view, Qt.Key.Key_Left)
    assert view.cursor_frequency() < 7.0  # type: ignore[operator]


def test_ctrl_arrows_step_between_marked_peaks(qtbot: QtBot) -> None:
    view = _view(qtbot)
    picked: list[Peak] = []
    view.peak_selected.connect(picked.append)
    view.set_selected_period(1 / 3.0)
    _key(view, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    assert view.cursor_frequency() == pytest.approx(7.0)
    _key(view, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    assert view.cursor_frequency() == pytest.approx(12.0)
    _key(view, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)  # no more
    assert view.cursor_frequency() == pytest.approx(12.0)
    _key(view, Qt.Key.Key_Enter)
    assert picked and picked[0].rank == 3  # the whole peak (a BLS epoch survives)


def test_shift_arrows_move_one_grid_sample(qtbot: QtBot) -> None:
    view = _view(qtbot)
    grid = view._pg.frequency  # type: ignore[union-attr]
    view.set_selected_period(1 / float(grid[1000]))
    _key(view, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert view.cursor_frequency() == pytest.approx(float(grid[1001]))
    _key(view, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
    _key(view, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
    assert view.cursor_frequency() == pytest.approx(float(grid[999]))


def test_period_axis_steps_follow_the_screen(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view._xaxis_combo.setCurrentText("period")
    view.set_selected_period(1 / 7.0)
    _key(view, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    # Right on a period axis is the longer period: 1/3 d, not 1/12 d.
    assert view.cursor_frequency() == pytest.approx(3.0)


def test_home_goes_to_the_tallest_and_minimise_methods_use_minima(qtbot: QtBot) -> None:
    view = _view(qtbot)
    _key(view, Qt.Key.Key_Home)
    assert view.cursor_frequency() == pytest.approx(3.0, abs=0.01)
    pdm = _view(qtbot, method="PDM", sense="min")
    _key(pdm, Qt.Key.Key_Home)
    assert pdm.cursor_frequency() == pytest.approx(3.0, abs=0.01)
    pdm.set_cursor(5.0)
    _key(pdm, Qt.Key.Key_Right)
    assert pdm.cursor_frequency() == pytest.approx(7.0, abs=0.01)  # the next dip


def test_without_a_cursor_the_first_step_starts_at_the_view_edge(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view._plot.getPlotItem().vb.setXRange(10.0, 14.0, padding=0.0)
    _key(view, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    assert view.cursor_frequency() == pytest.approx(12.0)


def test_zoom_keys(qtbot: QtBot) -> None:
    view = _view(qtbot)
    box = view._plot.getPlotItem().vb
    view.set_cursor(7.0)
    (x0, x1), _ = box.viewRange()
    _key(view, Qt.Key.Key_Plus)
    (a, b), _ = box.viewRange()
    assert b - a == pytest.approx(0.5 * (x1 - x0)) and a < 7.0 < b
    _key(view, Qt.Key.Key_Minus)
    (a, b), _ = box.viewRange()
    assert b - a == pytest.approx(x1 - x0)
    _key(view, Qt.Key.Key_Plus)
    _key(view, Qt.Key.Key_0)
    (a, b), _ = box.viewRange()
    assert b - a > 19.0


def test_cursor_moved_off_screen_is_brought_into_view(qtbot: QtBot) -> None:
    view = _view(qtbot)
    box = view._plot.getPlotItem().vb
    box.setXRange(2.0, 4.0, padding=0.0)
    view.set_cursor(3.0)
    _key(view, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    (a, b), _ = box.viewRange()
    assert a <= 7.0 <= b and b - a == pytest.approx(2.0)


def _pick_view(qtbot: QtBot) -> SpectrumView:
    view = _view(qtbot)
    frequency, power = _spectrum()
    view.set_periodogram(
        Periodogram.from_spectrum(
            method=PREWHITEN_METHOD, backend="numpy", frequency=frequency,
            power=power, objective_sense="max", n_samples=300, baseline=50.0,
        )
    )
    residual = 0.3 * np.exp(-(((frequency - 12.0) / 0.05) ** 2)) + 0.01
    view.set_overlay(frequency, residual)
    view.set_marker_states(
        [MarkerState(1, 3.0, "free", False), MarkerState(2, 7.0, "fixed", False)]
    )
    view.set_pick_enabled(True)
    view._plot.setFocus()
    return view


def test_n_moves_to_the_candidate_and_a_adds_at_the_cursor(qtbot: QtBot) -> None:
    view = _pick_view(qtbot)
    adds: list[tuple[float, bool]] = []
    view.add_requested.connect(lambda f, exact: adds.append((f, exact)))
    _key(view, Qt.Key.Key_N)  # no candidate yet: nothing happens
    assert view.cursor_frequency() is None
    view.set_candidate(CandidateView(12.0, 1 / 12.0, 0.31, 9.0, "next"))
    _key(view, Qt.Key.Key_N)
    assert view.cursor_frequency() == 12.0
    _key(view, Qt.Key.Key_A)
    _key(view, Qt.Key.Key_Insert)
    assert adds == [(12.0, False), (12.0, False)]
    # Home follows the residual spectrum in pick mode: what is left is at 12 c/d.
    view.set_cursor(None)
    _key(view, Qt.Key.Key_Home)
    assert view.cursor_frequency() == pytest.approx(12.0, abs=0.01)


def test_registry_shortcuts_keep_the_add_keys(qtbot: QtBot) -> None:
    # With the command registry's context shortcut on the spectrum, A goes there
    # (and follows its enabled state) instead of the plot adding by itself.
    view = _pick_view(qtbot)
    view.set_cursor(12.0)
    adds: list[tuple[float, bool]] = []
    view.add_requested.connect(lambda f, exact: adds.append((f, exact)))
    shortcut = QtGui.QShortcut(QtGui.QKeySequence("A"), view)
    shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
    fired: list[int] = []
    shortcut.activated.connect(lambda: fired.append(1))
    _key(view, Qt.Key.Key_A)
    assert fired == [1] and adds == []
    shortcut.setEnabled(False)  # e.g. a long job runs: nothing may add
    _key(view, Qt.Key.Key_A)
    assert fired == [1] and adds == []


def test_navigation_keys_win_over_window_shortcuts(qtbot: QtBot) -> None:
    view = _pick_view(qtbot)
    view.set_candidate(CandidateView(12.0, 1 / 12.0, 0.31, 9.0, "next"))
    shortcut = QtGui.QShortcut(QtGui.QKeySequence("N"), view.window())
    fired: list[int] = []
    shortcut.activated.connect(lambda: fired.append(1))
    _key(view, Qt.Key.Key_N)
    assert view.cursor_frequency() == 12.0 and fired == []


def test_enter_on_a_component_reports_it(qtbot: QtBot) -> None:
    view = _pick_view(qtbot)
    clicked: list[int] = []
    view.component_clicked.connect(clicked.append)
    view.set_cursor(3.5)
    _key(view, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    assert view.cursor_frequency() == 7.0
    _key(view, Qt.Key.Key_Return)
    assert clicked == [2]


def test_menu_key_opens_the_action_menu(qtbot: QtBot) -> None:
    view = _pick_view(qtbot)
    view.set_cursor(12.0)
    _key(view, Qt.Key.Key_Menu)
    menu = view._context_menu
    assert menu is not None and menu.isVisible()
    assert menu.actions()[0].text().startswith("Add peak at 12")
    menu.close()


def test_keys_do_nothing_without_a_spectrum(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    for key in (Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Plus, Qt.Key.Key_Menu):
        QTest.keyClick(view._plot, key)
    assert view.cursor_frequency() is None
    assert view._plot.accessibleDescription() == "No spectrum yet."


def test_focus_plot_gives_the_plot_the_keyboard(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view._xaxis_combo.setFocus()
    view.focus_plot()
    qtbot.waitUntil(lambda: view._plot.hasFocus(), timeout=2000)


def test_phased_view_folds_at_double_half_or_a_typed_period(qtbot: QtBot) -> None:
    from cuperiod.core.lightcurve import LightCurve
    from cuperiod.gui.widgets.phased_view import PhasedView

    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 20.0, 100))
    lc = LightCurve.from_arrays(t, np.sin(2 * np.pi * t / 1.5), np.full(100, 0.1))
    phased = PhasedView("dark")
    qtbot.addWidget(phased)
    phased.set_light_curve(lc)
    phased.set_fold(1.5, 0.0)
    phased._double.click()  # standing alone it refolds itself
    assert phased._period == pytest.approx(3.0)
    assert phased._period_label.text() == "P = 3 d"
    requested: list[float] = []
    phased.period_requested.connect(requested.append)
    phased._halve.click()
    assert requested == [pytest.approx(1.5)]
    phased._period_edit.setFocus()
    QTest.keyClicks(phased._period_edit, "2.25")
    QTest.keyClick(phased._period_edit, Qt.Key.Key_Return)
    assert requested[-1] == pytest.approx(2.25) and phased._period_edit.text() == ""
    for control in (phased._double, phased._halve, phased._period_edit):
        assert control.accessibleName() and control.toolTip()
    assert "folded at P = 3 d" in phased._plot.accessibleDescription()
