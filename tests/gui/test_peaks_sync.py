"""GUI-C-6 end to end: the result tables follow the spectrum's selection.

A marker click hands the whole peak to the controller (a BLS/TLS ``t0`` survives),
the Peaks and Frequencies highlights move with every selection change, and clicking
the row that used to be highlighted is never a dead click. Driven through the real
``MainWindow`` wiring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtTest
from pytestqt.qtbot import QtBot

from cuperiod.core.lightcurve import LightCurve
from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.main_window import MainWindow
from cuperiod.gui.qt import Qt, QtCore, QtWidgets
from cuperiod.gui.widgets.peaks_panel import PeaksTable

QSettings = QtCore.QSettings


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path: Path) -> None:
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path)
    )


def _peaks() -> list[Peak]:
    return [
        Peak(period=3.0, frequency=1 / 3.0, power=0.9, rank=1,
             extra={"t0": 2455000.75, "depth": 0.01}),
        Peak(period=2.24618, frequency=1 / 2.24618, power=0.6, rank=2,
             extra={"t0": 2455001.2835, "depth": 0.004}),
        Peak(period=1.5, frequency=1 / 1.5, power=0.3, rank=3,
             extra={"t0": 2455000.1, "depth": 0.002}),
    ]


def _periodogram() -> Periodogram:
    frequency = np.linspace(0.05, 2.0, 400)
    power = np.zeros_like(frequency)
    for peak in _peaks():
        power += peak.power * np.exp(-0.5 * ((frequency - peak.frequency) / 0.01) ** 2)
    return Periodogram.from_spectrum(
        method="BLS", backend="numpy", frequency=frequency, power=power,
        objective_sense="max", n_samples=200, baseline=40.0,
    )


def _window(qtbot: QtBot) -> MainWindow:
    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    window = MainWindow(app)
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    rng = np.random.default_rng(0)
    time = 2455000.0 + np.sort(rng.uniform(0.0, 40.0, 200))
    value = 1.0 + rng.normal(0.0, 0.001, 200)
    window._controller.set_light_curve(
        LightCurve.from_arrays(time, value, np.full(200, 0.001)), "src"
    )
    window._controller.periodogram_ready.emit(_periodogram())
    window._controller.peaks_ready.emit(_peaks())
    return window


def _marker(window: MainWindow, rank: int) -> object:
    spectrum = window._spectrum
    for spot in spectrum._markers.points():
        index = spot.data()
        if index is not None and spectrum._peaks[int(index)].rank == rank:
            return spot
    raise AssertionError(f"no marker for rank {rank}")


def _highlighted_ranks(table: PeaksTable) -> list[int]:
    rows = sorted({i.row() for i in table._table.selectedIndexes()})
    return [p.rank for p in (table._peak_for_row(r) for r in rows) if p is not None]


def _click_row(qtbot: QtBot, table: PeaksTable, rank: int) -> None:
    for row in range(table._table.rowCount()):
        peak = table._peak_for_row(row)
        if peak is not None and peak.rank == rank:
            rect = table._table.visualRect(table._table.model().index(row, 1))
            QtTest.QTest.mouseClick(
                table._table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center()
            )
            return
    raise AssertionError(f"no row for rank {rank}")


def test_marker_click_moves_the_highlight_and_keeps_the_epoch(qtbot: QtBot) -> None:
    window = _window(qtbot)
    folds: list[tuple[float, float]] = []
    window._controller.fold_changed.connect(lambda p, t0: folds.append((p, t0)))
    table = window._peaks_table
    assert _highlighted_ranks(table) == [1]

    window._spectrum._on_peak_clicked(None, [_marker(window, 2)])

    peaks = _peaks()
    assert _highlighted_ranks(table) == [2]
    assert folds[-1] == (peaks[1].period, peaks[1].extra["t0"])
    assert window._controller.state.selected_peak == peaks[1]


def test_the_previously_highlighted_row_is_never_a_dead_click(qtbot: QtBot) -> None:
    window = _window(qtbot)
    table = window._peaks_table
    window._spectrum._on_peak_clicked(None, [_marker(window, 2)])
    folds: list[tuple[float, float]] = []
    window._controller.fold_changed.connect(lambda p, t0: folds.append((p, t0)))

    _click_row(qtbot, table, 1)  # the row that was highlighted before

    peaks = _peaks()
    assert window._controller.state.selected_period == peaks[0].period
    assert folds == [(peaks[0].period, peaks[0].extra["t0"])]
    assert _highlighted_ranks(table) == [1]


def test_a_band_drag_off_every_peak_clears_the_highlight(qtbot: QtBot) -> None:
    window = _window(qtbot)
    table = window._peaks_table
    window._spectrum.period_selected.emit(7.77)
    assert _highlighted_ranks(table) == []
    _click_row(qtbot, table, 1)  # and the old row takes the selection back
    assert window._controller.state.selected_period == _peaks()[0].period


def test_peaks_table_is_named_and_its_menu_does_not_block(qtbot: QtBot) -> None:
    table = PeaksTable()
    qtbot.addWidget(table)
    table.set_peaks(_peaks())
    assert table._table.accessibleName() == "Peaks"
    assert table._table.accessibleDescription()
    assert table._table.toolTip() and table._table.statusTip()
    menu = table.context_menu()
    assert menu.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    texts = [a.text().split("\t")[0] for a in menu.actions()]
    assert texts == ["Copy selected rows", "Export CSV…"]
    table._table.selectRow(1)
    next(a for a in menu.actions() if a.text().startswith("Copy")).trigger()
    clip = QtWidgets.QApplication.clipboard().text().splitlines()
    assert clip[0].startswith("rank\tperiod (d)\tfrequency (c/d)")
    assert clip[1].startswith("2\t2.24618")
