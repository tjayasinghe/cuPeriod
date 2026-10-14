"""Tests for the GUI widget polish pass: info bar, peaks table, source browser,
spectrum view, phased view, and the pre-whitening result panels.
"""

from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest
from PySide6 import QtTest
from pytestqt.qtbot import QtBot

from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.meta import method_display_names, multiband_method_names
from cuperiod.gui.models import COMBINED_BAND, STACKED_BAND
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets
from cuperiod.gui.theme import palette
from cuperiod.gui.widgets._view_helpers import select_only_row
from cuperiod.gui.widgets.controls_panel import ControlsPanel
from cuperiod.gui.widgets.info_bar import RunInfoBar
from cuperiod.gui.widgets.peaks_panel import PeaksTable
from cuperiod.gui.widgets.solution_panel import SolutionPanel
from cuperiod.gui.widgets.source_panel import SourceBrowser
from cuperiod.gui.widgets.spacing_panel import SpacingPanel
from cuperiod.gui.widgets.spectrum_view import SpectrumView
from cuperiod.prewhiten.result import PreWhitenResult, Sinusoid


def _bump_periodogram() -> Periodogram:
    freq = np.linspace(0.1, 2.0, 500)
    power = np.exp(-((freq - 0.5) ** 2) / 0.001)
    return Periodogram.from_spectrum(
        method="GLS",
        backend="numpy",
        frequency=freq,
        power=power,
        objective_sense="max",
        n_samples=100,
        baseline=90.0,
    )


def _peaks() -> list[Peak]:
    return [
        Peak(period=2.0, frequency=0.5, power=10.0, rank=1, extra={"fap": 0.001}),
        Peak(period=1.0, frequency=1.0, power=8.0, rank=2, extra={"fap": 0.2}),
        Peak(period=5.0, frequency=0.2, power=3.0, rank=3, extra={"fap": 0.5}),
    ]


# -- info bar ------------------------------------------------------------------


def test_device_badge_elides_long_text_and_sets_tooltip(qtbot: QtBot) -> None:
    bar = RunInfoBar()
    qtbot.addWidget(bar)
    pg = _bump_periodogram()
    bar.set_result(pg)
    # a plain CPU backend badge should show the exact text (short enough not to elide)
    assert bar._device.text() == "device: CPU"
    assert bar._device.toolTip() == "device: CPU"

    long_name = "NVIDIA GeForce RTX 5070 Ti Super Extra Long Model Name Edition"
    bar._device.setText("placeholder")
    from cuperiod.gui.widgets.info_bar import _set_badge_text

    _set_badge_text(bar._device, f"device: {long_name}", max_width=100)
    # the full text is always recoverable via the tooltip...
    assert bar._device.toolTip() == f"device: {long_name}"
    # ...even though the visible label text is shorter (elided) and never loses its
    # leading characters (the observed bug: "evice: ..." with the leading "d" clipped).
    assert bar._device.text().startswith("device")
    assert len(bar._device.text()) < len(f"device: {long_name}")


def test_badge_tooltips_describe_meaning(qtbot: QtBot) -> None:
    bar = RunInfoBar()
    qtbot.addWidget(bar)
    assert "samples" in bar._samples.toolTip().lower()
    assert "baseline" in bar._baseline.toolTip().lower()
    assert "period" in bar._best.toolTip().lower()
    assert "cached" in bar._timing.toolTip().lower()


# -- peaks table -----------------------------------------------------------


def test_peaks_table_numeric_sort(qtbot: QtBot) -> None:
    table = PeaksTable()
    qtbot.addWidget(table)
    table.set_peaks(_peaks())
    period_col = table._columns.index("period (d)")
    table._table.sortItems(period_col)
    values = []
    for r in range(table._table.rowCount()):
        item = table._table.item(r, period_col)
        assert item is not None
        values.append(float(item.data(Qt.ItemDataRole.EditRole)))
    assert values == sorted(values)
    # numeric, not lexicographic: 5.0 must land after 1.0 and 2.0
    assert values == [1.0, 2.0, 5.0]


def test_peaks_table_min_width_hint(qtbot: QtBot) -> None:
    table = PeaksTable()
    qtbot.addWidget(table)
    assert table.minimumWidth() >= 360


def test_peaks_table_copy_selected_is_tsv(qtbot: QtBot) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    table = PeaksTable()
    qtbot.addWidget(table)
    table.set_peaks(_peaks())
    table._table.selectRow(0)
    table._copy_selected()
    clipboard = app.clipboard()
    text = clipboard.text()
    lines = text.splitlines()
    assert lines[0].split("\t") == table._columns
    assert len(lines) == 2  # header + one selected row
    assert "\t" in lines[1]


def test_peaks_table_export_csv(qtbot: QtBot, tmp_path) -> None:
    table = PeaksTable()
    qtbot.addWidget(table)
    table.set_peaks(_peaks())
    out = tmp_path / "peaks.csv"
    table.export_csv(str(out))
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == table._columns
    assert len(rows) == 1 + len(_peaks())


def test_peaks_table_header_tooltips(qtbot: QtBot) -> None:
    table = PeaksTable()
    qtbot.addWidget(table)
    table.set_peaks(_peaks())
    header_item = table._table.horizontalHeaderItem(table._columns.index("period (d)"))
    assert header_item is not None
    assert "period" in header_item.toolTip().lower()
    fap_item = table._table.horizontalHeaderItem(table._columns.index("fap"))
    assert fap_item is not None
    assert "false-alarm" in fap_item.toolTip().lower()


# -- source browser ----------------------------------------------------------


def test_source_browser_filter_updates_count(qtbot: QtBot) -> None:
    browser = SourceBrowser()
    qtbot.addWidget(browser)
    browser.set_sources(["alpha", "beta", "gamma"])
    assert browser._count_label.text() == "3 sources"
    browser._filter.setText("be")  # matches only "beta"
    assert browser._count_label.text().startswith("1 of 3")
    browser._filter.setText("")
    assert browser._count_label.text() == "3 sources"


def test_source_browser_item_tooltip(qtbot: QtBot) -> None:
    browser = SourceBrowser()
    qtbot.addWidget(browser)
    browser.set_sources(["a very long source label that might get elided"])
    item = browser._list.item(0)
    assert item is not None
    assert "very long source label" in item.toolTip()


# -- spectrum view -----------------------------------------------------------


def test_spectrum_export_csv_writes_expected_columns(qtbot: QtBot, tmp_path) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.set_periodogram(_bump_periodogram())
    out = tmp_path / "spectrum.csv"
    view.export_csv(str(out))
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["frequency", "period", "power"]
    assert len(rows) == 1 + view._pg.size


class _HoveredPoint:
    """Stand-in for the pyqtgraph spot under the cursor."""

    def __init__(self, index: int | None) -> None:
        self._index = index

    def data(self) -> int | None:
        return self._index


def test_hiding_the_peaks_takes_their_hover_label_with_them(qtbot: QtBot) -> None:
    # Regression: the hover label is anchored to a marker, but nothing hid it when the
    # markers went away — so unchecking "peaks" (the natural move right after hovering
    # one to read it) left the label stranded over an empty plot.
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    pg_result = _bump_periodogram()
    view.set_periodogram(pg_result)
    view.set_peaks(pg_result.best_periods(3))

    view._on_peak_hovered(view._markers, [_HoveredPoint(0)])
    assert view._hover_text.isVisible()

    view._show_peaks.setChecked(False)
    assert not view._hover_text.isVisible()
    assert len(view._markers.data) == 0

    # Same for a new result, and for an axis switch that moves the anchor.
    view._show_peaks.setChecked(True)
    view.set_peaks(pg_result.best_periods(3))
    view._on_peak_hovered(view._markers, [_HoveredPoint(0)])
    assert view._hover_text.isVisible()
    view.set_periodogram(pg_result)
    assert not view._hover_text.isVisible()

    view.set_peaks(pg_result.best_periods(3))
    view._on_peak_hovered(view._markers, [_HoveredPoint(0)])
    assert view._hover_text.isVisible()
    view._xaxis_combo.setCurrentText("period")
    assert not view._hover_text.isVisible()

    # A glow halo carries no index and must not leave another peak's label up.
    view._on_peak_hovered(view._markers, [_HoveredPoint(0)])
    assert view._hover_text.isVisible()
    view._on_peak_hovered(view._markers, [_HoveredPoint(None)])
    assert not view._hover_text.isVisible()


def test_peaks_toggle_hides_and_restores_every_marker(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    frequency = np.linspace(0.2, 20.0, 6000)
    power = sum(
        a * np.exp(-((frequency - f) ** 2) / 0.002)
        for f, a in ((3.0, 1.0), (7.0, 0.6), (11.0, 0.35))
    )
    pg_result = Periodogram.from_spectrum(
        method="GLS", backend="numpy", frequency=frequency, power=power,
        objective_sense="max", n_samples=500, baseline=40.0,
    )
    peaks = pg_result.best_periods(3)
    assert len(peaks) == 3
    view.set_periodogram(pg_result)
    view.set_peaks(peaks)
    view.set_selected_period(peaks[1].period)
    populated = len(view._markers.data)
    assert populated == 5  # three peaks plus the best-peak and selected halos

    view._show_peaks.setChecked(False)
    assert len(view._markers.data) == 0
    # Nothing that happens while it is off may bring them back.
    view.set_selected_period(peaks[2].period)
    view.set_peaks(peaks)
    view._logy.setChecked(True)
    assert len(view._markers.data) == 0

    view._show_peaks.setChecked(True)
    assert len(view._markers.data) == populated


def test_selection_band_follows_the_peaks_toggle(qtbot: QtBot) -> None:
    # The band shades a peak, so it belongs to the peak layer: leaving it behind was
    # the one bit of clutter the toggle is usually turned off to be rid of.
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    pg_result = _bump_periodogram()
    view.set_periodogram(pg_result)
    peak = pg_result.best_periods(1)[0]
    view.set_peaks([peak])
    view.set_selected_period(peak.period)
    assert view._sel_band.isVisible()

    view._show_peaks.setChecked(False)
    assert not view._sel_band.isVisible()
    # The selection itself is not lost, so the fold does not change under the user.
    assert view._sel_period == peak.period
    # Nor does re-selecting while hidden bring the band back.
    view.set_selected_period(peak.period)
    assert not view._sel_band.isVisible()

    view._show_peaks.setChecked(True)
    assert view._sel_band.isVisible()

    # With peaks shown but nothing selected there is still no band.
    view.clear_selection()
    assert not view._sel_band.isVisible()
    view._show_peaks.setChecked(False)
    view._show_peaks.setChecked(True)
    assert not view._sel_band.isVisible()


def test_spectrum_reset_button_label(qtbot: QtBot) -> None:
    # "Reset", not "Reset view": with the pre-whitening overlay toggles shown the
    # longer label pushed the toolbar past the dock width and Qt squeezed the buttons.
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    buttons = view.findChildren(QtWidgets.QPushButton)
    labels = [b.text() for b in buttons]
    assert "Reset" in labels


def test_spectrum_selected_marker_is_highlighted(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    freq = np.linspace(0.1, 2.0, 500)
    power = np.exp(-((freq - 0.5) ** 2) / 0.001) + 0.6 * np.exp(
        -((freq - 1.5) ** 2) / 0.001
    )
    pg_result = Periodogram.from_spectrum(
        method="GLS",
        backend="numpy",
        frequency=freq,
        power=power,
        objective_sense="max",
        n_samples=100,
        baseline=90.0,
    )
    view.set_periodogram(pg_result)
    peaks = pg_result.best_periods(5)
    view.set_peaks(peaks)
    assert len(peaks) > 1
    # select a non-best peak and confirm it is reported as the highlighted index
    other = peaks[1]
    view.set_selected_period(other.period)
    idx = view._selected_index()
    assert idx is not None
    assert view._peaks[idx].period == other.period


# -- controls panel: band sentinel --------------------------------------------


def test_current_band_real_band_named_combined(qtbot: QtBot) -> None:
    panel = ControlsPanel()
    qtbot.addWidget(panel)
    multi_methods = multiband_method_names()
    assert multi_methods  # sanity: at least one multiband-capable method exists
    panel._method_combo.setCurrentText(multi_methods[0])
    panel.set_bands(["combined", "g", "r"])
    # first entry is always the synthetic "combined (all bands)" sentinel
    assert panel._band_combo.itemData(0) == "__combined__"
    panel._band_combo.setCurrentIndex(0)
    # The synthetic entry hands over the controller's token, never the bare word, so
    # a real band named "combined" cannot stand in for the all-band analysis.
    assert panel.current_band() == COMBINED_BAND
    # selecting the *real* band literally named "combined" must read its own item data
    real_idx = panel._band_combo.findText("combined", Qt.MatchFlag.MatchExactly)
    assert real_idx >= 1  # not the sentinel at index 0
    assert panel._band_combo.itemData(real_idx) == "combined"
    panel._band_combo.setCurrentIndex(real_idx)
    assert panel.current_band() == "combined"


def test_current_band_real_band_prefixed_like_sentinel(qtbot: QtBot) -> None:
    # Regression: a real band merely *starting with* "stacked" used to be truncated to
    # "stacked" by the old `text.startswith("stacked")` heuristic.
    panel = ControlsPanel()
    qtbot.addWidget(panel)
    multi_methods = set(multiband_method_names())
    single_band_methods = [m for m in method_display_names() if m not in multi_methods]
    assert single_band_methods
    panel._method_combo.setCurrentText(single_band_methods[0])
    panel.set_bands(["stacked_g", "r"])
    idx = panel._band_combo.findText("stacked_g", Qt.MatchFlag.MatchExactly)
    assert idx >= 0
    panel._band_combo.setCurrentIndex(idx)
    assert panel.current_band() == "stacked_g"  # not truncated to "stacked"
    # the actual "stacked (all bands)" sentinel is still the last entry
    last = panel._band_combo.count() - 1
    assert panel._band_combo.itemData(last) == "__stacked__"
    panel._band_combo.setCurrentIndex(last)
    assert panel.current_band() == STACKED_BAND


# -- GUI-C-8: the peaks table keeps its own number formats -----------------------


def _bls_peaks() -> list[Peak]:
    return [
        Peak(period=3.123456789, frequency=1 / 3.123456789, power=0.123456789,
             rank=1, extra={"t0": 2459001.9060481642, "depth": 0.0123}),
        Peak(period=12345.678901, frequency=1 / 12345.678901, power=0.0987,
             rank=2, extra={"t0": 2459000.3791642073, "depth": float("nan")}),
    ]


def test_peaks_table_shows_its_own_formats_in_any_locale(qtbot: QtBot) -> None:
    # Regression: setData(EditRole, float) replaced the formatted text, so the
    # delegate painted QLocale's 6-digit rendering: t0 = 2459001.906 as "2.459e+06",
    # 12345.678901 as "12,345.7" (en_US) or "12.345,7" (de_DE).
    table = PeaksTable()
    qtbot.addWidget(table)
    table.set_peaks(_bls_peaks())
    col = table._columns.index
    assert table._table.item(0, col("t0")).text() == "2459001.90605"
    assert table._table.item(1, col("period (d)")).text() == "12345.6789"
    assert table._table.item(1, col("depth")).text() == "—"  # NaN
    delegate = table._table.itemDelegate()
    german = QtCore.QLocale(
        QtCore.QLocale.Language.German, QtCore.QLocale.Country.Germany
    )
    for locale in (german, QtCore.QLocale(QtCore.QLocale.Language.English)):
        for c in range(table._table.columnCount()):
            item = table._table.item(0, c)
            shown = delegate.displayText(item.data(Qt.ItemDataRole.DisplayRole), locale)
            assert shown == item.text()


def test_peaks_table_sorts_two_digit_ranks_and_periods_numerically(
    qtbot: QtBot,
) -> None:
    # Regression: a rank (or any value with <= 6 significant digits) stayed a string,
    # so 12 peaks sorted 1, 10, 11, 12, 2, ...
    table = PeaksTable()
    qtbot.addWidget(table)
    periods = [2.0, 10.0, 5.0, 100.0, 0.25, 1.0, 11.0, 3.5, 30.0, 40.0, 7.0, 9.0]
    peaks = [
        Peak(period=p, frequency=1 / p, power=1.0 / (i + 1), rank=i + 1)
        for i, p in enumerate(periods)
    ]
    table.set_peaks(peaks)
    table._table.sortItems(0, Qt.SortOrder.DescendingOrder)
    table._table.sortItems(0)
    ranks = [int(table._table.item(r, 0).text()) for r in range(12)]
    assert ranks == list(range(1, 13))
    period_col = table._columns.index("period (d)")
    table._table.sortItems(period_col)
    shown = [float(table._table.item(r, period_col).text()) for r in range(12)]
    assert shown == sorted(periods)
    # Selection still maps rows to their peaks after sorting.
    table._table.selectRow(0)
    picked = table._peak_for_row(table._table.currentRow())
    assert picked is not None and picked.period == 0.25


def test_peaks_copy_and_csv_keep_full_precision(qtbot: QtBot, tmp_path) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    table = PeaksTable()
    qtbot.addWidget(table)
    peaks = _bls_peaks()
    table.set_peaks(peaks)
    out = tmp_path / "peaks.csv"
    table.export_csv(str(out))
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        peak = next(p for p in peaks if p.rank == int(row["rank"]))
        assert float(row["period (d)"]) == peak.period
        assert float(row["frequency (c/d)"]) == peak.frequency
        assert float(row["t0"]) == peak.extra["t0"]
    table._table.selectRow(0)
    table._copy_selected()
    cells = app.clipboard().text().splitlines()[1].split("\t")
    assert float(cells[table._columns.index("t0")]) == peaks[0].extra["t0"]


def test_peaks_csv_writes_numpy_scalars_as_plain_numbers(
    qtbot: QtBot, tmp_path
) -> None:
    # repr(np.float64(2.5)) is "np.float64(2.5)" under NumPy 2: not a number to Excel.
    table = PeaksTable()
    qtbot.addWidget(table)
    table.set_peaks([
        Peak(period=np.float64(2.5), frequency=np.float64(0.4),
             power=np.float32(0.75), rank=1,
             extra={"t0": np.float64(2459000.5), "n_transits": np.int64(3)}),
    ])
    out = tmp_path / "peaks.csv"
    table.export_csv(str(out))
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[1] == ["1", "2.5", "0.4", "0.75", "2459000.5", "3"]


def test_peaks_rows_map_to_their_own_peak_when_ranks_repeat(qtbot: QtBot) -> None:
    # Rows used to be matched to peaks by rank, so of two peaks sharing a rank (two
    # merged results) only the first could ever be selected.
    table = PeaksTable()
    qtbot.addWidget(table)
    peaks = [
        Peak(period=2.0, frequency=0.5, power=1.0, rank=1),
        Peak(period=4.0, frequency=0.25, power=0.9, rank=1),
    ]
    table.set_peaks(peaks)
    table._table.sortItems(1)  # by period, ascending: 2.0 then 4.0
    picked: list[Peak] = []
    table.peak_selected.connect(picked.append)
    select_only_row(table._table, 1)
    select_only_row(table._table, 0)
    assert [p.period for p in picked] == [4.0, 2.0]
    table.select_period(4.0)
    assert {i.row() for i in table._table.selectedIndexes()} == {1}


def test_peaks_table_names_the_statistic_and_lists_frequency(qtbot: QtBot) -> None:
    from cuperiod.gui.widgets.peaks_panel import statistic_name

    table = PeaksTable()
    qtbot.addWidget(table)
    table.set_peaks(_peaks())
    assert "frequency (c/d)" in table._columns
    pdm = Periodogram.from_spectrum(
        method="PDM", backend="numpy", frequency=np.linspace(0.1, 2.0, 50),
        power=np.linspace(1.0, 0.5, 50), objective_sense="min", n_samples=10,
        baseline=10.0,
    )
    assert statistic_name(pdm) == "theta"
    assert statistic_name(_bump_periodogram()) == "power"
    table.set_statistic(statistic_name(pdm))
    header = table._table.horizontalHeaderItem(3)
    assert header is not None and header.text() == "theta"
    table.set_peaks(_peaks())  # kept for the next result's peaks
    assert table._columns[3] == "theta"


def test_result_tables_offer_visible_copy_and_export(qtbot: QtBot) -> None:
    for widget in (PeaksTable(), SolutionPanel()):
        qtbot.addWidget(widget)
        labels = {b.text() for b in widget.findChildren(QtWidgets.QPushButton)}
        assert {"Copy", "Export CSV…"} <= labels


def test_column_chooser_hides_and_restores_columns(qtbot: QtBot) -> None:
    from cuperiod.gui.widgets._view_helpers import column_menu

    table = PeaksTable()
    qtbot.addWidget(table)
    table.set_peaks(_peaks())
    header = table._table.horizontalHeader()
    assert header.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    actions = column_menu(table._table).actions()  # what a header right-click shows
    assert [a.text() for a in actions] == table._columns
    actions[2].setChecked(False)
    assert table._table.isColumnHidden(2)
    actions[2].setChecked(True)
    assert not table._table.isColumnHidden(2)


# -- GUI-C-12 / A11Y-01: keyboard behaviour of the result tables -----------------


def test_result_tables_do_not_trap_tab(qtbot: QtBot) -> None:
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    layout = QtWidgets.QVBoxLayout(host)
    table = PeaksTable()
    after = QtWidgets.QLineEdit()
    layout.addWidget(table)
    layout.addWidget(after)
    table.set_peaks(_peaks())
    host.show()
    host.activateWindow()
    qtbot.waitActive(host)
    solution = SolutionPanel()
    qtbot.addWidget(solution)
    assert not table._table.tabKeyNavigation()
    assert not solution._table.tabKeyNavigation()
    table._table.setFocus()
    qtbot.waitUntil(table._table.hasFocus, timeout=2000)
    QtTest.QTest.keyClick(table._table, Qt.Key.Key_Tab)
    assert QtWidgets.QApplication.focusWidget() is after


def test_copy_shortcut_belongs_to_its_own_table(qtbot: QtBot) -> None:
    # Regression: the Ctrl+C shortcuts were window-wide, so Ctrl+C in a sibling list
    # copied the peaks table, and with both tables visible neither copied anything.
    app = QtWidgets.QApplication.instance()
    assert app is not None
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    layout = QtWidgets.QHBoxLayout(host)
    sources = QtWidgets.QListWidget()
    sources.addItem("ASASSN-V J000000.00+000000.0")
    peaks = PeaksTable()
    solution = SolutionPanel()
    for widget in (sources, peaks, solution):
        layout.addWidget(widget)
    peaks.set_peaks(_peaks())
    solution.set_solution(_synthetic_solution(3))
    host.show()
    host.activateWindow()
    qtbot.waitActive(host)

    def ctrl_c(widget: QtWidgets.QWidget) -> str:
        app.clipboard().setText("<<untouched>>")
        widget.setFocus()
        qtbot.waitUntil(widget.hasFocus, timeout=2000)
        QtTest.QTest.keyClick(widget, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
        return app.clipboard().text()

    sources.setCurrentRow(0)
    assert ctrl_c(sources) == "ASASSN-V J000000.00+000000.0"
    # (not selectRow: QTest leaves Ctrl in the modifier state, so it would toggle)
    select_only_row(peaks._table, 0)
    assert ctrl_c(peaks._table).startswith("rank\t")
    select_only_row(solution._table, 0)
    assert ctrl_c(solution._table).startswith("ID\t")


# -- the frequency solution panel -------------------------------------------------


def _synthetic_solution(
    n_components: int = 3, stop_reason: str = "S/N 3.21 < 4"
) -> PreWhitenResult:
    components = tuple(
        Sinusoid(
            rank=i,
            label=f"F{i}",
            frequency=10.0 + 1.37 * i,
            frequency_error=1e-4,
            amplitude=0.01 / i,
            amplitude_error=1e-4,
            phase=0.5,
            phase_error=0.01,
            snr=20.0 / i,
            spectrum_amplitude=0.01 / i,
        )
        for i in range(1, n_components + 1)
    )
    return PreWhitenResult(
        components=components,
        combinations=(),
        offset=0.0,
        offset_error=0.0,
        t_ref=0.0,
        time=np.zeros(0),
        residuals=np.zeros(0),
        n_samples=500,
        baseline=30.0,
        stop_reason=stop_reason,
        n_iterations=n_components,
        n_pruned=0,
        rms=0.002049,
        chi2=500.0,
        reduced_chi2=1.05,
        bic=-100.0,
        correlation_factor=1.0,
        uncertainty_method="Montgomery & O'Donoghue <analytic>",
        backend="numpy",
    )


def _rendered(label: QtWidgets.QLabel) -> str:
    document = QtGui.QTextDocument()
    document.setHtml(label.text())
    return document.toPlainText()


@pytest.mark.parametrize(
    "reason",
    [
        "S/N 3.21 < 4; 2 insignificant components pruned",
        "amplitude 0.0001234 < 0.001",
        "S/N undefined < 4",
        "FAP 0.2 > 0.01",
    ],
)
def test_summary_shows_the_stop_reason_verbatim(qtbot: QtBot, reason: str) -> None:
    # Regression: the reason went into the label's HTML unescaped, so "< 4<br>" was
    # parsed as a tag and the threshold and line break vanished.
    panel = SolutionPanel()
    qtbot.addWidget(panel)
    panel.set_solution(_synthetic_solution(3, stop_reason=reason))
    assert panel._summary.textFormat() == Qt.TextFormat.RichText
    text = _rendered(panel._summary)
    assert f"stopped: {reason}" in text
    assert "<analytic>" in text
    assert len(text.splitlines()) == 3


def test_solution_id_column_sorts_in_extraction_order(qtbot: QtBot) -> None:
    panel = SolutionPanel()
    qtbot.addWidget(panel)
    panel.set_solution(_synthetic_solution(12))
    panel._table.sortItems(0)
    labels = [panel._table.item(r, 0).text() for r in range(12)]
    assert labels == [f"F{i}" for i in range(1, 13)]
    panel._table.sortItems(0, Qt.SortOrder.DescendingOrder)
    labels = [panel._table.item(r, 0).text() for r in range(12)]
    assert labels == [f"F{i}" for i in range(12, 0, -1)]
    with qtbot.waitSignal(panel.component_selected, timeout=2000) as blocker:
        panel._table.selectRow(2)
    assert blocker.args[0].label == "F10"


def test_solution_panel_follows_a_selection_made_elsewhere(qtbot: QtBot) -> None:
    panel = SolutionPanel()
    qtbot.addWidget(panel)
    solution = _synthetic_solution(4)
    panel.set_solution(solution)
    emitted: list[object] = []
    panel.component_selected.connect(emitted.append)
    panel.select_period(solution.components[2].period)  # an F3 marker click
    rows = sorted({i.row() for i in panel._table.selectedIndexes()})
    assert [panel._table.item(r, 0).text() for r in rows] == ["F3"]
    panel.select_period(123.456)  # a band drag off every component
    assert panel._table.selectedIndexes() == []
    assert emitted == []


def test_solution_columns_lead_with_what_identifies_a_mode(qtbot: QtBot) -> None:
    panel = SolutionPanel()
    qtbot.addWidget(panel)
    panel.set_solution(_synthetic_solution(2))
    headers = [
        panel._table.horizontalHeaderItem(i).text()
        for i in range(panel._table.columnCount())
    ]
    assert headers[0] == "ID"
    assert headers.index("S/N") < headers.index("phase")
    assert headers.index("combination") < headers.index("phase")


# -- GUI-C-14: one CSV schema --------------------------------------------------------


def test_solution_csv_has_one_schema_with_or_without_components(
    qtbot: QtBot, tmp_path
) -> None:
    from cuperiod.gui.widgets.solution_panel import CSV_FIELDS

    panel = SolutionPanel()
    qtbot.addWidget(panel)
    headers = []
    for state in ("never run", "populated", "zero components", "cleared"):
        if state == "populated":
            panel.set_solution(_synthetic_solution(2))
        elif state == "zero components":
            panel.set_solution(_synthetic_solution(0))
        elif state == "cleared":
            panel.clear()
        out = tmp_path / f"{state}.csv"
        panel.export_csv(str(out))
        headers.append(out.read_text(encoding="utf-8").splitlines()[0])
    assert len(set(headers)) == 1
    assert headers[0].split(",") == list(CSV_FIELDS)
    component = _synthetic_solution(1).components[0]
    assert tuple(component.to_dict()) == CSV_FIELDS  # tracks Sinusoid.to_dict


def test_amplitude_spectrum_export_names_its_column(qtbot: QtBot, tmp_path) -> None:
    from cuperiod.gui.widgets.spectrum_view import PREWHITEN_METHOD

    view = SpectrumView("dark")
    qtbot.addWidget(view)
    freq = np.linspace(0.1, 2.0, 50)
    view.set_periodogram(
        Periodogram.from_spectrum(
            method=PREWHITEN_METHOD, backend="numpy", frequency=freq,
            power=np.ones_like(freq), objective_sense="max", n_samples=10,
            baseline=10.0,
        )
    )
    out = tmp_path / "amplitude.csv"
    view.export_csv(str(out))
    assert out.read_text(encoding="utf-8").splitlines()[0] == (
        "frequency,period,amplitude"
    )


# -- GUI-C-13: the echelle diagram follows the theme ----------------------------------


def _gmode_solution() -> PreWhitenResult:
    base = _synthetic_solution(0)
    spacing = 2600.0 / 86400.0  # days
    periods = 0.9 + spacing * np.arange(12)
    components = tuple(
        Sinusoid(
            rank=i + 1, label=f"F{i + 1}", frequency=1.0 / p, frequency_error=1e-5,
            amplitude=0.005, amplitude_error=1e-4, phase=0.0, phase_error=0.01,
            snr=30.0, spectrum_amplitude=0.005,
        )
        for i, p in enumerate(periods)
    )
    return replace(base, components=components)


def _echelle_colours(panel: SpacingPanel) -> tuple[set[str], set[str]]:
    data = panel._echelle_points.data
    brushes = {QtGui.QBrush(b).color().name() for b in data["brush"]}
    pens = {QtGui.QPen(p).color().name() for p in data["pen"]}
    return brushes, pens


def test_echelle_points_follow_the_theme(qtbot: QtBot) -> None:
    panel = SpacingPanel("dark")
    qtbot.addWidget(panel)
    panel.set_solution(_gmode_solution())
    panel.search()
    assert panel._echelle_points.data.size > 0
    box = panel._echelle.getPlotItem().vb
    box.setXRange(0.0, 500.0, padding=0.0)  # the user zooms
    light = palette("light")
    panel.apply_theme(light)
    brushes, pens = _echelle_colours(panel)
    allowed = {QtGui.QColor(light.best_peak).name(), QtGui.QColor(light.muted).name()}
    assert brushes <= allowed
    assert QtGui.QColor(light.best_peak).name() in brushes
    assert pens == {QtGui.QColor(light.plot_bg).name()}
    assert box.viewRange()[0] == pytest.approx([0.0, 500.0])  # the zoom survives
    panel.clear()
    panel.apply_theme(palette("dark"))
    assert panel._echelle_points.data.size == 0  # nothing resurrected


def test_spacing_error_message_is_shown_verbatim(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cuperiod.gui.widgets.spacing_panel as spacing_module

    panel = SpacingPanel("dark")
    qtbot.addWidget(panel)
    panel.set_solution(_gmode_solution())

    def boom(*_a: object, **_k: object) -> None:
        raise ValueError("need <b>more</b> modes & a spacing < 5000 s")

    monkeypatch.setattr(spacing_module, "spacing_spectrum", boom)
    panel.search()
    assert "need <b>more</b> modes & a spacing < 5000 s" in _rendered(panel._summary)
