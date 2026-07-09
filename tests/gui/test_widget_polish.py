"""Tests for the GUI widget polish pass: info bar, peaks table, source browser,
spectrum view, and phased view.
"""

from __future__ import annotations

import csv

import numpy as np
from pytestqt.qtbot import QtBot

from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.meta import method_display_names, multiband_method_names
from cuperiod.gui.qt import Qt, QtWidgets
from cuperiod.gui.widgets.controls_panel import ControlsPanel
from cuperiod.gui.widgets.info_bar import RunInfoBar
from cuperiod.gui.widgets.peaks_panel import PeaksTable
from cuperiod.gui.widgets.source_panel import SourceBrowser
from cuperiod.gui.widgets.spectrum_view import SpectrumView


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


def test_spectrum_reset_button_label(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    buttons = view.findChildren(QtWidgets.QPushButton)
    labels = [b.text() for b in buttons]
    assert "Reset view" in labels


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
    assert panel.current_band() == "combined"
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
    assert panel.current_band() == "stacked"
