"""The peaks table: the N most significant peaks, click a row to investigate one.

Lists rank / period / frequency / the method's statistic plus whatever method-specific
extras a peak carries (BLS depth, duration, t0, SDE; GLS false-alarm probability).
Selecting a row emits :attr:`peak_selected`, which the controller turns into the active
period driving the spectrum line and the phased view; :meth:`PeaksTable.select_period`
moves the highlight when the selection changes elsewhere (a marker click, a band drag).
Rows can be sorted numerically by any column, copied as TSV (selection or the whole
table) and exported to CSV, both with full precision; the header's right-click menu
shows or hides columns.

Each column has its own display format — enough digits for a period, fixed decimals for
an epoch such as a BLS ``t0`` near JD 2459001.9 — and the shown text is exactly that
format, independent of the system locale.
"""

from __future__ import annotations

import csv
import math
from typing import TYPE_CHECKING

import numpy as np

from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, Signal
from cuperiod.gui.widgets._view_helpers import (
    NumericItem,
    ask_save_path,
    configure_result_table,
    install_column_chooser,
    report_failure,
    select_only_row,
    table_action_bar,
)

if TYPE_CHECKING:
    from cuperiod.gui.dialogs import DialogProvider

#: Column header tooltips: the fixed columns plus known method-specific extras. Any
#: extra key not listed here still gets a column, just without a tooltip.
_COLUMN_TOOLTIPS: dict[str, str] = {
    "rank": "Significance rank (1 = most significant)",
    "period (d)": "Period, in days",
    "frequency (c/d)": "Frequency, in cycles per day",
    "power": "The method's objective/statistic value at this peak",
    "fap": "False-alarm probability (GLS) — lower is more significant",
    "sde": "Signal detection efficiency (BLS/TLS) — higher is more significant",
    "depth": "Transit depth (BLS/TLS), in the light curve's flux units",
    "duration": "Transit duration (BLS/TLS), in days",
    "t0": "Transit epoch / reference time (BLS/TLS), in days",
    "depth_snr": "Transit depth signal-to-noise ratio (BLS/TLS)",
}

#: Minimum table/panel width so rank + period + power are readable without resizing.
_MIN_TABLE_WIDTH = 360

#: The fixed leading columns (the statistic column is renamed per method).
_FIXED = ("rank", "period (d)", "frequency (c/d)")

#: Statistic names of the minimise methods, whose "power" is a dispersion.
_MIN_STATISTICS = {"PDM": "theta", "CE": "entropy", "STRINGLENGTH": "string length"}

#: Item role of the rank cell holding the row's position in the peak list.
_POSITION_ROLE = Qt.ItemDataRole.UserRole + 1

#: Extras that are epochs: shown with fixed decimals, since ``.6g`` of a Julian date
#: (``2.459e+06``) is useless.
_EPOCH_KEYS = frozenset({"t0", "epoch", "t_ref"})


def statistic_name(pg_result: Periodogram) -> str:
    """The column name of a periodogram's statistic (``power``, ``theta``, ...)."""
    if pg_result.objective_sense == "min":
        return _MIN_STATISTICS.get(pg_result.method.upper(), "statistic")
    return "power"


def _format(key: str, value: float) -> str:
    """Display text of one cell; non-finite values show as an em dash."""
    if not math.isfinite(value):
        return "—"
    if key in _EPOCH_KEYS:
        return f"{value:.5f}"
    if key in {"period (d)", "frequency (c/d)"}:
        return f"{value:.9g}"
    return f"{value:.6g}"


def _full_precision(value: object) -> str:
    """Copy/CSV text: the shortest repr that round-trips (ints stay ints).

    NumPy scalars are converted first: ``repr(np.float64(3.0))`` is
    ``'np.float64(3.0)'`` under NumPy 2, which no spreadsheet reads as a number.
    """
    if isinstance(value, bool | np.bool_):
        return str(int(value))
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return repr(float(value))
    return str(value)


class PeaksTable(QtWidgets.QWidget):
    """A table of the best peaks; row selection emits the chosen :class:`Peak`."""

    peak_selected = Signal(object)  # Peak
    #: The CSV export could not be written; carries a message for the user.
    export_failed = Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._peaks: list[Peak] = []
        self._columns: list[str] = []
        self._extra_keys: list[str] = []
        self._statistic = "power"
        self.setMinimumWidth(_MIN_TABLE_WIDTH)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(
            table_action_bar(self._copy_selected, self._export_csv, "peaks")
        )
        self._table = QtWidgets.QTableWidget()
        self._table.setMinimumWidth(_MIN_TABLE_WIDTH)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setSortingEnabled(True)
        header = self._table.verticalHeader()
        if header is not None:
            header.setVisible(False)
        self._table.setObjectName("peaks_table")
        # Named and explained for screen readers and keyboard users (A11Y-04/18):
        # a tooltip alone needs a mouse.
        self._table.setAccessibleName("Peaks")
        self._table.setAccessibleDescription(
            "The most significant peaks of the periodogram. Selecting a row makes "
            "its period the active one; Ctrl+C copies the selected rows."
        )
        self._table.setToolTip(
            "Select a row to fold on that peak; right-click for copy and export, "
            "right-click the header to choose columns"
        )
        self._table.setStatusTip("The best peaks: select one to fold on its period")
        self._table.itemSelectionChanged.connect(self._on_selection)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        configure_result_table(self._table, self._copy_selected)
        install_column_chooser(self._table)

        layout.addWidget(self._table)

    def set_statistic(self, name: str) -> None:
        """Name the statistic column (``power``, ``theta``, ...; see statistic_name)."""
        self._statistic = name
        if self._columns:
            column = len(_FIXED)
            self._columns[column] = name
            item = self._table.horizontalHeaderItem(column)
            if item is not None:
                item.setText(name)

    def set_peaks(self, peaks: list[Peak]) -> None:
        """Populate the table from ``peaks`` (rank/period/power + method extras)."""
        self._peaks = list(peaks)
        extra_keys: list[str] = []
        for peak in self._peaks:
            for key in peak.extra:
                if key not in extra_keys:
                    extra_keys.append(key)
        self._extra_keys = extra_keys
        columns = [*_FIXED, self._statistic, *extra_keys]
        self._columns = columns

        self._table.setSortingEnabled(False)
        self._table.blockSignals(True)
        self._table.clear()
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.setRowCount(len(self._peaks))
        for r, peak in enumerate(self._peaks):
            rank_item = NumericItem(float(peak.rank), str(peak.rank))
            rank_item.setData(Qt.ItemDataRole.UserRole, peak.rank)
            rank_item.setData(_POSITION_ROLE, r)
            self._table.setItem(r, 0, rank_item)
            fixed = [peak.period, peak.frequency, peak.power]
            for c, value in enumerate(fixed, start=1):
                text = _format(columns[c], value)
                self._table.setItem(r, c, NumericItem(value, text))
            for c, key in enumerate(extra_keys, start=len(fixed) + 1):
                self._table.setItem(r, c, self._extra_item(peak, key))
        h_header = self._table.horizontalHeader()
        if h_header is not None:
            h_header.setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            self._set_header_tooltips(columns)
        self._table.resizeColumnsToContents()
        if self._peaks:
            select_only_row(self._table, 0)
        self._table.blockSignals(False)
        self._table.setSortingEnabled(True)

    @staticmethod
    def _extra_item(peak: Peak, key: str) -> QtWidgets.QTableWidgetItem:
        """A method-extra cell: numeric if the value is a number, else plain text."""
        if key not in peak.extra:
            return QtWidgets.QTableWidgetItem("")
        raw = peak.extra[key]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return QtWidgets.QTableWidgetItem(str(raw))
        return NumericItem(value, _format(key, value))

    def _set_header_tooltips(self, columns: list[str]) -> None:
        h_header = self._table.horizontalHeader()
        if h_header is None:
            return
        for c, name in enumerate(columns):
            key = "power" if c == len(_FIXED) else name
            tooltip = _COLUMN_TOOLTIPS.get(key, name)
            header_item = self._table.horizontalHeaderItem(c)
            if header_item is not None:
                header_item.setToolTip(tooltip)

    def clear(self) -> None:
        """Empty the table (e.g. when a new curve is being loaded)."""
        self.set_peaks([])

    # -- selection ---------------------------------------------------------------
    def _on_selection(self) -> None:
        row = self._table.currentRow()
        peak = self._peak_for_row(row)
        if peak is not None:
            self.peak_selected.emit(peak)

    def _peak_for_row(self, row: int) -> Peak | None:
        # By list position, not by rank: the rows are sortable, and peaks from merged
        # results can share a rank, which made every row after the first unreachable.
        item = self._table.item(row, 0)
        if item is None:
            return None
        position = item.data(_POSITION_ROLE)
        if isinstance(position, int) and 0 <= position < len(self._peaks):
            return self._peaks[position]
        return None

    def select_period(self, period: float) -> None:
        """Highlight the row of the peak at ``period`` without emitting anything.

        Connected to the controller's ``period_changed``, so a selection made elsewhere
        (a marker click, a band drag) moves the highlight: otherwise the old row stayed
        highlighted and clicking it was a dead click, the Qt selection not changing. A
        period that is no listed peak clears the highlight. A row that is already
        selected is left alone, so a multi-row selection survives the echo of its own
        current row.
        """
        target = None
        for row in range(self._table.rowCount()):
            peak = self._peak_for_row(row)
            if peak is not None and math.isclose(peak.period, period, rel_tol=1e-9):
                target = row
                break
        selected = {index.row() for index in self._table.selectedIndexes()}
        if target is not None and target in selected:
            return
        self._table.blockSignals(True)
        select_only_row(self._table, target)
        self._table.blockSignals(False)

    # -- copy / export -------------------------------------------------------
    def context_menu(self) -> QtWidgets.QMenu:
        """The table's right-click menu, deleted when it closes.

        Shown with ``popup()``: a modal ``exec()`` would block the event loop (and a
        headless session) until it closed.
        """
        menu = QtWidgets.QMenu(self._table)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        menu.setAccessibleName("Peaks table actions")
        copy_key = QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy).toString(
            QtGui.QKeySequence.SequenceFormat.NativeText
        )
        copy_action = menu.addAction(f"Copy selected rows\t{copy_key}")
        copy_action.triggered.connect(self._copy_selected)
        export_action = menu.addAction("Export CSV…")
        export_action.triggered.connect(self._export_csv)
        return menu

    def _on_context_menu(self, pos: QtCore.QPoint) -> None:
        viewport = self._table.viewport()
        anchor = viewport if viewport is not None else self._table
        self.context_menu().popup(anchor.mapToGlobal(pos))

    def _row_cells(self, row: int) -> list[str]:
        """One row's cells at full precision, in the table's current row order.

        Written from the :class:`Peak` itself rather than the display text, so a copy
        or an export never carries fewer digits than the result.
        """
        peak = self._peak_for_row(row)
        if peak is None:
            return [""] * len(self._columns)
        cells = [
            _full_precision(peak.rank),
            _full_precision(peak.period),
            _full_precision(peak.frequency),
            _full_precision(peak.power),
        ]
        cells += [
            _full_precision(peak.extra[key]) if key in peak.extra else ""
            for key in self._extra_keys
        ]
        return cells

    def _copy_selected(self) -> None:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        if not rows:
            return
        lines = ["\t".join(self._columns)]
        lines += ["\t".join(self._row_cells(row)) for row in rows]
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("\n".join(lines))

    #: The window's dialogs for the Export… prompts (None: a static file dialog).
    _dialogs: DialogProvider | None = None

    def set_dialog_provider(self, provider: DialogProvider | None) -> None:
        """Ask for export paths through ``provider`` (the window's dialogs)."""
        self._dialogs = provider

    def _export_csv(self) -> None:
        ask_save_path(
            self,
            self._dialogs,
            "Export peaks as CSV",
            "peaks.csv",
            "CSV files (*.csv)",
            self._write_csv,
        )

    def _write_csv(self, path: str) -> None:
        try:
            self.export_csv(path)
        except OSError as exc:
            report_failure(self, self.export_failed, f"Could not export {path}: {exc}")

    def export_csv(self, path: str) -> None:
        """Write the full table (all rows, current row order) to ``path`` as CSV."""
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(self._columns)
            for row in range(self._table.rowCount()):
                writer.writerow(self._row_cells(row))


__all__ = ["PeaksTable", "statistic_name"]
