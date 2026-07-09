"""The peaks table: the N most significant peaks, click a row to investigate one.

Lists rank / period / power plus whatever method-specific extras a peak carries (BLS
depth, duration, t0, SDE; GLS false-alarm probability). Selecting a row emits
:attr:`peak_selected`, which the controller turns into the active period driving the
spectrum line and the phased view. Rows can be sorted numerically by any column, copied
as TSV (selection or the whole table), and exported to CSV.
"""

from __future__ import annotations

import csv

from cuperiod.core.result import Peak
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, Signal

#: Column header tooltips: the fixed columns plus known method-specific extras. Any
#: extra key not listed here still gets a column, just without a tooltip.
_COLUMN_TOOLTIPS: dict[str, str] = {
    "rank": "Significance rank (1 = most significant)",
    "period (d)": "Period, in days",
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


def _fmt(value: float) -> str:
    return f"{value:.6g}"


class _NumericItem(QtWidgets.QTableWidgetItem):
    """A table item that sorts by its numeric value rather than its display text."""

    def __init__(self, value: float) -> None:
        super().__init__(_fmt(value))
        self.setData(Qt.ItemDataRole.EditRole, float(value))


class PeaksTable(QtWidgets.QWidget):
    """A table of the best peaks; row selection emits the chosen :class:`Peak`."""

    peak_selected = Signal(object)  # Peak

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._peaks: list[Peak] = []
        self._columns: list[str] = []
        self.setMinimumWidth(_MIN_TABLE_WIDTH)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
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
        self._table.itemSelectionChanged.connect(self._on_selection)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)

        copy_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Copy, self._table
        )
        copy_shortcut.activated.connect(self._copy_selected)

        layout.addWidget(self._table)

    def set_peaks(self, peaks: list[Peak]) -> None:
        """Populate the table from ``peaks`` (rank/period/power + method extras)."""
        self._peaks = list(peaks)
        extra_keys: list[str] = []
        for peak in self._peaks:
            for key in peak.extra:
                if key not in extra_keys:
                    extra_keys.append(key)
        columns = ["rank", "period (d)", "power", *extra_keys]
        self._columns = columns

        self._table.setSortingEnabled(False)
        self._table.blockSignals(True)
        self._table.clear()
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.setRowCount(len(self._peaks))
        for r, peak in enumerate(self._peaks):
            values = [float(peak.rank), peak.period, peak.power]
            values += [peak.extra.get(k, float("nan")) for k in extra_keys]
            for c, (key, value) in enumerate(zip(columns, values, strict=True)):
                if key in peak.extra or key in {"rank", "period (d)", "power"}:
                    self._table.setItem(r, c, _NumericItem(value))
                else:
                    self._table.setItem(r, c, QtWidgets.QTableWidgetItem(""))
        h_header = self._table.horizontalHeader()
        if h_header is not None:
            h_header.setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            self._set_header_tooltips(columns)
        self._table.resizeColumnsToContents()
        if self._peaks:
            self._table.selectRow(0)
        self._table.blockSignals(False)
        self._table.setSortingEnabled(True)

    def _set_header_tooltips(self, columns: list[str]) -> None:
        h_header = self._table.horizontalHeader()
        if h_header is None:
            return
        for c, name in enumerate(columns):
            tooltip = _COLUMN_TOOLTIPS.get(name, name)
            header_item = self._table.horizontalHeaderItem(c)
            if header_item is not None:
                header_item.setToolTip(tooltip)

    def clear(self) -> None:
        """Empty the table (e.g. when a new curve is being loaded)."""
        self.set_peaks([])

    def _on_selection(self) -> None:
        row = self._table.currentRow()
        peak = self._peak_for_row(row)
        if peak is not None:
            self.peak_selected.emit(peak)

    def _peak_for_row(self, row: int) -> Peak | None:
        item = self._table.item(row, 0)
        if item is None:
            return None
        rank = int(item.data(Qt.ItemDataRole.EditRole))
        for peak in self._peaks:
            if peak.rank == rank:
                return peak
        return None

    # -- copy / export -------------------------------------------------------
    def _on_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self._table)
        copy_action = menu.addAction("Copy selected rows")
        copy_action.triggered.connect(self._copy_selected)
        export_action = menu.addAction("Export CSV…")
        export_action.triggered.connect(self._export_csv)
        viewport = self._table.viewport()
        anchor = viewport if viewport is not None else self._table
        menu.exec(anchor.mapToGlobal(pos))

    def _copy_selected(self) -> None:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        if not rows:
            return
        lines = ["\t".join(self._columns)]
        for row in rows:
            cells = []
            for c in range(self._table.columnCount()):
                item = self._table.item(row, c)
                cells.append(item.text() if item is not None else "")
            lines.append("\t".join(cells))
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("\n".join(lines))

    def _export_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export peaks as CSV", "peaks.csv", "CSV files (*.csv)"
        )
        if path:
            self.export_csv(path)

    def export_csv(self, path: str) -> None:
        """Write the full table (all rows, current column order) to ``path`` as CSV."""
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(self._columns)
            for row in range(self._table.rowCount()):
                cells = []
                for c in range(self._table.columnCount()):
                    item = self._table.item(row, c)
                    cells.append(item.text() if item is not None else "")
                writer.writerow(cells)


__all__ = ["PeaksTable"]
