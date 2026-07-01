"""The peaks table: the N most significant peaks, click a row to investigate one.

Lists rank / period / power plus whatever method-specific extras a peak carries (BLS
depth, duration, t0, SDE; GLS false-alarm probability). Selecting a row emits
:attr:`peak_selected`, which the controller turns into the active period driving the
spectrum line and the phased view.
"""

from __future__ import annotations

from cuperiod.core.result import Peak
from cuperiod.gui.qt import QtWidgets, Signal


def _fmt(value: float) -> str:
    return f"{value:.6g}"


class PeaksTable(QtWidgets.QWidget):
    """A table of the best peaks; row selection emits the chosen :class:`Peak`."""

    peak_selected = Signal(object)  # Peak

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._peaks: list[Peak] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self._table = QtWidgets.QTableWidget()
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        header = self._table.verticalHeader()
        if header is not None:
            header.setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_selection)
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

        self._table.blockSignals(True)
        self._table.clear()
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.setRowCount(len(self._peaks))
        for r, peak in enumerate(self._peaks):
            cells = [str(peak.rank), _fmt(peak.period), _fmt(peak.power)]
            cells += [
                _fmt(peak.extra[k]) if k in peak.extra else "" for k in extra_keys
            ]
            for c, text in enumerate(cells):
                self._table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        h_header = self._table.horizontalHeader()
        if h_header is not None:
            h_header.setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        if self._peaks:
            self._table.selectRow(0)
        self._table.blockSignals(False)

    def clear(self) -> None:
        """Empty the table (e.g. when a new curve is being loaded)."""
        self.set_peaks([])

    def _on_selection(self) -> None:
        row = self._table.currentRow()
        if 0 <= row < len(self._peaks):
            self.peak_selected.emit(self._peaks[row])


__all__ = ["PeaksTable"]
