"""The frequency-solution panel: the extracted components and how the run ended.

The pre-whitening counterpart of the peaks table. Every row is one extracted sinusoid
with its uncertainty, signal-to-noise, false-alarm probability, and — where one was
identified — the combination that explains it. Selecting a row makes that component the
active period, so the phased view folds on it exactly as it does for a periodogram peak.

The ``A/Asp`` column is the reliability check the plot cannot show: the fitted amplitude
against the spectrum's own reading at that frequency. A row marked ✱ has an amplitude
entangled with a correlated neighbour, so it means something only alongside it.

The header line carries the part of the result that is easy to lose in a table: *why the
extraction stopped*, the residual scatter, and which uncertainty estimator produced the
error bars.
"""

from __future__ import annotations

import csv
import math

from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, Signal
from cuperiod.prewhiten.result import PreWhitenResult, Sinusoid

#: Column definitions: (header, tooltip, attribute, format).
_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("ID", "Extraction order (F1 was the strongest peak of the original data)",
     "label", "s"),
    ("frequency", "Frequency in cycles/day", "frequency", "g"),
    ("± f", "1-sigma frequency uncertainty (cycles/day)", "frequency_error", "e"),
    ("period (d)", "Period in days", "period", "g"),
    ("amplitude", "Amplitude in the light curve's units", "amplitude", "g"),
    ("± A", "1-sigma amplitude uncertainty", "amplitude_error", "e"),
    ("A/Asp",
     "Fitted amplitude over the spectrum's own reading at this frequency. 1.0 means "
     "they agree. Far from 1 (marked ✱) means this amplitude is entangled with a "
     "correlated neighbour — often the mode's own alias sidelobe — and should be "
     "quoted together with it, not alone. It is not a significance test.",
     "amplitude_ratio", "f"),
    ("phase", "Phase in radians at the solution's reference epoch", "phase", "f"),
    ("± ph", "1-sigma phase uncertainty (radians)", "phase_error", "e"),
    ("S/N", "Amplitude over the local noise of the residual spectrum (Breger)",
     "snr", "f"),
    ("FAP", "False-alarm probability of the peak when it was extracted", "fap", "e"),
    ("combination", "Identification as a combination of stronger components",
     "combination", "s"),
)

#: Minimum width so the frequency and its uncertainty are readable without resizing.
_MIN_TABLE_WIDTH = 460


def _format(value: object, kind: str) -> str:
    if kind == "s":
        return "" if value is None else str(value)
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        return "—"
    if kind == "e":
        return f"{number:.3g}"
    if kind == "f":
        return f"{number:.4f}"
    return f"{number:.9g}"


class _NumericItem(QtWidgets.QTableWidgetItem):
    """A table item that sorts by its numeric value rather than its display text.

    The sort key is kept on the instance rather than written to ``EditRole``:
    :class:`QTableWidgetItem` stores ``EditRole`` and ``DisplayRole`` in the same slot,
    so a float written there silently replaces the formatted text with Qt's own
    six-significant-digit rendering — which is not enough digits for a frequency and
    too many for an uncertainty.
    """

    def __init__(self, value: float, text: str) -> None:
        super().__init__(text)
        self._value = float(value)

    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:
        if isinstance(other, _NumericItem):
            return self._value < other._value
        return super().__lt__(other)


class SolutionPanel(QtWidgets.QWidget):
    """Summary line plus the table of extracted sinusoids."""

    component_selected = Signal(object)  # Sinusoid

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._components: list[Sinusoid] = []
        self._result: PreWhitenResult | None = None
        self.setMinimumWidth(_MIN_TABLE_WIDTH)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._summary = QtWidgets.QLabel("No frequency solution yet.")
        self._summary.setObjectName("muted")
        self._summary.setWordWrap(True)
        self._summary.setToolTip(
            "How the extraction ended, the residual scatter, and the error model"
        )
        layout.addWidget(self._summary)

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
        layout.addWidget(self._table, 1)

    # -- data --------------------------------------------------------------------
    def set_solution(self, result: PreWhitenResult) -> None:
        """Populate the summary and the component table from ``result``."""
        self._result = result
        self._components = list(result.components)
        self._summary.setText(self._summary_text(result))
        self._fill_table()

    @staticmethod
    def _summary_text(result: PreWhitenResult) -> str:
        pruned = (
            f"  ·  {result.n_pruned} pruned" if result.n_pruned else ""
        )
        combinations = (
            f"  ·  {len(result.combinations)} combination"
            f"{'' if len(result.combinations) == 1 else 's'}"
            if result.combinations
            else ""
        )
        blended = (
            f"  ·  <b>{result.n_blended}</b> blended ✱" if result.n_blended else ""
        )
        return (
            f"<b>{result.n_components}</b> components{pruned}{combinations}"
            f"{blended}<br>"
            f"stopped: {result.stop_reason}<br>"
            f"residual rms {result.rms:.4g}  ·  reduced χ² {result.reduced_chi2:.3g}"
            f"  ·  D = {result.correlation_factor:.2f}"
            f"  ·  errors: {result.uncertainty_method}"
        )

    def _fill_table(self) -> None:
        self._table.setSortingEnabled(False)
        self._table.blockSignals(True)
        self._table.clear()
        self._table.setColumnCount(len(_COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in _COLUMNS])
        self._table.setRowCount(len(self._components))
        for row, component in enumerate(self._components):
            for column, (_, tooltip, attribute, kind) in enumerate(_COLUMNS):
                value = getattr(component, attribute)
                text = _format(value, kind)
                if attribute == "amplitude_ratio" and component.blended:
                    text = f"{text} ✱"
                if kind == "s":
                    item: QtWidgets.QTableWidgetItem = QtWidgets.QTableWidgetItem(text)
                    if attribute == "label":
                        item.setData(Qt.ItemDataRole.UserRole, component.rank)
                else:
                    numeric = float(value)
                    item = _NumericItem(
                        numeric if math.isfinite(numeric) else float("inf"), text
                    )
                if component.blended and attribute in {"amplitude", "amplitude_ratio"}:
                    item.setToolTip(tooltip)
                self._table.setItem(row, column, item)
        h_header = self._table.horizontalHeader()
        if h_header is not None:
            h_header.setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            for column, spec in enumerate(_COLUMNS):
                header_item = self._table.horizontalHeaderItem(column)
                if header_item is not None:
                    header_item.setToolTip(spec[1])
        self._table.resizeColumnsToContents()
        if self._components:
            self._table.selectRow(0)
        self._table.blockSignals(False)
        self._table.setSortingEnabled(True)

    def clear(self) -> None:
        """Empty the panel (a new curve is loading, or the analysis changed)."""
        self._result = None
        self._components = []
        self._summary.setText("No frequency solution yet.")
        self._table.blockSignals(True)
        self._table.clear()
        self._table.setRowCount(0)
        self._table.setColumnCount(len(_COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in _COLUMNS])
        self._table.blockSignals(False)

    # -- selection ---------------------------------------------------------------
    def _on_selection(self) -> None:
        component = self._component_for_row(self._table.currentRow())
        if component is not None:
            self.component_selected.emit(component)

    def _component_for_row(self, row: int) -> Sinusoid | None:
        item = self._table.item(row, 0)
        if item is None:
            return None
        rank = item.data(Qt.ItemDataRole.UserRole)
        for component in self._components:
            if component.rank == rank:
                return component
        return None

    # -- copy / export -----------------------------------------------------------
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
        rows = sorted({index.row() for index in self._table.selectedIndexes()})
        if not rows:
            return
        lines = ["\t".join(c[0] for c in _COLUMNS)]
        lines += ["\t".join(self._row_text(row)) for row in rows]
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("\n".join(lines))

    def _row_text(self, row: int) -> list[str]:
        cells = []
        for column in range(self._table.columnCount()):
            item = self._table.item(row, column)
            cells.append(item.text() if item is not None else "")
        return cells

    def _export_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export frequency solution", "frequencies.csv", "CSV files (*.csv)"
        )
        if path:
            self.export_csv(path)

    def export_csv(self, path: str) -> None:
        """Write the full solution (all fields, full precision) to ``path`` as CSV."""
        rows = self._result.to_table() if self._result is not None else []
        fields = list(rows[0]) if rows else [c[2] for c in _COLUMNS]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


__all__ = ["SolutionPanel"]
