"""The frequency-solution panel: the extracted components and how the run ended.

The pre-whitening counterpart of the peaks table. Every row is one extracted sinusoid
with its uncertainty, signal-to-noise, false-alarm probability, and — where one was
identified — the combination that explains it. Selecting a row makes that component the
active period, so the phased view folds on it exactly as it does for a periodogram peak;
:meth:`SolutionPanel.select_period` moves the highlight when the selection changes
elsewhere. The columns that identify a mode (frequency, amplitude, S/N, combination)
come first; the header's right-click menu shows or hides any column.

The ``A/Asp`` column is the reliability check the plot cannot show: the fitted amplitude
against the spectrum's own reading at that frequency. A row marked ✱ has an amplitude
entangled with a correlated neighbour, so it means something only alongside it.

The header line carries the part of the result that is easy to lose in a table: *why the
extraction stopped*, the residual scatter, and which uncertainty estimator produced the
error bars.

Two modes share the table. :meth:`SolutionPanel.set_solution` shows a finished
:class:`~cuperiod.prewhiten.result.PreWhitenResult` read-only, exactly as before;
:meth:`SolutionPanel.set_view` shows a live interactive session
(:class:`~cuperiod.gui.viewmodels.SessionView`) in the editable
:class:`~cuperiod.gui.widgets.frequency_table.FrequencyTable`, whose edits leave through
:attr:`~SolutionPanel.edit_requested` and :attr:`~SolutionPanel.group_requested`.
"""

from __future__ import annotations

import csv
import html
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

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
from cuperiod.gui.widgets.frequency_table import (
    COLUMNS,
    EMPTY_HINT,
    FrequencyTable,
    format_value,
    summary_html,
)
from cuperiod.prewhiten.result import PreWhitenResult, Sinusoid

if TYPE_CHECKING:
    from cuperiod.gui.commands import CommandRegistry
    from cuperiod.gui.dialogs import DialogProvider
    from cuperiod.gui.theme import ThemePalette
    from cuperiod.gui.viewmodels import ComponentView, SessionView

#: Column definitions: (header, tooltip, attribute, format).
_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("ID", "Extraction order (F1 was the strongest peak of the original data)",
     "label", "s"),
    ("frequency", "Frequency in cycles/day", "frequency", "g"),
    ("± f", "1-sigma frequency uncertainty (cycles/day)", "frequency_error", "e"),
    ("amplitude", "Amplitude in the light curve's units", "amplitude", "g"),
    ("± A", "1-sigma amplitude uncertainty", "amplitude_error", "e"),
    ("S/N", "Amplitude over the local noise of the residual spectrum (Breger)",
     "snr", "f"),
    ("combination", "Identification as a combination of stronger components",
     "combination", "s"),
    ("period (d)", "Period in days", "period", "g"),
    ("A/Asp",
     "Fitted amplitude over the spectrum's own reading at this frequency. 1.0 means "
     "they agree. Far from 1 (marked ✱) means this amplitude is entangled with a "
     "correlated neighbour — often the mode's own alias sidelobe — and should be "
     "quoted together with it, not alone. It is not a significance test.",
     "amplitude_ratio", "f"),
    ("phase", "Phase in radians at the solution's reference epoch", "phase", "f"),
    ("± ph", "1-sigma phase uncertainty (radians)", "phase_error", "e"),
    ("FAP", "False-alarm probability of the peak when it was extracted", "fap", "e"),
)

#: The CSV columns, in order: the keys of :meth:`Sinusoid.to_dict`, which is also the
#: row layout of the batch tables. Fixed here so an export with no components (a run
#: that extracted nothing, or no run yet) has the same header as a populated one.
CSV_FIELDS: tuple[str, ...] = (
    "rank", "label", "frequency", "frequency_error", "period", "period_error",
    "amplitude", "amplitude_error", "spectrum_amplitude", "amplitude_ratio",
    "blended", "phase", "phase_error", "snr", "fap", "delta_bic", "combination",
)

#: Extra CSV columns of a session export, after :data:`CSV_FIELDS`: the constraints
#: and notes a result does not carry, so the file says what was held, tied or dropped.
SESSION_CSV_FIELDS: tuple[str, ...] = (
    "id", "active", "fix_frequency", "fix_amplitude", "fix_phase", "tie",
    "variation", "amplitude_rms", "note", "flags",
)

#: Minimum width so the ID, the frequency and its uncertainty are readable without
#: resizing; the rest scrolls. A wider minimum (460) took the plots' room in a
#: 1366 px window, crushing their toolbars (UX-04).
_MIN_TABLE_WIDTH = 300

#: The headers carry no units (they are looked up by name); this line does (UX-18).
_UNITS_TEXT = "frequency c/d · period d · phase rad"


def _format(value: object, kind: str) -> str:
    return format_value(value, kind)


def _session_row(view_row: ComponentView) -> dict[str, Any]:
    """One CSV row of a session component: the result's fields plus the edits."""
    f = view_row.frequency
    period_error = (
        view_row.frequency_error / (f * f) if f > 0.0 else math.nan
    )
    return {
        "rank": "" if view_row.rank is None else view_row.rank,
        "label": view_row.label,
        "frequency": f,
        "frequency_error": view_row.frequency_error,
        "period": view_row.period,
        "period_error": period_error,
        "amplitude": view_row.amplitude,
        "amplitude_error": view_row.amplitude_error,
        "spectrum_amplitude": view_row.spectrum_amplitude,
        "amplitude_ratio": view_row.amplitude_ratio,
        "blended": view_row.blended,
        "phase": view_row.phase,
        "phase_error": view_row.phase_error,
        "snr": view_row.snr,
        "fap": view_row.fap,
        "delta_bic": view_row.delta_bic,
        "combination": view_row.combination or "",
        "id": view_row.id,
        "active": view_row.active,
        "fix_frequency": view_row.fix_frequency,
        "fix_amplitude": view_row.fix_amplitude,
        "fix_phase": view_row.fix_phase,
        "tie": view_row.tie or "",
        "variation": view_row.variation,
        "amplitude_rms": "" if view_row.amplitude_rms is None
        else view_row.amplitude_rms,
        "note": view_row.note,
        "flags": ",".join(sorted(view_row.flags)),
    }


class SolutionPanel(QtWidgets.QWidget):
    """Summary line plus the table of extracted sinusoids.

    Signals
    -------
    component_selected(Sinusoid)
        A row of a finished result was selected (read-only mode).
    export_failed(str)
        The CSV export could not be written.
    edit_requested(str, object), group_requested(object), components_selected(object)
        The editable table's signals (see
        :class:`~cuperiod.gui.widgets.frequency_table.FrequencyTable`); every args
        mapping carries ``expect_id_epoch``.
    status_message(str)
        Feedback for the status bar (a rejected entry, an empty selection).
    """

    component_selected = Signal(object)  # Sinusoid
    #: The CSV export could not be written; carries a message for the user.
    export_failed = Signal(str)
    edit_requested = Signal(str, object)
    group_requested = Signal(object)
    components_selected = Signal(object)
    status_message = Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._components: list[Sinusoid] = []
        self._result: PreWhitenResult | None = None
        self._view: SessionView | None = None
        self.setMinimumWidth(_MIN_TABLE_WIDTH)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._summary = QtWidgets.QLabel("No frequency solution yet.")
        self._summary.setObjectName("muted")
        self._summary.setWordWrap(True)
        # Always rich text, never Qt's guess: every interpolated string is escaped.
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        self._summary.setToolTip(
            "How the extraction ended, the residual scatter, and the error model"
        )
        self._summary.setAccessibleName("Frequency solution summary")
        layout.addWidget(self._summary)

        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._units = QtWidgets.QLabel(_UNITS_TEXT)
        self._units.setObjectName("muted")
        self._units.setTextFormat(Qt.TextFormat.PlainText)
        self._units.setAccessibleName("Table units")
        # Wraps in a narrow pane, so the Copy and Export buttons keep their labels.
        self._units.setWordWrap(True)
        self._units.setToolTip(
            "Units of the table: frequencies (and their errors) in cycles per day, "
            "periods in days, phases in radians; amplitudes in the light curve's units"
        )
        row.addWidget(self._units)
        row.addWidget(
            table_action_bar(self._copy_selected, self._export_csv, "components"), 1
        )
        layout.addWidget(bar)

        self._hint = QtWidgets.QLabel(EMPTY_HINT)
        self._hint.setObjectName("muted")
        self._hint.setWordWrap(True)
        self._hint.setAccessibleName("How to start")
        self._hint.setVisible(False)
        layout.addWidget(self._hint)

        self._table = FrequencyTable()
        self._table.setMinimumWidth(_MIN_TABLE_WIDTH)
        self._table.setSortingEnabled(True)
        header = self._table.verticalHeader()
        if header is not None:
            header.setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_selection)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        self._table.edit_requested.connect(self.edit_requested)
        self._table.group_requested.connect(self.group_requested)
        self._table.components_selected.connect(self.components_selected)
        self._table.status_message.connect(self.status_message)
        configure_result_table(self._table, self._copy_selected)
        install_column_chooser(self._table)
        layout.addWidget(self._table, 1)

    # -- data --------------------------------------------------------------------
    @property
    def editable(self) -> bool:
        """Whether the panel shows a live session (``set_view``)."""
        return self._table.editable

    @property
    def table(self) -> FrequencyTable:
        """The frequency table (the host of the registry's table letters)."""
        return self._table

    def set_solution(self, result: PreWhitenResult) -> None:
        """Populate the summary and the component table from ``result``."""
        self._view = None
        self._table.leave_session()
        self._hint.setVisible(False)
        self._result = result
        self._components = list(result.components)
        self._summary.setText(self._summary_text(result))
        self._fill_table()

    def set_view(self, view: SessionView) -> None:
        """Show a live session, editable; cells update in place between views."""
        self._view = view
        self._result = view.result if view.result is not None else self._result
        self._components = []
        self._summary.setText(summary_html(view))
        self._hint.setVisible(not view.rows)
        self._table.show_view(view)

    @staticmethod
    def _summary_text(result: PreWhitenResult) -> str:
        # The engine's stop reasons read "S/N 3.62 < 4": unescaped, the "<" opened a
        # tag that swallowed the threshold and the line break after it.
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
            f"stopped: {html.escape(result.stop_reason, quote=False)}<br>"
            f"residual rms {result.rms:.4g}  ·  reduced χ² {result.reduced_chi2:.3g}"
            f"  ·  D = {result.correlation_factor:.2f}"
            f"  ·  errors: {html.escape(result.uncertainty_method, quote=False)}"
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
                item: QtWidgets.QTableWidgetItem
                if attribute == "label":
                    # Sorted by rank, not by text: "F10" must follow "F9".
                    item = NumericItem(float(component.rank), text)
                    item.setData(Qt.ItemDataRole.UserRole, component.rank)
                elif kind == "s":
                    item = QtWidgets.QTableWidgetItem(text)
                else:
                    item = NumericItem(float(value), text)
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
            select_only_row(self._table, 0)
        self._table.blockSignals(False)
        self._table.setSortingEnabled(True)

    def clear(self) -> None:
        """Empty the panel (a new curve is loading, or the analysis changed)."""
        self._result = None
        self._view = None
        self._components = []
        self._table.leave_session()
        self._hint.setVisible(False)
        self._summary.setText("No frequency solution yet.")
        self._table.blockSignals(True)
        self._table.clear()
        self._table.setRowCount(0)
        self._table.setColumnCount(len(_COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in _COLUMNS])
        self._table.blockSignals(False)

    # -- session -----------------------------------------------------------------
    def attach_commands(
        self, registry: CommandRegistry, *, route: bool = True
    ) -> None:
        """Create the table's context letters (Space, L, T, Del, ...) on the table.

        With ``route`` the table carries out the row commands itself whenever the
        registry triggers them (see
        :meth:`FrequencyTable.attach_commands
        <cuperiod.gui.widgets.frequency_table.FrequencyTable.attach_commands>`).
        """
        self._table.attach_commands(registry, route=route)

    def handle_command(self, command_id: str) -> bool:
        """Carry out a row command (``session.toggle_use``, ``edit.delete``, ...)."""
        return self._table.handle_command(command_id)

    def set_busy(self, busy: bool) -> None:
        """A long job runs: the table stays readable but takes no edits."""
        self._table.set_busy(busy)

    def resync(self) -> None:
        """Put the shown view's values back (after an edit was refused)."""
        self._table.resync()

    def selected_components(self) -> tuple[int, ...]:
        """The ids of the selected rows (current row first; empty in read-only)."""
        return self._table.selected_ids()

    def select_components(self, ids: Sequence[int]) -> None:
        """Select the rows of ``ids`` without emitting (the controller's echo)."""
        if self._table.editable:
            self._table.select_ids(ids)

    def set_amplitude_unit(self, unit: str | None) -> None:
        """Name the amplitude unit on the units line (``"mag"``, ``"flux"``, ...).

        The headers stay bare (they are addressed by name); ``None`` says the
        amplitudes are in the light curve's own units.
        """
        text = _UNITS_TEXT
        if unit:
            text += f" · amplitude {unit}"
        self._units.setText(text)

    def apply_theme(self, theme_palette: ThemePalette) -> None:
        """The warning colour of a failing S/N follows the theme (the check boxes
        follow the application style sheet)."""
        self._table.set_warning_color(theme_palette.qcolor(theme_palette.best_peak))

    # -- selection ---------------------------------------------------------------
    def _on_selection(self) -> None:
        if self._table.editable:
            return  # the table emits components_selected itself
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

    def _period_of_row(self, row: int) -> float | None:
        if self._table.editable:
            view = self._view
            ident = self._table.id_at(row)
            found = view.row(ident) if view is not None and ident is not None else None
            return None if found is None else found.period
        component = self._component_for_row(row)
        return None if component is None else component.period

    def select_period(self, period: float) -> None:
        """Highlight the component at ``period`` without emitting anything.

        The counterpart of :meth:`PeaksTable.select_period
        <cuperiod.gui.widgets.peaks_panel.PeaksTable.select_period>`: a marker click
        or a band drag elsewhere moves the highlight, so clicking the previously
        highlighted row is never a dead click.
        """
        target = None
        for row in range(self._table.rowCount()):
            row_period = self._period_of_row(row)
            if row_period is not None and math.isclose(
                row_period, period, rel_tol=1e-9
            ):
                target = row
                break
        selected = {index.row() for index in self._table.selectedIndexes()}
        if target is not None and target in selected:
            return
        self._table.blockSignals(True)
        select_only_row(self._table, target)
        self._table.blockSignals(False)

    # -- copy / export -----------------------------------------------------------
    def _on_context_menu(self, pos: QtCore.QPoint) -> None:
        self.context_menu().popup(self._menu_anchor().mapToGlobal(pos))

    def _menu_anchor(self) -> QtWidgets.QWidget:
        viewport = self._table.viewport()
        return viewport if viewport is not None else self._table

    def context_menu(self) -> QtWidgets.QMenu:
        """The table's right-click menu (row actions in a session, copy, export).

        Built fresh each time and deleted when it closes; shown with ``popup()`` so
        nothing blocks.
        """
        menu = QtWidgets.QMenu(self._table)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        menu.setAccessibleName("Frequency table actions")
        if self._table.editable:
            self._table.add_row_actions(menu)
            menu.addSeparator()
        copy_key = QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy).toString(
            QtGui.QKeySequence.SequenceFormat.NativeText
        )
        copy_action = menu.addAction(f"Copy selected rows\t{copy_key}")
        copy_action.triggered.connect(self._copy_selected)
        export_action = menu.addAction("Export CSV…")
        export_action.triggered.connect(self._export_csv)
        return menu

    def _headers(self) -> list[str]:
        if self._table.editable:
            return [c.header for c in COLUMNS]
        return [c[0] for c in _COLUMNS]

    def _copy_selected(self) -> None:
        rows = sorted({index.row() for index in self._table.selectedIndexes()})
        if not rows:
            return
        lines = ["\t".join(self._headers())]
        lines += ["\t".join(self._row_text(row)) for row in rows]
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("\n".join(lines))

    def _row_text(self, row: int) -> list[str]:
        if self._table.editable:
            return self._table.row_texts(row)
        cells = []
        for column in range(self._table.columnCount()):
            item = self._table.item(row, column)
            cells.append(item.text() if item is not None else "")
        return cells

    #: The window's dialogs for the Export… prompts (None: a static file dialog).
    _dialogs: DialogProvider | None = None

    def set_dialog_provider(self, provider: DialogProvider | None) -> None:
        """Ask for export paths through ``provider`` (the window's dialogs)."""
        self._dialogs = provider

    def _export_csv(self) -> None:
        ask_save_path(
            self,
            self._dialogs,
            "Export frequency solution",
            "frequencies.csv",
            "CSV files (*.csv)",
            self._write_csv,
        )

    def _write_csv(self, path: str) -> None:
        try:
            self.export_csv(path)
        except OSError as exc:
            report_failure(self, self.export_failed, f"Could not export {path}: {exc}")

    def export_csv(self, path: str) -> None:
        """Write the full solution (all fields, full precision) to ``path`` as CSV.

        The header is :data:`CSV_FIELDS`, with or without components (a field that a
        newer :meth:`Sinusoid.to_dict` adds is appended rather than dropped). A live
        session writes every component, the inactive ones included, with
        :data:`SESSION_CSV_FIELDS` appended.
        """
        fields = list(CSV_FIELDS)
        rows: list[dict[str, Any]]
        if self._table.editable and self._view is not None:
            rows = [_session_row(r) for r in self._view.rows]
            fields += list(SESSION_CSV_FIELDS)
        else:
            rows = self._result.to_table() if self._result is not None else []
            if rows:
                fields += [key for key in rows[0] if key not in CSV_FIELDS]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


__all__ = ["CSV_FIELDS", "SESSION_CSV_FIELDS", "SolutionPanel"]
