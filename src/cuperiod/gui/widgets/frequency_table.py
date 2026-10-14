"""The editable frequency table of an interactive pre-whitening session.

:class:`FrequencyTable` is the ``QTableWidget`` behind the Frequencies dock. It shows
the rows of a :class:`~cuperiod.gui.viewmodels.SessionView` — every component, the
inactive ones included — and turns what the user does to them into session operations:
the box in the ID column is "in use", the ``fix`` boxes hold a value, typing a number
into the frequency, amplitude or phase cell sets it, typing ``=F1+F2`` or ``2f1`` ties
it, and the ``per subset`` and ``note`` cells edit the variation mode and the note.

Nothing here touches a session. Every edit leaves as a signal — ``edit_requested(op,
args)`` or ``group_requested(actions)`` — whose argument mappings all carry
``expect_id_epoch``, the id epoch of the view the user was looking at, so the bridge can
refuse an edit aimed at a component list that has changed since (an undo across a
renumbering, say). :func:`split_epoch` and :func:`split_group` take it back out.

Rows are addressed by component id: sorting is display-only, and a new view updates the
cells in place (selection, scroll position and sort order survive); only a change in the
list of ids refills the table.

The table never holds an edit back: a text edit reverts to the view's value at once and
the committed value arrives with the next view, so a refused edit (a tie the session
rejects) leaves no stale text behind. Check boxes keep the user's click until then;
:meth:`FrequencyTable.resync` puts the view's state back when an edit was refused.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, Signal
from cuperiod.prewhiten.constraints import ConstraintError, format_tie, parse_tie

if TYPE_CHECKING:
    from cuperiod.gui.commands import CommandRegistry
    from cuperiod.gui.viewmodels import ComponentView, SessionView

#: Registry commands this table carries out on its selected rows (design 4.5). The
#: window routes them here (``attach_commands(..., route=True)`` does it itself).
TABLE_COMMANDS: frozenset[str] = frozenset(
    {
        "session.toggle_use",
        "session.fix_frequency",
        "session.fix_all",
        "session.tie",
        "session.improve_selected",
        "edit.delete",
    }
)

#: Commands that act on the table only while it has the keyboard focus: Delete also
#: removes points from the light curve, so its owner is whichever pane is focused.
FOCUS_COMMANDS: frozenset[str] = frozenset({"edit.delete"})

#: Text of each variation mode in the ``per subset`` column, and back.
VARIATION_TEXT: Mapping[str, str] = {
    "none": "–",
    "amplitude": "A",
    "phase": "φ",
    "both": "A+φ",
}
_VARIATION_OF_TEXT: Mapping[str, str] = {v: k for k, v in VARIATION_TEXT.items()}

#: Flags that a check box or a column already shows; every other flag is a warning
#: and puts ⚠ in front of the label.
_STATE_FLAGS = frozenset({"fixed_f", "fixed_a", "fixed_phi", "tied", "inactive"})

#: What each flag means, for the ID cell's tooltip.
FLAG_TEXT: Mapping[str, str] = {
    "fixed_f": "frequency held fixed",
    "fixed_a": "amplitude held fixed",
    "fixed_phi": "phase held fixed",
    "tied": "frequency follows its tie",
    "inactive": "not in use (excluded from the model)",
    "parent_inactive": "a component it is tied to is not in use; its frequency is "
    "held at that component's last value",
    "negative_amplitude": "negative amplitude under a fixed phase (the same sinusoid "
    "as |A| at phase + π)",
    "degenerate": "not identifiable from the data: its errors are undefined",
    "at_bound": "the frequency stopped at its refinement bound",
    "out_of_band": "outside the frequency grid",
    "unresolved": "closer than the frequency resolution (1/T) to another component",
    "low_frequency": "below 1/T: degenerate with the subset zero points",
    "weak_frequency": "frequency poorly constrained by the data",
    "short_subset": "a subset is too short to fit its own amplitude or phase",
}

#: The hint under the header of an empty session (design 4.4).
EMPTY_HINT = (
    "Click a peak in the spectrum and press A, or press Step to take the tallest. "
    "Auto runs until the stopping rule fires."
)

# The explicit Calculate (F8) and Improve solve with the covariance and fill these
# in; the light re-calculation after each edit does not, so a '–' says how to get them.
_ERRORS_TIP = (
    "Not computed yet: Calculate (F8) or Improve (F9, Shift+F9) gives the errors"
)

#: Item role holding the component id (column 0 of every row).
ID_ROLE = Qt.ItemDataRole.UserRole
#: Item role holding the text the table rendered, so an editor confirmed without a
#: change is not mistaken for an edit (a frequency rounded to nine digits must not be
#: written back as a new value).
_SHOWN_ROLE = Qt.ItemDataRole.UserRole + 2

_MINUS = "−"


@dataclass(frozen=True)
class _Column:
    key: str
    header: str
    tooltip: str


_ANY_TIE = "'=F1+F2' or '2f1' ties it"

#: Columns of the editable table: the values that identify a mode first (UX-18), each
#: ``fix`` box next to its value. Headers that the read-only table also has are spelled
#: exactly as there.
COLUMNS: tuple[_Column, ...] = (
    _Column("label", "ID",
            "Component id. The box says whether it is in the model (Space toggles the "
            "selected rows); ⚠ marks a fit warning — hover for the details."),
    _Column("frequency", "frequency",
            "Frequency in cycles/day. Type a number to move it, a period such as "
            f"'0.081 d', or a combination: {_ANY_TIE}."),
    _Column("fix_frequency", "fix f",
            "Hold the frequency at its value (L toggles the selected rows; Shift+L "
            "holds frequency, amplitude and phase)"),
    _Column("frequency_error", "± f",
            "1-sigma frequency uncertainty (cycles/day); '–' until Calculate (F8)"),
    _Column("amplitude", "amplitude",
            "Amplitude in the light curve's units; signed under a fixed phase. Typing "
            "a value also holds it."),
    _Column("fix_amplitude", "fix A", "Hold the amplitude at its value"),
    _Column("amplitude_error", "± A",
            "1-sigma amplitude uncertainty; '–' until Calculate (F8)"),
    _Column("snr", "S/N",
            "Amplitude over the local noise of the residual spectrum (Breger); '!' "
            "marks a component that fails an active stopping rule"),
    _Column("combination", "combination",
            "Identified combination of other components. Italic: not tied yet — "
            "double-click to tie the frequency to it."),
    _Column("tie", "tie",
            "Frequency tie such as F1 + F2 (T edits the selected row; empty unties)"),
    _Column("variation", "per subset",
            "Fit the amplitude (A), the phase (φ) or both (A+φ) per variation subset; "
            "– fits one value for all data. Double-click or F2 to choose."),
    _Column("period", "period (d)", "Period in days"),
    _Column("amplitude_ratio", "A/Asp",
            "Fitted amplitude over the spectrum's own reading at this frequency. 1.0 "
            "means they agree; far from 1 (marked ✱) means it is entangled with a "
            "correlated neighbour. It is not a significance test."),
    _Column("phase", "phase",
            "Phase in radians at the session's reference epoch. Typing a value also "
            "holds it."),
    _Column("fix_phase", "fix ph", "Hold the phase at its value"),
    _Column("phase_error", "± ph",
            "1-sigma phase uncertainty (radians); '–' until Calculate (F8)"),
    _Column("fap", "FAP", "False-alarm probability of the peak when it was extracted"),
    _Column("note", "note", "A free-text note, kept in the project and the script"),
)

COLUMN_KEYS: tuple[str, ...] = tuple(c.key for c in COLUMNS)
_COLUMN_OF: Mapping[str, int] = {c.key: i for i, c in enumerate(COLUMNS)}

_CHECK_COLUMNS = {
    "label": "active",
    "fix_frequency": "fix_frequency",
    "fix_amplitude": "fix_amplitude",
    "fix_phase": "fix_phase",
}
_TEXT_EDIT_COLUMNS = frozenset(
    {"frequency", "amplitude", "phase", "tie", "variation", "note"}
)


# -- parsing --------------------------------------------------------------------------
@dataclass(frozen=True)
class Entry:
    """What a typed frequency, period or tie means (see :func:`parse_entry`)."""

    kind: Literal["frequency", "tie"]
    frequency: float = math.nan
    expression: str = ""
    ids: tuple[int, ...] = ()
    coefficients: tuple[int, ...] = ()


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_QUANTITY = re.compile(rf"^({_NUMBER})\s*([^\s\d].*)?$")

#: Unit -> (is a period, factor to cycles/day or to days).
_UNITS: Mapping[str, tuple[bool, float]] = {
    "": (False, 1.0),
    "c/d": (False, 1.0),
    "cd": (False, 1.0),
    "1/d": (False, 1.0),
    "/d": (False, 1.0),
    "d-1": (False, 1.0),
    "uhz": (False, 86400e-6),
    "µhz": (False, 86400e-6),
    "μhz": (False, 86400e-6),
    "d": (True, 1.0),
    "day": (True, 1.0),
    "days": (True, 1.0),
    "h": (True, 1.0 / 24.0),
    "hr": (True, 1.0 / 24.0),
    "hours": (True, 1.0 / 24.0),
    "min": (True, 1.0 / 1440.0),
}

_LOOKS_LIKE_TIE = re.compile(r"[Ff]\s*\d")


def _plain(text: str) -> str:
    return text.replace(_MINUS, "-").strip()


def parse_number(text: str) -> float:
    """A finite float from table text (``−`` read as a minus sign).

    Raises
    ------
    ValueError
        With a message for the user.
    """
    raw = _plain(text)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"'{text.strip()}' is not a number") from None
    if not math.isfinite(value):
        raise ValueError("the value must be finite")
    return value


def parse_entry(text: str, *, child: int | None = None) -> Entry:
    """Read ``12.34`` (c/d), ``0.081 d`` (a period), ``=F1+F2`` or ``2f1`` (a tie).

    Frequencies may carry ``c/d`` or ``µHz``; periods ``d``, ``h`` or ``min``. A tie is
    anything with a label (``F1``) or a leading ``=``, parsed by
    :func:`~cuperiod.prewhiten.constraints.parse_tie` (never evaluated); ``child`` is
    the id of the component being tied, when there is one.

    Raises
    ------
    ValueError
        With the parser's message, ready for the status bar.
    """
    raw = _plain(text)
    if not raw:
        raise ValueError(
            "type a frequency (12.34), a period (0.081 d) or a tie (=F1+F2)"
        )
    if raw.startswith("=") or _LOOKS_LIKE_TIE.search(raw):
        expression = raw[1:] if raw.startswith("=") else raw
        try:
            ids, coefficients = parse_tie(expression, child=child)
        except ConstraintError as exc:
            raise ValueError(f"tie '{expression.strip()}': {exc}") from None
        return Entry(
            kind="tie",
            expression=format_tie(ids, coefficients),
            ids=ids,
            coefficients=coefficients,
        )
    match = _QUANTITY.match(raw)
    if match is None:
        raise ValueError(
            f"'{raw}' is not a frequency (12.34), a period (0.081 d) or a tie (=F1+F2)"
        )
    value = float(match.group(1))
    unit = (match.group(2) or "").strip().lower()
    if unit not in _UNITS:
        raise ValueError(f"unknown unit '{match.group(2)}' (use c/d, µHz, d, h or min)")
    is_period, factor = _UNITS[unit]
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("a frequency or period must be positive")
    frequency = 1.0 / (value * factor) if is_period else value * factor
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("a frequency or period must be positive")
    return Entry(kind="frequency", frequency=frequency)


def tie_frequency(entry: Entry, rows: Sequence[ComponentView]) -> float:
    """The frequency a tie ``entry`` resolves to with the frequencies of ``rows``.

    Raises
    ------
    ValueError
        A label that names no row, or a non-positive result.
    """
    by_id = {r.id: r for r in rows}
    total = 0.0
    for ident, coefficient in zip(entry.ids, entry.coefficients, strict=True):
        row = by_id.get(ident)
        if row is None:
            raise ValueError(f"F{ident} is not in the table")
        total += coefficient * row.frequency
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{entry.expression} gives {total:.6g} c/d, not a frequency")
    return total


def split_epoch(args: Mapping[str, Any]) -> tuple[dict[str, Any], int | None]:
    """``(op arguments, expect_id_epoch)`` of one emitted args mapping."""
    clean = dict(args)
    epoch = clean.pop("expect_id_epoch", None)
    return clean, None if epoch is None else int(epoch)


def split_group(
    actions: Iterable[tuple[str, Mapping[str, Any]]],
) -> tuple[list[tuple[str, dict[str, Any]]], int | None]:
    """``([(op, arguments), ...], expect_id_epoch)`` of an emitted group."""
    out: list[tuple[str, dict[str, Any]]] = []
    epoch: int | None = None
    for op, args in actions:
        clean, e = split_epoch(args)
        out.append((op, clean))
        if e is not None:
            epoch = e
    return out, epoch


# -- formatting -----------------------------------------------------------------------
def format_value(value: object, kind: str) -> str:
    """Display text: ``s`` text, ``e`` 3 significant digits, ``f`` 4 decimals, ``g`` 9
    significant digits; non-finite numbers show as an em dash."""
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


def _signed(value: float, kind: str) -> str:
    text = format_value(value, kind)
    return _MINUS + text[1:] if text.startswith("-") else text


def warning_flags(row: ComponentView) -> tuple[str, ...]:
    """The row's flags that are warnings (not states a check box already shows)."""
    return tuple(sorted(f for f in row.flags if f not in _STATE_FLAGS))


def _flag_lines(row: ComponentView) -> list[str]:
    return [f"{f}: {FLAG_TEXT.get(f, f.replace('_', ' '))}" for f in sorted(row.flags)]


@dataclass(frozen=True)
class _Cell:
    """Everything a cell shows; compared with the last one to update in place."""

    text: str
    sort: float | str
    check: bool | None = None
    editable: bool = False
    tooltip: str = ""
    bold: bool = False
    italic: bool = False
    muted: bool = False
    warn: bool = False


def _error_cell(value: float, fixed: str | None) -> _Cell:
    if fixed is not None:
        return _Cell(fixed, -1.0, tooltip=f"No uncertainty: the value is {fixed}")
    if not math.isfinite(value):
        return _Cell("–", math.inf, tooltip=_ERRORS_TIP)
    return _Cell(format_value(value, "e"), value)


def cell_for(row: ComponentView, key: str, position: int) -> _Cell:
    """The cell of column ``key`` for ``row`` (``position`` = table order)."""
    muted = not row.active
    tied = row.tie is not None
    if key == "label":
        warnings = warning_flags(row)
        lines = [f"{row.label}: {'in use' if row.active else 'not in use'}"]
        lines += [f"origin: {row.origin}"] if row.origin else []
        lines += _flag_lines(row)
        return _Cell(
            f"⚠ {row.label}" if warnings else row.label,
            float(position),
            check=row.active,
            tooltip="\n".join(lines),
            muted=muted,
        )
    if key == "frequency":
        return _Cell(
            format_value(row.frequency, "g"),
            row.frequency,
            editable=True,
            tooltip=f"Tied: {row.label} = {row.tie}" if tied else "",
            bold=tied,
            muted=muted,
        )
    if key == "fix_frequency":
        if tied:
            return _Cell(
                "tied", 2.0, tooltip="The frequency follows its tie; untie to fix it",
                muted=True,
            )
        return _Cell("", float(row.fix_frequency), check=row.fix_frequency)
    if key == "fix_amplitude":
        return _Cell("", float(row.fix_amplitude), check=row.fix_amplitude)
    if key == "fix_phase":
        return _Cell("", float(row.fix_phase), check=row.fix_phase)
    if key == "frequency_error":
        fixed = "tied" if tied else ("fixed" if row.fix_frequency else None)
        return _error_cell(row.frequency_error, fixed)
    if key == "amplitude_error":
        return _error_cell(row.amplitude_error, "fixed" if row.fix_amplitude else None)
    if key == "phase_error":
        return _error_cell(row.phase_error, "fixed" if row.fix_phase else None)
    if key == "amplitude":
        tip = (
            "Negative under a fixed phase: the same sinusoid as |A| at phase + π"
            if row.amplitude < 0.0
            else ""
        )
        return _Cell(
            _signed(row.amplitude, "g"), row.amplitude, editable=True, tooltip=tip,
            muted=muted,
        )
    if key == "phase":
        return _Cell(
            _signed(row.phase, "f"), row.phase, editable=True, muted=muted
        )
    if key == "period":
        return _Cell(format_value(row.period, "g"), row.period, muted=muted)
    if key == "snr":
        varying = row.amplitude_rms is not None
        source = "the RMS amplitude over the subsets" if varying else "|A|"
        text = format_value(row.snr, "f")
        failing = row.active and math.isfinite(row.snr) and not row.passes
        tip = f"S/N from {source}"
        if failing:
            text += " !"
            tip += "; fails an active stopping rule"
        return _Cell(text, row.snr, tooltip=tip, warn=failing, muted=muted)
    if key == "fap":
        return _Cell(format_value(row.fap, "e"), row.fap, muted=muted)
    if key == "amplitude_ratio":
        text = format_value(row.amplitude_ratio, "f")
        if row.blended:
            text += " ✱"
        tip = "Blended: quote this amplitude with its neighbour" if row.blended else ""
        return _Cell(text, row.amplitude_ratio, tooltip=tip, muted=muted)
    if key == "combination":
        text = row.combination or ""
        suggestion = bool(text) and not tied
        return _Cell(
            text,
            text,
            tooltip="Identified, not tied: double-click to tie the frequency to it"
            if suggestion
            else "",
            italic=suggestion,
            muted=muted,
        )
    if key == "tie":
        return _Cell(row.tie or "", row.tie or "", editable=True, muted=muted)
    if key == "variation":
        text = VARIATION_TEXT.get(row.variation, row.variation)
        return _Cell(text, text, editable=True, muted=muted)
    if key == "note":
        return _Cell(row.note, row.note, editable=True, tooltip=row.note, muted=muted)
    raise KeyError(key)


# -- items ----------------------------------------------------------------------------
class _Item(QtWidgets.QTableWidgetItem):
    """A cell that sorts by its own key (numbers numerically, NaN last)."""

    def __init__(self) -> None:
        super().__init__("")
        self.sort_key: float | str = ""
        self.cell: _Cell | None = None

    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:
        if isinstance(other, _Item):
            a, b = self.sort_key, other.sort_key
            if isinstance(a, float) and isinstance(b, float):
                return (math.inf if math.isnan(a) else a) < (
                    math.inf if math.isnan(b) else b
                )
            return str(a) < str(b)
        return super().__lt__(other)


class _VariationCombo(QtWidgets.QComboBox):
    """The ``per subset`` editor; its letters never reach the table's shortcuts."""

    def event(self, e: QtCore.QEvent) -> bool:
        if e.type() == QtCore.QEvent.Type.ShortcutOverride and isinstance(
            e, QtGui.QKeyEvent
        ):
            plain = e.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
            if plain == Qt.KeyboardModifier.NoModifier or (
                plain == Qt.KeyboardModifier.ShiftModifier and e.text()
            ):
                e.accept()
                return True
        return super().event(e)


class _VariationDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        combo = _VariationCombo(parent)
        combo.addItems(list(VARIATION_TEXT.values()))
        # A choice from the list is the edit: commit and close at once rather than
        # waiting for the focus to leave the editor.
        combo.activated.connect(lambda _index: self._commit(combo))
        combo.setAccessibleName("Per-subset fit")
        combo.setToolTip(COLUMNS[_COLUMN_OF["variation"]].tooltip)
        return combo

    def _commit(self, editor: QtWidgets.QWidget) -> None:
        self.commitData.emit(editor)
        self.closeEditor.emit(editor)

    def setEditorData(
        self,
        editor: QtWidgets.QWidget,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        if isinstance(editor, QtWidgets.QComboBox):
            text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
            position = editor.findText(text)
            editor.setCurrentIndex(max(position, 0))
            editor.showPopup()
            return
        super().setEditorData(editor, index)

    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractItemModel,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        if isinstance(editor, QtWidgets.QComboBox):
            model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
            return
        super().setModelData(editor, model, index)


# -- the table ------------------------------------------------------------------------
class FrequencyTable(QtWidgets.QTableWidget):
    """The Frequencies table; read-only by default, editable once given a view.

    Signals
    -------
    edit_requested(str, object)
        ``(op, args)`` of one session operation; ``args`` carries
        ``expect_id_epoch``.
    group_requested(object)
        A tuple of ``(op, args)`` pairs that undo as one action; each ``args`` carries
        ``expect_id_epoch``.
    components_selected(object)
        The selected component ids (tuple; the current row's id first), on a
        selection the user made in the editable table.
    status_message(str)
        Feedback for the status bar (a rejected entry, an empty selection).
    """

    edit_requested = Signal(str, object)
    group_requested = Signal(object)
    components_selected = Signal(object)
    status_message = Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._view: SessionView | None = None
        self._editable = False
        self._busy = False
        self._updating = False
        self._registry: CommandRegistry | None = None
        self._warn_color: QtGui.QColor | None = None
        self._variation_delegate = _VariationDelegate(self)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setObjectName("frequency_table")
        self.setAccessibleName("Frequency solution")
        self.setAccessibleDescription(
            "One row per extracted sinusoid. In a session: Space uses or drops the "
            "selected rows, L fixes their frequencies, T edits a tie, Del removes."
        )
        self.setStatusTip("The frequency solution; right-click for the row actions")
        self.itemChanged.connect(self._on_item_changed)
        self.itemDoubleClicked.connect(self._on_double_clicked)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        # The check boxes are drawn by the theme's QTableView::indicator rules
        # (dark.qss / light.qss), like every table's.

    # -- state ------------------------------------------------------------------------
    @property
    def editable(self) -> bool:
        """Whether the table shows a session (editable) rather than a result."""
        return self._editable

    @property
    def view(self) -> SessionView | None:
        """The session view shown (``None`` in read-only mode)."""
        return self._view

    @property
    def busy(self) -> bool:
        """A long job runs (``set_busy``, or the attached registry says so)."""
        registry = self._registry
        return self._busy or (registry is not None and registry.busy)

    def set_busy(self, busy: bool) -> None:
        """A long job runs: the table stays readable but takes no edits."""
        self._busy = bool(busy)
        self._apply_edit_triggers()

    def _apply_edit_triggers(self) -> None:
        triggers = QtWidgets.QAbstractItemView.EditTrigger
        if self._editable and not self._busy:
            self.setEditTriggers(triggers.DoubleClicked | triggers.EditKeyPressed)
        else:
            self.setEditTriggers(triggers.NoEditTriggers)

    def set_warning_color(self, color: QtGui.QColor | None) -> None:
        """The colour of an S/N that fails a stopping rule (the '!' says it too)."""
        self._warn_color = color
        self.resync()

    def leave_session(self) -> None:
        """Back to read-only mode: forget the view, drop the edit columns' delegate.

        The caller refills the table (``SolutionPanel.set_solution``).
        """
        if not self._editable and self._view is None:
            return
        self._view = None
        self._editable = False
        self._show_all_columns()
        default = self.itemDelegate()
        if default is not None:
            self.setItemDelegateForColumn(_COLUMN_OF["variation"], default)
        self._apply_edit_triggers()
        header = self.horizontalHeader()
        if header is not None:
            header.setSortIndicatorShown(False)

    def _show_all_columns(self) -> None:
        for column in range(self.columnCount()):
            self.setColumnHidden(column, False)

    # -- rendering --------------------------------------------------------------------
    def show_view(self, view: SessionView) -> None:
        """Show ``view``: in place when its ids match the rows shown, else refill."""
        ids = tuple(r.id for r in view.rows)
        structural = not self._editable or ids != self._shown_ids()
        previous = self._view
        self._view = view
        if structural:
            self._refill(view, keep_selection=previous is not None)
        else:
            self._update_in_place(view)

    def resync(self) -> None:
        """Re-render every cell from the current view (undo optimistic check boxes)."""
        if self._view is not None and self._editable:
            self._update_in_place(self._view, force=True)

    def fit_columns(self) -> None:
        """Fit the columns to their contents, with room for the ID check box.

        The size hint leaves out the styled check box, so a plain
        ``resizeColumnsToContents`` (after a zoom) elided "F1" to "…": column 0
        keeps room for the box and a two-digit label with its warning mark.
        """
        self.resizeColumnsToContents()
        room = self.fontMetrics().horizontalAdvance("⚠ F99") + 44
        self.setColumnWidth(0, max(self.columnWidth(0), room))

    def _shown_ids(self) -> tuple[int, ...]:
        """The ids in table (view) order, whatever the display sort."""
        found: list[tuple[float, int]] = []
        for r in range(self.rowCount()):
            item = self.item(r, 0)
            if item is None:
                continue
            ident = item.data(ID_ROLE)
            key = item.sort_key if isinstance(item, _Item) else 0.0
            if isinstance(ident, int):
                found.append((float(key) if isinstance(key, float) else 0.0, ident))
        return tuple(ident for _, ident in sorted(found))

    def _refill(self, view: SessionView, *, keep_selection: bool) -> None:
        selected = self.selected_ids() if keep_selection else ()
        bar = self.verticalScrollBar()
        scroll = bar.value() if bar is not None else 0
        entering = not self._editable
        self._editable = True
        self._updating = True
        sorting = self.isSortingEnabled()
        self.setSortingEnabled(False)
        self.blockSignals(True)
        try:
            self.clearContents()
            if entering:
                self._show_all_columns()  # hidden columns of the other mode's layout
            self.setColumnCount(len(COLUMNS))
            self.setHorizontalHeaderLabels([c.header for c in COLUMNS])
            for column, spec in enumerate(COLUMNS):
                header_item = self.horizontalHeaderItem(column)
                if header_item is not None:
                    header_item.setToolTip(spec.tooltip)
            self.setItemDelegateForColumn(
                _COLUMN_OF["variation"], self._variation_delegate
            )
            self.setRowCount(len(view.rows))
            for position, row in enumerate(view.rows):
                for column, spec in enumerate(COLUMNS):
                    item = _Item()
                    self._apply(item, cell_for(row, spec.key, position), spec.key)
                    if column == 0:
                        item.setData(ID_ROLE, row.id)
                    self.setItem(position, column, item)
            if entering:
                header = self.horizontalHeader()
                # Table order until the user sorts: Qt's default indicator (column
                # 0, descending) would otherwise reverse the rows.
                header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
                header.setSortIndicatorShown(True)
                header.setSectionResizeMode(
                    QtWidgets.QHeaderView.ResizeMode.Interactive
                )
                self.fit_columns()
            self.setSortingEnabled(sorting or entering)
            self.select_ids(selected)
        finally:
            self.blockSignals(False)
            self._updating = False
        self._apply_edit_triggers()
        if bar is not None:
            bar.setValue(scroll)

    def _update_in_place(self, view: SessionView, *, force: bool = False) -> None:
        rows_by_id = {r.id: (p, r) for p, r in enumerate(view.rows)}
        sorting = self.isSortingEnabled()
        self._updating = True
        self.setSortingEnabled(False)
        self.blockSignals(True)
        try:
            for r in range(self.rowCount()):
                first = self.item(r, 0)
                ident = first.data(ID_ROLE) if first is not None else None
                if not isinstance(ident, int) or ident not in rows_by_id:
                    continue
                position, row = rows_by_id[ident]
                for column, spec in enumerate(COLUMNS):
                    item = self.item(r, column)
                    if not isinstance(item, _Item):
                        continue
                    cell = cell_for(row, spec.key, position)
                    if force or cell != item.cell:
                        self._apply(item, cell, spec.key)
            # Re-enabling re-sorts by the user's column, if any; the selection
            # follows the rows (the selection model tracks them through the move),
            # and the move is no selection the user made.
            self.setSortingEnabled(sorting)
        finally:
            self.blockSignals(False)
            self._updating = False

    def _apply(self, item: _Item, cell: _Cell, key: str) -> None:
        item.cell = cell
        item.sort_key = cell.sort
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if cell.editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        if cell.check is not None:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        item.setFlags(flags)
        item.setText(cell.text)
        item.setData(_SHOWN_ROLE, cell.text)
        if cell.check is None:
            # No indicator at all (a tied row's "fix f" reads "tied").
            item.setData(Qt.ItemDataRole.CheckStateRole, None)
        else:
            item.setCheckState(
                Qt.CheckState.Checked if cell.check else Qt.CheckState.Unchecked
            )
        item.setToolTip(cell.tooltip)
        font = item.font()
        font.setBold(cell.bold)
        font.setItalic(cell.italic)
        item.setFont(font)
        if cell.warn:
            item.setForeground(QtGui.QBrush(self._warning_color()))
        elif cell.muted:
            item.setForeground(
                QtGui.QBrush(
                    self.palette().color(
                        QtGui.QPalette.ColorGroup.Disabled,
                        QtGui.QPalette.ColorRole.Text,
                    )
                )
            )
        else:
            item.setData(Qt.ItemDataRole.ForegroundRole, None)
        accessible = cell.text or (
            ("checked" if cell.check else "unchecked") if cell.check is not None else ""
        )
        item.setData(
            Qt.ItemDataRole.AccessibleTextRole,
            f"{COLUMNS[_COLUMN_OF[key]].header}: {accessible}",
        )

    def _warning_color(self) -> QtGui.QColor:
        if self._warn_color is not None:
            return self._warn_color
        dark = self.palette().color(QtGui.QPalette.ColorRole.Base).lightness() < 128
        # The theme's amber (ThemePalette.best_peak) for each background.
        return QtGui.QColor("#fbbf24" if dark else "#b45309")

    # -- addressing -------------------------------------------------------------------
    def id_at(self, row: int) -> int | None:
        """The component id shown in display row ``row``."""
        item = self.item(row, 0)
        ident = item.data(ID_ROLE) if item is not None else None
        return ident if isinstance(ident, int) else None

    def row_of(self, ident: int) -> int | None:
        """The display row of component ``ident``."""
        for r in range(self.rowCount()):
            if self.id_at(r) == ident:
                return r
        return None

    def selected_ids(self) -> tuple[int, ...]:
        """Selected ids: the current row first, then display order."""
        if not self._editable:
            return ()
        model = self.selectionModel()
        if model is None:
            return ()
        rows = sorted({index.row() for index in model.selectedRows()})
        ids = [i for i in (self.id_at(r) for r in rows) if i is not None]
        current = self.id_at(self.currentRow()) if self.currentRow() >= 0 else None
        if current is not None and current in ids:
            ids.remove(current)
            ids.insert(0, current)
        return tuple(ids)

    def select_ids(self, ids: Sequence[int]) -> None:
        """Select the rows of ``ids`` (the first becomes current) without emitting."""
        model = self.selectionModel()
        table_model = self.model()
        if model is None or table_model is None:
            return
        rows = [r for r in (self.row_of(i) for i in ids) if r is not None]
        blocked = self.blockSignals(True)
        try:
            model.clearSelection()
            flags = QtCore.QItemSelectionModel.SelectionFlag
            for n, r in enumerate(rows):
                index = table_model.index(r, 0)
                if n == 0:
                    model.setCurrentIndex(
                        index, flags.ClearAndSelect | flags.Rows
                    )
                else:
                    model.select(index, flags.Select | flags.Rows)
        finally:
            self.blockSignals(blocked)

    def _rows(self, ids: Iterable[int]) -> list[ComponentView]:
        view = self._view
        if view is None:
            return []
        return [r for r in (view.row(i) for i in ids) if r is not None]

    # -- emitting ---------------------------------------------------------------------
    def _args(self, **args: Any) -> dict[str, Any]:
        epoch = self._view.id_epoch if self._view is not None else None
        return {**args, "expect_id_epoch": epoch}

    def _emit(self, op: str, **args: Any) -> None:
        self.edit_requested.emit(op, self._args(**args))

    def _emit_group(self, actions: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
        if len(actions) == 1:
            op, args = actions[0]
            self._emit(op, **args)
            return
        self.group_requested.emit(
            tuple((op, self._args(**args)) for op, args in actions)
        )

    def _refuse(self, text: str) -> None:
        self.status_message.emit(text)

    # -- user edits -------------------------------------------------------------------
    def _on_selection_changed(self) -> None:
        if self._editable and not self._updating:
            self.components_selected.emit(self.selected_ids())

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._updating or not self._editable or not isinstance(item, _Item):
            return
        key = COLUMNS[item.column()].key
        ident = self.id_at(item.row())
        view = self._view
        row = view.row(ident) if view is not None and ident is not None else None
        if row is None:
            return
        if self.busy:
            self._restore(item, row, key)
            self._refuse("Busy: wait for the running job, or press Stop (Esc)")
            return
        if key in _CHECK_COLUMNS:
            self._on_check(item, row, key)
        elif key in _TEXT_EDIT_COLUMNS:
            text = item.text()
            unchanged = text == item.data(_SHOWN_ROLE)
            # Back to the view's value *before* anything is emitted: a listener may
            # deliver the next view synchronously, and the item must not be touched
            # after that (a refill deletes it).
            self._restore(item, row, key)
            if not unchanged:
                self._on_text(item, row, key, text)

    def _restore(
        self, item: _Item, row: ComponentView, key: str, *, error: str = ""
    ) -> None:
        self._updating = True
        blocked = self.blockSignals(True)
        sorting = self.isSortingEnabled()
        self.setSortingEnabled(False)
        try:
            self._apply(item, cell_for(row, key, self._position(row.id)), key)
            if error:
                item.setToolTip(error)
        finally:
            self.setSortingEnabled(sorting)
            self.blockSignals(blocked)
            self._updating = False

    def _position(self, ident: int) -> int:
        view = self._view
        if view is None:
            return 0
        for p, r in enumerate(view.rows):
            if r.id == ident:
                return p
        return 0

    def _on_check(self, item: _Item, row: ComponentView, key: str) -> None:
        if item.cell is None or item.cell.check is None:
            return
        checked = item.checkState() == Qt.CheckState.Checked
        if checked == item.cell.check:
            return
        # Kept as clicked until the next view arrives (resync() reverts a refusal).
        item.cell = replace(item.cell, check=checked)
        if key == "label":
            self._emit("set_active", ids=[row.id], active=checked)
        elif key == "fix_frequency":
            self._emit("fix", ids=[row.id], frequency=checked)
        elif key == "fix_amplitude":
            self._emit("fix", ids=[row.id], amplitude=checked)
        elif key == "fix_phase":
            self._emit("fix", ids=[row.id], phase=checked)

    def _fail(self, item: _Item, row: ComponentView, key: str, message: str) -> None:
        # Nothing was emitted, so the item is still alive: its tooltip keeps the
        # parser's message until the next view (the status bar gets it too).
        self._restore(item, row, key, error=message)
        self._refuse(message)

    def _on_text(self, item: _Item, row: ComponentView, key: str, text: str) -> None:
        try:
            if key == "frequency":
                entry = parse_entry(text, child=row.id)
                if entry.kind == "tie":
                    self._emit("tie", id=row.id, expression=entry.expression)
                else:
                    self._emit("set_values", id=row.id, frequency=entry.frequency)
            elif key in ("amplitude", "phase"):
                value = parse_number(text)
                self._emit_group(
                    [
                        ("set_values", {"id": row.id, key: value}),
                        ("fix", {"ids": [row.id], key: True}),
                    ]
                )
            elif key == "tie":
                stripped = _plain(text)
                if not stripped:
                    if row.tie is not None:
                        self._emit("tie", id=row.id, expression=None)
                    return
                body = stripped[1:] if stripped.startswith("=") else stripped
                try:
                    ids, coefficients = parse_tie(body, child=row.id)
                except ConstraintError as exc:
                    raise ValueError(f"tie '{body}': {exc}") from None
                self._emit("tie", id=row.id, expression=format_tie(ids, coefficients))
            elif key == "variation":
                mode = _VARIATION_OF_TEXT.get(text.strip())
                if mode is None:
                    raise ValueError(
                        f"per subset is one of {', '.join(VARIATION_TEXT.values())}"
                    )
                if mode != row.variation:
                    self._emit("set_variation", ids=[row.id], mode=mode)
            elif key == "note":
                note = text.strip()
                if len(note) > 200:
                    raise ValueError("a note has at most 200 characters")
                if note != row.note:
                    self._emit("set_note", id=row.id, note=note)
        except ValueError as exc:
            self._fail(item, row, key, f"{row.label}: {exc}")

    def _on_double_clicked(self, item: QtWidgets.QTableWidgetItem) -> None:
        if not self._editable or self.busy:
            return
        if COLUMNS[item.column()].key != "combination":
            return
        ident = self.id_at(item.row())
        rows = self._rows([] if ident is None else [ident])
        if not rows or not rows[0].combination or rows[0].tie is not None:
            return
        self.tie_to_combination(rows[0].id)

    def tie_to_combination(self, *ids: int) -> bool:
        """Tie each component of ``ids`` to its identified combination (one group)."""
        actions: list[tuple[str, Mapping[str, Any]]] = []
        for row in self._rows(ids):
            text = row.combination or ""
            if not text or row.tie is not None:
                continue
            body = text.split("=", 1)[1] if "=" in text else text
            try:
                parents, coefficients = parse_tie(body, child=row.id)
            except ConstraintError as exc:
                self._refuse(f"{row.label}: cannot tie to '{text}': {exc}")
                return False
            expression = format_tie(parents, coefficients)
            actions.append(("tie", {"id": row.id, "expression": expression}))
        if not actions:
            return False
        self._emit_group(actions)
        return True

    # -- commands ---------------------------------------------------------------------
    def attach_commands(
        self, registry: CommandRegistry, *, route: bool = True
    ) -> None:
        """Bind the registry's table letters (Space, L, Shift+L, T, Del, I, ...).

        The context shortcuts are created on this table. With ``route`` the table also
        carries out :data:`TABLE_COMMANDS` whenever the registry triggers them (menu,
        button or letter); Delete only while the table has the focus.
        """
        self._registry = registry
        registry.attach_context("table", self)
        if route:
            registry.triggered.connect(self._on_registry_command)

    def _on_registry_command(self, command_id: str) -> None:
        if command_id in FOCUS_COMMANDS and not self.has_focus_within():
            return
        if command_id in TABLE_COMMANDS:
            self.handle_command(command_id)

    def has_focus_within(self) -> bool:
        """Whether the keyboard focus is on the table or one of its editors."""
        focus = QtWidgets.QApplication.focusWidget()
        return focus is not None and (focus is self or self.isAncestorOf(focus))

    def handle_command(self, command_id: str) -> bool:
        """Carry out ``command_id`` on the selected rows; False when not applicable."""
        if not self._editable or command_id not in TABLE_COMMANDS:
            return False
        if self.busy:
            self._refuse("Busy: wait for the running job, or press Stop (Esc)")
            return False
        rows = self._rows(self.selected_ids())
        if not rows:
            self._refuse("Select one or more components in the table first")
            return False
        actions: dict[str, Callable[[list[ComponentView]], bool]] = {
            "session.toggle_use": self._toggle_use,
            "session.fix_frequency": self._toggle_fix_frequency,
            "session.fix_all": self._toggle_fix_all,
            "session.tie": self._start_tie,
            "session.improve_selected": self._improve,
            "edit.delete": self._delete,
        }
        return actions[command_id](rows)

    def _toggle_use(self, rows: list[ComponentView]) -> bool:
        # Mixed selection: use them all; all in use: drop them all.
        active = not all(r.active for r in rows)
        self._emit("set_active", ids=[r.id for r in rows], active=active)
        return True

    def _toggle_fix_frequency(self, rows: list[ComponentView]) -> bool:
        free = [r for r in rows if r.tie is None]
        if not free:
            self._refuse("A tied frequency follows its tie; untie it to fix it")
            return False
        fix = not all(r.fix_frequency for r in free)
        self._emit("fix", ids=[r.id for r in free], frequency=fix)
        return True

    def _toggle_fix_all(self, rows: list[ComponentView]) -> bool:
        def held(r: ComponentView) -> bool:
            frequency = r.fix_frequency or r.tie is not None
            return frequency and r.fix_amplitude and r.fix_phase

        fix = not all(held(r) for r in rows)
        free = [r.id for r in rows if r.tie is None]
        tied = [r.id for r in rows if r.tie is not None]
        actions: list[tuple[str, Mapping[str, Any]]] = []
        if free:
            actions.append(
                ("fix", {"ids": free, "frequency": fix, "amplitude": fix, "phase": fix})
            )
        if tied:
            actions.append(("fix", {"ids": tied, "amplitude": fix, "phase": fix}))
        self._emit_group(actions)
        return True

    def _start_tie(self, rows: list[ComponentView]) -> bool:
        return self.edit_cell(rows[0].id, "tie")

    def edit_cell(self, ident: int, key: str) -> bool:
        """Open the editor of column ``key`` in the row of ``ident``."""
        row = self.row_of(ident)
        if row is None or key not in _COLUMN_OF or self.busy:
            return False
        column = _COLUMN_OF[key]
        self.setColumnHidden(column, False)
        item = self.item(row, column)
        if item is None or not item.flags() & Qt.ItemFlag.ItemIsEditable:
            return False
        self.setCurrentItem(item, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
        self.scrollToItem(item)
        self.editItem(item)
        return True

    def _improve(self, rows: list[ComponentView]) -> bool:
        self._emit("improve", ids=[r.id for r in rows])
        return True

    def _delete(self, rows: list[ComponentView]) -> bool:
        self._emit("remove", ids=[r.id for r in rows])
        return True

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        # Without a registry the table still answers its own letters; with one, the
        # registry's shortcuts take them first (and a disabled one must stay inert).
        if self._registry is None and self._editable and self._key_command(e):
            e.accept()
            return
        super().keyPressEvent(e)

    def _key_command(self, e: QtGui.QKeyEvent) -> bool:
        mods = e.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        shift = mods == Qt.KeyboardModifier.ShiftModifier
        plain = mods == Qt.KeyboardModifier.NoModifier
        key = e.key()
        command = None
        if plain and key == Qt.Key.Key_Space:
            command = "session.toggle_use"
        elif plain and key == Qt.Key.Key_L:
            command = "session.fix_frequency"
        elif shift and key == Qt.Key.Key_L:
            command = "session.fix_all"
        elif plain and key == Qt.Key.Key_T:
            command = "session.tie"
        elif plain and key == Qt.Key.Key_Delete:
            command = "edit.delete"
        elif shift and key == Qt.Key.Key_I:
            command = "session.improve_selected"
        if command is None:
            return False
        self.handle_command(command)
        return True

    # -- context menu -----------------------------------------------------------------
    def add_row_actions(self, menu: QtWidgets.QMenu) -> None:
        """The editable rows' actions (design 4.4) for the selected rows."""
        rows = self._rows(self.selected_ids())
        enabled = bool(rows) and not self.busy

        def add(
            target: QtWidgets.QMenu, text: str, slot: Callable[[], object],
            *, tip: str = "", on: bool = True,
        ) -> QtGui.QAction:
            action = target.addAction(text)
            action.setEnabled(enabled and on)
            if tip:
                action.setStatusTip(tip)
                action.setToolTip(tip)
            action.triggered.connect(lambda _=False: slot())
            return action

        def run(command: str) -> Callable[[], object]:
            return lambda: self.handle_command(command)

        ids = [r.id for r in rows]
        add(menu, "Improve selected\tShift+I", run("session.improve_selected"),
            tip="Re-fit the selected components only")
        fix_menu = menu.addMenu("Fix")
        fix_menu.setEnabled(enabled)
        add(fix_menu, "Frequency\tL", run("session.fix_frequency"))
        add(fix_menu, "Amplitude", lambda: self._emit("fix", ids=ids, amplitude=True))
        add(fix_menu, "Phase", lambda: self._emit("fix", ids=ids, phase=True))
        add(fix_menu, "All three\tShift+L", run("session.fix_all"))
        add(fix_menu, "Release all",
            lambda: self._release_all(rows),
            on=any(r.fix_frequency or r.fix_amplitude or r.fix_phase for r in rows))
        add(menu, "Tie…\tT", run("session.tie"),
            tip="Tie the frequency to a combination such as F1+F2")
        tied = [r.id for r in rows if r.tie is not None]
        add(menu, "Untie", lambda: self._untie(tied), on=bool(tied))
        combo = [r.id for r in rows if r.combination and r.tie is None]
        add(menu, "Tie to identified combination",
            lambda: self.tie_to_combination(*combo), on=bool(combo))
        variation = menu.addMenu("Per subset")
        variation.setEnabled(enabled)
        for mode, text in VARIATION_TEXT.items():
            choose = partial(self._emit, "set_variation", ids=ids, mode=mode)
            add(variation, text, choose)
        add(menu, "Use\tSpace", lambda: self._emit("set_active", ids=ids, active=True),
            on=any(not r.active for r in rows))
        add(menu, "Don't use\tSpace",
            lambda: self._emit("set_active", ids=ids, active=False),
            on=any(r.active for r in rows))
        add(menu, "Delete\tDel", run("edit.delete"),
            tip="Remove the selected components (Undo brings them back)")

    def _release_all(self, rows: list[ComponentView]) -> None:
        free = [r.id for r in rows if r.tie is None]
        tied = [r.id for r in rows if r.tie is not None]
        actions: list[tuple[str, Mapping[str, Any]]] = []
        if free:
            actions.append(
                ("fix", {"ids": free, "frequency": False, "amplitude": False,
                         "phase": False})
            )
        if tied:
            actions.append(("fix", {"ids": tied, "amplitude": False, "phase": False}))
        if actions:
            self._emit_group(actions)

    def _untie(self, ids: Sequence[int]) -> None:
        if ids:
            self._emit_group([("tie", {"id": i, "expression": None}) for i in ids])

    # -- copy -------------------------------------------------------------------------
    def row_texts(self, row: int) -> list[str]:
        """The cells of display row ``row`` as text (a bare check box as 1 / 0)."""
        cells: list[str] = []
        for column in range(self.columnCount()):
            item = self.item(row, column)
            if item is None:
                cells.append("")
                continue
            text = item.text().removeprefix("⚠ ")
            if not text and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                text = "1" if item.checkState() == Qt.CheckState.Checked else "0"
            cells.append(text)
        return cells


def summary_html(view: SessionView) -> str:
    """The Frequencies header of a session view (the builder already escaped it)."""
    lines = [view.summary_html]
    lines += [
        f"⚠ {html.escape(w, quote=False)}" for w in view.warnings if w.strip()
    ]
    return "<br>".join(lines)



__all__ = [
    "COLUMNS",
    "COLUMN_KEYS",
    "EMPTY_HINT",
    "FLAG_TEXT",
    "FOCUS_COMMANDS",
    "ID_ROLE",
    "TABLE_COMMANDS",
    "VARIATION_TEXT",
    "Entry",
    "FrequencyTable",
    "cell_for",
    "format_value",
    "parse_entry",
    "parse_number",
    "split_epoch",
    "split_group",
    "summary_html",
    "tie_frequency",
    "warning_flags",
]
