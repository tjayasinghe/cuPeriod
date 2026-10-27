"""The import dialog: a file's first rows, and which column is which.

Opening a light curve shows what the reader found -- the first rows, the column each
role was detected in -- and lets the user say what detection could not: the time,
brightness, error and band columns (error and band may be "none"), whether the
numbers are magnitudes or fluxes, and for a plain-text table how a row is split (the
delimiter, whether the first row holds the column names). A second tab shows the
first rows *as they will load*, every number at full precision, so a Julian date
reads ``2458000.123456`` and not ``2.458e+06``; a note under the pickers says what
scale the time column looks like (BJD, MJD, seconds, years ...).

The dialog never blocks: the window shows it with :meth:`~QtWidgets.QDialog.open`
and receives the user's :class:`~cuperiod.gui.loaders.ImportChoice` through
:attr:`PreviewDialog.choice_made` (or reads :meth:`PreviewDialog.choice` after the
``finished`` signal), then loads with :func:`~cuperiod.gui.loaders.load_path`.

Everything shown here comes from the file -- its name, its column headers, its
cells -- so every label is plain text: a header such as ``<img src=...>`` must read as
those characters, not be rendered (or fetched) as markup.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from cuperiod.core.columns import ColumnMap, Domain, infer_domain
from cuperiod.gui.loaders import (
    DELIMITERS,
    NO_COLUMN,
    FilePreview,
    ImportChoice,
    ReadOptions,
    describe_time,
    guess_columns,
    is_text_cell,
    is_text_table,
    numeric_cells,
    preview_file,
    sniff_delimiter,
)
from cuperiod.gui.qt import Qt, QtGui, QtWidgets, Signal

#: What to rename columns to when detection fails, for messages *outside* this dialog
#: (a failed load from a folder, where nobody was asked). Inside it the pickers are
#: the way out, see :data:`CHOOSE_ADVICE`.
UNDETECTED_ADVICE = (
    "Name the columns so they can be recognised — for example time, mjd, jd or bjd "
    "for the time; mag or flux for the brightness; mag_err or flux_err for its "
    "error; band or filter for a multiband file — and open it again. From Python, "
    "pass a ColumnMap to cuperiod.LightCurve.from_file instead."
)

#: The dialog's own advice when detection fails: choose the columns here.
CHOOSE_ADVICE = (
    "Choose them below. Columns named time, mjd, jd or bjd (the time), mag or flux "
    "(the brightness), mag_err or flux_err (its error) and band or filter (a "
    "multiband file) are recognised automatically."
)

#: How many preview rows the "As loaded" tab converts.
_LOADED_ROWS = 12

_ROLES = ("time", "value", "error", "band")
_ROLE_TEXT = {"time": "time", "value": "brightness", "error": "error", "band": "band"}


def _summary_text(preview: FilePreview, *, guessed: bool = False) -> str:
    if not preview.mapping:
        text = (
            f"{preview.n_rows} rows — could not tell which columns hold the time and "
            f"the brightness.\n{CHOOSE_ADVICE}"
        )
        if guessed:
            text += " A first guess is filled in: check it."
        return text
    parts = "    ".join(f"{role} = {col}" for role, col in preview.mapping.items())
    kind = "multiband" if preview.multiband else "single-band"
    return f"{preview.n_rows} rows · {preview.domain} · {kind}\nMapping:  {parts}"


def _plain_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


def _set_style_role(widget: QtWidgets.QWidget, name: str) -> None:
    """Switch a label between the stylesheet's ``#muted`` / ``#error`` looks."""
    if widget.objectName() == name:
        return
    widget.setObjectName(name)
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)


def _number_text(cell: str) -> str:
    """A cell as the loader will read it: the float at full (round-trip) precision."""
    values = numeric_cells([cell])
    if not values:
        return "not a number" if cell.strip() else "(empty)"
    return repr(values[0])


def _headerless_looking(preview: FilePreview) -> bool:
    """Whether the file has no header row: its "column names" are numbers (a data
    row read as the header), or the reader's own col1, col2, ... (it found none,
    e.g. under a commented header, and the box must not claim otherwise).

    A data row may lack a value, so ``nan``, ``inf`` and the reader's name for an
    empty cell (``Unnamed: 2``, ``col2``) pass as numbers, and so does a number the
    reader renamed for repeating one before it (``0.02_1``); at least one name must
    be a finite number.
    """
    names = [str(name) for name in preview.columns]
    if not names:
        return False
    if all(re.fullmatch(r"col\d+", name) for name in names):
        return True
    finite = False
    for j, name in enumerate(names):
        if re.fullmatch(r"col\d+|Unnamed: \d+", name):
            continue  # an empty cell
        repeated = re.fullmatch(r"(.+)[._]\d+", name)
        if repeated is not None and repeated.group(1) in names[:j]:
            name = repeated.group(1)
        if is_text_cell(name):
            return False
        finite = finite or bool(numeric_cells([name]))
    return finite


def _text_over_numbers(preview: FilePreview, column: str) -> str | None:
    """The first cell of ``column`` when it is text over numbers below it, else None.

    A missing value (an empty cell, ``nan``) is no text, and a column of text
    throughout (a band) has no numbers below.
    """
    if column not in preview.columns:
        return None
    j = preview.columns.index(column)
    cells = [row[j] for row in preview.rows if j < len(row)]
    if len(cells) >= 2 and is_text_cell(cells[0]) and numeric_cells(cells[1:]):
        return cells[0]
    return None


def _names_read_as_data(
    preview: FilePreview, time: str, columns: Sequence[str]
) -> list[str]:
    """The first-row cells of ``columns`` that look like column names read as data.

    Read without a header row, a file that has one starts with its column names
    (``time``, ``flux`` ...) above the data. Such a row names every column, the time
    column too, so it is taken for one only when the first cell of ``time`` is text
    over numbers; a data row with a value missing (``nan``, an empty cell) or a
    stray word in another column is not. Returns the text-over-numbers cells of
    ``time`` and ``columns``, or nothing.
    """
    if _text_over_numbers(preview, time) is None:
        return []
    found: list[str] = []
    for column in dict.fromkeys([time, *columns]):
        cell = _text_over_numbers(preview, column)
        if cell is not None:
            found.append(cell)
    return found


def _splits(preview: FilePreview | None) -> bool:
    """Whether a preview split the file into columns (two or more)."""
    return preview is not None and len(preview.columns) >= 2


class PreviewDialog(QtWidgets.QDialog):
    """Preview a file and choose its columns; Load emits :attr:`choice_made`.

    Parameters
    ----------
    filename : str
        Shown as the heading and in the title.
    preview : FilePreview
        :func:`~cuperiod.gui.loaders.preview_file` of the file.
    parent : QWidget, optional
    path : Path, optional
        The file itself. With it, a plain-text table offers the delimiter and
        header-row choices (re-previewing the file on each change) and the dialog
        sniffs a delimiter or a missing header row when the first read clearly
        missed one.
    columns, domain, read : optional
        What the user chose for an earlier file (:meth:`ImportChoice.remembered
        <cuperiod.gui.loaders.ImportChoice.remembered>` of the last choice). What
        the file shows of itself comes first: a column applies only where the file
        has it, and a "none" (:data:`~cuperiod.gui.loaders.NO_COLUMN`) only where
        detection found no error or band column; the ``domain`` applies only to a
        brightness column whose name says neither magnitude nor flux (``col2``,
        ``V``); ``read`` only to a text file that its own detection leaves in a
        single column.

    :meth:`choice` carries, besides what loads this file, what the next import
    starts from: what the user picked here, and for what they left alone, the
    earlier choice passed in. What detection found or a column's name implied is
    the next file's to find again.
    """

    #: The user pressed Load: carries the :class:`~cuperiod.gui.loaders.ImportChoice`.
    choice_made = Signal(object)

    def __init__(
        self,
        filename: str,
        preview: FilePreview,
        parent: QtWidgets.QWidget | None = None,
        *,
        path: str | Path | None = None,
        columns: ColumnMap | None = None,
        domain: Domain | None = None,
        read: ReadOptions | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Load {filename}")
        self.setAccessibleName(f"Load {filename}")
        self.resize(860, 620)
        self._path = Path(path) if path is not None else None
        self._preview = preview
        self._text = self._path is not None and is_text_table(self._path)
        #: The user changed a choice (then a working choice makes Load the default).
        self._touched = False
        #: The user set the domain in this dialog (then the value column no longer
        #: infers it). A remembered domain is not such a choice: see below.
        self._domain_touched = False
        #: The domain the user chose for an earlier file, for a brightness column
        #: whose name does not say (a flux file opened after a magnitude one must
        #: still read as flux).
        self._fallback_domain = domain
        #: The columns the user chose for an earlier file, by role (unset roles
        #: left out): each applies where this file allows, see :meth:`_wanted`.
        self._presets: dict[str, str] = {
            role: str(getattr(columns, role))
            for role in _ROLES
            if columns is not None and getattr(columns, role) is not None
        }
        #: The columns the user picked in this dialog, by role.
        self._mine: dict[str, str] = {}
        self._filling = False

        layout = QtWidgets.QVBoxLayout(self)
        self._heading = _plain_label(filename)
        self._heading.setObjectName("heading")
        layout.addWidget(self._heading)
        self._summary = _plain_label("")
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._summary)

        layout.addWidget(self._build_choices())
        self._time_note = _plain_label("")
        self._time_note.setObjectName("muted")
        self._time_note.setWordWrap(True)
        self._time_note.setAccessibleName("Time column note")
        layout.addWidget(self._time_note)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setAccessibleName("Preview of the first rows")
        self._table = self._make_table("First rows of the file")
        self._loaded = self._make_table("First rows as they will load")
        self._tabs.addTab(self._table, "File")
        self._tabs.addTab(self._loaded, "As loaded")
        self._tabs.setTabToolTip(0, "The first rows exactly as the file has them")
        self._tabs.setTabToolTip(
            1,
            "The first rows as cuPeriod will read them with the columns chosen "
            "above, every number at full precision",
        )
        layout.addWidget(self._tabs, 1)

        # What the reader noticed about the file as a whole (a header-only file, a
        # one-row-per-object table): the reason Load may be unavailable.
        self._note = _plain_label("")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)
        # Why the current choice cannot load (no time column, a text column as the
        # brightness ...); hidden while it can.
        self._problem = _plain_label("")
        self._problem.setObjectName("error")
        self._problem.setWordWrap(True)
        self._problem.setAccessibleName("Why Load is unavailable")
        layout.addWidget(self._problem)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons = buttons
        load_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert load_button is not None
        load_button.setText("Load")
        load_button.setToolTip("Load the file with the columns chosen above")
        self._load_button = load_button
        self._cancel_button = buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._start(preview, read)

    # -- construction ------------------------------------------------------------
    def _make_table(self, name: str) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(0, 0)
        table.setAccessibleName(name)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # Tab must leave the table for the Load / Cancel buttons, not walk its cells.
        table.setTabKeyNavigation(False)
        vheader = table.verticalHeader()
        if vheader is not None:
            vheader.setVisible(False)
        hheader = table.horizontalHeader()
        if hheader is not None:
            hheader.setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        return table

    def _combo(self, role: str, name: str, tip: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.setAccessibleName(name)
        combo.setToolTip(tip)
        combo.setStatusTip(tip)
        combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(14)
        combo.currentIndexChanged.connect(lambda _i, r=role: self._on_role_changed(r))
        return combo

    def _build_choices(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Columns")
        grid = QtWidgets.QGridLayout(box)
        self._combos: dict[str, QtWidgets.QComboBox] = {
            "time": self._combo(
                "time",
                "Time column",
                "The column holding the time of each observation, in days (JD, "
                "MJD, BJD, ...)",
            ),
            "value": self._combo(
                "value",
                "Brightness column",
                "The column holding the measured magnitude or flux",
            ),
            "error": self._combo(
                "error",
                "Error column",
                "The 1-sigma uncertainty of the brightness, or none (every point "
                "then weighs the same)",
            ),
            "band": self._combo(
                "band",
                "Band column",
                "A filter/band label per row, which splits the file into one "
                "light curve per band; none loads every row as one band",
            ),
        }
        texts = {
            "time": "&Time:",
            "value": "&Brightness:",
            "error": "&Error:",
            "band": "Ba&nd:",
        }
        for i, role in enumerate(_ROLES):
            label = QtWidgets.QLabel(texts[role])
            label.setBuddy(self._combos[role])
            grid.addWidget(label, i // 2, (i % 2) * 2)
            grid.addWidget(self._combos[role], i // 2, (i % 2) * 2 + 1)

        self._domain_combo = QtWidgets.QComboBox()
        self._domain_combo.addItem("Magnitudes", Domain.MAGNITUDE)
        self._domain_combo.addItem("Fluxes", Domain.FLUX)
        tip = (
            "How to read the brightness: magnitudes grow fainter as they increase, "
            "fluxes grow brighter. It decides which way a transit dips and the "
            "amplitude unit."
        )
        self._domain_combo.setAccessibleName("Brightness values are")
        self._domain_combo.setToolTip(tip)
        self._domain_combo.setStatusTip(tip)
        self._domain_combo.currentIndexChanged.connect(self._on_domain_changed)
        domain_label = QtWidgets.QLabel("&Values are:")
        domain_label.setBuddy(self._domain_combo)
        grid.addWidget(domain_label, 2, 0)
        grid.addWidget(self._domain_combo, 2, 1)

        self._delimiter = QtWidgets.QComboBox()
        for text, value in DELIMITERS:
            self._delimiter.addItem(text, value)
        tip = (
            "How a row of this text file splits into columns. Automatic reads it "
            "as the extension says (comma for .csv, tab for .tsv, spaces otherwise)."
        )
        self._delimiter.setAccessibleName("Delimiter")
        self._delimiter.setToolTip(tip)
        self._delimiter.setStatusTip(tip)
        self._delimiter.currentIndexChanged.connect(self._on_read_changed)
        self._header = QtWidgets.QCheckBox("First row holds the column &names")
        tip = (
            "Untick for a file that starts straight with numbers (the JD mag err "
            "tables of OGLE or Period04); its columns are then col1, col2, ..."
        )
        self._header.setAccessibleName("First row holds the column names")
        self._header.setToolTip(tip)
        self._header.setStatusTip(tip)
        self._header.setChecked(True)
        self._header.toggled.connect(self._on_read_changed)
        delimiter_label = QtWidgets.QLabel("&Delimiter:")
        delimiter_label.setBuddy(self._delimiter)
        grid.addWidget(delimiter_label, 3, 0)
        grid.addWidget(self._delimiter, 3, 1)
        grid.addWidget(self._header, 3, 2, 1, 2)
        for widget in (delimiter_label, self._delimiter, self._header):
            widget.setVisible(self._text)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return box

    # -- state -------------------------------------------------------------------
    def _start(self, preview: FilePreview, read: ReadOptions | None) -> None:
        """Show ``preview``; with the file at hand, fix an obviously wrong first read.

        A text file read into a single column (a semicolon or tab file) is re-read
        with the delimiter it appears to use, and one whose "column names" are
        numbers (no header row) is re-read without a header, so the user starts from
        a table that looks like theirs. The previous file's options (``read``) are
        tried only when that still leaves a single column, and kept only if they
        split it: a file that shows how it reads is never read the last file's way
        (a header CSV opened after a headerless OGLE table keeps its header).
        """
        if self._text and self._path is not None:
            options, detected = self._detect_read(self._path, preview)
            if read is not None and read != options and not _splits(detected):
                remembered = preview_file(self._path, read=read)
                if remembered is not None and _splits(remembered):
                    # header=False on a file that has one reads its names as data.
                    time = remembered.columns[0]
                    preset_time = self._presets.get("time")
                    if preset_time is not None and preset_time in remembered.columns:
                        time = preset_time
                    names_as_data = not read.header and bool(
                        _names_read_as_data(remembered, time, remembered.columns)
                    )
                    if not names_as_data:
                        options, detected = read, remembered
            if detected is not None:
                preview = detected
            self._filling = True
            try:
                index = self._delimiter.findData(options.delimiter)
                self._delimiter.setCurrentIndex(max(index, 0))
                self._header.setChecked(options.header)
            finally:
                self._filling = False
        self._show_preview(preview)

    @staticmethod
    def _detect_read(
        path: Path, preview: FilePreview
    ) -> tuple[ReadOptions, FilePreview | None]:
        """How a text file splits by its own look, and its preview read that way.

        A single column is re-split on the delimiter :func:`sniff_delimiter` finds;
        "column names" that are numbers (or the reader's ``col1, col2, ...``) mean
        there is no header row.
        """
        options = ReadOptions()
        current: FilePreview | None = preview
        if len(preview.columns) == 1:
            sniffed = sniff_delimiter(path)
            if sniffed is not None:
                options = ReadOptions(delimiter=sniffed)
                current = preview_file(path, read=options)
        if current is not None and _headerless_looking(current):
            options = ReadOptions(delimiter=options.delimiter, header=False)
            current = preview_file(path, read=options)
        return options, current

    def _show_preview(self, preview: FilePreview) -> None:
        """Fill everything from ``preview``, keeping the choices that still apply."""
        self._preview = preview
        guess = guess_columns(preview)
        guessed = not preview.mapping and bool(guess)
        wanted = {role: self._wanted(role, preview, guess) for role in _ROLES}
        self._filling = True
        try:
            for role in _ROLES:
                self._fill_combo(role, wanted[role])
            if self._domain_touched:
                chosen_domain = self.domain()
            else:
                # The brightness column's name decides (flux, mag_g ...); the
                # previous file's domain fills in only for a name that says neither.
                value = wanted.get("value")
                inferred = infer_domain(value) if value else None
                chosen_domain = inferred or self._fallback_domain or Domain.MAGNITUDE
            self._show_domain(chosen_domain)
        finally:
            self._filling = False

        self._summary.setText(_summary_text(preview, guessed=guessed))
        _set_style_role(self._summary, "muted" if preview.mapping else "error")
        self._note.setText(preview.note)
        _set_style_role(self._note, "muted" if preview.loadable else "error")
        self._note.setVisible(bool(preview.note))
        self._fill_file_table()
        self._refresh()

    def _wanted(
        self, role: str, preview: FilePreview, guess: Mapping[str, str]
    ) -> str | None:
        """The column ``role`` shows for ``preview``.

        The user's pick in this dialog comes first while the file (re-read) still
        has it; then the earlier file's choice: a column the file has, or "none"
        where detection found no column for the role (the last file's lack of an
        error or band column must not drop this one's); then detection, or the
        guess when there was none.
        """
        mine = self._mine.get(role)
        if mine is not None and (mine == NO_COLUMN or mine in preview.columns):
            return mine
        preset = self._presets.get(role)
        if preset is not None and preset in preview.columns:
            return preset
        if (
            preset == NO_COLUMN
            and role in ("error", "band")
            and role not in preview.mapping
        ):
            return preset
        return guess.get(role)

    def _fill_combo(self, role: str, wanted: str | None) -> None:
        combo = self._combos[role]
        combo.clear()
        optional = role in ("error", "band")
        if optional:
            combo.addItem("none", NO_COLUMN)
        else:
            combo.addItem("(choose a column)", None)
        first = self._preview.rows[0] if self._preview.rows else []
        nested = set(self._preview.nested_columns)
        for j, name in enumerate(self._preview.columns):
            sample = first[j] if j < len(first) else ""
            text = f"{name}   e.g. {sample}" if sample else name
            combo.addItem(text, name)
            if name in nested and role != "band":
                index = combo.count() - 1
                combo.setItemData(
                    index,
                    "holds an array per row (one row per object), not one value",
                    Qt.ItemDataRole.ToolTipRole,
                )
                # Disabled: the reader cannot turn arrays into one value per row.
                model = combo.model()
                if isinstance(model, QtGui.QStandardItemModel):
                    item = model.item(index)
                    if item is not None:
                        item.setEnabled(False)
        index = combo.findData(wanted) if wanted is not None else -1
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _fill_file_table(self) -> None:
        table = self._table
        preview = self._preview
        table.clear()
        table.setRowCount(len(preview.rows))
        table.setColumnCount(len(preview.columns))
        for r, row in enumerate(preview.rows):
            for c, text in enumerate(row):
                table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        self._label_file_columns()

    def _label_file_columns(self) -> None:
        role_of = {
            col: _ROLE_TEXT[role]
            for role, col in self._chosen().items()
            if col is not None and col != NO_COLUMN
        }
        # The role sits in the header's text (not in a colour) so it is read out too.
        self._table.setHorizontalHeaderLabels(
            [
                f"{c}\n({role_of[c]})" if c in role_of else c
                for c in self._preview.columns
            ]
        )

    # -- choices -----------------------------------------------------------------
    def _chosen(self) -> dict[str, str | None]:
        """Role -> chosen column (:data:`NO_COLUMN` for "none", None for unset)."""
        out: dict[str, str | None] = {}
        for role, combo in self._combos.items():
            data = combo.currentData()
            out[role] = None if data is None else str(data)
        return out

    def _on_role_changed(self, role: str) -> None:
        if self._filling:
            return
        self._touched = True
        picked = self._chosen().get(role)
        if picked is None:
            self._mine.pop(role, None)
        else:
            self._mine[role] = picked
        if role == "value" and not self._domain_touched:
            value = self._chosen().get("value")
            inferred = infer_domain(value) if value else None
            if inferred is not None:
                self._filling = True
                try:
                    self._show_domain(inferred)
                finally:
                    self._filling = False
        self._label_file_columns()
        self._refresh()

    def _show_domain(self, domain: Domain) -> None:
        self._domain_combo.setCurrentIndex(self._domain_combo.findData(domain))

    def _on_domain_changed(self, _index: int) -> None:
        if self._filling:
            return
        self._domain_touched = True
        self._touched = True
        self._refresh()

    def _on_read_changed(self, *_args: object) -> None:
        if self._filling or self._path is None:
            return
        self._touched = True
        fresh = preview_file(self._path, read=self.read_options() or ReadOptions())
        if fresh is not None:
            self._show_preview(fresh)

    def _problem_text(self) -> str:
        """Why the current choice cannot load ("" when it can)."""
        preview = self._preview
        if not preview.loadable:
            return ""  # the note already says why, in the file's own terms
        chosen = self._chosen()
        time, value = chosen.get("time"), chosen.get("value")
        if time is None:
            return "Choose the time column."
        if value is None:
            return "Choose the brightness column."
        if time == value:
            return "The time and the brightness must be different columns."
        error, band = chosen.get("error"), chosen.get("band")
        if error not in (None, NO_COLUMN) and error in (time, value):
            return "The error column must differ from the time and brightness."
        if band not in (None, NO_COLUMN) and band in (time, value, error):
            return "The band column must differ from the other columns."
        nested = set(preview.nested_columns)
        for role in ("time", "value", "error"):
            column = chosen.get(role)
            if column in nested:
                return (
                    f"Column {column!r} holds an array per row (one row per object); "
                    "cuPeriod reads one observation per row."
                )
        for role, column in (("time", time), ("brightness", value)):
            cells = self._cells(column)
            if cells and not numeric_cells(cells):
                return (
                    f"Column {column!r} holds no numbers in the first rows, so it "
                    f"cannot be the {role}."
                )
        if self._text and not self._header.isChecked():
            numeric = [c for c in (value, error) if c not in (None, NO_COLUMN)]
            above = _names_read_as_data(preview, time, [str(c) for c in numeric])
            if above:
                listed = ", ".join(repr(cell) for cell in above)
                return (
                    f"The first row holds text ({listed}) above numbers: it looks "
                    "like the column names. Tick “First row holds the column names”."
                )
        return ""

    def _cells(self, column: str | None) -> list[str]:
        if column is None or column not in self._preview.columns:
            return []
        j = self._preview.columns.index(column)
        return [row[j] for row in self._preview.rows if j < len(row)]

    def _refresh(self) -> None:
        """Re-derive the time note, the as-loaded rows and whether Load is offered."""
        chosen = self._chosen()
        time = chosen.get("time")
        if time is not None and self._preview.rows:
            note = describe_time(time, self._cells(time))
            self._time_note.setText(note)
            _set_style_role(
                self._time_note, "error" if note.startswith("Warning") else "muted"
            )
            self._time_note.setVisible(True)
        else:
            self._time_note.setVisible(False)
        self._fill_loaded_table(chosen)

        problem = self._problem_text()
        self._problem.setText(problem)
        self._problem.setVisible(bool(problem))
        can_load = self._preview.loadable and not problem
        self._load_button.setEnabled(can_load)
        self._load_button.setToolTip(
            "Load the file with the columns chosen above"
            if can_load
            else (problem or self._preview.note or "This file cannot be loaded")
        )
        # Enter loads only a choice the user can trust: one detection found or one
        # they made themselves. A guess, or no choice at all, leaves Enter on Cancel.
        trusted = bool(self._preview.mapping) or self._touched
        primary = can_load and trusted
        self._load_button.setProperty("primary", primary)
        style = self._load_button.style()
        if style is not None:
            style.unpolish(self._load_button)
            style.polish(self._load_button)
        default = self._load_button if primary else self._cancel_button
        other = self._cancel_button if primary else self._load_button
        if other is not None:
            other.setDefault(False)
        if default is not None:
            default.setDefault(True)

    def _fill_loaded_table(self, chosen: dict[str, str | None]) -> None:
        table = self._loaded
        unit = "flux" if self.domain() is Domain.FLUX else "mag"
        headers = {
            "time": "time (d)",
            "value": f"brightness ({unit})",
            "error": "error",
            "band": "band",
        }
        roles = [
            role
            for role in _ROLES
            if chosen.get(role) is not None and chosen.get(role) != NO_COLUMN
        ]
        rows = self._preview.rows[:_LOADED_ROWS]
        table.clear()
        table.setColumnCount(len(roles))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(
            [f"{headers[role]}\n← {chosen[role]}" for role in roles]
        )
        for c, role in enumerate(roles):
            cells = self._cells(chosen[role])
            for r in range(len(rows)):
                cell = cells[r] if r < len(cells) else ""
                text = cell if role == "band" else _number_text(cell)
                table.setItem(r, c, QtWidgets.QTableWidgetItem(text))

    # -- results -----------------------------------------------------------------
    def domain(self) -> Domain:
        """The brightness domain chosen."""
        return Domain.FLUX if self._domain_combo.currentIndex() == 1 else (
            Domain.MAGNITUDE
        )

    def read_options(self) -> ReadOptions | None:
        """How to split a text table, or None for the reader's own default."""
        if not self._text:
            return None
        options = ReadOptions(
            delimiter=self._delimiter.currentData(), header=self._header.isChecked()
        )
        return None if options.is_default else options

    def column_map(self) -> ColumnMap:
        """The chosen columns (:data:`~cuperiod.gui.loaders.NO_COLUMN` for "none").

        A band column left as detected stays unpinned, so the loader's guard against
        an identifier column posing as a band (``MAX_AUTO_BANDS``) still applies.
        """
        chosen = self._chosen()
        band = chosen.get("band")
        if band is not None and band == self._preview.mapping.get("band"):
            band = None
        return ColumnMap(
            time=chosen.get("time"),
            value=chosen.get("value"),
            error=chosen.get("error"),
            band=band,
        )

    def choice(self) -> ImportChoice:
        """Everything :func:`~cuperiod.gui.loaders.load_path` needs, and what the next
        import starts from (:meth:`~cuperiod.gui.loaders.ImportChoice.remembered`)."""
        return ImportChoice(
            columns=self.column_map(),
            domain=self.domain(),
            read=self.read_options(),
            remember=self._carried_over(),
        )

    def _carried_over(self) -> ImportChoice:
        """What the next import starts from: the user's choices, never detection's.

        A column the user picked here (and still shown) replaces the earlier file's
        choice for its role; a role they left alone keeps that earlier choice, used
        here or not. A column detection found, or a guess, is not carried: the next
        file's own detection finds its own (a "none" carried from a file without an
        error column dropped the next file's errors). Likewise the domain: one set
        here, else the earlier choice, never one inferred from a column's name (a
        flux CSV made the next OGLE table load as fluxes). The read options are kept
        as used: they apply only where the next file's own look leaves one column.
        """
        chosen = self._chosen()
        columns: dict[str, str | None] = {}
        for role in _ROLES:
            mine = self._mine.get(role)
            shown = mine is not None and chosen.get(role) == mine
            columns[role] = mine if shown else self._presets.get(role)
        domain = self.domain() if self._domain_touched else self._fallback_domain
        return ImportChoice(
            columns=ColumnMap(**columns), domain=domain, read=self.read_options()
        )

    def can_load(self) -> bool:
        """Whether Load is offered for the current choice."""
        return self._load_button.isEnabled()

    def accept(self) -> None:
        if not self._load_button.isEnabled():
            return  # Enter on a disabled Load (or a stray accept()) loads nothing
        self.choice_made.emit(self.choice())
        super().accept()


#: The dialog's role in the window: what the user imports a file through.
ImportDialog = PreviewDialog

__all__ = ["CHOOSE_ADVICE", "UNDETECTED_ADVICE", "ImportDialog", "PreviewDialog"]
