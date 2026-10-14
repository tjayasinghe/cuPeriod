"""Private helpers shared by the plot views and the result tables.

Kept in one place because each of them guards against the same mistake in several
widgets: text from a data file reaching a rich-text renderer, a number format being
silently replaced by Qt's, a keyboard shortcut leaking out of its table, an export
failing without a word, and one-device-pixel strokes on a HiDPI screen.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pyqtgraph.exporters import ImageExporter

from cuperiod.gui.qt import QObject, Qt, QtCore, QtGui, QtWidgets

#: Longest band name a legend shows before eliding it. A legend sits on the data, so
#: a long label (or a hostile one, see :func:`legend_label`) must not cover the plot.
LEGEND_MAX_CHARS = 40


def legend_label(name: str) -> str:
    """A band name made safe for a pyqtgraph legend.

    pyqtgraph renders legend text as HTML, and the band names come straight from the
    band column of the opened file: ``r<22`` would lose everything after ``<``, and an
    ``<img src=//host/share/x.png>`` cell would make Qt open that path (an SMB request
    on Windows) as soon as the file loads. The name is elided *before* escaping, so the
    cut can never land inside an entity.
    """
    text = str(name)
    if len(text) > LEGEND_MAX_CHARS:
        text = text[: LEGEND_MAX_CHARS - 1] + "…"
    return html.escape(text, quote=False)


class NumericItem(QtWidgets.QTableWidgetItem):
    """A table item that sorts by a number while showing its own formatted text.

    The sort key is kept on the instance rather than written to ``EditRole``:
    :class:`QTableWidgetItem` stores ``EditRole`` and ``DisplayRole`` in the same slot,
    so a float written there silently replaces the formatted text with Qt's
    locale-grouped six-significant-digit rendering. A non-finite value sorts last.
    """

    def __init__(self, value: float, text: str) -> None:
        super().__init__(text)
        number = float(value)
        self.value = number
        self._key = number if not math.isnan(number) else math.inf

    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:
        if isinstance(other, NumericItem):
            return self._key < other._key
        return super().__lt__(other)


def configure_result_table(
    table: QtWidgets.QTableWidget, copy_slot: Callable[[], None]
) -> QtGui.QShortcut:
    """Keyboard behaviour shared by the peaks and frequency tables.

    Tab leaves the table instead of walking its cells (arrows, Home/End and PgUp/PgDn
    still move between cells), so a keyboard user is never trapped in it. Ctrl+C copies
    the selected rows only while the focus is in *this* table: a window-wide shortcut
    stole the key from every other widget, and two visible tables made it ambiguous so
    that neither copied anything.
    """
    table.setTabKeyNavigation(False)
    shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Copy, table)
    shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
    shortcut.activated.connect(copy_slot)
    return shortcut


def select_only_row(table: QtWidgets.QTableWidget, row: int | None) -> None:
    """Make ``row`` the one selected (and current) row; ``None`` clears the selection.

    :meth:`QTableView.selectRow` honours the keyboard modifiers held at the time, so in
    an extended-selection table a selection pushed in from elsewhere (a Ctrl-click on a
    plot marker) would *toggle* the row or add it to the old one instead.
    """
    selection = table.selectionModel()
    model = table.model()
    if selection is None or model is None:
        return
    if row is None:
        selection.clearSelection()
        return
    flags = (
        QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QtCore.QItemSelectionModel.SelectionFlag.Rows
    )
    selection.setCurrentIndex(model.index(row, 0), flags)


def table_action_bar(
    copy_slot: Callable[[], None], export_slot: Callable[[], None], what: str
) -> QtWidgets.QWidget:
    """A slim row of visible "Copy" / "Export CSV…" buttons above a result table.

    The same actions are on the table's right-click menu, where nobody finds them.
    """
    bar = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(bar)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    row.addStretch(1)
    copy_button = QtWidgets.QPushButton("Copy")
    copy_button.setToolTip(
        f"Copy the selected {what} as tab-separated text (Ctrl+C in the table)"
    )
    copy_button.clicked.connect(copy_slot)
    row.addWidget(copy_button)
    export_button = QtWidgets.QPushButton("Export CSV…")
    export_button.setToolTip(f"Save all the {what} with full precision as CSV")
    export_button.clicked.connect(export_slot)
    row.addWidget(export_button)
    return bar


def column_menu(table: QtWidgets.QTableWidget) -> QtWidgets.QMenu:
    """A menu of checkable column names that show or hide the table's columns."""
    menu = QtWidgets.QMenu(table)
    visible = [c for c in range(table.columnCount()) if not table.isColumnHidden(c)]
    for column in range(table.columnCount()):
        item = table.horizontalHeaderItem(column)
        action = menu.addAction(item.text() if item is not None else str(column))
        action.setCheckable(True)
        action.setChecked(not table.isColumnHidden(column))
        # Never let the last visible column be hidden: the table would vanish.
        action.setEnabled(visible != [column])
        action.toggled.connect(
            lambda checked, c=column: table.setColumnHidden(c, not checked)
        )
    return menu


def install_column_chooser(table: QtWidgets.QTableWidget) -> None:
    """Right-clicking the header offers a checkable list of the columns to show.

    Hiding a column only affects the view; copy and CSV export keep every column.
    """
    header = table.horizontalHeader()
    if header is None:
        return
    header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def show_menu(pos: QtCore.QPoint) -> None:
        # popup(), not exec(): a nested event loop would let queued session jobs
        # re-enter the controller while the menu is open.
        menu = column_menu(table)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        menu.popup(header.mapToGlobal(pos))

    header.customContextMenuRequested.connect(show_menu)


def is_connected(owner: QObject, signal: Any) -> bool:
    """Whether anything listens to ``signal`` (a signal instance of ``owner``)."""
    return bool(owner.isSignalConnected(QtCore.QMetaMethod.fromSignal(signal)))


def report_failure(widget: QtWidgets.QWidget, signal: Any, message: str) -> None:
    """Tell the user an action failed: through ``signal`` if it is wired, else a box.

    The main window turns the signal into its error banner. A widget used on its own
    still reports the failure, with a non-blocking message box rather than a modal
    ``exec()`` (which would stall a headless session).
    """
    if is_connected(widget, signal):
        signal.emit(message)
        return
    box = QtWidgets.QMessageBox(
        QtWidgets.QMessageBox.Icon.Warning, "cuPeriod", message, parent=widget
    )
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    box.open()


def writable_image_suffixes() -> frozenset[str]:
    """Lower-case file suffixes (``.png``, ``.jpg``, ...) Qt can write images to."""
    return frozenset(
        "." + bytes(fmt.data()).decode("ascii", "replace").lower()
        for fmt in QtGui.QImageWriter.supportedImageFormats()
    )


def image_target(path: str | Path) -> Path:
    """The file an image export to ``path`` writes: ``.png`` added unless writable.

    Qt picks the image format from the suffix and writes nothing at all for a name it
    does not recognise (``plot``, ``run1.v2``, ``plot.dat``), which is what a save
    dialog that does not append the filter's suffix hands back.
    """
    target = Path(path)
    if target.suffix.lower() not in writable_image_suffixes():
        target = target.with_name(target.name + ".png")
    return target


def device_scale(widget: QtWidgets.QWidget) -> int:
    """Whole device pixels per logical pixel of ``widget`` (1 on a standard screen).

    Rounded because the spectrum is drawn without antialiasing, where a fractional
    stroke width is rasterised inconsistently.
    """
    return max(1, round(widget.devicePixelRatioF()))


def stroke_width(widget: QtWidgets.QWidget, width: float = 1.0) -> float:
    """A pen width that keeps a stroke ``width`` *logical* pixels wide on any screen.

    pyqtgraph pens are cosmetic, so their width is in device pixels: a 1-px curve is a
    0.5-logical-pixel hairline at 200% scaling while text and widgets scale up.
    """
    return width * device_scale(widget)


def write_plot_image(
    item: Any, path: str | Path, *, scale: int = 1
) -> Path:
    """Render a plot item to an image file and return the path actually written.

    ``scale`` renders at that many pixels per logical pixel, matching the device-scaled
    strokes of a HiDPI screen. Raises :class:`OSError` when nothing could be written
    (an unwritable location, a read-only file); pyqtgraph's exporter reports that only
    through a return value that is easy to drop.
    """
    target = image_target(path)
    try:
        exporter = ImageExporter(item)
        if scale > 1:
            params = exporter.parameters()
            params["width"] = int(params["width"]) * scale
        image = exporter.export(toBytes=True)
    except Exception as exc:  # pyqtgraph raises a bare Exception for a 0-size plot
        raise OSError(f"Could not render the plot: {exc}") from exc
    if image is None or image.isNull() or not image.save(str(target)):
        raise OSError(f"Could not write the image to {target}")
    return target


def ask_save_path(
    widget: QtWidgets.QWidget,
    provider: Any,
    title: str,
    suggested: str,
    filters: str,
    done: Callable[[str], None],
) -> None:
    """Ask where to save, then call ``done(path)`` (never when cancelled).

    Through ``provider`` (the window's
    :class:`~cuperiod.gui.dialogs.DialogProvider`) when the window gave the view one:
    a dialog shown with ``open()`` that tests script. A view used on its own falls
    back to the static file dialog; the window's views used it too, a nested event
    loop the GUI otherwise never runs.
    """
    if provider is not None:

        def chosen(path: str | None) -> None:
            if path:
                done(path)

        provider.save_file(title, filters, suggested, chosen)
        return
    path, _ = QtWidgets.QFileDialog.getSaveFileName(widget, title, suggested, filters)
    if path:
        done(path)


def confirm_overwrite(
    parent: QtWidgets.QWidget, chosen: str, target: Path
) -> bool:
    """Ask before overwriting a file the save dialog never showed the user.

    The dialog's own overwrite prompt covered ``chosen``; when a suffix had to be
    appended, ``target`` is a different file that may already exist.
    """
    if str(target) == chosen or not target.exists():
        return True
    answer = QtWidgets.QMessageBox.question(
        parent,
        "cuPeriod",
        f"{target.name} already exists. Replace it?",
    )
    return answer == QtWidgets.QMessageBox.StandardButton.Yes


__all__ = [
    "LEGEND_MAX_CHARS",
    "NumericItem",
    "ask_save_path",
    "column_menu",
    "configure_result_table",
    "confirm_overwrite",
    "device_scale",
    "image_target",
    "install_column_chooser",
    "is_connected",
    "legend_label",
    "report_failure",
    "select_only_row",
    "stroke_width",
    "table_action_bar",
    "writable_image_suffixes",
    "write_plot_image",
]
