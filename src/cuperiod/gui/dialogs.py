"""Every dialog behind one injectable provider, shown without a nested event loop.

The controller and the window never construct a dialog themselves: they call a
:class:`DialogProvider` and receive the answer through a callback. The Qt
implementation shows each dialog with ``open()`` (window-modal, returns at once) and
answers from its ``finished`` / ``fileSelected`` signals, never with ``exec()``: a
nested event loop would let queued session jobs re-enter the controller mid-dialog,
and it hangs offscreen test runs. :class:`FakeDialogProvider` answers from a script so
tests can drive every prompt synchronously.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from cuperiod.gui.qt import QObject, Qt, QtWidgets

if TYPE_CHECKING:
    from cuperiod.gui.loaders import FilePreview, ImportChoice

DiscardChoice = Literal["save", "discard", "cancel"]
Severity = Literal["info", "warning", "error"]


@runtime_checkable
class DialogProvider(Protocol):
    """What the GUI may ask the user; every answer arrives through ``done``."""

    def confirm_discard(
        self, names: Sequence[str], done: Callable[[DiscardChoice], None]
    ) -> None:
        """Unsaved sessions would be lost: Save / Discard / Cancel."""

    def open_file(
        self,
        title: str,
        filters: str,
        done: Callable[[str | None], None],
        *,
        directory: str = "",
    ) -> None:
        """A path to open (starting in ``directory``), or ``None`` when cancelled."""

    def open_folder(
        self, title: str, done: Callable[[str | None], None], *, directory: str = ""
    ) -> None:
        """A folder to open (batch browsing), or ``None`` when cancelled."""

    def save_file(
        self,
        title: str,
        filters: str,
        suggested: str,
        done: Callable[[str | None], None],
    ) -> None:
        """A path to write, or ``None`` when cancelled."""

    def ask_text(
        self, title: str, label: str, text: str, done: Callable[[str | None], None]
    ) -> None:
        """A line of text (a tie expression, a note), or ``None``."""

    def ask_float(
        self,
        title: str,
        label: str,
        value: float,
        done: Callable[[float | None], None],
    ) -> None:
        """A number (a frequency to add), or ``None``."""

    def inform(self, title: str, text: str, severity: Severity = "info") -> None:
        """A message the user acknowledges; nothing waits for it."""

    def confirm(
        self, title: str, text: str, accept: str, done: Callable[[bool], None]
    ) -> None:
        """A yes-or-no question: ``done(True)`` when the user chose the ``accept``
        button, ``done(False)`` for Cancel, Esc or closing it."""

    def show_html(self, title: str, html: str) -> None:
        """A read-only rich-text sheet (the keyboard shortcuts)."""

    def import_file(
        self,
        path: str,
        preview: FilePreview,
        done: Callable[[ImportChoice | None], None],
        *,
        last: ImportChoice | None = None,
    ) -> None:
        """How to read ``path`` (columns, brightness scale, delimiter), or ``None``.

        ``last`` pre-fills the previous choice (names the file lacks are ignored).
        """


class QtDialogProvider(QObject):
    """:class:`DialogProvider` with Qt dialogs shown by ``open()``.

    Each live dialog is kept referenced until it finishes, then scheduled for
    deletion; a callback runs exactly once.
    """

    def __init__(self, parent: QtWidgets.QWidget, *, native_files: bool = True) -> None:
        super().__init__(parent)
        self._parent = parent
        self._native_files = native_files
        self._live: set[QtWidgets.QDialog] = set()

    @property
    def live_dialogs(self) -> tuple[QtWidgets.QDialog, ...]:
        """Dialogs currently shown (tests use it to answer them)."""
        return tuple(self._live)

    def _track(self, dialog: QtWidgets.QDialog) -> None:
        self._live.add(dialog)

        def _gone(*_: Any) -> None:
            self._live.discard(dialog)
            dialog.deleteLater()

        dialog.finished.connect(_gone)

    def confirm_discard(
        self, names: Sequence[str], done: Callable[[DiscardChoice], None]
    ) -> None:
        box = QtWidgets.QMessageBox(self._parent)
        box.setObjectName("dialog_confirm_discard")
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved pre-whitening work")
        listed = "\n".join(f"• {n}" for n in names)
        box.setText("These sessions have changes that are not saved:")
        box.setInformativeText(listed + "\n\nSave them before continuing?")
        buttons = QtWidgets.QMessageBox.StandardButton
        box.setStandardButtons(buttons.Save | buttons.Discard | buttons.Cancel)
        box.setDefaultButton(buttons.Save)
        box.setEscapeButton(buttons.Cancel)
        answered: list[bool] = []

        def _clicked(button: QtWidgets.QAbstractButton) -> None:
            if answered:
                return
            answered.append(True)
            role = box.standardButton(button)
            choice: DiscardChoice = (
                "save"
                if role == buttons.Save
                else "discard"
                if role == buttons.Discard
                else "cancel"
            )
            done(choice)

        def _finished(_: int) -> None:
            if not answered:  # closed by Esc or the title bar
                answered.append(True)
                done("cancel")

        box.buttonClicked.connect(_clicked)
        box.finished.connect(_finished)
        self._track(box)
        box.open()

    def _file_dialog(
        self,
        title: str,
        filters: str,
        mode: QtWidgets.QFileDialog.AcceptMode,
        suggested: str,
        done: Callable[[str | None], None],
        *,
        folder: bool = False,
    ) -> None:
        dialog = QtWidgets.QFileDialog(self._parent, title, suggested, filters)
        dialog.setObjectName(
            "dialog_save_file"
            if mode == QtWidgets.QFileDialog.AcceptMode.AcceptSave
            else "dialog_open_file"
        )
        dialog.setAcceptMode(mode)
        if not self._native_files:  # offscreen tests: Qt's own widget dialog
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setFileMode(
            QtWidgets.QFileDialog.FileMode.AnyFile
            if mode == QtWidgets.QFileDialog.AcceptMode.AcceptSave
            else QtWidgets.QFileDialog.FileMode.Directory
            if folder
            else QtWidgets.QFileDialog.FileMode.ExistingFile
        )
        if folder:
            dialog.setObjectName("dialog_open_folder")
            dialog.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly, True)
        answered: list[bool] = []

        def _finished(result: int) -> None:
            if answered:
                return
            answered.append(True)
            files = dialog.selectedFiles()
            accepted = result == QtWidgets.QDialog.DialogCode.Accepted
            done(files[0] if accepted and files else None)

        dialog.finished.connect(_finished)
        self._track(dialog)
        dialog.open()

    def open_file(
        self,
        title: str,
        filters: str,
        done: Callable[[str | None], None],
        *,
        directory: str = "",
    ) -> None:
        self._file_dialog(
            title, filters, QtWidgets.QFileDialog.AcceptMode.AcceptOpen, directory, done
        )

    def open_folder(
        self, title: str, done: Callable[[str | None], None], *, directory: str = ""
    ) -> None:
        self._file_dialog(
            title,
            "",
            QtWidgets.QFileDialog.AcceptMode.AcceptOpen,
            directory,
            done,
            folder=True,
        )

    def save_file(
        self,
        title: str,
        filters: str,
        suggested: str,
        done: Callable[[str | None], None],
    ) -> None:
        self._file_dialog(
            title, filters, QtWidgets.QFileDialog.AcceptMode.AcceptSave, suggested, done
        )

    def _input(
        self,
        title: str,
        label: str,
        configure: Callable[[QtWidgets.QInputDialog], None],
        read: Callable[[QtWidgets.QInputDialog], Any],
        done: Callable[[Any], None],
    ) -> None:
        dialog = QtWidgets.QInputDialog(self._parent)
        dialog.setObjectName("dialog_input")
        dialog.setWindowTitle(title)
        dialog.setLabelText(label)
        configure(dialog)
        answered: list[bool] = []

        def _finished(result: int) -> None:
            if answered:
                return
            answered.append(True)
            ok = result == QtWidgets.QDialog.DialogCode.Accepted
            done(read(dialog) if ok else None)

        dialog.finished.connect(_finished)
        self._track(dialog)
        dialog.open()

    def ask_text(
        self, title: str, label: str, text: str, done: Callable[[str | None], None]
    ) -> None:
        def configure(d: QtWidgets.QInputDialog) -> None:
            d.setInputMode(QtWidgets.QInputDialog.InputMode.TextInput)
            d.setTextValue(text)

        self._input(title, label, configure, lambda d: d.textValue(), done)

    def ask_float(
        self,
        title: str,
        label: str,
        value: float,
        done: Callable[[float | None], None],
    ) -> None:
        def configure(d: QtWidgets.QInputDialog) -> None:
            d.setInputMode(QtWidgets.QInputDialog.InputMode.DoubleInput)
            d.setDoubleDecimals(8)
            d.setDoubleRange(0.0, 1e9)
            d.setDoubleValue(value)

        self._input(title, label, configure, lambda d: float(d.doubleValue()), done)

    def inform(self, title: str, text: str, severity: Severity = "info") -> None:
        icon = {
            "info": QtWidgets.QMessageBox.Icon.Information,
            "warning": QtWidgets.QMessageBox.Icon.Warning,
            "error": QtWidgets.QMessageBox.Icon.Critical,
        }[severity]
        box = QtWidgets.QMessageBox(icon, title, text, parent=self._parent)
        box.setObjectName("dialog_inform")
        self._track(box)
        box.open()

    def confirm(
        self, title: str, text: str, accept: str, done: Callable[[bool], None]
    ) -> None:
        box = QtWidgets.QMessageBox(self._parent)
        box.setObjectName("dialog_confirm")
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setWindowTitle(title)
        # The question quotes component labels and numbers, never markup.
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(text)
        yes = box.addButton(accept, QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        no = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(yes)
        box.setEscapeButton(no)
        answered: list[bool] = []

        def _clicked(button: QtWidgets.QAbstractButton) -> None:
            if not answered:
                answered.append(True)
                done(button is yes)

        def _finished(_: int) -> None:
            if not answered:  # closed by the title bar
                answered.append(True)
                done(False)

        box.buttonClicked.connect(_clicked)
        box.finished.connect(_finished)
        self._track(box)
        box.open()

    def show_html(self, title: str, html: str) -> None:
        dialog = QtWidgets.QDialog(self._parent)
        dialog.setObjectName("dialog_sheet")
        dialog.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(dialog)
        view = QtWidgets.QTextBrowser(dialog)
        view.setAccessibleName(title)
        view.setHtml(html)
        layout.addWidget(view)
        close = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close, dialog
        )
        close.rejected.connect(dialog.reject)
        layout.addWidget(close)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.resize(520, 600)
        self._track(dialog)
        dialog.open()

    def import_file(
        self,
        path: str,
        preview: FilePreview,
        done: Callable[[ImportChoice | None], None],
        *,
        last: ImportChoice | None = None,
    ) -> None:
        from pathlib import Path

        from cuperiod.gui.widgets.preview_dialog import PreviewDialog

        dialog = PreviewDialog(
            Path(path).name,
            preview,
            self._parent,
            path=path,
            columns=None if last is None else last.columns,
            domain=None if last is None else last.domain,
            read=None if last is None else last.read,
        )
        dialog.setObjectName("dialog_import")
        answered: list[bool] = []

        def _chosen(choice: ImportChoice) -> None:
            if not answered:
                answered.append(True)
                done(choice)

        def _finished(_: int) -> None:
            if not answered:  # Cancel, Esc or the title bar
                answered.append(True)
                done(None)

        dialog.choice_made.connect(_chosen)
        dialog.finished.connect(_finished)
        self._track(dialog)
        dialog.open()


class FakeDialogProvider:
    """A scripted :class:`DialogProvider` for tests (answers synchronously).

    Parameters
    ----------
    answers : iterable, optional
        Replies consumed in order by the prompts (``confirm_discard``, ``open_file``,
        ``save_file``, ``ask_text``, ``ask_float``, ``confirm`` — where only
        ``True`` accepts). When exhausted, each prompt gives its "cancel" answer
        (``"cancel"`` / ``None`` / ``False``).

    Attributes
    ----------
    calls : list of (str, tuple)
        Every call, with its arguments (callbacks excluded).
    """

    def __init__(self, answers: Iterable[Any] = ()) -> None:
        self.answers: deque[Any] = deque(answers)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def _next(self, default: Any) -> Any:
        return self.answers.popleft() if self.answers else default

    def confirm_discard(
        self, names: Sequence[str], done: Callable[[DiscardChoice], None]
    ) -> None:
        self.calls.append(("confirm_discard", (tuple(names),)))
        done(self._next("cancel"))

    def open_file(
        self,
        title: str,
        filters: str,
        done: Callable[[str | None], None],
        *,
        directory: str = "",
    ) -> None:
        self.calls.append(("open_file", (title, filters)))
        done(self._next(None))

    def open_folder(
        self, title: str, done: Callable[[str | None], None], *, directory: str = ""
    ) -> None:
        self.calls.append(("open_folder", (title,)))
        done(self._next(None))

    def save_file(
        self,
        title: str,
        filters: str,
        suggested: str,
        done: Callable[[str | None], None],
    ) -> None:
        self.calls.append(("save_file", (title, filters, suggested)))
        done(self._next(None))

    def ask_text(
        self, title: str, label: str, text: str, done: Callable[[str | None], None]
    ) -> None:
        self.calls.append(("ask_text", (title, label, text)))
        done(self._next(None))

    def ask_float(
        self,
        title: str,
        label: str,
        value: float,
        done: Callable[[float | None], None],
    ) -> None:
        self.calls.append(("ask_float", (title, label, value)))
        done(self._next(None))

    def inform(self, title: str, text: str, severity: Severity = "info") -> None:
        self.calls.append(("inform", (title, text, severity)))

    def confirm(
        self, title: str, text: str, accept: str, done: Callable[[bool], None]
    ) -> None:
        self.calls.append(("confirm", (title, text, accept)))
        done(self._next(False) is True)

    def show_html(self, title: str, html: str) -> None:
        self.calls.append(("show_html", (title, html)))

    def import_file(
        self,
        path: str,
        preview: FilePreview,
        done: Callable[[ImportChoice | None], None],
        *,
        last: ImportChoice | None = None,
    ) -> None:
        self.calls.append(("import_file", (path, last)))
        done(self._next(None))


__all__ = [
    "DialogProvider",
    "DiscardChoice",
    "FakeDialogProvider",
    "QtDialogProvider",
    "Severity",
]
