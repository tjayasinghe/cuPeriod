"""Tests for the file preview dialog: literal file text and an actionable dead end."""

from __future__ import annotations

from pytestqt.qtbot import QtBot

from cuperiod.gui.loaders import FilePreview
from cuperiod.gui.qt import Qt, QtGui, QtWidgets
from cuperiod.gui.widgets.preview_dialog import PreviewDialog


def _preview(
    mapping: dict[str, str],
    columns: list[str],
    *,
    note: str = "",
    loadable: bool = True,
) -> FilePreview:
    return FilePreview(
        columns=columns,
        rows=[["1.0", "2.0"], ["3.0", "4.0"]],
        n_rows=2,
        mapping=mapping,
        domain="magnitude" if mapping else "?",
        multiband=False,
        note=note,
        loadable=loadable,
    )


# -- SEC-03: everything shown comes from the file, so nothing is markup -------------


def test_heading_and_summary_show_markup_literally(qtbot: QtBot) -> None:
    columns = ["<img src=x>", "<h1>HUGE</h1>"]
    mapping = {"time": columns[0], "value": columns[1]}
    dialog = PreviewDialog("<b>star</b>.csv", _preview(mapping, columns))
    qtbot.addWidget(dialog)
    assert dialog._heading.textFormat() == Qt.TextFormat.PlainText
    assert dialog._heading.text() == "<b>star</b>.csv"
    assert not dialog._heading.findChildren(QtGui.QTextDocument)
    summary = dialog._summary
    assert summary.textFormat() == Qt.TextFormat.PlainText
    assert "time = <img src=x>" in summary.text()
    # Left to guess (AutoText), Qt would have rendered this one as HTML.
    assert QtGui.Qt.mightBeRichText(summary.text().splitlines()[-1])


# -- A11Y-14 (a): undetected columns say what to do, and Enter does not load ---------


def test_undetected_columns_explain_the_way_out(qtbot: QtBot) -> None:
    dialog = PreviewDialog("star.dat", _preview({}, ["col1", "col2"]))
    qtbot.addWidget(dialog)
    text = dialog._summary.text()
    assert "could not tell which columns" in text
    assert "mag" in text and "time" in text  # names the headers that are recognised
    assert "ColumnMap(time=" not in text  # no Python-only instruction on its own
    buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
    cancel = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    assert cancel.isDefault()
    assert not dialog._load_button.isDefault()


def test_detected_columns_keep_load_as_the_default(qtbot: QtBot) -> None:
    dialog = PreviewDialog(
        "star.csv", _preview({"time": "time", "value": "mag"}, ["time", "mag"])
    )
    qtbot.addWidget(dialog)
    assert dialog._load_button.isDefault()
    assert "Mapping" in dialog._summary.text()


# -- GUI-B-16 / UX-21 (requested by B2-GUI-CORE): the reader's note and Load --------


def test_an_unloadable_file_shows_why_and_offers_no_load(qtbot: QtBot) -> None:
    note = "one row per object (<b>wide</b> table): not a light curve"
    dialog = PreviewDialog(
        "objects.parquet", _preview({}, ["id", "ra"], note=note, loadable=False)
    )
    qtbot.addWidget(dialog)
    assert not dialog._load_button.isEnabled()
    assert not dialog._note.isHidden() and dialog._note.text() == note
    assert dialog._note.textFormat() == Qt.TextFormat.PlainText


def test_an_undetected_but_loadable_file_keeps_load_available(qtbot: QtBot) -> None:
    dialog = PreviewDialog("star.dat", _preview({}, ["col1", "col2"]))
    qtbot.addWidget(dialog)
    assert dialog._load_button.isEnabled()
    assert dialog._note.isHidden()  # nothing to say
