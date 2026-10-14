"""Tests for the batch source browser: filter reset, failed rows, literal tooltips."""

from __future__ import annotations

from pytestqt.qtbot import QtBot

from cuperiod.gui.qt import Qt, QtGui, QtWidgets
from cuperiod.gui.widgets.source_panel import SourceBrowser, rich_tooltip


def _browser(qtbot: QtBot, labels: list[str]) -> SourceBrowser:
    browser = SourceBrowser()
    qtbot.addWidget(browser)
    browser.set_sources(labels)
    return browser


def _hidden(browser: SourceBrowser) -> list[bool]:
    return [browser._list.item(r).isHidden() for r in range(browser._list.count())]


# -- GUI-B-14 ----------------------------------------------------------------------


def test_a_new_batch_clears_the_previous_filter(qtbot: QtBot) -> None:
    browser = _browser(qtbot, ["alpha", "beta", "gamma"])
    browser._filter.setText("alp")
    assert browser._count_label.text() == "1 of 3 sources"
    browser.set_sources(["delta", "epsilon", "zeta", "eta"])
    assert browser._filter.text() == ""
    assert _hidden(browser) == [False, False, False, False]
    assert browser._count_label.text() == "4 sources"
    # ...and the filter still works on the new list.
    browser._filter.setText("ta")
    assert browser._count_label.text() == "3 of 4 sources"


# -- GUI-B-5 -----------------------------------------------------------------------


def test_a_failed_row_is_greyed_and_skipped_by_prev_next(qtbot: QtBot) -> None:
    browser = _browser(qtbot, ["a_star", "notes", "z_star"])
    selected: list[int] = []
    browser.source_selected.connect(selected.append)
    browser.mark_failed(1, "ColumnResolutionError: no time column")
    item = browser._list.item(1)
    assert not item.flags() & Qt.ItemFlag.ItemIsEnabled
    assert "no time column" in item.toolTip()
    assert browser.is_failed(1)
    assert "1 unreadable" in browser._count_label.text()

    browser._next_btn.click()
    assert selected == [2]  # straight past the unreadable row
    browser._prev_btn.click()
    assert selected == [2, 0]


def test_arrow_keys_skip_a_failed_row(qtbot: QtBot) -> None:
    browser = _browser(qtbot, ["a_star", "notes", "z_star"])
    browser.show()
    browser.mark_failed(1, "unreadable")
    selected: list[int] = []
    browser.source_selected.connect(selected.append)
    browser._list.setFocus()
    qtbot.keyClick(browser._list, Qt.Key.Key_Down)
    assert selected == [2]


def test_a_new_batch_forgets_failed_rows(qtbot: QtBot) -> None:
    browser = _browser(qtbot, ["a", "b"])
    browser.mark_failed(1, "unreadable")
    browser.set_sources(["c", "d"])
    assert not browser.is_failed(1)
    assert browser._list.item(1).flags() & Qt.ItemFlag.ItemIsEnabled
    assert browser._count_label.text() == "2 sources"


def test_prev_next_are_disabled_without_sources(qtbot: QtBot) -> None:
    browser = SourceBrowser()
    qtbot.addWidget(browser)
    assert not browser._next_btn.isEnabled() and not browser._prev_btn.isEnabled()
    browser.set_sources(["a"])
    assert browser._next_btn.isEnabled()


# -- SEC-03: file-derived labels are never rendered as markup -------------------------


def test_source_tooltips_show_markup_literally(qtbot: QtBot) -> None:
    browser = _browser(qtbot, ["<b>x</b>", "B&V", "<img src=missing.png>"])
    for row, label in enumerate(["<b>x</b>", "B&V", "<img src=missing.png>"]):
        tip = browser._list.item(row).toolTip()
        assert QtGui.Qt.mightBeRichText(tip)  # so Qt does not guess on its own...
        doc = QtGui.QTextDocument()
        doc.setHtml(tip)
        assert doc.toPlainText() == f"{row + 1}. {label}"  # ...and shows it as typed


def test_rich_tooltip_keeps_line_breaks() -> None:
    doc = QtGui.QTextDocument()
    doc.setHtml(rich_tooltip("one <two>\nthree"))
    assert doc.toPlainText() == "one <two>\nthree"


# -- UX-23: long labels elide in the middle, keeping their useful tail ----------------


def test_long_labels_elide_in_the_middle(qtbot: QtBot) -> None:
    browser = _browser(qtbot, ["8590948612 — RR Lyrae (P=0.567 d)"])
    assert browser._list.textElideMode() == Qt.TextElideMode.ElideMiddle
    assert (
        browser._list.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert isinstance(browser._list, QtWidgets.QListWidget)
