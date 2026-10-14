"""The batch source browser: scroll through many light curves.

A filterable list of sources with prev/next navigation. Selecting one (click, arrow
keys, or the buttons) emits :attr:`source_selected` with its index; the controller then
loads that curve and computes (or cache-hits) its periodogram on demand.

A source that fails to load is greyed out with the reason as its tooltip and is
skipped by Prev/Next and the arrow keys. Without that, a stray ``notes.txt`` in a
folder of light curves is a wall: every step lands on it again, the load fails, and
the selection snaps back to where it started.
"""

from __future__ import annotations

import html

from cuperiod.gui.qt import Qt, QtWidgets, Signal


def rich_tooltip(text: str) -> str:
    """``text`` as a tooltip that shows exactly what it says.

    Qt guesses whether a tooltip is rich text, so a label taken from a file name or a
    catalogue cell that happens to contain ``<b>`` or ``<img src=...>`` would be
    rendered (and an image fetched). Escaping alone is not enough — escaped text is
    not detected as rich, so ``B&V`` would show as ``B&amp;V`` — hence the wrapper tag.
    """
    escaped = html.escape(text).replace("\n", "<br>")
    return f"<span>{escaped}</span>"


class SourceBrowser(QtWidgets.QWidget):
    """A filterable, navigable list of batch sources."""

    source_selected = Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._total = 0
        self._failed: set[int] = set()
        self._count_label = QtWidgets.QLabel("No sources")
        self._count_label.setObjectName("muted")
        layout.addWidget(self._count_label)

        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText("filter…")
        self._filter.setAccessibleName("Filter sources")
        self._filter.setToolTip("Show only the sources whose label contains this text")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        self._list = QtWidgets.QListWidget()
        self._list.setAccessibleName("Sources")
        # Labels end in the useful part (variable type, catalogue period), so a
        # narrow dock elides the middle rather than the end; the full label stays in
        # the tooltip. Without a horizontal scrollbar the rows elide instead of
        # scrolling sideways.
        self._list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list, 1)

        nav = QtWidgets.QHBoxLayout()
        self._prev_btn = QtWidgets.QPushButton("◀ Prev")
        self._next_btn = QtWidgets.QPushButton("Next ▶")
        self._prev_btn.setAccessibleName("Previous source")
        self._next_btn.setAccessibleName("Next source")
        for button, tip in (
            (self._prev_btn, "Show the previous source of the batch"),
            (self._next_btn, "Show the next source of the batch"),
        ):
            button.setToolTip(tip)
            button.setStatusTip(tip)
        self._list.setStatusTip(
            "The batch's sources: pick one to show it (Up/Down move, type to filter)"
        )
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        self._next_btn.clicked.connect(lambda: self._step(1))
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)
        self._update_count()

    def set_sources(self, labels: list[str]) -> None:
        """Populate the list from ``labels`` and select the first (without emitting).

        A new batch is a new context, so the filter is cleared and nothing is marked
        as failed: keeping the old filter text while showing every row would make the
        box contradict the list (and a re-applied filter could hide the source that is
        about to be selected).
        """
        self._total = len(labels)
        self._failed.clear()
        self._filter.blockSignals(True)
        self._filter.clear()
        self._filter.blockSignals(False)
        self._list.blockSignals(True)
        self._list.clear()
        for i, label in enumerate(labels):
            text = f"{i + 1}. {label}"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setToolTip(rich_tooltip(text))  # full label, in case the row elides
            self._list.addItem(item)
        if labels:
            self._list.setCurrentRow(0)
        self._list.blockSignals(False)
        self._update_count()

    def set_current(self, index: int) -> None:
        """Highlight ``index`` without re-emitting (controller-driven sync)."""
        self._list.blockSignals(True)
        self._list.setCurrentRow(index)
        self._list.blockSignals(False)

    def mark_failed(self, index: int, message: str) -> None:
        """Grey out source ``index`` (it could not be loaded) and say why on hover.

        A disabled row is skipped by the arrow keys and by Prev/Next, so one unreadable
        file no longer blocks navigation past it. It stays marked until a new batch is
        loaded.
        """
        item = self._list.item(index)
        if item is None:
            return
        self._failed.add(index)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        item.setToolTip(rich_tooltip(f"{item.text()}\n{message}"))
        self._update_count()

    def clear_failed(self, index: int) -> None:
        """Make source ``index`` selectable again (its columns were chosen by hand)."""
        item = self._list.item(index)
        if item is None or index not in self._failed:
            return
        self._failed.discard(index)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
        item.setToolTip(rich_tooltip(item.text()))
        self._update_count()

    def is_failed(self, index: int) -> bool:
        """Whether source ``index`` has been marked as unloadable."""
        return index in self._failed

    def _on_row_changed(self, row: int) -> None:
        item = self._list.item(row)
        if item is None:
            return
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is not None:
            self.source_selected.emit(int(index))

    def _apply_filter(self, text: str) -> None:
        needle = text.lower()
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None:
                item.setHidden(needle not in item.text().lower())
        self._update_count()

    def _update_count(self) -> None:
        enabled = self._total > 0
        self._prev_btn.setEnabled(enabled)
        self._next_btn.setEnabled(enabled)
        if self._total == 0:
            self._count_label.setText("No sources")
            return
        shown = sum(
            1
            for row in range(self._list.count())
            if (item := self._list.item(row)) is not None and not item.isHidden()
        )
        if shown == self._total:
            text = f"{self._total} sources"
        else:
            text = f"{shown} of {self._total} sources"
        if self._failed:
            text += f" · {len(self._failed)} unreadable"
        self._count_label.setText(text)

    def _step(self, delta: int) -> None:
        count = self._list.count()
        if count == 0:
            return
        row = self._list.currentRow()
        for _ in range(count):
            row = (row + delta) % count
            item = self._list.item(row)
            if (
                item is not None
                and not item.isHidden()
                and item.flags() & Qt.ItemFlag.ItemIsEnabled
            ):
                self._list.setCurrentRow(row)
                return


__all__ = ["SourceBrowser", "rich_tooltip"]
