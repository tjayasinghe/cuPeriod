"""The batch source browser: scroll through many light curves.

A filterable list of sources with prev/next navigation. Selecting one (click, arrow
keys, or the buttons) emits :attr:`source_selected` with its index; the controller then
loads that curve and computes (or cache-hits) its periodogram on demand.
"""

from __future__ import annotations

from cuperiod.gui.qt import Qt, QtWidgets, Signal


class SourceBrowser(QtWidgets.QWidget):
    """A filterable, navigable list of batch sources."""

    source_selected = Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._count_label = QtWidgets.QLabel("No sources")
        self._count_label.setObjectName("muted")
        layout.addWidget(self._count_label)

        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText("filter…")
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        self._list = QtWidgets.QListWidget()
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list, 1)

        nav = QtWidgets.QHBoxLayout()
        self._prev_btn = QtWidgets.QPushButton("◀ Prev")
        self._next_btn = QtWidgets.QPushButton("Next ▶")
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        self._next_btn.clicked.connect(lambda: self._step(1))
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

    def set_sources(self, labels: list[str]) -> None:
        """Populate the list from ``labels`` and select the first (without emitting)."""
        self._list.blockSignals(True)
        self._list.clear()
        for i, label in enumerate(labels):
            item = QtWidgets.QListWidgetItem(f"{i + 1}. {label}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._list.addItem(item)
        if labels:
            self._list.setCurrentRow(0)
        self._list.blockSignals(False)
        self._count_label.setText(f"{len(labels)} sources")

    def set_current(self, index: int) -> None:
        """Highlight ``index`` without re-emitting (controller-driven sync)."""
        self._list.blockSignals(True)
        self._list.setCurrentRow(index)
        self._list.blockSignals(False)

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

    def _step(self, delta: int) -> None:
        count = self._list.count()
        if count == 0:
            return
        row = self._list.currentRow()
        for _ in range(count):
            row = (row + delta) % count
            item = self._list.item(row)
            if item is not None and not item.isHidden():
                self._list.setCurrentRow(row)
                return


__all__ = ["SourceBrowser"]
