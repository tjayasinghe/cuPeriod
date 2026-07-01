"""A dialog previewing a file's rows and detected column mapping before loading."""

from __future__ import annotations

from cuperiod.gui.loaders import FilePreview
from cuperiod.gui.qt import QtWidgets


def _summary_text(preview: FilePreview) -> str:
    if not preview.mapping:
        return f"{preview.n_rows} rows — could not auto-detect time/value columns."
    parts = "    ".join(f"{role} = {col}" for role, col in preview.mapping.items())
    kind = "multiband" if preview.multiband else "single-band"
    return f"{preview.n_rows} rows · {preview.domain} · {kind}\nMapping:  {parts}"


class PreviewDialog(QtWidgets.QDialog):
    """Show the first rows and column mapping; accept to load, reject to cancel."""

    def __init__(
        self,
        filename: str,
        preview: FilePreview,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Load {filename}")
        self.resize(780, 500)

        layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel(filename)
        heading.setObjectName("heading")
        layout.addWidget(heading)
        summary = QtWidgets.QLabel(_summary_text(preview))
        summary.setObjectName("muted")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        table = QtWidgets.QTableWidget(len(preview.rows), len(preview.columns))
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        vheader = table.verticalHeader()
        if vheader is not None:
            vheader.setVisible(False)
        role_of = {col: role for role, col in preview.mapping.items()}
        table.setHorizontalHeaderLabels(
            [f"{c}\n({role_of[c]})" if c in role_of else c for c in preview.columns]
        )
        for r, row in enumerate(preview.rows):
            for c, text in enumerate(row):
                table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        hheader = table.horizontalHeader()
        if hheader is not None:
            hheader.setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        layout.addWidget(table, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        load_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if load_button is not None:
            load_button.setText("Load")
            load_button.setProperty("primary", True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


__all__ = ["PreviewDialog"]
