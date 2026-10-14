"""DialogProvider.import_file: the import dialog behind the provider, never exec()."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pytestqt.qtbot import QtBot

from cuperiod.gui.dialogs import DialogProvider, FakeDialogProvider, QtDialogProvider
from cuperiod.gui.loaders import ImportChoice, preview_file
from cuperiod.gui.qt import QtWidgets


def _csv(tmp_path: Path) -> Path:
    path = tmp_path / "star.csv"
    t = np.linspace(0.0, 10.0, 30)
    rows = ["time,mag,mag_err"] + [
        f"{x:.5f},{12 + 0.1 * np.sin(x):.5f},0.01" for x in t
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_fake_provider_answers_from_its_script(tmp_path: Path) -> None:
    path = _csv(tmp_path)
    preview = preview_file(path)
    assert preview is not None
    fake = FakeDialogProvider(["chosen"])
    assert isinstance(fake, DialogProvider)
    got: list[object] = []
    fake.import_file(str(path), preview, got.append)
    fake.import_file(str(path), preview, got.append)
    assert got == ["chosen", None]
    assert fake.calls[0] == ("import_file", (str(path), None))


def test_qt_provider_opens_the_import_dialog_without_blocking(
    qtbot: QtBot, tmp_path: Path
) -> None:
    path = _csv(tmp_path)
    preview = preview_file(path)
    assert preview is not None
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    provider = QtDialogProvider(parent, native_files=False)
    got: list[object] = []
    provider.import_file(str(path), preview, got.append)
    (dialog,) = provider.live_dialogs
    assert dialog.objectName() == "dialog_import" and dialog.isVisible()
    assert got == []  # open() returned at once
    dialog.accept()  # Load
    assert len(got) == 1 and isinstance(got[0], ImportChoice)
    provider.import_file(str(path), preview, got.append)
    (dialog,) = provider.live_dialogs
    dialog.reject()  # Cancel
    assert got[-1] is None and len(got) == 2
