"""Application bootstrap: build the ``QApplication`` and show the main window."""

from __future__ import annotations

import sys

from cuperiod.gui.qt import QtWidgets
from cuperiod.gui.theme import ThemeName, app_icon

#: Theme used on first launch (overridden by a persisted choice once one exists).
DEFAULT_THEME: ThemeName = "dark"


def run_app(argv: list[str] | None = None) -> int:
    """Create the application, show the main window, and run the Qt event loop.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to :data:`sys.argv`.

    Returns
    -------
    int
        The Qt exit code.
    """
    from cuperiod.gui.main_window import MainWindow

    args = list(sys.argv if argv is None else argv)
    existing = QtWidgets.QApplication.instance()
    if existing is None:
        app = QtWidgets.QApplication(args)
    else:
        assert isinstance(existing, QtWidgets.QApplication)
        app = existing
    app.setApplicationName("cuPeriod")
    app.setApplicationDisplayName("cuPeriod")
    app.setOrganizationName("cuPeriod")
    app.setWindowIcon(app_icon())

    window = MainWindow(app, theme=DEFAULT_THEME)
    window.show()
    return app.exec()


__all__ = ["DEFAULT_THEME", "run_app"]
