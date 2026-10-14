"""Application bootstrap: build the ``QApplication`` and show the main window."""

from __future__ import annotations

import contextlib
import os
import sys

from cuperiod.gui.compute import compute_active
from cuperiod.gui.qt import QtCore, QtWidgets
from cuperiod.gui.theme import ThemeName, app_icon

#: Theme used on first launch (overridden by a persisted choice once one exists).
DEFAULT_THEME: ThemeName = "dark"


def _finish(code: int, *, hard_exit: bool) -> int:
    """Return the event loop's exit code, or end the process if a worker still runs.

    A periodogram or pre-whitening run cannot be interrupted once started, and its
    worker thread keeps the interpreter alive after the window has closed: the
    process would linger, windowless and still busy, until the computation ends.
    With ``hard_exit`` (the console entry point only), a still-running worker ends
    the process instead, after flushing the output streams and persisted settings
    (``os._exit`` skips atexit handlers and Qt's own teardown). Embedded use, such as
    IPython or a test session, must never be killed, so ``run_app`` defaults to off.
    """
    if hard_exit and compute_active():
        QtCore.QSettings().sync()
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(AttributeError, OSError, ValueError):
                stream.flush()  # a stream may be closed or missing (pythonw)
        os._exit(code)
    return code


def name_application(app: QtWidgets.QApplication) -> None:
    """Name the application (its settings, its dialogs) and give it its icon.

    No *display* name is set: Qt appends an explicitly set one to every window
    title (" - cuPeriod" on Windows, " — cuPeriod" on Linux), and the main window's
    title already ends with the application's name, so it read
    "cuPeriod — star.csv - cuPeriod". Unset, the display name (which dialogs show)
    is the application name all the same.
    """
    app.setApplicationName("cuPeriod")
    app.setOrganizationName("cuPeriod")
    app.setWindowIcon(app_icon())


def run_app(argv: list[str] | None = None, *, exit_when_closed: bool = False) -> int:
    """Create the application, show the main window, and run the Qt event loop.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to :data:`sys.argv`.
    exit_when_closed : bool, default False
        End the process as soon as the window closes, even while a computation is
        still running on a worker (it cannot be interrupted). Only the console entry
        point sets this; an embedding application keeps its process.

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
    name_application(app)

    window = MainWindow(app, theme=DEFAULT_THEME)
    window.show()
    return _finish(app.exec(), hard_exit=exit_when_closed)


__all__ = ["DEFAULT_THEME", "name_application", "run_app"]
