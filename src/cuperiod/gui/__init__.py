"""cuPeriod desktop GUI — the ``cuperiod-gui`` interactive periodogram explorer.

A PySide6 + pyqtgraph presentation layer over the cuPeriod API: load a light curve (or a
folder of them), run any method with all of its options, and explore the full-resolution
spectrum with a live phased light curve. Optional — ``pip install 'cuperiod[gui]'``.
"""

from __future__ import annotations


def main() -> None:
    """Console-script entry point for ``cuperiod-gui``.

    Sets the Windows OpenMP duplicate-load workaround (so a torch backend can import
    after numpy/MKL — mirrors the test/CLI stance) *before* importing the GUI, then
    launches it. If the ``[gui]`` extra is missing, exits with an actionable message
    rather than a raw ImportError traceback. Closing the window ends the process even
    while a computation is still running (it cannot be interrupted, and would
    otherwise keep a windowless process busy until it finished).
    """
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        from cuperiod.gui.app import run_app
    except ImportError as exc:
        raise SystemExit(
            "cuperiod-gui requires the GUI extra. Install it with:\n"
            "    pip install 'cuperiod[gui]'\n"
            f"(could not import: {exc.name})"
        ) from exc
    problem = _known_broken_qt()
    if problem:
        raise SystemExit(problem)
    raise SystemExit(run_app(exit_when_closed=True))


def _known_broken_qt() -> str:
    """Why the installed PySide6 cannot run the GUI here, or ``""``.

    PySide6 6.12 drops a reference to ``None`` whenever a Qt call returns a null
    pointer. ``None`` is immortal from Python 3.12, but on 3.11 its count reaches zero
    within seconds of normal use and the interpreter aborts ("deallocating None"),
    taking unsaved work with it. The ``[gui]`` extra keeps 3.11 below 6.12; this
    catches an environment that already had it.
    """
    import sys

    python = sys.version_info[:2]
    if python >= (3, 12):
        return ""
    import PySide6

    try:
        major, minor = (int(part) for part in PySide6.__version__.split(".")[:2])
    except ValueError:
        return ""
    if (major, minor) < (6, 12):
        return ""
    return (
        f"PySide6 {PySide6.__version__} crashes on Python "
        f"{python[0]}.{python[1]} (a reference-counting "
        "bug that aborts the process). Install an earlier release:\n"
        "    pip install 'PySide6<6.12'\n"
        "or run cuperiod-gui on Python 3.12 or newer."
    )


__all__ = ["main"]
