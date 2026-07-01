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
    rather than a raw ImportError traceback.
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
    raise SystemExit(run_app())


__all__ = ["main"]
