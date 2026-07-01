"""Single import surface for Qt (PySide6) and pyqtgraph.

Every GUI module imports its Qt classes and ``pg`` from here, so the binding choice and
the one-time pyqtgraph render config live in one place. Call sites use the module
namespaces (``QtWidgets.QPushButton``, ``QtGui.QAction``) plus the flat names below
(``Qt``, ``Signal``, ``Slot``, ``QObject``).

Importing this module configures pyqtgraph globally: antialiasing is **off** because the
spectrum can carry 10^5-10^6 points and per-sample AA destroys pan/zoom latency; the
phased and raw scatter plots opt back in per item. OpenGL is left off for cross-platform
reliability.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QObject, Qt, Signal, Slot

# Global render defaults. Per-item antialiasing is re-enabled on the low-count scatters.
pg.setConfigOptions(antialias=False, useOpenGL=False)

__all__ = [
    "QObject",
    "Qt",
    "QtCore",
    "QtGui",
    "QtWidgets",
    "Signal",
    "Slot",
    "pg",
]
