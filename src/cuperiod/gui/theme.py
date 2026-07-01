"""Theme palettes, Qt stylesheet loading, and application.

Two cohesive themes ship with the app: a deep-slate ``"dark"`` (the default — ideal
contrast for spectra and folded curves) and a clean ``"light"`` one. A
:class:`ThemePalette` carries the colours the pyqtgraph views need (backgrounds, the
accent line, the best-peak glow, the grid, and a colourblind-safe band cycle) so each
view can re-pen itself on a live theme switch without rebuilding. The widget chrome is
styled by a Qt stylesheet (``resources/<theme>.qss``) layered on the Fusion style + a
matching ``QPalette`` so native popups and tooltips match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal

from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, pg

#: Plot font point sizes (ticks / axis labels) — larger for readability.
_TICK_PT = 10
_LABEL_PT = 12

ThemeName = Literal["dark", "light"]

#: Okabe-Ito colourblind-safe qualitative palette, used to colour photometric bands.
_BAND_CYCLE: tuple[str, ...] = (
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#999999",  # grey
)


@dataclass(frozen=True)
class ThemePalette:
    """Colours a pyqtgraph view needs to render and re-skin itself.

    All fields are hex strings; :meth:`qcolor` converts. ``plot_bg``/``plot_fg`` are the
    background and axis colours; ``accent`` the selected-period line; ``best_peak`` the
    rank-1 glow; ``peak`` other markers; ``grid`` the grid; ``muted`` secondary text.
    """

    name: ThemeName
    plot_bg: str
    plot_fg: str
    accent: str
    best_peak: str
    peak: str
    grid: str
    muted: str
    curve: str
    band_cycle: tuple[str, ...] = field(default=_BAND_CYCLE)

    def qcolor(self, hex_str: str, alpha: int = 255) -> QtGui.QColor:
        """A :class:`~PySide6.QtGui.QColor` for ``hex_str`` with optional ``alpha``."""
        color = QtGui.QColor(hex_str)
        color.setAlpha(alpha)
        return color

    def band_color(self, index: int, alpha: int = 255) -> QtGui.QColor:
        """Stable colour for band ``index`` from the colourblind-safe cycle."""
        return self.qcolor(self.band_cycle[index % len(self.band_cycle)], alpha)


_PALETTES: dict[ThemeName, ThemePalette] = {
    "dark": ThemePalette(
        name="dark",
        plot_bg="#0b0f14",
        plot_fg="#c9d3df",
        accent="#5eead4",  # teal — selected-period line / primary
        best_peak="#fbbf24",  # amber — rank-1 peak glow
        peak="#7dd3fc",  # light blue — other peaks
        grid="#7689a3",
        muted="#8b98ab",
        curve="#9fb4cc",
    ),
    "light": ThemePalette(
        name="light",
        plot_bg="#ffffff",
        plot_fg="#1f2328",
        accent="#0d9488",  # teal (darker for light bg)
        best_peak="#b45309",  # amber (darker)
        peak="#0369a1",
        grid="#9aa7b8",
        muted="#57606a",
        curve="#334155",
    ),
}


def palette(theme: ThemeName) -> ThemePalette:
    """Return the :class:`ThemePalette` for ``theme``."""
    return _PALETTES[theme]


@lru_cache(maxsize=2)
def load_qss(theme: ThemeName) -> str:
    """Read the ``resources/<theme>.qss`` stylesheet text (cached)."""
    return (
        files("cuperiod.gui.resources")
        .joinpath(f"{theme}.qss")
        .read_text(encoding="utf-8")
    )


def _qpalette(pal: ThemePalette) -> QtGui.QPalette:
    """Build a Fusion ``QPalette`` matching ``pal`` for native chrome (popups)."""
    Role = QtGui.QPalette.ColorRole
    qp = QtGui.QPalette()
    if pal.name == "dark":
        window, base, text, button = "#11161d", "#0d1117", "#e6edf3", "#1b222c"
        disabled = "#5b6776"
    else:
        window, base, text, button = "#eef1f5", "#ffffff", "#1f2328", "#e6eaf0"
        disabled = "#a8b0bb"
    qp.setColor(Role.Window, QtGui.QColor(window))
    qp.setColor(Role.Base, QtGui.QColor(base))
    qp.setColor(Role.AlternateBase, QtGui.QColor(window))
    qp.setColor(Role.Text, QtGui.QColor(text))
    qp.setColor(Role.WindowText, QtGui.QColor(text))
    qp.setColor(Role.Button, QtGui.QColor(button))
    qp.setColor(Role.ButtonText, QtGui.QColor(text))
    qp.setColor(Role.ToolTipBase, QtGui.QColor(base))
    qp.setColor(Role.ToolTipText, QtGui.QColor(text))
    highlight_text = "#06231f" if pal.name == "dark" else "#ffffff"
    qp.setColor(Role.Highlight, QtGui.QColor(pal.accent))
    qp.setColor(Role.HighlightedText, QtGui.QColor(highlight_text))
    qp.setColor(Role.PlaceholderText, QtGui.QColor(pal.muted))
    group = QtGui.QPalette.ColorGroup.Disabled
    qp.setColor(group, Role.Text, QtGui.QColor(disabled))
    qp.setColor(group, Role.ButtonText, QtGui.QColor(disabled))
    qp.setColor(group, Role.WindowText, QtGui.QColor(disabled))
    return qp


def plot_label_style(pal: ThemePalette) -> dict[str, str]:
    """CSS kwargs for a pyqtgraph axis label (colour + larger font)."""
    return {"color": pal.plot_fg, "font-size": f"{_LABEL_PT}pt"}


def apply_plot_theme(plot: Any, pal: ThemePalette) -> None:
    """Style a pyqtgraph PlotWidget: background, axis colours, and readable fonts.

    Sets the tick text/line colour to the palette foreground and enlarges the tick and
    axis-label fonts, then re-applies each existing axis label in the new style. Call at
    creation and on every theme change so light-theme axes stay legible.
    """
    plot.setBackground(pal.plot_bg)
    foreground = pal.qcolor(pal.plot_fg)
    tick_font = QtGui.QFont()
    tick_font.setPointSize(_TICK_PT)
    plot_item = plot.getPlotItem()
    for side in ("bottom", "left"):
        axis = plot_item.getAxis(side)
        axis.setTextPen(foreground)
        axis.setPen(foreground)
        axis.setStyle(tickFont=tick_font)
        text = axis.labelText
        if text:
            axis.setLabel(text, **plot_label_style(pal))


def app_icon() -> QtGui.QIcon:
    """A generated cuPeriod icon (a teal spectrum with a glowing amber peak)."""
    pixmap = QtGui.QPixmap(64, 64)
    pixmap.fill(QtGui.QColor(0, 0, 0, 0))
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor("#0b0f14"))
    painter.drawRoundedRect(0, 0, 64, 64, 14, 14)

    pen = QtGui.QPen(QtGui.QColor("#5eead4"))
    pen.setWidthF(2.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    points = [
        (8, 46), (14, 44), (18, 40), (22, 45), (26, 39), (30, 47),
        (34, 20), (38, 47), (42, 41), (46, 44), (50, 39), (56, 45),
    ]
    path = QtGui.QPainterPath()
    path.moveTo(float(points[0][0]), float(points[0][1]))
    for x, y in points[1:]:
        path.lineTo(float(x), float(y))
    painter.drawPath(path)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(251, 191, 36, 60))
    painter.drawEllipse(QtCore.QPointF(34.0, 16.0), 10.0, 10.0)
    painter.setBrush(QtGui.QColor("#fbbf24"))
    painter.drawEllipse(QtCore.QPointF(34.0, 16.0), 5.5, 5.5)
    painter.end()
    return QtGui.QIcon(pixmap)


def apply_theme(app: QtWidgets.QApplication, theme: ThemeName) -> ThemePalette:
    """Apply ``theme`` to ``app``: Fusion style, palette, stylesheet, pyqtgraph globals.

    Returns the active :class:`ThemePalette` so the caller can hand it to views for
    re-skinning.
    """
    pal = palette(theme)
    app.setStyle("Fusion")
    app.setPalette(_qpalette(pal))
    app.setStyleSheet(load_qss(theme))
    pg.setConfigOption("background", pal.plot_bg)
    pg.setConfigOption("foreground", pal.plot_fg)
    return pal


__all__ = [
    "ThemeName",
    "ThemePalette",
    "app_icon",
    "apply_plot_theme",
    "apply_theme",
    "load_qss",
    "palette",
    "plot_label_style",
]
