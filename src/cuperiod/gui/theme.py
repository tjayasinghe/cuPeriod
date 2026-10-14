"""Theme palettes, Qt stylesheet loading, UI zoom, and application.

Two cohesive themes ship with the app: a deep-slate ``"dark"`` (the default — ideal
contrast for spectra and folded curves) and a clean ``"light"`` one. A
:class:`ThemePalette` carries the colours the pyqtgraph views need (backgrounds, the
accent line, the best-peak glow, the grid, the residual trace, and a colourblind-safe
band cycle with a matching symbol cycle) so each view can re-pen itself on a live theme
switch without rebuilding. The widget chrome is styled by a Qt stylesheet
(``resources/<theme>.qss``) layered on the Fusion style + a matching ``QPalette`` so
native popups and tooltips match.

Accessibility contract (pinned by ``tests/gui/test_theme.py``): every text/background
pair is at least 4.5:1 and every control boundary, state cue and keyboard-focus ring at
least 3:1 (WCAG 2.1 SC 1.4.3, 1.4.11, 2.4.7), in both themes. Combo and spin boxes draw
arrows and a checked box draws a tick (theme-coloured SVGs next to the stylesheets), so
no state is told by fill colour alone.

The stylesheets are written in px at the designed size (13 px body text). Every length
in them, and the plot fonts, scale with the UI zoom (:func:`apply_theme`'s
``font_scale``) and with an application font larger than 13 px, so the whole interface
can be enlarged (WCAG 1.4.4) without the sheet overriding the user's text size.
"""

from __future__ import annotations

import atexit
import contextlib
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Literal

from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, pg

#: Plot font point sizes (ticks / axis labels) at zoom 1.0 — larger for readability.
_TICK_PT = 10
_LABEL_PT = 12

#: Body text size (px) the stylesheets are drawn at; zoom 1.0 renders the sheet as is.
BASE_FONT_PX = 13

#: Zoom levels a View > Zoom In / Zoom Out pair walks through (1.0 = designed size).
#: :func:`apply_theme` clamps ``font_scale`` to the first and last entries.
FONT_SCALE_STEPS: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0)

#: ``QDir`` search-path prefix the stylesheets use for their images
#: (``url(cuperiodtheme:chevron-down-dark.svg)``). Qt silently ignores a prefix with
#: anything but letters and digits in it, so no hyphen.
RESOURCE_PREFIX = "cuperiodtheme"

ThemeName = Literal["dark", "light"]

#: Okabe-Ito colourblind-safe qualitative palette, used to colour photometric bands on
#: the dark plot background. Its grey (#999999) is replaced by a light grey: under
#: deuteranopia the grey and the reddish purple were only dE00 6 apart.
_BAND_CYCLE: tuple[str, ...] = (
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#DDDDDD",  # light grey
)

#: Band cycle for the white plot background. Okabe-Ito's yellow, orange, sky blue and
#: grey are 1.3-2.9:1 on white (WCAG 1.4.11 asks 3:1 for data marks), and simply
#: darkening them collapses orange onto yellow under red-green deficiencies. This keeps
#: the Okabe-Ito colours that pass on white, adds three dark hues, and orders the eight
#: so the first six (Rubin ugrizy) stay >= dE00 11.8 apart under protanopia,
#: deuteranopia and tritanopia; band 0 stays a blue and band 1 a warm hue, as on dark.
_BAND_CYCLE_LIGHT: tuple[str, ...] = (
    "#389DD1",  # sky blue, darkened to 3:1
    "#D55E00",  # vermillion
    "#332288",  # indigo (Tol)
    "#767676",  # grey, darkened to 4.5:1
    "#0072B2",  # blue
    "#882255",  # wine (Tol)
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
)

#: pyqtgraph scatter symbols paired with the band colours, so bands stay apart for
#: anyone who cannot tell two colours apart (WCAG 1.4.1). Band 0 keeps the circle;
#: the filled shapes come first because they survive small marker sizes best.
_BAND_SYMBOLS: tuple[str, ...] = ("o", "s", "t1", "d", "t", "star", "x", "+")


@dataclass(frozen=True)
class ThemePalette:
    """Colours a pyqtgraph view needs to render and re-skin itself.

    All fields are hex strings; :meth:`qcolor` converts. ``plot_bg``/``plot_fg`` are the
    background and axis colours; ``accent`` the selected-period line; ``best_peak`` the
    rank-1 glow; ``peak`` other markers; ``grid`` the grid; ``muted`` secondary text;
    ``curve`` the spectrum; ``residual`` the pre-whitened residual spectrum drawn over
    it (a hue apart from ``curve`` under every colour-vision deficiency). ``band_cycle``
    / ``band_symbols`` give each photometric band a colour and a marker shape.
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
    residual: str
    band_cycle: tuple[str, ...] = field(default=_BAND_CYCLE)
    band_symbols: tuple[str, ...] = field(default=_BAND_SYMBOLS)

    def qcolor(self, hex_str: str, alpha: int = 255) -> QtGui.QColor:
        """A :class:`~PySide6.QtGui.QColor` for ``hex_str`` with optional ``alpha``."""
        color = QtGui.QColor(hex_str)
        color.setAlpha(alpha)
        return color

    def band_color(self, index: int, alpha: int = 255) -> QtGui.QColor:
        """Stable colour for band ``index`` from the colourblind-safe cycle."""
        return self.qcolor(self.band_cycle[index % len(self.band_cycle)], alpha)

    def band_symbol(self, index: int) -> str:
        """Stable pyqtgraph scatter symbol for band ``index`` (backs up its colour)."""
        return self.band_symbols[index % len(self.band_symbols)]


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
        residual="#f87171",  # coral red: dE00 >= 22 from the curve under any CVD
    ),
    "light": ThemePalette(
        name="light",
        plot_bg="#ffffff",
        plot_fg="#1f2328",
        # Deep teal: 6.4:1 solid, and still >= 3:1 at the 170 alpha of the selection
        # band edges (the previous #0d9488 was 3.7:1 solid, 2.3:1 there).
        accent="#0d6b63",
        best_peak="#b45309",  # amber (darker)
        peak="#0369a1",
        grid="#9aa7b8",
        muted="#57606a",
        curve="#334155",
        residual="#e11d48",  # rose: dE00 >= 22 from the curve under any CVD
        band_cycle=_BAND_CYCLE_LIGHT,
    ),
}


def palette(theme: ThemeName) -> ThemePalette:
    """Return the :class:`ThemePalette` for ``theme``."""
    return _PALETTES[theme]


def contrast_ratio(
    foreground: str | QtGui.QColor, background: str | QtGui.QColor
) -> float:
    """WCAG 2.1 contrast ratio (1-21) of two opaque colours (hex strings or QColors).

    Text needs 4.5:1 against its background and non-text UI (control borders, focus
    rings, plot marks) 3:1 (SC 1.4.3 / 1.4.11).
    """

    def luminance(color: str | QtGui.QColor) -> float:
        c = QtGui.QColor(color)

        def channel(v: float) -> float:
            return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

        return (
            0.2126 * channel(c.redF())
            + 0.7152 * channel(c.greenF())
            + 0.0722 * channel(c.blueF())
        )

    lighter, darker = sorted((luminance(foreground), luminance(background)))[::-1]
    return (lighter + 0.05) / (darker + 0.05)


@lru_cache(maxsize=2)
def load_qss(theme: ThemeName) -> str:
    """Read the ``resources/<theme>.qss`` stylesheet text (cached, unscaled)."""
    return (
        files("cuperiod.gui.resources")
        .joinpath(f"{theme}.qss")
        .read_text(encoding="utf-8")
    )


#: A (possibly negative) px length in a stylesheet, not part of a name or colour.
_PX = re.compile(r"(?<![\w.#])(-?)(\d+(?:\.\d+)?)px\b")


def _scale_lengths(qss: str, factor: float) -> str:
    """Multiply every px length in ``qss`` by ``factor``.

    Lengths round half up and a non-zero length never rounds to zero, so hairline
    borders survive a zoom-out; offsets keep their sign.
    """
    if factor == 1.0:
        return qss

    def scaled(match: re.Match[str]) -> str:
        sign, value = match.group(1), float(match.group(2))
        if value == 0:
            return "0px"
        return f"{sign}{max(1, math.floor(value * factor + 0.5))}px"

    return _PX.sub(scaled, qss)


def stylesheet(theme: ThemeName, scale: float = 1.0) -> str:
    """The ``theme`` stylesheet with every px length (fonts, padding, indicator and
    arrow sizes, radii, borders) multiplied by ``scale``; ``1.0`` is the file verbatim.
    """
    return _scale_lengths(load_qss(theme), scale)


@dataclass
class _UIScale:
    """Process-wide zoom state shared by the stylesheet and the plot fonts."""

    zoom: float = 1.0  # the user's zoom (FONT_SCALE_STEPS)
    factor: float = 1.0  # zoom x max(1, application font px / BASE_FONT_PX)


_SCALE = _UIScale()


def font_scale() -> float:
    """The current UI zoom (1.0 = designed size), as last set by :func:`apply_theme`."""
    return _SCALE.zoom


def step_font_scale(scale: float, steps: int) -> float:
    """The zoom ``steps`` notches from ``scale`` along :data:`FONT_SCALE_STEPS`.

    Positive ``steps`` zoom in, negative zoom out, clamped at either end; a ``scale``
    between two notches moves to the next notch in that direction. Use it for
    View > Zoom In (``+1``) / Zoom Out (``-1``); Reset is ``1.0``.
    """
    levels = FONT_SCALE_STEPS
    if steps == 0:
        return scale
    if steps > 0:
        above = [i for i, v in enumerate(levels) if v > scale + 1e-9]
        if not above:
            return levels[-1]
        return levels[min(len(levels) - 1, above[0] + steps - 1)]
    below = [i for i, v in enumerate(levels) if v < scale - 1e-9]
    if not below:
        return levels[0]
    return levels[max(0, below[-1] + steps + 1)]


def _font_px(font: QtGui.QFont) -> float:
    """Pixel size of ``font``: as rendered, else as requested (no fonts installed, as
    on a bare offscreen platform), else its point size at the screen's logical DPI."""
    rendered = QtGui.QFontInfo(font).pixelSize()
    if rendered > 0:
        return float(rendered)
    if font.pixelSize() > 0:
        return float(font.pixelSize())
    screen = QtGui.QGuiApplication.primaryScreen()
    dpi = screen.logicalDotsPerInchY() if screen is not None else 96.0
    return font.pointSizeF() * dpi / 72.0 if font.pointSizeF() > 0 else BASE_FONT_PX


def _checked_scale(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"font_scale must be a positive number, got {value!r}")
    return min(max(value, FONT_SCALE_STEPS[0]), FONT_SCALE_STEPS[-1])


def _resource_dir() -> Path | None:
    """A filesystem directory holding the stylesheet images (extracted if zipped)."""
    root = files("cuperiod.gui.resources")
    if isinstance(root, Path):
        return root
    try:  # a zipped install: extract once for the life of the process
        stack = contextlib.ExitStack()
        path = stack.enter_context(as_file(root))
    except (OSError, TypeError, NotImplementedError):  # pragma: no cover - odd loaders
        return None
    atexit.register(stack.close)  # pragma: no cover
    return Path(path)  # pragma: no cover


@lru_cache(maxsize=1)
def _register_resources() -> bool:
    """Point the ``cuperiodtheme:`` url prefix at the packaged images (once)."""
    directory = _resource_dir()
    if directory is None:  # pragma: no cover - images then simply do not draw
        return False
    QtCore.QDir.setSearchPaths(RESOURCE_PREFIX, [str(directory)])
    return bool(QtCore.QDir.searchPaths(RESOURCE_PREFIX))


def qpalette(pal: ThemePalette) -> QtGui.QPalette:
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


#: Former private name, kept for callers that reached for it.
_qpalette = qpalette


def plot_label_style(pal: ThemePalette) -> dict[str, str]:
    """CSS kwargs for a pyqtgraph axis label (colour + larger font, zoom-scaled)."""
    return {"color": pal.plot_fg, "font-size": f"{_LABEL_PT * _SCALE.factor:g}pt"}


def apply_plot_theme(plot: Any, pal: ThemePalette) -> None:
    """Style a pyqtgraph PlotWidget: background, axis colours, and readable fonts.

    Sets the tick text/line colour to the palette foreground and enlarges the tick and
    axis-label fonts (by the current UI zoom), then re-applies each existing axis label
    in the new style. Call at creation and on every theme or zoom change so light-theme
    axes stay legible.

    Automatic SI-prefix scaling is switched off: pyqtgraph otherwise relabels a
    unitless axis whose values stay below 1 as ``"Power (x0.001)"`` and draws a power
    of 0.73 as ``730``, which no table or readout in the app agrees with.
    """
    plot.setBackground(pal.plot_bg)
    foreground = pal.qcolor(pal.plot_fg)
    tick_font = QtGui.QFont()
    tick_font.setPointSizeF(_TICK_PT * _SCALE.factor)
    plot_item = plot.getPlotItem()
    for side in ("bottom", "left"):
        axis = plot_item.getAxis(side)
        axis.setTextPen(foreground)
        axis.setPen(foreground)
        axis.setStyle(tickFont=tick_font)
        # pyqtgraph's enableAutoSIPrefix(False) still recomputes the scale once from the
        # current range (and then keeps it): with data already on screen it can leave a
        # x1000 tick scale behind with no "(x0.001)" in the label. An empty set of
        # enable-ranges pins that computation to 1 first; setLabel() later resets the
        # ranges but, with the prefix disabled, never recomputes the scale. pyqtgraph
        # before 0.13.4 has no enable-ranges hook (the pin allows 0.13), so the scale
        # it computed is also reset by hand; on newer versions that is a no-op.
        if hasattr(axis, "setSIPrefixEnableRanges"):
            axis.setSIPrefixEnableRanges(())
        axis.enableAutoSIPrefix(False)
        axis.autoSIPrefixScale = 1.0
        axis.labelUnitPrefix = ""
        axis.picture = None  # redraw the ticks at the reset scale
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


def apply_theme(
    app: QtWidgets.QApplication, theme: ThemeName, *, font_scale: float | None = None
) -> ThemePalette:
    """Apply ``theme`` to ``app``: Fusion style, palette, stylesheet, pyqtgraph globals.

    Parameters
    ----------
    app : QApplication
        The application to style.
    theme : {"dark", "light"}
        The theme to apply.
    font_scale : float, optional
        UI zoom (1.0 = designed size), clamped to :data:`FONT_SCALE_STEPS`; ``None``
        keeps the current zoom. Every stylesheet length and the plot fonts scale with
        it, and with an application font larger than :data:`BASE_FONT_PX`. Views must
        re-run :func:`apply_plot_theme` (their ``apply_theme``) to pick a new zoom up.

    Returns
    -------
    ThemePalette
        The active palette, so the caller can hand it to views for re-skinning.

    Raises
    ------
    ValueError
        If ``font_scale`` is not a positive finite number.
    """
    if font_scale is not None:
        _SCALE.zoom = _checked_scale(font_scale)
    # The sheet sets every widget's font size, so the application font is read here
    # (it is never overwritten): a user or platform font above 13 px enlarges the UI.
    _SCALE.factor = _SCALE.zoom * max(1.0, _font_px(app.font()) / BASE_FONT_PX)
    pal = palette(theme)
    _register_resources()
    app.setStyle("Fusion")
    app.setPalette(qpalette(pal))
    app.setStyleSheet(stylesheet(theme, _SCALE.factor))
    pg.setConfigOption("background", pal.plot_bg)
    pg.setConfigOption("foreground", pal.plot_fg)
    return pal


__all__ = [
    "BASE_FONT_PX",
    "FONT_SCALE_STEPS",
    "RESOURCE_PREFIX",
    "ThemeName",
    "ThemePalette",
    "app_icon",
    "apply_plot_theme",
    "apply_theme",
    "contrast_ratio",
    "font_scale",
    "load_qss",
    "palette",
    "plot_label_style",
    "qpalette",
    "step_font_scale",
    "stylesheet",
]
