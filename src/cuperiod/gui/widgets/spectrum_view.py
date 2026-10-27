"""The interactive spectrum view (pyqtgraph).

Renders the **full-resolution** spectrum, overlays the significant peaks, and marks the
selection with a **translucent shaded band** drawn behind the curve — so the peak stays
visible — that is draggable and emits :attr:`period_selected` (snapped to the nearest
grid sample). Clicking a peak marker emits :attr:`peak_selected` with the whole
:class:`~cuperiod.core.result.Peak`, so method extras such as a BLS epoch survive the
click. Markers and band form one layer that the *peaks* toggle shows or hides together;
the selection itself survives being hidden. A crosshair reads out frequency, period and
the plotted quantity under the cursor.

10^5-10^6 samples stay smooth to pan and zoom because each trace is decimated for the
screen: the samples are binned uniformly in the *plotted* x coordinate (the one the
screen is linear in, so log10 on a log axis) at about two bins per pixel, and every bin
keeps its first, last, lowest and highest sample. Every peak and dip is drawn at any
zoom, and zooming in re-bins the visible slice down to every sample. pyqtgraph's own
auto-downsampling cannot be used here: it strides by sample index assuming uniform x,
which on a period or log axis drops the whole long-period end of the curve.

The x-axis toggles frequency/period (arrays are reversed in period mode so x stays
ascending) and each axis toggles linear/log; a log axis labels only the ticks it has
room for. The axis choice is kept across results of the same kind (a box search, a
pre-whitening spectrum, any other periodogram) and reset to that kind's default only
when the kind changes. Double-clicking anywhere restores the default, auto-ranged view.
Display adapts to the objective sense: for minimise methods (PDM/CE/string-length) the
peaks are minima and the y-axis is labelled accordingly.

The same view serves pre-whitening: the main curve becomes the amplitude spectrum of
the data, an optional **overlay** curve shows the spectrum of the residuals once every
extracted component has been subtracted, and the peak markers become the components.
That side-by-side is the whole point of the method — what was there, and what is left.
A third, dashed trace can show the **spectral window** of the sampling (scaled to the
tallest peak, Period04-style) so an alias lobe is recognisable at a glance. The dense
data curve draws over both, so it has its own toggle: hide it and the residual and
window become readable on their own.

Picking on the plot. A plain click selects: it folds at the peak marker within 12 px
of the click, else at the tallest local maximum within 12 px (Alt: exactly where
clicked). The click is acted on only after the double-click interval, because pyqtgraph
reports the first half of every double click as a single click, and a double click must
reset the view without moving the fold first. In **pick mode** (interactive
pre-whitening, :meth:`SpectrumView.set_pick_enabled`) a click moves the *pick cursor*
instead — a dash-dot line separate from the selection that says where "add" acts and
never moves the fold; Shift+click asks to add the peak there (Ctrl+Shift: exactly
there), a click on a component marker selects that component, and the right button
opens a menu of the same actions. Component markers carry their state by shape (filled
circle free, square fixed frequency, diamond tied, hollow not used), a hollow triangle
marks the "next" candidate, and a dash-dot trace the S/N acceptance limit.

The plot is keyboard operable once it has focus: Left/Right move the cursor to the
previous/next local maximum (Ctrl: marked peak, Shift: one grid sample), Home to the
tallest, N to the next candidate, Enter folds at the cursor, +/- zoom about it, 0 resets
the view and the Menu key opens the action menu. Its accessible description summarises
the spectrum (range, peaks, best period, selection, cursor).

When there is no spectrum to show, :meth:`SpectrumView.set_placeholder` writes why
inside the plot (a failed pre-whitening preview: "Spectrum unavailable: …"); the next
spectrum removes it.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.core.columns import Domain
from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, Signal, pg
from cuperiod.gui.theme import (
    ThemePalette,
    apply_plot_theme,
    palette,
    plot_label_style,
)
from cuperiod.gui.widgets._view_helpers import (
    ElidedLabel,
    ask_save_path,
    confirm_overwrite,
    device_scale,
    image_target,
    is_connected,
    report_failure,
    stroke_width,
    write_plot_image,
)

if TYPE_CHECKING:
    from cuperiod.gui.dialogs import DialogProvider

_MIN_OBJECTIVE_LABEL = "dispersion (lower = better)"

#: Method name carried by the synthetic Periodogram that wraps an amplitude spectrum.
PREWHITEN_METHOD = "Pre-whitening"

#: Methods whose results open on a log period axis (box/transit searches).
_BOX_METHODS = frozenset({"BLS", "TLS"})

#: Decimation bins per horizontal pixel of the plot. Two keep a vertical stroke per
#: pixel column for both the top and the bottom of the envelope.
_BINS_PER_PIXEL = 2

#: Plot width assumed before the widget has been laid out (or when it is very narrow),
#: so a curve decimated early is never coarser than a typical screen needs.
_MIN_PLOT_PIXELS = 1000

#: A visible slice with at most this many samples per bin is drawn as is.
_FULL_RESOLUTION_PER_BIN = 4

#: Fraction of the view height below the top in which a peak's hover label is hung
#: under the marker instead of above it (where it would be clipped).
_HOVER_FLIP_TOP = 0.2

#: Fraction of the view width from the right edge in which the label grows leftwards.
_HOVER_FLIP_RIGHT = 0.3

#: Screen distance (px) within which a click snaps to a marker or a local maximum. A
#: marker symbol is only 9-15 px across; a near miss used to start a pan instead.
SNAP_PX = 12.0

#: Scatter symbol of a component marker per state: the shape, not the colour, carries
#: the state (WCAG 1.4.1). An inactive component is the hollow circle.
_STATE_SYMBOLS = {"free": "o", "fixed": "s", "tied": "d", "inactive": "o"}

#: Words for the marker states, for the hover label and the accessible summary.
_STATE_WORDS = {
    "free": "free",
    "fixed": "frequency fixed",
    "tied": "tied",
    "inactive": "not used",
}

#: Keys the focused plot handles itself; they must win over any shortcut.
_NAV_KEYS = frozenset(
    {
        Qt.Key.Key_Left,
        Qt.Key.Key_Right,
        Qt.Key.Key_Home,
        Qt.Key.Key_Return,
        Qt.Key.Key_Enter,
        Qt.Key.Key_N,
        Qt.Key.Key_Plus,
        Qt.Key.Key_Equal,
        Qt.Key.Key_Minus,
        Qt.Key.Key_0,
        Qt.Key.Key_Menu,
    }
)

#: The keys the focused spectrum acts on itself, as ``(what, keys)`` for the F1 sheet
#: (the registry's A / Insert and the rest are listed with their commands).
SPECTRUM_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Cursor to the previous / next local maximum", ("Left", "Right")),
    (
        "Cursor to the previous / next marked peak (pre-whitening: component)",
        ("Ctrl+Left", "Ctrl+Right"),
    ),
    ("Cursor one sample left / right", ("Shift+Left", "Shift+Right")),
    ("Cursor to the tallest peak", ("Home",)),
    ("Cursor to the next candidate (pre-whitening)", ("N",)),
    ("Fold at the cursor", ("Enter",)),
    ("Zoom in / out", ("+", "-")),
    ("Reset the view", ("0",)),
    ("The plot's menu", ("Menu",)),
)

#: The keyboard and mouse gestures, for the plot's status tip and What's This text.
_GESTURES_PERIODOGRAM = (
    "Click: fold at the nearest peak (Alt: exactly there) · drag the shaded band · "
    "double-click: reset the view · Left/Right: cursor to the previous/next local "
    "maximum (Ctrl: marked peak, Shift: one sample) · Home: tallest · Enter: fold at "
    "the cursor · +/-: zoom · 0: reset · Menu key or right-click: more"
)
_GESTURES_PICK = (
    "Click: move the pick cursor to the nearest peak (Alt: exactly there) · click a "
    "marker: select that component · Shift+click: add the peak there (Ctrl+Shift: "
    "exactly there) · Left/Right: cursor to the previous/next local maximum (Ctrl: "
    "component, Shift: one sample) · N: next candidate · A or Insert: add at the "
    "cursor · Enter: fold at the cursor · +/-: zoom · 0: reset · right-click: more"
)


def local_maxima(values: FloatArray, *, maximise: bool = True) -> IntArray:
    """Indices of the local maxima of ``values`` (minima when ``maximise`` is False).

    A sample is a maximum when it is above its left neighbour and not below its right
    one, so a plateau counts once (its first sample) and an end sample counts when it
    beats its only neighbour. Non-finite samples are never extrema and never beat a
    finite neighbour.
    """
    y = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(y)
    score = np.where(finite, y if maximise else -y, -np.inf)
    n = score.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    rises = np.ones(n, dtype=np.bool_)
    rises[1:] = score[1:] > score[:-1]
    holds = np.ones(n, dtype=np.bool_)
    holds[:-1] = score[:-1] >= score[1:]
    return np.flatnonzero(rises & holds & finite).astype(np.int64)


def _nearest_index(grid: FloatArray, value: float) -> int:
    """Index of the sample of the ascending, non-empty ``grid`` nearest ``value``."""
    n = int(grid.size)
    idx = int(np.searchsorted(grid, value))
    if idx <= 0:
        return 0
    if idx >= n:
        return n - 1
    return idx - 1 if value - grid[idx - 1] <= grid[idx] - value else idx


def _candidate_symbol() -> QtGui.QPainterPath:
    """A downward triangle whose *tip* is the symbol's origin.

    pyqtgraph centres a symbol on its point, so the built-in triangle would sit over
    the peak and hide it; this one stands on the peak and points at it.
    """
    path = QtGui.QPainterPath()
    path.moveTo(0.0, 0.0)
    path.lineTo(-0.36, -0.5)
    path.lineTo(0.36, -0.5)
    path.closeSubpath()
    return path


def _unit_word(unit: Domain | str | None) -> str | None:
    """``"mag"`` / ``"flux"`` for a light-curve domain (or a unit given as text)."""
    if unit is None:
        return None
    if isinstance(unit, Domain):
        return "mag" if unit is Domain.MAGNITUDE else "flux"
    text = str(unit).strip()
    try:
        return _unit_word(Domain(text))
    except ValueError:
        return text or None


def envelope_indices(
    plot_x: FloatArray, y: FloatArray, lo: float, hi: float, n_bins: int
) -> IntArray:
    """Indices of the samples that draw ``y`` over ``[lo, hi]`` at ``n_bins`` columns.

    ``plot_x`` must be finite and ascending, in the coordinate the screen is linear in
    (log10 of the data on a log axis). The slice covering ``[lo, hi]`` — plus one
    sample beyond each end, so the line runs to the edges — is split into ``n_bins``
    bins of equal width in that coordinate, and each bin keeps its first, last, lowest
    and highest sample; NaN values never win a bin. A slice with no more than
    :data:`_FULL_RESOLUTION_PER_BIN` samples per bin is returned whole.

    The result is ascending and free of duplicates, so it indexes the original arrays
    directly: the drawn curve is made of real samples only. Binning by index instead
    (pyqtgraph's auto-downsampling) assumes uniform x and, on a period or log axis,
    merges the sparse long-period end into a few vertices and drops its remainder.
    """
    n = int(plot_x.size)
    start = max(int(np.searchsorted(plot_x, lo, side="left")) - 1, 0)
    stop = min(int(np.searchsorted(plot_x, hi, side="right")) + 1, n)
    if stop <= start:
        return np.zeros(0, dtype=np.int64)
    m = stop - start
    if m <= _FULL_RESOLUTION_PER_BIN * n_bins:
        return np.arange(start, stop, dtype=np.int64)
    xs = plot_x[start:stop]
    ys = y[start:stop]
    span = float(xs[-1] - xs[0])
    if not span > 0.0:  # every sample at one x: its extremes are all there is
        bins = np.zeros(m, dtype=np.int64)
    else:
        bins = ((xs - xs[0]) * (n_bins / span)).astype(np.int64)
    starts = np.flatnonzero(np.diff(bins)) + 1
    starts = np.concatenate((np.zeros(1, dtype=np.int64), starts))
    counts = np.diff(np.append(starts, m))
    ends = starts + counts - 1
    if np.isnan(ys).any():
        high = np.where(np.isnan(ys), -np.inf, ys)
        low = np.where(np.isnan(ys), np.inf, ys)
    else:
        high = low = ys
    position = np.arange(m, dtype=np.int64)
    top = np.repeat(np.maximum.reduceat(high, starts), counts)
    arg_top = np.minimum.reduceat(np.where(high == top, position, m), starts)
    bottom = np.repeat(np.minimum.reduceat(low, starts), counts)
    arg_bottom = np.minimum.reduceat(np.where(low == bottom, position, m), starts)
    keep = np.unique(np.concatenate((starts, ends, arg_top, arg_bottom)))
    return (keep + start).astype(np.int64)


class _Trace:
    """One curve's full-resolution data and the decimated slice drawn of it."""

    def __init__(self, item: Any) -> None:
        self.item = item
        self._x: FloatArray = np.zeros(0)
        self._y: FloatArray = np.zeros(0)
        self._plot_x: FloatArray = np.zeros(0)
        self._keep: IntArray | None = None
        self._log_x: bool | None = None
        self._version = 0
        self._drawn: tuple[Any, ...] | None = None

    @property
    def empty(self) -> bool:
        return self._x.size == 0

    def load(self, x: FloatArray, y: FloatArray) -> None:
        """Hold ``(x, y)`` in data coordinates (x ascending) for later drawing."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size > 1 and not bool(np.all(x[1:] >= x[:-1])):
            order = np.argsort(x, kind="stable")
            x, y = x[order], y[order]
        self._x, self._y = x, y
        self._log_x = None
        self._version += 1

    def clear(self) -> None:
        self.load(np.zeros(0), np.zeros(0))
        self.item.setData([], [])
        self._drawn = None

    def _prepare(self, log_x: bool) -> None:
        """Plot coordinates of the samples a ``log_x`` axis can show (cached)."""
        if self._log_x is log_x:
            return
        if log_x:
            with np.errstate(divide="ignore", invalid="ignore"):
                plot_x = np.log10(self._x)
        else:
            plot_x = self._x
        drawable = np.isfinite(plot_x)
        self._keep = None if bool(drawable.all()) else np.flatnonzero(drawable)
        self._plot_x = plot_x if self._keep is None else plot_x[self._keep]
        self._log_x = log_x

    def draw(
        self, log_x: bool, lo: float | None, hi: float | None, n_bins: int
    ) -> None:
        """Hand pyqtgraph the samples needed for ``[lo, hi]`` (None: everything)."""
        self._prepare(log_x)
        if self._plot_x.size == 0:
            if self._drawn is not None:
                self.item.setData([], [])
                self._drawn = None
            return
        if lo is None or hi is None:
            lo, hi = float(self._plot_x[0]), float(self._plot_x[-1])
        key = (self._version, log_x, lo, hi, n_bins)
        if key == self._drawn:
            return
        ys = self._y if self._keep is None else self._y[self._keep]
        xs = self._x if self._keep is None else self._x[self._keep]
        index = envelope_indices(self._plot_x, ys, lo, hi, n_bins)
        self.item.setData(xs[index], ys[index])
        self._drawn = key


def _tick_text(log_value: float) -> str:
    """A log-axis tick label: plain digits from 0.001 to 10^4, else m·10^k."""
    exponent = math.floor(log_value + 1e-9)
    mantissa = round(10.0 ** (log_value - exponent))
    if mantissa >= 10:
        mantissa, exponent = 1, exponent + 1
    if -3 <= exponent <= 4:
        if exponent >= 0:
            return str(mantissa * 10**exponent)
        return f"{mantissa * 10.0**exponent:.{-exponent}f}"
    digits = str(abs(exponent)).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    power = f"10{'⁻' if exponent < 0 else ''}{digits}"
    return power if mantissa == 1 else f"{mantissa}·{power}"


#: Order in which log-axis mantissas claim room for a label: decades first, then the
#: 5s and 2s that subdivide them most evenly, then the rest.
_MANTISSA_PRIORITY = {1: 0, 5: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 9: 8}


class _LogTickAxis(pg.AxisItem):
    """An axis whose log-scale labels never overprint one another.

    pyqtgraph labels every 1-9 multiple of a decade once there is room for them *on
    average*, but on a log scale they crowd at the top of each decade (7, 8, 9) and the
    labels overprint ("0.50.6 0.7 0.8.91"). Here a label is kept only if it clears the
    ones already placed, claimed in the order decades, 5s, 2s, then the rest; every
    tick mark is still drawn. Labels read as plain numbers where that is short.
    """

    _GAP_PX = 6.0

    def __init__(self, orientation: str, **kwargs: Any) -> None:
        super().__init__(orientation, **kwargs)
        self._px_per_unit = 0.0
        self._major: set[float] = set()

    def logTickValues(
        self, minVal: float, maxVal: float, size: float, stdTicks: list[Any]
    ) -> list[Any]:
        ticks = super().logTickValues(minVal, maxVal, size, stdTicks)
        self._px_per_unit = size / (maxVal - minVal) if maxVal > minVal else 0.0
        major = [values for spacing, values in ticks if spacing is not None]
        self._major = {round(v, 9) for values in major for v in values}
        # The minor list repeats the decades of the major levels; drop the copies, or
        # the same label is drawn twice and claims its own room twice.
        return [
            (spacing, values)
            if spacing is not None
            else (spacing, [v for v in values if round(v, 9) not in self._major])
            for spacing, values in ticks
        ]

    def logTickStrings(
        self, values: list[float], scale: float, spacing: float | None
    ) -> list[str]:
        texts = [_tick_text(float(v) + math.log10(scale)) for v in values]
        if spacing is not None or self._px_per_unit <= 0.0:
            return texts
        font = self.style.get("tickFont") or QtGui.QFont()
        metrics = QtGui.QFontMetricsF(font)
        horizontal = self.orientation in ("bottom", "top")

        def extent(text: str) -> float:
            if horizontal:
                return float(metrics.horizontalAdvance(text))
            return float(metrics.height())

        placed: list[tuple[float, float]] = [
            (v * self._px_per_unit, extent(_tick_text(v + math.log10(scale))))
            for v in self._major
        ]
        keep = [False] * len(values)

        def priority(i: int) -> int:
            mantissa = round(10.0 ** (values[i] - math.floor(values[i] + 1e-9)))
            return _MANTISSA_PRIORITY.get(int(mantissa), 9)

        for i in sorted(range(len(values)), key=priority):
            centre = values[i] * self._px_per_unit
            width = extent(texts[i])
            if all(
                abs(centre - other) >= (width + other_width) / 2.0 + self._GAP_PX
                for other, other_width in placed
            ):
                placed.append((centre, width))
                keep[i] = True
        return [text if kept else "" for text, kept in zip(texts, keep, strict=True)]


class _ReadoutLabel(ElidedLabel):
    """The readout line: elided at its right end to the width it is given.

    Right-aligned text wider than the label lost its *start* ("cursor  f = 15.4")
    in a narrow centre column; eliding the end keeps the frequency, which matters
    most. :meth:`text` and the accessible description hold the full line.
    """


class SpectrumView(QtWidgets.QWidget):
    """Full-resolution spectrum with peak markers, a selection band, and a crosshair."""

    #: An arbitrary period chosen on the plot (the selection band was dragged), or a
    #: marker click when nothing is connected to :attr:`peak_selected`.
    period_selected = Signal(float)
    #: The :class:`~cuperiod.core.result.Peak` whose marker was clicked.
    peak_selected = Signal(object)
    #: A CSV/PNG export could not be written; carries a message for the user.
    export_failed = Signal(str)
    #: Pick mode: add a component at ``(frequency, exact)`` — ``exact`` False means
    #: "the peak near this frequency", True "exactly this frequency".
    add_requested = Signal(float, bool)
    #: The pick cursor moved by a user gesture (its frequency, cycles/day).
    cursor_changed = Signal(float)
    #: A component marker was clicked (pick mode); carries the component id.
    component_clicked = Signal(int)

    def __init__(
        self, theme: str = "dark", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._pal: ThemePalette = palette(theme)  # type: ignore[arg-type]
        self._pg: Periodogram | None = None
        self._peaks: list[Peak] = []
        self._x_mode = "frequency"
        self._axis_family: str | None = None
        self._sel_period: float | None = None
        self._suppress_band = False
        # Pick mode (interactive pre-whitening) and its state; see the module notes.
        self._pick_enabled = False
        self._cursor_freq: float | None = None
        self._marker_states: tuple[Any, ...] | None = None
        self._candidate: Any | None = None
        self._threshold_xy: tuple[np.ndarray, np.ndarray] | None = None
        self._snr_threshold: float | None = None
        self._amplitude_unit: str | None = None
        self._extrema_key: tuple[Any, ...] | None = None
        self._extrema: tuple[FloatArray, FloatArray, IntArray] | None = None
        self._pending_click: tuple[QtCore.QPointF, Any] | None = None
        self._press_origin: QtCore.QPointF | None = None
        self._press_dragged = False
        self._context_menu: QtWidgets.QMenu | None = None
        self._placeholder_text = ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Tight: in a 768 px window the plot gets what the rows around it leave.
        layout.setSpacing(2)
        layout.addWidget(self._build_toolbar())

        self._plot = pg.PlotWidget(
            axisItems={
                "bottom": _LogTickAxis("bottom"),
                "left": _LogTickAxis("left"),
            }
        )
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        style = plot_label_style(self._pal)
        self._plot.setLabel("bottom", "Frequency (cycles/day)", **style)
        self._plot.setLabel("left", "Power", **style)
        layout.addWidget(self._plot, 1)

        # The crosshair readout gets a row of its own under the plot, with its size
        # hint ignored: in the toolbar its text set the row's minimum width, so the
        # first hover resized the docks around it, and it was still clipped.
        self._readout = _ReadoutLabel("—")
        self._readout.setObjectName("muted")
        self._readout.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._readout.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self._readout.setContentsMargins(6, 0, 6, 0)
        self._readout.setToolTip(
            "Frequency, period and value under the mouse, or at the pick cursor"
        )
        self._readout.setAccessibleName("Spectrum readout")
        layout.addWidget(self._readout)

        # Translucent selection band, behind the curve so the peak shows through.
        self._sel_band = pg.LinearRegionItem(
            movable=True,
            brush=pg.mkBrush(self._pal.qcolor(self._pal.accent, 45)),
            pen=self._band_pen(),
        )
        self._sel_band.setZValue(-20)
        self._sel_band.sigRegionChangeFinished.connect(self._on_band_moved)
        self._plot.addItem(self._sel_band)
        self._sel_band.setVisible(False)

        # No pyqtgraph downsampling or clip-to-view: _Trace decimates for the screen.
        self._curve = self._plot.plot([], [], pen=self._curve_pen())
        self._curve.setZValue(0)

        # Optional second trace (the pre-whitened residual spectrum), drawn over the
        # main curve so what is *left* stays readable against what was there.
        self._overlay = self._plot.plot([], [], pen=self._overlay_pen())
        self._overlay.setZValue(1)
        self._overlay.setVisible(False)
        self._overlay_xy: tuple[np.ndarray, np.ndarray] | None = None

        # The spectral window of the sampling (dashed, behind the data curve): the
        # alias-lobe pattern every real peak is convolved with.
        self._window_curve = self._plot.plot([], [], pen=self._window_pen())
        self._window_curve.setZValue(-5)
        self._window_curve.setVisible(False)
        self._window_xy: tuple[np.ndarray, np.ndarray] | None = None

        # The S/N acceptance limit over the residual spectrum (dash-dot, pick mode).
        self._threshold_curve = self._plot.plot([], [], pen=self._threshold_pen())
        self._threshold_curve.setZValue(2)
        self._threshold_curve.setVisible(False)

        self._data_trace = _Trace(self._curve)
        self._overlay_trace = _Trace(self._overlay)
        self._window_trace = _Trace(self._window_curve)
        self._threshold_trace = _Trace(self._threshold_curve)

        # Names the traces once there is more than one to tell apart (pre-whitening).
        self._legend = self._plot.addLegend(offset=(-10, 10))
        self._legend.setBrush(self._legend_brush())
        self._legend.setVisible(False)

        self._markers = pg.ScatterPlotItem(hoverable=True, pxMode=True)
        self._markers.setZValue(5)
        self._markers.sigClicked.connect(self._on_peak_clicked)
        self._markers.sigHovered.connect(self._on_peak_hovered)
        self._plot.addItem(self._markers)

        self._vline = pg.InfiniteLine(angle=90, pen=self._crosshair_pen())
        self._hline = pg.InfiniteLine(angle=0, pen=self._crosshair_pen())
        for crosshair in (self._vline, self._hline):
            crosshair.setVisible(False)
            self._plot.addItem(crosshair, ignoreBounds=True)
        self._hover_text = pg.TextItem(color=self._pal.plot_fg, anchor=(0, 1))
        self._hover_text.setVisible(False)
        self._plot.addItem(self._hover_text, ignoreBounds=True)

        # The pick cursor: its own line, never part of the peaks or the selection band.
        self._cursor_line = pg.InfiniteLine(angle=90, pen=self._cursor_pen())
        self._cursor_line.setZValue(4)
        self._cursor_line.setVisible(False)
        self._plot.addItem(self._cursor_line, ignoreBounds=True)

        # The "next" candidate: a hollow triangle standing on its peak, with a label
        # above it. Kept out of _markers so it is never mistaken for a component.
        self._candidate_item = pg.ScatterPlotItem(pxMode=True)
        self._candidate_item.setZValue(6)
        self._plot.addItem(self._candidate_item)
        # The empty first line keeps the text clear of the triangle under it.
        self._candidate_label = pg.TextItem(
            "next\n", color=self._pal.plot_fg, anchor=(0.5, 1.0)
        )
        self._candidate_label.setVisible(False)
        self._plot.addItem(self._candidate_label, ignoreBounds=True)

        # A plain click acts only once the double-click interval has passed (module
        # notes); a double click or a new press cancels it.
        self._click_timer = QtCore.QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(QtWidgets.QApplication.doubleClickInterval())
        self._click_timer.timeout.connect(self._on_click_timeout)

        # Re-decimating on every range-changed signal would redo the work several
        # times per pan step; a zero-interval single shot coalesces them.
        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(0)
        self._redraw_timer.timeout.connect(self._draw_traces)

        self._proxy = pg.SignalProxy(
            self._plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )
        # Double-click anywhere in the plot restores the default view, the shortcut
        # every plotting tool has and pyqtgraph leaves unbound.
        self._plot.scene().sigMouseClicked.connect(self._on_scene_clicked)
        # Selecting and picking with single clicks is a second, separate slot.
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        box = self._plot.getPlotItem().vb
        box.sigXRangeChanged.connect(self._on_xrange_changed)
        box.sigResized.connect(self._redraw_timer.start)
        # The crosshair must not stay frozen where the cursor left the plot; the
        # keyboard and drag detection hang off the same filter.
        self._plot.installEventFilter(self)
        viewport = self._plot.viewport()
        if viewport is not None:
            viewport.installEventFilter(self)
        # Text shown inside the plot when there is no spectrum to show (a failed
        # pre-whitening preview): over the plot, not taking its clicks.
        self._placeholder = QtWidgets.QLabel(self._plot)
        self._placeholder.setObjectName("spectrum_placeholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setTextFormat(Qt.TextFormat.PlainText)
        self._placeholder.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._placeholder.setAccessibleName("Spectrum message")
        self._placeholder.hide()
        self._style_placeholder()
        # A keyboard user reaches the plot with Tab and drives it with the keys above;
        # assistive technology reads its name and the summary in its description.
        self._plot.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._plot.setAccessibleName("Spectrum plot")
        self._sync_gesture_help()
        self._update_accessible()
        apply_plot_theme(self._plot, self._pal)

    # -- toolbar -----------------------------------------------------------------
    def _build_toolbar(self) -> QtWidgets.QWidget:
        from cuperiod.gui.widgets.info_bar import FlowLayout

        bar = QtWidgets.QWidget()
        bar.setObjectName("plotToolbar")  # compact controls (theme sheets)
        # A wrapping row: with the pre-whitening toggles shown the toolbar is the
        # widest thing in the pane, and in a narrow window an unwrapped row squeezed
        # the check boxes and buttons into unreadable stubs (UX-04).
        row = FlowLayout(hspacing=6, vspacing=2)
        bar.setLayout(row)
        row.setContentsMargins(6, 2, 6, 2)

        x_label = QtWidgets.QLabel("x:")
        row.addWidget(x_label)
        self._xaxis_combo = QtWidgets.QComboBox()
        self._xaxis_combo.addItems(["frequency", "period"])
        self._xaxis_combo.setToolTip("Plot the x-axis as frequency or period")
        self._xaxis_combo.setAccessibleName("Spectrum x axis")
        x_label.setBuddy(self._xaxis_combo)
        # Never squeeze the combo into an elided "frequen…" when the row gets tight;
        # shortage lands on the trailing stretch instead. The explicit minimum is
        # needed because a combo's minimum size ignores its contents by default.
        self._xaxis_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._xaxis_combo.setMinimumWidth(self._xaxis_combo.sizeHint().width())
        self._xaxis_combo.currentTextChanged.connect(self._on_xaxis_changed)
        row.addWidget(self._xaxis_combo)

        self._logx = QtWidgets.QCheckBox("log x")
        self._logy = QtWidgets.QCheckBox("log y")
        self._logx.setToolTip("Plot the x-axis on a logarithmic scale")
        self._logy.setToolTip("Plot the y-axis on a logarithmic scale")
        self._logx.toggled.connect(self._apply_log)
        self._logy.toggled.connect(self._apply_log)
        row.addWidget(self._logx)
        row.addWidget(self._logy)

        self._show_peaks = QtWidgets.QCheckBox("peaks")
        self._show_peaks.setChecked(True)
        self._show_peaks.setToolTip(
            "Mark the significant peaks, and shade the selected one. Turn it off for "
            "an unobstructed view of the spectrum itself"
        )
        self._show_peaks.toggled.connect(self._on_peaks_toggled)
        row.addWidget(self._show_peaks)

        # Only offered once there is an overlay to read: hiding the data curve of a
        # plain periodogram would just leave an empty plot.
        self._show_data = QtWidgets.QCheckBox("data")
        self._show_data.setChecked(True)
        self._show_data.setToolTip(
            "Show the amplitude spectrum of the data. Turn it off to read the "
            "residual and window traces, which it otherwise draws over"
        )
        self._show_data.setVisible(False)
        self._show_data.toggled.connect(self._redraw_curve)
        row.addWidget(self._show_data)

        self._show_residual = QtWidgets.QCheckBox("residual")
        self._show_residual.setChecked(True)
        self._show_residual.setToolTip(
            "Overlay the amplitude spectrum of the residuals after pre-whitening"
        )
        self._show_residual.setVisible(False)
        self._show_residual.toggled.connect(self._redraw_overlay)
        row.addWidget(self._show_residual)

        self._show_window = QtWidgets.QCheckBox("window")
        self._show_window.setChecked(False)
        self._show_window.setToolTip(
            "Overlay the spectral window of the sampling (scaled to the tallest "
            "peak): a peak sitting on another's window lobe is likely an alias"
        )
        self._show_window.setVisible(False)
        self._show_window.toggled.connect(self._redraw_window)
        row.addWidget(self._show_window)

        reset = QtWidgets.QPushButton("Reset")
        reset.setToolTip(
            "Auto-range the plot back to the full spectrum "
            "(or just double-click the plot)"
        )
        reset.clicked.connect(self._autorange)
        row.addWidget(reset)

        export = QtWidgets.QToolButton()
        export.setText("Export…")
        export.setToolTip("Export the spectrum data or a plot image")
        export.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QtWidgets.QMenu(export)
        csv_action = export_menu.addAction("Spectrum as CSV…")
        csv_action.triggered.connect(self._export_csv)
        png_action = export_menu.addAction("Image as PNG…")
        png_action.triggered.connect(self._export_png)
        export.setMenu(export_menu)
        row.addWidget(export)
        return bar

    # -- data --------------------------------------------------------------------
    def set_periodogram(self, pg_result: Periodogram) -> None:
        """Show a new spectrum (clears peaks, the overlay, and the selection band).

        A new spectrum also drops the pick cursor, the component markers, the
        candidate and the threshold: they belonged to the previous one. To refresh the
        traces of the same session use :meth:`update_traces`.
        """
        self._pg = pg_result
        self._peaks = []
        self._sel_period = None
        self.set_placeholder(None)  # there is a spectrum to show now
        self._reset_pick_state()
        self._markers.clear()
        self._sync_band()
        self._hide_crosshair()  # its numbers belonged to the previous spectrum
        self._overlay_xy = None
        self._show_residual.setVisible(False)
        self._overlay_trace.clear()
        self._overlay.setVisible(False)
        self._window_xy = None
        self._show_window.setVisible(False)
        self._window_trace.clear()
        self._window_curve.setVisible(False)
        self._show_data.setVisible(False)
        self._show_data.setChecked(True)
        self._default_axis_for(pg_result)
        self._redraw_curve()
        self._update_y_label()
        self._autorange()
        self._update_accessible()

    def _reset_pick_state(self) -> None:
        """Forget the cursor, component markers, candidate and threshold."""
        self._click_timer.stop()
        self._pending_click = None
        self._cursor_freq = None
        self._cursor_line.setVisible(False)
        self._marker_states = None
        self._candidate = None
        self._candidate_item.clear()
        self._candidate_label.setVisible(False)
        self._threshold_xy = None
        self._snr_threshold = None
        self._threshold_trace.clear()
        self._threshold_curve.setVisible(False)
        self._extrema_key = None
        self._extrema = None

    # -- pick mode, session traces and markers (interactive pre-whitening) ---------
    def set_pick_enabled(self, enabled: bool) -> None:
        """Turn pick mode on (interactive pre-whitening) or off (periodograms).

        In pick mode a click moves the pick cursor instead of the selection, Shift+click
        and the A / Insert keys ask to add a component, and markers report the
        component they stand for (:attr:`component_clicked`).
        """
        enabled = bool(enabled)
        if enabled == self._pick_enabled:
            return
        self._pick_enabled = enabled
        self._click_timer.stop()
        self._pending_click = None
        self._extrema_key = None  # pick mode follows the residual trace
        self._place_cursor()
        self._sync_gesture_help()
        self._update_accessible()

    def pick_enabled(self) -> bool:
        """Whether pick mode is on."""
        return self._pick_enabled

    def focus_plot(self) -> None:
        """Give the plot the keyboard focus (View ▸ Focus spectrum, Ctrl+1)."""
        self._plot.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def update_traces(
        self, data: Any, residual: Any = None, *, keep_range: bool = True
    ) -> None:
        """Replace the data and residual spectra of the same session in place.

        ``data`` is an amplitude spectrum (anything with ``frequency`` and
        ``amplitude``, e.g. :class:`~cuperiod.prewhiten.AmplitudeSpectrum`) or a
        :class:`~cuperiod.core.result.Periodogram`; ``residual`` likewise, or a
        ``(frequency, values)`` pair, or ``None`` to drop the overlay. Peaks, component
        markers, the selection and the pick cursor are kept, and with ``keep_range``
        the view does not re-range: an edit must not throw away the user's zoom. Only
        the first spectrum (or ``keep_range=False``) auto-ranges.
        """
        wrapped = self._as_periodogram(data)
        if self._pg is None or self._axis_family_of(wrapped) != self._axis_family:
            keep_range = False
        self.set_placeholder(None)  # there is a spectrum to show now
        box = self._plot.getPlotItem().vb
        auto_x, auto_y = box.autoRangeEnabled()
        x_range, y_range = box.viewRange()
        self._pg = wrapped
        self._default_axis_for(wrapped)
        self._overlay_xy = None if residual is None else self._as_xy(residual)
        self._show_residual.setVisible(self._overlay_xy is not None)
        self._show_data.setVisible(
            self._overlay_xy is not None or self._window_xy is not None
        )
        if self._overlay_xy is None:
            self._overlay_trace.clear()
            self._overlay.setVisible(False)
            self._sync_data_toggle()
        self._extrema_key = None
        x, y = self._curve_xy()
        self._data_trace.load(x, y)
        self._curve.setVisible(self._show_data.isChecked())
        self._load_trace(self._overlay_trace, self._overlay_xy, self._show_residual)
        self._load_trace(self._window_trace, self._window_xy, self._show_window)
        self._load_trace(self._threshold_trace, self._threshold_xy, None)
        self._update_y_label()
        if keep_range:
            self._draw_traces()
            if not auto_x:
                box.setXRange(*x_range, padding=0.0)
            if not auto_y:
                box.setYRange(*y_range, padding=0.0)
        else:
            self._apply_log()
            self._autorange()
        self._sync_legend()
        self._update_band()
        self._redraw_markers()
        self._redraw_candidate()
        self._place_cursor()
        self._update_accessible()

    def set_marker_states(self, states: Sequence[Any] | None) -> None:
        """Draw the components as markers whose shape is their state.

        ``states`` are :class:`~cuperiod.gui.viewmodels.MarkerState` records (``id``,
        ``frequency``, ``shape`` in ``free`` / ``fixed`` / ``tied`` / ``inactive``,
        ``selected``): a filled circle is free, a square has its frequency fixed, a
        diamond is tied and a hollow circle is not used. Markers stand on the data
        spectrum. ``None`` returns to drawing :meth:`set_peaks` peaks.
        """
        self._marker_states = None if states is None else tuple(states)
        self._redraw_markers()
        self._update_accessible()

    def set_candidate(self, candidate: Any | None) -> None:
        """Mark what Step would try next (a hollow "next" triangle), or nothing.

        ``candidate`` is a :class:`~cuperiod.gui.viewmodels.CandidateView` (anything
        with ``frequency``, optionally ``amplitude`` and ``hint``). It is not a peak:
        it never enters the peak list, the selection or the hover readout.
        """
        self._candidate = candidate
        self._redraw_candidate()
        self._update_accessible()

    def candidate_frequency(self) -> float | None:
        """Frequency of the "next" candidate, if one is shown."""
        if self._candidate is None:
            return None
        frequency = float(self._candidate.frequency)
        return frequency if math.isfinite(frequency) and frequency > 0.0 else None

    def set_cursor(self, frequency: float | None) -> None:
        """Place the pick cursor at ``frequency`` (``None`` hides it); no echo."""
        if frequency is not None:
            frequency = float(frequency)
            if not (math.isfinite(frequency) and frequency > 0.0):
                frequency = None
        self._cursor_freq = frequency
        self._place_cursor()
        self._show_cursor_readout()
        self._update_accessible()

    def cursor_frequency(self) -> float | None:
        """Frequency of the pick cursor (cycles/day), or ``None``."""
        return self._cursor_freq

    def set_threshold(
        self,
        frequency: np.ndarray,
        values: np.ndarray,
        *,
        snr_threshold: float | None = None,
    ) -> None:
        """Show the S/N acceptance limit (``snr_threshold x noise``) as a trace.

        With ``snr_threshold`` the context menu also estimates a peak's S/N as its
        residual amplitude over the noise (``values / snr_threshold``).
        """
        self._threshold_xy = (
            np.asarray(frequency, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
        self._snr_threshold = None
        if (
            snr_threshold is not None
            and math.isfinite(snr_threshold)
            and snr_threshold > 0.0
        ):
            self._snr_threshold = float(snr_threshold)
        self._load_trace(self._threshold_trace, self._threshold_xy, None)
        self._draw_traces()
        self._sync_legend()

    def clear_threshold(self) -> None:
        """Remove the S/N acceptance trace."""
        self._threshold_xy = None
        self._snr_threshold = None
        self._threshold_trace.clear()
        self._threshold_curve.setVisible(False)
        self._sync_legend()

    def set_amplitude_unit(self, unit: Domain | str | None) -> None:
        """Name the amplitude unit from the light curve's domain (mag / flux).

        The pre-whitening y-axis then reads "Amplitude (mag)" or "Amplitude (flux)" and
        the readout carries the unit; ``None`` leaves the amplitude unitless.
        """
        self._amplitude_unit = _unit_word(unit)
        self._update_y_label()

    def set_session_view(
        self,
        view: Any,
        *,
        selected: Sequence[int] = (),
        keep_range: bool = True,
    ) -> None:
        """Show one :class:`~cuperiod.gui.viewmodels.SessionView` in one call.

        Updates the data and residual traces (keeping the range), the spectral
        window, the S/N limit, the candidate and the component markers, with the
        components in ``selected`` drawn as selected. The pick cursor and the selection
        band are left alone.
        """
        from cuperiod.gui.viewmodels import marker_states

        self.update_traces(
            view.data_spectrum, view.residual_spectrum, keep_range=keep_range
        )
        window = getattr(view, "window", None)
        if window is not None and np.size(window.frequency):
            amplitude = np.asarray(view.data_spectrum.amplitude, dtype=np.float64)
            finite = amplitude[np.isfinite(amplitude)]
            scale = float(finite.max()) if finite.size else 1.0
            self.set_window(window.frequency, window.amplitude, scale=scale)
        else:
            self.clear_window()
        settings = getattr(view, "settings", None)
        snr = getattr(settings, "snr_threshold", None)
        if view.threshold is not None:
            self.set_threshold(
                view.residual_spectrum.frequency, view.threshold, snr_threshold=snr
            )
        else:
            self.clear_threshold()
        self.set_candidate(view.candidate)
        self.set_marker_states(marker_states(view.rows, selected))

    def request_add_at_cursor(self, exact: bool = False) -> bool:
        """Emit :attr:`add_requested` for the pick cursor; False when there is none.

        The command registry's "Add peak at cursor" can be wired here, so the A and
        Insert keys and the menu item go through one path.
        """
        if not self._pick_enabled or self._cursor_freq is None:
            return False
        self.add_requested.emit(self._cursor_freq, bool(exact))
        return True

    @staticmethod
    def _as_xy(data: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(frequency, values)`` of a spectrum, a periodogram or a pair."""
        if isinstance(data, Periodogram):
            frequency, values = data.frequency, data.power
        elif hasattr(data, "frequency") and hasattr(data, "amplitude"):
            frequency, values = data.frequency, data.amplitude
        else:
            frequency, values = data
        return (
            np.asarray(frequency, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )

    @staticmethod
    def _as_periodogram(data: Any) -> Periodogram:
        """A pre-whitening amplitude spectrum wrapped as a :class:`Periodogram`."""
        if isinstance(data, Periodogram):
            return data
        frequency, amplitude = SpectrumView._as_xy(data)
        return Periodogram.from_spectrum(
            method=PREWHITEN_METHOD,
            backend=str(getattr(data, "backend", "")),
            frequency=frequency,
            power=amplitude,
            objective_sense="max",
            n_samples=int(getattr(data, "n_samples", 0)),
            baseline=float(getattr(data, "baseline", 0.0)),
        )

    def set_overlay(self, frequency: np.ndarray, values: np.ndarray) -> None:
        """Add a second trace on the same grid (the pre-whitened residual spectrum)."""
        self._overlay_xy = (
            np.asarray(frequency, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
        # The local-maximum cache is keyed by array identity, which a freed array can
        # hand on to its replacement: drop it whenever the trace is replaced.
        self._extrema_key = None
        self._show_residual.setVisible(True)
        self._show_data.setVisible(True)
        self._redraw_overlay()
        self._redraw_candidate()

    def clear_overlay(self) -> None:
        """Remove the second trace and hide its toggle."""
        self._overlay_xy = None
        self._extrema_key = None
        self._show_residual.setVisible(False)
        self._overlay.setVisible(False)
        self._overlay_trace.clear()
        self._sync_data_toggle()
        self._sync_legend()

    def set_window(
        self, frequency: np.ndarray, amplitude: np.ndarray, *, scale: float = 1.0
    ) -> None:
        """Provide the sampling's spectral window ``|W(f)|``, scaled by ``scale``.

        ``|W|`` is dimensionless (1 at zero frequency); ``scale`` is normally the
        tallest amplitude of the displayed spectrum, which is how Period04 overlays the
        two. The trace stays hidden until the ``window`` toggle is checked.
        """
        factor = float(scale) if np.isfinite(scale) and scale > 0.0 else 1.0
        self._window_xy = (
            np.asarray(frequency, dtype=np.float64),
            np.asarray(amplitude, dtype=np.float64) * factor,
        )
        self._show_window.setVisible(True)
        self._show_data.setVisible(True)
        self._redraw_window()

    def clear_window(self) -> None:
        """Remove the spectral-window trace and hide its toggle."""
        self._window_xy = None
        self._show_window.setVisible(False)
        self._window_curve.setVisible(False)
        self._window_trace.clear()
        self._sync_data_toggle()
        self._sync_legend()

    def _sync_data_toggle(self) -> None:
        """Offer the data toggle only while an overlay could be read underneath it."""
        if self._overlay_xy is None and self._window_xy is None:
            self._show_data.setVisible(False)
            self._show_data.setChecked(True)  # never leave the plot empty

    def set_peaks(self, peaks: list[Peak]) -> None:
        """Overlay the significant peaks as markers."""
        self._peaks = list(peaks)
        self._redraw_markers()
        self._update_accessible()

    def set_selected_period(self, period: float) -> None:
        """Highlight ``period`` with the shaded band (no signal echo).

        Outside pick mode the keyboard cursor follows the selection, so the arrow keys
        step on from whatever was last selected (in the table or on the plot).
        """
        if self._pg is None or not np.isfinite(period) or period <= 0.0:
            return
        self._sel_period = period
        if not self._pick_enabled:
            self._cursor_freq = 1.0 / period
        self._sync_band()
        self._update_band()
        self._redraw_markers()  # re-place the selected-peak halo
        self._place_cursor()
        self._update_accessible()

    def _sync_band(self) -> None:
        """The band is shown iff something is selected *and* peaks are being marked.

        It shades a peak, so it belongs to the peak layer: leaving it behind when the
        markers are hidden would keep the one piece of clutter the toggle is usually
        turned off to be rid of. The selection itself survives, band and all, and
        comes back with the markers.
        """
        self._sel_band.setVisible(
            self._sel_period is not None and self._show_peaks.isChecked()
        )

    def clear(self) -> None:
        """Clear the spectrum, peaks, and selection (e.g. when a new curve loads).

        The axis choice is deliberately kept: switching sources clears the view first,
        and the next result of the same kind should open on the axis the user chose.
        """
        self._pg = None
        self._peaks = []
        self._sel_period = None
        self._reset_pick_state()
        self._data_trace.clear()
        self._markers.clear()
        self._sync_band()
        self._hover_text.setVisible(False)
        self._hide_crosshair()
        self.clear_overlay()
        self.clear_window()
        self.set_placeholder(None)
        self._update_accessible()

    def set_placeholder(self, text: str | None) -> None:
        """Show ``text`` inside the plot in place of a spectrum (``None`` removes it).

        For a spectrum that could not be computed, e.g. a failed pre-whitening
        preview ("Spectrum unavailable: …"). The text is also the plot's accessible
        description; the next spectrum shown (or :meth:`clear`) removes it.
        """
        text = (text or "").strip()
        self._placeholder_text = text
        self._placeholder.setText(text)
        self._placeholder.setVisible(bool(text))
        if text:
            self._place_placeholder()
            self._placeholder.raise_()
        self._update_accessible()

    def placeholder(self) -> str:
        """The text shown in place of a spectrum ("" when none)."""
        return self._placeholder_text

    def _place_placeholder(self) -> None:
        """Centre the placeholder over the plot, inset from its edges."""
        margin = 24
        self._placeholder.setGeometry(
            self._plot.rect().adjusted(margin, margin, -margin, -margin)
        )

    def _style_placeholder(self) -> None:
        self._placeholder.setStyleSheet(
            f"QLabel#spectrum_placeholder {{ color: {self._pal.plot_fg}; "
            "background: transparent; }"
        )

    def clear_selection(self) -> None:
        """Hide the selection band (e.g. a compute finished with zero peaks).

        Unlike :meth:`clear`, the spectrum curve/peaks themselves are left alone.
        """
        self._sel_period = None
        self._sync_band()
        self._redraw_markers()
        self._place_cursor()
        self._update_accessible()

    # -- drawing -----------------------------------------------------------------
    def _curve_xy(self) -> tuple[np.ndarray, np.ndarray]:
        assert self._pg is not None
        if self._x_mode == "period":
            return self._pg.period[::-1], self._pg.power[::-1]
        return self._pg.frequency, self._pg.power

    def _redraw_curve(self) -> None:
        if self._pg is None:
            self._data_trace.clear()
            return
        x, y = self._curve_xy()
        self._data_trace.load(x, y)
        self._curve.setVisible(self._show_data.isChecked())
        self._load_trace(self._overlay_trace, self._overlay_xy, self._show_residual)
        self._load_trace(self._window_trace, self._window_xy, self._show_window)
        self._load_trace(self._threshold_trace, self._threshold_xy, None)
        self._extrema_key = None  # the x orientation may have changed
        self._apply_log()  # sets the log mode, then draws every trace
        self._sync_legend()

    def _trace_xy(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """An auxiliary trace re-oriented for the current x mode."""
        frequency, values = data
        if self._x_mode == "period":
            with np.errstate(divide="ignore"):
                return (1.0 / frequency)[::-1], values[::-1]
        return frequency, values

    def _load_trace(
        self,
        trace: _Trace,
        data: tuple[np.ndarray, np.ndarray] | None,
        toggle: QtWidgets.QCheckBox | None,
    ) -> None:
        """Load an auxiliary trace for the current x mode and show it, or hide it."""
        if data is None or (toggle is not None and not toggle.isChecked()):
            trace.item.setVisible(False)
            return
        x, y = self._trace_xy(data)
        trace.load(x, y)
        trace.item.setVisible(True)

    def _redraw_overlay(self) -> None:
        """Re-place the second trace for the current x mode, or hide it."""
        self._load_trace(self._overlay_trace, self._overlay_xy, self._show_residual)
        self._draw_traces()
        self._sync_legend()

    def _redraw_window(self) -> None:
        """Re-place the spectral-window trace for the current x mode, or hide it."""
        self._load_trace(self._window_trace, self._window_xy, self._show_window)
        self._draw_traces()
        self._sync_legend()

    def _draw_traces(self) -> None:
        """Hand pyqtgraph each visible trace, decimated for the current view.

        While the x-axis auto-ranges (the default view) every trace is drawn end to
        end; once the user zooms or pans, the visible range plus one view width on each
        side is drawn, so a pan does not run off the decimated data before the next
        redraw catches up.
        """
        box = self._plot.getPlotItem().vb
        log_x = self._logx.isChecked()
        pixels = max(int(box.width()), _MIN_PLOT_PIXELS)
        lo: float | None = None
        hi: float | None = None
        n_bins = _BINS_PER_PIXEL * pixels
        if not box.autoRangeEnabled()[0]:
            (x0, x1), _ = box.viewRange()
            width = x1 - x0
            lo, hi = x0 - width, x1 + width
            n_bins *= 3
        traces = (
            self._data_trace,
            self._overlay_trace,
            self._window_trace,
            self._threshold_trace,
        )
        for trace in traces:
            if trace.item.isVisible() and not trace.empty:
                trace.draw(log_x, lo, hi, n_bins)

    def _sync_legend(self) -> None:
        """Name the traces once the residual or window can be told apart from data."""
        self._legend.clear()
        entries = [
            (self._curve, "data", self._show_data),
            (self._overlay, "residual", self._show_residual),
            (self._window_curve, "window", self._show_window),
        ]
        shown = [
            (item, name)
            for item, name, toggle in entries
            if toggle.isVisibleTo(self) and toggle.isChecked()
        ]
        if self._threshold_xy is not None:
            shown.append((self._threshold_curve, "S/N limit"))
        if not (
            self._overlay_xy is not None
            or self._window_xy is not None
            or self._threshold_xy is not None
        ):
            shown = []
        for item, name in shown:
            self._legend.addItem(item, name)
        self._legend.setVisible(bool(shown))

    def _selected_index(self) -> int | None:
        """Index into :attr:`_peaks` matching the current selection, if any."""
        if self._sel_period is None or not self._peaks:
            return None
        periods = np.array([p.period for p in self._peaks])
        idx = int(np.argmin(np.abs(periods - self._sel_period)))
        if np.isclose(periods[idx], self._sel_period, rtol=1e-6, atol=0.0):
            return idx
        return None

    def _redraw_markers(self) -> None:
        # The hover label is anchored to a marker in plot coordinates, so any redraw
        # invalidates it: hiding the peaks would stranded it over an empty plot, a new
        # result would leave the old one's numbers floating, and a log/axis switch
        # would park it at coordinates that no longer mean anything. It reappears on
        # the next hover event, which fires on every mouse move over a marker.
        self._hover_text.setVisible(False)
        if self._pg is None or not self._show_peaks.isChecked():
            self._markers.clear()
            return
        if self._marker_states is not None:
            self._markers.setData(self._state_spots())
            return
        sel_idx = self._selected_index()
        spots: list[dict[str, Any]] = []
        if self._peaks:  # a soft glow halo behind the rank-1 peak
            best = self._peaks[0]
            bx = best.frequency if self._x_mode == "frequency" else best.period
            spots.append(
                {
                    "pos": (self._to_plot_x(bx), self._to_plot_y(best.power)),
                    "size": 30,
                    "symbol": "o",
                    "brush": pg.mkBrush(self._pal.qcolor(self._pal.best_peak, 55)),
                    "pen": None,
                    "data": None,
                }
            )
        if sel_idx is not None and sel_idx != 0:  # halo for the selected peak
            sel = self._peaks[sel_idx]
            sx = sel.frequency if self._x_mode == "frequency" else sel.period
            spots.append(
                {
                    "pos": (self._to_plot_x(sx), self._to_plot_y(sel.power)),
                    "size": 26,
                    "symbol": "o",
                    "brush": pg.mkBrush(self._pal.qcolor(self._pal.accent, 60)),
                    "pen": None,
                    "data": None,
                }
            )
        for i, peak in enumerate(self._peaks):
            x = peak.frequency if self._x_mode == "frequency" else peak.period
            is_best = i == 0
            is_selected = i == sel_idx
            color = self._pal.best_peak if is_best else self._pal.peak
            size = 15 if is_best else 9
            pen = pg.mkPen(self._pal.plot_bg, width=stroke_width(self))
            if is_selected:
                size += 4
                pen = pg.mkPen(
                    self._pal.qcolor(self._pal.accent), width=stroke_width(self, 2.0)
                )
            spots.append(
                {
                    "pos": (self._to_plot_x(x), self._to_plot_y(peak.power)),
                    "size": size,
                    "symbol": "star" if is_best else "o",
                    "brush": pg.mkBrush(color),
                    "pen": pen,
                    "data": i,
                }
            )
        self._markers.setData(spots)

    def _state_spots(self) -> list[dict[str, Any]]:
        """Scatter spots of the component markers (shape = state, see the notes)."""
        assert self._marker_states is not None
        spots: list[dict[str, Any]] = []
        outline = pg.mkPen(self._pal.plot_bg, width=stroke_width(self))
        ring = pg.mkPen(self._pal.qcolor(self._pal.accent), width=stroke_width(self, 2))
        hollow = pg.mkPen(self._pal.peak, width=stroke_width(self, 2))
        for state in self._marker_states:
            frequency = float(state.frequency)
            if not (math.isfinite(frequency) and frequency > 0.0):
                continue
            pos = (
                self._to_plot_x(self._x_for_frequency(frequency)),
                self._to_plot_y(self._height_at(frequency)),
            )
            if state.selected:  # the selection halo, behind the marker
                spots.append(
                    {
                        "pos": pos,
                        "size": 26,
                        "symbol": "o",
                        "brush": pg.mkBrush(self._pal.qcolor(self._pal.accent, 60)),
                        "pen": None,
                        "data": None,
                    }
                )
            inactive = state.shape == "inactive"
            pen = ring if state.selected else (hollow if inactive else outline)
            spots.append(
                {
                    "pos": pos,
                    "size": 15 if state.selected else 11,
                    "symbol": _STATE_SYMBOLS.get(str(state.shape), "o"),
                    "brush": None if inactive else pg.mkBrush(self._pal.peak),
                    "pen": pen,
                    "data": ("component", int(state.id)),
                }
            )
        return spots

    def _height_at(self, frequency: float) -> float:
        """The data spectrum at ``frequency`` (nearest sample): where markers stand."""
        if self._pg is None or self._pg.frequency.size == 0:
            return 0.0
        value = float(self._pg.power[_nearest_index(self._pg.frequency, frequency)])
        return value if math.isfinite(value) else 0.0

    def _redraw_candidate(self) -> None:
        """Place the "next" triangle on the residual spectrum at the candidate."""
        frequency = self.candidate_frequency()
        if self._pg is None or frequency is None:
            self._candidate_item.clear()
            self._candidate_label.setVisible(False)
            return
        height = self._pick_value_at(frequency)
        if height is None:
            amplitude = getattr(self._candidate, "amplitude", float("nan"))
            height = float(amplitude) if math.isfinite(amplitude) else 0.0
        x = self._to_plot_x(self._x_for_frequency(frequency))
        y = self._to_plot_y(height)
        self._candidate_item.setData(
            [
                {
                    "pos": (x, y),
                    "size": 22,
                    "symbol": _candidate_symbol(),
                    "brush": None,
                    "pen": pg.mkPen(self._pal.plot_fg, width=stroke_width(self, 2)),
                    "data": None,
                }
            ]
        )
        self._candidate_label.setPos(x, y)
        self._candidate_label.setVisible(True)

    def _y_quantity(self) -> str:
        """The plotted y quantity, in the words the readout and CSV header use."""
        if self._pg is not None and self._pg.method == PREWHITEN_METHOD:
            return "amplitude"
        if self._pg is not None and self._pg.objective_sense == "min":
            return "dispersion"
        return "power"

    def _y_unit_suffix(self) -> str:
        """`` mag`` / `` flux`` after a pre-whitening amplitude, else nothing."""
        if self._pg is not None and self._pg.method == PREWHITEN_METHOD:
            return f" {self._amplitude_unit}" if self._amplitude_unit else ""
        return ""

    def _update_y_label(self) -> None:
        if self._pg is None:
            return
        style = plot_label_style(self._pal)
        if self._pg.method == PREWHITEN_METHOD:
            unit = self._amplitude_unit
            text = f"Amplitude ({unit})" if unit else "Amplitude"
        elif self._pg.objective_sense == "min":
            text = f"{self._pg.method} {_MIN_OBJECTIVE_LABEL}"
        else:
            text = f"{self._pg.method} power"
        self._plot.setLabel("left", text, **style)

    @staticmethod
    def _axis_family_of(pg_result: Periodogram) -> str:
        if pg_result.method in _BOX_METHODS:
            return "box"
        if pg_result.method == PREWHITEN_METHOD:
            return "prewhiten"
        return "spectrum"

    def _default_axis_for(self, pg_result: Periodogram) -> None:
        # Box/transit searches read most naturally on a *log period* axis (their period
        # grids are log-spaced), so default BLS/TLS to period + log-x. The default is
        # applied only when the kind of result changes: the next source, a re-run or a
        # cache hit of the same kind keeps whatever axis the user picked for it.
        family = self._axis_family_of(pg_result)
        if family == self._axis_family:
            return
        self._axis_family = family
        is_box = family == "box"
        want = "period" if is_box else "frequency"
        if self._xaxis_combo.currentText() != want:
            self._xaxis_combo.blockSignals(True)
            self._xaxis_combo.setCurrentText(want)
            self._xaxis_combo.blockSignals(False)
        self._x_mode = want
        self._logx.blockSignals(True)
        self._logx.setChecked(is_box)
        self._logx.blockSignals(False)
        self._plot.setLabel("bottom", self._x_label(), **plot_label_style(self._pal))

    # -- axis / scale ------------------------------------------------------------
    def _x_label(self) -> str:
        return "Period (days)" if self._x_mode == "period" else "Frequency (cycles/day)"

    def _on_xaxis_changed(self, mode: str) -> None:
        self._x_mode = mode
        # A period axis reads best on a log scale — on a uniform-frequency grid the
        # samples are sparse in period at long P, so a linear period axis looks jagged.
        # A frequency axis reads best linear.
        self._logx.blockSignals(True)
        self._logx.setChecked(mode == "period")
        self._logx.blockSignals(False)
        self._plot.setLabel("bottom", self._x_label(), **plot_label_style(self._pal))
        self._redraw_curve()  # applies log mode + re-places markers
        if self._sel_period is not None:
            self.set_selected_period(self._sel_period)
        self._autorange()

    def _apply_log(self) -> None:
        # Toggling a log mode makes pyqtgraph re-enable auto-ranging, so the traces
        # are then drawn end to end in the new coordinates.
        self._plot.setLogMode(x=self._logx.isChecked(), y=self._logy.isChecked())
        self._draw_traces()
        self._update_band()
        self._redraw_markers()  # markers live in plot coords; re-place on log change
        self._redraw_candidate()
        self._place_cursor()

    def _on_peaks_toggled(self, _checked: bool) -> None:
        self._redraw_markers()
        self._sync_band()

    def _autorange(self) -> None:
        # Fit the view to the full data extent: re-enable auto-ranging, hand pyqtgraph
        # every trace end to end, then range on that.
        self._plot.enableAutoRange()
        self._draw_traces()
        self._plot.autoRange()

    # -- selection band ----------------------------------------------------------
    def _update_band(self) -> None:
        if self._sel_period is None:
            return
        center = self._to_plot_x(self._x_for_period(self._sel_period))
        (x0, x1), _ = self._plot.getPlotItem().vb.viewRange()
        half = max(abs(x1 - x0) * 0.01, 1e-12)
        self._suppress_band = True
        self._sel_band.setRegion((center - half, center + half))
        self._suppress_band = False

    def _on_band_moved(self) -> None:
        if self._suppress_band or self._pg is None:
            return
        region = self._sel_band.getRegion()
        data_x = self._from_plot_x(0.5 * (region[0] + region[1]))
        period = self._snap_period(self._period_for_x(data_x))
        if np.isfinite(period) and period > 0.0:
            self.set_selected_period(period)
            self.period_selected.emit(period)

    def _on_xrange_changed(self, *_: Any) -> None:
        self._update_band()
        self._redraw_timer.start()  # re-decimate for the new range (coalesced)

    def _on_scene_clicked(self, event: Any) -> None:
        """Restore the default view on a double-click (same as the Reset button)."""
        if event.double():
            self._autorange()
            event.accept()

    # -- conversions -------------------------------------------------------------
    def _x_for_period(self, period: float) -> float:
        if self._x_mode == "period":
            return period
        return 1.0 / period if period > 0.0 else 0.0

    def _x_for_frequency(self, frequency: float) -> float:
        if self._x_mode == "frequency":
            return frequency
        return 1.0 / frequency if frequency > 0.0 else float("inf")

    def _frequency_for_x(self, x: float) -> float:
        if self._x_mode == "frequency":
            return x
        return 1.0 / x if x > 0.0 else float("inf")

    def _plot_x_of(self, frequency: FloatArray) -> FloatArray:
        """Plot x coordinates of an array of frequencies (NaN where undrawable)."""
        f = np.asarray(frequency, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = f if self._x_mode == "frequency" else 1.0 / f
            if self._logx.isChecked():
                x = np.where(x > 0.0, np.log10(np.where(x > 0.0, x, 1.0)), np.nan)
        return np.asarray(x, dtype=np.float64)

    def _period_for_x(self, x: float) -> float:
        if self._x_mode == "period":
            return x
        return 1.0 / x if x > 0.0 else float("inf")

    def _to_plot_x(self, data_x: float) -> float:
        if self._logx.isChecked() and data_x > 0.0:
            return float(np.log10(data_x))
        return float(data_x)

    def _from_plot_x(self, plot_x: float) -> float:
        return float(10.0**plot_x) if self._logx.isChecked() else float(plot_x)

    def _to_plot_y(self, data_y: float) -> float:
        # pyqtgraph log-transforms the curve but not the ScatterPlotItem markers, so we
        # place markers in the plot's coordinate space ourselves (log10 in log mode).
        if self._logy.isChecked() and data_y > 0.0:
            return float(np.log10(data_y))
        return float(data_y)

    def _snap_period(self, raw_period: float) -> float:
        """Snap a raw period to the nearest grid sample's period."""
        assert self._pg is not None
        if not np.isfinite(raw_period) or raw_period <= 0.0:
            return raw_period
        freq = 1.0 / raw_period
        idx = int(np.argmin(np.abs(self._pg.frequency - freq)))
        return float(self._pg.period[idx])

    # -- interaction -------------------------------------------------------------
    def _on_peak_clicked(self, _item: Any, points: Any, *_: Any) -> None:
        pts = list(points) if points is not None else []
        if not pts:
            return
        self._click_timer.stop()  # a marker hit wins over a pending plain click
        self._pending_click = None
        self._activate_marker(pts[0].data())

    def _activate_marker(self, data: Any) -> None:
        """Act on a marker: select its peak, or report its component."""
        if isinstance(data, tuple) and len(data) == 2 and data[0] == "component":
            self._activate_component(int(data[1]))
            return
        if not isinstance(data, int | np.integer) or not 0 <= data < len(self._peaks):
            return
        self._select_peak(self._peaks[int(data)])

    def _select_peak(self, peak: Peak) -> None:
        """Select a listed peak (fold at it) and hand the whole peak on."""
        self.set_selected_period(peak.period)
        # The whole peak goes out, so a BLS/TLS epoch is folded on rather than
        # replaced by the first timestamp. A view wired only to period_selected (the
        # pre-1.3 interface) still gets the click.
        if is_connected(self, self.peak_selected):
            self.peak_selected.emit(peak)
        else:
            self.period_selected.emit(peak.period)

    def _activate_component(self, component_id: int) -> None:
        """Report a component marker; standing alone, fold at its frequency instead."""
        if is_connected(self, self.component_clicked):
            self.component_clicked.emit(component_id)
            return
        for state in self._marker_states or ():
            if int(state.id) == component_id and float(state.frequency) > 0.0:
                self._select_frequency(float(state.frequency))
                return

    def _select_frequency(self, frequency: float) -> None:
        """Fold at ``frequency`` (snapped to the grid); a listed peak goes out whole."""
        if self._pg is None or not (math.isfinite(frequency) and frequency > 0.0):
            return
        peak = self._peak_at(frequency)
        if peak is not None:
            self._select_peak(peak)
            return
        period = self._snap_period(1.0 / frequency)
        if not (math.isfinite(period) and period > 0.0):
            return
        self.set_selected_period(period)
        self.period_selected.emit(period)

    def _peak_at(self, frequency: float) -> Peak | None:
        """The listed peak at ``frequency``: within half a grid step (it may be refined
        off the grid, while a snapped click lands on the nearest sample)."""
        if self._pg is None or not self._peaks or self._pg.frequency.size == 0:
            return None
        grid = self._pg.frequency
        idx = _nearest_index(grid, frequency)
        lo, hi = max(idx - 1, 0), min(idx + 1, grid.size - 1)
        step = float(grid[hi] - grid[lo]) / max(hi - lo, 1)
        tolerance = max(0.5 * step, 1e-9 * frequency)
        best = min(self._peaks, key=lambda p: abs(p.frequency - frequency))
        return best if abs(best.frequency - frequency) <= tolerance else None

    def _on_peak_hovered(self, _item: Any, points: Any) -> None:
        pts = list(points) if points is not None else []
        if not pts:
            self._hover_text.setVisible(False)
            return
        data = pts[0].data()
        if isinstance(data, tuple) and len(data) == 2 and data[0] == "component":
            label = self._component_label(int(data[1]))
            if label is None:
                self._hover_text.setVisible(False)
                return
            text, frequency = label
            self._show_hover(
                text,
                self._to_plot_x(self._x_for_frequency(frequency)),
                self._to_plot_y(self._height_at(frequency)),
            )
            return
        idx = data
        if idx is None or idx >= len(self._peaks):
            # A glow halo carries no index; keeping the previous peak's label up while
            # the cursor sits on a different one would simply be wrong.
            self._hover_text.setVisible(False)
            return
        peak = self._peaks[int(idx)]
        data_x = peak.frequency if self._x_mode == "frequency" else peak.period
        self._show_hover(
            f"#{peak.rank}  P={peak.period:.6g} d\n{self._peak_detail(peak)}",
            self._to_plot_x(data_x),
            self._to_plot_y(peak.power),
        )

    def _component_label(self, component_id: int) -> tuple[str, float] | None:
        """Hover text of a component marker and its frequency."""
        for state in self._marker_states or ():
            if int(state.id) != component_id:
                continue
            frequency = float(state.frequency)
            word = _STATE_WORDS.get(str(state.shape), str(state.shape))
            text = f"F{component_id}  f={frequency:.6g} c/d  P={1.0 / frequency:.6g} d"
            for peak in self._peaks:  # the fitted amplitude, when the peaks carry it
                if math.isclose(peak.frequency, frequency, rel_tol=1e-9):
                    text += f"\n{self._peak_detail(peak)}"
                    break
            return f"{text}\n{word}", frequency
        return None

    def _show_hover(self, text: str, x: float, y: float) -> None:
        """Show the hover label at plot coordinates ``(x, y)``, kept inside the view."""
        self._hover_text.setText(text)
        # Hang the label under a marker near the top of the view (the best peak sits
        # there after auto-ranging, and a label above it was clipped to its last
        # line), and grow it leftwards near the right edge.
        (x0, x1), (y0, y1) = self._plot.getPlotItem().vb.viewRange()
        near_top = y1 > y0 and (y1 - y) < _HOVER_FLIP_TOP * (y1 - y0)
        near_right = x1 > x0 and (x1 - x) < _HOVER_FLIP_RIGHT * (x1 - x0)
        self._hover_text.setAnchor(
            (1.0 if near_right else 0.0, 0.0 if near_top else 1.0)
        )
        self._hover_text.setPos(x, y)
        self._hover_text.setVisible(True)

    @staticmethod
    def _peak_detail(peak: Peak) -> str:
        """The hover line under a marker: fitted amplitude and S/N when present.

        A pre-whitening marker is drawn at the height of the *spectrum*, so its fitted
        amplitude — which can differ once components are correlated — has to be read
        out here rather than inferred from where the marker sits.
        """
        amplitude = peak.extra.get("amplitude")
        if amplitude is None:
            return f"power={peak.power:.4g}"
        detail = f"A={amplitude:.4g}"
        snr = peak.extra.get("snr")
        if snr is not None and np.isfinite(snr):
            detail += f"   S/N={snr:.1f}"
        if peak.extra.get("blended"):
            detail += "\nblended — amplitude disagrees with the spectrum"
        return detail

    def _on_mouse_moved(self, event: Any) -> None:
        if self._pg is None:
            return
        pos = event[0]
        view = self._plot.getPlotItem().vb
        if not self._plot.sceneBoundingRect().contains(pos):
            self._hide_crosshair()
            return
        point = view.mapSceneToView(pos)
        x, y = self._mouse_xy(point)
        self._vline.setPos(point.x())
        self._hline.setPos(point.y())
        self._vline.setVisible(True)
        self._hline.setVisible(True)
        freq = 1.0 / x if (self._x_mode == "period" and x > 0) else x
        period = x if self._x_mode == "period" else (1.0 / x if x > 0 else float("inf"))
        self._readout.setText(
            f"f = {freq:.6g} c/d    P = {period:.6g} d    "
            f"{self._y_quantity()} = {y:.4g}{self._y_unit_suffix()}"
        )

    def _hide_crosshair(self) -> None:
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._show_cursor_readout()

    def _show_cursor_readout(self) -> None:
        """Read out the pick cursor (keyboard users get its numbers here), or "—"."""
        frequency = self._cursor_freq
        if self._pg is None or frequency is None or self._vline.isVisible():
            if not self._vline.isVisible():
                self._readout.setText("—")
            return
        value = self._pick_value_at(frequency)
        shown = "–" if value is None else f"{value:.4g}{self._y_unit_suffix()}"
        self._readout.setText(
            f"cursor  f = {frequency:.6g} c/d    P = {1.0 / frequency:.6g} d    "
            f"{self._y_quantity()} = {shown}"
        )

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        kind = event.type()
        if watched is self._plot and kind == QtCore.QEvent.Type.Resize:
            self._place_placeholder()
        # The scene reports moves only while the cursor is over it, so without this the
        # crosshair and readout stayed frozen where the cursor left the plot.
        if kind == QtCore.QEvent.Type.Leave:
            self._hide_crosshair()
        elif watched is self._plot and isinstance(event, QtGui.QKeyEvent):
            if kind == QtCore.QEvent.Type.ShortcutOverride and self._handles_key(event):
                event.accept()  # deliver as a key press: the plot owns these keys
                return True
            if kind == QtCore.QEvent.Type.KeyPress and self._on_key(event):
                return True
        elif isinstance(event, QtGui.QMouseEvent):
            self._track_press(kind, event)
        return super().eventFilter(watched, event)

    # -- clicks ---------------------------------------------------------------------
    def _track_press(self, kind: QtCore.QEvent.Type, event: QtGui.QMouseEvent) -> None:
        """Note when a press turns into a drag, so its release is not a click.

        A new press also cancels a click still waiting for the double-click interval:
        either it is the start of a drag, or a new click that supersedes it.
        """
        if kind == QtCore.QEvent.Type.MouseButtonPress:
            self._press_origin = event.position()
            self._press_dragged = False
            self._click_timer.stop()
            self._pending_click = None
        elif (
            kind == QtCore.QEvent.Type.MouseMove
            and self._press_origin is not None
            and event.buttons() != Qt.MouseButton.NoButton
        ):
            moved = (event.position() - self._press_origin).manhattanLength()
            if moved >= QtWidgets.QApplication.startDragDistance():
                self._press_dragged = True
                self._click_timer.stop()
                self._pending_click = None

    def _on_plot_clicked(self, event: Any) -> None:
        """Single clicks: select (periodograms) or pick (pick mode); see the notes.

        Runs beside :meth:`_on_scene_clicked` (the double-click reset). Clicks a marker
        already took, middle clicks, clicks outside the data area and the release of
        a drag are ignored; a right click opens the menu at once; a left click waits
        for the double-click interval.
        """
        if event.double():
            self._click_timer.stop()  # the reset runs alone
            self._pending_click = None
            return
        if self._pg is None or event.isAccepted():
            return
        button = event.button()
        if button not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        scene_pos = QtCore.QPointF(event.scenePos())
        box = self._plot.getPlotItem().vb
        if not box.sceneBoundingRect().contains(scene_pos):
            return
        if button == Qt.MouseButton.RightButton:
            self._click_timer.stop()
            self._pending_click = None
            screen = event.screenPos()
            self._popup_menu_at(
                scene_pos, QtCore.QPoint(int(screen.x()), int(screen.y()))
            )
            event.accept()
            return
        if self._press_dragged:
            return
        mods = event.modifiers()
        if self._pick_enabled and mods & Qt.KeyboardModifier.ShiftModifier:
            # Adding is deliberate (a modifier is held), so it need not wait.
            self._click_timer.stop()
            self._pending_click = None
            exact = bool(mods & Qt.KeyboardModifier.ControlModifier)
            frequency = self._pick_frequency(scene_pos, snap=not exact)
            if frequency is not None:
                self._move_cursor(frequency)
                self.add_requested.emit(frequency, exact)
            event.accept()
            return
        self._pending_click = (scene_pos, mods)
        self._click_timer.start()

    def _on_click_timeout(self) -> None:
        """The deferred single click: no double click followed it."""
        pending, self._pending_click = self._pending_click, None
        if pending is None or self._pg is None:
            return
        scene_pos, mods = pending
        marker = self._nearest_marker(scene_pos)
        if marker is not None:
            self._activate_marker(marker)
            return
        snap = not bool(mods & Qt.KeyboardModifier.AltModifier)
        frequency = self._pick_frequency(scene_pos, snap=snap)
        if frequency is None:
            return
        if self._pick_enabled:
            self._move_cursor(frequency)
        else:
            self._select_frequency(frequency)

    def _nearest_marker(self, scene_pos: QtCore.QPointF) -> Any:
        """Data of the marker within :data:`SNAP_PX` of a scene point, if any."""
        box = self._plot.getPlotItem().vb
        best: Any = None
        best_distance = SNAP_PX
        for spot in self._markers.points():
            data = spot.data()
            if data is None:
                continue
            where = box.mapViewToScene(spot.pos())
            distance = math.hypot(where.x() - scene_pos.x(), where.y() - scene_pos.y())
            if distance <= best_distance:
                best, best_distance = data, distance
        return best

    def _pick_frequency(
        self, scene_pos: QtCore.QPointF, *, snap: bool
    ) -> float | None:
        """Frequency picked at a scene point: snapped to a local maximum, or exact.

        Snapping takes the tallest local maximum of the pick trace (the residual
        spectrum in pick mode, else the spectrum) within :data:`SNAP_PX` of the click
        horizontally; with nothing in reach, the nearest grid sample.
        """
        box = self._plot.getPlotItem().vb
        plot_x = float(box.mapSceneToView(scene_pos).x())
        frequency = self._frequency_for_x(self._from_plot_x(plot_x))
        if not (math.isfinite(frequency) and frequency > 0.0):
            return None
        if not snap:
            return frequency
        arrays = self._pick_arrays()
        if arrays is None:
            return frequency
        grid, values, extrema = arrays
        (x0, x1), _ = box.viewRange()
        width = float(box.width())
        if extrema.size and x1 > x0 and width > 0.0:
            reach = SNAP_PX * (x1 - x0) / width
            near = extrema[np.abs(self._plot_x_of(grid[extrema]) - plot_x) <= reach]
            if near.size:
                best = near[int(np.argmax(self._score(values[near])))]
                return float(grid[best])
        return float(grid[_nearest_index(grid, frequency)])

    def _score(self, values: FloatArray) -> FloatArray:
        """Values oriented so that larger is better (minimise methods flipped)."""
        if self._minimising():
            return np.asarray(-values, dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    def _minimising(self) -> bool:
        return (
            not self._pick_or_residual()
            and self._pg is not None
            and self._pg.objective_sense == "min"
        )

    def _pick_or_residual(self) -> bool:
        """Whether the cursor follows the residual spectrum (pick mode with one)."""
        return self._pick_enabled and self._overlay_xy is not None

    def _pick_arrays(self) -> tuple[FloatArray, FloatArray, IntArray] | None:
        """``(frequency, values, local-maximum indices)`` of the trace picks follow.

        The residual spectrum in pick mode (what is left to extract), else the
        displayed spectrum; frequencies ascending. Cached until the trace changes.
        """
        if self._pick_or_residual():
            assert self._overlay_xy is not None
            frequency, values = self._overlay_xy
        elif self._pg is not None:
            frequency, values = self._pg.frequency, self._pg.power
        else:
            return None
        key = (id(frequency), id(values), self._minimising())
        if key != self._extrema_key or self._extrema is None:
            grid = np.asarray(frequency, dtype=np.float64)
            vals = np.asarray(values, dtype=np.float64)
            if grid.size > 1 and not bool(np.all(grid[1:] >= grid[:-1])):
                order = np.argsort(grid, kind="stable")
                grid, vals = grid[order], vals[order]
            extrema = local_maxima(vals, maximise=not self._minimising())
            self._extrema = (grid, vals, extrema)
            self._extrema_key = key
        if self._extrema[0].size == 0:
            return None
        return self._extrema

    def _pick_value_at(self, frequency: float) -> float | None:
        """The pick trace at ``frequency`` (linear interpolation), if defined."""
        arrays = self._pick_arrays()
        if arrays is None:
            return None
        grid, values, _ = arrays
        value = float(np.interp(frequency, grid, values))
        return value if math.isfinite(value) else None

    # -- the pick cursor ------------------------------------------------------------
    def _move_cursor(self, frequency: float, *, reveal: bool = False) -> None:
        """Move the cursor by a user gesture and announce it."""
        self._cursor_freq = float(frequency)
        self._place_cursor()
        if reveal:
            self._reveal(frequency)
        self._show_cursor_readout()
        self._update_accessible()
        self.cursor_changed.emit(float(frequency))

    def _place_cursor(self) -> None:
        """Draw the cursor line; outside pick mode only when it left the selection."""
        frequency = self._cursor_freq
        if self._pg is None or frequency is None:
            self._cursor_line.setVisible(False)
            return
        on_selection = self._sel_period is not None and math.isclose(
            1.0 / frequency, self._sel_period, rel_tol=1e-9
        )
        x = self._to_plot_x(self._x_for_frequency(frequency))
        self._cursor_line.setPos(x)
        self._cursor_line.setVisible(
            math.isfinite(x) and (self._pick_enabled or not on_selection)
        )

    def _reveal(self, frequency: float) -> None:
        """Pan the view (zoom kept) so a cursor moved by the keyboard stays visible."""
        box = self._plot.getPlotItem().vb
        x = self._to_plot_x(self._x_for_frequency(frequency))
        (x0, x1), _ = box.viewRange()
        if not math.isfinite(x) or x0 <= x <= x1:
            return
        half = 0.5 * (x1 - x0)
        box.setXRange(x - half, x + half, padding=0.0)

    # -- keyboard -------------------------------------------------------------------
    def _handles_key(self, event: QtGui.QKeyEvent) -> bool:
        """Whether the focused plot acts on this key itself (no shortcut gets it)."""
        key = Qt.Key(event.key())
        mods = event.modifiers()
        if key not in _NAV_KEYS:
            return False
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            return not mods & ~(
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.ShiftModifier
                | Qt.KeyboardModifier.KeypadModifier
            )
        # Everything else only bare (Ctrl+Return is Auto, Ctrl+0 resets the UI zoom).
        return not mods & ~(
            Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.KeypadModifier
        )

    def _on_key(self, event: QtGui.QKeyEvent) -> bool:
        """Act on a key pressed while the plot has focus; True when handled."""
        if self._pg is None:
            return False
        key = Qt.Key(event.key())
        mods = event.modifiers()
        if key in (Qt.Key.Key_Insert, Qt.Key.Key_A) and self._pick_enabled:
            # The command registry binds these on the spectrum; only standing alone
            # (no such shortcut around) does the plot add by itself.
            if self._shortcut_owns(event):
                return False
            return self.request_add_at_cursor(exact=False)
        if not self._handles_key(event):
            return False
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            step = 1 if key == Qt.Key.Key_Right else -1
            if mods & Qt.KeyboardModifier.ShiftModifier:
                target = self._step_sample(step)
            elif mods & Qt.KeyboardModifier.ControlModifier:
                target = self._step_marker(step)
            else:
                target = self._step_extremum(step)
            if target is not None:
                self._move_cursor(target, reveal=True)
            return True
        if key == Qt.Key.Key_Home:
            target = self._tallest()
            if target is not None:
                self._move_cursor(target, reveal=True)
            return True
        if key == Qt.Key.Key_N:
            candidate = self.candidate_frequency()
            if candidate is not None:
                self._move_cursor(candidate, reveal=True)
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._fold_at_cursor()
            return True
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Minus):
            self._zoom(0.5 if key != Qt.Key.Key_Minus else 2.0)
            return True
        if key == Qt.Key.Key_0:
            self._autorange()
            return True
        if key == Qt.Key.Key_Menu:
            self._popup_menu_at_cursor()
            return True
        return False

    def _shortcut_owns(self, event: QtGui.QKeyEvent) -> bool:
        """Whether a ``QShortcut`` on the plot or around it is bound to this key."""
        pressed = QtGui.QKeySequence(event.keyCombination())
        widget: QtWidgets.QWidget | None = self._plot
        while widget is not None:
            for shortcut in widget.findChildren(
                QtGui.QShortcut, options=Qt.FindChildOption.FindDirectChildrenOnly
            ):
                bound = shortcut.keys()
                if any(
                    seq.matches(pressed) == QtGui.QKeySequence.SequenceMatch.ExactMatch
                    for seq in bound
                ):
                    return True
            widget = widget.parentWidget()
        return False

    def _cursor_origin(self) -> float | None:
        """Where keyboard steps start: the cursor, else the selection."""
        if self._cursor_freq is not None:
            return self._cursor_freq
        if self._sel_period is not None and self._sel_period > 0.0:
            return 1.0 / self._sel_period
        return None

    def _step_in_plot(self, frequencies: FloatArray, step: int) -> float | None:
        """The frequency ``step`` places to the right (+1) / left (-1) on screen."""
        if frequencies.size == 0:
            return None
        x = self._plot_x_of(frequencies)
        ok = np.isfinite(x)
        frequencies, x = frequencies[ok], x[ok]
        if x.size == 0:
            return None
        order = np.argsort(x, kind="stable")
        frequencies, x = frequencies[order], x[order]
        origin = self._cursor_origin()
        if origin is None:  # start at the view's edge facing the step
            (x0, x1), _ = self._plot.getPlotItem().vb.viewRange()
            idx = (
                int(np.searchsorted(x, x0, side="left"))
                if step > 0
                else int(np.searchsorted(x, x1, side="right")) - 1
            )
        else:
            here = float(self._plot_x_of(np.array([origin]))[0])
            tol = 1e-12 * max(1.0, abs(here))
            idx = (
                int(np.searchsorted(x, here + tol, side="right"))
                if step > 0
                else int(np.searchsorted(x, here - tol, side="left")) - 1
            )
        if not 0 <= idx < x.size:
            return None
        return float(frequencies[idx])

    def _step_extremum(self, step: int) -> float | None:
        arrays = self._pick_arrays()
        if arrays is None:
            return None
        grid, _, extrema = arrays
        return self._step_in_plot(grid[extrema], step)

    def _step_marker(self, step: int) -> float | None:
        if self._marker_states is not None:
            marks = [float(s.frequency) for s in self._marker_states]
        else:
            marks = [float(p.frequency) for p in self._peaks]
        return self._step_in_plot(np.asarray(marks, dtype=np.float64), step)

    def _step_sample(self, step: int) -> float | None:
        arrays = self._pick_arrays()
        if arrays is None:
            return None
        return self._step_in_plot(arrays[0], step)

    def _tallest(self) -> float | None:
        arrays = self._pick_arrays()
        if arrays is None:
            return None
        grid, values, extrema = arrays
        if extrema.size == 0:
            return None
        return float(grid[extrema[int(np.argmax(self._score(values[extrema])))]])

    def _fold_at_cursor(self) -> None:
        """Enter: fold at the cursor (a component or listed peak there goes out)."""
        frequency = self._cursor_freq
        if frequency is None:
            return
        for state in self._marker_states or ():
            if math.isclose(float(state.frequency), frequency, rel_tol=1e-9):
                self._activate_component(int(state.id))
                return
        self._select_frequency(frequency)

    def _zoom(self, factor: float) -> None:
        """Zoom the x-axis by ``factor`` about the cursor (else the view's centre)."""
        box = self._plot.getPlotItem().vb
        (x0, x1), _ = box.viewRange()
        centre = 0.5 * (x0 + x1)
        if self._cursor_freq is not None:
            x = self._to_plot_x(self._x_for_frequency(self._cursor_freq))
            if math.isfinite(x) and x0 <= x <= x1:
                centre = x
        half = 0.5 * (x1 - x0) * factor
        box.setXRange(centre - half, centre + half, padding=0.0)

    # -- context menu ---------------------------------------------------------------
    def build_context_menu(
        self, frequency: float, *, exact_frequency: float | None = None
    ) -> QtWidgets.QMenu:
        """The plot's action menu for a peak at ``frequency`` (cycles/day).

        ``exact_frequency`` is where the user actually pointed (defaults to
        ``frequency``): "Add exactly here" and the zoom act there. In pick mode the
        menu offers *Add peak at ... (S/N ≈ ...)*, *Add exactly here*, *Fold here*,
        *Move cursor here*, *Zoom to ±1 c/d* and *Copy frequency*; outside it the add
        items are left out.
        """
        exact = frequency if exact_frequency is None else float(exact_frequency)
        menu = QtWidgets.QMenu(self)
        menu.setObjectName("spectrumContextMenu")
        menu.setAccessibleName("Spectrum actions")

        def add(text: str, tip: str, slot: Any) -> None:
            action = menu.addAction(text)
            action.setStatusTip(tip)
            action.setToolTip(tip)
            action.triggered.connect(slot)

        if self._pick_enabled:
            add(
                f"Add peak at {frequency:.6g} c/d{self._snr_text(frequency)}",
                "Add a component at this peak (its frequency is then refined)",
                lambda: self._add_from_menu(frequency, False),
            )
            add(
                "Add exactly here",
                f"Add a component at exactly {exact:.6g} c/d",
                lambda: self._add_from_menu(exact, True),
            )
            menu.addSeparator()
        add(
            "Fold here",
            f"Fold the light curve at P = {1.0 / frequency:.6g} d",
            lambda: self._select_frequency(frequency),
        )
        add(
            "Move cursor here",
            "Put the pick cursor on this peak",
            lambda: self._move_cursor(frequency),
        )
        add(
            "Zoom to ±1 c/d",
            "Show one cycle per day either side of this frequency",
            lambda: self._zoom_to(exact, 1.0),
        )
        add(
            "Copy frequency",
            f"Copy {frequency:.10g} to the clipboard",
            lambda: self._copy_frequency(frequency),
        )
        menu.addSeparator()
        add("Reset view", "Auto-range the plot back to the full spectrum",
            self._autorange)
        return menu

    def _snr_text(self, frequency: float) -> str:
        """`` (S/N ≈ 7.9)`` from the residual amplitude over the noise, when known."""
        if self._threshold_xy is None or self._snr_threshold is None:
            return ""
        value = self._pick_value_at(frequency)
        grid, limit = self._threshold_xy
        if value is None or grid.size == 0:
            return ""
        noise = float(np.interp(frequency, grid, limit)) / self._snr_threshold
        if not (math.isfinite(noise) and noise > 0.0):
            return ""
        return f" (S/N ≈ {value / noise:.1f})"

    def _add_from_menu(self, frequency: float, exact: bool) -> None:
        self._move_cursor(frequency)
        self.add_requested.emit(float(frequency), bool(exact))

    def _zoom_to(self, frequency: float, half_width: float) -> None:
        """Show ``frequency ± half_width`` cycles/day (clamped to the grid)."""
        if self._pg is None or self._pg.frequency.size == 0:
            return
        f_lo = max(frequency - half_width, float(self._pg.frequency[0]))
        f_hi = min(frequency + half_width, float(self._pg.frequency[-1]))
        if not (f_lo > 0.0 and f_hi > f_lo):
            return
        ends = self._plot_x_of(np.array([f_lo, f_hi]))
        if not bool(np.all(np.isfinite(ends))):
            return
        self._plot.getPlotItem().vb.setXRange(
            float(ends.min()), float(ends.max()), padding=0.02
        )

    @staticmethod
    def _copy_frequency(frequency: float) -> None:
        clipboard = QtGui.QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(f"{frequency:.10g}")

    def _popup_menu_at(self, scene_pos: QtCore.QPointF, screen: QtCore.QPoint) -> None:
        """Build the menu for a scene point and pop it up (never ``exec()``)."""
        exact = self._pick_frequency(scene_pos, snap=False)
        snapped = self._pick_frequency(scene_pos, snap=True)
        if exact is None or snapped is None:
            return
        self._show_menu(self.build_context_menu(snapped, exact_frequency=exact), screen)

    def _popup_menu_at_cursor(self) -> None:
        """The Menu key: the action menu for the cursor, under the cursor line."""
        frequency = self._cursor_origin()
        if frequency is None:
            return
        box = self._plot.getPlotItem().vb
        x = self._to_plot_x(self._x_for_frequency(frequency))
        (x0, x1), (y0, y1) = box.viewRange()
        if not x0 <= x <= x1:
            x = 0.5 * (x0 + x1)
        scene = box.mapViewToScene(QtCore.QPointF(x, 0.5 * (y0 + y1)))
        local = self._plot.mapFromScene(scene)
        viewport = self._plot.viewport()
        screen = (viewport or self._plot).mapToGlobal(local)
        self._show_menu(self.build_context_menu(frequency), screen)

    def _show_menu(self, menu: QtWidgets.QMenu, screen: QtCore.QPoint) -> None:
        if self._context_menu is not None:
            self._context_menu.deleteLater()
        self._context_menu = menu
        menu.popup(screen)

    # -- accessibility --------------------------------------------------------------
    def _sync_gesture_help(self) -> None:
        """Status tip and What's This of the plot for the current mode."""
        text = _GESTURES_PICK if self._pick_enabled else _GESTURES_PERIODOGRAM
        self._plot.setStatusTip(text)
        self._plot.setWhatsThis(text)

    def accessible_summary(self) -> str:
        """One paragraph describing what the plot shows (its accessible description)."""
        if self._pg is None:
            return self._placeholder_text or "No spectrum yet."
        pg_result = self._pg
        quantity = self._y_quantity()
        parts = [f"{pg_result.method} {quantity} versus {self._x_mode}"]
        if pg_result.frequency.size:
            parts.append(
                f"{pg_result.frequency.size} samples from "
                f"{float(pg_result.frequency[0]):.4g} to "
                f"{float(pg_result.frequency[-1]):.4g} cycles/day"
            )
        if self._marker_states is not None:
            active = sum(1 for s in self._marker_states if s.shape != "inactive")
            parts.append(
                f"{len(self._marker_states)} components marked, {active} in use"
            )
        elif self._peaks:
            best = self._peaks[0]
            parts.append(
                f"{len(self._peaks)} peaks marked; best P = {best.period:.6g} d "
                f"(f = {best.frequency:.6g} c/d, {quantity} {best.power:.4g})"
            )
        else:
            parts.append("no peaks marked")
        candidate = self.candidate_frequency()
        if candidate is not None:
            parts.append(f"next candidate f = {candidate:.6g} c/d")
        if self._sel_period is not None:
            parts.append(f"selected P = {self._sel_period:.6g} d")
        if self._cursor_freq is not None:
            parts.append(f"cursor f = {self._cursor_freq:.6g} c/d")
        return "; ".join(parts) + "."

    def _update_accessible(self) -> None:
        self._plot.setAccessibleDescription(self.accessible_summary())

    def _mouse_xy(self, point: Any) -> tuple[float, float]:
        # In log mode the view coordinates are log10 of the data values.
        x = 10.0 ** point.x() if self._logx.isChecked() else point.x()
        y = 10.0 ** point.y() if self._logy.isChecked() else point.y()
        return float(x), float(y)

    # -- export --------------------------------------------------------------
    #: The window's dialogs for the Export… prompts (None: a static file dialog).
    _dialogs: DialogProvider | None = None

    def set_dialog_provider(self, provider: DialogProvider | None) -> None:
        """Ask for export paths through ``provider`` (the window's dialogs)."""
        self._dialogs = provider

    def _export_csv(self) -> None:
        if self._pg is None:
            return
        ask_save_path(
            self,
            self._dialogs,
            "Export spectrum as CSV",
            "spectrum.csv",
            "CSV files (*.csv)",
            self._write_csv,
        )

    def _write_csv(self, path: str) -> None:
        if self._pg is None:
            return
        try:
            self.export_csv(path)
        except OSError as exc:
            report_failure(self, self.export_failed, f"Could not export {path}: {exc}")

    def export_csv(self, path: str) -> None:
        """Write the full-resolution spectrum to ``path`` as CSV.

        The columns are ``frequency``, ``period`` and the plotted quantity: ``power``
        for a periodogram, ``amplitude`` for a pre-whitening amplitude spectrum.
        """
        assert self._pg is not None
        quantity = "amplitude" if self._pg.method == PREWHITEN_METHOD else "power"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frequency", "period", quantity])
            for freq, period, power in zip(
                self._pg.frequency, self._pg.period, self._pg.power, strict=True
            ):
                writer.writerow([freq, period, power])

    def _export_png(self) -> None:
        ask_save_path(
            self,
            self._dialogs,
            "Export spectrum as PNG",
            "spectrum.png",
            "PNG images (*.png)",
            self._write_png,
        )

    def _write_png(self, path: str) -> None:
        confirm_overwrite(
            self, self._dialogs, path, image_target(path), lambda: self._save_png(path)
        )

    def _save_png(self, path: str) -> None:
        try:
            self.export_png(path)
        except OSError as exc:
            report_failure(self, self.export_failed, str(exc))

    def export_png(self, path: str) -> Path:
        """Render the current plot to an image file; returns the path written.

        ``.png`` is appended unless ``path`` already ends in an image suffix Qt can
        write (``.png``, ``.jpg``, ...). Raises :class:`OSError` if nothing could be
        written.
        """
        return write_plot_image(
            self._plot.getPlotItem(), path, scale=device_scale(self)
        )

    # -- theme -------------------------------------------------------------------
    def _curve_pen(self) -> object:
        return pg.mkPen(self._pal.curve, width=stroke_width(self))

    def _overlay_pen(self) -> object:
        # The residual colour is a hue apart from the curve under every colour-vision
        # deficiency (the accent was only dE00 7.8 from it under deuteranopia), and
        # the wider stroke tells the two apart without colour at all.
        return pg.mkPen(self._pal.residual, width=max(2.0, stroke_width(self)))

    def _cursor_pen(self) -> object:
        return pg.mkPen(
            self._pal.plot_fg,
            width=stroke_width(self, 1.5),
            style=Qt.PenStyle.DashDotLine,
        )

    def _threshold_pen(self) -> object:
        return pg.mkPen(
            self._pal.best_peak,
            width=stroke_width(self, 1.5),
            style=Qt.PenStyle.DashDotDotLine,
        )

    def _band_pen(self) -> object:
        return pg.mkPen(
            self._pal.qcolor(self._pal.accent, 170), width=stroke_width(self)
        )

    def _crosshair_pen(self) -> object:
        return pg.mkPen(
            self._pal.muted, width=stroke_width(self), style=Qt.PenStyle.DashLine
        )

    def _window_pen(self) -> object:
        return pg.mkPen(
            self._pal.qcolor(self._pal.muted, 200), width=stroke_width(self),
            style=Qt.PenStyle.DashLine,
        )

    def _legend_brush(self) -> object:
        return pg.mkBrush(self._pal.qcolor(self._pal.plot_bg, 200))

    def _repen(self) -> None:
        """Re-create every stroke for the current palette and device-pixel ratio."""
        self._curve.setPen(self._curve_pen())
        self._overlay.setPen(self._overlay_pen())
        self._window_curve.setPen(self._window_pen())
        band_pen = self._band_pen()
        for line in self._sel_band.lines:
            line.setPen(band_pen)
        self._vline.setPen(self._crosshair_pen())
        self._hline.setPen(self._crosshair_pen())
        self._cursor_line.setPen(self._cursor_pen())
        self._threshold_curve.setPen(self._threshold_pen())

    def event(self, event: QtCore.QEvent) -> bool:
        # pyqtgraph pens are cosmetic (device pixels), so moving to a screen with a
        # different scale factor needs new pens to keep the strokes' logical width.
        if event.type() == QtCore.QEvent.Type.DevicePixelRatioChange:
            self._repen()
            self._redraw_markers()  # their outlines are strokes too
            self._redraw_candidate()
        return super().event(event)

    def apply_theme(self, theme_palette: ThemePalette) -> None:
        """Re-pen the plot items for a new theme (live re-skin)."""
        self._pal = theme_palette
        self._repen()
        self._sel_band.setBrush(
            pg.mkBrush(theme_palette.qcolor(theme_palette.accent, 45))
        )
        self._legend.setBrush(self._legend_brush())
        self._legend.setLabelTextColor(theme_palette.plot_fg)
        self._sync_legend()
        self._hover_text.setColor(theme_palette.plot_fg)
        self._candidate_label.setColor(theme_palette.plot_fg)
        self._style_placeholder()
        apply_plot_theme(self._plot, theme_palette)
        self._redraw_markers()
        self._redraw_candidate()


__all__ = [
    "PREWHITEN_METHOD",
    "SNAP_PX",
    "SPECTRUM_KEYS",
    "SpectrumView",
    "envelope_indices",
    "local_maxima",
]
