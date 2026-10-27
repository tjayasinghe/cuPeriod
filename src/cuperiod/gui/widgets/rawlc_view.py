"""The raw (unfolded) light-curve view — spot checks, and point editing in a session.

Plots brightness against time (days since the first sample) for the loaded curve,
overlaid per band for multiband data, each band with its own colour *and* marker shape
so bands never differ by colour alone. Like the phased view it inverts the y-axis for
magnitudes and leaves flux as-is, leaves out rows whose time or value is not finite,
and draws a large curve as a fixed subset (see :class:`DisplayPoints`).

During interactive pre-whitening (:meth:`RawLightCurveView.set_session_view`) the
view edits the session's points: it shows the data, the residuals or the data with
the model, coloured and shaped by subset; the Rectangle and Lasso tools (and Shift-drag
in Pan mode) select points, Delete and Restore act on the selection (its count is on
the title line), and "Select beyond N σ" selects the outliers of every zero-point
subset from the keyboard. The controls take one compact row that wraps. Deleted
points stay on the plot as hollow grey crosses, so they can be selected and restored.
The view only asks:
:attr:`~RawLightCurveView.delete_requested` and
:attr:`~RawLightCurveView.restore_requested` carry point indices for the session, which
makes one delete one undoable action.

A session view also has a *time cursor*, a labelled dash-dot line: a plain click puts
it where the mouse is (with any tool — pyqtgraph reports no click for a drag, so
panning and the selection tools are unaffected) and Left/Right move it from the
keyboard. :attr:`~RawLightCurveView.time_cursor_changed` announces it in the session's
own time (not the axis's days since the first sample): the Subsets pane's *Split at*
cuts there.

The panel is hidden until "Show Raw LC" is ticked, so it builds its scatters only while
it is visible: loading a curve or switching the theme just marks it out of date.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.core.columns import Domain
from cuperiod.gui.models import LoadedCurve
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, Signal, pg
from cuperiod.gui.selection import SelectMode, SelectViewBox, select_points
from cuperiod.gui.theme import (
    ThemePalette,
    apply_plot_theme,
    palette,
    plot_label_style,
)
from cuperiod.gui.widgets._view_helpers import ElidedLabel, legend_label, stroke_width
from cuperiod.gui.widgets.phased_view import (
    MAX_DRAWN_POINTS,
    THINNED_TOOLTIP,
    DisplayPoints,
    brightness_label,
    curve_bands,
)

BoolArray = npt.NDArray[np.bool_]

_TITLE = "Raw light curve"

#: What the session view plots on the y-axis.
Y_MODES: tuple[tuple[str, str], ...] = (
    ("data", "Data"),
    ("residuals", "Residuals"),
    ("model", "Data + model"),
)

#: Seed of the display subset of a large session curve (same points every time).
_SUBSET_SEED = 0

#: Robust sigma of a normal sample: 1.4826 x the median absolute deviation.
_MAD_TO_SIGMA = 1.4826

#: Keys the plot acts on itself in a session when no command shortcut owns them.
_TOOL_KEYS: dict[Qt.Key, SelectMode] = {
    Qt.Key.Key_V: "pan",
    Qt.Key.Key_X: "rect",
    Qt.Key.Key_G: "lasso",
}

#: Left/Right move the time cursor by this share of the visible time range (Shift:
#: the second value).
_CURSOR_STEP = (0.01, 0.1)

#: The keys the focused light curve acts on itself in pre-whitening, as ``(what,
#: keys)`` for the F1 sheet (V / X / G, Del, U and Esc are commands, listed there).
LIGHT_CURVE_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Move the time cursor (pre-whitening)", ("Left", "Right")),
    ("Move it ten times further", ("Shift+Left", "Shift+Right")),
)

#: The session plot's mouse and keyboard gestures (status tip and What's This).
_GESTURES_SESSION = (
    "Click: put the time cursor there (Subsets ▸ Split at cuts there) · Left/Right: "
    "move it (Shift: ten times further) · drag: pan, or select with the Rectangle or "
    "Lasso tool (Shift-drag selects in Pan) · V / X / G: Pan, Rectangle, Lasso · "
    "Del: delete the selected points · U: restore them · Esc: clear the selection"
)


def outliers_beyond(
    residuals: FloatArray,
    codes: IntArray | None,
    usable: BoolArray,
    n_sigma: float,
) -> BoolArray:
    """Points more than ``n_sigma`` robust sigmas from their subset's median residual.

    Each zero-point subset (``codes``; one subset when ``None``) is judged on its own
    ``usable`` points: sigma is ``1.4826 x MAD`` (the standard deviation when the MAD
    is zero), so the outliers being hunted do not inflate their own yardstick.
    """
    r = np.asarray(residuals, dtype=np.float64)
    usable = np.asarray(usable, dtype=np.bool_) & np.isfinite(r)
    groups = np.zeros(r.size, dtype=np.int64) if codes is None else np.asarray(codes)
    out = np.zeros(r.size, dtype=np.bool_)
    for code in np.unique(groups[usable]):
        members = usable & (groups == code)
        values = r[members]
        if values.size < 3:
            continue
        centre = float(np.median(values))
        sigma = _MAD_TO_SIGMA * float(np.median(np.abs(values - centre)))
        if not sigma > 0.0:
            sigma = float(np.std(values))
        if not sigma > 0.0:
            continue
        out |= members & (np.abs(r - centre) > n_sigma * sigma)
    return out


class _PointsViewBox(SelectViewBox):
    """A :class:`SelectViewBox` whose selection gestures work only in a session.

    Outside a session the raw view is a plain plot: Shift-drag must not draw a
    rubber band that selects nothing.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.selection_enabled = False

    def mouseDragEvent(self, ev: Any, axis: int | None = None) -> None:  # noqa: N802
        if not self.selection_enabled and not self.selecting:
            pg.ViewBox.mouseDragEvent(self, ev, axis)
            return
        super().mouseDragEvent(ev, axis)


class RawLightCurveView(QtWidgets.QWidget):
    """The unfolded time series (spot checks; point selection in a session)."""

    #: Delete these points (an int array of indices into the session's data).
    delete_requested = Signal(object)
    #: Restore these deleted points (an int array of indices).
    restore_requested = Signal(object)
    #: The number of selected points changed.
    selection_changed = Signal(int)
    #: The drag tool changed (``"pan"``, ``"rect"`` or ``"lasso"``).
    tool_changed = Signal(str)
    #: The time cursor moved by a click or Left/Right (session mode): its time in the
    #: session's own units (not days since the first sample); NaN when it went away
    #: (another session or curve).
    time_cursor_changed = Signal(float)

    def __init__(
        self, theme: str = "dark", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._pal: ThemePalette = palette(theme)  # type: ignore[arg-type]
        self._lc: LoadedCurve | None = None
        self._scatters: dict[str, Any] = {}
        self._stale = False
        # Session mode (interactive pre-whitening); see the module notes.
        self._session: Any = None
        self._session_domain: Domain | None = None
        self._selected: BoolArray = np.zeros(0, dtype=np.bool_)
        self._session_items: list[Any] = []
        self._rings: Any = None
        self._display: IntArray | None = None
        self._display_key: tuple[Any, ...] | None = None
        self._range_key: tuple[Any, ...] | None = None
        self._range_pending = False
        self._cursor_time: float | None = None
        self._title_base = _TITLE

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)  # tight, like the other plot panes
        # One line: the title with the counts (elided, never wrapping) and, in a
        # session, the point selection's count at its right end.
        title_row = QtWidgets.QHBoxLayout()
        title_row.setContentsMargins(0, 0, 6, 0)
        title_row.setSpacing(12)
        self._title = ElidedLabel(_TITLE)
        self._title.setObjectName("muted")
        self._title.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        title_row.addWidget(self._title, 1)
        layout.addLayout(title_row)
        self._session_bar = self._build_session_bar()
        self._session_bar.setVisible(False)
        title_row.addWidget(self._selection_label)
        self._selection_label.setVisible(False)
        layout.addWidget(self._session_bar)

        self._box = _PointsViewBox()
        self._box.region_selected.connect(self._on_region_selected)
        self._box.mode_changed.connect(self._on_mode_changed)
        self._plot = pg.PlotWidget(viewBox=self._box)
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        style = plot_label_style(self._pal)
        self._plot.setLabel("bottom", "Time (days since first sample)", **style)
        self._plot.setLabel("left", "Brightness", **style)
        self._plot.setAccessibleName("Light curve plot")
        # A Tab stop only in a session, where the plot has keys (tools, Del, U, Esc).
        self._plot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._plot.installEventFilter(self)
        self._legend = self._plot.addLegend(offset=(-10, 10))
        self._style_legend()
        # The time cursor: a labelled dash-dot line (not told apart by colour alone),
        # kept out of the auto-range and of the session items a rebuild replaces.
        self._cursor_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=self._cursor_pen(),
            label="cursor",
            labelOpts={"position": 0.92, "color": self._pal.plot_fg},
        )
        self._cursor_line.setZValue(8)
        self._cursor_line.setVisible(False)
        self._plot.addItem(self._cursor_line, ignoreBounds=True)
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        layout.addWidget(self._plot, 1)
        apply_plot_theme(self._plot, self._pal)
        self._update_accessible()

    # -- the session toolbar --------------------------------------------------------
    def _build_session_bar(self) -> QtWidgets.QWidget:
        """The session controls: one wrapping row of four groups.

        What to plot, the drag tools, the outlier selection and Delete / Restore /
        Clear each stay together, and the row wraps between groups when the pane is
        narrow (UX-04); the selection's count sits on the title line. They used to
        take three rows of their own: under the fold in a 1366 x 768 window that
        left the plot 45 px tall.
        """
        from cuperiod.gui.widgets.info_bar import FlowLayout

        bar = QtWidgets.QWidget()
        bar.setObjectName("plotToolbar")  # compact controls (theme sheets)
        flow = FlowLayout(hspacing=12, vspacing=2)
        flow.setContentsMargins(6, 0, 6, 0)
        bar.setLayout(flow)

        def group(*widgets: QtWidgets.QWidget) -> None:
            box = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(box)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            for widget in widgets:
                row.addWidget(widget)
            flow.addWidget(box)

        show_label = QtWidgets.QLabel("Show:")
        self._mode_combo = QtWidgets.QComboBox()
        for key, text in Y_MODES:
            self._mode_combo.addItem(text, key)
        self._mode_combo.setToolTip(
            "Plot the data, the residuals after subtracting the model, or the data "
            "with the model drawn over it"
        )
        self._mode_combo.setAccessibleName("Light curve y values")
        self._mode_combo.currentIndexChanged.connect(self._on_y_mode_changed)
        show_label.setBuddy(self._mode_combo)
        group(show_label, self._mode_combo)

        self._tool_group = QtWidgets.QButtonGroup(bar)
        self._tool_group.setExclusive(True)
        self._tool_buttons: dict[str, QtWidgets.QToolButton] = {}
        for mode, text, tip in (
            ("pan", "Pan", "Drag to pan the light curve; Shift-drag selects (V)"),
            ("rect", "Rectangle", "Drag a rectangle to select points; Ctrl adds (X)"),
            ("lasso", "Lasso", "Draw a lasso to select points; Ctrl adds (G)"),
        ):
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setStatusTip(tip)
            button.setAccessibleName(f"{text} tool")
            button.clicked.connect(lambda _=False, m=mode: self.set_tool(m))
            self._tool_group.addButton(button)
            self._tool_buttons[mode] = button
        self._tool_buttons["pan"].setChecked(True)
        group(*self._tool_buttons.values())

        self._beyond_button = QtWidgets.QPushButton("Select beyond")
        self._beyond_button.setToolTip(
            "Select every point whose residual lies more than this many sigma from "
            "its zero-point subset's median (sigma = 1.4826 x MAD of the subset)"
        )
        self._beyond_button.setAccessibleName("Select outliers beyond sigma")
        self._beyond_button.clicked.connect(lambda: self.select_beyond())
        self._sigma_spin = QtWidgets.QDoubleSpinBox()
        self._sigma_spin.setRange(1.0, 20.0)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setValue(4.0)
        self._sigma_spin.setSuffix(" σ")
        self._sigma_spin.setToolTip("Outlier threshold in robust sigmas")
        self._sigma_spin.setAccessibleName("Outlier threshold in sigma")
        group(self._beyond_button, self._sigma_spin)

        self._selection_label = QtWidgets.QLabel("No points selected")
        self._selection_label.setObjectName("muted")
        self._selection_label.setAccessibleName("Point selection")
        self._delete_button = QtWidgets.QPushButton("Delete")
        self._delete_button.setToolTip(
            "Leave the selected points out of the analysis (Del). They stay on the "
            "plot as grey crosses and can be restored; Undo also brings them back"
        )
        self._delete_button.clicked.connect(self.delete_selected)
        self._restore_button = QtWidgets.QPushButton("Restore")
        self._restore_button.setToolTip("Bring the selected deleted points back (U)")
        self._restore_button.clicked.connect(self.restore_selected)
        self._clear_button = QtWidgets.QPushButton("Clear")
        self._clear_button.setToolTip("Deselect every point (Esc)")
        self._clear_button.clicked.connect(self.clear_selection)
        for push, name in (
            (self._delete_button, "Delete selected points"),
            (self._restore_button, "Restore selected points"),
            (self._clear_button, "Clear point selection"),
        ):
            push.setAccessibleName(name)
            push.setStatusTip(push.toolTip())
        group(self._delete_button, self._restore_button, self._clear_button)
        self._sync_selection_bar()
        return bar

    # -- plain light curve ----------------------------------------------------------
    def set_light_curve(self, lc: LoadedCurve) -> None:
        """Store a new light curve; it is drawn now if shown, else when next shown.

        A new curve ends session mode: the session's points belonged to the old one.
        """
        self._lc = lc
        self._leave_session()
        self._refresh()

    def _refresh(self) -> None:
        self._stale = True
        if self.isVisible():
            self._rebuild()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._stale:
            self._rebuild()
        # Range on the data once laid out, whenever the panel comes back: a range
        # taken while it was hidden (zero-sized) showed a 0-2400 d curve as
        # -1800-4200 d (UX-05: the raw view was not re-ranged on show).
        QtCore.QTimer.singleShot(0, self, self._auto_range)

    def _auto_range(self) -> None:
        """Range on the data now, or as soon as the plot is shown with a size."""
        box = self._plot.getPlotItem().vb
        if not self.isVisible() or box.width() < 2 or box.height() < 2:
            self._range_pending = True
            return
        self._range_pending = False
        box.autoRange()

    def _style_legend(self) -> None:
        self._legend.setBrush(pg.mkBrush(self._pal.qcolor(self._pal.plot_bg, 200)))
        self._legend.setLabelTextColor(self._pal.plot_fg)

    def _clear_items(self) -> None:
        for scatter in self._scatters.values():
            self._plot.removeItem(scatter)
        self._scatters.clear()
        for item in self._session_items:
            self._plot.removeItem(item)
        self._session_items = []
        self._rings = None
        self._legend.clear()

    def _rebuild(self) -> None:
        self._stale = False
        self._clear_items()
        if self._session is not None:
            self._rebuild_session()
            return
        bands = curve_bands(self._lc)
        points = DisplayPoints(self._lc)
        summary = points.summary()
        self._title.setText(f"{_TITLE}  ·  {summary}" if summary else _TITLE)
        self._title.setToolTip(THINNED_TOOLTIP if summary else "")
        self._update_accessible()
        if not bands:
            return
        # Times are measured from the first *finite* sample: one NaN timestamp used to
        # make every offset NaN, and an empty curve made min() raise in the slot.
        multi = len(bands) > 1
        default_color = self._pal.qcolor(self._pal.peak)
        for i, (name, (time, value)) in enumerate(points.bands.items()):
            color = self._pal.band_color(i) if multi else default_color
            # Shape backs up colour, so bands stay apart for every reader (WCAG 1.4.1).
            symbol = self._pal.band_symbol(i) if multi else "o"
            scatter = pg.ScatterPlotItem(
                size=5, brush=pg.mkBrush(color), pen=None, symbol=symbol, pxMode=True
            )
            scatter.setData(time - points.first_time, value)
            self._plot.addItem(scatter)
            if multi:
                self._legend.addItem(scatter, legend_label(name))
            self._scatters[name] = scatter
        domain = next(iter(bands.values())).domain
        self._plot.getPlotItem().vb.invertY(domain is Domain.MAGNITUDE)
        self._plot.setLabel(
            "left", brightness_label(domain), **plot_label_style(self._pal)
        )
        self._auto_range()

    # -- session mode ---------------------------------------------------------------
    def set_session_view(self, view: Any, *, domain: Domain | None = None) -> None:
        """Show and edit the points of a :class:`~cuperiod.gui.viewmodels.SessionView`.

        ``domain`` (magnitude or flux) sets the axis direction and label; by default
        it comes from the light curve last given to :meth:`set_light_curve`. The point
        selection is kept while the session's points stay the same (an edit, a delete,
        an undo) and dropped when another session or curve arrives. The range is kept
        across edits of the same session.
        """
        n = int(np.asarray(view.time).size)
        same = (
            self._session is not None
            and self._session.key == view.key
            and self._selected.size == n
        )
        self._session = view
        if domain is not None:
            self._session_domain = domain
        if not same:
            self._selected = np.zeros(n, dtype=np.bool_)
            self._display_key = None
            self._emit_selection()
            self._drop_time_cursor()  # its time belonged to the other session
        self._box.selection_enabled = True
        self._plot.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._plot.setStatusTip(_GESTURES_SESSION)
        self._plot.setWhatsThis(_GESTURES_SESSION)
        self._session_bar.setVisible(True)
        self._selection_label.setVisible(True)
        self._sync_selection_bar()
        self._refresh()

    def clear_session(self) -> None:
        """Leave session mode and show the plain light curve again."""
        if self._leave_session():
            self._refresh()

    def _leave_session(self) -> bool:
        if self._session is None:
            return False
        self._session = None
        self._session_domain = None
        self._selected = np.zeros(0, dtype=np.bool_)
        self._display_key = None
        self._range_key = None
        self._title_base = _TITLE
        self._box.cancel_selection()
        self._box.set_mode("pan")
        self._box.selection_enabled = False
        self._plot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._plot.setStatusTip("")
        self._plot.setWhatsThis("")
        self._session_bar.setVisible(False)
        self._selection_label.setVisible(False)
        self._emit_selection()
        self._sync_selection_bar()
        self._drop_time_cursor()
        return True

    def focus_plot(self) -> None:
        """Give the plot the keyboard focus (View ▸ Focus light curve, Ctrl+3).

        Outside a session the plot takes no focus: it has no keys there.
        """
        if self._session is not None:
            self._plot.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def y_mode(self) -> str:
        """``"data"``, ``"residuals"`` or ``"model"`` (data with the model)."""
        return str(self._mode_combo.currentData())

    def set_y_mode(self, mode: str) -> None:
        """Plot the data, the residuals or the data with the model."""
        index = self._mode_combo.findData(mode)
        if index < 0:
            raise ValueError(f"unknown y mode {mode!r}")
        self._mode_combo.setCurrentIndex(index)

    def tool(self) -> SelectMode:
        """The active drag tool."""
        return self._box.mode

    def set_tool(self, mode: SelectMode) -> None:
        """Pan, Rectangle or Lasso (also the V / X / G keys)."""
        self._box.set_mode(mode)
        self._tool_buttons[mode].setChecked(True)

    def _on_mode_changed(self, mode: str) -> None:
        button = self._tool_buttons.get(mode)
        if button is not None:
            button.setChecked(True)
        self.tool_changed.emit(mode)

    def _on_y_mode_changed(self, _index: int) -> None:
        self._range_key = None  # another quantity: range on it afresh
        if self._session is not None:
            self._refresh()

    def _domain(self) -> Domain | None:
        if self._session_domain is not None:
            return self._session_domain
        bands = curve_bands(self._lc)
        return next(iter(bands.values())).domain if bands else None

    def _time_origin(self) -> float:
        """The session time at x = 0 (its first finite sample, deleted ones too)."""
        assert self._session is not None
        time = np.asarray(self._session.time, dtype=np.float64)
        finite_time = time[np.isfinite(time)]
        return float(finite_time.min()) if finite_time.size else 0.0

    def _plot_xy(self) -> tuple[FloatArray, FloatArray]:
        """Plot coordinates of every session point in the current y mode."""
        assert self._session is not None
        time = np.asarray(self._session.time, dtype=np.float64)
        if self.y_mode() == "residuals":
            y = np.asarray(self._session.residuals, dtype=np.float64)
        else:
            y = np.asarray(self._session.value, dtype=np.float64)
        return time - self._time_origin(), y

    def _subset_labels(self) -> dict[int, str]:
        """Labels of the subsets the points are coloured by (index -> label)."""
        assert self._session is not None
        view = self._session
        partition = view.offsets_by or (view.partitions[0] if view.partitions else None)
        return {
            int(s.index): str(s.label)
            for s in view.subsets
            if partition is not None and s.partition == partition
        }

    def _display_points(self, drawable: BoolArray) -> BoolArray:
        """The points drawn: all, or a fixed random subset of a large curve."""
        n = drawable.size
        count = int(drawable.sum())
        if count <= MAX_DRAWN_POINTS:
            return drawable
        key = (n, count)
        if self._display is None or self._display_key != key:
            rng = np.random.default_rng(_SUBSET_SEED)
            chosen = rng.choice(np.flatnonzero(drawable), MAX_DRAWN_POINTS, False)
            self._display = np.sort(chosen).astype(np.int64)
            self._display_key = key
        shown = np.zeros(n, dtype=np.bool_)
        shown[self._display] = True
        return shown & drawable

    def _rebuild_session(self) -> None:
        assert self._session is not None
        view = self._session
        x, y = self._plot_xy()
        n = x.size
        deleted = np.asarray(view.deleted, dtype=np.bool_)
        if deleted.size != n:
            deleted = np.zeros(n, dtype=np.bool_)
        drawable = np.isfinite(x) & np.isfinite(y)
        shown = self._display_points(drawable)
        codes = (
            np.zeros(n, dtype=np.int64)
            if view.point_codes is None
            else np.asarray(view.point_codes, dtype=np.int64)
        )
        labels = self._subset_labels()
        live = shown & ~deleted
        present = sorted(int(c) for c in np.unique(codes[drawable]))
        multi = len(present) > 1
        default_color = self._pal.qcolor(self._pal.peak)
        for code in present:
            members = live & (codes == code)
            color = self._pal.band_color(code) if multi else default_color
            symbol = self._pal.band_symbol(code) if multi else "o"
            scatter = pg.ScatterPlotItem(
                size=5, brush=pg.mkBrush(color), pen=None, symbol=symbol, pxMode=True
            )
            scatter.setData(x[members], y[members])
            self._add(scatter)
            if multi:
                self._legend.addItem(
                    scatter, legend_label(labels.get(code, f"subset {code}"))
                )
        gone = deleted & drawable
        if gone.any():
            crosses = pg.ScatterPlotItem(
                size=8,
                symbol="x",
                brush=None,
                pen=pg.mkPen(self._pal.muted, width=stroke_width(self, 1.5)),
                pxMode=True,
            )
            crosses.setData(x[gone], y[gone])
            self._add(crosses)
            self._legend.addItem(crosses, f"deleted ({int(gone.sum()):,})")
        if self.y_mode() == "model":
            self._add_model_line(x, deleted)
        # Selected points are ringed, every one of them (thinning never hides them).
        self._rings = pg.ScatterPlotItem(
            size=11,
            symbol="o",
            brush=None,
            pen=pg.mkPen(self._pal.accent, width=stroke_width(self, 2.0)),
            pxMode=True,
        )
        self._rings.setZValue(10)
        self._add(self._rings)
        self._update_rings()
        domain = self._domain()
        box = self._plot.getPlotItem().vb
        inverted = domain is Domain.MAGNITUDE
        box.invertY(inverted)
        # The cursor's label sits at the top of the plot either way up.
        self._cursor_line.label.setPosition(0.08 if inverted else 0.92)
        label = self._y_label(domain)
        self._plot.setLabel("left", label, **plot_label_style(self._pal))
        n_drawn, n_finite = int(shown.sum()), int(drawable.sum())
        parts = [_TITLE, f"{n_finite:,} points", f"{int(deleted.sum()):,} deleted"]
        if n_drawn < n_finite:
            parts.append(f"{n_drawn:,} of {n_finite:,} shown")
        self._title_base = "  ·  ".join(parts)
        self._title.setToolTip(THINNED_TOOLTIP if n_drawn < n_finite else "")
        self._place_time_cursor()
        range_key = (view.key, n, self.y_mode())
        if range_key != self._range_key:  # a new session or quantity: range on it
            self._range_key = range_key
            self._auto_range()
        self._update_accessible()

    def _add(self, item: Any) -> None:
        self._plot.addItem(item)
        self._session_items.append(item)

    def _update_rings(self) -> None:
        """Ring the selected points; cheap enough to follow every selection change."""
        if self._rings is None or self._session is None:
            return
        x, y = self._plot_xy()
        picked = self._selected & np.isfinite(x) & np.isfinite(y)
        self._rings.setData(x[picked], y[picked])

    def _add_model_line(self, x: FloatArray, deleted: BoolArray) -> None:
        """The model through the points in time order, broken across long gaps."""
        assert self._session is not None
        model = np.asarray(self._session.model, dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(model)
        if model.size != x.size or int(ok.sum()) < 2:
            return
        index = np.flatnonzero(ok)
        if index.size > MAX_DRAWN_POINTS:
            index = index[self._display_points(ok)[index]]
        order = index[np.argsort(x[index], kind="stable")]
        xs, ys = x[order], model[order]
        gaps = np.diff(xs)
        typical = float(np.median(gaps)) if gaps.size else 0.0
        connect = np.ones(xs.size, dtype=np.bool_)
        if typical > 0.0:
            connect[:-1] = gaps <= 5.0 * typical
        # Wider than the points' outline and a hue apart from them, so the model reads
        # as a line under any colour-vision deficiency.
        line = pg.PlotDataItem(
            xs,
            ys,
            pen=pg.mkPen(self._pal.residual, width=max(2.0, stroke_width(self))),
            connect=connect,
        )
        line.setZValue(5)
        self._add(line)
        self._legend.addItem(line, "model")

    def _y_label(self, domain: Domain | None) -> str:
        if self.y_mode() == "residuals":
            if domain is None:
                return "Residual"
            return "Residual (mag)" if domain is Domain.MAGNITUDE else "Residual (flux)"
        return "Brightness" if domain is None else brightness_label(domain)

    # -- selection ------------------------------------------------------------------
    def selected_indices(self) -> IntArray:
        """Indices of the selected points (into the session's data)."""
        return np.flatnonzero(self._selected).astype(np.int64)

    def selection_count(self) -> int:
        return int(self._selected.sum())

    def set_selection(self, indices: Any, *, additive: bool = False) -> None:
        """Select points by index (or by a boolean mask over every point)."""
        if self._session is None:
            return
        mask = np.zeros(self._selected.size, dtype=np.bool_)
        chosen = np.asarray(indices)
        if chosen.dtype == np.bool_ and chosen.size == mask.size:
            mask = chosen.copy()
        elif chosen.size:
            mask[chosen.astype(np.int64)] = True
        self._change_selection(self._selected | mask if additive else mask)

    def clear_selection(self) -> None:
        """Deselect every point."""
        self._box.cancel_selection()
        if self._selected.any():
            self._change_selection(np.zeros_like(self._selected))

    def select_beyond(self, n_sigma: float | None = None) -> int:
        """Select the points beyond ``n_sigma`` (default: the spin box) per subset.

        Judged on the residuals of each zero-point subset's points in use (see
        :func:`outliers_beyond`); returns the number selected.
        """
        if self._session is None:
            return 0
        threshold = float(self._sigma_spin.value() if n_sigma is None else n_sigma)
        view = self._session
        deleted = np.asarray(view.deleted, dtype=np.bool_)
        if deleted.size != self._selected.size:
            deleted = np.zeros(self._selected.size, dtype=np.bool_)
        mask = outliers_beyond(
            np.asarray(view.residuals, dtype=np.float64),
            None if view.point_codes is None else np.asarray(view.point_codes),
            ~deleted,
            threshold,
        )
        self._change_selection(mask)
        return int(mask.sum())

    def delete_selected(self) -> None:
        """Ask to delete the selected points that are still in use (one action)."""
        index = self._selected_with(deleted=False)
        if index.size:
            self.delete_requested.emit(index)
            self.clear_selection()

    def restore_selected(self) -> None:
        """Ask to restore the selected points that are deleted (one action)."""
        index = self._selected_with(deleted=True)
        if index.size:
            self.restore_requested.emit(index)
            self.clear_selection()

    def _selected_with(self, *, deleted: bool) -> IntArray:
        if self._session is None:
            return np.zeros(0, dtype=np.int64)
        gone = np.asarray(self._session.deleted, dtype=np.bool_)
        if gone.size != self._selected.size:
            gone = np.zeros(self._selected.size, dtype=np.bool_)
        state = gone if deleted else ~gone
        return np.flatnonzero(self._selected & state).astype(np.int64)

    def _on_region_selected(self, polygon: Any, additive: bool) -> None:
        if self._session is None:
            return
        x, y = self._plot_xy()
        inside = select_points(x, y, polygon) & np.isfinite(x) & np.isfinite(y)
        self._change_selection(self._selected | inside if additive else inside)

    def _change_selection(self, mask: BoolArray) -> None:
        changed = mask.shape != self._selected.shape or bool(
            np.any(mask != self._selected)
        )
        self._selected = np.asarray(mask, dtype=np.bool_)
        if not changed:
            return
        self._emit_selection()
        self._sync_selection_bar()
        if self._stale or not self.isVisible():
            self._stale = True  # drawn in full when next shown
        else:
            self._update_rings()
            self._update_accessible()

    def _emit_selection(self) -> None:
        self.selection_changed.emit(int(self._selected.sum()))

    def _sync_selection_bar(self) -> None:
        count = int(self._selected.sum())
        self._selection_label.setText(
            "No points selected"
            if count == 0
            else f"{count:,} point{'s' if count != 1 else ''} selected"
        )
        self._delete_button.setEnabled(self._selected_with(deleted=False).size > 0)
        self._restore_button.setEnabled(self._selected_with(deleted=True).size > 0)
        self._clear_button.setEnabled(count > 0)

    # -- the time cursor ------------------------------------------------------------
    def time_cursor(self) -> float | None:
        """The time cursor in the session's own time units, or ``None``."""
        return self._cursor_time

    def set_time_cursor(self, time: float | None) -> None:
        """Put the time cursor at ``time`` (session time; ``None`` hides it), no echo.

        Only a session view has a cursor: outside one this hides it.
        """
        value = None if time is None else float(time)
        if self._session is None or (value is not None and not math.isfinite(value)):
            value = None
        self._cursor_time = value
        self._place_time_cursor()
        self._update_accessible()

    def _move_time_cursor(self, time: float) -> None:
        """Move the cursor by a user gesture and announce it."""
        self._cursor_time = float(time)
        self._place_time_cursor()
        self._update_accessible()
        self.time_cursor_changed.emit(self._cursor_time)

    def _drop_time_cursor(self) -> None:
        """Forget the cursor (another session or curve); NaN tells the listeners."""
        had = self._cursor_time is not None
        self._cursor_time = None
        self._place_time_cursor()
        if had:
            self.time_cursor_changed.emit(math.nan)

    def _place_time_cursor(self) -> None:
        """Draw the cursor line where the cursor is (in days since the first sample)."""
        time = self._cursor_time
        if self._session is None or time is None:
            self._cursor_line.setVisible(False)
        else:
            self._cursor_line.setPos(time - self._time_origin())
            self._cursor_line.setVisible(True)
        self._sync_title()

    def _sync_title(self) -> None:
        """The session title with the cursor's time (plain mode titles itself)."""
        if self._session is None:
            return
        text = self._title_base
        if self._cursor_time is not None:
            text += f"  ·  cursor t = {self._cursor_time:.5f}"
        self._title.setText(text)

    def _on_plot_clicked(self, event: Any) -> None:
        """A plain left click in a session puts the time cursor under the mouse.

        pyqtgraph reports a click only when the mouse did not move between press and
        release, so a pan (or a rectangle / lasso drag) never moves the cursor; a
        click with a modifier, a double click and a click on an axis are left alone.
        """
        if self._session is None or event.double():
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        held = event.modifiers() & (
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        )
        if held:
            return
        box = self._plot.getPlotItem().vb
        pos = event.scenePos()
        if not box.sceneBoundingRect().contains(pos):
            return
        x = float(box.mapSceneToView(pos).x())
        if math.isfinite(x):
            self._move_time_cursor(self._time_origin() + x)

    def _step_time_cursor(self, direction: int, *, coarse: bool) -> None:
        """Left/Right: move the cursor by 1 % of the visible range (Shift: 10 %).

        The first press puts it in the middle of the view; the view pans (zoom kept)
        when the cursor would leave it.
        """
        box = self._plot.getPlotItem().vb
        (x0, x1), _ = box.viewRange()
        origin = self._time_origin()
        if self._cursor_time is None:
            x = 0.5 * (x0 + x1)
        else:
            step = (x1 - x0) * _CURSOR_STEP[1 if coarse else 0]
            x = self._cursor_time - origin + direction * step
        if not x0 <= x <= x1:
            half = 0.5 * (x1 - x0)
            box.setXRange(x - half, x + half, padding=0.0)
        self._move_time_cursor(origin + x)

    def _cursor_pen(self) -> Any:
        # Dash-dot like the spectrum's pick cursor: a cursor reads the same everywhere.
        return pg.mkPen(
            self._pal.plot_fg,
            width=stroke_width(self, 1.5),
            style=Qt.PenStyle.DashDotLine,
        )

    # -- keyboard -------------------------------------------------------------------
    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            watched is self._plot
            and event.type() == QtCore.QEvent.Type.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and self._on_key(event)
        ):
            return True
        return super().eventFilter(watched, event)

    def _on_key(self, event: QtGui.QKeyEvent) -> bool:
        """The session keys, when no command shortcut around the plot owns them.

        The command registry binds Del / U / Esc / V / X / G on the light curve and
        follows its enabled state; these are the fallback of a view standing alone.
        Left/Right (Shift: further) move the time cursor; no command uses them.
        """
        extra = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        key = Qt.Key(event.key())
        if self._session is not None and key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            if extra & ~Qt.KeyboardModifier.ShiftModifier or self._shortcut_owns(event):
                return False
            self._step_time_cursor(
                1 if key == Qt.Key.Key_Right else -1,
                coarse=bool(extra & Qt.KeyboardModifier.ShiftModifier),
            )
            return True
        if self._session is None or extra:
            return False
        if key == Qt.Key.Key_Escape and self._box.selecting:
            self._box.cancel_selection()
            return True
        handled = key in _TOOL_KEYS or key in (
            Qt.Key.Key_Delete,
            Qt.Key.Key_U,
            Qt.Key.Key_Escape,
        )
        if not handled or self._shortcut_owns(event):
            return False
        if key in _TOOL_KEYS:
            self.set_tool(_TOOL_KEYS[key])
        elif key == Qt.Key.Key_Delete:
            self.delete_selected()
        elif key == Qt.Key.Key_U:
            self.restore_selected()
        else:
            self.clear_selection()
        return True

    def _shortcut_owns(self, event: QtGui.QKeyEvent) -> bool:
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

    # -- accessibility --------------------------------------------------------------
    def accessible_summary(self) -> str:
        """What the plot shows, for its accessible description."""
        if self._session is not None:
            view = self._session
            n = int(np.asarray(view.time).size)
            deleted = int(np.asarray(view.deleted, dtype=np.bool_).sum())
            shown = dict(Y_MODES)[self.y_mode()].lower()
            text = (
                f"Light curve, {shown} versus time: {n:,} points, {deleted:,} "
                f"deleted, {self.selection_count():,} selected; {self._box.mode} tool"
            )
            if self._cursor_time is not None:
                text += f"; time cursor at t = {self._cursor_time:.5f}"
            return text + "."
        bands = curve_bands(self._lc)
        if not bands:
            return "No light curve loaded."
        n = sum(int(band.time.size) for band in bands.values())
        names = ", ".join(name for name in bands if name)
        tail = f" in {len(bands)} bands ({names})" if len(bands) > 1 else ""
        return f"Light curve, brightness versus time: {n:,} points{tail}."

    def _update_accessible(self) -> None:
        self._plot.setAccessibleDescription(self.accessible_summary())

    def apply_theme(self, theme_palette: ThemePalette) -> None:
        """Re-skin for a new theme."""
        self._pal = theme_palette
        apply_plot_theme(self._plot, theme_palette)
        self._style_legend()
        self._cursor_line.setPen(self._cursor_pen())
        self._cursor_line.label.setColor(theme_palette.plot_fg)
        self._refresh()


__all__ = ["LIGHT_CURVE_KEYS", "Y_MODES", "RawLightCurveView", "outliers_beyond"]
