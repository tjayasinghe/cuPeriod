"""The phased light-curve view (pyqtgraph).

Shows the light curve folded at the selected period; it updates live from the
controller's fold signal by re-running only the O(n) fold, never the periodogram.
Multiband curves overlay one colour per band with a legend. Magnitude curves invert the
y-axis (brighter up); flux curves (e.g. a BLS/TLS transit) stay as dips. The "2 cycles"
toggle repeats the fold over ``[0, 2)``, the usual convention for reading shape.

Rows whose time or value is not finite are left out of the plot (loaders keep them, as
the periodogram engines drop them on their own), and so is a non-finite epoch: the fold
then starts at the first finite timestamp. A curve longer than
:data:`MAX_DRAWN_POINTS` is drawn as a fixed random subset of that size, chosen once per
loaded curve, so selecting another peak refolds without stalling the window; the panel
says when it is doing so. The analysis itself always uses every point.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import Domain
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.gui.fold import fold_series
from cuperiod.gui.models import LoadedCurve
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, Signal, pg
from cuperiod.gui.theme import (
    ThemePalette,
    apply_plot_theme,
    palette,
    plot_label_style,
)
from cuperiod.gui.widgets._view_helpers import (
    ask_save_path,
    confirm_overwrite,
    device_scale,
    image_target,
    is_connected,
    legend_label,
    report_failure,
    write_plot_image,
)

if TYPE_CHECKING:
    from cuperiod.gui.dialogs import DialogProvider

#: Most points the light-curve panels draw (all bands together). Past this, every
#: refold spends its time in pyqtgraph's per-point styling, not in the fold: 10^5
#: points took ~0.2 s per selection change and 5x10^5 more than a second.
MAX_DRAWN_POINTS = 20_000

#: Seed of the display subset, so the same curve always shows the same points.
_SUBSET_SEED = 0


def brightness_label(domain: Domain) -> str:
    """Y-axis label for a light-curve domain (flux is unit-agnostic in cuPeriod)."""
    return "Magnitude" if domain is Domain.MAGNITUDE else "Flux (arbitrary units)"


def curve_bands(lc: LoadedCurve | None) -> dict[str, LightCurve]:
    """The bands of a loaded curve, a single-band curve as one unnamed band."""
    if isinstance(lc, MultiBandLightCurve):
        return dict(lc.bands)
    if isinstance(lc, LightCurve):
        return {"": lc}
    return {}


class DisplayPoints:
    """The finite points of every band, thinned once to at most ``limit`` in total.

    Attributes
    ----------
    bands : dict of str to (time, value)
        Per band, the time and value arrays to draw (finite rows only).
    n_finite : int
        Finite points over all bands, before thinning.
    n_drawn : int
        Points kept for display over all bands.
    first_time : float
        The earliest finite timestamp of all bands (0.0 when there is none) — the
        epoch of last resort and the zero of the raw view's time axis.
    """

    def __init__(self, lc: LoadedCurve | None, limit: int = MAX_DRAWN_POINTS) -> None:
        source = curve_bands(lc)
        finite = {
            name: np.flatnonzero(np.isfinite(band.time) & np.isfinite(band.value))
            for name, band in source.items()
        }
        self.n_finite = int(sum(index.size for index in finite.values()))
        starts = [
            float(np.min(source[name].time[index]))
            for name, index in finite.items()
            if index.size
        ]
        self.first_time = min(starts) if starts else 0.0
        rng = np.random.default_rng(_SUBSET_SEED)
        self.bands: dict[str, tuple[FloatArray, FloatArray]] = {}
        for name, index in finite.items():
            if self.n_finite > limit and index.size:
                # Each band keeps its share of the budget, so a sparse band is not
                # thinned away next to a dense one.
                share = round(limit * index.size / self.n_finite)
                count = min(index.size, max(1, share))
                index = np.sort(rng.choice(index, size=count, replace=False))
            band = source[name]
            self.bands[name] = (
                np.asarray(band.time, dtype=np.float64)[index],
                np.asarray(band.value, dtype=np.float64)[index],
            )
        self.n_drawn = int(sum(time.size for time, _ in self.bands.values()))

    @property
    def thinned(self) -> bool:
        """Whether fewer points are drawn than the curve has."""
        return self.n_drawn < self.n_finite

    def summary(self) -> str:
        """``'20,000 of 500,000 points shown'``, or ``''`` when nothing was thinned."""
        if not self.thinned:
            return ""
        return f"{self.n_drawn:,} of {self.n_finite:,} points shown"


#: Tooltip of the "points shown" note in both light-curve panels.
THINNED_TOOLTIP = (
    "Large light curves are drawn as a fixed random subset so the plot stays "
    "responsive. The periodogram and every fit use all the points."
)


class PhasedView(QtWidgets.QWidget):
    """The folded light curve, updated live as the selected period changes."""

    show_raw_toggled = Signal(bool)
    #: The PNG export could not be written; carries a message for the user.
    export_failed = Signal(str)
    #: Fold at this period instead (×2, ÷2 or a typed period); unconnected, the view
    #: just refolds itself.
    period_requested = Signal(float)

    def __init__(
        self, theme: str = "dark", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._pal: ThemePalette = palette(theme)  # type: ignore[arg-type]
        self._lc: LoadedCurve | None = None
        self._points = DisplayPoints(None)
        self._period: float | None = None
        self._t0 = 0.0
        self._scatters: dict[str, Any] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)  # tight, like the other plot panes

        from cuperiod.gui.widgets.info_bar import FlowLayout

        bar = QtWidgets.QWidget()
        bar.setObjectName("plotToolbar")  # compact controls (theme sheets)
        # Wraps instead of squeezing its labels when the pane is narrow (UX-05).
        row = FlowLayout(hspacing=6, vspacing=2)
        bar.setLayout(row)
        row.setContentsMargins(6, 2, 6, 2)
        title = QtWidgets.QLabel("Phased light curve")
        title.setObjectName("muted")
        row.addWidget(title)
        self._points_label = QtWidgets.QLabel("")
        self._points_label.setObjectName("muted")
        self._points_label.setToolTip(THINNED_TOOLTIP)
        row.addWidget(self._points_label)
        self._period_label = QtWidgets.QLabel("")
        self._period_label.setObjectName("badgeAccent")
        self._period_label.setVisible(False)
        row.addWidget(self._period_label)
        # Folding at P/2 or 2P (an eclipsing binary found at half its period) or at a
        # typed period must not need a mouse drag on the spectrum.
        self._halve = QtWidgets.QToolButton()
        self._halve.setText("÷2")
        self._halve.setToolTip("Fold at half the period (P/2)")
        self._halve.setAccessibleName("Fold at half the period")
        self._halve.clicked.connect(lambda: self._scale_period(0.5))
        row.addWidget(self._halve)
        self._double = QtWidgets.QToolButton()
        self._double.setText("×2")
        self._double.setToolTip(
            "Fold at twice the period (2P): the usual check for an eclipsing binary, "
            "whose strongest peak is often at half its period"
        )
        self._double.setAccessibleName("Fold at twice the period")
        self._double.clicked.connect(lambda: self._scale_period(2.0))
        row.addWidget(self._double)
        self._period_edit = QtWidgets.QLineEdit()
        self._period_edit.setPlaceholderText("P (d)")
        validator = QtGui.QDoubleValidator(1e-9, 1e9, 12, self._period_edit)
        validator.setLocale(QtCore.QLocale.c())  # a dot, whatever the system locale
        validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        self._period_edit.setValidator(validator)
        self._period_edit.setToolTip(
            "Type a period in days and press Enter to fold at it"
        )
        self._period_edit.setAccessibleName("Fold at period in days")
        self._period_edit.setMaximumWidth(
            self._period_edit.fontMetrics().horizontalAdvance("0.000000000") + 16
        )
        self._period_edit.returnPressed.connect(self._on_period_entered)
        row.addWidget(self._period_edit)
        self._show_raw = QtWidgets.QCheckBox("Show Raw LC")
        self._show_raw.setToolTip("Show the raw (unfolded) light-curve panel")
        self._show_raw.toggled.connect(self.show_raw_toggled)
        row.addWidget(self._show_raw)
        self._two_cycles = QtWidgets.QCheckBox("2 cycles")
        self._two_cycles.setChecked(True)  # two cycles is the default view
        self._two_cycles.setToolTip(
            "Repeat the fold over phases 0-2, so a minimum or transit at phase 0 "
            "is seen whole"
        )
        self._two_cycles.toggled.connect(self.refold)
        row.addWidget(self._two_cycles)
        export_btn = QtWidgets.QPushButton("Image as PNG…")
        export_btn.setToolTip("Save the phased plot as a PNG image")
        export_btn.clicked.connect(self._export_png)
        row.addWidget(export_btn)
        layout.addWidget(bar)

        self._plot = pg.PlotWidget()
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        style = plot_label_style(self._pal)
        self._plot.setLabel("bottom", "Phase", **style)
        self._plot.setLabel("left", "Brightness", **style)
        self._legend = self._plot.addLegend(offset=(-10, 10))
        self._style_legend()
        # The fold has no keys of its own, so it is not a Tab stop (a keyboard user
        # would only pass through it); assistive technology still reads what it shows.
        self._plot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._plot.setAccessibleName("Phased light curve plot")
        layout.addWidget(self._plot, 1)
        apply_plot_theme(self._plot, self._pal)
        self._update_accessible()

    # -- data --------------------------------------------------------------------
    def set_light_curve(self, lc: LoadedCurve) -> None:
        """Store a new light curve and rebuild the per-band scatters."""
        self._lc = lc
        self._points = DisplayPoints(lc)
        self._points_label.setText(self._points.summary())
        self._rebuild_scatters()
        self.refold()
        self._update_accessible()

    def set_fold(self, period: float, t0: float) -> None:
        """Fold at ``period`` with epoch ``t0`` (from the controller).

        A non-finite ``t0`` (the epoch of a curve with a NaN timestamp) falls back to
        the first finite timestamp rather than turning every phase into NaN.
        """
        self._period = period
        self._t0 = float(t0) if np.isfinite(t0) else self._points.first_time
        self.refold()

    def refold(self) -> None:
        """Re-fold every band at the current period (O(n); no periodogram re-run)."""
        if self._lc is None or self._period is None:
            return
        period = self._period
        if not (np.isfinite(period) and period > 0.0):
            for scatter in self._scatters.values():
                scatter.setData([], [])
            return
        two = self._two_cycles.isChecked()
        for name, (time, value) in self._points.bands.items():
            scatter = self._scatters.get(name)
            if scatter is None:
                continue
            if time.size == 0:
                scatter.setData([], [])
                continue
            phase, values = fold_series(time, value, period, self._t0, two_cycles=two)
            scatter.setData(phase, values)
        self._show_period_text(f"P = {period:.6g} d")
        self._plot.getPlotItem().vb.autoRange()
        self._update_accessible()

    def _scale_period(self, factor: float) -> None:
        if self._period is not None and np.isfinite(self._period) and self._period > 0:
            self._request_period(self._period * factor)

    def _on_period_entered(self) -> None:
        try:
            period = float(self._period_edit.text())
        except ValueError:
            return
        if np.isfinite(period) and period > 0.0:
            self._request_period(period)
            self._period_edit.clear()

    def _request_period(self, period: float) -> None:
        """Ask the controller to fold at ``period``; standing alone, refold here."""
        if self._lc is None:
            return
        if is_connected(self, self.period_requested):
            self.period_requested.emit(float(period))
        else:
            self.set_fold(float(period), self._t0)

    # -- helpers -----------------------------------------------------------------
    def accessible_summary(self) -> str:
        """What the plot shows, for its accessible description."""
        if self._lc is None:
            return "No light curve loaded."
        n = self._points.n_drawn
        bands = [name for name in self._points.bands if name]
        tail = f" in {len(bands)} bands ({', '.join(bands)})" if len(bands) > 1 else ""
        if self._period is None or not (
            np.isfinite(self._period) and self._period > 0.0
        ):
            return f"Light curve of {n:,} points{tail}, not folded yet."
        cycles = "two cycles" if self._two_cycles.isChecked() else "one cycle"
        return (
            f"Light curve of {n:,} points{tail} folded at P = {self._period:.6g} d, "
            f"{cycles} shown."
        )

    def _update_accessible(self) -> None:
        self._plot.setAccessibleDescription(self.accessible_summary())

    def _bands(self) -> dict[str, LightCurve]:
        return curve_bands(self._lc)

    def _style_legend(self) -> None:
        """A semi-opaque legend background, so the entries read over the points."""
        self._legend.setBrush(pg.mkBrush(self._pal.qcolor(self._pal.plot_bg, 200)))
        self._legend.setLabelTextColor(self._pal.plot_fg)

    def _rebuild_scatters(self) -> None:
        for scatter in self._scatters.values():
            self._plot.removeItem(scatter)
        self._scatters.clear()
        self._legend.clear()
        bands = self._bands()
        multi = len(bands) > 1
        default_color = self._pal.qcolor(self._pal.peak)
        for i, name in enumerate(bands):
            color = self._pal.band_color(i) if multi else default_color
            # Each band has its own marker shape as well as its own colour, so the
            # bands (and their legend swatches) never differ by colour alone.
            symbol = self._pal.band_symbol(i) if multi else "o"
            scatter = pg.ScatterPlotItem(
                size=6, brush=pg.mkBrush(color), pen=None, symbol=symbol, pxMode=True
            )
            self._plot.addItem(scatter)
            if multi:
                # The legend renders HTML; the names come from the opened file.
                self._legend.addItem(scatter, legend_label(name))
            self._scatters[name] = scatter
        domain = self._domain()
        self._plot.getPlotItem().vb.invertY(domain is Domain.MAGNITUDE)
        self._plot.setLabel(
            "left", brightness_label(domain), **plot_label_style(self._pal)
        )

    # -- export ----------------------------------------------------------------
    #: The window's dialogs for the Export… prompts (None: a static file dialog).
    _dialogs: DialogProvider | None = None

    def set_dialog_provider(self, provider: DialogProvider | None) -> None:
        """Ask for export paths through ``provider`` (the window's dialogs)."""
        self._dialogs = provider

    def _export_png(self) -> None:
        ask_save_path(
            self,
            self._dialogs,
            "Export phased plot as PNG",
            "phased.png",
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
        """Render the current phased plot to an image file; returns the path written.

        ``.png`` is appended unless ``path`` already ends in an image suffix Qt can
        write. Raises :class:`OSError` if nothing could be written.
        """
        return write_plot_image(
            self._plot.getPlotItem(), path, scale=device_scale(self)
        )

    def _domain(self) -> Domain:
        bands = self._bands()
        if not bands:
            return Domain.MAGNITUDE
        return next(iter(bands.values())).domain

    def _show_period_text(self, text: str) -> None:
        """The period badge, hidden while there is no fold (an empty pill showed)."""
        self._period_label.setText(text)
        self._period_label.setVisible(bool(text))

    def clear(self) -> None:
        """Remove all folded points (e.g. when a new curve is being loaded)."""
        self._lc = None
        self._points = DisplayPoints(None)
        self._period = None
        self._rebuild_scatters()
        self._show_period_text("")
        self._update_accessible()
        self._points_label.setText("")

    def clear_fold(self) -> None:
        """Clear the current fold display (e.g. a compute finished with zero peaks).

        Unlike :meth:`clear`, the loaded light curve itself is left alone.
        """
        self._period = None
        self._t0 = 0.0
        for scatter in self._scatters.values():
            scatter.setData([], [])
        self._show_period_text("")
        self._update_accessible()

    # -- theme -------------------------------------------------------------------
    def apply_theme(self, theme_palette: ThemePalette) -> None:
        """Re-skin for a new theme."""
        self._pal = theme_palette
        apply_plot_theme(self._plot, theme_palette)
        self._style_legend()
        self._rebuild_scatters()
        self.refold()


__all__ = [
    "MAX_DRAWN_POINTS",
    "THINNED_TOOLTIP",
    "DisplayPoints",
    "PhasedView",
    "brightness_label",
    "curve_bands",
]
