"""The interactive spectrum view (pyqtgraph).

Renders the **full-resolution** spectrum (auto peak-preserving downsampling keeps
pan/zoom smooth on 10^5-10^6 points), overlays the significant peaks, and marks the
selection with a **translucent shaded band** drawn behind the curve — so the peak stays
visible — that is draggable and emits :attr:`period_selected` (snapped to the nearest
grid sample). A crosshair reads out frequency/period/power under the cursor.

The x-axis toggles frequency/period (arrays are reversed in period mode so x stays
ascending, as pyqtgraph's clip/downsample require) and each axis toggles linear/log.
Display adapts to the objective sense: for minimise methods (PDM/CE/string-length) the
peaks are minima and the y-axis is labelled accordingly.

The same view serves pre-whitening: the main curve becomes the amplitude spectrum of
the data, an optional **overlay** curve shows the spectrum of the residuals once every
extracted component has been subtracted, and the peak markers become the components.
That side-by-side is the whole point of the method — what was there, and what is left.
"""

from __future__ import annotations

import csv
from typing import Any

import numpy as np
from pyqtgraph.exporters import ImageExporter

from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.qt import Qt, QtWidgets, Signal, pg
from cuperiod.gui.theme import (
    ThemePalette,
    apply_plot_theme,
    palette,
    plot_label_style,
)

_MIN_OBJECTIVE_LABEL = "dispersion (lower = better)"

#: Method name carried by the synthetic Periodogram that wraps an amplitude spectrum.
PREWHITEN_METHOD = "Pre-whitening"


class SpectrumView(QtWidgets.QWidget):
    """Full-resolution spectrum with peak markers, a selection band, and a crosshair."""

    period_selected = Signal(float)

    def __init__(
        self, theme: str = "dark", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._pal: ThemePalette = palette(theme)  # type: ignore[arg-type]
        self._pg: Periodogram | None = None
        self._peaks: list[Peak] = []
        self._x_mode = "frequency"
        self._sel_period: float | None = None
        self._suppress_band = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._build_toolbar())

        self._plot = pg.PlotWidget()
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        style = plot_label_style(self._pal)
        self._plot.setLabel("bottom", "Frequency (cycles/day)", **style)
        self._plot.setLabel("left", "Power", **style)
        layout.addWidget(self._plot, 1)

        # Translucent selection band, behind the curve so the peak shows through.
        self._sel_band = pg.LinearRegionItem(
            movable=True,
            brush=pg.mkBrush(self._pal.qcolor(self._pal.accent, 45)),
            pen=pg.mkPen(self._pal.qcolor(self._pal.accent, 170), width=1),
        )
        self._sel_band.setZValue(-20)
        self._sel_band.sigRegionChangeFinished.connect(self._on_band_moved)
        self._plot.addItem(self._sel_band)
        self._sel_band.setVisible(False)

        self._curve = self._plot.plot([], [], pen=pg.mkPen(self._pal.curve, width=1))
        self._curve.setDownsampling(auto=True, method="peak")
        self._curve.setClipToView(True)
        self._curve.setZValue(0)

        # Optional second trace (the pre-whitened residual spectrum), drawn over the
        # main curve so what is *left* stays readable against what was there.
        self._overlay = self._plot.plot(
            [], [], pen=pg.mkPen(self._pal.qcolor(self._pal.accent, 220), width=1)
        )
        self._overlay.setDownsampling(auto=True, method="peak")
        self._overlay.setClipToView(True)
        self._overlay.setZValue(1)
        self._overlay.setVisible(False)
        self._overlay_xy: tuple[np.ndarray, np.ndarray] | None = None

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

        self._proxy = pg.SignalProxy(
            self._plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )
        self._plot.getPlotItem().vb.sigXRangeChanged.connect(self._on_xrange_changed)
        apply_plot_theme(self._plot, self._pal)

    # -- toolbar -----------------------------------------------------------------
    def _build_toolbar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(8)

        row.addWidget(QtWidgets.QLabel("x:"))
        self._xaxis_combo = QtWidgets.QComboBox()
        self._xaxis_combo.addItems(["frequency", "period"])
        self._xaxis_combo.setToolTip("Plot the x-axis as frequency or period")
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
        self._show_peaks.toggled.connect(self._on_peaks_toggled)
        row.addWidget(self._show_peaks)

        self._show_residual = QtWidgets.QCheckBox("residual")
        self._show_residual.setChecked(True)
        self._show_residual.setToolTip(
            "Overlay the amplitude spectrum of the residuals after pre-whitening"
        )
        self._show_residual.setVisible(False)
        self._show_residual.toggled.connect(self._redraw_overlay)
        row.addWidget(self._show_residual)

        reset = QtWidgets.QPushButton("Reset view")
        reset.setToolTip("Auto-range the plot back to the full spectrum")
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

        row.addStretch(1)
        self._readout = QtWidgets.QLabel("—")
        self._readout.setObjectName("muted")
        row.addWidget(self._readout)
        return bar

    # -- data --------------------------------------------------------------------
    def set_periodogram(self, pg_result: Periodogram) -> None:
        """Show a new spectrum (clears peaks, the overlay, and the selection band)."""
        self._pg = pg_result
        self._peaks = []
        self._sel_period = None
        self._markers.clear()
        self._sel_band.setVisible(False)
        self._overlay_xy = None
        self._show_residual.setVisible(False)
        self._default_axis_for(pg_result)
        self._redraw_curve()
        self._update_y_label()
        self._autorange()

    def set_overlay(self, frequency: np.ndarray, values: np.ndarray) -> None:
        """Add a second trace on the same grid (the pre-whitened residual spectrum)."""
        self._overlay_xy = (
            np.asarray(frequency, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
        self._show_residual.setVisible(True)
        self._redraw_overlay()

    def clear_overlay(self) -> None:
        """Remove the second trace and hide its toggle."""
        self._overlay_xy = None
        self._show_residual.setVisible(False)
        self._overlay.setVisible(False)
        self._overlay.setData([], [])

    def set_peaks(self, peaks: list[Peak]) -> None:
        """Overlay the significant peaks as markers."""
        self._peaks = list(peaks)
        self._redraw_markers()

    def set_selected_period(self, period: float) -> None:
        """Highlight ``period`` with the shaded band (no signal echo)."""
        if self._pg is None or not np.isfinite(period) or period <= 0.0:
            return
        self._sel_period = period
        self._sel_band.setVisible(True)
        self._update_band()
        self._redraw_markers()  # re-place the selected-peak halo

    def clear(self) -> None:
        """Clear the spectrum, peaks, and selection (e.g. when a new curve loads)."""
        self._pg = None
        self._peaks = []
        self._sel_period = None
        self._curve.setData([], [])
        self._markers.clear()
        self._sel_band.setVisible(False)
        self._hover_text.setVisible(False)
        self._readout.setText("—")
        self.clear_overlay()

    def clear_selection(self) -> None:
        """Hide the selection band (e.g. a compute finished with zero peaks).

        Unlike :meth:`clear`, the spectrum curve/peaks themselves are left alone.
        """
        self._sel_period = None
        self._sel_band.setVisible(False)
        self._redraw_markers()

    # -- drawing -----------------------------------------------------------------
    def _curve_xy(self) -> tuple[np.ndarray, np.ndarray]:
        assert self._pg is not None
        if self._x_mode == "period":
            return self._pg.period[::-1], self._pg.power[::-1]
        return self._pg.frequency, self._pg.power

    def _redraw_curve(self) -> None:
        if self._pg is None:
            self._curve.setData([], [])
            return
        x, y = self._curve_xy()
        self._curve.setData(x, y)
        self._redraw_overlay()
        self._apply_log()

    def _redraw_overlay(self) -> None:
        """Re-place the second trace for the current x mode, or hide it."""
        if self._overlay_xy is None or not self._show_residual.isChecked():
            self._overlay.setVisible(False)
            return
        frequency, values = self._overlay_xy
        if self._x_mode == "period":
            with np.errstate(divide="ignore"):
                x = (1.0 / frequency)[::-1]
            y = values[::-1]
        else:
            x, y = frequency, values
        self._overlay.setData(x, y)
        self._overlay.setVisible(True)

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
        if self._pg is None or not self._show_peaks.isChecked():
            self._markers.clear()
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
            pen = pg.mkPen(self._pal.plot_bg, width=1)
            if is_selected:
                size += 4
                pen = pg.mkPen(self._pal.qcolor(self._pal.accent), width=2)
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

    def _update_y_label(self) -> None:
        if self._pg is None:
            return
        style = plot_label_style(self._pal)
        if self._pg.method == PREWHITEN_METHOD:
            text = "Amplitude"
        elif self._pg.objective_sense == "min":
            text = f"{self._pg.method} {_MIN_OBJECTIVE_LABEL}"
        else:
            text = f"{self._pg.method} power"
        self._plot.setLabel("left", text, **style)

    def _default_axis_for(self, pg_result: Periodogram) -> None:
        # Box/transit searches read most naturally on a *log period* axis (their period
        # grids are log-spaced), so default BLS/TLS to period + log-x.
        is_box = pg_result.method in {"BLS", "TLS"}
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
        self._plot.setLogMode(x=self._logx.isChecked(), y=self._logy.isChecked())
        self._update_band()
        self._redraw_markers()  # markers live in plot coords; re-place on log change

    def _on_peaks_toggled(self, _checked: bool) -> None:
        self._redraw_markers()

    def _autorange(self) -> None:
        # Disable clip-to-view first: with it on, auto-ranging after a data-range change
        # fits only the previously-visible slice, so the view sticks (e.g. a BLS peak
        # ends up off-screen). Re-enable it once the view spans the full data.
        self._curve.setClipToView(False)
        self._plot.enableAutoRange()
        self._plot.autoRange()
        self._curve.setClipToView(True)

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

    # -- conversions -------------------------------------------------------------
    def _x_for_period(self, period: float) -> float:
        if self._x_mode == "period":
            return period
        return 1.0 / period if period > 0.0 else 0.0

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
        idx = pts[0].data()
        if idx is None or idx >= len(self._peaks):
            return
        period = self._peaks[int(idx)].period
        self.set_selected_period(period)
        self.period_selected.emit(period)

    def _on_peak_hovered(self, _item: Any, points: Any) -> None:
        pts = list(points) if points is not None else []
        if not pts:
            self._hover_text.setVisible(False)
            return
        idx = pts[0].data()
        if idx is None or idx >= len(self._peaks):
            return
        peak = self._peaks[int(idx)]
        x = peak.frequency if self._x_mode == "frequency" else peak.period
        self._hover_text.setText(
            f"#{peak.rank}  P={peak.period:.6g} d\npower={peak.power:.4g}"
        )
        self._hover_text.setPos(self._to_plot_x(x), self._to_plot_y(peak.power))
        self._hover_text.setVisible(True)

    def _on_mouse_moved(self, event: Any) -> None:
        if self._pg is None:
            return
        pos = event[0]
        view = self._plot.getPlotItem().vb
        if not self._plot.sceneBoundingRect().contains(pos):
            self._vline.setVisible(False)
            self._hline.setVisible(False)
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
            f"f = {freq:.6g} /d    P = {period:.6g} d    power = {y:.4g}"
        )

    def _mouse_xy(self, point: Any) -> tuple[float, float]:
        # In log mode the view coordinates are log10 of the data values.
        x = 10.0 ** point.x() if self._logx.isChecked() else point.x()
        y = 10.0 ** point.y() if self._logy.isChecked() else point.y()
        return float(x), float(y)

    # -- export --------------------------------------------------------------
    def _export_csv(self) -> None:
        if self._pg is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export spectrum as CSV", "spectrum.csv", "CSV files (*.csv)"
        )
        if path:
            self.export_csv(path)

    def export_csv(self, path: str) -> None:
        """Write the full-resolution spectrum (frequency, period, power) to ``path``."""
        assert self._pg is not None
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frequency", "period", "power"])
            for freq, period, power in zip(
                self._pg.frequency, self._pg.period, self._pg.power, strict=True
            ):
                writer.writerow([freq, period, power])

    def _export_png(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export spectrum as PNG", "spectrum.png", "PNG images (*.png)"
        )
        if path:
            self.export_png(path)

    def export_png(self, path: str) -> None:
        """Render the current plot to ``path`` as a PNG image."""
        exporter = ImageExporter(self._plot.getPlotItem())
        exporter.export(path)

    # -- theme -------------------------------------------------------------------
    def _crosshair_pen(self) -> object:
        return pg.mkPen(self._pal.muted, width=1, style=Qt.PenStyle.DashLine)

    def apply_theme(self, theme_palette: ThemePalette) -> None:
        """Re-pen the plot items for a new theme (live re-skin)."""
        self._pal = theme_palette
        self._curve.setPen(pg.mkPen(theme_palette.curve, width=1))
        self._overlay.setPen(
            pg.mkPen(theme_palette.qcolor(theme_palette.accent, 220), width=1)
        )
        band_brush = pg.mkBrush(theme_palette.qcolor(theme_palette.accent, 45))
        self._sel_band.setBrush(band_brush)
        band_pen = pg.mkPen(theme_palette.qcolor(theme_palette.accent, 170), width=1)
        for line in self._sel_band.lines:
            line.setPen(band_pen)
        self._vline.setPen(self._crosshair_pen())
        self._hline.setPen(self._crosshair_pen())
        self._hover_text.setColor(theme_palette.plot_fg)
        apply_plot_theme(self._plot, theme_palette)
        self._redraw_markers()


__all__ = ["PREWHITEN_METHOD", "SpectrumView"]
