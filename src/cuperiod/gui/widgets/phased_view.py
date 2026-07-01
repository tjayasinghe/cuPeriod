"""The phased light-curve view (pyqtgraph).

Shows the light curve folded at the selected period; it updates live from the
controller's fold signal by re-running only the O(n) fold, never the periodogram.
Multiband curves overlay one colour per band with a legend. Magnitude curves invert the
y-axis (brighter up); flux curves (e.g. a BLS/TLS transit) stay as dips. The "2 cycles"
toggle repeats the fold over ``[0, 2)``, the usual convention for reading shape.
"""

from __future__ import annotations

from typing import Any

from cuperiod.core.columns import Domain
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.gui.fold import fold_series
from cuperiod.gui.models import LoadedCurve
from cuperiod.gui.qt import QtWidgets, Signal, pg
from cuperiod.gui.theme import (
    ThemePalette,
    apply_plot_theme,
    palette,
    plot_label_style,
)


def brightness_label(domain: Domain) -> str:
    """Y-axis label for a light-curve domain (flux is unit-agnostic in cuPeriod)."""
    return "Magnitude" if domain is Domain.MAGNITUDE else "Flux (arbitrary units)"


class PhasedView(QtWidgets.QWidget):
    """The folded light curve, updated live as the selected period changes."""

    show_raw_toggled = Signal(bool)

    def __init__(
        self, theme: str = "dark", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._pal: ThemePalette = palette(theme)  # type: ignore[arg-type]
        self._lc: LoadedCurve | None = None
        self._period: float | None = None
        self._t0 = 0.0
        self._scatters: dict[str, Any] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        title = QtWidgets.QLabel("Phased light curve")
        title.setObjectName("muted")
        row.addWidget(title)
        row.addStretch(1)
        self._period_label = QtWidgets.QLabel("")
        self._period_label.setObjectName("badgeAccent")
        row.addWidget(self._period_label)
        self._show_raw = QtWidgets.QCheckBox("Show Raw LC")
        self._show_raw.setToolTip("Show the raw (unfolded) light-curve panel")
        self._show_raw.toggled.connect(self.show_raw_toggled)
        row.addWidget(self._show_raw)
        self._two_cycles = QtWidgets.QCheckBox("2 cycles")
        self._two_cycles.setChecked(True)  # two cycles is the default view
        self._two_cycles.toggled.connect(self.refold)
        row.addWidget(self._two_cycles)
        layout.addWidget(bar)

        self._plot = pg.PlotWidget()
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        style = plot_label_style(self._pal)
        self._plot.setLabel("bottom", "Phase", **style)
        self._plot.setLabel("left", "Brightness", **style)
        self._legend = self._plot.addLegend(offset=(-10, 10))
        layout.addWidget(self._plot, 1)
        apply_plot_theme(self._plot, self._pal)

    # -- data --------------------------------------------------------------------
    def set_light_curve(self, lc: LoadedCurve) -> None:
        """Store a new light curve and rebuild the per-band scatters."""
        self._lc = lc
        self._rebuild_scatters()
        self.refold()

    def set_fold(self, period: float, t0: float) -> None:
        """Fold at ``period`` with epoch ``t0`` (from the controller)."""
        self._period = period
        self._t0 = t0
        self.refold()

    def refold(self) -> None:
        """Re-fold every band at the current period (O(n); no periodogram re-run)."""
        if self._lc is None or self._period is None:
            return
        two = self._two_cycles.isChecked()
        for name, band in self._bands().items():
            scatter = self._scatters.get(name)
            if scatter is None:
                continue
            phase, value = fold_series(
                band.time, band.value, self._period, self._t0, two_cycles=two
            )
            scatter.setData(phase, value)
        self._period_label.setText(f"P = {self._period:.6g} d")
        self._plot.getPlotItem().vb.autoRange()

    # -- helpers -----------------------------------------------------------------
    def _bands(self) -> dict[str, LightCurve]:
        if isinstance(self._lc, MultiBandLightCurve):
            return dict(self._lc.bands)
        if isinstance(self._lc, LightCurve):
            return {"": self._lc}
        return {}

    def _rebuild_scatters(self) -> None:
        for scatter in self._scatters.values():
            self._plot.removeItem(scatter)
        self._scatters.clear()
        self._legend.clear()
        bands = self._bands()
        multi = len(bands) > 1
        default_color = self._pal.qcolor(self._pal.peak)
        for i, (name, _band) in enumerate(bands.items()):
            color = self._pal.band_color(i) if multi else default_color
            scatter = pg.ScatterPlotItem(
                size=6, brush=pg.mkBrush(color), pen=None, pxMode=True
            )
            self._plot.addItem(scatter)
            if multi:
                self._legend.addItem(scatter, name)
            self._scatters[name] = scatter
        domain = self._domain()
        self._plot.getPlotItem().vb.invertY(domain is Domain.MAGNITUDE)
        self._plot.setLabel(
            "left", brightness_label(domain), **plot_label_style(self._pal)
        )

    def _domain(self) -> Domain:
        bands = self._bands()
        if not bands:
            return Domain.MAGNITUDE
        return next(iter(bands.values())).domain

    def clear(self) -> None:
        """Remove all folded points (e.g. when a new curve is being loaded)."""
        self._lc = None
        self._period = None
        self._rebuild_scatters()
        self._period_label.setText("")

    # -- theme -------------------------------------------------------------------
    def apply_theme(self, theme_palette: ThemePalette) -> None:
        """Re-skin for a new theme."""
        self._pal = theme_palette
        apply_plot_theme(self._plot, theme_palette)
        self._rebuild_scatters()
        self.refold()


__all__ = ["PhasedView"]
