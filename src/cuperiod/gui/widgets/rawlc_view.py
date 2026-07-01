"""The raw (unfolded) light-curve view — an optional panel for quick spot checks.

Plots brightness against time (days since the first sample) for the loaded curve,
overlaid per band for multiband data. Like the phased view it inverts the y-axis for
magnitudes and leaves flux as-is. It is static with respect to the selected period.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cuperiod.core.columns import Domain
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.gui.models import LoadedCurve
from cuperiod.gui.qt import QtWidgets, pg
from cuperiod.gui.theme import (
    ThemePalette,
    apply_plot_theme,
    palette,
    plot_label_style,
)
from cuperiod.gui.widgets.phased_view import brightness_label


class RawLightCurveView(QtWidgets.QWidget):
    """The unfolded time series (optional spot-check panel)."""

    def __init__(
        self, theme: str = "dark", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._pal: ThemePalette = palette(theme)  # type: ignore[arg-type]
        self._lc: LoadedCurve | None = None
        self._scatters: dict[str, Any] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        title = QtWidgets.QLabel("Raw light curve")
        title.setObjectName("muted")
        layout.addWidget(title)

        self._plot = pg.PlotWidget()
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        style = plot_label_style(self._pal)
        self._plot.setLabel("bottom", "Time (days since first sample)", **style)
        self._plot.setLabel("left", "Brightness", **style)
        self._legend = self._plot.addLegend(offset=(-10, 10))
        layout.addWidget(self._plot, 1)
        apply_plot_theme(self._plot, self._pal)

    def set_light_curve(self, lc: LoadedCurve) -> None:
        """Store and draw a new light curve."""
        self._lc = lc
        self._rebuild()

    def _bands(self) -> dict[str, LightCurve]:
        if isinstance(self._lc, MultiBandLightCurve):
            return dict(self._lc.bands)
        if isinstance(self._lc, LightCurve):
            return {"": self._lc}
        return {}

    def _rebuild(self) -> None:
        for scatter in self._scatters.values():
            self._plot.removeItem(scatter)
        self._scatters.clear()
        self._legend.clear()
        bands = self._bands()
        if not bands:
            return
        t_min = min(float(band.time.min()) for band in bands.values() if band.n)
        multi = len(bands) > 1
        default_color = self._pal.qcolor(self._pal.peak)
        for i, (name, band) in enumerate(bands.items()):
            color = self._pal.band_color(i) if multi else default_color
            scatter = pg.ScatterPlotItem(
                size=5, brush=pg.mkBrush(color), pen=None, pxMode=True
            )
            scatter.setData(np.asarray(band.time) - t_min, band.value)
            self._plot.addItem(scatter)
            if multi:
                self._legend.addItem(scatter, name)
            self._scatters[name] = scatter
        domain = next(iter(bands.values())).domain
        self._plot.getPlotItem().vb.invertY(domain is Domain.MAGNITUDE)
        self._plot.setLabel(
            "left", brightness_label(domain), **plot_label_style(self._pal)
        )
        self._plot.getPlotItem().vb.autoRange()

    def apply_theme(self, theme_palette: ThemePalette) -> None:
        """Re-skin for a new theme."""
        self._pal = theme_palette
        apply_plot_theme(self._plot, theme_palette)
        self._rebuild()


__all__ = ["RawLightCurveView"]
