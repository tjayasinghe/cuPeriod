"""The g-mode period-spacing explorer.

Given a frequency solution, this panel answers the two questions a γ Dor or SPB analysis
asks next: *is there a regular spacing?* and *which modes belong to it?* The comb
spectrum on the left scans trial spacings; the échelle diagram on the right folds the
member periods on the spacing that was found, where a clean series shows up as a
near-vertical ridge and the tilt from rotation as a slant.

It runs on demand rather than automatically — the search is only meaningful once the
extraction has finished, and only for g-mode pulsators — and always on the *independent*
components, so identified combination frequencies cannot pollute the pattern.
"""

from __future__ import annotations

import numpy as np

from cuperiod.core.config import SpacingSettings
from cuperiod.gui.qt import QtWidgets, pg
from cuperiod.gui.theme import ThemePalette, apply_plot_theme, palette, plot_label_style
from cuperiod.prewhiten.result import PreWhitenResult
from cuperiod.prewhiten.spacing import (
    echelle,
    find_period_spacing,
    spacing_spectrum,
)

#: Seconds per day, for reporting spacings in the units the literature uses.
_SECONDS_PER_DAY = 86400.0

#: Fewest independent modes worth running a comb search on.
_MIN_MODES = 4

#: How many multiples of the best spacing the comb plot frames after a search.
_COMB_VIEW_FACTOR = 6.0


class SpacingPanel(QtWidgets.QWidget):
    """Comb-spectrum scan, échelle diagram, and the fitted period-spacing series."""

    def __init__(
        self, theme: str = "dark", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._pal: ThemePalette = palette(theme)  # type: ignore[arg-type]
        self._result: PreWhitenResult | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        root.addWidget(self._build_controls())

        self._summary = QtWidgets.QLabel(
            "Run pre-whitening, then search for a spacing."
        )
        self._summary.setObjectName("muted")
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)

        splitter = QtWidgets.QSplitter()
        self._comb = pg.PlotWidget()
        self._comb.setMenuEnabled(False)
        self._comb.showGrid(x=True, y=True, alpha=0.18)
        style = plot_label_style(self._pal)
        self._comb.setLabel("bottom", "Trial spacing ΔP (s)", **style)
        self._comb.setLabel("left", "Comb response", **style)
        self._comb_curve = self._comb.plot(
            [], [], pen=pg.mkPen(self._pal.curve, width=1)
        )
        self._comb_marker = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(self._pal.qcolor(self._pal.accent, 200), width=2)
        )
        self._comb_marker.setVisible(False)
        self._comb.addItem(self._comb_marker)
        splitter.addWidget(self._comb)

        self._echelle = pg.PlotWidget()
        self._echelle.setMenuEnabled(False)
        self._echelle.showGrid(x=True, y=True, alpha=0.18)
        self._echelle.setLabel("bottom", "P mod ΔP (s)", **style)
        self._echelle.setLabel("left", "Period (d)", **style)
        self._echelle_points = pg.ScatterPlotItem(pxMode=True)
        self._echelle.addItem(self._echelle_points)
        splitter.addWidget(self._echelle)
        splitter.setSizes([500, 400])
        root.addWidget(splitter, 1)

        for plot in (self._comb, self._echelle):
            apply_plot_theme(plot, self._pal)
        self._set_enabled(False)

    # -- construction ------------------------------------------------------------
    def _build_controls(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        row.addWidget(QtWidgets.QLabel("ℓ:"))
        self._ell_spin = QtWidgets.QSpinBox()
        self._ell_spin.setRange(1, 4)
        self._ell_spin.setValue(1)
        self._ell_spin.setToolTip(
            "Spherical degree assumed when converting the mean spacing to a "
            "buoyancy radius Π₀ (dipole modes dominate γ Dor and SPB spectra)"
        )
        row.addWidget(self._ell_spin)

        row.addWidget(QtWidgets.QLabel("tolerance:"))
        self._tolerance_spin = QtWidgets.QDoubleSpinBox()
        self._tolerance_spin.setRange(0.01, 1.0)
        self._tolerance_spin.setSingleStep(0.05)
        self._tolerance_spin.setValue(SpacingSettings().tolerance)
        self._tolerance_spin.setToolTip(
            "How far an observed spacing may sit from the tilted model, as a fraction "
            "of the local spacing"
        )
        row.addWidget(self._tolerance_spin)

        row.addWidget(QtWidgets.QLabel("max gap:"))
        self._gap_spin = QtWidgets.QSpinBox()
        self._gap_spin.setRange(1, 8)
        self._gap_spin.setValue(SpacingSettings().max_gap)
        self._gap_spin.setToolTip(
            "Largest number of consecutive missing radial orders bridged in a series"
        )
        row.addWidget(self._gap_spin)

        self._search_btn = QtWidgets.QPushButton("Find period spacing")
        self._search_btn.setToolTip(
            "Scan for a regular period spacing among the independent components"
        )  # noqa: E501
        self._search_btn.clicked.connect(self.search)
        row.addWidget(self._search_btn)
        row.addStretch(1)
        return bar

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (
            self._search_btn,
            self._ell_spin,
            self._tolerance_spin,
            self._gap_spin,
        ):
            widget.setEnabled(enabled)

    # -- data --------------------------------------------------------------------
    def set_solution(self, result: PreWhitenResult) -> None:
        """Attach a new frequency solution (does not search until asked)."""
        self._result = result
        n = len(result.independent())
        self._set_enabled(n >= _MIN_MODES)
        self._clear_plots()
        if n < _MIN_MODES:
            self._summary.setText(
                f"Only {n} independent component{'' if n == 1 else 's'} — a period-"
                f"spacing search needs at least {_MIN_MODES}."
            )
        else:
            self._summary.setText(
                f"{n} independent components ready — press “Find period spacing”."
            )

    def clear(self) -> None:
        """Reset the panel (a new curve is loading, or the analysis changed)."""
        self._result = None
        self._set_enabled(False)
        self._clear_plots()
        self._summary.setText("Run pre-whitening, then search for a spacing.")

    def _clear_plots(self) -> None:
        self._comb_curve.setData([], [])
        self._comb_marker.setVisible(False)
        self._echelle_points.clear()

    # -- search ------------------------------------------------------------------
    def search(self) -> None:
        """Run the comb scan and the series extraction on the current solution."""
        result = self._result
        if result is None:
            return
        components = result.independent()
        if len(components) < _MIN_MODES:
            return
        periods = np.asarray([c.period for c in components], dtype=np.float64)
        amplitudes = np.asarray([c.amplitude for c in components], dtype=np.float64)
        settings = SpacingSettings(
            tolerance=self._tolerance_spin.value(),
            max_gap=self._gap_spin.value(),
            ell=self._ell_spin.value(),
        )
        try:
            comb = spacing_spectrum(
                periods,
                weights=amplitudes,
                minimum_spacing=settings.minimum_spacing,
                maximum_spacing=settings.maximum_spacing,
                oversample=settings.oversample,
            )
            series = find_period_spacing(periods, amplitudes, settings=settings)
        except ValueError as exc:
            self._summary.setText(f"Could not search for a spacing: {exc}")
            self._clear_plots()
            return

        best_seconds = comb.best_spacing * _SECONDS_PER_DAY
        self._comb_curve.setData(comb.spacing * _SECONDS_PER_DAY, comb.power)
        self._comb_marker.setPos(best_seconds)
        self._comb_marker.setVisible(True)
        self._comb.enableAutoRange(axis="y")
        # The trial range runs out to the full period span, which crushes the peak into
        # the left edge; frame it instead, and leave panning/zooming to the user.
        self._comb.setXRange(0.0, _COMB_VIEW_FACTOR * best_seconds, padding=0.02)

        if series is None:
            self._summary.setText(
                f"Comb peak at ΔP = {comb.best_spacing * _SECONDS_PER_DAY:.1f} s "
                f"(response {comb.best_power:.2f}), but no chain of at least "
                f"{settings.min_length} modes follows it — try a larger tolerance."
            )
            self._echelle_points.clear()
            return

        member = set(series.indices)
        spacing = series.mean_spacing
        x_all, y_all = echelle(periods, spacing)
        spots = [
            {
                "pos": (float(x * _SECONDS_PER_DAY), float(y)),
                "size": 12 if i in member else 8,
                "symbol": "o",
                "brush": pg.mkBrush(
                    self._pal.best_peak if i in member else self._pal.muted
                ),
                "pen": pg.mkPen(self._pal.plot_bg, width=1),
            }
            for i, (x, y) in enumerate(zip(x_all, y_all, strict=True))
        ]
        self._echelle_points.setData(spots)
        self._echelle.enableAutoRange()
        self._echelle.autoRange()
        self._summary.setText(
            f"<b>{series.n_modes}</b> of {len(components)} modes in one series  ·  "
            f"⟨ΔP⟩ = {spacing * _SECONDS_PER_DAY:.1f} s  ·  "
            f"slope {series.slope:+.4g}  ·  "
            f"rms {series.rms * _SECONDS_PER_DAY:.1f} s  ·  "
            f"Π₀(ℓ={series.ell}) = {series.buoyancy_radius:.0f} s"
        )

    # -- theme -------------------------------------------------------------------
    def apply_theme(self, theme_palette: ThemePalette) -> None:
        """Re-pen the plots for a new theme (live re-skin)."""
        self._pal = theme_palette
        self._comb_curve.setPen(pg.mkPen(theme_palette.curve, width=1))
        self._comb_marker.setPen(
            pg.mkPen(theme_palette.qcolor(theme_palette.accent, 200), width=2)
        )
        for plot in (self._comb, self._echelle):
            apply_plot_theme(plot, theme_palette)


__all__ = ["SpacingPanel"]
