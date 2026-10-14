"""The g-mode period-spacing explorer.

Given a frequency solution, this panel answers the two questions a γ Dor or SPB analysis
asks next: *is there a regular spacing?* and *which modes belong to it?* The comb
spectrum on the left scans trial spacings; the échelle diagram on the right folds the
member periods on the spacing that was found, where a clean series shows up as a
near-vertical ridge and the tilt from rotation as a slant.

It runs on demand rather than automatically — the search is only meaningful once the
extraction has finished, and only for g-mode pulsators — and always on the *independent*
components, so identified combination frequencies cannot pollute the pattern. A theme
switch redraws the échelle points in the new palette and keeps the user's zoom.
"""

from __future__ import annotations

import html

import numpy as np

from cuperiod.core.config import SpacingSettings
from cuperiod.gui.qt import Qt, QtCore, QtWidgets, pg
from cuperiod.gui.theme import ThemePalette, apply_plot_theme, palette, plot_label_style
from cuperiod.gui.widgets._view_helpers import stroke_width
from cuperiod.gui.widgets.info_bar import FlowLayout
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


def _labelled(text: str, field: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """``text`` beside ``field`` (its buddy), wrapping as one."""
    box = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    label = QtWidgets.QLabel(text)
    label.setBuddy(field)
    row.addWidget(label)
    row.addWidget(field)
    return box


class SpacingPanel(QtWidgets.QWidget):
    """Comb-spectrum scan, échelle diagram, and the fitted period-spacing series."""

    def __init__(
        self, theme: str = "dark", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._pal: ThemePalette = palette(theme)  # type: ignore[arg-type]
        self._result: PreWhitenResult | None = None
        #: The last échelle drawn: x (s), y (d) and the indices in the series. Kept
        #: so a theme switch can redraw the points in the new palette.
        self._echelle_data: tuple[np.ndarray, np.ndarray, frozenset[int]] | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        root.addWidget(self._build_controls())

        self._summary = QtWidgets.QLabel(
            "Run pre-whitening, then search for a spacing."
        )
        self._summary.setObjectName("muted")
        self._summary.setWordWrap(True)
        # Always rich text: the error branch escapes the exception text instead of
        # leaving Qt to guess whether "<" in it starts a tag.
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self._summary)

        splitter = QtWidgets.QSplitter()
        self._comb = pg.PlotWidget()
        self._comb.setMenuEnabled(False)
        self._comb.showGrid(x=True, y=True, alpha=0.18)
        style = plot_label_style(self._pal)
        self._comb.setLabel("bottom", "Trial spacing ΔP (s)", **style)
        self._comb.setLabel("left", "Comb response", **style)
        self._comb_curve = self._comb.plot([], [])
        self._comb_marker = pg.InfiniteLine(angle=90)
        self._repen()
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
        # A wrapping row: in one line the controls made the pane 632 px wide, so
        # in a 400 px dock "Find period spacing" sat off to the right. Each field
        # is named (screen readers announced three anonymous spin boxes) and its
        # label is its buddy.
        bar = QtWidgets.QWidget()
        row = FlowLayout(hspacing=8, vspacing=4)
        row.setContentsMargins(0, 0, 0, 0)
        bar.setLayout(row)

        self._ell_spin = QtWidgets.QSpinBox()
        self._ell_spin.setRange(1, 4)
        self._ell_spin.setValue(1)
        self._name(
            self._ell_spin,
            "Spherical degree ℓ",
            "Spherical degree assumed when converting the mean spacing to a "
            "buoyancy radius Π₀ (dipole modes dominate γ Dor and SPB spectra)",
        )
        row.addWidget(_labelled("ℓ:", self._ell_spin))

        self._tolerance_spin = QtWidgets.QDoubleSpinBox()
        self._tolerance_spin.setRange(0.01, 1.0)
        self._tolerance_spin.setSingleStep(0.05)
        self._tolerance_spin.setValue(SpacingSettings().tolerance)
        self._name(
            self._tolerance_spin,
            "Spacing tolerance",
            "How far an observed spacing may sit from the tilted model, as a fraction "
            "of the local spacing",
        )
        row.addWidget(_labelled("tolerance:", self._tolerance_spin))

        self._gap_spin = QtWidgets.QSpinBox()
        self._gap_spin.setRange(1, 8)
        self._gap_spin.setValue(SpacingSettings().max_gap)
        self._name(
            self._gap_spin,
            "Maximum gap in radial order",
            "Largest number of consecutive missing radial orders bridged in a series",
        )
        row.addWidget(_labelled("max gap:", self._gap_spin))

        self._search_btn = QtWidgets.QPushButton("Find period spacing")
        self._name(
            self._search_btn,
            "Find period spacing",
            "Scan for a regular period spacing among the independent components",
        )
        self._search_btn.clicked.connect(self.search)
        row.addWidget(self._search_btn)
        return bar

    @staticmethod
    def _name(widget: QtWidgets.QWidget, name: str, tip: str) -> None:
        widget.setAccessibleName(name)
        widget.setToolTip(tip)
        widget.setStatusTip(tip)

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
        self._echelle_data = None
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
            message = html.escape(str(exc), quote=False)
            self._summary.setText(f"Could not search for a spacing: {message}")
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
            self._echelle_data = None
            self._echelle_points.clear()
            return

        spacing = series.mean_spacing
        x_all, y_all = echelle(periods, spacing)
        self._echelle_data = (
            np.asarray(x_all, dtype=np.float64) * _SECONDS_PER_DAY,
            np.asarray(y_all, dtype=np.float64),
            frozenset(int(i) for i in series.indices),
        )
        self._draw_echelle()
        self._echelle.enableAutoRange()
        self._echelle.autoRange()
        self._summary.setText(
            f"<b>{series.n_modes}</b> of {len(components)} modes in one series  ·  "
            f"⟨ΔP⟩ = {spacing * _SECONDS_PER_DAY:.1f} s  ·  "
            f"slope {series.slope:+.4g}  ·  "
            f"rms {series.rms * _SECONDS_PER_DAY:.1f} s  ·  "
            f"Π₀(ℓ={series.ell}) = {series.buoyancy_radius:.0f} s"
        )

    def _draw_echelle(self) -> None:
        """Draw the last échelle in the current palette (never touches the view)."""
        if self._echelle_data is None:
            self._echelle_points.clear()
            return
        x_seconds, y_days, member = self._echelle_data
        spots = [
            {
                "pos": (float(x), float(y)),
                "size": 12 if i in member else 8,
                "symbol": "o",
                "brush": pg.mkBrush(
                    self._pal.best_peak if i in member else self._pal.muted
                ),
                "pen": pg.mkPen(self._pal.plot_bg, width=stroke_width(self)),
            }
            for i, (x, y) in enumerate(zip(x_seconds, y_days, strict=True))
        ]
        self._echelle_points.setData(spots)

    # -- theme -------------------------------------------------------------------
    def _repen(self) -> None:
        """Pen the comb curve and marker for the palette and device-pixel ratio.

        pyqtgraph pens are cosmetic (device pixels), so the widths are scaled to stay
        1 and 2 *logical* pixels wide on a HiDPI screen.
        """
        self._comb_curve.setPen(pg.mkPen(self._pal.curve, width=stroke_width(self)))
        self._comb_marker.setPen(
            pg.mkPen(
                self._pal.qcolor(self._pal.accent, 200), width=stroke_width(self, 2.0)
            )
        )

    def event(self, event: QtCore.QEvent) -> bool:
        # Moved to a screen with another scale factor: keep the logical stroke widths.
        if event.type() == QtCore.QEvent.Type.DevicePixelRatioChange:
            self._repen()
            self._draw_echelle()
        return super().event(event)

    def apply_theme(self, theme_palette: ThemePalette) -> None:
        """Re-pen the plots for a new theme (live re-skin), keeping the user's zoom."""
        self._pal = theme_palette
        self._repen()
        for plot in (self._comb, self._echelle):
            apply_plot_theme(plot, theme_palette)
        # The échelle spots carry their colours, so they are rebuilt — otherwise they
        # kept the old theme's brushes and an outline in the old background colour.
        self._draw_echelle()


__all__ = ["SpacingPanel"]
