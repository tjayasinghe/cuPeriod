"""Qt-driven tests (offscreen): form, compute manager, controller, and spectrum."""

from __future__ import annotations

import stat
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtTest
from pytestqt.qtbot import QtBot

from cuperiod.core.config import (
    BLSSettings,
    GLSSettings,
    MHAOVSettings,
    PDMSettings,
    PreWhitenSettings,
)
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.compute import ComputeManager
from cuperiod.gui.models import LoadedCurve, ResultKey, SourceItem, settings_hash
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets
from cuperiod.gui.settingsform import PydanticSettingsForm
from cuperiod.gui.state import AppController
from cuperiod.gui.theme import palette
from cuperiod.gui.widgets.peaks_panel import PeaksTable
from cuperiod.gui.widgets.phased_view import PhasedView
from cuperiod.gui.widgets.rawlc_view import RawLightCurveView
from cuperiod.gui.widgets.spectrum_view import PREWHITEN_METHOD, SpectrumView


def _light_curve() -> LightCurve:
    rng = np.random.default_rng(0)
    time = np.sort(rng.uniform(0.0, 40.0, 200))
    value = 14.0 + 0.4 * np.sin(2 * np.pi * time / 2.0) + rng.normal(0.0, 0.02, 200)
    return LightCurve.from_arrays(time, value, np.full(200, 0.02))


def _bump_periodogram() -> Periodogram:
    freq = np.linspace(0.1, 2.0, 500)
    power = np.exp(-((freq - 0.5) ** 2) / 0.001)
    return Periodogram.from_spectrum(
        method="GLS",
        backend="numpy",
        frequency=freq,
        power=power,
        objective_sense="max",
        n_samples=100,
        baseline=90.0,
    )


def test_optional_spin_shows_auto_value(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(GLSSettings)
    qtbot.addWidget(form)
    form.set_auto_value("maximum_frequency", 10.0)
    opt = form._optional["maximum_frequency"]
    assert abs(opt._spin.value() - 10.0) < 1e-6  # displayed in the greyed box
    assert opt.value() is None  # still auto -> contributes None to the settings


def test_period_axis_defaults_to_log(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.set_periodogram(_bump_periodogram())
    view._xaxis_combo.setCurrentText("period")
    assert view._logx.isChecked()
    view._xaxis_combo.setCurrentText("frequency")
    assert not view._logx.isChecked()


def test_settings_form_build(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(GLSSettings)
    qtbot.addWidget(form)
    settings = form.build()
    assert isinstance(settings, GLSSettings)
    assert settings.minimum_frequency is None  # "auto" checkbox contributes None


def test_compute_manager_emits_result(qtbot: QtBot) -> None:
    manager = ComputeManager()
    key = ResultKey("s", "GLS", settings_hash(GLSSettings()), "auto")
    with qtbot.waitSignal(manager.signals.finished, timeout=30000) as blocker:
        manager.submit(key, _light_curve(), "GLS", GLSSettings(), "auto")
    got_key, pg, elapsed_ms = blocker.args
    assert got_key == key
    assert isinstance(pg, Periodogram)
    assert pg.size > 0
    assert elapsed_ms >= 0.0


def test_controller_cache_hit_is_synchronous(qtbot: QtBot) -> None:
    controller = AppController()
    controller.set_light_curve(_light_curve(), "src")
    with qtbot.waitSignal(controller.periodogram_ready, timeout=30000):
        controller.run("GLS", "auto", GLSSettings(), 10)
    # a second identical run hits the cache and emits without the event loop
    got: list[object] = []
    controller.periodogram_ready.connect(got.append)
    controller.run("GLS", "auto", GLSSettings(), 10)
    assert len(got) == 1


def test_auto_grid_tuning_for_sparse_data(qtbot: QtBot) -> None:
    controller = AppController()
    # sparse, long-baseline sampling -> low pseudo-Nyquist
    time = np.sort(np.random.default_rng(0).uniform(0.0, 2000.0, 200))
    lc = LightCurve.from_arrays(time, np.sin(time), np.full(200, 0.01))

    # The frequency-domain analyses share the delta Scuti / HADS floor (50 c/d): a
    # trial frequency costs them a trig sum, and the 10 c/d ceiling put the bundled
    # HADS demo (f = 11.14 c/d) out of band, aliasing it to a wrong period.
    for settings in (GLSSettings(), MHAOVSettings(), PreWhitenSettings()):
        tuned = controller._tune_auto_grid(lc, settings)
        assert tuned.maximum_frequency is not None
        assert tuned.maximum_frequency >= 50.0, type(settings).__name__
    assert controller._tune_auto_grid(lc, GLSSettings()).samples_per_peak >= 10

    # Fold-based methods pay a full fold per trial frequency and keep the 10 c/d
    # floor; their sub-day coverage is unchanged.
    pdm = controller._tune_auto_grid(lc, PDMSettings())
    assert pdm.maximum_frequency is not None
    assert pdm.maximum_frequency == 10.0

    # a method without frequency-grid fields (BLS) is returned unchanged
    bls = BLSSettings()
    assert controller._tune_auto_grid(lc, bls) is bls


def test_spectrum_placeholder_sits_in_the_plot_until_a_spectrum(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.resize(700, 400)
    view.show()
    qtbot.waitExposed(view)
    text = "Spectrum unavailable: no usable time baseline"
    view.set_placeholder(text)
    label = view._placeholder
    assert view.placeholder() == text and label.isVisible() and label.text() == text
    assert label.parentWidget() is view._plot  # inside the plot, over it
    assert view._plot.rect().contains(label.geometry())
    assert label.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert label.textFormat() == Qt.TextFormat.PlainText  # file text stays literal
    assert view._plot.accessibleDescription() == text
    view.resize(500, 300)  # follows the plot's size
    qtbot.wait(20)
    assert view._plot.rect().contains(label.geometry())
    view.apply_theme(palette("light"))
    assert palette("light").plot_fg in label.styleSheet()
    view.set_periodogram(_bump_periodogram())  # a spectrum: the text goes
    assert view.placeholder() == "" and not label.isVisible()
    view.set_placeholder(text)
    view.clear()
    assert view.placeholder() == "" and not label.isVisible()
    view.set_placeholder(text)
    view.set_placeholder(None)
    assert not label.isVisible()


def test_spectrum_axis_toggle_keeps_ascending(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    pg = _bump_periodogram()
    view.set_periodogram(pg)
    view.set_peaks(pg.best_periods(5))

    freq_x, _ = view._curve.getData()
    assert np.all(np.diff(freq_x) >= 0.0)  # ascending in frequency mode
    view._xaxis_combo.setCurrentText("period")
    period_x, _ = view._curve.getData()
    assert np.all(np.diff(period_x) >= 0.0)  # still ascending in period mode


def test_spectrum_markers_follow_log_mode(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    pg = _bump_periodogram()
    view.set_periodogram(pg)
    view.set_peaks(pg.best_periods(3))
    view._xaxis_combo.setCurrentText("period")
    view._logx.setChecked(True)
    # markers must be placed at log10(period) so they align with the log-scaled curve
    for spot in view._markers.points():
        idx = spot.data()
        if idx is None:  # the glow halo
            continue
        expected = np.log10(view._peaks[int(idx)].period)
        assert abs(spot.pos().x() - expected) < 1e-6


def test_select_source_load_failure_reverts_index_and_names_source(
    qtbot: QtBot,
) -> None:
    controller = AppController()
    good = SourceItem(key="good", label="Good Star", loader=_light_curve)

    def _boom() -> LoadedCurve:
        raise ValueError("bad file")

    bad = SourceItem(key="bad", label="Bad Star", loader=_boom)
    controller.load_sources([good, bad])
    controller.select_source(0)
    assert controller.state.current_source_index == 0

    failures: list[str] = []
    controller.compute_failed.connect(failures.append)
    reverted: list[int] = []
    controller.source_selected.connect(reverted.append)
    controller.select_source(1)

    # index/state stayed on the previously-loaded (good) source, not the failed one
    assert controller.state.current_source_index == 0
    assert reverted[-1] == 0  # browser highlight snaps back to the previous source
    assert len(failures) == 1
    assert "Bad Star" in failures[0]  # failure message names the failed source


def test_select_source_success_after_prior_failure(qtbot: QtBot) -> None:
    controller = AppController()
    good = SourceItem(key="good", label="Good Star", loader=_light_curve)

    def _boom() -> LoadedCurve:
        raise ValueError("bad file")

    bad = SourceItem(key="bad", label="Bad Star", loader=_boom)
    controller.load_sources([bad, good])
    controller.select_source(0)  # fails; index stays at the default 0
    assert controller.state.current_source_index == 0

    controller.select_source(1)  # succeeds
    assert controller.state.current_source_index == 1
    assert controller.state.current_lc is not None


def test_zero_peaks_clears_stale_spectrum_and_phased_selection(qtbot: QtBot) -> None:
    controller = AppController()
    lc = _light_curve()
    controller.set_light_curve(lc, "src")

    spectrum = SpectrumView("dark")
    qtbot.addWidget(spectrum)
    phased = PhasedView("dark")
    qtbot.addWidget(phased)
    controller.periodogram_ready.connect(spectrum.set_periodogram)
    controller.peaks_ready.connect(spectrum.set_peaks)
    controller.period_changed.connect(spectrum.set_selected_period)
    controller.selection_cleared.connect(spectrum.clear_selection)
    controller.fold_changed.connect(phased.set_fold)
    controller.selection_cleared.connect(phased.clear_fold)
    phased.set_light_curve(lc)

    # first, a normal run with a real peak selects something in both views
    with qtbot.waitSignal(controller.periodogram_ready, timeout=30000):
        controller.run("GLS", "auto", GLSSettings(), 5)
    assert spectrum._sel_period is not None
    assert phased._period is not None

    # a compute that yields zero peaks must clear both stale selections
    empty_pg = Periodogram.from_spectrum(
        method="GLS",
        backend="numpy",
        frequency=np.array([]),
        power=np.array([]),
        objective_sense="max",
        n_samples=100,
        baseline=90.0,
    )
    assert empty_pg.best_periods(5) == []  # sanity: this really yields zero peaks
    cleared: list[None] = []
    controller.selection_cleared.connect(lambda: cleared.append(None))
    key = ResultKey("src", "GLS", settings_hash(GLSSettings(nyquist_factor=6)), "auto")
    controller._pending_key = key
    controller._on_finished(key, empty_pg, 1.0)

    assert cleared  # signal fired
    assert controller.state.selected_peak is None
    assert controller.state.selected_period is None
    assert spectrum._sel_period is None
    assert not spectrum._sel_band.isVisible()
    assert phased._period is None


def test_spectrum_marker_click_selects_period(qtbot: QtBot) -> None:
    # With nothing listening to peak_selected, a marker click still drives the
    # selection through period_selected (a widget wired the old way keeps working).
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    pg = _bump_periodogram()
    view.set_periodogram(pg)
    view.set_peaks(pg.best_periods(5))
    # points()[0] is the glow halo; points()[1] is the rank-1 peak marker
    marker = view._markers.points()[1]
    with qtbot.waitSignal(view.period_selected, timeout=2000) as blocker:
        view._on_peak_clicked(None, [marker])
    assert blocker.args[0] > 0.0


# -- GUI-C-6: marker clicks keep the peak; the tables follow the selection ------


def _bls_like_peaks() -> list[Peak]:
    return [
        Peak(period=3.0, frequency=1 / 3.0, power=0.9, rank=1,
             extra={"t0": 2455000.75, "depth": 0.01}),
        Peak(period=2.24618, frequency=1 / 2.24618, power=0.6, rank=2,
             extra={"t0": 2455001.2835, "depth": 0.004}),
        Peak(period=1.5, frequency=1 / 1.5, power=0.3, rank=3,
             extra={"t0": 2455000.1, "depth": 0.002}),
    ]


def _marker_for(view: SpectrumView, rank: int) -> object:
    for spot in view._markers.points():
        idx = spot.data()
        if idx is not None and view._peaks[int(idx)].rank == rank:
            return spot
    raise AssertionError(f"no marker for rank {rank}")


def test_marker_click_hands_over_the_whole_peak(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.set_periodogram(_bump_periodogram())
    peaks = _bls_like_peaks()
    view.set_peaks(peaks)
    picked: list[Peak] = []
    periods: list[float] = []
    view.peak_selected.connect(picked.append)
    view.period_selected.connect(periods.append)

    view._on_peak_clicked(None, [_marker_for(view, 2)])

    assert picked == [peaks[1]]  # extras such as the BLS epoch come along
    assert periods == []  # and the period-only path does not run as well
    assert view._sel_period == peaks[1].period


def test_marker_click_folds_at_the_peaks_own_epoch(qtbot: QtBot) -> None:
    controller = AppController()
    controller.set_light_curve(_light_curve(), "src")
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.peak_selected.connect(controller.select_peak)
    view.period_selected.connect(controller.select_period)
    folds: list[tuple[float, float]] = []
    controller.fold_changed.connect(lambda p, t0: folds.append((p, t0)))
    view.set_periodogram(_bump_periodogram())
    peaks = _bls_like_peaks()
    view.set_peaks(peaks)

    view._on_peak_clicked(None, [_marker_for(view, 2)])

    assert folds[-1] == (peaks[1].period, peaks[1].extra["t0"])
    assert controller.state.selected_peak == peaks[1]


def _row_of_rank(table: PeaksTable, rank: int) -> int:
    for row in range(table._table.rowCount()):
        peak = table._peak_for_row(row)
        if peak is not None and peak.rank == rank:
            return row
    raise AssertionError(f"no row for rank {rank}")


def _selected_rows(table: QtWidgets.QTableWidget) -> list[int]:
    return sorted({index.row() for index in table.selectedIndexes()})


def test_peaks_table_follows_a_selection_made_elsewhere(qtbot: QtBot) -> None:
    table = PeaksTable()
    qtbot.addWidget(table)
    peaks = _bls_like_peaks()
    table.set_peaks(peaks)
    emitted: list[Peak] = []
    table.peak_selected.connect(emitted.append)

    # A marker click elsewhere selected rank 2: the highlight moves, silently.
    table.select_period(peaks[1].period)
    assert _selected_rows(table._table) == [_row_of_rank(table, 2)]
    assert emitted == []

    # Sorted by another column, the right row is still found.
    table._table.sortItems(table._columns.index("period (d)"))
    table.select_period(peaks[2].period)
    assert _selected_rows(table._table) == [_row_of_rank(table, 3)]

    # A period that is no listed peak (a band drag) clears the highlight.
    table.select_period(7.77)
    assert _selected_rows(table._table) == []
    assert emitted == []


def test_clicking_the_highlighted_row_after_a_marker_click_takes_effect(
    qtbot: QtBot,
) -> None:
    # Regression: the table kept rank 1 highlighted after a marker click selected rank
    # 2, so clicking rank 1 changed no Qt selection and emitted nothing (a dead click).
    table = PeaksTable()
    qtbot.addWidget(table)
    table.resize(500, 300)
    table.show()
    qtbot.waitExposed(table)
    peaks = _bls_like_peaks()
    table.set_peaks(peaks)  # rank 1 starts selected
    table.select_period(peaks[1].period)  # what period_changed now delivers

    row = _row_of_rank(table, 1)
    rect = table._table.visualRect(table._table.model().index(row, 0))
    with qtbot.waitSignal(table.peak_selected, timeout=2000) as blocker:
        QtTest.QTest.mouseClick(
            table._table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center()
        )
    assert blocker.args[0] == peaks[0]


def test_peaks_table_keeps_a_multi_row_selection(qtbot: QtBot) -> None:
    table = PeaksTable()
    qtbot.addWidget(table)
    peaks = _bls_like_peaks()
    table.set_peaks(peaks)
    table._table.selectAll()
    selected = _selected_rows(table._table)
    assert len(selected) == 3
    # The period the table itself just reported must not collapse the selection.
    current = table._peak_for_row(table._table.currentRow())
    assert current is not None
    table.select_period(current.period)
    assert _selected_rows(table._table) == selected


# -- GUI-C-1 / UX-02: the curve is drawn end to end on every axis ----------------


def _long_period_spectrum(n: int = 500_000) -> tuple[Periodogram, Peak]:
    """A dense uniform-frequency spectrum whose best peak sits in its lowest bins."""
    baseline = 2000.0
    frequency = np.linspace(1.0 / baseline, 50.0, n)
    step = frequency[1] - frequency[0]
    rng = np.random.default_rng(1)
    power = 0.05 * rng.random(n)
    power += 0.9 * np.exp(-0.5 * ((frequency - frequency[20]) / (2.0 * step)) ** 2)
    pg = Periodogram.from_spectrum(
        method="GLS", backend="numpy", frequency=frequency, power=power,
        objective_sense="max", n_samples=1000, baseline=baseline,
    )
    best = pg.best_periods(1)[0]
    assert best.frequency < frequency[50]  # sanity: the peak is in the lowest bins
    return pg, best


def _painted(view: SpectrumView) -> tuple[np.ndarray, np.ndarray]:
    """What pyqtgraph hands the painter for the data curve (plot coordinates)."""
    x, y = view._curve.curve.getData()
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


@pytest.mark.parametrize(
    ("mode", "log_x"),
    [("period", True), ("period", False), ("frequency", True)],
)
def test_spectrum_curve_reaches_both_ends_on_any_axis(
    qtbot: QtBot, mode: str, log_x: bool
) -> None:
    # Regression: pyqtgraph's auto-downsampling assumes uniform x spacing, so on a
    # period or log axis it dropped the long-period tail, and the best peak's marker
    # floated over an empty plot.
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.resize(1000, 500)
    view.show()
    qtbot.waitExposed(view)
    pg, best = _long_period_spectrum()
    view.set_periodogram(pg)
    view.set_peaks([best])
    view._xaxis_combo.setCurrentText(mode)
    view._logx.setChecked(log_x)
    view._autorange()

    data_x = pg.period if mode == "period" else pg.frequency
    plot_x = np.log10(data_x) if log_x else data_x
    x, y = _painted(view)
    assert x.size < pg.size // 10  # still decimated...
    assert x.min() == pytest.approx(plot_x.min(), rel=1e-12, abs=1e-12)
    assert x.max() == pytest.approx(plot_x.max(), rel=1e-12, abs=1e-12)  # ...end to end

    marker_x = best.period if mode == "period" else best.frequency
    marker = np.log10(marker_x) if log_x else marker_x
    box = view._plot.getPlotItem().vb
    (x0, x1), _ = box.viewRange()
    units_per_px = (x1 - x0) / box.width()
    near = np.abs(x - marker) < 3.0 * units_per_px
    assert near.any()
    assert float(np.max(y[near])) == pytest.approx(best.power)  # the peak is drawn


def test_spectrum_zoom_shows_every_sample(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.resize(1000, 500)
    view.show()
    qtbot.waitExposed(view)
    pg, best = _long_period_spectrum()
    view.set_periodogram(pg)
    box = view._plot.getPlotItem().vb
    lo, hi = best.frequency * 0.5, best.frequency * 3.0
    box.setXRange(lo, hi, padding=0.0)
    qtbot.waitUntil(lambda: _painted(view)[0].size < 2000, timeout=2000)

    x, _ = _painted(view)
    inside = (pg.frequency >= lo) & (pg.frequency <= hi)
    shown = (x >= lo) & (x <= hi)
    assert int(shown.sum()) == int(inside.sum())  # full resolution when zoomed in

    # And Reset brings the whole spectrum back.
    view._autorange()
    x, _ = _painted(view)
    assert x.max() == pytest.approx(pg.frequency.max())


def test_envelope_keeps_every_extreme() -> None:
    from cuperiod.gui.widgets.spectrum_view import envelope_indices

    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0.0, 1.0, 200_000))
    y = rng.normal(size=x.size)
    y[[10, 150_000]] = [50.0, -50.0]
    y[777] = np.nan
    keep = envelope_indices(x, y, 0.0, 1.0, 500)
    assert np.all(np.diff(keep) > 0)  # ascending, no duplicates
    assert keep.size <= 4 * 501
    assert {0, x.size - 1, 10, 150_000} <= set(keep.tolist())
    # every bin's extremes survive
    bins = np.minimum((x * 500).astype(int), 499)
    for b in (0, 123, 499):
        members = np.flatnonzero(bins == b)
        assert np.nanargmax(y[members]) + members[0] in keep
        assert np.nanargmin(y[members]) + members[0] in keep
    # a short slice is passed through untouched
    small = envelope_indices(x, y, 0.5, 0.5 + 1e-3, 500)
    assert np.array_equal(small, np.arange(small[0], small[-1] + 1))


# -- GUI-C-10: a new result keeps the user's axis choice -------------------------


def _named_periodogram(method: str) -> Periodogram:
    freq = np.linspace(0.1, 2.0, 500)
    power = np.exp(-((freq - 0.5) ** 2) / 0.001)
    return Periodogram.from_spectrum(
        method=method, backend="numpy", frequency=freq, power=power,
        objective_sense="max", n_samples=100, baseline=90.0,
    )


def test_new_result_keeps_the_users_axis_choice(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.set_periodogram(_named_periodogram("GLS"))
    view._xaxis_combo.setCurrentText("period")
    view._logx.setChecked(False)  # period, linear
    view._logy.setChecked(True)

    view.set_periodogram(_named_periodogram("GLS"))  # next source / re-run
    assert view._xaxis_combo.currentText() == "period"
    assert view._x_mode == "period"
    assert not view._logx.isChecked()
    assert view._logy.isChecked()
    view.clear()  # a source switch clears the view first
    view.set_periodogram(_named_periodogram("GLS"))
    assert view._x_mode == "period" and not view._logx.isChecked()

    # A box search still opens on its own default...
    view.set_periodogram(_named_periodogram("BLS"))
    assert view._x_mode == "period" and view._logx.isChecked()
    # ...and keeps a choice made for it across box-search results.
    view._xaxis_combo.setCurrentText("frequency")
    view.set_periodogram(_named_periodogram("TLS"))
    assert view._x_mode == "frequency"
    # A pre-whitening spectrum opens on frequency, linear.
    view._xaxis_combo.setCurrentText("period")
    view.set_periodogram(_named_periodogram(PREWHITEN_METHOD))
    assert view._x_mode == "frequency" and not view._logx.isChecked()


# -- GUI-C-2 / GUI-C-3 / GUI-A-23: non-finite and empty light curves -------------


def _curve_with_nan_times(n: int = 400) -> LightCurve:
    rng = np.random.default_rng(5)
    time = np.sort(rng.uniform(0.0, 40.0, n))
    value = 14.0 + 0.4 * np.sin(2 * np.pi * time / 2.0) + rng.normal(0.0, 0.02, n)
    time[[3, 150, 333]] = np.nan  # SPOC FITS gap cadences / blank CSV time cells
    return LightCurve.from_arrays(time, value, np.full(n, 0.02))


def _finite_points(scatter: object) -> int:
    x, y = scatter.getData()  # type: ignore[attr-defined]
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return int((np.isfinite(x) & np.isfinite(y)).sum())


def test_phased_view_folds_around_nan_timestamps(qtbot: QtBot) -> None:
    lc = _curve_with_nan_times()
    controller = AppController()
    controller.set_light_curve(lc, "src")
    phased = PhasedView("dark")
    qtbot.addWidget(phased)
    folds: list[tuple[float, float]] = []
    controller.fold_changed.connect(lambda p, t0: folds.append((p, t0)))
    controller.fold_changed.connect(phased.set_fold)
    phased.set_light_curve(lc)

    controller.select_period(2.0)

    assert _finite_points(phased._scatters[""]) == 2 * (lc.n - 3)
    # Even a non-finite epoch handed in directly falls back to the first finite time.
    phased.set_fold(2.0, float("nan"))
    assert _finite_points(phased._scatters[""]) == 2 * (lc.n - 3)
    assert np.isfinite(phased._t0)
    assert phased._t0 == float(np.nanmin(lc.time))


def test_raw_view_draws_around_nan_timestamps(qtbot: QtBot) -> None:
    lc = _curve_with_nan_times()
    raw = RawLightCurveView("dark")
    qtbot.addWidget(raw)
    raw.show()
    raw.set_light_curve(lc)
    assert _finite_points(raw._scatters[""]) == lc.n - 3


def test_raw_and_phased_views_accept_empty_curves(qtbot: QtBot) -> None:
    raw = RawLightCurveView("dark")
    qtbot.addWidget(raw)
    raw.show()
    phased = PhasedView("dark")
    qtbot.addWidget(phased)
    raw.set_light_curve(_light_curve())
    assert _finite_points(raw._scatters[""]) == 200

    empty = LightCurve.from_arrays([], [], [])
    all_empty = MultiBandLightCurve.from_light_curves({"g": empty, "r": empty})
    for curve in (empty, all_empty):
        raw.set_light_curve(curve)  # used to raise "min() iterable argument is empty"
        assert sum(_finite_points(s) for s in raw._scatters.values()) == 0
        raw.apply_theme(palette("light"))  # the theme toggle rebuilt it too
        phased.set_light_curve(curve)
        phased.set_fold(2.0, 0.0)
        phased.apply_theme(palette("light"))
        assert sum(_finite_points(s) for s in phased._scatters.values()) == 0


# -- GUI-C-11: big curves stay interactive --------------------------------------


def _big_curve(n: int) -> LightCurve:
    rng = np.random.default_rng(9)
    time = np.sort(rng.uniform(0.0, 90.0, n))
    value = 1.0 + 0.01 * np.sin(2 * np.pi * time / 3.3) + rng.normal(0.0, 1e-3, n)
    return LightCurve.from_arrays(time, value, np.full(n, 1e-3), domain="flux")


def test_phased_view_caps_the_points_it_draws(qtbot: QtBot) -> None:
    from cuperiod.gui.widgets.phased_view import MAX_DRAWN_POINTS

    phased = PhasedView("dark")
    qtbot.addWidget(phased)
    lc = _big_curve(4 * MAX_DRAWN_POINTS)
    phased.set_light_curve(lc)
    phased.set_fold(3.3, 0.0)
    x, y = phased._scatters[""].getData()
    assert len(x) == 2 * MAX_DRAWN_POINTS  # two cycles of the capped subset
    assert f"{MAX_DRAWN_POINTS:,}" in phased._points_label.text()
    assert f"{lc.n:,}" in phased._points_label.text()
    # The subset is fixed per curve, so refolding does not make points flicker.
    phased.set_fold(3.31, 0.0)
    assert np.array_equal(phased._scatters[""].getData()[1], y)

    small = _light_curve()
    phased.set_light_curve(small)
    phased.set_fold(2.0, 0.0)
    assert len(phased._scatters[""].getData()[0]) == 2 * small.n
    assert phased._points_label.text() == ""


def test_raw_view_builds_only_when_shown(qtbot: QtBot) -> None:
    raw = RawLightCurveView("dark")
    qtbot.addWidget(raw)
    raw.set_light_curve(_light_curve())  # hidden, its default state in the window
    assert sum(_finite_points(s) for s in raw._scatters.values()) == 0
    raw.show()
    assert _finite_points(raw._scatters[""]) == 200


# -- SEC-02: band names from the file are text, never markup ---------------------


def _legend_texts(legend: object) -> list[str]:
    return [label.item.toPlainText() for _, label in legend.items]  # type: ignore[attr-defined]


def test_legends_show_band_names_literally(qtbot: QtBot) -> None:
    from cuperiod.gui.widgets._view_helpers import LEGEND_MAX_CHARS

    rng = np.random.default_rng(2)
    time = np.sort(rng.uniform(0.0, 20.0, 50))
    band = LightCurve.from_arrays(time, rng.normal(14.0, 0.1, 50), np.full(50, 0.1))
    names = ["g", "<b>x</b>", "r<22", "<img src=missing.png>", "B&V 'q'", "w" * 80]
    mblc = MultiBandLightCurve.from_light_curves(dict.fromkeys(names, band))
    phased = PhasedView("dark")
    qtbot.addWidget(phased)
    raw = RawLightCurveView("dark")
    qtbot.addWidget(raw)
    raw.show()
    expected = [*names[:-1], "w" * (LEGEND_MAX_CHARS - 1) + "…"]
    for _ in range(2):  # and again after a theme switch rebuilds the legends
        phased.set_light_curve(mblc)
        raw.set_light_curve(mblc)
        for legend in (phased._legend, raw._legend):
            assert sorted(_legend_texts(legend)) == sorted(expected)
            for _, label in legend.items:
                assert "<img" not in label.item.document().toHtml()
        phased.apply_theme(palette("light"))
        raw.apply_theme(palette("light"))
    assert set(phased._scatters) == set(names)  # keyed by the real band names


# -- GUI-C-7: exports never fail silently ----------------------------------------


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_png_export_appends_the_suffix_it_needs(qtbot: QtBot, tmp_path: Path) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.set_periodogram(_bump_periodogram())
    phased = PhasedView("dark")
    qtbot.addWidget(phased)
    for widget, stem in ((view, "spectrum"), (phased, "phased")):
        written = widget.export_png(str(tmp_path / stem))
        assert written == tmp_path / f"{stem}.png"
        assert written.read_bytes().startswith(_PNG_MAGIC)
        odd = widget.export_png(str(tmp_path / f"{stem}.v2"))
        assert odd == tmp_path / f"{stem}.v2.png" and odd.is_file()
    jpeg = view.export_png(str(tmp_path / "plot.jpg"))  # a real image suffix is kept
    assert jpeg == tmp_path / "plot.jpg"
    assert jpeg.read_bytes().startswith(b"\xff\xd8")


@pytest.fixture
def read_only_file(tmp_path: Path) -> Iterator[Callable[[str], Path]]:
    made: list[Path] = []

    def make(name: str) -> Path:
        path = tmp_path / name
        path.write_bytes(b"old")
        path.chmod(stat.S_IREAD)
        made.append(path)
        return path

    yield make
    for path in made:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_png_export_raises_when_nothing_is_written(
    qtbot: QtBot, read_only_file: Callable[[str], Path]
) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.set_periodogram(_bump_periodogram())
    target = read_only_file("locked.png")
    with pytest.raises(OSError, match="locked.png"):
        view.export_png(str(target))
    assert target.read_bytes() == b"old"


def test_export_slots_report_failures(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    read_only_file: Callable[[str], Path],
) -> None:
    # Regression: a locked target (a CSV open in Excel) raised inside the Qt slot,
    # which only printed a traceback to stderr; the user saw nothing.
    spectrum = SpectrumView("dark")
    qtbot.addWidget(spectrum)
    spectrum.set_periodogram(_bump_periodogram())
    phased = PhasedView("dark")
    qtbot.addWidget(phased)
    table = PeaksTable()
    qtbot.addWidget(table)
    table.set_peaks(_bls_like_peaks())
    csv_target = read_only_file("locked.csv")
    png_target = read_only_file("locked.png")
    slots = [
        (spectrum, spectrum._export_csv, csv_target),
        (spectrum, spectrum._export_png, png_target),
        (phased, phased._export_png, png_target),
        (table, table._export_csv, csv_target),
    ]
    for widget, slot, target in slots:
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *a, _t=target, **k: (str(_t), "")),
        )
        messages: list[str] = []
        widget.export_failed.connect(messages.append)
        slot()  # must not raise
        assert len(messages) == 1 and target.name in messages[0]


# -- UX-06: the crosshair readout never resizes the layout -----------------------


def test_readout_does_not_drive_the_layout(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.set_periodogram(_bump_periodogram())
    before = view.minimumSizeHint().width()
    view._readout.setText("f = 12.3456 /d    P = 0.0810005 d    power = 0.123456 " * 3)
    assert view.minimumSizeHint().width() == before


def test_readout_names_the_plotted_quantity_and_clears(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.resize(800, 400)
    view.show()
    qtbot.waitExposed(view)
    view.set_periodogram(_named_periodogram(PREWHITEN_METHOD))
    box = view._plot.getPlotItem().vb
    scene_pos = box.mapViewToScene(QtCore.QPointF(0.5, 0.5))
    view._on_mouse_moved((scene_pos,))
    assert "amplitude" in view._readout.text() and "power" not in view._readout.text()
    assert view._vline.isVisible()

    leave = QtCore.QEvent(QtCore.QEvent.Type.Leave)
    QtWidgets.QApplication.sendEvent(view._plot, leave)
    assert not view._vline.isVisible() and view._readout.text() == "—"

    view._on_mouse_moved((scene_pos,))
    view.set_periodogram(_named_periodogram("GLS"))  # a new result: no stale numbers
    assert not view._vline.isVisible() and view._readout.text() == "—"
    view._on_mouse_moved((scene_pos,))
    assert "power" in view._readout.text()


# -- UX-24: annotations do not collide ------------------------------------------


def test_hover_label_of_the_top_peak_hangs_below_it(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.resize(800, 400)
    view.show()
    qtbot.waitExposed(view)
    freq = np.linspace(0.1, 2.0, 2000)
    power = np.exp(-((freq - 0.5) ** 2) / 0.001) + 0.3 * np.exp(
        -((freq - 1.2) ** 2) / 0.001
    )
    pg = Periodogram.from_spectrum(
        method="GLS", backend="numpy", frequency=freq, power=power,
        objective_sense="max", n_samples=100, baseline=90.0,
    )
    view.set_periodogram(pg)
    peaks = pg.best_periods(2)
    view.set_peaks(peaks)

    class _Spot:
        def __init__(self, index: int) -> None:
            self._index = index

        def data(self) -> int:
            return self._index

    view._on_peak_hovered(view._markers, [_Spot(0)])  # the tallest: top of the view
    assert tuple(view._hover_text.anchor) == (0.0, 0.0)
    view._on_peak_hovered(view._markers, [_Spot(1)])  # a low one: label above it
    assert tuple(view._hover_text.anchor) == (0.0, 1.0)


def test_log_axis_tick_labels_do_not_overlap(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.resize(520, 360)
    view.show()
    qtbot.waitExposed(view)
    freq = np.geomspace(1 / 3000.0, 3.0, 4000)
    power = np.exp(-((np.log10(freq) + 1.0) ** 2) / 0.01)
    view.set_periodogram(
        Periodogram.from_spectrum(
            method="BLS", backend="numpy", frequency=freq, power=power,
            objective_sense="max", n_samples=100, baseline=3000.0,
        )
    )
    assert view._x_mode == "period" and view._logx.isChecked()
    for zoom in (None, (np.log10(0.4), np.log10(40.0))):
        if zoom is not None:
            view._plot.getPlotItem().vb.setXRange(*zoom, padding=0.0)
        axis = view._plot.getPlotItem().getAxis("bottom")
        image = QtGui.QImage(800, 100, QtGui.QImage.Format.Format_ARGB32)
        painter = QtGui.QPainter(image)
        try:
            _, _, texts = axis.generateDrawSpecs(painter)
        finally:
            painter.end()
        rects = [rect for rect, _flags, text in texts if text]
        assert len(rects) >= 2
        rects.sort(key=lambda r: r.left())
        for left, right in zip(rects, rects[1:], strict=False):
            assert left.right() <= right.left(), [t for _, _, t in texts]


def test_overlay_traces_get_a_legend(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    freq = np.linspace(0.1, 2.0, 500)
    view.set_periodogram(_named_periodogram(PREWHITEN_METHOD))
    assert not view._legend.isVisible()  # a plain spectrum needs no legend
    view.set_overlay(freq, 0.1 * np.ones_like(freq))
    assert _legend_texts(view._legend) == ["data", "residual"]
    view.set_window(freq, np.ones_like(freq))
    view._show_window.setChecked(True)
    assert _legend_texts(view._legend) == ["data", "residual", "window"]
    view._show_data.setChecked(False)
    assert _legend_texts(view._legend) == ["residual", "window"]
    view.set_periodogram(_named_periodogram("GLS"))
    assert not view._legend.isVisible()
    phased = PhasedView("dark")
    qtbot.addWidget(phased)
    raw = RawLightCurveView("dark")
    qtbot.addWidget(raw)
    for legend in (view._legend, phased._legend, raw._legend):
        assert 0 < legend.brush().color().alpha() < 255  # readable over the data


# -- A11Y-20: strokes keep their logical width on a HiDPI screen -----------------


def test_strokes_scale_with_the_device_pixel_ratio(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.set_periodogram(_bump_periodogram())
    assert view._curve.opts["pen"].widthF() == 1.0
    view.devicePixelRatioF = lambda: 2.0  # type: ignore[method-assign]
    QtWidgets.QApplication.sendEvent(
        view, QtCore.QEvent(QtCore.QEvent.Type.DevicePixelRatioChange)
    )
    assert view._curve.opts["pen"].widthF() == 2.0
    assert view._vline.pen.widthF() == 2.0
    for line in view._sel_band.lines:
        assert line.pen.widthF() == 2.0
    # The peak markers' outlines are strokes too.
    view.set_peaks([Peak(period=2.0, frequency=0.5, power=1.0, rank=1)])
    outlines = [
        spot.pen().widthF() for spot in view._markers.points() if spot.data() == 0
    ]
    assert outlines == [2.0]


def test_spacing_panel_strokes_scale_with_the_device_pixel_ratio(
    qtbot: QtBot,
) -> None:
    from cuperiod.gui.widgets.spacing_panel import SpacingPanel

    panel = SpacingPanel("dark")
    qtbot.addWidget(panel)
    assert panel._comb_curve.opts["pen"].widthF() == 1.0
    assert panel._comb_marker.pen.widthF() == 2.0
    panel.devicePixelRatioF = lambda: 2.0  # type: ignore[method-assign]
    QtWidgets.QApplication.sendEvent(
        panel, QtCore.QEvent(QtCore.QEvent.Type.DevicePixelRatioChange)
    )
    assert panel._comb_curve.opts["pen"].widthF() == 2.0
    assert panel._comb_marker.pen.widthF() == 4.0
    panel.apply_theme(palette("light"))  # a theme switch keeps the scaled widths
    assert panel._comb_curve.opts["pen"].widthF() == 2.0
