"""Qt-driven tests (offscreen): form, compute manager, controller, and spectrum."""

from __future__ import annotations

import numpy as np
from pytestqt.qtbot import QtBot

from cuperiod.core.config import BLSSettings, GLSSettings
from cuperiod.core.lightcurve import LightCurve
from cuperiod.core.result import Periodogram
from cuperiod.gui.compute import ComputeManager
from cuperiod.gui.models import LoadedCurve, ResultKey, SourceItem, settings_hash
from cuperiod.gui.settingsform import PydanticSettingsForm
from cuperiod.gui.state import AppController
from cuperiod.gui.widgets.phased_view import PhasedView
from cuperiod.gui.widgets.spectrum_view import SpectrumView


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

    tuned = controller._tune_auto_grid(lc, GLSSettings())
    assert tuned.maximum_frequency is not None
    assert tuned.maximum_frequency >= 10.0  # reaches at least P = 0.1 d
    assert tuned.samples_per_peak >= 10  # densified default grid

    # a method without frequency-grid fields (BLS) is returned unchanged
    bls = BLSSettings()
    assert controller._tune_auto_grid(lc, bls) is bls


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
