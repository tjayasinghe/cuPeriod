"""GUI pre-whitening: analysis switching, off-thread runs, and the new panels.

The point of these tests is that adding pre-whitening did not cost the periodogram
workflow anything: the same controller, the same spectrum/phased views, the same source
browser, with the result panels swapped underneath.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from cuperiod.core.config import GLSSettings, PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve
from cuperiod.gui.compute import ComputeManager
from cuperiod.gui.meta import prewhiten_backend_options, resolved_prewhiten_backend
from cuperiod.gui.models import ResultCache, ResultKey
from cuperiod.gui.state import PREWHITEN_KEY, AppController
from cuperiod.gui.widgets.controls_panel import (
    _PREWHITEN_CACHE_KEY,
    ControlsPanel,
)
from cuperiod.gui.widgets.solution_panel import SolutionPanel
from cuperiod.gui.widgets.spacing_panel import SpacingPanel
from cuperiod.gui.widgets.spectrum_view import PREWHITEN_METHOD, SpectrumView
from cuperiod.prewhiten import prewhiten
from cuperiod.prewhiten.result import PreWhitenResult
from synth import synthetic_gmode, synthetic_pulsator

_SETTINGS = PreWhitenSettings(
    backend="finufft", max_frequencies=3, samples_per_peak=6
)


def _curve(n: int = 600) -> LightCurve:
    time, value, error = synthetic_pulsator(n=n, span=20.0)
    return LightCurve.from_arrays(time, value, error)


def _solution(n: int = 600) -> PreWhitenResult:
    return prewhiten(_curve(n), settings=_SETTINGS)


# --- controls ----------------------------------------------------------------


def test_analysis_picker_swaps_the_settings_model(qtbot: QtBot) -> None:
    panel = ControlsPanel()
    qtbot.addWidget(panel)
    assert panel.analysis == "periodogram"
    with qtbot.waitSignal(panel.analysis_changed, timeout=2000) as blocker:
        panel.set_analysis("prewhiten")
    assert blocker.args == ["prewhiten"]
    assert panel.analysis == "prewhiten"
    assert isinstance(panel._form.build(), PreWhitenSettings)
    panel.set_analysis("periodogram")
    assert isinstance(panel._form.build(), GLSSettings)


def test_compute_emits_the_analysis_specific_signal(qtbot: QtBot) -> None:
    panel = ControlsPanel()
    qtbot.addWidget(panel)
    panel.set_enabled(True)
    with qtbot.waitSignal(panel.run_requested, timeout=2000):
        panel.request_compute()
    panel.set_analysis("prewhiten")
    with qtbot.waitSignal(panel.prewhiten_requested, timeout=2000) as blocker:
        panel.request_compute()
    backend, settings = blocker.args
    assert backend in prewhiten_backend_options()
    assert isinstance(settings, PreWhitenSettings)


def test_settings_are_remembered_across_analysis_switches(qtbot: QtBot) -> None:
    panel = ControlsPanel()
    qtbot.addWidget(panel)
    # Seed the pre-whitening slot while the periodogram form is up, so switching to
    # pre-whitening has to rebuild its form from the cache rather than from defaults.
    panel._settings_cache[_PREWHITEN_CACHE_KEY] = PreWhitenSettings(max_frequencies=7)
    panel.set_analysis("prewhiten")
    assert panel._form.build().max_frequencies == 7
    # Switching away stashes whatever the form currently holds, per analysis.
    panel.set_analysis("periodogram")
    assert isinstance(panel._settings_cache[_PREWHITEN_CACHE_KEY], PreWhitenSettings)
    assert isinstance(panel._settings_cache["GLS"], GLSSettings)


def test_prewhiten_backend_helpers() -> None:
    options = prewhiten_backend_options()
    assert "auto" in options and "numpy" in options
    resolved = resolved_prewhiten_backend("cpu")
    assert resolved is not None
    assert resolved[0] in {"finufft", "numpy"} and resolved[1] is False
    assert resolved_prewhiten_backend("nonsense") is None


# --- controller --------------------------------------------------------------


def test_controller_runs_prewhiten_off_thread(qtbot: QtBot) -> None:
    controller = AppController()
    controller.set_light_curve(_curve(), "star")
    with qtbot.waitSignal(controller.solution_ready, timeout=60000) as blocker:
        controller.run_prewhiten("finufft", _SETTINGS)
    (result,) = blocker.args
    assert isinstance(result, PreWhitenResult)
    assert result.n_components >= 1
    assert controller.state.current_solution is result
    assert controller.state.method == PREWHITEN_KEY
    # The strongest component becomes the active period for the phased view.
    assert controller.state.selected_period == pytest.approx(
        result.components[0].period
    )
    controller.shutdown()


def test_a_repeat_run_is_served_from_the_cache(qtbot: QtBot) -> None:
    controller = AppController()
    controller.set_light_curve(_curve(), "star")
    with qtbot.waitSignal(controller.solution_ready, timeout=60000):
        controller.run_prewhiten("finufft", _SETTINGS)
    with qtbot.waitSignal(controller.solution_ready, timeout=2000):
        controller.run_prewhiten("finufft", _SETTINGS)
    assert controller.state.last_from_cache
    assert controller.state.last_compute_ms == 0.0
    controller.shutdown()


def test_switching_analysis_keeps_both_caches(qtbot: QtBot) -> None:
    controller = AppController()
    controller.set_light_curve(_curve(), "star")
    with qtbot.waitSignal(controller.periodogram_ready, timeout=60000):
        controller.run("GLS", "finufft", GLSSettings(), 5)
    controller.set_analysis("prewhiten")
    with qtbot.waitSignal(controller.solution_ready, timeout=60000):
        controller.run_prewhiten("finufft", _SETTINGS)
    controller.set_analysis("periodogram")
    with qtbot.waitSignal(controller.periodogram_ready, timeout=2000):
        controller.run("GLS", "finufft", GLSSettings(), 5)
    assert controller.state.last_from_cache  # the earlier spectrum survived
    controller.shutdown()


def test_analysis_change_emits_once_and_is_idempotent(qtbot: QtBot) -> None:
    controller = AppController()
    with qtbot.waitSignal(controller.analysis_changed, timeout=2000):
        controller.set_analysis("prewhiten")
    assert controller.state.analysis == "prewhiten"
    with qtbot.assertNotEmitted(controller.analysis_changed):
        controller.set_analysis("prewhiten")


def test_a_failed_prewhiten_run_surfaces_as_a_message(qtbot: QtBot) -> None:
    controller = AppController()
    flat = LightCurve.from_arrays(
        np.full(60, 2458000.0), np.ones(60), np.full(60, 0.01)
    )
    controller.set_light_curve(flat, "flat")
    with qtbot.waitSignal(controller.compute_failed, timeout=30000) as blocker:
        controller.run_prewhiten("finufft", _SETTINGS)
    assert "InsufficientDataError" in blocker.args[0]
    controller.shutdown()


def test_compute_manager_gates_superseded_prewhiten_tasks(qtbot: QtBot) -> None:
    manager = ComputeManager()
    key = ResultKey("star", PREWHITEN_KEY, "hash", "finufft")
    manager.cancel_all()  # bump the generation so the next submit is stale on arrival
    manager.submit_prewhiten(key, _curve(200), _SETTINGS, "finufft")
    manager.cancel_all()
    assert manager.wait_for_done(30000)


def test_result_cache_is_generic_over_its_value_type() -> None:
    cache: ResultCache[PreWhitenResult] = ResultCache(maxsize=1)
    key_a = ResultKey("a", PREWHITEN_KEY, "h", "cpu")
    key_b = ResultKey("b", PREWHITEN_KEY, "h", "cpu")
    solution = _solution(300)
    cache.put(key_a, solution)
    assert cache.get(key_a) is solution
    cache.put(key_b, solution)
    assert cache.get(key_a) is None  # evicted


# --- panels ------------------------------------------------------------------


def test_solution_panel_lists_components_and_emits_selection(qtbot: QtBot) -> None:
    panel = SolutionPanel()
    qtbot.addWidget(panel)
    solution = _solution()
    panel.set_solution(solution)
    assert panel._table.rowCount() == solution.n_components
    assert panel._table.item(0, 0).text() == "F1"
    assert "stopped" in panel._summary.text()
    with qtbot.waitSignal(panel.component_selected, timeout=2000) as blocker:
        panel._table.selectRow(1)
    assert blocker.args[0].rank == 2
    panel.clear()
    assert panel._table.rowCount() == 0


def test_solution_panel_exports_full_precision_csv(qtbot: QtBot, tmp_path) -> None:
    panel = SolutionPanel()
    qtbot.addWidget(panel)
    solution = _solution()
    panel.set_solution(solution)
    path = tmp_path / "frequencies.csv"
    panel.export_csv(str(path))
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("rank,label,frequency")
    assert len(text) == solution.n_components + 1


def test_spacing_panel_finds_a_series(qtbot: QtBot) -> None:
    time, value, error, periods = synthetic_gmode(n=1500, n_modes=12)
    solution = prewhiten(
        LightCurve.from_arrays(time, value, error),
        settings=PreWhitenSettings(backend="finufft", max_frequencies=14),
    )
    panel = SpacingPanel()
    qtbot.addWidget(panel)
    panel.set_solution(solution)
    assert panel._search_btn.isEnabled()
    panel.search()
    assert "modes in one series" in panel._summary.text()
    assert panel._echelle_points.data.size > 0


def test_spacing_panel_declines_a_short_solution(qtbot: QtBot) -> None:
    panel = SpacingPanel()
    qtbot.addWidget(panel)
    panel.set_solution(_solution())  # only three components
    assert not panel._search_btn.isEnabled()
    assert "at least" in panel._summary.text()
    panel.search()  # a no-op, must not raise
    panel.clear()
    assert "Run pre-whitening" in panel._summary.text()


def test_spectrum_view_overlays_the_residual_spectrum(qtbot: QtBot) -> None:
    from cuperiod.core.result import Periodogram

    view = SpectrumView()
    qtbot.addWidget(view)
    solution = _solution()
    assert solution.spectrum is not None and solution.residual_spectrum is not None
    wrapped = Periodogram.from_spectrum(
        method=PREWHITEN_METHOD,
        backend=solution.backend,
        frequency=solution.spectrum.frequency,
        power=solution.spectrum.amplitude,
        objective_sense="max",
        n_samples=solution.n_samples,
        baseline=solution.baseline,
    )
    view.set_periodogram(wrapped)
    assert not view._show_residual.isVisible() or view._overlay_xy is None
    view.set_overlay(
        solution.residual_spectrum.frequency, solution.residual_spectrum.amplitude
    )
    assert view._overlay.isVisible()
    # The residual spectrum must sit below the original everywhere it is drawn.
    assert np.max(solution.residual_spectrum.amplitude) < np.max(
        solution.spectrum.amplitude
    )
    view.set_periodogram(wrapped)  # a new result drops the stale overlay
    assert view._overlay_xy is None


# --- regressions -------------------------------------------------------------


def test_switching_analysis_mid_run_releases_the_busy_state(qtbot: QtBot) -> None:
    # Regression: set_analysis() cleared the pending key, so the completion handler for
    # the in-flight run never fired busy_changed(False) and Compute stayed disabled for
    # the rest of the session.
    controller = AppController()
    panel = ControlsPanel()
    qtbot.addWidget(panel)
    panel.set_enabled(True)
    controller.busy_changed.connect(panel.set_busy)
    controller.set_light_curve(_curve(4000), "star")
    with qtbot.waitSignal(controller.busy_changed, timeout=5000):
        controller.run_prewhiten("finufft", _SETTINGS)
    assert not panel.can_compute()  # busy
    with qtbot.waitSignal(controller.busy_changed, timeout=5000) as blocker:
        controller.set_analysis("periodogram")
    assert blocker.args == [False]
    assert panel.can_compute()
    controller.shutdown()


def test_prewhiten_defaults_to_one_band_for_a_multiband_curve(qtbot: QtBot) -> None:
    # Regression: set_bands() passed the (hidden) method combo's text, so a multiband
    # curve loaded *after* switching to pre-whitening offered and selected
    # "combined (all bands)" and silently analysed a raw all-band stack.
    panel = ControlsPanel()
    qtbot.addWidget(panel)
    panel.set_analysis("prewhiten")
    panel.set_bands(["g", "r"])
    assert panel.current_band() == "g"
    assert "combined" not in panel._band_combo.itemText(0)
    # ...and the same is true whichever order the two happen in.
    other = ControlsPanel()
    qtbot.addWidget(other)
    other.set_bands(["g", "r"])
    other.set_analysis("prewhiten")
    assert other.current_band() == "g"


def test_a_stacked_multiband_prewhiten_removes_the_band_offsets(qtbot: QtBot) -> None:
    from cuperiod.core.lightcurve import MultiBandLightCurve

    rng = np.random.default_rng(0)
    time = np.sort(rng.uniform(0.0, 20.0, 900)) + 2458000.0
    signal = 0.01 * np.sin(2 * np.pi * 1.3 * (time - time.min()))
    bands = {
        "g": LightCurve.from_arrays(
            time, 15.0 + signal + rng.normal(0, 5e-4, 900), np.full(900, 5e-4)
        ),
        "r": LightCurve.from_arrays(
            time, 14.2 + signal + rng.normal(0, 5e-4, 900), np.full(900, 5e-4)
        ),
    }
    mblc = MultiBandLightCurve.from_light_curves(bands)
    controller = AppController()
    controller.set_light_curve(mblc, "multiband")
    with qtbot.waitSignal(controller.solution_ready, timeout=60000) as blocker:
        controller.run_prewhiten("finufft", _SETTINGS, band="stacked")
    (result,) = blocker.args
    # A raw concatenation would bury the 0.01 mag signal under a 0.8 mag offset.
    assert result.n_components >= 1
    assert result.components[0].frequency == pytest.approx(1.3, abs=0.01)
    controller.shutdown()
