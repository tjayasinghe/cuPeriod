"""Regression tests for :class:`cuperiod.gui.state.AppController` (offscreen, CPU).

Each test drives the controller through its public slots and signals, the way the
main window does, and checks what reaches the views: which result is applied to which
curve, what the cache serves, and which settings a run actually uses.
"""

from __future__ import annotations

import threading
import time as _time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from pydantic_settings import BaseSettings
from pytestqt.qtbot import QtBot

import cuperiod
from cuperiod.core.config import BLSSettings, GLSSettings, PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.compute import compute_active
from cuperiod.gui.models import (
    COMBINED_BAND,
    STACKED_BAND,
    LoadedCurve,
    ResultKey,
    SourceItem,
)
from cuperiod.gui.state import AppController
from cuperiod.prewhiten.result import PreWhitenResult


def _sine(period: float, seed: int, n: int = 400, span: float = 80.0) -> LightCurve:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n))
    m = 12.0 + 0.3 * np.sin(2 * np.pi * t / period) + rng.normal(0.0, 0.01, n)
    return LightCurve.from_arrays(t, m, np.full(n, 0.01))


def _two_mode(n: int = 600) -> LightCurve:
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 60.0, n))
    m = (
        0.30 * np.sin(2 * np.pi * 1.3 * t)
        + 0.20 * np.sin(2 * np.pi * 1.37 * t + 1.0)
        + rng.normal(0.0, 0.01, n)
    )
    return LightCurve.from_arrays(t, m, np.full(n, 0.01))


class _FakeCompute:
    """Stands in for ``cup.periodogram``: records calls, can block, can fail."""

    def __init__(self, real: Callable[..., Any]) -> None:
        self._real = real
        self.calls: list[tuple[int, str, BaseSettings]] = []
        self.gate = threading.Event()
        self.gate.set()
        self.result: Periodogram | None = None
        self.error: Exception | None = None

    def block(self) -> None:
        self.gate.clear()

    def release(self) -> None:
        self.gate.set()

    def __call__(self, lc: Any, method: str, **kwargs: Any) -> Any:
        n = lc.n if isinstance(lc, LightCurve) else sum(b.n for b in lc.bands.values())
        self.calls.append((n, method, kwargs["settings"]))
        self.gate.wait(30)
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return self._real(lc, method, **kwargs)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeCompute]:
    stub = _FakeCompute(cuperiod.periodogram)
    monkeypatch.setattr(cuperiod, "periodogram", stub)
    yield stub
    stub.release()


@pytest.fixture
def controller() -> Iterator[AppController]:
    ctl = AppController()
    yield ctl
    ctl._compute.cancel_all()
    ctl._compute.wait_for_done(30000)


def _drain(ctl: AppController, qtbot: QtBot) -> None:
    """Let the pool finish and deliver its queued signals."""
    assert ctl._compute.wait_for_done(60000)
    qtbot.wait(150)


def _submissions(
    ctl: AppController, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[ResultKey, BaseSettings]]:
    """Record every ``ComputeManager.submit``/``submit_prewhiten`` call."""
    seen: list[tuple[ResultKey, BaseSettings]] = []
    submit, submit_pw = ctl._compute.submit, ctl._compute.submit_prewhiten

    def spy(key: ResultKey, lc: Any, method: str, settings: Any, backend: str) -> None:
        seen.append((key, settings))
        submit(key, lc, method, settings, backend)

    def spy_pw(key: ResultKey, lc: Any, settings: Any, backend: str) -> None:
        seen.append((key, settings))
        submit_pw(key, lc, settings, backend)

    monkeypatch.setattr(ctl._compute, "submit", spy)
    monkeypatch.setattr(ctl._compute, "submit_prewhiten", spy_pw)
    return seen


# -- GUI-A-1: cached results belong to the data, not to the file stem --------------


def test_same_source_id_with_different_data_is_not_a_cache_hit(
    controller: AppController, qtbot: QtBot
) -> None:
    controller.set_light_curve(_sine(3.7, 1), "star_0001")
    with qtbot.waitSignal(controller.periodogram_ready, timeout=60000):
        controller.run("GLS", "cpu", GLSSettings(), 5)
    assert controller.state.selected_period == pytest.approx(3.7, rel=0.02)

    controller.set_light_curve(_sine(0.61, 2, n=520), "star_0001")  # another star
    with qtbot.waitSignal(controller.periodogram_ready, timeout=60000):
        controller.run("GLS", "cpu", GLSSettings(), 5)
    assert not controller.state.last_from_cache
    assert controller.state.selected_period == pytest.approx(0.61, rel=0.02)


def test_same_source_id_with_different_data_prewhiten(
    controller: AppController, qtbot: QtBot
) -> None:
    settings = PreWhitenSettings(backend="finufft", max_frequencies=1)
    controller.set_analysis("prewhiten")
    controller.set_light_curve(_sine(3.7, 1), "star_0001")
    with qtbot.waitSignal(controller.solution_ready, timeout=60000):
        controller.run_prewhiten("cpu", settings)
    controller.set_light_curve(_sine(0.61, 2), "star_0001")
    with qtbot.waitSignal(controller.solution_ready, timeout=60000):
        controller.run_prewhiten("cpu", settings)
    assert not controller.state.last_from_cache
    solution = controller.state.current_solution
    assert solution is not None
    assert solution.components[0].frequency == pytest.approx(1 / 0.61, rel=0.01)


def test_reloading_identical_data_is_still_an_instant_cache_hit(
    controller: AppController, qtbot: QtBot
) -> None:
    # batch mode re-runs the loader on every visit: a new object with equal data
    controller.set_light_curve(_sine(3.7, 1), "star")
    with qtbot.waitSignal(controller.periodogram_ready, timeout=60000):
        controller.run("GLS", "cpu", GLSSettings(), 5)
    controller.set_light_curve(_sine(3.7, 1), "star")
    got: list[object] = []
    controller.periodogram_ready.connect(got.append)
    controller.run("GLS", "cpu", GLSSettings(), 5)
    assert len(got) == 1 and controller.state.last_from_cache


# -- GUI-A-2 / GUI-B-2: an in-flight run never lands on a newly loaded curve --------


def test_in_flight_result_is_not_applied_to_a_new_curve(
    controller: AppController, qtbot: QtBot
) -> None:
    applied: list[str] = []
    controller.periodogram_ready.connect(lambda _pg: applied.append("pg"))
    controller.fold_changed.connect(lambda *_: applied.append("fold"))
    controller.compute_failed.connect(lambda _m: applied.append("failed"))
    busy: list[bool] = []
    controller.busy_changed.connect(busy.append)

    star_b = _sine(0.61, 2)
    controller.set_light_curve(_sine(3.7, 1), "A")
    controller.run("GLS", "cpu", GLSSettings(), 5)
    controller.set_light_curve(star_b, "B")  # File > Open while "Computing..."
    assert busy[-1] is False  # Compute is usable for B straight away
    _drain(controller, qtbot)

    assert applied == []
    assert controller.state.current_lc is star_b
    assert controller.state.current_pg is None
    assert controller.state.selected_period is None
    assert busy[-1] is False


def test_in_flight_failure_is_not_reported_over_a_new_curve(
    controller: AppController, qtbot: QtBot, fake: _FakeCompute
) -> None:
    fake.error = ValueError("A failed")
    failures: list[str] = []
    controller.compute_failed.connect(failures.append)
    controller.set_light_curve(_sine(3.7, 1), "A")
    controller.run("GLS", "cpu", GLSSettings(), 5)
    controller.set_light_curve(_sine(0.61, 2), "B")
    _drain(controller, qtbot)
    assert failures == []


def test_in_flight_solution_is_not_applied_to_a_new_curve(
    controller: AppController, qtbot: QtBot
) -> None:
    controller.set_analysis("prewhiten")
    applied: list[object] = []
    controller.solution_ready.connect(applied.append)
    controller.set_light_curve(_sine(3.7, 1), "A")
    controller.run_prewhiten(
        "cpu", PreWhitenSettings(backend="finufft", max_frequencies=1)
    )
    controller.set_light_curve(_sine(0.61, 2), "B")
    _drain(controller, qtbot)
    assert applied == []
    assert controller.state.current_solution is None


def test_a_run_started_from_the_load_signal_survives_the_reset(
    controller: AppController, qtbot: QtBot
) -> None:
    # Batch mode starts the next run from an lc_loaded slot, so the reset of the
    # previous run has to happen before that signal, not after it.
    controller.lc_loaded.connect(
        lambda _lc: controller.run("GLS", "cpu", GLSSettings(), 5)
    )
    controller.set_light_curve(_sine(3.7, 1), "A")
    controller.set_light_curve(_sine(0.61, 2), "B")
    with qtbot.waitSignal(controller.periodogram_ready, timeout=60000):
        pass
    assert controller.state.selected_period == pytest.approx(0.61, rel=0.02)


# -- GUI-A-13: a key already in flight is not computed twice ------------------------


def test_rerequesting_an_in_flight_key_computes_once(
    controller: AppController, qtbot: QtBot, fake: _FakeCompute
) -> None:
    fake.block()
    controller.set_light_curve(_sine(3.7, 1), "s1")
    with qtbot.waitSignal(controller._compute.signals.started, timeout=10000):
        controller.run("GLS", "cpu", GLSSettings(), 5)
    controller.set_light_curve(_sine(0.61, 2), "s2")
    controller.run("GLS", "cpu", GLSSettings(), 5)
    controller.set_light_curve(_sine(3.7, 1), "s1")  # back to s1 while it runs
    busy: list[bool] = []
    controller.busy_changed.connect(busy.append)
    controller.run("GLS", "cpu", GLSSettings(), 5)
    assert busy == [True]
    ready: list[object] = []
    controller.periodogram_ready.connect(ready.append)
    fake.release()
    _drain(controller, qtbot)
    assert [n for n, _, _ in fake.calls] == [400]  # s1 only; s2 was superseded
    assert len(ready) == 1
    assert busy[-1] is False
    assert controller.state.selected_period == pytest.approx(3.7, rel=0.02)


def test_a_late_duplicate_does_not_reset_the_selection(
    controller: AppController, qtbot: QtBot
) -> None:
    controller.set_light_curve(_two_mode(), "s1")
    with qtbot.waitSignal(controller._compute.signals.finished, timeout=60000) as done:
        controller.run("GLS", "cpu", GLSSettings(), 5)
    qtbot.wait(50)
    key, pg, _ = done.args
    assert isinstance(key, ResultKey) and controller.state.current_pg is not None
    peaks = controller.state.current_pg.best_periods(5)
    controller.select_peak(peaks[1])  # the user picks the rank-2 peak
    ready: list[object] = []
    controller.periodogram_ready.connect(ready.append)
    controller._on_finished(key, pg, 1.0)  # a duplicate delivery of the same key
    assert ready == []
    assert controller.state.selected_period == peaks[1].period


# -- GUI-A-14: a failing source neither strands navigation nor leaks a curve --------


def _boom() -> LoadedCurve:
    raise ValueError("unreadable")


def test_a_failed_source_is_reported_by_index(
    controller: AppController, qtbot: QtBot
) -> None:
    good = SourceItem("good", "good", lambda: _sine(3.7, 1))
    bad = SourceItem("bad", "bad", _boom)
    controller.load_sources([good, bad])
    controller.select_source(0)
    failed: list[int] = []
    controller.source_failed.connect(failed.append)
    controller.select_source(1)
    assert failed == [1]
    assert controller.state.current_source_index == 0


def test_a_new_batch_drops_the_previous_curve(
    controller: AppController, qtbot: QtBot
) -> None:
    items = [SourceItem(f"a{i}", f"a{i}", lambda: _sine(3.7, 1)) for i in range(3)]
    controller.load_sources(items)
    controller.select_source(2)
    assert controller.state.current_source_index == 2
    cleared: list[None] = []
    controller.lc_cleared.connect(lambda: cleared.append(None))
    busy: list[bool] = []
    controller.busy_changed.connect(busy.append)
    controller.load_sources(
        [SourceItem("bad", "bad", _boom), SourceItem("ok", "ok", lambda: _sine(1, 2))]
    )
    assert cleared == [None]
    assert controller.state.current_source_index == 0
    assert controller.state.current_lc is None
    assert controller.state.source_id == ""
    assert controller.state.current_pg is None
    assert busy == [False]
    controller.select_source(0)  # fails: nothing of the old batch comes back
    assert controller.state.current_lc is None
    # Compute on the cleared state is a no-op rather than a run on the old curve
    controller.run("GLS", "cpu", GLSSettings(), 5)
    assert controller._pending_key is None


# -- GUI-A-15: a real band named like a sentinel cannot hijack all-bands ------------


def _sentinel_named_bands(name: str) -> MultiBandLightCurve:
    return MultiBandLightCurve.from_light_curves(
        {name: _sine(2.0, 1, n=150), "V": _sine(2.0, 2, n=400)}
    )


def test_combined_sentinel_beats_a_band_named_combined(
    controller: AppController, qtbot: QtBot
) -> None:
    controller.set_light_curve(_sentinel_named_bands("combined"), "mb")
    with qtbot.waitSignal(controller.periodogram_ready, timeout=60000) as blocker:
        controller.run("GLS", "cpu", GLSSettings(), 5, band=COMBINED_BAND)
    assert blocker.args[0].n_samples == 550  # the joint multiband fit
    with qtbot.waitSignal(controller.periodogram_ready, timeout=60000) as blocker:
        controller.run("GLS", "cpu", GLSSettings(), 5, band="combined")
    assert blocker.args[0].n_samples == 150  # the real band of that name


def test_stacked_sentinel_beats_a_band_named_stacked(
    controller: AppController, qtbot: QtBot
) -> None:
    settings = PreWhitenSettings(backend="finufft", max_frequencies=1)
    controller.set_analysis("prewhiten")
    controller.set_light_curve(_sentinel_named_bands("stacked"), "mb")
    with qtbot.waitSignal(controller.solution_ready, timeout=60000) as blocker:
        controller.run_prewhiten("cpu", settings, band=STACKED_BAND)
    assert blocker.args[0].n_samples == 550
    with qtbot.waitSignal(controller.solution_ready, timeout=60000) as blocker:
        controller.run_prewhiten("cpu", settings, band="stacked")
    assert blocker.args[0].n_samples == 150


# -- GUI-A-8: the peak-selection settings are honoured, without a recompute ---------


def test_peak_separation_setting_changes_peaks_from_the_cache(
    controller: AppController, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _submissions(controller, monkeypatch)
    controller.set_light_curve(_two_mode(), "two")
    with qtbot.waitSignal(controller.peaks_ready, timeout=60000) as blocker:
        controller.run("GLS", "cpu", GLSSettings(), 10)
    narrow: list[Peak] = blocker.args[0]
    wide_settings = GLSSettings(peak_separation_rayleigh=1000.0)
    with qtbot.waitSignal(controller.peaks_ready, timeout=2000) as blocker:
        controller.run("GLS", "cpu", wide_settings, 10)
    wide: list[Peak] = blocker.args[0]
    assert len(seen) == 1  # the second run was served from the cache
    assert controller.state.last_from_cache
    assert len(wide) < len(narrow)
    freqs = np.array(sorted(p.frequency for p in wide))
    assert np.all(np.diff(freqs) >= 1000.0 / 60.0 * 0.9)


def test_the_applied_spectrum_defaults_to_the_peaks_spin(
    controller: AppController, qtbot: QtBot
) -> None:
    # Views that ask the shown spectrum for its peaks get the Peaks spin's count, not
    # the settings' n_peaks (the form no longer shows that field).
    controller.set_light_curve(_two_mode(), "two")
    with qtbot.waitSignal(controller.periodogram_ready, timeout=60000) as blocker:
        controller.run("GLS", "cpu", GLSSettings(n_peaks=10), 3)
    assert len(blocker.args[0].best_periods()) == 3


def test_alias_settings_reach_box_search_peaks(
    controller: AppController, qtbot: QtBot, fake: _FakeCompute
) -> None:
    freq = np.linspace(0.05, 3.0, 6000)
    power = sum(
        np.exp(-((freq - f0) ** 2) / 2e-6) / k
        for k, f0 in enumerate((0.4, 0.8, 1.2), 1)
    )
    fake.result = Periodogram.from_spectrum(
        method="BLS",
        backend="numpy",
        frequency=freq,
        power=power,
        objective_sense="max",
        n_samples=400,
        baseline=80.0,
    )
    controller.set_light_curve(_sine(2.5, 1), "box")
    settings = BLSSettings(harmonic_max=1, alias_freq_tolerance=0.001)
    with qtbot.waitSignal(controller.peaks_ready, timeout=60000) as blocker:
        controller.run("BLS", "cpu", settings, 3)
    expected = fake.result.best_periods(
        3, alias_diverse=True, harmonic_max=1, alias_tolerance=0.001
    )
    assert [p.frequency for p in blocker.args[0]] == [p.frequency for p in expected]
    default = fake.result.best_periods(3, alias_diverse=True)
    assert len(expected) != len(default)  # the setting really changes the answer


# -- GUI-A-19: an explicit samples_per_peak reaches the run unchanged ---------------


def test_explicit_samples_per_peak_is_honoured(
    controller: AppController, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _submissions(controller, monkeypatch)
    controller.set_light_curve(_sine(3.7, 1), "s")
    for spp in (5, 4, 6):
        controller.run("GLS", "cpu", GLSSettings(samples_per_peak=spp), 5)
    # left unset (a programmatic default), the GUI's denser grid still applies
    controller.run("GLS", "cpu", GLSSettings(), 5)
    assert [s.samples_per_peak for _, s in seen] == [5, 4, 6, 10]  # type: ignore[attr-defined]


# -- UX-12: pre-whitening always keeps the spectra the display needs ----------------


def test_prewhiten_forces_store_spectra(
    controller: AppController, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _submissions(controller, monkeypatch)
    controller.set_light_curve(_sine(3.7, 1), "s")
    settings = PreWhitenSettings(
        backend="finufft", max_frequencies=1, store_spectra=False
    )
    with qtbot.waitSignal(controller.solution_ready, timeout=60000) as blocker:
        controller.run_prewhiten("cpu", settings)
    assert seen[0][1].store_spectra is True  # type: ignore[attr-defined]
    assert isinstance(blocker.args[0], PreWhitenResult)
    assert blocker.args[0].spectrum is not None


# -- GUI-A-4: NaN timestamps do not blank the fold ----------------------------------


def test_nan_time_keeps_the_fold_epoch_finite(
    controller: AppController, qtbot: QtBot
) -> None:
    base = _sine(3.7, 1)
    time = base.time.copy()
    time[[0, 5, 9]] = np.nan
    lc = LightCurve.from_arrays(time, base.value, base.error)
    controller.set_light_curve(lc, "gappy")
    assert np.all(np.isfinite(controller.current_time()))
    assert controller.current_time().size == lc.n - 3
    folds: list[tuple[float, float]] = []
    controller.fold_changed.connect(lambda p, t0: folds.append((p, t0)))
    with qtbot.waitSignal(controller.periodogram_ready, timeout=60000):
        controller.run("GLS", "cpu", GLSSettings(), 5)
    controller.select_period(1.234)
    assert folds and all(np.isfinite(t0) for _, t0 in folds)


# -- GUI-A-24: a marker click on a box-search peak keeps its transit epoch ----------


def _bls_like() -> Periodogram:
    freq = np.linspace(0.05, 2.0, 4000)
    power = np.exp(-((freq - 0.4) ** 2) / 1e-5)
    pg = Periodogram.from_spectrum(
        method="BLS",
        backend="numpy",
        frequency=freq,
        power=power,
        objective_sense="max",
        n_samples=400,
        baseline=80.0,
        extras={"t0": np.full(freq.size, 12.345)},
    )
    return pg


def test_marker_click_on_a_peak_uses_its_t0(
    controller: AppController, qtbot: QtBot, fake: _FakeCompute
) -> None:
    fake.result = _bls_like()
    controller.set_light_curve(_sine(2.5, 1), "box")
    with qtbot.waitSignal(controller.peaks_ready, timeout=60000) as blocker:
        controller.run("BLS", "cpu", BLSSettings(), 3)
    peak = blocker.args[0][0]
    assert peak.extra.get("t0") == pytest.approx(12.345)
    folds: list[tuple[float, float]] = []
    controller.fold_changed.connect(lambda p, t0: folds.append((p, t0)))
    controller.select_period(peak.period)  # what a spectrum-marker click emits
    assert folds[-1] == (peak.period, pytest.approx(12.345))
    assert controller.state.selected_peak == peak
    controller.select_period(peak.period * 1.01)  # an arbitrary period
    assert folds[-1][1] == pytest.approx(float(np.min(controller.current_time())))
    assert controller.state.selected_peak is None


def test_prewhiten_components_never_take_a_periodogram_epoch(
    controller: AppController, qtbot: QtBot, fake: _FakeCompute
) -> None:
    fake.result = _bls_like()
    controller.set_light_curve(_sine(2.5, 1), "box")
    with qtbot.waitSignal(controller.peaks_ready, timeout=60000) as blocker:
        controller.run("BLS", "cpu", BLSSettings(), 3)
    peak = blocker.args[0][0]
    controller.set_analysis("prewhiten")
    folds: list[tuple[float, float]] = []
    controller.fold_changed.connect(lambda p, t0: folds.append((p, t0)))
    controller.select_period(peak.period)
    assert folds[-1][1] != pytest.approx(12.345)


# -- GUI-A-20: closing the window does not wait on an uninterruptible run -----------


def test_shutdown_returns_promptly_while_a_run_is_in_flight(
    controller: AppController, qtbot: QtBot, fake: _FakeCompute
) -> None:
    fake.block()
    controller.set_light_curve(_sine(3.7, 1), "s")
    with qtbot.waitSignal(controller._compute.signals.started, timeout=10000):
        controller.run("GLS", "cpu", GLSSettings(), 5)
    assert controller._compute.active and compute_active()
    start = _time.perf_counter()
    controller.shutdown()
    assert _time.perf_counter() - start < 0.5
    fake.release()
    _drain(controller, qtbot)
    assert not controller._compute.active
    assert not compute_active()
