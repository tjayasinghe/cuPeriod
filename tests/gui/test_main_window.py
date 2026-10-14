"""Tests for the top-level window: busy state, shutdown, banner, drag-and-drop."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from cuperiod.core.config import GLSSettings, PDMSettings
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.gui.main_window import MainWindow, explain_failure
from cuperiod.gui.models import COMBINED_BAND, SourceItem
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets

QSettings = QtCore.QSettings


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path: Path) -> None:
    """Route ``QSettings()`` to a throwaway ini file so tests never touch (or depend
    on) the user's real persisted settings (registry on Windows)."""
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path)
    )


def _window(qtbot: QtBot) -> MainWindow:
    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    window = MainWindow(app)
    qtbot.addWidget(window)
    window.show()  # children's isVisible() is meaningful only in a shown window
    return window


def _light_curve() -> LightCurve:
    rng = np.random.default_rng(0)
    time = np.sort(rng.uniform(0.0, 40.0, 200))
    value = 14.0 + 0.4 * np.sin(2 * np.pi * time / 2.0) + rng.normal(0.0, 0.02, 200)
    return LightCurve.from_arrays(time, value, np.full(200, 0.02))


def _write_csv(path: Path) -> None:
    rng = np.random.default_rng(1)
    time = np.sort(rng.uniform(0.0, 40.0, 100))
    mag = 14.0 + 0.3 * np.sin(2 * np.pi * time / 3.0)
    lines = ["time,mag,mag_err"]
    lines += [f"{t:.6f},{m:.6f},0.02" for t, m in zip(time, mag, strict=True)]
    path.write_text("\n".join(lines), encoding="utf-8")


# -- Compute button busy state -----------------------------------------------------


def test_compute_button_disabled_while_busy(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._controller.set_light_curve(_light_curve(), "src")
    assert window._controls.can_compute()  # curve loaded, not busy
    window._controls.set_busy(True)
    assert not window._controls.can_compute()
    window._controls.set_busy(False)
    assert window._controls.can_compute()


def test_busy_changed_signal_reenables_compute_on_success(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._controller.set_light_curve(_light_curve(), "src")
    with qtbot.waitSignal(window._controller.periodogram_ready, timeout=30000):
        window._controller.run("GLS", "auto", GLSSettings(), 5)
    assert window._controls.can_compute()  # busy_changed(False) re-enabled it


def test_busy_changed_signal_reenables_compute_on_failure(qtbot: QtBot) -> None:
    from cuperiod.gui.models import ResultKey, settings_hash

    window = _window(qtbot)
    window._controller.set_light_curve(_light_curve(), "src")
    key = ResultKey("src", "GLS", settings_hash(GLSSettings()), "auto")
    window._controller._pending_key = key
    window._controls.set_busy(True)
    assert not window._controls.can_compute()
    window._controller._on_failed(key, "boom")  # emits busy_changed(False)
    assert window._controls.can_compute()


# -- closeEvent: shutdown + geometry persistence ------------------------------------


def test_close_event_calls_shutdown_and_persists_geometry(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    called: list[bool] = []
    monkeypatch.setattr(window._controller, "shutdown", lambda: called.append(True))
    window.close()
    assert called == [True]
    settings = QSettings()
    assert settings.value("geometry") is not None
    assert settings.value("windowState") is not None


def test_restored_geometry_is_applied_on_next_open(qtbot: QtBot) -> None:
    # Sizes must respect both the window minimum (1024x680) and the offscreen
    # platform's virtual screen (1024x768) — restoreGeometry clamps to the screen.
    first = _window(qtbot)
    first.resize(1024, 700)
    first.close()

    second = _window(qtbot)
    assert second.size().height() == 700  # not the 860 default -> restored


# -- dismissable error banner --------------------------------------------------------


def test_banner_dismiss_button_hides_banner(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._show_banner("something went wrong")
    assert window._banner_container.isVisible()
    qtbot.mouseClick(window._banner_dismiss, Qt.MouseButton.LeftButton)
    assert not window._banner_container.isVisible()


def _active_timers(window: MainWindow) -> set[int]:
    return {id(t) for t in window.findChildren(QtCore.QTimer) if t.isActive()}


def test_banner_stays_until_dismissed_or_superseded(qtbot: QtBot) -> None:
    # Regression (UX-14, A11Y-14): the banner hid itself after 12 s, so anyone who
    # looked away during a run never saw why it failed.
    window = _window(qtbot)
    window._controller.set_light_curve(_light_curve(), "src")
    before = _active_timers(window)
    window._show_banner("oops")
    assert _active_timers(window) <= before  # nothing scheduled to hide it
    qtbot.wait(20)
    assert window._banner_container.isVisible()
    # A new attempt supersedes it, valid or not.
    window._controls._form.setEnabled(True)
    window._controller.run = lambda *a, **k: None  # type: ignore[method-assign]
    window._controls.request_compute()
    assert not window._banner_container.isVisible()
    # So does a new curve.
    window._show_banner("oops again")
    window._controller.set_light_curve(_light_curve(), "other")
    assert not window._banner_container.isVisible()


# -- drag and drop --------------------------------------------------------------------


def test_drop_single_file_loads_it(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    csv = tmp_path / "curve.csv"
    _write_csv(csv)
    # skip the modal preview dialog (its exec() would block the headless test)
    monkeypatch.setattr("cuperiod.gui.main_window.preview_file", lambda _path: None)
    with qtbot.waitSignal(window._controller.lc_loaded, timeout=10000):
        window._handle_dropped_paths([csv])
    assert window._controller.state.current_lc is not None
    assert csv.name in window.windowTitle()


def test_drop_folder_loads_batch(qtbot: QtBot, tmp_path: Path) -> None:
    window = _window(qtbot)
    _write_csv(tmp_path / "a.csv")
    _write_csv(tmp_path / "b.csv")
    window._handle_dropped_paths([tmp_path])
    assert window._controller.state.mode == "batch"
    assert tmp_path.name in window.windowTitle()


def test_drop_multiple_files_loads_batch(qtbot: QtBot, tmp_path: Path) -> None:
    window = _window(qtbot)
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a)
    _write_csv(b)
    window._handle_dropped_paths([a, b])
    assert window._controller.state.mode == "batch"
    assert window._controller.state.sources is not None
    assert len(window._controller.state.sources) == 2


def test_drop_unrecognized_extension_warns_and_does_not_load(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    warned: list[str] = []
    monkeypatch.setattr(window, "_warn", warned.append)
    bogus = tmp_path / "notes.md"
    bogus.write_text("nope", encoding="utf-8")
    window._handle_dropped_paths([bogus])
    assert warned
    assert window._controller.state.current_lc is None


def test_is_acceptable_drop_accepts_directory_and_known_extensions(
    qtbot: QtBot, tmp_path: Path
) -> None:
    window = _window(qtbot)
    csv = tmp_path / "curve.csv"
    _write_csv(csv)

    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(csv))])
    assert window._is_acceptable_drop(mime)

    mime_dir = QtCore.QMimeData()
    mime_dir.setUrls([QtCore.QUrl.fromLocalFile(str(tmp_path))])
    assert window._is_acceptable_drop(mime_dir)

    bogus = tmp_path / "notes.md"
    bogus.write_text("nope", encoding="utf-8")
    mime_bad = QtCore.QMimeData()
    mime_bad.setUrls([QtCore.QUrl.fromLocalFile(str(bogus))])
    assert not window._is_acceptable_drop(mime_bad)


# -- last-used directory / file filter -----------------------------------------------


def test_file_filter_includes_all_files_fallback() -> None:
    from cuperiod.gui.main_window import _FILE_FILTER

    assert "All files (*)" in _FILE_FILTER
    assert ";;" in _FILE_FILTER


def test_placeholder_text_uses_straight_quotes(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._controller.set_light_curve(_light_curve(), "src")
    text = window._placeholder.text()
    assert "“" not in text and "”" not in text
    assert '"Compute periodogram"' in text


def test_analysis_switch_swaps_the_result_docks(qtbot: QtBot) -> None:
    window = _window(qtbot)
    assert window._peaks_dock.isVisible()
    assert not window._solution_dock.isVisible()
    window._controls.set_analysis("prewhiten")
    assert window._solution_dock.isVisible()
    assert window._spacing_dock.isVisible()
    assert not window._peaks_dock.isVisible()
    window._controls.set_analysis("periodogram")
    assert window._peaks_dock.isVisible()
    assert not window._solution_dock.isVisible()
    window._controller.shutdown()


def test_analysis_switch_keeps_the_loaded_light_curve(qtbot: QtBot) -> None:
    # Regression: switching analysis used to clear the phased view's light curve,
    # leaving an empty panel until the next file load.
    window = _window(qtbot)
    window._controller.set_light_curve(_light_curve(), "star")
    window._controls.set_analysis("prewhiten")
    assert window._phased._lc is not None
    assert window._controls.can_compute()
    window._controller.shutdown()


def test_prewhiten_run_populates_every_panel(qtbot: QtBot) -> None:
    from cuperiod.core.config import PreWhitenSettings
    from cuperiod.gui.widgets.spectrum_view import PREWHITEN_METHOD
    from synth import synthetic_pulsator

    window = _window(qtbot)
    time, value, error = synthetic_pulsator(n=600, span=20.0)
    window._controller.set_light_curve(
        LightCurve.from_arrays(time, value, error), "pulsator"
    )
    window._controls.set_analysis("prewhiten")
    settings = PreWhitenSettings(
        backend="finufft", max_frequencies=3, samples_per_peak=6
    )
    with qtbot.waitSignal(window._controller.solution_ready, timeout=60000):
        window._controller.run_prewhiten("finufft", settings)
    solution = window._controller.state.current_solution
    assert solution is not None and solution.n_components >= 1
    assert window._solution_panel._table.rowCount() == solution.n_components
    assert window._spectrum._pg is not None
    assert window._spectrum._pg.method == PREWHITEN_METHOD
    assert window._spectrum._overlay_xy is not None
    assert window._stack.currentIndex() == 1
    window._controller.shutdown()


def test_component_markers_are_drawn_on_the_spectrum_curve(qtbot: QtBot) -> None:
    # Regression: markers were placed at the component's *fitted* amplitude while the
    # curve shows the single-frequency amplitude spectrum. The two diverge as soon as
    # components are correlated (a HADS harmonic and its yearly alias sidelobes trade
    # amplitude in the joint fit), leaving dots floating in empty space above the
    # curve — a peak the spectrum does not have.
    from dataclasses import replace

    from cuperiod.core.config import PreWhitenSettings
    from cuperiod.prewhiten import prewhiten
    from synth import synthetic_pulsator

    window = _window(qtbot)
    time, value, error = synthetic_pulsator(n=600, span=20.0)
    solution = prewhiten(
        (time, value, error),
        settings=PreWhitenSettings(
            backend="finufft", max_frequencies=3, samples_per_peak=6
        ),
    )
    assert solution.spectrum is not None and solution.n_components >= 2
    # Force the pathological case rather than hoping for it: an amplitude nothing like
    # the spectrum at that frequency, in both directions.
    components = list(solution.components)
    components[0] = replace(components[0], amplitude=components[0].amplitude * 5.0)
    components[1] = replace(components[1], amplitude=components[1].amplitude * 0.1)
    solution = replace(solution, components=tuple(components))

    window._show_solution_spectrum(solution)
    grid, curve = solution.spectrum.frequency, solution.spectrum.amplitude
    peaks = window._spectrum._peaks
    assert len(peaks) == solution.n_components
    for peak in peaks:
        nearest = int(np.argmin(np.abs(grid - peak.frequency)))
        assert peak.power == pytest.approx(float(curve[nearest]))
        assert peak.power <= float(curve.max())
    # The fitted amplitude is not lost — it moves to the hover readout.
    assert peaks[0].extra["amplitude"] == pytest.approx(components[0].amplitude)
    assert peaks[0].power != pytest.approx(components[0].amplitude)
    window._controller.shutdown()


# == B2-GUI-SHELL regressions ==========================================================


def _multiband(seed: int = 0, bands: tuple[str, ...] = ("g", "r", "i")) -> object:
    from cuperiod.core.lightcurve import MultiBandLightCurve

    rng = np.random.default_rng(seed)
    curves = {}
    for i, name in enumerate(bands):
        time = np.sort(rng.uniform(0.0, 40.0, 120))
        value = 14.0 + i + 0.3 * np.sin(2 * np.pi * time / 3.0)
        curves[name] = LightCurve.from_arrays(time, value, np.full(120, 0.02))
    return MultiBandLightCurve.from_light_curves(curves)


def _record_runs(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, str, str]]:
    """Capture what each Compute would run, without computing anything."""
    runs: list[tuple[str, str, str]] = []

    def run(
        method: str, backend: str, settings: object, n_peaks: int, band: str = ""
    ) -> None:
        runs.append((method, backend, band))

    monkeypatch.setattr(window._controller, "run", run)
    return runs


def _pick(combo: QtWidgets.QComboBox, text: str) -> None:
    index = combo.findText(text, Qt.MatchFlag.MatchExactly)
    assert index >= 0, text
    combo.setCurrentIndex(index)


def _items(n: int, factory: Callable[[int], object]) -> list[SourceItem]:
    return [
        SourceItem(key=f"s{i}", label=f"s{i}", loader=partial(factory, i))  # type: ignore[arg-type]
        for i in range(n)
    ]


def _periodogram_for_badges() -> Periodogram:
    freq = np.linspace(0.1, 2.0, 200)
    return Periodogram.from_spectrum(
        method="GLS",
        backend="numpy",
        frequency=freq,
        power=np.exp(-((freq - 0.5) ** 2) / 0.001),
        objective_sense="max",
        n_samples=100,
        baseline=40.0,
    )


# -- GUI-B-3 / GUI-A-10 / UX-10: a batch step keeps the chosen backend and band -------


def test_batch_step_runs_with_the_chosen_backend_and_band(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    runs = _record_runs(window, monkeypatch)
    window._controller.load_sources(_items(3, _multiband))
    assert runs[-1] == ("GLS", "auto", COMBINED_BAND)
    _pick(window._controls._backend_combo, "cpu")
    _pick(window._controls._band_combo, "r")
    window._controls.request_compute()
    assert runs[-1] == ("GLS", "cpu", "r")
    window._source_browser._next_btn.click()
    assert window._controller.state.current_source_index == 1
    assert runs[-1] == ("GLS", "cpu", "r")  # was ("GLS", "auto", combined)


# -- GUI-B-5: an unreadable file does not block Prev/Next ---------------------------


def _folder_with_a_bad_file(tmp_path: Path) -> Path:
    _write_csv(tmp_path / "a_star.csv")
    (tmp_path / "notes.txt").write_text("just some notes\n", encoding="utf-8")
    _write_csv(tmp_path / "z_star.csv")
    return tmp_path


def test_next_steps_past_an_unreadable_source(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    _record_runs(window, monkeypatch)
    window._load_folder(_folder_with_a_bad_file(tmp_path))
    labels = [item.label for item in window._controller.state.sources or []]
    assert labels == ["a_star", "notes", "z_star"]
    assert window._controller.state.current_source_index == 0
    window._info.set_result(_periodogram_for_badges())
    shown = window._info._method.text()

    window._source_browser._next_btn.click()  # lands on notes.txt, which fails
    assert window._controller.state.current_source_index == 0
    assert window._source_browser.is_failed(1)
    assert window._info._method.text() == shown  # badges of what is still displayed
    assert window._banner_container.isVisible()
    assert window.statusBar().currentMessage().startswith("Load failed")

    window._source_browser._next_btn.click()  # ...and the next press passes it
    assert window._controller.state.current_source_index == 2


def test_clicking_an_unreadable_source_leaves_the_highlight_on_the_shown_one(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    _record_runs(window, monkeypatch)
    window._load_folder(_folder_with_a_bad_file(tmp_path))
    view = window._source_browser._list
    rect = view.visualItemRect(view.item(1))
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    qtbot.waitUntil(
        lambda: [view.row(i) for i in view.selectedItems()] == [view.currentRow()],
        timeout=2000,
    )
    assert view.currentRow() == window._controller.state.current_source_index == 0


def test_a_batch_starts_on_the_first_source_that_loads(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    _record_runs(window, monkeypatch)
    (tmp_path / "a_notes.txt").write_text("not a light curve\n", encoding="utf-8")
    _write_csv(tmp_path / "b_star.csv")
    window._load_folder(tmp_path)
    assert window._controller.state.current_source_index == 1
    assert window._source_browser.is_failed(0)
    assert window._controller.state.current_lc is not None


# -- GUI-B-8: the Sources dock and junk window state ---------------------------------


def test_sources_dock_does_not_come_back_empty_on_the_next_launch(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _window(qtbot)
    _record_runs(first, monkeypatch)
    first._controller.load_sources(_items(2, lambda _i: _light_curve()))
    assert first._sources_dock.isVisible()
    first.close()
    second = _window(qtbot)
    assert not second._sources_dock.isVisible()


@pytest.mark.parametrize("junk", ["garbage", 5])
def test_junk_saved_window_state_does_not_stop_the_window(
    qtbot: QtBot, junk: object
) -> None:
    settings = QSettings()
    settings.setValue("geometry", junk)
    settings.setValue("windowState", junk)
    settings.sync()
    window = _window(qtbot)  # used to raise TypeError in restoreGeometry(str)
    assert window.isVisible()


# -- GUI-B-9: the grid hint describes the band that will be analysed ----------------


def _dense_and_sparse() -> MultiBandLightCurve:
    rng = np.random.default_rng(3)
    dense = np.sort(rng.uniform(0.0, 5.0, 4000))
    sparse = np.sort(rng.uniform(0.0, 5.0, 80))
    curves = {
        name: LightCurve.from_arrays(t, 14.0 + 0.1 * np.sin(t), np.full(t.size, 0.01))
        for name, t in (("g", dense), ("r", sparse))
    }
    return MultiBandLightCurve.from_light_curves(curves)


def test_grid_hint_matches_the_grid_of_the_selected_band(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    submitted: list[float] = []

    def submit(*args: Any) -> None:  # (key, lc, method, settings, backend)
        submitted.append(float(args[3].maximum_frequency))

    monkeypatch.setattr(window._controller._compute, "submit", submit)
    window._controller.set_light_curve(_dense_and_sparse(), "mb")
    _pick(window._controls._band_combo, "r")
    hint_r = window._controls._grid_hint.text()
    window._controls.request_compute()
    assert submitted and f"{submitted[-1]:.3g} c/d" in hint_r
    _pick(window._controls._band_combo, "g")
    assert window._controls._grid_hint.text() != hint_r  # refreshed on band change
    window._controller.busy_changed.emit(False)


def test_nan_times_put_no_nan_in_the_loaded_line_or_the_hint(qtbot: QtBot) -> None:
    window = _window(qtbot)
    lc = _light_curve()
    time = lc.time.copy()
    time[::10] = np.nan
    window._controller.set_light_curve(
        LightCurve.from_arrays(time, lc.value, lc.error), "nan-times"
    )
    for text in (window._placeholder.text(), window._controls._grid_hint.text()):
        assert "nan" not in text and "inf" not in text


# -- GUI-B-11 / A11Y-14 (d): guidance names the button that is there ---------------


def test_prewhitening_guidance_names_the_run_button(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._controls.set_analysis("prewhiten")
    window._controller.set_light_curve(_light_curve(), "src")
    text = window._placeholder.text()
    assert '"Run pre-whitening"' in text and "Compute periodogram" not in text
    assert "Run pre-whitening" in window.statusBar().currentMessage()
    # ...and switching analysis with a curve loaded rewrites it.
    window._controls.set_analysis("periodogram")
    assert '"Compute periodogram"' in window._placeholder.text()
    window._controller.shutdown()


# -- GUI-B-13: drops return at once and mixed selections load what they can ---------


def _mime(*paths: Path) -> QtCore.QMimeData:
    data = QtCore.QMimeData()
    data.setUrls([QtCore.QUrl.fromLocalFile(str(p)) for p in paths])
    return data


def test_drop_accepts_mixed_selections(qtbot: QtBot, tmp_path: Path) -> None:
    window = _window(qtbot)
    csv = tmp_path / "a.csv"
    _write_csv(csv)
    readme = tmp_path / "README.md"
    readme.write_text("hi", encoding="utf-8")
    assert window._is_acceptable_drop(_mime(csv, readme))
    assert window._is_acceptable_drop(_mime(tmp_path, csv))
    assert not window._is_acceptable_drop(_mime(readme))


def test_drop_event_loads_after_the_drop_has_returned(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    csv = tmp_path / "a.csv"
    _write_csv(csv)
    handled: list[list[Path]] = []
    monkeypatch.setattr(window, "_handle_dropped_paths", handled.append)
    data = _mime(csv)
    event = QtGui.QDropEvent(
        QtCore.QPointF(20.0, 20.0),
        Qt.DropAction.CopyAction,
        data,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.dropEvent(event)
    assert event.isAccepted()
    assert handled == []  # nothing modal can open inside the drop handler
    qtbot.waitUntil(lambda: handled == [[csv]], timeout=2000)


def test_mixed_drop_loads_the_light_curves_and_reports_the_rest(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    _record_runs(window, monkeypatch)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_csv(a)
    _write_csv(b)
    readme = tmp_path / "README.md"
    readme.write_text("hi", encoding="utf-8")
    window._handle_dropped_paths([a, b, readme])
    assert window._controller.state.mode == "batch"
    assert len(window._controller.state.sources or []) == 2
    assert "README.md" in window.statusBar().currentMessage()


def test_drop_of_a_folder_and_a_file_takes_both(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    _record_runs(window, monkeypatch)
    sub = tmp_path / "field"
    sub.mkdir()
    _write_csv(sub / "s1.csv")
    _write_csv(sub / "s2.csv")
    single = tmp_path / "extra.csv"
    _write_csv(single)
    window._handle_dropped_paths([sub, single])
    labels = [item.label for item in window._controller.state.sources or []]
    assert sorted(labels) == ["extra", "field/s1", "field/s2"]


# -- GUI-B-15: the demo menu is filled when it opens, not at start-up -----------------


def test_demo_menu_is_built_on_first_open(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> list[SourceItem]:
        raise AssertionError("demo_sources ran at start-up")

    monkeypatch.setattr("cuperiod.gui.main_window.demo_sources", boom)
    window = _window(qtbot)  # must not touch the demo files
    menu = window._demo_menu
    assert [a.text() for a in menu.actions() if not a.isSeparator()] == [
        "Batch: browse all demo sources"
    ]
    demo = SourceItem("d", "Demo star", _light_curve)
    monkeypatch.setattr("cuperiod.gui.main_window.demo_sources", lambda: [demo])
    menu.aboutToShow.emit()
    menu.aboutToShow.emit()  # a second opening adds nothing
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert texts == ["Demo star", "Batch: browse all demo sources"]


def test_unreadable_demo_files_leave_a_working_menu(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> list[SourceItem]:
        raise OSError("bad parquet")

    monkeypatch.setattr("cuperiod.gui.main_window.demo_sources", boom)
    window = _window(qtbot)
    window._demo_menu.aboutToShow.emit()
    entries = [a for a in window._demo_menu.actions() if not a.isSeparator()]
    assert not entries[0].isEnabled() and "could not be read" in entries[0].text()


# -- SEC-03 / GUI-A-17: file-derived error text is never rendered as markup ----------


def test_banner_shows_markup_literally(qtbot: QtBot) -> None:
    window = _window(qtbot)
    raw = "ValueError: could not convert string to float: '<h1>big</h1> <img src=x>'"
    window._show_banner(raw)
    assert window._banner.textFormat() == Qt.TextFormat.PlainText
    assert "<h1>big</h1> <img src=x>" in window._banner.text()
    assert QtGui.Qt.mightBeRichText(window._banner.text())  # AutoText would render it


def test_a_hostile_header_in_a_batch_is_reported_literally(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    _record_runs(window, monkeypatch)
    header = "<img src=x>,<h1>License expired</h1>"
    (tmp_path / "a_bad.csv").write_text(f"{header}\n1,2\n3,4\n", encoding="utf-8")
    window._load_folder(tmp_path)
    assert window._banner_container.isVisible()
    assert window._banner.textFormat() == Qt.TextFormat.PlainText
    # The full error, column names included, is in the tooltip — escaped, so the
    # tooltip shows the tags as text.
    doc = QtGui.QTextDocument()
    doc.setHtml(window._banner.toolTip())
    assert "<img src=x>" in doc.toPlainText()
    assert "<h1>License expired</h1>" in doc.toPlainText()


def test_warning_box_is_plain_text_and_keeps_long_detail_out_of_the_way(
    qtbot: QtBot,
) -> None:
    window = _window(qtbot)
    table = "\n".join(f"format {i}  |  <b>yes</b>" for i in range(60))
    box = window._warning_box(f"Could not load star.fits:\nIORegistryError: {table}")
    qtbot.addWidget(box)
    assert box.textFormat() == Qt.TextFormat.PlainText
    assert len(box.text().splitlines()) <= 6
    assert box.detailedText().endswith("format 59  |  <b>yes</b>")


# -- A11Y-14 / UX-14: errors say what to do -----------------------------------------


def test_known_failures_are_explained_with_a_next_step() -> None:
    summary, details = explain_failure(
        "Could not load star: ColumnResolutionError: could not resolve a time "
        "column; pass ColumnMap(time=...) explicitly."
    )
    assert summary.startswith("Could not load star: ")
    assert "ColumnMap(time=" not in summary and "mag" in summary
    assert "ColumnResolutionError" in details
    summary, _ = explain_failure("InsufficientDataError: 3 finite points")
    assert "not enough usable data" in summary.lower()
    summary, details = explain_failure("RuntimeError: odd")
    assert summary == "RuntimeError: odd" and details == ""


def test_banner_is_named_for_assistive_technology(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._show_banner("InsufficientDataError: 3 finite points")
    assert window._banner.accessibleName() == "Error"
    assert "InsufficientDataError" in window._banner.accessibleDescription()
    assert window._banner_dismiss.accessibleName() == "Dismiss error"
    assert window._banner.textInteractionFlags() & (
        Qt.TextInteractionFlag.TextSelectableByKeyboard
    )


def test_a_header_only_file_is_refused_with_a_reason(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    warned: list[str] = []
    monkeypatch.setattr(window, "_warn", warned.append)
    monkeypatch.setattr("cuperiod.gui.main_window.preview_file", lambda _path: None)
    empty = tmp_path / "empty.csv"
    empty.write_text("time,mag,mag_err\n", encoding="utf-8")
    window._load_file(empty)
    assert warned and "no data rows" in warned[0]
    assert window._controller.state.current_lc is None
    assert not window._controls.can_compute()


def test_a_failed_run_keeps_the_displayed_results_badges(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._controller.set_light_curve(_light_curve(), "src")
    with qtbot.waitSignal(window._controller.periodogram_ready, timeout=30000):
        window._controller.run("GLS", "cpu", GLSSettings(), 5)
    shown = window._info._method.text()
    window._on_compute_failed("InsufficientDataError: 3 finite points")
    assert window._info._method.text() == shown
    assert window._info._backend.isVisible()
    assert window.statusBar().currentMessage().startswith("Compute periodogram failed")
    window._controller.shutdown()


# -- UX-09: the loaded curve is shown before anything is computed -------------------


def test_loaded_curve_is_shown_before_any_compute(qtbot: QtBot) -> None:
    window = _window(qtbot)
    assert window._stack.currentIndex() == 0  # nothing loaded: the welcome page
    assert "drop" in window._welcome.text()
    window._controller.set_light_curve(_light_curve(), "src")
    assert window._stack.currentIndex() == 1
    assert window._raw.isVisible() and not window._spectrum.isVisible()
    assert window._placeholder.isVisible() and "Loaded:" in window._placeholder.text()

    with qtbot.waitSignal(window._controller.periodogram_ready, timeout=30000):
        window._controller.run("GLS", "cpu", GLSSettings(), 5)
    assert window._spectrum.isVisible() and window._phased.isVisible()
    assert not window._placeholder.isVisible()
    assert not window._raw.isVisible()  # "Show Raw LC" is off by default
    window._phased._show_raw.setChecked(True)
    assert window._raw.isVisible()
    window._controller.shutdown()


# -- UX-23: a result that no longer matches the controls is flagged ----------------


def test_changing_the_method_flags_the_displayed_result(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._controller.set_light_curve(_light_curve(), "src")
    with qtbot.waitSignal(window._controller.periodogram_ready, timeout=30000):
        window._controller.run("GLS", "cpu", GLSSettings(), 5)
    assert not window._info.is_stale()
    _pick(window._controls._method_combo, "PDM")
    assert window._info.is_stale()
    assert "Settings changed" in window.statusBar().currentMessage()
    window._controller.shutdown()


# -- UX-25: a demo's suggested method is applied -------------------------------------


def test_a_demo_suggests_its_method(qtbot: QtBot) -> None:
    window = _window(qtbot)
    demo = SourceItem("kepler", "Kepler transit", _light_curve, {"hint_method": "BLS"})
    window._load_demo(demo)
    assert window._controls.current_method() == "BLS"
    assert "BLS" in window.statusBar().currentMessage()
    # Not in pre-whitening, which has no method picker.
    window._controls.set_analysis("prewhiten")
    window._controls.set_method("GLS")
    window._load_demo(demo)
    assert window._controls.current_method() == "GLS"


# -- GUI-B-1 / GUI-B-6: a cached result is reused only for the same data ------------


def _write_periodic_csv(path: Path, period: float) -> None:
    rng = np.random.default_rng(7)  # the same time stamps for every period
    time = np.sort(rng.uniform(0.0, 40.0, 200))
    mag = 14.0 + 0.4 * np.sin(2 * np.pi * time / period)
    lines = ["time,mag,mag_err"]
    lines += [f"{t:.6f},{m:.6f},0.02" for t, m in zip(time, mag, strict=True)]
    path.write_text("\n".join(lines), encoding="utf-8")


def _best_period(window: MainWindow) -> float:
    pg = window._controller.state.current_pg
    assert pg is not None
    return float(pg.best_period())


def test_same_named_files_in_two_folders_get_their_own_periodograms(
    qtbot: QtBot, tmp_path: Path
) -> None:
    window = _window(qtbot)
    _pick(window._controls._backend_combo, "cpu")
    folders = {2.0: tmp_path / "fieldA", 5.0: tmp_path / "fieldB"}
    for period, folder in folders.items():
        folder.mkdir()
        _write_periodic_csv(folder / "star_0001.csv", period)
    for period, folder in folders.items():
        with qtbot.waitSignal(window._controller.periodogram_ready, timeout=30000):
            window._load_folder(folder)  # batch mode computes the first source
        assert not window._controller.state.last_from_cache
        assert _best_period(window) == pytest.approx(period, rel=0.02)
    # Sources that share a key outright (as file stems used to) are told apart too.
    same_key = [
        SourceItem(
            "star", "star", partial(LightCurve.from_file, folders[p] / "star_0001.csv")
        )
        for p in (2.0, 5.0)
    ]
    with qtbot.waitSignal(window._controller.periodogram_ready, timeout=30000):
        window._controller.load_sources(same_key)
    with qtbot.waitSignal(window._controller.periodogram_ready, timeout=30000):
        window._source_browser._next_btn.click()
    assert not window._controller.state.last_from_cache
    assert _best_period(window) == pytest.approx(5.0, rel=0.02)
    window._controller.shutdown()


def test_reopening_a_rewritten_file_recomputes(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("cuperiod.gui.main_window.preview_file", lambda _path: None)
    window = _window(qtbot)
    _pick(window._controls._backend_combo, "cpu")
    path = tmp_path / "target.csv"

    def open_and_compute() -> None:
        window._load_file(path)
        with qtbot.waitSignal(window._controller.periodogram_ready, timeout=30000):
            window._controls.request_compute()

    _write_periodic_csv(path, 2.0)
    open_and_compute()
    assert _best_period(window) == pytest.approx(2.0, rel=0.02)
    _write_periodic_csv(path, 7.0)  # same name, same times, different star
    open_and_compute()
    assert not window._controller.state.last_from_cache
    assert _best_period(window) == pytest.approx(7.0, rel=0.02)
    open_and_compute()  # unchanged: the cached result is the right one
    assert window._controller.state.last_from_cache
    window._controller.shutdown()


# -- GUI-B-5 follow-up: a new batch whose sources all fail shows nothing of the old --


def test_a_new_batch_that_fails_to_load_leaves_nothing_of_the_old_one(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> LightCurve:
        raise OSError("unreadable")

    window = _window(qtbot)
    _record_runs(window, monkeypatch)
    window._controller.load_sources(_items(2, lambda _i: _light_curve()))
    assert window._stack.currentIndex() == 1 and window._controls.can_compute()
    window._controller.load_sources([SourceItem("bad", "bad", boom)])
    assert window._controller.state.current_lc is None
    assert window._stack.currentIndex() == 0  # the welcome page, not the old plots
    assert not window._controls.can_compute()
    assert window._source_browser.is_failed(0)
    assert window._banner_container.isVisible()


# -- requests from the other GUI groups --------------------------------------------


def test_load_notes_are_shown_with_the_loaded_curve(qtbot: QtBot) -> None:
    import dataclasses

    from cuperiod.gui.loaders import LOAD_NOTES_KEY

    window = _window(qtbot)
    note = "40 of 240 rows have no label in band column 'filter' and were dropped"
    lc = dataclasses.replace(_light_curve(), meta={LOAD_NOTES_KEY: (note,)})
    window._controller.set_light_curve(lc, "noted")
    assert note in window._placeholder.text()
    assert note in window.statusBar().currentMessage()


def test_views_are_wired_to_the_selection_and_the_banner(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._controller.set_light_curve(_light_curve(), "src")
    with qtbot.waitSignal(window._controller.periodogram_ready, timeout=30000):
        window._controller.run("PDM", "cpu", PDMSettings(), 5)
    # The Peaks column names the statistic: PDM minimises theta, not a power.
    assert window._peaks_table._statistic != "power"
    peaks = window._controller._peaks
    assert len(peaks) >= 2
    window._spectrum.peak_selected.emit(peaks[1])  # a marker click
    assert window._controller.state.selected_peak == peaks[1]
    rows = {i.row() for i in window._peaks_table._table.selectedIndexes()}
    assert rows == {1}  # the table follows the selection
    window._spectrum.export_failed.emit("Could not write spectrum.png: disk full")
    assert window._banner_container.isVisible()
    assert "disk full" in window._banner.text()
    window._controller.shutdown()


def test_zoom_shortcuts_scale_the_ui_and_are_remembered(qtbot: QtBot) -> None:
    from cuperiod.gui.theme import apply_theme, font_scale

    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    window = _window(qtbot)
    zoom_in, _zoom_out, reset = window._zoom_actions
    assert QtGui.QKeySequence("Ctrl+0") in reset.shortcuts()
    try:
        zoom_in.trigger()
        zoomed = font_scale()
        assert zoomed > 1.0
        assert float(QSettings().value("ui_scale")) == pytest.approx(zoomed)
        apply_theme(app, "dark", font_scale=1.0)
        _window(qtbot)  # the next launch comes back at the saved zoom
        assert font_scale() == pytest.approx(zoomed)
        reset.trigger()
        assert font_scale() == pytest.approx(1.0)
    finally:
        apply_theme(app, "dark", font_scale=1.0)  # never leak a zoom into other tests


def test_whats_this_is_reachable_from_the_keyboard(qtbot: QtBot) -> None:
    window = _window(qtbot)
    action = window._whats_this_action
    assert action in window.actions()
    assert QtGui.QKeySequence("Shift+F1") in action.shortcuts()
