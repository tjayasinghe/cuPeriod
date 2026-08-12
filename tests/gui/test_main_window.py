"""Tests for the top-level window: busy state, shutdown, banner, drag-and-drop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from cuperiod.core.config import GLSSettings
from cuperiod.core.lightcurve import LightCurve
from cuperiod.gui.main_window import MainWindow
from cuperiod.gui.qt import Qt, QtCore, QtWidgets

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


def test_hide_banner_stops_auto_hide_timer(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._show_banner("oops")
    assert window._banner_timer.isActive()
    window._hide_banner()
    assert not window._banner_timer.isActive()


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
