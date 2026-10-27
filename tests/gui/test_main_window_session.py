"""The main window around a pre-whitening session (design 4.4-4.9, stage G8b).

Docks and their tab order, the session bar above the table, the title bar's dirty
mark, the Save / Discard / Cancel prompt on close / open / new, Stop while busy, the
import dialog, persistence across launches, keeping each analysis's result
(UX-17), hidden tabs out of the Tab chain (A11Y-15) and the screen-aware size.
Sessions run on ``numpy``; prompts go to a FakeDialogProvider.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import GLSSettings, PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve
from cuperiod.gui.dialogs import FakeDialogProvider
from cuperiod.gui.loaders import ImportChoice, SourceItem
from cuperiod.gui.main_window import MainWindow
from cuperiod.gui.qt import QtCore, QtGui, QtWidgets
from cuperiod.gui.viewmodels import SessionView

SETTINGS = PreWhitenSettings(max_frequencies=2, samples_per_peak=5)


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path: Path) -> None:
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        str(tmp_path / "settings"),
    )


def _curve(n: int = 300) -> LightCurve:
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0.0, 40.0, n))
    y = (
        12.0
        + 0.3 * np.sin(2 * np.pi * 1.3 * t)
        + 0.1 * np.sin(2 * np.pi * 2.71 * t + 1.0)
        + rng.normal(0.0, 0.01, n)
    )
    return LightCurve.from_arrays(t, y, np.full(n, 0.01))


def _window(qtbot: QtBot, provider: FakeDialogProvider | None = None) -> MainWindow:
    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    window = MainWindow(app)
    qtbot.addWidget(window)
    if provider is not None:
        window.set_dialog_provider(provider)
    window.show()
    return window


def _settle(qtbot: QtBot, window: MainWindow) -> SessionView:
    ctl = window.controller
    bridge = ctl.session_bridge
    qtbot.waitUntil(
        lambda: bridge.is_idle() and ctl._compute.session_jobs == 0, timeout=60_000
    )
    qtbot.wait(20)
    view = ctl.state.session_view
    assert isinstance(view, SessionView)
    return view


def _auto(qtbot: QtBot, window: MainWindow) -> SessionView:
    window._controls._backend_combo.setCurrentText("numpy")
    window._controls.set_prewhiten_settings(SETTINGS)
    with qtbot.waitSignal(window.controller.solution_ready, timeout=60_000):
        window._controls.request_compute()
    return _settle(qtbot, window)


def _dirty_session(qtbot: QtBot, window: MainWindow) -> SessionView:
    """A curve in pre-whitening, Auto, then one edit: unsaved work."""
    window._load_demo(SourceItem("star", "Star", _curve))
    window._controls.set_analysis("prewhiten")
    view = _auto(qtbot, window)
    window.controller.add_frequency(4.4)
    view = _settle(qtbot, window)
    assert window.controller.is_session_dirty()
    return view


# -- docks ---
def test_prewhitening_docks_are_tabbed_with_the_session_bar_above_the_table(
    qtbot: QtBot,
) -> None:
    window = _window(qtbot)
    window._controls.set_analysis("prewhiten")
    shown = [
        window._solution_dock,
        window._subsets_dock,
        window._variations_dock,
        window._history_dock,
        window._spacing_dock,
    ]
    assert all(d.isVisible() for d in shown)
    assert not window._peaks_dock.isVisible()
    tabbed = set(window.tabifiedDockWidgets(window._solution_dock))
    assert set(shown[1:]) <= tabbed
    pane = window._solution_dock.widget()
    assert isinstance(pane, QtWidgets.QScrollArea)
    inner = pane.widget()
    assert inner is not None
    assert inner.isAncestorOf(window._session_bar)
    assert inner.isAncestorOf(window._solution_panel)
    layout = inner.layout()
    assert layout is not None
    assert layout.indexOf(window._session_bar) < layout.indexOf(window._solution_panel)


def test_hidden_dock_tabs_leave_the_tab_chain(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._controls.set_analysis("prewhiten")
    window._solution_dock.raise_()
    qtbot.wait(50)
    assert window._solution_dock.widget().isEnabled()
    behind = window._spacing_dock.widget()
    assert behind is not None and not behind.isEnabled()
    window._spacing_dock.raise_()
    qtbot.waitUntil(lambda: window._spacing_dock.widget().isEnabled())
    assert not window._solution_dock.widget().isEnabled()


def test_narrow_docks_scroll_instead_of_squeezing_the_plots(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window._controls.set_analysis("prewhiten")
    for dock in window._result_docks():
        assert dock.minimumSizeHint().width() <= 320, dock.objectName()


# -- the title bar and unsaved work ---
def test_title_bar_marks_unsaved_work_and_names_the_project(
    qtbot: QtBot, tmp_path: Path
) -> None:
    provider = FakeDialogProvider()
    window = _window(qtbot, provider)
    _dirty_session(qtbot, window)
    qtbot.waitUntil(lambda: _shown_title(window) == "Star* — cuPeriod")
    assert window.isWindowModified()
    project = tmp_path / "star.cupw.json"
    provider.answers.append(str(project))
    window.commands.action("file.save_project").trigger()
    qtbot.waitUntil(lambda: project.is_file())
    qtbot.waitUntil(
        lambda: _shown_title(window) == "Star — star.cupw.json — cuPeriod"
    )
    assert not window.isWindowModified()
    window.controller.shutdown()


def _shown_title(window: MainWindow) -> str:
    """The title the platform is given (Qt's ``[*]`` marker resolved)."""
    handle = window.windowHandle()
    assert handle is not None
    return handle.title()


def test_the_title_names_the_application_once(qtbot: QtBot) -> None:
    # Qt appends an explicitly set application display name to every title; the
    # app set one, so on Windows the bar read "cuPeriod — Star - cuPeriod".
    from cuperiod.gui.app import name_application
    from cuperiod.gui.main_window import window_title

    named: list[tuple[str, object]] = []

    class Recorder:
        def __getattr__(self, attribute: str) -> object:
            return lambda *args: named.append((attribute, args))

    name_application(Recorder())  # type: ignore[arg-type]
    assert [n for n, _ in named] == [
        "setApplicationName", "setOrganizationName", "setWindowIcon"
    ]
    window = _window(qtbot)
    assert _shown_title(window).count("cuPeriod") == 1
    window._load_demo(SourceItem("star", "Star", _curve))
    assert _shown_title(window) == "Star — cuPeriod"
    assert not window.isWindowModified()
    # The document first, the application once and last; a name holding Qt's
    # marker cannot move the "*".
    assert window_title("star.csv", "star.cupw.json") == (
        "star.csv — star.cupw.json[*] — cuPeriod"
    )
    assert window_title("a[*].csv") == "a[*][*].csv[*] — cuPeriod"
    assert window_title("", "").count("cuPeriod") == 1
    window.controller.shutdown()


def test_close_asks_about_unsaved_work_and_cancel_keeps_the_window(
    qtbot: QtBot,
) -> None:
    provider = FakeDialogProvider()
    window = _window(qtbot, provider)
    _dirty_session(qtbot, window)
    provider.answers.append("cancel")
    assert not window.close()
    assert window.isVisible()
    kind, (names,) = provider.calls[-1]
    # The source's label (what the Load demo menu called it), not its key "star".
    assert kind == "confirm_discard" and names == ("Star",)
    window._populate_sessions_menu()  # File ▸ Unsaved sessions says the same
    assert [a.text() for a in window._sessions_menu.actions()] == ["&1 Star"]
    provider.answers.append("discard")
    assert window.close()
    assert not window.isVisible()


def test_opening_a_curve_over_unsaved_work_asks_first(qtbot: QtBot) -> None:
    provider = FakeDialogProvider()
    window = _window(qtbot, provider)
    _dirty_session(qtbot, window)
    provider.calls.clear()
    provider.answers.append("cancel")
    window.commands.action("file.open_curve").trigger()
    assert [c[0] for c in provider.calls] == ["confirm_discard"]  # no file dialog
    provider.answers.extend(["discard", None])
    window.commands.action("file.open_curve").trigger()
    assert [c[0] for c in provider.calls][-2:] == ["confirm_discard", "open_file"]
    assert not window.controller.is_session_dirty()
    window.controller.shutdown()


def test_new_session_over_unsaved_work_asks_first(qtbot: QtBot) -> None:
    provider = FakeDialogProvider()
    window = _window(qtbot, provider)
    view = _dirty_session(qtbot, window)
    provider.answers.append("cancel")
    window.commands.action("file.new_session").trigger()
    assert provider.calls[-1][0] == "confirm_discard"
    assert _settle(qtbot, window).serial == view.serial  # nothing was dropped
    window.controller.shutdown()


# -- busy, Stop, feedback ---
def test_stop_is_offered_while_a_run_is_busy_and_the_strip_is_labelled(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _window(qtbot)
    window.controller.set_light_curve(_curve(), "star")
    stopped: list[bool] = []
    monkeypatch.setattr(
        window.controller, "cancel_compute", lambda: stopped.append(True)
    )
    window._on_compute_started()
    window._on_busy_changed(True)
    assert window._controls._stop_btn.isEnabled()
    assert window.commands.action("session.stop").isEnabled()
    assert "running" in window._info.activity_text()
    assert window._info._busy.accessibleName()
    window._on_session_progress(
        type(
            "R", (), {"kind": "accepted", "component_id": 7, "statistics": {"snr": 5.3}}
        )()
    )
    assert window._info.activity_text().startswith("Auto — F7 accepted, S/N 5.3")
    QtGui.QGuiApplication.processEvents()
    window.commands.action("session.stop").trigger()
    assert stopped == [True]
    window._on_busy_changed(False)
    assert not window._controls._stop_btn.isEnabled()
    assert window._info.activity_text() == ""


def test_session_messages_reach_the_status_bar_and_errors_the_banner(
    qtbot: QtBot,
) -> None:
    window = _window(qtbot)
    window.controller.session_message.emit("warning", "Nothing was changed.")
    status = window.statusBar()
    assert status is not None and status.currentMessage() == "Nothing was changed."
    assert not window._banner_container.isVisible()
    window.controller.session_message.emit("error", "The fit failed.")
    assert window._banner_container.isVisible()
    assert "The fit failed." in window._banner.text()


def test_a_failed_preview_says_so_where_the_spectrum_would_be(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window.controller.set_light_curve(_curve(), "star")
    window.controller.preview_failed.emit("Spectrum unavailable: no memory")
    assert window._placeholder.isVisible()
    assert "Spectrum unavailable: no memory" in window._placeholder.text()
    # Inside the spectrum pane too, above the light curve, which stays on show.
    spectrum = window._spectrum
    assert spectrum.isVisible() and window._raw.isVisible()
    assert spectrum.placeholder() == "Spectrum unavailable: no memory"
    assert spectrum._placeholder.isVisible()
    assert spectrum._plot.accessibleDescription() == "Spectrum unavailable: no memory"


def test_a_failed_preview_takes_down_a_result_of_another_session(
    qtbot: QtBot,
) -> None:
    # With a result on show (another band's session, say), the failure of the
    # displayed session's preview reached only the status bar, under that result.
    window = _window(qtbot)
    window._load_demo(SourceItem("star", "Star", _curve))
    window._controls.set_analysis("prewhiten")
    _auto(qtbot, window)
    assert window._showing_result and window._solution_panel.table.rowCount() > 0
    text = "Spectrum unavailable: not enough points in band r"
    window.controller.preview_failed.emit(text)
    status = window.statusBar()
    assert status is not None and status.currentMessage() == text
    spectrum = window._spectrum
    assert spectrum.isVisible() and spectrum.placeholder() == text
    assert spectrum._placeholder.isVisible()
    assert window._placeholder.isVisible() and text in window._placeholder.text()
    # As with no result: the curve full size, and nothing of the other session.
    assert window.lc_layout == "preview" and window._raw.isVisible()
    assert not window._phased.isVisible()
    assert window._solution_panel.table.rowCount() == 0
    assert not window._session_bar.isVisible()
    assert window._info.is_idle()
    window.controller.shutdown()


def test_a_failed_preview_disables_the_session_commands(qtbot: QtBot) -> None:
    """State 4.9: preview failed -> everything but the session commands."""
    window = _window(qtbot)
    window._controls.set_analysis("prewhiten")
    window._controls._backend_combo.setCurrentText("numpy")
    ctl, commands = window.controller, window.commands
    # Every time the same: no frequency grid, so the session cannot open.
    flat = LightCurve.from_arrays(np.full(40, 5.0), np.ones(40), np.full(40, 0.01))
    with qtbot.waitSignal(ctl.preview_failed, timeout=10_000):
        window._load_demo(SourceItem("flat", "Flat", lambda: flat))
    assert ctl.state.preview_error.startswith("Spectrum unavailable:")
    assert window._spectrum.placeholder() == ctl.state.preview_error
    for command_id in (
        "session.calculate", "session.improve_all", "session.add_peak",
        "file.save_project", "file.export_script", "edit.undo", "data.tool_pan",
        "file.new_session",
    ):
        assert not commands.action(command_id).isEnabled(), command_id
    assert not window._errors_action.isEnabled()
    # Auto and Step stay, like the Controls buttons: they say in the banner why not.
    assert commands.action("session.auto").isEnabled()
    assert commands.action("session.step").isEnabled()
    assert commands.action("file.open_curve").isEnabled()  # the rest works

    # Another curve: the failure is forgotten at once (the commands follow as the
    # curve arrives), and its session opens.
    seen: list[bool] = []
    ctl.lc_loaded.connect(
        lambda _lc: seen.append(commands.action("session.calculate").isEnabled())
    )
    with qtbot.waitSignal(ctl.session_changed, timeout=30_000):
        window._load_demo(SourceItem("star", "Star", _curve))
    assert seen == [True]
    assert ctl.state.preview_error == "" and window._spectrum.placeholder() == ""
    assert commands.action("file.save_project").isEnabled()


def test_the_light_curve_cursor_is_where_split_at_cuts(qtbot: QtBot) -> None:
    window = _window(qtbot)
    subsets = window._subsets_panel
    window._raw.time_cursor_changed.emit(2458004.25)
    assert subsets._split_time.value() == pytest.approx(2458004.25)
    assert subsets._time_cursor == pytest.approx(2458004.25)
    status = window.statusBar()
    assert status is not None and "Split at" in status.currentMessage()
    window._raw.time_cursor_changed.emit(float("nan"))  # another session
    assert subsets._time_cursor is None


# -- the import dialog (UX-03) ---
def _write_semicolon_csv(path: Path) -> None:
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0.0, 20.0, 80))
    v = 13.0 + 0.2 * np.sin(2 * np.pi * t / 1.7)
    lines = ["HJD;Vmag;err"] + [
        f"{a:.5f};{b:.5f};0.01" for a, b in zip(t, v, strict=True)
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_open_goes_through_the_import_dialog_and_remembers_the_choice(
    qtbot: QtBot, tmp_path: Path
) -> None:
    csv = tmp_path / "odd.csv"
    _write_semicolon_csv(csv)
    from cuperiod.gui.loaders import ReadOptions

    choice = ImportChoice(
        columns=ColumnMap(time="HJD", value="Vmag", error="err"),
        domain=Domain.MAGNITUDE,
        read=ReadOptions(delimiter=";"),
    )
    provider = FakeDialogProvider([None])  # first: cancelled
    window = _window(qtbot, provider)
    window._load_file(csv)
    assert provider.calls[-1][0] == "import_file"
    assert window.controller.state.current_lc is None
    provider.answers.append(choice)
    window._load_file(csv)
    lc = window.controller.state.current_lc
    assert isinstance(lc, LightCurve) and lc.n == 80
    # The next import starts from this choice.
    provider.answers.append(None)
    window._load_file(csv)
    kind, (path, last) = provider.calls[-1]
    assert kind == "import_file" and last == choice
    window.controller.shutdown()


def test_a_batch_file_that_will_not_load_offers_to_choose_its_columns(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = tmp_path / "a_good.csv"
    good.write_text(
        "time,mag,mag_err\n"
        + "\n".join(f"{i},{14 + 0.1 * (i % 3)},0.01" for i in range(40)),
        encoding="utf-8",
    )
    bad = tmp_path / "b_bad.csv"
    _write_semicolon_csv(bad)
    provider = FakeDialogProvider()
    window = _window(qtbot, provider)
    monkeypatch.setattr(window._controls, "request_compute", lambda: None)
    window._load_folder(tmp_path)
    window._select_source(1)
    assert window._banner_container.isVisible()
    assert window._banner_action.isVisible()
    assert "column" in window._banner_action.text().lower()
    window._banner_action.click()
    assert provider.calls[-1][0] == "import_file"
    window.controller.shutdown()


def test_a_late_result_does_not_hide_the_choose_columns_banner(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The previous source's result arriving after a file failed to load hid the
    banner, and with it the only way to that file's import dialog (its row is
    disabled)."""
    from cuperiod.core.result import Periodogram

    good = tmp_path / "a_good.csv"
    good.write_text(
        "time,mag,mag_err\n"
        + "\n".join(f"{i},{14 + 0.1 * (i % 3)},0.01" for i in range(40)),
        encoding="utf-8",
    )
    _write_semicolon_csv(tmp_path / "b_bad.csv")
    provider = FakeDialogProvider()
    window = _window(qtbot, provider)
    monkeypatch.setattr(window._controls, "request_compute", lambda: None)
    window._load_folder(tmp_path)
    window._select_source(1)  # fails; the window stays on source 0
    assert window._banner_container.isVisible() and window._banner_action.isVisible()
    assert window._source_browser.is_failed(1)
    frequency = np.linspace(0.05, 2.0, 200)
    late = Periodogram.from_spectrum(
        method="GLS",
        backend="numpy",
        frequency=frequency,
        power=np.exp(-0.5 * ((frequency - 0.7) / 0.01) ** 2),
        objective_sense="max",
        n_samples=40,
        baseline=39.0,
    )
    window.controller.compute_started.emit()  # source 0's run, still going ...
    window.controller.periodogram_ready.emit(late)  # ... and its late result
    assert window._banner_container.isVisible() and window._banner_action.isVisible()
    window._banner_action.click()
    assert provider.calls[-1][0] == "import_file"
    # A run error, by contrast, is superseded by the next result as before.
    window._hide_banner()
    window._show_banner("InsufficientDataError: 3 finite points")
    window.controller.periodogram_ready.emit(late)
    assert not window._banner_container.isVisible()
    window.controller.shutdown()


def _write_table(path: Path, header: str | None, sep: str = ",") -> Path:
    """A 40-row ``time, brightness, error`` table (``header`` None: no header row)."""
    rows = [
        sep.join((f"{2458000 + 0.37 * i:.5f}", f"{14 + 0.1 * (i % 3):.3f}", "0.010"))
        for i in range(40)
    ]
    lines = rows if header is None else [header, *rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _import_through_the_dialog(window: MainWindow, path: Path) -> object:
    """Open ``path`` with the real import dialog, press Load, return the dialog."""
    from cuperiod.gui.widgets.preview_dialog import PreviewDialog

    window._load_file(path)
    live = getattr(window._dialogs, "live_dialogs", ())
    dialogs = [d for d in live if isinstance(d, PreviewDialog)]
    assert len(dialogs) == 1
    dialog = dialogs[0]
    assert dialog.can_load(), dialog._problem.text()
    dialog.accept()
    return dialog


def test_a_flux_file_after_a_magnitude_file_loads_as_flux(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """The remembered domain must not override a file whose column names say what
    it holds (a flux CSV opened after a magnitude one loaded as magnitudes)."""
    window = _window(qtbot)
    _import_through_the_dialog(
        window, _write_table(tmp_path / "star.csv", "time,mag,mag_err")
    )
    first = window.controller.state.current_lc
    assert isinstance(first, LightCurve) and first.domain is Domain.MAGNITUDE
    _import_through_the_dialog(
        window, _write_table(tmp_path / "kepler.csv", "time,flux,flux_err")
    )
    lc = window.controller.state.current_lc
    assert isinstance(lc, LightCurve) and lc.domain is Domain.FLUX
    # And back: a magnitude file after the flux one is magnitudes again.
    _import_through_the_dialog(
        window, _write_table(tmp_path / "again.csv", "time,mag,mag_err")
    )
    lc = window.controller.state.current_lc
    assert isinstance(lc, LightCurve) and lc.domain is Domain.MAGNITUDE
    window.controller.shutdown()


def test_a_header_file_after_a_headerless_one_keeps_its_header(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """The remembered read options must not override what the next file shows of
    itself: a header CSV after an OGLE-style ``JD mag err`` table, a semicolon file."""
    window = _window(qtbot)
    _import_through_the_dialog(window, _write_table(tmp_path / "ogle.dat", None, " "))
    assert window.controller.state.current_lc is not None
    remembered = window._last_import()
    assert remembered is not None and remembered.read is not None
    assert not remembered.read.header
    dialog = _import_through_the_dialog(
        window, _write_table(tmp_path / "kepler.csv", "time,flux,flux_err")
    )
    assert dialog._header.isChecked()  # type: ignore[attr-defined]
    lc = window.controller.state.current_lc
    assert isinstance(lc, LightCurve) and lc.n == 40 and lc.domain is Domain.FLUX
    # A semicolon file is still split on opening, after a headerless one.
    _import_through_the_dialog(window, _write_table(tmp_path / "ogle2.dat", None, " "))
    dialog = _import_through_the_dialog(
        window, _write_table(tmp_path / "eu.csv", "time;mag;mag_err", ";")
    )
    assert dialog._delimiter.currentData() == ";"  # type: ignore[attr-defined]
    lc = window.controller.state.current_lc
    assert isinstance(lc, LightCurve) and lc.n == 40
    # The choice persists across launches; detection still wins there.
    _import_through_the_dialog(window, _write_table(tmp_path / "ogle3.dat", None, " "))
    window.controller.shutdown()
    window.close()
    second = _window(qtbot)
    dialog = _import_through_the_dialog(
        second, _write_table(tmp_path / "next.csv", "time,mag,mag_err")
    )
    assert dialog._header.isChecked()  # type: ignore[attr-defined]
    assert isinstance(second.controller.state.current_lc, LightCurve)
    second.controller.shutdown()


# -- UX-17: each analysis keeps its result ---
def test_switching_analysis_keeps_each_result(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window.controller.set_light_curve(_curve(), "star")
    window._controls._backend_combo.setCurrentText("finufft")
    with qtbot.waitSignal(window.controller.periodogram_ready, timeout=30_000):
        window._controls.request_compute()
    rows = window._peaks_table._table.rowCount()
    assert rows > 0
    window._controls.set_analysis("prewhiten")
    _auto(qtbot, window)
    assert window._solution_panel.table.rowCount() >= 1
    # Back to the periodogram: the spectrum and the peaks come back at once.
    window._controls.set_analysis("periodogram")
    assert window._spectrum._pg is not None and window._spectrum._pg.method == "GLS"
    assert window._peaks_table._table.rowCount() == rows
    assert window.controller.state.last_from_cache
    # And to pre-whitening again: its session is shown, editable.
    window._controls.set_analysis("prewhiten")
    view = _settle(qtbot, window)
    assert window._solution_panel.editable
    assert window._solution_panel.table.rowCount() == len(view.rows)
    window.controller.shutdown()


# -- A11Y-19: the layout, the analysis and the options persist ---
def test_layout_analysis_and_options_survive_a_relaunch(qtbot: QtBot) -> None:
    first = _window(qtbot)
    first._controls._peaks_spin.setValue(17)
    first._controls.set_analysis("prewhiten")
    first._main_split.setSizes([310, 700])
    sizes = first._main_split.sizes()
    first.close()
    second = _window(qtbot)
    assert second._controls.analysis == "prewhiten"
    assert second._controls._peaks_spin.value() == 17
    assert second._solution_dock.isVisible()
    assert second._main_split.sizes()[0] == sizes[0]


def test_options_of_a_method_survive_a_relaunch(qtbot: QtBot) -> None:
    first = _window(qtbot)
    form = first._controls._form
    assert form is not None
    first._controls._settings_cache["GLS"] = GLSSettings(nyquist_factor=9)
    first._controls._current_method = None  # the cached value is the one to keep
    first.close()
    second = _window(qtbot)
    built = second._controls._form
    assert built is not None and built.build().nyquist_factor == 9


# -- UX-20 / A11Y-12: the size follows the screen ---
def test_first_launch_fits_the_screen(qtbot: QtBot) -> None:
    window = _window(qtbot)
    screen = window.screen()
    assert screen is not None
    avail = screen.availableGeometry()
    assert window.minimumWidth() <= avail.width()
    assert window.minimumHeight() <= avail.height()
    assert window.minimumWidth() <= 1024 and window.minimumHeight() <= 680


def test_verify_project_asks_for_a_file_and_opens_it_verified(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeDialogProvider(["C:/data/star.cupw.json"])
    window = _window(qtbot, provider)
    opened: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        window.controller,
        "open_project",
        lambda path, verify=False: opened.append((str(path), verify)),
    )
    window.commands.action("file.verify_project").trigger()
    assert provider.calls[-1][0] == "open_file"
    assert opened == [("C:/data/star.cupw.json", True)]


def _saved_project(qtbot: QtBot, path: Path) -> None:
    """A pre-whitening project at ``path``, saved from a window that then closes."""
    window = _window(qtbot)
    window.controller.set_light_curve(_curve(), "star")
    window._controls.set_analysis("prewhiten")
    _auto(qtbot, window)
    window.controller.save_project(str(path))
    qtbot.waitUntil(
        lambda: path.is_file() and not window.controller.is_session_dirty(),
        timeout=30_000,
    )
    window.controller.shutdown()
    window.close()


def _status_log(window: MainWindow) -> list[str]:
    seen: list[str] = []
    bar = window.statusBar()
    assert bar is not None
    bar.messageChanged.connect(lambda text: seen.append(text) if text else None)
    return seen


def test_opening_a_project_shows_its_warnings_and_the_run_to_press(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What opening a project warned about (its GPU backend is not here, its stored
    values were re-fitted) reached only stderr; and the status then said "press
    Compute periodogram" while the window showed pre-whitening."""
    import warnings

    from cuperiod.prewhiten.session.session import PreWhitenSession
    from cuperiod.prewhiten.session.state import SessionWarning

    path = tmp_path / "p.cupw.json"
    _saved_project(qtbot, path)
    text = (
        "the project ran on 'cufinufft', which is not available here; using the "
        "best CPU backend instead"
    )
    real_load = PreWhitenSession.load

    def load(cls: object, target: object, **kwargs: object) -> PreWhitenSession:
        warnings.warn(text, SessionWarning, stacklevel=2)
        return real_load(target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(PreWhitenSession, "load", classmethod(load))
    window = _window(qtbot)
    status = _status_log(window)
    window.controller.open_project(path)
    qtbot.waitUntil(
        lambda: window.controller.state.source_id.startswith("project:"),
        timeout=30_000,
    )
    view = _settle(qtbot, window)
    assert text in view.warnings  # beside the table, with the session's first view
    assert status[-1] == "The project ran on 'cufinufft', which is not available " \
        "here; using the best CPU backend instead."
    assert any("press Run pre-whitening" in s for s in status), status
    assert not any("press Compute periodogram" in s for s in status), status
    window.controller.shutdown()


def test_a_project_that_does_not_verify_is_not_a_failed_run(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cuperiod.prewhiten.session.project import ProjectError
    from cuperiod.prewhiten.session.session import PreWhitenSession

    path = tmp_path / "p.cupw.json"
    _saved_project(qtbot, path)
    real_load = PreWhitenSession.load

    def load(cls: object, target: object, **kwargs: object) -> PreWhitenSession:
        if kwargs.get("verify"):
            raise ProjectError(
                "verify failed: replaying all 3 log entries gives F1: frequency "
                "1.31 instead of 1.3"
            )
        return real_load(target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(PreWhitenSession, "load", classmethod(load))
    window = _window(qtbot)
    status = _status_log(window)
    badge = window._info._method.text()
    window.controller.open_project(path, verify=True)
    qtbot.waitUntil(window._banner_container.isVisible, timeout=30_000)
    assert status[-1].startswith("Could not verify p.cupw.json: ProjectError: verify")
    runs = ("Compute periodogram failed", "Run pre-whitening failed")
    assert not any(s.startswith(runs) for s in status), status
    assert window._info._method.text() == badge  # no run is marked failed
    assert "Could not verify p.cupw.json" in window._banner.text()
    # Opening it (no verify) still works, and says so in its own words.
    window._hide_banner()
    window.controller.open_project(path)
    qtbot.waitUntil(
        lambda: window.controller.state.source_id.startswith("project:"),
        timeout=30_000,
    )
    _settle(qtbot, window)
    assert not window._banner_container.isVisible()
    window.controller.shutdown()
