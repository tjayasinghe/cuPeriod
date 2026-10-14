"""The light-curve pane: side by side or as tabs, and every plot readable.

With "Show Raw LC" on in a 1366 x 768 window, the fold and the raw curve used to be
stacked one above the other, each with a 45 px strip of plot under its toolbar rows
(and the spectrum not much more). They now sit side by side where the pane is wide
enough for both toolbars and take turns under Phased / Raw light curve tabs where it
is not; the raw view's tools take one wrapping row instead of three; and while the
raw curve is on, the spectrum and the light curve share the height so that their
plots are equally tall. Sessions run on ``numpy``; prompts go to a
FakeDialogProvider.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtTest
from pytestqt.qtbot import QtBot

from cuperiod.core.config import PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve
from cuperiod.gui.dialogs import FakeDialogProvider
from cuperiod.gui.loaders import SourceItem
from cuperiod.gui.main_window import MainWindow
from cuperiod.gui.qt import QtCore, QtWidgets
from cuperiod.gui.viewmodels import SessionView

Qt = QtCore.Qt
SETTINGS = PreWhitenSettings(
    max_frequencies=3, samples_per_peak=5, maximum_frequency=30.0
)

#: The least plot height (the plot widget, axes included) any visible plot gets.
MIN_PLOT = 200


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path: Path) -> None:
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        str(tmp_path / "settings"),
    )


def _campaign() -> LightCurve:
    """Three sites, nightly gaps, a few modes: what a pulsator campaign looks like."""
    rng = np.random.default_rng(11)
    t = np.concatenate(
        [
            night + site / 3 + np.sort(rng.uniform(0.0, 0.25, 25))
            for site in range(3)
            for night in range(20)
        ]
    )
    t.sort()
    y = 10.0 + rng.normal(0.0, 0.002, t.size)
    for f, a in ((9.6555, 0.03), (12.7162, 0.012), (10.4515, 0.006)):
        y += a * np.sin(2 * np.pi * f * t)
    return LightCurve.from_arrays(t, y, np.full(t.size, 0.002))


def _window(qtbot: QtBot, size: tuple[int, int]) -> MainWindow:
    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    window = MainWindow(app)
    qtbot.addWidget(window)
    window.set_dialog_provider(FakeDialogProvider())
    window.resize(*size)  # before the first show, which lays out the panes for it
    window.show()
    qtbot.waitExposed(window)
    return window


def _settle(qtbot: QtBot, window: MainWindow) -> SessionView:
    ctl = window.controller
    bridge = ctl.session_bridge
    qtbot.waitUntil(
        lambda: bridge.is_idle() and ctl._compute.session_jobs == 0, timeout=60_000
    )
    qtbot.wait(60)  # the queued fit of the pane, after the last view
    view = ctl.state.session_view
    assert isinstance(view, SessionView)
    return view


def _session(qtbot: QtBot, window: MainWindow) -> None:
    """A pre-whitening session after Auto, with the raw curve and its tools shown."""
    window._load_demo(SourceItem("campaign", "Campaign", _campaign))
    window._analysis_actions["prewhiten"].trigger()
    window._controls._backend_combo.setCurrentText("numpy")
    window._controls.set_prewhiten_settings(SETTINGS)
    with qtbot.waitSignal(window.controller.solution_ready, timeout=90_000):
        window._controls.request_compute()
    _settle(qtbot, window)
    window.commands.action("data.tool_lasso").trigger()  # shows the raw curve
    qtbot.wait(60)


def _heights(window: MainWindow) -> dict[str, int]:
    """The visible plots' heights (the plot widget, axes included), in px."""
    views = {"spectrum": window._spectrum, "phased": window._phased, "raw": window._raw}
    return {
        name: view._plot.height()
        for name, view in views.items()
        if view._plot.isVisible()
    }


def test_every_plot_is_readable_at_1366x768_with_the_raw_curve_shown(
    qtbot: QtBot,
) -> None:
    window = _window(qtbot, (1366, 768))
    _session(qtbot, window)
    tabs = window._lc_tabs
    for theme in ("dark", "light"):
        if theme == "light":
            window._toggle_theme()
            qtbot.wait(60)
        # Too narrow for both side by side: the two take turns as tabs, the raw
        # curve in front (the lasso was asked for).
        assert window.lc_layout == "tabs", theme
        assert tabs.isVisible() and tabs.accessibleName()
        assert tabs.tabText(tabs.currentIndex()) == "Raw light curve"
        shown = _heights(window)
        assert set(shown) == {"spectrum", "raw"}, theme
        assert min(shown.values()) >= MIN_PLOT, (theme, shown)
        assert abs(shown["spectrum"] - shown["raw"]) <= 1, (theme, shown)
        # The fold, from the keyboard: the split does not move.
        tabs.setFocus(Qt.FocusReason.TabFocusReason)
        QtTest.QTest.keyClick(tabs, Qt.Key.Key_Left)
        qtbot.wait(30)
        folded = _heights(window)
        assert set(folded) == {"spectrum", "phased"}, theme
        assert folded["spectrum"] == shown["spectrum"]
        assert folded["phased"] >= MIN_PLOT, (theme, folded)
        QtTest.QTest.keyClick(tabs, Qt.Key.Key_Right)
        qtbot.wait(30)
        assert window._raw.isVisible() and not window._phased.isVisible()
    window.controller.shutdown()


def test_a_wide_window_shows_the_fold_and_the_raw_curve_side_by_side(
    qtbot: QtBot,
) -> None:
    window = _window(qtbot, (1920, 1080))
    _session(qtbot, window)
    assert window.lc_layout == "side" and not window._lc_tabs.isVisible()
    shown = _heights(window)
    assert set(shown) == {"spectrum", "phased", "raw"}
    assert min(shown.values()) >= MIN_PLOT, shown
    assert abs(shown["spectrum"] - min(shown["phased"], shown["raw"])) <= 1, shown
    # Narrowed below room for both: tabs, and the plots are shared again.
    window.resize(1366, 1080)
    qtbot.waitUntil(lambda: window.lc_layout == "tabs")
    qtbot.wait(60)
    shown = _heights(window)
    assert abs(shown["spectrum"] - shown["raw"]) <= 1, shown
    window.controller.shutdown()


def test_turning_the_raw_curve_off_gives_the_spectrum_its_share_back(
    qtbot: QtBot,
) -> None:
    window = _window(qtbot, (1366, 768))
    window._load_demo(SourceItem("campaign", "Campaign", _campaign))
    window._analysis_actions["prewhiten"].trigger()
    window._controls._backend_combo.setCurrentText("numpy")
    window._controls.set_prewhiten_settings(SETTINGS)
    with qtbot.waitSignal(window.controller.solution_ready, timeout=90_000):
        window._controls.request_compute()
    _settle(qtbot, window)
    assert window.lc_layout == "single" and not window._raw.isVisible()
    before = window._plot_split.sizes()
    raw = window._raw_action  # View ▸ Show raw light curve (Ctrl+R)
    raw.trigger()
    qtbot.wait(60)
    assert window.lc_layout == "tabs" and window._raw.isVisible()
    assert window._plot_split.sizes() != before  # the light curve got more
    raw.trigger()
    qtbot.wait(60)
    assert window.lc_layout == "single" and not window._lc_tabs.isVisible()
    assert window._phased.isVisible() and window._plot_split.sizes() == before
    window.controller.shutdown()


def test_a_split_the_user_dragged_stays_until_the_layout_changes(
    qtbot: QtBot,
) -> None:
    window = _window(qtbot, (1366, 768))
    _session(qtbot, window)
    split = window._plot_split
    total = sum(split.sizes())
    split.setSizes([total - 250, 250])
    split.splitterMoved.emit(total - 250, 1)  # what a drag of the handle sends
    window.resize(1366, 800)  # the window grows: the split is not re-shared
    qtbot.wait(80)
    spectrum, lower = split.sizes()
    assert lower < spectrum  # still the user's (it was not evened out)
    window._raw_action.trigger()  # the raw curve off and on: shared afresh
    window._raw_action.trigger()
    qtbot.wait(80)
    shown = _heights(window)
    assert abs(shown["spectrum"] - shown["raw"]) <= 1, shown
    window.controller.shutdown()


def test_point_tools_bring_the_raw_curve_in_front_of_the_fold(qtbot: QtBot) -> None:
    window = _window(qtbot, (1366, 768))
    _session(qtbot, window)
    window._lc_tabs.setCurrentIndex(0)  # the fold in front
    assert window._phased.isVisible() and not window._raw.isVisible()
    window.commands.action("data.tool_rect").trigger()
    assert window._raw.isVisible() and window._raw.tool() == "rect"
    assert window._lc_tabs.tabText(window._lc_tabs.currentIndex()) == "Raw light curve"
    window._lc_tabs.setCurrentIndex(0)
    window.commands.action("view.focus_light_curve").trigger()
    assert window._raw.isVisible() and window._raw._plot.hasFocus()
    window.controller.shutdown()


def test_before_a_result_the_curve_fills_the_pane_alone(qtbot: QtBot) -> None:
    window = _window(qtbot, (1366, 768))
    window._load_demo(SourceItem("campaign", "Campaign", _campaign))
    assert window.lc_layout == "preview"
    assert window._raw.isVisible() and not window._phased.isVisible()
    assert not window._lc_tabs.isVisible() and not window._spectrum.isVisible()
    window._phased._show_raw.setChecked(True)  # no tabs before there is a fold
    assert window.lc_layout == "preview" and not window._lc_tabs.isVisible()
    window.controller.shutdown()
