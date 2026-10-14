"""Picking on the spectrum: the deferred click, the pick cursor, markers, the menu.

Every gesture is a real ``QMouseEvent`` sequence sent to the plot's viewport, so the
tests exercise pyqtgraph's own click/double-click/drag classification: pyqtgraph
reports the first half of a double click as a single click, and the view must wait
out the double-click interval before acting on it (design 4.3).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from cuperiod.core.columns import Domain
from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.qt import QtCore, QtGui, QtWidgets
from cuperiod.gui.theme import palette
from cuperiod.gui.viewmodels import CandidateView, MarkerState
from cuperiod.gui.widgets.spectrum_view import (
    PREWHITEN_METHOD,
    SNAP_PX,
    SpectrumView,
    local_maxima,
)

Qt = QtCore.Qt
L = Qt.MouseButton.LeftButton
R = Qt.MouseButton.RightButton
NONE = Qt.MouseButton.NoButton
NO_MODS = Qt.KeyboardModifier.NoModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier
CTRL = Qt.KeyboardModifier.ControlModifier
ALT = Qt.KeyboardModifier.AltModifier

PEAKS = (3.0, 7.0, 12.0)


def _bumps(frequency: np.ndarray, centres: tuple[float, ...], scale: float) -> Any:
    out = np.full(frequency.shape, 0.01)
    for i, centre in enumerate(centres):
        out += scale * (1.0 - 0.2 * i) * np.exp(-(((frequency - centre) / 0.04) ** 2))
    return out


def _data(n: int = 6000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency = np.linspace(0.05, 20.0, n)
    data = _bumps(frequency, PEAKS, 1.0)
    residual = _bumps(frequency, (12.0,), 0.3)  # what is left: the 12 c/d peak
    return frequency, data, residual


def _wrapped(frequency: np.ndarray, data: np.ndarray, method: str = PREWHITEN_METHOD,
             sense: str = "max") -> Periodogram:
    return Periodogram.from_spectrum(
        method=method, backend="numpy", frequency=frequency, power=data,
        objective_sense=sense,  # type: ignore[arg-type]
        n_samples=500, baseline=40.0,
    )


def _view(qtbot: QtBot, *, pick: bool = True) -> SpectrumView:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    view.resize(900, 460)
    view.show()
    qtbot.waitExposed(view)
    frequency, data, residual = _data()
    view.set_periodogram(_wrapped(frequency, data))
    view.set_overlay(frequency, residual)
    view.set_pick_enabled(pick)
    _settle(qtbot)
    return view


def _settle(qtbot: QtBot) -> None:
    """Let pending layout and range updates land before positions are computed."""
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
        qtbot.wait(15)


def _vp(view: SpectrumView, frequency: float, y: float) -> QtCore.QPointF:
    """Viewport position of the data point ``(frequency, y)`` (frequency axis)."""
    box = view._plot.getPlotItem().vb
    scene = box.mapViewToScene(QtCore.QPointF(view._to_plot_x(frequency), y))
    return QtCore.QPointF(view._plot.mapFromScene(scene))


def _mouse(
    view: SpectrumView,
    kind: QtCore.QEvent.Type,
    pos: QtCore.QPointF,
    button: Any = L,
    mods: Any = NO_MODS,
) -> None:
    viewport = view._plot.viewport()
    if kind == QtCore.QEvent.Type.MouseMove:
        # pyqtgraph drops moves closer together than its rate limit (10 ms at the
        # default 100/s), so a synthetic drag faster than that never starts.
        QtCore.QThread.msleep(12)
        buttons = button
        button = NONE
    elif kind == QtCore.QEvent.Type.MouseButtonRelease:
        buttons = NONE
    else:
        buttons = button
    event = QtGui.QMouseEvent(
        kind, pos, viewport.mapToGlobal(pos.toPoint()).toPointF(),
        button, buttons, mods,
    )
    QtWidgets.QApplication.sendEvent(viewport, event)
    QtWidgets.QApplication.processEvents()


def _hover(view: SpectrumView, pos: QtCore.QPointF) -> None:
    _mouse(view, QtCore.QEvent.Type.MouseMove, pos, NONE)


def _click(view: SpectrumView, pos: QtCore.QPointF, button: Any = L,
           mods: Any = NO_MODS) -> None:
    _hover(view, pos)
    _mouse(view, QtCore.QEvent.Type.MouseButtonPress, pos, button, mods)
    _mouse(view, QtCore.QEvent.Type.MouseButtonRelease, pos, button, mods)


def _double_click(view: SpectrumView, pos: QtCore.QPointF) -> None:
    _hover(view, pos)
    _mouse(view, QtCore.QEvent.Type.MouseButtonPress, pos)
    _mouse(view, QtCore.QEvent.Type.MouseButtonRelease, pos)
    _mouse(view, QtCore.QEvent.Type.MouseButtonDblClick, pos)
    _mouse(view, QtCore.QEvent.Type.MouseButtonRelease, pos)


def _drag(view: SpectrumView, start: QtCore.QPointF, end: QtCore.QPointF) -> None:
    _hover(view, start)
    _mouse(view, QtCore.QEvent.Type.MouseButtonPress, start)
    steps = 10
    for i in range(1, steps + 1):
        point = start + (end - start) * (i / steps)
        _mouse(view, QtCore.QEvent.Type.MouseMove, point, L)
    _mouse(view, QtCore.QEvent.Type.MouseButtonRelease, end)


def _wait_out_double_click(qtbot: QtBot) -> None:
    qtbot.wait(QtWidgets.QApplication.doubleClickInterval() + 150)


class _Recorder:
    def __init__(self, view: SpectrumView) -> None:
        self.cursor: list[float] = []
        self.periods: list[float] = []
        self.adds: list[tuple[float, bool]] = []
        self.components: list[int] = []
        view.cursor_changed.connect(self.cursor.append)
        view.period_selected.connect(self.periods.append)
        view.add_requested.connect(lambda f, exact: self.adds.append((f, exact)))
        view.component_clicked.connect(self.components.append)


# -- the deferred single click -------------------------------------------------------


def test_click_moves_the_cursor_after_the_double_click_interval(qtbot: QtBot) -> None:
    view = _view(qtbot)
    seen = _Recorder(view)
    _click(view, _vp(view, 12.0 + 0.05, 0.05))  # a little off the residual peak
    assert seen.cursor == []  # deferred: nothing yet
    with qtbot.waitSignal(view.cursor_changed, timeout=3000):
        pass
    cursor = view.cursor_frequency()
    assert cursor is not None and abs(cursor - 12.0) < 0.01  # snapped to the peak
    assert seen.cursor == [cursor]
    assert seen.periods == []  # the pick cursor never moves the fold
    assert view._cursor_line.isVisible()


def test_alt_click_puts_the_cursor_exactly_where_clicked(qtbot: QtBot) -> None:
    view = _view(qtbot)
    _click(view, _vp(view, 12.05, 0.05), mods=ALT)
    _wait_out_double_click(qtbot)
    cursor = view.cursor_frequency()
    assert cursor is not None and abs(cursor - 12.05) < 0.02 and abs(cursor - 12) > 0.02


def test_double_click_resets_the_view_and_leaves_the_cursor(qtbot: QtBot) -> None:
    view = _view(qtbot)
    box = view._plot.getPlotItem().vb
    box.setXRange(11.0, 13.0, padding=0.0)
    seen = _Recorder(view)
    _double_click(view, _vp(view, 12.3, 0.05))
    _wait_out_double_click(qtbot)
    assert seen.cursor == [] and view.cursor_frequency() is None
    (x0, x1), _ = box.viewRange()
    assert x1 - x0 > 15.0  # back to the full spectrum


def test_a_drag_pans_and_does_not_move_the_cursor(qtbot: QtBot) -> None:
    view = _view(qtbot)
    box = view._plot.getPlotItem().vb
    box.setXRange(4.0, 10.0, padding=0.0)
    before = box.viewRange()[0]
    seen = _Recorder(view)
    _drag(view, _vp(view, 8.0, 0.3), _vp(view, 6.0, 0.3))
    _wait_out_double_click(qtbot)
    assert seen.cursor == [] and seen.periods == []
    assert box.viewRange()[0] != before  # it panned


def test_right_click_builds_the_menu_at_once(qtbot: QtBot) -> None:
    view = _view(qtbot)
    seen = _Recorder(view)
    _click(view, _vp(view, 12.02, 0.1), button=R)
    menu = view._context_menu
    assert menu is not None and menu.isVisible()  # no wait for the interval
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert texts[0].startswith("Add peak at ")
    assert abs(float(texts[0].split()[3]) - 12.0) < 0.01
    assert "Add exactly here" in texts and "Fold here" in texts
    menu.close()
    _wait_out_double_click(qtbot)
    assert seen.cursor == [] and seen.periods == []  # a right click picks nothing


def test_shift_click_asks_to_add_the_peak_there(qtbot: QtBot) -> None:
    view = _view(qtbot)
    seen = _Recorder(view)
    _click(view, _vp(view, 12.04, 0.05), mods=SHIFT)
    assert len(seen.adds) == 1  # immediate: a held modifier is deliberate
    frequency, exact = seen.adds[0]
    assert not exact and abs(frequency - 12.0) < 0.01
    assert view.cursor_frequency() == pytest.approx(frequency)
    _click(view, _vp(view, 12.04, 0.05), mods=SHIFT | CTRL)
    frequency, exact = seen.adds[1]
    assert exact and abs(frequency - 12.04) < 0.02 and abs(frequency - 12.0) > 0.01
    _wait_out_double_click(qtbot)
    assert seen.periods == []


def test_shift_click_outside_pick_mode_only_selects(qtbot: QtBot) -> None:
    view = _view(qtbot, pick=False)
    seen = _Recorder(view)
    _click(view, _vp(view, 7.02, 0.05), mods=SHIFT)
    _wait_out_double_click(qtbot)
    assert seen.adds == []
    assert len(seen.periods) == 1 and abs(1.0 / seen.periods[0] - 7.0) < 0.01


# -- markers -----------------------------------------------------------------------


def _states(*, selected: int | None = None) -> list[MarkerState]:
    return [
        MarkerState(1, 3.0, "free", selected == 1),
        MarkerState(2, 7.0, "fixed", selected == 2),
        MarkerState(3, 12.0, "tied", selected == 3),
        MarkerState(4, 15.0, "inactive", selected == 4),
    ]


def _marker_pos(view: SpectrumView, component_id: int) -> QtCore.QPointF:
    for spot in view._markers.points():
        if spot.data() == ("component", component_id):
            return _spot_pos(view, spot)
    raise AssertionError(f"no marker for F{component_id}")


def _spot_pos(view: SpectrumView, spot: Any) -> QtCore.QPointF:
    box = view._plot.getPlotItem().vb
    return QtCore.QPointF(view._plot.mapFromScene(box.mapViewToScene(spot.pos())))


def test_marker_shapes_carry_the_component_state(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.set_marker_states(_states(selected=2))
    data = view._markers.data
    by_id = {d["data"][1]: d for d in data if d["data"] is not None}
    assert {k: by_id[k]["symbol"] for k in by_id} == {1: "o", 2: "s", 3: "d", 4: "o"}
    assert by_id[4]["brush"].style() == Qt.BrushStyle.NoBrush  # hollow: not used
    assert all(by_id[k]["brush"].style() != Qt.BrushStyle.NoBrush for k in (1, 2, 3))
    assert by_id[2]["size"] > by_id[1]["size"]  # selected is larger, with a halo
    halos = [d for d in data if d["data"] is None]
    assert len(halos) == 1
    # Markers stand on the data spectrum.
    assert by_id[1]["y"] == pytest.approx(view._height_at(3.0))
    view.set_marker_states(None)  # back to plain peaks
    assert len(view._markers.data) == 0


def test_marker_click_reports_the_component(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.set_marker_states(_states())
    seen = _Recorder(view)
    _click(view, _marker_pos(view, 2))
    assert seen.components == [2]  # a direct hit is immediate
    _wait_out_double_click(qtbot)
    assert seen.cursor == [] and seen.periods == []


def test_a_near_miss_on_a_marker_still_selects_it(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.set_marker_states(_states())
    seen = _Recorder(view)
    near = _marker_pos(view, 3) + QtCore.QPointF(SNAP_PX - 3.0, 0.0)
    _click(view, near)
    with qtbot.waitSignal(view.component_clicked, timeout=3000) as blocker:
        pass
    assert blocker.args == [3] and seen.cursor == []


def test_near_miss_selects_a_periodogram_peak(qtbot: QtBot) -> None:
    view = _view(qtbot, pick=False)
    frequency, data, _ = _data()
    view.set_peaks(
        [
            Peak(period=1 / f, frequency=f, power=float(np.interp(f, frequency, data)),
                 rank=i + 1)
            for i, f in enumerate(PEAKS)
        ]
    )
    picked: list[Peak] = []
    view.peak_selected.connect(picked.append)
    (spot,) = [s for s in view._markers.points() if s.data() == 1]  # the 7 c/d peak
    _click(view, _spot_pos(view, spot) + QtCore.QPointF(0.0, -(SNAP_PX - 3.0)))
    qtbot.waitUntil(lambda: bool(picked), timeout=3000)
    assert picked[0].frequency == pytest.approx(7.0)


# -- periodogram mode: a click selects (UX-19) --------------------------------------


def test_periodogram_click_folds_at_the_nearest_peak(qtbot: QtBot) -> None:
    view = _view(qtbot, pick=False)
    seen = _Recorder(view)
    _click(view, _vp(view, 7.03, 0.02))  # near the baseline beside the 7 c/d peak
    assert seen.periods == []  # deferred
    qtbot.waitUntil(lambda: bool(seen.periods), timeout=3000)
    assert abs(1.0 / seen.periods[0] - 7.0) < 0.005
    assert view._sel_period == pytest.approx(seen.periods[0])
    # The keyboard cursor follows the selection and is not drawn over the band.
    assert view.cursor_frequency() == pytest.approx(1.0 / seen.periods[0])
    assert not view._cursor_line.isVisible()


def test_periodogram_click_selects_minima_for_minimise_methods(qtbot: QtBot) -> None:
    view = _view(qtbot, pick=False)
    frequency, data, _ = _data()
    view.set_periodogram(_wrapped(frequency, 1.0 - data, method="PDM", sense="min"))
    seen = _Recorder(view)
    _click(view, _vp(view, 3.03, 0.9))
    qtbot.waitUntil(lambda: bool(seen.periods), timeout=3000)
    assert abs(1.0 / seen.periods[0] - 3.0) < 0.005


def test_band_drag_still_selects_a_period(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.set_selected_period(1.0 / 7.0)
    view.set_cursor(12.0)
    seen = _Recorder(view)
    lo, hi = view._sel_band.getRegion()
    centre = 0.5 * (lo + hi)
    _drag(view, _vp(view, centre, 0.4), _vp(view, centre + 1.0, 0.4))
    _wait_out_double_click(qtbot)
    assert seen.periods and abs(1.0 / seen.periods[-1] - 8.0) < 0.1
    assert view.cursor_frequency() == 12.0 and seen.cursor == []


# -- the context menu ----------------------------------------------------------------


def test_context_menu_texts_and_actions(qtbot: QtBot) -> None:
    view = _view(qtbot)
    frequency, _, residual = _data()
    view.set_threshold(frequency, np.full_like(frequency, 0.04), snr_threshold=4.0)
    menu = view.build_context_menu(12.0, exact_frequency=12.03)
    actions = {a.text(): a for a in menu.actions() if not a.isSeparator()}
    add_text = next(t for t in actions if t.startswith("Add peak at"))
    # residual 0.31 over a noise of 0.04 / 4 = 0.01
    assert add_text == f"Add peak at 12 c/d (S/N ≈ {residual.max() / 0.01:.1f})"
    assert list(actions)[1:] == [
        "Add exactly here", "Fold here", "Move cursor here", "Zoom to ±1 c/d",
        "Copy frequency", "Reset view",
    ]
    assert all(a.statusTip() for a in actions.values())
    seen = _Recorder(view)
    actions["Add exactly here"].trigger()
    assert seen.adds == [(12.03, True)]
    actions["Fold here"].trigger()
    assert seen.periods and abs(1.0 / seen.periods[0] - 12.0) < 0.005
    actions["Move cursor here"].trigger()
    assert view.cursor_frequency() == 12.0
    actions["Copy frequency"].trigger()
    assert QtGui.QGuiApplication.clipboard().text() == "12"
    actions["Zoom to ±1 c/d"].trigger()
    (x0, x1), _ = view._plot.getPlotItem().vb.viewRange()
    assert 10.9 < x0 < 11.1 and 12.9 < x1 < 13.2


def test_context_menu_outside_pick_mode_has_no_add(qtbot: QtBot) -> None:
    view = _view(qtbot, pick=False)
    texts = [a.text() for a in view.build_context_menu(7.0).actions()]
    assert not any(t.startswith("Add") for t in texts) and "Fold here" in texts


# -- candidate, cursor, threshold, traces ------------------------------------------


def test_candidate_and_cursor_stay_out_of_the_peaks(qtbot: QtBot) -> None:
    view = _view(qtbot)
    peaks = [Peak(period=1 / 7.0, frequency=7.0, power=0.8, rank=1)]
    view.set_peaks(peaks)
    view.set_candidate(CandidateView(12.0, 1 / 12.0, 0.31, 9.5, "next"))
    view.set_cursor(12.0)
    assert view._peaks == peaks
    assert all(spot.data() in (None, 0) for spot in view._markers.points())
    assert view.candidate_frequency() == 12.0
    label = view._candidate_label
    assert label.isVisible() and label.toPlainText().strip() == "next"
    (spot,) = view._candidate_item.points()
    assert spot.pos().x() == pytest.approx(12.0)
    assert spot.brush().style() == Qt.BrushStyle.NoBrush  # a hollow triangle
    view.set_candidate(None)
    assert len(view._candidate_item.points()) == 0 and not label.isVisible()


def test_n_is_not_needed_to_read_the_candidate(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.set_candidate(CandidateView(12.0, 1 / 12.0, 0.31, 9.5, "next"))
    assert "next candidate f = 12" in view.accessible_summary()


def test_update_traces_keeps_range_peaks_selection_and_cursor(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.set_marker_states(_states())
    view.set_selected_period(1.0 / 7.0)
    view.set_cursor(12.0)
    box = view._plot.getPlotItem().vb
    box.setXRange(10.0, 14.0, padding=0.0)
    box.setYRange(0.0, 0.5, padding=0.0)
    before = box.viewRange()
    frequency, data, _ = _data()
    flatter = _bumps(frequency, (12.0,), 0.05)
    view.update_traces(_wrapped(frequency, data * 1.1), (frequency, flatter))
    assert np.ravel(box.viewRange()) == pytest.approx(np.ravel(before))
    assert view._sel_period == pytest.approx(1.0 / 7.0)
    assert view.cursor_frequency() == 12.0
    assert len([s for s in view._markers.points() if s.data() is not None]) == 4
    assert view._overlay_xy is not None and view._overlay_xy[1].max() < 0.1
    view.update_traces(_wrapped(frequency, data), None)
    assert view._overlay_xy is None and not view._overlay.isVisible()
    assert np.ravel(box.viewRange()) == pytest.approx(np.ravel(before))


def test_update_traces_accepts_amplitude_spectra(qtbot: QtBot) -> None:
    view = SpectrumView("dark")
    qtbot.addWidget(view)
    frequency, data, residual = _data(800)

    class _Spectrum:
        def __init__(self, amplitude: np.ndarray) -> None:
            self.frequency = frequency
            self.amplitude = amplitude
            self.backend = "numpy"
            self.n_samples = 400
            self.baseline = 30.0

    view.update_traces(_Spectrum(data), _Spectrum(residual))  # the first: auto-ranges
    assert view._pg is not None and view._pg.method == PREWHITEN_METHOD
    assert np.array_equal(view._overlay_xy[1], residual)  # type: ignore[index]


def test_threshold_trace_is_named_in_the_legend(qtbot: QtBot) -> None:
    view = _view(qtbot)
    frequency, _, _ = _data()
    view.set_threshold(frequency, np.full_like(frequency, 0.05), snr_threshold=4.0)
    assert view._threshold_curve.isVisible()
    names = [label.item.toPlainText() for _, label in view._legend.items]
    assert "S/N limit" in names
    pen = view._threshold_curve.opts["pen"]
    assert pen.style() == Qt.PenStyle.DashDotDotLine  # told apart by dash, not colour
    view.clear_threshold()
    assert not view._threshold_curve.isVisible()
    names = [label.item.toPlainText() for _, label in view._legend.items]
    assert "S/N limit" not in names


def test_new_spectrum_drops_the_pick_state(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.set_marker_states(_states())
    view.set_cursor(12.0)
    view.set_candidate(CandidateView(12.0, 1 / 12.0, 0.31, 9.5, "next"))
    frequency, data, _ = _data()
    view.set_periodogram(_wrapped(frequency, data))
    assert view.cursor_frequency() is None and view.candidate_frequency() is None
    assert view._marker_states is None and not view._cursor_line.isVisible()


# -- axis labels and pens (UX-16, A11Y-10) -------------------------------------------


def test_amplitude_axis_names_its_unit(qtbot: QtBot) -> None:
    view = _view(qtbot)
    axis = view._plot.getPlotItem().getAxis("left")
    assert axis.labelText == "Amplitude"
    view.set_amplitude_unit(Domain.MAGNITUDE)
    assert axis.labelText == "Amplitude (mag)"
    view.set_amplitude_unit("flux")
    assert axis.labelText == "Amplitude (flux)"
    view.set_cursor(12.0)
    assert view._readout.text().endswith("flux")
    view.set_amplitude_unit(None)
    assert axis.labelText == "Amplitude"
    # The tick values are the plotted values: no "(x0.001)" rescaling.
    assert axis.autoSIPrefixScale == 1.0


def test_residual_overlay_uses_the_residual_colour(qtbot: QtBot) -> None:
    view = _view(qtbot)
    for name in ("dark", "light"):
        pal = palette(name)
        view.apply_theme(pal)
        pen = view._overlay.opts["pen"]
        assert pen.color().name() == QtGui.QColor(pal.residual).name()
        assert pen.color().alpha() == 255 and pen.widthF() >= 2.0


def test_local_maxima_counts_plateaus_once_and_skips_nan() -> None:
    values = np.array([0.0, 2.0, 1.0, 3.0, 3.0, 1.0, np.nan, 5.0, 4.0])
    assert local_maxima(values).tolist() == [1, 3, 7]
    assert local_maxima(values, maximise=False).tolist() == [0, 2, 5, 8]
    assert local_maxima(np.zeros(0)).size == 0


def test_a_real_session_view_fills_the_spectrum(qtbot: QtBot) -> None:
    import cuperiod as cup
    from cuperiod.gui.viewmodels import SessionKey, build_session_view

    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0.0, 30.0, 400))
    y = 0.3 * np.sin(2 * np.pi * 1.3 * t) + 0.1 * np.sin(2 * np.pi * 2.71 * t)
    y += rng.normal(0.0, 0.01, t.size)
    session = cup.PreWhitenSession((t, y, np.full(t.size, 0.01)), backend="numpy")
    session.step()
    view = build_session_view(session, key=SessionKey("s"), id_epoch=0)
    spectrum = SpectrumView("dark")
    qtbot.addWidget(spectrum)
    spectrum.set_pick_enabled(True)
    spectrum.set_session_view(view, selected=[view.rows[0].id])
    components = [s for s in spectrum._markers.points() if s.data() is not None]
    assert len(components) == len(view.rows) == 1
    assert view.candidate is not None
    assert spectrum.candidate_frequency() == pytest.approx(view.candidate.frequency)
    assert spectrum._threshold_curve.isVisible() == (view.threshold is not None)
    assert spectrum._show_window.isVisibleTo(spectrum)
    assert spectrum._overlay_xy is not None
    # Enter on the cursor at the candidate goes to the selection, not to an add.
    spectrum.set_cursor(spectrum.candidate_frequency())
    with qtbot.waitSignal(spectrum.period_selected):
        spectrum._fold_at_cursor()
