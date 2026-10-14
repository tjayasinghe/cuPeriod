"""Selecting, deleting and restoring light-curve points in a session (design 4.7).

The raw light-curve view in session mode plots the data, the residuals or the data
with the model, coloured and shaped by subset; Rectangle / Lasso (and Shift-drag in
Pan mode) select points, the selection bar deletes or restores them, and deleted points
stay visible as hollow grey crosses. The view only *asks*: it emits point indices.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

import cuperiod as cup
from cuperiod.core.columns import Domain
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.gui.qt import QtCore, QtGui, QtWidgets
from cuperiod.gui.theme import palette
from cuperiod.gui.viewmodels import SessionKey, build_session_view
from cuperiod.gui.widgets.phased_view import MAX_DRAWN_POINTS, PhasedView
from cuperiod.gui.widgets.rawlc_view import RawLightCurveView, outliers_beyond

Qt = QtCore.Qt
QTest = pytest.importorskip("PySide6.QtTest").QTest

N = 200
OUTLIERS = (17, 90, 150)


def _fake_view(*, key: str = "s", n: int = N, deleted: tuple[int, ...] = (5, 6)) -> Any:
    """A fabricated session view: two nights, a sinusoid, three outliers."""
    rng = np.random.default_rng(3)
    time = 2450000.0 + np.sort(
        np.concatenate((rng.uniform(0, 0.4, n // 2), rng.uniform(1.0, 1.4, n - n // 2)))
    )
    model = 12.0 + 0.2 * np.sin(2 * np.pi * 3.0 * time)
    value = model + rng.normal(0.0, 0.01, n)
    for i in OUTLIERS:
        if i < n:
            value[i] += 0.3
    codes = (time > time[0] + 0.7).astype(np.int64)
    mask = np.zeros(n, dtype=np.bool_)
    mask[list(deleted)] = True
    return SimpleNamespace(
        key=key,
        time=time,
        value=value,
        model=model,
        residuals=value - model,
        deleted=mask,
        point_codes=codes,
        partitions=("night",),
        subsets=(
            SimpleNamespace(partition="night", index=0, label="night 1"),
            SimpleNamespace(partition="night", index=1, label="night 2"),
        ),
        offsets_by="night",
    )


def _raw(qtbot: QtBot, view: Any | None = None) -> RawLightCurveView:
    raw = RawLightCurveView("dark")
    qtbot.addWidget(raw)
    raw.resize(900, 500)
    raw.show()
    qtbot.waitExposed(raw)
    shown = _fake_view() if view is None else view
    raw.set_session_view(shown, domain=Domain.MAGNITUDE)
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
        qtbot.wait(15)
    return raw


def _items(raw: RawLightCurveView) -> list[Any]:
    return list(raw._session_items)


def _legend(raw: RawLightCurveView) -> list[str]:
    return [label.item.toPlainText() for _, label in raw._legend.items]


def _rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def test_session_points_are_coloured_and_shaped_by_subset(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    assert raw._session_bar.isVisibleTo(raw)
    scatters = [i for i in _items(raw) if isinstance(i, type(raw._rings))]
    symbols = [s.opts["symbol"] for s in scatters]
    assert symbols[:2] == ["o", "s"]  # the two nights differ by shape too
    assert scatters[0].opts["brush"].color() != scatters[1].opts["brush"].color()
    assert _legend(raw) == ["night 1", "night 2", "deleted (2)"]
    assert raw._plot.getPlotItem().getAxis("left").labelText == "Magnitude"
    assert raw._plot.getPlotItem().vb.state["yInverted"]
    assert "200 points" in raw._title.text() and "2 deleted" in raw._title.text()
    assert "200 points, 2 deleted, 0 selected" in raw._plot.accessibleDescription()


def test_deleted_points_are_hollow_grey_crosses(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    (crosses,) = [i for i in _items(raw) if getattr(i, "opts", {}).get("symbol") == "x"]
    assert crosses.opts["brush"].style() == Qt.BrushStyle.NoBrush
    assert crosses.opts["pen"].color().name() == QtGui.QColor(raw._pal.muted).name()
    x, _ = crosses.getData()
    assert len(x) == 2


def test_rectangle_selects_and_delete_asks_for_those_points(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    counts: list[int] = []
    raw.selection_changed.connect(counts.append)
    x, y = raw._plot_xy()
    # The first night's points fainter than 12.1 mag (deleted ones included).
    raw._box.region_selected.emit(_rect(-1.0, 12.1, 0.5, 13.0), False)
    expected = np.flatnonzero((x <= 0.5) & (y >= 12.1))
    assert {5, 6} <= set(expected.tolist()) and expected.size > 3
    assert raw.selected_indices().tolist() == expected.tolist()
    assert counts == [expected.size]
    assert raw._selection_label.text() == f"{expected.size} points selected"
    assert raw._delete_button.isEnabled() and raw._restore_button.isEnabled()
    rx, _ = raw._rings.getData()
    assert len(rx) == expected.size  # every selected point is ringed
    with qtbot.waitSignal(raw.delete_requested) as blocker:
        raw._delete_button.click()
    # Points already deleted are not deleted twice.
    assert blocker.args[0].tolist() == [i for i in expected.tolist() if i not in (5, 6)]
    assert raw.selection_count() == 0 and counts[-1] == 0


def test_lasso_and_additive_selection(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    x, y = raw._plot_xy()
    i, j = OUTLIERS[0], OUTLIERS[2]
    triangle = np.array(
        [[x[i] - 1e-4, y[i] - 0.05], [x[i] + 1e-4, y[i] - 0.05], [x[i], y[i] + 0.05]]
    )
    raw._box.region_selected.emit(triangle, False)
    assert raw.selected_indices().tolist() == [i]
    box = _rect(x[j] - 1e-4, y[j] - 0.01, x[j] + 1e-4, y[j] + 0.01)
    raw._box.region_selected.emit(box, True)  # Ctrl: add to the selection
    assert raw.selected_indices().tolist() == [i, j]
    raw._box.region_selected.emit(box, False)  # replace
    assert raw.selected_indices().tolist() == [j]


def test_deleted_points_can_be_selected_and_restored(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    raw.set_selection([5, 6, 7])
    assert raw._restore_button.isEnabled() and raw._delete_button.isEnabled()
    restored: list[Any] = []
    deleted: list[Any] = []
    raw.restore_requested.connect(restored.append)
    raw.delete_requested.connect(deleted.append)
    raw.restore_selected()
    assert restored[0].tolist() == [5, 6] and deleted == []
    assert raw.selection_count() == 0


def test_residual_mode_selects_on_the_residuals(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    raw.set_y_mode("residuals")
    x, y = raw._plot_xy()
    assert np.allclose(y, raw._session.residuals)
    assert raw._plot.getPlotItem().getAxis("left").labelText == "Residual (mag)"
    raw._box.region_selected.emit(_rect(-1.0, 0.2, 5.0, 0.5), False)
    assert raw.selected_indices().tolist() == list(OUTLIERS)
    assert "residuals versus time" in raw._plot.accessibleDescription()


def test_model_mode_draws_the_model_broken_at_gaps(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    raw.set_y_mode("model")
    assert "model" in _legend(raw)
    (line,) = [i for i in _items(raw) if isinstance(i, type(raw._plot.plot()))]
    connect = line.opts["connect"]
    xs, _ = line.getData()
    night_gap = int(np.argmax(np.diff(xs)))
    assert isinstance(connect, np.ndarray) and not connect[night_gap]
    assert int((~connect[:-1]).sum()) < 10  # within a night the line is unbroken
    pen = line.opts["pen"]
    assert pen.color().name() == QtGui.QColor(raw._pal.residual).name()
    assert pen.widthF() >= 2.0


def test_select_beyond_sigma_works_per_subset(qtbot: QtBot) -> None:
    view = _fake_view()
    view.residuals[view.point_codes == 1] += 1.0  # night 2 sits 1 mag off: no matter
    raw = _raw(qtbot, view)
    assert raw.select_beyond(4.0) == 3
    assert raw.selected_indices().tolist() == list(OUTLIERS)
    raw._sigma_spin.setValue(1.0)  # the spin box drives the button
    raw._beyond_button.click()
    assert raw.selection_count() > 20


def test_outliers_beyond_ignores_unusable_points() -> None:
    r = np.random.default_rng(5).normal(0.0, 0.01, 300)
    r[3] = 5.0
    usable = np.ones(300, dtype=np.bool_)
    usable[3] = False  # deleted: not judged, not selected
    assert not outliers_beyond(r, None, usable, 5.0).any()
    usable[3] = True
    assert np.flatnonzero(outliers_beyond(r, None, usable, 5.0)).tolist() == [3]
    flat = np.zeros(10)  # no scatter at all: nothing is an outlier
    assert not outliers_beyond(flat, None, np.ones(10, dtype=np.bool_), 3.0).any()


def test_a_real_rectangle_drag_selects(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    raw.set_tool("rect")
    box = raw._plot.getPlotItem().vb
    x, y = raw._plot_xy()
    i = OUTLIERS[1]

    def vp(px: float, py: float) -> QtCore.QPointF:
        scene = box.mapViewToScene(QtCore.QPointF(px, py))
        return QtCore.QPointF(raw._plot.mapFromScene(scene))

    start, end = vp(x[i] - 0.02, y[i] + 0.04), vp(x[i] + 0.02, y[i] - 0.04)
    viewport = raw._plot.viewport()

    def send(kind: QtCore.QEvent.Type, pos: QtCore.QPointF, buttons: Any) -> None:
        if kind == QtCore.QEvent.Type.MouseMove:
            QtCore.QThread.msleep(12)  # pyqtgraph's mouse rate limit
        moving = kind == QtCore.QEvent.Type.MouseMove
        button = Qt.MouseButton.NoButton if moving else Qt.MouseButton.LeftButton
        event = QtGui.QMouseEvent(
            kind, pos, viewport.mapToGlobal(pos.toPoint()).toPointF(), button,
            buttons, Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(viewport, event)
        QtWidgets.QApplication.processEvents()

    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    send(QtCore.QEvent.Type.MouseMove, start, none)
    send(QtCore.QEvent.Type.MouseButtonPress, start, left)
    for k in range(1, 9):
        send(QtCore.QEvent.Type.MouseMove, start + (end - start) * (k / 8), left)
    send(QtCore.QEvent.Type.MouseButtonRelease, end, none)
    assert i in raw.selected_indices().tolist()


def _send(
    raw: RawLightCurveView, kind: QtCore.QEvent.Type, pos: QtCore.QPointF,
    buttons: Any, mods: Any = Qt.KeyboardModifier.NoModifier,
) -> None:
    viewport = raw._plot.viewport()
    moving = kind == QtCore.QEvent.Type.MouseMove
    if moving:
        QtCore.QThread.msleep(12)  # pyqtgraph's mouse rate limit
    button = Qt.MouseButton.NoButton if moving else Qt.MouseButton.LeftButton
    event = QtGui.QMouseEvent(
        kind, pos, viewport.mapToGlobal(pos.toPoint()).toPointF(), button, buttons,
        mods,
    )
    QtWidgets.QApplication.sendEvent(viewport, event)
    QtWidgets.QApplication.processEvents()


def _at(raw: RawLightCurveView, x: float, y: float) -> QtCore.QPointF:
    """Viewport position of data point (x, y)."""
    scene = raw._plot.getPlotItem().vb.mapViewToScene(QtCore.QPointF(x, y))
    return QtCore.QPointF(raw._plot.mapFromScene(scene))


def _click_at(
    raw: RawLightCurveView, pos: QtCore.QPointF,
    mods: Any = Qt.KeyboardModifier.NoModifier,
) -> None:
    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    _send(raw, QtCore.QEvent.Type.MouseMove, pos, none)
    _send(raw, QtCore.QEvent.Type.MouseButtonPress, pos, left, mods)
    _send(raw, QtCore.QEvent.Type.MouseButtonRelease, pos, none, mods)


def test_a_click_puts_the_time_cursor_there_and_a_pan_does_not(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    times: list[float] = []
    raw.time_cursor_changed.connect(times.append)
    box = raw._plot.getPlotItem().vb
    origin = float(np.min(raw._session.time))
    (x0, x1), (y0, y1) = box.viewRange()
    target = x0 + 0.3 * (x1 - x0)
    _click_at(raw, _at(raw, target, 0.5 * (y0 + y1)))
    pixel = (x1 - x0) / box.width()
    assert len(times) == 1 and times[0] == pytest.approx(origin + target, abs=3 * pixel)
    assert raw.time_cursor() == times[0]  # in the session's own time, not the axis's
    line = raw._cursor_line
    assert line.isVisible() and line.value() == pytest.approx(times[0] - origin)
    assert line.label.toPlainText() == "cursor"  # named, not told by colour alone
    assert line.pen.style() == Qt.PenStyle.DashDotLine
    assert f"cursor t = {times[0]:.5f}" in raw._title.text()
    assert "time cursor at t =" in raw._plot.accessibleDescription()
    assert "Click: put the time cursor there" in raw._plot.statusTip()

    # A plain drag in Pan mode pans and leaves the cursor where it was.
    assert raw.tool() == "pan"
    start = _at(raw, x0 + 0.6 * (x1 - x0), 0.5 * (y0 + y1))
    end = start + QtCore.QPointF(-120.0, 0.0)
    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    _send(raw, QtCore.QEvent.Type.MouseMove, start, none)
    _send(raw, QtCore.QEvent.Type.MouseButtonPress, start, left)
    for k in range(1, 9):
        _send(raw, QtCore.QEvent.Type.MouseMove, start + (end - start) * (k / 8), left)
    _send(raw, QtCore.QEvent.Type.MouseButtonRelease, end, none)
    assert box.viewRange()[0][0] > x0  # panned to later times
    assert len(times) == 1 and raw.selection_count() == 0

    # A modified click (Shift: select in Pan) leaves the cursor alone.
    _click_at(raw, _at(raw, x0, 0.5 * (y0 + y1)), Qt.KeyboardModifier.ShiftModifier)
    assert len(times) == 1


def test_a_click_sets_the_cursor_with_the_selection_tools_too(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    raw.set_tool("rect")
    times: list[float] = []
    raw.time_cursor_changed.connect(times.append)
    (x0, x1), (y0, y1) = raw._plot.getPlotItem().vb.viewRange()
    _click_at(raw, _at(raw, 0.5 * (x0 + x1), 0.5 * (y0 + y1)))
    assert len(times) == 1 and raw.selection_count() == 0


def test_left_right_move_the_time_cursor(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    times: list[float] = []
    raw.time_cursor_changed.connect(times.append)
    box = raw._plot.getPlotItem().vb
    origin = float(np.min(raw._session.time))
    (x0, x1), _ = box.viewRange()
    span = x1 - x0
    raw._plot.setFocus()
    QTest.keyClick(raw._plot, Qt.Key.Key_Right)  # the first press: mid-view
    assert times[-1] == pytest.approx(origin + 0.5 * (x0 + x1))
    QTest.keyClick(raw._plot, Qt.Key.Key_Right)
    assert times[-1] - times[-2] == pytest.approx(0.01 * span)
    QTest.keyClick(raw._plot, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
    assert times[-2] - times[-1] == pytest.approx(0.1 * span)
    # Past the edge of the view, the view follows (its zoom kept).
    for _ in range(8):
        QTest.keyClick(raw._plot, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    (n0, n1), _ = box.viewRange()
    assert n0 <= times[-1] - origin <= n1
    assert n1 - n0 == pytest.approx(span)


def test_the_time_cursor_belongs_to_its_session(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    times: list[float] = []
    raw.time_cursor_changed.connect(times.append)
    raw.set_time_cursor(2450001.2)  # no echo
    assert times == [] and raw._cursor_line.isVisible()
    raw.set_session_view(_fake_view(deleted=(5, 6, 17)))  # an edit: kept
    assert raw.time_cursor() == pytest.approx(2450001.2)
    raw.set_session_view(_fake_view(key="other"))  # another session: gone
    assert raw.time_cursor() is None and not raw._cursor_line.isVisible()
    assert len(times) == 1 and np.isnan(times[0])
    raw.set_time_cursor(2450000.5)
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 10, 50))
    raw.set_light_curve(LightCurve.from_arrays(t, rng.normal(10, 0.1, 50), None))
    assert raw.time_cursor() is None and not raw._cursor_line.isVisible()
    raw.set_time_cursor(3.0)  # no session, no cursor
    assert raw.time_cursor() is None
    (x0, x1), (y0, y1) = raw._plot.getPlotItem().vb.viewRange()
    count = len(times)
    _click_at(raw, _at(raw, 0.5 * (x0 + x1), 0.5 * (y0 + y1)))
    assert len(times) == count  # a plain light curve has no cursor


def test_tools_buttons_and_fallback_keys(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    tools: list[str] = []
    raw.tool_changed.connect(tools.append)
    raw._tool_buttons["lasso"].click()
    assert raw.tool() == "lasso" and tools == ["lasso"]
    raw._plot.setFocus()
    QTest.keyClick(raw._plot, Qt.Key.Key_X)
    assert raw.tool() == "rect" and raw._tool_buttons["rect"].isChecked()
    QTest.keyClick(raw._plot, Qt.Key.Key_V)
    assert raw.tool() == "pan"
    raw.set_selection(list(OUTLIERS))
    with qtbot.waitSignal(raw.delete_requested):
        QTest.keyClick(raw._plot, Qt.Key.Key_Delete)
    raw.set_selection([5])
    with qtbot.waitSignal(raw.restore_requested):
        QTest.keyClick(raw._plot, Qt.Key.Key_U)
    raw.set_selection([1, 2])
    QTest.keyClick(raw._plot, Qt.Key.Key_Escape)
    assert raw.selection_count() == 0


def test_registry_shortcuts_keep_their_keys(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    shortcut = QtGui.QShortcut(QtGui.QKeySequence("Del"), raw)
    shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
    shortcut.setEnabled(False)  # e.g. busy: the plot must not delete on its own
    asked: list[Any] = []
    raw.delete_requested.connect(asked.append)
    raw.set_selection(list(OUTLIERS))
    QTest.keyClick(raw._plot, Qt.Key.Key_Delete)
    assert asked == [] and raw.selection_count() == 3


def test_every_new_control_is_named_and_explained(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    controls = [
        raw._mode_combo, raw._beyond_button, raw._sigma_spin, raw._delete_button,
        raw._restore_button, raw._clear_button, *raw._tool_buttons.values(),
    ]
    for control in controls:
        assert control.accessibleName(), control
        assert control.toolTip(), control
        assert control.focusPolicy() & Qt.FocusPolicy.TabFocus, control


def test_selection_survives_edits_and_resets_on_a_new_session(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    raw.set_selection([1, 2, 3])
    box = raw._plot.getPlotItem().vb
    box.setXRange(0.1, 0.2, padding=0.0)
    edited = _fake_view(deleted=(5, 6, 17))
    raw.set_session_view(edited)
    assert raw.selected_indices().tolist() == [1, 2, 3]
    assert box.viewRange()[0] == pytest.approx([0.1, 0.2])  # the zoom is kept
    raw.set_session_view(_fake_view(key="other"))
    assert raw.selection_count() == 0


def test_plain_light_curve_mode_has_no_selection(qtbot: QtBot) -> None:
    raw = _raw(qtbot)
    raw.set_selection([1])
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 10, 50))
    raw.set_light_curve(LightCurve.from_arrays(t, rng.normal(10, 0.1, 50), None))
    assert not raw._session_bar.isVisibleTo(raw)
    assert not raw._box.selection_enabled and raw.selection_count() == 0
    raw._box.region_selected.emit(_rect(-1, 0, 20, 20), False)
    assert raw.selection_count() == 0
    assert len(raw._scatters[""].getData()[0]) == 50


def test_large_session_curves_are_thinned_but_selected_in_full(qtbot: QtBot) -> None:
    n = 3 * MAX_DRAWN_POINTS
    raw = _raw(qtbot, _fake_view(n=n))
    drawn = sum(
        len(i.getData()[0])
        for i in _items(raw)
        if getattr(i, "opts", {}).get("symbol") in ("o", "s") and i is not raw._rings
    )
    assert drawn <= MAX_DRAWN_POINTS
    assert f"of {n:,} shown" in raw._title.text()
    raw._box.region_selected.emit(_rect(-1.0, -100.0, 5.0, 100.0), False)
    assert raw.selection_count() == n  # the hit test sees every point


def test_band_symbols_back_up_band_colours(qtbot: QtBot) -> None:
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0.0, 20.0, 60))
    band = LightCurve.from_arrays(t, rng.normal(14.0, 0.1, 60), np.full(60, 0.1))
    mblc = MultiBandLightCurve.from_light_curves({"g": band, "r": band, "i": band})
    phased = PhasedView("dark")
    qtbot.addWidget(phased)
    raw = RawLightCurveView("dark")
    qtbot.addWidget(raw)
    raw.show()
    phased.set_light_curve(mblc)
    raw.set_light_curve(mblc)
    pal = palette("dark")
    for widget in (phased, raw):
        symbols = [s.opts["symbol"] for s in widget._scatters.values()]
        assert symbols == [pal.band_symbol(i) for i in range(3)]
    assert phased._plot.accessibleName() == "Phased light curve plot"
    assert "3 bands" in phased._plot.accessibleDescription()
    single = PhasedView("dark")
    qtbot.addWidget(single)
    single.set_light_curve(band)
    assert single._scatters[""].opts["symbol"] == "o"


def test_a_real_session_view_drives_the_view(qtbot: QtBot) -> None:
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 30.0, 300))
    y = 0.2 * np.sin(2 * np.pi * 1.7 * t) + rng.normal(0.0, 0.01, 300)
    y[[10, 200]] += 0.5
    session = cup.PreWhitenSession((t, y, np.full(300, 0.01)), backend="numpy")
    session.step()
    session.delete_points([10])
    view = build_session_view(session, key=SessionKey("real"), id_epoch=0)
    raw = _raw(qtbot, view)
    assert raw._legend and "deleted (1)" in _legend(raw)
    raw.set_y_mode("residuals")
    assert raw.select_beyond(5.0) >= 1 and 200 in raw.selected_indices().tolist()
    assert 10 not in raw.selected_indices().tolist()  # deleted points are not judged
