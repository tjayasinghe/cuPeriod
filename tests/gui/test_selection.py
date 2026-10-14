"""Selection geometry (vs a reference) and the SelectViewBox gestures (offscreen)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from cuperiod.gui.qt import QtCore, pg
from cuperiod.gui.selection import (
    MAX_LASSO_VERTICES,
    SelectViewBox,
    points_in_polygon,
    points_in_rect,
    select_points,
    simplify_path,
)


def _reference(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """The textbook even-odd crossing test, one point at a time."""
    out = np.zeros(x.size, dtype=bool)
    n = len(poly)
    for i in range(x.size):
        px, py = x[i], y[i]
        inside = False
        for j in range(n):
            ax, ay = poly[j]
            bx, by = poly[(j + 1) % n]
            if (ay > py) != (by > py):
                cross = ax + (py - ay) * (bx - ax) / (by - ay)
                if px < cross:
                    inside = not inside
        out[i] = inside
    return out


def _star(n: int = 7, r0: float = 1.0, r1: float = 0.4) -> np.ndarray:
    angles = np.linspace(0.0, 2 * np.pi, 2 * n, endpoint=False)
    radii = np.where(np.arange(2 * n) % 2 == 0, r0, r1)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def test_points_in_rect_any_corner_order_and_nan() -> None:
    x = np.array([0.0, 1.0, 2.0, np.nan, 1.5])
    y = np.array([0.0, 1.0, 2.0, 1.0, np.nan])
    a = points_in_rect(x, y, 0.5, 0.5, 2.0, 2.0)
    b = points_in_rect(x, y, 2.0, 2.0, 0.5, 0.5)
    assert a.tolist() == [False, True, True, False, False]
    assert np.array_equal(a, b)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_points_in_polygon_matches_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, 3000)
    y = rng.uniform(-1.5, 1.5, 3000)
    for poly in (_star(), _star(5, 1.2, 0.2)[::-1], rng.uniform(-1, 1, (9, 2))):
        got = points_in_polygon(x, y, poly)
        assert np.array_equal(got, _reference(x, y, poly))


def test_points_in_polygon_degenerate_and_nan() -> None:
    x = np.array([0.0, np.nan, 0.1])
    y = np.array([0.0, 0.0, np.nan])
    assert not points_in_polygon(x, y, [[0, 0], [1, 1]]).any()
    square = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    assert points_in_polygon(x, y, square).tolist() == [True, False, False]
    with pytest.raises(ValueError):
        points_in_polygon([1.0, 2.0], [1.0], square)


def test_select_points_rectangle_is_closed() -> None:
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 1.0, 0.5])
    box = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert select_points(x, y, box).all()
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert select_points(x, y, tri).tolist() == [True, False, False]


def test_simplify_path_limits_vertices_and_keeps_shape() -> None:
    theta = np.linspace(0.0, 2 * np.pi, 2000)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    simple = simplify_path(circle)
    assert 3 <= simple.shape[0] <= MAX_LASSO_VERTICES
    # Endpoints are kept, order preserved, and the area hardly changes.
    assert np.array_equal(simple[0], circle[0])
    assert np.array_equal(simple[-1], circle[-1])

    def area(p: np.ndarray) -> float:
        return 0.5 * abs(
            float(np.dot(p[:, 0], np.roll(p[:, 1], -1)))
            - float(np.dot(p[:, 1], np.roll(p[:, 0], -1)))
        )

    assert abs(area(simple) - np.pi) / np.pi < 0.01
    short = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [np.nan, 1.0], [1.0, 1.0]])
    assert simplify_path(short).tolist() == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    assert simplify_path(circle, 10).shape[0] == 10


def test_million_points_within_budget() -> None:
    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 100.0, 1_000_000)
    y = rng.normal(0.0, 1.0, 1_000_000)
    theta = np.linspace(0.0, 2 * np.pi, 5000)
    lasso = np.column_stack([50 + 40 * np.cos(theta), 1.5 * np.sin(theta)])
    start = time.perf_counter()
    poly = simplify_path(lasso)
    mask = points_in_polygon(x, y, poly)
    elapsed = time.perf_counter() - start
    assert poly.shape[0] <= MAX_LASSO_VERTICES
    assert 0 < mask.sum() < x.size
    # The naive O(N x E) test took 2.7 s; generous for slow CI machines.
    assert elapsed < 1.5, elapsed
    sample = rng.choice(x.size, 2000, replace=False)
    assert np.array_equal(mask[sample], _reference(x[sample], y[sample], poly))


# -- SelectViewBox ---------------------------------------------------------------
class _Drag:
    """Duck-typed pyqtgraph MouseDragEvent in scene coordinates."""

    def __init__(
        self,
        start: QtCore.QPointF,
        pos: QtCore.QPointF,
        last_pos: QtCore.QPointF | None = None,
        *,
        first: bool = False,
        last: bool = False,
        mods: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
        button: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.LeftButton,
    ) -> None:
        # pyqtgraph's own pan arithmetic needs its Point type (numpy-multipliable).
        self._start, self._pos = pg.Point(start), pg.Point(pos)
        self._last_pos = pg.Point(pos if last_pos is None else last_pos)
        self._first, self._last = first, last
        self._mods, self._button = mods, button
        self.accepted = False

    def button(self) -> QtCore.Qt.MouseButton:
        return self._button

    def buttons(self) -> QtCore.Qt.MouseButton:
        return self._button

    def modifiers(self) -> QtCore.Qt.KeyboardModifier:
        return self._mods

    def isStart(self) -> bool:  # noqa: N802
        return self._first

    def isFinish(self) -> bool:  # noqa: N802
        return self._last

    def buttonDownPos(self, btn: Any = None) -> Any:  # noqa: N802
        return self._start

    def pos(self) -> Any:
        return self._pos

    def lastPos(self) -> Any:  # noqa: N802
        return self._last_pos

    def scenePos(self) -> Any:  # noqa: N802
        return self._pos

    def lastScenePos(self) -> Any:  # noqa: N802
        return self._last_pos

    def buttonDownScenePos(self, btn: Any = None) -> Any:  # noqa: N802
        return self._start

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.accepted = False


@pytest.fixture
def viewbox(qtbot: QtBot) -> Iterator[SelectViewBox]:
    box = SelectViewBox()
    widget = pg.PlotWidget(viewBox=box)
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    box.setRange(xRange=(0.0, 10.0), yRange=(0.0, 10.0), padding=0.0)
    widget.show()
    qtbot.waitExposed(widget)
    yield box  # the plot widget (which owns the box) stays alive until teardown
    del widget


def _drag(
    box: SelectViewBox,
    points: list[tuple[float, float]],
    mods: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> None:
    local = [box.mapFromView(QtCore.QPointF(px, py)) for px, py in points]
    for i, p in enumerate(local):
        box.mouseDragEvent(
            _Drag(
                local[0],
                p,
                local[max(i - 1, 0)],
                first=i == 0,
                last=i == len(local) - 1,
                mods=mods,
            )
        )


def test_rect_mode_emits_rectangle_in_data_coords(
    viewbox: SelectViewBox, qtbot: QtBot
) -> None:
    viewbox.set_mode("rect")
    with qtbot.waitSignal(viewbox.region_selected) as blocker:
        _drag(viewbox, [(2.0, 2.0), (4.0, 5.0), (6.0, 7.0)])
    polygon, additive = blocker.args
    assert additive is False
    assert polygon.shape == (4, 2)
    np.testing.assert_allclose(polygon.min(axis=0), [2.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(polygon.max(axis=0), [6.0, 7.0], atol=1e-6)
    assert not viewbox.selecting


def test_lasso_mode_simplifies_and_reports_ctrl(
    viewbox: SelectViewBox, qtbot: QtBot
) -> None:
    viewbox.set_mode("lasso")
    theta = np.linspace(0.0, 2 * np.pi, 300)
    pts = [(5 + 3 * np.cos(a), 5 + 3 * np.sin(a)) for a in theta]
    with qtbot.waitSignal(viewbox.region_selected) as blocker:
        _drag(viewbox, pts, mods=QtCore.Qt.KeyboardModifier.ControlModifier)
    polygon, additive = blocker.args
    assert additive is True
    assert 3 <= polygon.shape[0] <= MAX_LASSO_VERTICES
    inside = select_points(np.array([5.0, 9.5]), np.array([5.0, 5.0]), polygon)
    assert inside.tolist() == [True, False]


def test_pan_mode_pans_and_shift_drag_selects(
    viewbox: SelectViewBox, qtbot: QtBot
) -> None:
    assert viewbox.mode == "pan"
    before = viewbox.viewRange()
    with qtbot.assertNotEmitted(viewbox.region_selected):
        _drag(viewbox, [(5.0, 5.0), (6.0, 5.0), (7.0, 5.0)])
    assert viewbox.viewRange() != before  # a plain drag still pans
    with qtbot.waitSignal(viewbox.region_selected):
        _drag(
            viewbox,
            [(2.0, 2.0), (3.0, 3.0)],
            mods=QtCore.Qt.KeyboardModifier.ShiftModifier,
        )


def test_mode_switch_and_cancel(viewbox: SelectViewBox, qtbot: QtBot) -> None:
    with qtbot.waitSignal(viewbox.mode_changed) as blocker:
        viewbox.set_mode("lasso")
    assert blocker.args == ["lasso"]
    local = viewbox.mapFromView(QtCore.QPointF(1.0, 1.0))
    viewbox.mouseDragEvent(_Drag(local, local, first=True))
    assert viewbox.selecting
    viewbox.cancel_selection()
    assert not viewbox.selecting
    with pytest.raises(ValueError):
        viewbox.set_mode("zoom")  # type: ignore[arg-type]
