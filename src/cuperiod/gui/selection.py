"""Point selection geometry and the rectangle / lasso view box of the light curve.

Deleting outliers interactively needs a hit test that keeps up with a million-point
curve. A naive even-odd test is ``O(N x E)`` and took 2.7 s for ``1e6`` points; here
the points are first cut to the polygon's bounding box, sorted by ``y`` once, and every
edge then touches only the contiguous slice of points whose ``y`` it spans, so the cost
is a sort plus ``O(N x k)`` with ``k`` the number of edges crossing a typical scan
line. Lasso paths are simplified (Ramer-Douglas-Peucker) to at most 64 vertices first.

:class:`SelectViewBox` adds the gestures: in **pan** mode a plain drag pans as before
and Shift-drag selects with a rectangle; **rect** and **lasso** modes select with a left
drag. Selections are emitted in data coordinates as a closed polygon.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from cuperiod.core._typing import FloatArray
from cuperiod.gui.qt import QtCore, QtGui, QtWidgets, Signal, pg

BoolArray = npt.NDArray[np.bool_]

SelectMode = Literal["pan", "rect", "lasso"]

#: Lasso paths are simplified to at most this many vertices before the hit test.
MAX_LASSO_VERTICES = 64


def _as_xy(x: Any, y: Any) -> tuple[FloatArray, FloatArray]:
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    if xa.shape != ya.shape:
        raise ValueError("x and y must have the same length")
    return xa, ya


def points_in_rect(
    x: Any, y: Any, x0: float, y0: float, x1: float, y1: float
) -> BoolArray:
    """Mask of the points inside the closed rectangle spanned by two corners.

    The corners may come in any order; NaN coordinates are never inside.
    """
    xa, ya = _as_xy(x, y)
    lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
    lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
    return (xa >= lo_x) & (xa <= hi_x) & (ya >= lo_y) & (ya <= hi_y)


def points_in_polygon(x: Any, y: Any, polygon: Any) -> BoolArray:
    """Mask of the points strictly inside ``polygon`` (even-odd rule).

    Parameters
    ----------
    x, y : array_like
        Point coordinates (NaN is never inside).
    polygon : array_like, shape (M, 2)
        Vertices in order; the polygon closes itself (the last vertex joins the
        first). Fewer than three distinct vertices select nothing.

    Notes
    -----
    Bounding-box prefilter, then one ``argsort`` by ``y``: each edge updates only the
    slice of points whose ``y`` lies in its half-open span ``[min y, max y)``, the
    usual crossing rule, so points exactly on a vertex are counted once.
    """
    xa, ya = _as_xy(x, y)
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    out = np.zeros(xa.shape, dtype=np.bool_)
    poly = poly[np.all(np.isfinite(poly), axis=1)]
    if poly.shape[0] < 3:
        return out
    px, py = poly[:, 0], poly[:, 1]
    box = (
        (xa >= px.min()) & (xa <= px.max()) & (ya >= py.min()) & (ya <= py.max())
    )
    candidates = np.flatnonzero(box)
    if candidates.size == 0:
        return out
    cy = ya[candidates]
    order = np.argsort(cy, kind="stable")
    ys = cy[order]
    xs = xa[candidates][order]
    inside = np.zeros(ys.shape, dtype=np.bool_)
    qx = np.roll(px, -1)
    qy = np.roll(py, -1)
    for ax, ay, bx, by in zip(px, py, qx, qy, strict=True):
        if ay == by:
            continue  # a horizontal edge never crosses a horizontal ray
        lo, hi = (ay, by) if ay < by else (by, ay)
        start = int(np.searchsorted(ys, lo, side="left"))
        stop = int(np.searchsorted(ys, hi, side="left"))
        if stop <= start:
            continue
        seg_y = ys[start:stop]
        crossing = ax + (seg_y - ay) * ((bx - ax) / (by - ay))
        inside[start:stop] ^= xs[start:stop] < crossing
    out[candidates[order]] = inside
    return out


def simplify_path(path: Any, max_vertices: int = MAX_LASSO_VERTICES) -> FloatArray:
    """Reduce a polyline to at most ``max_vertices`` vertices (Ramer-Douglas-Peucker).

    Every vertex gets the RDP significance at which it would be kept (clamped to its
    parent's, so the ranking is a valid RDP result for some tolerance); the endpoints
    and the most significant interior vertices are kept, in their original order.
    Consecutive duplicates and non-finite vertices are dropped first.
    """
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] > 1:
        keep = np.ones(pts.shape[0], dtype=np.bool_)
        keep[1:] = np.any(np.diff(pts, axis=0) != 0.0, axis=1)
        pts = pts[keep]
    n = pts.shape[0]
    limit = max(int(max_vertices), 2)
    if n <= limit:
        return pts.copy()
    significance = np.full(n, -1.0)
    significance[0] = significance[-1] = np.inf
    stack: list[tuple[int, int, float]] = [(0, n - 1, np.inf)]
    while stack:
        first, last, parent = stack.pop()
        if last - first < 2:
            continue
        a, b = pts[first], pts[last]
        inner = pts[first + 1 : last]
        d = b - a
        norm = float(np.hypot(d[0], d[1]))
        if norm == 0.0:  # a closed loop: distance to the shared endpoint
            dist = np.hypot(inner[:, 0] - a[0], inner[:, 1] - a[1])
        else:
            dist = np.abs(d[0] * (inner[:, 1] - a[1]) - d[1] * (inner[:, 0] - a[0]))
            dist = dist / norm
        k = int(np.argmax(dist))
        split = first + 1 + k
        sig = min(float(dist[k]), parent)
        significance[split] = sig
        stack.append((first, split, sig))
        stack.append((split, last, sig))
    ranked = np.argsort(-significance, kind="stable")[:limit]
    return pts[np.sort(ranked)]


class SelectViewBox(pg.ViewBox):  # type: ignore[misc]
    """A ``ViewBox`` with pan / rectangle / lasso selection.

    Signals
    -------
    region_selected(object, bool)
        ``(polygon, additive)``: the selected region in data coordinates as an
        ``(M, 2)`` array (a rectangle is four vertices), and whether Ctrl was held
        (add to the current selection rather than replace it).
    mode_changed(str)
        The tool changed (``"pan"``, ``"rect"``, ``"lasso"``).
    """

    region_selected = Signal(object, bool)
    mode_changed = Signal(str)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mode: SelectMode = "pan"
        self._drag_kind: Literal["rect", "lasso"] | None = None
        self._origin: QtCore.QPointF | None = None
        self._path: list[tuple[float, float]] = []
        pen = pg.mkPen(color=(255, 140, 0), width=1, style=QtCore.Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._band = QtWidgets.QGraphicsRectItem()
        self._band.setPen(pen)
        self._band.setBrush(pg.mkBrush(255, 140, 0, 40))
        self._band.setZValue(1e9)
        self._band.hide()
        self._lasso = QtWidgets.QGraphicsPathItem()
        self._lasso.setPen(pen)
        self._lasso.setBrush(pg.mkBrush(255, 140, 0, 30))
        self._lasso.setZValue(1e9)
        self._lasso.hide()
        self.addItem(self._band, ignoreBounds=True)
        self.addItem(self._lasso, ignoreBounds=True)

    # -- mode ------------------------------------------------------------------------
    @property
    def mode(self) -> SelectMode:
        return self._mode

    def set_mode(self, mode: SelectMode) -> None:
        """Select the drag tool; an unfinished selection is abandoned."""
        if mode not in ("pan", "rect", "lasso"):
            raise ValueError(f"unknown selection mode {mode!r}")
        self.cancel_selection()
        if mode != self._mode:
            self._mode = mode
            self.mode_changed.emit(mode)

    @property
    def selecting(self) -> bool:
        """Whether a selection drag is in progress."""
        return self._drag_kind is not None

    def cancel_selection(self) -> None:
        """Abandon a selection drag in progress (Esc)."""
        self._drag_kind = None
        self._origin = None
        self._path = []
        self._band.hide()
        self._lasso.hide()

    # -- gestures --------------------------------------------------------------------
    def mouseDragEvent(self, ev: Any, axis: int | None = None) -> None:  # noqa: N802
        """Route a left drag to the active selection tool, else pan as usual."""
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis)
            return
        mods = ev.modifiers()
        shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if ev.isStart():
            kind: Literal["rect", "lasso"] | None
            if self._mode == "lasso":
                kind = "lasso"
            elif self._mode == "rect" or shift:
                kind = "rect"
            else:
                kind = None
            if kind is None:
                super().mouseDragEvent(ev, axis)
                return
            self._begin(kind, self.mapToView(ev.buttonDownPos()))
        elif self._drag_kind is None:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        point = self.mapToView(ev.pos())
        if ev.isFinish():
            additive = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
            self._finish(point, additive)
        else:
            self._extend(point)

    def _begin(self, kind: Literal["rect", "lasso"], origin: QtCore.QPointF) -> None:
        self._drag_kind = kind
        self._origin = QtCore.QPointF(origin)
        self._path = [(origin.x(), origin.y())]

    def _extend(self, point: QtCore.QPointF) -> None:
        if self._drag_kind == "rect" and self._origin is not None:
            rect = QtCore.QRectF(self._origin, point).normalized()
            self._band.setRect(rect)
            self._band.show()
        elif self._drag_kind == "lasso":
            self._path.append((point.x(), point.y()))
            qpath = QtGui.QPainterPath()
            qpath.moveTo(*self._path[0])
            for px, py in self._path[1:]:
                qpath.lineTo(px, py)
            qpath.closeSubpath()
            self._lasso.setPath(qpath)
            self._lasso.show()

    def _finish(self, point: QtCore.QPointF, additive: bool) -> None:
        kind, origin = self._drag_kind, self._origin
        self._extend(point)
        path = list(self._path)
        self.cancel_selection()
        if kind == "rect" and origin is not None:
            x0, y0, x1, y1 = origin.x(), origin.y(), point.x(), point.y()
            polygon = np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
            )
        elif kind == "lasso":
            polygon = simplify_path(path)
            if polygon.shape[0] < 3:
                return
        else:
            return
        self.region_selected.emit(polygon, additive)


def select_points(x: Any, y: Any, polygon: Any) -> BoolArray:
    """Mask of the points in ``polygon``: a 4-vertex axis-aligned box is a rectangle
    test (closed: points on the border count), anything else the even-odd test."""
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] == 4:
        xs, ys = poly[:, 0], poly[:, 1]
        if (
            xs[0] == xs[3] and xs[1] == xs[2] and ys[0] == ys[1] and ys[2] == ys[3]
        ):
            return points_in_rect(x, y, xs[0], ys[0], xs[2], ys[2])
    return points_in_polygon(x, y, poly)


__all__ = [
    "MAX_LASSO_VERTICES",
    "SelectMode",
    "SelectViewBox",
    "points_in_polygon",
    "points_in_rect",
    "select_points",
    "simplify_path",
]
