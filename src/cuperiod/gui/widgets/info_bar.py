"""The run-info bar: badges showing the method, backend/device, and curve stats.

Reads what it shows off the computed :class:`~cuperiod.core.result.Periodogram` —
method, backend, n_samples, baseline, best period — plus a GPU probe, so the user sees
*what ran where*. An indeterminate progress strip indicates a compute in flight.
"""

from __future__ import annotations

import math

from cuperiod.core.device import gpu_info
from cuperiod.core.result import Periodogram
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets

#: Natural-size cap so one long badge (e.g. a GPU model name) cannot starve the rest.
_BADGE_MAX_WIDTH = 260
#: Horizontal chrome from the badge QSS box: 10 px padding + 1 px border per side.
_BADGE_HPAD = 22


class _BadgeLabel(QtWidgets.QLabel):
    """Badge label that elides its text to the width the layout actually grants.

    Centre-aligned ``QLabel`` text that overflows its widget clips glyphs off *both*
    ends rather than adding an ellipsis, and a fixed elide budget cannot help when the
    layout crushes the row below it — so re-elide on every resize instead.
    """

    def __init__(self, object_name: str = "badge", *, elastic: bool = False) -> None:
        super().__init__()
        self._full_text = ""
        self._elastic = elastic
        self.setObjectName(object_name)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("—")

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        self._full_text = text
        self.updateGeometry()
        self._apply_elide()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt override
        hint = super().sizeHint()
        width = self.fontMetrics().horizontalAdvance(self._full_text) + _BADGE_HPAD
        if self._elastic:
            width = min(width, _BADGE_MAX_WIDTH)
        return QtCore.QSize(width, hint.height())

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt override
        # Elastic badges may shrink (they re-elide); the rest keep their full width.
        if not self._elastic:
            return self.sizeHint()
        hint = super().minimumSizeHint()
        width = self.fontMetrics().horizontalAdvance("…") + _BADGE_HPAD
        return QtCore.QSize(width, hint.height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self) -> None:
        available = max(16, self.width() - _BADGE_HPAD)
        elided = self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideRight, available
        )
        if elided != super().text():
            super().setText(elided)

    def text(self) -> str:
        """Return the full (unelided) badge text."""
        return self._full_text


def _badge(object_name: str = "badge", *, elastic: bool = False) -> _BadgeLabel:
    return _BadgeLabel(object_name, elastic=elastic)


class _FlowLayout(QtWidgets.QLayout):
    """A left-to-right layout that wraps items onto new rows when width runs out.

    The classic Qt flow-layout example, trimmed to what the badge strip needs: the
    full set of badges rarely fits one row once the docks take their share of the
    window, and a plain ``QHBoxLayout`` responds by crushing the badges instead of
    wrapping them.
    """

    def __init__(self, hspacing: int = 8, vspacing: int = 6) -> None:
        super().__init__()
        self._hspace = hspacing
        self._vspace = vspacing
        self._items: list[QtWidgets.QLayoutItem] = []

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QtCore.QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:  # noqa: N802
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QtCore.QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def _do_layout(self, rect: QtCore.QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x, y, line_height = effective.x(), effective.y(), 0
        for item in self._items:
            if item.isEmpty():  # hidden widgets take no space
                continue
            hint = item.sizeHint()
            if x + hint.width() > effective.right() + 1 and line_height > 0:
                x = effective.x()
                y += line_height + self._vspace
                line_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self._hspace
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


def _set_badge_text(
    label: QtWidgets.QLabel, text: str, *, max_width: int = _BADGE_MAX_WIDTH
) -> None:
    """Set ``text`` on ``label`` with a full-text tooltip, eliding to ``max_width``."""
    label.setToolTip(text)
    metrics = label.fontMetrics()
    label.setText(metrics.elidedText(text, Qt.TextElideMode.ElideRight, max_width))


class RunInfoBar(QtWidgets.QWidget):
    """A horizontal strip of status badges for the current run."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = _FlowLayout()
        layout.setContentsMargins(8, 4, 8, 4)
        self.setLayout(layout)

        self._method = _badge("badgeAccent")
        self._backend = _badge(elastic=True)
        self._device = _badge(elastic=True)
        self._samples = _badge()
        self._baseline = _badge()
        self._best = _badge()
        self._timing = _badge()

        self._busy = QtWidgets.QProgressBar()
        self._busy.setRange(0, 0)  # indeterminate
        self._busy.setFixedWidth(120)
        self._busy.setTextVisible(False)
        self._busy.setVisible(False)

        for widget in (
            self._method,
            self._backend,
            self._device,
            self._samples,
            self._baseline,
            self._best,
            self._timing,
        ):
            layout.addWidget(widget)
        layout.addWidget(self._busy)

        self._samples.setToolTip("N = number of light-curve samples used in this run")
        self._baseline.setToolTip("Time baseline (span) of the light curve, in days")
        self._best.setToolTip("Best P = the most significant period found in this run")
        self._timing.setToolTip(
            "Wall-clock compute time for this run (or 'cached' if reused)"
        )

        self.set_idle()

    def set_idle(self) -> None:
        """Reset to the no-result placeholder state."""
        self._method.setText("no run yet")
        for badge in (
            self._backend,
            self._device,
            self._samples,
            self._baseline,
            self._best,
            self._timing,
        ):
            badge.setText("—")
        self._busy.setVisible(False)

    def set_timing(self, text: str) -> None:
        """Set the execution-time badge (e.g. ``"142 ms"`` or ``"cached"``)."""
        self._timing.setText(text)

    def set_busy(self, busy: bool, method: str | None = None) -> None:
        """Show/hide the busy strip; optionally name the running method."""
        self._busy.setVisible(busy)
        if busy and method:
            self._method.setText(f"{method} · computing…")

    def set_result(self, pg: Periodogram) -> None:
        """Populate the badges from a finished periodogram."""
        self._busy.setVisible(False)
        self._method.setText(pg.method)
        _set_badge_text(self._backend, f"backend: {pg.backend}")
        _set_badge_text(self._device, self._device_text(pg.backend))
        self._samples.setText(f"N = {pg.n_samples}")
        self._baseline.setText(f"baseline: {pg.baseline:.1f} d")
        best = pg.best_period()
        self._best.setText(
            f"best P = {best:.6g} d" if math.isfinite(best) else "best P = —"
        )

    @staticmethod
    def _device_text(backend: str) -> str:
        """Describe the device for ``backend`` (GPU name on a CUDA path)."""
        if backend in {"cufinufft", "cupy", "gpu"} or backend.startswith("torch:"):
            info = gpu_info()
            if info is not None:
                # Drop the marketing-name vendor prefix; the badge is short on space
                # and the full name is preserved in the tooltip.
                name = info.name
                for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD "):
                    if name.startswith(prefix):
                        name = name[len(prefix) :]
                        break
                free, total = info.free_gib, info.total_gib
                return f"device: {name} ({free:.0f}/{total:.0f} GiB)"
            return "device: GPU"
        return "device: CPU"


__all__ = ["RunInfoBar"]
