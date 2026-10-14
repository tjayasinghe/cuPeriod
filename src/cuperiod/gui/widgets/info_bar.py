"""The run-info bar: badges showing the method, backend/device, and curve stats.

Reads what it shows off the computed :class:`~cuperiod.core.result.Periodogram` —
method, backend, n_samples, baseline, best period — plus a GPU probe, so the user sees
*what ran where*. An indeterminate progress strip indicates a compute in flight.

Until the first result arrives the data badges are hidden: six dashes describe
nothing, so the bar shows one line saying what to do instead. Once a result is on
screen the badges keep describing *that* result — a later run that fails, or a change
of settings that makes it stale, is flagged on the leading badge rather than wiping
the badges of a spectrum that is still displayed.
"""

from __future__ import annotations

import math

from cuperiod.core.device import gpu_info
from cuperiod.core.result import Periodogram
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets
from cuperiod.gui.theme import BASE_FONT_PX

#: Natural-size cap so one long badge (e.g. a GPU model name) cannot starve the rest.
_BADGE_MAX_WIDTH = 260
#: Horizontal chrome from the badge QSS box: 10 px padding + 1 px border per side.
_BADGE_HPAD = 22

#: Backends whose device is the CUDA card :func:`~cuperiod.core.device.gpu_info`
#: reports on. Other torch devices (MPS, XPU) are named from the backend string itself;
#: probing CUDA for them would put an unrelated NVIDIA card's name on the badge.
_CUDA_BACKENDS = frozenset({"cufinufft", "cupy", "gpu"})

#: What the leading badge says before anything has run.
IDLE_TEXT = "no result yet"


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
        # Badge text comes from results and backend names, never markup; plain text
        # keeps a stray '<' in a method or device name from being parsed as a tag.
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setText("—")

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        self._full_text = text
        self.updateGeometry()
        self._apply_elide()

    def _chrome(self) -> int:
        """The badge's padding and border, which the style sheet scales with the
        zoom like the font (a fixed 22 px clipped the text at 125 %)."""
        px = self.font().pixelSize()
        scale = px / BASE_FONT_PX if px > 0 else 1.0
        return math.ceil(_BADGE_HPAD * max(1.0, scale))

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt override
        hint = super().sizeHint()
        width = self.fontMetrics().horizontalAdvance(self._full_text) + self._chrome()
        if self._elastic:
            width = min(width, _BADGE_MAX_WIDTH)
        return QtCore.QSize(width, hint.height())

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt override
        # Elastic badges may shrink (they re-elide); the rest keep their full width.
        if not self._elastic:
            return self.sizeHint()
        hint = super().minimumSizeHint()
        width = self.fontMetrics().horizontalAdvance("…") + self._chrome()
        return QtCore.QSize(width, hint.height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_elide()

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        # A zoom or theme switch changes the font through the style sheet: the size
        # hint is measured with it, so the layout must ask again (the badges kept
        # their old width and were clipped to "ackend: finuff" at 125 %).
        if event.type() in (
            QtCore.QEvent.Type.FontChange,
            QtCore.QEvent.Type.StyleChange,
        ):
            self.updateGeometry()
            self._apply_elide()

    def _apply_elide(self) -> None:
        available = max(16, self.width() - self._chrome())
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


class FlowLayout(QtWidgets.QLayout):
    """A left-to-right layout that wraps items onto new rows when width runs out.

    The classic Qt flow-layout example, trimmed to what the badge strip needs: the
    full set of badges rarely fits one row once the docks take their share of the
    window, and a plain ``QHBoxLayout`` responds by crushing the badges instead of
    wrapping them. The plot toolbars use it for the same reason: in a narrow window
    their check boxes lost their labels ("pea", "res") and the buttons became stubs.
    """

    def __init__(self, hspacing: int = 8, vspacing: int = 6) -> None:
        super().__init__()
        self._hspace = hspacing
        self._vspace = vspacing
        self._items: list[QtWidgets.QLayoutItem] = []
        #: The width last laid out at, and the wrapped height there: the minimum
        #: height follows the wrap (see minimumSize).
        self._wrap: tuple[int, int] = (0, 0)

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
        width = rect.width()
        if width > 0:
            # Measured at every layout pass, not only on a new width: a zoom changes
            # the items' sizes at the same width.
            height = self.heightForWidth(width)
            changed = height != self._wrap[1]
            self._wrap = (width, height)
            if changed:
                # Ask the parents again for room: a splitter (which ignores
                # height-for-width) otherwise squeezed the wrapped rows over the
                # widgets below them. Posted, so this never recurses.
                self.invalidate()

    def invalidate(self) -> None:
        """Forget the items' cached size hints too.

        ``QLayout.invalidate`` leaves them (the box layouts drop theirs in their
        own override), so after a zoom the badges and buttons kept their old
        widths and their text was clipped ("ackend: finuff").
        """
        for item in self._items:
            item.invalidate()
        super().invalidate()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:  # noqa: N802
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QtCore.QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )
        # At the width last laid out, the rows that wrapped are part of the minimum.
        return size.expandedTo(QtCore.QSize(0, self._wrap[1]))

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


#: The previous private name (kept for anything that imported it).
_FlowLayout = FlowLayout


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
        layout = FlowLayout()
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
        self._busy.setFixedWidth(self.fontMetrics().horizontalAdvance("0") * 12)
        self._busy.setTextVisible(False)
        self._busy.setAccessibleName("Computation in progress")
        self._busy.setToolTip("A computation is running; Stop (Esc) ends it")
        self._busy.setVisible(False)

        # What is running and for how long, in words next to the strip: an
        # anonymous moving bar says nothing to a screen reader, and nothing about
        # whether a long pre-whitening run is still making progress.
        self._activity = QtWidgets.QLabel("")
        self._activity.setObjectName("muted")
        self._activity.setTextFormat(Qt.TextFormat.PlainText)
        self._activity.setAccessibleName("Run status")
        self._activity.setVisible(False)
        self._activity_text = ""
        self._elapsed = QtCore.QElapsedTimer()
        self._ticker = QtCore.QTimer(self)
        self._ticker.setInterval(1000)
        self._ticker.timeout.connect(self._refresh_activity)
        self._finishing = False
        self._working = False

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
        layout.addWidget(self._activity)

        self._samples.setToolTip("N = number of light-curve samples used in this run")
        self._baseline.setToolTip("Time baseline (span) of the light curve, in days")
        self._best.setToolTip("Best P = the most significant period found in this run")
        self._timing.setToolTip(
            "Wall-clock compute time for this run (or 'cached' if reused)"
        )

        #: The method of the result the badges currently describe (None: nothing).
        self._result_method: str | None = None
        self._stale = False
        self.set_idle()

    @property
    def _data_badges(self) -> tuple[_BadgeLabel, ...]:
        return (
            self._backend,
            self._device,
            self._samples,
            self._baseline,
            self._best,
            self._timing,
        )

    def set_idle(self, message: str = IDLE_TEXT) -> None:
        """Reset to the no-result state: one leading badge saying ``message``.

        The data badges are hidden rather than dashed out, and the backend/device
        tooltips of the previous run are dropped with them (they would otherwise
        still name a backend while the badge shows nothing).
        """
        self._result_method = None
        self._stale = False
        self._method.setText(message)
        for badge in self._data_badges:
            badge.setText("—")
            badge.setVisible(False)
        self._backend.setToolTip("")
        self._device.setToolTip("")
        self._busy.setVisible(False)
        self._ticker.stop()
        self._refresh_activity()

    def set_timing(self, text: str) -> None:
        """Set the execution-time badge (e.g. ``"142 ms"`` or ``"cached"``)."""
        self._timing.setText(text)

    def set_busy(self, busy: bool, method: str | None = None) -> None:
        """Show/hide the busy strip; optionally name the running method.

        While busy the strip is followed by the elapsed time in words (and whatever
        :meth:`set_activity` narrates), ticking once a second.
        """
        was_busy = self._busy.isVisibleTo(self)
        self._busy.setVisible(busy)
        if busy and method:
            self._method.setText(f"{method} · computing…")
        elif not busy:
            self._restore_leading_badge()
        if busy and not was_busy:
            self._elapsed.start()
            self._activity_text = ""
            self._ticker.start()
        elif not busy:
            self._ticker.stop()
            self._activity_text = ""
        self._refresh_activity()

    def set_activity(self, text: str) -> None:
        """Narrate the running job ("Auto — F7 accepted, S/N 5.3")."""
        self._activity_text = text
        self._refresh_activity()

    def set_finishing(self, finishing: bool) -> None:
        """A stopped run still occupies the worker: say so until it has ended."""
        self._finishing = finishing
        self._refresh_activity()

    def set_working(self, working: bool) -> None:
        """A quick session job is taking a while (non-blocking: nothing is disabled)."""
        self._working = working
        self._refresh_activity()

    def activity_text(self) -> str:
        """What the status words next to the strip say (empty when hidden)."""
        return self._activity.text() if self._activity.isVisibleTo(self) else ""

    def _refresh_activity(self) -> None:
        if self._busy.isVisibleTo(self):
            seconds = self._elapsed.elapsed() / 1000.0 if self._elapsed.isValid() else 0
            parts = [self._activity_text] if self._activity_text else ["running"]
            parts.append(f"{seconds:.0f} s")
            text = " · ".join(parts)
            self._busy.setAccessibleDescription(text)
        elif self._finishing:
            text = "finishing previous run…"
        elif self._working:
            text = "working…"
        else:
            text = ""
        self._activity.setText(text)
        self._activity.setVisible(bool(text))

    def set_failed(self, method: str | None = None) -> None:
        """A run failed: stop the busy strip, keep describing what is on screen.

        If a result is still displayed the badges go on describing it (the error is
        reported elsewhere); otherwise the leading badge says which run failed.
        """
        self._busy.setVisible(False)
        self._ticker.stop()
        self._refresh_activity()
        if self._result_method is not None:
            self._restore_leading_badge()
        else:
            self._method.setText(f"{method} · failed" if method else "run failed")

    def set_stale(self, stale: bool) -> None:
        """Flag (or clear) that the displayed result no longer matches the controls."""
        self._stale = stale and self._result_method is not None
        if not self._busy.isVisible():
            self._restore_leading_badge()

    def is_idle(self) -> bool:
        """Whether no result is described (the badges show one idle line)."""
        return self._result_method is None

    def is_stale(self) -> bool:
        """Whether the displayed result is flagged as out of date."""
        return self._stale

    def _restore_leading_badge(self) -> None:
        if self._result_method is None:
            return
        suffix = " · settings changed" if self._stale else ""
        self._method.setText(f"{self._result_method}{suffix}")
        self._method.setToolTip(
            "The options, method or band changed since this result was computed; "
            "compute again to update it."
            if self._stale
            else ""
        )

    def set_result(self, pg: Periodogram) -> None:
        """Populate the badges from a finished periodogram."""
        self._busy.setVisible(False)
        self._ticker.stop()
        self._refresh_activity()
        self._result_method = pg.method
        self._stale = False
        self._restore_leading_badge()
        for badge in self._data_badges:
            badge.setVisible(True)
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
        """Describe the device ``backend`` ran on (the card's name on a CUDA path).

        Torch results carry their device in the backend string (``"torch:cpu"``,
        ``"torch:cuda"``, ``"torch:mps"``), so only a CUDA device is probed; the
        portable CPU path must read "CPU" however many GPUs the machine has.
        """
        name = backend.strip().lower()
        if name.startswith("torch:"):
            kind = name.split(":")[1]  # "torch:cuda:0" -> "cuda"
            if kind in {"", "cpu"}:  # a bare "torch:" names no accelerator
                return "device: CPU"
            if kind != "cuda":
                return f"device: GPU ({kind.upper()})"
        elif name not in _CUDA_BACKENDS:
            return "device: CPU"
        info = gpu_info()
        if info is None:
            return "device: GPU"
        # Drop the marketing-name vendor prefix; the badge is short on space and the
        # full name is preserved in the tooltip.
        card = info.name
        for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD "):
            if card.startswith(prefix):
                card = card[len(prefix) :]
                break
        free, total = info.free_gib, info.total_gib
        return f"device: {card} ({free:.0f}/{total:.0f} GiB)"


__all__ = ["FlowLayout", "RunInfoBar"]
