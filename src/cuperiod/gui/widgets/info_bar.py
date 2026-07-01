"""The run-info bar: badges showing the method, backend/device, and curve stats.

Reads what it shows off the computed :class:`~cuperiod.core.result.Periodogram` —
method, backend, n_samples, baseline, best period — plus a GPU probe, so the user sees
*what ran where*. An indeterminate progress strip indicates a compute in flight.
"""

from __future__ import annotations

import math

from cuperiod.core.device import gpu_info
from cuperiod.core.result import Periodogram
from cuperiod.gui.qt import Qt, QtWidgets


def _badge(object_name: str = "badge") -> QtWidgets.QLabel:
    label = QtWidgets.QLabel("—")
    label.setObjectName(object_name)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


class RunInfoBar(QtWidgets.QWidget):
    """A horizontal strip of status badges for the current run."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self._method = _badge("badgeAccent")
        self._backend = _badge()
        self._device = _badge()
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
        layout.addStretch(1)
        layout.addWidget(self._busy)

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
        self._backend.setText(f"backend: {pg.backend}")
        self._device.setText(self._device_text(pg.backend))
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
                free, total = info.free_gib, info.total_gib
                return f"device: {info.name} ({free:.0f}/{total:.0f} GiB)"
            return "device: GPU"
        return "device: CPU"


__all__ = ["RunInfoBar"]
