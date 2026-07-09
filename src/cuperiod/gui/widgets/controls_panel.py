"""The left-hand controls: method/backend pickers, the options form, and Compute.

The method picker drives a dynamically rebuilt :class:`PydanticSettingsForm` (so every
option of the chosen method is exposed) and a backend picker limited to usable backends.
Settings are cached per method, so switching back restores your values.
``Compute`` validates and emits :attr:`run_requested`; invalid input is shown inline
instead of starting a run.
"""

from __future__ import annotations

import contextlib

from pydantic import ValidationError
from pydantic_settings import BaseSettings

from cuperiod.core._typing import FloatArray
from cuperiod.gui.meta import (
    auto_max_frequency,
    backend_options,
    method_display_names,
    multiband_method_names,
    natural_domain,
    objective_sense,
    resolved_backend,
    settings_class,
    supports_multiband,
)
from cuperiod.gui.qt import Qt, QtWidgets, Signal
from cuperiod.gui.settingsform import PydanticSettingsForm

_DEFAULT_METHOD = "GLS"

#: Item-data sentinels for the synthetic "all bands" combo entries, distinguishing them
#: from a real band that happens to be named "combined" or "stacked".
_COMBINED_SENTINEL = "__combined__"
_STACKED_SENTINEL = "__stacked__"


def _first_error_message(exc: ValidationError) -> str:
    """A compact ``loc: msg`` summary of the first validation error."""
    errors = exc.errors()
    if not errors:
        return str(exc)
    first = errors[0]
    loc = ".".join(str(part) for part in first.get("loc", ()))
    msg = str(first.get("msg", "invalid value"))
    return f"{loc}: {msg}" if loc else msg


class ControlsPanel(QtWidgets.QWidget):
    """Method/backend pickers, the auto-generated options form, and Compute."""

    # Emitted on Compute: (method, backend, settings, n_peaks).
    run_requested = Signal(str, str, object, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings_cache: dict[str, BaseSettings] = {}
        self._current_method: str | None = None
        self._form: PydanticSettingsForm | None = None
        self._curve_time: FloatArray | None = None
        self._has_curve = False
        self._busy = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QtWidgets.QLabel("Controls")
        header.setObjectName("heading")
        root.addWidget(header)

        self._top = QtWidgets.QFormLayout()
        self._top.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItems(method_display_names())
        self._method_combo.setToolTip("Periodogram method to run")
        self._band_combo = QtWidgets.QComboBox()
        self._band_combo.setToolTip("Which band(s) to analyse for multiband data")
        self._backend_combo = QtWidgets.QComboBox()
        self._backend_combo.setToolTip(
            "Where to run — auto (best available), cpu/gpu (best CPU/GPU path, not "
            "necessarily PyTorch), torch (portable), or a concrete backend."
        )
        self._peaks_spin = QtWidgets.QSpinBox()
        self._peaks_spin.setRange(1, 100)
        self._peaks_spin.setValue(10)
        self._peaks_spin.setToolTip("Number of significant peaks to find and list")
        self._top.addRow("Method", self._method_combo)
        self._top.addRow("Band", self._band_combo)
        self._top.addRow("Backend", self._backend_combo)
        self._top.addRow("Peaks", self._peaks_spin)
        self._top.setRowVisible(self._band_combo, False)
        self._band_names: list[str] | None = None
        root.addLayout(self._top)

        self._note = QtWidgets.QLabel("")
        self._note.setObjectName("muted")
        self._note.setWordWrap(True)
        root.addWidget(self._note)

        self._backend_hint = QtWidgets.QLabel("")
        self._backend_hint.setObjectName("muted")
        self._backend_hint.setWordWrap(True)
        root.addWidget(self._backend_hint)

        self._grid_hint = QtWidgets.QLabel("")
        self._grid_hint.setObjectName("muted")
        self._grid_hint.setWordWrap(True)
        self._grid_hint.setToolTip("The trial grid the auto settings will search")
        root.addWidget(self._grid_hint)

        options_group = QtWidgets.QGroupBox("Options")
        og_layout = QtWidgets.QVBoxLayout(options_group)
        og_layout.setContentsMargins(6, 6, 6, 6)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        og_layout.addWidget(self._scroll)
        root.addWidget(options_group, 1)

        self._error = QtWidgets.QLabel("")
        self._error.setObjectName("error")
        self._error.setWordWrap(True)
        self._error.setVisible(False)
        root.addWidget(self._error)

        self._compute_btn = QtWidgets.QPushButton("Compute periodogram")
        self._compute_btn.setProperty("primary", True)
        self._compute_btn.setToolTip("Compute the periodogram  (Ctrl+Enter)")
        self._compute_btn.clicked.connect(self._on_compute)
        root.addWidget(self._compute_btn)

        if _DEFAULT_METHOD in method_display_names():
            self._method_combo.setCurrentText(_DEFAULT_METHOD)
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        self._backend_combo.currentTextChanged.connect(self._update_backend_hint)
        self._on_method_changed(self._method_combo.currentText())
        self.set_enabled(False)

    # -- public ------------------------------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the Compute action (a light curve must be loaded first)."""
        self._has_curve = enabled
        self._update_compute_enabled()

    def set_busy(self, busy: bool) -> None:
        """Disable just the Compute button while a run is in flight.

        The rest of the form (method/backend/options) stays interactive so the user
        can prepare the next run while the current one finishes.
        """
        self._busy = busy
        self._update_compute_enabled()

    def can_compute(self) -> bool:
        """Whether Compute would currently do anything (curve loaded, not busy)."""
        return self._compute_btn.isEnabled()

    def request_compute(self) -> None:
        """Trigger a compute programmatically (batch auto-run on source change)."""
        self._on_compute()

    def _update_compute_enabled(self) -> None:
        self._compute_btn.setEnabled(self._has_curve and not self._busy)

    def set_multiband(self, is_multiband: bool) -> None:
        """Restrict the method list to multiband-capable methods for multiband data."""
        names = multiband_method_names() if is_multiband else method_display_names()
        current = self._method_combo.currentText()
        self._method_combo.blockSignals(True)
        self._method_combo.clear()
        self._method_combo.addItems(names)
        if current in names:
            self._method_combo.setCurrentText(current)
        elif _DEFAULT_METHOD in names:
            self._method_combo.setCurrentText(_DEFAULT_METHOD)
        self._method_combo.blockSignals(False)
        self._on_method_changed(self._method_combo.currentText())

    def set_bands(self, band_names: list[str] | None) -> None:
        """Show a band selector for multiband curves; hide it for single-band ones."""
        self._band_names = list(band_names) if band_names else None
        self._rebuild_band_combo(self._method_combo.currentText())

    def set_curve_time(self, time: FloatArray) -> None:
        """Provide the active curve's times so the auto grid range can be shown."""
        self._curve_time = time if time.size >= 2 else None
        self._update_grid_hint()

    def current_band(self) -> str:
        """The selected band: a band name, ``"stacked"``, ``"combined"``, or ``""``."""
        if self._band_names is None:
            return ""
        data = self._band_combo.currentData()
        if data == _COMBINED_SENTINEL:
            return "combined"
        if data == _STACKED_SENTINEL:
            return "stacked"
        return self._band_combo.currentText()

    # -- method switching --------------------------------------------------------
    def _on_method_changed(self, method: str) -> None:
        if self._current_method is not None and self._form is not None:
            with contextlib.suppress(ValidationError):
                self._settings_cache[self._current_method] = self._form.build()
        self._current_method = method
        self._rebuild_backend_combo(method)
        self._rebuild_band_combo(method)
        self._rebuild_form(method)
        self._update_note(method)
        self._update_backend_hint()
        self._update_grid_hint()
        self._clear_error()

    def _rebuild_band_combo(self, method: str) -> None:
        if self._band_names is None:
            self._top.setRowVisible(self._band_combo, False)
            return
        self._top.setRowVisible(self._band_combo, True)
        self._band_combo.blockSignals(True)
        self._band_combo.clear()
        if supports_multiband(method):
            # combined multiband analysis is the default for capable methods
            self._band_combo.addItem("combined (all bands)", _COMBINED_SENTINEL)
            for name in self._band_names:
                self._band_combo.addItem(name, name)
        else:
            # single-band methods analyse one band (safest) or a stacked merge
            for name in self._band_names:
                self._band_combo.addItem(name, name)
            self._band_combo.addItem("stacked (all bands)", _STACKED_SENTINEL)
        self._band_combo.setCurrentIndex(0)
        self._band_combo.blockSignals(False)

    def _rebuild_backend_combo(self, method: str) -> None:
        self._backend_combo.blockSignals(True)
        self._backend_combo.clear()
        self._backend_combo.addItems(backend_options(method))
        self._backend_combo.setCurrentText("auto")
        self._backend_combo.blockSignals(False)

    def _rebuild_form(self, method: str) -> None:
        form = PydanticSettingsForm(
            settings_class(method), initial=self._settings_cache.get(method)
        )
        form.changed.connect(self._clear_error)
        form.changed.connect(self._update_grid_hint)
        self._scroll.setWidget(form)  # takes ownership and deletes the previous form
        self._form = form

    def _update_note(self, method: str) -> None:
        sense = objective_sense(method)
        kind = "peaks are maxima" if sense == "max" else "peaks are minima"
        bits = [
            f"objective: {sense} ({kind})",
            "multiband-capable" if supports_multiband(method) else "single-band",
        ]
        domain = natural_domain(method)
        if domain is not None:
            bits.append(f"domain: {domain.value}")
        self._note.setText("  ·  ".join(bits))

    def _update_backend_hint(self) -> None:
        chosen = self._backend_combo.currentText()
        info = resolved_backend(self._method_combo.currentText(), chosen)
        if info is None:
            self._backend_hint.setText("")
            return
        resolved, is_gpu = info
        where = "GPU" if is_gpu else "CPU"
        prefix = "" if resolved == chosen else f"{chosen} → "
        self._backend_hint.setText(f"runs on:  {prefix}{resolved}  ({where})")

    def _update_grid_hint(self) -> None:
        method = self._method_combo.currentText()
        if (
            self._curve_time is None
            or self._form is None
            or "maximum_frequency" not in settings_class(method).model_fields
        ):
            self._grid_hint.setText("")
            return
        time = self._curve_time
        baseline = float(time.max() - time.min())
        nyquist_factor = int(self._form.value_of("nyquist_factor") or 5)
        auto_min = 1.0 / baseline if baseline > 0.0 else 0.0
        auto_max = auto_max_frequency(time, nyquist_factor)
        # fill the greyed 'auto' spin boxes with the values that will actually be used
        self._form.set_auto_value("minimum_frequency", auto_min)
        self._form.set_auto_value("maximum_frequency", auto_max)
        # the hint uses the user's values where set, else the auto ones
        user_min = self._form.value_of("minimum_frequency")
        user_max = self._form.value_of("maximum_frequency")
        fmin = user_min if user_min is not None else auto_min
        fmax = user_max if user_max is not None else auto_max
        p_long = 1.0 / fmin if fmin > 0.0 else float("inf")
        p_short = 1.0 / fmax if fmax > 0.0 else 0.0
        self._grid_hint.setText(
            f"grid → f: {fmin:.3g}–{fmax:.3g} /d   ·   P: {p_short:.3g}–{p_long:.3g} d"
        )

    # -- compute -----------------------------------------------------------------
    def _on_compute(self) -> None:
        if self._form is None:
            return
        try:
            settings = self._form.build()
        except ValidationError as exc:
            self._show_error(_first_error_message(exc))
            return
        self._clear_error()
        self.run_requested.emit(
            self._method_combo.currentText(),
            self._backend_combo.currentText(),
            settings,
            self._peaks_spin.value(),
        )

    def _show_error(self, message: str) -> None:
        self._error.setText(f"Invalid setting — {message}")
        self._error.setVisible(True)

    def _clear_error(self) -> None:
        self._error.setVisible(False)


__all__ = ["ControlsPanel"]
