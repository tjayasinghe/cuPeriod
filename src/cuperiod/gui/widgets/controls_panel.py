"""The left-hand controls: analysis/method/backend pickers, options form, and Compute.

The panel drives both analyses from one layout. In **Periodogram** mode the method
picker selects one of the registered methods; in **Pre-whitening** mode the method and
peak-count rows are hidden and the same machinery serves
:class:`~cuperiod.PreWhitenSettings` instead. Either way the options form is a
dynamically rebuilt :class:`PydanticSettingsForm` — every field of the chosen settings
model is exposed automatically, so the pre-whitening controls needed no bespoke
widgets — and the backend picker is limited to backends that actually resolve here.

Settings are cached per analysis/method, so switching back restores your values.
``Compute`` validates and emits :attr:`run_requested` or :attr:`prewhiten_requested`;
invalid input is shown inline instead of starting a run. In pre-whitening the button
is *Auto* (run to the stopping rule) and a **Step** button beside it tries one peak
(:attr:`step_requested`). **Stop** is enabled while anything runs and asks the window
to stop it (:attr:`stop_requested`; Esc does the same). The panel's choices — analysis,
method, backend, peak count and every form's values — can be saved and restored as
one JSON text (:meth:`saved_state` / :meth:`restore_state`), so a relaunch comes back
as it was left.

The backend and band pickers remember what the *user* chose and re-select it whenever
they are rebuilt — on a curve load, a batch step, a method or analysis switch — as
long as it is still offered. Rebuilding them used to snap back to ``auto`` and the
first band, so a batch run chosen for the CPU went back to the GPU on the next source.
"""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from cuperiod.core._typing import FloatArray
from cuperiod.core.config import PreWhitenSettings
from cuperiod.gui.meta import (
    auto_max_frequency,
    backend_options,
    method_display_names,
    multiband_method_names,
    natural_domain,
    objective_sense,
    prewhiten_backend_options,
    reaches_short_periods,
    resolved_backend,
    resolved_prewhiten_backend,
    settings_class,
    supports_multiband,
)
from cuperiod.gui.models import COMBINED_BAND, STACKED_BAND
from cuperiod.gui.qt import Qt, QtWidgets, Signal
from cuperiod.gui.settingsform import PydanticSettingsForm
from cuperiod.prewhiten.engine import default_maximum_frequency

_DEFAULT_METHOD = "GLS"

#: Display labels for the analysis picker, in order.
_PERIODOGRAM_LABEL = "Periodogram"
_PREWHITEN_LABEL = "Pre-whitening"

#: Labels of the Compute button, one per analysis. The window's guidance text names
#: the button through :attr:`ControlsPanel.action_label`, so the two always agree.
COMPUTE_LABEL = "Compute periodogram"
PREWHITEN_RUN_LABEL = "Run pre-whitening"
STEP_LABEL = "Step"
STOP_LABEL = "Stop"

#: Cache key for the pre-whitening settings form (methods use their own names).
_PREWHITEN_CACHE_KEY = "__prewhiten__"

#: Item data of the synthetic "all bands" combo entries — the controller's own tokens,
#: which :meth:`ControlsPanel.current_band` hands over unchanged. Mapping them back to
#: the bare words "combined"/"stacked" let a real band of that name take over the
#: all-band analysis (the controller tests ``band in lc.bands`` first) and share its
#: cache key.
_COMBINED_SENTINEL = COMBINED_BAND
_STACKED_SENTINEL = STACKED_BAND

#: Which kind of band list a picker was built for. The selection is remembered per
#: kind: a single-band analysis defaults to the first band and a joint one to
#: "combined", and one must not leak into the other.
BandKind = Literal["joint", "single", "prewhiten"]

#: One-line descriptions of the registered methods, shown as item tooltips and in the
#: note under the pickers (the picker itself shows the registry key, which is what the
#: Python API and the CLI take).
METHOD_DESCRIPTIONS: dict[str, str] = {
    "BLS": "Box Least Squares — periodic box-shaped dips: transits and detached "
    "eclipsing binaries (flux).",
    "CE": "Conditional Entropy — non-parametric, any light-curve shape; robust to "
    "outliers.",
    "GLS": "Generalised Lomb–Scargle — least-squares sinusoid; smooth, "
    "near-sinusoidal variables.",
    "MHAOV": "Multi-harmonic analysis of variance — a Fourier series per trial "
    "period; non-sinusoidal pulsators such as RRab stars.",
    "PDM": "Phase Dispersion Minimisation — non-parametric, any light-curve shape.",
    "STRINGLENGTH": "String length (Lafler–Kinman) — length of the phase-ordered "
    "curve; any shape, sparse data.",
    "SUPERSMOOTHER": "SuperSmoother — adaptive smoother fit; any shape, eclipsing "
    "binaries.",
    "TLS": "Transit Least Squares — limb-darkened transit template; small planets "
    "(flux, single band).",
}


def method_description(method: str) -> str:
    """A one-line description of ``method`` (empty for an unknown name)."""
    return METHOD_DESCRIPTIONS.get(method.upper(), "")


def _field_label(name: str) -> str:
    """The options-form label for settings field ``name`` (see settingsform)."""
    return name.replace("_", " ")


def _first_error_message(
    exc: ValidationError, fields: tuple[str, ...] | list[str] = ()
) -> str:
    """The first validation error, worded with the form's labels.

    Pydantic reports ``minimum_frequency: ...`` for a field and ``Value error,
    minimum_frequency (5.0) must be < maximum_frequency (1.0)`` for a model-level
    check. The form labels those fields ``minimum frequency``, so the message uses the
    same words and drops pydantic's ``Value error,`` preamble.
    """
    errors = exc.errors()
    if not errors:
        return str(exc)
    first = errors[0]
    loc = ".".join(_field_label(str(part)) for part in first.get("loc", ()))
    msg = str(first.get("msg", "invalid value"))
    msg = re.sub(r"^(Value|Assertion) error,\s*", "", msg)
    for name in sorted(fields, key=len, reverse=True):
        msg = re.sub(rf"\b{re.escape(name)}\b", _field_label(name), msg)
    return f"{loc}: {msg}" if loc else msg


def _plain_muted_label() -> QtWidgets.QLabel:
    label = QtWidgets.QLabel("")
    label.setObjectName("muted")
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


class ControlsPanel(QtWidgets.QWidget):
    """Method/backend pickers, the auto-generated options form, and Compute."""

    # Emitted on Compute in periodogram mode: (method, backend, settings, n_peaks).
    run_requested = Signal(str, str, object, int)
    # Emitted on Compute in pre-whitening mode: (backend, PreWhitenSettings).
    prewhiten_requested = Signal(str, object)
    # Emitted when the analysis picker changes: "periodogram" | "prewhiten".
    analysis_changed = Signal(str)
    # Emitted whenever the band selection may have changed (the user picked one, or
    # the picker was rebuilt); carries current_band(). The window answers with the
    # times of that band, so the grid hint describes the input that will be analysed.
    band_changed = Signal(str)
    # Emitted when the user changes anything that alters the next run (method,
    # backend, band, peak count or an option), so a displayed result can be flagged
    # as out of date. Programmatic rebuilds do not emit it.
    settings_edited = Signal()
    # Emitted whenever Compute is pressed (or requested), before the options are
    # validated: a new attempt supersedes the previous run's error message.
    compute_attempted = Signal()
    # Emitted on Step (pre-whitening only): (backend, PreWhitenSettings).
    step_requested = Signal(str, object)
    # Emitted on Stop: the window stops whatever runs (a session job or a one-shot).
    stop_requested = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings_cache: dict[str, BaseSettings] = {}
        self._current_method: str | None = None
        self._form: PydanticSettingsForm | None = None
        self._curve_time: FloatArray | None = None
        self._has_curve = False
        self._busy = False
        self._prewhiten = False
        #: The backend the user last picked; re-selected after every rebuild of the
        #: backend picker while it is offered, else the picker shows "auto".
        self._backend_choice = "auto"
        #: The band (item data) the user last picked, per kind of band list.
        self._band_choice: dict[BandKind, str] = {}
        self._band_kind: BandKind | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QtWidgets.QLabel("Controls")
        header.setObjectName("heading")
        root.addWidget(header)

        self._top = QtWidgets.QFormLayout()
        self._top.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._analysis_combo = QtWidgets.QComboBox()
        self._analysis_combo.addItems([_PERIODOGRAM_LABEL, _PREWHITEN_LABEL])
        self._analysis_combo.setToolTip(
            "Periodogram: one period-search statistic over a trial grid.\n"
            "Pre-whitening: iterative sinusoid extraction with uncertainties, "
            "combination frequencies, and g-mode period spacings."
        )
        self._method_combo = QtWidgets.QComboBox()
        self._fill_method_combo(method_display_names())
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
        self._top.addRow("Analysis", self._analysis_combo)
        self._top.addRow("Method", self._method_combo)
        self._top.addRow("Band", self._band_combo)
        self._top.addRow("Backend", self._backend_combo)
        self._top.addRow("Peaks", self._peaks_spin)
        self._top.setRowVisible(self._band_combo, False)
        self._band_names: list[str] | None = None
        root.addLayout(self._top)

        # Everything below is plain text: hints quote numbers and backend names, and
        # the error line quotes values the user typed.
        self._note = _plain_muted_label()
        root.addWidget(self._note)
        self._backend_hint = _plain_muted_label()
        root.addWidget(self._backend_hint)
        self._grid_hint = _plain_muted_label()
        self._grid_hint.setToolTip("The trial grid the auto settings will search")
        root.addWidget(self._grid_hint)

        options_group = QtWidgets.QGroupBox("Options")
        og_layout = QtWidgets.QVBoxLayout(options_group)
        og_layout.setContentsMargins(6, 6, 6, 6)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # The scroll area is a container, not a control: as a Tab stop it only adds
        # an invisible focus position between the pickers and the first option.
        self._scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        og_layout.addWidget(self._scroll)
        root.addWidget(options_group, 1)

        self._error = QtWidgets.QLabel("")
        self._error.setObjectName("error")
        self._error.setWordWrap(True)
        self._error.setTextFormat(Qt.TextFormat.PlainText)
        self._error.setVisible(False)
        root.addWidget(self._error)

        self._compute_btn = QtWidgets.QPushButton(COMPUTE_LABEL)
        self._compute_btn.setProperty("primary", True)
        self._compute_tip = "Compute the periodogram  (Ctrl+Enter)"
        self._compute_btn.clicked.connect(self._on_compute)
        # Step sits beside Auto in pre-whitening; Stop is enabled only while a run is
        # in flight. Compute keeps its own disabled state while busy (the next run can
        # be prepared, not started), so Stop is a button of its own, not a relabel.
        self._step_btn = QtWidgets.QPushButton(STEP_LABEL)
        self._step_btn.setToolTip(
            "Step: try the tallest untested peak once, one iteration of Auto  (F7)"
        )
        self._step_btn.setStatusTip("Try the tallest untested peak once")
        self._step_btn.clicked.connect(self._on_step)
        self._step_btn.setVisible(False)
        self._stop_btn = QtWidgets.QPushButton(STOP_LABEL)
        self._stop_btn.setAccessibleName("Stop the running computation")
        self._stop_btn.setToolTip(
            "Stop the running computation; what is done so far is kept  (Esc)"
        )
        self._stop_btn.setStatusTip("Stop the running computation")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_requested)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self._compute_btn, 1)
        buttons.addWidget(self._step_btn)
        buttons.addWidget(self._stop_btn)
        root.addLayout(buttons)

        if _DEFAULT_METHOD in method_display_names():
            self._method_combo.setCurrentText(_DEFAULT_METHOD)
        self._method_combo.currentTextChanged.connect(self._on_method_picked)
        self._backend_combo.currentTextChanged.connect(self._on_backend_picked)
        self._band_combo.currentIndexChanged.connect(self._on_band_picked)
        self._peaks_spin.valueChanged.connect(lambda *_: self.settings_edited.emit())
        self._analysis_combo.currentTextChanged.connect(self._on_analysis_changed)
        self._on_method_changed(self._method_combo.currentText())
        self.set_enabled(False)

    # -- analysis switching ------------------------------------------------------
    @property
    def analysis(self) -> str:
        """``"periodogram"`` or ``"prewhiten"``."""
        return "prewhiten" if self._prewhiten else "periodogram"

    @property
    def action_label(self) -> str:
        """The Compute button's label for the current analysis."""
        return PREWHITEN_RUN_LABEL if self._prewhiten else COMPUTE_LABEL

    def set_analysis(self, analysis: str) -> None:
        """Programmatically switch the analysis picker."""
        label = _PREWHITEN_LABEL if analysis == "prewhiten" else _PERIODOGRAM_LABEL
        if self._analysis_combo.currentText() != label:
            self._analysis_combo.setCurrentText(label)

    def _on_analysis_changed(self, label: str) -> None:
        self._stash_current_settings()
        self._prewhiten = label == _PREWHITEN_LABEL
        self._top.setRowVisible(self._method_combo, not self._prewhiten)
        self._top.setRowVisible(self._peaks_spin, not self._prewhiten)
        self._compute_btn.setText(self.action_label)
        self._compute_tip = (
            "Auto: extract frequencies until the stopping rule fires  (F5, Ctrl+Enter)"
            if self._prewhiten
            else "Compute the periodogram  (Ctrl+Enter)"
        )
        self._step_btn.setVisible(self._prewhiten)
        if self._prewhiten:
            self._current_method = _PREWHITEN_CACHE_KEY
            self._rebuild_backend_combo(None)
            self._rebuild_band_combo(None)
            self._rebuild_form(_PREWHITEN_CACHE_KEY, PreWhitenSettings)
            self._note.setText(
                "iterative extraction  ·  single-band  ·  amplitude spectrum"
            )
            self._notify_band()
        else:
            self._current_method = None
            self._on_method_changed(self._method_combo.currentText())
        self._update_compute_enabled()
        self._update_backend_hint()
        self._update_grid_hint()
        self._clear_error()
        self.analysis_changed.emit(self.analysis)

    def _stash_current_settings(self) -> None:
        """Remember the current form's values under the key it was built for."""
        if self._current_method is not None and self._form is not None:
            with contextlib.suppress(ValidationError):
                self._settings_cache[self._current_method] = self._form.build()

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

    def request_step(self) -> None:
        """Step programmatically (F7, the registry's ``session.step``)."""
        self._on_step()

    @property
    def current_backend(self) -> str:
        """The backend picked in the Backend combo."""
        return self._backend_combo.currentText() or "auto"

    def prewhiten_request(
        self, *, report: bool = True
    ) -> tuple[str, PreWhitenSettings] | None:
        """``(backend, settings)`` of the pre-whitening form, or None.

        None outside pre-whitening and while the form does not validate; with
        ``report`` the reason is shown under the form, as on Compute. Step, Improve
        and the lazy spectrum preview read the form through this.
        """
        if not self._prewhiten or self._form is None:
            return None
        try:
            settings = self._form.build()
        except ValidationError as exc:
            if report:
                fields = list(PreWhitenSettings.model_fields)
                self._show_error(_first_error_message(exc, fields))
            return None
        if not isinstance(settings, PreWhitenSettings):  # pragma: no cover - typing
            return None
        if report:
            self._clear_error()
        return self.current_backend, settings

    def set_prewhiten_settings(self, settings: PreWhitenSettings) -> None:
        """Load ``settings`` into the pre-whitening form (an opened project's).

        Not a user edit: :attr:`settings_edited` is not emitted.
        """
        self._settings_cache[_PREWHITEN_CACHE_KEY] = settings
        if self._prewhiten:
            self._rebuild_form(_PREWHITEN_CACHE_KEY, PreWhitenSettings)
            self._notify_band()
            self._update_backend_hint()
            self._update_grid_hint()

    # -- persistence (A11Y-19) ----------------------------------------------------
    def saved_state(self) -> str:
        """The panel's choices as JSON: analysis, method, backend, peaks, forms.

        Every form's values are kept (the one on show included), so a relaunch
        restores the options of each method and of pre-whitening.
        """
        self._stash_current_settings()
        forms: dict[str, Any] = {}
        for key, settings in self._settings_cache.items():
            with contextlib.suppress(Exception):  # an exotic field must not block
                forms[key] = settings.model_dump(mode="json")
        state = {
            "analysis": self.analysis,
            "method": self._method_combo.currentText(),
            "backend": self._backend_choice,
            "n_peaks": self._peaks_spin.value(),
            "forms": forms,
        }
        return json.dumps(state)

    def restore_state(self, text: str) -> None:
        """Restore :meth:`saved_state` output; anything that does not validate (an
        older version's field, a hand-edited file) is skipped, never raised."""
        try:
            state = json.loads(text)
        except (TypeError, ValueError):
            return
        if not isinstance(state, dict):
            return
        forms = state.get("forms")
        if isinstance(forms, dict):
            for key, values in forms.items():
                if not isinstance(values, dict):
                    continue
                model: type[BaseSettings] | None
                if key == _PREWHITEN_CACHE_KEY:
                    model = PreWhitenSettings
                else:
                    try:
                        model = settings_class(str(key))
                    except Exception:  # noqa: BLE001 - a method no longer offered
                        model = None
                if model is None:
                    continue
                with contextlib.suppress(Exception):
                    self._settings_cache[str(key)] = model.model_validate(values)
        backend = state.get("backend")
        if isinstance(backend, str) and backend:
            self._backend_choice = backend
        n_peaks = state.get("n_peaks")
        if isinstance(n_peaks, int) and not isinstance(n_peaks, bool):
            self._peaks_spin.blockSignals(True)
            self._peaks_spin.setValue(n_peaks)
            self._peaks_spin.blockSignals(False)
        # The form on show was built before the cache was restored: forget which key
        # it belongs to, or the switches below would stash its defaults over the
        # restored values.
        self._current_method = None
        method = state.get("method")
        if (
            isinstance(method, str)
            and self._method_combo.findText(method, Qt.MatchFlag.MatchExactly) >= 0
        ):
            self._method_combo.blockSignals(True)
            self._method_combo.setCurrentText(method)
            self._method_combo.blockSignals(False)
        if state.get("analysis") == "prewhiten" and not self._prewhiten:
            self.set_analysis("prewhiten")
        elif self._prewhiten:
            self._on_analysis_changed(_PREWHITEN_LABEL)
        else:
            self._on_method_changed(self._method_combo.currentText())

    def _update_compute_enabled(self) -> None:
        self._compute_btn.setEnabled(self._has_curve and not self._busy)
        self._step_btn.setEnabled(self._has_curve and not self._busy)
        self._stop_btn.setEnabled(self._busy)
        # A disabled button's tooltip is the only place left to say *why*.
        if not self._has_curve:
            tip = "Load a light curve first — Open…, Open folder…, Load demo, or drop "
            tip += "a file on the window."
        elif self._busy:
            tip = "A run is in progress; the next one can start when it finishes."
        else:
            tip = self._compute_tip
        self._compute_btn.setToolTip(tip)

    def set_method(self, method: str) -> bool:
        """Select ``method`` in the picker if it is offered; return whether it was.

        Used to apply a demo's suggested method. Not a user edit, so
        :attr:`settings_edited` is not emitted.
        """
        if self._method_combo.findText(method, Qt.MatchFlag.MatchExactly) < 0:
            return False
        if self._method_combo.currentText() != method:
            self._method_combo.blockSignals(True)
            self._method_combo.setCurrentText(method)
            self._method_combo.blockSignals(False)
            self._on_method_changed(method)
        return True

    def current_method(self) -> str:
        """The method selected in the picker (meaningful in periodogram mode)."""
        return self._method_combo.currentText()

    def set_multiband(self, is_multiband: bool) -> None:
        """Restrict the method list to multiband-capable methods for multiband data.

        Nothing is rebuilt when the list is already the one required: this runs on
        every curve load and batch step, and rebuilding reset the form (its scroll
        position and any edit in progress) along with the pickers.
        """
        names = multiband_method_names() if is_multiband else method_display_names()
        shown = [
            self._method_combo.itemText(i) for i in range(self._method_combo.count())
        ]
        if names == shown:
            return
        current = self._method_combo.currentText()
        self._method_combo.blockSignals(True)
        self._fill_method_combo(names)
        if current in names:
            self._method_combo.setCurrentText(current)
        elif _DEFAULT_METHOD in names:
            self._method_combo.setCurrentText(_DEFAULT_METHOD)
        self._method_combo.blockSignals(False)
        self._on_method_changed(self._method_combo.currentText())

    def set_bands(self, band_names: list[str] | None) -> None:
        """Show a band selector for multiband curves; hide it for single-band ones."""
        self._band_names = list(band_names) if band_names else None
        # ``None`` means pre-whitening, which is single-band by definition. Passing
        # the (hidden) method combo's text would offer "combined (all bands)" and
        # select it, so loading a multiband curve *after* switching analysis would
        # silently analyse a raw all-band stack instead of one band.
        self._rebuild_band_combo(
            None if self._prewhiten else self._method_combo.currentText()
        )
        self._notify_band()

    def set_curve_time(self, time: FloatArray) -> None:
        """Provide the analysed input's times so the auto grid range can be shown.

        Non-finite times (blank cells, FITS gap cadences) are ignored, as the methods
        ignore them: a NaN would otherwise turn the baseline, and with it the whole
        hint, into ``nan`` / ``inf``.
        """
        finite = np.asarray(time, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        self._curve_time = finite if finite.size >= 2 else None
        self._update_grid_hint()

    def current_band(self) -> str:
        """The selected band: a band name, the all-band token
        :data:`~cuperiod.gui.models.COMBINED_BAND` or
        :data:`~cuperiod.gui.models.STACKED_BAND`, or ``""`` for a single-band curve.

        The synthetic entries return the controller's tokens rather than the words
        "combined"/"stacked", so a real band of that name cannot stand in for the
        all-band analyses.
        """
        if self._band_names is None:
            return ""
        data = self._band_combo.currentData()
        if data in (_COMBINED_SENTINEL, _STACKED_SENTINEL):
            return str(data)
        return self._band_combo.currentText()

    # -- method switching --------------------------------------------------------
    def _fill_method_combo(self, names: list[str]) -> None:
        self._method_combo.clear()
        for i, name in enumerate(names):
            self._method_combo.addItem(name)
            self._method_combo.setItemData(
                i, method_description(name), Qt.ItemDataRole.ToolTipRole
            )

    def _on_method_picked(self, method: str) -> None:
        self._on_method_changed(method)
        self.settings_edited.emit()

    def _on_method_changed(self, method: str) -> None:
        if self._prewhiten:
            return
        self._stash_current_settings()
        self._current_method = method
        self._method_combo.setToolTip(
            method_description(method) or "Periodogram method to run"
        )
        self._rebuild_backend_combo(method)
        self._rebuild_band_combo(method)
        self._rebuild_form(method, settings_class(method))
        self._update_note(method)
        self._update_backend_hint()
        self._notify_band()
        self._update_grid_hint()
        self._clear_error()

    def _rebuild_band_combo(self, method: str | None) -> None:
        """Rebuild the band picker; ``method=None`` means the pre-whitening analysis.

        Re-selects the band the user last picked for this kind of list when the new
        list still has it, else the kind's default (the first entry).
        """
        if self._band_names is None:
            self._top.setRowVisible(self._band_combo, False)
            self._band_kind = None
            return
        kind: BandKind
        if method is None:
            kind = "prewhiten"
        else:
            kind = "joint" if supports_multiband(method) else "single"
        self._top.setRowVisible(self._band_combo, True)
        self._band_combo.blockSignals(True)
        self._band_combo.clear()
        if kind == "joint":
            # combined multiband analysis is the default for capable methods
            self._band_combo.addItem("combined (all bands)", _COMBINED_SENTINEL)
            for name in self._band_names:
                self._band_combo.addItem(name, name)
        else:
            # single-band methods analyse one band (safest) or a stacked merge
            for name in self._band_names:
                self._band_combo.addItem(name, name)
            self._band_combo.addItem("stacked (all bands)", _STACKED_SENTINEL)
        wanted = self._band_choice.get(kind)
        index = self._band_combo.findData(wanted) if wanted is not None else -1
        self._band_combo.setCurrentIndex(max(index, 0))
        self._band_combo.blockSignals(False)
        self._band_kind = kind

    def _on_band_picked(self, _index: int) -> None:
        data = self._band_combo.currentData()
        if self._band_kind is not None and isinstance(data, str):
            self._band_choice[self._band_kind] = data
        self._notify_band()
        self.settings_edited.emit()

    def _notify_band(self) -> None:
        band = self.current_band()
        if self._form is not None:
            # The mb_* options tune only the joint multiband fit; for one band or a
            # single-band curve they are inert, so they are hidden (they still build).
            self._form.set_multiband(
                self._band_names is not None and band == COMBINED_BAND
            )
        self.band_changed.emit(band)

    def _rebuild_backend_combo(self, method: str | None) -> None:
        """Rebuild the backend picker; ``method=None`` means pre-whitening.

        The user's last pick survives whenever the new method offers it; only an
        unavailable choice falls back to ``auto``.
        """
        options = (
            prewhiten_backend_options() if method is None else backend_options(method)
        )
        self._backend_combo.blockSignals(True)
        self._backend_combo.clear()
        self._backend_combo.addItems(options)
        choice = self._backend_choice if self._backend_choice in options else "auto"
        self._backend_combo.setCurrentText(choice)
        self._backend_combo.blockSignals(False)

    def _on_backend_picked(self, backend: str) -> None:
        if backend:
            self._backend_choice = backend
        self._update_backend_hint()
        self.settings_edited.emit()

    def _rebuild_form(self, cache_key: str, model: type[BaseSettings]) -> None:
        cached = self._settings_cache.get(cache_key)
        form = PydanticSettingsForm(
            model, initial=cached if isinstance(cached, model) else None
        )
        form.changed.connect(self._on_form_changed)
        self._scroll.setWidget(form)  # takes ownership and deletes the previous form
        self._form = form
        self._restitch_tab_order(form)

    def focusNextPrevChild(self, next: bool) -> bool:  # noqa: N802, A002 - Qt override
        """Tab, and scroll the Options so the field it lands on is in view.

        The scroll area only does that for moves that start inside it, so Tab from
        the Peaks box (or Shift+Tab from Compute) could focus an option scrolled
        out of sight.
        """
        moved = super().focusNextPrevChild(next)
        focus = QtWidgets.QApplication.focusWidget()
        form = self._scroll.widget()
        inside = focus is not None and form is not None and form.isAncestorOf(focus)
        if moved and inside and focus is not None:
            self._scroll.ensureWidgetVisible(focus)
        return moved

    def _restitch_tab_order(self, form: QtWidgets.QWidget) -> None:
        """Put the new form's fields between the pickers and Compute in Tab order.

        Qt appends a newly parented widget tree to the *end* of the window's focus
        chain, so after every rebuild the options came after the plots and the result
        tables instead of before Compute.
        """
        fields: list[QtWidgets.QWidget] = []
        widget = form.nextInFocusChain()
        for _ in range(10_000):  # the chain is a cycle; this only bounds a bad one
            if widget is form or widget is None:
                break
            if (
                form.isAncestorOf(widget)
                and widget.focusPolicy() & Qt.FocusPolicy.TabFocus
                and widget.focusProxy() is None
            ):
                fields.append(widget)
            widget = widget.nextInFocusChain()
        previous: QtWidgets.QWidget = self._peaks_spin
        for field in fields:
            QtWidgets.QWidget.setTabOrder(previous, field)
            previous = field
        QtWidgets.QWidget.setTabOrder(previous, self._compute_btn)
        # A compound field (an "auto" box beside its spin, a group of check boxes)
        # is a NoFocus container whose focus proxy is its first box, and Qt 6 takes
        # it as a Tab stop at its *own* place in the chain. The moves above left
        # those containers behind Compute, so Tab from Compute jumped back into
        # the options for ever (a keyboard trap). Setting the proxy again puts each
        # container right in front of its proxy.
        for widget in form.findChildren(QtWidgets.QWidget):
            proxy = widget.focusProxy()
            if proxy is not None and widget.isAncestorOf(proxy):
                widget.setFocusProxy(None)  # type: ignore[arg-type]  # clears it
                widget.setFocusProxy(proxy)

    def _on_form_changed(self) -> None:
        self._clear_error()
        self._update_backend_hint()  # 'auto' can defer to the form's backend
        self._update_grid_hint()
        self.settings_edited.emit()

    def _update_note(self, method: str) -> None:
        sense = objective_sense(method)
        kind = "peaks are maxima" if sense == "max" else "peaks are minima"
        description = method_description(method)
        bits = [description.split(" — ")[0]] if description else []
        bits += [
            kind,
            "multiband-capable" if supports_multiband(method) else "single-band",
        ]
        domain = natural_domain(method)
        if domain is not None:
            bits.append(f"domain: {domain.value}")
        self._note.setText("  ·  ".join(bits))

    def _update_backend_hint(self) -> None:
        chosen = self._backend_combo.currentText()
        # Resolve with the form's own settings, as the run does: 'auto' defers to
        # settings.backend and a bare 'torch' runs on settings.device.
        settings = self._form_settings()
        info = (
            resolved_prewhiten_backend(chosen, settings)
            if self._prewhiten
            else resolved_backend(self._method_combo.currentText(), chosen, settings)
        )
        if info is None:
            self._backend_hint.setText("")
            return
        resolved, is_gpu = info
        where = "GPU" if is_gpu else "CPU"
        prefix = "" if resolved == chosen else f"{chosen} → "
        self._backend_hint.setText(f"runs on:  {prefix}{resolved}  ({where})")

    def _form_settings(self) -> BaseSettings | None:
        """The form's current settings, or None while they do not validate."""
        if self._form is None:
            return None
        try:
            return self._form.build()
        except ValidationError:
            return None

    def _settings_model(self) -> type[BaseSettings]:
        """The settings model backing the current form."""
        if self._prewhiten:
            return PreWhitenSettings
        return settings_class(self._method_combo.currentText())

    def _update_grid_hint(self) -> None:
        if (
            self._curve_time is None
            or self._form is None
            or "maximum_frequency" not in self._settings_model().model_fields
        ):
            self._grid_hint.setText("")
            return
        time = self._curve_time
        baseline = float(time.max() - time.min())
        nyquist_factor = int(self._form.value_of("nyquist_factor") or 5)
        auto_min = 1.0 / baseline if baseline > 0.0 else 0.0
        # Frequency-domain analyses floor their auto band higher (delta Scuti / HADS
        # coverage); mirror the controller exactly so the greyed value is the one
        # that will run.
        auto_max = (
            default_maximum_frequency(time, nyquist_factor)
            if reaches_short_periods(self._settings_model())
            else auto_max_frequency(time, nyquist_factor)
        )
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
            f"grid → f: {fmin:.3g}–{fmax:.3g} c/d   ·   P: {p_short:.3g}–{p_long:.3g} d"
        )

    # -- compute -----------------------------------------------------------------
    def _on_compute(self) -> None:
        if self._form is None:
            return
        self.compute_attempted.emit()
        try:
            settings = self._form.build()
        except ValidationError as exc:
            fields = list(self._settings_model().model_fields)
            self._show_error(_first_error_message(exc, fields))
            return
        self._clear_error()
        if self._prewhiten:
            self.prewhiten_requested.emit(self._backend_combo.currentText(), settings)
            return
        self.run_requested.emit(
            self._method_combo.currentText(),
            self._backend_combo.currentText(),
            settings,
            self._peaks_spin.value(),
        )

    def _on_step(self) -> None:
        if not (self._has_curve and not self._busy):
            return
        self.compute_attempted.emit()
        request = self.prewhiten_request()
        if request is not None:
            self.step_requested.emit(*request)

    def _show_error(self, message: str) -> None:
        self._error.setText(f"Invalid setting — {message}")
        self._error.setVisible(True)

    def _clear_error(self) -> None:
        self._error.setVisible(False)


__all__ = [
    "COMPUTE_LABEL",
    "METHOD_DESCRIPTIONS",
    "PREWHITEN_RUN_LABEL",
    "STEP_LABEL",
    "STOP_LABEL",
    "ControlsPanel",
    "method_description",
]
