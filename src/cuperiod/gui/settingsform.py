"""Auto-generate a settings form from a pydantic model.

Each periodogram method has a pydantic settings model (``GLSSettings``, ``BLSSettings``,
...). Rather than hand-write a form per method, :func:`field_to_spec` reduces each model
field to a small, Qt-free :class:`FieldSpec` (kind, choices, bounds, default), and
:class:`PydanticSettingsForm` turns those into labelled widgets, read back into a
validated settings object. The descriptor step is pure, so it is unit-tested per method.

Every field gets a widget, tuples included (a check-box group for a tuple of choices
such as pre-whitening's ``stop_criteria``, a comma-separated list for a tuple of
numbers). Fields are grouped into labelled, collapsible sections -- the search range
first, the performance knobs in a collapsed *Advanced* section -- with readable labels
and units, so the handful of fields that usually matter are not buried in a flat list.
Only :data:`GUI_OWNED_FIELDS` are left out, each because another control owns it. The
Backend picker stands in for ``backend``: ``backend=`` overrides ``settings.backend``,
and ``"auto"`` defers to it, so a ``CUPERIOD_<METHOD>_BACKEND`` pin still applies.

The form starts from the GUI's defaults
(:func:`~cuperiod.gui.meta.gui_default_settings`): the library's settings including any
``CUPERIOD_<METHOD>_<FIELD>`` environment overrides, with the GUI's denser frequency
oversampling shown as such, so the values on screen are the values that run.
"""

from __future__ import annotations

import math
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Final, Literal, Union, get_args, get_origin

import annotated_types as at
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from cuperiod.core.config import PreWhitenSettings
from cuperiod.gui.meta import gui_default_settings
from cuperiod.gui.qt import Qt, QtGui, QtWidgets, Signal

_NONE_TYPE = type(None)

FieldKind = Literal[
    "combo",
    "bool",
    "int",
    "float",
    "optional_int",
    "optional_float",
    "multi_choice",
    "float_list",
]

#: Fields the options form leaves out because another part of the GUI owns them:
#: ``backend`` (the Backend picker; ``backend=`` overrides ``settings.backend`` and
#: ``"auto"`` defers to it),
#: ``n_peaks`` (the Peaks spin, which the controller passes explicitly),
#: ``store_spectra`` (the GUI draws the spectra, so the controller always keeps them;
#: unticked, the plot was left empty) and ``downsample_points`` (the size of a stored
#: downsampled spectrum; the GUI plots the full resolution and never downsamples).
GUI_OWNED_FIELDS: Final[frozenset[str]] = frozenset(
    {"backend", "n_peaks", "store_spectra", "downsample_points"}
)

#: QDoubleSpinBox rounds every value it holds to ``decimals`` places, so 6 decimals
#: turned a typed 1e-12 FAP threshold into 0 (and invalid). Qt's maximum keeps any
#: double exactly; the *display* is :meth:`_GSpinBox.textFromValue`, not ``decimals``.
_HELD_DECIMALS: Final = 323

# -- sections ---------------------------------------------------------------------

SEARCH: Final = "Search range"
EXTRACTION: Final = "Extraction"
STOPPING: Final = "Stopping"
MODEL: Final = "Model"
PEAKS: Final = "Peaks"
MULTIBAND: Final = "Multiband"
UNCERTAINTIES: Final = "Uncertainties"
COMBINATIONS: Final = "Combinations"
OTHER: Final = "Other"
ADVANCED: Final = "Advanced"

#: Section display order.
SECTION_ORDER: Final[tuple[str, ...]] = (
    SEARCH,
    EXTRACTION,
    STOPPING,
    MODEL,
    PEAKS,
    MULTIBAND,
    UNCERTAINTIES,
    COMBINATIONS,
    OTHER,
    ADVANCED,
)

#: Sections that start collapsed: secondary or performance knobs.
COLLAPSED_SECTIONS: Final[frozenset[str]] = frozenset(
    {MULTIBAND, UNCERTAINTIES, COMBINATIONS, ADVANCED}
)

#: What the help line under the form says before an option has had the focus.
HELP_PLACEHOLDER: Final = "Tab to an option to read what it does here."

#: Section of each known field (periodogram methods). Unknown fields go to "Other".
_SECTION_OF: Final[Mapping[str, str]] = {
    **dict.fromkeys(
        (
            "minimum_frequency",
            "maximum_frequency",
            "nyquist_factor",
            "samples_per_peak",
            "min_period_days",
            "max_period_days",
            "min_transits",
            "oversample",
            "grid_oversample",
            "grid_duration_frac",
            "segment_factor",
        ),
        SEARCH,
    ),
    **dict.fromkeys(
        (
            "fit_mean",
            "fap_method",
            "n_harmonics",
            "n_bins",
            "n_covers",
            "n_phase_bins",
            "n_mag_bins",
            "primary_spans",
            "middle_span",
            "final_span",
            "bass_enhancement",
            "objective",
            "duration_min_frac",
            "duration_max_frac",
            "n_durations",
            "min_duration_days",
            "bins_per_duration",
            "limb_dark_u1",
            "limb_dark_u2",
        ),
        MODEL,
    ),
    **dict.fromkeys(
        ("peak_separation_rayleigh", "alias_freq_tolerance", "harmonic_max"), PEAKS
    ),
    **dict.fromkeys(
        (
            "mb_model",
            "mb_nterms_base",
            "mb_nterms_band",
            "mb_reg_base",
            "mb_reg_band",
            "mb_regularize_by_trace",
            "mb_fap_bootstrap",
            "mb_fap_seed",
        ),
        MULTIBAND,
    ),
    **dict.fromkeys(
        (
            "device",
            "precision",
            "min_detections",
            "batch_periods",
            "period_batch",
            "nufft_eps",
            "direct_freq_batch",
        ),
        ADVANCED,
    ),
}

#: Pre-whitening's own grouping (it shares a few field names with the periodograms).
_PREWHITEN_SECTION_OF: Final[Mapping[str, str]] = {
    **dict.fromkeys(
        (
            "max_frequencies",
            "normalization",
            "fit_mean",
            "min_separation_rayleigh",
            "refine",
            "refine_bound_rayleigh",
            "sweeps",
            "final_refine",
            "max_simultaneous",
            "max_nfev",
        ),
        EXTRACTION,
    ),
    **dict.fromkeys(
        (
            "stop_criteria",
            "snr_threshold",
            "snr_window",
            "noise_estimator",
            "fap_threshold",
            "min_delta_bic",
            "min_amplitude",
            "prune",
            "blend_tolerance",
        ),
        STOPPING,
    ),
    **dict.fromkeys(
        ("uncertainty", "correlation_correction", "n_resamples", "seed"),
        UNCERTAINTIES,
    ),
    **dict.fromkeys(
        (
            "combinations",
            "combination_max_order",
            "combination_parents",
            "combination_tolerance_rayleigh",
            "combination_sigma",
        ),
        COMBINATIONS,
    ),
}

#: Display labels (with units) where the field name alone is cryptic. Anything not
#: listed is shown as its name with spaces; the tooltip always carries the full
#: description from the settings model.
_LABELS: Final[Mapping[str, str]] = {
    "minimum_frequency": "min frequency (c/d)",
    "maximum_frequency": "max frequency (c/d)",
    "nyquist_factor": "Nyquist factor",
    "min_period_days": "min period (d)",
    "max_period_days": "max period (d)",
    "oversample": "frequency oversampling",
    "grid_oversample": "frequency oversampling",
    "grid_duration_frac": "grid step (x duration)",
    "segment_factor": "segment period ratio",
    "fit_mean": "floating mean",
    "fap_method": "FAP method",
    "n_harmonics": "harmonics",
    "n_bins": "phase bins",
    "n_covers": "bin covers",
    "n_phase_bins": "phase bins",
    "n_mag_bins": "magnitude bins",
    "duration_min_frac": "min duration (x period)",
    "duration_max_frac": "max duration (x period)",
    "n_durations": "trial durations",
    "min_duration_days": "min duration (d)",
    "limb_dark_u1": "limb darkening u1",
    "limb_dark_u2": "limb darkening u2",
    "peak_separation_rayleigh": "peak separation (Rayleigh)",
    "alias_freq_tolerance": "alias tolerance (c/d)",
    "harmonic_max": "max harmonic order",
    "mb_model": "multiband model",
    "mb_nterms_base": "shared harmonics",
    "mb_nterms_band": "per-band harmonics",
    "mb_reg_base": "shared ridge",
    "mb_reg_band": "per-band ridge",
    "mb_regularize_by_trace": "ridge scaled by trace",
    "mb_fap_bootstrap": "FAP bootstrap draws",
    "mb_fap_seed": "bootstrap seed",
    "device": "torch device",
    "min_detections": "min points",
    "batch_periods": "trial batch",
    "period_batch": "trial batch",
    "nufft_eps": "NUFFT tolerance",
    "direct_freq_batch": "direct-sum batch",
    "normalization": "amplitude convention",
    "min_separation_rayleigh": "min separation (Rayleigh)",
    "refine": "refinement",
    "refine_bound_rayleigh": "refine bound (Rayleigh)",
    "final_refine": "final polish",
    "max_nfev": "max evaluations",
    "snr_threshold": "S/N threshold",
    "snr_window": "S/N window (c/d)",
    "fap_threshold": "max FAP",
    "min_delta_bic": "min ΔBIC",
    "uncertainty": "estimator",
    "n_resamples": "bootstrap draws",
    "seed": "bootstrap seed",
    "combinations": "find combinations",
    "combination_max_order": "max order",
    "combination_parents": "parents",
    "combination_tolerance_rayleigh": "tolerance (Rayleigh)",
    "combination_sigma": "tolerance (σ)",
}


#: Pre-whitening thresholds that act only while their criterion is ticked in
#: ``stop_criteria`` (the engine tests a criterion only when it is named there, and
#: ``prune`` re-tests S/N). The form greys them out otherwise, so editing one can no
#: longer look like it should change the result.
CRITERION_FIELDS: Final[Mapping[str, tuple[str, ...]]] = {
    "snr": ("snr_threshold", "prune"),
    "fap": ("fap_threshold",),
    "bic": ("min_delta_bic",),
    "amplitude": ("min_amplitude",),
}

#: The criterion each dependent field needs (inverse of :data:`CRITERION_FIELDS`).
_NEEDS_CRITERION: Final[Mapping[str, str]] = {
    name: criterion for criterion, names in CRITERION_FIELDS.items() for name in names
}


def field_label(name: str) -> str:
    """The form label of field ``name`` (with units where they help)."""
    return _LABELS.get(name, name.replace("_", " "))


def field_section(model_cls: type[BaseSettings], name: str) -> str:
    """The form section field ``name`` of ``model_cls`` is shown in."""
    if issubclass(model_cls, PreWhitenSettings) and name in _PREWHITEN_SECTION_OF:
        return _PREWHITEN_SECTION_OF[name]
    return _SECTION_OF.get(name, OTHER)


# -- field descriptors ------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """A Qt-free description of one settings field, enough to build/read a widget."""

    name: str
    kind: FieldKind
    default: Any
    choices: tuple[str, ...] = ()
    lo: float | None = None
    hi: float | None = None
    help: str = dataclass_field(default="")


def _is_optional(annotation: Any) -> bool:
    """Whether ``annotation`` is ``X | None`` (PEP 604 or typing.Optional)."""
    if get_origin(annotation) in (types.UnionType, Union):
        return _NONE_TYPE in get_args(annotation)
    return False


def _optional_inner(annotation: Any) -> Any:
    """The non-None member of an optional annotation."""
    return next(a for a in get_args(annotation) if a is not _NONE_TYPE)


def _variadic_tuple_item(annotation: Any) -> Any | None:
    """``X`` for a ``tuple[X, ...]`` annotation, else None."""
    if get_origin(annotation) is not tuple:
        return None
    args = get_args(annotation)
    if len(args) == 2 and args[1] is Ellipsis:
        return args[0]
    return None


def _as_float(value: Any) -> float:
    """Coerce an annotated-types constraint value (typed as a protocol) to ``float``."""
    return float(value)


def _bounds(field: FieldInfo) -> tuple[float | None, float | None]:
    """Extract ``(lo, hi)`` from a field's annotated-types ge/gt/le/lt constraints."""
    lo: float | None = None
    hi: float | None = None
    for marker in field.metadata:
        if isinstance(marker, at.Ge):
            lo = _as_float(marker.ge)
        elif isinstance(marker, at.Gt):
            lo = _as_float(marker.gt)
        elif isinstance(marker, at.Le):
            hi = _as_float(marker.le)
        elif isinstance(marker, at.Lt):
            hi = _as_float(marker.lt)
    return lo, hi


def field_to_spec(name: str, field: FieldInfo) -> FieldSpec | None:
    """Reduce a pydantic field to a :class:`FieldSpec`, or None if it has no widget.

    ``bool`` is tested before ``int`` because ``bool`` is an ``int`` subclass. A
    ``tuple[Literal[...], ...]`` becomes a ``multi_choice`` (one check box per choice)
    and a ``tuple[float, ...]`` (or of ints) a ``float_list`` (comma-separated text).
    """
    annotation = field.annotation
    help_text = field.description or ""
    default = field.default
    lo, hi = _bounds(field)
    if get_origin(annotation) is Literal:
        choices = tuple(str(a) for a in get_args(annotation))
        return FieldSpec(name, "combo", str(default), choices=choices, help=help_text)
    item = _variadic_tuple_item(annotation)
    if item is not None:
        values = tuple(default) if isinstance(default, (tuple, list)) else ()
        if get_origin(item) is Literal:
            choices = tuple(str(a) for a in get_args(item))
            return FieldSpec(
                name,
                "multi_choice",
                tuple(str(v) for v in values),
                choices=choices,
                help=help_text,
            )
        if item in (float, int):
            return FieldSpec(name, "float_list", values, help=help_text)
        return None
    if _is_optional(annotation):
        inner = _optional_inner(annotation)
        if inner is int:
            return FieldSpec(
                name, "optional_int", default, lo=lo, hi=hi, help=help_text
            )
        if inner is float:
            return FieldSpec(
                name, "optional_float", default, lo=lo, hi=hi, help=help_text
            )
        return None
    if annotation is bool:
        return FieldSpec(name, "bool", bool(default), help=help_text)
    if annotation is int:
        return FieldSpec(name, "int", default, lo=lo, hi=hi, help=help_text)
    if annotation is float:
        return FieldSpec(name, "float", default, lo=lo, hi=hi, help=help_text)
    return None


def _decimals_for(value: Any) -> int:
    """Display decimals for ``value``'s magnitude (handles tiny defaults like 1e-9).

    Used to size the arrow-key step; the value itself is held at full precision.
    """
    try:
        magnitude = abs(float(value))
    except (TypeError, ValueError):
        return 6
    if magnitude == 0.0 or magnitude >= 1.0e-3:
        return 6
    return min(12, max(6, int(math.ceil(-math.log10(magnitude))) + 3))


def _float_step(value: Any, decimals: int) -> float:
    """A sensible single-step for a float spin given its default magnitude."""
    try:
        magnitude = abs(float(value))
    except (TypeError, ValueError):
        magnitude = 0.0
    if magnitude == 0.0:
        return 10.0 ** (-min(decimals, 3))
    if magnitude >= 1.0:
        return max(0.1, magnitude / 10.0)
    return max(10.0 ** (-decimals), magnitude / 10.0)


def shortest_float_text(value: float) -> str:
    """The shortest ``%g`` text that reads back as exactly ``value``.

    ``3.0`` is ``"3"`` and ``1e-9`` is ``"1e-09"`` as before, but ``1234567.5`` stays
    ``"1234567.5"`` instead of a 6-digit ``"1.23457e+06"`` that Qt would re-parse,
    and so silently truncate the value, the next time Return is pressed.
    """
    text = f"{value:g}"
    if not math.isfinite(value):
        return text
    for digits in range(7, 18):
        if float(text) == value:
            break
        text = f"{value:.{digits}g}"
    return text


def _snap_step(value: float, step: float) -> float:
    """``value`` rounded to 12 significant digits of ``max(|value|, |step|)``."""
    scale = max(abs(value), abs(step))
    if scale == 0.0 or not math.isfinite(scale):
        return value
    return round(value, 11 - math.floor(math.log10(scale)))


class _GSpinBox(QtWidgets.QDoubleSpinBox):
    """A ``QDoubleSpinBox`` rendering a compact ``%g`` instead of fixed decimals.

    The stock spinbox always shows :meth:`decimals` places, so a peak-separation
    default of ``3.0`` renders as ``"3.000000"`` and an eps of ``1e-9`` renders as
    ``"0.000000001000"``. Overriding ``textFromValue``/``valueFromText`` keeps the
    underlying value/precision machinery (range, step) intact while displaying a
    compact string (scientific notation for tiny/huge values) that always round-trips:
    pressing Return re-parses the displayed text, so a display that dropped digits
    used to replace the held value. ``validate`` is intentionally permissive so typing
    (e.g. ``"1e-"`` while entering ``"1e-9"``) is never blocked mid-edit.
    """

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt override
        return shortest_float_text(value)

    def stepBy(self, steps: int) -> None:  # noqa: N802 - Qt override
        """Step like the stock box, without the binary noise of a repeated sum.

        The value is held at full precision (:data:`_HELD_DECIMALS`), so stepping
        1.1 up by 0.1 read ``1.2000000000000002``. The sum is rounded to 12
        significant digits of the larger of value and step: far below anything a
        person types, far above the noise.
        """
        self.interpretText()  # step from a typed, not yet committed value, as Qt does
        step = self.singleStep()
        self.setValue(_snap_step(self.value() + steps * step, step))
        style = self.style()
        hint = QtWidgets.QStyle.StyleHint.SH_SpinBox_SelectOnStep
        if style is not None and style.styleHint(hint, None, self, None):
            self.selectAll()

    def valueFromText(self, text: str) -> float:  # noqa: N802 - Qt override
        try:
            return float(text)
        except ValueError:
            return 0.0

    def validate(  # noqa: N802 - Qt override
        self, text: str, pos: int
    ) -> tuple[QtGui.QValidator.State, str, int]:
        if text.strip() in ("", "-", "+"):
            return QtGui.QValidator.State.Intermediate, text, pos
        try:
            float(text)
        except ValueError:
            return QtGui.QValidator.State.Intermediate, text, pos
        return QtGui.QValidator.State.Acceptable, text, pos


def _configure_float_spin(
    spin: QtWidgets.QDoubleSpinBox, spec: FieldSpec, effective_default: Any
) -> None:
    spin.setDecimals(_HELD_DECIMALS)
    spin.setRange(
        spec.lo if spec.lo is not None else -1.0e12,
        spec.hi if spec.hi is not None else 1.0e12,
    )
    spin.setSingleStep(_float_step(effective_default, _decimals_for(effective_default)))


def _configure_int_spin(spin: QtWidgets.QSpinBox, spec: FieldSpec) -> None:
    spin.setRange(
        int(spec.lo) if spec.lo is not None else -10_000_000,
        int(spec.hi) if spec.hi is not None else 10_000_000,
    )


def _describe(widget: QtWidgets.QWidget, name: str, help_text: str) -> None:
    """Give a widget its accessible name and keyboard-reachable help.

    Tooltips need a mouse hover; ``whatsThis`` (Shift+F1) and ``statusTip`` (the
    status bar, on focus) reach keyboard and screen-reader users too.
    """
    widget.setAccessibleName(name)
    widget.setAccessibleDescription(help_text)
    widget.setToolTip(help_text)
    widget.setWhatsThis(help_text)
    widget.setStatusTip(help_text)


class _OptionalSpin(QtWidgets.QWidget):
    """An ``auto`` checkbox + a spin: contributes ``None`` when ``auto`` is checked.

    While ``auto`` is ticked the spin shows the value the run will actually use
    (:meth:`set_auto_value`). It is read-only rather than disabled: the value is
    information to read, and a disabled box is drawn in low-contrast text and skipped
    by keyboard focus and screen readers. The dynamic property ``auto`` lets a style
    sheet tint it; the italic font marks it without one.
    """

    changed = Signal()

    def __init__(
        self,
        spec: FieldSpec,
        value: Any,
        label: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._is_float = spec.kind == "optional_float"
        self._auto_value: float | None = None
        label = label or field_label(spec.name)
        help_text = spec.help or label
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._auto = QtWidgets.QCheckBox("auto")
        spin: QtWidgets.QAbstractSpinBox
        if self._is_float:
            fspin = _GSpinBox()
            # Only sizes the arrow-key step. A 0 lower bound (frequency limits, ridge
            # strengths, amplitude floors) would shrink the step to 0.001.
            fallback = value if value is not None else (spec.lo or 1.0)
            _configure_float_spin(fspin, spec, fallback)
            if value is not None:
                fspin.setValue(float(value))
            fspin.valueChanged.connect(lambda *_: self.changed.emit())
            spin = fspin
        else:
            ispin = QtWidgets.QSpinBox()
            _configure_int_spin(ispin, spec)
            if value is not None:
                ispin.setValue(int(value))
            ispin.valueChanged.connect(lambda *_: self.changed.emit())
            spin = ispin
        self._spin = spin
        self._label = label
        self.describe(help_text)
        layout.addWidget(self._auto)
        layout.addWidget(self._spin, 1)
        self.setFocusProxy(self._auto)

        is_none = value is None
        self._auto.setChecked(is_none)
        self._apply_auto(is_none)
        self._auto.toggled.connect(self._on_auto)
        self._auto.toggled.connect(lambda *_: self.changed.emit())

    def describe(self, help_text: str) -> None:
        """Name both halves and give each the field's help (tooltip, status tip,
        *What's This?*); the group itself is named after the field too."""
        label = self._label
        _describe(
            self._auto,
            f"{label}: automatic",
            f"{help_text}\nTicked: chosen automatically (the value shown is what "
            "the run uses). Untick to set it yourself.",
        )
        _describe(self._spin, label, help_text)
        self.setAccessibleName(label)
        self.setToolTip(help_text)
        self.setWhatsThis(help_text)
        self.setStatusTip(help_text)

    def _apply_auto(self, auto: bool) -> None:
        """Make the spin read-only (not disabled) while ``auto`` is ticked."""
        spin = self._spin
        spin.setReadOnly(auto)
        spin.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
            if auto
            else QtWidgets.QAbstractSpinBox.ButtonSymbols.UpDownArrows
        )
        spin.setProperty("auto", auto)
        font = spin.font()
        font.setItalic(auto)
        spin.setFont(font)
        style = spin.style()
        if style is not None:  # re-evaluate [auto="true"] style-sheet rules
            style.unpolish(spin)
            style.polish(spin)

    def _on_auto(self, checked: bool) -> None:
        self._apply_auto(checked)
        if checked and self._auto_value is not None:
            self._show(self._auto_value)

    def _show(self, value: float) -> None:
        self._spin.blockSignals(True)
        if self._is_float:
            self._spin.setValue(float(value))  # type: ignore[attr-defined,arg-type]
        else:
            self._spin.setValue(int(value))  # type: ignore[attr-defined,arg-type]
        self._spin.blockSignals(False)

    def set_auto_value(self, value: float | None) -> None:
        """Display ``value`` in the (read-only) spin while ``auto`` is checked.

        Shown to six significant digits: the full float ("0.04068016549568509")
        did not fit the field, and the run uses its own exact value anyway.
        """
        if value is not None and math.isfinite(value):
            value = float(f"{value:.6g}")
        self._auto_value = value
        if self._auto.isChecked() and value is not None:
            self._show(value)

    def value(self) -> float | int | None:
        """The current value, or ``None`` when ``auto`` is checked."""
        if self._auto.isChecked():
            return None
        if self._is_float:
            return float(self._spin.value())  # type: ignore[attr-defined]
        return int(self._spin.value())  # type: ignore[attr-defined]

    def set_value(self, value: float | int | None) -> None:
        """Set the value; ``None`` ticks ``auto``."""
        if value is None:
            self._auto.setChecked(True)
            return
        self._show(value)
        self._auto.setChecked(False)


class _MultiChoice(QtWidgets.QWidget):
    """One check box per choice of a ``tuple[Literal[...], ...]`` field.

    Reads back the ticked choices in the choices' own order (an empty tuple is a valid
    value, e.g. pre-whitening that runs to its component cap). The boxes sit in a grid
    of :attr:`COLUMNS` columns: a single row of four is wider than the options dock.
    """

    changed = Signal()

    #: Check boxes per grid row.
    COLUMNS: Final = 2

    def __init__(
        self,
        spec: FieldSpec,
        value: Iterable[str],
        label: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(2)
        help_text = spec.help or label
        selected = {str(v) for v in value}
        self._boxes: dict[str, QtWidgets.QCheckBox] = {}
        for position, choice in enumerate(spec.choices):
            box = QtWidgets.QCheckBox(choice)
            box.setObjectName(f"{spec.name}:{choice}")
            box.setChecked(choice in selected)
            _describe(box, f"{label}: {choice}", help_text)
            box.toggled.connect(lambda *_: self.changed.emit())
            row, column = divmod(position, self.COLUMNS)
            layout.addWidget(box, row, column)
            self._boxes[choice] = box
        layout.setColumnStretch(self.COLUMNS, 1)
        # The group is named too, so a screen reader announces what the boxes choose.
        self.setAccessibleName(label)
        self.setWhatsThis(help_text)
        self.setStatusTip(help_text)
        if self._boxes:
            self.setFocusProxy(next(iter(self._boxes.values())))

    def value(self) -> tuple[str, ...]:
        """The ticked choices, in declaration order."""
        return tuple(name for name, box in self._boxes.items() if box.isChecked())

    def set_value(self, value: Iterable[str]) -> None:
        selected = {str(v) for v in value}
        for name, box in self._boxes.items():
            box.setChecked(name in selected)


def _float_list_text(values: Iterable[Any]) -> str:
    return ", ".join(
        shortest_float_text(float(v)) if isinstance(v, (int, float)) else str(v)
        for v in values
    )


def _split_list(text: str) -> list[str]:
    """Comma-separated entries, handed to pydantic as text (it validates/coerces)."""
    return [part.strip() for part in text.split(",") if part.strip()]


class _Section(QtWidgets.QWidget):
    """A titled, collapsible group of form rows."""

    def __init__(
        self, title: str, *, expanded: bool, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.title = title
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._toggle = QtWidgets.QToolButton()
        self._toggle.setObjectName("section_toggle")
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setAutoRaise(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        _describe(
            self._toggle,
            f"{title} options",
            f"Show or hide the {title.lower()} options (Space toggles).",
        )
        layout.addWidget(self._toggle)
        self._body = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self._body)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        # A field moves under its label when both do not fit side by side, so a long
        # label plus an 'auto' spin never forces a horizontal scrollbar on the
        # narrow options dock (which clipped every field's right edge).
        self.form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.form.setContentsMargins(12, 2, 4, 6)
        self.form.setSpacing(8)
        layout.addWidget(self._body)
        self._toggle.toggled.connect(self.set_expanded)
        self._toggle.setChecked(expanded)
        self.set_expanded(expanded)

    @property
    def expanded(self) -> bool:
        return self._toggle.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        if self._toggle.isChecked() != expanded:
            self._toggle.setChecked(expanded)  # re-enters via ``toggled``
            return
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._body.setVisible(expanded)


class PydanticSettingsForm(QtWidgets.QWidget):
    """A form built from a pydantic settings model; read back via :meth:`build`.

    Fields are grouped into the sections of :data:`SECTION_ORDER`; each field label is
    the buddy of its widget, every input has an accessible name, and the model's field
    description is its tooltip, status tip and *What's This?* text.
    """

    changed = Signal()

    def __init__(
        self,
        model_cls: type[BaseSettings],
        *,
        initial: BaseSettings | None = None,
        exclude: frozenset[str] = GUI_OWNED_FIELDS,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model_cls = model_cls
        self._readers: dict[str, Callable[[], Any]] = {}
        self._writers: dict[str, Callable[[Any], None]] = {}
        self._widgets: dict[str, QtWidgets.QWidget] = {}
        self._optional: dict[str, _OptionalSpin] = {}
        self._sections: dict[str, _Section] = {}
        self._hidden_sections: set[str] = set()
        self._section_of: dict[str, str] = {}
        self._labels: dict[str, QtWidgets.QLabel] = {}
        self._quiet = False
        seed = initial if initial is not None else gui_default_settings(model_cls)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        rows: dict[str, list[tuple[str, FieldSpec]]] = {}
        for name, info in model_cls.model_fields.items():
            if name in exclude:
                continue
            spec = field_to_spec(name, info)
            if spec is None:
                continue
            rows.setdefault(field_section(model_cls, name), []).append((name, spec))
        order = [s for s in SECTION_ORDER if s in rows]
        order += [s for s in rows if s not in order]
        for title in order:
            section = _Section(title, expanded=title not in COLLAPSED_SECTIONS)
            self._sections[title] = section
            layout.addWidget(section)
            for name, spec in rows[title]:
                self._add_row(section, name, spec, getattr(seed, name, spec.default))

        reset = QtWidgets.QPushButton("Reset to defaults")
        reset.setObjectName("reset_defaults")
        _describe(
            reset,
            "Reset options to defaults",
            "Put every option back to its default (environment overrides included).",
        )
        reset.clicked.connect(self.reset_to_defaults)
        layout.addWidget(reset, 0, Qt.AlignmentFlag.AlignLeft)
        # The focused option's description, in words: tooltips need a mouse hover and
        # What's This a key chord, so a keyboard user tabbing through the form reads
        # each option's purpose here (A11Y-18).
        self._help = QtWidgets.QLabel(HELP_PLACEHOLDER)
        self._help.setObjectName("field_help")
        self._help.setWordWrap(True)
        self._help.setTextFormat(Qt.TextFormat.PlainText)
        self._help.setAccessibleName("Option help")
        self._help.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        layout.addWidget(self._help)
        layout.addStretch(1)
        app = QtWidgets.QApplication.instance()
        if isinstance(app, QtWidgets.QApplication):
            app.focusChanged.connect(self._on_focus_changed)

        criteria = self._widgets.get("stop_criteria")
        if isinstance(criteria, _MultiChoice):
            criteria.changed.connect(self._sync_criteria)
            self._sync_criteria()

    def _add_row(
        self, section: _Section, name: str, spec: FieldSpec, value: Any
    ) -> None:
        text = field_label(name)
        widget, reader, writer = self._build_widget(spec, value, text)
        self._readers[name] = reader
        self._writers[name] = writer
        self._widgets[name] = widget
        self._section_of[name] = section.title
        help_text = spec.help or text
        criterion = _NEEDS_CRITERION.get(name)
        if criterion is not None and "stop_criteria" in self._model_cls.model_fields:
            help_text += f"\n(Used only while stop criterion '{criterion}' is ticked.)"
        if isinstance(widget, _OptionalSpin):
            self._optional[name] = widget
            # Again, now with the stop-criterion note: both halves carry it.
            widget.describe(help_text)
        elif not isinstance(widget, _MultiChoice):
            _describe(widget, text, help_text)
        label = QtWidgets.QLabel(text)
        label.setObjectName("field_label")
        label.setToolTip(help_text)
        label.setBuddy(widget)
        self._labels[name] = label
        section.form.addRow(label, widget)

    def _on_focus_changed(
        self, _old: QtWidgets.QWidget | None, new: QtWidgets.QWidget | None
    ) -> None:
        """Show the description of the option that just took the focus."""
        if new is None or not self.isAncestorOf(new):
            return  # focus left the form: the last description stays readable
        widget: QtWidgets.QWidget | None = new
        while widget is not None and widget is not self:
            text = widget.whatsThis()
            if text:
                self._help.setText(text)
                return
            widget = widget.parentWidget()

    @property
    def help_text(self) -> str:
        """What the help line under the form says (the focused option's purpose)."""
        return self._help.text()

    def _sync_criteria(self) -> None:
        """Enable each stop-criterion threshold only while its criterion is ticked."""
        reader = self._readers.get("stop_criteria")
        if reader is None:
            return
        active = set(reader())
        for name, criterion in _NEEDS_CRITERION.items():
            widget = self._widgets.get(name)
            if widget is None:
                continue
            widget.setEnabled(criterion in active)
            self._labels[name].setEnabled(criterion in active)

    def _emit(self, *_: Any) -> None:
        if not self._quiet:
            self.changed.emit()

    def _build_widget(
        self, spec: FieldSpec, value: Any, label: str
    ) -> tuple[QtWidgets.QWidget, Callable[[], Any], Callable[[Any], None]]:
        if spec.kind == "combo":
            combo = QtWidgets.QComboBox()
            combo.addItems(list(spec.choices))
            combo.setCurrentText(str(value))
            combo.currentTextChanged.connect(self._emit)
            return combo, combo.currentText, lambda v: combo.setCurrentText(str(v))
        if spec.kind == "bool":
            check = QtWidgets.QCheckBox()
            check.setChecked(bool(value))
            check.toggled.connect(self._emit)
            return check, check.isChecked, lambda v: check.setChecked(bool(v))
        if spec.kind == "int":
            ispin = QtWidgets.QSpinBox()
            _configure_int_spin(ispin, spec)
            ispin.setValue(int(value))
            ispin.valueChanged.connect(self._emit)
            return ispin, ispin.value, lambda v: ispin.setValue(int(v))
        if spec.kind == "float":
            fspin = _GSpinBox()
            _configure_float_spin(fspin, spec, value)
            fspin.setValue(float(value))
            fspin.valueChanged.connect(self._emit)
            return fspin, fspin.value, lambda v: fspin.setValue(float(v))
        if spec.kind == "multi_choice":
            multi = _MultiChoice(spec, value or (), label)
            multi.changed.connect(self._emit)
            return multi, multi.value, multi.set_value
        if spec.kind == "float_list":
            edit = QtWidgets.QLineEdit(_float_list_text(value or ()))
            edit.setObjectName(spec.name)
            edit.setPlaceholderText("comma-separated numbers")
            edit.textChanged.connect(self._emit)
            return (
                edit,
                lambda: _split_list(edit.text()),
                lambda v: edit.setText(_float_list_text(v or ())),
            )
        # optional_int / optional_float
        opt = _OptionalSpin(spec, value, label)
        opt.changed.connect(self._emit)
        return opt, opt.value, opt.set_value

    # -- reading -----------------------------------------------------------------
    def values(self) -> dict[str, Any]:
        """The current widget values as a plain dict (excluded fields omitted)."""
        return {name: reader() for name, reader in self._readers.items()}

    def value_of(self, name: str) -> Any:
        """Current value of one field (None if excluded or an unchecked optional)."""
        reader = self._readers.get(name)
        return reader() if reader is not None else None

    def build(self) -> BaseSettings:
        """Construct and validate the settings object (raises ``ValidationError``)."""
        return self._model_cls(**self.values())

    # -- writing -----------------------------------------------------------------
    def set_auto_value(self, name: str, value: float | None) -> None:
        """Fill an optional field's read-only 'auto' spin with its effective value."""
        widget = self._optional.get(name)
        if widget is not None:
            widget.set_auto_value(value)

    def reset_to_defaults(self) -> None:
        """Put every field back to the GUI default (one ``changed`` emission)."""
        defaults = gui_default_settings(self._model_cls)
        self._quiet = True
        try:
            for name, writer in self._writers.items():
                writer(getattr(defaults, name))
        finally:
            self._quiet = False
        self.changed.emit()

    # -- layout ------------------------------------------------------------------
    def section_names(self) -> list[str]:
        """Titles of the sections on show (not hidden), in display order."""
        return [t for t in self._sections if t not in self._hidden_sections]

    def section_of(self, name: str) -> str | None:
        """The section field ``name`` is in (None if the form has no such field)."""
        return self._section_of.get(name)

    def field_widget(self, name: str) -> QtWidgets.QWidget | None:
        """The input widget of field ``name`` (None if the form has no such field)."""
        return self._widgets.get(name)

    def set_section_expanded(self, title: str, expanded: bool) -> None:
        """Expand or collapse section ``title`` (no-op if absent)."""
        section = self._sections.get(title)
        if section is not None:
            section.set_expanded(expanded)

    def set_section_visible(self, title: str, visible: bool) -> None:
        """Show or hide section ``title`` entirely (its values still build)."""
        section = self._sections.get(title)
        if section is None:
            return
        section.setVisible(visible)
        if visible:
            self._hidden_sections.discard(title)
        else:
            self._hidden_sections.add(title)

    def set_multiband(self, is_multiband: bool) -> None:
        """Show the multiband options only while a multiband curve is analysed.

        They are inert for a single-band curve; hidden, they still build with
        their current values.
        """
        self.set_section_visible(MULTIBAND, is_multiband)


__all__ = [
    "ADVANCED",
    "COLLAPSED_SECTIONS",
    "CRITERION_FIELDS",
    "GUI_OWNED_FIELDS",
    "HELP_PLACEHOLDER",
    "SECTION_ORDER",
    "FieldKind",
    "FieldSpec",
    "PydanticSettingsForm",
    "field_label",
    "field_section",
    "field_to_spec",
    "shortest_float_text",
]
