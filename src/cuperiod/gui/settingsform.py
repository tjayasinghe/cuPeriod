"""Auto-generate a settings form from a pydantic model.

Each periodogram method has a pydantic settings model (``GLSSettings``, ``BLSSettings``,
...). Rather than hand-write a form per method, :func:`field_to_spec` reduces each model
field to a small, Qt-free :class:`FieldSpec` (kind, choices, bounds, default), and
:class:`PydanticSettingsForm` turns those into labelled widgets, read back into a
validated settings object. The descriptor step is pure, so it is unit-tested per method.

The ``backend`` field is excluded by default: it is owned by a dedicated picker (the
``backend=`` argument to :func:`cuperiod.periodogram` overrides ``settings.backend``).
``device`` and ``precision`` stay in the form so every other knob remains exposed.
"""

from __future__ import annotations

import math
import types
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal, Union, get_args, get_origin

import annotated_types as at
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from cuperiod.gui.qt import Qt, QtWidgets, Signal

_NONE_TYPE = type(None)

FieldKind = Literal["combo", "bool", "int", "float", "optional_int", "optional_float"]


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

    ``bool`` is tested before ``int`` because ``bool`` is an ``int`` subclass.
    """
    annotation = field.annotation
    help_text = field.description or ""
    default = field.default
    lo, hi = _bounds(field)
    if get_origin(annotation) is Literal:
        choices = tuple(str(a) for a in get_args(annotation))
        return FieldSpec(name, "combo", str(default), choices=choices, help=help_text)
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
    """Spinbox decimals enough to show ``value`` (handles tiny defaults like 1e-9)."""
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


def _configure_float_spin(
    spin: QtWidgets.QDoubleSpinBox, spec: FieldSpec, effective_default: Any
) -> None:
    decimals = _decimals_for(effective_default)
    spin.setDecimals(decimals)
    spin.setRange(
        spec.lo if spec.lo is not None else -1.0e12,
        spec.hi if spec.hi is not None else 1.0e12,
    )
    spin.setSingleStep(_float_step(effective_default, decimals))


def _configure_int_spin(spin: QtWidgets.QSpinBox, spec: FieldSpec) -> None:
    spin.setRange(
        int(spec.lo) if spec.lo is not None else 0,
        int(spec.hi) if spec.hi is not None else 10_000_000,
    )


class _OptionalSpin(QtWidgets.QWidget):
    """An ``auto`` checkbox + a spin: contributes ``None`` when ``auto`` is checked."""

    changed = Signal()

    def __init__(
        self, spec: FieldSpec, value: Any, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._is_float = spec.kind == "optional_float"
        self._auto_value: float | None = None
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._auto = QtWidgets.QCheckBox("auto")
        spin: QtWidgets.QAbstractSpinBox
        if self._is_float:
            fspin = QtWidgets.QDoubleSpinBox()
            fallback = value if value is not None else spec.lo
            if fallback is None:
                fallback = 1.0
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
        layout.addWidget(self._auto)
        layout.addWidget(self._spin, 1)

        is_none = value is None
        self._auto.setChecked(is_none)
        self._spin.setEnabled(not is_none)
        self._auto.toggled.connect(self._on_auto)
        self._auto.toggled.connect(lambda *_: self.changed.emit())

    def _on_auto(self, checked: bool) -> None:
        self._spin.setEnabled(not checked)
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
        """Display ``value`` in the (disabled) spin while ``auto`` is checked."""
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


class PydanticSettingsForm(QtWidgets.QWidget):
    """A form built from a pydantic settings model; read back via :meth:`build`."""

    changed = Signal()

    def __init__(
        self,
        model_cls: type[BaseSettings],
        *,
        initial: BaseSettings | None = None,
        exclude: frozenset[str] = frozenset({"backend"}),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model_cls = model_cls
        self._readers: dict[str, Callable[[], Any]] = {}
        self._optional: dict[str, _OptionalSpin] = {}

        layout = QtWidgets.QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        for name, info in model_cls.model_fields.items():
            if name in exclude:
                continue
            spec = field_to_spec(name, info)
            if spec is None:
                continue
            value = getattr(initial, name) if initial is not None else spec.default
            widget, reader = self._build_widget(spec, value)
            self._readers[name] = reader
            if isinstance(widget, _OptionalSpin):
                self._optional[name] = widget
            label = QtWidgets.QLabel(name.replace("_", " "))
            tip = spec.help or name.replace("_", " ")
            label.setToolTip(tip)
            widget.setToolTip(tip)
            layout.addRow(label, widget)

    def _emit(self, *_: Any) -> None:
        self.changed.emit()

    def _build_widget(
        self, spec: FieldSpec, value: Any
    ) -> tuple[QtWidgets.QWidget, Callable[[], Any]]:
        if spec.kind == "combo":
            combo = QtWidgets.QComboBox()
            combo.addItems(list(spec.choices))
            combo.setCurrentText(str(value))
            combo.currentTextChanged.connect(self._emit)
            return combo, combo.currentText
        if spec.kind == "bool":
            check = QtWidgets.QCheckBox()
            check.setChecked(bool(value))
            check.toggled.connect(self._emit)
            return check, check.isChecked
        if spec.kind == "int":
            ispin = QtWidgets.QSpinBox()
            _configure_int_spin(ispin, spec)
            ispin.setValue(int(value))
            ispin.valueChanged.connect(self._emit)
            return ispin, ispin.value
        if spec.kind == "float":
            fspin = QtWidgets.QDoubleSpinBox()
            _configure_float_spin(fspin, spec, value)
            fspin.setValue(float(value))
            fspin.valueChanged.connect(self._emit)
            return fspin, fspin.value
        # optional_int / optional_float
        opt = _OptionalSpin(spec, value)
        opt.changed.connect(self._emit)
        return opt, opt.value

    def values(self) -> dict[str, Any]:
        """The current widget values as a plain dict (excluded fields omitted)."""
        return {name: reader() for name, reader in self._readers.items()}

    def value_of(self, name: str) -> Any:
        """Current value of one field (None if excluded or an unchecked optional)."""
        reader = self._readers.get(name)
        return reader() if reader is not None else None

    def set_auto_value(self, name: str, value: float | None) -> None:
        """Fill an optional field's greyed 'auto' spin with its effective value."""
        widget = self._optional.get(name)
        if widget is not None:
            widget.set_auto_value(value)

    def build(self) -> BaseSettings:
        """Construct and validate the settings object (raises ``ValidationError``)."""
        return self._model_cls(**self.values())


__all__ = ["FieldKind", "FieldSpec", "PydanticSettingsForm", "field_to_spec"]
