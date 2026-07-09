"""Tests for the pure ``field_to_spec`` descriptor over every settings model."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings
from pytestqt.qtbot import QtBot

from cuperiod.core.config import (
    BLSSettings,
    CESettings,
    GLSSettings,
    MHAOVSettings,
    PDMSettings,
    StringLengthSettings,
    TLSSettings,
)
from cuperiod.gui.qt import QtGui, QtWidgets
from cuperiod.gui.settingsform import (
    FieldSpec,
    PydanticSettingsForm,
    _configure_int_spin,
    _GSpinBox,
    field_to_spec,
)

ALL_SETTINGS: list[type[BaseSettings]] = [
    BLSSettings,
    CESettings,
    GLSSettings,
    MHAOVSettings,
    PDMSettings,
    StringLengthSettings,
    TLSSettings,
]

_KINDS = {"combo", "bool", "int", "float", "optional_int", "optional_float"}


@pytest.mark.parametrize("model_cls", ALL_SETTINGS)
def test_every_field_maps_or_skips(model_cls: type[BaseSettings]) -> None:
    for name, info in model_cls.model_fields.items():
        spec = field_to_spec(name, info)
        assert spec is None or spec.kind in _KINDS


def test_gls_specific_fields() -> None:
    fields = GLSSettings.model_fields
    backend = field_to_spec("backend", fields["backend"])
    assert backend is not None and backend.kind == "combo"
    assert "auto" in backend.choices

    fit_mean = field_to_spec("fit_mean", fields["fit_mean"])
    assert fit_mean is not None and fit_mean.kind == "bool"

    spp = field_to_spec("samples_per_peak", fields["samples_per_peak"])
    assert spp is not None and spp.kind == "int"
    assert spp.lo == 1.0

    min_freq = field_to_spec("minimum_frequency", fields["minimum_frequency"])
    assert min_freq is not None and min_freq.kind == "optional_float"
    assert min_freq.default is None


def test_literal_choices_match_get_args() -> None:
    field = GLSSettings.model_fields["fap_method"]
    spec = field_to_spec("fap_method", field)
    assert spec is not None
    assert set(spec.choices) == {str(a) for a in get_args(field.annotation)}


@pytest.mark.parametrize("model_cls", ALL_SETTINGS)
def test_defaults_build_valid(model_cls: type[BaseSettings]) -> None:
    assert model_cls() is not None


def test_bad_bounds_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        GLSSettings(minimum_frequency=10.0, maximum_frequency=1.0)


# -- float spinbox display formatting (%g instead of fixed decimals) ---------------


def test_gspinbox_formats_compactly(qtbot: QtBot) -> None:
    spin = _GSpinBox()
    qtbot.addWidget(spin)
    spin.setDecimals(12)
    spin.setRange(-1.0e12, 1.0e12)
    spin.setValue(3.0)
    assert spin.textFromValue(spin.value()) == "3"
    spin.setValue(1e-9)
    assert spin.textFromValue(spin.value()) == "1e-09"
    assert isinstance(spin, QtWidgets.QDoubleSpinBox)


def test_gspinbox_valuefromtext_parses_scientific_and_plain() -> None:
    spin = _GSpinBox()
    assert spin.valueFromText("1e-9") == pytest.approx(1e-9)
    assert spin.valueFromText("3") == pytest.approx(3.0)
    assert spin.valueFromText("not-a-number") == 0.0  # never raises


def test_gspinbox_validate_never_blocks_partial_scientific_input() -> None:
    spin = _GSpinBox()
    for text in ("", "-", "1e-", "1e", "1.", "3", "1e-09"):
        state, _, _ = spin.validate(text, len(text))
        assert state != QtGui.QValidator.State.Invalid


def test_float_spin_uses_compact_g_formatting_for_gls_defaults(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(GLSSettings)
    qtbot.addWidget(form)
    sep_spin = form._readers["peak_separation_rayleigh"].__self__  # type: ignore[attr-defined]
    assert isinstance(sep_spin, _GSpinBox)
    assert sep_spin.text() == "3"  # not "3.000000"

    eps_spin = form._readers["nufft_eps"].__self__  # type: ignore[attr-defined]
    assert isinstance(eps_spin, _GSpinBox)
    assert eps_spin.text() == "1e-09"  # not "0.000000001000"


@pytest.mark.parametrize("model_cls", ALL_SETTINGS)
def test_form_float_values_roundtrip_defaults(
    model_cls: type[BaseSettings], qtbot: QtBot
) -> None:
    """Every default float field, read back through its (now %g-formatted) spin box,
    still equals the pydantic default — the display format change must not lose
    precision or corrupt the round-trip.
    """
    defaults = model_cls()
    form = PydanticSettingsForm(model_cls, initial=defaults)
    qtbot.addWidget(form)
    built = form.build()
    for name, info in model_cls.model_fields.items():
        spec = field_to_spec(name, info)
        if spec is None or spec.kind not in ("float", "optional_float"):
            continue
        got = getattr(built, name)
        want = getattr(defaults, name)
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want, rel=1e-9)


# -- int spinbox fallback bounds -----------------------------------------------------


def test_configure_int_spin_negative_fallback_when_lo_none(qtbot: QtBot) -> None:
    spin = QtWidgets.QSpinBox()
    qtbot.addWidget(spin)
    spec = FieldSpec(name="x", kind="int", default=0, lo=None, hi=None)
    _configure_int_spin(spin, spec)
    assert spin.minimum() == -10_000_000
    assert spin.maximum() == 10_000_000
