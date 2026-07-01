"""Tests for the pure ``field_to_spec`` descriptor over every settings model."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from cuperiod.core.config import (
    BLSSettings,
    CESettings,
    GLSSettings,
    MHAOVSettings,
    PDMSettings,
    StringLengthSettings,
    TLSSettings,
)
from cuperiod.gui.settingsform import field_to_spec

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
