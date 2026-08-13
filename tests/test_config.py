"""Settings models: cross-field bound validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import cuperiod as cup


def test_defaults_construct() -> None:
    # Every settings model must build with no arguments.
    for cls in (
        cup.GLSSettings,
        cup.BLSSettings,
        cup.PDMSettings,
        cup.CESettings,
        cup.MHAOVSettings,
        cup.StringLengthSettings,
        cup.SuperSmootherSettings,
        cup.TLSSettings,
        cup.BatchSettings,
    ):
        cls()


def test_frequency_bounds_validated() -> None:
    for cls in (cup.GLSSettings, cup.PDMSettings, cup.CESettings,
                cup.MHAOVSettings, cup.StringLengthSettings,
                cup.SuperSmootherSettings):
        with pytest.raises(ValidationError, match="must be <"):
            cls(minimum_frequency=10.0, maximum_frequency=1.0)
        # a valid ordering is accepted
        cls(minimum_frequency=0.01, maximum_frequency=5.0)


def test_period_bounds_validated() -> None:
    for cls in (cup.BLSSettings, cup.TLSSettings):
        with pytest.raises(ValidationError, match="must be <"):
            cls(min_period_days=100.0, max_period_days=1.0)


def test_duration_fraction_bounds_validated() -> None:
    for cls in (cup.BLSSettings, cup.TLSSettings):
        with pytest.raises(ValidationError, match="must be <"):
            cls(duration_min_frac=0.5, duration_max_frac=0.01)
