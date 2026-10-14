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


def test_defaults_construct_for_every_exported_model() -> None:
    from cuperiod.core import config

    for name in config.__all__:
        if name.endswith("Settings"):
            getattr(config, name)()


# --- cuperiod.core re-exports (CORE-A-21) ---------------------------------------------


def test_core_reexports_every_settings_model() -> None:
    import cuperiod.core as core
    from cuperiod.core import config

    for name in config.__all__:
        if name.endswith("Settings"):
            assert name in core.__all__, name
            assert getattr(core, name) is getattr(config, name)
    from cuperiod.core import (  # noqa: F401 - the import itself is the test
        PdotSettings,
        PreWhitenSettings,
        SpacingSettings,
        SuperSmootherSettings,
    )


# --- value bounds (CORE-A-13, METH-A-17) ---------------------------------------------

_NAN, _INF = float("nan"), float("inf")

_FREQUENCY_MODELS = (
    cup.GLSSettings,
    cup.PDMSettings,
    cup.CESettings,
    cup.MHAOVSettings,
    cup.StringLengthSettings,
    cup.SuperSmootherSettings,
    cup.PreWhitenSettings,
)


@pytest.mark.parametrize("cls", _FREQUENCY_MODELS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("field", ["minimum_frequency", "maximum_frequency"])
@pytest.mark.parametrize("bad", [-3.0, _NAN, _INF, -_INF])
def test_frequency_bounds_reject_negative_and_non_finite(
    cls: type, field: str, bad: float
) -> None:
    with pytest.raises(ValidationError):
        cls(**{field: bad})


@pytest.mark.parametrize("cls", _FREQUENCY_MODELS[:-1], ids=lambda c: c.__name__)
def test_zero_minimum_frequency_stays_legal(cls: type) -> None:
    # 0 means "auto" for the fold methods and f=0 is a valid GLS grid start; both
    # worked before and must keep working.
    assert cls(minimum_frequency=0.0).minimum_frequency == 0.0


def test_prewhiten_minimum_frequency_must_be_positive() -> None:
    # f = 0 is not a pulsation: it used to be extracted as a bogus component.
    with pytest.raises(ValidationError):
        cup.PreWhitenSettings(minimum_frequency=0.0)


def test_negative_frequency_from_the_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUPERIOD_GLS_MINIMUM_FREQUENCY", "-3")
    with pytest.raises(ValidationError):
        cup.GLSSettings()


@pytest.mark.parametrize("cls", [cup.BLSSettings, cup.TLSSettings])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_max_frac": 1.0},
        {"duration_max_frac": 1.5},
        {"duration_min_frac": 1.2, "duration_max_frac": 2.0},
        {"duration_max_frac": _NAN},
    ],
)
def test_duration_fractions_must_be_below_one(
    cls: type, kwargs: dict[str, float]
) -> None:
    # A box at least as long as the period is meaningless; BLS used to report a
    # wrong best period on numba/numpy while astropy raised.
    with pytest.raises(ValidationError):
        cls(**kwargs)
    cls(duration_max_frac=0.5)


def test_period_cap_may_stay_infinite() -> None:
    assert cup.BLSSettings(max_period_days=_INF).max_period_days == _INF
    assert cup.TLSSettings(max_period_days=_INF).max_period_days == _INF


@pytest.mark.parametrize("field", ["mb_reg_base", "mb_reg_band"])
@pytest.mark.parametrize("bad", [-1.0, _NAN, _INF])
def test_flex_ridges_must_be_finite_and_non_negative(field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        cup.GLSSettings(**{field: bad})
    assert getattr(cup.GLSSettings(**{field: 0.0}), field) == 0.0
    assert getattr(cup.GLSSettings(**{field: None}), field) is None


@pytest.mark.parametrize("bad", [-0.1, _NAN, _INF])
def test_min_amplitude_must_be_finite_and_non_negative(bad: float) -> None:
    with pytest.raises(ValidationError):
        cup.PreWhitenSettings(min_amplitude=bad)
    assert cup.PreWhitenSettings(min_amplitude=0.0).min_amplitude == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limb_dark_u1": _NAN},
        {"limb_dark_u2": _INF},
        {"limb_dark_u1": 5.0, "limb_dark_u2": 5.0},
        {"limb_dark_u1": 0.9, "limb_dark_u2": 0.2},
    ],
)
def test_limb_darkening_must_be_finite_and_physical(kwargs: dict[str, float]) -> None:
    # NaN gave an all-zero TLS spectrum; u1 + u2 > 1 makes the limb intensity
    # negative (u1 = u2 = 5 reported a 0.0018 depth for a true 0.01).
    with pytest.raises(ValidationError):
        cup.TLSSettings(**kwargs)


def test_tabulated_negative_u2_stays_legal() -> None:
    s = cup.TLSSettings(limb_dark_u1=0.7, limb_dark_u2=-0.1)
    assert s.limb_dark_u2 == -0.1


@pytest.mark.parametrize("field", ["minimum_spacing", "maximum_spacing"])
@pytest.mark.parametrize("bad", [-1.0, 0.0])
def test_spacings_must_be_positive(field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        cup.SpacingSettings(**{field: bad})


def test_spacing_minimum_must_be_finite_and_maximum_may_be_infinite() -> None:
    with pytest.raises(ValidationError):
        cup.SpacingSettings(minimum_spacing=_NAN)
    assert cup.SpacingSettings(maximum_spacing=_INF).maximum_spacing == _INF


# --- retired no-op fields (PW-A-12, CORE-A-10, METH-A-18) ----------------------------


def test_prewhiten_downsample_points_is_retired() -> None:
    # The field was accepted, validated and env-overridable but never read.
    assert "downsample_points" not in cup.PreWhitenSettings.model_fields
    with pytest.warns(DeprecationWarning, match="downsample_points"):
        s = cup.PreWhitenSettings(downsample_points=5)
    assert s == cup.PreWhitenSettings()


def test_retired_field_in_the_environment_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUPERIOD_PREWHITEN_DOWNSAMPLE_POINTS", "7")
    assert cup.PreWhitenSettings() == cup.PreWhitenSettings.model_construct()


def test_retired_field_warning_points_at_the_caller() -> None:
    with pytest.warns(DeprecationWarning) as record:
        cup.PreWhitenSettings(downsample_points=5)
    assert record[0].filename == __file__


# --- phase-tracking segment floor (PDOT-A-10) -----------------------------------------


@pytest.mark.parametrize("bad", [2, 3])
def test_pdot_segments_below_the_ephemeris_minimum_are_rejected(bad: int) -> None:
    # The quadratic O-C ephemeris needs four points, so 2 or 3 segments could only
    # fail later, inside phase_track.
    with pytest.raises(ValidationError, match="n_segments"):
        cup.PdotSettings(n_segments=bad)


def test_pdot_segments_from_the_environment_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUPERIOD_PDOT_N_SEGMENTS", "3")
    with pytest.raises(ValidationError, match="n_segments"):
        cup.PdotSettings()
    monkeypatch.setenv("CUPERIOD_PDOT_N_SEGMENTS", "4")
    assert cup.PdotSettings().n_segments == 4
    assert cup.PdotSettings(n_segments=None).n_segments is None
