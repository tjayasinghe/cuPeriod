"""Period-evolution models: the warps, their identities, and the registry."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.pdot.models import available_models, get_model


def _dt() -> np.ndarray:
    return np.linspace(-400.0, 400.0, 501)


@pytest.mark.parametrize("name", available_models())
def test_zero_rate_is_the_identity_warp(name: str) -> None:
    dt = _dt()
    assert np.array_equal(get_model(name).warp(dt, 0.0), dt)


@pytest.mark.parametrize("name", available_models())
def test_warp_is_continuous_at_zero_rate(name: str) -> None:
    """log1p / expm1 keep precision as a -> 0; a naive form would not."""
    dt = _dt()
    model = get_model(name)
    assert np.allclose(model.warp(dt, 1e-14), dt, rtol=0, atol=1e-8)


@pytest.mark.parametrize("name", available_models())
def test_delay_leading_term_is_a_over_two(name: str) -> None:
    """Every model's O-C starts as (a/2)*dt**2 — that is what makes ``a`` shared."""
    a = 3e-7
    dt = np.array([50.0, 100.0, 200.0])
    delay = get_model(name).delay(dt, a)
    assert np.allclose(delay, 0.5 * a * dt**2, rtol=2e-3)


@pytest.mark.parametrize("name", available_models())
def test_phase_matches_direct_integration(name: str) -> None:
    """The warp is exactly ``P0 * integral(dt/P(t))`` — the defining property."""
    model = get_model(name)
    a, period = -2.5e-6, 1.4
    grid = np.linspace(0.0, 900.0, 400_001)
    p_of_t = model.period_at(grid, period, a)
    integral = np.concatenate(
        [[0.0], np.cumsum(np.diff(grid) / (0.5 * (p_of_t[1:] + p_of_t[:-1])))]
    )
    probe = np.array([100.0, 400.0, 900.0])
    expected = np.interp(probe, grid, integral)
    assert np.allclose(model.warp(probe, a) / period, expected, rtol=1e-6)


def test_period_and_pdot_agree_at_the_reference_epoch() -> None:
    """Both models are parameterized by the same a = (dP/dt)/P at dt = 0."""
    a, period = 4e-7, 0.75
    zero = np.zeros(1)
    for name in available_models():
        model = get_model(name)
        assert model.period_at(zero, period, a)[0] == pytest.approx(period)
        assert model.pdot_at(zero, period, a)[0] == pytest.approx(a * period)


def test_constant_pdot_keeps_pdot_constant() -> None:
    model = get_model("constant-pdot")
    dt = np.array([-300.0, 0.0, 300.0])
    pdot = model.pdot_at(dt, 1.0, 5e-7)
    assert np.allclose(pdot, pdot[0])


def test_exponential_pdot_tracks_the_period() -> None:
    """dP/dt proportional to P: an inspiral accelerates as it shortens."""
    model = get_model("exponential")
    dt = np.array([-300.0, 0.0, 300.0])
    a, period = -5e-6, 1.4
    ratio = model.pdot_at(dt, period, a) / model.period_at(dt, period, a)
    assert np.allclose(ratio, a)


def test_the_two_models_differ_in_the_cubic_term() -> None:
    """constant-pdot: c3 = -a^2/3; exponential: c3 = -a^2/6."""
    a = 1e-6
    dt = np.linspace(-500.0, 500.0, 401)
    for name, expected in (("constant-pdot", -1 / 3), ("exponential", -1 / 6)):
        coef = np.polyfit(dt, get_model(name).delay(dt, a), 3)
        assert coef[0] == pytest.approx(expected * a * a, rel=2e-3)


def test_constant_pdot_refuses_its_singular_epoch() -> None:
    model = get_model("constant-pdot")
    with pytest.raises(ValueError, match="passes through zero|1 \\+ a\\*dt"):
        model.warp(np.array([1000.0]), -1.0 / 500.0)


def test_exponential_has_no_singular_epoch() -> None:
    model = get_model("exponential")
    assert np.all(np.isfinite(model.warp(np.array([1e4]), -1e-3)))
    assert model.max_abs_rate(500.0) > get_model("constant-pdot").max_abs_rate(500.0)


def test_fractional_change_integrates_the_law() -> None:
    a = -1e-5
    exp = get_model("exponential")
    assert exp.fractional_change(0.0, 1000.0, a) == pytest.approx(np.expm1(a * 1000.0))
    lin = get_model("constant-pdot")
    assert lin.fractional_change(0.0, 1000.0, a) == pytest.approx(a * 1000.0)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("constant-pdot", "constant-pdot"),
        ("quadratic", "constant-pdot"),
        ("linear", "constant-pdot"),
        ("Exponential", "exponential"),
        ("exp", "exponential"),
        ("constant_pdot", "constant-pdot"),
    ],
)
def test_registry_aliases(alias: str, expected: str) -> None:
    assert get_model(alias).name == expected


def test_registry_default_and_passthrough() -> None:
    default = get_model(None)
    assert default.name == "constant-pdot"
    assert get_model(default) is default


def test_unknown_model_names_the_alternatives() -> None:
    with pytest.raises(ValueError, match="unknown period-evolution model"):
        get_model("power-law")
