"""Parametric period laws: recovery, ranking, degeneracy, and the edges."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.core.errors import InsufficientDataError
from cuperiod.pdot.laws import (
    AUTO_LAWS,
    available_laws,
    compare_period_laws,
    fit_period_law,
)

#: Tylenda et al. (2011) Eq. (1) for V1309 Sco: P = P0*exp(b/(t - t0)).
V1309 = (1.4456, 15.29, 2455233.5)


def singular_track(
    n: int = 60, sigma: float = 0.0008, seed: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A published-quality P(t) track drawn from the V1309 Sco law."""
    p0, b, t0 = V1309
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2452100.0, 2454300.0, n))
    period = p0 * np.exp(b / (t - t0)) + rng.normal(0.0, sigma, n)
    return t, period, np.full(n, sigma)


def linear_track(
    n: int = 40, pdot: float = -3e-6, sigma: float = 5e-5, seed: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 2000.0, n)) + 2455000.0
    period = 0.8 + pdot * (t - t.mean()) + rng.normal(0.0, sigma, n)
    return t, period, np.full(n, sigma)


# --- recovery -----------------------------------------------------------------


def test_singular_law_recovers_the_published_constants() -> None:
    t, period, sigma = singular_track()
    fit = fit_period_law(t, period, sigma, law="singular")
    p0, b, t0 = V1309
    assert fit.converged
    assert fit.params["period"] == pytest.approx(p0, abs=4.0 * fit.errors["period"])
    assert fit.params["b"] == pytest.approx(b, abs=4.0 * fit.errors["b"])
    assert fit.params["t0"] == pytest.approx(t0, abs=4.0 * fit.errors["t0"])


def test_singular_law_reproduces_the_total_change_and_timescales() -> None:
    """The derived quantities survive even though b and t0 are degenerate."""
    t, period, sigma = singular_track()
    fit = fit_period_law(t, period, sigma, law="singular")
    p0, b, t0 = V1309
    exact = p0 * np.exp(b / (np.array([t[0], t[-1]]) - t0))
    assert fit.fractional_change(t[0], t[-1]) == pytest.approx(
        exact[1] / exact[0] - 1.0, rel=0.15
    )
    # P/|dP/dt| = (t - t0)**2 / b in closed form for this law.
    probe = np.array([t[0], t[-1]])
    assert fit.timescale_yr_at(probe) == pytest.approx(
        (probe - t0) ** 2 / b / 365.25, rel=0.35
    )


def test_linear_law_recovers_a_constant_rate() -> None:
    t, period, sigma = linear_track(pdot=-3e-6)
    fit = fit_period_law(t, period, sigma, law="linear")
    assert fit.params["pdot"] == pytest.approx(-3e-6, abs=3.0 * fit.errors["pdot"])


def test_exponential_law_recovers_a_constant_fractional_rate() -> None:
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(0.0, 3000.0, 50)) + 2455000.0
    a = -4e-6
    period = 1.2 * np.exp(a * (t - t.mean())) + rng.normal(0.0, 3e-5, t.size)
    fit = fit_period_law(t, period, np.full(t.size, 3e-5), law="exponential")
    assert fit.params["dlnp_dt"] == pytest.approx(a, abs=3.0 * fit.errors["dlnp_dt"])


def test_pejcha_law_recovers_the_exponent_of_a_singular_track() -> None:
    """delta = 2 is exactly the singular law, with amplitude = b."""
    t, period, sigma = singular_track(n=90, sigma=0.0004, seed=17)
    fit = fit_period_law(t, period, sigma, law="pejcha")
    assert fit.params["delta"] == pytest.approx(2.0, abs=max(1.0, fit.errors["delta"]))


# --- ranking ------------------------------------------------------------------


def test_bic_prefers_the_law_that_generated_the_track() -> None:
    t, period, sigma = singular_track()
    ranked = compare_period_laws(t, period, sigma, laws=available_laws())
    assert ranked[0].law in {"singular", "pejcha"}
    by_name = {f.law: f for f in ranked}
    assert by_name["linear"].bic - by_name["singular"].bic > 10.0


def test_bic_prefers_the_simple_law_on_a_simple_track() -> None:
    t, period, sigma = linear_track()
    ranked = compare_period_laws(t, period, sigma)
    assert ranked[0].law in {"linear", "exponential"}


def test_constant_wins_when_nothing_is_changing() -> None:
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0.0, 1000.0, 30)) + 2455000.0
    period = 0.5 + rng.normal(0.0, 1e-5, t.size)
    assert compare_period_laws(t, period, np.full(t.size, 1e-5))[0].law == "constant"


def test_auto_returns_the_lowest_bic_of_the_auto_laws() -> None:
    t, period, sigma = singular_track()
    auto = fit_period_law(t, period, sigma, law="auto")
    ranked = compare_period_laws(t, period, sigma, laws=AUTO_LAWS)
    assert auto.law == ranked[0].law
    assert "pejcha" not in AUTO_LAWS  # too free for a routine automatic choice


def test_compare_skips_laws_it_cannot_constrain() -> None:
    """Three points cannot support a four-parameter law; that is not an error."""
    t = np.array([2455000.0, 2455500.0, 2456000.0])
    fits = compare_period_laws(t, np.array([1.0, 0.999, 0.998]), laws=available_laws())
    assert {f.law for f in fits} <= {"constant", "linear", "exponential"}
    assert fits


# --- structure ----------------------------------------------------------------


def test_singular_epoch_is_kept_outside_the_data() -> None:
    t, period, sigma = singular_track()
    fit = fit_period_law(t, period, sigma, law="singular")
    assert not (t.min() <= fit.params["t0"] <= t.max())
    assert fit.singular_epoch == fit.params["t0"]


def test_both_singular_branches_are_tried() -> None:
    """A period *rising* toward a singularity puts t0 before the data."""
    rng = np.random.default_rng(23)
    t = np.sort(rng.uniform(2453000.0, 2455000.0, 60))
    t0 = 2452500.0
    period = 1.0 * np.exp(-6.0 / (t - t0)) + rng.normal(0.0, 2e-4, t.size)
    fit = fit_period_law(t, period, np.full(t.size, 2e-4), law="singular")
    assert fit.params["t0"] < t.min()


def test_errors_are_inflated_by_a_poor_chi_squared() -> None:
    t, period, sigma = linear_track(sigma=5e-5)
    tenth = sigma / 10.0  # errors ten times too small
    scaled = fit_period_law(t, period, tenth, law="linear", scale_errors=True)
    raw = fit_period_law(t, period, tenth, law="linear", scale_errors=False)
    assert scaled.chi2_reduced > 10.0
    assert scaled.errors["pdot"] == pytest.approx(
        raw.errors["pdot"] * np.sqrt(scaled.chi2_reduced), rel=1e-6
    )


def test_equal_weights_when_no_errors_are_given() -> None:
    t, period, _ = linear_track()
    fit = fit_period_law(t, period, None, law="linear")
    assert fit.n_used == t.size
    assert fit.params["pdot"] == pytest.approx(-3e-6, rel=0.1)


@pytest.mark.parametrize("law", available_laws())
def test_pdot_at_is_the_derivative_of_period_at(law: str) -> None:
    """Central difference, not np.gradient: these curves are steep."""
    t, period, sigma = singular_track()
    fit = fit_period_law(t, period, sigma, law=law)
    probe = np.linspace(t.min(), t.max(), 9)
    h = 0.5
    numeric = (fit.period_at(probe + h) - fit.period_at(probe - h)) / (2.0 * h)
    assert np.allclose(fit.pdot_at(probe), numeric, rtol=1e-4, atol=1e-14)


def test_to_dict_carries_params_and_errors() -> None:
    t, period, sigma = singular_track()
    row = fit_period_law(t, period, sigma, law="singular").to_dict(prefix="law_")
    assert row["law_law"] == "singular"
    assert {"law_b", "law_b_err", "law_t0", "law_t0_err", "law_bic"} <= set(row)


# --- edges --------------------------------------------------------------------


def test_non_finite_points_are_dropped() -> None:
    t, period, sigma = linear_track()
    period = period.copy()
    period[3] = np.nan
    sigma = sigma.copy()
    sigma[7] = 0.0  # a zero error is unusable, not fatal
    fit = fit_period_law(t, period, sigma, law="linear")
    assert fit.n_used == t.size - 2


def test_too_few_points_raises_for_a_named_law() -> None:
    t = np.array([2455000.0, 2455500.0])
    with pytest.raises(InsufficientDataError, match="cannot constrain"):
        fit_period_law(t, np.array([1.0, 0.99]), law="singular")


def test_all_points_unusable_raises() -> None:
    with pytest.raises(InsufficientDataError, match="no usable period points"):
        fit_period_law(np.array([1.0, 2.0, 3.0]), np.array([np.nan] * 3))


def test_unknown_law_names_the_alternatives() -> None:
    t, period, sigma = linear_track()
    with pytest.raises(ValueError, match="unknown period law"):
        fit_period_law(t, period, sigma, law="powerlaw")


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_period_law(np.arange(5.0), np.arange(4.0))
