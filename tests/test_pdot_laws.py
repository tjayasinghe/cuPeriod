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


# --- boundary-pinned fits -----------------------------------------------------


def flat_track_with_one_outlier() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A constant period, but the first (most precise) point sits 4 sigma low.

    This is the shape a real null looks like — a track whose scatter exceeds its
    formal errors — and it is exactly what a finite-time singularity will chase:
    push ``t0`` up against the data and ``b/(t - t0)`` becomes a spike on that
    one point.
    """
    t = np.linspace(2456900.0, 2461200.0, 19)
    period = np.full(t.size, 0.3034290)
    sigma = np.full(t.size, 3.0e-6)
    sigma[:3] = 8.0e-7  # the early points are far more precise
    rng = np.random.default_rng(5)
    period = period + rng.normal(0.0, sigma)
    period[0] -= 4.0 * sigma[0]
    return t, period, sigma


def test_singular_epoch_keeps_its_distance_from_the_data() -> None:
    t, period, sigma = flat_track_with_one_outlier()
    fit = fit_period_law(t, period, sigma, law="singular")
    gap = min(abs(fit.params["t0"] - t.min()), abs(fit.params["t0"] - t.max()))
    spacing = float(np.median(np.diff(np.sort(t))))
    assert gap >= spacing * (1.0 - 1e-6)


def test_a_pinned_fit_is_flagged() -> None:
    t, period, sigma = flat_track_with_one_outlier()
    assert fit_period_law(t, period, sigma, law="singular").at_bound
    assert not fit_period_law(t, period, sigma, law="constant").at_bound


def test_a_pinned_fit_never_wins_the_comparison() -> None:
    """Even at a lower BIC: a fit on its constraint has no interior optimum."""
    t, period, sigma = flat_track_with_one_outlier()
    ranked = compare_period_laws(t, period, sigma, laws=available_laws())
    assert ranked[0].law == "constant"
    assert not ranked[0].at_bound
    pinned = [f for f in ranked if f.at_bound]
    assert pinned, "this track is built to pin the singular law"
    # ...and demotion is on the flag, not on the BIC, which still favours them.
    assert min(f.bic for f in pinned) < ranked[0].bic
    assert all(ranked.index(f) > ranked.index(ranked[0]) for f in pinned)
    assert fit_period_law(t, period, sigma, law="auto").law == "constant"


def test_a_real_singularity_is_not_demoted() -> None:
    """The floor must not block the case the law exists for."""
    t, period, sigma = singular_track()
    fit = fit_period_law(t, period, sigma, law="singular")
    assert not fit.at_bound
    assert compare_period_laws(t, period, sigma, laws=AUTO_LAWS)[0].law == "singular"


def test_at_bound_is_serialized() -> None:
    t, period, sigma = flat_track_with_one_outlier()
    row = fit_period_law(t, period, sigma, law="singular").to_dict()
    assert row["at_bound"] is True


# --- the edge test measures the law, not the sampling --------------------------


def test_a_long_final_gap_does_not_demote_a_real_singularity() -> None:
    """The edge test must not read the sampling as the law.

    Eleven even epochs, then one 600 d later, on the exact V1309 law: the
    singular epoch sits ~6 median spacings past the data and the fit is good,
    but the model's step between the last two *samples* is over half its swing
    because those two samples are far apart.
    """
    p0, b, t0 = V1309
    t = np.concatenate([np.linspace(2452100.0, 2453700.0, 11), [2454300.0]])
    rng = np.random.default_rng(0)
    period = p0 * np.exp(b / (t - t0)) + rng.normal(0.0, 1e-4, t.size)
    sigma = np.full(t.size, 1e-4)
    fit = fit_period_law(t, period, sigma, law="singular")
    assert not fit.at_bound
    assert fit.chi2_reduced < 3.0
    assert fit_period_law(t, period, sigma, law="auto").law == "singular"


def test_an_end_outlier_on_uneven_epochs_is_caught() -> None:
    """A spike on an end point stays a spike when the end step is short.

    Flat tracks on random epochs with one 4-sigma end point: the sample-based
    test flagged the singular absorber on 11 of these 24 and let ``auto`` pick
    ``singular`` on 8. Measured on the law at the track's resolution it is
    flagged on 21 and wins once (an absorber spread over the last two points,
    which no edge test can tell from a trend).
    """
    flagged = wins = 0
    for seed in range(24):
        rng = np.random.default_rng(7000 + seed)
        n = 25
        t = np.sort(rng.uniform(0.0, 3000.0, n)) + 2455000.0
        sigma = np.full(n, 1e-6)
        period = 0.4 + rng.normal(0.0, 1e-6, n)
        end = 0 if seed % 2 == 0 else n - 1
        period[end] += 4e-6 * (1 if (seed // 2) % 2 == 0 else -1)
        flagged += fit_period_law(t, period, sigma, law="singular").at_bound
        wins += fit_period_law(t, period, sigma, law="auto").law == "singular"
    assert flagged >= 18
    assert wins <= 2


# --- ranking that survives understated errors, and fits without errors ---------


def test_understated_errors_do_not_buy_a_period_change() -> None:
    """Constant tracks scattering 5x beyond their quoted errors stay constant.

    The weighted BIC multiplies every chi-squared difference by 25 here and
    ranked a period change first on most of them; the scaled BIC divides that
    factor back out using the scatter of the most flexible free law.
    """
    rng = np.random.default_rng(9050)
    wins = decisive = 0
    trials = 40
    for _ in range(trials):
        t = np.sort(rng.uniform(0.0, 3000.0, 30)) + 2455000.0
        period = 0.4 + rng.normal(0.0, 5e-6, t.size)
        ranked = compare_period_laws(t, period, np.full(t.size, 1e-6))
        wins += ranked[0].law == "constant"
        by_name = {f.law: f for f in ranked}
        decisive += by_name["constant"].bic_scaled - ranked[0].bic_scaled > 10.0
    assert wins >= 0.85 * trials
    assert decisive == 0


def test_honest_errors_rank_exactly_as_the_plain_bic() -> None:
    """With chi2_reduced <= 1 the scaled BIC *is* the BIC."""
    t, period, sigma = singular_track()
    ranked = compare_period_laws(t, period, sigma * 3.0, laws=available_laws())
    for fit in ranked:
        assert fit.bic_scaled == pytest.approx(fit.bic, rel=1e-12)
    assert np.isnan(fit_period_law(t, period, sigma, law="linear").bic_scaled)


def test_unweighted_fits_find_the_trend_and_its_errors() -> None:
    """Without period_err the laws are compared on the residuals' own scale.

    With one-day stand-in errors every chi-squared was ~1e-10, the BIC was the
    parameter penalty alone, and 'auto' returned 'constant' for any track; the
    parameter errors assumed a one-day scatter (a million times too large here).
    """
    t, period, sigma = linear_track()
    auto = fit_period_law(t, period, None, law="auto")
    assert auto.law in {"linear", "exponential"}
    assert not auto.weighted and auto.chi2_reduced == pytest.approx(1.0)
    weighted = fit_period_law(t, period, sigma, law="linear")
    plain = fit_period_law(t, period, None, law="linear")
    assert plain.params["pdot"] == pytest.approx(weighted.params["pdot"], rel=1e-6)
    assert plain.errors["pdot"] == pytest.approx(weighted.errors["pdot"], rel=0.3)
    st, sp, _ = singular_track()
    assert fit_period_law(st, sp, None, law="auto").law == "singular"
    rng = np.random.default_rng(2)
    ct = np.sort(rng.uniform(0.0, 1000.0, 30)) + 2455000.0
    cp = 0.5 + rng.normal(0.0, 1e-5, ct.size)
    assert fit_period_law(ct, cp, None, law="auto").law == "constant"


def test_unweighted_ranking_ignores_the_units() -> None:
    t, period, _ = linear_track()
    days = compare_period_laws(t, period)
    seconds = compare_period_laws(t, period * 86400.0)
    assert [f.law for f in days] == [f.law for f in seconds]
    for a, b in zip(days, seconds, strict=True):
        assert a.bic_scaled - days[0].bic_scaled == pytest.approx(
            b.bic_scaled - seconds[0].bic_scaled, abs=1e-6
        )


# --- caller mistakes are not skipped laws ---------------------------------------


def test_compare_refuses_caller_mistakes() -> None:
    t, period, sigma = linear_track()
    with pytest.raises(ValueError, match="same length"):
        compare_period_laws(np.arange(5.0), np.arange(4.0))
    with pytest.raises(ValueError, match="unknown period law"):
        compare_period_laws(t, period, sigma, laws=("constant", "linar"))
    with pytest.raises(ValueError, match="unknown period law"):
        compare_period_laws(t, period, sigma, laws=("linar",))
    # A bare name is one law, not a sequence of letters.
    assert [f.law for f in compare_period_laws(t, period, sigma, laws="linear")] == [
        "linear"
    ]


# --- errors -------------------------------------------------------------------


def _physical_covariance_errors(fit, t, period, sigma) -> np.ndarray:
    """Gauss-Newton errors computed directly in the published parameters."""
    names = list(fit.params)
    theta = np.array([fit.params[k] for k in names])

    def model(th: np.ndarray) -> np.ndarray:
        p0, amp, t0, delta = th
        gap = np.abs(t - t0)
        return p0 * np.exp((amp / (1.0 - delta)) * gap ** (1.0 - delta))

    steps = np.maximum(np.abs(theta) * 1e-6, 1e-6)
    steps[2] = 1e-4 * min(abs(theta[2] - t.min()), abs(theta[2] - t.max()))
    jac = np.empty((t.size, theta.size))
    for j in range(theta.size):
        up, down = theta.copy(), theta.copy()
        up[j] += steps[j]
        down[j] -= steps[j]
        jac[:, j] = (model(up) - model(down)) / (2.0 * steps[j] * sigma)
    return np.sqrt(np.diag(np.linalg.inv(jac.T @ jac)))


def test_pejcha_errors_include_the_exponent_dependence() -> None:
    """amplitude = a' * T**(delta-1) depends on delta too; its error must say so."""
    t, period, sigma = singular_track(n=90, sigma=0.0004, seed=17)
    fit = fit_period_law(t, period, sigma, law="pejcha", scale_errors=False)
    direct = _physical_covariance_errors(fit, t, period, sigma)
    got = np.array([fit.errors[k] for k in fit.params])
    assert np.allclose(got, direct, rtol=0.05)


def test_unconstrained_parameters_report_nan_not_zero() -> None:
    t = np.linspace(2455000.0, 2456000.0, 8)
    flat = np.full(t.size, 0.5)
    sigma = np.full(t.size, 1e-5)
    singular = fit_period_law(t, flat, sigma, law="singular")
    assert np.isnan(singular.errors["t0"])
    pejcha = fit_period_law(t, flat, sigma, law="pejcha")
    assert np.isnan(pejcha.errors["t0"]) and np.isnan(pejcha.errors["delta"])
    assert all(v != 0.0 for v in singular.errors.values())
    # A singular epoch held on the floor has no Gaussian error either.
    ft, fp, fs = flat_track_with_one_outlier()
    pinned = fit_period_law(ft, fp, fs, law="singular")
    assert pinned.at_bound and np.isnan(pinned.errors["t0"])


def test_singular_errors_are_unchanged_by_the_reparametrization() -> None:
    """Pinned from the published-coordinate fit this replaced."""
    t, period, sigma = singular_track()
    fit = fit_period_law(t, period, sigma, law="singular")
    assert fit.chi2 == pytest.approx(69.39418, abs=1e-4)
    assert fit.errors["period"] == pytest.approx(0.001131, rel=2e-3)
    assert fit.errors["b"] == pytest.approx(2.60, rel=5e-3)
    assert fit.errors["t0"] == pytest.approx(146.2, rel=2e-3)


@pytest.mark.parametrize("make", ["v1309", "rising", "pejcha"])
def test_parameters_reproduce_the_published_form(make: str) -> None:
    """``params`` must mean what the law's formula says, on both sides of t0."""
    if make == "v1309":
        t, period, sigma = singular_track()
        law = "singular"
    elif make == "rising":
        rng = np.random.default_rng(23)
        t = np.sort(rng.uniform(2453000.0, 2455000.0, 60))
        period = np.exp(-6.0 / (t - 2452500.0)) + rng.normal(0.0, 2e-4, t.size)
        sigma = np.full(t.size, 2e-4)
        law = "singular"
    else:
        t, period, sigma = singular_track(n=90, sigma=0.0004, seed=17)
        law = "pejcha"
    fit = fit_period_law(t, period, sigma, law=law)
    probe = np.linspace(t.min(), t.max(), 7)
    q = fit.params
    if law == "singular":
        formula = q["period"] * np.exp(q["b"] / (probe - q["t0"]))
    else:
        gap = np.abs(probe - q["t0"])
        formula = q["period"] * np.exp(
            (q["amplitude"] / (1.0 - q["delta"])) * gap ** (1.0 - q["delta"])
        )
    assert np.allclose(fit.period_at(probe), formula, rtol=1e-9)


# --- cost -----------------------------------------------------------------------


def test_singular_fits_converge_without_burning_the_budget(monkeypatch) -> None:
    """The losing t0 branch used to creep to max_nfev=4000 on every trend."""
    import scipy.optimize

    calls: list[int] = []
    real = scipy.optimize.least_squares

    def counting(*args, **kwargs):
        sol = real(*args, **kwargs)
        calls.append(int(sol.nfev))
        return sol

    monkeypatch.setattr(scipy.optimize, "least_squares", counting)
    t, period, sigma = singular_track()
    for law in ("singular", "pejcha"):
        calls.clear()
        fit = fit_period_law(t, period, sigma, law=law)
        assert sum(calls) < 400, (law, calls)
        assert fit.converged
    calls.clear()
    lt, lp, ls = linear_track()
    fit = fit_period_law(lt, lp, ls, law="singular")
    assert sum(calls) < 400 and fit.chi2 <= 34.58012


def test_track_constants_are_computed_once_per_fit(monkeypatch) -> None:
    from cuperiod.pdot import laws

    count = 0
    real = laws._epoch_floor

    def counting(t: np.ndarray) -> float:
        nonlocal count
        count += 1
        return real(t)

    monkeypatch.setattr(laws, "_epoch_floor", counting)
    t, period, sigma = singular_track()
    fit_period_law(t, period, sigma, law="pejcha")
    assert count <= 2


# --- equality -------------------------------------------------------------------


@pytest.mark.parametrize("law", ["linear", "singular"])
def test_identical_fits_compare_equal(law: str) -> None:
    t, period, sigma = singular_track()
    f1 = fit_period_law(t, period, sigma, law=law)
    f2 = fit_period_law(t, period, sigma, law=law)
    assert f1 == f2
    assert f2 in [f1]
    f3 = fit_period_law(t, period + 1e-4, sigma, law=law)
    assert f1 != f3
