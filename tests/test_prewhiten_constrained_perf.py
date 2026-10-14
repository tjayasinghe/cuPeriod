"""Zero points per group are projected out of the constrained solves (FWL).

With a zero point per night the dense design of every evaluation carried an
``N x S_o`` indicator block and a pivoted QR of ``N x (2K + S_o)``. Large fits now
remove the groups by weighted centring (exact by Frisch-Waugh-Lovell) and solve the
component columns only. These tests force each path and hold the projected one to
the dense one (round-off), to the brute-force oracle, to Kaufman's gradient, and to
its promise that nothing of size ``N x S_o`` is formed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import test_prewhiten_constrained_fit as base
from cuperiod.prewhiten import _constrained
from cuperiod.prewhiten._constrained import fit_constrained
from cuperiod.prewhiten.constraints import (
    ComponentConstraint,
    FitConstraints,
    FrequencyTie,
)
from cuperiod.prewhiten.fit import MultiSineFit

GROUPED = tuple(s for s in base.SCENARIOS if s.offsets == "groups")
DENSE = 1 << 62  # an indicator block no fit reaches: always the dense solve
PROJECTED = 0  # every fit with groups is projected


@pytest.fixture
def path(monkeypatch: pytest.MonkeyPatch) -> Any:
    """``path("dense" | "projected")`` selects how group zero points are solved."""

    def choose(which: str) -> None:
        value = DENSE if which == "dense" else PROJECTED
        monkeypatch.setattr(_constrained, "_PROJECT_GROUPS_ELEMENTS", value)

    return choose


def _phase_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def _in_sigma(a: np.ndarray, b: np.ndarray, sigma: np.ndarray) -> float:
    """Largest ``|a - b|`` in units of ``sigma`` (a fixed value, sigma 0, exactly)."""
    diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    scale = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, 1e-12)
    return float(np.max(diff / scale, initial=0.0))


def assert_same_fit(
    dense: MultiSineFit, projected: MultiSineFit, *, iterative: bool = True
) -> None:
    """The two solves of one problem agree.

    A single solve (``iterative=False``) agrees to round-off. An optimised one stops
    where the trust region's ``ftol`` says, and round-off may end it one evaluation
    earlier or later: then the two agree to a small fraction of the errors, the
    level the oracle comparison holds the kernel to.
    """
    d, p = dense.detail, projected.detail
    assert d is not None and p is not None
    tol = 1e-4 if iterative else 1e-7
    if iterative:
        # In units of the reduced chi-square: 1e-4 is a parameter shift of ~1e-2
        # sigma at most (a single cyclic sweep stops short of the optimum, so where
        # it stops depends on each block's stopping evaluation).
        scale = dense.rss / max(dense.n_samples - dense.n_parameters, 1)
        assert abs(projected.rss - dense.rss) <= 1e-4 * scale
    else:
        assert projected.rss == pytest.approx(dense.rss, rel=1e-12, abs=1e-12)
    sigma = np.sqrt(np.clip(np.diag(dense.covariance), 0.0, None))
    k = dense.n_components
    assert _in_sigma(projected.frequency, dense.frequency, sigma[:k]) < tol
    assert _in_sigma(projected.amplitude, dense.amplitude, sigma[k : 2 * k]) < tol
    phase = _phase_difference(projected.phase, dense.phase)
    assert _in_sigma(phase, np.zeros_like(phase), sigma[2 * k : 3 * k]) < tol
    if d.subset_offset is not None:
        assert p.subset_offset is not None and d.subset_offset_error is not None
        assert _in_sigma(p.subset_offset, d.subset_offset, d.subset_offset_error) < tol
    assert d.parameter_names == p.parameter_names
    if d.parameter_covariance.size:
        cov_tol = 1e-5 if iterative else 1e-8
        sd = np.sqrt(np.diag(d.parameter_covariance))
        sp = np.sqrt(np.diag(p.parameter_covariance))
        np.testing.assert_allclose(sp, sd, rtol=cov_tol, equal_nan=True)
        live = np.isfinite(sd) & (sd > 0.0)
        corr_d = d.parameter_covariance[np.ix_(live, live)] / np.outer(
            sd[live], sd[live]
        )
        corr_p = p.parameter_covariance[np.ix_(live, live)] / np.outer(
            sp[live], sp[live]
        )
        np.testing.assert_allclose(corr_p, corr_d, atol=cov_tol)
        assert p.rank == d.rank
        # The condition estimate is |R_00 / R_rr| of a pivoted QR of the
        # column-scaled design: every column has norm one there, so round-off picks
        # the first pivot and the ratio is an estimate that may differ by tens of
        # percent between two factorisations of the same design.
        assert 1.0 <= p.condition <= 2.0 * d.condition
        assert 1.0 <= d.condition <= 2.0 * p.condition
    assert d.flags == p.flags


# --- the oracle scenarios -------------------------------------------------------------


@pytest.mark.parametrize("refine", ["none", "last", "cyclic", "simultaneous"])
@pytest.mark.parametrize("baseline", ["short", "long"])
@pytest.mark.parametrize("scenario", GROUPED, ids=lambda s: s.name)
def test_projection_equals_the_dense_solve(
    scenario: base.Scenario, baseline: str, refine: str, path: Any
) -> None:
    built = base.build(scenario, baseline)
    path("dense")
    dense = base.run_package(built, refine=refine)
    path("projected")
    projected = base.run_package(built, refine=refine)
    assert_same_fit(dense, projected, iterative=refine != "none")


@pytest.mark.parametrize("baseline", ["short", "long"])
@pytest.mark.parametrize("scenario", GROUPED, ids=lambda s: s.name)
def test_projection_matches_the_oracle(
    scenario: base.Scenario, baseline: str, path: Any
) -> None:
    path("projected")
    base.test_matches_the_oracle(scenario, baseline)


@pytest.mark.parametrize("baseline", ["short", "long"])
@pytest.mark.parametrize("scenario", GROUPED, ids=lambda s: s.name)
def test_projected_policies_keep_their_promises(
    scenario: base.Scenario, baseline: str, path: Any
) -> None:
    import test_prewhiten_constrained_policies as policies

    path("projected")
    policies.test_block_schedule_is_monotone(scenario, baseline)
    policies.test_last_is_never_worse_than_its_start(scenario, baseline)
    policies.test_cyclic_sweeps_reach_the_simultaneous_optimum(scenario, baseline)


def test_projected_kaufman_gradient_matches_finite_differences(path: Any) -> None:
    path("projected")
    built = base.build(base.SCENARIOS[-1], "short")
    problem, alpha = base._problem_for(built)
    assert problem.projected
    alpha = alpha + np.r_[np.full(len(built.roots), 0.02 / 122.0),
                          np.full(alpha.size - len(built.roots), 0.05)]
    residual = problem.fun(alpha)
    gradient = problem.jac(alpha).T @ residual
    numeric = np.empty_like(gradient)
    for i in range(alpha.size):
        # Steps ten times those of the dense test: there the finite difference sits
        # on the round-off floor of the cost (~1e-6 relative for either solve).
        h = 1e-6 / 122.0 if i < len(built.roots) else 1e-5
        up, down = alpha.copy(), alpha.copy()
        up[i] += h
        down[i] -= h
        numeric[i] = (0.5 * np.sum(problem.fun(up) ** 2)
                      - 0.5 * np.sum(problem.fun(down) ** 2)) / (2 * h)
    assert np.max(np.abs(gradient - numeric) / np.abs(numeric)) < 1e-6


def test_small_fits_keep_the_dense_solve() -> None:
    built = base.build(GROUPED[0], "short")
    problem, _ = base._problem_for(built)
    n_o = int(np.max(built.constraints.offset_groups)) + 1
    assert built.t.size * n_o < _constrained._PROJECT_GROUPS_ELEMENTS
    assert not problem.projected


# --- large nightly data ---------------------------------------------------------------


def nightly(n: int, nights: int, k: int, seed: int = 1) -> dict[str, Any]:
    """``n`` points over ``nights`` nights, ``k`` modes and a zero point per night."""
    rng = np.random.default_rng(seed)
    per = n // nights
    t = np.concatenate(
        [night + 0.1 + 0.3 * np.sort(rng.random(per)) for night in range(nights)]
    )
    codes = np.repeat(np.arange(nights), per).astype(np.int32)
    freqs = np.linspace(5.3, 31.7, k) + rng.normal(0.0, 0.01, k)
    amps = np.linspace(0.05, 0.005, k)
    phases = rng.uniform(0.0, 2.0 * np.pi, k)
    y = (amps[:, None] * np.sin(2.0 * np.pi * freqs[:, None] * t
                                + phases[:, None])).sum(axis=0)
    y = y + rng.normal(0.0, 0.02, nights)[codes]
    error = 0.002 * (1.0 + 0.2 * rng.random(t.size))
    y = y + rng.normal(0.0, 1.0, t.size) * error
    start = freqs + rng.normal(0.0, 0.2 / float(np.ptp(t)), k)
    return {"t": t, "y": y, "error": error, "codes": codes, "start": start}


def _nightly_fit(data: dict[str, Any], constraints: FitConstraints,
                 **kw: Any) -> MultiSineFit:
    return fit_constrained(
        data["t"], data["y"], data["error"], data["start"],
        constraints=constraints, t_ref=float(np.mean(data["t"])), **kw,
    )


@pytest.mark.parametrize("refine", ["none", "last", "simultaneous"])
def test_many_nights_projected_equals_dense(refine: str, path: Any) -> None:
    data = nightly(6000, 60, 4)
    k = data["start"].size
    comps = [ComponentConstraint() for _ in range(k)]
    comps[1] = ComponentConstraint(frequency="fixed")
    comps[2] = ComponentConstraint(
        frequency=FrequencyTie(parents=(0,), coefficients=(2,))
    )
    comps[3] = ComponentConstraint(phase=0.7)
    constraints = FitConstraints(components=tuple(comps), offset_groups=data["codes"])
    data["start"][2] = 2.0 * data["start"][0]
    path("dense")
    dense = _nightly_fit(data, constraints, refine=refine)
    path("projected")
    projected = _nightly_fit(data, constraints, refine=refine)
    assert_same_fit(dense, projected, iterative=refine != "none")


def test_large_fits_never_form_the_indicator_block(monkeypatch: Any) -> None:
    data = nightly(20_000, 100, 6)
    k = data["start"].size
    widths: list[int] = []
    jac_columns: list[int] = []
    lstsq = _constrained._lstsq
    rows = _constrained._jacobian_rows

    def spy_lstsq(design_w: Any, rhs: Any, full_support: bool) -> Any:
        widths.append(int(design_w.shape[1]))
        return lstsq(design_w, rhs, full_support)

    def spy_rows(*args: Any) -> Any:
        out = rows(*args)
        jac_columns.append(int(out.shape[1]))
        return out

    monkeypatch.setattr(_constrained, "_lstsq", spy_lstsq)
    monkeypatch.setattr(_constrained, "_jacobian_rows", spy_rows)
    fit = _nightly_fit(
        data, FitConstraints(offset_groups=data["codes"]), refine="simultaneous"
    )
    assert fit.detail is not None and fit.detail.subset_offset is not None
    assert fit.detail.subset_offset.shape == (100,)
    assert widths and max(widths) == 2 * k  # cos/sin pairs only, no indicator
    assert jac_columns and max(jac_columns) == 3 * k  # f, A, phi; offsets closed-form
    assert fit.detail.rank == fit.detail.n_columns == 3 * k + 100


def test_projected_session_matches_the_dense_session(path: Any) -> None:
    """A nightly session: zero points per night, a fixed frequency, Improve all."""
    import cuperiod as cup
    from cuperiod.prewhiten.subsets import Partition

    data = nightly(4000, 40, 3, seed=5)
    light_curve = (data["t"], data["y"], data["error"])
    settings = cup.PreWhitenSettings(max_frequencies=3, backend="finufft")
    automatic = cup.PreWhitenSession(light_curve, settings=settings)
    automatic.run()
    # Both paths continue from one stored head (a threaded finufft spectrum is not
    # reproducible to the last bit, so two automatic runs could seed differently).
    snapshot = automatic.snapshot()

    def take_over() -> Any:
        session = cup.PreWhitenSession.restore(snapshot, light_curve)
        assert session.define_partition(
            Partition.from_gaps(session.data.time, min_gap=0.5, name="night")
        ).committed
        assert session.use_offsets("night").committed
        assert session.fix(1, frequency=True).committed
        assert session.improve().committed
        return session

    path("dense")
    dense = take_over()
    path("projected")
    projected = take_over()
    a, b = dense.state, projected.state
    assert [c.id for c in a.components] == [c.id for c in b.components]
    for x, y in zip(a.components, b.components, strict=True):
        assert y.frequency == pytest.approx(x.frequency, rel=1e-10)
        assert y.amplitude == pytest.approx(x.amplitude, rel=1e-6)
        assert _phase_difference(np.array(y.phase), np.array(x.phase)) < 1e-6
    assert a.subset_offsets is not None and b.subset_offsets is not None
    np.testing.assert_allclose(b.subset_offsets, a.subset_offsets, atol=1e-8)
    assert b.stats.rss == pytest.approx(a.stats.rss, rel=1e-9)
