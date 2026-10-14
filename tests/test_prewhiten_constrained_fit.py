"""The constrained multi-sine kernel against a brute-force oracle (E1, spec 5.1).

Every scenario is fitted twice: by ``fit_multisine(..., constraints=...)`` (variable
projection over the free frequencies and the non-linear phases only) and by
``oracle_multisine.solve`` (all free natural parameters at once, analytic Jacobian,
started at the truth plus 5 sigma). They must agree on the residual sum of squares,
on every parameter to a thousandth of its sigma, on the sigmas to 1e-5 and on the
correlation matrix to 1e-5 — on a single-season baseline (T ~ 122 d) and on a
multi-season one (T > 1000 d), the per-season use case.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pytest

import oracle_multisine as oracle
from cuperiod.prewhiten import _constrained
from cuperiod.prewhiten import fit as fit_module
from cuperiod.prewhiten._constrained import fit_constrained
from cuperiod.prewhiten.constraints import (
    ComponentConstraint,
    ConstraintError,
    FitConstraints,
    FitStart,
    FrequencyTie,
)
from cuperiod.prewhiten.fit import MultiSineFit, fit_multisine

TWO_PI = 2.0 * np.pi

# --- scenarios -----------------------------------------------------------------------


@dataclass(frozen=True)
class Comp:
    """One true component and how the fit constrains it.

    ``freq``: ``"free"``, ``"fixed"`` or ``("tie", ((parent, n), ...), constant)``;
    ``amp``: ``"free"``, ``"fixed"`` or ``"fixed_wrong"`` (held at 0.8 A);
    ``phase``: ``"free"``, ``"fixed"`` or ``"fixed_pi"`` (held pi away: negative A).
    """

    f: float
    A: float
    phi: float
    freq: Any = "free"
    amp: str = "free"
    phase: str = "free"
    vary: str = "none"


@dataclass(frozen=True)
class Scenario:
    name: str
    comps: tuple[Comp, ...]
    offsets: str = "single"  # "single" | "groups" | "none"
    variation: bool = False
    weights: bool = False


F1, F2 = 4.37, 6.91
BASE = (Comp(F1, 0.020, 0.4), Comp(F2, 0.012, 2.0))


def _sc(name: str, *extra: Comp, base: tuple[Comp, ...] = BASE, **kw: Any) -> Scenario:
    return Scenario(name, base + extra, **kw)


SCENARIOS: tuple[Scenario, ...] = (
    _sc("free"),
    _sc("sum-tie", Comp(F1 + F2, 0.006, 1.0, freq=("tie", ((0, 1), (1, 1)), 0.0))),
    _sc("harmonic-tie", Comp(2 * F1, 0.005, 3.0, freq=("tie", ((0, 2),), 0.0))),
    _sc(
        "tie-fixed-parent",
        Comp(F1 + F2, 0.006, 1.0, freq=("tie", ((0, 1), (1, 1)), 0.0)),
        base=(Comp(F1, 0.020, 0.4), Comp(F2, 0.012, 2.0, freq="fixed")),
    ),
    _sc("tie-constant",
        Comp(2 * F1 + 0.0013, 0.005, 3.0, freq=("tie", ((0, 2),), 0.0013))),
    _sc("fixed-frequency", Comp(9.83, 0.006, 5.0, freq="fixed")),
    _sc("fixed-amplitude", Comp(9.83, 0.006, 5.0, amp="fixed")),
    _sc("fixed-amplitude-wrong", Comp(9.83, 0.006, 5.0, amp="fixed_wrong")),
    _sc("fixed-phase", Comp(9.83, 0.006, 5.0, phase="fixed")),
    _sc("fixed-phase-pi", Comp(9.83, 0.006, 5.0, phase="fixed_pi")),
    _sc("zero-points", Comp(9.83, 0.006, 5.0), offsets="groups"),
    _sc("amplitude-per-subset", Comp(9.83, 0.008, 5.0, vary="amplitude"),
        variation=True),
    _sc("phase-per-subset", Comp(9.83, 0.008, 5.0, vary="phase"), variation=True),
    _sc("both-per-subset", Comp(9.83, 0.008, 5.0, vary="both"), variation=True),
    _sc("both-per-subset-fixed-f",
        Comp(9.83, 0.008, 5.0, vary="both", freq="fixed"), variation=True),
    _sc("fixed-amplitude-phase-per-subset",
        Comp(9.83, 0.008, 5.0, vary="phase", amp="fixed"), variation=True),
    _sc("two-partitions-weights",
        Comp(9.83, 0.008, 5.0, vary="amplitude"),
        offsets="groups", variation=True, weights=True),
    _sc(
        "psi-only",
        Comp(9.83, 0.008, 5.0, vary="phase", amp="fixed", freq="fixed"),
        base=(Comp(F1, 0.020, 0.4, freq="fixed", amp="fixed"),
              Comp(F2, 0.012, 2.0, freq="fixed")),
        variation=True,
    ),
    _sc(
        "everything-fixed",
        base=(Comp(F1, 0.020, 0.4, freq="fixed", amp="fixed", phase="fixed"),
              Comp(F2, 0.012, 2.0, freq="fixed", amp="fixed", phase="fixed")),
    ),
    _sc(
        "everything-fixed-offsets",
        base=(Comp(F1, 0.020, 0.4, freq="fixed", amp="fixed", phase="fixed"),
              Comp(F2, 0.012, 2.0, freq="fixed", amp="fixed", phase="fixed")),
        offsets="groups",
    ),
    _sc(
        "everything-fixed-no-mean",
        base=(Comp(F1, 0.020, 0.4, freq="fixed", amp="fixed", phase="fixed"),
              Comp(F2, 0.012, 2.0, freq="fixed", amp="fixed", phase="fixed")),
        offsets="none",
    ),
    _sc("no-mean-tie", Comp(F1 + F2, 0.006, 1.0, freq=("tie", ((0, 1), (1, 1)), 0.0)),
        offsets="none"),
    _sc(
        "kitchen-sink",
        Comp(F1 + F2, 0.006, 1.0, freq=("tie", ((0, 1), (1, 1)), 0.0)),
        Comp(9.83, 0.008, 5.0, vary="both"),
        Comp(12.41, 0.007, 2.5, vary="amplitude", phase="fixed"),
        base=(Comp(F1, 0.020, 0.4, vary="amplitude"), Comp(F2, 0.012, 2.0)),
        offsets="groups", variation=True, weights=True,
    ),
)


@dataclass
class Built:
    """A generated data set, its constraints, the start and the oracle spec."""

    t: np.ndarray
    y: np.ndarray
    error: np.ndarray
    t_ref: float
    constraints: FitConstraints
    start: np.ndarray
    fit_mean: bool
    spec: oracle.OracleSpec
    truth: np.ndarray
    roots: list[int]
    extra: dict[str, Any] = field(default_factory=dict)


def sampling(baseline: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Times plus offset and variation codes for a short or a multi-season baseline."""
    rng = np.random.default_rng(seed)
    if baseline == "short":
        t = np.sort(rng.uniform(0.0, 122.0, 700))
        ocodes = np.minimum((t // 30.5).astype(int), 3)
        vcodes = np.minimum((t // 40.7).astype(int), 2)
    else:
        t = np.sort(np.concatenate([rng.uniform(s, s + 120.0, 300)
                                    for s in (0.0, 500.0, 1000.0)]))
        season = np.searchsorted([250.0, 750.0], t)
        vcodes = season
        ocodes = 2 * season + (t - np.array([0.0, 500.0, 1000.0])[season] > 60.0)
    return t + 2459000.0, ocodes.astype(int), vcodes.astype(int)


def _per_subset(value: float, n: int, step: float, *, scale: bool) -> np.ndarray:
    offsets = np.arange(n) - (n - 1) / 2.0
    return value * (1.0 + step * offsets) if scale else value + step * offsets


def build(scenario: Scenario, baseline: str = "short", seed: int = 3) -> Built:
    t, ocodes, vcodes = sampling(baseline, seed)
    rng = np.random.default_rng(seed + 100)
    n = t.size
    t_ref = float(np.round(np.mean(t), 1))
    tau = t - t_ref
    n_v = int(vcodes.max()) + 1
    n_o = int(ocodes.max()) + 1
    span = float(np.ptp(t))
    noise = 0.002
    error = np.full(n, noise)
    omega = np.ones(n)
    if scenario.weights:
        omega = np.where(vcodes == 1, 0.5, 1.0) * np.where(ocodes % 2 == 0, 1.0, 2.0)
    y = np.zeros(n)
    offsets_true = 10.0 + 0.003 * (np.arange(n_o) - (n_o - 1) / 2.0)
    if scenario.offsets == "groups":
        y += offsets_true[ocodes]
    elif scenario.offsets == "single":
        y += 10.0
    comps: list[ComponentConstraint] = []
    start = np.zeros(len(scenario.comps))
    roots = [k for k, c in enumerate(scenario.comps) if c.freq == "free"]
    matrix = np.zeros((len(scenario.comps), len(roots)))
    constant = np.zeros(len(scenario.comps))
    amp_modes: list[Any] = []
    phase_modes: list[Any] = []
    truth: dict[tuple[str, int, int], float] = {}
    for k, c in enumerate(scenario.comps):
        amp_s = (_per_subset(c.A, n_v, 0.2, scale=True)
                 if c.vary in ("amplitude", "both") else np.full(n_v, c.A))
        phase_s = (_per_subset(c.phi, n_v, 0.35, scale=False)
                   if c.vary in ("phase", "both") else np.full(n_v, c.phi))
        y += amp_s[vcodes] * np.sin(TWO_PI * c.f * tau + phase_s[vcodes])
        # frequency
        if c.freq == "free":
            frequency: Any = "free"
            matrix[k, roots.index(k)] = 1.0
            start[k] = c.f + (0.05 if k % 2 else -0.05) / span
            truth[("f", roots.index(k), -1)] = c.f
        elif c.freq == "fixed":
            frequency = "fixed"
            constant[k] = c.f
            start[k] = c.f
        else:
            _, terms, const = c.freq
            frequency = FrequencyTie(
                tuple(p for p, _ in terms), tuple(m for _, m in terms), const
            )
            constant[k] = const
            for parent, m in terms:
                if scenario.comps[parent].freq == "free":
                    matrix[k, roots.index(parent)] += m
                else:
                    constant[k] += m * scenario.comps[parent].f
            start[k] = 0.0
        # amplitude
        sign = -1.0 if c.phase == "fixed_pi" else 1.0
        if c.amp == "fixed":
            amplitude: float | None = c.A
            amp_modes.append(c.A)
        elif c.amp == "fixed_wrong":
            amplitude = 0.8 * c.A
            amp_modes.append(0.8 * c.A)
        elif c.vary in ("amplitude", "both"):
            amplitude = None
            amp_modes.append("subset")
            for g in range(n_v):
                truth[("A", k, g)] = sign * amp_s[g]
        else:
            amplitude = None
            amp_modes.append("common")
            truth[("A", k, -1)] = sign * c.A
        # phase
        if c.phase in ("fixed", "fixed_pi"):
            phase: float | None = c.phi + (np.pi if c.phase == "fixed_pi" else 0.0)
            phase_modes.append(phase)
        elif c.vary in ("phase", "both"):
            phase = None
            phase_modes.append("subset")
            for g in range(n_v):
                truth[("phi", k, g)] = phase_s[g]
        else:
            phase = None
            phase_modes.append("common")
            truth[("phi", k, -1)] = c.phi
        comps.append(ComponentConstraint(frequency=frequency, amplitude=amplitude,
                                         phase=phase, variation=c.vary))  # type: ignore[arg-type]
    if scenario.offsets == "groups":
        for g in range(n_o):
            truth[("o", -1, g)] = offsets_true[g]
    elif scenario.offsets == "single":
        truth[("o", -1, -1)] = 10.0
    y = y + rng.normal(0.0, 1.0, n) * noise / np.sqrt(omega)
    constraints = FitConstraints(
        components=tuple(comps),
        offset_groups=ocodes if scenario.offsets == "groups" else None,
        variation_groups=vcodes if scenario.variation else None,
        point_weights=omega if scenario.weights else None,
    )
    weights = omega / omega.mean() / noise**2
    spec = oracle.OracleSpec(
        t=t, y=y, w=weights, t_ref=t_ref, matrix=matrix, constant=constant,
        amp=amp_modes, phase=phase_modes,
        vcodes=vcodes if scenario.variation else None,
        offsets=(ocodes if scenario.offsets == "groups" else scenario.offsets),
    )
    truth_vector = np.array([truth[key] for key in spec.layout])
    return Built(t, y, error, t_ref, constraints, start,
                 scenario.offsets != "none", spec, truth_vector, roots)


def package_params(fit: MultiSineFit, built: Built) -> np.ndarray:
    """The package's solution in the oracle's parameter order."""
    detail = fit.detail
    assert detail is not None
    out = []
    for kind, a, g in built.spec.layout:
        if kind == "f":
            out.append(fit.frequency[built.roots[a]])
        elif kind in ("A", "phi"):
            table = detail.subset_amplitude if kind == "A" else detail.subset_phase
            headline = fit.amplitude if kind == "A" else fit.phase
            if table is None:
                out.append(headline[a])
            else:
                out.append(table[a, max(g, 0)])
        elif g < 0:
            out.append(fit.offset)
        else:
            assert detail.subset_offset is not None
            out.append(detail.subset_offset[g])
    return np.asarray(out)


def expected_names(built: Built) -> tuple[str, ...]:
    names = []
    for kind, a, g in built.spec.layout:
        if kind == "f":
            names.append(f"f[{built.roots[a]}]")
        elif kind == "o":
            names.append("offset" if g < 0 else f"offset[{g}]")
        else:
            names.append(f"{kind}[{a}]" if g < 0 else f"{kind}[{a}|{g}]")
    return tuple(names)


def run_package(built: Built, **kwargs: Any) -> MultiSineFit:
    options: dict[str, Any] = {"refine": "simultaneous", "max_nfev": 500}
    options.update(kwargs)
    # A free scenario would take the legacy path; force it through the general kernel.
    fitter = fit_constrained if built.constraints.is_trivial() else fit_multisine
    return fitter(  # type: ignore[operator]
        built.t, built.y, built.error, built.start, fit_mean=built.fit_mean,
        t_ref=built.t_ref, constraints=built.constraints, **options,
    )


def _phase_aware_difference(a: np.ndarray, b: np.ndarray,
                            layout: list[tuple[str, int, int]]) -> np.ndarray:
    diff = a - b
    for i, (kind, _, _) in enumerate(layout):
        if kind == "phi":
            diff[i] = (diff[i] + np.pi) % TWO_PI - np.pi
    return diff


# --- the oracle comparison -----------------------------------------------------------


@pytest.mark.parametrize("baseline", ["short", "long"])
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_matches_the_oracle(scenario: Scenario, baseline: str) -> None:
    built = build(scenario, baseline)
    fit = run_package(built)
    detail = fit.detail
    assert detail is not None
    assert detail.parameter_names == expected_names(built)
    assert fit.n_parameters == built.spec.n_params
    assert np.allclose(fit.residuals, built.y - fit.model(
        built.t, offset_groups=built.constraints.offset_groups,
        variation_groups=built.constraints.variation_groups), atol=1e-12)

    if built.spec.n_params == 0:
        expected = built.spec.rss(np.zeros(0))
        assert fit.rss == pytest.approx(expected, rel=1e-12)
        assert fit.covariance.shape == (6, 6)
        assert np.all(fit.covariance == 0.0)
        return

    assert oracle.jacobian_error(built.spec, built.truth) < 1e-4
    ref = oracle.solve(built.spec, built.truth, seed=7)
    assert ref.status > 0
    assert ref.step_sigma < 1e-4

    assert abs(fit.rss - ref.rss) / ref.rss < 1e-9
    mine = package_params(fit, built)
    diff = _phase_aware_difference(mine, ref.params, built.spec.layout)
    assert np.max(np.abs(diff) / ref.sigma) < 1e-3
    sigma = np.sqrt(np.diag(detail.parameter_covariance))
    assert np.max(np.abs(sigma / ref.sigma - 1.0)) < 1e-5
    corr = detail.parameter_covariance / np.outer(sigma, sigma)
    ref_corr = ref.covariance / np.outer(ref.sigma, ref.sigma)
    assert np.max(np.abs(corr - ref_corr)) < 1e-5
    assert detail.rank == detail.n_columns == built.spec.n_params
    assert np.isfinite(detail.condition) and detail.condition >= 1.0


def test_the_oracle_jacobian_is_analytic_and_exact() -> None:
    for baseline in ("short", "long"):
        built = build(SCENARIOS[-1], baseline)
        assert oracle.jacobian_error(built.spec, built.truth) < 1e-4


# --- trivial constraints through the general kernel ----------------------------------


def _free_data(seed: int = 1) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    n = 500
    t = np.sort(rng.uniform(0, 30, n)) + 2459000.0
    f = np.array([3.1, 5.7, 8.8])
    a = np.array([0.02, 0.012, 0.006])
    p = np.array([0.4, 2.0, 1.0])
    y = 10 + sum(a[k] * np.sin(TWO_PI * f[k] * (t - 2459015.0) + p[k])
                 for k in range(3))
    y = y + rng.normal(0, 0.002, n)
    return t, y, np.full(n, 0.002), f + np.array([0.003, -0.002, 0.004])


@pytest.mark.parametrize("error", [True, False])
@pytest.mark.parametrize("refine", ["simultaneous", "none", "cyclic", "last"])
def test_trivial_constraints_reproduce_the_legacy_fit(refine: str, error: bool) -> None:
    t, y, e, start = _free_data()
    err = e if error else None
    legacy = fit_multisine(t, y, err, start, refine=refine, sweeps=3)
    general = fit_constrained(t, y, err, start, constraints=FitConstraints.free(3),
                              refine=refine, sweeps=3)
    assert general.t_ref == legacy.t_ref
    assert general.n_parameters == legacy.n_parameters
    if refine in ("simultaneous", "none"):
        assert np.allclose(general.frequency, legacy.frequency, rtol=1e-10, atol=0)
        assert np.allclose(general.amplitude, legacy.amplitude, rtol=1e-10, atol=0)
        assert np.allclose(general.phase, legacy.phase, rtol=1e-10, atol=0)
        assert general.offset == pytest.approx(legacy.offset, rel=1e-10)
        assert general.rss == pytest.approx(legacy.rss, rel=1e-10)
        assert np.allclose(general.covariance, legacy.covariance, rtol=1e-8,
                           atol=1e-10 * np.max(np.abs(legacy.covariance)))
    else:
        sigma = legacy.frequency_error
        assert np.max(np.abs(general.frequency - legacy.frequency) / sigma) < 1e-6
        assert np.max(np.abs(general.amplitude - legacy.amplitude)
                      / legacy.amplitude_error) < 1e-6


def test_trivial_constraints_take_the_legacy_path() -> None:
    t, y, e, start = _free_data()
    for constraints in (FitConstraints(), FitConstraints.free(3)):
        fit = fit_multisine(t, y, e, start, constraints=constraints)
        assert fit.detail is None


# --- Kaufman gradient -----------------------------------------------------------------


def test_kaufman_gradient_matches_finite_differences() -> None:
    built = build(SCENARIOS[-1], "short")
    fit = run_package(built, refine="none", covariance=False)
    assert fit.detail is not None
    # Rebuild the internal problem away from the optimum and compare J^T r with the
    # finite-difference gradient of 1/2 |r|^2 over [theta; psi].
    problem, alpha = _problem_for(built)
    alpha = alpha + np.r_[np.full(len(built.roots), 0.02 / 122.0),
                          np.full(alpha.size - len(built.roots), 0.05)]
    residual = problem.fun(alpha)
    gradient = problem.jac(alpha).T @ residual
    numeric = np.empty_like(gradient)
    for i in range(alpha.size):
        h = 1e-7 / 122.0 if i < len(built.roots) else 1e-6
        up, down = alpha.copy(), alpha.copy()
        up[i] += h
        down[i] -= h
        numeric[i] = (0.5 * np.sum(problem.fun(up) ** 2)
                      - 0.5 * np.sum(problem.fun(down) ** 2)) / (2 * h)
    assert np.max(np.abs(gradient - numeric) / np.abs(numeric)) < 1e-6


def _problem_for(built: Built) -> tuple[Any, np.ndarray]:
    """The full variable-projection problem of a scenario at its start."""
    captured: dict[str, Any] = {}
    original = _constrained._Solver.run

    def spy(self: Any, refine: str, sweeps: int, theta: np.ndarray,
            psi: np.ndarray) -> Any:
        captured["solver"] = self
        captured["theta"] = theta.copy()
        captured["psi"] = psi.copy()
        return original(self, refine, sweeps, theta, psi)

    _constrained._Solver.run = spy  # type: ignore[method-assign]
    try:
        run_package(built, refine="none", covariance=False)
    finally:
        _constrained._Solver.run = original  # type: ignore[method-assign]
    solver = captured["solver"]
    p = solver.s.n_theta
    problem = solver.full(captured["theta"], captured["psi"], active=range(p),
                          psi_active=solver.s.n_psi > 0)
    return problem, problem.pack()


# --- covariance of ties, headline map, rms amplitude ----------------------------------


def test_tied_sigma_is_the_propagated_parent_covariance() -> None:
    built = build(SCENARIOS[1], "short")  # f3 = f1 + f2
    fit = run_package(built)
    detail = fit.detail
    assert detail is not None
    c_theta = detail.parameter_covariance[:2, :2]
    row = np.array([1.0, 1.0])
    assert fit.frequency_error[2] == pytest.approx(np.sqrt(row @ c_theta @ row),
                                                   rel=1e-12)
    assert fit.covariance[0, 2] == pytest.approx(c_theta[0, 0] + c_theta[0, 1],
                                                 rel=1e-12)
    assert fit.frequency[2] == pytest.approx(fit.frequency[0] + fit.frequency[1],
                                             rel=1e-15)
    assert "tied" in detail.flags[2]


def _headline_map(fit: MultiSineFit, k: int, amp: np.ndarray,
                  phase: np.ndarray) -> tuple[float, float]:
    vcodes = fit.detail.constraints.variation_groups  # type: ignore[union-attr]
    w = np.bincount(vcodes, minlength=amp.size).astype(float) / 0.002**2
    zbar = np.sum(w * amp * np.exp(1j * phase)) / w.sum()
    return abs(zbar), float(np.angle(zbar) % TWO_PI)


@pytest.mark.parametrize("name", ["amplitude-per-subset", "phase-per-subset",
                                  "both-per-subset"])
def test_headline_rows_match_central_differences(name: str) -> None:
    scenario = next(s for s in SCENARIOS if s.name == name)
    built = build(scenario, "short")
    fit = run_package(built)
    detail = fit.detail
    assert detail is not None
    assert detail.subset_amplitude is not None and detail.subset_phase is not None
    k = 2
    amp, phase = detail.subset_amplitude[k].copy(), detail.subset_phase[k].copy()
    a0, p0 = _headline_map(fit, k, amp, phase)
    assert fit.amplitude[k] == pytest.approx(a0, rel=1e-12)
    assert fit.phase[k] == pytest.approx(p0, abs=1e-12)
    # sigma(A_k)^2 from the delta method equals the CD-propagated parameter variance.
    names = detail.parameter_names
    columns = [i for i, nm in enumerate(names) if nm.startswith((f"A[{k}", f"phi[{k}"))]
    jac = np.zeros((2, len(names)))
    for col in columns:
        nm = names[col]
        g = int(nm.split("|")[1][:-1]) if "|" in nm else None
        h = 1e-7
        for sign in (1.0, -1.0):
            a2, p2 = amp.copy(), phase.copy()
            target = a2 if nm.startswith("A") else p2
            if g is None:
                target += sign * h
            else:
                target[g] += sign * h
            a_h, p_h = _headline_map(fit, k, a2, p2)
            jac[0, col] += sign * a_h / (2 * h)
            jac[1, col] += sign * (((p_h - p0 + np.pi) % TWO_PI) - np.pi) / (2 * h)
    cov = jac @ detail.parameter_covariance @ jac.T
    assert fit.amplitude_error[k] == pytest.approx(np.sqrt(cov[0, 0]), rel=1e-6)
    assert fit.phase_error[k] == pytest.approx(np.sqrt(cov[1, 1]), rel=1e-6)
    # rms amplitude: the per-subset RMS for the varying component, |A| otherwise.
    w = np.bincount(built.constraints.variation_groups).astype(float)
    assert detail.rms_amplitude[k] == pytest.approx(
        np.sqrt(np.sum(w * amp**2) / w.sum()), rel=1e-12)
    assert detail.rms_amplitude[0] == pytest.approx(abs(fit.amplitude[0]), rel=1e-15)
    # A common component's headline row is the identity.
    assert fit.amplitude_error[0] == pytest.approx(
        np.sqrt(detail.parameter_covariance[names.index("A[0]"),
                                            names.index("A[0]")]), rel=1e-12)


# --- linear solves --------------------------------------------------------------------


def test_indicator_designs_never_reach_the_normal_equations(monkeypatch) -> None:
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0.0, 4.0, 400))  # four nights, a 0.3 c/d component
    ocodes = np.minimum(t.astype(int), 3)
    y = 0.05 * np.sin(TWO_PI * 0.3 * t + 0.5) + 0.01 * ocodes + rng.normal(0, 1e-3,
                                                                            t.size)
    seen: list[np.ndarray] = []
    original = fit_module._solve

    def spy(design: np.ndarray, rhs: np.ndarray) -> Any:
        seen.append(design.copy())
        return original(design, rhs)

    monkeypatch.setattr(fit_module, "_solve", spy)
    constraints = FitConstraints(offset_groups=ocodes)
    fit = fit_multisine(t, y, None, [0.3], refine="none", t_ref=2.0,
                        constraints=constraints)
    for design in seen:
        support = np.count_nonzero(design, axis=0)
        assert np.all(support == design.shape[0])
    # beta against an SVD solve of the same design.
    theta = TWO_PI * 0.3 * (t - 2.0)
    design = np.column_stack([np.cos(theta), np.sin(theta)]
                             + [(ocodes == g).astype(float) for g in range(4)])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    detail = fit.detail
    assert detail is not None and detail.subset_offset is not None
    assert np.allclose(detail.subset_offset, beta[2:], rtol=0, atol=1e-12)
    a, b = beta[0], beta[1]
    assert fit.amplitude[0] == pytest.approx(np.hypot(a, b), rel=1e-12)
    assert "low_frequency" in detail.flags[0]


# --- starting phases ------------------------------------------------------------------


@pytest.mark.parametrize("name", ["amplitude-per-subset", "phase-per-subset",
                                  "fixed-amplitude-phase-per-subset"])
def test_start_rebuilds_a_stored_solution(name: str) -> None:
    scenario = next(s for s in SCENARIOS if s.name == name)
    built = build(scenario, "long")
    stored = run_package(built)
    detail = stored.detail
    assert detail is not None
    start = FitStart(amplitude=stored.amplitude, phase=stored.phase,
                     subset_amplitude=detail.subset_amplitude,
                     subset_phase=detail.subset_phase)
    rebuilt = fit_multisine(
        built.t, built.y, built.error, stored.frequency, refine="none",
        t_ref=built.t_ref, constraints=replace(built.constraints, start=start),
    )
    assert rebuilt.detail is not None
    assert np.allclose(rebuilt.frequency, stored.frequency, rtol=0, atol=0)
    assert np.allclose(rebuilt.detail.subset_amplitude, detail.subset_amplitude,
                       rtol=1e-9, atol=1e-12)
    assert np.allclose(rebuilt.detail.subset_phase, detail.subset_phase,
                       rtol=1e-9, atol=1e-9)
    assert rebuilt.rss == pytest.approx(stored.rss, rel=1e-9)


# --- edge-case table (spec 1.8) -------------------------------------------------------


def test_fixed_phase_needs_t_ref() -> None:
    built = build(SCENARIOS[8], "short")  # fixed phase
    with pytest.raises(ConstraintError, match="explicit t_ref"):
        fit_multisine(built.t, built.y, built.error, built.start,
                      constraints=built.constraints)


def test_negative_amplitude_under_a_fixed_phase_is_signed_and_flagged() -> None:
    built = build(next(s for s in SCENARIOS if s.name == "fixed-phase-pi"), "short")
    fit = run_package(built)
    assert fit.amplitude[2] < 0.0
    assert fit.amplitude[2] == pytest.approx(-0.006, rel=0.15)
    assert fit.phase[2] == pytest.approx((5.0 + np.pi) % TWO_PI)
    assert fit.phase_error[2] == 0.0
    assert np.isfinite(fit.amplitude_error[2]) and fit.amplitude_error[2] > 0
    assert {"negative_amplitude", "fixed_phi"} <= fit.detail.flags[2]  # type: ignore[union-attr]


def test_fixed_parameters_report_zero_error() -> None:
    built = build(next(s for s in SCENARIOS if s.name == "psi-only"), "short")
    fit = run_package(built)
    assert np.all(fit.frequency_error == 0.0)
    assert fit.amplitude_error[0] == 0.0
    assert fit.amplitude_error[1] > 0.0 and fit.phase_error[0] > 0.0
    flags = fit.detail.flags  # type: ignore[union-attr]
    assert {"fixed_f", "fixed_a"} <= flags[0]
    assert flags[1] == {"fixed_f"}
    assert {"fixed_f", "fixed_a"} <= flags[2] and "weak_frequency" not in flags[2]


def test_tie_to_fixed_parents_is_constant() -> None:
    built = build(SCENARIOS[3], "short")  # f3 = f1 + f2 with f2 fixed
    fit = run_package(built)
    assert fit.frequency[2] == pytest.approx(fit.frequency[0] + F2, rel=1e-15)
    fixed_parent = FitConstraints(components=(
        ComponentConstraint(frequency="fixed"),
        ComponentConstraint(frequency="fixed"),
        ComponentConstraint(frequency=FrequencyTie((0, 1), (1, 1))),
    ))
    fit2 = fit_multisine(built.t, built.y, built.error, [F1, F2, 0.0],
                         refine="simultaneous", t_ref=built.t_ref,
                         constraints=fixed_parent)
    assert fit2.frequency[2] == pytest.approx(F1 + F2, rel=1e-15)
    assert {"tied", "fixed_f"} <= fit2.detail.flags[2]  # type: ignore[union-attr]
    assert np.all(fit2.frequency_error == 0.0)


def test_degenerate_design_reports_nan_and_rank() -> None:
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0.0, 20.0, 300))
    y = 0.01 * np.sin(TWO_PI * 3.0 * t) + rng.normal(0, 1e-3, t.size)
    # Two fixed frequencies one ulp apart: the pair columns are collinear.
    fixed = ComponentConstraint(frequency="fixed")
    constraints = FitConstraints(components=(fixed, fixed))
    fit = fit_multisine(t, y, None, [3.0, np.nextafter(3.0, 4.0)], refine="none",
                        t_ref=10.0, constraints=constraints)
    detail = fit.detail
    assert detail is not None
    assert detail.rank < detail.n_columns
    assert "degenerate" in detail.flags[0] and "degenerate" in detail.flags[1]
    assert np.isnan(fit.amplitude_error[0])
    assert np.all(np.isfinite(fit.residuals))


def test_close_tie_and_out_of_band_flags() -> None:
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0.0, 20.0, 400))
    y = (0.02 * np.sin(TWO_PI * 4.0 * t) + 0.01 * np.sin(TWO_PI * 8.01 * t)
         + rng.normal(0, 1e-3, t.size))
    constraints = FitConstraints(components=(
        ComponentConstraint(),
        ComponentConstraint(),
        ComponentConstraint(frequency=FrequencyTie((0,), (2,))),
    ))
    fit = fit_multisine(t, y, None, [4.0, 8.01, 0.0], refine="none", t_ref=10.0,
                        frequency_bounds=(1.0, 7.0), constraints=constraints)
    flags = fit.detail.flags  # type: ignore[union-attr]
    assert "unresolved" in flags[2]  # 8.0 is within 0.5/T of 8.01
    assert "out_of_band" in flags[2]  # outside the (1, 7) box


def test_short_subset_and_weak_frequency_flags() -> None:
    rng = np.random.default_rng(6)
    t = np.sort(rng.uniform(0.0, 20.0, 400))
    vcodes = (t > 19.7).astype(int)  # a 0.3-day subset: shorter than one cycle
    y = 0.02 * np.sin(TWO_PI * 1.5 * t + 0.3) + rng.normal(0, 1e-3, t.size)
    constraints = FitConstraints(
        components=(ComponentConstraint(variation="phase"),),
        variation_groups=vcodes, min_subset_points=3,
    )
    fit = fit_multisine(t, y, None, [1.5], refine="simultaneous", t_ref=10.0,
                        constraints=constraints)
    assert {"short_subset", "weak_frequency"} <= fit.detail.flags[0]  # type: ignore[union-attr]


def test_at_bound_flag() -> None:
    rng = np.random.default_rng(8)
    t = np.sort(rng.uniform(0.0, 20.0, 400))
    y = 0.02 * np.sin(TWO_PI * 5.0 * t) + rng.normal(0, 1e-3, t.size)
    constraints = FitConstraints(point_weights=np.ones(t.size))
    fit = fit_multisine(t, y, None, [4.99], refine="simultaneous", t_ref=10.0,
                        frequency_bounds=(4.98, 4.995), constraints=constraints)
    assert fit.frequency[0] == pytest.approx(4.995)
    assert "at_bound" in fit.detail.flags[0]  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"offset_groups": np.zeros(10, dtype=int)}, "one code per point"),
        ({"components": (ComponentConstraint(variation="phase"),)},
         "no variation_groups"),
    ],
)
def test_validation_errors_surface_from_the_fit(kwargs, message: str) -> None:
    t = np.linspace(0, 10, 50)
    with pytest.raises(ConstraintError, match=message):
        fit_multisine(t, np.sin(t), None, [0.2], t_ref=5.0,
                      constraints=FitConstraints(**kwargs))


def test_offset_groups_need_fit_mean() -> None:
    t = np.linspace(0, 10, 50)
    with pytest.raises(ConstraintError, match="fit_mean=True"):
        fit_multisine(t, np.sin(t), None, [0.2], fit_mean=False, t_ref=5.0,
                      constraints=FitConstraints(offset_groups=np.repeat([0, 1], 25)))


def test_nothing_to_fit_is_refused() -> None:
    t = np.linspace(0, 10, 50)
    with pytest.raises(ValueError, match="at least one frequency"):
        fit_multisine(t, np.sin(t), None, [], fit_mean=False,
                      constraints=FitConstraints(point_weights=np.ones(50)))


def test_offsets_alone() -> None:
    t = np.linspace(0, 10, 50)
    codes = np.repeat([0, 1], 25)
    y = np.where(codes == 0, 1.0, 2.0) + 0.01 * np.cos(t)
    fit = fit_multisine(t, y, None, [], constraints=FitConstraints(offset_groups=codes))
    assert fit.detail is not None and fit.detail.subset_offset is not None
    assert np.allclose(fit.detail.subset_offset, [1.0, 2.0], atol=0.01)
    assert fit.n_parameters == 2


def test_nothing_is_infinite_across_scenarios() -> None:
    for scenario in SCENARIOS:
        built = build(scenario, "short")
        fit = run_package(built, refine="last")
        for array in (fit.frequency, fit.amplitude, fit.phase, fit.residuals,
                      fit.frequency_error, fit.amplitude_error, fit.phase_error):
            assert not np.any(np.isinf(array)), scenario.name
        assert np.isfinite(fit.rss) and np.isfinite(fit.bic)


# --- invariances ----------------------------------------------------------------------


def _sink_with_fixed_frequency_under_the_fixed_phase() -> Scenario:
    # A phase fixed at one epoch is the same constraint at another epoch only for a
    # fixed frequency: with a free one the frame would drag the phase along.
    sink = SCENARIOS[-1]
    comps = list(sink.comps)
    comps[4] = replace(comps[4], freq="fixed")
    return replace(sink, comps=tuple(comps))


@pytest.mark.parametrize(
    "scenario",
    [_sink_with_fixed_frequency_under_the_fixed_phase(), SCENARIOS[1]],
    ids=["sink", "tie"],
)
def test_shifting_t_ref_re_references_the_phases(scenario: Scenario) -> None:
    built = build(scenario, "short")
    fit_a = run_package(built)
    shift = 3.7  # t_ref moves 3.7 d earlier: every phase drops by 2 pi f shift
    moved_components = tuple(
        replace(comp, phase=None if comp.phase is None
                else comp.phase - TWO_PI * fit_a.frequency[k] * shift)
        for k, comp in enumerate(built.constraints.components)
    )
    moved = replace(built, t_ref=built.t_ref - shift,
                    constraints=replace(built.constraints,
                                        components=moved_components),
                    start=fit_a.frequency.copy())
    fit_b = run_package(moved)
    assert fit_b.rss == pytest.approx(fit_a.rss, rel=1e-9)
    assert np.allclose(fit_b.frequency, fit_a.frequency, rtol=1e-10)
    expected = (fit_a.phase - TWO_PI * fit_a.frequency * shift) % TWO_PI
    diff = (fit_b.phase - expected + np.pi) % TWO_PI - np.pi
    assert np.max(np.abs(diff) / fit_a.phase_error.clip(1e-300)) < 1e-4
    assert np.allclose(fit_b.amplitude_error, fit_a.amplitude_error, rtol=1e-6)
    assert np.allclose(fit_b.frequency_error, fit_a.frequency_error, rtol=1e-6)


def test_relabelling_subsets_changes_nothing() -> None:
    built = build(SCENARIOS[-1], "short")
    fit_a = run_package(built)
    c = built.constraints
    assert c.offset_groups is not None and c.variation_groups is not None
    o_perm = np.array([2, 0, 3, 1])
    v_perm = np.array([1, 2, 0])
    relabelled = replace(c, offset_groups=o_perm[c.offset_groups],
                         variation_groups=v_perm[c.variation_groups])
    fit_b = run_package(replace(built, constraints=relabelled))
    assert fit_b.rss == pytest.approx(fit_a.rss, rel=1e-9)
    assert np.allclose(fit_b.frequency, fit_a.frequency, rtol=1e-10)
    assert np.allclose(fit_b.amplitude, fit_a.amplitude, rtol=1e-7)
    assert np.allclose(fit_b.amplitude_error, fit_a.amplitude_error, rtol=1e-6)
    da, db = fit_a.detail, fit_b.detail
    assert da is not None and db is not None
    assert np.allclose(db.subset_offset[o_perm], da.subset_offset, rtol=1e-9)  # type: ignore[index]
    assert np.allclose(db.subset_amplitude[:, v_perm], da.subset_amplitude,  # type: ignore[index]
                       rtol=1e-7)


def test_scaling_all_weights_changes_nothing() -> None:
    built = build(SCENARIOS[-1], "short")
    fit_a = run_package(built)
    c = built.constraints
    assert c.point_weights is not None
    fit_b = run_package(replace(built, constraints=replace(
        c, point_weights=c.point_weights * 7.5)))
    assert fit_b.rss == pytest.approx(fit_a.rss, rel=1e-10)
    assert np.allclose(fit_b.frequency, fit_a.frequency, rtol=1e-12)
    assert np.allclose(fit_b.covariance, fit_a.covariance, rtol=1e-8,
                       atol=1e-12 * np.nanmax(np.abs(fit_a.covariance)))


def test_splitting_a_subset_without_variation_changes_nothing() -> None:
    built = build(SCENARIOS[1], "short")  # a tie, a single offset, no variation
    fit_a = run_package(built)
    split = np.minimum((built.t - built.t.min()) // 30.0, 4).astype(int)
    constraints = replace(built.constraints, variation_groups=split)
    fit_b = run_package(replace(built, constraints=constraints))
    assert fit_b.rss == pytest.approx(fit_a.rss, rel=1e-12)
    assert np.allclose(fit_b.frequency, fit_a.frequency, rtol=1e-12)
    assert np.allclose(fit_b.covariance, fit_a.covariance, rtol=1e-8,
                       atol=1e-12 * np.max(np.abs(fit_a.covariance)))


# --- Monte-Carlo calibration ----------------------------------------------------------


def _mc(make: Callable[[np.random.Generator], tuple[Any, ...]], n_real: int,
        pick: Callable[[MultiSineFit], tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    values, sigmas = [], []
    for i in range(n_real):
        fit = make(np.random.default_rng(1000 + i))
        v, s = pick(fit)
        values.append(v)
        sigmas.append(s)
    values_a, sigmas_a = np.array(values), np.array(sigmas)
    spread = values_a.std(axis=0, ddof=1)
    if values_a.ndim > 1:
        spread = np.where(np.ptp(values_a, axis=0) > np.pi, np.nan, spread)
    return np.median(sigmas_a, axis=0) / spread


def test_monte_carlo_calibration() -> None:
    """Predicted / empirical scatter in [0.75, 1.35] over 60 noise realisations."""
    t = np.sort(np.random.default_rng(0).uniform(0.0, 60.0, 300))
    vcodes = np.minimum((t // 20.0).astype(int), 2)
    t_ref = 30.0

    def make(rng: np.random.Generator) -> MultiSineFit:
        amp = np.array([0.008, 0.010, 0.012])
        y = (0.02 * np.sin(TWO_PI * 3.1 * (t - t_ref) + 0.4)
             + 0.012 * np.sin(TWO_PI * 4.7 * (t - t_ref) + 1.0)
             + 0.006 * np.sin(TWO_PI * 7.8 * (t - t_ref) + 2.0)
             + amp[vcodes] * np.sin(TWO_PI * 9.9 * (t - t_ref) + 1.3)
             + rng.normal(0, 0.003, t.size))
        constraints = FitConstraints(
            components=(ComponentConstraint(), ComponentConstraint(),
                        ComponentConstraint(frequency=FrequencyTie((0, 1), (1, 1))),
                        ComponentConstraint(variation="amplitude")),
            variation_groups=vcodes,
        )
        return fit_multisine(t, y, None, [3.1, 4.7, 0.0, 9.9], refine="simultaneous",
                             t_ref=t_ref, constraints=constraints)

    def pick(fit: MultiSineFit) -> tuple[np.ndarray, np.ndarray]:
        d = fit.detail
        assert d is not None and d.subset_amplitude is not None
        assert d.subset_amplitude_error is not None
        names = d.parameter_names
        phi = names.index("phi[3]")
        sigma_phi = np.sqrt(d.parameter_covariance[phi, phi])
        value = np.r_[fit.frequency[2], d.subset_amplitude[3], fit.phase[3]]
        sigma = np.r_[fit.frequency_error[2], d.subset_amplitude_error[3], sigma_phi]
        return value, sigma

    ratio = _mc(make, 60, pick)
    assert np.all((ratio > 0.75) & (ratio < 1.35)), ratio


# --- more rows of the edge-case table ------------------------------------------------


def test_everything_fixed_still_solves_the_zero_points() -> None:
    for name, n_par in (("everything-fixed", 1), ("everything-fixed-offsets", 4),
                        ("everything-fixed-no-mean", 0)):
        built = build(next(s for s in SCENARIOS if s.name == name), "short")
        fit = run_package(built)
        assert fit.n_parameters == n_par
        k = fit.n_components
        assert np.all(fit.covariance[: 3 * k, :] == 0.0)
        if built.fit_mean:
            assert fit.offset_error > 0.0
            assert fit.offset == pytest.approx(10.0, abs=1e-3)
        assert fit.detail is not None and fit.detail.nfev == 1


def test_relative_weights_without_error_bars() -> None:
    built = build(next(s for s in SCENARIOS if s.name == "two-partitions-weights"),
                  "short")
    fit = fit_multisine(built.t, built.y, None, built.start, t_ref=built.t_ref,
                        refine="simultaneous", constraints=built.constraints)
    assert fit.weighted is False
    omega = built.constraints.point_weights
    assert omega is not None
    w = omega / omega.mean()
    assert fit.rss == pytest.approx(float(np.sum(w * fit.residuals**2)), rel=1e-12)
    n, m = fit.n_samples, fit.n_parameters
    assert fit.bic == pytest.approx(n * np.log(fit.rss / n) + m * np.log(n),
                                    rel=1e-12)


def _varying(phases: np.ndarray, amps: np.ndarray, vary: str) -> MultiSineFit:
    rng = np.random.default_rng(12)
    t = np.sort(rng.uniform(0.0, 120.0, 1200))
    vcodes = np.minimum((t // 20.0).astype(int), 5)
    y = amps[vcodes] * np.sin(TWO_PI * 2.3 * (t - 60.0) + phases[vcodes])
    y = y + rng.normal(0, 0.01, t.size)
    constraints = FitConstraints(components=(ComponentConstraint(variation=vary),),
                                 variation_groups=vcodes)
    return fit_multisine(t, y, None, [2.3], refine="none", t_ref=60.0,
                         constraints=constraints)


def test_incoherent_phases_collapse_the_headline_but_not_the_rms() -> None:
    reversal = np.array([0.5, 0.5, 0.5, 0.5 + np.pi, 0.5 + np.pi, 0.5 + np.pi])
    fit = _varying(reversal, np.ones(6), "phase")
    detail = fit.detail
    assert detail is not None
    assert fit.amplitude[0] < 0.2
    assert detail.rms_amplitude[0] == pytest.approx(1.0, rel=0.01)
    assert detail.subset_phase is not None
    assert np.allclose(np.cos(detail.subset_phase[0] - reversal), 1.0, atol=1e-3)


def test_negative_subset_amplitude_under_a_common_phase() -> None:
    amps = np.array([1.0, 0.9, -0.8, 1.1, 1.0, 0.95])
    fit = _varying(np.full(6, 0.5), amps, "amplitude")
    detail = fit.detail
    assert detail is not None and detail.subset_amplitude is not None
    assert np.allclose(detail.subset_amplitude[0], amps, atol=5e-3)
    assert fit.phase[0] == pytest.approx(0.5, abs=5e-3)
    # The global sign is canonical: the same signal written with every amplitude
    # flipped and the phase moved by pi gives the same signed table and phase.
    flipped = _varying(np.full(6, 0.5 + np.pi), -amps, "amplitude")
    assert flipped.detail is not None
    assert np.allclose(flipped.detail.subset_amplitude, detail.subset_amplitude,
                       atol=1e-9)
    assert np.cos(flipped.phase[0] - fit.phase[0]) == pytest.approx(1.0, abs=1e-9)
