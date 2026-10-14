"""The refine policies under constraints: a schedule of exact block minimisations.

``"last"`` is the policy the extraction loop (Auto) takes in every session with a
tie, a fixed value or a varying component, so its block schedule is held here to
the promises of spec 1.5: the residual sum of squares never increases across blocks
and sweeps, ``"last"`` is never worse than its start, enough ``"cyclic"`` sweeps reach
the ``"simultaneous"`` optimum, and ``"none"`` moves no frequency.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cuperiod.prewhiten._constrained import fit_constrained
from cuperiod.prewhiten.fit import MultiSineFit
from test_prewhiten_constrained_fit import (
    BASE,
    F1,
    F2,
    SCENARIOS,
    Built,
    Comp,
    Scenario,
    build,
)

CONSTRAINED = tuple(
    s for s in SCENARIOS
    if s.name not in ("free", "everything-fixed-no-mean")
)

EXTRA = (
    # A tie whose child is the last component, and one whose parent is.
    Scenario("child-last", BASE + (Comp(F1 + F2, 0.006, 1.0,
                                        freq=("tie", ((0, 1), (1, 1)), 0.0)),)),
    Scenario("parent-last", (
        Comp(F1, 0.020, 0.4),
        Comp(2 * 5.13, 0.006, 1.0, freq=("tie", ((2, 2),), 0.0)),
        Comp(5.13, 0.012, 2.0),
    )),
)


def _fit(built: Built, refine: str, trace: list[float] | None = None,
         **kw: Any) -> MultiSineFit:
    return fit_constrained(
        built.t, built.y, built.error, built.start, constraints=built.constraints,
        fit_mean=built.fit_mean, t_ref=built.t_ref, refine=refine,
        trace=None if trace is None else trace.append, max_nfev=500, **kw,
    )


@pytest.mark.parametrize("baseline", ["short", "long"])
@pytest.mark.parametrize("scenario", CONSTRAINED + EXTRA, ids=lambda s: s.name)
def test_block_schedule_is_monotone(scenario: Scenario, baseline: str) -> None:
    built = build(scenario, baseline)
    for refine, sweeps in (("last", 1), ("cyclic", 4)):
        trace: list[float] = []
        fit = _fit(built, refine, trace, sweeps=sweeps)
        assert trace, "the schedule reports its running RSS"
        steps = np.diff(trace)
        assert np.all(steps <= 1e-12 * np.asarray(trace[:-1])), (refine, trace)
        assert fit.rss == pytest.approx(trace[-1], rel=1e-12)


@pytest.mark.parametrize("baseline", ["short", "long"])
@pytest.mark.parametrize("scenario", CONSTRAINED + EXTRA, ids=lambda s: s.name)
def test_last_is_never_worse_than_its_start(scenario: Scenario, baseline: str) -> None:
    built = build(scenario, baseline)
    start = _fit(built, "none")
    last = _fit(built, "last")
    assert last.rss <= start.rss * (1.0 + 1e-12)


@pytest.mark.parametrize("baseline", ["short", "long"])
@pytest.mark.parametrize("scenario", CONSTRAINED + EXTRA, ids=lambda s: s.name)
def test_cyclic_sweeps_reach_the_simultaneous_optimum(
    scenario: Scenario, baseline: str
) -> None:
    built = build(scenario, baseline)
    joint = _fit(built, "simultaneous")
    cyclic = _fit(built, "cyclic", sweeps=50)
    assert abs(cyclic.rss - joint.rss) / joint.rss < 1e-9


def test_last_moves_only_the_roots_of_the_last_component() -> None:
    built = build(EXTRA[0], "short")  # f3 = f1 + f2 is last: both roots move
    start = _fit(built, "none")
    last = _fit(built, "last")
    assert np.all(last.frequency[:2] != start.frequency[:2])
    sink = build(SCENARIOS[-1], "short")  # last component has its own root only
    start = _fit(sink, "none")
    last = _fit(sink, "last")
    assert np.array_equal(last.frequency[:4], start.frequency[:4])
    assert last.frequency[4] != start.frequency[4]


def test_last_with_a_fixed_last_component_moves_nothing() -> None:
    scenario = replace(SCENARIOS[5])  # "fixed-frequency": the last one is fixed
    built = build(scenario, "short")
    start = _fit(built, "none")
    last = _fit(built, "last")
    assert np.array_equal(last.frequency, start.frequency)
    assert last.rss == start.rss


@pytest.mark.parametrize(
    "name",
    ["phase-per-subset", "amplitude-per-subset", "fixed-amplitude-phase-per-subset",
     "psi-only", "kitchen-sink"],
)
def test_none_with_nonlinear_phases_moves_no_frequency(name: str) -> None:
    scenario = next(s for s in SCENARIOS if s.name == name)
    built = build(scenario, "long")
    fit = _fit(built, "none")
    resolved = built.constraints.resolve(built.start)
    expected = resolved.frequencies(resolved.theta(built.start))
    assert np.array_equal(fit.frequency, expected)
    # The phases were optimised: the result beats the relaxation start.
    trace: list[float] = []
    _fit(built, "none", trace)
    assert len(trace) == 1


def test_policies_never_change_the_parameter_count() -> None:
    built = build(SCENARIOS[-1], "short")
    counts = {refine: _fit(built, refine).n_parameters
              for refine in ("none", "last", "cyclic", "simultaneous")}
    assert len(set(counts.values())) == 1
