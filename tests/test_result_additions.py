"""The trailing, defaulted fields Sinusoid / PreWhitenResult gained for sessions."""

from __future__ import annotations

import json
import math

import numpy as np

from cuperiod.prewhiten.result import PreWhitenResult, Sinusoid, SubsetSolution

V13_KEYS = [
    "rank",
    "label",
    "frequency",
    "frequency_error",
    "period",
    "period_error",
    "amplitude",
    "amplitude_error",
    "spectrum_amplitude",
    "amplitude_ratio",
    "blended",
    "phase",
    "phase_error",
    "snr",
    "fap",
    "delta_bic",
    "combination",
]


def _sinusoid(**kw: object) -> Sinusoid:
    base: dict[str, object] = dict(
        rank=1,
        label="F1",
        frequency=2.0,
        frequency_error=1e-4,
        amplitude=0.1,
        amplitude_error=0.01,
        phase=1.0,
        phase_error=0.1,
        snr=12.0,
        fap=1e-9,
        delta_bic=-50.0,
        spectrum_amplitude=0.1,
    )
    base.update(kw)
    return Sinusoid(**base)  # type: ignore[arg-type]


def _result(components: tuple[Sinusoid, ...]) -> PreWhitenResult:
    return PreWhitenResult(
        components=components,
        combinations=(),
        offset=0.0,
        offset_error=0.0,
        t_ref=0.0,
        time=np.zeros(3),
        residuals=np.zeros(3),
        n_samples=3,
        baseline=1.0,
        stop_reason="x",
        n_iterations=1,
        n_pruned=0,
        rms=0.0,
        chi2=0.0,
        reduced_chi2=0.0,
        bic=0.0,
        correlation_factor=1.0,
        uncertainty_method="covariance",
        backend="numpy",
    )


def test_defaults_keep_the_v13_row() -> None:
    c = _sinusoid()
    assert list(c.to_dict()) == V13_KEYS
    assert c.fixed == "" and c.tied is None and c.variation == "none"
    assert c.amplitude_rms is None and c.flags == ()
    assert not c.has_constraint
    r = _result((c,))
    assert [list(row) for row in r.to_table()] == [V13_KEYS]
    assert r.to_table() == [c.to_dict()]
    assert r.subsets is None and r.n_deleted == 0
    assert list(r.to_dict()["components"][0]) == V13_KEYS


def test_constraint_keys_appear_only_when_asked_or_present() -> None:
    plain = _sinusoid()
    fixed = _sinusoid(rank=2, label="F2", fixed="fp", flags=("fixed_f", "fixed_phi"))
    assert list(plain.to_dict(constraints=True))[len(V13_KEYS) :] == [
        "fixed",
        "tied",
        "variation",
        "amplitude_rms",
        "flags",
    ]
    assert fixed.to_dict(constraints=True)["flags"] == "fixed_f,fixed_phi"
    assert list(_result((plain,)).to_table()[0]) == V13_KEYS
    both = _result((plain, fixed)).to_table()
    assert all("fixed" in row for row in both)
    assert list(_result((plain, fixed)).to_table(constraints=False)[1]) == V13_KEYS
    tied = _sinusoid(tied="F1 + F2")
    varying = _sinusoid(variation="phase", amplitude_rms=0.2)
    assert tied.has_constraint and varying.has_constraint


def test_value_equality_still_holds() -> None:
    assert _sinusoid() == _sinusoid()
    assert _sinusoid(flags=("tied",)) != _sinusoid()


def test_subset_solution_round_trip() -> None:
    nan = float("nan")
    s = SubsetSolution(
        offsets_by="night",
        offset_labels=("N1", "N2", "N3"),
        offset=np.array([0.1, nan, -0.2]),
        offset_error=np.array([0.01, nan, 0.02]),
        variation_by="season",
        variation_labels=("S1", "S2"),
        amplitude=np.array([[0.1, -0.05], [nan, nan]]),
        amplitude_error=np.array([[0.01, 0.02], [nan, nan]]),
        phase=np.array([[1.0, 2.0], [nan, nan]]),
        phase_error=np.array([[0.1, 0.2], [nan, nan]]),
    )
    data = s.to_dict()
    back = SubsetSolution.from_dict(json.loads(json.dumps(data)))
    assert back.to_dict().keys() == data.keys()
    for name in ("offset", "offset_error", "amplitude", "phase", "phase_error"):
        np.testing.assert_array_equal(getattr(back, name), getattr(s, name))
    assert back.offset_labels == s.offset_labels and back.variation_by == "season"
    assert math.isnan(back.amplitude[1, 0])


def test_subset_solution_without_variation() -> None:
    s = SubsetSolution(
        offsets_by="night",
        offset_labels=("N1",),
        offset=np.array([0.1]),
        offset_error=np.array([0.01]),
        variation_by=None,
        variation_labels=(),
        amplitude=np.zeros((2, 0)),
        amplitude_error=np.zeros((2, 0)),
        phase=np.zeros((2, 0)),
        phase_error=np.zeros((2, 0)),
    )
    back = SubsetSolution.from_dict(s.to_dict())
    assert back.variation_by is None and back.offset.tolist() == [0.1]
