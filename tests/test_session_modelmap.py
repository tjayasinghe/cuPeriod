"""State -> UsedData / FitConstraints / FitStart, and fit -> state (no solver)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.core.columns import Domain
from cuperiod.prewhiten.constraints import FitDetail, FrequencyTie
from cuperiod.prewhiten.fit import MultiSineFit
from cuperiod.prewhiten.session import modelmap
from cuperiod.prewhiten.session.sampling import used_data
from cuperiod.prewhiten.session.state import (
    Component,
    FitStats,
    PointMask,
    SessionData,
    SessionState,
    Tie,
    UsedData,
    content_sha256,
)
from cuperiod.prewhiten.subsets import Partition

N = 60
T_REF = 15.0


def _data(errors: bool = True) -> SessionData:
    t = np.linspace(0.0, 30.0, N)
    y = np.sin(t)
    e = np.full(N, 0.1) if errors else None
    return SessionData(
        time=t, value=y, error=e, domain=Domain.MAGNITUDE,
        source_index=np.arange(N), labels={}, meta={},
        content_sha256=content_sha256(t, y, e),
    )


def _thirds(name: str = "season", weights: tuple[float, ...] = (1.0, 1.0, 1.0)
            ) -> Partition:
    return Partition.from_runs(N, [(0, 20, 0), (20, 40, 1), (40, 60, 2)],
                               name=name, labels=("A", "B", "C"), weights=weights)


def _state(components: tuple[Component, ...], **kw: object) -> SessionState:
    base: dict[str, object] = dict(
        serial=1, components=components, offset=0.0, subset_offsets=None,
        stats=FitStats(1.0, 0.1, 0.0, N, 1, True), mask=PointMask.none(N),
        partitions=(), excluded=(), offsets_by=None, variation_by=None,
        weights_by=None, settings=cup.PreWhitenSettings(), grid=(0.1, 0.01, 100),
        blocked=(), n_iterations=0, budget_origin=0, n_pruned=0,
        status="searching", stop_reason="", next_id=10,
    )
    base.update(kw)
    return SessionState(**base)  # type: ignore[arg-type]


C1 = Component(id=1, frequency=1.0, amplitude=0.5, phase=0.1)
C2 = Component(id=2, frequency=1.7, amplitude=0.3, phase=0.2)
C3 = Component(id=3, frequency=2.7, amplitude=0.1, phase=0.3)


def test_plain_state_maps_to_none_and_the_data_arrays() -> None:
    data = _data()
    state = _state((C1, C2))
    used = used_data(data, state)
    assert used.time is data.time and used.value is data.value
    assert used.error is data.error and used.error_eff is data.error
    assert used.point_weights is None and used.offset_codes is None
    full = UsedData.full(data)
    assert full.time is data.time and full.error is data.error
    mm = modelmap.build(state, used, t_ref=T_REF)
    assert mm.constraints is None and mm.ids == (1, 2)
    np.testing.assert_array_equal(mm.frequencies, [1.0, 1.7])
    assert mm.start is None


def test_only_an_inactive_untied_component_is_trivial_but_keeps_the_id_map() -> None:
    state = _state((C1, dataclasses.replace(C2, active=False), C3))
    mm = modelmap.build(state, used_data(_data(), state), t_ref=T_REF)
    assert mm.constraints is None and mm.ids == (1, 3)


def test_ties_fixes_and_inactive_parents() -> None:
    c3 = dataclasses.replace(C3, tie=Tie(parents=(1, 2), coefficients=(1, 1)))
    c2 = dataclasses.replace(C2, fix_amplitude=True, fix_phase=True)
    state = _state((dataclasses.replace(C1, fix_frequency=True), c2, c3))
    mm = modelmap.build(state, used_data(_data(), state), t_ref=T_REF)
    assert mm.constraints is not None
    comps = mm.constraints.components
    assert comps[0].frequency == "fixed"
    assert comps[1].amplitude == 0.3 and comps[1].phase == 0.2
    assert comps[2].frequency == FrequencyTie(parents=(0, 1), coefficients=(1, 1))
    assert mm.inactive_parent == frozenset()
    assert mm.start is not None and mm.start.amplitude.tolist() == [0.5, 0.3, 0.1]

    # Parent F2 inactive: its 1 * f2 folds into the tie's constant.
    inactive = _state((C1, dataclasses.replace(C2, active=False), c3))
    mm2 = modelmap.build(inactive, used_data(_data(), inactive), t_ref=T_REF)
    assert mm2.ids == (1, 3)
    assert mm2.constraints is not None
    tie = mm2.constraints.components[1].frequency
    assert tie == FrequencyTie(parents=(0,), coefficients=(1,), constant=1.7)
    assert mm2.inactive_parent == frozenset({3})

    # Every parent inactive: the child is held at the constant.
    orphan = _state((dataclasses.replace(C1, active=False),
                     dataclasses.replace(C2, active=False), c3))
    mm3 = modelmap.build(orphan, used_data(_data(), orphan), t_ref=T_REF)
    assert mm3.constraints is not None
    assert mm3.constraints.components[0].frequency == "fixed"
    assert mm3.frequencies.tolist() == [pytest.approx(2.7)]


def test_extra_free_and_hold() -> None:
    state = _state((C1, dataclasses.replace(C2, fix_phase=True)))
    mm = modelmap.build(state, used_data(_data(), state), t_ref=T_REF, extra_free=1,
                        hold={1})
    assert mm.constraints is not None
    assert len(mm.constraints.components) == 3
    assert mm.constraints.components[0].frequency == "fixed"
    assert mm.constraints.components[2].is_free
    assert mm.start is not None and mm.start.amplitude.shape == (3,)
    held = modelmap.build(_state((C1, C2)), used_data(_data(), _state((C1, C2))),
                          t_ref=T_REF, hold={2})
    assert held.constraints is not None
    assert held.constraints.components[1].frequency == "fixed"


def test_deleted_points_excluded_subsets_and_compacted_codes() -> None:
    data = _data()
    season = _thirds()
    night = Partition.from_runs(N, [(0, 30, 0), (30, 60, 1)], name="night",
                                labels=("N1", "N2"))
    varying = dataclasses.replace(
        C1, variation="amplitude", subset_amplitude=(0.4, 0.5, 0.6)
    )
    state = _state(
        (varying, C2),
        partitions=(season, night),
        mask=PointMask.from_runs(N, [(0, 2)]),
        excluded=(("season", (1,)),),  # B excluded: 20 points out
        variation_by="season",
        offsets_by="night",
    )
    used = used_data(data, state)
    assert used.n == N - 2 - 20
    assert used.index.tolist() == [*range(2, 20), *range(40, 60)]
    np.testing.assert_array_equal(used.time, data.time[used.index])
    assert not used.used.flags.writeable and not used.index.flags.writeable
    assert used.variation_labels == ("A", "C")
    assert used.variation_codes is not None
    assert set(used.variation_codes.tolist()) == {0, 1}
    assert used.variation_subsets is not None
    assert used.variation_subsets.tolist() == [0, 2]
    assert used.offset_labels == ("N1", "N2")
    mm = modelmap.build(state, used, t_ref=T_REF, min_subset_points=5)
    assert mm.constraints is not None
    assert mm.constraints.variation_groups is used.variation_codes
    assert mm.constraints.offset_groups is used.offset_codes
    assert mm.constraints.min_subset_points == 5
    assert mm.start is not None and mm.start.subset_amplitude is not None
    np.testing.assert_array_equal(mm.start.subset_amplitude[0], [0.4, 0.6])
    assert np.isnan(mm.start.subset_amplitude[1]).all()
    assert mm.n_variation_subsets == 3
    mm.constraints.validate(np.array([1.0, 1.7]), used.n, fit_mean=True, t_ref=T_REF)


def test_variation_groups_only_when_something_varies() -> None:
    state = _state((C1,), partitions=(_thirds(),), variation_by="season")
    mm = modelmap.build(state, used_data(_data(), state), t_ref=T_REF)
    assert mm.constraints is None


@pytest.mark.parametrize("errors", [True, False])
def test_subset_weights(errors: bool) -> None:
    data = _data(errors)
    state = _state((C1,), partitions=(_thirds(weights=(1.0, 2.0, 4.0)),),
                   weights_by="season")
    used = used_data(data, state)
    assert used.point_weights is not None
    assert float(np.mean(used.point_weights)) == pytest.approx(1.0)
    omega = np.repeat([1.0, 2.0, 4.0], 20) / (7.0 / 3.0)
    np.testing.assert_allclose(used.point_weights, omega)
    sigma = data.error if errors else 1.0
    np.testing.assert_allclose(used.error_eff, sigma / np.sqrt(omega))
    assert used.error is data.error  # the fit keeps the error bars
    mm = modelmap.build(state, used, t_ref=T_REF)
    assert mm.constraints is not None
    assert mm.constraints.point_weights is used.point_weights


def _fit(frequency: list[float], amplitude: list[float], phase: list[float],
         detail: FitDetail | None = None) -> MultiSineFit:
    return MultiSineFit(
        frequency=np.array(frequency), amplitude=np.array(amplitude),
        phase=np.array(phase), offset=0.25, t_ref=T_REF, residuals=np.zeros(N),
        covariance=np.zeros((0, 0)), rss=1.0, n_samples=N, n_parameters=7,
        weighted=True, detail=detail,
    )


def test_apply_writes_back_by_id() -> None:
    state = _state((C1, dataclasses.replace(C2, active=False), C3))
    mm = modelmap.build(state, used_data(_data(), state), t_ref=T_REF)
    update = modelmap.apply(state, _fit([1.01, 2.71], [0.6, -0.2], [1.0, 2.0]), mm)
    by_id = {c.id: c for c in update.components}
    assert by_id[1].frequency == 1.01 and by_id[1].amplitude == 0.6
    assert by_id[3].amplitude == -0.2  # signed, as the fit returned it
    assert by_id[2] is state.components[1]  # inactive: untouched
    assert update.offset == 0.25 and update.subset_offsets is None


def test_apply_expands_subset_values_onto_the_full_partition() -> None:
    season = _thirds()
    night = Partition.from_runs(N, [(0, 30, 0), (30, 60, 1)], name="night",
                                labels=("N1", "N2"))
    state = _state(
        (dataclasses.replace(C1, variation="phase"), C2),
        partitions=(season, night), excluded=(("season", (1,)),),
        variation_by="season", offsets_by="night",
    )
    used = used_data(_data(), state)
    mm = modelmap.build(state, used, t_ref=T_REF, min_subset_points=5)
    assert mm.constraints is not None
    nan = np.nan
    detail = FitDetail(
        constraints=mm.constraints, parameter_names=(), parameter_covariance=np.zeros(
            (0, 0)), flags=(frozenset(), frozenset()),
        rms_amplitude=np.array([0.5, 0.3]),
        subset_amplitude=np.array([[0.5, 0.5], [0.3, 0.3]]),
        subset_amplitude_error=np.full((2, 2), nan),
        subset_phase=np.array([[0.1, 0.4], [0.2, 0.2]]),
        subset_phase_error=np.full((2, 2), 0.01),
        subset_offset=np.array([0.01, -0.02]),
        subset_offset_error=np.array([0.001, 0.001]),
        rank=8, n_columns=8, condition=1.0, nfev=3,
    )
    update = modelmap.apply(state, _fit([1.0, 1.7], [0.5, 0.3], [0.2, 0.2], detail),
                            mm)
    c1, c2 = update.components
    assert c1.subset_amplitude is None  # amplitude common
    assert c1.subset_phase is not None
    assert c1.subset_phase[0] == 0.1 and np.isnan(c1.subset_phase[1])
    assert c1.subset_phase[2] == 0.4
    assert c2.subset_phase is None and c2.subset_amplitude is None
    assert update.subset_offsets == (0.01, -0.02)
