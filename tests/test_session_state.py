"""Session state types: content equality, digests, PointMask, classification."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.core.columns import Domain
from cuperiod.prewhiten.session.state import (
    Component,
    Derived,
    FitStats,
    LogEntry,
    PointMask,
    SessionData,
    SessionError,
    SessionState,
    StepReport,
    Tie,
    UsedData,
    content_sha256,
    parse_ref,
)
from cuperiod.prewhiten.subsets import Partition

N = 40


def make_state(**overrides: object) -> SessionState:
    base: dict[str, object] = dict(
        serial=1,
        components=(
            Component(id=1, frequency=1.5, amplitude=0.2, phase=0.3, fap=1e-9),
            Component(id=2, frequency=3.25, amplitude=0.05, phase=2.0),
        ),
        offset=0.01,
        subset_offsets=None,
        stats=FitStats(rss=1.0, rms=0.1, bic=-5.0, n_samples=N, n_parameters=7,
                       weighted=True),
        mask=PointMask.none(N),
        partitions=(),
        excluded=(),
        offsets_by=None,
        variation_by=None,
        weights_by=None,
        settings=cup.PreWhitenSettings(backend="numpy"),
        grid=(0.1, 0.01, 500),
        blocked=(),
        n_iterations=3,
        budget_origin=0,
        n_pruned=0,
        status="searching",
        stop_reason="",
        next_id=3,
    )
    base.update(overrides)
    return SessionState(**base)  # type: ignore[arg-type]


def test_same_content_ignores_serial_only() -> None:
    a = make_state()
    b = make_state(serial=99)
    assert a.same_content(b) and a.content_digest() == b.content_digest()
    assert a.same_content(a)


@pytest.mark.parametrize(
    "change",
    [
        {"offset": 0.02},
        {"n_iterations": 4},
        {"budget_origin": 1},
        {"n_pruned": 1},
        {"status": "stopped"},
        {"stop_reason": "x"},
        {"next_id": 4},
        {"blocked": (1.0,)},
        {"grid": (0.1, 0.01, 501)},
        {"offsets_by": "night"},
        {"variation_by": "night"},
        {"weights_by": "night"},
        {"subset_offsets": (0.1,)},
        {"excluded": (("night", (1,)),)},
        {"settings": cup.PreWhitenSettings(backend="numpy", snr_threshold=5.0)},
        {"mask": PointMask.from_runs(N, [(0, 2)])},
        {"stats": FitStats(rss=1.5, rms=0.1, bic=-5.0, n_samples=N, n_parameters=7,
                           weighted=True)},
        {"partitions": (Partition.from_runs(N, [(0, 20, 0), (20, 40, 1)],
                                            name="night", labels=("A", "B")),)},
    ],
)
def test_same_content_sees_every_other_field(change: dict[str, object]) -> None:
    a = make_state()
    b = make_state(**change)
    assert not a.same_content(b)
    assert a.content_digest() != b.content_digest()


def test_component_fields_and_nan_compare_by_value() -> None:
    a = make_state()
    comps = list(a.components)
    comps[1] = dataclasses.replace(comps[1], note="x")
    assert not a.same_content(make_state(components=tuple(comps)))
    # NaN fap on both sides: same content (bitwise), although nan != nan.
    assert math.isnan(a.components[1].fap)
    assert a.same_content(make_state())
    # One ulp away from the stored amplitude is a different state.
    comps[1] = dataclasses.replace(a.components[1], amplitude=0.05000000000000001)
    assert not a.same_content(make_state(components=tuple(comps)))


def test_partitions_compare_by_codes_not_identity() -> None:
    p1 = Partition.from_runs(N, [(0, 20, 0), (20, 40, 1)], name="night",
                             labels=("A", "B"))
    p2 = Partition.from_runs(N, [(0, 20, 0), (20, 40, 1)], name="night",
                             labels=("A", "B"))
    p3 = Partition.from_runs(N, [(0, 21, 0), (21, 40, 1)], name="night",
                             labels=("A", "B"))
    assert make_state(partitions=(p1,)).same_content(make_state(partitions=(p2,)))
    assert not make_state(partitions=(p1,)).same_content(make_state(partitions=(p3,)))


def test_classification() -> None:
    plain = make_state()
    assert plain.is_plain()
    assert not plain.has_model_constraints() and not plain.has_data_edits()
    for comp_change in (
        {"active": False},
        {"fix_frequency": True},
        {"fix_amplitude": True},
        {"fix_phase": True},
        {"tie": Tie(parents=(1,), coefficients=(2,))},
        {"variation": "amplitude"},
    ):
        comps = (plain.components[0],
                 dataclasses.replace(plain.components[1], **comp_change))
        state = make_state(components=comps)
        assert state.has_model_constraints() and not state.has_data_edits()
    assert make_state(offsets_by="night").has_model_constraints()
    assert make_state(mask=PointMask.from_runs(N, [(3, 4)])).has_data_edits()
    assert make_state(excluded=(("night", (0,)),)).has_data_edits()
    assert not make_state(excluded=(("night", ()),)).has_data_edits()
    assert make_state(weights_by="night").has_data_edits()


def test_component_lookup() -> None:
    s = make_state()
    assert s.component(2).frequency == 3.25
    assert s.component("F1").id == 1 and s.component("f2").id == 2
    assert s.components[0].label == "F1"
    with pytest.raises(SessionError, match="F9"):
        s.component(9)
    with pytest.raises(SessionError):
        parse_ref("G1")
    with pytest.raises(SessionError):
        parse_ref(True)  # type: ignore[arg-type]


def test_point_mask_round_trip_and_ops() -> None:
    mask = PointMask.from_runs(N, [(0, 3), (10, 12), (39, 40)])
    assert mask.runs() == [(0, 3), (10, 12), (39, 40)]
    assert mask.n_deleted == 6
    assert not mask.deleted.flags.writeable
    again = PointMask.from_runs(N, mask.runs())
    assert again.digest == mask.digest
    more = mask.without(np.array([5, 6]))
    assert more.runs() == [(0, 3), (5, 7), (10, 12), (39, 40)]
    assert more.restored(np.array([5, 6])).digest == mask.digest
    assert more.restored().digest == PointMask.none(N).digest
    boolean = np.zeros(N, dtype=bool)
    boolean[20] = True
    assert mask.without(boolean).runs()[-2] == (20, 21)
    assert PointMask.none(N).runs() == []
    with pytest.raises(SessionError):
        PointMask.from_runs(N, [(5, 5)])
    with pytest.raises(SessionError):
        PointMask.from_runs(N, [(38, 41)])
    with pytest.raises(SessionError):
        mask.without(np.array([N]))


def test_session_data_readonly_views_leave_the_source_writable() -> None:
    t = np.linspace(0, 1, 5)
    y = np.ones(5)
    data = SessionData(
        time=t, value=y, error=None, domain=Domain.MAGNITUDE,
        source_index=np.arange(5), labels={"band": np.array(["V"] * 5, dtype=object)},
        meta={}, content_sha256=content_sha256(t, y, None),
    )
    view = data.readonly()
    assert not view.time.flags.writeable and not view.labels["band"].flags.writeable
    assert t.flags.writeable and y.flags.writeable
    assert np.shares_memory(view.time, t)
    full = UsedData.full(data)
    assert full.time is t and full.value is y and full.error is None
    assert full.is_full and full.index.tolist() == list(range(5))
    assert content_sha256(t, y, None) != content_sha256(t, y, np.ones(5))


def test_report_and_entry_round_trip() -> None:
    report = StepReport(
        kind="accepted", message="F1 at 1.5", component_id=1, frequency=1.5,
        statistics={"snr": 12.0, "fap": float("nan")},
    )
    entry = LogEntry(seq=1, group=1, op="step", args={"seed": 1.5, "index": 3},
                     report=report, timestamp="2026-10-14T00:00:00Z")
    data = entry.to_dict()
    assert data["report"]["statistics"] == {"fap": None, "snr": 12.0}
    back = LogEntry.from_dict(data)
    assert back.report.kind == "accepted" and back.report.frequency == 1.5
    assert math.isnan(back.report.statistics["fap"])
    assert back.args == {"seed": 1.5, "index": 3}


def test_derived_materialises_the_residual_spectrum_once() -> None:
    from cuperiod.core.grid import uniform_frequency_grid
    from cuperiod.prewhiten.spectrum import SpectrumEngine

    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 20, 200))
    y = np.sin(2 * np.pi * 1.3 * t)
    engine = SpectrumEngine(t, None, grid=uniform_frequency_grid(20.0,
                            maximum_frequency=5.0), backend="numpy")
    from cuperiod.prewhiten import fit_multisine

    fit = fit_multisine(t, y, None, np.zeros(0), refine="none")
    derived = Derived(fit=fit, scan=engine._scan(y), sampling_key="k")
    first = derived.residual_spectrum
    assert derived.residual_spectrum is first
    np.testing.assert_array_equal(first.amplitude, engine.spectrum(y).amplitude)
