"""PreWhitenSession: the manual component operations (add, remove, fix, tie, ...)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import prewhiten_corpus as pc
from cuperiod.core.config import SessionSettings
from cuperiod.prewhiten.fit import fit_multisine
from cuperiod.prewhiten.session import PreWhitenSession
from cuperiod.prewhiten.session.state import StepReport
from cuperiod.prewhiten.subsets import Partition

CASE = pc.case_by_name("triple-last")
SETTINGS = CASE.resolved_settings()
DRIFT = pc.case_by_name("drift-blocked")


def _data(case: pc.Case = CASE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (t, y, e), _, _ = pc.load(case)
    assert e is not None
    return t, y, e


def _halves(t: np.ndarray) -> Partition:
    return Partition.from_ranges(t, [float(np.median(t))], name="half")


def _three(**kw: Any) -> PreWhitenSession:
    """The triple after three accepted steps: F1 12.34, F2 17.81, F3 24.68 (= 2 F1)."""
    t, y, e = _data()
    session = PreWhitenSession(
        (t, y, e), settings=SETTINGS, partitions=[_halves(t)], **kw
    )
    session.run(max_steps=3)
    assert [c.id for c in session.components] == [1, 2, 3]
    return session


def _refused(
    session: PreWhitenSession, call: Callable[[], StepReport], match: str = ""
) -> StepReport:
    """The error contract: refused, nothing committed or logged, head unchanged."""
    head, log = session.state, session.history
    report = call()
    assert report.kind == "refused", report
    assert re.search(match, report.message), report.message
    assert session.state is head and session.history == log
    return report


def _freqs(session: PreWhitenSession) -> np.ndarray:
    return np.array([c.frequency for c in session.components])


# --- add / calculate ----------------------------------------------------------------


def test_add_then_calculate_is_the_v13_linear_fit() -> None:
    session = _three()
    report = session.add(5.0, snap=False, refine="none")
    assert report.kind == "edited" and report.component_id == 4
    assert report.frequency == 5.0 and {"components", "fit", "ids"} <= report.changed
    new = session.state.component(4)
    assert new.origin == "manual" and new.frequency == 5.0
    assert np.isfinite(new.delta_bic) and np.isnan(new.fap)
    (entry,) = session.history[-1:]
    assert entry.op == "add" and entry.args["frequency"] == 5.0
    assert session.calculate().kind == "edited"
    t, y, e = _data()
    fit = fit_multisine(
        t, y, e, _freqs(session), fit_mean=True, t_ref=session.t_ref, refine="none"
    )
    head = session.state
    np.testing.assert_array_equal([c.amplitude for c in head.components], fit.amplitude)
    np.testing.assert_array_equal([c.phase for c in head.components], fit.phase)
    np.testing.assert_array_equal(session.fit.residuals, fit.residuals)
    assert head.offset == fit.offset


def test_add_snaps_to_the_residual_peak_and_refines() -> None:
    t, y, e = _data()
    session = PreWhitenSession((t, y, e), settings=SETTINGS)
    session.run(max_steps=2)
    report = session.add(24.66)  # snapped within 1/T, then refined ("last")
    assert report.kind == "edited"
    logged = session.history[-1].args["frequency"]
    assert logged != 24.66 and abs(logged - 24.68) < 0.02
    assert abs(session.state.component(3).frequency - 24.68) < 0.005
    exact = PreWhitenSession((t, y, e), settings=SETTINGS)
    exact.run(max_steps=2)
    assert exact.add(24.66, snap=False, refine="none").frequency == 24.66


def test_add_refusals() -> None:
    session = _three()
    f1 = session.state.component(1).frequency
    _refused(session, lambda: session.add(f1 + 0.001, snap=False), "allow_close")
    _refused(session, lambda: session.add(100.0, snap=False), "outside the frequency")
    _refused(session, lambda: session.add(-1.0, snap=False), "frequency")
    _refused(session, lambda: session.add(float("nan"), snap=False), "frequency")
    close = session.add(f1 + 0.001, snap=False, refine="none", allow_close=True)
    assert close.kind == "edited" and close.frequency == f1 + 0.001
    assert session.add(40.0, snap=False, refine="none").kind == "edited"


def test_add_peak_walks_the_spectrum() -> None:
    session = PreWhitenSession(_data(DRIFT), settings=DRIFT.resolved_settings())
    for _ in range(3):
        assert session.add_peak().kind == "accepted"
    assert session.state.components[-1].origin == "manual"
    seeds: list[float] = []
    kinds = []
    for _ in range(3):
        before = session.state
        report = session.add_peak()
        kinds.append(report.kind)
        after = session.state
        assert after.n_iterations == before.n_iterations + 1
        assert session.history[-1].op == "add_peak"
        seed = session.history[-1].args["seed"]
        assert seed not in seeds  # a blocked seed is never tried again
        seeds.append(seed)
        if report.kind == "blocked":
            assert after.blocked == (*before.blocked, seed)
            assert after.budget_origin == before.budget_origin
            assert after.status == before.status
            assert len(after.components) == len(before.components)
        else:
            assert report.kind == "accepted"
            assert after.blocked == before.blocked  # adding never un-collides a peak
            assert after.budget_origin == after.n_iterations
    assert kinds == ["blocked", "blocked", "accepted"]


def test_add_peak_without_a_peak_is_refused() -> None:
    case = pc.case_by_name("narrow-no-peak")
    (t, y, e), cfg, grid = pc.load(case)
    session = PreWhitenSession((t, y, e), settings=cfg, grid=grid)
    assert session.step().kind == "accepted"
    _refused(session, session.add_peak, "no peak")
    _refused(session, lambda: session.do("add_peak", seed=5.0), "together")


# --- remove / set_active ----------------------------------------------------------


def test_deactivate_is_remove_for_the_fit() -> None:
    off, gone = _three(), _three()
    f2 = off.state.component(2).frequency
    assert off.set_active(2, False).kind == "edited"
    assert gone.remove(2).kind == "edited"
    np.testing.assert_array_equal(off.fit.residuals, gone.fit.residuals)
    np.testing.assert_array_equal(off.fit.amplitude, gone.fit.amplitude)
    inactive = off.state.component(2)
    assert not inactive.active and inactive.frequency == f2  # keeps its values
    candidate = off.candidate()
    assert candidate is None or abs(candidate.frequency - f2) > off.kernel.separation(
        off.state
    )
    assert [c.id for c in gone.components] == [1, 3]
    assert off.set_active([2], True).kind == "edited"
    np.testing.assert_allclose(off.fit.residuals, _three().fit.residuals, atol=1e-12)


def test_ids_are_stable_under_remove_and_compacted_by_renumber() -> None:
    session = _three()
    assert session.add(5.0, snap=False, refine="none").component_id == 4
    report = session.tie(4, "F3 - F2")
    assert report.kind == "edited", report
    assert session.remove(1).kind == "edited"
    assert [c.id for c in session.components] == [2, 3, 4]
    assert session.add(8.0, snap=False, refine="none").component_id == 5
    assert session.renumber().kind == "edited"
    assert [c.id for c in session.components] == [1, 2, 3, 4]
    tied = session.state.component(3)
    assert tied.tie is not None and tied.tie.text() == "F2 - F1"
    assert session.state.next_id == 5


# --- fix / set_values ---------------------------------------------------------------


def test_fix_and_free() -> None:
    session = _three()
    f1 = session.state.component(1).frequency
    assert session.fix(1, frequency=True).kind == "edited"
    assert session.state.component(1).fix_frequency
    assert session.improve().kind == "edited"
    assert session.state.component(1).frequency == f1  # held bitwise
    result = session.result()
    assert result[0].fixed == "f" and result[0].frequency_error == 0.0
    assert "fixed_f" in result[0].flags
    assert session.fix(1, frequency=False).kind == "edited"
    assert not session.state.component(1).is_constrained
    assert session.fix([1, 2], amplitude=True, phase=True).kind == "edited"
    a1 = session.state.component(1).amplitude
    assert session.calculate().kind == "edited"
    assert session.state.component(1).amplitude == a1
    assert session.result()[1].fixed == "Ap"


def test_set_values() -> None:
    session = _three()
    c2 = session.state.component(2)
    assert session.set_values(2, frequency=17.9).kind == "edited"
    edited = session.state.component(2)
    assert edited.frequency == 17.9 and not edited.fix_frequency
    assert edited.amplitude != c2.amplitude  # re-solved at the new frequency
    # A free amplitude is re-solved by the implicit Calculate; a fixed one is kept.
    a1 = session.state.component(1).amplitude
    session.set_values(1, amplitude=0.02)
    assert session.state.component(1).amplitude == pytest.approx(a1, rel=1e-6)
    session.fix(1, amplitude=True, phase=True)
    session.set_values(1, amplitude=0.02, phase=7.0)
    c1 = session.state.component(1)
    assert c1.amplitude == 0.02 and c1.phase == pytest.approx(7.0 - 2 * np.pi)
    assert session.set_values(1, amplitude=0.02).kind == "noop"
    f1 = c1.frequency
    _refused(session, lambda: session.set_values(2, frequency=f1 + 1e-4), "Rayleigh")
    _refused(session, lambda: session.set_values(9, amplitude=1.0), "F9")
    # A frequency unties the row.
    session.tie(3, "2F1")
    assert session.set_values(3, frequency=24.7).kind == "edited"
    assert session.state.component(3).tie is None


# --- ties ---------------------------------------------------------------------------


def test_tie_sets_the_predicted_frequency_and_reports() -> None:
    session = _three()
    f1 = session.state.component(1).frequency
    report = session.tie(3, "2F1")
    assert report.kind == "edited" and report.frequency == 2 * f1
    c3 = session.state.component(3)
    assert c3.tie is not None and c3.tie.text() == "2F1"
    assert c3.frequency == pytest.approx(2 * session.state.component(1).frequency)
    session.improve()
    head = session.state
    assert head.component(3).frequency == pytest.approx(
        2 * head.component(1).frequency, rel=1e-14
    )
    row = session.result()[2]
    assert row.tied == "2F1" and row.combination == "F3 = 2F1" and "tied" in row.flags
    assert session.tie(3, "F3 = 2*f1").kind == "noop"
    assert session.tie(3, None).kind == "edited"
    assert session.state.component(3).tie is None
    assert session.tie(3, None).kind == "noop"


def test_tie_refusals() -> None:
    session = _three()
    _refused(session, lambda: session.tie(1, "0.5F2"), "non-integer")
    _refused(session, lambda: session.tie(1, "F2 F3"), "missing operator")
    _refused(session, lambda: session.tie(1, "F2 +"), "trailing")
    _refused(session, lambda: session.tie(1, "F2 - F2"), "zero")
    _refused(session, lambda: session.tie(1, "F9"), "F9")
    _refused(session, lambda: session.tie(1, "F1 + F2"), "cyclic")
    _refused(session, lambda: session.tie(1, "F2 = F3"), "not the edited")
    _refused(session, lambda: session.tie(1, "21F2"), "order")
    _refused(session, lambda: session.tie(1, "F2 + F3"), "outside the frequency")
    _refused(session, lambda: session.tie(2, "2F1"), "allow_close")  # lands on F3
    _refused(session, lambda: session.tie(1, "import os"), "labels")
    assert session.tie(3, "2F1").kind == "edited"
    _refused(session, lambda: session.tie(1, "F3 - F2"), r"cyclic tie: F\d")
    _refused(session, lambda: session.fix(3, frequency=True), "untie")
    session.add(29.0, snap=False, refine="none")
    _refused(  # the same tie twice: identical design columns
        session, lambda: session.tie(4, "2F1", allow_close=True), "identical"
    )


def test_tie_parents_removal_inactivity_and_cascade() -> None:
    session = _three()
    session.tie(3, "2F1")
    _refused(session, lambda: session.remove(1), "cascade")
    tied_f = session.state.component(3).frequency
    assert session.set_active(1, False).kind == "edited"
    held = session.state.component(3)
    assert held.frequency == tied_f  # the inactive parent holds the child exactly
    assert "parent_inactive" in session.result()[1].flags
    session.set_active(1, True)
    assert session.remove(1, cascade=True).kind == "edited"
    child = session.state.component(3)
    assert child.tie is None and not child.fix_frequency
    assert [c.id for c in session.components] == [2, 3]


def test_a_fixed_row_gives_way_to_a_tie() -> None:
    session = _three()
    session.fix(3, frequency=True)
    assert session.tie(3, "2F1").kind == "edited"
    c3 = session.state.component(3)
    assert c3.tie is not None and not c3.fix_frequency


# --- improve ------------------------------------------------------------------------


def test_improve_all_is_the_polish() -> None:
    manual, auto = _three(), _three()
    assert manual.improve().kind == "edited"
    assert auto.polish().kind == "edited"
    for a, b in zip(manual.components, auto.components, strict=True):
        assert a.frequency == b.frequency and a.amplitude == b.amplitude
        assert a.phase == b.phase
    assert not manual.is_pristine and auto.is_pristine


def test_improve_selected_holds_the_others_and_reports_free_errors() -> None:
    session = _three()
    before = _freqs(session)
    assert session.improve(["F2"]).kind == "edited"
    after = _freqs(session)
    assert after[0] == before[0] and after[2] == before[2]
    assert after[1] != before[1]
    assert not any(c.is_constrained for c in session.components)
    improved = session.result()
    assert all(c.frequency_error > 0.0 and c.fixed == "" for c in improved)
    session.calculate()
    recalculated = session.result()
    np.testing.assert_allclose(
        [c.frequency_error for c in improved],
        [c.frequency_error for c in recalculated],
        rtol=1e-9,
    )
    _refused(session, lambda: session.improve(["F9"]), "F9")
    session.fix(2, frequency=True)
    _refused(session, lambda: session.improve([2]), "no free frequency")
    session.set_active(1, False)
    _refused(session, lambda: session.improve([1]), "inactive")


# --- variation ----------------------------------------------------------------------


def test_set_variation_needs_a_partition_and_holds_the_frequency() -> None:
    session = _three()
    _refused(session, lambda: session.set_variation(1, "phase"), "use_variation")
    assert session.use_variation("half").kind == "edited"
    assert session.set_variation(1, "phase").kind == "edited"
    c1 = session.state.component(1)
    assert c1.variation == "phase" and c1.fix_frequency
    assert c1.subset_phase is not None and len(c1.subset_phase) == 2
    assert session.set_variation(2, "amplitude", hold_frequency=False).kind == "edited"
    assert not session.state.component(2).fix_frequency
    _refused(session, lambda: session.fix(1, phase=True), "per subset")
    _refused(session, lambda: session.fix(2, amplitude=True), "per subset")
    session.tie(3, "2F1")
    assert session.set_variation(3, "both").kind == "edited"
    c3 = session.state.component(3)
    assert c3.tie is not None and not c3.fix_frequency
    assert session.set_variation(1, "none").kind == "edited"
    c1 = session.state.component(1)
    assert c1.subset_phase is None and c1.variation == "none" and c1.fix_frequency
    assert session.set_variation(1, "none").kind == "noop"
    session.fix(2, amplitude=False)
    session.set_variation(2, "none")
    session.fix(2, amplitude=True)
    _refused(session, lambda: session.set_variation(2, "both"), "fixed amplitude")


# --- the error contract and no-ops --------------------------------------------------

INVALID: list[tuple[str, dict[str, Any], str]] = [
    ("add_peak", {"seed": 5.0}, "together"),
    ("add", {"frequency": -1.0}, "frequency"),
    ("add", {"frequency": 5.0, "bogus": 1}, "bogus"),
    ("remove", {"ids": [99]}, "F99"),
    ("remove", {"ids": []}, "ids"),
    ("set_active", {"ids": [99], "active": False}, "F99"),
    ("fix", {"ids": [1]}, "frequency=, amplitude="),
    ("fix", {"ids": ["G1"], "frequency": True}, "reference"),
    ("set_values", {"id": 99, "amplitude": 1.0}, "F99"),
    ("set_values", {"id": 1, "amplitude": float("inf")}, "amplitude"),
    ("tie", {"id": 1, "expression": "F2 +"}, "trailing"),
    ("set_variation", {"ids": [1], "mode": "both"}, "use_variation"),
    ("set_variation", {"ids": [1], "mode": "sometimes"}, "mode"),
    ("calculate", {"x": 1}, "x"),
    ("improve", {"ids": [99]}, "F99"),
    ("set_note", {"id": 99, "note": "x"}, "F99"),
    ("set_note", {"id": 1, "note": "x" * 201}, "note"),
]


@pytest.mark.parametrize(("op", "args", "match"), INVALID)
def test_every_op_refuses_bad_input(op: str, args: dict[str, Any], match: str) -> None:
    session = _three()
    _refused(session, lambda: session.do(op, **args), match)


def test_noops_are_not_logged() -> None:
    session = _three()
    log = session.history
    head = session.state
    for report in (
        session.set_active(1, True),
        session.fix(1, frequency=False),
        session.tie(1, None),
        session.set_values(1, frequency=head.component(1).frequency),
        session.set_note(1, ""),
    ):
        assert report.kind == "noop", report
    assert session.history == log and session.state is head
    empty = PreWhitenSession(_data(), settings=SETTINGS)
    assert empty.improve().kind == "noop" and empty.history == ()


def test_set_note_is_neutral() -> None:
    session = PreWhitenSession(_data(DRIFT), settings=DRIFT.resolved_settings())
    session.run()
    before = session.state
    assert session.set_note(1, "the main mode").kind == "edited"
    after = session.state
    assert after.component(1).note == "the main mode"
    assert after.status == before.status and after.blocked == before.blocked
    assert after.budget_origin == before.budget_origin
    assert session.kernel.cached(after) is session.kernel.cached(before)
    assert not session.is_pristine


# --- op-class side effects ----------------------------------------------------------

MANUAL_EDITS: dict[str, Callable[[PreWhitenSession], StepReport]] = {
    "add": lambda s: s.add(3.0, snap=False, refine="none"),
    "remove": lambda s: s.remove(s.components[-1].id),
    "set_active": lambda s: s.set_active(s.components[-1].id, False),
    "fix": lambda s: s.fix(1, frequency=True),
    "set_values": lambda s: s.set_values(1, phase=1.0),
    "tie": lambda s: s.tie(s.components[-1].id, f"F{s.components[0].id} + F1"),
    "set_variation": lambda s: (s.use_variation("half"), s.set_variation(1, "both"))[1],
    "calculate": lambda s: s.calculate(),
    "improve": lambda s: s.improve(),
    "improve_selected": lambda s: s.improve([1]),
}


@pytest.mark.parametrize("name", sorted(MANUAL_EDITS))
def test_manual_edits_reset_the_search(name: str) -> None:
    t, y, e = _data(DRIFT)
    session = PreWhitenSession(
        (t, y, e), settings=DRIFT.resolved_settings(), partitions=[_halves(t)]
    )
    session.run()
    before = session.state
    assert before.status == "stopped" and before.blocked
    report = MANUAL_EDITS[name](session)
    assert report.kind == "edited", report
    after = session.state
    assert after.status == "searching" and after.stop_reason == ""
    assert after.blocked == () and after.budget_origin == after.n_iterations
    assert after.n_iterations == before.n_iterations
    assert not session.is_pristine
    assert session.kernel.cached(after) is not None  # it fitted (or Calculate did)


def test_auto_then_manual_then_auto_resumes() -> None:
    session = PreWhitenSession(_data(DRIFT), settings=DRIFT.resolved_settings())
    session.run()
    stopped = session.state
    assert stopped.status == "stopped"
    session.remove(stopped.components[-1].id)
    assert session.state.status == "searching"
    entries = len(session.history)
    session.run()
    assert session.status == "stopped"
    steps = [e for e in session.history[entries:] if e.op == "step"]
    assert steps and steps[0].report.kind in ("accepted", "blocked", "rejected")
    assert session.state.budget_origin == stopped.n_iterations


# --- prune protection ---------------------------------------------------------------


def _insignificant(session: PreWhitenSession) -> None:
    assert session.configure(snr_threshold=1e6).kind == "edited"


@pytest.mark.parametrize("protect_manual", [True, False])
def test_prune_protects_constrained_and_varying_components(
    protect_manual: bool,
) -> None:
    session = _three(session_settings=SessionSettings(protect_manual=protect_manual))
    f = _freqs(session)
    session.add(5.0, snap=False, refine="none")  # manual
    session.fix(2, frequency=True)  # fixed
    session.tie(3, "2F1")  # tied child, F1 a tie parent
    session.use_variation("half")
    session.add(8.0, snap=False, refine="none")
    session.set_variation(5, "amplitude", hold_frequency=False)  # varying, manual
    session.add(11.0, snap=False, refine="none")
    session.set_variation(6, "both")  # varying, held
    _insignificant(session)
    report = session.prune()
    kept = _freqs(session)
    assert any(abs(k - 8.0) < 0.1 for k in kept)  # varying: always kept
    assert any(abs(k - 11.0) < 0.1 for k in kept)
    if protect_manual:
        assert report.kind == "noop"
        assert len(kept) == 6
        assert np.isclose(kept, f[1]).any() and np.isclose(kept, 5.0).any()
    else:
        assert report.kind == "edited"
        assert [c.id for c in session.components] == list(range(1, len(kept) + 1))
        assert not np.isclose(kept, f[1]).any()


def test_prune_renumbers_and_rewrites_ties() -> None:
    session = _three()
    f2, f3 = session.state.component(2).frequency, session.state.component(3).frequency
    session.add(5.0, snap=False, refine="none")
    assert session.tie(4, "F3 - F2").kind == "edited"
    _insignificant(session)
    # F1 is the only unprotected component (F2 and F3 are tie parents, F4 manual).
    report = session.prune()
    assert report.kind == "edited" and session.state.n_pruned == 1
    assert [c.id for c in session.components] == [1, 2, 3]
    assert session.state.component(1).frequency == pytest.approx(f2, abs=0.01)
    assert session.state.component(2).frequency == pytest.approx(f3, abs=0.01)
    child = session.state.component(3)
    assert child.tie is not None and child.tie.text() == "F2 - F1"
    assert child.frequency == pytest.approx(
        session.state.component(2).frequency - session.state.component(1).frequency,
        rel=1e-14,
    )


def test_without_implicit_calculate_edits_are_still_validated() -> None:
    session = _three(session_settings=SessionSettings(recalculate_on_edit=False))
    before = session.state
    assert session.fix(1, frequency=True).kind == "edited"
    head = session.state
    assert session.kernel.cached(head) is None  # nothing was fitted
    assert head.stats == before.stats and head.status == "searching"
    assert session.tie(3, "2F1").kind == "edited"
    _refused(session, lambda: session.tie(1, "F3 - F2"), "cyclic")
    _refused(session, lambda: session.set_variation(2, "both"), "use_variation")
    assert session.calculate().kind == "edited"
    assert session.kernel.cached(session.state) is not None
    c1, c3 = session.state.component(1), session.state.component(3)
    assert c3.frequency == 2 * c1.frequency
