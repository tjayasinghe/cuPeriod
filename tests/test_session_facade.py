"""PreWhitenSession: do(), the error contract, history, caches, threads, snapshots."""

from __future__ import annotations

import dataclasses
import importlib.util
import threading
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

import cuperiod as cup
import prewhiten_corpus as pc
from cuperiod.core.grid import uniform_frequency_grid
from cuperiod.prewhiten.constraints import ConstraintError
from cuperiod.prewhiten.session import PreWhitenSession, actions
from cuperiod.prewhiten.session.state import (
    SessionError,
    SessionState,
    StepReport,
)
from cuperiod.prewhiten.spectrum import SpectrumEngine


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data, _, _ = pc.load(pc.case_by_name("triple-last"))
    t, y, e = data
    assert e is not None
    return t, y, e


SETTINGS = pc.case_by_name("triple-last").resolved_settings()


def _session(**kw: Any) -> PreWhitenSession:
    return PreWhitenSession(_data(), settings=SETTINGS, **kw)


@pytest.fixture
def local_reducer() -> Any:
    """Register test-local reducers; the registry is restored afterwards."""
    from cuperiod.prewhiten.session.session import _load_ops

    _load_ops()  # so a later-stage module cannot re-register over the local one
    saved: dict[str, Any] = {}

    def install(op: str, reducer: Any) -> None:
        if op not in saved:
            saved[op] = actions.reducer(op)
        actions.register(op)(reducer)

    yield install
    for op, previous in saved.items():
        actions.unregister(op)
        if previous is not None:
            actions.register(op)(previous)


# --- do(), log, undo --------------------------------------------------------------


def test_do_commits_logs_and_undoes() -> None:
    session = _session()
    root = session.state
    report = session.do("step")
    assert report.kind == "accepted" and report.resolved is not None
    (entry,) = session.history
    assert entry.seq == 1 and entry.op == "step" and entry.report.resolved is None
    assert set(entry.args) == {"seed", "index"} and entry.report.kind == "accepted"
    assert entry.report.component_id == 1 and entry.timestamp.endswith("Z")
    head = session.state
    assert head is not root and head.serial > root.serial
    assert session.can_undo and session.undo() and session.state is root
    assert session.can_redo and session.redo() and session.state is head
    assert not session.redo()
    assert session.is_pristine


def test_a_whole_run_undoes_as_one_group() -> None:
    session = _session()
    root = session.state
    session.run()
    assert len(session.history) > 2
    assert len({e.group for e in session.history}) == 1
    assert session.undo() and session.state is root
    assert session.redo() and session.state.status == "stopped"
    session.finalize()
    groups = [e.group for e in session.history]
    assert groups[-1] != groups[0]  # finalize is its own group
    assert session.undo() and session.state.status == "stopped"


def test_goto_and_reachable() -> None:
    session = _session()
    session.run()
    first = session.history[0]
    report = session.goto(first.seq)
    assert report.kind == "edited" and len(session.history) == 1
    assert session.reachable(0) and session.reachable(first.seq)
    assert session.goto(999).kind == "refused"


def test_no_log_without_record() -> None:
    session = _session(record=False)
    session.run()
    assert session.history == () and not session.can_undo and not session.undo()
    assert session.goto(0).kind == "refused"
    assert session.is_pristine


# --- the error contract ---------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        SessionError("bad id"),
        ConstraintError("cyclic tie"),
        ValueError("bad value"),
        np.linalg.LinAlgError("singular"),
        "validation",
    ],
)
def test_reducer_errors_roll_back(local_reducer: Any, error: Any) -> None:
    def failing(kernel: Any, state: SessionState, derived: Any, args: Any) -> Any:
        if error == "validation":
            class Strict(BaseModel):
                x: int

            Strict.model_validate({"x": "not an int"})
        raise error

    local_reducer("polish", failing)
    session = _session()
    session.step()
    head, entries = session.state, session.history
    report = session.polish()
    assert report.kind == "refused" and report.message
    assert session.state is head and session.history == entries


def test_programming_errors_propagate_and_commit_nothing(local_reducer: Any) -> None:
    def broken(kernel: Any, state: SessionState, derived: Any, args: Any) -> Any:
        raise TypeError("a bug")

    local_reducer("polish", broken)
    session = _session()
    session.step()
    head, entries = session.state, session.history
    with pytest.raises(TypeError):
        session.polish()
    assert session.state is head and session.history == entries


def test_invalid_arguments_are_refused() -> None:
    session = _session()
    head = session.state
    for op, args in (
        ("step", {"seed": 3.0}),
        ("step", {"seed": float("nan"), "index": 1}),
        ("nonsense", {}),
        ("polish", {"extra": 1}),
    ):
        report = session.do(op, **args)
        assert report.kind == "refused", (op, args)
    assert session.state is head and session.history == ()


def test_an_unregistered_op_is_refused(local_reducer: Any) -> None:
    session = _session()
    saved = actions.unregister("renumber")
    try:
        report = session.renumber()
        assert report.kind == "refused" and "not available" in report.message
    finally:
        if saved is not None:
            actions.register("renumber")(saved)


def test_noop_is_not_logged() -> None:
    session = _session()
    assert session.polish().kind == "noop"  # nothing to polish yet
    assert session.renumber().kind == "noop"
    assert session.history == ()


def test_manual_class_side_effects(local_reducer: Any) -> None:
    # A manual edit: status back to searching, blocked cleared, the budget restarts
    # at n_iterations, an implicit Calculate when the reducer did not fit.
    def edit(kernel: Any, state: SessionState, derived: Any, args: Any) -> Any:
        return dataclasses.replace(state), None, StepReport(kind="edited", message="ok")

    local_reducer("calculate", edit)
    session = PreWhitenSession(
        _data(), settings=pc.case_by_name("drift-blocked").resolved_settings()
    )
    session.run()
    assert session.status == "stopped"
    before = session.state
    report = session.calculate()
    assert report.kind == "edited"
    after = session.state
    assert after.status == "searching" and after.stop_reason == ""
    assert after.blocked == () and after.budget_origin == before.n_iterations
    assert after.stats.n_parameters == before.stats.n_parameters
    assert not session.is_pristine
    assert session.kernel.cached(after) is not None  # the implicit Calculate's fit
    assert session.undo() and session.state is before


# --- derived cache --------------------------------------------------------------------


def test_eviction_rebuilds_bitwise() -> None:
    session = _session()
    session.run(max_steps=3)
    head = session.state
    cached = session.kernel.cached(head)
    assert cached is not None
    session.kernel.forget()
    rebuilt = session.kernel.derive(head)
    assert rebuilt is not cached
    np.testing.assert_array_equal(rebuilt.fit.residuals, cached.fit.residuals)
    np.testing.assert_array_equal(rebuilt.fit.amplitude, cached.fit.amplitude)
    np.testing.assert_array_equal(rebuilt.scan.amplitude, cached.scan.amplitude)
    assert rebuilt.drift == 0.0


def test_k0_rebuild_is_the_data_spectrum_and_reruns_match() -> None:
    session = _session(session_settings=cup.SessionSettings(derived_cache=1))
    root = session.state
    session.run()
    assert session.undo() and session.state is root
    assert session.kernel.cached(root) is None  # evicted (cache of one)
    derived = session.kernel.derive(root)
    assert derived.scan is session.kernel.data_scan(root)
    np.testing.assert_array_equal(
        derived.residual_spectrum.amplitude, session.spectrum().amplitude
    )
    session.run()
    again = session.finalize()
    fresh = _session()
    fresh.run()
    expected = fresh.finalize()
    for name in ("frequency", "amplitude", "phase", "frequency_error", "snr"):
        np.testing.assert_array_equal(getattr(again, name), getattr(expected, name))
    np.testing.assert_array_equal(again.residuals, expected.residuals)


# --- threads --------------------------------------------------------------------------


def test_two_threads_are_refused() -> None:
    session = _session()
    holding = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with session._guard():
            holding.set()
            release.wait(5)

    worker = threading.Thread(target=hold)
    worker.start()
    try:
        assert holding.wait(5)
        with pytest.raises(SessionError, match="two threads"):
            session.step()
        with pytest.raises(SessionError, match="two threads"):
            session.result()
    finally:
        release.set()
        worker.join(5)
    assert session.step().kind == "accepted"


# --- snapshots and data ---------------------------------------------------------------


def test_snapshot_restore_in_memory() -> None:
    session = _session()
    session.run()
    snap = session.snapshot(with_states=True)
    restored = PreWhitenSession.restore(snap, _data())
    assert restored.state.same_content(session.state)
    assert [e.seq for e in restored.history] == [e.seq for e in session.history]
    for entry in session.history:
        assert restored.reachable(entry.seq) == session.reachable(entry.seq)
    assert restored.undo() and session.undo()
    assert restored.state.same_content(session.state)
    assert restored.redo() and session.redo()
    assert restored.t_ref == session.t_ref and restored.backend == session.backend
    bare = PreWhitenSession.restore(session.snapshot(), _data())
    assert not bare.can_undo and bare.state.same_content(session.state)
    t, y, e = _data()
    with pytest.raises(cup.ProjectError):
        PreWhitenSession.restore(snap, (t, y + 1.0, e))


def test_restored_session_continues_like_the_original() -> None:
    session = _session()
    session.run(max_steps=2)
    restored = PreWhitenSession.restore(session.snapshot(), _data())
    session.run()
    restored.run()
    a, b = session.finalize(), restored.finalize()
    np.testing.assert_array_equal(a.frequency, b.frequency)
    np.testing.assert_array_equal(a.residuals, b.residuals)


def test_data_views_are_read_only_and_inputs_stay_writable() -> None:
    t, y, e = _data()
    session = PreWhitenSession((t, y, e), settings=SETTINGS)
    view = session.data
    assert not view.time.flags.writeable and not view.value.flags.writeable
    assert t.flags.writeable and y.flags.writeable and e.flags.writeable
    session.run()
    result = session.finalize()
    assert result.time.flags.writeable and result.residuals.flags.writeable
    assert t.flags.writeable
    assert view.source_index.tolist() == list(range(t.size))


def test_labels_follow_the_finite_filter() -> None:
    t, y, e = _data()
    y = y.copy()
    y[[3, 7]] = np.nan
    site = np.array(["A", "B"] * (t.size // 2), dtype=object)
    session = PreWhitenSession((t, y, e), settings=SETTINGS, labels={"site": site},
                               partitions=["site"])
    keep = np.isfinite(y)
    assert session.data.labels["site"].tolist() == site[keep].tolist()
    assert session.data.source_index.tolist() == np.flatnonzero(keep).tolist()
    lc = cup.LightCurve.from_arrays(t, y, e).finite()
    np.testing.assert_array_equal(session.data.time, lc.time)
    (partition,) = session.state.partitions
    assert partition.name == "site" and partition.n_points == lc.n
    with pytest.raises(ValueError):
        PreWhitenSession((t, y, e), settings=SETTINGS, labels={"site": site[:5]})


def test_grid_as_a_tuple_is_exact() -> None:
    session = _session()
    f0, df, nf = session.state.grid
    again = PreWhitenSession(_data(), settings=SETTINGS, grid=(f0, df, nf))
    np.testing.assert_array_equal(
        again.kernel.engine(again.state).frequency,
        session.kernel.engine(session.state).frequency,
    )


def test_delegations_are_clear_when_missing(tmp_path: Any) -> None:
    session = _session()
    if importlib.util.find_spec("cuperiod.prewhiten.session.project") is None:
        with pytest.raises(SessionError, match="not available"):
            session.save(tmp_path / "x.cupw.json")
    if importlib.util.find_spec("cuperiod.prewhiten.session.script") is None:
        with pytest.raises(SessionError, match="not available"):
            session.to_script()
    if importlib.util.find_spec("cuperiod.prewhiten.session.variation") is None:
        with pytest.raises(SessionError, match="not available"):
            session.variations()


def test_errors_of_the_constructor_match_prewhiten() -> None:
    t, y, e = _data()
    with pytest.raises(cup.InsufficientDataError):
        PreWhitenSession((t[:5], y[:5], e[:5]), settings=SETTINGS)
    with pytest.raises(cup.InsufficientDataError, match="error"):
        PreWhitenSession((t, y, np.zeros(t.size)), settings=SETTINGS)
    with pytest.raises(ValidationError):
        cup.PreWhitenSettings(snr_threshold=-1.0)


# --- spectrum additions (E3) ----------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "finufft"])
def test_noise_curve_and_release(backend: str) -> None:
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 30, 400))
    y = np.sin(2 * np.pi * 2.3 * t) + rng.normal(0, 0.3, t.size)
    grid = uniform_frequency_grid(30.0, maximum_frequency=10.0, samples_per_peak=5)
    engine = SpectrumEngine(t, None, grid=grid, backend=backend)
    spectrum = engine.spectrum(y)
    for estimator in ("mean", "median"):
        for window in (1.0, 0.01, 50.0):
            curve = spectrum.noise_curve(window=window, estimator=estimator)
            point = [
                spectrum.noise_at(f, window=window, estimator=estimator)  # type: ignore[arg-type]
                for f in spectrum.frequency
            ]
            np.testing.assert_allclose(curve, point, rtol=1e-12)
    engine.release()  # a no-op on CPU
    assert not engine._released
    engine.release(force=True)
    assert engine._released
    np.testing.assert_array_equal(engine.spectrum(y).amplitude, spectrum.amplitude)
    assert not engine._released


# --- reports of hand-edited states (the reducers that make them come later) ----------


def test_report_of_a_negative_amplitude_under_a_fixed_phase() -> None:
    from cuperiod.prewhiten.session.report import build_result

    session = _session()
    session.run(max_steps=3)
    kernel, head = session.kernel, session.state
    c1 = head.components[0]
    flipped = dataclasses.replace(
        c1, fix_phase=True, phase=float((c1.phase + np.pi) % (2 * np.pi))
    )
    edited = dataclasses.replace(head, components=(flipped, *head.components[1:]))
    new, derived = kernel.recalculate(edited)
    assert new.components[0].amplitude < 0.0  # signed where the user edits it
    result = build_result(kernel, new, derived)
    first = result[0]
    assert first.amplitude == pytest.approx(c1.amplitude, rel=1e-9)
    assert first.phase == pytest.approx(c1.phase, abs=1e-9)
    assert first.fixed == "p" and "negative_amplitude" in first.flags
    assert first.snr > 0 and not first.blended
    assert "flags" in result.to_table()[0]
    kernel.forget()
    assert kernel.derive(new).drift <= 1e-9


def test_report_of_a_tie_to_an_inactive_parent_and_rows() -> None:
    from cuperiod.prewhiten.session.report import build_result, rows
    from cuperiod.prewhiten.session.state import Tie

    session = _session()
    session.run(max_steps=3)
    kernel, head = session.kernel, session.state
    c1, c2, c3 = head.components
    edited = dataclasses.replace(
        head,
        components=(
            c1,
            dataclasses.replace(c2, active=False),
            dataclasses.replace(c3, tie=Tie(parents=(1, 2), coefficients=(1, 1))),
        ),
    )
    new, derived = kernel.recalculate(edited)
    expected = c1.frequency + c2.frequency
    assert new.components[2].frequency == pytest.approx(expected, rel=1e-12)
    result = build_result(kernel, new, derived)
    assert [c.label for c in result] == ["F1", "F3"]
    tied = result[1]
    assert tied.tied == "F1 + F2" and tied.combination == "F3 = F1 + F2"
    assert {"tied", "parent_inactive"} <= set(tied.flags)
    assert all(m.index != 1 for m in result.combinations)
    table = rows(kernel, new, derived)
    assert [r.id for r in table] == [1, 2, 3]
    assert np.isnan(table[1].snr) and not table[1].passes
    assert table[0].passes and table[2].combination == "F3 = F1 + F2"
