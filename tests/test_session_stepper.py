"""The stepper: step() is one iteration of the automatic loop, and it composes.

``cup.prewhiten`` is itself a session run and finalized, held to the pre-session
numbers by tests/test_prewhiten_golden.py; these tests pin the ways of driving a
session that must reach the very same numbers (bitwise, numpy backend).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

import cuperiod as cup
import prewhiten_corpus as pc
from cuperiod.prewhiten.session import PreWhitenSession
from cuperiod.prewhiten.session.kernel import SessionKernel
from cuperiod.prewhiten.spectrum import AmplitudeSpectrum


def _payload(case: pc.Case) -> tuple[Any, cup.PreWhitenSettings, Any]:
    data, settings, grid = pc.load(case)
    t, y, e = data
    return ((t, y) if e is None else (t, y, e)), settings, grid


def _session(name: str, **kw: Any) -> PreWhitenSession:
    payload, settings, grid = _payload(pc.case_by_name(name))
    overrides = kw.pop("settings", None)
    if overrides:
        settings = settings.model_copy(update=overrides)
    return PreWhitenSession(payload, settings=settings, grid=grid, **kw)


def _diff(a: Any, b: Any, path: str = "") -> list[str]:
    """Paths where two results differ (arrays and floats compared bitwise)."""
    if isinstance(a, AmplitudeSpectrum) or isinstance(b, AmplitudeSpectrum):
        if (a is None) != (b is None):
            return [path]
        return [
            f"{path}.{f}"
            for f in ("frequency", "amplitude", "phase", "power")
            if not np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True)
        ]
    if isinstance(a, np.ndarray):
        return [] if np.array_equal(a, b, equal_nan=True) else [path]
    if isinstance(a, float):
        same = a == b or (math.isnan(a) and math.isnan(b))
        return [] if same else [f"{path}: {a!r} != {b!r}"]
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        out: list[str] = []
        for f in dataclasses.fields(a):
            out += _diff(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
        return out
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}"]
        pairs = enumerate(zip(a, b, strict=True))
        return [d for i, (x, y) in pairs for d in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def _reference(name: str) -> cup.PreWhitenResult:
    payload, settings, grid = _payload(pc.case_by_name(name))
    return cup.prewhiten(payload, settings=settings, grid=grid)


CASES = ["triple-last", "triple-fit-mean-false", "drift-blocked", "pair-tight-guard",
         "narrow-no-peak", "noerr-bootstrap", "modulated-prune"]


@pytest.mark.parametrize("name", CASES)
def test_steps_one_by_one_equal_run(name: str) -> None:
    session = _session(name)
    steps = 0
    while session.status != "stopped":
        session.step()
        steps += 1
        assert steps < 100
    assert _diff(session.finalize(), _reference(name)) == []


@pytest.mark.parametrize("name", ["triple-last", "drift-blocked"])
def test_run_in_pieces_and_cancelled_runs_resume(name: str) -> None:
    pieces = _session(name)
    pieces.run(max_steps=2)
    assert pieces.status == "searching"
    pieces.run()
    assert _diff(pieces.finalize(), _reference(name)) == []

    calls = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    cancelled = _session(name)
    cancelled.run(should_stop=stop_after_two)
    assert cancelled.status == "searching" and len(cancelled.history) == 2
    cancelled.run()
    assert _diff(cancelled.finalize(), _reference(name)) == []


@pytest.mark.parametrize("name", ["triple-last", "drift-budget"])
def test_record_false_equals_record_true(name: str) -> None:
    quiet = _session(name, record=False)
    quiet.run()
    loud = _session(name)
    loud.run()
    # every attempt is logged, plus the pre-check stop when that ended the run
    n = loud.state.n_iterations
    assert quiet.history == () and len(loud.history) in (n, n + 1)
    assert quiet.state.same_content(loud.state)
    assert _diff(quiet.finalize(), loud.finalize()) == []


def test_result_is_pure() -> None:
    session = _session("triple-last")
    session.run()
    head = session.state
    entries = session.history
    first = session.result()
    assert session.state is head and session.history == entries
    assert _diff(first, session.result()) == []


def test_n_iterations_after_every_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    # accepted, blocked and the stopping rule (rejected) on the drifting mode
    session = _session("drift-blocked")
    seen: dict[str, int] = {}
    while session.status != "stopped":
        before = session.state.n_iterations
        report = session.step()
        if report.message.startswith("reached max_frequencies"):
            assert session.state.n_iterations == before  # the pre-check
        else:
            assert session.state.n_iterations == before + 1, report.kind
        seen[report.kind] = seen.get(report.kind, 0) + 1
    assert seen.get("accepted") and seen.get("blocked")
    # a stopped session: step() is a no-op and logs nothing
    n = session.state.n_iterations
    assert session.step().kind == "noop" and session.state.n_iterations == n

    rejected = _session("triple-last")
    rejected.run()
    assert rejected.history[-1].report.kind == "rejected"
    assert rejected.state.stop_reason == rejected.history[-1].report.message

    # no peak resolved from the existing components: +1, stopped
    narrow = _session("narrow-no-peak")
    narrow.run()
    last = narrow.history[-1]
    assert last.report.kind == "stopped" and "no peak" in last.report.message
    assert last.args == {"seed": None, "index": None}
    assert narrow.state.n_iterations == len(narrow.history)

    # the pre-check (max_frequencies reached) leaves n_iterations unchanged
    capped = _session("triple-max-frequencies-2")
    capped.run()
    assert capped.state.stop_reason == "reached max_frequencies (2)"
    assert capped.state.n_iterations == 2 and len(capped.history) == 3
    zero = _session("triple-max-frequencies-0")
    zero.run()
    assert zero.state.n_iterations == 0 and zero.state.stop_reason == (
        "max_frequencies is 0"
    )

    # a failed trial fit stops the search, as v1.3 did (+1)
    failing = _session("triple-last")
    original = SessionKernel.fit

    def broken(self: SessionKernel, state: Any, frequencies: Any, **kw: Any) -> Any:
        if np.size(frequencies):
            raise np.linalg.LinAlgError("singular")
        return original(self, state, frequencies, **kw)

    monkeypatch.setattr(SessionKernel, "fit", broken)
    report = failing.step()
    assert report.kind == "stopped" and report.message == "fit failed: singular"
    assert failing.state.n_iterations == 1
    assert failing.history[-1].args["seed"] is not None


def test_the_budget_stop() -> None:
    session = _session("drift-budget")
    session.run()
    assert session.state.stop_reason == (
        "too many candidates merged with existing components"
    )
    assert session.state.n_iterations == session.settings.max_frequencies + 20


def test_fit_mean_false_uses_the_zero_model_and_centred_trials() -> None:
    session = _session("triple-fit-mean-false")
    root_fit = session.fit
    assert root_fit.n_parameters == 0 and not root_fit.fit_mean
    session.step()
    assert not session.fit.fit_mean
    zero_point = session.kernel.zero_point(session.state)
    assert zero_point != 0.0
    assert session.state.offset == pytest.approx(zero_point)


def test_every_committed_step_logs_its_seed() -> None:
    session = _session("drift-blocked")
    session.run()
    for entry in session.history:
        assert entry.op == "step"
        if entry.report.kind in ("accepted", "blocked", "rejected"):
            assert isinstance(entry.args["seed"], float)
            assert isinstance(entry.args["index"], int)
    accepted = [e for e in session.history if e.report.kind == "accepted"]
    assert [e.report.component_id for e in accepted] == list(
        range(1, len(accepted) + 1)
    )


def test_seeded_replay_is_bitwise_on_another_backend() -> None:
    fast = _session("triple-last", settings={"backend": "finufft"})
    fast.run()
    seeds = [
        None if e.args["seed"] is None else (e.args["seed"], e.args["index"])
        for e in fast.history
    ]
    replay = _session("triple-last", settings={"backend": "numpy"})
    replay.run(seeds=seeds)
    assert [e.report.kind for e in replay.history] == [
        e.report.kind for e in fast.history
    ]
    a, b = fast.state.components, replay.state.components
    assert [c.frequency for c in a] == [c.frequency for c in b]
    assert [c.amplitude for c in a] == [c.amplitude for c in b]
    assert [c.phase for c in a] == [c.phase for c in b]


def test_a_seed_far_from_the_peak_warns() -> None:
    session = _session("triple-last")
    cand = session.candidate()
    assert cand is not None
    report = session.step(seed=cand.frequency * 1.01, index=0)
    assert "not this run's tallest" in report.warning


def test_candidate_previews_the_next_step() -> None:
    session = _session("triple-last")
    cand = session.candidate()
    assert cand is not None
    head = session.state
    report = session.step()
    assert session.history[-1].args == {"seed": cand.frequency, "index": cand.index}
    assert head.serial < session.state.serial and report.kind == "accepted"
