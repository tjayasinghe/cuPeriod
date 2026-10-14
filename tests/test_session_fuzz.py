"""Seeded random operation sequences against the session contract.

Valid and invalid operations, garbage arguments and history moves are applied in a
random order. After every call:

* ``do()`` (and every typed wrapper) returned a report: bad input is refused, never
  raised (D21); only a programming error would escape, and fails the test;
* a refused or no-op call changed nothing;
* every committed head passes :func:`project.validate_state` (the invariants a
  project file is held to);
* undo, redo and goto land on exactly the state that was committed there, and a
  full walk back to the root and forward again restores the head.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.prewhiten.session import PreWhitenSession, project
from cuperiod.prewhiten.session.state import SessionState, StepReport
from cuperiod.prewhiten.subsets import Partition

N_NIGHTS, PER_NIGHT = 6, 50
SETTINGS = cup.PreWhitenSettings(
    backend="numpy", max_frequencies=4, maximum_frequency=25.0, max_nfev=60
)
GARBAGE: tuple[Any, ...] = (
    None, -1, 0, True, "F", "Fx", "F-1", "1e3", 1.5, float("nan"), float("inf"),
    -float("inf"), 10**30, [], [[]], {}, "x" * 300, object(),
)


def _light_curve() -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    rng = np.random.default_rng(7)
    night = np.repeat(np.arange(N_NIGHTS), PER_NIGHT)
    t = night * 1.0 + 0.3 * rng.random(night.size)
    order = np.argsort(t)
    t, night = t[order], night[order]
    y = (
        0.040 * np.sin(2 * np.pi * 5.21 * t + 0.3)
        + 0.020 * np.sin(2 * np.pi * 8.77 * t + 1.7)
        + 0.010 * np.sin(2 * np.pi * (5.21 + 8.77) * t + 2.2)
        + rng.normal(0.0, 0.003, N_NIGHTS)[night]
        + rng.normal(0.0, 0.004, t.size)
    )
    return (t, y, np.full(t.size, 0.004)), night


class Mirror:
    """What the history must hold: the state committed after each log ``seq``."""

    def __init__(self, session: PreWhitenSession) -> None:
        self.session = session
        self.root = session.state
        self.after: dict[int, SessionState] = {}

    def expected_head(self) -> SessionState:
        log = self.session.history
        return self.after[log[-1].seq] if log else self.root

    def check_head(self) -> None:
        head = self.session.state
        expected = self.expected_head()
        assert head is expected or head.same_content(expected)


def _ids(session: PreWhitenSession, rng: np.random.Generator) -> list[Any]:
    known = [c.id for c in session.state.components]
    if known and rng.random() < 0.7:  # mostly a valid selection, ints or labels
        size = int(rng.integers(1, len(known) + 1))
        chosen = rng.choice(known, size=size, replace=False).tolist()
        return [f"F{i}" if rng.random() < 0.3 else int(i) for i in chosen]
    pool: list[Any] = [*known, *(f"F{i}" for i in known), session.state.next_id,
                       session.state.next_id + 5, 0, -3]
    if rng.random() < 0.15:
        pool.append(GARBAGE[int(rng.integers(len(GARBAGE)))])
    size = int(rng.integers(0, 4))
    return [pool[int(rng.integers(len(pool)))] for _ in range(size)]


def _one_id(session: PreWhitenSession, rng: np.random.Generator) -> Any:
    ids = _ids(session, rng)
    return ids[0] if ids else 1


def _maybe(rng: np.random.Generator, value: Any) -> Any:
    """``value`` mostly, now and then a garbage value."""
    return GARBAGE[int(rng.integers(len(GARBAGE)))] if rng.random() < 0.1 else value


def _tie_text(session: PreWhitenSession, rng: np.random.Generator) -> Any:
    ids = [c.id for c in session.state.components] or [1]
    a = ids[int(rng.integers(len(ids)))]
    b = ids[int(rng.integers(len(ids)))]
    choices = [
        f"F{a} + F{b}", f"2F{a}", f"F{a} - F{b}", f"3F{a} + F{b}", f"F{a}",
        "F99", "F1 + F1", "import os", "", None, "F1 = F2 + F3", "2 * F1 + 1",
        "F1" * 150,
    ]
    return choices[int(rng.integers(len(choices)))]


def _tie(session: PreWhitenSession, rng: np.random.Generator) -> StepReport:
    """Mostly a well-formed tie of one row to two others; else anything."""
    ids = [c.id for c in session.state.components]
    if len(ids) >= 3 and rng.random() < 0.6:
        child, a, b = (int(i) for i in rng.choice(ids, size=3, replace=False))
        text = [f"F{a} + F{b}", f"2F{a}", f"F{a} - F{b}"][int(rng.integers(3))]
        return session.tie(child, text, allow_close=bool(rng.integers(2)))
    return session.tie(
        _one_id(session, rng), _tie_text(session, rng),
        allow_close=bool(rng.integers(2)),
    )


def _vary(session: PreWhitenSession, rng: np.random.Generator) -> StepReport:
    """Per-subset amplitude/phase of valid rows (choosing a partition first)."""
    if session.state.variation_by is None:
        return session.use_variation(_name(session, rng, ("night",)))
    known = [c.id for c in session.state.active]
    ids = [int(i) for i in rng.choice(known, size=1)] if known else [1]
    mode = ["amplitude", "phase", "both", "none"][int(rng.integers(4))]
    return session.set_variation(
        ids, mode, hold_frequency=bool(rng.integers(2))  # type: ignore[arg-type]
    )


def _name(session: PreWhitenSession, rng: np.random.Generator,
          extra: tuple[Any, ...] = ("nope",)) -> Any:
    """A defined partition's name mostly, else an undefined one (or ``None``)."""
    defined = [p.name for p in session.state.partitions]
    if defined and rng.random() < 0.7:
        return defined[int(rng.integers(len(defined)))]
    return extra[int(rng.integers(len(extra)))]


def _partition(session: PreWhitenSession, rng: np.random.Generator) -> Partition:
    time = np.asarray(session.data.time)
    kind = int(rng.integers(3))
    if kind == 0:
        return Partition.from_gaps(time, min_gap=0.5, name="night", min_points=5)
    if kind == 1:
        return Partition.from_gaps(time, min_gap=1.5, name="pair", min_points=5)
    return Partition.from_chunks(time, length=2.0, name="chunk", min_points=5)


def _runs(n: int, rng: np.random.Generator) -> Any:
    if rng.random() < 0.2:
        return [[int(rng.integers(-5, n + 5)), int(rng.integers(-5, n + 5))]]
    start = int(rng.integers(0, n - 1))
    return [[start, int(min(n, start + rng.integers(1, 6)))]]


def _settings_update(rng: np.random.Generator) -> dict[str, Any]:
    choices: list[dict[str, Any]] = [
        {"snr_threshold": float(rng.uniform(2.5, 6.0))},
        {"max_frequencies": int(rng.integers(1, 7))},
        {"refine": ["none", "last", "cyclic", "simultaneous"][int(rng.integers(4))]},
        {"prune": bool(rng.integers(2))},
        {"min_separation_rayleigh": float(rng.uniform(0.8, 2.0))},
        {"maximum_frequency": float(rng.choice([20.0, 25.0]))},
        {"fit_mean": bool(rng.integers(2))},
        {"snr_threshold": -1.0},
        {"max_frequencies": 0},
        {"refine": "bogus"},
        {"backend": "cpu"},
        {"no_such_field": 1},
    ]
    return choices[int(rng.integers(len(choices)))]


Action = Callable[[PreWhitenSession, np.random.Generator], StepReport]


def _do(op: str, **args: Any) -> Action:
    return lambda s, rng: s.do(op, **args)


ACTIONS: dict[str, Action] = {
    "step": lambda s, rng: s.step(),
    "step_seeded": lambda s, rng: s.do(
        "step", seed=_maybe(rng, float(rng.uniform(1.0, 20.0))),
        index=_maybe(rng, int(rng.integers(0, 500))),
    ),
    "add_peak": lambda s, rng: s.add_peak(),
    "add": lambda s, rng: s.add(
        _maybe(rng, float(rng.uniform(0.5, 24.0))),
        snap=bool(rng.integers(2)),
        refine=["none", "last", None][int(rng.integers(3))],  # type: ignore[arg-type]
        allow_close=bool(rng.integers(2)),
    ),
    "add_existing": lambda s, rng: s.add(
        s.state.components[0].frequency if s.state.components else 5.21
    ),
    "remove": lambda s, rng: s.remove(*_ids(s, rng), cascade=bool(rng.integers(2))),
    "set_active": lambda s, rng: s.set_active(
        _ids(s, rng), _maybe(rng, bool(rng.integers(2)))
    ),
    "fix": lambda s, rng: s.fix(
        _ids(s, rng),
        frequency=[None, True, False][int(rng.integers(3))],
        amplitude=[None, True, False][int(rng.integers(3))],
        phase=_maybe(rng, [None, True, False][int(rng.integers(3))]),
    ),
    "set_values": lambda s, rng: s.set_values(
        _one_id(s, rng),
        frequency=_maybe(rng, [None, float(rng.uniform(0.5, 24.0))][
            int(rng.integers(2))]),
        amplitude=_maybe(rng, [None, float(rng.normal(0.0, 0.03))][
            int(rng.integers(2))]),
        phase=[None, float(rng.uniform(-10.0, 10.0))][int(rng.integers(2))],
    ),
    "tie": _tie,
    "vary": _vary,
    "set_variation": lambda s, rng: s.set_variation(
        _ids(s, rng),
        ["none", "amplitude", "phase", "both", "bogus"][  # type: ignore[arg-type]
            int(rng.integers(5))],
        hold_frequency=bool(rng.integers(2)),
    ),
    "calculate": lambda s, rng: s.calculate(),
    "improve": lambda s, rng: s.improve(None if rng.random() < 0.5 else _ids(s, rng)),
    "set_note": lambda s, rng: s.set_note(
        _one_id(s, rng),
        _maybe(rng, ["ok", 'x\n"""__import__("os")', " ", "n" * 250][
            int(rng.integers(4))]),
    ),
    "delete_points": lambda s, rng: s.delete_points(
        _maybe(rng, rng.choice(s.data.n, size=int(rng.integers(1, 8)), replace=False))
    ),
    "delete_runs": lambda s, rng: s.do("delete_points", runs=_runs(s.data.n, rng)),
    "restore_points": lambda s, rng: s.restore_points(
        None if rng.random() < 0.3 else _runs(s.data.n, rng)
    ),
    "exclude_subsets": lambda s, rng: s.exclude_subsets(
        _name(s, rng),
        [_maybe(rng, int(rng.integers(-1, 8))) for _ in range(int(rng.integers(1, 3)))],
        excluded=bool(rng.integers(2)),
    ),
    "define_partition": lambda s, rng: s.define_partition(_partition(s, rng)),
    "define_bad_partition": lambda s, rng: s.do(
        "define_partition", name="bad", labels=["A"], runs=[[0, 3, 0]],
        weights=[1.0], origin={"kind": "manual"},
    ),
    "drop_partition": lambda s, rng: s.drop_partition(
        ["night", "pair", "chunk", "nope"][int(rng.integers(4))]
    ),
    "set_subset_weights": lambda s, rng: s.set_subset_weights(
        ["night", "pair", "nope"][int(rng.integers(3))],
        [_maybe(rng, float(w)) for w in rng.uniform(0.2, 3.0,
                                                    int(rng.integers(1, 8)))],
    ),
    "use_offsets": lambda s, rng: s.use_offsets(_name(s, rng, (None, "nope"))),
    "use_variation": lambda s, rng: s.use_variation(_name(s, rng, (None, "nope"))),
    "use_weights": lambda s, rng: s.use_weights(
        ["night", "pair", None, "nope"][int(rng.integers(4))]
    ),
    "configure": lambda s, rng: s.configure(**_settings_update(rng)),
    "configure_raw": lambda s, rng: s.do(
        "configure", settings=_maybe(rng, {"snr_threshold": 3.0})
    ),
    "polish": lambda s, rng: s.polish(),
    "prune": lambda s, rng: s.prune(),
    "renumber": lambda s, rng: s.renumber(),
    "unknown_op": _do("explode"),
    "extra_args": lambda s, rng: s.do("calculate", surprise=1),
    "wrong_types": lambda s, rng: s.do("remove", ids="F1", cascade="maybe"),
}
NAMES = sorted(ACTIONS)


def _session() -> PreWhitenSession:
    light_curve, night = _light_curve()
    return PreWhitenSession(
        light_curve, settings=SETTINGS, labels={"night": night},
        partitions=["night"],
        session_settings=cup.SessionSettings(undo_depth=1000),
    )


def _apply(session: PreWhitenSession, mirror: Mirror, name: str,
           rng: np.random.Generator) -> None:
    before = session.state
    log_before = session.history
    report = ACTIONS[name](session, rng)  # must not raise
    assert isinstance(report, StepReport), name
    if report.committed:
        head = session.state
        assert head is not before and head.serial > before.serial, name
        project.validate_state(head, session.kernel.data)
        log = session.history
        assert len(log) == len(log_before) + 1, name
        mirror.after[log[-1].seq] = head
    else:
        assert report.kind in ("refused", "noop"), (name, report)
        assert session.state is before, (name, report)
        assert session.history == log_before, name
        if report.kind == "refused":
            assert report.message, name
    mirror.check_head()


def _move(session: PreWhitenSession, mirror: Mirror, rng: np.random.Generator) -> None:
    roll = rng.random()
    if roll < 0.45:
        session.undo()
    elif roll < 0.9:
        session.redo()
    else:
        entries = session.log_entries
        seq = 0 if not entries else entries[int(rng.integers(len(entries)))].seq
        report = session.goto(seq)
        assert report.kind == "edited", report
    project.validate_state(session.state, session.kernel.data)
    mirror.check_head()


@pytest.mark.parametrize("seed", range(8))
def test_random_operation_sequences_keep_the_contract(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    session = _session()
    mirror = Mirror(session)
    for _ in range(60):
        if rng.random() < 0.15:
            _move(session, mirror, rng)
        else:
            _apply(session, mirror, NAMES[int(rng.integers(len(NAMES)))], rng)
    # A full walk back to the root and forward to the end of the log (past the
    # head when a redo tail is left) meets every committed state again, and goto
    # brings the head back.
    final = session.state
    final_seq = session.history[-1].seq if session.history else 0
    entries = session.log_entries
    while session.undo():
        mirror.check_head()
    assert session.history == () and session.state is mirror.root
    while session.redo():
        mirror.check_head()
    assert session.history == entries == session.log_entries
    assert session.goto(final_seq).kind == "edited"
    assert session.state.same_content(final)
    if session.state.components:
        result = session.result()
        assert all(math.isfinite(c.frequency) for c in result.components)


def test_every_action_name_is_exercised() -> None:
    """The vocabulary covers every registered op (plus garbage)."""
    from cuperiod.prewhiten.session import actions

    ops: set[str] = set()
    for name in NAMES:
        session = _session()
        seen: list[str] = []
        original = session._do

        def record(op: str, raw: Any, _seen: list[str] = seen,
                   _orig: Any = original) -> StepReport:
            _seen.append(op)
            return _orig(op, raw)

        session._do = record  # type: ignore[method-assign]
        for seed in range(6):
            ACTIONS[name](session, np.random.default_rng(seed))
        ops.update(seen)
    assert set(actions.ARGS) - ops == set()
