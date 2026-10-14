"""History: undo/redo by group, truncation, goto, eviction and reachability."""

from __future__ import annotations

import dataclasses

import pytest

from cuperiod.prewhiten.session.history import History
from cuperiod.prewhiten.session.state import (
    LogEntry,
    SessionError,
    SessionSnapshot,
    SessionState,
    StepReport,
)
from test_session_state import make_state


def _entry(history: History, op: str = "step", group: int | None = None) -> LogEntry:
    return LogEntry(
        seq=history.next_seq(),
        group=history.allocate_group() if group is None else group,
        op=op,
        args={},
        report=StepReport(kind="edited", message=op),
        timestamp="t",
    )


def _push(history: History, n: int = 1) -> list[SessionState]:
    out = []
    for _ in range(n):
        state = dataclasses.replace(history.head, serial=history.head.serial + 1,
                                    n_iterations=history.head.n_iterations + 1)
        history.push(_entry(history), state)
        out.append(state)
    return out


def test_undo_then_redo_returns_the_identical_state() -> None:
    root = make_state()
    h = History(root, depth=10)
    s1, s2 = _push(h, 2)
    assert h.head is s2 and h.cursor == 2
    assert h.undo() is s1 and h.head is s1
    assert h.undo() is root
    assert h.undo() is None and h.head is root and not h.can_undo
    assert h.redo() is s1 and h.redo() is s2 and h.redo() is None
    assert not h.can_redo


def test_a_new_action_truncates_redo() -> None:
    h = History(make_state(), depth=10)
    s1, _ = _push(h, 2)
    h.undo()
    (s3,) = _push(h, 1)
    assert h.head is s3 and not h.can_redo
    assert [e.seq for e in h.entries] == [1, 2]
    assert h.effective()[0].seq == 1
    assert h.undo() is s1


def test_groups_undo_as_one() -> None:
    root = make_state()
    h = History(root, depth=10)
    with h.group() as g:
        with h.group() as inner:
            assert inner == g
        _push(h, 3)
    (single,) = _push(h, 1)
    assert [e.group for e in h.entries] == [g, g, g, g + 1]
    before_group = h.undo()
    assert before_group is not None and h.cursor == 3
    assert h.undo() is root and h.cursor == 0
    assert h.redo() is not None and h.cursor == 3
    assert h.redo() is single


def test_goto_and_reachable() -> None:
    root = make_state()
    h = History(root, depth=10)
    states = _push(h, 4)
    assert h.goto(2) is states[1] and h.cursor == 2
    assert h.goto(4) is states[3]  # forward into the redo tail
    assert h.goto(0) is root
    assert h.goto(99) is None and h.head is root
    assert h.reachable(0) and h.reachable(3) and not h.reachable(99)


def test_depth_eviction() -> None:
    root = make_state()
    h = History(root, depth=2)
    states = _push(h, 5)
    assert [h.reachable(seq) for seq in range(6)] == [
        False, False, False, True, True, True
    ]
    assert h.retained() == tuple(states[2:])
    assert h.undo() is states[3] and h.undo() is states[2]
    assert h.undo() is None and h.head is states[2]
    assert len(h.effective()) == 3  # entries stay in the log
    assert len(h.entries) == 5
    with pytest.raises(SessionError):
        History(root, depth=0)


def test_from_snapshot() -> None:
    root = make_state()
    h = History(root, depth=10)
    states = _push(h, 3)
    snap_full = SessionSnapshot(creation=None, head=h.head,  # type: ignore[arg-type]
                                log=h.effective(), states=h.aligned())
    again = History.from_snapshot(snap_full, depth=10)
    assert again.head is states[-1] and again.undo() is states[-2]
    assert again.reachable(0)
    snap_head = SessionSnapshot(creation=None, head=h.head,  # type: ignore[arg-type]
                                log=h.effective())
    bare = History.from_snapshot(snap_head, depth=10)
    assert bare.head is states[-1] and bare.undo() is None
    assert not bare.reachable(2) and bare.reachable(3)
    assert bare.next_seq() == 4 and bare.allocate_group() > max(
        e.group for e in h.entries
    )
