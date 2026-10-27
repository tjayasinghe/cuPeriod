"""Linear undo/redo over the states of a session.

The history holds the root state, one state per logged action and a cursor. Undo
and redo move the cursor across whole *groups* (an Auto run, a macro), so nothing is
recomputed; a new action truncates the redo tail. Only the newest ``depth`` states are
retained: an older entry stays in the log but becomes *unreachable* — its state can
only be re-created by replaying the log, which is never done implicitly.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace

from cuperiod.prewhiten.session.state import (
    LogEntry,
    SessionError,
    SessionSnapshot,
    SessionState,
)


class History:
    """Retained states, the log and the cursor of one session.

    Parameters
    ----------
    root : SessionState
        The state before any action.
    depth : int
        How many states before the head are retained.
    """

    def __init__(self, root: SessionState, *, depth: int) -> None:
        if depth < 1:
            raise SessionError("undo depth must be at least 1")
        self._depth = int(depth)
        self._states: list[SessionState | None] = [root]
        self._entries: list[LogEntry] = []
        self._cursor = 0
        self._open: list[int] = []
        self._next_group = 1
        # States before this position were never in memory (a project opened from a
        # file keeps only its head); later gaps were evicted by ``depth``.
        self._unloaded = 0

    # -- read-only -----------------------------------------------------------------
    @property
    def head(self) -> SessionState:
        """The state at the cursor (always retained)."""
        state = self._states[self._cursor]
        assert state is not None  # the head is never evicted
        return state

    @property
    def cursor(self) -> int:
        """How many log entries the head has applied."""
        return self._cursor

    @property
    def entries(self) -> tuple[LogEntry, ...]:
        """Every entry, including the redo tail."""
        return tuple(self._entries)

    def effective(self) -> tuple[LogEntry, ...]:
        """The entries from the root to the head."""
        return tuple(self._entries[: self._cursor])

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0 and self._target_back() is not None

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._entries) and self._target_forward() is not None

    def next_seq(self) -> int:
        """The ``seq`` of the next pushed entry."""
        return self._entries[self._cursor - 1].seq + 1 if self._cursor else 1

    def retained(self) -> tuple[SessionState, ...]:
        """The states still in memory, oldest first."""
        return tuple(s for s in self._states if s is not None)

    def aligned(self) -> tuple[SessionState | None, ...]:
        """States aligned with ``(root, *effective())``, ``None`` where evicted."""
        return tuple(self._states[: self._cursor + 1])

    def _position(self, seq: int) -> int | None:
        """Index into the states of the state after entry ``seq`` (0 = root)."""
        if seq == 0:
            return 0
        for i, entry in enumerate(self._entries):
            if entry.seq == seq:
                return i + 1
        return None

    def reachable(self, seq: int) -> bool:
        """Whether the state after entry ``seq`` (``0`` = root) is in memory."""
        position = self._position(seq)
        return position is not None and self._states[position] is not None

    def unreachable(self, seq: int) -> str | None:
        """Why the state after entry ``seq`` is not in memory (``None`` when it is)."""
        position = self._position(seq)
        if position is None:
            return f"no log entry {seq}"
        if self._states[position] is not None:
            return None
        what = "the root state" if position == 0 else f"the state after entry {seq}"
        if position < self._unloaded:
            return (
                f"{what} is not in memory: this project was opened from a file; use "
                "Verify project to replay its log"
            )
        return (
            f"{what} is no longer in memory: only the newest {self._depth} states are "
            "kept (SessionSettings.undo_depth)"
        )

    # -- groups --------------------------------------------------------------------
    def allocate_group(self) -> int:
        """The open group's id, or a fresh one for a single action."""
        if self._open:
            return self._open[-1]
        group = self._next_group
        self._next_group += 1
        return group

    def group(self) -> AbstractContextManager[int]:
        """Actions inside share one group (undone as one); nesting joins the outer."""
        return self._group()

    @contextmanager
    def _group(self) -> Iterator[int]:
        if self._open:
            yield self._open[-1]
            return
        group = self._next_group
        self._next_group += 1
        self._open.append(group)
        try:
            yield group
        finally:
            self._open.pop()

    # -- mutation ------------------------------------------------------------------
    def push(self, entry: LogEntry, state: SessionState) -> None:
        """Append an action and its resulting state; truncates the redo tail."""
        del self._entries[self._cursor :]
        del self._states[self._cursor + 1 :]
        self._entries.append(entry)
        self._states.append(state)
        self._cursor += 1
        self._next_group = max(self._next_group, entry.group + 1)
        oldest = self._cursor - self._depth
        for i in range(max(oldest, 0)):
            self._states[i] = None

    def _target_back(self) -> int | None:
        if self._cursor == 0:
            return None
        group = self._entries[self._cursor - 1].group
        target = self._cursor - 1
        while target > 0 and self._entries[target - 1].group == group:
            target -= 1
        return target if self._states[target] is not None else None

    def _target_forward(self) -> int | None:
        if self._cursor >= len(self._entries):
            return None
        group = self._entries[self._cursor].group
        target = self._cursor + 1
        while target < len(self._entries) and self._entries[target].group == group:
            target += 1
        return target if self._states[target] is not None else None

    def undo(self) -> SessionState | None:
        """Move back one group; ``None`` (and no move) when nothing is reachable."""
        target = self._target_back()
        if target is None:
            return None
        self._cursor = target
        return self.head

    def redo(self) -> SessionState | None:
        """Move forward one group; ``None`` at the end of the redo tail."""
        target = self._target_forward()
        if target is None:
            return None
        self._cursor = target
        return self.head

    def goto(self, seq: int) -> SessionState | None:
        """Move to the state after entry ``seq`` (``0`` = root), if retained."""
        position = self._position(seq)
        if position is None or self._states[position] is None:
            return None
        self._cursor = position
        return self.head

    def replace_head(self, state: SessionState) -> None:
        """Put ``state`` at the cursor in place of the head (no entry is added)."""
        self._states[self._cursor] = state

    def restamp_last(self, timestamp: str) -> None:
        """Give the entry just before the cursor another ``timestamp``."""
        if self._cursor:
            entry = self._entries[self._cursor - 1]
            self._entries[self._cursor - 1] = replace(entry, timestamp=timestamp)

    # -- persistence -------------------------------------------------------------------
    @classmethod
    def from_snapshot(cls, s: SessionSnapshot, *, depth: int) -> History:
        """The history of a snapshot: its log, its retained states (or the head)."""
        history = cls.__new__(cls)
        history._depth = int(depth)
        history._entries = list(s.log)
        if s.states is not None and len(s.states) == len(s.log) + 1:
            states: list[SessionState | None] = list(s.states)
            states[-1] = s.head
            history._unloaded = 0
        else:
            states = [None] * len(s.log) + [s.head]
            history._unloaded = len(s.log)
        history._states = states
        history._cursor = len(s.log)
        history._open = []
        history._next_group = max((e.group for e in s.log), default=0) + 1
        return history


__all__ = ["History"]
