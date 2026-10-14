"""Interactive, step-wise pre-whitening: :class:`PreWhitenSession`.

A session is Period04's workflow as a library object: step through the extraction
one peak at a time, add, remove, fix or tie frequencies, delete points, fit zero
points per night, let amplitudes vary per season, undo any of it, and report the
result as a :class:`~cuperiod.PreWhitenResult`. :func:`cuperiod.prewhiten` is the same
session run to its stopping rule without any edit.
"""

from __future__ import annotations

from cuperiod.prewhiten.session.session import ComponentRef, PreWhitenSession
from cuperiod.prewhiten.session.state import (
    Candidate,
    Component,
    CreationArgs,
    Derived,
    FitStats,
    LogEntry,
    PointMask,
    ProjectError,
    SessionData,
    SessionError,
    SessionSnapshot,
    SessionState,
    SessionWarning,
    StepReport,
    Tie,
    UsedData,
    VariationResult,
)

__all__ = [
    "Candidate",
    "ComponentRef",
    "Component",
    "CreationArgs",
    "Derived",
    "FitStats",
    "LogEntry",
    "PointMask",
    "PreWhitenSession",
    "ProjectError",
    "SessionData",
    "SessionError",
    "SessionSnapshot",
    "SessionState",
    "SessionWarning",
    "StepReport",
    "Tie",
    "UsedData",
    "VariationResult",
]
