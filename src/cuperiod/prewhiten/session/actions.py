"""The operation registry of an interactive pre-whitening session.

Every change to a session is one *operation*: a name, a pydantic argument model
(``extra="forbid"``, finite floats only) and a pure *reducer*
``(kernel, state, derived, args) -> (state, derived | None, report)``. The argument
models and the op-class table live here for all operations; reducers register
themselves with :func:`register` from the modules that implement them, so the
registry is open (a test may register a local reducer).

The log stores :func:`canonical` arguments: resolved facts only (the seed a step
used, the frequency a click snapped to, the index runs a lasso selected), never the
gesture that produced them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from cuperiod.core.config import PreWhitenSettings
from cuperiod.prewhiten.constraints import MAX_TIE_LENGTH, Variation
from cuperiod.prewhiten.session.state import (
    Derived,
    LogEntry,
    SessionError,
    SessionState,
    StepReport,
    parse_ref,
)

if TYPE_CHECKING:
    from cuperiod.prewhiten.session.kernel import SessionKernel

OpClass = Literal["auto", "attempt", "manual", "neutral"]
RefineName = Literal["none", "last", "cyclic", "simultaneous"]

#: Largest serialized ``origin`` a ``define_partition`` may carry.
MAX_ORIGIN_BYTES = 4096

_TIE_ALPHABET = re.compile(r"^[Ff0-9eE.+\-−*=\s]*$")


# --- argument models -----------------------------------------------------------------


class OpArgs(BaseModel):
    """Base of every operation's arguments: strict, immutable, finite."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _id(value: Any) -> Any:
    """Accept ``"F3"``, ``"3"`` and numpy integers for an id; leave anything else to
    pydantic (``True`` is no id)."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (str, bool)):
        return parse_ref(value)
    return value


def _ids(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_id(v) for v in value]
    return value


class SeedArgs(OpArgs):
    """``step`` / ``add_peak``: an optional logged seed and its grid index."""

    seed: float | None = Field(default=None, gt=0.0)
    index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _both_or_neither(self) -> SeedArgs:
        if (self.seed is None) != (self.index is None):
            raise ValueError("seed and index are given together or not at all")
        return self


class NoArgs(OpArgs):
    """``polish``, ``prune``, ``renumber``, ``calculate``."""


class AddArgs(OpArgs):
    frequency: float = Field(gt=0.0)
    refine: RefineName | None = None
    allow_close: bool = False


class _IdsArgs(OpArgs):
    ids: list[int] = Field(min_length=1)

    @field_validator("ids", mode="before")
    @classmethod
    def _refs(cls, value: Any) -> Any:
        return _ids(value)


class RemoveArgs(_IdsArgs):
    cascade: bool = False


class SetActiveArgs(_IdsArgs):
    active: bool


class FixArgs(_IdsArgs):
    frequency: bool | None = None
    amplitude: bool | None = None
    phase: bool | None = None


class _IdArgs(OpArgs):
    id: int

    @field_validator("id", mode="before")
    @classmethod
    def _ref(cls, value: Any) -> Any:
        return _id(value)


class SetValuesArgs(_IdArgs):
    frequency: float | None = Field(default=None, gt=0.0)
    amplitude: float | None = None
    phase: float | None = None


class TieArgs(_IdArgs):
    expression: str | None = Field(default=None, max_length=MAX_TIE_LENGTH)
    allow_close: bool = False

    @field_validator("expression")
    @classmethod
    def _alphabet(cls, value: str | None) -> str | None:
        if value is not None and not _TIE_ALPHABET.match(value):
            raise ValueError(
                "a tie uses only labels (F1), integers, + - * = and spaces"
            )
        return value


class SetVariationArgs(_IdsArgs):
    mode: Variation
    hold_frequency: bool = True


class ImproveArgs(OpArgs):
    ids: list[int] | None = None

    @field_validator("ids", mode="before")
    @classmethod
    def _refs(cls, value: Any) -> Any:
        return _ids(value)


class SetNoteArgs(_IdArgs):
    note: str = Field(max_length=200)


Run = tuple[int, int]


def _check_runs(runs: list[Run]) -> list[Run]:
    for start, stop in runs:
        if start < 0 or stop <= start:
            raise ValueError(f"run ({start}, {stop}) is not a half-open index range")
    return runs


class DeletePointsArgs(OpArgs):
    runs: list[Run] = Field(min_length=1)

    @field_validator("runs")
    @classmethod
    def _valid(cls, value: list[Run]) -> list[Run]:
        return _check_runs(value)


class RestorePointsArgs(OpArgs):
    runs: list[Run] | None = None

    @field_validator("runs")
    @classmethod
    def _valid(cls, value: list[Run] | None) -> list[Run] | None:
        return None if value is None else _check_runs(value)


class ExcludeSubsetsArgs(OpArgs):
    name: str = Field(min_length=1, max_length=64)
    subsets: list[int] = Field(min_length=1)
    excluded: bool = True


class DefinePartitionArgs(OpArgs):
    name: str = Field(min_length=1, max_length=64)
    labels: list[str] = Field(min_length=1)
    runs: list[tuple[int, int, int]]
    weights: list[float]
    origin: dict[str, Any] = Field(default_factory=lambda: {"kind": "manual"})

    @field_validator("labels")
    @classmethod
    def _labels(cls, value: list[str]) -> list[str]:
        if any(len(label) > 64 for label in value):
            raise ValueError("subset labels are at most 64 characters")
        return value

    @field_validator("weights")
    @classmethod
    def _weights(cls, value: list[float]) -> list[float]:
        if not all(w > 0.0 for w in value):
            raise ValueError("subset weights must be positive")
        return value

    @field_validator("origin")
    @classmethod
    def _origin(cls, value: dict[str, Any]) -> dict[str, Any]:
        from cuperiod.prewhiten.session.state import canonical_json

        try:
            size = len(canonical_json(value).encode())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"origin is not plain JSON data: {exc}") from exc
        if size > MAX_ORIGIN_BYTES:
            raise ValueError(f"origin exceeds {MAX_ORIGIN_BYTES} bytes")
        return value


class NameArgs(OpArgs):
    """``drop_partition``."""

    name: str = Field(min_length=1, max_length=64)


class SetSubsetWeightsArgs(OpArgs):
    name: str = Field(min_length=1, max_length=64)
    weights: list[float] = Field(min_length=1)

    @field_validator("weights")
    @classmethod
    def _weights(cls, value: list[float]) -> list[float]:
        if not all(w > 0.0 for w in value):
            raise ValueError("subset weights must be positive")
        return value


class UseArgs(OpArgs):
    """``use_offsets``, ``use_variation``, ``use_weights``: a partition or none."""

    name: str | None = Field(default=None, min_length=1, max_length=64)


class ConfigureArgs(OpArgs):
    """``configure``: the complete requested settings (every field)."""

    settings: dict[str, Any]

    @field_validator("settings")
    @classmethod
    def _known(cls, value: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(value) - set(PreWhitenSettings.model_fields))
        if unknown:
            raise ValueError(f"unknown settings {unknown}")
        for key, item in value.items():
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"setting {key!r} must be finite")
        return value


#: Argument model of every operation (the 25 of the log schema).
ARGS: Mapping[str, type[OpArgs]] = {
    "step": SeedArgs,
    "polish": NoArgs,
    "prune": NoArgs,
    "renumber": NoArgs,
    "add_peak": SeedArgs,
    "add": AddArgs,
    "remove": RemoveArgs,
    "set_active": SetActiveArgs,
    "fix": FixArgs,
    "set_values": SetValuesArgs,
    "tie": TieArgs,
    "set_variation": SetVariationArgs,
    "calculate": NoArgs,
    "improve": ImproveArgs,
    "set_note": SetNoteArgs,
    "delete_points": DeletePointsArgs,
    "restore_points": RestorePointsArgs,
    "exclude_subsets": ExcludeSubsetsArgs,
    "define_partition": DefinePartitionArgs,
    "drop_partition": NameArgs,
    "set_subset_weights": SetSubsetWeightsArgs,
    "use_offsets": UseArgs,
    "use_variation": UseArgs,
    "use_weights": UseArgs,
    "configure": ConfigureArgs,
}

#: The op-class table of the specification (section 2.4): what an op does to the
#: status, the blocked seeds, the attempt budget and the pristine flag.
OP_CLASS: Mapping[str, OpClass] = {
    "step": "auto",
    "polish": "auto",
    "prune": "auto",
    "renumber": "auto",
    "add_peak": "attempt",
    "add": "manual",
    "remove": "manual",
    "set_active": "manual",
    "fix": "manual",
    "set_values": "manual",
    "tie": "manual",
    "set_variation": "manual",
    "calculate": "manual",
    "improve": "manual",
    "delete_points": "manual",
    "restore_points": "manual",
    "exclude_subsets": "manual",
    "set_subset_weights": "manual",
    "use_offsets": "manual",
    "use_variation": "manual",
    "use_weights": "manual",
    "configure": "manual",
    "set_note": "neutral",
    "define_partition": "neutral",
    "drop_partition": "neutral",
}

#: Ops whose arguments name component ids (a renumbering invalidates them).
ID_ADDRESSED: frozenset[str] = frozenset(
    {
        "remove",
        "set_active",
        "fix",
        "set_values",
        "tie",
        "set_variation",
        "improve",
        "set_note",
    }
)

#: Ops a pristine session may log (the auto class): a session whose effective log
#: holds only these is still the automatic run, so Auto may restart it.
PRISTINE_OPS: frozenset[str] = frozenset(
    op for op, kind in OP_CLASS.items() if kind == "auto"
)


# --- reducers ---------------------------------------------------------------------

Reducer: TypeAlias = Callable[
    ["SessionKernel", SessionState, Derived, Any],
    tuple[SessionState, "Derived | None", StepReport],
]

_REDUCERS: dict[str, Reducer] = {}


def register(op: str) -> Callable[[Reducer], Reducer]:
    """Decorator registering ``reducer`` for ``op`` (replacing any previous one)."""
    if op not in ARGS:
        raise SessionError(f"unknown op {op!r}")

    def decorate(reducer: Reducer) -> Reducer:
        _REDUCERS[op] = reducer
        return reducer

    return decorate


def unregister(op: str) -> Reducer | None:
    """Remove and return the reducer of ``op`` (tests use it to restore a registry)."""
    return _REDUCERS.pop(op, None)


def reducer(op: str) -> Reducer | None:
    """The registered reducer of ``op``, or ``None``."""
    return _REDUCERS.get(op)


def parse(op: str, raw: Mapping[str, Any]) -> OpArgs:
    """Validate ``raw`` arguments of ``op``.

    Raises
    ------
    SessionError
        For an unknown op or invalid arguments (the message names the field).
    """
    model = ARGS.get(op)
    if model is None:
        raise SessionError(f"unknown op {op!r}")
    try:
        return model.model_validate(dict(raw))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or op}: {err['msg']}"
            for err in exc.errors()
        )
        raise SessionError(f"{op}: {problems}") from exc
    except SessionError as exc:  # a bad "F3"-style reference
        raise SessionError(f"{op}: {exc}") from exc


def canonical(args: OpArgs) -> dict[str, Any]:
    """Plain JSON data of validated arguments (what the log stores)."""
    out: dict[str, Any] = args.model_dump(mode="json")
    return out


# --- describe -------------------------------------------------------------------------


def _labels(ids: Any) -> str:
    return ", ".join(f"F{i}" for i in ids) if ids else "all"


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:.9g}"


def _runs_count(runs: Any) -> int:
    return sum(int(b) - int(a) for a, b in runs or ())


_DESCRIBE: Mapping[str, Callable[[Mapping[str, Any]], str]] = {
    "step": lambda a: "Step"
    + ("" if a.get("seed") is None else f" (seed {_fmt(a['seed'])} c/d)"),
    "polish": lambda a: "Polish (refine all frequencies)",
    "prune": lambda a: "Prune insignificant components",
    "renumber": lambda a: "Renumber components",
    "add_peak": lambda a: "Add tallest peak"
    + ("" if a.get("seed") is None else f" (seed {_fmt(a['seed'])} c/d)"),
    "add": lambda a: f"Add {_fmt(a.get('frequency'))} c/d",
    "remove": lambda a: f"Remove {_labels(a.get('ids'))}"
    + (" (dissolving ties)" if a.get("cascade") else ""),
    "set_active": lambda a: (
        f"{'Activate' if a.get('active') else 'Deactivate'} {_labels(a.get('ids'))}"
    ),
    "fix": lambda a: f"Fix/free {_labels(a.get('ids'))} ("
    + ", ".join(
        f"{name} {'fixed' if a[name] else 'free'}"
        for name in ("frequency", "amplitude", "phase")
        if a.get(name) is not None
    )
    + ")",
    "set_values": lambda a: f"Set values of F{a.get('id')}",
    "tie": lambda a: (
        f"Untie F{a.get('id')}"
        if a.get("expression") is None
        else f"Tie F{a.get('id')} to {a['expression']}"
    ),
    "set_variation": lambda a: f"Vary {_labels(a.get('ids'))}: {a.get('mode')}",
    "calculate": lambda a: "Calculate amplitudes and phases",
    "improve": lambda a: f"Improve {_labels(a.get('ids'))}",
    "set_note": lambda a: f"Note on F{a.get('id')}",
    "delete_points": lambda a: f"Delete {_runs_count(a.get('runs'))} point(s)",
    "restore_points": lambda a: (
        "Restore all points"
        if a.get("runs") is None
        else f"Restore {_runs_count(a.get('runs'))} point(s)"
    ),
    "exclude_subsets": lambda a: (
        f"{'Exclude' if a.get('excluded', True) else 'Include'} "
        f"{len(a.get('subsets') or ())} subset(s) of {a.get('name')!r}"
    ),
    "define_partition": lambda a: (
        f"Define partition {a.get('name')!r} ({len(a.get('labels') or ())} subsets)"
    ),
    "drop_partition": lambda a: f"Drop partition {a.get('name')!r}",
    "set_subset_weights": lambda a: f"Set subset weights of {a.get('name')!r}",
    "use_offsets": lambda a: (
        "No subset zero points"
        if a.get("name") is None
        else f"Zero points per {a['name']!r}"
    ),
    "use_variation": lambda a: (
        "No variation partition"
        if a.get("name") is None
        else f"Variation per {a['name']!r}"
    ),
    "use_weights": lambda a: (
        "No subset weights" if a.get("name") is None else f"Weights of {a['name']!r}"
    ),
    "configure": lambda a: "Change settings",
}


_WORD = re.compile(r"[A-Za-z0-9.]+")


def _says(text: str, other: str) -> bool:
    """Whether ``text`` names every content word of ``other`` (case-insensitive).

    A word with a digit (``"F3"``, ``"22"``, ``"15.4"``) must appear as it is; any
    other word of three or more letters is named by a word of ``text`` that starts
    with it, so an inflected restatement counts (``"restored 22 points"`` names
    ``"Restore 22 point(s)"``) while a different number, label or quantity does not
    (``"F4 added at 15.3966"`` does not name ``"Add 15.4"``).
    """
    words = {w.lower().rstrip(".") for w in _WORD.findall(text)}
    for word in _WORD.findall(other):
        word = word.lower().rstrip(".")
        if any(ch.isdigit() for ch in word):
            if word not in words:
                return False
        elif len(word) >= 3 and not any(w.startswith(word) for w in words):
            return False
    return True


def describe(entry: LogEntry) -> str:
    """One human-readable line for a log entry (History dock, script comments).

    ``"<action>: <outcome>"``, or only the outcome when it restates every word of the
    action (``"Restored 22 points"``, not ``"Restore 22 point(s): restored 22
    points"``); the outcome is kept because it says what happened, with the numbers
    of the run.
    """
    formatter = _DESCRIBE.get(entry.op)
    head = formatter(entry.args) if formatter is not None else entry.op
    message = entry.report.message
    if not message:
        return head
    if formatter is not None and _says(message, head):
        return message[:1].upper() + message[1:]
    return f"{head}: {message}"


__all__ = [
    "ARGS",
    "ID_ADDRESSED",
    "MAX_ORIGIN_BYTES",
    "OP_CLASS",
    "PRISTINE_OPS",
    "AddArgs",
    "ConfigureArgs",
    "DefinePartitionArgs",
    "DeletePointsArgs",
    "ExcludeSubsetsArgs",
    "FixArgs",
    "ImproveArgs",
    "NameArgs",
    "NoArgs",
    "OpArgs",
    "OpClass",
    "Reducer",
    "RemoveArgs",
    "RestorePointsArgs",
    "SeedArgs",
    "SetActiveArgs",
    "SetNoteArgs",
    "SetSubsetWeightsArgs",
    "SetValuesArgs",
    "SetVariationArgs",
    "TieArgs",
    "UseArgs",
    "canonical",
    "describe",
    "parse",
    "reducer",
    "register",
    "unregister",
]
