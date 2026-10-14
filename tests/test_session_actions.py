"""The op registry: argument models, op classes, canonical JSON and describe()."""

from __future__ import annotations

import json
from typing import Any

import pytest

from cuperiod.prewhiten.session import actions
from cuperiod.prewhiten.session.state import LogEntry, SessionError, StepReport

LOG_OPS = {
    "step", "polish", "prune", "renumber", "add_peak", "add", "remove", "set_active",
    "fix", "set_values", "tie", "set_variation", "calculate", "improve", "set_note",
    "delete_points", "restore_points", "exclude_subsets", "define_partition",
    "drop_partition", "set_subset_weights", "use_offsets", "use_variation",
    "use_weights", "configure",
}

#: One valid argument set per op (the round-trip fixtures).
VALID: dict[str, dict[str, Any]] = {
    "step": {"seed": 12.345678901234567, "index": 17},
    "polish": {},
    "prune": {},
    "renumber": {},
    "add_peak": {},
    "add": {"frequency": 7.25, "refine": "none", "allow_close": True},
    "remove": {"ids": [3, 4], "cascade": True},
    "set_active": {"ids": [2], "active": False},
    "fix": {"ids": [1], "frequency": True, "amplitude": None, "phase": False},
    "set_values": {"id": 2, "frequency": 3.5, "amplitude": -0.01, "phase": 1.0},
    "tie": {"id": 3, "expression": "F1 + F2", "allow_close": False},
    "set_variation": {"ids": [1], "mode": "phase", "hold_frequency": False},
    "calculate": {},
    "improve": {"ids": [1, 2]},
    "set_note": {"id": 1, "note": "dominant mode"},
    "delete_points": {"runs": [[0, 3], [10, 11]]},
    "restore_points": {"runs": None},
    "exclude_subsets": {"name": "night", "subsets": [0, 2], "excluded": True},
    "define_partition": {
        "name": "night", "labels": ["N1", "N2"], "runs": [[0, 5, 0], [5, 9, 1]],
        "weights": [1.0, 2.0], "origin": {"kind": "gaps", "min_gap": 0.3},
    },
    "drop_partition": {"name": "night"},
    "set_subset_weights": {"name": "night", "weights": [1.0, 0.5]},
    "use_offsets": {"name": "night"},
    "use_variation": {"name": None},
    "use_weights": {"name": "night"},
    "configure": {"settings": {"snr_threshold": 4.5, "maximum_frequency": None}},
}


def test_registry_covers_exactly_the_log_ops() -> None:
    assert set(actions.ARGS) == LOG_OPS
    assert set(actions.OP_CLASS) == LOG_OPS
    assert set(VALID) == LOG_OPS
    assert actions.ID_ADDRESSED <= LOG_OPS


def test_op_class_table() -> None:
    by_class: dict[str, set[str]] = {}
    for op, kind in actions.OP_CLASS.items():
        by_class.setdefault(kind, set()).add(op)
    assert by_class["auto"] == {"step", "polish", "prune", "renumber"}
    assert by_class["attempt"] == {"add_peak"}
    assert by_class["neutral"] == {"set_note", "define_partition", "drop_partition"}
    assert len(by_class["manual"]) == 17
    assert by_class["auto"] == actions.PRISTINE_OPS


@pytest.mark.parametrize("op", sorted(LOG_OPS))
def test_valid_args_round_trip_through_canonical_json(op: str) -> None:
    parsed = actions.parse(op, VALID[op])
    data = actions.canonical(parsed)
    again = actions.parse(op, json.loads(json.dumps(data, allow_nan=False)))
    assert actions.canonical(again) == data
    if op == "step":
        assert data["seed"] == VALID["step"]["seed"]  # repr-exact


@pytest.mark.parametrize("op", sorted(LOG_OPS))
def test_extras_are_forbidden(op: str) -> None:
    with pytest.raises(SessionError, match="extra|Extra"):
        actions.parse(op, {**VALID[op], "surprise": 1})


@pytest.mark.parametrize(
    ("op", "args"),
    [
        ("step", {"seed": float("nan"), "index": 1}),
        ("step", {"seed": float("inf"), "index": 1}),
        ("add", {"frequency": float("inf")}),
        ("set_values", {"id": 1, "amplitude": float("nan")}),
        ("set_subset_weights", {"name": "n", "weights": [1.0, float("inf")]}),
        ("define_partition", {"name": "n", "labels": ["a"], "runs": [[0, 1, 0]],
                              "weights": [float("nan")]}),
        ("configure", {"settings": {"snr_threshold": float("nan")}}),
    ],
)
def test_non_finite_floats_are_rejected(op: str, args: dict[str, Any]) -> None:
    with pytest.raises(SessionError):
        actions.parse(op, args)


@pytest.mark.parametrize(
    ("op", "args"),
    [
        ("step", {"seed": 1.0}),  # index missing
        ("step", {"index": 3}),
        ("step", {"seed": -1.0, "index": 3}),
        ("add", {"frequency": 0.0}),
        ("remove", {"ids": []}),
        ("tie", {"id": 1, "expression": "__import__('os')"}),
        ("tie", {"id": 1, "expression": "F1 +" * 100}),
        ("set_note", {"id": 1, "note": "x" * 201}),
        ("delete_points", {"runs": [[5, 5]]}),
        ("set_variation", {"ids": [1], "mode": "sometimes"}),
        ("configure", {"settings": {"not_a_setting": 1}}),
        ("define_partition", {"name": "n", "labels": ["a"], "runs": [[0, 1, 0]],
                              "weights": [1.0], "origin": {"x": "y" * 5000}}),
        ("nonexistent", {}),
    ],
)
def test_invalid_args_raise_session_error(op: str, args: dict[str, Any]) -> None:
    with pytest.raises(SessionError):
        actions.parse(op, args)


def test_ids_accept_labels() -> None:
    parsed = actions.parse("remove", {"ids": ["F3", 4]})
    assert actions.canonical(parsed) == {"ids": [3, 4], "cascade": False}
    assert actions.canonical(actions.parse("set_note", {"id": "f2", "note": ""}))[
        "id"
    ] == 2


@pytest.mark.parametrize("op", sorted(LOG_OPS))
def test_describe_has_a_line_for_every_op(op: str) -> None:
    args = actions.canonical(actions.parse(op, VALID[op]))
    entry = LogEntry(seq=1, group=1, op=op, args=args,
                     report=StepReport(kind="edited", message="done"),
                     timestamp="t")
    line = actions.describe(entry)
    assert line.endswith(": done") and "\n" not in line
    assert not line.startswith(op + ":")  # a real description, not the op name


@pytest.mark.parametrize(
    ("op", "args", "message", "line"),
    [
        # The outcome restates the action: only the outcome, which says what happened.
        ("restore_points", {"runs": [[0, 22]]}, "restored 22 points",
         "Restored 22 points"),
        ("improve", {"ids": None}, "improved all 4 frequencies",
         "Improved all 4 frequencies"),
        ("calculate", {}, "calculated amplitudes, phases and zero points",
         "Calculated amplitudes, phases and zero points"),
        ("tie", {"id": 3, "expression": "F1 + F2"}, "tied F3 = F1 + F2 (24.7 c/d)",
         "Tied F3 = F1 + F2 (24.7 c/d)"),
        # The action says something the outcome does not: both are kept.
        ("restore_points", {"runs": None}, "restored 22 points",
         "Restore all points: restored 22 points"),
        ("add", {"frequency": 15.4}, "F4 added at 15.3966 c/d",
         "Add 15.4 c/d: F4 added at 15.3966 c/d"),
        ("fix", {"ids": [1], "frequency": True}, "fixed/freed F1",
         "Fix/free F1 (frequency fixed): fixed/freed F1"),
        ("improve", {"ids": [1, 2]}, "improved F2",
         "Improve F1, F2: improved F2"),
    ],
)
def test_describe_does_not_say_everything_twice(
    op: str, args: dict[str, Any], message: str, line: str
) -> None:
    entry = LogEntry(
        seq=1, group=1, op=op, args=actions.canonical(actions.parse(op, args)),
        report=StepReport(kind="edited", message=message), timestamp="t",
    )
    assert actions.describe(entry) == line


def test_register_and_reducer() -> None:
    assert actions.reducer("set_note") is None or callable(actions.reducer("set_note"))
    previous = actions.unregister("set_note")
    try:
        @actions.register("set_note")
        def local(kernel: Any, state: Any, derived: Any, args: Any) -> Any:
            return state, derived, StepReport(kind="noop", message="")

        assert actions.reducer("set_note") is local
    finally:
        actions.unregister("set_note")
        if previous is not None:
            actions.register("set_note")(previous)
    with pytest.raises(SessionError):
        actions.register("bogus")
