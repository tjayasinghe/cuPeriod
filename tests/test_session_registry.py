"""The operation registry is complete: no op lacks a piece of its contract.

Every op named in ``actions.ARGS`` has an op class, a reducer, a ``describe()`` line of
its own, a script renderer whose statement parses, a typed facade wrapper that the
rendered statement calls and that forwards exactly the logged arguments to
``do()``, and a log entry that round-trips through the project format. A new op
that misses any of these fails here, by name.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
from typing import Any

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.prewhiten.session import PreWhitenSession, actions, project, script
from cuperiod.prewhiten.session import session as session_module
from cuperiod.prewhiten.session.state import LogEntry, StepReport
from cuperiod.prewhiten.subsets import parse_origin

N = 240
NIGHTS = 4


def _light_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    night = np.repeat(np.arange(NIGHTS), N // NIGHTS)
    t = night * 1.0 + 0.4 * np.sort(rng.random(N)) + 0.05
    t = np.sort(t)
    y = 0.03 * np.sin(2 * np.pi * 7.3 * t + 0.4) + 0.012 * np.sin(
        2 * np.pi * 11.9 * t + 1.1
    )
    error = np.full(N, 0.002)
    y = y + rng.normal(0.0, 0.002, N)
    return t, y, error, night.astype(str)


@pytest.fixture(scope="module")
def session() -> PreWhitenSession:
    t, y, error, night = _light_curve()
    return PreWhitenSession(
        (t, y, error),
        settings=cup.PreWhitenSettings(backend="numpy", max_frequencies=4),
        grid=(0.1, 0.01, 3000),
        labels={"night": night},
        partitions=["night"],
    )


def examples(settings: cup.PreWhitenSettings) -> dict[str, dict[str, Any]]:
    """One valid argument set per op (the form the log stores)."""
    half = N // 2
    changed = settings.model_copy(update={"snr_threshold": 4.5})
    return {
        "step": {"seed": 7.31, "index": 721},
        "polish": {},
        "prune": {},
        "renumber": {},
        "add_peak": {"seed": None, "index": None},
        "add": {"frequency": 3.25, "refine": "last", "allow_close": True},
        "remove": {"ids": [2, 3], "cascade": True},
        "set_active": {"ids": [1], "active": False},
        "fix": {"ids": [1, 2], "frequency": True, "amplitude": None, "phase": False},
        "set_values": {"id": 1, "frequency": 2.0, "amplitude": None, "phase": 0.5},
        "tie": {"id": 3, "expression": "F1 + F2", "allow_close": False},
        "set_variation": {"ids": [1], "mode": "both", "hold_frequency": False},
        "calculate": {},
        "improve": {"ids": [1, 2]},
        "set_note": {"id": 1, "note": "a note"},
        "delete_points": {"runs": [[1, 4], [9, 10]]},
        "restore_points": {"runs": [[1, 4]]},
        "exclude_subsets": {"name": "night", "subsets": [0, 2], "excluded": True},
        "define_partition": {
            "name": "half",
            "labels": ["H1", "H2"],
            "runs": [[0, half, 0], [half, N, 1]],
            "weights": [1.0, 2.0],
            "origin": {"kind": "manual"},
        },
        "drop_partition": {"name": "night"},
        "set_subset_weights": {"name": "night", "weights": [1.0, 0.5, 2.0, 1.0]},
        "use_offsets": {"name": "night"},
        "use_variation": {"name": None},
        "use_weights": {"name": "night"},
        "configure": {"settings": changed.model_dump(mode="json")},
    }


def _entry(op: str, args: dict[str, Any]) -> LogEntry:
    return LogEntry(
        seq=1,
        group=1,
        op=op,
        args=actions.canonical(actions.parse(op, args)),
        report=StepReport(kind="edited", message="ok"),
        timestamp="2026-10-14T00:00:00Z",
    )


def test_every_registry_table_names_the_same_ops(session: PreWhitenSession) -> None:
    ops = set(actions.ARGS)
    assert len(ops) == 25
    assert set(actions.OP_CLASS) == ops
    assert ops >= actions.ID_ADDRESSED
    assert {
        op for op, kind in actions.OP_CLASS.items() if kind == "auto"
    } == actions.PRISTINE_OPS
    assert set(script.RENDERERS) == ops
    assert set(examples(session.settings)) == ops


@pytest.mark.parametrize("op", sorted(actions.ARGS))
def test_every_op_has_a_reducer(op: str) -> None:
    session_module._load_ops()
    assert actions.reducer(op) is not None, op


@pytest.mark.parametrize("op", sorted(actions.ARGS))
def test_every_op_has_a_typed_facade_wrapper(op: str) -> None:
    wrapper = getattr(PreWhitenSession, op, None)
    assert callable(wrapper), op
    assert wrapper.__qualname__ == f"PreWhitenSession.{op}"


@pytest.mark.parametrize("op", sorted(actions.ARGS))
def test_every_op_describes_itself(op: str, session: PreWhitenSession) -> None:
    entry = _entry(op, examples(session.settings)[op])
    text = actions.describe(entry)
    head = text.removesuffix(": ok")
    assert head and head != op, op  # not the bare-op fallback
    assert "\n" not in text


@pytest.mark.parametrize("op", sorted(actions.ARGS))
def test_every_op_round_trips_through_the_project_format(
    op: str, session: PreWhitenSession
) -> None:
    entry = _entry(op, examples(session.settings)[op])
    # Canonical arguments are a fixed point of parse.
    assert actions.canonical(actions.parse(op, entry.args)) == entry.args
    assert LogEntry.from_dict(entry.to_dict()).to_dict() == entry.to_dict()
    assert project.log_entry(entry.to_dict()).to_dict() == entry.to_dict()


@pytest.mark.parametrize("op", sorted(actions.ARGS))
def test_every_rendered_statement_calls_the_wrapper_with_the_logged_args(
    op: str, session: PreWhitenSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """entry -> script statement -> facade wrapper -> do(op, args) == entry."""
    entry = _entry(op, examples(session.settings)[op])
    statement = script.render_entry(entry)
    tree = ast.parse(statement)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr), op
    call = tree.body[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    assert call.func.attr == op

    seen: list[tuple[str, dict[str, Any]]] = []

    def record(name: str, raw: Any) -> StepReport:
        seen.append((name, dict(raw)))
        return StepReport(kind="noop", message="recorded")

    monkeypatch.setattr(session, "_do", record)
    code = compile(tree, "<statement>", "exec")
    exec(code, {"session": session, "cup": cup, "parse_origin": parse_origin})
    assert [name for name, _ in seen] == [op]
    forwarded = actions.canonical(actions.parse(op, seen[0][1]))
    assert forwarded == entry.args


def test_wrapper_signatures_accept_the_renderer_keywords() -> None:
    """Each keyword a renderer writes is a parameter of the wrapper."""
    for op, render in script.RENDERERS.items():
        parameters = inspect.signature(getattr(PreWhitenSession, op)).parameters
        call = ast.parse(render(_example_args(op))).body[0]
        assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
        for keyword in call.value.keywords:
            assert keyword.arg in parameters, (op, keyword.arg)


def _example_args(op: str) -> dict[str, Any]:
    return actions.canonical(
        actions.parse(op, examples(cup.PreWhitenSettings())[op])
    )


# --- no tolerated missing module ---------------------------------------------------


def test_a_missing_ops_module_is_an_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "cuperiod.prewhiten.session.ops_data"
    monkeypatch.setattr(session_module, "_ops_loaded", False)
    monkeypatch.setitem(sys.modules, name, None)  # type: ignore[arg-type]
    with pytest.raises(ImportError):
        session_module._load_ops()
    assert session_module._ops_loaded is False  # a later call tries again


def test_a_missing_delegate_module_is_an_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules, "cuperiod.prewhiten.session.script", None  # type: ignore[arg-type]
    )
    with pytest.raises(ImportError):
        session_module._delegate("script")


def test_ops_modules_load_lazily_and_register_everything() -> None:
    for name in session_module._OP_MODULES:
        module = importlib.import_module(f"cuperiod.prewhiten.session.{name}")
        assert module is not None
    session_module._load_ops()
    missing = [op for op in actions.ARGS if actions.reducer(op) is None]
    assert missing == []
