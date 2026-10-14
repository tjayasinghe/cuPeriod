"""Exported session scripts: they run, reproduce the session, and resist injection."""

from __future__ import annotations

import ast
import dataclasses
import io
import os
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cuperiod as cup
import prewhiten_corpus as pc
from cuperiod.prewhiten.session import PreWhitenSession, actions, project, script
from cuperiod.prewhiten.session.state import (
    LogEntry,
    SessionError,
    SessionState,
    StepReport,
)
from cuperiod.prewhiten.subsets import Partition

SETTINGS = pc.case_by_name("triple-last").resolved_settings()


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data, _, _ = pc.load(pc.case_by_name("triple-last"))
    t, y, e = data
    assert e is not None
    return t, y, e


def _env() -> dict[str, str]:
    """The caller's environment without any cuPeriod setting (a CPU-only run)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUPERIOD_")}
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _execute(
    path: Path, *, save_to: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a script with ``sys.executable`` from an unrelated working directory.

    With ``save_to`` the script is run with one extra line that saves its session, so
    the test can compare the reproduction with the original exactly.
    """
    target = path
    if save_to is not None:
        target = path.with_name(path.stem + "_check.py")
        target.write_text(
            path.read_text(encoding="utf-8") + f"\nsession.save({str(save_to)!r})\n",
            encoding="utf-8",
        )
    return subprocess.run(
        [sys.executable, str(target)],
        capture_output=True,
        text=True,
        env=_env(),
        cwd=str(path.parent.parent),
        timeout=300,
        check=False,
    )


ALLOWED_IMPORTS = {"numpy", "cuperiod"}


def _check_public_only(text: str) -> ast.Module:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {a.name.split(".")[0] for a in node.names} <= ALLOWED_IMPORTS
            assert all(
                not part.startswith("_")
                for a in node.names
                for part in a.name.split(".")
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.split(".")[0] in ALLOWED_IMPORTS
            assert all(not part.startswith("_") for part in module.split("."))
            assert all(not a.name.startswith("_") for a in node.names)
        elif isinstance(node, ast.Attribute):
            assert not node.attr.startswith("_"), node.attr
        elif isinstance(node, ast.Name):
            assert node.id == "__file__" or not node.id.startswith("_"), node.id
            assert node.id not in {"eval", "exec", "compile", "open", "getattr"}
    return tree


# --- the script runs and reproduces the session ---------------------------------------


def test_exported_script_reproduces_the_session_bitwise(tmp_path: Path) -> None:
    session = PreWhitenSession(_data(), settings=SETTINGS)
    session.run()
    session.finalize()
    out = session.export_script(tmp_path / "triple.py", title="triple")
    assert out == tmp_path / "triple.py"
    assert (tmp_path / "triple.data.csv").exists()
    text = out.read_text(encoding="utf-8")
    _check_public_only(text)
    assert "BACKEND = None" in text and "session.run(seeds=[" in text
    saved = tmp_path / "again.cupw.json"
    run = _execute(out, save_to=saved)
    assert run.returncode == 0, run.stderr
    assert "Reproduced the exported session." in run.stdout
    again = PreWhitenSession.load(saved)
    assert again.state.same_content(session.state)
    assert [e.args for e in again.history] == [e.args for e in session.history]


def test_a_finufft_session_replays_on_numpy(tmp_path: Path) -> None:
    pytest.importorskip("finufft")
    session = PreWhitenSession(
        _data(), settings=SETTINGS.model_copy(update={"backend": "finufft"})
    )
    session.run()
    session.finalize()
    out = session.export_script(tmp_path / "f.py")
    text = out.read_text(encoding="utf-8")
    assert "# the session ran on finufft" in text
    out.write_text(
        text.replace("BACKEND = None", "BACKEND = 'numpy'"), encoding="utf-8"
    )
    saved = tmp_path / "f.cupw.json"
    run = _execute(out, save_to=saved)
    assert run.returncode == 0, run.stderr
    assert "Reproduced the exported session." in run.stdout
    again = PreWhitenSession.load(saved)
    assert again.backend == "numpy"
    for a, b in zip(session.components, again.components, strict=True):
        assert (a.frequency, a.amplitude, a.phase) == (
            b.frequency,
            b.amplitude,
            b.phase,
        )


def test_a_from_bands_script_reads_the_centred_data_as_they_are(tmp_path: Path) -> None:
    t, y, e = _data()
    bands = cup.MultiBandLightCurve.from_light_curves(
        {
            "V": cup.LightCurve.from_arrays(t[::2], y[::2], e[::2]),
            "B": cup.LightCurve.from_arrays(t[1::2], y[1::2] + 0.3, e[1::2]),
        }
    )
    session = PreWhitenSession.from_bands(bands, settings=SETTINGS)
    session.run(max_steps=2)
    out = session.export_script(tmp_path / "bands.py")
    text = out.read_text(encoding="utf-8")
    assert "from_csv(" in text and "partitions=['band']" in text
    lc, labels = project.read_session_csv(tmp_path / "bands.data.csv")
    np.testing.assert_array_equal(lc.value, session.data.value)  # still centred
    np.testing.assert_array_equal(lc.time, session.data.time)
    assert labels["band"].tolist() == [str(b) for b in session.data.labels["band"]]
    saved = tmp_path / "b.cupw.json"
    run = _execute(out, save_to=saved)
    assert run.returncode == 0, run.stderr
    again = PreWhitenSession.load(saved)
    assert again.state.same_content(session.state)


def test_the_script_refuses_altered_data(tmp_path: Path) -> None:
    session = PreWhitenSession(_data(), settings=SETTINGS)
    session.run(max_steps=1)
    out = session.export_script(tmp_path / "s.py")
    csv_path = tmp_path / "s.data.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    time, value, error = lines[5].split(",")
    lines[5] = ",".join([time, repr(float(value) + 1e-3), error])
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run = _execute(out)
    assert run.returncode == 1
    assert "the data differ from the exported session's" in run.stderr
    assert "Reproduced" not in run.stdout


def test_a_script_reads_the_original_file_when_it_still_holds_the_data(
    tmp_path: Path,
) -> None:
    t, y, e = _data()
    source = tmp_path / "star.csv"
    rows = "\n".join(
        f"{float(a)!r},{float(b)!r},{float(c)!r}"
        for a, b, c in zip(t, y, e, strict=True)
    )
    source.write_text("time,mag,mag_err\n" + rows + "\n", encoding="utf-8")
    session = PreWhitenSession(cup.LightCurve.from_file(source), settings=SETTINGS)
    session.run(max_steps=2)
    out = session.export_script(tmp_path / "star.py")
    text = out.read_text(encoding="utf-8")
    assert "cup.LightCurve.from_file(DATA" in text and "'/star.csv'" in text
    assert not (tmp_path / "star.data.csv").exists()
    run = _execute(out)
    assert run.returncode == 0, run.stderr
    assert "Reproduced the exported session." in run.stdout
    # Without a target directory the script names the file by its absolute path.
    assert repr(source.resolve().as_posix()) in session.to_script()


def test_to_script_needs_a_data_source_or_an_expression() -> None:
    session = PreWhitenSession(_data(), settings=SETTINGS)
    session.run(max_steps=1)
    with pytest.raises(SessionError, match="data_expr"):
        session.to_script()
    text = session.to_script(data_expr="cup.LightCurve.from_file('star.csv')")
    assert "DATA = cup.LightCurve.from_file('star.csv')" in text
    _check_public_only(text)
    for bad in ("x\nimport os", "x = 1", "import os", "x y", "(1,"):
        with pytest.raises(SessionError):
            session.to_script(data_expr=bad)


# --- one renderer per op --------------------------------------------------------------

EXAMPLES: dict[str, dict[str, Any]] = {
    "step": {"seed": 12.5, "index": 40},
    "polish": {},
    "prune": {},
    "renumber": {},
    "add_peak": {"seed": None, "index": None},
    "add": {"frequency": 3.25, "refine": "last", "allow_close": False},
    "remove": {"ids": [2, 3], "cascade": True},
    "set_active": {"ids": [1], "active": False},
    "fix": {"ids": [1], "frequency": True, "amplitude": None, "phase": False},
    "set_values": {"id": 1, "frequency": 2.0, "amplitude": None, "phase": 0.5},
    "tie": {"id": 3, "expression": "F1 + F2", "allow_close": False},
    "set_variation": {"ids": [1], "mode": "both", "hold_frequency": True},
    "calculate": {},
    "improve": {"ids": None},
    "set_note": {"id": 1, "note": "x"},
    "delete_points": {"runs": [[1, 4], [9, 10]]},
    "restore_points": {"runs": None},
    "exclude_subsets": {"name": "night", "subsets": [0, 2], "excluded": True},
    "define_partition": {
        "name": "night",
        "labels": ["N1", "N2"],
        "runs": [[0, 5, 0], [5, 9, 1]],
        "weights": [1.0, 2.0],
        "origin": {"kind": "manual"},
    },
    "drop_partition": {"name": "night"},
    "set_subset_weights": {"name": "night", "weights": [1.0, 0.5]},
    "use_offsets": {"name": "night"},
    "use_variation": {"name": None},
    "use_weights": {"name": "night"},
    "configure": {"settings": SETTINGS.model_dump(mode="json")},
}


def test_every_op_has_a_renderer_whose_statement_parses() -> None:
    assert set(script.RENDERERS) == set(actions.ARGS) == set(EXAMPLES)
    for op, raw in EXAMPLES.items():
        args = actions.canonical(actions.parse(op, raw))
        entry = LogEntry(
            seq=1,
            group=1,
            op=op,
            args=args,
            report=StepReport(kind="edited", message="ok"),
            timestamp="t",
        )
        statement = script.render_entry(entry)
        tree = ast.parse(statement)
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr), op
        call = tree.body[0].value
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        assert call.func.attr == op


def test_sanitize_comment_and_literal() -> None:
    hostile = 'a\nb\r"""c\\d e f\x00g\x85h' + "x" * 300
    clean = script.sanitize_comment(hostile)
    assert "\n" not in clean and "\r" not in clean and " " not in clean
    assert '"""' not in clean.replace('\\"', "")
    assert len(clean) <= 2 * script.COMMENT_LENGTH
    assert (
        script.literal([1, 2.5, "x", None, True, (1,)])
        == "[1, 2.5, 'x', None, True, (1,)]"
    )
    with pytest.raises(SessionError):
        script.literal(float("nan"))
    with pytest.raises(SessionError):
        script.literal(object())


# --- hostile text ---------------------------------------------------------------------

HOSTILE = 'x\n"""\\  __import__(\'os\').system("echo pwned")\r\n#'


@pytest.fixture
def local_reducer() -> Any:
    from cuperiod.prewhiten.session.session import _load_ops

    _load_ops()
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


def _note(kernel: Any, state: SessionState, derived: Any, args: Any) -> Any:
    """A note that also lands in the report and the stop reason (free text x3)."""
    components = tuple(
        dataclasses.replace(c, note=args.note) if c.id == args.id else c
        for c in state.components
    )
    new = dataclasses.replace(
        state, components=components, stop_reason=args.note, status="stopped"
    )
    return new, derived, StepReport(kind="edited", message=args.note)


def _hostile_script(text: str, tmp_path: Path) -> str:
    t, y, e = _data()
    lc = cup.LightCurve.from_arrays(
        t, y, e, meta={"source": text + ".csv", "observer": text}
    )
    n = lc.n
    partition = Partition.from_runs(
        n,
        [(0, n // 2, 0), (n // 2, n, 1)],
        name=text[:64],
        labels=[text[:60] + "1", text[:60] + "2"],
    )
    session = PreWhitenSession(lc, settings=SETTINGS, partitions=[partition])
    session.run(max_steps=1)
    assert session.set_note(1, text).kind == "edited"
    out = session.export_script(tmp_path / f"s{len(text)}.py", title=text)
    return out.read_text(encoding="utf-8")


class _Blank(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        return ast.Constant("S") if isinstance(node.value, str) else node


def test_hostile_text_never_becomes_code(tmp_path: Path, local_reducer: Any) -> None:
    local_reducer("set_note", _note)
    benign = "plain text"
    hostile_text = _hostile_script(HOSTILE, tmp_path)
    benign_text = _hostile_script(benign.ljust(len(HOSTILE), "y"), tmp_path)
    hostile_tree = _check_public_only(hostile_text)
    benign_tree = _check_public_only(benign_text)
    # Exactly the statements a benign session produces: nothing was injected.
    blank = _Blank()
    assert ast.dump(blank.visit(ast.parse(hostile_text))) == ast.dump(
        blank.visit(benign_tree)
    )
    # Every occurrence of the payload is inside a comment or a string literal.
    for token in tokenize.generate_tokens(io.StringIO(hostile_text).readline):
        if "__import__" in token.string:
            assert token.type in (tokenize.COMMENT, tokenize.STRING), token
    # Values placed in code come back exactly.
    strings = {
        node.value
        for node in ast.walk(hostile_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert HOSTILE in strings  # the note, through repr()
    assert HOSTILE[:64] in strings  # the partition name
    assert any(s.startswith(HOSTILE[:60]) for s in strings)  # the labels
