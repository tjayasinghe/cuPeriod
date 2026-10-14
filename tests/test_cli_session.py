"""CLI: prewhiten --save-project / --export-script and the session command group."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import cuperiod as cup
import prewhiten_corpus as pc
from cuperiod.cli.app import app
from cuperiod.prewhiten.session import PreWhitenSession

runner = CliRunner()

OPTIONS = ["--backend", "numpy", "--fmax", "30", "-n", "8"]


def _write_star(path: Path) -> Path:
    data, _, _ = pc.load(pc.case_by_name("triple-last"))
    t, y, e = data
    assert e is not None
    rows = "\n".join(
        f"{float(a)!r},{float(b)!r},{float(c)!r}"
        for a, b, c in zip(t, y, e, strict=True)
    )
    path.write_text("time,mag,mag_err\n" + rows + "\n", encoding="utf-8")
    return path


@pytest.fixture
def saved(tmp_path: Path) -> tuple[Path, Path, str]:
    """A pre-whitening run saved as a project and a script, plus its printed output."""
    star = _write_star(tmp_path / "star.csv")
    project = tmp_path / "star.cupw.json"
    script = tmp_path / "star.py"
    result = runner.invoke(
        app,
        [
            "prewhiten",
            str(star),
            *OPTIONS,
            "--save-project",
            str(project),
            "--export-script",
            str(script),
        ],
    )
    assert result.exit_code == 0, result.output
    return project, script, result.stdout


def test_prewhiten_saves_a_project_and_a_script(
    saved: tuple[Path, Path, str], tmp_path: Path
) -> None:
    project, script, output = saved
    assert f"Wrote {project}" in output and f"Wrote {script}" in output
    doc = json.loads(project.read_text(encoding="utf-8"))
    assert doc["data"] == {**doc["data"], "mode": "reference", "path": "star.csv"}
    # The recorded run is the plain one: same table as without the options.
    plain = runner.invoke(app, ["prewhiten", str(tmp_path / "star.csv"), *OPTIONS])
    assert plain.exit_code == 0, plain.output
    assert plain.stdout.splitlines()[:8] == output.splitlines()[:8]
    session = PreWhitenSession.load(project)
    reference = cup.prewhiten(
        cup.LightCurve.from_file(tmp_path / "star.csv"),
        settings=session.kernel.creation.settings,
    )
    np.testing.assert_array_equal(session.result().frequency, reference.frequency)
    # The script runs on its own and reproduces the run.
    ast.parse(script.read_text(encoding="utf-8"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUPERIOD_")}
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    run = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "Reproduced the exported session." in run.stdout


def test_session_verify_accepts_the_project(saved: tuple[Path, Path, str]) -> None:
    project, _, _ = saved
    result = runner.invoke(app, ["session", "verify", str(project)])
    assert result.exit_code == 0, result.output
    assert (
        "Verified" in result.stdout and "reproduce the stored solution" in result.stdout
    )
    result = runner.invoke(
        app,
        [
            "session",
            "verify",
            str(project),
            "--backend",
            "numpy",
            "--data",
            str(project.with_name("star.csv")),
        ],
    )
    assert result.exit_code == 0, result.output


def test_session_replay_reproduces_the_project(
    saved: tuple[Path, Path, str], tmp_path: Path
) -> None:
    project, _, _ = saved
    out = tmp_path / "replayed.cupw.json"
    result = runner.invoke(app, ["session", "replay", str(project), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "the stored solution is reproduced" in result.stdout
    original, replayed = PreWhitenSession.load(project), PreWhitenSession.load(out)
    assert replayed.state.same_content(original.state)
    assert [e.to_dict() for e in replayed.history] == [
        e.to_dict() for e in original.history
    ]


def test_session_script_writes_a_script(
    saved: tuple[Path, Path, str], tmp_path: Path
) -> None:
    project, _, _ = saved
    out = tmp_path / "again.py"
    result = runner.invoke(app, ["session", "script", str(project), "-o", str(out)])
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    ast.parse(text)
    assert "session.run(seeds=[" in text and "'/star.csv'" in text


@pytest.mark.parametrize("command", ["verify", "replay"])
def test_malformed_projects_exit_with_the_project_error(
    command: str, saved: tuple[Path, Path, str], tmp_path: Path
) -> None:
    project, _, _ = saved
    bad = tmp_path / "bad.cupw.json"
    bad.write_text('{"format": "cuperiod-prewhiten-project", "version": 7}')
    result = runner.invoke(app, ["session", command, str(bad)])
    assert result.exit_code == 1, result.output
    assert "Error:" in result.output and "newer cuPeriod" in result.output
    assert "Traceback" not in result.output
    doc = json.loads(project.read_text(encoding="utf-8"))
    doc["state"]["components"][0]["frequency"] *= 1.0 + 1e-6
    tampered = tmp_path / "tampered.cupw.json"
    tampered.write_text(json.dumps(doc), encoding="utf-8")
    result = runner.invoke(app, ["session", command, str(tampered)])
    assert result.exit_code == 1, result.output
    assert "Error:" in result.output
    assert "warning: the stored component values differ" in result.output


def test_a_missing_project_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["session", "verify", str(tmp_path / "none.cupw.json")])
    assert result.exit_code == 2, result.output


def test_the_session_group_lists_its_commands() -> None:
    result = runner.invoke(app, ["session", "--help"])
    assert result.exit_code == 0, result.output
    for name in ("verify", "script", "replay"):
        assert name in result.stdout
