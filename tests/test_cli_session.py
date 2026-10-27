"""CLI: prewhiten --save-project / --export-script and the session command group."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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


# --- release gate (1.3.0): --data and --save-project ------------------------------


def _write_renamed_star(path: Path) -> Path:
    """The star with columns the reader cannot guess: only a mapping finds them."""
    star = _write_star(path)
    text = star.read_text(encoding="utf-8")
    star.write_text(text.replace("time,mag,mag_err", "t_obs,m,dm", 1), encoding="utf-8")
    return star


@pytest.mark.parametrize("command", ["verify", "script", "replay"])
def test_session_data_is_read_with_the_projects_columns(
    command: str, tmp_path: Path
) -> None:
    star = _write_renamed_star(tmp_path / "odd.csv")
    project = tmp_path / "odd.cupw.json"
    created = runner.invoke(
        app,
        ["prewhiten", str(star), *OPTIONS, "--time", "t_obs", "--value", "m",
         "--error", "dm", "--save-project", str(project)],
    )
    assert created.exit_code == 0, created.output
    moved = tmp_path / "moved" / "odd_copy.csv"
    moved.parent.mkdir()
    moved.write_bytes(star.read_bytes())
    extra = ["-o", str(tmp_path / "odd.py")] if command == "script" else []
    result = runner.invoke(
        app, ["session", command, str(project), "--data", str(moved), *extra]
    )
    assert result.exit_code == 0, result.output


def test_session_data_keeps_the_projects_domain(tmp_path: Path) -> None:
    from cuperiod.cli.app import _project_data

    star = _write_star(tmp_path / "star.csv")
    project = tmp_path / "star.cupw.json"
    created = runner.invoke(
        app,
        ["prewhiten", str(star), *OPTIONS, "--domain", "flux",
         "--save-project", str(project)],
    )
    assert created.exit_code == 0, created.output
    # A "mag" column alone would be read as magnitudes.
    lc = _project_data(project, star)
    assert lc is not None and lc.domain == cup.Domain.FLUX
    result = runner.invoke(
        app, ["session", "verify", str(project), "--data", str(star)]
    )
    assert result.exit_code == 0, result.output


def test_save_project_embeds_large_data_kept_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The automatic-embed limit, made small: the data file is outside the project's
    # folder, so it cannot be referenced and auto-embedding refuses it.
    monkeypatch.setenv("CUPERIOD_SESSION_EMBED_DATA_MAX_POINTS", "10")
    (tmp_path / "data").mkdir()
    (tmp_path / "project").mkdir()
    star = _write_star(tmp_path / "data" / "star.csv")
    project = tmp_path / "project" / "star.cupw.json"
    script = tmp_path / "project" / "star.py"
    result = runner.invoke(
        app,
        ["prewhiten", str(star), *OPTIONS, "--save-project", str(project),
         "--export-script", str(script)],
    )
    assert result.exit_code == 0, result.output
    assert "note: embedded all" in result.stdout
    assert f"Wrote {project}" in result.stdout and f"Wrote {script}" in result.stdout
    doc = json.loads(project.read_text(encoding="utf-8"))
    assert doc["data"]["mode"] == "embedded"
    verified = runner.invoke(app, ["session", "verify", str(project)])
    assert verified.exit_code == 0, verified.output


def test_export_script_survives_a_failed_project_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUPERIOD_SESSION_EMBED_DATA_MAX_POINTS", "10")
    monkeypatch.setenv("CUPERIOD_SESSION_MAX_PROJECT_BYTES", "2000")
    (tmp_path / "data").mkdir()
    (tmp_path / "project").mkdir()
    star = _write_star(tmp_path / "data" / "star.csv")
    project = tmp_path / "project" / "star.cupw.json"
    script = tmp_path / "project" / "star.py"
    result = runner.invoke(
        app,
        ["prewhiten", str(star), *OPTIONS, "--save-project", str(project),
         "--export-script", str(script)],
    )
    assert result.exit_code == 1, result.output
    assert "Error:" in result.output and "max_project_bytes" in result.output
    assert not project.exists()
    assert script.exists() and f"Wrote {script}" in result.stdout


def test_session_commands_pass_other_warnings_through_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A non-session warning (a column read, say) used to be re-emitted inside the
    # recording context, recorded again, and re-emitted forever: the command hung.
    import warnings

    from cuperiod.cli.app import _session_warnings

    calls: list[object] = []
    real = warnings.warn_explicit

    def counting(*args: Any, **kwargs: Any) -> None:
        calls.append(args)
        if len(calls) > 5:
            raise RuntimeError("the warning is re-emitted in a loop")
        real(*args, **kwargs)

    monkeypatch.setattr(warnings, "warn_explicit", counting)
    with pytest.warns(UserWarning, match="a column read"), _session_warnings():
        warnings.warn("a column read", UserWarning, stacklevel=1)
    assert len(calls) == 1


# --- release gate (1.3.0), round 2 -------------------------------------------------


def test_export_script_survives_a_project_folder_that_does_not_exist(
    tmp_path: Path,
) -> None:
    # An OSError from --save-project (here: no such folder) escaped before the
    # script was written, and the error named the temporary file, not the project.
    star = _write_star(tmp_path / "star.csv")
    project = tmp_path / "no_such_dir" / "star.cupw.json"
    script = tmp_path / "star.py"
    result = runner.invoke(
        app,
        ["prewhiten", str(star), *OPTIONS, "--save-project", str(project),
         "--export-script", str(script)],
    )
    assert result.exit_code == 2, result.output
    assert script.exists() and f"Wrote {script}" in result.stdout
    assert "--save-project" in result.output and "does not exist" in result.output
    assert "star.cupw.json" in result.output and ".tmp" not in result.output
    assert "Traceback" not in result.output


def test_a_failed_embed_fallback_keeps_the_first_refusals_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Room for a project that references its data file, not for one embedding it.
    monkeypatch.setenv("CUPERIOD_SESSION_EMBED_DATA_MAX_POINTS", "10")
    monkeypatch.setenv("CUPERIOD_SESSION_MAX_PROJECT_BYTES", "15000")
    (tmp_path / "data").mkdir()
    (tmp_path / "project").mkdir()
    star = _write_star(tmp_path / "data" / "star.csv")
    project = tmp_path / "project" / "star.cupw.json"
    result = runner.invoke(
        app, ["prewhiten", str(star), *OPTIONS, "--save-project", str(project)]
    )
    assert result.exit_code == 1, result.output
    assert "max_project_bytes (15000)" in result.output
    assert "exceed embed_data_max_points (10)" in result.output
    assert "keep the data file next to the project" in result.output
    # The remedy works: saved beside the data, the file is referenced.
    beside = tmp_path / "data" / "star.cupw.json"
    result = runner.invoke(
        app, ["prewhiten", str(star), *OPTIONS, "--save-project", str(beside)]
    )
    assert result.exit_code == 0, result.output
    assert "note: embedded" not in result.output
    # A referenced project that is too large is not retried as an embedded one.
    monkeypatch.setenv("CUPERIOD_SESSION_MAX_PROJECT_BYTES", "2000")
    result = runner.invoke(
        app, ["prewhiten", str(star), *OPTIONS, "--save-project", str(beside)]
    )
    assert result.exit_code == 1, result.output
    assert "max_project_bytes (2000)" in result.output
    assert "embedded" not in result.output


def test_session_data_under_another_header_falls_back_to_detected_columns(
    tmp_path: Path,
) -> None:
    # A project made with explicit columns records them; the same points exported
    # under another (detectable) header were refused, with no way to override.
    star = _write_star(tmp_path / "star.csv")
    project = tmp_path / "star.cupw.json"
    created = runner.invoke(
        app,
        ["prewhiten", str(star), *OPTIONS, "--time", "time", "--value", "mag",
         "--error", "mag_err", "--save-project", str(project)],
    )
    assert created.exit_code == 0, created.output
    assert json.loads(project.read_text(encoding="utf-8"))["data"]["columns"]
    text = star.read_text(encoding="utf-8")
    moved = tmp_path / "moved" / "star_hjd.csv"
    moved.parent.mkdir()
    moved.write_text(text.replace("time,mag,mag_err", "hjd,vmag,e_vmag", 1),
                     encoding="utf-8")
    result = runner.invoke(
        app, ["session", "verify", str(project), "--data", str(moved)]
    )
    assert result.exit_code == 0, result.output
    assert "note: star_hjd.csv lacks the project's recorded columns" in result.output
    # The content hash still guards the points: other values are refused.
    lines = text.splitlines()
    first = lines[1].split(",")
    first[1] = repr(float(first[1]) + 0.5)
    lines[1] = ",".join(first)
    changed = tmp_path / "moved" / "changed.csv"
    changed.write_text(
        "\n".join(["hjd,vmag,e_vmag", *lines[1:]]) + "\n", encoding="utf-8"
    )
    result = runner.invoke(
        app, ["session", "verify", str(project), "--data", str(changed)]
    )
    assert result.exit_code == 1, result.output
    assert "Error:" in result.output
    # Neither the recorded nor any detectable columns: the recorded ones are named.
    odd = tmp_path / "moved" / "odd.csv"
    odd.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    result = runner.invoke(app, ["session", "verify", str(project), "--data", str(odd)])
    assert result.exit_code == 2, result.output
    assert "'time'" in result.output
