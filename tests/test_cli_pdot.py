"""CLI tests for the ``pdot`` command (joint search + phase tracking).

The planted signal is `synthetic_period_change`'s exact constant-dP/dt model, so
the command is checked against the same truth dict the library-level tests use;
here the concern is the CLI surface — exit codes, the printed table, and the
files ``--out`` / ``--save-map`` write.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from cuperiod.cli.app import app
from synth import synthetic_period_change

runner = CliRunner()

#: A drift big enough that both the search grid and the O-C parabola see it
#: (~1.5 cycles accumulated over the span), on a light curve small enough to
#: keep the CLI round trip quick.
PDOT = 5.0e-7
N_POINTS = 1200
SPAN = 1500.0


def _write_drifting_csv(path: Path) -> tuple[Path, dict[str, float]]:
    time, mag, err, truth = synthetic_period_change(
        n=N_POINTS, span=SPAN, pdot=PDOT, seed=3
    )
    pd.DataFrame({"hjd": time, "mag": mag, "mag_err": err}).to_csv(path, index=False)
    return path, truth


def test_pdot_command_reports_the_drift(tmp_path: Path) -> None:
    csv, truth = _write_drifting_csv(tmp_path / "drift.csv")
    out = tmp_path / "pdot.json"
    result = runner.invoke(
        app,
        ["pdot", str(csv), "--period", str(truth["period"]),
         "--backend", "cpu", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "s/yr" in result.stdout
    # The printed period must be the recovered one, not an echo of the input.
    printed = re.search(r"period \(d\)\s+([0-9.]+)", result.stdout)
    assert printed is not None, result.output
    assert float(printed.group(1)) == pytest.approx(truth["period"], rel=1e-3)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["method"] == "GLS"
    assert abs(payload["best"]["pdot"] - truth["pdot"]) < 0.5 * truth["pdot"]
    # Tracking ran by default and is serialized alongside the search.
    assert abs(payload["track"]["pdot"] - truth["pdot"]) < 0.5 * truth["pdot"]


def test_pdot_no_track_skips_the_tracking_section(tmp_path: Path) -> None:
    csv, truth = _write_drifting_csv(tmp_path / "drift.csv")
    result = runner.invoke(
        app,
        ["pdot", str(csv), "--period", str(truth["period"]),
         "--backend", "cpu", "--no-track"],
    )
    assert result.exit_code == 0, result.output
    assert "dP/dt search" in result.stdout
    assert "phase tracking" not in result.stdout
    assert "F-test" not in result.stdout


def test_pdot_rejects_an_unparseable_range(tmp_path: Path) -> None:
    csv, _ = _write_drifting_csv(tmp_path / "drift.csv")
    result = runner.invoke(
        app, ["pdot", str(csv), "--backend", "cpu", "--pdot-range", "garbage"]
    )
    assert result.exit_code == 2  # typer BadParameter (usage), not a crash


def test_pdot_saves_the_map(tmp_path: Path) -> None:
    csv, truth = _write_drifting_csv(tmp_path / "drift.csv")
    npz = tmp_path / "map.npz"
    result = runner.invoke(
        app,
        ["pdot", str(csv), "--period", str(truth["period"]), "--backend", "cpu",
         "--no-track", "--save-map", str(npz)],
    )
    assert result.exit_code == 0, result.output
    with np.load(npz) as data:
        assert {"frequency", "pdot_over_p", "power"} <= set(data)
        assert data["power"].shape == (data["pdot_over_p"].size, data["frequency"].size)


def test_pdot_unknown_method_fails(tmp_path: Path) -> None:
    csv, truth = _write_drifting_csv(tmp_path / "drift.csv")
    result = runner.invoke(
        app,
        ["pdot", str(csv), "--period", str(truth["period"]),
         "--backend", "cpu", "--method", "NOPE"],
    )
    assert result.exit_code != 0
