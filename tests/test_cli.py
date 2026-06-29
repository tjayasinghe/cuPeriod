"""CLI smoke tests via Typer's CliRunner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from cuperiod.cli.app import app
from synth import synthetic_sine

runner = CliRunner()


def _write_csv(path: Path) -> Path:
    t, mag, err = synthetic_sine(n=150)
    pd.DataFrame({"jd": t, "mag": mag, "mag_err": err}).to_csv(path, index=False)
    return path


def test_methods_command() -> None:
    result = runner.invoke(app, ["methods"])
    assert result.exit_code == 0
    assert "GLS" in result.stdout
    assert "BLS" in result.stdout


def test_run_command(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "star.csv")
    result = runner.invoke(app, ["run", str(csv), "--method", "GLS", "--n-best", "5"])
    assert result.exit_code == 0
    assert "period" in result.stdout
    assert "GLS" in result.stdout


def test_run_writes_outputs(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "star.csv")
    out_json = tmp_path / "out.json"
    out_npz = tmp_path / "spec.npz"
    result = runner.invoke(
        app,
        ["run", str(csv), "-m", "GLS", "--out", str(out_json),
         "--save-periodogram", str(out_npz)],
    )
    assert result.exit_code == 0
    assert out_json.exists()
    assert out_npz.exists()


def test_batch_command(tmp_path: Path) -> None:
    for i in range(2):
        _write_csv(tmp_path / f"s{i}.csv")
    out = tmp_path / "results.parquet"
    result = runner.invoke(
        app,
        ["batch", str(tmp_path / "*.csv"), "-m", "GLS", "--out", str(out)],
    )
    assert result.exit_code == 0
    assert out.exists()


def test_gpu_info_command() -> None:
    result = runner.invoke(app, ["gpu-info"])
    assert result.exit_code == 0


def test_grid_info_command(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "star.csv")
    result = runner.invoke(app, ["grid-info", str(csv), "--method", "GLS"])
    assert result.exit_code == 0
    assert "samples" in result.stdout
