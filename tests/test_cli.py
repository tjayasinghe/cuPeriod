"""CLI smoke tests via Typer's CliRunner."""

from __future__ import annotations

import json
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


def test_run_out_is_valid_json(tmp_path: Path) -> None:
    # Regression (M1): --out must be standard JSON, never bare NaN/Infinity tokens.
    csv = _write_csv(tmp_path / "star.csv")
    out_json = tmp_path / "out.json"
    runner.invoke(app, ["run", str(csv), "-m", "GLS", "--out", str(out_json)])
    text = out_json.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    json.loads(text)  # must parse as standard JSON


def test_run_bad_domain_is_usage_error(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "star.csv")
    result = runner.invoke(app, ["run", str(csv), "-m", "GLS", "--domain", "lux"])
    assert result.exit_code == 2  # typer BadParameter (usage), not a ValueError crash


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


def test_doctor_command() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "backends installed" in result.stdout
    assert "backend='auto' resolves to" in result.stdout


def test_grid_info_command(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "star.csv")
    result = runner.invoke(app, ["grid-info", str(csv), "--method", "GLS"])
    assert result.exit_code == 0
    assert "samples" in result.stdout


def _write_pulsator_csv(path: Path) -> Path:
    from synth import synthetic_pulsator

    time, mag, err = synthetic_pulsator(n=600, span=20.0)
    pd.DataFrame({"hjd": time, "mag": mag, "mag_err": err}).to_csv(path, index=False)
    return path


def test_prewhiten_command(tmp_path: Path) -> None:
    csv = _write_pulsator_csv(tmp_path / "pulsator.csv")
    out = tmp_path / "solution.json"
    table = tmp_path / "components.csv"
    result = runner.invoke(
        app,
        [
            "prewhiten", str(csv), "--backend", "finufft", "-n", "4",
            "--out", str(out), "--csv", str(table),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Pre-whitening" in result.stdout and "F1" in result.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_components"] >= 1
    assert payload["components"][0]["label"] == "F1"
    header = table.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("rank,label,frequency")


def test_prewhiten_command_saves_spectra(tmp_path: Path) -> None:
    import numpy as np

    csv = _write_pulsator_csv(tmp_path / "pulsator.csv")
    npz = tmp_path / "spectra.npz"
    result = runner.invoke(
        app,
        ["prewhiten", str(csv), "--backend", "finufft", "-n", "2",
         "--save-spectrum", str(npz)],
    )
    assert result.exit_code == 0, result.output
    with np.load(npz) as data:
        assert {
            "frequency", "amplitude", "residual_amplitude", "window_amplitude"
        } <= set(data)
        assert np.all(data["window_amplitude"] <= 1.0 + 1e-12)


def test_prewhiten_command_reports_a_period_spacing(tmp_path: Path) -> None:
    from synth import synthetic_gmode

    time, mag, err, _ = synthetic_gmode(n=1500, n_modes=12)
    csv = tmp_path / "gdor.csv"
    pd.DataFrame({"hjd": time, "mag": mag, "mag_err": err}).to_csv(csv, index=False)
    result = runner.invoke(
        app,
        ["prewhiten", str(csv), "--backend", "finufft", "-n", "14", "--spacing"],
    )
    assert result.exit_code == 0, result.output
    assert "Period spacing" in result.stdout


def test_prewhiten_command_rejects_a_bad_criterion(tmp_path: Path) -> None:
    csv = _write_pulsator_csv(tmp_path / "pulsator.csv")
    result = runner.invoke(
        app, ["prewhiten", str(csv), "--stop", "nonsense", "--backend", "finufft"]
    )
    assert result.exit_code != 0


def test_batch_prewhiten_command(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    for index in range(2):
        _write_pulsator_csv(tmp_path / f"p{index}.csv")
    out = tmp_path / "modes.parquet"
    result = runner.invoke(
        app,
        ["batch-prewhiten", str(tmp_path / "*.csv"), "--out", str(out),
         "--backend", "finufft", "-n", "3", "--workers", "1"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    table = pq.read_table(out).to_pylist()
    assert {row["key"] for row in table}
    assert "frequency" in table[0]
