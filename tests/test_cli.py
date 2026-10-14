"""CLI smoke tests via Typer's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
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


# --- the run printout and its JSON agree (BATCH-CLI-18) ---------------------------


def _write_transit_csv(path: Path) -> Path:
    from synth import synthetic_eclipser

    t, flux, err = synthetic_eclipser(n=600, span=60.0)
    pd.DataFrame({"jd": t, "flux": flux, "flux_err": err}).to_csv(path, index=False)
    return path


def _printed_periods(stdout: str, method: str) -> list[float]:
    block = stdout.split(f"\n{method}  (", 1)[1].split("\n\n", 1)[0]
    rows = [line.split() for line in block.splitlines()[2:] if line.strip()]
    return [float(row[1]) for row in rows]


def test_run_json_lists_the_printed_bls_peaks(tmp_path: Path) -> None:
    import cuperiod as cup

    csv = _write_transit_csv(tmp_path / "transit.csv")
    out = tmp_path / "bls.json"
    result = runner.invoke(
        app,
        ["run", str(csv), "-m", "BLS,GLS", "--backend", "cpu", "--domain", "flux",
         "--n-best", "5", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    for method in ("BLS", "GLS"):
        printed = _printed_periods(result.stdout, method)
        saved = [peak["period"] for peak in payload[method]["peaks"]]
        assert saved == pytest.approx(printed, rel=1e-7), method
    batch = cup.batch_periodograms(
        str(csv), "BLS", backend="cpu", workers=0, domain="flux", n_best=5
    )
    assert batch.rows is not None
    row = batch.rows[0]
    saved_bls = [peak["period"] for peak in payload["BLS"]["peaks"]]
    assert [row[f"period_{k}"] for k in range(1, 6)] == pytest.approx(saved_bls)


# --- user errors are usage errors, not tracebacks (BATCH-CLI-20) -------------------


@pytest.mark.parametrize(
    "args",
    [
        ["run", "missing.csv"],
        ["run", "{csv}", "-m", "NOPE"],
        ["run", "{csv}", "--backend", "bogus"],
        ["run", "{csv}", "-m", ","],
        ["run", "{csv}", "--n-best", "0"],
        ["grid-info", "missing.csv"],
        ["batch", "{dir}/*.csv", "--out", "{dir}/r.parquet", "--device", "tpu"],
        ["batch", "{dir}/*.csv", "--out", "{dir}/r.json"],
        ["batch", "{dir}/*.csv", "--out", "{dir}/r.parquet", "--backend", "bogus"],
        ["batch", "{dir}/*.csv", "--out", "{dir}/r.parquet", "-m", "NOPE"],
        ["batch", "{dir}/*.parqet", "--out", "{dir}/r.parquet"],
        ["batch", "{dir}/*.csv", "--out", "{dir}/r.parquet", "--device", "cpu",
         "--backend", "cufinufft"],
    ],
    ids=[
        "missing-file", "unknown-method", "unknown-backend", "no-method", "n-best-0",
        "grid-info-missing", "bad-device", "bad-sink", "batch-unknown-backend",
        "batch-unknown-method", "empty-glob", "gpu-backend-on-cpu",
    ],
)
def test_user_errors_exit_2_without_a_traceback(
    tmp_path: Path, args: list[str]
) -> None:
    csv = _write_csv(tmp_path / "star.csv")
    filled = [a.format(csv=csv, dir=tmp_path) for a in args]
    result = runner.invoke(app, filled)
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "r.parquet").exists()


def test_batch_where_everything_fails_exits_1_and_lists_every_error(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    for i in range(13):
        pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).to_csv(
            bad / f"x{i:02d}.csv", index=False
        )
    out = tmp_path / "out.parquet"
    result = runner.invoke(
        app,
        ["batch", str(bad), "--out", str(out), "--backend", "cpu", "--workers", "0"],
    )
    assert result.exit_code == 1, result.output
    assert "(0 skipped, 13 failed" in result.stdout
    assert result.stdout.count("  ! ") == 10
    assert "... and 3 more" in result.stdout
    listed = pd.read_csv(tmp_path / "out.errors.csv")
    assert len(listed) == 13 and {"key", "error"} <= set(listed.columns)


def test_batch_with_no_inputs_keeps_the_previous_errors_list(tmp_path: Path) -> None:
    # Zero inputs leave the sink untouched, so the list of its last run's failures,
    # which describes the sink, must survive too.
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out.parquet"
    listed = tmp_path / "out.errors.csv"
    listed.write_text("key,error\nstar0,ValueError: earlier\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["batch", str(empty), "--out", str(out), "--backend", "cpu", "--workers", "0"],
    )
    assert result.exit_code == 0, result.output
    assert "no inputs" in result.output
    assert listed.exists() and not out.exists()


def test_batch_accepts_a_chunk_size(tmp_path: Path) -> None:
    for i in range(3):
        _write_csv(tmp_path / f"s{i}.csv")
    out = tmp_path / "parts"
    result = runner.invoke(
        app,
        ["batch", str(tmp_path / "*.csv"), "--out", str(out), "--backend", "cpu",
         "--workers", "0", "--chunk-size", "1"],
    )
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("part-*.parquet"))) == 3
    assert not (out / "_errors.csv").exists()


def test_batch_prewhiten_accepts_a_chunk_size(tmp_path: Path) -> None:
    for index in range(2):
        _write_pulsator_csv(tmp_path / f"p{index}.csv")
    out = tmp_path / "parts"
    result = runner.invoke(
        app,
        ["batch-prewhiten", str(tmp_path / "*.csv"), "--out", str(out),
         "--backend", "finufft", "-n", "2", "--workers", "1", "--chunk-size", "1"],
    )
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("part-*.parquet"))) == 2


@pytest.mark.parametrize("command", ["prewhiten", "batch-prewhiten"])
def test_prewhiten_commands_leave_the_backend_to_the_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    # Without --backend the CLI must not pin settings.backend to "auto": that hid
    # CUPERIOD_PREWHITEN_BACKEND, which cup.prewhiten and batch_prewhiten honour.
    import pyarrow.parquet as pq

    monkeypatch.setenv("CUPERIOD_PREWHITEN_BACKEND", "numpy")
    csv = _write_pulsator_csv(tmp_path / "p0.csv")
    out = tmp_path / "modes.parquet"
    if command == "prewhiten":
        args = ["prewhiten", str(csv), "-n", "2"]
    else:
        args = ["batch-prewhiten", str(csv), "--out", str(out), "-n", "2",
                "--workers", "1"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    if command == "prewhiten":
        assert "backend=numpy" in result.stdout
    else:
        assert {row["backend"] for row in pq.read_table(out).to_pylist()} == {"numpy"}


def test_save_periodogram_reports_the_file_it_wrote(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "star.csv")
    target = tmp_path / "spec.dat"
    result = runner.invoke(
        app, ["run", str(csv), "-m", "GLS", "--save-periodogram", str(target)]
    )
    assert result.exit_code == 0, result.output
    written = tmp_path / "spec.dat.npz"
    assert written.exists() and f"Wrote {written}" in result.stdout
