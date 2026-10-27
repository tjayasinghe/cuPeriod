"""CLI, 1.3.0 release gate round 2: Windows argument expansion and batch reruns.

* The console script expands ``~``, ``%VAR%`` and wildcards on Windows as click
  did in 1.2.0, but leaves a batch command's ``INPUTS`` pattern whole (CLI-1). The
  Windows branch is selected through ``_on_windows`` and the expansion uses
  ``ntpath``, so these tests run on any OS.
* A batch command never reads its own sink, errors list, ``.corrupt-*`` copies or
  temporary files as light curves, so rerunning it into a sink that its own glob
  matches works (REL-2).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from cuperiod.batch.io import _resolve_input_files
from synth import synthetic_period_change, synthetic_pulsator, synthetic_sine

runner = CliRunner()


def _cli() -> ModuleType:
    # The module, not cuperiod.cli.app (the package re-exports the Typer app there).
    return importlib.import_module("cuperiod.cli.app")


def _write_csv(path: Path, seed: int = 0) -> Path:
    t, mag, err = synthetic_sine(n=150, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"jd": t, "mag": mag, "mag_err": err}).to_csv(path, index=False)
    return path


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch user folder: ``~`` (USERPROFILE, as Windows reads it) and %MYLC%."""
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("USERPROFILE", str(folder))
    monkeypatch.setenv("MYLC", str(folder))
    return folder


def _same(a: str, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(str(b))


# --- CLI-1: the expansion itself -----------------------------------------------------


def test_windows_expansion_resolves_tilde_variables_and_a_single_match(
    home: Path,
) -> None:
    cli = _cli()
    lc = _write_csv(home / "lc.csv")
    for word in ("~/lc.csv", "%MYLC%/lc.csv", "$MYLC/lc.csv", str(home / "l*.csv")):
        command, path = cli._expand_windows_args(["run", word])
        assert command == "run" and _same(path, lc), word


def test_windows_expansion_reaches_option_values(home: Path) -> None:
    cli = _cli()
    expanded = cli._expand_windows_args(
        ["batch", "x.csv", "--out", "~/r.parquet", "--sink=~/s.parquet", "-o~/t"]
    )
    assert _same(expanded[3], home / "r.parquet")
    name, _, value = expanded[4].partition("=")
    assert name == "--sink" and _same(value, home / "s.parquet")
    assert expanded[5] == "-o~/t"  # as click: only a leading ~ is a home folder


def test_windows_expansion_keeps_a_batch_glob_whole(tmp_path: Path) -> None:
    cli = _cli()
    for i in range(2):
        _write_csv(tmp_path / f"s{i}.csv", seed=i)
    pattern = str(tmp_path / "*.csv")
    for command in ("batch", "batch-prewhiten", "pdot-batch"):
        argv = [command, "--workers", "1", pattern, "--out", "o.parquet"]
        assert cli._batch_inputs_index(argv) == 3
        assert cli._expand_windows_args(argv, keep_whole=3) == argv
    # Any other command's argument is globbed, one argument per match, as by click.
    expanded = cli._expand_windows_args(["run", pattern])
    assert sorted(Path(p).name for p in expanded[1:]) == ["s0.csv", "s1.csv"]


@pytest.mark.parametrize(
    ("argv", "index"),
    [
        (["batch", "in", "--out", "o"], 1),
        (["batch", "--out", "o", "-m", "GLS", "in"], 5),
        (["batch", "--store-raw", "in", "--out", "o"], 2),
        (["batch", "--no-resume", "--out=o", "-mGLS", "in"], 4),
        (["batch", "--out", "o", "--", "in"], 4),
        (["batch-prewhiten", "-n", "3", "--combinations", "in"], 4),
        (["pdot-batch", "--sink", "s", "--no-ltte", "--profile", "in"], 5),
        (["pdot-batch", "--out", "s", "--plots", "p", "in"], 5),
        (["--help", "batch", "in"], 2),
        (["batch", "--help"], None),
        (["run", "in"], None),
        (["session", "verify", "p.cupw.json"], None),
        ([], None),
    ],
)
def test_the_batch_inputs_argument_is_found(argv: list[str], index: int | None) -> None:
    assert _cli()._batch_inputs_index(argv) == index


# --- CLI-1: through the console script -----------------------------------------------


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str, windows: bool = True) -> int:
    cli = _cli()
    monkeypatch.setattr(cli, "_on_windows", lambda: windows)
    monkeypatch.setattr("sys.argv", ["cuperiod", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    code = exit_info.value.code
    return 0 if code is None else int(code)


def test_console_script_expands_tilde_on_windows(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_csv(home / "lc.csv")
    code = _main(
        monkeypatch, "run", "~/lc.csv", "-m", "GLS", "--backend", "cpu",
        "--n-best", "1", "--out", "~/peaks.json",
    )
    assert code == 0
    assert (home / "peaks.json").is_file()


def test_console_script_expands_variables_and_a_single_match_on_windows(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_csv(home / "lc.csv")
    for word in ("%MYLC%/lc.csv", str(home / "l*.csv")):
        assert _main(monkeypatch, "grid-info", word) == 0, word


def test_console_script_batch_keeps_a_quoted_glob_and_expands_its_out(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(2):
        _write_csv(tmp_path / "lcs" / f"s{i}.csv", seed=i)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    code = _main(
        monkeypatch, "batch", str(tmp_path / "lcs" / "*.csv"), "-m", "GLS",
        "--backend", "cpu", "--workers", "0", "--out=~/results.parquet",
    )
    assert code == 0
    rows = pq.read_table(home / "results.parquet").to_pylist()
    assert sorted(row["key"] for row in rows) == ["s0", "s1"]
    assert not (work / "~").exists()  # 1.3.0's main() wrote into a literal ~ folder


def test_console_script_leaves_posix_arguments_to_typer(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _cli()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_app(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(cli, "app", fake_app)
    monkeypatch.setattr("sys.argv", ["cuperiod", "run", "~/lc.csv"])
    monkeypatch.setattr(cli, "_on_windows", lambda: False)
    cli.main()
    monkeypatch.setattr(cli, "_on_windows", lambda: True)
    cli.main()
    assert calls[0] == ((), {})  # typer's own default, untouched
    args, kwargs = calls[1]
    assert args == () and kwargs["windows_expand_args"] is False
    assert kwargs["args"][0] == "run" and not kwargs["args"][1].startswith("~")


# --- REL-2: a batch's own outputs are not inputs -------------------------------------


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n", encoding="utf-8")
    return path


def test_a_file_sink_and_its_side_files_are_not_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lcs = [_touch(tmp_path / name) for name in ("a.csv", "b.csv", "results_old.csv")]
    sink = tmp_path / "results.csv"
    errors = tmp_path / "results.errors.csv"
    for name in (
        "results.csv",
        "results.errors.csv",
        "results.csv.corrupt-20261027T101010",
        "results.csv.corrupt-20261027T101010-1",
        ".results.csv.123.abcd1234.tmp",
        ".results.errors.csv.123.abcd1234.tmp",
    ):
        _touch(tmp_path / name)
    outputs = [sink, errors]
    assert _resolve_input_files(str(tmp_path / "*"), outputs=outputs) == lcs
    assert _resolve_input_files(str(tmp_path / ".*"), outputs=outputs) == []
    assert _resolve_input_files(tmp_path, outputs=outputs) == lcs
    # Without outputs nothing is dropped; a relative sink names the same file.
    assert len(_resolve_input_files(str(tmp_path / "*"))) == 7
    monkeypatch.chdir(tmp_path)
    found = _resolve_input_files(str(tmp_path / "*"), outputs=[Path("results.csv")])
    assert sink not in found and len(found) == 4


def test_a_directory_sink_drops_only_what_it_writes(tmp_path: Path) -> None:
    lc = _touch(tmp_path / "a.csv")
    sink = tmp_path / "results"
    kept = _touch(sink / "mine.csv")  # the user's own file, not the sink's
    for name in (
        "part-00000.parquet",
        ".part-00001.parquet.corrupt-20261027T101010",
        "_manifest.json",
        "_errors.csv",
        "._errors.csv.1.ab.tmp",
    ):
        _touch(sink / name)
    pattern = str(tmp_path / "**" / "*.*")
    found = _resolve_input_files(pattern, outputs=[sink, sink / "_errors.csv"])
    assert sorted(found) == sorted([lc, kept])
    # A directory sink that is the input folder itself keeps the light curves.
    _touch(tmp_path / "part-00000.parquet")
    assert _resolve_input_files(tmp_path, outputs=[tmp_path]) == [lc]


@pytest.mark.skipif(os.name != "nt", reason="Windows paths ignore letter case")
def test_a_sink_named_in_another_case_is_still_the_sink(tmp_path: Path) -> None:
    lc = _touch(tmp_path / "a.csv")
    _touch(tmp_path / "results.csv")
    found = _resolve_input_files(
        str(tmp_path / "*.csv"), outputs=[tmp_path / "RESULTS.CSV"]
    )
    assert found == [lc]


def test_batch_rerun_into_a_sink_inside_its_glob(tmp_path: Path) -> None:
    from cuperiod.cli.app import app

    for i in range(3):
        _write_csv(tmp_path / f"lc_{i}.csv", seed=i)
    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).to_csv(
        tmp_path / "bad.csv", index=False
    )
    out = tmp_path / "results.csv"
    command = ["batch", str(tmp_path / "*.csv"), "--out", str(out), "-m", "GLS",
               "--backend", "cpu", "--workers", "0"]
    first = runner.invoke(app, command)
    assert first.exit_code == 0, first.output
    assert "Processed 3 results from 4 inputs (0 skipped, 1 failed" in first.stdout
    assert (tmp_path / "results.errors.csv").exists()
    # The rerun sees the same four inputs: the results and the errors list are its
    # own outputs. Only bad.csv runs (and fails) again, so it exits 1 as before.
    second = runner.invoke(app, command)
    assert "from 4 inputs (3 skipped, 1 failed" in second.stdout, second.output
    assert "! results" not in second.stdout
    listed = pd.read_csv(tmp_path / "results.errors.csv")
    assert list(listed["key"]) == ["bad"]
    # With the bad file gone, a rerun of finished work is a success.
    (tmp_path / "bad.csv").unlink()
    third = runner.invoke(app, command)
    assert third.exit_code == 0, third.output
    assert "from 3 inputs (3 skipped, 0 failed" in third.stdout
    assert not (tmp_path / "results.errors.csv").exists()


def test_batch_prewhiten_rerun_into_a_sink_inside_its_glob(tmp_path: Path) -> None:
    from cuperiod.cli.app import app

    for i in range(2):
        time, mag, err = synthetic_pulsator(n=600, span=20.0, seed=i)
        pd.DataFrame({"hjd": time, "mag": mag, "mag_err": err}).to_csv(
            tmp_path / f"p{i}.csv", index=False
        )
    out = tmp_path / "modes.csv"
    command = ["batch-prewhiten", str(tmp_path / "*.csv"), "--out", str(out),
               "--backend", "finufft", "-n", "2", "--workers", "1"]
    first = runner.invoke(app, command)
    assert first.exit_code == 0, first.output
    assert "from 2 inputs (0 skipped, 0 failed)" in first.stdout
    second = runner.invoke(app, command)
    assert second.exit_code == 0, second.output
    assert "from 2 inputs (2 skipped, 0 failed)" in second.stdout


def test_pdot_batch_rerun_into_a_sink_inside_its_glob(tmp_path: Path) -> None:
    from cuperiod.cli.app import app

    time, mag, err, truth = synthetic_period_change(
        n=1200, span=1500.0, pdot=5.0e-7, seed=3
    )
    pd.DataFrame({"hjd": time, "mag": mag, "mag_err": err}).to_csv(
        tmp_path / "star0.csv", index=False
    )
    out = tmp_path / "summary.csv"
    command = [
        "pdot-batch", str(tmp_path / "*.csv"), "--sink", str(out),
        "--oc-sink", str(tmp_path / "oc.csv"), "--period", str(truth["period"]),
        "--backend", "cpu", "--workers", "1", "--no-profile", "--no-ltte",
        "--max-cycle-drift", "2", "--window-rayleigh", "10",
        "--samples-per-peak", "5",
    ]
    first = runner.invoke(app, command)
    assert first.exit_code == 0, first.output
    assert "Measured 1 light curves from 1 inputs" in first.stdout
    second = runner.invoke(app, command)
    assert second.exit_code == 0, second.output
    assert "from 1 inputs (1 skipped, 0 failed)" in second.stdout
