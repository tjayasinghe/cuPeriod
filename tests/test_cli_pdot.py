"""CLI tests for the ``pdot`` and ``pdot-batch`` commands.

The planted signal is `synthetic_period_change`'s exact constant-dP/dt model, so
the commands are checked against the same truth dict the library-level tests use;
here the concern is the CLI surface — exit codes, the printed table, and the
files ``--out`` / ``--save-map`` / the three ``pdot-batch`` sinks write.
"""

from __future__ import annotations

import json
import re
import sys
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

#: Search knobs that shrink the (dP/dt)/P x frequency grid about thirtyfold. The
#: planted drift still sits well inside the smaller grid, and these tests are
#: about the CLI plumbing rather than the last digit of the recovered rate.
FAST_GRID = [
    "--max-cycle-drift", "2",
    "--window-rayleigh", "10",
    "--samples-per-peak", "5",
]


def _write_drifting_csv(path: Path) -> tuple[Path, dict[str, float]]:
    time, mag, err, truth = synthetic_period_change(
        n=N_POINTS, span=SPAN, pdot=PDOT, seed=3
    )
    pd.DataFrame({"hjd": time, "mag": mag, "mag_err": err}).to_csv(path, index=False)
    return path, truth


def _write_batch_inputs(
    tmp_path: Path, n_stars: int = 2
) -> tuple[str, dict[str, float]]:
    """A glob over ``n_stars`` drifting light curves in their own directory.

    The light curves get a directory of their own so the sinks written into
    ``tmp_path`` can never be picked up as inputs by a later resolve.
    """
    lc_dir = tmp_path / "lightcurves"
    lc_dir.mkdir(exist_ok=True)
    truth: dict[str, float] = {}
    for index in range(n_stars):
        _, truth = _write_drifting_csv(lc_dir / f"star{index}.csv")
    return str(lc_dir / "*.csv"), truth


def _write_unusable_csv(path: Path) -> Path:
    """Far too few points for the search: a light curve that has to fail."""
    pd.DataFrame(
        {
            "hjd": [1.0, 2.0, 3.0, 4.0, 5.0],
            "mag": [13.0, 13.2, 13.0, 13.2, 13.0],
            "mag_err": [0.05] * 5,
        }
    ).to_csv(path, index=False)
    return path


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


# --- the new pdot options: --model, --detrend-width, --profile ----------------


def test_pdot_model_and_detrend_options(tmp_path: Path) -> None:
    csv, truth = _write_drifting_csv(tmp_path / "drift.csv")
    result = runner.invoke(
        app,
        ["pdot", str(csv), "--period", str(truth["period"]), "--backend", "cpu",
         "--model", "exponential", "--detrend-width", "60", "--no-track", *FAST_GRID],
    )
    assert result.exit_code == 0, result.output
    assert "detrended with a 60 d running median" in result.stdout
    # The two models agree to first order, so the exponential warp must still
    # land on the planted period rather than an alias of it.
    printed = re.search(r"period \(d\)\s+([0-9.]+)", result.stdout)
    assert printed is not None, result.output
    assert float(printed.group(1)) == pytest.approx(truth["period"], rel=1e-3)


def test_pdot_rejects_an_unknown_model(tmp_path: Path) -> None:
    csv, _ = _write_drifting_csv(tmp_path / "drift.csv")
    result = runner.invoke(
        app, ["pdot", str(csv), "--backend", "cpu", "--model", "wishful"]
    )
    assert result.exit_code == 2  # typer BadParameter (usage), not a ValueError


def test_pdot_rejects_a_nonpositive_detrend_width(tmp_path: Path) -> None:
    csv, _ = _write_drifting_csv(tmp_path / "drift.csv")
    result = runner.invoke(
        app, ["pdot", str(csv), "--backend", "cpu", "--detrend-width", "0"]
    )
    assert result.exit_code == 2


def test_pdot_profile_ranks_period_laws(tmp_path: Path) -> None:
    csv, truth = _write_drifting_csv(tmp_path / "drift.csv")
    out = tmp_path / "pdot.json"
    result = runner.invoke(
        app,
        ["pdot", str(csv), "--period", str(truth["period"]), "--backend", "cpu",
         "--profile", "--no-track", "--out", str(out), *FAST_GRID],
    )
    assert result.exit_code == 0, result.output
    assert "period profile" in result.stdout
    assert "period laws over P(t)" in result.stdout
    # A linearly drifting period: the constant law must not win the ranking.
    ranked = re.findall(r"^\s{2}(constant|linear|exponential|singular|pejcha)\s",
                        result.stdout, flags=re.MULTILINE)
    assert ranked, result.output
    assert ranked[0] != "constant"

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["profile"]["n_intervals"] >= 2
    assert len(payload["profile"]["points"]) == payload["profile"]["n_intervals"]
    assert {fit["law"] for fit in payload["laws"]} <= {
        "constant", "linear", "exponential", "singular"
    }


# --- pdot-batch ---------------------------------------------------------------


def test_pdot_batch_writes_a_summary_table(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    inputs, truth = _write_batch_inputs(tmp_path)
    out = tmp_path / "summary.parquet"
    result = runner.invoke(
        app,
        ["pdot-batch", inputs, "--sink", str(out), "--period", str(truth["period"]),
         "--backend", "cpu", "--workers", "1", *FAST_GRID],
    )
    assert result.exit_code == 0, result.output
    assert "Measured 2 light curves from 2 inputs" in result.stdout

    rows = pq.read_table(out).to_pylist()
    assert {row["key"] for row in rows} == {"star0", "star1"}
    assert all(row["status"] == "ok" for row in rows)
    for row in rows:
        assert row["search_period"] == pytest.approx(truth["period"], rel=1e-3)
        assert abs(row["track_pdot"] - truth["pdot"]) < 0.5 * truth["pdot"]
        # The P(t) track and its law ranking run by default.
        assert row["n_intervals"] >= 2
        assert row["law"] in {"constant", "linear", "exponential", "singular"}


def test_pdot_batch_writes_all_three_tables(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    inputs, truth = _write_batch_inputs(tmp_path, n_stars=1)
    summary = tmp_path / "summary.parquet"
    oc = tmp_path / "oc.parquet"
    profile = tmp_path / "profile.parquet"
    result = runner.invoke(
        app,
        ["pdot-batch", inputs, "--sink", str(summary), "--oc-sink", str(oc),
         "--profile-sink", str(profile), "--period", str(truth["period"]),
         "--backend", "cpu", "--workers", "1", *FAST_GRID],
    )
    assert result.exit_code == 0, result.output
    assert str(oc) in result.stdout and str(profile) in result.stdout

    summary_rows = pq.read_table(summary).to_pylist()
    oc_rows = pq.read_table(oc).to_pylist()
    profile_rows = pq.read_table(profile).to_pylist()
    assert len(summary_rows) == 1
    assert {"key", "t_mid", "delay", "delay_err"} <= set(oc_rows[0])
    assert {"key", "t_mid", "period", "period_err"} <= set(profile_rows[0])
    # The side tables are the rows behind the summary's counts, star by star.
    assert len(oc_rows) == summary_rows[0]["n_segments"]
    assert len(profile_rows) == summary_rows[0]["n_intervals"]
    assert {row["key"] for row in oc_rows + profile_rows} == {"star0"}


def test_pdot_batch_no_profile_leaves_the_track_columns_empty(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    inputs, truth = _write_batch_inputs(tmp_path, n_stars=1)
    out = tmp_path / "summary.parquet"
    result = runner.invoke(
        app,
        ["pdot-batch", inputs, "--sink", str(out), "--no-profile",
         "--period", str(truth["period"]), "--backend", "cpu", "--workers", "1",
         *FAST_GRID],
    )
    assert result.exit_code == 0, result.output
    row = pq.read_table(out).to_pylist()[0]
    assert row["status"] == "ok"
    assert row["n_intervals"] == 0
    assert row["law"] == ""
    assert np.isnan(row["profile_total_fractional_change"])
    # Only the profile is skipped — the search and the O-C measurement still ran.
    assert np.isfinite(row["track_pdot"])


def test_pdot_batch_records_a_failing_light_curve(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    inputs, truth = _write_batch_inputs(tmp_path, n_stars=1)
    _write_unusable_csv(tmp_path / "lightcurves" / "tiny.csv")
    out = tmp_path / "summary.parquet"
    result = runner.invoke(
        app,
        ["pdot-batch", inputs, "--sink", str(out), "--no-profile",
         "--period", str(truth["period"]), "--backend", "cpu", "--workers", "1",
         *FAST_GRID],
    )
    # One pathological star must not take the run down with it.
    assert result.exit_code == 0, result.output
    assert "1 failed" in result.stdout
    assert "! tiny:" in result.stdout

    by_key = {row["key"]: row for row in pq.read_table(out).to_pylist()}
    assert set(by_key) == {"star0", "tiny"}
    assert by_key["star0"]["status"] == "ok"
    assert by_key["tiny"]["status"] != "ok"
    assert "InsufficientDataError" in by_key["tiny"]["status"]
    assert np.isnan(by_key["tiny"]["search_period"])


def test_pdot_batch_rejects_an_unknown_law(tmp_path: Path) -> None:
    inputs, _ = _write_batch_inputs(tmp_path, n_stars=1)
    result = runner.invoke(
        app,
        ["pdot-batch", inputs, "--sink", str(tmp_path / "summary.parquet"),
         "--laws", "linear,parabolic", "--backend", "cpu", "--workers", "1"],
    )
    assert result.exit_code == 2  # usage error, not a silently unfitted catalogue
    assert not (tmp_path / "summary.parquet").exists()


def test_pdot_batch_plots_without_matplotlib_fails_up_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # batch_pdot swallows per-star plotting failures by design, so a missing
    # matplotlib would otherwise show up only as empty `figure` cells at the end
    # of a survey-scale run. The CLI has to refuse before doing any work.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    inputs, _ = _write_batch_inputs(tmp_path, n_stars=1)
    out = tmp_path / "summary.parquet"
    result = runner.invoke(
        app,
        ["pdot-batch", inputs, "--sink", str(out), "--plots", str(tmp_path / "figs"),
         "--backend", "cpu", "--workers", "1", *FAST_GRID],
    )
    assert result.exit_code == 1
    assert 'pip install "cuperiod[plot]"' in result.output
    assert not out.exists()
