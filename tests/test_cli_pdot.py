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


def test_pdot_runs_the_light_time_search_by_default(tmp_path: Path) -> None:
    csv, truth = _write_drifting_csv(tmp_path / "drift.csv")
    out = tmp_path / "pdot.json"
    result = runner.invoke(
        app,
        ["pdot", str(csv), "--period", str(truth["period"]), "--backend", "cpu",
         *FAST_GRID, "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "light-time orbit search" in result.stdout
    assert "outer period" in result.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ltte"]["best"]["outer_period_yr"] > 0.0
    assert "delta_bic" in payload["ltte"]


def test_pdot_no_ltte_skips_the_light_time_section(tmp_path: Path) -> None:
    csv, truth = _write_drifting_csv(tmp_path / "drift.csv")
    result = runner.invoke(
        app,
        ["pdot", str(csv), "--period", str(truth["period"]), "--backend", "cpu",
         *FAST_GRID, "--no-ltte"],
    )
    assert result.exit_code == 0, result.output
    assert "phase tracking" in result.stdout
    assert "light-time orbit" not in result.stdout



# --- usage errors (BATCH-CLI-20) -------------------------------------------------


@pytest.mark.parametrize("bad", ["0", "-1", "nan"])
def test_pdot_rejects_a_nonpositive_period(tmp_path: Path, bad: str) -> None:
    csv, _ = _write_drifting_csv(tmp_path / "drift.csv")
    result = runner.invoke(
        app, ["pdot", str(csv), "--backend", "cpu", "--period", bad]
    )
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output


def test_pdot_batch_rejects_a_bad_period_and_device(tmp_path: Path) -> None:
    inputs, _ = _write_batch_inputs(tmp_path, n_stars=1)
    out = tmp_path / "summary.parquet"
    # Fewer than 4 tracking segments can never fit the quadratic ephemeris.
    for extra in (["--period", "0"], ["--device", "tpu"], ["--segments", "3"]):
        result = runner.invoke(
            app, ["pdot-batch", inputs, "--sink", str(out), "--backend", "cpu", *extra]
        )
        assert result.exit_code == 2, result.output
    assert not out.exists()


def test_pdot_missing_file_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["pdot", str(tmp_path / "missing.csv")])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_pdot_batch_where_everything_fails_exits_1(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lightcurves"
    lc_dir.mkdir()
    for i in range(2):
        _write_unusable_csv(lc_dir / f"tiny{i}.csv")
    out = tmp_path / "summary.parquet"
    result = runner.invoke(
        app,
        ["pdot-batch", str(lc_dir / "*.csv"), "--sink", str(out), "--period", "1.0",
         "--backend", "cpu", "--workers", "1", "--no-profile", *FAST_GRID],
    )
    assert result.exit_code == 1, result.output
    assert "2 failed" in result.stdout
    assert out.exists()  # the failures are still recorded, in the status column


def test_pdot_save_map_reports_the_file_it_wrote(tmp_path: Path) -> None:
    csv, truth = _write_drifting_csv(tmp_path / "drift.csv")
    target = tmp_path / "map.dat"
    result = runner.invoke(
        app,
        ["pdot", str(csv), "--period", str(truth["period"]), "--backend", "cpu",
         "--no-track", *FAST_GRID, "--save-map", str(target)],
    )
    assert result.exit_code == 0, result.output
    written = tmp_path / "map.dat.npz"
    assert written.exists() and f"Wrote {written}" in result.stdout


def test_pdot_batch_rerun_replaces_failure_rows(tmp_path: Path) -> None:
    # Regression (BATCH-CLI-8): a corrected re-run into the same summary file must
    # leave one row per star, not the old failure row beside the new "ok" one.
    import pyarrow.parquet as pq

    inputs, truth = _write_batch_inputs(tmp_path, n_stars=1)
    broken = _write_unusable_csv(tmp_path / "lightcurves" / "star1.csv")
    out = tmp_path / "summary.parquet"
    command = [
        "pdot-batch", inputs, "--sink", str(out), "--period", str(truth["period"]),
        "--backend", "cpu", "--workers", "1", "--no-profile", *FAST_GRID,
    ]
    first = runner.invoke(app, command)
    assert first.exit_code == 0 and "1 failed" in first.stdout, first.output
    _write_drifting_csv(broken)  # the bad file is fixed between the runs
    second = runner.invoke(app, command)
    assert second.exit_code == 0, second.output
    rows = pq.read_table(out).to_pylist()
    assert sorted(r["key"] for r in rows) == ["star0", "star1"]
    assert {r["status"] for r in rows} == {"ok"}


def test_period_law_table_prints_the_ranking_criterion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """BIC / dBIC columns are bic_scaled, the value the laws are sorted by.

    The track's errors are understated ten-fold, so bic_scaled and bic differ by
    about a factor of a hundred and the old plain-bic table would fail here.
    """
    import cuperiod as cup
    from cuperiod.cli.app import _print_period_profile

    time, mag, err, truth = synthetic_period_change(
        n=N_POINTS, span=SPAN, pdot=PDOT, seed=3
    )
    prof = cup.period_profile((time, mag, err), period=truth["period"])
    fits = cup.compare_period_laws(prof.t_mid, prof.period, 0.1 * prof.period_err)
    assert len(fits) >= 2
    assert not np.isclose(fits[1].bic_scaled, fits[1].bic, rtol=0.1)
    _print_period_profile(prof, fits)
    rows = re.findall(
        r"^\s{2}(constant|linear|exponential|singular|pejcha)\s+(\S+)\s+(\S+)\s",
        capsys.readouterr().out, flags=re.MULTILINE,
    )
    assert len(rows) == len(fits)
    best = fits[0].bic_scaled
    for (law, bic, dbic), fit in zip(rows, fits, strict=True):
        assert law == fit.law
        assert float(bic) == pytest.approx(fit.bic_scaled, rel=1e-3)
        assert float(dbic) == pytest.approx(fit.bic_scaled - best, rel=1e-2,
                                            abs=1e-3)
        if not fit.at_bound:
            assert float(dbic) >= 0.0


# --- release gate (1.3.0): usage errors, the environment, one error convention ---


@pytest.mark.parametrize(
    "extra", [["-m", "BLS"], ["-m", "TLS"], ["--pdot-range=-1e-2,1e-2"]]
)
def test_pdot_options_the_search_cannot_honour_are_usage_errors(
    tmp_path: Path, extra: list[str]
) -> None:
    # pdot_search raises ValueError for a box-search method and for a rate range
    # outside the period law's domain; `pdot` used to end in a traceback.
    csv, truth = _write_drifting_csv(tmp_path / "drift.csv")
    result = runner.invoke(
        app,
        ["pdot", str(csv), "--period", str(truth["period"]), "--backend", "cpu",
         *extra],
    )
    assert result.exit_code == 2, result.output
    assert not isinstance(result.exception, ValueError)
    assert "Traceback" not in result.output


def test_pdot_too_few_points_is_still_an_error_not_a_usage_error(
    tmp_path: Path,
) -> None:
    # InsufficientDataError is a ValueError too; it must keep its "Error:" / 1.
    csv = _write_unusable_csv(tmp_path / "tiny.csv")
    result = runner.invoke(
        app, ["pdot", str(csv), "--period", "1.0", "--backend", "cpu"]
    )
    assert result.exit_code == 1, result.output
    assert "Error:" in result.output


def test_pdot_honours_the_pdot_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An option left unset must not pass the CLI's default over the environment.
    monkeypatch.setenv("CUPERIOD_PDOT_MAX_CYCLE_DRIFT", "2")
    monkeypatch.setenv("CUPERIOD_PDOT_N_SEGMENTS", "12")
    csv, truth = _write_drifting_csv(tmp_path / "drift.csv")
    command = [
        "pdot", str(csv), "--period", str(truth["period"]), "--backend", "cpu",
        "--window-rayleigh", "10", "--samples-per-peak", "5", "--no-ltte",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    # 4 * samples_per_drift * max_cycle_drift + 1 rows: 41 for 2 cycles, not 201.
    assert re.search(r"grid\s+41 x \d+", result.stdout), result.stdout
    assert re.search(r"segments\s+12\b", result.stdout), result.stdout
    # An option given on the command line still beats the environment.
    result = runner.invoke(app, [*command, "--segments", "6"])
    assert result.exit_code == 0, result.output
    assert re.search(r"segments\s+6\b", result.stdout), result.stdout


def test_pdot_batch_honours_the_pdot_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow.parquet as pq

    monkeypatch.setenv("CUPERIOD_PDOT_N_SEGMENTS", "12")
    inputs, truth = _write_batch_inputs(tmp_path, n_stars=1)
    out = tmp_path / "summary.parquet"
    result = runner.invoke(
        app,
        ["pdot-batch", inputs, "--sink", str(out), "--period", str(truth["period"]),
         "--backend", "cpu", "--workers", "1", "--no-profile", "--no-ltte",
         *FAST_GRID],
    )
    assert result.exit_code == 0, result.output
    (row,) = pq.read_table(out).to_pylist()
    assert row["status"] == "ok"
    assert row["n_segments"] == 12


def test_pdot_options_help_names_the_environment() -> None:
    result = runner.invoke(app, ["pdot", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "CUPERIOD_PDOT_MAX_CYCLE_DRIFT, else 10" in result.stdout
    assert "CUPERIOD_PDOT_N_SEGMENTS, else auto" in result.stdout


def test_ltte_block_quotes_every_error_in_one_convention(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """P2, A and the secular dP/dt all carry the formal error x error_inflation."""
    import cuperiod as cup
    from cuperiod.cli.app import _SECONDS_PER_YEAR, _print_ltte

    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 5000.0, 40)) + 2450000.0
    dt = t - t.mean()
    oc = (
        0.5 * (3e-8 / 0.4) * dt**2
        + 0.003 * np.sin(2.0 * np.pi * dt / 1300.0 + 0.4)
        + rng.normal(0.0, 5e-4, t.size)
    )
    # Quoted errors three times too small: chi2_reduced ~ 9, inflation ~ 3.
    res = cup.ltte_search(t, oc=oc, errors=np.full(t.size, 5e-4 / 3), period=0.4)
    best = res.best
    scale = best.error_inflation
    assert scale > 2.0
    _print_ltte(res)
    out = capsys.readouterr().out
    assert f"+- {best.outer_period_err_yr * scale:.3f} yr" in out
    assert f"+- {best.amplitude_err_s * scale:.1f} s" in out
    assert f"+- {best.pdot_err_scaled * _SECONDS_PER_YEAR:.2g} s/yr" in out
    assert f"formal x {scale:.3g}" in out and "error_inflation" in out
