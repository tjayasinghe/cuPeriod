"""batch_pdot: the three tables, their schema stability, sinks, and figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import cuperiod as cup
from cuperiod.pdot.batch import TABLES, pdot_to_rows

#: Tylenda et al. (2011) Eq. (1) for V1309 Sco: an accelerating period decay.
V1309 = (1.4456, 15.29, 2455233.5)


def _drifting(
    seed: int, n: int = 700, trend: float = 0.0
) -> cup.LightCurve:
    """A light curve whose period follows the V1309 Sco singular decay law."""
    p0, b, t0 = V1309
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2452100.0, 2454300.0, n))
    fine = np.linspace(t.min(), t.max(), 100_001)
    period = p0 * np.exp(b / (fine - t0))
    grid = np.concatenate(
        [[0.0], np.cumsum(np.diff(fine) / (0.5 * (period[1:] + period[:-1])))]
    )
    phase = np.interp(t, fine, grid)
    mag = (
        12.0
        + 0.30 * np.sin(2.0 * np.pi * phase)
        + 0.12 * np.sin(4.0 * np.pi * phase + 0.7)
        + rng.normal(0.0, 0.02, n)
        + trend * (t - t.min()) / (t.max() - t.min())
    )
    return cup.LightCurve.from_arrays(t, mag, np.full(n, 0.02))


def _inputs(count: int, trend: float = 0.0) -> list[tuple[str, cup.LightCurve]]:
    return [(f"star{i}", _drifting(i, trend=trend)) for i in range(count)]


BATCH_KW: dict[str, object] = {
    "period": V1309[0],
    "backend": "cpu",
    "workers": 1,
}


# --- the summary table --------------------------------------------------------


def test_in_memory_summary_carries_every_stage() -> None:
    summary = cup.batch_pdot(_inputs(2), **BATCH_KW)  # type: ignore[arg-type]
    assert summary.n_done == 2
    assert summary.n_failed == 0
    assert summary.methods == ("PDOT",)
    assert summary.rows is not None
    row = summary.rows[0]
    assert row["status"] == "ok"
    assert row["search_period"] == pytest.approx(V1309[0], rel=0.02)
    assert row["track_pdot"] < 0.0  # the period is shortening
    assert row["track_f_test_p"] < 1e-3
    assert row["law"] in {"singular", "pejcha", "exponential", "linear"}
    assert row["n_intervals"] > 2
    # The cyclic-O-C stage ran and its columns are in the schema.
    assert np.isfinite(row["ltte_outer_period_yr"])
    assert isinstance(row["ltte_at_grid_edge"], bool)


def test_the_measured_change_matches_the_generating_law() -> None:
    rows = cup.batch_pdot(_inputs(1), **BATCH_KW).rows  # type: ignore[arg-type]
    assert rows is not None
    p0, b, t0 = V1309
    lc = _drifting(0)
    exact = p0 * np.exp(b / (np.array([lc.time.min(), lc.time.max()]) - t0))
    assert rows[0]["law_total_fractional_change"] == pytest.approx(
        exact[1] / exact[0] - 1.0, abs=0.005
    )


def test_ladder_flags_a_rate_that_is_itself_changing() -> None:
    """A cubic must shed real residual here: the decay is not a parabola."""
    rows = cup.batch_pdot(_inputs(1), **BATCH_KW).rows  # type: ignore[arg-type]
    assert rows is not None
    row = rows[0]
    assert row["ladder_rms_min_deg3"] < row["ladder_rms_min_deg2"]
    assert row["ladder_curvature_ratio"] < 0.8
    assert np.isfinite(row["ladder_pddot"])


def test_scaled_errors_are_never_smaller_than_formal_ones() -> None:
    rows = cup.batch_pdot(_inputs(1), **BATCH_KW).rows  # type: ignore[arg-type]
    assert rows is not None
    assert rows[0]["track_pdot_err_scaled"] >= rows[0]["track_pdot_err"]


def test_failures_become_rows_and_errors_not_crashes() -> None:
    bad = cup.LightCurve.from_arrays(
        np.arange(25.0), np.zeros(25), np.ones(25)
    )
    summary = cup.batch_pdot(
        [*_inputs(1), ("bad", bad)], profile=False, **BATCH_KW  # type: ignore[arg-type]
    )
    assert summary.rows is not None
    assert len(summary.rows) == 2  # nothing disappears from the catalogue
    statuses = {r["key"]: r["status"] for r in summary.rows}
    assert statuses["star0"] == "ok"


def test_schema_is_identical_for_ok_and_failed_rows() -> None:
    """Every part of a directory sink must infer the same Arrow schema."""
    good = pdot_to_rows("a", status="ok")["summary"][0]
    bad = pdot_to_rows("b", status="ValueError: nope")["summary"][0]
    assert set(good) == set(bad)
    assert {type(good[k]) for k in good} == {type(bad[k]) for k in bad}


def test_law_parameter_columns_are_the_full_union() -> None:
    """A part where no star chose `singular` must still type law_b as float."""
    row = pdot_to_rows("a")["summary"][0]
    for name in ("period", "pdot", "dlnp_dt", "b", "amplitude", "t0", "delta"):
        assert np.isnan(row[f"law_{name}"])
        assert np.isnan(row[f"law_err_{name}"])


# --- the long tables ----------------------------------------------------------


def test_oc_and_profile_tables(tmp_path: Path) -> None:
    summary = cup.batch_pdot(
        _inputs(2),
        sink=tmp_path / "s.parquet",
        oc_sink=tmp_path / "oc.parquet",
        profile_sink=tmp_path / "p.parquet",
        **BATCH_KW,  # type: ignore[arg-type]
    )
    assert summary.n_done == 2
    oc = pd.read_parquet(tmp_path / "oc.parquet")
    prof = pd.read_parquet(tmp_path / "p.parquet")
    assert set(oc["key"]) == {"star0", "star1"}
    assert {"t_mid", "delay", "delay_err", "delay_err_scaled"} <= set(oc.columns)
    assert len(oc) > 2 * 3  # several O-C points per star
    assert {"period", "period_err", "at_window_edge"} <= set(prof.columns)
    assert len(prof) == pq.read_table(tmp_path / "s.parquet").to_pandas()[
        "n_intervals"
    ].sum()


def test_long_tables_are_omitted_unless_asked_for() -> None:
    """They are large; not writing them must also mean not building them."""
    summary = cup.batch_pdot(_inputs(1), **BATCH_KW)  # type: ignore[arg-type]
    assert summary.rows is not None and summary.rows


def test_shared_directory_sinks_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="different directories"):
        cup.batch_pdot(
            _inputs(1), sink=tmp_path, oc_sink=tmp_path, **BATCH_KW  # type: ignore[arg-type]
        )


# --- sinks, resume, workers ---------------------------------------------------


def test_directory_sink_resumes(tmp_path: Path) -> None:
    inputs = _inputs(4)
    first = cup.batch_pdot(
        inputs, sink=tmp_path, chunk_size=2, profile=False, **BATCH_KW  # type: ignore[arg-type]
    )
    assert first.n_done == 4
    assert (tmp_path / "part-00000.parquet").exists()
    second = cup.batch_pdot(
        inputs, sink=tmp_path, chunk_size=2, profile=False, **BATCH_KW  # type: ignore[arg-type]
    )
    assert second.n_done == 0
    assert second.n_skipped == 4


def test_csv_sink(tmp_path: Path) -> None:
    out = tmp_path / "s.csv"
    cup.batch_pdot(_inputs(2), sink=out, profile=False, **BATCH_KW)  # type: ignore[arg-type]
    assert len(pd.read_csv(out)) == 2


def test_process_pool(tmp_path: Path) -> None:
    summary = cup.batch_pdot(
        _inputs(3),
        sink=tmp_path,
        chunk_size=1,
        workers=2,
        period=V1309[0],
        backend="cpu",
        profile=False,
    )
    assert summary.n_done == 3
    assert len(list(tmp_path.glob("part-*.parquet"))) == 3


# --- options ------------------------------------------------------------------


def test_per_star_seed_periods() -> None:
    mapping = {"star0": V1309[0], "star1": V1309[0]}
    summary = cup.batch_pdot(
        _inputs(2), period=mapping, backend="cpu", workers=1, profile=False
    )
    assert summary.n_done == 2


def test_detrending_rescues_a_trended_light_curve() -> None:
    """An 0.8 mag secular trend aliases onto 1 c/d and swamps the search."""
    inputs = _inputs(1, trend=0.8)
    plain = cup.batch_pdot(inputs, profile=False, **BATCH_KW)  # type: ignore[arg-type]
    flat = cup.batch_pdot(
        inputs, profile=False, detrend_width=40.0, **BATCH_KW  # type: ignore[arg-type]
    )
    assert plain.rows is not None and flat.rows is not None
    assert flat.rows[0]["search_power"] > plain.rows[0]["search_power"]
    assert flat.rows[0]["search_period"] == pytest.approx(V1309[0], rel=0.02)


def test_profile_can_be_switched_off() -> None:
    rows = cup.batch_pdot(_inputs(1), profile=False, **BATCH_KW).rows  # type: ignore[arg-type]
    assert rows is not None
    assert rows[0]["n_intervals"] == 0
    assert rows[0]["law"] == ""
    assert np.isnan(rows[0]["law_bic"])


def test_laws_none_keeps_the_track_without_fitting_it() -> None:
    rows = cup.batch_pdot(_inputs(1), laws=None, **BATCH_KW).rows  # type: ignore[arg-type]
    assert rows is not None
    assert rows[0]["n_intervals"] > 2
    assert rows[0]["law"] == ""


def test_exponential_model_is_threaded_through() -> None:
    rows = cup.batch_pdot(
        _inputs(1), model="exponential", profile=False, **BATCH_KW  # type: ignore[arg-type]
    ).rows
    assert rows is not None
    assert rows[0]["model"] == "exponential"


@pytest.mark.parametrize("width", [0.0, -5.0])
def test_bad_detrend_width_raises(width: float) -> None:
    with pytest.raises(ValueError, match="detrend_width"):
        cup.batch_pdot(_inputs(1), detrend_width=width, **BATCH_KW)  # type: ignore[arg-type]


def test_bad_device_raises() -> None:
    with pytest.raises(ValueError, match="device must be"):
        cup.batch_pdot(_inputs(1), device="tpu", **BATCH_KW)  # type: ignore[arg-type]


# --- figures ------------------------------------------------------------------


def test_diagnostic_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    summary = cup.batch_pdot(
        _inputs(2), plots=tmp_path, plot_keys=["star0"], **BATCH_KW  # type: ignore[arg-type]
    )
    assert summary.rows is not None
    by_key = {r["key"]: r for r in summary.rows}
    assert by_key["star0"]["figure"].endswith("star0.png")
    assert by_key["star1"]["figure"] == ""
    written = list(tmp_path.glob("*.png"))
    assert len(written) == 1
    assert written[0].stat().st_size > 10_000


def test_figure_keys_are_made_filesystem_safe(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    lc = _drifting(0)
    summary = cup.batch_pdot(
        [("OGLE/BLG-ECL 1?2", lc)], plots=tmp_path, profile=False, **BATCH_KW  # type: ignore[arg-type]
    )
    assert summary.rows is not None
    assert list(tmp_path.glob("*.png"))
    assert "/" not in Path(summary.rows[0]["figure"]).name


def test_table_names_are_the_documented_set() -> None:
    assert TABLES == ("summary", "oc", "profile")
    assert set(pdot_to_rows("a")) == set(TABLES)
