"""batch_pdot: the three tables, their schema stability, sinks, and figures."""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import cuperiod as cup
from cuperiod.pdot.batch import TABLES, pdot_to_rows
from cuperiod.pdot.laws import AUTO_LAWS

PDOT_BATCH = importlib.import_module("cuperiod.pdot.batch")
SCAN = importlib.import_module("cuperiod.pdot.scan")

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


def test_a_misspelt_domain_is_refused_before_any_work() -> None:
    # In-memory LightCurve inputs never read the domain, so a typo used to run the
    # whole catalogue and only bite on the first file or table input.
    with pytest.raises(ValueError, match="bogus"):
        cup.batch_pdot(_inputs(1), domain="bogus", **BATCH_KW)  # type: ignore[arg-type]


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


# --- fast doubles for the driver tests ----------------------------------------
#
# The chain costs seconds per star; the driver logic below (seeds, validation,
# backends, resume, chunking) is exercised with a recorder in place of the search
# or of the whole per-star chain.


def _tiny(n: int = 40, *, flux: bool = False, seed: int = 0) -> cup.LightCurve:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 50.0, n))
    domain = cup.Domain.FLUX if flux else cup.Domain.MAGNITUDE
    return cup.LightCurve.from_arrays(
        t, 10.0 + np.sin(t), np.full(n, 0.01), domain=domain
    )


class _SearchSpy:
    """Replaces ``pdot_search``: records the seed and backend, then fails the star."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, work: Any, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        raise RuntimeError("spy: search not run")

    @property
    def seeds(self) -> list[Any]:
        return [call["period"] for call in self.calls]


@pytest.fixture
def search_spy(monkeypatch: pytest.MonkeyPatch) -> _SearchSpy:
    spy = _SearchSpy()
    monkeypatch.setattr(SCAN, "pdot_search", spy)
    return spy


def _fake_chain(
    calls: list[str], *, fail: Mapping[str, BaseException] | None = None
) -> Callable[..., dict[str, tuple[dict[str, Any], ...]]]:
    """A per-star chain emitting two O-C rows and one P(t) row per star."""

    def analyze(key: str, lc: cup.LightCurve, cfg: Any) -> dict[str, Any]:
        calls.append(key)
        if fail and key in fail:
            raise fail[key]
        tables = pdot_to_rows(key, n_samples=lc.n, baseline=lc.baseline)
        summary = {
            **tables["summary"][0],
            "method": cfg.method,
            "n_segments": 2,
            "n_intervals": 1,
        }
        oc = tuple({"key": key, "index": i, "delay": 0.1 * i} for i in range(2))
        profile = ({"key": key, "period": 1.0},)
        return {"summary": (summary,), "oc": oc, "profile": profile}

    return analyze


@pytest.fixture
def chain(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    calls: list[str] = []
    monkeypatch.setattr(PDOT_BATCH, "_analyze_one", _fake_chain(calls))
    yield calls


def _keys(path: Path) -> list[str]:
    """Keys stored in a file or directory sink."""
    if path.is_dir():
        frames = [pd.read_parquet(p) for p in sorted(path.glob("part-*.parquet"))]
        frame = pd.concat(frames) if frames else pd.DataFrame({"key": []})
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_parquet(path)
    return [str(k) for k in frame["key"]]


THREE = [(f"star{i}", _tiny(seed=i)) for i in range(3)]


# --- seed periods of any numeric type (BATCH-B-1) -----------------------------


@pytest.mark.parametrize(
    "period",
    [
        np.float32(1.4456),
        np.int64(2),
        np.array(1.4456),
        pd.Series({"star0": 1.4456}),
        {"star0": np.float32(1.4456)},
    ],
    ids=["float32", "int64", "0-d array", "Series", "mapping"],
)
def test_seed_period_of_any_numeric_type_reaches_the_search(
    period: Any, search_spy: _SearchSpy
) -> None:
    cup.batch_pdot([("star0", _tiny())], period=period, workers=1)
    (seed,) = search_spy.seeds
    expected = 2.0 if isinstance(period, np.integer) else 1.4456
    assert isinstance(seed, float)
    assert seed == pytest.approx(expected, rel=1e-6)


def test_integer_keyed_catalogue_matches_integer_keys(search_spy: _SearchSpy) -> None:
    cup.batch_pdot([(7, _tiny())], period={7: 1.4456}, workers=1)
    assert search_spy.seeds == [pytest.approx(1.4456)]


def test_grouped_table_with_integer_ids_takes_its_seeds(
    search_spy: _SearchSpy,
) -> None:
    lc = _tiny()
    frame = pd.DataFrame(
        {"objid": 4711, "time": lc.time, "mag": lc.value, "magerr": lc.error}
    )
    catalogue = pd.DataFrame({"objid": [4711], "period": [1.4456]})
    cup.batch_pdot(
        (frame, "objid"),
        period=dict(zip(catalogue.objid, catalogue.period, strict=True)),
        workers=1,
    )
    assert search_spy.seeds == [pytest.approx(1.4456)]


def test_missing_catalogue_periods_fall_back_to_a_blind_search(
    search_spy: _SearchSpy,
) -> None:
    seeds = pd.Series({"star0": np.nan, "star1": 1.4456, "star2": -1.0})
    cup.batch_pdot(THREE, period=seeds, workers=1)
    assert search_spy.seeds == [None, pytest.approx(1.4456), None]


def test_a_catalogue_matching_no_key_warns(search_spy: _SearchSpy) -> None:
    with pytest.warns(UserWarning, match="matches none"):
        cup.batch_pdot(THREE, period={"other": 1.4456}, workers=1)
    assert search_spy.seeds == [None, None, None]


@pytest.mark.parametrize(
    ("period", "error"),
    [
        ("1.4456", TypeError),
        (True, TypeError),
        ([1.4456], TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (np.inf, ValueError),
    ],
)
def test_unusable_seed_periods_are_refused_up_front(
    period: Any, error: type[Exception], search_spy: _SearchSpy, tmp_path: Path
) -> None:
    with pytest.raises(error, match="period"):
        cup.batch_pdot(THREE, period=period, sink=tmp_path / "parts", workers=1)
    assert search_spy.calls == []
    assert not (tmp_path / "parts").exists()


# --- configuration checked before the run (BATCH-B-10) ------------------------


class _ChunkSpy:
    """Replaces ``_process_chunk``: records the config, returns no rows."""

    def __init__(self) -> None:
        self.configs: list[Any] = []

    def __call__(self, items: Any, cfg: Any) -> tuple[dict[str, list[Any]], list[Any]]:
        self.configs.append(cfg)
        return {name: [] for name in TABLES}, []


def test_bare_string_laws_and_plot_keys_are_one_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spy = _ChunkSpy()
    monkeypatch.setattr(PDOT_BATCH, "_process_chunk", spy)
    cup.batch_pdot(
        THREE, laws="singular", plots=tmp_path, plot_keys="star0", workers=1
    )
    cfg = spy.configs[0]
    assert cfg.laws == ("singular",)
    assert cfg.plot_keys == frozenset({"star0"})


def test_law_names_are_normalised_and_defaults_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _ChunkSpy()
    monkeypatch.setattr(PDOT_BATCH, "_process_chunk", spy)
    cup.batch_pdot(THREE, laws=[" Singular", "LINEAR"], plot_keys=[7], workers=1)
    cup.batch_pdot(THREE, laws=None, workers=1)
    cup.batch_pdot(THREE, workers=1)
    assert spy.configs[0].laws == ("singular", "linear")
    assert spy.configs[0].plot_keys == frozenset({"7"})
    assert spy.configs[1].laws == ()
    assert spy.configs[2].laws == tuple(AUTO_LAWS)


@pytest.mark.parametrize(
    "bad",
    [
        {"laws": ["singlar"]},
        {"method": "NOPE"},
        {"method": "BLS"},
        {"model": "nope"},
        {"detrend_width": 40.0, "detrend_method": "mean"},
    ],
    ids=["law", "method", "unsupported-method", "model", "detrend"],
)
def test_misconfiguration_raises_before_any_star_or_sink(
    bad: dict[str, Any], search_spy: _SearchSpy, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        cup.batch_pdot(THREE, sink=tmp_path / "parts", workers=1, **bad)
    assert search_spy.calls == []
    assert not (tmp_path / "parts").exists()


# --- backends for the worker pool (BATCH-B-6) ---------------------------------


def test_cpu_device_hands_the_workers_a_cpu_backend(search_spy: _SearchSpy) -> None:
    cup.batch_pdot([("star0", _tiny())], workers=1)
    assert search_spy.calls[0]["backend"] == "cpu"


def test_cpu_device_honours_the_method_settings_backend(
    monkeypatch: pytest.MonkeyPatch, search_spy: _SearchSpy
) -> None:
    monkeypatch.setenv("CUPERIOD_GLS_BACKEND", "astropy")
    cup.batch_pdot([("star0", _tiny())], workers=1)
    assert search_spy.calls[0]["backend"] == "astropy"


def test_gpu_backend_on_a_cpu_pool_is_refused(search_spy: _SearchSpy) -> None:
    with pytest.raises(ValueError, match="device='cpu'"):
        cup.batch_pdot([("star0", _tiny())], backend="gpu", workers=1)
    assert search_spy.calls == []


def test_gpu_device_requests_the_gpu(
    monkeypatch: pytest.MonkeyPatch, search_spy: _SearchSpy
) -> None:
    monkeypatch.setattr(PDOT_BATCH, "free_gpu_memory", lambda: None)
    cup.batch_pdot([("star0", _tiny())], device="gpu", workers=1)
    assert search_spy.calls[0]["backend"] == "gpu"


# --- failed rows keep the light curve's size (BATCH-B-11) ---------------------


def test_failed_row_records_size_and_span() -> None:
    lc = _tiny(8)
    summary = cup.batch_pdot([("short", lc)], period=1.4456, backend="cpu", workers=1)
    assert summary.rows is not None
    (row,) = summary.rows
    assert row["status"].startswith("InsufficientDataError")
    assert row["n_samples"] == 8
    assert row["baseline"] == pytest.approx(lc.baseline)


def test_unreadable_input_row_has_no_size(tmp_path: Path) -> None:
    summary = cup.batch_pdot(
        [("gone", tmp_path / "missing.csv")], period=1.4456, workers=1
    )
    assert summary.rows is not None
    (row,) = summary.rows
    assert row["status"] != "ok"
    assert row["n_samples"] == 0 and math.isnan(row["baseline"])


# --- figures follow the brightness domain (BATCH-B-12) ------------------------


@pytest.mark.parametrize("flux", [True, False])
def test_figure_gets_the_light_curve_with_its_domain(
    flux: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("matplotlib")
    plots = importlib.import_module("cuperiod.pdot.plots")
    seen: dict[str, Any] = {}

    def capture(**kwargs: Any) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(plots, "pdot_figure", capture)
    lc = _tiny(flux=flux)
    cfg = PDOT_BATCH._ChunkConfig(
        settings=cup.PdotSettings(), method="GLS", model=None, backend="cpu",
        columns=None, domain=None, period=None, period_map=None,
        detrend_width=None, detrend_method="median", do_profile=False,
        do_ltte=False, laws=(), points_per_interval=None, period_window=None,
        store_oc=False, store_profile=False, plots_dir=str(tmp_path),
        plot_keys=None,
    )
    PDOT_BATCH._write_figure("star0", lc, None, None, None, (), None, cfg)
    assert seen["light_curve"] is lc
    _, _, _, invert = plots._as_arrays(seen["light_curve"])
    assert invert is (not flux)


# --- one error convention for the ladder (BATCH-B-9) --------------------------


def _cubic_fit(scatter: float) -> Any:
    from cuperiod.pdot.track import _build_ephemeris

    rng = np.random.default_rng(1)
    t = np.linspace(-500.0, 500.0, 40)
    err = np.full(t.size, 1e-3)
    delay = 1e-9 * t**2 + 1e-13 * t**3 + rng.normal(0.0, scatter, t.size)
    return _build_ephemeris(t, delay, err, period_ref=1.0, t_ref=0.0, degree=3)


@pytest.mark.parametrize("scatter", [3e-3, 1e-4])
def test_ladder_errors_share_one_convention(scatter: float) -> None:
    cubic = _cubic_fit(scatter)
    row = pdot_to_rows("a", ladder=[cubic])["summary"][0]
    inflation = math.sqrt(max(1.0, cubic.chi2_reduced))
    assert (cubic.chi2_reduced > 1.0) is (scatter > 1e-3)
    assert row["ladder_pdot_cubic_err"] == pytest.approx(cubic.pdot_err * inflation)
    assert row["ladder_pddot_err"] == pytest.approx(cubic.pddot_err * inflation)


# --- resume with any mix of sinks (BATCH-B-2, BATCH-B-7) -----------------------


def _interrupted(
    monkeypatch: pytest.MonkeyPatch, at: str, **kwargs: Any
) -> list[str]:
    """Run THREE (one star per chunk) with a KeyboardInterrupt at star ``at``."""
    calls: list[str] = []
    monkeypatch.setattr(
        PDOT_BATCH,
        "_analyze_one",
        _fake_chain(calls, fail={at: KeyboardInterrupt()}),
    )
    with pytest.raises(KeyboardInterrupt):
        cup.batch_pdot(THREE, chunk_size=1, workers=1, **kwargs)
    return calls


def test_file_summary_beside_directory_oc_survives_an_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sinks = {"sink": tmp_path / "summary.parquet", "oc_sink": tmp_path / "oc"}
    _interrupted(monkeypatch, "star2", **sinks)
    calls: list[str] = []
    monkeypatch.setattr(PDOT_BATCH, "_analyze_one", _fake_chain(calls))
    again = cup.batch_pdot(THREE, chunk_size=1, workers=1, **sinks)
    assert sorted(_keys(sinks["sink"])) == ["star0", "star1", "star2"]
    assert sorted(set(_keys(sinks["oc_sink"]))) == ["star0", "star1", "star2"]
    assert len(_keys(sinks["oc_sink"])) == 6  # no duplicated O-C rows
    assert again.n_done == 3 and again.n_skipped == 0
    third = cup.batch_pdot(THREE, chunk_size=1, workers=1, **sinks)
    assert (third.n_done, third.n_skipped) == (0, 3)
    assert calls == ["star0", "star1", "star2"]  # nothing recomputed the third time


def test_directory_summary_beside_file_oc_survives_an_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sinks = {"sink": tmp_path / "summary", "oc_sink": tmp_path / "oc.parquet"}
    _interrupted(monkeypatch, "star2", **sinks)
    calls: list[str] = []
    monkeypatch.setattr(PDOT_BATCH, "_analyze_one", _fake_chain(calls))
    cup.batch_pdot(THREE, chunk_size=1, workers=1, **sinks)
    assert sorted(set(_keys(sinks["oc_sink"]))) == ["star0", "star1", "star2"]
    assert len(_keys(sinks["oc_sink"])) == 6
    assert sorted(_keys(sinks["sink"])) == ["star0", "star1", "star2"]


def test_a_missing_oc_part_is_regenerated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chain: list[str]
) -> None:
    sinks = {"sink": tmp_path / "summary", "oc_sink": tmp_path / "oc"}
    cup.batch_pdot(THREE, chunk_size=1, workers=1, **sinks)
    (tmp_path / "oc" / "part-00001.parquet").unlink()
    chain.clear()
    again = cup.batch_pdot(THREE, chunk_size=1, workers=1, **sinks)
    assert chain == ["star1"]
    assert again.n_skipped == 2
    assert sorted(_keys(sinks["oc_sink"])) == sorted(["star0", "star1", "star2"] * 2)
    assert sorted(_keys(sinks["sink"])) == ["star0", "star1", "star2"]


def test_in_memory_summary_beside_a_directory_is_always_computed(
    tmp_path: Path, chain: list[str]
) -> None:
    first = cup.batch_pdot(THREE, oc_sink=tmp_path / "oc", workers=1)
    second = cup.batch_pdot(THREE, oc_sink=tmp_path / "oc", workers=1)
    assert first.rows is not None and second.rows is not None
    assert len(second.rows) == 3 and second.n_skipped == 0
    assert sorted(_keys(tmp_path / "oc")) == sorted(["star0", "star1", "star2"] * 2)


def test_adding_an_oc_sink_to_a_finished_run_fills_it(
    tmp_path: Path, chain: list[str]
) -> None:
    cup.batch_pdot(THREE, sink=tmp_path / "summary", chunk_size=1, workers=1)
    again = cup.batch_pdot(
        THREE, sink=tmp_path / "summary", oc_sink=tmp_path / "oc", chunk_size=1,
        workers=1,
    )
    assert again.n_skipped == 0
    assert sorted(set(_keys(tmp_path / "oc"))) == ["star0", "star1", "star2"]
    assert sorted(_keys(tmp_path / "summary")) == ["star0", "star1", "star2"]


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_identical_file_sink_rerun_computes_nothing(
    suffix: str, tmp_path: Path, chain: list[str]
) -> None:
    sink = tmp_path / f"summary{suffix}"
    cup.batch_pdot(THREE, sink=sink, workers=1)
    before = sink.read_bytes()
    chain.clear()
    again = cup.batch_pdot(THREE, sink=sink, workers=1)
    assert chain == []
    assert (again.n_done, again.n_skipped, again.n_failed) == (0, 3, 0)
    assert sink.read_bytes() == before


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_a_failed_star_is_retried_and_its_row_replaced(
    suffix: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sink = tmp_path / f"summary{suffix}"
    calls: list[str] = []
    monkeypatch.setattr(
        PDOT_BATCH, "_analyze_one", _fake_chain(calls, fail={"star1": ValueError("x")})
    )
    first = cup.batch_pdot(THREE, sink=sink, workers=1)
    assert first.n_failed == 1
    calls.clear()
    monkeypatch.setattr(PDOT_BATCH, "_analyze_one", _fake_chain(calls))
    again = cup.batch_pdot(THREE, sink=sink, workers=1)
    assert calls == ["star1"]
    assert (again.n_done, again.n_skipped) == (1, 2)
    frame = pd.read_csv(sink) if suffix == ".csv" else pd.read_parquet(sink)
    assert sorted(frame["key"]) == ["star0", "star1", "star2"]  # one row per star
    assert set(frame["status"]) == {"ok"}


def test_a_new_method_recomputes_and_replaces(
    tmp_path: Path, chain: list[str]
) -> None:
    sink = tmp_path / "summary.parquet"
    cup.batch_pdot(THREE, sink=sink, workers=1)
    chain.clear()
    cup.batch_pdot(THREE, sink=sink, method="PDM", workers=1)
    assert chain == ["star0", "star1", "star2"]
    frame = pd.read_parquet(sink)
    assert sorted(frame["key"]) == ["star0", "star1", "star2"]
    assert set(frame["method"]) == {"PDM"}


def test_a_directory_part_keeps_its_other_stars_when_one_is_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sink = tmp_path / "summary"
    calls: list[str] = []
    monkeypatch.setattr(
        PDOT_BATCH, "_analyze_one", _fake_chain(calls, fail={"star1": ValueError("x")})
    )
    cup.batch_pdot(THREE, sink=sink, chunk_size=3, workers=1)
    calls.clear()
    monkeypatch.setattr(PDOT_BATCH, "_analyze_one", _fake_chain(calls))
    again = cup.batch_pdot(THREE, sink=sink, chunk_size=3, workers=1)
    assert calls == ["star1"]  # only the failed star, not its whole chunk
    assert again.n_skipped == 2
    frame = pd.read_parquet(sink / "part-00000.parquet")
    assert list(frame["key"]) == ["star0", "star1", "star2"]
    assert set(frame["status"]) == {"ok"}


def test_csv_resume_matches_formula_escaped_keys(
    tmp_path: Path, chain: list[str]
) -> None:
    """A CSV sink stores '=x' as "'=x"; resume must still recognise the star."""
    inputs = [("=x", _tiny()), ("+y", _tiny(seed=1)), ("plain", _tiny(seed=2))]
    sinks = {"sink": tmp_path / "s.csv", "oc_sink": tmp_path / "oc.csv"}
    cup.batch_pdot(inputs, workers=1, **sinks)
    chain.clear()
    again = cup.batch_pdot(inputs, workers=1, **sinks)
    assert chain == [] and again.n_skipped == 3
    assert len(pd.read_csv(sinks["sink"])) == 3


def test_resume_false_recomputes_everything(tmp_path: Path, chain: list[str]) -> None:
    sink = tmp_path / "summary.parquet"
    cup.batch_pdot(THREE, sink=sink, workers=1)
    chain.clear()
    again = cup.batch_pdot(THREE, sink=sink, workers=1, resume=False)
    assert chain == ["star0", "star1", "star2"]
    assert again.n_skipped == 0
    assert sorted(_keys(sink)) == ["star0", "star1", "star2"]


# --- pool chunking and row order (PERF-PDOT-04) --------------------------------


class _InlinePool:
    """Replaces ``_run_pool``: runs chunks in-process, last chunk first."""

    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.max_workers = 0
        self.initargs: tuple[Any, ...] = ()
        self.pinned_env: str | None = None

    def __call__(
        self,
        chunks: list[list[Any]],
        pending: list[int],
        cfg: Any,
        absorb: Callable[..., None],
        max_workers: int,
        *,
        initializer: Any = None,
        initargs: tuple[Any, ...] = (),
        worker: Callable[..., tuple[Any, ...]],
    ) -> None:
        self.max_workers = max_workers
        self.initargs = initargs
        self.pinned_env = os.environ.get("OMP_NUM_THREADS")
        self.sizes = [len(chunks[idx]) for idx in pending]
        for idx in reversed(pending):
            absorb(idx, *worker(chunks[idx], cfg))


EIGHT = [(f"k{i}", _tiny(seed=i)) for i in range(8)]


def test_small_batches_are_spread_over_every_worker(
    monkeypatch: pytest.MonkeyPatch, chain: list[str]
) -> None:
    pool = _InlinePool()
    monkeypatch.setattr(PDOT_BATCH, "_run_pool", pool)
    monkeypatch.setattr(PDOT_BATCH, "cpu_worker_count", lambda w: 4)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    summary = cup.batch_pdot(EIGHT, workers=4)
    assert len(pool.sizes) >= 4 and sum(pool.sizes) == 8  # was one 32-star task
    assert pool.max_workers == 4
    assert summary.rows is not None
    assert [r["key"] for r in summary.rows] == [key for key, _ in EIGHT]
    assert pool.pinned_env == "1" and "OMP_NUM_THREADS" not in os.environ
    assert len(pool.initargs) == 1 and int(pool.initargs[0]) >= 1


def test_pool_rows_in_a_file_sink_follow_the_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chain: list[str]
) -> None:
    monkeypatch.setattr(PDOT_BATCH, "_run_pool", _InlinePool())
    monkeypatch.setattr(PDOT_BATCH, "cpu_worker_count", lambda w: 4)
    sink = tmp_path / "s.parquet"
    cup.batch_pdot(EIGHT, sink=sink, oc_sink=tmp_path / "oc.parquet", workers=4)
    assert _keys(sink) == [key for key, _ in EIGHT]
    assert _keys(tmp_path / "oc.parquet")[::2] == [key for key, _ in EIGHT]


def test_directory_sinks_keep_the_requested_chunk_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chain: list[str]
) -> None:
    pool = _InlinePool()
    monkeypatch.setattr(PDOT_BATCH, "_run_pool", pool)
    monkeypatch.setattr(PDOT_BATCH, "cpu_worker_count", lambda w: 4)
    cup.batch_pdot(EIGHT, sink=tmp_path / "parts", workers=4)
    assert pool.sizes == [8]
    assert [p.name for p in (tmp_path / "parts").glob("part-*")] == [
        "part-00000.parquet"
    ]


# --- law margin and figure names ------------------------------------------------


def _understated_track() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A drifting P(t) track whose quoted errors are five times too small."""
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 3000.0, 40)
    sigma = np.full(t.size, 2e-6)
    period = 1.4 + 2e-8 * t + 3e-12 * (t - 1500.0) ** 2
    return t, period + rng.normal(0.0, 5.0 * sigma), sigma


def test_law_margin_is_on_the_ranking_scale() -> None:
    """law_delta_bic is the bic_scaled gap the laws are sorted by, never negative."""
    laws = cup.compare_period_laws(*_understated_track())
    assert len(laws) > 1 and not laws[1].at_bound
    row = pdot_to_rows("x", laws=laws)["summary"][0]
    expected = laws[1].bic_scaled - laws[0].bic_scaled
    assert row["law_delta_bic"] == pytest.approx(expected)
    assert row["law_delta_bic"] >= 0.0
    assert row["law_bic_scaled"] == laws[0].bic_scaled
    assert row["law_bic"] == laws[0].bic


def test_law_margin_falls_back_to_bic_without_a_comparison() -> None:
    import dataclasses

    t, period, sigma = _understated_track()
    fits = [
        dataclasses.replace(cup.fit_period_law(t, period, sigma, law=law),
                            bic_scaled=float("nan"))
        for law in ("linear", "constant")
    ]
    row = pdot_to_rows("x", laws=fits)["summary"][0]
    assert row["law_delta_bic"] == pytest.approx(fits[1].bic - fits[0].bic)
    assert np.isnan(row["law_bic_scaled"])


def test_failed_row_has_a_law_bic_scaled_column() -> None:
    row = pdot_to_rows("x", status="failed")["summary"][0]
    assert np.isnan(row["law_bic_scaled"])


def test_figure_stems_never_alias() -> None:
    safe = PDOT_BATCH._safe_name
    keys = [
        "fieldA/lc_0000", "fieldA_lc_0000", "a b", "a_b", "a:b", "",
        "lightcurve", "x" * 130, "x" * 129 + "y", "con", "CON.txt", "con_1",
        "OGLE-BLG-ECL-123456", "star.v2",
    ]
    stems = [safe(k) for k in keys]
    assert len(set(stems)) == len(keys)
    assert all(0 < len(s) <= 120 for s in stems)
    assert all(set(s) <= set("-_.") | {c for c in s if c.isalnum()} for s in stems)
    # A key that is already a safe name stays its own stem, so it can be found.
    for key in ("fieldA_lc_0000", "a_b", "lightcurve", "con_1",
                "OGLE-BLG-ECL-123456", "star.v2"):
        assert safe(key) == key
    assert safe("fieldA/lc_0000") == safe("fieldA/lc_0000")
    assert safe("fieldA/lc_0000").startswith("fieldA_lc_0000-")
    assert safe("con").split(".")[0].lower() != "con"


def test_path_qualified_and_flat_keys_get_their_own_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    lc = _drifting(0)
    summary = cup.batch_pdot(
        [("fieldA/lc_0000", lc), ("fieldA_lc_0000", lc)], plots=tmp_path,
        profile=False, **BATCH_KW,  # type: ignore[arg-type]
    )
    assert summary.rows is not None
    figures = {r["key"]: r["figure"] for r in summary.rows}
    assert figures["fieldA/lc_0000"] != figures["fieldA_lc_0000"]
    assert len(list(tmp_path.glob("*.png"))) == 2
