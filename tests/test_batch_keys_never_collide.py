"""Same-named batch inputs never collide or merge in a driver or a sink (SEC-05).

A batch key is a light curve's identity in every sink: its rows, the resume check,
``batch_pdot``'s period catalogue and figure names. Colliding file stems were already
path-qualified; these tests cover the rest: explicit or derived keys that repeat,
keys a CSV sink would store identically, the pre-whitening and period-change sinks
recording which file a row came from (so a resume from a same-named file elsewhere
is reported, as ``batch_periodograms`` reports it), and figure names that differ
only in letter case.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

import cuperiod as cup
from cuperiod.batch.io import resolve_inputs, source_label
from cuperiod.pdot import batch as pdot_batch
from cuperiod.pdot.batch import pdot_to_rows
from synth import synthetic_sine


def _lc(seed: int = 0, **meta: Any) -> cup.LightCurve:
    t, mag, err = synthetic_sine(n=120, period=0.5 + 0.05 * seed, seed=seed)
    return cup.LightCurve.from_arrays(t, mag, err, meta=meta or None)


def _write_csv(path: Path, seed: int = 0) -> Path:
    t, mag, err = synthetic_sine(n=120, period=0.5 + 0.05 * seed, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"jd": t, "mag": mag, "mag_err": err}).to_csv(path, index=False)
    return path


# --- keys are made unique at resolution ----------------------------------------------


def test_repeated_explicit_keys_are_renamed_not_merged() -> None:
    a, b, c = _lc(0), _lc(1), _lc(2)
    with pytest.warns(UserWarning, match="more than once.*'x#3'") as rec:
        items = resolve_inputs([("x", a), ("x", b), ("x#2", c)])
    # The first occurrence keeps its key; the repeat skips the real key "x#2".
    assert [k for k, _ in items] == ["x", "x#3", "x#2"]
    assert [src for _, src in items] == [a, b, c]
    assert rec[0].filename == __file__  # attributed to the caller
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # unique keys: untouched and silent
        assert [k for k, _ in resolve_inputs([("a", a), ("b", b)])] == ["a", "b"]


def test_derived_keys_are_made_unique() -> None:
    # Two light curves of one star (one meta id), and a meta id equal to another
    # entry's position: both used to give two inputs one key.
    with pytest.warns(UserWarning, match="more than once"):
        same_id = resolve_inputs([_lc(0, id="s1"), _lc(1, id="s1")])
    assert [k for k, _ in same_id] == ["s1", "s1#2"]
    with pytest.warns(UserWarning, match="more than once"):
        mixed = resolve_inputs([_lc(0, id="1"), _lc(1)])  # index 1 -> key "1"
    assert [k for k, _ in mixed] == ["1", "1#2"]


def test_keys_a_csv_sink_would_store_identically_are_kept_apart(
    tmp_path: Path,
) -> None:
    # "=x" is formula-escaped to "'=x" in a CSV cell: the same text as the key "'=x".
    with pytest.warns(UserWarning, match="more than once"):
        items = resolve_inputs([("=x", _lc(0)), ("'=x", _lc(1))])
    assert [k for k, _ in items] == ["=x", "'=x#2"]
    sink = tmp_path / "rows.csv"
    with pytest.warns(UserWarning, match="more than once"):
        cup.batch_periodograms(
            [("=x", _lc(0)), ("'=x", _lc(1))], "GLS", workers=0, sink=sink
        )
    stored = pd.read_csv(sink, dtype=str)["key"].tolist()
    assert len(stored) == len(set(stored)) == 2


def test_a_repeat_is_not_skipped_as_done_by_a_resumed_sink(tmp_path: Path) -> None:
    # Run 1 wrote "x" for one light curve; run 2 has two light curves keyed "x".
    # The second used to be skipped as done although it was never computed.
    sink = tmp_path / "rows.parquet"
    cup.batch_periodograms([("x", _lc(0))], "GLS", workers=0, sink=sink)
    with pytest.warns(UserWarning, match="more than once"):
        again = cup.batch_periodograms(
            [("x", _lc(0)), ("x", _lc(1))], "GLS", workers=0, sink=sink
        )
    assert (again.n_done, again.n_skipped) == (1, 1)
    assert sorted(pq.read_table(sink).column("key").to_pylist()) == ["x", "x#2"]


# --- pre-whitening rows record their file; a clashing resume is reported --------------


def _two_fields(tmp_path: Path) -> tuple[Path, Path]:
    a = _write_csv(tmp_path / "fieldA" / "lc.csv", seed=0)
    b = _write_csv(tmp_path / "fieldB" / "lc.csv", seed=3)
    return a, b


@pytest.mark.parametrize("sink_name", ["rows.parquet", "rows.csv", "rows_dir"])
def test_prewhiten_reports_a_same_named_file_from_elsewhere(
    tmp_path: Path, sink_name: str
) -> None:
    a, b = _two_fields(tmp_path)
    sink = tmp_path / sink_name
    settings = cup.PreWhitenSettings(max_frequencies=1)
    first = cup.batch_prewhiten(str(a), settings=settings, workers=0, sink=sink)
    assert first.n_done >= 1
    rows = (
        pd.read_parquet(sink) if sink.suffix != ".csv" else pd.read_csv(sink, dtype=str)
    )
    assert set(rows["source"]) == {source_label(a)}
    with pytest.warns(UserWarning, match="different file") as rec:
        second = cup.batch_prewhiten(str(b), settings=settings, workers=0, sink=sink)
    assert (second.n_done, second.n_skipped) == (0, 1)  # reported, still skipped
    assert any(w.filename == __file__ for w in rec)
    # The same file again is a plain resume: no report.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*different file")
        cup.batch_prewhiten(str(a), settings=settings, workers=0, sink=sink)


def test_prewhiten_rows_carry_the_source_column() -> None:
    summary = cup.batch_prewhiten(
        [("mem", _lc(0))], settings=cup.PreWhitenSettings(max_frequencies=1),
        workers=0,
    )
    assert summary.rows is not None and summary.rows
    assert all(row["source"] == "" for row in summary.rows)  # in memory: no file
    assert list(summary.rows[0])[:2] == ["key", "source"]


# --- period-change rows record their file; a clashing resume is reported -------------


def _fake_chain(key: str, lc: cup.LightCurve, cfg: Any) -> dict[str, Any]:
    """One finished summary row per star, without running the (slow) chain."""
    row = pdot_to_rows(key, n_samples=lc.n, baseline=lc.baseline)["summary"][0]
    return {"summary": ({**row, "method": cfg.method},), "oc": (), "profile": ()}


@pytest.mark.parametrize("sink_name", ["summary.parquet", "summary.csv", "summary_dir"])
def test_pdot_reports_a_same_named_file_from_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sink_name: str
) -> None:
    monkeypatch.setattr(pdot_batch, "_analyze_one", _fake_chain)
    a, b = _two_fields(tmp_path)
    sink = tmp_path / sink_name
    cup.batch_pdot(str(a), workers=0, sink=sink)
    with pytest.warns(UserWarning, match="different file"):
        second = cup.batch_pdot(str(b), workers=0, sink=sink)
    assert (second.n_done, second.n_skipped) == (0, 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*different file")
        cup.batch_pdot(str(a), workers=0, sink=sink)


def test_pdot_summary_rows_carry_the_source_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pdot_batch, "_analyze_one", _fake_chain)
    a = _write_csv(tmp_path / "lc.csv")
    summary = cup.batch_pdot([a, ("mem", _lc(0))], workers=0)
    assert summary.rows is not None
    assert [row["source"] for row in summary.rows] == [source_label(a), ""]
    # The public row builder carries the column too, so a hand-made row matches.
    assert list(pdot_to_rows("k", source="/x.csv")["summary"][0])[:3] == [
        "key", "source", "status",
    ]
    failed = pdot_to_rows("k", status="ValueError: x")["summary"][0]
    assert failed["source"] == ""


# --- figure files that differ only in letter case ------------------------------------


def test_figure_stems_that_differ_only_in_case_are_told_apart() -> None:
    stems = pdot_batch._figure_stems(["V1", "v1", "star", "fieldA/lc"])
    assert set(stems) == {"V1", "v1"}  # every other key keeps _safe_name
    assert stems["V1"].startswith("V1-") and stems["v1"].startswith("v1-")
    assert stems["V1"].casefold() != stems["v1"].casefold()
    assert pdot_batch._figure_stems(["a", "b"]) == {}


def test_case_clashing_keys_get_two_figure_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cuperiod.pdot import plots

    written: list[Path] = []

    def capture(**kwargs: Any) -> None:
        path = Path(kwargs["path"])
        path.write_bytes(b"png")
        written.append(path)

    monkeypatch.setattr(plots, "pdot_figure", capture)
    cfg = pdot_batch._ChunkConfig(
        settings=cup.PdotSettings(), method="GLS", model=None, backend="cpu",
        columns=None, domain=None, period=None, period_map=None,
        detrend_width=None, detrend_method="median", do_profile=False,
        do_ltte=False, laws=(), points_per_interval=None, period_window=None,
        store_oc=False, store_profile=False, plots_dir=str(tmp_path),
        plot_keys=None, figure_stems=pdot_batch._figure_stems(["V1", "v1"]),
    )
    lc = _lc(0)
    for key in ("V1", "v1"):
        pdot_batch._write_figure(key, lc, None, None, None, (), None, cfg)
    assert len({p.name.casefold() for p in written}) == 2
    assert len(list(tmp_path.glob("*.png"))) == 2  # neither overwrote the other
