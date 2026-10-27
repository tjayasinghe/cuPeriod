"""batch_pdot figures: the O-C panel's light-time orbit, and a missing matplotlib.

A figure draws the cyclic search's best orbit only when the O-C supports it (ΔBIC
over the polynomial alone above 10, and the outer period not at the edge of the
scan): the search always returns a best trial, and drawn and labelled with its
period and amplitude a non-detection read as a fitted orbit.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cuperiod as cup

PDOT_BATCH = importlib.import_module("cuperiod.pdot.batch")


def _oc(orbit: bool, seed: int = 1) -> cup.LTTESearchResult:
    """A cyclic search over 40 O-C points: a parabola, with or without an orbit."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 5000.0, 40)) + 2450000.0
    dt = t - t.mean()
    oc = 0.5 * (3e-8 / 0.4) * dt**2 + rng.normal(0.0, 5e-4, t.size)
    if orbit:
        oc += 0.003 * np.sin(2.0 * np.pi * dt / 1300.0 + 0.4)
    return cup.ltte_search(t, oc=oc, errors=np.full(t.size, 5e-4), period=0.4)


def _drawn_orbit(
    cyclic: cup.LTTESearchResult, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Any:
    """The ``ltte=`` that ``_write_figure`` hands to ``pdot_figure``."""
    pytest.importorskip("matplotlib")
    plots = importlib.import_module("cuperiod.pdot.plots")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(plots, "pdot_figure", lambda **kw: seen.update(kw))
    t = np.linspace(0.0, 100.0, 200)
    lc = cup.LightCurve.from_arrays(t, np.sin(t), np.full(t.size, 0.1))
    cfg = PDOT_BATCH._ChunkConfig(
        settings=cup.PdotSettings(), method="GLS", model=None, backend="cpu",
        columns=None, domain=None, period=None, period_map=None,
        detrend_width=None, detrend_method="median", do_profile=False,
        do_ltte=True, laws=(), points_per_interval=None, period_window=None,
        store_oc=False, store_profile=False, plots_dir=str(tmp_path),
        plot_keys=None,
    )
    PDOT_BATCH._write_figure("star0", lc, None, None, None, (), cyclic, cfg)
    return seen["ltte"]


def test_a_detected_orbit_is_drawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cyclic = _oc(orbit=True)
    assert cyclic.delta_bic > 10.0 and not cyclic.best.at_grid_edge
    assert _drawn_orbit(cyclic, monkeypatch, tmp_path) is cyclic.best


def test_an_undetected_orbit_is_not_drawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cyclic = _oc(orbit=False)
    assert cyclic.delta_bic < 10.0
    assert _drawn_orbit(cyclic, monkeypatch, tmp_path) is None


def test_an_orbit_at_the_edge_of_the_scan_is_not_drawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detected = _oc(orbit=True)
    at_edge = replace(detected, best=replace(detected.best, at_grid_edge=True))
    assert _drawn_orbit(at_edge, monkeypatch, tmp_path) is None


def test_plots_without_matplotlib_fail_before_any_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Per-star plotting failures are swallowed, so a missing matplotlib used to
    # finish "ok" with an empty figure column and no word of the [plot] extra.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    t = np.linspace(0.0, 100.0, 200)
    lc = cup.LightCurve.from_arrays(t, np.sin(t), np.full(t.size, 0.1))
    sink = tmp_path / "summary.parquet"
    with pytest.raises(ImportError, match=r"cuperiod\[plot\]"):
        cup.batch_pdot(
            [("star0", lc)], period=1.0, backend="cpu", workers=1, sink=sink,
            plots=tmp_path / "figs",
        )
    assert not sink.exists()
    assert not (tmp_path / "figs").exists()
