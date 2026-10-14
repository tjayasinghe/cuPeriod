"""Batched rows of a period-derivative scan (PERF-PDOT-10).

``pdot_search`` evaluates the method once per trial rate on warped copies of the
curve. GLS on cufinufft and PDM on cupy now take blocks of rows at once
(:meth:`PeriodogramMethod.power_rows`): one upload, one transform or launch per row,
one download. The rows must be the ones the per-row loop produces, and a method
without the path -- or one declining a block -- must fall back to the loop.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu, requires_numba
from cuperiod.methods.base import get_method
from cuperiod.pdot import scan as scan_module
from cuperiod.pdot.models import get_model
from synth import synthetic_period_change


def _warped_rows(n: int = 900, n_rows: int = 7, seed: int = 4):
    t, mag, err, truth = synthetic_period_change(n=n, seed=seed)
    lc = cup.LightCurve.from_arrays(t, mag, err).finite()
    t_ref = float(np.mean(lc.time))
    model = get_model(None)
    rates = np.linspace(-3e-6, 3e-6, n_rows)
    times = np.stack([model.warp(lc.time - t_ref, float(a)) for a in rates])
    return lc, times, truth


def test_methods_without_a_batched_path_decline() -> None:
    lc, times, _ = _warped_rows(n_rows=2)
    grid = cup.uniform_frequency_grid(float(np.ptp(lc.time)), maximum_frequency=3.0)
    for name, backend in (("GLS", "finufft"), ("PDM", "numpy"), ("CE", "numpy")):
        method = get_method(name)
        assert not method.supports_power_rows(grid, backend)
        settings = method.coerce_settings(None)
        assert method.power_rows(grid, lc, times, settings, backend) is None


def test_a_declined_block_falls_back_to_the_loop(monkeypatch) -> None:
    # A method that batches the first block and declines the next: the scan must
    # finish the remaining rows with the per-row loop, giving the loop's map.
    t, mag, err, truth = synthetic_period_change(n=400, seed=9)
    ref = cup.pdot_search((t, mag, err), period=truth["period"], backend="finufft")
    method = get_method("GLS")
    calls = {"n": 0}

    def fake_rows(grid, lc, times, settings, backend, engine=None):
        calls["n"] += 1
        if calls["n"] > 1:
            return None
        return np.stack([
            method.power(grid, replace(lc, time=row), settings, backend, engine).power
            for row in times
        ])

    monkeypatch.setattr(scan_module, "_ROW_BLOCK_VALUES", 400 * 50)  # 50-row blocks
    monkeypatch.setattr(type(method), "supports_power_rows", lambda self, g, b: True)
    monkeypatch.setattr(method, "power_rows", fake_rows)
    got = cup.pdot_search((t, mag, err), period=truth["period"], backend="finufft")
    assert calls["n"] == 2
    np.testing.assert_allclose(got.power, ref.power, rtol=1e-12, atol=1e-14)


@requires_gpu
def test_gls_rows_equal_per_row_periodograms() -> None:
    lc, times, truth = _warped_rows()
    method = get_method("GLS")
    settings = cup.GLSSettings(fap_method="none")
    f_c = 1.0 / truth["period"]
    df = 1.0 / (5.0 * float(np.ptp(lc.time)))
    grid = cup.GridSpec(
        kind="frequency", values=f_c + df * np.arange(-300, 301), uniform=True
    )
    rows = method.power_rows(grid, lc, times, settings, "cufinufft")
    assert rows is not None and rows.shape == (times.shape[0], grid.size)
    for r, row_times in enumerate(times):
        loop = method.power(grid, replace(lc, time=row_times), settings, "finufft")
        np.testing.assert_allclose(rows[r], loop.power, rtol=1e-7, atol=1e-9)


@requires_gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_pdm_rows_equal_per_row_periodograms(precision: str) -> None:
    lc, times, truth = _warped_rows(n=700)
    method = get_method("PDM")
    settings = cup.PDMSettings(precision=precision)  # type: ignore[arg-type]
    f_c = 1.0 / truth["period"]
    df = 1.0 / (5.0 * float(np.ptp(lc.time)))
    grid = cup.GridSpec(
        kind="frequency", values=f_c + df * np.arange(-150, 151), uniform=True
    )
    rows = method.power_rows(grid, lc, times, settings, "cupy")
    assert rows is not None
    for r, row_times in enumerate(times):
        loop = method.power(grid, replace(lc, time=row_times), settings, "cupy")
        # The same blocks as a single-row launch; only atomic order may differ.
        np.testing.assert_allclose(rows[r], loop.power, rtol=1e-10, atol=1e-12)


@requires_gpu
@requires_numba
def test_gpu_pdm_scan_matches_the_cpu_scan() -> None:
    t, mag, err, truth = synthetic_period_change(n=800, seed=11)
    kwargs = {"period": truth["period"], "method": "PDM"}
    cpu = cup.pdot_search((t, mag, err), backend="numba", **kwargs)
    gpu = cup.pdot_search((t, mag, err), backend="cupy", **kwargs)
    assert gpu.backend == "cupy"
    np.testing.assert_allclose(gpu.power, cpu.power, rtol=1e-9, atol=1e-12)
    assert gpu.best.pdot_over_p == pytest.approx(cpu.best.pdot_over_p, abs=1e-15)


@requires_gpu
def test_gpu_scan_of_a_long_curve_spans_several_blocks(monkeypatch) -> None:
    # Blocks of a few rows exercise the chunked upload/download.
    monkeypatch.setattr(scan_module, "_ROW_BLOCK_VALUES", 1200 * 16)
    t, mag, err, truth = synthetic_period_change(n=1200, seed=42)
    cpu = cup.pdot_search((t, mag, err), period=truth["period"], backend="finufft")
    gpu = cup.pdot_search((t, mag, err), period=truth["period"], backend="cufinufft")
    np.testing.assert_allclose(gpu.power, cpu.power, rtol=1e-7, atol=1e-9)
