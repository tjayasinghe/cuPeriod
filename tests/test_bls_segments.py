"""The segmented BLS search with one upload and one download (PERF-METHODS-12).

``bls_power_segments`` searches every log period segment of a light curve; on cupy
the segments share one upload of their period grids and one download of every
result, where each segment used to wait on its own transfers. On every backend it
must return exactly what :func:`bls_power` returns segment by segment.
"""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu, requires_numba
from cuperiod.methods._bls_core import bls_power, bls_power_segments
from cuperiod.methods.bls import _grid_segments

_FIELDS = ("period", "power", "depth", "depth_err", "depth_snr", "duration",
           "transit_time", "log_likelihood")


def _transit_curve(n: int = 800, seed: int = 2):
    rng = np.random.default_rng(seed)
    t = 2458000.0 + np.sort(rng.uniform(0.0, 60.0, n))
    flux = 1.0 + rng.normal(0.0, 0.002, n)
    phase = np.mod(t - 2458001.3, 3.7) / 3.7
    flux[phase < 0.03] -= 0.01
    return t, flux, np.full(n, 0.002)


def _segments(t, flux, err):
    lc = cup.LightCurve.from_arrays(t, flux, err, domain="flux")
    settings = cup.BLSSettings()
    grid = cup.get_method("BLS").default_grid(lc, settings)
    segments = _grid_segments(grid, settings)
    # An empty segment in the middle must come back empty, in its place.
    empty = (np.zeros(0), segments[0][1])
    return [segments[0], empty, *segments[1:]], settings


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("numba", marks=requires_numba)]
)
def test_segments_equal_per_segment_calls_on_the_cpu(backend: str) -> None:
    t, flux, err = _transit_curve()
    segments, settings = _segments(t, flux, err)
    got = bls_power_segments(
        t, flux, err, segments, settings.bins_per_duration, backend=backend
    )
    assert len(got) == len(segments)
    for (periods, durations), power in zip(segments, got, strict=True):
        if periods.size == 0:
            assert power.power.size == 0
            continue
        ref = bls_power(
            t, flux, err, periods, durations, settings.bins_per_duration,
            backend=backend,
        )
        for name in _FIELDS:
            np.testing.assert_array_equal(getattr(power, name), getattr(ref, name))


@requires_gpu
@pytest.mark.parametrize("objective", ["snr", "likelihood"])
def test_cuda_segments_equal_per_segment_launches(objective) -> None:
    t, flux, err = _transit_curve(seed=5)
    segments, settings = _segments(t, flux, err)
    got = bls_power_segments(
        t, flux, err, segments, settings.bins_per_duration, backend="cupy",
        objective=objective, precision="float64",
    )
    for (periods, durations), power in zip(segments, got, strict=True):
        if periods.size == 0:
            assert power.power.size == 0
            continue
        ref = bls_power(
            t, flux, err, periods, durations, settings.bins_per_duration,
            backend="cupy", objective=objective, precision="float64",
        )
        # Same kernel and inputs; only shared-memory atomic order may differ.
        for name in _FIELDS:
            np.testing.assert_allclose(
                getattr(power, name), getattr(ref, name), rtol=1e-9, atol=1e-12,
                err_msg=name,
            )


@requires_gpu
def test_cuda_segments_in_float32() -> None:
    # float32 binning is order-sensitive at ~1e-6 (the atomics' order varies from
    # launch to launch) and flips rare near-tie boxes, as for any float32 BLS run.
    t, flux, err = _transit_curve(seed=5)
    segments, settings = _segments(t, flux, err)
    got = bls_power_segments(
        t, flux, err, segments, settings.bins_per_duration, backend="cupy",
        precision="float32",
    )
    ref = [
        bls_power(t, flux, err, p, d, settings.bins_per_duration, backend="numpy")
        for p, d in segments
    ]
    power = np.concatenate([g.power for g in got])
    reference = np.concatenate([r.power for r in ref])
    close = np.isclose(power, reference, rtol=1e-4, atol=1e-6)
    assert close.mean() > 0.99
    assert np.argmax(power) == np.argmax(reference)


@requires_gpu
@pytest.mark.parametrize("n", [45, 300])
def test_cuda_breaks_exact_box_ties_like_the_host(n: int) -> None:
    # Boxes that differ only by empty bins hold the same points and tie exactly; a
    # sparse curve has them at most periods. The kernel used to keep the tied box of
    # the lowest thread, so the GPU reported another (equally good) mid-transit time
    # and duration than numpy/astropy's first-wins box at ~70% of the periods.
    rng = np.random.default_rng(n)
    t = 2458000.0 + np.sort(rng.uniform(0.0, 90.0, n))
    flux = 1.0 + rng.normal(0.0, 0.005, n)
    flux[np.mod(t - 2458000.3, 2.5) < 0.1] -= 0.01
    err = np.full(n, 0.005)
    periods = np.linspace(1.0, 6.0, 3000)
    durations = np.geomspace(0.02, 0.4, 6)
    cpu = bls_power(t, flux, err, periods, durations, 5, backend="numpy")
    gpu = bls_power(t, flux, err, periods, durations, 5, backend="cupy")
    np.testing.assert_allclose(gpu.power, cpu.power, rtol=1e-12, atol=1e-12)
    same_t0 = np.isclose(gpu.transit_time, cpu.transit_time, rtol=0.0, atol=1e-9)
    same_duration = gpu.duration == cpu.duration
    # Only round-off near-ties may still differ.
    assert same_t0.mean() > 0.995
    assert same_duration.mean() > 0.995


@requires_gpu
def test_gpu_bls_periodogram_matches_the_cpu_reference() -> None:
    t, flux, err = _transit_curve(n=1500, seed=8)
    data = cup.LightCurve.from_arrays(t, flux, err, domain="flux")
    gpu = cup.periodogram(data, "BLS", backend="cupy")
    ref = cup.periodogram(data, "BLS", backend="numpy")
    np.testing.assert_allclose(gpu.power, ref.power, rtol=1e-9, atol=1e-9)
    assert gpu.best_period() == pytest.approx(ref.best_period(), rel=1e-12)
