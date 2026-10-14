"""PDM: parity with a direct reference, period recovery, minimization sense."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu, requires_numba, requires_torch
from cuperiod.methods.pdm import pdm_theta
from synth import synthetic_eclipser, synthetic_sine


def _offset_curve(mean: float, *, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """A 0.1-amplitude sine with 0.02 scatter around ``mean`` (27 d, random cadence)."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 27.0, n))
    y = mean + 0.1 * np.sin(2 * np.pi * t / 0.6234) + rng.normal(0.0, 0.02, n)
    return t, y


def _reference_theta(
    t: np.ndarray, y: np.ndarray, period: float, n_bins: int, n_covers: int
) -> float:
    """Transparent, slow Stellingwerf Theta for one period (the algorithm of record)."""
    tau = t - t.min()
    n = t.size
    sigma2 = float(np.var(y, ddof=1))
    phase = (tau / period) % 1.0
    ssd = 0.0
    nonempty = 0
    for cover in range(n_covers):
        offset = cover / (n_bins * n_covers)
        b = np.clip((((phase + offset) % 1.0) * n_bins).astype(int), 0, n_bins - 1)
        for j in range(n_bins):
            yj = y[b == j]
            if yj.size > 0:
                ssd += float(((yj - yj.mean()) ** 2).sum())
                nonempty += 1
    den = n * n_covers - nonempty
    s2 = ssd / den if den > 0 else np.inf
    return s2 / sigma2


@pytest.mark.parametrize("n_covers", [1, 3])
def test_pdm_matches_reference(n_covers: int) -> None:
    t, mag, _ = synthetic_sine(n=400)
    periods = np.linspace(0.3, 3.0, 150)
    mine = pdm_theta(t, mag, periods, n_bins=10, n_covers=n_covers)
    ref = np.array(
        [_reference_theta(t, mag, float(p), 10, n_covers) for p in periods]
    )
    assert np.max(np.abs(mine - ref)) < 1e-9


def test_pdm_recovers_sine_period() -> None:
    t, mag, err = synthetic_sine(period=0.6234)
    pg = cup.periodogram((t, mag, err), "PDM")
    assert pg.objective_sense == "min"
    assert pg.best_period() == pytest.approx(0.6234, rel=2e-3)


def test_pdm_recovers_eclipser_period() -> None:
    t, flux, err = synthetic_eclipser(period=2.5)
    pg = cup.periodogram((t, flux, err), "PDM", domain="flux")
    assert pg.best_period() == pytest.approx(2.5, rel=5e-3)


def test_pdm_theta_drops_at_true_period() -> None:
    t, mag, _ = synthetic_sine(period=0.6234)
    at_true = pdm_theta(t, mag, np.array([0.6234]))[0]
    off = pdm_theta(t, mag, np.array([0.45]))[0]
    assert at_true < 0.5 < off  # dispersion collapses only at the true period


def test_pdm_constant_signal_is_flat() -> None:
    t = np.linspace(0.0, 100.0, 200)
    theta = pdm_theta(t, np.full(200, 5.0), np.linspace(0.5, 5.0, 50))
    assert np.allclose(theta, 1.0)


@requires_torch
@pytest.mark.parametrize("mean", [15.0, 1e3, 1e5])
def test_pdm_torch_float32_tracks_float64_at_any_mean(mean: float) -> None:
    # Regression: the array-API path binned raw y and y^2 in the working dtype and
    # formed sum(y^2) - sum(y)^2/n there; in float32 that cancellation swamped the
    # within-bin variance (negative Theta, wrong period) once mean/scatter neared
    # 1e3. The CUDA kernel keeps double accumulators; this path now centres instead.
    t, y = _offset_curve(mean)
    periods = np.linspace(0.4, 1.0, 300)
    ref = pdm_theta(t, y, periods, backend="numpy")
    f32 = pdm_theta(t, y, periods, backend="torch:cpu", precision="float32")
    assert np.all(f32 >= 0.0)
    assert float(np.max(np.abs(f32 - ref))) < 2e-3
    best32 = float(periods[int(np.argmin(f32))])
    assert best32 == pytest.approx(float(periods[int(np.argmin(ref))]), rel=5e-3)


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("numba", marks=requires_numba)]
)
def test_pdm_is_offset_invariant(backend: str) -> None:
    t, y = _offset_curve(0.0)
    periods = np.linspace(0.4, 1.0, 200)
    base = pdm_theta(t, y, periods, backend=backend)
    shifted = pdm_theta(t, y + 1e6, periods, backend=backend)
    assert float(np.max(np.abs(shifted - base))) < 1e-6


@requires_gpu
def test_pdm_gpu_matches_cpu() -> None:
    t, mag, _ = synthetic_sine()
    periods = np.linspace(0.3, 3.0, 500)
    cpu = pdm_theta(t, mag, periods, backend="numpy")
    gpu = pdm_theta(t, mag, periods, backend="cupy")  # one-block-per-period CUDA kernel
    # atomicAdd reorders the summation, so allow a small relative tolerance.
    assert np.allclose(cpu, gpu, rtol=1e-6, atol=1e-9)


@requires_gpu
def test_pdm_gpu_shared_memory_overflow_is_clean() -> None:
    # Regression: a bin count whose per-block shared memory exceeds any device's opt-in
    # limit must raise a clear BackendUnavailableError, not a raw CUDADriverError.
    t, mag, err = synthetic_sine(n=300)
    lc = cup.LightCurve.from_arrays(t, mag, err)
    with pytest.raises(cup.BackendUnavailableError, match="shared memory"):
        cup.periodogram(
            lc, "PDM", backend="gpu",
            settings=cup.PDMSettings(n_bins=10000, n_covers=3),
        )
