"""Conditional entropy: parity with a direct reference, recovery, min sense."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu
from cuperiod.methods.conditional_entropy import conditional_entropy
from synth import synthetic_eclipser, synthetic_sine


def _reference_ce(
    t: np.ndarray, y: np.ndarray, period: float, n_phase: int, n_mag: int
) -> float:
    """Transparent Graham-2013 conditional entropy for one period."""
    tau = t - t.min()
    span = y.max() - y.min()
    mag_bin = np.clip(((y - y.min()) / span * n_mag).astype(int), 0, n_mag - 1)
    phase = (tau / period) % 1.0
    pb = np.clip((phase * n_phase).astype(int), 0, n_phase - 1)
    count = np.zeros((n_phase, n_mag))
    for i in range(t.size):
        count[pb[i], mag_bin[i]] += 1.0
    n = t.size
    h = 0.0
    for i in range(n_phase):
        ci = count[i].sum()
        for j in range(n_mag):
            cij = count[i, j]
            if cij > 0:
                h += cij * (np.log(ci) - np.log(cij))
    return h / n


def test_ce_matches_reference() -> None:
    t, mag, _ = synthetic_sine(n=300)
    periods = np.linspace(0.3, 3.0, 120)
    mine = conditional_entropy(t, mag, periods, n_phase_bins=10, n_mag_bins=10)
    ref = np.array([_reference_ce(t, mag, float(p), 10, 10) for p in periods])
    assert np.max(np.abs(mine - ref)) < 1e-9


def test_ce_recovers_sine() -> None:
    t, mag, err = synthetic_sine(period=0.6234)
    pg = cup.periodogram((t, mag, err), "CE")
    assert pg.objective_sense == "min"
    assert pg.best_period() == pytest.approx(0.6234, rel=2e-3)


def test_ce_recovers_eclipser() -> None:
    t, flux, err = synthetic_eclipser(period=2.5)
    pg = cup.periodogram((t, flux, err), "ce", domain="flux")
    assert pg.best_period() == pytest.approx(2.5, rel=5e-3)


def test_ce_low_at_true_period() -> None:
    t, mag, _ = synthetic_sine(period=0.6234)
    at_true = conditional_entropy(t, mag, np.array([0.6234]))[0]
    off = conditional_entropy(t, mag, np.array([0.43]))[0]
    assert at_true < off


@requires_gpu
def test_ce_gpu_matches_cpu() -> None:
    t, mag, _ = synthetic_sine()
    periods = np.linspace(0.3, 3.0, 500)
    cpu = conditional_entropy(t, mag, periods, backend="numpy")
    gpu = conditional_entropy(t, mag, periods, backend="cupy")  # CUDA kernel
    # atomicAdd reorders the histogram summation, so allow a small tolerance.
    assert np.allclose(cpu, gpu, rtol=1e-6, atol=1e-9)
