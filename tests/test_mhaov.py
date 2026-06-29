"""MHAOV: parity with a direct least-squares reference, recovery, max sense."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu
from cuperiod.methods.mhaov import aov_power
from synth import synthetic_eclipser, synthetic_sine


def _reference_aov(t: np.ndarray, y: np.ndarray, freq: float, h: int) -> float:
    """Transparent least-squares AOV F-statistic for one frequency."""
    n = t.size
    d = 2 * h + 1
    tau = t - t.min()
    ang = 2 * np.pi * freq * tau
    design = np.ones((n, d))
    for k in range(1, h + 1):
        design[:, 2 * k - 1] = np.cos(k * ang)
        design[:, 2 * k] = np.sin(k * ang)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    yhat = design @ beta
    ybar = y.mean()
    model = float(((yhat - ybar) ** 2).sum())
    resid = float(((y - yhat) ** 2).sum())
    return (model / (2 * h)) / (resid / (n - d))


def test_mhaov_matches_reference() -> None:
    t, mag, _ = synthetic_sine(n=300)
    freqs = np.linspace(0.2, 3.0, 120)
    mine = aov_power(t, mag, freqs, n_harmonics=3)
    ref = np.array([_reference_aov(t, mag, float(f), 3) for f in freqs])
    assert np.allclose(mine, ref, rtol=1e-5, atol=1e-4)


def test_mhaov_recovers_eclipser() -> None:
    # Sharp eclipses need several harmonics, which is exactly MHAOV's strength.
    t, flux, err = synthetic_eclipser(period=2.5)
    pg = cup.periodogram((t, flux, err), "MHAOV", domain="flux")
    assert pg.objective_sense == "max"
    assert pg.best_period() == pytest.approx(2.5, rel=5e-3)


def test_mhaov_detects_sine_in_bounded_range() -> None:
    # On a pure sinusoid the fundamental and its subharmonics share the same AOV, so
    # the search is bounded above the first subharmonic to isolate the fundamental.
    t, mag, err = synthetic_sine(period=0.6234)
    pg = cup.periodogram(
        (t, mag, err), "MHAOV", settings=cup.MHAOVSettings(minimum_frequency=1.0)
    )
    assert pg.best_period() == pytest.approx(0.6234, rel=2e-3)


def test_mhaov_peaks_at_true_frequency() -> None:
    t, flux, _ = synthetic_eclipser(period=2.5)
    at_true = aov_power(t, flux, np.array([1.0 / 2.5]))[0]
    off = aov_power(t, flux, np.array([1.0 / 1.7]))[0]
    assert at_true > 5.0 * off


def test_mhaov_too_few_points() -> None:
    t = np.linspace(0.0, 10.0, 5)
    out = aov_power(t, np.arange(5.0), np.array([0.5, 1.0]), n_harmonics=3)
    assert np.all(out == 0.0)


@requires_gpu
def test_mhaov_gpu_matches_cpu() -> None:
    t, mag, _ = synthetic_sine()
    freqs = np.linspace(0.2, 3.0, 400)
    cpu = aov_power(t, mag, freqs, backend="numpy")
    gpu = aov_power(t, mag, freqs, backend="cupy")
    assert np.allclose(cpu, gpu, rtol=1e-6, atol=1e-6)
