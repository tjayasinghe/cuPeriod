"""Synthetic light-curve factories with planted signals for recovery tests."""

from __future__ import annotations

import numpy as np

from cuperiod.core._typing import FloatArray


def synthetic_sine(
    *,
    n: int = 600,
    span: float = 320.0,
    period: float = 0.6234,
    amp: float = 0.4,
    noise: float = 0.02,
    base_jd: float = 2458000.0,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Irregularly-sampled single-sine light curve on real-scale JDs (magnitudes)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n)) + base_jd
    mag = 12.0 + amp * np.sin(2 * np.pi * t / period) + rng.normal(0.0, noise, n)
    err = np.full(n, noise)
    return t, mag, err


def synthetic_eclipser(
    *,
    n: int = 800,
    span: float = 200.0,
    period: float = 2.5,
    depth: float = 0.05,
    duration_frac: float = 0.04,
    epoch: float = 2458000.3,
    noise: float = 0.005,
    base_jd: float = 2458000.0,
    seed: int = 1,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Flat flux light curve with periodic box dips (a transit/eclipse)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n)) + base_jd
    phase = ((t - epoch) % period) / period
    flux = np.ones(n)
    in_transit = (phase < duration_frac) | (phase > 1.0 - duration_frac)
    flux[in_transit] -= depth
    flux += rng.normal(0.0, noise, n)
    err = np.full(n, noise)
    return t, flux, err
