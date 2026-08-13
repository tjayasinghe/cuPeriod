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


def synthetic_pulsator(
    *,
    n: int = 1500,
    span: float = 27.0,
    frequencies: tuple[float, ...] = (12.34, 17.81, 24.68),
    amplitudes: tuple[float, ...] = (0.012, 0.007, 0.004),
    phases: tuple[float, ...] = (0.5, 2.0, 1.0),
    noise: float = 0.0015,
    base_jd: float = 2458000.0,
    seed: int = 4,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """A multiperiodic pulsator: several coherent sinusoids plus white noise.

    The default is a δ Scuti-like triple whose third frequency is deliberately the exact
    second harmonic of the first, so combination-frequency identification has something
    real to find.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n)) + base_jd
    mag = np.full(n, 10.0)
    for freq, amp, phase in zip(frequencies, amplitudes, phases, strict=True):
        mag += amp * np.sin(2 * np.pi * freq * (t - t.min()) + phase)
    err = np.full(n, noise)
    mag = mag + rng.normal(0.0, noise, n)
    return t, mag, err


def synthetic_gmode(
    *,
    n: int = 2500,
    span: float = 90.0,
    n_modes: int = 14,
    first_period: float = 0.55,
    spacing: float = 0.028,
    spacing_slope: float = 0.008,
    amplitude: float = 0.006,
    noise: float = 0.0006,
    base_jd: float = 2458000.0,
    seed: int = 9,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """A γ Dor-like g-mode series with a rotationally tilted period spacing.

    Consecutive periods follow ``dP(P) = spacing + spacing_slope * P``. Returns
    ``(time, mag, err, periods)`` so a test can compare against the planted comb.
    """
    rng = np.random.default_rng(seed)
    periods = [first_period]
    for _ in range(n_modes - 1):
        periods.append(periods[-1] + spacing + spacing_slope * periods[-1])
    period_array = np.asarray(periods, dtype=np.float64)
    t = np.sort(rng.uniform(0.0, span, n)) + base_jd
    mag = np.full(n, 9.0)
    for index, period in enumerate(period_array):
        amp = amplitude * (1.0 - 0.04 * index)
        mag += amp * np.sin(
            2 * np.pi * (t - t.min()) / period + rng.uniform(0.0, 2 * np.pi)
        )
    err = np.full(n, noise)
    mag = mag + rng.normal(0.0, noise, n)
    return t, mag, err, period_array


def synthetic_multiband_sine(
    *,
    band_points: tuple[int, ...] = (60, 45, 30),
    period: float = 0.7365,
    amplitudes: tuple[float, ...] = (0.30, 0.22, 0.15),
    offsets: tuple[float, ...] = (15.0, 14.2, 13.9),
    phase: float = 0.3,
    span: float = 180.0,
    noise: float = 0.05,
    base_jd: float = 2458000.0,
    seed: int = 0,
) -> dict[str, tuple[FloatArray, FloatArray, FloatArray]]:
    """Several bands of one shared-phase sine: per-band ``(t, mag, err)`` arrays.

    Bands share the period and phase but differ in amplitude, mean magnitude,
    sampling, and per-point noise — the sparse multi-band regime the joint models
    are for. Band labels count up from ``"b0"``.
    """
    out: dict[str, tuple[FloatArray, FloatArray, FloatArray]] = {}
    for index, (n, amp, offset) in enumerate(
        zip(band_points, amplitudes, offsets, strict=True)
    ):
        rng = np.random.default_rng(seed + index)
        t = np.sort(rng.uniform(0.0, span, n)) + base_jd
        err = noise * (0.8 + 0.4 * rng.random(n))
        mag = offset + amp * np.sin(2 * np.pi * t / period + phase)
        mag = mag + rng.normal(0.0, err)
        out[f"b{index}"] = (t, mag, err)
    return out


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
