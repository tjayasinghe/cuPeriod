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


def synthetic_period_change(
    *,
    n: int = 2000,
    span: float = 1500.0,
    period: float = 0.5,
    pdot: float = 1.0e-7,
    amp: float = 0.30,
    amp2: float = 0.10,
    noise: float = 0.05,
    base_jd: float = 2457000.0,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, float]]:
    """A two-harmonic sine whose period drifts linearly: ``(t, mag, err, truth)``.

    The phase is the exact constant-dP/dt model about the mean epoch:
    ``phi = f0 * tau`` with ``tau = ln(1 + a*dt)/a``, ``a = pdot/period`` (the
    logarithmic phase of Mikulasek et al. 2012, written out independently here
    so the tests do not share code with the implementation under test).
    Recovery tests compare against the planted ``truth`` dict (``period``,
    ``pdot``, ``pdot_over_p``, ``t_ref``) without re-deriving conventions.
    Positive ``pdot`` = lengthening period = late arrivals (upward O-C
    parabola).
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n)) + base_jd
    t_ref = float(t.mean())
    a = pdot / period
    dt = t - t_ref
    tau = np.log1p(a * dt) / a if pdot != 0.0 else dt
    f0 = 1.0 / period
    mag = (
        13.0
        + amp * np.sin(2 * np.pi * f0 * tau)
        + amp2 * np.sin(4 * np.pi * f0 * tau + 1.0)
        + rng.normal(0.0, noise, n)
    )
    err = np.full(n, noise)
    truth = {
        "period": period,
        "pdot": pdot,
        "pdot_over_p": a,
        "t_ref": t_ref,
    }
    return t, mag, err, truth


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


def _true_anomaly_reference(mean_anomaly: FloatArray, e: float) -> FloatArray:
    """Kepler's equation by bisection-free fixed-point iteration (test-side)."""
    m = np.mod(mean_anomaly, 2.0 * np.pi)
    ecc = m.copy()
    for _ in range(200):
        ecc = m + e * np.sin(ecc)
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(0.5 * ecc), np.sqrt(1.0 - e) * np.cos(0.5 * ecc)
    )


def ltte_delay_reference(
    dt: FloatArray,
    outer_period: float,
    amplitude: float,
    omega: float,
    eccentricity: float = 0.0,
    t_periastron: float = 0.0,
) -> FloatArray:
    """Irwin's (1952) light-time delay, written independently of the package."""
    m = 2.0 * np.pi * (dt - t_periastron) / outer_period
    nu = _true_anomaly_reference(m, eccentricity)
    e = eccentricity
    return amplitude * (
        (1.0 - e * e) / (1.0 + e * np.cos(nu)) * np.sin(nu + omega) + e * np.sin(omega)
    )


def synthetic_ltte(
    *,
    n: int = 2500,
    span: float = 3000.0,
    period: float = 0.5,
    outer_period: float = 800.0,
    amplitude: float = 0.004,
    omega: float = 0.7,
    eccentricity: float = 0.0,
    t_periastron_frac: float = 0.2,
    pdot: float = 0.0,
    amp: float = 0.30,
    amp2: float = 0.10,
    noise: float = 0.03,
    base_jd: float = 2455000.0,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, float]]:
    """A two-harmonic signal whose arrivals carry a light-travel-time delay.

    The binary sits in an outer orbit: what is observed at ``t`` was emitted
    at ``t - delay(t)`` with Irwin's (1952) delay from an independent Kepler
    solver, so the O-C is ``+delay`` (positive = late) plus the constant-dP/dt
    parabola when ``pdot`` is nonzero. ``truth`` carries ``outer_period``,
    ``amplitude`` (days), ``omega``, ``eccentricity``, ``t_periastron``,
    ``period``, ``pdot`` and ``t_ref``; compare *delay curves* rather than
    phase parameters, which are degenerate for a circular orbit.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n)) + base_jd
    t_ref = float(t.mean())
    dt = t - t_ref
    t_peri = t_ref + t_periastron_frac * outer_period
    delay = ltte_delay_reference(
        dt, outer_period, amplitude, omega, eccentricity, t_peri - t_ref
    )
    a = pdot / period
    retarded = dt - delay
    tau = np.log1p(a * retarded) / a if pdot != 0.0 else retarded
    f0 = 1.0 / period
    mag = (
        13.0
        + amp * np.sin(2 * np.pi * f0 * tau)
        + amp2 * np.sin(4 * np.pi * f0 * tau + 1.0)
        + rng.normal(0.0, noise, n)
    )
    err = np.full(n, noise)
    truth = {
        "period": period,
        "pdot": pdot,
        "outer_period": outer_period,
        "amplitude": amplitude,
        "omega": omega,
        "eccentricity": eccentricity,
        "t_periastron": t_peri,
        "t_ref": t_ref,
    }
    return t, mag, err, truth
