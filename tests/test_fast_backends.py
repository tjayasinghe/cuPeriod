"""Backend parity for the performance tiers: numba kernels, float32 CUDA, caches.

The per-method test modules pin each statistic against a transparent reference; this
module pins the *fast* backends against the vectorized numpy path — the numba CPU
kernels for all five ported methods, the float32-templated CUDA kernels, the shared
scatter shims, and the BLS per-light-curve device cache.
"""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu, requires_numba, requires_torch
from cuperiod.core._arrayapi import scatter_add_rows, scatter_counts_rows
from cuperiod.methods._bls_core import bls_power
from cuperiod.methods.conditional_entropy import conditional_entropy
from cuperiod.methods.mhaov import aov_power
from cuperiod.methods.pdm import pdm_theta
from cuperiod.methods.string_length import string_length
from cuperiod.methods.tls import tls_power
from synth import synthetic_eclipser, synthetic_sine

PERIODS = np.linspace(0.3, 3.0, 400)
FREQS = np.linspace(0.4, 3.0, 400)


def _tied_sine(n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """A sine curve with duplicated times, so phase folds tie exactly."""
    t, mag, _ = synthetic_sine(n=n)
    t[50:60] = t[49]
    return t, mag


# --- numba CPU kernels vs the vectorized numpy path ---------------------------


@requires_numba
def test_numba_pdm_matches_numpy() -> None:
    t, mag, _ = synthetic_sine(n=300)
    cpu = pdm_theta(t, mag, PERIODS, backend="numpy")
    fast = pdm_theta(t, mag, PERIODS, backend="numba")
    assert np.max(np.abs(cpu - fast)) < 1e-12


@requires_numba
def test_numba_ce_matches_numpy() -> None:
    t, mag, _ = synthetic_sine(n=300)
    cpu = conditional_entropy(t, mag, PERIODS, backend="numpy")
    fast = conditional_entropy(t, mag, PERIODS, backend="numba")
    assert np.max(np.abs(cpu - fast)) < 1e-12


@requires_numba
def test_numba_string_length_matches_numpy_with_ties() -> None:
    # Duplicated times force equal phases: the numba mergesort must break ties in
    # the same (stable) order as the array-API paths' stable argsort.
    t, mag = _tied_sine()
    cpu = string_length(t, mag, PERIODS, backend="numpy")
    fast = string_length(t, mag, PERIODS, backend="numba")
    assert np.max(np.abs(cpu - fast)) < 1e-9


def _stable_string_length(tau: np.ndarray, m: np.ndarray, period: float) -> float:
    """The numba kernel's string length, written out with numpy's stable argsort.

    Same fold and the same left-to-right sum, so it is bit-identical to the kernel
    exactly when the kernel visits the points in the stable order.
    """
    q = tau / period
    phase = q - np.floor(q)
    order = np.argsort(phase, kind="stable")
    ph, mm = phase[order], m[order]
    total = 0.0
    for j in range(1, ph.size):
        dphi, dmag = ph[j] - ph[j - 1], mm[j] - mm[j - 1]
        total += np.sqrt(dphi * dphi + dmag * dmag)
    wrap_phi, wrap_mag = (ph[0] + 1.0) - ph[-1], mm[0] - mm[-1]
    return float(total + np.sqrt(wrap_phi * wrap_phi + wrap_mag * wrap_mag))


def _sl_cases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(11)
    t, mag = _tied_sine()
    nights = (np.arange(3.0)[:, None] + rng.uniform(0.0, 0.3, (3, 400))).ravel()
    lattice = np.repeat(np.arange(120.0), 3)  # whole days, three points per epoch
    return {
        "ties": (t, mag),
        "three dense nights": (np.sort(nights), rng.normal(size=nights.size)),
        "integer-day lattice": (lattice, rng.normal(size=lattice.size)),
    }


@requires_numba
@pytest.mark.parametrize("case", ["ties", "three dense nights", "integer-day lattice"])
def test_numba_string_length_bucket_sort_is_the_stable_order(case: str) -> None:
    # PERF-METHODS-05: the kernel sorts each fold with an O(N) bucket + insertion
    # sort (falling back to mergesort on clumped folds). The permutation must be the
    # stable one, or tied phases would pair different neighbours: bit-identical to
    # a stable-argsort reference on ties, clumped folds, commensurate periods (every
    # point at phase 0) and periods longer than the baseline.
    from cuperiod.methods.string_length import _numba_sl_kernel

    t, y = _sl_cases()[case]
    tau = t - t.min()
    span = float(y.max() - y.min())
    m = (y - y.min()) / span * 0.5 - 0.25
    baseline = float(tau.max())
    periods = np.concatenate([
        np.linspace(0.3, 3.0, 40), [0.5, 1.0, 2.0, 1.5],
        baseline * np.array([1.0, 1.7, 5.0]),
    ])
    got = _numba_sl_kernel()(tau, m, periods)
    want = np.array([_stable_string_length(tau, m, float(p)) for p in periods])
    assert np.array_equal(got, want)
    # ...and the public entry point agrees with the vectorized numpy path.
    cpu = string_length(t, y, periods, backend="numpy")
    fast = string_length(t, y, periods, backend="numba")
    assert np.max(np.abs(cpu - fast)) < 1e-11


@requires_numba
def test_numba_string_length_kernel_survives_unfoldable_periods() -> None:
    # The public entry point never passes these, but the kernel itself must not
    # bucket a NaN phase (an out-of-range bucket index would write out of bounds).
    from cuperiod.methods.string_length import _numba_sl_kernel

    t, mag, _ = synthetic_sine(n=200)
    tau = t - t.min()
    m = (mag - mag.min()) / float(np.ptp(mag)) * 0.5 - 0.25
    out = _numba_sl_kernel()(tau, m, np.array([0.5, np.nan, 0.0, 2.0]))
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert np.isfinite(out[0]) and np.isfinite(out[3])


@requires_numba
def test_numba_mhaov_matches_numpy() -> None:
    t, mag, _ = synthetic_sine(n=300)
    cpu = aov_power(t, mag, FREQS, n_harmonics=3, backend="numpy")
    fast = aov_power(t, mag, FREQS, n_harmonics=3, backend="numba")
    assert np.allclose(cpu, fast, rtol=1e-8, atol=1e-8)


@requires_numba
def test_numba_tls_matches_numpy() -> None:
    t, flux, err = synthetic_eclipser(period=2.5)
    periods = np.linspace(1.5, 4.0, 300)
    settings = cup.TLSSettings()
    cpu = tls_power(t, flux, err, periods, settings=settings, backend="numpy")
    fast = tls_power(t, flux, err, periods, settings=settings, backend="numba")
    for key in ("sde", "sr", "depth", "duration", "t0"):
        assert np.allclose(cpu[key], fast[key], rtol=1e-10, atol=1e-10), key


@requires_numba
def test_cpu_request_resolves_to_numba() -> None:
    for name in ("PDM", "CE", "STRINGLENGTH", "MHAOV", "TLS", "BLS",
                 "SUPERSMOOTHER"):
        assert cup.get_method(name).resolve_backend("cpu") == "numba"


# --- float32 CUDA kernels ------------------------------------------------------


@requires_gpu
def test_bls_cupy_float32_close_to_float64() -> None:
    t, flux, err = synthetic_eclipser(period=2.5)
    periods = np.linspace(1.5, 4.0, 500)
    durations = np.asarray([0.05, 0.1])
    p64 = bls_power(t, flux, err, periods, durations, 10, backend="cupy",
                    precision="float64")
    p32 = bls_power(t, flux, err, periods, durations, 10, backend="cupy",
                    precision="float32")
    # Detection-grade agreement: float32 rounding may flip which of two near-tied
    # boxes wins at isolated periods, so allow a small fraction of outliers but
    # require the same detected period.
    close = np.isclose(p64.power, p32.power, rtol=1e-3, atol=1e-3)
    assert close.mean() > 0.99
    assert int(np.argmax(p64.power)) == int(np.argmax(p32.power))


@requires_gpu
def test_pdm_ce_tls_cupy_float32_close_to_float64() -> None:
    t, mag, _ = synthetic_sine(n=400)
    theta64 = pdm_theta(t, mag, PERIODS, backend="cupy", precision="float64")
    theta32 = pdm_theta(t, mag, PERIODS, backend="cupy", precision="float32")
    assert np.allclose(theta64, theta32, rtol=5e-3, atol=5e-3)

    h64 = conditional_entropy(t, mag, PERIODS, backend="cupy", precision="float64")
    h32 = conditional_entropy(t, mag, PERIODS, backend="cupy", precision="float32")
    assert np.allclose(h64, h32, rtol=5e-3, atol=5e-3)

    tt, flux, err = synthetic_eclipser(period=2.5)
    periods = np.linspace(1.5, 4.0, 300)
    r64 = tls_power(tt, flux, err, periods,
                    settings=cup.TLSSettings(precision="float64"), backend="cupy")
    r32 = tls_power(tt, flux, err, periods,
                    settings=cup.TLSSettings(precision="float32"), backend="cupy")
    assert int(np.argmax(r64["sr"])) == int(np.argmax(r32["sr"]))


@requires_gpu
def test_string_length_gpu_kernel_handles_ties() -> None:
    # The in-block bitonic sort breaks equal phases by original index — the same
    # stable order as the CPU paths.
    t, mag = _tied_sine()
    cpu = string_length(t, mag, PERIODS, backend="numpy")
    gpu = string_length(t, mag, PERIODS, backend="cupy")
    assert np.max(np.abs(cpu - gpu)) < 1e-9


# --- scatter shims ---------------------------------------------------------------


def test_scatter_add_rows_numpy_matches_manual() -> None:
    rng = np.random.default_rng(0)
    index = rng.integers(0, 7, size=(4, 50))
    values = rng.normal(size=50)
    target = np.zeros((4, 7))
    scatter_add_rows(target, index, values)
    expected = np.zeros((4, 7))
    for p in range(4):
        np.add.at(expected[p], index[p], values)
    assert np.allclose(target, expected)

    counts = np.zeros((4, 7))
    scatter_counts_rows(counts, index)
    expected_counts = np.zeros((4, 7))
    for p in range(4):
        np.add.at(expected_counts[p], index[p], 1.0)
    assert np.allclose(counts, expected_counts)


@requires_torch
def test_scatter_add_rows_torch_matches_numpy() -> None:
    import torch

    rng = np.random.default_rng(1)
    index = rng.integers(0, 9, size=(3, 40))
    values = rng.normal(size=40)
    np_target = np.zeros((3, 9))
    scatter_add_rows(np_target, index, values)
    t_target = torch.zeros((3, 9), dtype=torch.float64)
    scatter_add_rows(t_target, torch.as_tensor(index), torch.as_tensor(values))
    assert np.allclose(np_target, t_target.numpy())

    np_counts = np.zeros((3, 9))
    scatter_counts_rows(np_counts, index)
    t_counts = torch.zeros((3, 9), dtype=torch.float64)
    scatter_counts_rows(t_counts, torch.as_tensor(index))
    assert np.allclose(np_counts, t_counts.numpy())


# --- BLS device cache -------------------------------------------------------------


@requires_torch
def test_bls_device_cache_reuse_matches_fresh() -> None:
    t, flux, err = synthetic_eclipser(period=2.5)
    seg1 = np.linspace(1.5, 2.4, 200)
    seg2 = np.linspace(2.4, 4.0, 200)
    durations = np.asarray([0.05, 0.1])
    cache: dict[str, object] = {}
    with_cache = [
        bls_power(t, flux, err, seg, durations, 10, backend="torch:cpu",
                  device_cache=cache)
        for seg in (seg1, seg2)
    ]
    fresh = [
        bls_power(t, flux, err, seg, durations, 10, backend="torch:cpu")
        for seg in (seg1, seg2)
    ]
    assert "torch" in cache  # the upload happened once and was kept
    for got, want in zip(with_cache, fresh, strict=True):
        assert np.allclose(got.power, want.power)
        assert np.allclose(got.transit_time, want.transit_time)
