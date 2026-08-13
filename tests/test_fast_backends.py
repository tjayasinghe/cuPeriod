"""Backend parity for the performance tiers: numba kernels, float32 CUDA, caches.

The per-method test modules pin each statistic against a transparent reference; this
module pins the *fast* backends against the vectorized numpy path — the numba CPU
kernels for all five ported methods, the float32-templated CUDA kernels, the shared
scatter shims, and the BLS per-light-curve device cache.
"""

from __future__ import annotations

import numpy as np

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
