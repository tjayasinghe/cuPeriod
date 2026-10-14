"""String-length: parity with a direct reference, recovery, minimization sense."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import cuperiod as cup
import cuperiod.methods.string_length as sl_mod
from conftest import requires_gpu
from cuperiod.methods.string_length import string_length
from synth import synthetic_eclipser, synthetic_sine


def _fake_cupy(optin: int) -> types.SimpleNamespace:
    """A host-only stand-in for cupy reporting ``optin`` bytes of opt-in shared mem."""
    device = types.SimpleNamespace(attributes={"MaxSharedMemoryPerBlockOptin": optin})
    return types.SimpleNamespace(cuda=types.SimpleNamespace(Device=lambda: device))


def _reference_length(t: np.ndarray, y: np.ndarray, period: float) -> float:
    """Transparent Dworetsky string length for one period."""
    tau = t - t.min()
    span = y.max() - y.min()
    m = (y - y.min()) / span * 0.5 - 0.25 if span > 0 else np.zeros_like(y)
    phase = (tau / period) % 1.0
    order = np.argsort(phase)
    ph, mm = phase[order], m[order]
    total = float(np.sqrt(np.diff(ph) ** 2 + np.diff(mm) ** 2).sum())
    total += float(np.sqrt((ph[0] + 1 - ph[-1]) ** 2 + (mm[0] - mm[-1]) ** 2))
    return total


def test_string_length_matches_reference() -> None:
    t, mag, _ = synthetic_sine(n=300)
    periods = np.linspace(0.3, 3.0, 120)
    mine = string_length(t, mag, periods)
    ref = np.array([_reference_length(t, mag, float(p)) for p in periods])
    assert np.max(np.abs(mine - ref)) < 1e-9


def test_string_length_recovers_sine() -> None:
    # String length also minimizes at integer *multiples* of the true period (overlaid
    # copies pack points tightly), so — as in practice — the search is bounded to keep
    # the fundamental the global minimum.
    t, mag, err = synthetic_sine(period=0.6234)
    pg = cup.periodogram(
        (t, mag, err), "STRINGLENGTH",
        settings=cup.StringLengthSettings(minimum_frequency=1.0),
    )
    assert pg.objective_sense == "min"
    assert pg.best_period() == pytest.approx(0.6234, rel=2e-3)


def test_string_length_recovers_eclipser() -> None:
    t, flux, err = synthetic_eclipser(period=2.5)
    pg = cup.periodogram(
        (t, flux, err), "stringlength", domain="flux",
        settings=cup.StringLengthSettings(minimum_frequency=0.3),
    )
    assert pg.best_period() == pytest.approx(2.5, rel=5e-3)


def test_string_length_short_at_true_period() -> None:
    t, mag, _ = synthetic_sine(period=0.6234)
    at_true = string_length(t, mag, np.array([0.6234]))[0]
    off = string_length(t, mag, np.array([0.43]))[0]
    assert at_true < off


@requires_gpu
def test_string_length_gpu_matches_cpu() -> None:
    t, mag, _ = synthetic_sine()
    periods = np.linspace(0.3, 3.0, 500)
    cpu = string_length(t, mag, periods, backend="numpy")
    gpu = string_length(t, mag, periods, backend="cupy")
    assert float(np.max(np.abs(cpu - gpu))) < 1e-9


@pytest.mark.parametrize("optin_kb", [48, 64, 96, 99, 163, 227])
def test_sl_cuda_capacity_covers_the_padded_sort(
    optin_kb: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: the dispatch compared the raw point count with the shared-memory
    # capacity, but the bitonic sort pads to the next power of two, so every length
    # between the largest fitting power of two and the capacity (8193..8277 points
    # on a 99 KB device; 4097..5290 on 64 KB) raised instead of falling back.
    optin = optin_kb * 1024
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(optin))
    capacity = sl_mod._sl_cuda_capacity()
    n_pad = 1 << max(0, (capacity - 1).bit_length())  # what the kernel allocates
    static = sl_mod.CUDA_BLOCK * 8  # the kernel's reduction buffer
    assert n_pad * sl_mod._SL_BYTES_PER_POINT + static <= optin
    # ...and it is the longest such curve: one more point doubles the padding.
    assert 2 * n_pad * sl_mod._SL_BYTES_PER_POINT + static > optin


@requires_gpu
def test_string_length_gpu_at_the_kernel_capacity_boundary() -> None:
    # The longest curve the in-block sort takes, and one point more (the vectorized
    # fallback), both match the CPU.
    capacity = sl_mod._sl_cuda_capacity()
    periods = np.linspace(0.3, 3.0, 64)
    for n in (capacity, capacity + 1):
        t, mag, _ = synthetic_sine(n=n, seed=n % 7)
        cpu = string_length(t, mag, periods, backend="numpy")
        gpu = string_length(t, mag, periods, backend="cupy")
        assert float(np.max(np.abs(cpu - gpu))) < 1e-9, n
