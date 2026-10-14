"""The cufinufft amplitude scan forms the amplitude on the device (PERF-PREWHITEN-05).

The extraction loop's scan solves the normal equations, applies the ill-conditioned
fallback and the non-finite guard on the GPU and brings back only the amplitude and
the data sums. These tests pin it to the host path: the same amplitude to the NUFFT
tolerance, the same lazily formed phase/power, the same conditioning decisions, and a
scan that holds host memory only.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from cuperiod.core.grid import uniform_frequency_grid
from cuperiod.prewhiten.spectrum import SpectrumEngine


def _curve(n: int = 600, span: float = 40.0, seed: int = 4):
    rng = np.random.default_rng(seed)
    time = 2459000.0 + np.sort(rng.uniform(0.0, span, n))
    error = rng.uniform(0.01, 0.05, n)
    value = (
        0.3 * np.sin(2 * np.pi * 1.7 * time + 0.9)
        + 0.1 * np.sin(2 * np.pi * 4.31 * time)
        + rng.normal(0.0, error)
    )
    return time, value, error


def _regular_noise(seed: int = 0):
    """White noise on an exact 30-minute cadence: in-band ill-conditioned zones."""
    time = 2459000.0 + np.arange(1296) / 48.0
    rng = np.random.default_rng(seed)
    return time, 10.0 + rng.normal(0.0, 0.001, time.size), np.full(time.size, 0.001)


def _grid(time, maximum: float, samples: int = 5):
    return uniform_frequency_grid(
        float(time.max() - time.min()),
        maximum_frequency=maximum,
        samples_per_peak=samples,
    )


@requires_gpu
@pytest.mark.parametrize("normalization", ["lsq", "dft"])
def test_device_scan_matches_the_host_scan(normalization: str) -> None:
    time, value, error = _curve()
    grid = _grid(time, maximum=12.0)
    gpu = SpectrumEngine(
        time, error, grid=grid, backend="cufinufft", normalization=normalization
    )
    cpu = SpectrumEngine(
        time, error, grid=grid, backend="finufft", normalization=normalization
    )
    g, c = gpu._scan(value), cpu._scan(value)
    assert isinstance(g.amplitude, np.ndarray)
    assert np.max(np.abs(g.amplitude - c.amplitude)) < 1e-9
    index = c.peak_index()
    assert g.peak_index() == index
    assert index is not None
    assert g.power_at(index) == pytest.approx(c.power_at(index), abs=1e-9)
    full_g, full_c = g.to_spectrum(), c.to_spectrum()
    for name in ("amplitude", "power"):
        assert np.max(np.abs(getattr(full_g, name) - getattr(full_c, name))) < 1e-9
    big = full_c.amplitude > 1e-3 * full_c.amplitude.max()
    wrapped = np.angle(np.exp(1j * (full_g.phase - full_c.phase)))[big]
    assert np.max(np.abs(wrapped)) < 1e-6


@requires_gpu
def test_device_scan_is_consistent_with_its_own_full_spectrum() -> None:
    # The lean scan and the spectrum built from it must agree exactly: the loop
    # compares a scan's power_at with the kept spectrum's power.
    time, value, error = _curve(seed=7)
    engine = SpectrumEngine(
        time, error, grid=_grid(time, maximum=9.0), backend="cufinufft"
    )
    scan = engine._scan(value)
    full = scan.to_spectrum()
    assert np.array_equal(scan.amplitude, full.amplitude)
    assert all(scan.power_at(i) == full.power[i] for i in range(0, full.size, 13))
    public = engine.spectrum(value)
    assert np.allclose(public.amplitude, full.amplitude, rtol=0.0, atol=1e-12)


@requires_gpu
def test_device_scan_applies_the_ill_conditioned_fallback() -> None:
    time, value, error = _regular_noise(seed=2)
    grid = _grid(time, maximum=50.0, samples=10)
    gpu = SpectrumEngine(time, error, grid=grid, backend="cufinufft")
    cpu = SpectrumEngine(time, error, grid=grid, backend="finufft")
    terms = cpu._normal_terms()
    assert terms.ill.any()
    g, c = gpu._scan(value), cpu._scan(value)
    scale = float(np.max(c.amplitude))
    assert np.max(np.abs(g.amplitude - c.amplitude)) < 1e-6 * scale
    # Same conditioning decisions: the device terms are the host terms, uploaded.
    assert np.array_equal(gpu._normal_terms().ill, terms.ill)
    np.testing.assert_array_equal(g._bad, c._bad)


@requires_gpu
def test_device_gram_terms_are_the_host_arithmetic_bit_for_bit() -> None:
    # The window terms are formed on the device; from the same transform output the
    # host expression must give the same bits (IEEE steps only). One transform pair
    # feeds both: cufinufft's spreading is not bitwise reproducible across calls.
    import cupy

    time, _, error = _regular_noise(seed=5)
    engine = SpectrumEngine(
        time, error, grid=_grid(time, maximum=50.0, samples=10), backend="cufinufft"
    )
    w = engine._weight
    base = engine._plan_transform(w)
    doubled = engine._plan_transform(w, doubled=True)
    on_device = engine._gram_terms(
        cupy, base.real, base.imag, doubled.real, doubled.imag
    )
    b, d = base.get(), doubled.get()
    with np.errstate(divide="ignore", invalid="ignore"):
        on_host = engine._gram_terms(np, b.real, b.imag, d.real, d.imag)
    for name, dev, host in zip(
        ("cc", "ss", "cs", "inv", "ill"), on_device, on_host, strict=True
    ):
        np.testing.assert_array_equal(dev.get(), host, err_msg=name)
    assert on_host[4].any()  # the regular cadence puts ill-conditioned zones in band
    # The scans' pre-scaled terms are those products.
    terms = engine._normal_terms()
    ssi, _, _, index = engine._device_terms()
    np.testing.assert_array_equal(ssi.get(), terms.ss * terms.inv)
    np.testing.assert_array_equal(index.get(), terms.ill_index)


@requires_gpu
def test_device_window_matches_the_host_window() -> None:
    time, _, error = _curve(seed=9)
    grid = _grid(time, maximum=8.0)
    gpu = SpectrumEngine(time, error, grid=grid, backend="cufinufft").window()
    cpu = SpectrumEngine(time, error, grid=grid, backend="finufft").window()
    assert np.max(np.abs(gpu.amplitude - cpu.amplitude)) < 1e-9
    assert np.max(np.abs(gpu.power - cpu.power)) < 1e-9
    big = cpu.amplitude > 1e-3
    wrapped = np.angle(np.exp(1j * (gpu.phase - cpu.phase)))[big]
    assert np.max(np.abs(wrapped)) < 1e-6


@requires_gpu
def test_full_spectrum_on_the_device_equals_the_host_fallback() -> None:
    time, value, error = _curve(seed=13)
    engine = SpectrumEngine(
        time, error, grid=_grid(time, maximum=9.0), backend="cufinufft"
    )
    scan = engine._scan(value)
    on_device = scan.to_spectrum()  # engine live: formed on the device
    engine.release()
    on_host = scan.to_spectrum()  # released: the host computes the same thing
    np.testing.assert_array_equal(on_device.power, on_host.power)
    wrapped = np.angle(np.exp(1j * (on_device.phase - on_host.phase)))
    assert np.max(np.abs(wrapped)) < 1e-12


@requires_gpu
def test_device_scan_keeps_no_device_memory_and_survives_release() -> None:
    time, value, error = _curve(seed=11)
    engine = SpectrumEngine(
        time, error, grid=_grid(time, maximum=6.0), backend="cufinufft"
    )
    first = engine._scan(value)
    for name in ("amplitude", "_yc", "_ys", "_bad"):
        assert isinstance(getattr(first, name), np.ndarray), name
    assert engine._dev_terms is not None
    engine.release()
    assert engine._dev_terms is None and engine._plan is None
    # The scan built before the release still answers from host memory.
    spectrum = first.to_spectrum()
    assert spectrum.power.shape == first.amplitude.shape
    # The next scan re-plans and re-uploads lazily, with the same result.
    again = engine._scan(value)
    assert np.max(np.abs(again.amplitude - first.amplitude)) < 1e-12
