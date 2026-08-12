"""Amplitude spectrum: exactness against brute-force least squares, backend parity."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu, requires_torch
from cuperiod.core.errors import BackendUnavailableError, InsufficientDataError
from cuperiod.core.grid import GridSpec, uniform_frequency_grid
from cuperiod.prewhiten.spectrum import (
    SpectrumEngine,
    amplitude_spectrum,
    noise_level,
    resolve_spectrum_backend,
    weighted_epoch,
)
from synth import synthetic_pulsator


def _curve(n: int = 200, span: float = 10.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0.0, span, n))
    error = rng.uniform(0.01, 0.05, n)
    value = 0.3 * np.sin(2 * np.pi * 1.7 * time + 0.9) + rng.normal(0.0, error)
    return time, value, error


def _grid(time, *, maximum=5.0, samples=3) -> GridSpec:
    return uniform_frequency_grid(
        float(time.max() - time.min()),
        maximum_frequency=maximum,
        minimum_frequency=0.1,
        samples_per_peak=samples,
    )


def _brute_force(time, value, error, frequency, t_ref):
    """The weighted least-squares (amplitude, phase) at one frequency, transparently."""
    dt = time - t_ref
    design = np.column_stack(
        [np.cos(2 * np.pi * frequency * dt), np.sin(2 * np.pi * frequency * dt),
         np.ones(time.size)]
    )
    weight = 1.0 / error
    beta, *_ = np.linalg.lstsq(design * weight[:, None], value * weight, rcond=None)
    return float(np.hypot(beta[0], beta[1])), float(
        np.mod(np.arctan2(beta[0], beta[1]), 2 * np.pi)
    )


def test_lsq_amplitude_matches_brute_force_least_squares() -> None:
    time, value, error = _curve()
    grid = _grid(time)
    spectrum = amplitude_spectrum(time, value, error, grid=grid, backend="numpy")
    for index in range(0, grid.size, 37):
        amp, phase = _brute_force(
            time, value, error, grid.values[index], spectrum.t_ref
        )
        assert spectrum.amplitude[index] == pytest.approx(amp, abs=1e-12)
        wrapped = np.angle(np.exp(1j * (phase - spectrum.phase[index])))
        assert abs(wrapped) < 1e-9


def test_numpy_and_finufft_backends_agree() -> None:
    time, value, error = _curve()
    grid = _grid(time)
    direct = amplitude_spectrum(time, value, error, grid=grid, backend="numpy")
    nufft = amplitude_spectrum(time, value, error, grid=grid, backend="finufft")
    assert np.max(np.abs(direct.amplitude - nufft.amplitude)) < 1e-9
    assert np.max(np.abs(direct.power - nufft.power)) < 1e-9


def test_dft_normalization_matches_deeming() -> None:
    time, value, _ = _curve()
    grid = _grid(time)
    spectrum = amplitude_spectrum(
        time, value, None, grid=grid, backend="numpy", normalization="dft"
    )
    centred = value - value.mean()
    for index in range(0, grid.size, 53):
        dt = time - spectrum.t_ref
        phasor = np.exp(2j * np.pi * grid.values[index] * dt)
        expected = 2.0 * abs(np.sum(centred * phasor)) / time.size
        assert spectrum.amplitude[index] == pytest.approx(expected, rel=1e-9)


def test_engine_reuses_sampling_terms_across_calls() -> None:
    # The window terms depend only on the times, so evaluating two different value
    # arrays on one engine must match two independent one-shot spectra exactly.
    time, value, error = _curve()
    grid = _grid(time)
    engine = SpectrumEngine(time, error, grid=grid, backend="numpy")
    other = value * 2.0 + 1.0
    first, second = engine.spectrum(value), engine.spectrum(other)
    assert np.array_equal(
        first.amplitude,
        amplitude_spectrum(time, value, error, grid=grid, backend="numpy").amplitude,
    )
    # Amplitude is linear in the data and blind to an added constant.
    assert np.allclose(second.amplitude, 2.0 * first.amplitude, rtol=1e-12)


def test_peak_index_honours_the_exclusion_zone() -> None:
    time, value, error = synthetic_pulsator(n=600, span=20.0)
    grid = _grid(time, maximum=30.0, samples=6)
    spectrum = amplitude_spectrum(time, value, error, grid=grid, backend="finufft")
    top = spectrum.peak_index()
    assert top is not None
    strongest = spectrum.frequency[top]
    blocked = spectrum.peak_index(
        exclude=np.asarray([strongest]), separation=1.0
    )
    assert blocked is not None
    assert abs(spectrum.frequency[blocked] - strongest) >= 1.0


def test_refine_peak_beats_the_grid_sample() -> None:
    time, value, error = synthetic_pulsator(
        n=800, span=20.0, frequencies=(7.1234,), amplitudes=(0.05,), phases=(0.3,)
    )
    grid = _grid(time, maximum=12.0, samples=2)  # deliberately coarse
    spectrum = amplitude_spectrum(time, value, error, grid=grid, backend="finufft")
    index = spectrum.peak_index()
    assert index is not None
    refined, apex = spectrum.refine_peak(index)
    assert abs(refined - 7.1234) < abs(spectrum.frequency[index] - 7.1234)
    assert apex >= spectrum.amplitude[index]


def test_noise_level_estimators() -> None:
    frequency = np.linspace(0.0, 10.0, 1001)
    amplitude = np.full_like(frequency, 2.0)
    amplitude[500] = 100.0  # one huge peak inside the box
    assert noise_level(frequency, amplitude, 5.0, window=1.0, estimator="median") == 2.0
    assert noise_level(frequency, amplitude, 5.0, window=1.0, estimator="mean") > 2.0


def test_noise_level_widens_a_box_that_is_too_narrow() -> None:
    frequency = np.linspace(0.0, 10.0, 101)
    amplitude = np.arange(101, dtype=float)
    # A box of half-width 0.01 holds one sample; the estimator must use min_samples.
    value = noise_level(frequency, amplitude, 5.0, window=0.01, min_samples=25)
    assert np.isfinite(value)
    assert value == pytest.approx(np.mean(amplitude[38:63]), rel=1e-9)


def test_weighted_epoch_decorrelates_phase_from_frequency() -> None:
    time = np.array([0.0, 1.0, 2.0, 10.0])
    assert weighted_epoch(time) == pytest.approx(3.25)
    weights = np.array([1.0, 1.0, 1.0, 0.0])
    assert weighted_epoch(time, weights) == pytest.approx(1.0)


def test_engine_rejects_a_non_uniform_grid() -> None:
    time, _, error = _curve()
    grid = GridSpec(kind="period", values=np.geomspace(0.1, 10.0, 50))
    with pytest.raises(ValueError, match="uniform frequency grid"):
        SpectrumEngine(time, error, grid=grid)


def test_engine_rejects_a_short_curve() -> None:
    with pytest.raises(InsufficientDataError):
        SpectrumEngine(np.array([1.0, 2.0]), grid=_grid(np.array([0.0, 10.0])))


def test_spectrum_rejects_mismatched_values() -> None:
    time, value, error = _curve()
    engine = SpectrumEngine(time, error, grid=_grid(time), backend="numpy")
    with pytest.raises(ValueError, match="does not match"):
        engine.spectrum(value[:-1])


def test_backend_resolution() -> None:
    assert resolve_spectrum_backend("numpy") == "numpy"
    assert resolve_spectrum_backend("astropy") == "numpy"
    assert resolve_spectrum_backend("cpu") in {"finufft", "numpy"}
    assert resolve_spectrum_backend("auto") in {
        "cufinufft", "torch", "finufft", "numpy"
    }
    with pytest.raises(BackendUnavailableError, match="unknown backend"):
        resolve_spectrum_backend("nonsense")


@requires_torch
def test_torch_backend_matches_numpy() -> None:
    time, value, error = _curve()
    grid = _grid(time)
    reference = amplitude_spectrum(time, value, error, grid=grid, backend="numpy")
    torch_result = amplitude_spectrum(
        time, value, error, grid=grid, backend="torch:cpu"
    )
    assert torch_result.backend == "torch:cpu"
    assert np.max(np.abs(reference.amplitude - torch_result.amplitude)) < 1e-9


@requires_gpu
def test_cufinufft_backend_matches_finufft() -> None:
    time, value, error = _curve(n=500, span=30.0)
    grid = _grid(time, maximum=20.0, samples=5)
    cpu = amplitude_spectrum(time, value, error, grid=grid, backend="finufft")
    gpu = amplitude_spectrum(time, value, error, grid=grid, backend="cufinufft")
    assert gpu.backend == "cufinufft"
    assert np.max(np.abs(cpu.amplitude - gpu.amplitude)) < 1e-9
