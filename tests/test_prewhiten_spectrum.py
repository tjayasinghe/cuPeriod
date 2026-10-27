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
    spectral_window,
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


# --- the spectral window -------------------------------------------------------


def test_window_matches_the_direct_sum_definition() -> None:
    time, _, error = _curve()
    grid = _grid(time)
    engine = SpectrumEngine(time, error, grid=grid, backend="numpy")
    window = engine.window()
    weight = 1.0 / error**2
    weight = weight / weight.sum()
    dt = time - engine.t_ref
    for index in range(0, grid.size, 29):
        w_f = np.sum(weight * np.exp(2j * np.pi * grid.values[index] * dt))
        assert window.amplitude[index] == pytest.approx(abs(w_f), abs=1e-12)
    assert np.all(window.amplitude <= 1.0 + 1e-12)
    assert np.array_equal(window.power, window.amplitude**2)


def test_window_peaks_at_the_sampling_alias() -> None:
    # Perfectly regular sampling at dt = 0.05 d puts an exact alias at 20 c/d: the
    # window there is 1, and it is small away from the aliases.
    time = 0.05 * np.arange(400)
    grid = uniform_frequency_grid(
        float(time.max()), maximum_frequency=22.0, minimum_frequency=0.5,
        samples_per_peak=5,
    )
    window = spectral_window(time, grid=grid)
    alias = int(np.argmin(np.abs(window.frequency - 20.0)))
    # The nearest grid sample sits a fraction of a Rayleigh width off the exact alias.
    assert window.amplitude[alias] > 0.99
    assert window.refine_peak(alias)[1] == pytest.approx(1.0, abs=1e-3)
    midway = int(np.argmin(np.abs(window.frequency - 10.0)))
    assert window.amplitude[midway] < 0.05


def test_nightly_gaps_put_sidelobes_at_one_cycle_per_day() -> None:
    # Single-site ground-based sampling: observations only during a fraction of each
    # night alias every peak at +/- 1 c/d, which is the classic reason to look at the
    # window before believing a close pair.
    rng = np.random.default_rng(3)
    time = np.sort(
        np.concatenate([day + rng.uniform(0.0, 0.3, 12) for day in range(25)])
    )
    grid = uniform_frequency_grid(
        float(time.max() - time.min()), maximum_frequency=3.0,
        minimum_frequency=0.05, samples_per_peak=10,
    )
    window = spectral_window(time, grid=grid)
    lobe = int(np.argmin(np.abs(window.frequency - 1.0)))
    assert window.amplitude[lobe] > 0.5
    assert np.all(window.amplitude <= 1.0 + 1e-12)


def test_window_is_identical_for_lsq_and_dft_engines() -> None:
    # The window is a property of sampling and weights alone; the amplitude
    # normalization must not touch it (the dft engine just computes it lazily).
    time, _, error = _curve()
    grid = _grid(time)
    lsq = SpectrumEngine(time, error, grid=grid, backend="numpy")
    dft = SpectrumEngine(time, error, grid=grid, backend="numpy",
                         normalization="dft")
    assert np.allclose(lsq.window().amplitude, dft.window().amplitude, atol=1e-14)


def test_window_backends_agree() -> None:
    time, _, error = _curve()
    grid = _grid(time)
    direct = spectral_window(time, error, grid=grid, backend="numpy")
    nufft = spectral_window(time, error, grid=grid, backend="finufft")
    assert np.max(np.abs(direct.amplitude - nufft.amplitude)) < 1e-9


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


# --- regressions ---------------------------------------------------------------


def _reference_spectrum(engine: SpectrumEngine, values):
    """The amplitude/phase/power of the floating-mean LSQ spectrum, written out in full.

    The pre-fix ``SpectrumEngine.spectrum`` arithmetic, kept here so the lean
    extraction-loop path can be pinned against it bit for bit away from the
    ill-conditioned zones.
    """
    from cuperiod.prewhiten.spectrum import _wrapped_angles  # noqa: F401 (import check)

    w = engine._weight
    y = np.asarray(values, dtype=np.float64)
    centred = y - float(np.dot(w, y))
    yy = float(np.dot(w, centred * centred))
    yc, ys = engine._base_sums(w * centred)
    terms = engine._normal_terms()
    a = (terms.ss * yc - terms.cs * ys) * terms.inv
    b = (terms.cc * ys - terms.cs * yc) * terms.inv
    amplitude = np.hypot(a, b)
    phase = np.mod(np.arctan2(a, b), 2.0 * np.pi)
    power = np.clip((a * yc + b * ys) / yy, 0.0, 1.0)
    return amplitude, phase, power, terms


@pytest.mark.parametrize("backend", ["numpy", "finufft"])
def test_lean_scan_matches_the_full_spectrum_bit_for_bit(backend: str) -> None:
    # PERF-PREWHITEN-02 / PW-A-9: the extraction loop reads only the amplitude and one
    # power value; the lean scan must give exactly what spectrum() gives.
    time, value, error = _curve(n=300, span=12.0)
    engine = SpectrumEngine(time, error, grid=_grid(time, maximum=8.0), backend=backend)
    scan = engine._scan(value)
    full = scan.to_spectrum()
    assert np.array_equal(scan.amplitude, full.amplitude)
    assert np.array_equal(scan.frequency, full.frequency)
    index = scan.peak_index()
    assert index == full.peak_index()
    assert index is not None
    assert scan.power_at(index) == full.power[index]
    assert all(scan.power_at(i) == full.power[i] for i in range(0, full.size, 17))
    assert scan.refine_peak(index) == full.refine_peak(index)
    assert scan.noise_at(1.7) == full.noise_at(1.7)
    # spectrum() is the same computation; finufft's threaded spreading is not
    # bitwise reproducible from one transform to the next, numpy is.
    public = engine.spectrum(value)
    for name in ("amplitude", "phase", "power", "frequency"):
        if backend == "numpy":
            assert np.array_equal(getattr(public, name), getattr(full, name)), name
        else:
            assert np.allclose(
                getattr(public, name), getattr(full, name), rtol=0.0, atol=1e-13
            ), name
    if backend != "numpy":
        return
    # ...and both equal the plain normal-equation arithmetic where it is conditioned.
    amplitude, phase, power, terms = _reference_spectrum(engine, value)
    good = ~terms.ill
    assert np.array_equal(full.amplitude[good], amplitude[good])
    assert np.array_equal(full.phase[good], phase[good])
    assert np.array_equal(full.power, power)


def test_gram_terms_are_built_lazily_and_only_once(monkeypatch) -> None:
    # PERF-PREWHITEN-07: the window needs only the base sums; the doubled-grid
    # transform of the Gram terms belongs to spectrum() and runs once per engine.
    time, value, error = _curve()
    grid = _grid(time)
    calls = {"n": 0}
    original = SpectrumEngine._doubled_sums

    def spy(self, strengths):
        calls["n"] += 1
        return original(self, strengths)

    monkeypatch.setattr(SpectrumEngine, "_doubled_sums", spy)
    window = spectral_window(time, error, grid=grid, backend="finufft")
    assert calls["n"] == 0
    assert window.normalization == "lsq"
    engine = SpectrumEngine(time, error, grid=grid, backend="finufft")
    assert calls["n"] == 0
    engine.spectrum(value)
    engine.spectrum(value * 2.0)
    assert calls["n"] == 1
    # The window of an engine that already ran spectrum() is the same window (to the
    # last bits finufft's threaded spreading leaves free between two transforms).
    assert np.allclose(engine.window().amplitude, window.amplitude, rtol=0, atol=1e-15)
    numpy_engine = SpectrumEngine(time, error, grid=grid, backend="numpy")
    first = numpy_engine.window().amplitude
    numpy_engine.spectrum(value)
    assert np.array_equal(numpy_engine.window().amplitude, first)
    assert np.array_equal(
        spectral_window(time, error, grid=grid, backend="numpy").amplitude, first
    )


def test_reused_plan_gives_the_same_doubled_sums_as_a_fresh_one() -> None:
    # PERF-PREWHITEN-05: the doubled-grid sums re-point the base plan instead of
    # building a second one; the base plan must be left exactly as it was.
    time, value, error = _curve()
    grid = _grid(time)
    engine = SpectrumEngine(time, error, grid=grid, backend="finufft")
    before = engine._base_sums(engine._weight)
    doubled = engine._doubled_sums(engine._weight)
    after = engine._base_sums(engine._weight)
    # Bit-for-bit up to finufft's own run-to-run spreading noise (~1e-17).
    assert np.max(np.abs(before[0] - after[0])) < 1e-15
    assert np.max(np.abs(before[1] - after[1])) < 1e-15
    reference = SpectrumEngine(
        time, error, grid=GridSpec(
            kind="frequency",
            values=2.0 * grid.values,
            uniform=True,
        ), backend="numpy",
    )
    direct = reference._base_sums(reference._weight)
    assert np.max(np.abs(doubled[0] - direct[0])) < 1e-9
    assert np.max(np.abs(doubled[1] - direct[1])) < 1e-9


def test_cpu_backends_resolve_without_probing_for_a_gpu(monkeypatch) -> None:
    # PERF-PREWHITEN-06: an explicit CPU request must not import cupy / probe CUDA.
    import importlib

    spectrum_module = importlib.import_module("cuperiod.prewhiten.spectrum")

    def boom() -> set[str]:
        raise AssertionError("available_backends() probed the GPU stack")

    monkeypatch.setattr(spectrum_module, "available_backends", boom)
    assert resolve_spectrum_backend("numpy") == "numpy"
    assert resolve_spectrum_backend("astropy") == "numpy"
    assert resolve_spectrum_backend(" NumPy ") == "numpy"
    assert resolve_spectrum_backend("cpu") in {"finufft", "numpy"}
    assert resolve_spectrum_backend("CPU") in {"finufft", "numpy"}
    assert resolve_spectrum_backend("finufft") == "finufft"
    with pytest.raises(BackendUnavailableError, match="unknown backend"):
        resolve_spectrum_backend("nonsense")


@pytest.mark.parametrize("backend", ["numpy", "finufft"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_times_raise_instead_of_crashing(backend: str, bad: float) -> None:
    # PW-A-8: one NaN/inf timestamp used to give an all-zero spectrum (numpy) or kill
    # the interpreter inside finufft. It must be a ValueError naming the problem.
    time, value, error = _curve()
    time = time.copy()
    time[7] = bad
    grid = _grid(np.array([0.0, 10.0]))
    with pytest.raises(ValueError, match="time"):
        amplitude_spectrum(time, value, error, grid=grid, backend=backend)
    with pytest.raises(ValueError, match="time"):
        spectral_window(time, error, grid=grid, backend=backend)


@pytest.mark.parametrize("bad", [0.0, -0.01, np.nan, np.inf])
def test_unusable_errors_raise(bad: float) -> None:
    # PW-A-8: a zero or NaN error turned every sum into NaN and the spectrum into an
    # all-zero array with no exception.
    time, value, error = _curve()
    error = error.copy()
    error[3] = bad
    with pytest.raises(ValueError, match="error"):
        amplitude_spectrum(time, value, error, grid=_grid(time), backend="numpy")


def test_non_finite_values_raise() -> None:
    time, value, error = _curve()
    engine = SpectrumEngine(time, error, grid=_grid(time), backend="numpy")
    value = value.copy()
    value[5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        engine.spectrum(value)


def _regular_noise(seed: int = 0):
    """Pure white noise on an exact 30-minute cadence (Nyquist 24 c/d)."""
    time = 2459000.0 + np.arange(1296) / 48.0
    rng = np.random.default_rng(seed)
    return time, 10.0 + rng.normal(0.0, 0.001, time.size), np.full(time.size, 0.001)


def test_ill_conditioned_frequencies_do_not_produce_giant_lsq_peaks() -> None:
    # PW-A-1: near k/dt and (k + 1/2)/dt the floating-mean normal equations are nearly
    # singular; dividing noise by the tiny eigenvalue made peaks 30x the noise there.
    time, value, error = _regular_noise()
    grid = uniform_frequency_grid(
        float(time.max() - time.min()), maximum_frequency=120.0, samples_per_peak=10
    )
    spectrum = amplitude_spectrum(time, value, error, grid=grid, backend="finufft")
    assert spectrum.amplitude.max() < 8.0 * np.median(spectrum.amplitude)
    # The Deeming (DFT) amplitude is the bounded reference everywhere.
    deeming = amplitude_spectrum(
        time, value, error, grid=grid, backend="finufft", normalization="dft"
    )
    assert spectrum.amplitude.max() < 3.0 * deeming.amplitude.max()


def test_ill_conditioned_zone_reports_the_power_equivalent_amplitude() -> None:
    time, value, error = _regular_noise(seed=2)
    grid = uniform_frequency_grid(
        float(time.max() - time.min()), maximum_frequency=50.0, samples_per_peak=10
    )
    engine = SpectrumEngine(time, error, grid=grid, backend="finufft")
    spectrum = engine.spectrum(value)
    terms = engine._normal_terms()
    assert terms.ill.any()  # the Nyquist and 1/dt zones are in band
    w = engine._weight
    centred = value - float(np.dot(w, value))
    yy = float(np.dot(w, centred * centred))
    expected = np.sqrt(2.0 * yy * spectrum.power[terms.ill])
    assert np.allclose(spectrum.amplitude[terms.ill], expected, rtol=1e-12, atol=0.0)
    # A well-conditioned frequency keeps the exact least-squares amplitude.
    assert not terms.ill[np.searchsorted(grid.values, 7.3)]


def test_numpy_direct_sums_match_a_brute_force_sum() -> None:
    # PERF-PREWHITEN-09: the numpy direct path is a mat-vec now; it must still be the
    # transparent reference and must not depend on the frequency chunking.
    from cuperiod.core._arrayapi import array_namespace
    from cuperiod.prewhiten.spectrum import _direct_trig_sums

    rng = np.random.default_rng(4)
    tau = np.sort(rng.uniform(-15.0, 15.0, 700))
    strengths = rng.normal(size=tau.size) * rng.uniform(0.2, 1.0, tau.size)
    f0, df, nf = 0.05, 0.013, 900
    expected = np.exp(
        2j * np.pi * (f0 + df * np.arange(nf))[:, None] * tau[None, :]
    ) @ strengths
    # The array-API namespace, as production passes: raw numpy < 2 has no device=.
    xp = array_namespace(tau)
    cos_a, sin_a = _direct_trig_sums(xp, tau, strengths, f0, df, nf, 4096)
    cos_b, sin_b = _direct_trig_sums(xp, tau, strengths, f0, df, nf, 37)
    assert np.max(np.abs(cos_a - expected.real)) < 1e-12
    assert np.max(np.abs(sin_a - expected.imag)) < 1e-12
    scale = float(np.abs(strengths).sum())
    assert np.max(np.abs(cos_a - cos_b)) < 1e-15 * scale
    assert np.max(np.abs(sin_a - sin_b)) < 1e-15 * scale


@requires_torch
def test_float32_direct_path_keeps_its_phase_at_large_f_times_t() -> None:
    # PW-A-10: forming 2*pi*f*t in float32 loses the phase as f*T grows (14% error at
    # f*T = 1.4e6); the phases are now reduced in float64 on the host.
    rng = np.random.default_rng(0)
    span, frequency = 2000.0, 700.0
    time = np.sort(rng.uniform(0.0, span, 4000))
    value = 0.01 * np.sin(2 * np.pi * frequency * time + 0.4) + rng.normal(
        0.0, 0.001, time.size
    )
    df = 1.0 / (10 * span)
    grid = GridSpec(
        kind="frequency",
        values=frequency - 200 * df + df * np.arange(401),
        uniform=True,
    )
    reference = amplitude_spectrum(time, value, None, grid=grid, backend="numpy")
    single = amplitude_spectrum(
        time, value, None, grid=grid, backend="torch:cpu", precision="float32"
    )
    peak = float(reference.amplitude.max())
    assert np.max(np.abs(single.amplitude - reference.amplitude)) / peak < 1e-3
    assert single.peak_index() == reference.peak_index()


@requires_torch
def test_float32_direct_path_short_baseline() -> None:
    time, value, error = _curve(n=400, span=27.0)
    grid = _grid(time, maximum=20.0, samples=5)
    reference = amplitude_spectrum(time, value, error, grid=grid, backend="numpy")
    single = amplitude_spectrum(
        time, value, error, grid=grid, backend="torch:cpu", precision="float32"
    )
    peak = float(reference.amplitude.max())
    assert np.max(np.abs(single.amplitude - reference.amplitude)) / peak < 1e-3


def test_spectra_compare_by_identity_without_raising() -> None:
    # PW-B-10: a frozen dataclass of arrays raised on == and hash().
    time, value, error = _curve()
    engine = SpectrumEngine(time, error, grid=_grid(time), backend="numpy")
    first, second = engine.spectrum(value), engine.spectrum(value)
    assert first == first
    assert (first == second) is False
    assert len({first, second}) == 2
    assert [first, second].index(second) == 1


# --- finufft thread count ---------------------------------------------------------


class _PlanSpy:
    """Records the keyword arguments every ``finufft.Plan`` is built with."""

    def __init__(self, real: object) -> None:
        self.real = real
        self.kwargs: list[dict[str, object]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.kwargs.append(kwargs)
        return self.real(*args, **kwargs)  # type: ignore[operator]


def test_small_finufft_spectrum_runs_on_one_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waking finufft's thread pool cost ~7x a small transform; one thread is used."""
    finufft = pytest.importorskip("finufft")
    spy = _PlanSpy(finufft.Plan)
    monkeypatch.setattr(finufft, "Plan", spy)
    time, value, error = _curve(n=300, span=12.0)
    grid = _grid(time, maximum=8.0)
    single = SpectrumEngine(time, error, grid=grid, backend="finufft").spectrum(value)
    assert spy.kwargs and all(k.get("nthreads") == 1 for k in spy.kwargs)
    direct = amplitude_spectrum(time, value, error, grid=grid, backend="numpy")
    assert np.max(np.abs(direct.amplitude - single.amplitude)) < 1e-9
    assert np.max(np.abs(direct.power - single.power)) < 1e-9


def test_large_finufft_spectrum_keeps_every_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finufft = pytest.importorskip("finufft")
    spy = _PlanSpy(finufft.Plan)
    monkeypatch.setattr(finufft, "Plan", spy)
    rng = np.random.default_rng(5)
    time = np.sort(rng.uniform(0.0, 20.0, 150_001))  # the points alone exceed it
    SpectrumEngine(time, None, grid=_grid(time, maximum=1.0), backend="finufft")
    assert spy.kwargs and all("nthreads" not in k for k in spy.kwargs)
