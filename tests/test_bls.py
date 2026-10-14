"""BLS: astropy parity, period recovery, multi-band, and GPU agreement."""

from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest
from astropy.timeseries import BoxLeastSquares

import cuperiod as cup
from conftest import requires_gpu, requires_numba, requires_torch, torch_auto_device
from cuperiod.core.errors import BackendUnavailableError
from cuperiod.core.grid import GridSpec, GridTooLargeError, log_period_grid
from cuperiod.methods import _bls_core
from cuperiod.methods._bls_core import _duration_bins, bls_power
from cuperiod.methods.bls import BLSMethod
from synth import synthetic_eclipser


def _grid_and_lc() -> tuple[object, cup.LightCurve, cup.BLSSettings]:
    t, flux, err = synthetic_eclipser()
    lc = cup.LightCurve.from_arrays(t, flux, err, domain=cup.Domain.FLUX)
    settings = cup.BLSSettings()
    grid = BLSMethod().default_grid(lc, settings)
    return grid, lc, settings


def test_bls_numpy_matches_astropy() -> None:
    grid, lc, settings = _grid_and_lc()
    method = BLSMethod()
    mine = method.power(grid, lc, settings, "numpy")
    astro = method.power(grid, lc, settings, "astropy")
    finite = np.isfinite(mine.power) & np.isfinite(astro.power)
    assert float(np.max(np.abs(mine.power[finite] - astro.power[finite]))) < 1e-7
    assert int(np.argmax(mine.power)) == int(np.argmax(astro.power))
    assert np.max(np.abs(mine.extras["duration"] - astro.extras["duration"])) < 1e-9


def test_bls_numba_matches_astropy() -> None:
    pytest.importorskip("numba")
    grid, lc, settings = _grid_and_lc()
    method = BLSMethod()
    mine = method.power(grid, lc, settings, "numba")
    astro = method.power(grid, lc, settings, "astropy")
    finite = np.isfinite(mine.power) & np.isfinite(astro.power)
    assert float(np.max(np.abs(mine.power[finite] - astro.power[finite]))) < 1e-7
    assert int(np.argmax(mine.power)) == int(np.argmax(astro.power))
    assert np.max(np.abs(mine.extras["duration"] - astro.extras["duration"])) < 1e-9


def test_bls_cpu_prefers_numba_when_available() -> None:
    pytest.importorskip("numba")
    assert BLSMethod().resolve_backend("cpu") == "numba"


def test_bls_power_function_matches_astropy() -> None:
    from astropy.timeseries import BoxLeastSquares

    t, flux, err = synthetic_eclipser()
    periods = np.linspace(1.5, 4.0, 400)
    durations = np.array([0.05, 0.1, 0.2])
    mine = bls_power(t, flux, err, periods, durations, 10, backend="numpy")
    ref = BoxLeastSquares(t, flux, err).power(
        periods, durations, objective="snr", oversample=10
    )
    finite = np.isfinite(mine.power) & np.isfinite(np.asarray(ref.power))
    assert np.max(np.abs(mine.power[finite] - np.asarray(ref.power)[finite])) < 1e-7
    assert np.max(np.abs(mine.depth - np.asarray(ref.depth))) < 1e-6


def test_bls_recovers_period_and_depth() -> None:
    t, flux, err = synthetic_eclipser(period=2.5, depth=0.05)
    pg = cup.periodogram((t, flux, err), "BLS", domain=cup.Domain.FLUX)
    peak = pg.best_periods(1, alias_diverse=True)[0]
    assert peak.period == pytest.approx(2.5, rel=3e-3)
    assert peak.extra["depth"] == pytest.approx(0.05, abs=0.01)
    assert peak.extra["sde"] > 7.0


def test_bls_converts_magnitude_to_flux() -> None:
    # An eclipse given as magnitudes (fainter = larger) must still be found.
    t, flux, err = synthetic_eclipser(depth=0.1)
    mag = -2.5 * np.log10(flux)
    pg = cup.periodogram((t, mag, 0.01 * np.ones_like(mag)), "BLS", domain="magnitude")
    peak = pg.best_periods(1, alias_diverse=True)[0]
    assert peak.period == pytest.approx(2.5, rel=5e-3)


def test_bls_multiband_recovers_period() -> None:
    tg, fg, eg = synthetic_eclipser(n=500, period=3.0, seed=4)
    tr, fr, er = synthetic_eclipser(n=500, period=3.0, depth=0.08, seed=5)
    mb = cup.MultiBandLightCurve.from_light_curves(
        {
            "g": cup.LightCurve.from_arrays(tg, fg, eg, domain=cup.Domain.FLUX),
            "r": cup.LightCurve.from_arrays(tr, fr, er, domain=cup.Domain.FLUX),
        }
    )
    pg = cup.periodogram(mb, "BLS")
    peak = pg.best_periods(1, alias_diverse=True)[0]
    assert peak.period == pytest.approx(3.0, rel=5e-3)


@requires_gpu
def test_bls_gpu_matches_numpy() -> None:
    grid, lc, settings = _grid_and_lc()
    method = BLSMethod()
    cpu = method.power(grid, lc, settings, "numpy")
    gpu = method.power(grid, lc, settings, "cupy")
    finite = np.isfinite(cpu.power) & np.isfinite(gpu.power)
    assert float(np.max(np.abs(cpu.power[finite] - gpu.power[finite]))) < 1e-7


@requires_torch
def test_bls_torch_cpu_matches_numpy() -> None:
    # The torch box search shares the array-API body with numpy: bit-for-bit close.
    grid, lc, settings = _grid_and_lc()
    method = BLSMethod()
    ref = method.power(grid, lc, settings, "numpy")
    tor = method.power(grid, lc, settings, "torch:cpu")
    assert tor.backend == "torch:cpu"
    finite = np.isfinite(ref.power) & np.isfinite(tor.power)
    assert float(np.max(np.abs(ref.power[finite] - tor.power[finite]))) < 1e-7
    assert int(np.argmax(ref.power)) == int(np.argmax(tor.power))


@requires_torch
@torch_auto_device
def test_bls_torch_recovers_period_and_depth() -> None:
    t, flux, err = synthetic_eclipser(period=2.5, depth=0.05)
    pg = cup.periodogram((t, flux, err), "BLS", domain=cup.Domain.FLUX, backend="torch")
    peak = pg.best_periods(1, alias_diverse=True)[0]
    assert peak.period == pytest.approx(2.5, rel=3e-3)
    assert peak.extra["depth"] == pytest.approx(0.05, abs=0.01)


@requires_torch
def test_bls_torch_float32_tracks_float64_on_bjd_times() -> None:
    # Regression: the absolute time origin must be subtracted in float64 *before* the
    # float32 device cast. Synthetic times are real-scale BJD (~2.458e6), where float32
    # spacing is ~0.25 d — coarser than a transit bin — so a raw cast destroys the phase
    # fold and float32 power diverges by ~order-1 from float64. float32 is the default
    # precision on Apple MPS, so this is the out-of-the-box Mac path. After the fix the
    # only difference is float32 round-off.
    t, flux, err = synthetic_eclipser(period=2.5, depth=0.05)
    assert float(t.min()) > 2.4e6  # the catastrophic-cancellation regime
    common = dict(domain=cup.Domain.FLUX, backend="torch:cpu")
    p64 = cup.periodogram(
        (t, flux, err), "BLS", settings=cup.BLSSettings(precision="float64"), **common
    )
    p32 = cup.periodogram(
        (t, flux, err), "BLS", settings=cup.BLSSettings(precision="float32"), **common
    )
    assert p32.backend == "torch:cpu"
    finite = np.isfinite(p64.power) & np.isfinite(p32.power)
    scale = max(float(np.max(np.abs(p64.power[finite]))), 1e-30)
    rel = float(np.max(np.abs(p64.power[finite] - p32.power[finite]))) / scale
    assert rel < 5e-2  # pre-fix the lost timing made this ~order-1
    best = p32.best_periods(1, alias_diverse=True)[0]
    assert best.period == pytest.approx(2.5, rel=5e-3)


# --- regressions --------------------------------------------------------------------


def _flux_lc(t: np.ndarray, flux: np.ndarray, err: np.ndarray) -> cup.LightCurve:
    return cup.LightCurve.from_arrays(t, flux, err, domain=cup.Domain.FLUX)


def _two_runs(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure noise in two 3-night runs a year apart (clustered sampling)."""
    rng = np.random.default_rng(seed)
    t = np.concatenate(
        [np.sort(rng.uniform(0.0, 3.0, 150)), np.sort(rng.uniform(365.0, 368.0, 150))]
    ) + 2458000.0
    err = rng.uniform(0.003, 0.008, t.size)
    flux = 1.0 + err * rng.standard_normal(t.size)
    return t, flux, err


@pytest.mark.parametrize(
    "backend",
    [
        "numpy",
        pytest.param("numba", marks=requires_numba),
        pytest.param("torch:cpu", marks=requires_torch),
        pytest.param("cupy", marks=requires_gpu),
    ],
)
def test_bls_rejects_boxes_that_swallow_every_point(backend: str) -> None:
    # METH-A-3: with clustered sampling one box can hold every point. Its "outside"
    # inverse variance is then cumulative-sum round-off, which passed the absolute
    # 2.2e-16 floor, and the out-of-box mean noise/noise was of the order of the flux:
    # a likelihood of ~1e6 at depth ~1 on pure noise. The same floor rejected every
    # box once the errors were large (flux and errors x 1e12 gave power 0 everywhere).
    settings = cup.BLSSettings(objective="likelihood", min_period_days=20.0)
    method = BLSMethod()
    for seed in (1, 5):
        t, flux, err = _two_runs(seed)
        grid = method.default_grid(_flux_lc(t, flux, err), settings)
        base = method.power(grid, _flux_lc(t, flux, err), settings, backend)
        assert base.best_periods(1)[0].extra["depth"] < 0.1
        assert float(base.power.max()) < 1e3
        for shifted in (
            _flux_lc(t, flux + 1000.0, err),
            _flux_lc(t, flux * 1e12, err * 1e12),
        ):
            other = method.power(grid, shifted, settings, backend)
            np.testing.assert_allclose(other.power, base.power, rtol=1e-6, atol=1e-9)


def test_duration_bins_round_half_away_from_zero() -> None:
    # METH-A-13: astropy's run_bls uses C round(); Python's round() is half-to-even.
    assert _duration_bins(np.array([0.2, 0.25, 0.45]), 0.1) == [2, 3, 5]


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("numba", marks=requires_numba)]
)
@pytest.mark.parametrize("durations", [(0.2, 0.25), (0.2, 0.45)])
def test_bls_half_bin_durations_match_astropy(
    backend: str, durations: tuple[float, float]
) -> None:
    # METH-A-13: a duration of exactly 2.5 (or 4.5) bins was searched as 2 (4) bins,
    # so with oversample=2 one trial duration vanished and ~1/3 of the periods
    # differed from BoxLeastSquares by up to 7.8 in SNR.
    t, flux, err = synthetic_eclipser()
    periods = np.linspace(1.5, 4.0, 300)
    mine = bls_power(t, flux, err, periods, np.array(durations), 2, backend=backend)
    ref = BoxLeastSquares(t, flux, err).power(
        periods, np.array(durations), objective="snr", oversample=2
    )
    assert float(np.max(np.abs(mine.power - np.asarray(ref.power)))) < 1e-7
    assert float(np.max(np.abs(mine.duration - np.asarray(ref.duration)))) < 1e-9


@requires_torch
def test_bls_float32_tracks_float64_on_ppm_photometry() -> None:
    # METH-A-19: the float32 device path (forced on Apple MPS) summed the raw flux,
    # so the cumulative sums of y*ivar carried 6e-8 of the whole flux level into
    # every in-box mean -- order-1 SNR errors and 10% depth errors on a 300 ppm
    # transit. The flux is now centred on its median in float64 first.
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 27.0, 6000)) + 2458000.0
    flux = 1.0 + 1e-4 * rng.standard_normal(t.size)
    flux[((t - 2458000.4) % 3.1) / 3.1 < 0.03] -= 3e-4
    err = np.full(t.size, 1e-4)
    periods = np.linspace(2.5, 4.0, 300)
    durations = np.array([0.05, 0.1, 0.15])
    common = dict(backend="torch:cpu")
    p64 = bls_power(t, flux, err, periods, durations, 10, precision="float64", **common)
    p32 = bls_power(t, flux, err, periods, durations, 10, precision="float32", **common)
    assert float(np.max(np.abs(p64.power - p32.power))) < 0.05
    # (A near-tie can still pick another box at a few periods, so the median.)
    assert float(np.median(np.abs(p64.depth - p32.depth))) < 1e-8
    best = int(np.argmax(p64.power))
    assert int(np.argmax(p32.power)) == best
    assert abs(float(p64.depth[best] - p32.depth[best])) < 1e-7


@pytest.mark.parametrize(
    "backend", ["astropy", "numpy", pytest.param("numba", marks=requires_numba)]
)
def test_bls_searches_exactly_the_grid_it_is_given(backend: str) -> None:
    # METH-A-4: a default grid only flagged "segmented", and power() rebuilt the
    # periods from the light curve and the *current* settings, discarding a grid made
    # with other settings or for another light curve.
    t, flux, err = synthetic_eclipser(span=80.0)
    lc = _flux_lc(t, flux, err)
    method = BLSMethod()
    settings = cup.BLSSettings(min_period_days=1.0, max_period_days=4.0)
    narrow = method.default_grid(lc, cup.BLSSettings(min_period_days=2.0,
                                                     max_period_days=3.0))
    pg = method.power(narrow, lc, settings, backend)
    np.testing.assert_allclose(np.sort(pg.period), narrow.period, rtol=1e-12)
    assert pg.best_periods(1)[0].period == pytest.approx(2.5, rel=3e-3)
    subset = _flux_lc(t[:300], flux[:300], err[:300])
    other = method.default_grid(subset, settings)
    pg = method.power(other, lc, settings, backend)
    np.testing.assert_allclose(np.sort(pg.period), other.period, rtol=1e-12)
    # The light curve's own default grid is searched exactly as before.
    own = method.default_grid(lc, settings)
    assert own.meta["segmented"]
    assert sum(own.meta["segment_sizes"]) == own.size
    pg = method.power(own, lc, settings, backend)
    np.testing.assert_allclose(np.sort(pg.period), own.period, rtol=1e-12)


def _long_period_eclipser() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0.0, 400.0, 3000)) + 2458000.0
    flux = 1.0 + 0.002 * rng.standard_normal(t.size)
    phase = ((t - 2458011.0) % 37.3) / 37.3
    flux[(phase < 0.45 / 37.3) | (phase > 1.0 - 0.45 / 37.3)] -= 0.01
    return t, flux, np.full(t.size, 0.002)


@requires_numba
def test_bls_custom_grid_durations_track_the_period() -> None:
    # METH-A-5: a custom grid was one segment whose box durations were 1-12% of its
    # *shortest* period, so on a 1-100 d log grid a 0.9 d eclipse at 37.3 d was fitted
    # with 0.12 d boxes at half the SNR (and a 1e5-bin phase grid per period).
    t, flux, err = _long_period_eclipser()
    lc = _flux_lc(t, flux, err)
    method = BLSMethod()
    settings = cup.BLSSettings()
    default = method.power(method.default_grid(lc, settings), lc, settings, "numba")
    custom = method.power(
        log_period_grid(minimum_period=1.0, maximum_period=100.0, n_periods=4000),
        lc, settings, "numba",
    )
    ref = default.best_periods(1, alias_diverse=True)[0]
    got = custom.best_periods(1, alias_diverse=True)[0]
    assert ref.period == pytest.approx(37.3, rel=2e-3)
    assert got.period == pytest.approx(37.3, rel=2e-3)
    assert got.extra["depth_snr"] >= 0.9 * ref.extra["depth_snr"]
    assert 0.45 <= got.extra["duration"] <= 1.8


def test_bls_array_search_memory_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    # METH-A-15: the numpy/torch search materialised ~(4N + 14*width) float64 values
    # per trial period for a fixed 2048-period batch -- gigabytes for long curves.
    budget = 16 << 20
    monkeypatch.setattr(_bls_core, "_HOST_BATCH_BYTES", budget)
    t, flux, err = synthetic_eclipser(n=4000)
    periods = np.linspace(2.0, 4.0, 600)
    durations = np.array([0.05, 0.1, 0.2])
    tracemalloc.start()
    try:
        got = bls_power(t, flux, err, periods, durations, 10, backend="numpy")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1.5 * budget + (2 << 20)
    ref = bls_power(t, flux, err, periods, durations, 10, backend="numpy", batch=64)
    for name in ("power", "depth", "duration", "transit_time"):
        assert np.array_equal(getattr(got, name), getattr(ref, name))


def _two_band_eclipser() -> cup.MultiBandLightCurve:
    tg, fg, eg = synthetic_eclipser(n=400, period=3.0, seed=4)
    tr, fr, er = synthetic_eclipser(n=400, period=3.0, depth=0.08, seed=5)
    return cup.MultiBandLightCurve.from_light_curves(
        {"g": _flux_lc(tg, fg, eg), "r": _flux_lc(tr, fr, er)}
    )


@requires_torch
def test_bls_multiband_torch_resolves_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # METH-A-7 / MULTIBAND-8: multi-band BLS passed a bare "torch" straight through,
    # so it ran on torch's CPU device whatever settings.device said (on AMD/Intel/
    # Mac GPUs, where auto resolves to bare torch, too), never validated the device
    # and recorded the backend without it.
    mb = _two_band_eclipser()
    settings = cup.BLSSettings(device="cpu")
    grid = log_period_grid(minimum_period=2.0, maximum_period=4.0, n_periods=200)
    pg = cup.periodogram(mb, "BLS", backend="torch", settings=settings, grid=grid)
    assert pg.backend == "torch:cpu"
    ref = cup.periodogram(mb, "BLS", backend="numpy", settings=settings, grid=grid)
    np.testing.assert_allclose(pg.power, ref.power, rtol=0.0, atol=1e-7)

    import cuperiod.core.backend as backend_module

    monkeypatch.setattr(backend_module, "torch_devices", lambda: {"cpu"})
    with pytest.raises(BackendUnavailableError, match="cuda"):
        BLSMethod().multiband_power(
            grid, mb, cup.BLSSettings(device="cuda"), "torch"
        )


def _adversarial_cases() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(11)
    t, flux, err = synthetic_eclipser(n=300, span=60.0)
    n = t.size
    cases = [
        ("eclipser", t, flux, err),
        ("offset 1e6", t, flux + 1e6, err),
        ("offset 1e9", t, flux + 1e9, err),
        ("constant", t, np.ones(n), err),
        ("quantised ties", t, rng.integers(0, 3, n) * 0.01, err),
        ("wild errors", t, flux, 10.0 ** rng.uniform(-3.0, 1.0, n)),
        ("five points", t[:: n // 5][:5], flux[:: n // 5][:5], err[:: n // 5][:5]),
        ("pure noise", t, 1.0 + err * rng.standard_normal(n), err),
        ("huge errors", t, flux, np.full(n, 1e150)),
        ("both x 1e-60", t, flux * 1e-60, err * 1e-60),
        ("both x 1e60", t, flux * 1e60, err * 1e60),
    ]
    # Flux scales inside and outside the range the screen runs in.
    for scale in (1e-140, 1e-158, 1e-200, 1e140, 1e160):
        cases.append((f"flux x {scale:g}", t, flux * scale, err))
    return cases


@requires_numba
@pytest.mark.parametrize("obj_flag", [0, 1])
def test_numba_box_screen_is_exact(obj_flag: int) -> None:
    # PERF-METHODS-03: the numba sweep skips boxes whose division-free bound cannot
    # beat the incumbent. Every skip must be a proof: the screened sweep has to pick
    # bit-for-bit the boxes (and first-wins ties) of evaluating every box exactly.
    periods = np.linspace(0.6, 9.0, 160)
    durations = np.array([0.02, 0.05, 0.11, 0.3])
    oversample = 3
    bin_duration = float(durations.min()) / oversample
    dur_bins = _duration_bins(durations, bin_duration)
    width = math.ceil(float(periods.max()) / bin_duration) + oversample + 1
    keys = ("power", "depth", "depth_err", "depth_snr", "duration",
            "transit_time", "log_likelihood")
    for name, t, y, dy in _adversarial_cases():
        ivar = 1.0 / dy**2
        kwargs = dict(bin_duration=bin_duration, dur_bins=dur_bins,
                      oversample=oversample, width=width, obj_flag=obj_flag)
        exact = _bls_core._bls_search_numba(t, y, ivar, periods, screen=False, **kwargs)
        fast = _bls_core._bls_search_numba(t, y, ivar, periods, **kwargs)
        for key in keys:
            assert np.array_equal(fast[key], exact[key], equal_nan=True), (name, key)
    # Errors so large that every weight underflows: nothing admissible, no exception.
    t, flux, _ = synthetic_eclipser(n=200)
    with np.errstate(over="ignore"):
        out = bls_power(t, flux, np.full(t.size, 1e200), periods, durations, 3,
                        backend="numba")
    assert np.all(np.isneginf(out.power))


def test_bls_default_grid_has_no_overshooting_period() -> None:
    # CORE-B-9: np.arange can overshoot its stop by a rounding quantum, which put a
    # period just below min_period_days (or a duplicate of the previous segment's
    # longest period) into the default grid -- here the first segment.
    t = 2458000.0 + np.linspace(0.0, 115.35, 400)
    rng = np.random.default_rng(0)
    lc = _flux_lc(t, 1.0 + 0.01 * rng.standard_normal(t.size), np.full(t.size, 0.01))
    settings = cup.BLSSettings()
    grid = BLSMethod().default_grid(lc, settings)
    assert float(grid.period.min()) > settings.min_period_days
    assert bool(np.all(np.diff(grid.period) > 0.0))


def test_bls_default_grid_is_size_checked_before_allocation() -> None:
    # SEC-04: one t=0 row multiplies the baseline ~1e4-fold, and the segment grids
    # (4.9e8 periods here) were allocated without a check.
    t, flux, err = synthetic_eclipser(n=200)
    t[0] = 0.0
    with pytest.raises(GridTooLargeError, match="BLS default grid"):
        BLSMethod().default_grid(_flux_lc(t, flux, err), cup.BLSSettings())


def test_bls_empty_grid_raises() -> None:
    t, flux, err = synthetic_eclipser()
    with pytest.raises(cup.InsufficientDataError, match="empty trial grid"):
        BLSMethod().power(
            GridSpec(kind="period", values=np.zeros(0)), _flux_lc(t, flux, err),
            cup.BLSSettings(), "numpy",
        )


def test_bls_frequency_grid_through_zero_is_refused() -> None:
    # Frequency 0 is a valid GridSpec sample but an infinite period: the log
    # segmentation of a custom grid used to divide by zero on it (an OverflowError
    # before that), rather than saying what is wrong.
    t, flux, err = synthetic_eclipser()
    grid = GridSpec(kind="frequency", values=np.linspace(0.0, 0.5, 200))
    with pytest.raises(ValueError, match="finite"):
        BLSMethod().power(grid, _flux_lc(t, flux, err), cup.BLSSettings(), "numpy")
