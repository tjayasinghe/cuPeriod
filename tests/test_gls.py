"""GLS: astropy parity, period recovery, multi-band, and GPU agreement."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.timeseries import LombScargle

import cuperiod as cup
from conftest import requires_gpu, requires_torch, torch_auto_device
from cuperiod.core.grid import GridSpec, GridTooLargeError
from cuperiod.methods.gls import (
    FinufftGLS,
    GLSMethod,
    lombscargle_power,
)
from synth import synthetic_sine


def test_gls_empty_grid_raises() -> None:
    # Regression: an empty trial grid must raise InsufficientDataError, not IndexError.
    from cuperiod.core.errors import InsufficientDataError
    from cuperiod.core.grid import GridSpec

    t, mag, err = synthetic_sine(n=200)
    lc = cup.LightCurve.from_arrays(t, mag, err)
    method = GLSMethod()
    with pytest.raises(InsufficientDataError):
        method.power(
            GridSpec(kind="frequency", values=np.zeros(0), uniform=True),
            lc, cup.GLSSettings(), method.resolve_backend("cpu"),
        )


def test_gls_matches_astropy_cython() -> None:
    t, mag, err = synthetic_sine()
    pg = cup.periodogram((t, mag, err), "GLS", backend="finufft")
    ref = LombScargle(t, mag, err, fit_mean=True).power(
        pg.frequency, method="cython", normalization="standard"
    )
    assert float(np.max(np.abs(pg.power - np.asarray(ref)))) < 1e-6


def test_gls_recovers_period() -> None:
    t, mag, err = synthetic_sine(period=0.6234)
    pg = cup.periodogram((t, mag, err), "GLS")
    assert pg.best_period() == pytest.approx(0.6234, rel=1e-3)


def test_gls_peaks_have_fap() -> None:
    t, mag, err = synthetic_sine()
    pg = cup.periodogram((t, mag, err), "GLS")
    top = pg.best_periods(5)
    assert top and "fap" in top[0].extra
    assert top[0].extra["fap"] < 1e-3  # a strong signal is highly significant


def test_gls_default_grid_matches_astropy_autofrequency() -> None:
    t, mag, err = synthetic_sine()
    lc = cup.LightCurve.from_arrays(t, mag, err)
    grid = GLSMethod().default_grid(lc, cup.GLSSettings())
    ref = LombScargle(t, mag, err).autofrequency(
        minimum_frequency=1.0 / (t.max() - t.min()),
        samples_per_peak=5,
        nyquist_factor=5,
    )
    assert np.allclose(grid.frequency, np.asarray(ref))


def test_gls_multiband_recovers_period() -> None:
    period = 0.81
    tg, mg, eg = synthetic_sine(n=300, period=period, seed=2)
    tr, mr, er = synthetic_sine(n=300, period=period, amp=0.3, seed=3)
    mb = cup.MultiBandLightCurve.from_light_curves(
        {
            "g": cup.LightCurve.from_arrays(tg, mg, eg),
            "r": cup.LightCurve.from_arrays(tr, mr + 1.0, er),
        }
    )
    pg = cup.periodogram(mb, "GLS")
    assert pg.best_period() == pytest.approx(period, rel=2e-3)


def test_lombscargle_power_empty_grid() -> None:
    out = lombscargle_power(np.arange(5.0), np.arange(5.0), None, 0.1, 0.0, 0)
    assert out.size == 0


@requires_gpu
def test_gls_gpu_matches_cpu() -> None:
    t, mag, err = synthetic_sine()
    cpu = cup.periodogram((t, mag, err), "GLS", backend="finufft")
    gpu = cup.periodogram((t, mag, err), "GLS", backend="cufinufft")
    assert float(np.max(np.abs(cpu.power - gpu.power))) < 1e-7


@requires_torch
def test_gls_torch_cpu_matches_finufft() -> None:
    # The portable direct trig-sum path must match the NUFFT path in float64.
    t, mag, err = synthetic_sine()
    fi = cup.periodogram((t, mag, err), "GLS", backend="finufft")
    to = cup.periodogram((t, mag, err), "GLS", backend="torch:cpu")
    assert to.backend == "torch:cpu"
    assert float(np.max(np.abs(fi.power - to.power))) < 1e-6


@requires_torch
@torch_auto_device
def test_gls_torch_recovers_period() -> None:
    t, mag, err = synthetic_sine(period=0.6234)
    pg = cup.periodogram((t, mag, err), "GLS", backend="torch")
    assert pg.best_period() == pytest.approx(0.6234, rel=1e-3)


@requires_torch
def test_gls_torch_float32_recovers_period() -> None:
    # float32 is the forced precision on Apple MPS; the direct trig-sum's power drifts
    # but the peak must stay robust (the float32 path previously had no test coverage).
    t, mag, err = synthetic_sine(period=0.6234)
    pg = cup.periodogram(
        (t, mag, err), "GLS", backend="torch:cpu",
        settings=cup.GLSSettings(precision="float32"),
    )
    assert pg.backend == "torch:cpu"
    assert pg.best_period() == pytest.approx(0.6234, rel=1e-3)


# --- regressions --------------------------------------------------------------------


def _long_period_curve(
    freq: float, *, n: int = 400, span: float = 100.0, seed: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n)) + 2458000.0
    dy = np.full(n, 0.03)
    y = 12.0 + 0.4 * np.sin(2 * np.pi * freq * t) + dy * rng.standard_normal(n)
    return t, y, dy


def test_gls_peak_on_a_grid_edge_carries_a_fap() -> None:
    # CORE-B-5 / METH-A-14: a period longer than the baseline peaks at the lowest
    # grid frequency, and best_periods() reports that edge sample -- but the FAP was
    # computed only at interior maxima, so the best peak came back with fap = NaN and
    # any "fap < threshold" cut silently dropped every long-period variable.
    t, y, dy = _long_period_curve(1.0 / 160.0)
    pg = cup.periodogram((t, y, dy), "GLS", backend="finufft")
    peaks = pg.best_periods(3)
    assert peaks[0].frequency == pg.frequency[0]
    assert peaks[0].extra["fap"] < 1e-6
    assert all(np.isfinite(p.extra["fap"]) for p in peaks)
    # Interior peaks keep exactly astropy's value for their power.
    ls = LombScargle(t - t.min(), y, dy, fit_mean=True)
    ref = ls.false_alarm_probability(
        np.array([p.power for p in peaks[1:]]),
        minimum_frequency=float(pg.frequency[0]),
        maximum_frequency=float(pg.frequency[-1]),
    )
    np.testing.assert_allclose([p.extra["fap"] for p in peaks[1:]], ref, rtol=1e-9)

    # The top edge, when maximum_frequency cuts through a peak.
    t, y, dy = _long_period_curve(5.004)
    pg = cup.periodogram(
        (t, y, dy), "GLS", backend="finufft",
        settings=cup.GLSSettings(maximum_frequency=5.0),
    )
    top = pg.best_periods(1)[0]
    assert top.frequency == pg.frequency[-1]
    assert np.isfinite(top.extra["fap"]) and top.extra["fap"] < 1e-6


def test_gls_fap_does_not_depend_on_the_time_origin() -> None:
    # METH-A-1: astropy's Baluev FAP takes the effective baseline from the one-pass
    # variance <t^2> - <t>^2 of the times it is given. On absolute JDs that is
    # rounding noise for a one-hour run (zero or negative: NaN FAPs, or orders of
    # magnitude off). Dyadic times make both origins exactly representable.
    rng = np.random.default_rng(1)
    ticks = np.unique(rng.integers(0, 2**30 // 24, 700))[:600]
    t_rel = ticks * 2.0**-30
    y = rng.normal(0.0, 0.01, t_rel.size)
    dy = np.full(t_rel.size, 0.01)
    rel = cup.periodogram((t_rel, y, dy), "GLS", backend="finufft")
    absolute = cup.periodogram((t_rel + 2.0**21, y, dy), "GLS", backend="finufft")
    fap_rel, fap_abs = rel.extras["fap"], absolute.extras["fap"]
    assert int(np.isfinite(fap_rel).sum()) > 10
    assert np.array_equal(np.isfinite(fap_abs), np.isfinite(fap_rel))
    finite = np.isfinite(fap_rel)
    np.testing.assert_allclose(fap_abs[finite], fap_rel[finite], rtol=1e-12)


def _lattice_curve(
    timing: float = 0.0, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    dt = 1728.0 / 86400.0
    period = 1.25011
    t = 2460000.5 + dt * np.arange(1500)
    # Optional smooth barycentric-like timing term (seconds), coherent over the run.
    t = t + timing / 86400.0 * np.sin(2 * np.pi * (t - t[0]) / 365.25 + 0.3)
    rng = np.random.default_rng(seed)
    y = 0.1 * np.sin(2 * np.pi * t / period) + rng.normal(0.0, 0.05, t.size)
    return t, y, np.full(t.size, 0.05), period, 0.5 / dt


def test_gls_default_grid_stops_at_the_nyquist_of_a_regular_cadence() -> None:
    # METH-A-2: on a strictly regular cadence the spectrum above 0.5/cadence mirrors
    # the band below, every peak has images of the same height, and the default
    # grid (5x astropy's average Nyquist) returned an image as the best period.
    for timing in (0.0, 400.0):
        t, y, dy, period, nyquist = _lattice_curve(timing)
        pg = cup.periodogram((t, y, dy), "GLS", backend="finufft")
        assert pg.frequency[-1] <= nyquist * (1.0 + 1e-3)
        assert pg.best_period() == pytest.approx(period, rel=1e-3)
    # An explicit nyquist_factor (or maximum_frequency) is honoured as before.
    t, y, dy, _, nyquist = _lattice_curve()
    wide = GLSMethod().default_grid(
        cup.LightCurve.from_arrays(t, y, dy), cup.GLSSettings(nyquist_factor=5)
    )
    assert wide.frequency[-1] > 4.0 * nyquist


def test_gls_default_grid_is_size_checked_before_allocation() -> None:
    # SEC-04: one t=0 row multiplies the baseline ~1e4-fold; with an explicit
    # maximum frequency astropy's autofrequency then allocated a 6e8-sample grid.
    t, mag, err = synthetic_sine(n=200)
    t[0] = 0.0
    lc = cup.LightCurve.from_arrays(t, mag, err)
    with pytest.raises(GridTooLargeError, match="GLS default grid"):
        GLSMethod().default_grid(lc, cup.GLSSettings(maximum_frequency=50.0))
    # A minimum above the automatic maximum is named, not an empty grid.
    t, mag, err = synthetic_sine(n=200)
    with pytest.raises(cup.InsufficientDataError, match="minimum frequency"):
        GLSMethod().default_grid(
            cup.LightCurve.from_arrays(t, mag, err),
            cup.GLSSettings(minimum_frequency=1000.0),
        )


def test_gls_power_is_accurate_on_a_large_mean() -> None:
    # METH-A-12: the trig sums used to be formed on un-centred data and corrected by
    # subtracting mean*sum(w*cos) afterwards -- a difference of two large,
    # NUFFT-approximated numbers that lost (mean/scatter)^2 of the precision.
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0.0, 50.0, 3000))
    y = 1.0 + 1e-6 * rng.standard_normal(t.size)
    dy = np.full(t.size, 1e-6)
    pg = cup.periodogram((t, y, dy), "GLS", backend="finufft")
    ref = LombScargle(t, y, dy, fit_mean=True).power(
        pg.frequency[::7], method="cython", normalization="standard"
    )
    assert float(np.max(np.abs(pg.power[::7] - np.asarray(ref)))) < 1e-8

    # A noise-free sine on a 1e4 pedestal: power stays in [0, 1] with a finite FAP.
    t, _, _ = synthetic_sine(n=1300)
    y = 1e4 + 1e-3 * np.sin(2 * np.pi * t / 0.6234)
    pg = cup.periodogram((t, y, np.full(t.size, 1e-4)), "GLS", backend="finufft")
    assert float(pg.power.min()) >= 0.0
    assert float(pg.power.max()) <= 1.0
    best = pg.best_periods(1)[0]
    assert best.period == pytest.approx(0.6234, rel=1e-3)
    assert np.isfinite(best.extra["fap"])


@pytest.mark.parametrize("backend", ["finufft", "astropy", "engine"])
def test_gls_constant_light_curve_has_no_power(backend: str) -> None:
    # Centring a constant series leaves rounding noise of ~1e-16, and a power formed
    # from it is a ratio of rounding errors: astropy (and the centred NUFFT path)
    # reported peaks of ~0.08 that the FAP called overwhelmingly significant.
    t, _, err = synthetic_sine()
    lc = cup.LightCurve.from_arrays(t, np.ones_like(t), err)
    method, settings = GLSMethod(), cup.GLSSettings()
    grid = method.default_grid(lc, settings)
    if backend == "engine":
        engine = method.make_engine("finufft", settings)
        pg = method.power(grid, lc, settings, "finufft", engine=engine)
    else:
        pg = method.power(grid, lc, settings, backend)
    assert float(np.max(pg.power)) == 0.0
    assert pg.best_periods(1)[0].extra["fap"] == pytest.approx(1.0)


def test_gls_power_is_never_negative_on_a_lattice() -> None:
    # METH-A-12: at f = k/cadence the GLS denominator degenerates and round-off made
    # the power slightly negative.
    t = np.arange(400) * 0.02
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 1.0, t.size)
    grid = GridSpec(
        kind="frequency", values=0.5 + 0.25 * np.arange(800), uniform=True
    )
    pg = cup.periodogram((t, y, None), "GLS", backend="finufft", grid=grid)
    assert float(pg.power.min()) >= 0.0
    assert float(pg.power.max()) <= 1.0


def test_small_finufft_transforms_run_on_one_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # METH-A-6 / PDOT-A-6 / PERF-METHODS-02 / PERF-PDOT-01: a light-curve-sized
    # transform is dominated by the OpenMP start-up of a threaded finufft call
    # (~3 ms on a many-core desktop against ~0.3 ms single-threaded).
    import finufft

    seen: list[object] = []
    real = finufft.nufft1d1

    def spy(*args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("nthreads"))
        return real(*args, **kwargs)

    monkeypatch.setattr(finufft, "nufft1d1", spy)
    t, mag, err = synthetic_sine(n=300)
    small = lombscargle_power(t, mag, err, 0.01, 0.002, 4000)
    assert seen == [1, 1]
    ref = LombScargle(t, mag, err, fit_mean=True).power(
        0.01 + 0.002 * np.arange(4000), method="cython", normalization="standard"
    )
    assert float(np.max(np.abs(small - np.asarray(ref)))) < 1e-6

    # A long light curve, or a very fine grid, keeps finufft's default threading.
    seen.clear()
    rng = np.random.default_rng(0)
    long_t = np.sort(rng.uniform(0.0, 100.0, 200_000))
    lombscargle_power(long_t, rng.normal(size=long_t.size), None, 0.01, 0.01, 601)
    lombscargle_power(t, mag, err, 0.01, 1e-4, 300_000)
    assert seen == [None, None, None, None]


def test_finufft_engine_matches_the_one_shot_transform() -> None:
    # PERF-METHODS-09 / PERF-PDOT-01: the plan-reusing CPU engine (what pdot scans
    # and other repeated callers get from make_engine("finufft")).
    t, mag, err = synthetic_sine(n=300)
    method = GLSMethod()
    settings = cup.GLSSettings()
    engine = method.make_engine("finufft", settings)
    assert isinstance(engine, FinufftGLS)
    for nf in (601, 4000, 4001, 601):
        for fit_mean in (True, False):
            ref = lombscargle_power(t, mag, err, 0.01, 0.002, nf, fit_mean=fit_mean)
            got = engine.power(t, mag, err, 0.01, 0.002, nf, fit_mean=fit_mean)
            assert float(np.max(np.abs(got - ref))) < 1e-8
    # 601 and 4000/4001 modes share two plans each (n_trans 1 and 2), reused.
    assert len(engine._plans) == 4
    lc = cup.LightCurve.from_arrays(t, mag, err)
    grid = method.default_grid(lc, settings)
    plain = method.power(grid, lc, settings, "finufft")
    cached = method.power(grid, lc, settings, "finufft", engine=engine)
    assert cached.backend == "finufft"
    assert float(np.max(np.abs(cached.power - plain.power))) < 1e-8
    # The cache is bounded by the modes it holds.
    small = FinufftGLS(max_cached_modes=3000)
    small.power(t, mag, err, 0.01, 0.002, 601)
    small.power(t, mag, err, 0.01, 0.002, 1201)
    assert sum(m * k for m, k in small._plans) <= 3000


def test_gls_uniform_flag_needs_a_frequency_grid() -> None:
    # METH-A-16: the NUFFT path derives (f0, df, nf) from the first two samples. A
    # one-sample period grid flagged uniform reached uniform_frequency_params and
    # raised; multi-sample grids that are not a frequency progression are refused by
    # GridSpec itself.
    t, mag, err = synthetic_sine()
    lc = cup.LightCurve.from_arrays(t, mag, err)
    grid = GridSpec(kind="period", values=np.array([0.6234]), uniform=True)
    pg = GLSMethod().power(grid, lc, cup.GLSSettings(), "finufft")
    assert pg.backend == "astropy"
    assert pg.size == 1 and pg.power[0] > 0.5
    with pytest.raises(ValueError, match="arithmetic progression"):
        GridSpec(kind="frequency", values=np.geomspace(0.01, 1.0, 50), uniform=True)
