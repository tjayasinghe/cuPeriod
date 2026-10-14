"""MHAOV: parity with a direct least-squares reference, recovery, max sense."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
import cuperiod.methods.mhaov as mhaov_mod
from conftest import requires_gpu, requires_numba, requires_torch
from cuperiod.methods.mhaov import aov_multiband_power, aov_power
from synth import synthetic_eclipser, synthetic_sine


def _offset_sine(
    mean: float, *, n: int = 600, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """A 0.1-amplitude sine, 0.02 scatter, around ``mean`` (100 d, random cadence)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 100.0, n))
    y = mean + 0.1 * np.sin(2 * np.pi * t / 0.6234) + rng.normal(0.0, 0.02, n)
    return t, y


def _reference_aov(t: np.ndarray, y: np.ndarray, freq: float, h: int) -> float:
    """Transparent least-squares AOV F-statistic for one frequency."""
    n = t.size
    d = 2 * h + 1
    tau = t - t.min()
    ang = 2 * np.pi * freq * tau
    design = np.ones((n, d))
    for k in range(1, h + 1):
        design[:, 2 * k - 1] = np.cos(k * ang)
        design[:, 2 * k] = np.sin(k * ang)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    yhat = design @ beta
    ybar = y.mean()
    model = float(((yhat - ybar) ** 2).sum())
    resid = float(((y - yhat) ** 2).sum())
    return (model / (2 * h)) / (resid / (n - d))


def test_mhaov_matches_reference() -> None:
    t, mag, _ = synthetic_sine(n=300)
    freqs = np.linspace(0.2, 3.0, 120)
    mine = aov_power(t, mag, freqs, n_harmonics=3)
    ref = np.array([_reference_aov(t, mag, float(f), 3) for f in freqs])
    assert np.allclose(mine, ref, rtol=1e-5, atol=1e-4)


def test_mhaov_recovers_eclipser() -> None:
    # Sharp eclipses need several harmonics, which is exactly MHAOV's strength.
    t, flux, err = synthetic_eclipser(period=2.5)
    pg = cup.periodogram((t, flux, err), "MHAOV", domain="flux")
    assert pg.objective_sense == "max"
    assert pg.best_period() == pytest.approx(2.5, rel=5e-3)


def test_mhaov_detects_sine_in_bounded_range() -> None:
    # On a pure sinusoid the fundamental and its subharmonics share the same AOV, so
    # the search is bounded above the first subharmonic to isolate the fundamental.
    t, mag, err = synthetic_sine(period=0.6234)
    pg = cup.periodogram(
        (t, mag, err), "MHAOV", settings=cup.MHAOVSettings(minimum_frequency=1.0)
    )
    assert pg.best_period() == pytest.approx(0.6234, rel=2e-3)


def test_mhaov_peaks_at_true_frequency() -> None:
    t, flux, _ = synthetic_eclipser(period=2.5)
    at_true = aov_power(t, flux, np.array([1.0 / 2.5]))[0]
    off = aov_power(t, flux, np.array([1.0 / 1.7]))[0]
    assert at_true > 5.0 * off


def test_mhaov_multiband_recovers_period() -> None:
    period = 0.6234
    tg, mg, eg = synthetic_sine(n=300, period=period, seed=2)
    tr, mr, er = synthetic_sine(n=300, period=period, amp=0.3, seed=3)
    mb = cup.MultiBandLightCurve.from_light_curves(
        {
            "g": cup.LightCurve.from_arrays(tg, mg, eg),
            "r": cup.LightCurve.from_arrays(tr, mr + 1.0, er),
        }
    )
    # bounded above the first subharmonic, as for the single-band sine
    pg = cup.periodogram(
        mb, "MHAOV", settings=cup.MHAOVSettings(minimum_frequency=1.0)
    )
    assert "bands" in pg.meta
    assert pg.best_period() == pytest.approx(period, rel=2e-3)


def test_mhaov_supports_multiband() -> None:
    assert cup.get_method("MHAOV").supports_multiband


def test_mhaov_too_few_points() -> None:
    t = np.linspace(0.0, 10.0, 5)
    out = aov_power(t, np.arange(5.0), np.array([0.5, 1.0]), n_harmonics=3)
    assert np.all(out == 0.0)


def test_aov_multiband_power_pools_each_band_in_its_own_variance_units() -> None:
    # Cross-group MULTIBAND-1: the public pooled statistic summed raw sums of squares,
    # so a band in flux units (or just a noisier band) outweighed a magnitude band;
    # each band now enters in its own variance units, and an exactly constant band
    # (whose rounding-level total SS used to count towards the dof) is left out.
    t1, m1, _ = synthetic_sine(n=300, seed=1)
    t2, m2, _ = synthetic_sine(n=250, amp=0.2, seed=2)
    freqs = np.linspace(0.2, 3.0, 300)

    def unit(y: np.ndarray) -> np.ndarray:
        return (y - y.mean()) / y.std(ddof=1)

    want = aov_multiband_power(freqs, [(t1, unit(m1)), (t2, unit(m2))])
    raw = aov_multiband_power(freqs, [(t1, m1), (t2, 1e4 * m2 + 3e5)])
    np.testing.assert_allclose(raw, want, rtol=1e-8)
    t3 = np.linspace(0.0, 30.0, 40)
    with_flat = aov_multiband_power(
        freqs, [(t1, m1), (t2, 1e4 * m2 + 3e5), (t3, np.full(40, 0.1))]
    )
    np.testing.assert_allclose(with_flat, want, rtol=1e-8)


def test_mhaov_reports_min_detections_and_harmonics_separately() -> None:
    # Regression: both limits used to say "too few for 3 harmonics", although 15
    # points are plenty for 3 harmonics (7 terms); the limit hit was min_detections.
    t, mag, err = synthetic_sine(n=15)
    with pytest.raises(cup.InsufficientDataError, match="< min_detections 20"):
        cup.periodogram((t, mag, err), "MHAOV", backend="numpy")
    t, mag, err = synthetic_sine(n=7)
    with pytest.raises(
        cup.InsufficientDataError, match="too few for 3 harmonics.*needs at least 8"
    ):
        cup.periodogram(
            (t, mag, err), "MHAOV", backend="numpy",
            settings=cup.MHAOVSettings(min_detections=5),
        )


@requires_torch
@pytest.mark.parametrize("mean", [1.0, 15.0, 1e5])
def test_mhaov_float32_tracks_float64_at_any_mean(mean: float) -> None:
    # Regression: the normal equations were solved on the raw values, and the ridge on
    # the constant column biased the projection by ~ridge*mean^2 -- in float32 more
    # than the whole variance once mean/scatter exceeds ~90, so clip() returned an
    # all-zero spectrum (magnitudes, normalised flux; float32 is automatic on MPS).
    t, y = _offset_sine(mean)
    freqs = np.linspace(1.0, 3.0, 400)
    ref = aov_power(t, y, freqs, backend="numpy")
    f32 = aov_power(t, y, freqs, backend="torch:cpu", precision="float32")
    assert int(np.argmax(f32)) == int(np.argmax(ref))
    assert float(f32.max()) == pytest.approx(float(ref.max()), rel=0.05)
    assert np.count_nonzero(f32 > 0.0) > 0.9 * f32.size


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("numba", marks=requires_numba)]
)
def test_mhaov_is_offset_invariant(backend: str) -> None:
    # The statistic does not depend on the mean level; in float64 the same ridge bias
    # used to halve the peak at mean 1e4 and zero the spectrum at 1e6.
    t, y = _offset_sine(0.0)
    freqs = np.linspace(1.0, 3.0, 300)
    base = aov_power(t, y, freqs, backend=backend)
    for offset in (15.0, 1e6):
        shifted = aov_power(t, y + offset, freqs, backend=backend)
        np.testing.assert_allclose(shifted, base, rtol=1e-8, atol=1e-8 * base.max())


def test_mhaov_numpy_backend_runs_without_device_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: the numpy branch handed the raw numpy module to the array-API kernel,
    # whose xp.empty/xp.zeros(..., device=) need NumPy >= 2.0 (the declared floor is
    # 1.24). Emulate NumPy 1.x creation functions: the result must not change.
    t, mag, _ = synthetic_sine(n=200)
    freqs = np.linspace(0.5, 3.0, 60)
    want = aov_power(t, mag, freqs, backend="numpy")

    def numpy1(fn):  # type: ignore[no-untyped-def]
        def creation(*args, **kwargs):  # type: ignore[no-untyped-def]
            if "device" in kwargs:
                raise TypeError(
                    f"{fn.__name__}() got an unexpected keyword argument 'device'"
                )
            return fn(*args, **kwargs)

        return creation

    monkeypatch.setattr(np, "empty", numpy1(np.empty))
    monkeypatch.setattr(np, "zeros", numpy1(np.zeros))
    assert np.array_equal(aov_power(t, mag, freqs, backend="numpy"), want)
    # ...also on a grid that is not uniform (the direct vectorized sums).
    periods = np.linspace(0.4, 2.0, 60)
    monkeypatch.undo()
    want = aov_power(t, mag, 1.0 / periods, backend="numpy")
    monkeypatch.setattr(np, "empty", numpy1(np.empty))
    monkeypatch.setattr(np, "zeros", numpy1(np.zeros))
    assert np.array_equal(aov_power(t, mag, 1.0 / periods, backend="numpy"), want)


# --- NUFFT harmonic sums on uniform grids (PERF-METHODS-08) -----------------------


def _force_path(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    """Make the CPU backends take the NUFFT sums ("nufft") or never ("direct")."""
    if path == "nufft":
        monkeypatch.setattr(mhaov_mod, "_NUFFT_MIN_FREQS", 0)
        monkeypatch.setattr(mhaov_mod, "_NUFFT_POINTS_PER_NUMBA_THREAD", 0)
    else:
        monkeypatch.setattr(mhaov_mod, "_NUFFT_MIN_FREQS", 10**12)


@pytest.mark.parametrize(
    ("label", "mean", "amp", "noise"),
    [("magnitudes", 12.0, 0.3, 0.05), ("flux at 100 ppm", 1.0, 1e-4, 5e-5)],
)
@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("numba", marks=requires_numba)]
)
def test_mhaov_nufft_sums_match_the_direct_sums(
    backend: str, label: str, mean: float, amp: float, noise: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The type-1 NUFFT path (uniform grids on the CPU backends) solves the same
    # normal equations from sums good to 1e-13; on centred values that agrees with
    # the direct O(F*N*H) sums to ~1e-11, also where mean^2/variance is ~1e8.
    # (From 1/T, as the default grids: below one cycle per baseline the normal
    # equations themselves lose ~1e-5 -- both sums alike -- to the ridge conditioning.)
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0.0, 300.0, 700)) + 2458000.0
    y = mean + amp * np.sin(2 * np.pi * t / 0.6234) + rng.normal(0.0, noise, t.size)
    baseline = float(t.max() - t.min())
    freqs = cup.uniform_frequency_grid(
        baseline, maximum_frequency=4.0, minimum_frequency=1.0 / baseline
    ).frequency
    _force_path(monkeypatch, "direct")
    direct = aov_power(t, y, freqs, backend=backend)
    _force_path(monkeypatch, "nufft")
    nufft = aov_power(t, y, freqs, backend=backend)
    np.testing.assert_allclose(nufft, direct, rtol=1e-9, atol=0.0)
    assert int(np.argmax(nufft)) == int(np.argmax(direct))


def test_mhaov_nufft_matches_the_least_squares_reference() -> None:
    # The default numpy path on a uniform grid is the NUFFT one; pin it to the
    # transparent least-squares fit, as the direct path is pinned above.
    t, mag, _ = synthetic_sine(n=300)
    freqs = np.linspace(0.2, 3.0, 400)
    assert mhaov_mod._uniform_grid_params(freqs) is not None
    mine = aov_power(t, mag, freqs, n_harmonics=3, backend="numpy")
    ref = np.array([_reference_aov(t, mag, float(f), 3) for f in freqs[::20]])
    np.testing.assert_allclose(mine[::20], ref, rtol=1e-8)


def test_mhaov_nufft_chunks_and_multiband(monkeypatch: pytest.MonkeyPatch) -> None:
    # Grids longer than one NUFFT chunk are summed chunk by chunk, and the pooled
    # multiband statistic goes through the same per-band sums.
    t, mag, _ = synthetic_sine(n=300)
    freqs = np.linspace(0.2, 3.0, 1000)
    whole = aov_power(t, mag, freqs, backend="numpy")
    monkeypatch.setattr(mhaov_mod, "_NUFFT_CHUNK", 97)
    np.testing.assert_allclose(
        aov_power(t, mag, freqs, backend="numpy"), whole, rtol=1e-10
    )
    tr, mr, _ = synthetic_sine(n=250, amp=0.2, seed=5)
    pooled = aov_multiband_power(freqs, [(t, mag), (tr, mr + 1.0)], backend="numpy")
    _force_path(monkeypatch, "direct")
    direct = aov_multiband_power(freqs, [(t, mag), (tr, mr + 1.0)], backend="numpy")
    np.testing.assert_allclose(pooled, direct, rtol=1e-9)


def test_mhaov_nufft_only_for_uniform_grids_and_where_it_pays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uniform = cup.uniform_frequency_grid(100.0, maximum_frequency=5.0).frequency
    assert mhaov_mod._uniform_grid_params(uniform) == pytest.approx(
        (uniform[0], uniform[1] - uniform[0], uniform.size)
    )
    assert mhaov_mod._uniform_grid_params(np.linspace(0.1, 3.0, 500)) is not None
    periods = np.linspace(0.3, 3.0, 500)
    assert mhaov_mod._uniform_grid_params(1.0 / periods) is None  # a period grid
    assert mhaov_mod._uniform_grid_params(np.array([1.0])) is None
    holed = np.delete(np.linspace(0.1, 3.0, 500), 250)
    assert mhaov_mod._uniform_grid_params(holed) is None
    # numpy always (above a token grid size); numba only for long curves.
    assert mhaov_mod._use_nufft("numpy", 50, 1000)
    assert not mhaov_mod._use_nufft("numpy", 50, 10)
    assert not mhaov_mod._use_nufft("torch:cpu", 10**6, 10**6)
    assert not mhaov_mod._use_nufft("cupy", 10**6, 10**6)
    calls: list[int] = []
    real = mhaov_mod._model_ss_nufft

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mhaov_mod, "_model_ss_nufft", spy)
    t, mag, _ = synthetic_sine(n=200)
    aov_power(t, mag, uniform, backend="numpy")
    assert calls == [1]
    aov_power(t, mag, 1.0 / periods[::-1], backend="numpy")
    assert calls == [1]  # the period grid took the direct sums


@requires_numba
def test_mhaov_numba_switches_to_nufft_by_points_per_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numba

    threads = numba.get_num_threads()
    per = mhaov_mod._NUFFT_POINTS_PER_NUMBA_THREAD
    assert not mhaov_mod._use_nufft("numba", per * threads - 1, 10_000)
    assert mhaov_mod._use_nufft("numba", per * threads, 10_000)


@requires_gpu
def test_mhaov_gpu_matches_cpu() -> None:
    t, mag, _ = synthetic_sine()
    freqs = np.linspace(0.2, 3.0, 400)
    cpu = aov_power(t, mag, freqs, backend="numpy")
    gpu = aov_power(t, mag, freqs, backend="cupy")
    assert np.allclose(cpu, gpu, rtol=1e-6, atol=1e-6)
