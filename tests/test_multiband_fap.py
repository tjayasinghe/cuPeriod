"""Within-band bootstrap false-alarm statistics for the multi-band GLS."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu
from cuperiod.core.grid import GridSpec
from cuperiod.multiband.fap_mb import MultibandFAP
from synth import synthetic_multiband_sine

PERIOD = 0.7365


def make_grid(nf: int = 3001) -> GridSpec:
    return GridSpec(
        kind="frequency", values=np.linspace(0.05, 6.0, nf), uniform=True
    )


def signal_mblc() -> cup.MultiBandLightCurve:
    bands = synthetic_multiband_sine(period=PERIOD)
    return cup.MultiBandLightCurve.from_light_curves(
        {n: cup.LightCurve.from_arrays(t, m, e) for n, (t, m, e) in bands.items()}
    )


def noise_mblc(seed: int = 5) -> cup.MultiBandLightCurve:
    bands = synthetic_multiband_sine(amplitudes=(0.0, 0.0, 0.0), seed=seed)
    return cup.MultiBandLightCurve.from_light_curves(
        {n: cup.LightCurve.from_arrays(t, m, e) for n, (t, m, e) in bands.items()}
    )


def test_planted_signal_hits_resolution_floor() -> None:
    grid = make_grid()
    calib = cup.multiband_fap(
        signal_mblc(), grid=grid, backend="finufft", n_bootstrap=100, seed=1
    )
    pg = cup.periodogram(signal_mblc(), "GLS", backend="finufft", grid=grid)
    assert isinstance(pg, cup.Periodogram)
    fap = calib.fap(float(pg.power.max()))
    assert fap == pytest.approx(1.0 / 101.0)


def test_noise_is_not_flagged() -> None:
    grid = make_grid()
    calib = cup.multiband_fap(
        noise_mblc(), grid=grid, backend="finufft", n_bootstrap=100, seed=1
    )
    pg = cup.periodogram(noise_mblc(), "GLS", backend="finufft", grid=grid)
    assert isinstance(pg, cup.Periodogram)
    assert calib.fap(float(pg.power.max())) > 0.02


def test_seed_determinism_and_level() -> None:
    grid = make_grid(1001)
    a = cup.multiband_fap(
        noise_mblc(), grid=grid, backend="finufft", n_bootstrap=60, seed=9
    )
    b = cup.multiband_fap(
        noise_mblc(), grid=grid, backend="finufft", n_bootstrap=60, seed=9
    )
    assert np.array_equal(a.null_max, b.null_max)
    level = a.level(0.1)
    assert a.fap(level) <= 0.2
    with pytest.raises(ValueError, match="n_bootstrap"):
        a.level(1e-4)


def test_settings_wire_fap_into_extras() -> None:
    grid = make_grid(2001)
    settings = cup.GLSSettings(mb_fap_bootstrap=50, mb_fap_seed=2)
    pg = cup.periodogram(
        signal_mblc(), "GLS", backend="finufft", grid=grid, settings=settings
    )
    assert isinstance(pg, cup.Periodogram)
    fap = pg.extras["fap"]
    best = int(np.argmax(pg.power))
    assert np.isfinite(fap[best]) and fap[best] <= 1.0 / 51.0 + 1e-12
    assert np.isnan(fap).sum() > fap.size // 2  # NaN away from local maxima
    assert pg.meta["fap_method"] == "bootstrap"
    assert "fap_level_10pct" in pg.meta


@requires_gpu
def test_gpu_null_distribution_matches_cpu() -> None:
    grid = make_grid(2001)
    cpu = cup.multiband_fap(
        signal_mblc(), grid=grid, backend="finufft", n_bootstrap=50, seed=3
    )
    gpu = cup.multiband_fap(
        signal_mblc(), grid=grid, backend="cufinufft", n_bootstrap=50, seed=3
    )
    assert np.allclose(cpu.null_max, gpu.null_max, atol=1e-7)


def test_single_band_input_supported() -> None:
    bands = synthetic_multiband_sine(band_points=(70,), amplitudes=(0.3,),
                                     offsets=(14.0,))
    (t, m, e) = bands["b0"]
    lc = cup.LightCurve.from_arrays(t, m, e)
    calib = cup.multiband_fap(
        lc, grid=make_grid(1001), backend="finufft", n_bootstrap=30, seed=0
    )
    assert calib.null_max.size == 30
    assert calib.model == "offsets"


def test_astropy_backend_rejected() -> None:
    with pytest.raises(ValueError, match="native backend"):
        cup.multiband_fap(signal_mblc(), backend="astropy", n_bootstrap=10)


def test_settings_backend_is_honoured() -> None:
    # backend="auto" defers to settings.backend, as cup.periodogram does.
    with pytest.raises(ValueError, match="native backend"):
        cup.multiband_fap(
            signal_mblc(), cup.GLSSettings(backend="astropy"), n_bootstrap=3
        )


# --- MULTIBAND-9: the settings route never drops a requested FAP ---------------


@pytest.mark.parametrize(
    ("backend", "uniform"), [("astropy", True), ("finufft", False)]
)
def test_settings_bootstrap_runs_on_every_route(backend: str, uniform: bool) -> None:
    # The astropy backend, and any non-uniform grid (which reroutes to astropy),
    # used to skip the bootstrap without a trace: no extras["fap"], no meta.
    grid = GridSpec(
        kind="frequency", values=np.linspace(0.05, 6.0, 200), uniform=uniform
    )
    settings = cup.GLSSettings(mb_fap_bootstrap=5, mb_fap_seed=1)
    pg = cup.periodogram(
        signal_mblc(), "GLS", backend=backend, grid=grid, settings=settings
    )
    assert isinstance(pg, cup.Periodogram)
    assert pg.backend == "astropy"
    assert "fap" in pg.extras
    assert pg.meta["fap_method"] == "bootstrap"
    assert np.isfinite(pg.best_periods(1)[0].extra["fap"])


def test_nonuniform_grid_calibration_is_labelled_astropy() -> None:
    grid = GridSpec(kind="frequency", values=np.geomspace(0.1, 6.0, 150))
    calib = cup.multiband_fap(
        signal_mblc(), grid=grid, backend="finufft", n_bootstrap=3, seed=0
    )
    assert calib.backend == "astropy"
    assert np.all(np.isfinite(calib.null_max))


# --- MULTIBAND-10: resamples are drawn batch by batch ---------------------------


def _block_draw_null(
    mblc: cup.MultiBandLightCurve, grid: GridSpec, n_bootstrap: int, seed: int
) -> np.ndarray:
    """The bootstrap null by brute force: one block draw per band from a single
    stream, then one ordinary power evaluation per resample."""
    from cuperiod.multiband.gls_mb import gls_multiband_power

    rng = np.random.default_rng(seed)
    bands = [(k, lc) for k, lc in mblc.finite().bands.items() if lc.n > 0]
    draws = {k: rng.integers(0, lc.n, size=(n_bootstrap, lc.n)) for k, lc in bands}
    out = np.empty(n_bootstrap)
    for r in range(n_bootstrap):
        resampled = cup.MultiBandLightCurve.from_light_curves(
            {
                k: cup.LightCurve.from_arrays(
                    lc.time, lc.value[draws[k][r]], lc.error[draws[k][r]]
                )
                for k, lc in bands
            }
        )
        pg = gls_multiband_power(grid, resampled, cup.GLSSettings(), "finufft")
        out[r] = pg.power.max()
    return out


def test_batched_null_is_chunk_size_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cuperiod.multiband.fap_mb as fap_mb

    grid = make_grid(801)
    big = cup.multiband_fap(
        signal_mblc(), grid=grid, backend="finufft", n_bootstrap=12, seed=4
    ).null_max
    # A budget this small forces one resample per batch.
    monkeypatch.setattr(fap_mb, "_FAP_CHUNK_BYTES", 1 << 12)
    small = cup.multiband_fap(
        signal_mblc(), grid=grid, backend="finufft", n_bootstrap=12, seed=4
    ).null_max
    np.testing.assert_array_equal(small, big)
    # Same draws as the original one-block-per-band definition of the resamples.
    reference = _block_draw_null(signal_mblc(), grid, 12, 4)
    np.testing.assert_allclose(small, reference, rtol=1e-9, atol=1e-12)


def test_batched_null_memory_does_not_grow_with_n_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tracemalloc

    import cuperiod.multiband.fap_mb as fap_mb

    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 300.0, 10_000))
    lc = cup.LightCurve.from_arrays(t, rng.normal(12.0, 0.05, t.size),
                                    np.full(t.size, 0.05))
    grid = make_grid(201)
    # ~2 MB per batch: every run below needs several batches.
    monkeypatch.setattr(fap_mb, "_FAP_CHUNK_BYTES", 1 << 21)

    def peak(n_bootstrap: int) -> int:
        tracemalloc.start()
        try:
            cup.multiband_fap(lc, grid=grid, backend="finufft",
                              n_bootstrap=n_bootstrap, seed=0)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    small, large = peak(16), peak(128)
    # Materializing every resample up front grew the peak ~8x here.
    assert large < 1.5 * small


# --- MULTIBAND-13: FAP accessors at their edges ---------------------------------


def _calibration(null_max: np.ndarray) -> MultibandFAP:
    return MultibandFAP(
        null_max=null_max, n_bootstrap=null_max.size, seed=0, model="offsets",
        backend="finufft",
    )


def test_fap_of_nan_is_nan() -> None:
    calib = _calibration(np.linspace(0.1, 0.5, 99))
    assert np.isnan(calib.fap(float("nan")))
    out = calib.fap(np.array([0.05, np.nan, 0.6]))
    assert out[0] == pytest.approx(1.0) and np.isnan(out[1])
    assert out[2] == pytest.approx(0.01)


def test_level_is_the_order_statistic_fap_implies() -> None:
    rng = np.random.default_rng(1)
    for _ in range(300):
        n_bootstrap = int(rng.integers(1, 400))
        calib = _calibration(rng.random(n_bootstrap))
        p = float(rng.uniform(1.0 / (n_bootstrap + 1), 1.0))
        level = calib.level(p)
        # Any power above the level is at least as significant as asked for.
        assert calib.fap(np.nextafter(level, np.inf)) <= p + 1e-12


def test_level_at_the_resolution_floor() -> None:
    rng = np.random.default_rng(2)
    for n_bootstrap in range(1, 401):
        null_max = rng.random(n_bootstrap)
        calib = _calibration(null_max)
        assert calib.level(1.0 / (n_bootstrap + 1)) == null_max.max()
        assert np.isfinite(calib.level(1.0))


@pytest.mark.parametrize("n_bootstrap", [0, -1])
def test_n_bootstrap_must_be_positive(n_bootstrap: int) -> None:
    from cuperiod.multiband.fap_mb import bootstrap_null_max

    with pytest.raises(ValueError, match="n_bootstrap"):
        cup.multiband_fap(signal_mblc(), grid=make_grid(101), backend="finufft",
                          n_bootstrap=n_bootstrap)
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_null_max(signal_mblc(), make_grid(101), cup.GLSSettings(),
                           "finufft", n_bootstrap, 0)


def test_meta_levels_only_where_resolved() -> None:
    grid = make_grid(301)
    low = cup.periodogram(signal_mblc(), "GLS", backend="finufft", grid=grid,
                          settings=cup.GLSSettings(mb_fap_bootstrap=3))
    assert isinstance(low, cup.Periodogram)
    assert "fap_level_10pct" not in low.meta  # 1/(3+1) > 0.1: not resolvable
    settings = cup.GLSSettings(mb_fap_bootstrap=9, mb_fap_seed=5)
    high = cup.periodogram(signal_mblc(), "GLS", backend="finufft", grid=grid,
                           settings=settings)
    assert isinstance(high, cup.Periodogram)
    calib = cup.multiband_fap(signal_mblc(), settings, grid=grid,
                              backend="finufft", n_bootstrap=9, seed=5)
    assert high.meta["fap_level_10pct"] == calib.level(0.1)


def test_peak_on_the_grid_edge_gets_a_fap() -> None:
    # A grid ending at the true frequency puts the best peak on its last sample,
    # which best_periods() reports but the FAP annotation used to skip (NaN).
    grid = GridSpec(
        kind="frequency", values=np.linspace(0.05, 1.0 / PERIOD, 600), uniform=True
    )
    settings = cup.GLSSettings(mb_fap_bootstrap=20, mb_fap_seed=0)
    pg = cup.periodogram(signal_mblc(), "GLS", backend="finufft", grid=grid,
                         settings=settings)
    assert isinstance(pg, cup.Periodogram)
    best = pg.best_periods(1)[0]
    assert best.frequency == pytest.approx(grid.values[-1])
    assert np.isfinite(best.extra["fap"])


def test_flex_model_uses_loop_path() -> None:
    settings = cup.GLSSettings(mb_model="flex")
    calib = cup.multiband_fap(
        signal_mblc(), settings, grid=make_grid(501), backend="finufft",
        n_bootstrap=5, seed=4,
    )
    assert calib.model == "flex"
    assert calib.null_max.size == 5
    assert np.all(calib.null_max > 0.0)
