"""Within-band bootstrap false-alarm statistics for the multi-band GLS."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu
from cuperiod.core.grid import GridSpec
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


def test_flex_model_uses_loop_path() -> None:
    settings = cup.GLSSettings(mb_model="flex")
    calib = cup.multiband_fap(
        signal_mblc(), settings, grid=make_grid(501), backend="finufft",
        n_bootstrap=5, seed=4,
    )
    assert calib.model == "flex"
    assert calib.null_max.size == 5
    assert np.all(calib.null_max > 0.0)
