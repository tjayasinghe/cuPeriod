"""BLS: astropy parity, period recovery, multi-band, and GPU agreement."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu, requires_torch
from cuperiod.methods._bls_core import bls_power
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
def test_bls_torch_recovers_period_and_depth() -> None:
    t, flux, err = synthetic_eclipser(period=2.5, depth=0.05)
    pg = cup.periodogram((t, flux, err), "BLS", domain=cup.Domain.FLUX, backend="torch")
    peak = pg.best_periods(1, alias_diverse=True)[0]
    assert peak.period == pytest.approx(2.5, rel=3e-3)
    assert peak.extra["depth"] == pytest.approx(0.05, abs=0.01)
