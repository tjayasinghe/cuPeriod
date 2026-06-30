"""GLS: astropy parity, period recovery, multi-band, and GPU agreement."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.timeseries import LombScargle

import cuperiod as cup
from conftest import requires_gpu, requires_torch
from cuperiod.methods.gls import GLSMethod, lombscargle_power
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
def test_gls_torch_recovers_period() -> None:
    t, mag, err = synthetic_sine(period=0.6234)
    pg = cup.periodogram((t, mag, err), "GLS", backend="torch")
    assert pg.best_period() == pytest.approx(0.6234, rel=1e-3)
