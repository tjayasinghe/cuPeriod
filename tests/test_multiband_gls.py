"""Native multi-band GLS: astropy parity, backend parity, and model behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu, requires_torch
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.multiband.gls_mb import gls_multiband_power
from synth import synthetic_multiband_sine

PERIOD = 0.7365
F_TRUE = 1.0 / PERIOD


def make_mblc(**kwargs: object) -> cup.MultiBandLightCurve:
    bands = synthetic_multiband_sine(period=PERIOD, **kwargs)  # type: ignore[arg-type]
    return cup.MultiBandLightCurve.from_light_curves(
        {name: cup.LightCurve.from_arrays(t, m, e) for name, (t, m, e) in bands.items()}
    )


def make_grid(nf: int = 4001, f_max: float = 6.0) -> GridSpec:
    freq = np.linspace(0.05, f_max, nf)
    return GridSpec(kind="frequency", values=freq, uniform=True)


# --- model math vs the astropy reference -------------------------------------


def test_offsets_matches_astropy_flex_1_0() -> None:
    """The closed-form offsets model equals astropy's ridge-regularized (1, 0)."""
    mb, grid = make_mblc(), make_grid()
    s = cup.GLSSettings()
    native = gls_multiband_power(grid, mb, s, "finufft").power
    reference = gls_multiband_power(grid, mb, s, "astropy").power
    assert np.allclose(native, reference, atol=2e-4)


@pytest.mark.parametrize(("nb", "nk"), [(1, 1), (2, 1), (0, 1), (1, 0)])
def test_flex_matches_astropy(nb: int, nk: int) -> None:
    """The sufficient-statistics flex path reproduces astropy's flexible solver."""
    mb, grid = make_mblc(), make_grid()
    s = cup.GLSSettings(mb_model="flex", mb_nterms_base=nb, mb_nterms_band=nk)
    native = gls_multiband_power(grid, mb, s, "finufft").power
    reference = gls_multiband_power(grid, mb, s, "astropy").power
    assert np.allclose(native, reference, atol=1e-8)


def test_flex_absolute_ridge_matches_astropy() -> None:
    """regularize_by_trace=False needs astropy's raw 1/dy^2 normal-matrix scale."""
    mb, grid = make_mblc(), make_grid()
    s = cup.GLSSettings(
        mb_model="flex",
        mb_reg_base=1e-3,
        mb_reg_band=1e-2,
        mb_regularize_by_trace=False,
    )
    native = gls_multiband_power(grid, mb, s, "finufft").power
    reference = gls_multiband_power(grid, mb, s, "astropy").power
    assert np.allclose(native, reference, atol=1e-8)


def test_perband_close_to_astropy_flex_0_1() -> None:
    """The chi2_0-weighted per-band combination is the (0, 1) model up to ridge."""
    mb, grid = make_mblc(), make_grid()
    s = cup.GLSSettings(mb_model="perband")
    native = gls_multiband_power(grid, mb, s, "finufft").power
    reference = gls_multiband_power(grid, mb, s, "astropy").power
    assert np.allclose(native, reference, atol=5e-3)


def test_offsets_single_band_collapses_to_single_band_gls() -> None:
    """With one band, the offsets model is exactly the floating-mean GLS."""
    bands = synthetic_multiband_sine(band_points=(80,), amplitudes=(0.25,),
                                     offsets=(12.0,), seed=7)
    (t, m, e) = bands["b0"]
    lc = cup.LightCurve.from_arrays(t, m, e)
    mb1 = cup.MultiBandLightCurve.from_light_curves({"g": lc})
    grid = make_grid()
    pg_mb = gls_multiband_power(grid, mb1, cup.GLSSettings(), "finufft")
    pg_sb = cup.periodogram(lc, "GLS", backend="finufft", grid=grid)
    assert isinstance(pg_sb, cup.Periodogram)
    assert np.allclose(pg_mb.power, pg_sb.power, atol=1e-9)


@pytest.mark.parametrize("model", ["offsets", "perband", "flex"])
def test_models_recover_planted_period(model: str) -> None:
    mb, grid = make_mblc(), make_grid()
    s = cup.GLSSettings(mb_model=model)  # type: ignore[arg-type]
    pg = gls_multiband_power(grid, mb, s, "finufft")
    assert pg.best_period() == pytest.approx(PERIOD, rel=2e-3)


def test_unweighted_bands_match_astropy() -> None:
    """Bands without errors run unweighted and still match the reference."""
    bands = synthetic_multiband_sine()
    mb = cup.MultiBandLightCurve.from_light_curves(
        {name: cup.LightCurve.from_arrays(t, m) for name, (t, m, _) in bands.items()}
    )
    grid = make_grid(2001)
    s = cup.GLSSettings()
    native = gls_multiband_power(grid, mb, s, "finufft").power
    reference = gls_multiband_power(grid, mb, s, "astropy").power
    assert np.allclose(native, reference, atol=2e-4)


# --- backend parity ----------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("model", ["offsets", "perband", "flex"])
def test_cufinufft_matches_finufft(model: str) -> None:
    mb, grid = make_mblc(), make_grid()
    s = cup.GLSSettings(mb_model=model)  # type: ignore[arg-type]
    cpu = gls_multiband_power(grid, mb, s, "finufft").power
    gpu = gls_multiband_power(grid, mb, s, "cufinufft").power
    assert np.allclose(gpu, cpu, atol=1e-8)


@requires_torch
@pytest.mark.parametrize("model", ["offsets", "perband", "flex"])
def test_torch_cpu_matches_finufft(model: str) -> None:
    mb, grid = make_mblc(), make_grid()
    s = cup.GLSSettings(mb_model=model)  # type: ignore[arg-type]
    cpu = gls_multiband_power(grid, mb, s, "finufft").power
    torch_power = gls_multiband_power(grid, mb, s, "torch:cpu").power
    assert np.allclose(torch_power, cpu, atol=1e-8)


# --- batch engine reuse ------------------------------------------------------


@pytest.mark.parametrize("model", ["offsets", "perband", "flex"])
def test_foreign_engine_ignored_on_cpu(model: str) -> None:
    """A non-cufinufft engine (any object) must not change the CPU result."""
    mb, grid = make_mblc(), make_grid(1001)
    s = cup.GLSSettings(mb_model=model)  # type: ignore[arg-type]
    plain = gls_multiband_power(grid, mb, s, "finufft").power
    with_engine = gls_multiband_power(
        grid, mb, s, "finufft", engine=object()
    ).power
    np.testing.assert_array_equal(with_engine, plain)


@requires_gpu
@pytest.mark.parametrize("model", ["offsets", "perband", "flex"])
def test_cufinufft_engine_matches_planless(model: str) -> None:
    """The plan-cached engine path returns the plan-per-call cufinufft result."""
    from cuperiod.methods.gls import CufinufftGLS

    mb, grid = make_mblc(), make_grid()
    s = cup.GLSSettings(mb_model=model)  # type: ignore[arg-type]
    engine = CufinufftGLS(eps=s.nufft_eps)
    plain = gls_multiband_power(grid, mb, s, "cufinufft").power
    reused = gls_multiband_power(grid, mb, s, "cufinufft", engine=engine).power
    # A second run through the now-warm plan cache must be deterministic.
    again = gls_multiband_power(grid, mb, s, "cufinufft", engine=engine).power
    assert np.allclose(reused, plain, atol=1e-8)
    np.testing.assert_array_equal(again, reused)


# --- API wiring and guards ---------------------------------------------------


def test_periodogram_api_uses_native_backend() -> None:
    """cup.periodogram on a MultiBandLightCurve no longer falls back to astropy."""
    mb = make_mblc()
    pg = cup.periodogram(mb, "GLS", backend="cpu")
    assert isinstance(pg, cup.Periodogram)
    assert pg.backend == "finufft"
    assert pg.best_period() == pytest.approx(PERIOD, rel=2e-3)
    assert pg.meta["mb_model"] == "offsets"
    assert tuple(pg.meta["bands"]) == ("b0", "b1", "b2")


def test_astropy_backend_still_available() -> None:
    mb = make_mblc()
    pg = cup.periodogram(mb, "GLS", backend="astropy", grid=make_grid(1001))
    assert isinstance(pg, cup.Periodogram)
    assert pg.backend == "astropy"


def test_non_uniform_grid_falls_back_to_astropy() -> None:
    mb = make_mblc()
    freq = np.geomspace(0.1, 6.0, 800)
    grid = GridSpec(kind="frequency", values=freq, uniform=False)
    pg = gls_multiband_power(grid, mb, cup.GLSSettings(), "finufft")
    assert pg.backend == "astropy"


def test_empty_grid_raises() -> None:
    mb = make_mblc()
    grid = GridSpec(kind="frequency", values=np.empty(0), uniform=True)
    with pytest.raises(InsufficientDataError):
        gls_multiband_power(grid, mb, cup.GLSSettings(), "finufft")


def test_too_few_points_raises() -> None:
    bands = synthetic_multiband_sine(band_points=(3, 2), amplitudes=(0.3, 0.2),
                                     offsets=(15.0, 14.0))
    mb = cup.MultiBandLightCurve.from_light_curves(
        {n: cup.LightCurve.from_arrays(t, m, e) for n, (t, m, e) in bands.items()}
    )
    with pytest.raises(InsufficientDataError):
        gls_multiband_power(make_grid(101), mb, cup.GLSSettings(), "finufft")


def test_flex_zero_terms_rejected() -> None:
    with pytest.raises(ValueError, match="mb_nterms"):
        cup.GLSSettings(mb_model="flex", mb_nterms_base=0, mb_nterms_band=0)
