"""Multi-band Box Least Squares: error scale, custom grids, and the box geometry.

The multi-band statistic is the quadrature sum of the bands' depth-SNRs, so every
band's SNR must be dimensionless: a band without errors gets a robust scatter as its
error (MULTIBAND-2), and a set mixing bands with and without errors runs all bands
that way (MULTIBAND-6). A user grid is searched as given (MULTIBAND-4), and the
band that supplies each period's box geometry is recorded (MULTIBAND-5/12).
"""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup

PERIOD = 2.5
BASE_JD = 2458000.0

#: A period range around the planted eclipse that keeps every run small.
SETTINGS = cup.BLSSettings(min_period_days=0.5, max_period_days=10.0)

#: A flat (user) period grid bracketing the planted period.
FLAT_GRID = cup.GridSpec(kind="period", values=np.linspace(2.0, 3.0, 401))


def _eclipser(
    n: int,
    depth: float,
    noise: float,
    mag0: float,
    seed: int,
    *,
    eclipse: bool = True,
    span: float = 120.0,
    duty: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t, mag, err)`` of a box eclipse (or a flat star) with fractional noise."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n)) + BASE_JD
    phase = ((t - BASE_JD - 0.3) / PERIOD) % 1.0
    flux = np.ones(n)
    if eclipse:
        flux[(phase < duty / 2) | (phase > 1 - duty / 2)] -= depth
    flux = flux * (1.0 + rng.normal(0.0, noise, n))
    mag = mag0 - 2.5 * np.log10(flux)
    err = np.full(n, 2.5 / np.log(10.0) * noise)
    return t, mag, err


def _mb(**bands: cup.LightCurve) -> cup.MultiBandLightCurve:
    return cup.MultiBandLightCurve.from_light_curves(bands)


def _faint_eclipse_bright_noise(
    seed: int = 1, *, errors: tuple[bool, bool] = (False, False)
) -> cup.MultiBandLightCurve:
    """A faint, precise eclipsing g band next to a bright, noisy, flat r band."""
    tg, mg, eg = _eclipser(500, 0.03, 0.003, 18.0, seed)
    tr, mr, er = _eclipser(500, 0.03, 0.03, 13.0, seed + 900, eclipse=False)
    return _mb(
        g=cup.LightCurve.from_arrays(tg, mg, eg if errors[0] else None),
        r=cup.LightCurve.from_arrays(tr, mr, er if errors[1] else None),
    )


def _near(period: float) -> bool:
    return abs(period / PERIOD - 1.0) < 0.01


# --- error scale (MULTIBAND-2, MULTIBAND-6) -----------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bands_without_errors_are_weighted_by_their_own_scatter(seed: int) -> None:
    # Unit errors made depth_snr a raw flux depth, so the 5-mag-brighter noise band
    # outweighed the eclipse by ~1e4 and BLS returned its noise peak.
    mb = _faint_eclipse_bright_noise(seed)
    pg = cup.periodogram(mb, "BLS", backend="cpu", settings=SETTINGS)
    assert _near(pg.best_period())


def test_power_is_invariant_to_one_bands_zero_point() -> None:
    # A constant magnitude shift is a flux scale: it must not change any band's SNR.
    tg, mg, _ = _eclipser(400, 0.03, 0.005, 15.0, 4)
    tr, mr, _ = _eclipser(400, 0.02, 0.01, 14.0, 5)
    a = _mb(g=cup.LightCurve.from_arrays(tg, mg), r=cup.LightCurve.from_arrays(tr, mr))
    b = _mb(
        g=cup.LightCurve.from_arrays(tg, mg),
        r=cup.LightCurve.from_arrays(tr, mr - 5.0),
    )
    pa = cup.periodogram(a, "BLS", backend="cpu", settings=SETTINGS)
    pb = cup.periodogram(b, "BLS", backend="cpu", settings=SETTINGS)
    np.testing.assert_allclose(pb.power, pa.power, rtol=1e-8)


def test_mixed_error_bands_run_unweighted() -> None:
    # One band with errors and one without: the package-wide rule is that errors are
    # used only when every band has them, so the mixed set equals the error-less one.
    mixed = _faint_eclipse_bright_noise(6, errors=(False, True))
    bare = _faint_eclipse_bright_noise(6, errors=(False, False))
    pm = cup.periodogram(mixed, "BLS", backend="cpu", settings=SETTINGS)
    pn = cup.periodogram(bare, "BLS", backend="cpu", settings=SETTINGS)
    np.testing.assert_array_equal(pm.power, pn.power)
    assert _near(pm.best_period())
    # The signal band carrying errors is recovered as well.
    swapped = _faint_eclipse_bright_noise(6, errors=(True, False))
    assert _near(
        cup.periodogram(swapped, "BLS", backend="cpu", settings=SETTINGS).best_period()
    )


def test_bands_with_errors_keep_them() -> None:
    # Honest errors are used as given (the robust scatter is only a fallback): the
    # stack is exactly the sum of the bands' own single-band depth-SNR^2.
    mb = _faint_eclipse_bright_noise(7, errors=(True, True))
    pg = cup.periodogram(mb, "BLS", backend="cpu", grid=FLAT_GRID, settings=SETTINGS)
    assert _near(pg.best_period())
    expected = np.zeros(FLAT_GRID.size)
    for lc in mb.bands.values():
        one = cup.periodogram(lc, "BLS", backend="cpu", grid=FLAT_GRID,
                              settings=SETTINGS)
        expected += np.clip(one.extras["depth_snr"], 0.0, None) ** 2
    np.testing.assert_allclose(pg.power, expected, rtol=1e-12)


def test_constant_band_without_errors_does_not_raise() -> None:
    tg, mg, _ = _eclipser(300, 0.03, 0.003, 15.0, 8)
    tr = np.sort(np.random.default_rng(9).uniform(0.0, 120.0, 300)) + BASE_JD
    mb = _mb(
        g=cup.LightCurve.from_arrays(tg, mg),
        r=cup.LightCurve.from_arrays(tr, np.full(tr.size, 14.0)),
    )
    pg = cup.periodogram(mb, "BLS", backend="cpu", settings=SETTINGS)
    assert np.all(np.isfinite(pg.power))
    assert _near(pg.best_period())


# --- custom grids (MULTIBAND-4) -------------------------------------------------


def _with_errors() -> cup.MultiBandLightCurve:
    return _faint_eclipse_bright_noise(11, errors=(True, True))


def test_custom_period_grid_is_searched_as_given() -> None:
    grid = cup.GridSpec(kind="period", values=np.linspace(2.4, 2.6, 301))
    pg = cup.periodogram(_with_errors(), "BLS", backend="cpu", grid=grid,
                         settings=SETTINGS)
    assert pg.size == grid.size
    np.testing.assert_allclose(np.sort(pg.period), grid.period, rtol=1e-12)
    assert _near(pg.best_period())


@pytest.mark.parametrize(
    "grid",
    [
        FLAT_GRID,
        # Wide enough to span several log segments (box durations per segment).
        cup.GridSpec(kind="period", values=np.geomspace(0.6, 9.0, 1500)),
    ],
    ids=["narrow", "wide"],
)
def test_one_band_on_a_custom_grid_equals_single_band_bls(
    grid: cup.GridSpec,
) -> None:
    lc = _with_errors().bands["g"]
    mb = cup.periodogram(_mb(g=lc), "BLS", backend="cpu", grid=grid,
                         settings=SETTINGS)
    single = cup.periodogram(lc, "BLS", backend="cpu", grid=grid, settings=SETTINGS)
    np.testing.assert_array_equal(mb.frequency, single.frequency)
    np.testing.assert_allclose(
        mb.power, np.clip(single.extras["depth_snr"], 0.0, None) ** 2, rtol=1e-12
    )
    np.testing.assert_array_equal(mb.extras["duration"], single.extras["duration"])


def test_default_grid_passed_explicitly_matches_grid_none() -> None:
    from cuperiod.api import _multiband_grid

    mb = _with_errors()
    method = cup.get_method("BLS")
    grid = _multiband_grid(method, mb, SETTINGS)
    explicit = cup.periodogram(mb, "BLS", backend="cpu", grid=grid, settings=SETTINGS)
    default = cup.periodogram(mb, "BLS", backend="cpu", settings=SETTINGS)
    np.testing.assert_array_equal(explicit.power, default.power)
    np.testing.assert_array_equal(explicit.frequency, default.frequency)


# --- box geometry provenance (MULTIBAND-5, MULTIBAND-12) ------------------------


def test_band_index_names_the_band_supplying_the_box() -> None:
    mb = _with_errors()
    pg = cup.periodogram(mb, "BLS", backend="cpu", grid=FLAT_GRID, settings=SETTINGS)
    index = pg.extras["band_index"]
    assert index.shape == pg.power.shape
    assert set(np.unique(index)) <= {0.0, 1.0}
    best = pg.best_periods(1)[0]
    band = pg.meta["bands"][int(best.extra["band_index"])]
    assert band == "g"  # the eclipsing band, not the flat one
    # The reported box is that band's own single-band box at the best period.
    single = cup.periodogram(mb.bands["g"], "BLS", backend="cpu", grid=FLAT_GRID,
                             settings=SETTINGS)
    at = int(np.argmin(np.abs(single.period - best.period)))
    assert best.extra["depth"] == pytest.approx(float(single.extras["depth"][at]))
    assert best.extra["t0"] == pytest.approx(float(single.extras["t0"][at]))
