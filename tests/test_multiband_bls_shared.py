"""Joint-ephemeris multi-band BLS (MULTIBAND-5).

``ephemeris="shared"`` places one box -- one duration and one mid-transit time on a
phase grid common to every band -- and sums the bands' squared depth-SNRs inside it;
the default ``"independent"`` stack (each band's own best box) must stay exactly as
it was. Pinned here: the default's bits, the backends' agreement (numba = numpy =
torch bit for bit, CUDA to round-off), the one-band limit, the refusal of astropy,
phase-scrambled bands, and a weak planted multi-band transit that the joint
statistic recovers where the independent stack does not.
"""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu, requires_numba, requires_torch
from cuperiod.core.errors import BackendUnavailableError
from cuperiod.methods.base import get_method
from cuperiod.multiband.bls_mb import bls_multiband_power

BLS = get_method("BLS")

#: A small search: a 1-6 d period range with few durations keeps every backend fast.
SETTINGS = cup.BLSSettings(
    min_period_days=1.0, max_period_days=6.0, n_durations=4, bins_per_duration=5,
    min_detections=10,
)


def _star(
    seed: int,
    *,
    n_band: int = 45,
    n_bands: int = 6,
    depth: float = 0.02,
    noise: float = 0.005,
    span: float = 90.0,
    chromatic: bool = True,
):
    """Bands of one transiting star; returns (mblc, P).

    The depths run from 0.7x to 1.3x ``depth`` across the bands when ``chromatic``.
    """
    rng = np.random.default_rng(seed)
    period = rng.uniform(1.5, 4.5)
    t0 = 2458000.0 + rng.uniform(0.0, period)
    bands = {}
    for b in range(n_bands):
        t = 2458000.0 + np.sort(rng.uniform(0.0, span, n_band))
        flux = 1.0 + rng.normal(0.0, noise, n_band)
        phase = np.mod(t - t0 + 0.02 * period, period) / period
        scale = 0.7 + 0.6 * b / max(1, n_bands - 1) if chromatic else 1.0
        flux[phase < 0.04] -= depth * scale
        bands[f"b{b}"] = cup.LightCurve.from_arrays(
            t, flux, np.full(n_band, noise), domain="flux"
        )
    return cup.MultiBandLightCurve.from_light_curves(bands), period


def _grid(mblc):
    time, value, error, _ = mblc.finite().stacked()
    return BLS.default_grid(
        cup.LightCurve.from_arrays(time, value, error, domain="flux"), SETTINGS
    )


def _old_independent(grid, mblc, settings, backend):
    """The pre-1.3 independent stack, verbatim in substance (the reference)."""
    from cuperiod.core.columns import Domain
    from cuperiod.methods.bls import _grid_segments, _segment_power
    from cuperiod.multiband.bls_mb import _band_errors

    finite = mblc.finite()
    bands = [lc.in_domain(Domain.FLUX) for lc in finite.bands.values() if lc.n >= 3]
    errors = _band_errors(bands)
    caches = [{} for _ in bands]
    power, depth, t0, band = [], [], [], []
    for periods, durations in _grid_segments(grid, settings):
        per_band = [
            _segment_power(backend, lc.time, lc.value, err, periods, durations,
                           settings, cache)
            for lc, err, cache in zip(bands, errors, caches, strict=True)
        ]
        dsnr = np.vstack([np.clip(s["depth_snr"], 0.0, None) for s in per_band])
        best = np.argmax(dsnr, axis=0)
        cols = np.arange(dsnr.shape[1])
        power.append(np.sum(dsnr**2, axis=0))
        depth.append(np.vstack([s["depth"] for s in per_band])[best, cols])
        t0.append(np.vstack([s["transit_time"] for s in per_band])[best, cols])
        band.append(best.astype(np.float64))
    return [np.concatenate(v) for v in (power, depth, t0, band)]


# --- the default is unchanged ---------------------------------------------------


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("numba", marks=requires_numba)]
)
def test_default_independent_stack_is_unchanged(backend: str) -> None:
    mblc, _ = _star(3)
    grid = _grid(mblc)
    pg = bls_multiband_power(grid, mblc, SETTINGS, backend, BLS)
    power, depth, t0, band = _old_independent(grid, mblc, SETTINGS, backend)
    # from_spectrum sorts by frequency: compare in period order.
    order = np.argsort(1.0 / grid.period, kind="stable")
    np.testing.assert_array_equal(pg.power, power[order])
    np.testing.assert_array_equal(pg.extras["depth"], depth[order])
    np.testing.assert_array_equal(pg.extras["t0"], t0[order])
    np.testing.assert_array_equal(pg.extras["band_index"], band[order])
    assert "mb_ephemeris" not in pg.meta
    assert list(pg.extras) == ["depth", "depth_snr", "duration", "t0", "sde",
                               "band_index"]


def test_unknown_ephemeris_and_astropy_are_refused() -> None:
    mblc, _ = _star(1, n_bands=2)
    grid = _grid(mblc)
    with pytest.raises(ValueError, match="ephemeris"):
        bls_multiband_power(grid, mblc, SETTINGS, "numpy", BLS, ephemeris="joint")
    with pytest.raises(BackendUnavailableError, match="astropy"):
        bls_multiband_power(grid, mblc, SETTINGS, "astropy", BLS, ephemeris="shared")


# --- backends agree ---------------------------------------------------------------


@requires_numba
@pytest.mark.parametrize("objective", ["snr", "likelihood"])
def test_shared_numba_equals_numpy_bit_for_bit(objective: str) -> None:
    mblc, _ = _star(5, n_bands=4)
    grid = _grid(mblc)
    settings = SETTINGS.model_copy(update={"objective": objective})
    a = bls_multiband_power(grid, mblc, settings, "numpy", BLS, ephemeris="shared")
    b = bls_multiband_power(grid, mblc, settings, "numba", BLS, ephemeris="shared")
    np.testing.assert_array_equal(a.power, b.power)
    for name in ("depth", "duration", "t0", "band_index", "depth_snr"):
        np.testing.assert_array_equal(a.extras[name], b.extras[name], err_msg=name)
    assert b.meta["mb_ephemeris"] == "shared"


@requires_torch
def test_shared_torch_equals_numpy() -> None:
    mblc, _ = _star(6, n_bands=3)
    grid = _grid(mblc)
    a = bls_multiband_power(grid, mblc, SETTINGS, "numpy", BLS, ephemeris="shared")
    b = bls_multiband_power(grid, mblc, SETTINGS, "torch:cpu", BLS, ephemeris="shared")
    np.testing.assert_allclose(b.power, a.power, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(a.extras["t0"], b.extras["t0"])


@requires_gpu
@pytest.mark.parametrize("objective", ["snr", "likelihood"])
def test_shared_cuda_matches_numpy(objective: str) -> None:
    mblc, _ = _star(7)
    grid = _grid(mblc)
    settings = SETTINGS.model_copy(update={"objective": objective})
    ref = bls_multiband_power(grid, mblc, settings, "numpy", BLS, ephemeris="shared")
    gpu = bls_multiband_power(grid, mblc, settings, "cupy", BLS, ephemeris="shared")
    np.testing.assert_allclose(gpu.power, ref.power, rtol=1e-9, atol=1e-9)
    # The same box nearly everywhere (an exact tie may break the other way).
    same = np.isclose(gpu.extras["t0"], ref.extras["t0"], rtol=0.0, atol=1e-9)
    assert same.mean() > 0.999
    assert gpu.best_period() == ref.best_period()


@requires_gpu
def test_shared_cuda_float32_finds_the_same_transit() -> None:
    mblc, period = _star(8)
    grid = _grid(mblc)
    settings = SETTINGS.model_copy(update={"precision": "float32"})
    gpu = bls_multiband_power(grid, mblc, settings, "cupy", BLS, ephemeris="shared")
    ref = bls_multiband_power(grid, mblc, SETTINGS, "numpy", BLS, ephemeris="shared")
    assert gpu.best_period() == pytest.approx(ref.best_period(), rel=1e-9)
    assert np.isclose(gpu.power, ref.power, rtol=1e-4, atol=1e-4).mean() > 0.99


# --- what the statistic means ---------------------------------------------------


@pytest.mark.parametrize(
    "backend",
    ["numpy", pytest.param("numba", marks=requires_numba),
     pytest.param("cupy", marks=requires_gpu)],
)
def test_one_band_shared_is_the_single_band_search(backend: str) -> None:
    mblc, _ = _star(9, n_bands=1, n_band=120)
    grid = _grid(mblc)
    shared = bls_multiband_power(grid, mblc, SETTINGS, backend, BLS, ephemeris="shared")
    single = BLS.power(grid, mblc.bands["b0"], SETTINGS, backend)
    expected = np.clip(single.extras["depth_snr"], 0.0, None) ** 2
    np.testing.assert_allclose(shared.power, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shared.extras["t0"], single.extras["t0"], atol=1e-9)


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("numba", marks=requires_numba)]
)
def test_phase_scrambled_bands_no_longer_add_up(backend: str) -> None:
    # Shifting one band's times by 0.375 P moves its transit to another phase of
    # the common clock. Each band's own search is blind to a time shift (up to the
    # rounding of the shifted times), so the independent stack does not change; the
    # shared box loses that band.
    aligned, period = _star(11, n_bands=2, n_band=150, depth=0.03, chromatic=False)
    shifted = aligned.bands["b1"]
    scrambled = cup.MultiBandLightCurve.from_light_curves({
        "b0": aligned.bands["b0"],
        "b1": cup.LightCurve.from_arrays(
            shifted.time + 0.375 * period, shifted.value, shifted.error,
            domain="flux",
        ),
    })
    grid = _grid(aligned)
    i_al = bls_multiband_power(grid, aligned, SETTINGS, backend, BLS)
    i_sc = bls_multiband_power(grid, scrambled, SETTINGS, backend, BLS)
    s_al = bls_multiband_power(
        grid, aligned, SETTINGS, backend, BLS, ephemeris="shared"
    )
    s_sc = bls_multiband_power(
        grid, scrambled, SETTINGS, backend, BLS, ephemeris="shared"
    )
    assert i_sc.power.max() == pytest.approx(i_al.power.max(), rel=1e-6)
    assert np.median(np.abs(i_sc.power - i_al.power)) < 1e-9
    i = int(np.argmax(s_al.power))
    assert s_al.power[i] == pytest.approx(i_al.power[i], rel=0.05)
    assert s_al.best_period() == pytest.approx(period, rel=0.01)
    # At the true period the scrambled shared power is about one band's SNR^2.
    single = [
        BLS.power(grid, aligned.bands[b], SETTINGS, backend).power.max() ** 2
        for b in ("b0", "b1")
    ]
    assert s_sc.power[i] < 0.75 * s_al.power[i]
    assert s_sc.power[i] < 1.3 * max(single)


@requires_numba
def test_weak_multiband_transit_is_recovered_by_the_shared_box() -> None:
    # Six sparse bands (45 points each), a 1.5% transit in 0.5% noise: no band
    # alone shows it; the independent stack also adds up noise boxes at unrelated
    # phases, the shared box does not. Calibrated over 40 stars: 39/40 vs 24/40.
    n_stars = 30
    hits = {"independent": 0, "shared": 0}
    for seed in range(n_stars):
        mblc, period = _star(seed, depth=0.015)
        grid = _grid(mblc)
        for mode in hits:
            pg = bls_multiband_power(grid, mblc, SETTINGS, "numba", BLS, ephemeris=mode)
            hits[mode] += abs(pg.best_period() / period - 1.0) < 0.01
    assert hits["shared"] >= 0.9 * n_stars
    assert hits["shared"] - hits["independent"] >= 0.2 * n_stars


def test_shared_geometry_is_one_box_for_every_band() -> None:
    mblc, period = _star(12, n_bands=3, n_band=200, depth=0.03)
    grid = _grid(mblc)
    pg = bls_multiband_power(grid, mblc, SETTINGS, "numpy", BLS, ephemeris="shared")
    i = int(np.argmax(pg.power))
    assert pg.period[i] == pytest.approx(period, rel=0.01)
    assert 0.0 < pg.extras["duration"][i] < 0.2 * period
    assert pg.extras["depth"][i] > 0.0
    assert pg.extras["depth_snr"][i] == pytest.approx(np.sqrt(pg.power[i]))
    assert pg.meta["bands"] == ("b0", "b1", "b2")


@pytest.mark.skipif(
    "mb_ephemeris" not in cup.BLSSettings.model_fields,
    reason="BLSSettings.mb_ephemeris not added yet",
)
def test_settings_select_the_shared_ephemeris() -> None:
    mblc, _ = _star(13, n_bands=2)
    settings = SETTINGS.model_copy(update={"mb_ephemeris": "shared"})
    pg = cup.periodogram(mblc, "BLS", settings=settings, backend="numpy")
    assert pg.meta["mb_ephemeris"] == "shared"
    default = cup.periodogram(mblc, "BLS", settings=SETTINGS, backend="numpy")
    assert "mb_ephemeris" not in default.meta
