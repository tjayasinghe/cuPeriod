"""Pooled multi-band fold methods (PDM, CE, string-length, SuperSmoother, MHAOV).

Period recovery across two filters, the single-band collapse (the pooling weights must
reduce to the single-band statistic exactly), the too-few-points guard, and parity of
the pooled statistic across every backend. The guards shared by every multi-band
method (``min_detections`` and the ``meta["bands"]`` record) are checked here for all
seven of them.
"""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import (
    requires_gpu,
    requires_numba,
    requires_torch,
    torch_auto_device,
)
from cuperiod.core.errors import InsufficientDataError
from synth import synthetic_sine

#: Planted period of the two-band synthetic (days); 1/0.6234 = 1.604 cycles/day.
PERIOD = 0.6234

#: Explicit trial grid bracketing the true frequency. Bounded above the first
#: subharmonic (as in the single-band string-length test, where overlaid copies of the
#: fold also shorten the string) and small enough to stay fast on every backend.
GRID = cup.GridSpec(kind="frequency", values=np.linspace(1.0, 2.5, 1200), uniform=True)

#: The pooled fold methods, spelled as a user would.
FOLD_METHODS = ("PDM", "CE", "StringLength", "SuperSmoother")


def _two_band(period: float = PERIOD) -> cup.MultiBandLightCurve:
    """Two bands of one star: different amplitude, offset mean, independent noise."""
    tg, mg, eg = synthetic_sine(n=300, period=period, amp=0.4, seed=2)
    tr, mr, er = synthetic_sine(n=300, period=period, amp=0.25, seed=3)
    return cup.MultiBandLightCurve.from_light_curves(
        {
            "g": cup.LightCurve.from_arrays(tg, mg, eg),
            "r": cup.LightCurve.from_arrays(tr, mr + 1.3, er),
        }
    )


# --- period recovery ----------------------------------------------------------


@pytest.mark.parametrize("method", FOLD_METHODS)
def test_multiband_fold_recovers_period(method: str) -> None:
    pg = cup.periodogram(_two_band(), method, grid=GRID)
    assert pg.objective_sense == cup.get_method(method).objective_sense
    assert pg.meta["bands"] == ("g", "r")
    assert pg.n_samples == 600  # both bands' points
    assert pg.best_period() == pytest.approx(PERIOD, rel=2e-2)


# --- single-band collapse -----------------------------------------------------


@pytest.mark.parametrize("method", FOLD_METHODS)
def test_single_band_collapses_to_the_single_band_statistic(method: str) -> None:
    # With one band the pooling weights cancel (w/w = 1), so the multi-band statistic
    # must reproduce the single-band periodogram on the same grid to round-off.
    t, mag, err = synthetic_sine(n=300, period=PERIOD, seed=2)
    lc = cup.LightCurve.from_arrays(t, mag, err)
    mb = cup.MultiBandLightCurve.from_light_curves({"g": lc})
    pooled = cup.periodogram(mb, method, grid=GRID, backend="numpy")
    single = cup.periodogram(lc, method, grid=GRID, backend="numpy")
    assert np.allclose(pooled.power, single.power, rtol=1e-12)
    assert np.array_equal(pooled.frequency, single.frequency)
    assert pooled.n_samples == single.n_samples


# --- guards -------------------------------------------------------------------


@pytest.mark.parametrize("method", FOLD_METHODS)
def test_bands_below_the_minimum_point_count_raise(method: str) -> None:
    # Five points per band is below every method's floor (n_bins + 2 for PDM,
    # max(n_phase_bins, 8) for CE, 8 for string length).
    rng = np.random.default_rng(0)
    bands = {
        name: cup.LightCurve.from_arrays(
            np.sort(rng.uniform(0.0, 5.0, 5)), rng.normal(12.0, 0.1, 5)
        )
        for name in ("g", "r")
    }
    mb = cup.MultiBandLightCurve.from_light_curves(bands)
    with pytest.raises(InsufficientDataError):
        cup.periodogram(mb, method, grid=GRID)


# --- backend parity -----------------------------------------------------------
#
# ``precision="auto"`` is float64 on CUDA, so the fast backends agree with the
# vectorized numpy path to atomic-reordering round-off, not to float32 precision.


def _pooled_power(method: str, backend: str) -> np.ndarray:
    pg = cup.periodogram(_two_band(), method, grid=GRID, backend=backend)
    return pg.power


@requires_numba
@pytest.mark.parametrize("method", FOLD_METHODS)
def test_numba_multiband_matches_numpy(method: str) -> None:
    cpu = _pooled_power(method, "numpy")
    fast = _pooled_power(method, "numba")
    assert np.allclose(cpu, fast, rtol=1e-6, atol=1e-9)


@requires_gpu
@pytest.mark.parametrize("method", FOLD_METHODS)
def test_cupy_multiband_matches_numpy(method: str) -> None:
    cpu = _pooled_power(method, "numpy")
    gpu = _pooled_power(method, "cupy")
    assert np.allclose(cpu, gpu, rtol=1e-6, atol=1e-9)


@requires_torch
@torch_auto_device
@pytest.mark.parametrize("method", FOLD_METHODS)
def test_torch_multiband_matches_numpy(method: str) -> None:
    # A bare "torch" request must be normalized to "torch:<device>" by the method's
    # multiband_power, exactly as its single-band power does.
    cpu = _pooled_power(method, "numpy")
    pg = cup.periodogram(_two_band(), method, grid=GRID, backend="torch")
    assert pg.backend.startswith("torch:")
    assert np.allclose(cpu, pg.power, rtol=1e-6, atol=1e-9)


# --- MHAOV: per-band standardization (MULTIBAND-1) ------------------------------
#
# The pooled MHAOV F-statistic adds the bands' regression and residual sums of
# squares. Pooled raw, those sums are in each band's own units, so the noisiest or
# largest-scale band decided the period on its own; each band's sums are now divided
# by that band's variance first, like every other pooled method.

_MB_FREQ = cup.GridSpec(
    kind="frequency", values=np.linspace(1.0, 2.5, 1500), uniform=True
)

_MHAOV_BACKENDS = ("numpy", pytest.param("numba", marks=requires_numba))


def _mhaov_band(
    n: int, amp: float, noise: float, offset: float, seed: int
) -> cup.LightCurve:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 200.0, n)) + 2458000.0
    y = offset + amp * np.sin(2 * np.pi * (t - 2458000.0) / PERIOD)
    return cup.LightCurve.from_arrays(t, y + rng.normal(0.0, noise, n))


@pytest.mark.parametrize("backend", _MHAOV_BACKENDS)
def test_mhaov_noisy_band_does_not_decide_the_period(backend: str) -> None:
    # The same 0.03-mag signal in both bands, r 15x noisier: the raw pooled sums let
    # r's noise swamp g's clean detection (3 stars in 40 recovered).
    for seed in range(5):
        mb = cup.MultiBandLightCurve.from_light_curves(
            {
                "g": _mhaov_band(150, 0.03, 0.01, 15.0, seed),
                "r": _mhaov_band(150, 0.03, 0.15, 14.0, seed + 500),
            }
        )
        pg = cup.periodogram(mb, "MHAOV", grid=_MB_FREQ, backend=backend)
        assert pg.best_period() == pytest.approx(PERIOD, rel=1e-2)


@pytest.mark.parametrize("backend", _MHAOV_BACKENDS)
def test_mhaov_pooled_power_ignores_one_bands_scale(backend: str) -> None:
    g = _mhaov_band(150, 0.03, 0.01, 15.0, 1)
    r = _mhaov_band(150, 0.03, 0.02, 14.0, 501)
    scaled = cup.LightCurve.from_arrays(r.time, 100.0 * r.value)
    a = cup.MultiBandLightCurve.from_light_curves({"g": g, "r": r})
    b = cup.MultiBandLightCurve.from_light_curves({"g": g, "r": scaled})
    pa = cup.periodogram(a, "MHAOV", grid=_MB_FREQ, backend=backend).power
    pb = cup.periodogram(b, "MHAOV", grid=_MB_FREQ, backend=backend).power
    np.testing.assert_allclose(pb, pa, rtol=1e-9)


@pytest.mark.parametrize("backend", _MHAOV_BACKENDS)
def test_mhaov_single_band_collapses_to_the_single_band_statistic(
    backend: str,
) -> None:
    # The pooled path fits the centered, scaled band. AOV is shift- and
    # scale-invariant, so it must equal the single-band statistic of the centered
    # magnitudes (the raw ones differ only by the kernel's float64 cancellation of
    # a large mean against a small scatter, ~1e-8 absolute here).
    t, mag, err = synthetic_sine(n=300, period=PERIOD, seed=2)
    lc = cup.LightCurve.from_arrays(t, mag, err)
    centered = cup.LightCurve.from_arrays(t, mag - mag.mean(), err)
    mb = cup.MultiBandLightCurve.from_light_curves({"g": lc})
    pooled = cup.periodogram(mb, "MHAOV", grid=GRID, backend=backend)
    single = cup.periodogram(centered, "MHAOV", grid=GRID, backend=backend)
    np.testing.assert_allclose(pooled.power, single.power, rtol=1e-10)
    assert pooled.n_samples == single.n_samples
    raw = cup.periodogram(lc, "MHAOV", grid=GRID, backend=backend)
    np.testing.assert_allclose(pooled.power, raw.power, rtol=1e-6, atol=1e-7)


def test_mhaov_constant_band_contributes_nothing() -> None:
    # An exactly constant band has no variance to standardize by: it is left out of
    # the pooled sums rather than inflating rounding residue to unit variance.
    # (15.3 x 50 leaves a ~6e-28 rounding variance about its float mean.)
    g = _mhaov_band(150, 0.03, 0.01, 15.0, 3)
    flat = cup.LightCurve.from_arrays(
        np.sort(np.random.default_rng(4).uniform(0.0, 200.0, 50)) + 2458000.0,
        np.full(50, 15.3),
    )
    both = cup.MultiBandLightCurve.from_light_curves({"g": g, "r": flat})
    alone = cup.MultiBandLightCurve.from_light_curves({"g": g})
    pb = cup.periodogram(both, "MHAOV", grid=_MB_FREQ, backend="numpy").power
    pa = cup.periodogram(alone, "MHAOV", grid=_MB_FREQ, backend="numpy").power
    np.testing.assert_allclose(pb, pa, rtol=1e-12)


# --- SuperSmoother: errors and constant bands (MULTIBAND-6, METH-C-1) -------------


def _signal_and_noise(
    seed: int, *, g_err: bool, r_err: bool
) -> cup.MultiBandLightCurve:
    """g: 60 points of a 0.05-mag signal; r: 200 points of pure noise (0.03 mag)."""
    g = _mhaov_band(60, 0.05, 0.03, 15.0, seed)
    r = _mhaov_band(200, 0.0, 0.03, 14.0, seed + 500)
    return cup.MultiBandLightCurve.from_light_curves(
        {
            "g": cup.LightCurve.from_arrays(
                g.time, g.value, np.full(g.n, 0.03) if g_err else None
            ),
            "r": cup.LightCurve.from_arrays(
                r.time, r.value, np.full(r.n, 0.03) if r_err else None
            ),
        }
    )


def test_supersmoother_mixed_error_bands_run_unweighted() -> None:
    # A band without errors got B_k in magnitudes while a band with errors got it in
    # sigmas, so the error-less signal band carried ~4% of the weight and the pooled
    # score followed the noise band. Errors now count only when every band has them.
    for seed in range(4):
        mixed = _signal_and_noise(seed, g_err=False, r_err=True)
        bare = _signal_and_noise(seed, g_err=False, r_err=False)
        pm = cup.periodogram(mixed, "SuperSmoother", grid=_MB_FREQ, backend="numpy")
        pn = cup.periodogram(bare, "SuperSmoother", grid=_MB_FREQ, backend="numpy")
        np.testing.assert_array_equal(pm.power, pn.power)
        assert pm.best_period() == pytest.approx(PERIOD, rel=1e-2)


def test_supersmoother_constant_band_is_left_out() -> None:
    # 1234.567 x 50 with unequal errors is not bit-equal to its own weighted mean,
    # so B_k was ~2e-11 > 0 and the band's degenerate score entered the mix.
    g = _mhaov_band(150, 0.05, 0.01, 15.0, 3)
    rng = np.random.default_rng(4)
    t_flat = np.sort(rng.uniform(0.0, 200.0, 50)) + 2458000.0
    dy_flat = 0.01 * (0.8 + 0.4 * np.random.default_rng(0).random(50))
    flat = cup.LightCurve.from_arrays(t_flat, np.full(50, 1234.567), dy_flat)
    g_err = cup.LightCurve.from_arrays(g.time, g.value, np.full(g.n, 0.01))
    both = cup.MultiBandLightCurve.from_light_curves({"g": g_err, "r": flat})
    alone = cup.MultiBandLightCurve.from_light_curves({"g": g_err})
    pb = cup.periodogram(both, "SuperSmoother", grid=_MB_FREQ, backend="numpy")
    pa = cup.periodogram(alone, "SuperSmoother", grid=_MB_FREQ, backend="numpy")
    np.testing.assert_allclose(pb.power, pa.power, rtol=1e-12)
    flat2 = cup.LightCurve.from_arrays(t_flat, np.full(50, 14.7), np.full(50, 0.02))
    constant = cup.MultiBandLightCurve.from_light_curves({"g": flat, "r": flat2})
    pc = cup.periodogram(constant, "SuperSmoother", grid=_MB_FREQ, backend="numpy")
    assert np.count_nonzero(pc.power) == 0


def test_supersmoother_band_constant_to_rounding_is_left_out() -> None:
    # Not bit-constant, so the old max == min guard kept it: its score is all zeros
    # (the score-level guard is relative), but its rounding-level B_k still diluted
    # the weight sum. Both guards now use the same relative test.
    g = _mhaov_band(150, 0.05, 0.01, 15.0, 3)
    rng = np.random.default_rng(4)
    t_flat = np.sort(rng.uniform(0.0, 200.0, 50)) + 2458000.0
    wobble = 14.7 + 2e-15 * rng.standard_normal(50)
    assert wobble.max() != wobble.min()
    flat = cup.LightCurve.from_arrays(t_flat, wobble, np.full(50, 0.02))
    g_err = cup.LightCurve.from_arrays(g.time, g.value, np.full(g.n, 0.01))
    both = cup.MultiBandLightCurve.from_light_curves({"g": g_err, "r": flat})
    alone = cup.MultiBandLightCurve.from_light_curves({"g": g_err})
    pb = cup.periodogram(both, "SuperSmoother", grid=_MB_FREQ, backend="numpy")
    pa = cup.periodogram(alone, "SuperSmoother", grid=_MB_FREQ, backend="numpy")
    np.testing.assert_array_equal(pb.power, pa.power)


# --- guards shared by every multi-band method (MULTIBAND-7, MULTIBAND-12) -------

ALL_MB_METHODS = ("GLS", "BLS", "MHAOV", "PDM", "CE", "StringLength", "SuperSmoother")


@pytest.mark.parametrize("method", ALL_MB_METHODS)
def test_min_detections_counts_the_usable_bands(method: str) -> None:
    # A sparse multi-band object is skipped exactly like its single-band twin: the
    # batch runners rely on InsufficientDataError to leave it out.
    settings = cup.get_method(method).settings_cls(min_detections=5000)
    with pytest.raises(InsufficientDataError, match="min_detections"):
        cup.periodogram(_two_band(), method, grid=GRID, settings=settings,
                        backend="cpu")


def test_default_bls_skips_a_handful_of_points() -> None:
    # Two bands of five points clear BLS's per-band floor (3) but not the default
    # min_detections of 20, which single-band BLS already enforced.
    rng = np.random.default_rng(1)
    bands = {
        name: cup.LightCurve.from_arrays(
            np.sort(rng.uniform(0.0, 30.0, 5)), rng.normal(12.0, 0.1, 5)
        )
        for name in ("g", "r")
    }
    mb = cup.MultiBandLightCurve.from_light_curves(bands)
    with pytest.raises(InsufficientDataError, match="min_detections"):
        cup.periodogram(mb, "BLS", backend="cpu")


def _with_dropped_bands() -> cup.MultiBandLightCurve:
    """``g``/``r`` of the two-band set plus a 2-point band and an empty band."""
    two = _two_band()
    u = two.bands["g"]
    return cup.MultiBandLightCurve.from_light_curves(
        {
            **two.bands,
            "u": cup.LightCurve.from_arrays(u.time[:2], u.value[:2], u.error[:2]),
            "none": cup.LightCurve.from_arrays(np.empty(0), np.empty(0), np.empty(0)),
        }
    )


@pytest.mark.parametrize(
    ("method", "settings", "backend", "bands"),
    [
        ("PDM", None, "cpu", ("g", "r")),
        ("CE", None, "cpu", ("g", "r")),
        ("StringLength", None, "cpu", ("g", "r")),
        ("SuperSmoother", None, "cpu", ("g", "r")),
        ("MHAOV", None, "cpu", ("g", "r")),
        ("BLS", None, "cpu", ("g", "r")),
        # Every band with a point carries an offset in the joint GLS models ...
        ("GLS", cup.GLSSettings(mb_model="offsets"), "finufft", ("g", "r", "u")),
        ("GLS", cup.GLSSettings(mb_model="flex"), "finufft", ("g", "r", "u")),
        ("GLS", cup.GLSSettings(mb_model="offsets"), "astropy", ("g", "r", "u")),
        # ... while a per-band sinusoid needs three points.
        ("GLS", cup.GLSSettings(mb_model="perband"), "finufft", ("g", "r")),
    ],
)
def test_meta_bands_lists_only_contributing_bands(
    method: str, settings: object, backend: str, bands: tuple[str, ...]
) -> None:
    mb = _with_dropped_bands()
    pg = cup.periodogram(mb, method, grid=GRID, settings=settings,  # type: ignore[arg-type]
                         backend=backend)
    assert isinstance(pg, cup.Periodogram)
    assert tuple(pg.meta["bands"]) == bands
    assert pg.n_samples == sum(mb.bands[name].n for name in bands)


# --- registry -----------------------------------------------------------------


def test_fold_methods_report_multiband_support() -> None:
    support = {info.name: info.supports_multiband for info in cup.list_methods()}
    assert support["PDM"] and support["CE"] and support["STRINGLENGTH"]
    assert support["SUPERSMOOTHER"]
    for name in FOLD_METHODS:
        assert cup.get_method(name).supports_multiband
