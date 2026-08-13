"""Pooled multi-band fold methods (PDM, CE, string-length).

Period recovery across two filters, the single-band collapse (the pooling weights must
reduce to the single-band statistic exactly), the too-few-points guard, and parity of
the pooled statistic across every backend.
"""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu, requires_numba, requires_torch
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
@pytest.mark.parametrize("method", FOLD_METHODS)
def test_torch_multiband_matches_numpy(method: str) -> None:
    # A bare "torch" request must be normalized to "torch:<device>" by the method's
    # multiband_power, exactly as its single-band power does.
    cpu = _pooled_power(method, "numpy")
    pg = cup.periodogram(_two_band(), method, grid=GRID, backend="torch")
    assert pg.backend.startswith("torch:")
    assert np.allclose(cpu, pg.power, rtol=1e-6, atol=1e-9)


# --- registry -----------------------------------------------------------------


def test_fold_methods_report_multiband_support() -> None:
    support = {info.name: info.supports_multiband for info in cup.list_methods()}
    assert support["PDM"] and support["CE"] and support["STRINGLENGTH"]
    assert support["SUPERSMOOTHER"]
    for name in FOLD_METHODS:
        assert cup.get_method(name).supports_multiband
