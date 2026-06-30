"""Torch-CPU parity for the bonus methods (CE, String-Length, MHAOV, PDM, TLS).

Each method's portable torch path must match its numpy path (both float64 on CPU) and
pick the same best period. MHAOV goes through ``linalg.solve`` so its agreement is
relative (peak-normalized), not bit-exact; the rest are near-identical. GPU devices
are validated separately in PR2; these run on the CPU torch device.
"""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_torch
from cuperiod.methods.mhaov import aov_power
from synth import synthetic_eclipser, synthetic_sine

FREQ_METHODS = ["CE", "STRINGLENGTH", "MHAOV", "PDM"]


@requires_torch
@pytest.mark.parametrize("method", FREQ_METHODS)
def test_bonus_torch_matches_numpy(method: str) -> None:
    t, mag, err = synthetic_sine(period=0.6234)
    lc = cup.LightCurve.from_arrays(t, mag, err)
    m = cup.get_method(method)
    settings = m.settings_cls()
    grid = m.default_grid(lc, settings)
    ref = m.power(grid, lc, settings, "numpy")
    tor = m.power(grid, lc, settings, "torch:cpu")
    assert tor.backend == "torch:cpu"
    finite = np.isfinite(ref.power) & np.isfinite(tor.power)
    scale = max(float(np.max(np.abs(ref.power[finite]))), 1e-30)
    max_abs = float(np.max(np.abs(ref.power[finite] - tor.power[finite])))
    assert max_abs / scale < 1e-6
    # Same best period off the identical grid (whichever sense the method is).
    assert ref.best_period() == pytest.approx(tor.best_period(), rel=1e-9)


@requires_torch
def test_tls_torch_matches_numpy() -> None:
    t, flux, err = synthetic_eclipser(period=2.5, depth=0.05)
    lc = cup.LightCurve.from_arrays(t, flux, err, domain=cup.Domain.FLUX)
    m = cup.get_method("TLS")
    settings = m.settings_cls(min_period_days=1.5, max_period_days=4.0)
    grid = m.default_grid(lc, settings)
    ref = m.power(grid, lc, settings, "numpy")
    tor = m.power(grid, lc, settings, "torch:cpu")
    assert tor.backend == "torch:cpu"
    finite = np.isfinite(ref.power) & np.isfinite(tor.power)
    scale = max(float(np.max(np.abs(ref.power[finite]))), 1e-30)
    assert float(np.max(np.abs(ref.power[finite] - tor.power[finite]))) / scale < 1e-6
    assert int(np.argmax(ref.power)) == int(np.argmax(tor.power))


@requires_torch
@pytest.mark.parametrize("method", FREQ_METHODS)
def test_bonus_torch_recovers_period(method: str) -> None:
    # The portable torch path recovers the period (or a small-integer harmonic —
    # String-Length and MHAOV legitimately lock onto 3P here, identically to numpy).
    t, mag, err = synthetic_sine(period=0.6234)
    pg = cup.periodogram((t, mag, err), method, backend="torch")
    assert pg.backend == "torch:cpu"
    ratio = pg.best_period() / 0.6234
    harmonics = (1.0, 0.5, 2.0, 1 / 3, 3.0, 2 / 3, 1.5)
    assert min(abs(ratio / r - 1.0) for r in harmonics) < 5e-3


@requires_torch
def test_string_length_torch_matches_numpy_with_tied_phases() -> None:
    # Regression: equal folded phases must sort *stably* and identically on numpy and
    # torch. An unstable sort (raw numpy quicksort) breaks ties differently from the
    # array-API stable sort the torch path uses — and quicksort's tie order is
    # platform-dependent, so this passed locally but failed in CI. Force heavy ties
    # (times on a period/8 lattice fold to only 8 distinct phases) to pin it everywhere.
    from cuperiod.methods.string_length import string_length

    rng = np.random.default_rng(0)
    period = 2.0
    t = 2458000.0 + (rng.integers(0, 400, 400) * (period / 8)).astype(float)
    y = rng.normal(0.0, 1.0, t.size)
    periods = np.linspace(1.5, 3.0, 200)
    ref = string_length(t, y, periods, backend="numpy")
    tor = string_length(t, y, periods, backend="torch:cpu")
    assert float(np.max(np.abs(ref - tor))) < 1e-9


@requires_torch
def test_mhaov_multiband_torch_honours_precision() -> None:
    # Regression: the multiband wrapper must forward ``precision`` to the torch compute.
    # It previously dropped it (defaulting to auto→float64), so an explicit float32 was
    # silently ignored — bit-identical to float64 (and on MPS this turned the mandatory
    # float64 raise into a silent float32 downcast). After the fix the two precisions
    # produce different bits, while both still recover the period.
    tg, mg, eg = synthetic_sine(n=300, period=0.6234, seed=2)
    tr, mr, er = synthetic_sine(n=300, period=0.6234, amp=0.3, seed=3)
    mb = cup.MultiBandLightCurve.from_light_curves(
        {
            "g": cup.LightCurve.from_arrays(tg, mg, eg),
            "r": cup.LightCurve.from_arrays(tr, mr + 1.0, er),
        }
    )
    m = cup.get_method("MHAOV")
    p64 = cup.periodogram(
        mb, "MHAOV", backend="torch:cpu", settings=m.settings_cls(precision="float64")
    )
    p32 = cup.periodogram(
        mb, "MHAOV", backend="torch:cpu", settings=m.settings_cls(precision="float32")
    )
    assert p32.backend == "torch:cpu"
    assert not np.array_equal(p64.power, p32.power)  # float32 was actually used
    assert p64.best_period() == pytest.approx(p32.best_period(), rel=1e-2)


@requires_torch
def test_mhaov_torch_float32_survives_degenerate_frequency() -> None:
    # Regression: at f→0 the harmonic basis collapses onto the constant column; the
    # diagonal ridge must survive the float32 cast. A fixed absolute 1e-10 underflowed
    # against the ~N-sized Gram diagonal in float32, leaving the matrix singular so
    # ``linalg.solve`` raised _LinAlgError. The ridge is now eps(dtype)·N-scaled.
    t, mag, err = synthetic_sine(n=400, period=0.6234)
    freqs = np.array([1e-8, 1e-6, 1.0 / 0.6234])  # first is near-degenerate
    power = aov_power(
        t, mag, freqs, n_harmonics=3, backend="torch:cpu", precision="float32"
    )
    assert np.all(np.isfinite(power))  # no crash, no NaN/inf
    assert power[-1] > power[0]  # the real signal beats the degenerate frequency
