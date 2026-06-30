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
