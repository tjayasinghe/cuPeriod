"""TLS: limb-darkened template, transit recovery, and backend wiring."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu
from cuperiod.core.errors import BackendUnavailableError
from cuperiod.methods.tls import limb_darkened_template, tls_power


def _inject_transit(
    *,
    n: int = 1200,
    span: float = 120.0,
    period: float = 3.0,
    t0: float = 2458001.0,
    depth: float = 0.02,
    dur_frac: float = 0.05,
    u1: float = 0.4,
    u2: float = 0.3,
    noise: float = 0.002,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A flux light curve with a planted limb-darkened transit."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n)) + 2458000.0
    phase = ((t - t0) / period) % 1.0
    x = np.where(phase > 0.5, phase - 1.0, phase)
    inside = np.abs(x) < dur_frac / 2.0
    z = np.clip(2.0 * np.abs(x) / dur_frac, 0.0, 1.0)
    mu = np.sqrt(1.0 - z * z)
    prof = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    prof = prof / prof.max()
    flux = np.ones(n)
    flux[inside] -= depth * prof[inside]
    flux += rng.normal(0.0, noise, n)
    return t, flux, np.full(n, noise)


def test_limb_darkened_template_shape() -> None:
    g = limb_darkened_template(21, 0.4, 0.3)
    assert g.size == 21
    assert g.max() == pytest.approx(1.0)
    assert g[10] == pytest.approx(1.0, abs=1e-6)  # peak at the centre
    assert g[0] < g[10] and g[-1] < g[10]  # tapers toward the limb


def test_tls_recovers_transit() -> None:
    # A single-transit signal also aligns at sub-multiples P/k, so — as with every
    # fold method — the search is bounded to a sensible period range.
    t, flux, err = _inject_transit(period=3.0, depth=0.02)
    pg = cup.periodogram(
        (t, flux, err), "TLS", domain="flux",
        settings=cup.TLSSettings(min_period_days=2.0),
    )
    assert pg.objective_sense == "max"
    peak = pg.best_periods(1, alias_diverse=True)[0]
    assert peak.period == pytest.approx(3.0, rel=5e-3)
    assert peak.extra["depth"] > 0.0
    assert peak.power > 7.0  # SDE of a clear transit


def test_tls_peak_has_transit_parameters() -> None:
    t, flux, err = _inject_transit()
    pg = cup.periodogram(
        (t, flux, err), "tls", domain="flux",
        settings=cup.TLSSettings(min_period_days=2.0),
    )
    peak = pg.best_periods(1, alias_diverse=True)[0]
    assert {"depth", "duration", "t0", "sr"} <= set(peak.extra)
    assert 0.0 < peak.extra["duration"] < 3.0


def test_tls_converts_magnitude() -> None:
    t, flux, _ = _inject_transit(depth=0.03)
    mag = -2.5 * np.log10(flux)
    pg = cup.periodogram(
        (t, mag, 0.003 * np.ones_like(mag)), "TLS", domain="magnitude",
        settings=cup.TLSSettings(min_period_days=2.0),
    )
    peak = pg.best_periods(1, alias_diverse=True)[0]
    assert peak.period == pytest.approx(3.0, rel=1e-2)


def test_tls_gpu_backend_unavailable() -> None:
    from cuperiod.core.backend import cuda_available

    if cuda_available():
        pytest.skip("a GPU is present")
    with pytest.raises(BackendUnavailableError):
        cup.get_method("TLS").resolve_backend("gpu")


@requires_gpu
def test_tls_gpu_matches_cpu() -> None:
    t, flux, err = _inject_transit(period=3.0, depth=0.02)
    settings = cup.TLSSettings(min_period_days=2.0)
    grid = cup.get_method("TLS").default_grid(
        cup.LightCurve.from_arrays(t, flux, err, domain=cup.Domain.FLUX), settings
    )
    periods = grid.period
    cpu = tls_power(t, flux, err, periods, settings=settings, backend="numpy")
    gpu = tls_power(t, flux, err, periods, settings=settings, backend="cupy")
    assert np.allclose(cpu["sr"], gpu["sr"], rtol=1e-6, atol=1e-6)
    assert np.allclose(cpu["sde"], gpu["sde"], rtol=1e-6, atol=1e-6)
