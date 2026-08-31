"""Tests for the joint (period, dP/dt) time-warp search (`cuperiod.pdot_search`).

The algorithm of record is the warp identity itself: a signal with constant
dP/dt has phase ``f0 * tau`` with ``tau = dt*(1 - a/2*dt)``, ``a = pdot/P`` —
so evaluating an ordinary periodogram on warped times at the true ``a`` must
recover exactly the constant-period problem. `synthetic_period_change` plants
signals with that exact phase, which pins both axes and the sign convention.
"""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu, requires_numba, requires_torch
from cuperiod.core.errors import InsufficientDataError
from cuperiod.pdot.scan import _pdot_axis, weighted_epoch
from synth import synthetic_multiband_sine, synthetic_period_change, synthetic_sine

# --- recovery ----------------------------------------------------------------


def test_gls_recovers_planted_drift():
    # pdot=1e-6 accumulates ~1.1 cycles of drift: enough to defocus the
    # constant-period row, which is the regime the 2-D search exists for.
    t, mag, err, truth = synthetic_period_change(pdot=1.0e-6, seed=42)
    res = cup.pdot_search((t, mag, err), period=truth["period"], backend="cpu")
    step = float(res.pdot_over_p[1] - res.pdot_over_p[0])
    assert res.best.period == pytest.approx(truth["period"], rel=2e-4)
    assert res.best.pdot_over_p == pytest.approx(truth["pdot_over_p"], abs=2 * step)
    assert res.best.pdot == pytest.approx(truth["pdot"], rel=0.15)
    # The drift is large enough that the constant-period row lost the signal.
    assert res.best.power > 2.0 * res.best.zero_pdot_power


def test_negative_pdot_sign_convention():
    t, mag, err, truth = synthetic_period_change(pdot=-1.0e-6, seed=3)
    res = cup.pdot_search((t, mag, err), period=truth["period"], backend="cpu")
    assert res.best.pdot < 0.0
    assert res.best.pdot == pytest.approx(truth["pdot"], rel=0.2)
    # fdot has the opposite sign of pdot.
    assert res.best.fdot > 0.0


def test_zero_pdot_null():
    t, mag, err, truth = synthetic_period_change(pdot=0.0, seed=11)
    res = cup.pdot_search((t, mag, err), period=truth["period"], backend="cpu")
    step = float(res.pdot_over_p[1] - res.pdot_over_p[0])
    assert abs(res.best.pdot_over_p) <= 2 * step
    assert res.best.power == pytest.approx(res.best.zero_pdot_power, rel=0.05)


@requires_numba
def test_pdm_recovers_planted_drift():
    t, mag, err, truth = synthetic_period_change(pdot=1.0e-6, seed=42)
    res = cup.pdot_search(
        (t, mag, err), period=truth["period"], method="PDM", backend="cpu"
    )
    step = float(res.pdot_over_p[1] - res.pdot_over_p[0])
    assert res.objective_sense == "min"
    assert res.best.period == pytest.approx(truth["period"], rel=2e-4)
    assert res.best.pdot_over_p == pytest.approx(truth["pdot_over_p"], abs=2 * step)
    # Theta at the solution is far below the drift-smeared zero row.
    assert res.best.power < 0.5 * res.best.zero_pdot_power


def test_exact_warp_recovers_on_short_cycle_baseline():
    # 160 cycles over the span, drift near the edge of the default range: the
    # truncated quadratic warp would mis-phase by ~1 cycle here (the regime
    # flagged by the v1.3 literature review); the exact log warp must not.
    t, mag, err, truth = synthetic_period_change(
        n=1500, span=800.0, period=5.0, pdot=5.0e-4 * 5.0, seed=21
    )
    res = cup.pdot_search((t, mag, err), period=truth["period"], backend="cpu")
    step = float(res.pdot_over_p[1] - res.pdot_over_p[0])
    assert res.best.pdot_over_p == pytest.approx(truth["pdot_over_p"], abs=2 * step)
    assert res.best.period == pytest.approx(truth["period"], rel=5e-4)
    assert res.best.power > 2.0 * res.best.zero_pdot_power


def test_seeding_run_when_no_period_given():
    t, mag, err = synthetic_sine(seed=5)
    res = cup.pdot_search((t, mag, err), backend="cpu")
    assert res.meta["pdot_search"]["seeded"] is True
    assert res.best.period == pytest.approx(0.6234, rel=1e-3)
    assert abs(res.best.pdot_over_p) <= 3 * float(
        res.pdot_over_p[1] - res.pdot_over_p[0]
    )


# --- structure and consistency ------------------------------------------------


def test_zero_row_is_the_standard_periodogram():
    t, mag, err, truth = synthetic_period_change(seed=42)
    res = cup.pdot_search((t, mag, err), period=truth["period"], backend="cpu")
    zero = res.zero_pdot_periodogram()
    grid = cup.GridSpec(kind="frequency", values=res.frequency, uniform=True)
    direct = cup.periodogram((t, mag, err), "GLS", grid=grid, backend="cpu")
    assert np.allclose(zero.power, direct.power, rtol=1e-9, atol=1e-12)
    assert zero.method == "GLS"


def test_axes_and_unit_conversions():
    t, mag, err, truth = synthetic_period_change(seed=42)
    res = cup.pdot_search((t, mag, err), period=truth["period"], backend="cpu")
    # The axis contains an exact zero and is uniform and ascending.
    assert np.any(res.pdot_over_p == 0.0)
    steps = np.diff(res.pdot_over_p)
    assert np.allclose(steps, steps[0])
    assert res.power.shape == res.shape == (
        res.pdot_over_p.size,
        res.frequency.size,
    )
    b = res.best
    assert b.pdot == pytest.approx(b.pdot_over_p / b.frequency)
    assert b.fdot == pytest.approx(-b.pdot_over_p * b.frequency)
    assert b.pdot_s_per_yr == pytest.approx(b.pdot * 86400.0 * 365.25)
    assert b.pdot_d_per_myr == pytest.approx(b.pdot * 365.25e6)
    assert b.timescale_yr == pytest.approx(abs(b.period / b.pdot) / 365.25)
    # t_ref is the weighted mean epoch of the finite points.
    lc = cup.LightCurve.from_arrays(t, mag, err)
    assert res.t_ref == pytest.approx(weighted_epoch(lc))
    # The 2-D pdot grid matches the axis / frequency outer ratio.
    assert res.pdot[3, 7] == pytest.approx(res.pdot_over_p[3] / res.frequency[7])


def test_profiles_and_to_dict():
    t, mag, err, truth = synthetic_period_change(n=800, span=800.0, seed=2)
    res = cup.pdot_search((t, mag, err), period=truth["period"], backend="cpu")
    assert res.pdot_profile().shape == (res.pdot_over_p.size,)
    assert res.frequency_profile().shape == (res.frequency.size,)
    assert np.max(res.pdot_profile()) == pytest.approx(np.max(res.power))
    d = res.to_dict()
    assert d["best"]["pdot"] == pytest.approx(res.best.pdot)
    assert "power" not in d
    assert "power" in res.to_dict(include_map=True)


# --- grids and settings -------------------------------------------------------


def test_explicit_pdot_range_and_grid():
    t, mag, err, truth = synthetic_period_change(n=800, span=800.0, seed=2)
    grid = cup.uniform_frequency_grid(
        800.0, maximum_frequency=2.02, minimum_frequency=1.98
    )
    settings = cup.PdotSettings(pdot_over_p_range=(-1e-6, 1e-6))
    res = cup.pdot_search((t, mag, err), grid=grid, settings=settings, backend="cpu")
    assert res.pdot_over_p[0] >= -1e-6 - 1e-15
    assert res.pdot_over_p[-1] <= 1e-6 + 1e-15
    assert np.array_equal(res.frequency, grid.frequency)


def test_axis_resolution_scales_as_derived():
    # d_beta = 4 / (s * f * T^2): doubling samples_per_drift halves the step.
    a1 = _pdot_axis(cup.PdotSettings(samples_per_drift=5), 2.0, 1000.0, 1.0)
    a2 = _pdot_axis(cup.PdotSettings(samples_per_drift=10), 2.0, 1000.0, 1.0)
    assert (a1[1] - a1[0]) == pytest.approx(2.0 * (a2[1] - a2[0]))
    # max_cycle_drift sets the half-range in accumulated drift cycles.
    a3 = _pdot_axis(cup.PdotSettings(max_cycle_drift=5.0), 2.0, 1000.0, 1.0)
    assert a3[-1] == pytest.approx(8.0 * 5.0 / (2.0 * 1000.0**2), rel=0.02)


def test_harmonic_scale_refines_step():
    from cuperiod.pdot.scan import _harmonic_scale

    assert _harmonic_scale(cup.get_method("GLS"), cup.GLSSettings()) == 1.0
    assert _harmonic_scale(cup.get_method("PDM"), cup.PDMSettings()) == 5.0
    assert (
        _harmonic_scale(cup.get_method("MHAOV"), cup.MHAOVSettings())
        == cup.MHAOVSettings().n_harmonics
    )
    a_sine = _pdot_axis(cup.PdotSettings(), 2.0, 1000.0, 1.0)
    a_fold = _pdot_axis(cup.PdotSettings(), 2.0, 1000.0, 5.0)
    assert (a_sine[1] - a_sine[0]) == pytest.approx(5.0 * (a_fold[1] - a_fold[0]))
    # The range is physical and does not grow with the scale.
    assert a_fold[-1] == pytest.approx(a_sine[-1], rel=0.02)
    # Explicit override wins over the per-method automatic value.
    a_over = _pdot_axis(cup.PdotSettings(harmonic_scale=2.0), 2.0, 1000.0, 5.0)
    assert (a_over[1] - a_over[0]) == pytest.approx(
        0.5 * (a_sine[1] - a_sine[0])
    )


def test_extreme_explicit_range_rejected():
    t, mag, err, truth = synthetic_period_change(
        n=600, span=800.0, period=5.0, seed=2
    )
    settings = cup.PdotSettings(pdot_over_p_range=(-3e-3, 3e-3))
    with pytest.raises(ValueError, match="domain"):
        cup.pdot_search(
            (t, mag, err), period=truth["period"], settings=settings, backend="cpu"
        )


def test_unit_conversions_wasp12_fixture():
    # Patra et al. (2017): dP/dt = -9.3e-10 for WASP-12b is -29 +- 3 ms/yr.
    # Guards the exact conversion a refereed paper once got backwards
    # (multiplying by the period instead of dividing, Wong et al. 2022).
    from cuperiod.pdot.result import PdotBest

    period = 1.0914203
    best = PdotBest(
        period=period,
        pdot=-9.3e-10,
        frequency=1.0 / period,
        fdot=9.3e-10 / period**2,
        pdot_over_p=-9.3e-10 / period,
        power=1.0,
        zero_pdot_power=1.0,
        t_ref=0.0,
    )
    assert best.pdot_ms_per_yr == pytest.approx(-29.35, abs=0.05)
    assert best.pdot_s_per_yr == pytest.approx(-0.02935, abs=5e-5)
    assert best.dlnp_dt_per_yr == pytest.approx(-9.3e-10 / period * 365.25)


def test_grid_cell_guard():
    t, mag, err, truth = synthetic_period_change(n=800, span=800.0, seed=2)
    settings = cup.PdotSettings(max_grid_cells=100)
    with pytest.raises(ValueError, match="max_grid_cells"):
        cup.pdot_search(
            (t, mag, err), period=truth["period"], settings=settings, backend="cpu"
        )


# --- guards -------------------------------------------------------------------


def test_bls_and_tls_rejected():
    t, mag, err, truth = synthetic_period_change(n=400, span=400.0, seed=2)
    for name in ("BLS", "TLS"):
        with pytest.raises(ValueError, match="phase_track"):
            cup.pdot_search((t, mag, err), period=1.0, method=name)


def test_insufficient_data():
    with pytest.raises(InsufficientDataError):
        cup.pdot_search((np.arange(8.0), np.zeros(8), np.ones(8)), period=1.0)


def test_bad_period():
    t, mag, err, _ = synthetic_period_change(n=400, span=400.0, seed=2)
    with pytest.raises(ValueError, match="positive"):
        cup.pdot_search((t, mag, err), period=-1.0)


# --- multiband ----------------------------------------------------------------


def test_multiband_scan_recovers_drift():
    t, mag, err, truth = synthetic_period_change(n=1600, pdot=1.0e-6, seed=8)
    bands = {
        "g": (t[0::2], mag[0::2] + 0.0, err[0::2]),
        "r": (t[1::2], mag[1::2] + 0.8, err[1::2]),
    }
    mblc = cup.MultiBandLightCurve.from_light_curves(
        {k: cup.LightCurve.from_arrays(*v) for k, v in bands.items()}
    )
    res = cup.pdot_search(mblc, period=truth["period"], backend="cpu")
    step = float(res.pdot_over_p[1] - res.pdot_over_p[0])
    assert res.best.period == pytest.approx(truth["period"], rel=2e-4)
    assert res.best.pdot_over_p == pytest.approx(truth["pdot_over_p"], abs=2 * step)


def test_multiband_constant_period():
    bands = synthetic_multiband_sine(seed=0)
    data = {k: cup.LightCurve.from_arrays(*v) for k, v in bands.items()}
    res = cup.pdot_search(data, period=0.7365, backend="cpu")
    step = float(res.pdot_over_p[1] - res.pdot_over_p[0])
    assert res.best.period == pytest.approx(0.7365, rel=1e-3)
    assert abs(res.best.pdot_over_p) <= 3 * step


# --- backends -----------------------------------------------------------------


@requires_gpu
def test_gpu_scan_matches_cpu():
    t, mag, err, truth = synthetic_period_change(n=1200, seed=42)
    cpu = cup.pdot_search((t, mag, err), period=truth["period"], backend="finufft")
    gpu = cup.pdot_search((t, mag, err), period=truth["period"], backend="cufinufft")
    assert gpu.backend == "cufinufft"
    assert np.allclose(cpu.power, gpu.power, rtol=1e-7, atol=1e-9)
    assert gpu.best.pdot_over_p == pytest.approx(cpu.best.pdot_over_p, abs=1e-12)


@requires_torch
def test_torch_scan_matches_cpu():
    t, mag, err, truth = synthetic_period_change(n=1200, seed=42)
    cpu = cup.pdot_search((t, mag, err), period=truth["period"], backend="finufft")
    tor = cup.pdot_search((t, mag, err), period=truth["period"], backend="torch:cpu")
    assert tor.backend.startswith("torch:")
    assert np.allclose(cpu.power, tor.power, rtol=1e-6, atol=1e-8)
