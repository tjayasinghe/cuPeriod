"""Sliding-window (dynamic) periodograms: window placement, map, and drift trend."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.core.errors import InsufficientDataError, UnknownMethodError
from cuperiod.core.grid import GridSpec
from cuperiod.pdot.sliding import _MAX_GRID_POINTS, _shared_grid
from synth import synthetic_multiband_sine, synthetic_period_change, synthetic_sine

PERIOD = 0.6234
MB_PERIOD = 0.7365


def steady() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A constant-period sine, long enough for two dozen default windows."""
    return synthetic_sine(n=900, span=360.0, period=PERIOD, noise=0.02, seed=3)


def gapped() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same curve with a 70-day hole: several windows hold no points at all."""
    t, mag, err = steady()
    keep = ~((t - t[0] >= 150.0) & (t - t[0] < 220.0))
    return t[keep], mag[keep], err[keep]


def multiband() -> cup.MultiBandLightCurve:
    bands = synthetic_multiband_sine(
        band_points=(300, 250, 200), span=400.0, period=MB_PERIOD, noise=0.03
    )
    return cup.MultiBandLightCurve.from_light_curves(
        {name: cup.LightCurve.from_arrays(*arrays) for name, arrays in bands.items()}
    )


# --- a constant period stays put ---------------------------------------------


def test_constant_period_every_window() -> None:
    """No drift planted: every window recovers the true period, and so does the fit."""
    t, mag, err = steady()
    res = cup.sliding_periodogram((t, mag, err))
    assert res.n_windows >= 10
    assert res.power.shape == (res.n_windows, res.frequency.size)
    assert res.shape == res.power.shape
    assert np.all(np.abs(res.peak_period / PERIOD - 1.0) < 2e-3)
    pdot_over_p, err_trend = res.period_trend()
    assert err_trend > 0.0
    assert abs(pdot_over_p) < 5.0 * err_trend


def test_window_epochs_and_starts_are_ordered() -> None:
    """Rows are in time order, and each mid-epoch sits inside its own window."""
    t, mag, err = steady()
    res = cup.sliding_periodogram((t, mag, err))
    assert np.all(np.diff(res.t_start) > 0.0)
    assert np.all(np.diff(res.t_mid) > 0.0)
    assert np.all(res.t_mid >= res.t_start)
    assert np.all(res.t_mid <= res.t_start + res.window)
    assert res.t_start[0] == pytest.approx(float(t.min()))


# --- a drifting period drifts ------------------------------------------------


def test_drifting_period_trend() -> None:
    """A planted (dP/dt)/P is recovered to a factor of ~1.5, with the right sign."""
    t, mag, err, truth = synthetic_period_change(
        n=3000, span=1500.0, period=0.5, pdot=1.0e-6, seed=1
    )
    assert truth["pdot_over_p"] == pytest.approx(2.0e-6)
    res = cup.sliding_periodogram((t, mag, err))
    pdot_over_p, err_trend = res.period_trend()
    assert pdot_over_p > 0.0
    assert 1.0 / 1.5 < pdot_over_p / truth["pdot_over_p"] < 1.5
    assert err_trend < 0.5 * pdot_over_p
    # A lengthening period is a falling frequency, end to end.
    assert res.peak_frequency[-1] < res.peak_frequency[0]
    assert res.peak_period[-1] > res.peak_period[0]


# --- the map is made of real periodograms ------------------------------------


def test_window_periodogram_matches_its_row() -> None:
    """Row i as a Periodogram peaks where the stored per-window peak says it does."""
    t, mag, err = steady()
    res = cup.sliding_periodogram((t, mag, err))
    pg = res.window_periodogram(0)
    assert isinstance(pg, cup.Periodogram)
    assert pg.method == res.method
    assert np.array_equal(pg.power, res.power[0])
    resolution = float(res.frequency[1] - res.frequency[0])
    assert abs(1.0 / pg.best_period() - res.peak_frequency[0]) < resolution
    assert pg.meta["window_index"] == 0
    assert pg.meta["t_start"] == pytest.approx(float(res.t_start[0]))
    assert pg.meta["window"] == pytest.approx(res.window)


def test_to_dict_round_trip() -> None:
    """to_dict carries the per-window peaks always, the 2-D map on request."""
    t, mag, err = steady()
    res = cup.sliding_periodogram((t, mag, err))
    plain = res.to_dict()
    assert plain["method"] == "GLS"
    assert plain["n_windows"] == res.n_windows
    assert len(plain["peak_frequency"]) == res.n_windows
    assert "power" not in plain
    full = res.to_dict(include_map=True)
    assert np.asarray(full["power"]).shape == res.power.shape
    assert len(full["frequency"]) == res.frequency.size


# --- grids -------------------------------------------------------------------


def test_explicit_grid_is_honored() -> None:
    """An explicit GridSpec becomes the map's frequency axis verbatim."""
    t, mag, err = steady()
    grid = GridSpec(
        kind="frequency", values=np.linspace(1.0, 2.5, 501), uniform=True
    )
    res = cup.sliding_periodogram((t, mag, err), grid=grid)
    assert np.array_equal(res.frequency, grid.frequency)
    assert res.power.shape[1] == 501
    assert res.meta["sliding_periodogram"]["explicit_grid"] is True


def test_shared_grid_step_and_coarsening() -> None:
    """The shared axis samples the window peak 5x, and is capped in width."""
    grid, coarsened = _shared_grid(0.01, 5.0, 30.0)
    assert not coarsened
    assert float(grid.values[1] - grid.values[0]) == pytest.approx(1.0 / (5.0 * 30.0))
    assert grid.values[0] == pytest.approx(0.01)
    assert grid.values[-1] >= 5.0
    # A long window over a wide band would need millions of columns: coarsen.
    wide, coarsened = _shared_grid(0.01, 500.0, 3000.0)
    assert coarsened
    assert wide.size == _MAX_GRID_POINTS
    assert wide.values[-1] == pytest.approx(500.0)


def test_period_band_is_narrow_and_centered() -> None:
    """period= builds a +/-6 window-Rayleigh band at 10 samples per window peak."""
    t, mag, err = steady()
    res = cup.sliding_periodogram((t, mag, err), period=PERIOD)
    f_true = 1.0 / PERIOD
    step = float(res.frequency[1] - res.frequency[0])
    assert step == pytest.approx(1.0 / (10.0 * res.window))
    half_width = 0.5 * float(res.frequency[-1] - res.frequency[0])
    assert half_width == pytest.approx(6.0 / res.window)
    assert np.min(np.abs(res.frequency - f_true)) < 0.5 * step
    assert np.all(np.abs(res.peak_period / PERIOD - 1.0) < 2e-3)


# --- window placement --------------------------------------------------------


def test_default_window_and_step() -> None:
    """window=None -> baseline/12, step=None -> window/2."""
    t, mag, err = steady()
    baseline = float(t.max() - t.min())
    res = cup.sliding_periodogram((t, mag, err))
    assert res.baseline == pytest.approx(baseline)
    assert res.window == pytest.approx(baseline / 12.0)
    assert res.step == pytest.approx(res.window / 2.0)
    assert res.n_samples == t.size


def test_tail_window_is_flush_with_the_last_epoch() -> None:
    """A tail longer than half a step gets its own window, ending exactly at t_max."""
    t, mag, err = steady()
    t_max = float(t.max())
    res = cup.sliding_periodogram((t, mag, err), window=100.0, step=100.0)
    assert res.t_start[-1] + res.window == pytest.approx(t_max)
    assert res.t_start[-1] - res.t_start[-2] < res.step
    # The flush-right window is closed on the right, so it holds the last point.
    assert res.t_mid[-1] > res.t_start[-1]


# --- skipping and guards -----------------------------------------------------


def test_sparse_windows_are_skipped_and_counted() -> None:
    """Empty/sparse windows drop out of the map and are recorded in meta."""
    t, mag, err = gapped()
    res = cup.sliding_periodogram((t, mag, err), min_window_points=5)
    provenance = res.meta["sliding_periodogram"]
    assert provenance["n_windows_skipped"] >= 3
    assert res.n_windows + provenance["n_windows_skipped"] == (
        provenance["n_windows_placed"]
    )
    tight = cup.sliding_periodogram((t, mag, err), min_window_points=60)
    assert tight.meta["sliding_periodogram"]["min_window_points"] == 60
    assert (
        tight.meta["sliding_periodogram"]["n_windows_skipped"]
        > provenance["n_windows_skipped"]
    )
    assert tight.n_windows < res.n_windows


def test_too_few_windows_raises() -> None:
    """One window is not a time series: both placement failures are the same error."""
    t, mag, err = steady()
    with pytest.raises(InsufficientDataError, match="usable window"):
        cup.sliding_periodogram((t, mag, err), window=330.0, step=200.0)
    with pytest.raises(InsufficientDataError, match="does not fit"):
        cup.sliding_periodogram((t, mag, err), window=5000.0)
    with pytest.raises(InsufficientDataError, match="usable window"):
        cup.sliding_periodogram((t, mag, err), min_window_points=10_000)


def test_box_methods_are_rejected() -> None:
    """BLS/TLS trial grids carry box structure a shared frequency axis cannot hold."""
    t, mag, err = steady()
    with pytest.raises(ValueError, match="does not support BLS"):
        cup.sliding_periodogram((t, mag, err), method="BLS")


def test_unknown_method_raises_registry_error() -> None:
    t, mag, err = steady()
    with pytest.raises(UnknownMethodError, match="unknown method"):
        cup.sliding_periodogram((t, mag, err), method="NotAMethod")


def test_bad_window_step_and_period() -> None:
    t, mag, err = steady()
    with pytest.raises(ValueError, match="window must be positive"):
        cup.sliding_periodogram((t, mag, err), window=-1.0)
    with pytest.raises(ValueError, match="step must be positive"):
        cup.sliding_periodogram((t, mag, err), step=0.0)
    with pytest.raises(ValueError, match="period must be positive"):
        cup.sliding_periodogram((t, mag, err), period=0.0)


def test_period_trend_needs_three_windows() -> None:
    """The linear fit refuses to run on two points (it would have no residuals)."""
    t, mag, err = steady()
    res = cup.sliding_periodogram((t, mag, err))
    two = cup.SlidingPeriodogram(
        method=res.method,
        backend=res.backend,
        frequency=res.frequency,
        t_mid=res.t_mid[:2],
        t_start=res.t_start[:2],
        power=res.power[:2],
        peak_frequency=res.peak_frequency[:2],
        peak_power=res.peak_power[:2],
        objective_sense=res.objective_sense,
        window=res.window,
        step=res.step,
        n_samples=res.n_samples,
        baseline=res.baseline,
    )
    assert two.n_windows == 2
    with pytest.raises(InsufficientDataError, match="at least 3 windows"):
        two.period_trend()


# --- multi-band --------------------------------------------------------------


def test_multiband_smoke() -> None:
    """Bands are sliced on the same time range and pooled into one map."""
    res = cup.sliding_periodogram(multiband())
    assert res.method == "GLS"
    assert res.n_windows >= 10
    assert res.power.shape == (res.n_windows, res.frequency.size)
    assert np.all(np.isfinite(res.power))
    assert np.all(np.abs(res.peak_period / MB_PERIOD - 1.0) < 5e-3)


# --- other methods -----------------------------------------------------------


def test_min_objective_method_uses_the_right_sense() -> None:
    """PDM minimizes: each window's peak cell is the row's minimum, not its maximum."""
    t, mag, err = steady()
    res = cup.sliding_periodogram((t, mag, err), method="PDM", period=PERIOD)
    assert res.objective_sense == "min"
    assert res.method == "PDM"
    for i in range(res.n_windows):
        assert res.peak_power[i] == pytest.approx(float(np.nanmin(res.power[i])))
    assert np.all(np.abs(res.peak_period / PERIOD - 1.0) < 5e-3)
