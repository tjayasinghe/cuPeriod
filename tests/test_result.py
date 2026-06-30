"""Periodogram result object: peak ranking and objective sense."""

from __future__ import annotations

import numpy as np

from cuperiod.core.result import Periodogram


def _make(power: np.ndarray, sense: str) -> Periodogram:
    freq = np.linspace(0.1, 2.0, power.size)
    return Periodogram.from_spectrum(
        method="TEST",
        backend="numpy",
        frequency=freq,
        power=power,
        objective_sense=sense,  # type: ignore[arg-type]
        n_samples=100,
        baseline=100.0,
    )


def test_best_periods_max_objective() -> None:
    power = np.zeros(200)
    power[50] = 5.0
    power[150] = 3.0
    pg = _make(power, "max")
    peaks = pg.best_periods(2)
    assert peaks[0].power == 5.0
    assert peaks[0].rank == 1
    assert peaks[1].power == 3.0


def test_best_periods_min_objective() -> None:
    stat = np.ones(200)
    stat[70] = 0.1  # a dip is the best peak for a minimized statistic
    pg = _make(stat, "min")
    peaks = pg.best_periods(1)
    assert peaks[0].frequency == pg.frequency[70]


def test_extras_attached_to_peaks() -> None:
    power = np.zeros(50)
    power[10] = 1.0
    freq = np.linspace(0.1, 2.0, 50)
    pg = Periodogram.from_spectrum(
        method="BLS",
        backend="numpy",
        frequency=freq,
        power=power,
        objective_sense="max",
        n_samples=100,
        baseline=100.0,
        extras={"depth": np.linspace(0.0, 0.5, 50)},
    )
    peak = pg.best_periods(1)[0]
    assert "depth" in peak.extra


def test_sorted_ascending_frequency() -> None:
    freq = np.array([2.0, 0.5, 1.0])
    power = np.array([1.0, 2.0, 3.0])
    pg = Periodogram.from_spectrum(
        method="T", backend="n", frequency=freq, power=power,
        objective_sense="max", n_samples=10, baseline=10.0,
    )
    assert np.all(np.diff(pg.frequency) > 0)
    # power must travel with its frequency
    assert pg.power[np.argmin(pg.frequency)] == 2.0


def test_best_period_at_low_frequency_edge() -> None:
    # Regression: the global maximum sitting on the first grid sample (a signal whose
    # period ~ the baseline) must not be discarded in favor of a smaller interior bump.
    power = np.zeros(200)
    power[0] = 10.0   # true peak at the low-frequency / long-period edge
    power[80] = 4.0   # a smaller interior local maximum
    pg = _make(power, "max")
    assert pg.best_period() == pg.period[0]
    assert pg.best_periods(2)[0].power == 10.0


def test_best_period_at_high_frequency_edge() -> None:
    power = np.zeros(200)
    power[-1] = 10.0  # true peak at the high-frequency / short-period edge
    power[80] = 4.0
    pg = _make(power, "max")
    assert pg.best_period() == pg.period[-1]


def test_min_objective_edge_peak() -> None:
    stat = np.ones(200)
    stat[0] = 0.0     # deepest (most significant) minimum at the edge
    stat[120] = 0.5
    pg = _make(stat, "min")
    assert pg.best_period() == pg.period[0]


def test_non_finite_samples_never_selected() -> None:
    # A frequency==0 sample (period inf) and a NaN-power sample must never be reported.
    freq = np.array([0.0, 0.1, 0.2, 0.3])
    power = np.array([np.nan, 1.0, 5.0, 2.0])
    pg = Periodogram.from_spectrum(
        method="X", backend="b", frequency=freq, power=power,
        objective_sense="max", n_samples=10, baseline=10.0,
    )
    peaks = pg.best_periods(4)
    assert all(np.isfinite(p.period) and np.isfinite(p.power) for p in peaks)
    assert peaks[0].power == 5.0


def test_all_non_finite_spectrum_is_safe() -> None:
    freq = np.array([0.1, 0.2, 0.3])
    power = np.array([np.nan, np.nan, np.nan])
    pg = Periodogram.from_spectrum(
        method="X", backend="b", frequency=freq, power=power,
        objective_sense="max", n_samples=10, baseline=10.0,
    )
    assert pg.best_periods(3) == []
    assert np.isnan(pg.best_period())
