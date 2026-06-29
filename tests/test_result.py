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
