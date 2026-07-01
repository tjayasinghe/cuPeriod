"""Tests for the pure-numpy fold helpers."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.core.result import Peak
from cuperiod.gui.fold import epoch_for_peak, fold_phase, fold_series


def test_fold_phase_in_unit_interval() -> None:
    time = np.linspace(0.0, 10.0, 101)
    phase = fold_phase(time, 2.0)
    assert phase.min() >= 0.0
    assert phase.max() < 1.0
    assert phase.shape == time.shape


def test_fold_phase_epoch_shift() -> None:
    time = np.array([0.0, 0.5, 1.0])
    phase = fold_phase(time, 1.0, t0=0.0)
    np.testing.assert_allclose(phase, [0.0, 0.5, 0.0], atol=1e-12)


def test_fold_phase_two_cycles_length() -> None:
    time = np.linspace(0.0, 5.0, 50)
    phase = fold_phase(time, 1.3, two_cycles=True)
    assert phase.shape[0] == 2 * time.shape[0]
    assert phase.max() < 2.0


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_fold_phase_invalid_period(bad: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        fold_phase(np.linspace(0.0, 1.0, 10), bad)


def test_fold_series_two_cycles_tiles_value() -> None:
    time = np.linspace(0.0, 3.0, 30)
    value = np.arange(30.0)
    phase, tiled = fold_series(time, value, 1.0, two_cycles=True)
    assert phase.shape[0] == 60
    np.testing.assert_array_equal(tiled[:30], value)
    np.testing.assert_array_equal(tiled[30:], value)


def test_epoch_for_peak_prefers_t0() -> None:
    time = np.array([5.0, 6.0, 7.0])
    peak = Peak(period=1.0, frequency=1.0, power=1.0, rank=1, extra={"t0": 5.5})
    assert epoch_for_peak(peak, time) == 5.5


def test_epoch_for_peak_falls_back_to_min_time() -> None:
    time = np.array([5.0, 6.0, 7.0])
    peak = Peak(period=1.0, frequency=1.0, power=1.0, rank=1, extra={})
    assert epoch_for_peak(peak, time) == 5.0
    assert epoch_for_peak(None, time) == 5.0
