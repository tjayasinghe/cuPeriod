"""Grid construction."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.core.grid import GridSpec, log_period_grid, uniform_frequency_grid


def test_uniform_frequency_grid_spacing() -> None:
    grid = uniform_frequency_grid(
        100.0, maximum_frequency=2.0, minimum_frequency=0.01, samples_per_peak=5
    )
    assert grid.kind == "frequency"
    assert grid.uniform
    df = np.diff(grid.values)
    assert np.allclose(df, df[0])
    assert df[0] == pytest.approx(1.0 / (5 * 100.0))
    assert grid.values[-1] >= 2.0


def test_grid_frequency_period_views() -> None:
    grid = GridSpec(kind="frequency", values=np.array([0.5, 1.0, 2.0]), uniform=False)
    assert np.allclose(grid.frequency, [0.5, 1.0, 2.0])
    assert np.allclose(grid.period, [0.5, 1.0, 2.0])  # 1/f reversed -> ascending


def test_uniform_frequency_params() -> None:
    grid = uniform_frequency_grid(50.0, maximum_frequency=5.0, samples_per_peak=4)
    f0, df, nf = grid.uniform_frequency_params()
    assert nf == grid.size
    assert df == pytest.approx(1.0 / (4 * 50.0))


def test_log_period_grid() -> None:
    grid = log_period_grid(minimum_period=0.5, maximum_period=50.0, n_periods=100)
    assert grid.kind == "period"
    assert grid.size == 100
    assert grid.values[0] == pytest.approx(0.5)
    assert grid.values[-1] == pytest.approx(50.0)


def test_invalid_grids_raise() -> None:
    with pytest.raises(ValueError):
        uniform_frequency_grid(-1.0, maximum_frequency=2.0)
    with pytest.raises(ValueError):
        log_period_grid(minimum_period=5.0, maximum_period=1.0, n_periods=10)


def test_uniform_frequency_grid_rejects_min_ge_max() -> None:
    # Regression: transposed bounds must raise, not return a 1-sample grid above max.
    with pytest.raises(ValueError, match="must be <"):
        uniform_frequency_grid(10.0, maximum_frequency=0.5, minimum_frequency=2.0)
