"""A session grid given as ``(f0, df, nf)`` goes through the size guard (SEC-04).

Every grid builder refuses more than ``MAX_GRID_SIZE`` samples before allocating
(``GridTooLargeError``). ``PreWhitenSession(grid=(f0, df, nf))`` built its grid
straight from the tuple, so ``nf = 10**12`` asked numpy for 8 TB; a non-positive
step or an empty grid failed later, deep in the spectrum engine.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.prewhiten.session import PreWhitenSession
from cuperiod.prewhiten.session.kernel import ExactUniformGrid
from synth import synthetic_sine


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return synthetic_sine(n=200, span=30.0, period=0.3, seed=1)


def test_an_oversized_grid_tuple_is_refused_before_allocating() -> None:
    data = _data()
    tracemalloc.start()
    try:
        with pytest.raises(cup.GridTooLargeError, match="pre-whitening session grid"):
            PreWhitenSession(data, grid=(0.01, 1e-6, 10**12))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 64e6  # nothing near the 8 TB the tuple asked for


@pytest.mark.parametrize(
    ("grid", "match"),
    [
        ((0.01, 0.0, 100), "df > 0"),
        ((0.01, -0.01, 100), "df > 0"),
        ((-1.0, 0.01, 100), "f0 >= 0"),
        ((float("nan"), 0.01, 100), "f0 >= 0"),
        ((0.01, float("inf"), 100), "df > 0"),
        ((0.01, 0.01, 0), "nf >= 1"),
    ],
)
def test_a_malformed_grid_tuple_is_refused(
    grid: tuple[float, float, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        PreWhitenSession(_data(), grid=grid)


def test_a_valid_grid_tuple_is_exact() -> None:
    grid = ExactUniformGrid.of((0.5, 0.005, 3000))
    assert grid.uniform_frequency_params() == (0.5, 0.005, 3000)
    assert grid.values.size == 3000 and grid.values[0] == 0.5
    session = PreWhitenSession(_data(), grid=(0.5, 0.005, 3000))
    assert tuple(session.state.grid) == (0.5, 0.005, 3000)
