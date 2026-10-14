"""GPU batch pools are sized for the most demanding method and the real grid.

Every pool worker runs every requested method, so the worker count must fit the
largest per-worker footprint, not just the first method listed; an explicit grid is
the real footprint. The probe itself (free memory, CUDA context size, MPS) is tested
in tests/test_device_workers.py; here it is replaced by fixed per-method answers.
"""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.batch import runner
from cuperiod.core.grid import GridSpec


@pytest.fixture
def fake_suggest(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int | None]]:
    calls: list[tuple[str, int | None]] = []
    per_method = {"GLS": 4, "BLS": 2, "PDM": 3}

    def suggest(method: str, *, grid_size: int | None = None) -> int:
        calls.append((method, grid_size))
        return per_method[method]

    monkeypatch.setattr(runner, "suggest_gpu_workers", suggest)
    return calls


def test_the_most_demanding_method_sets_the_count(fake_suggest) -> None:
    assert runner._gpu_worker_count(("GLS", "BLS", "PDM"), None, None) == 2
    assert [m for m, _ in fake_suggest] == ["GLS", "BLS", "PDM"]
    assert all(size is None for _, size in fake_suggest)


def test_an_explicit_grid_is_the_footprint(fake_suggest) -> None:
    grid = GridSpec(kind="frequency", values=np.linspace(0.1, 10.0, 12345))
    runner._gpu_worker_count(("GLS",), grid, None)
    assert fake_suggest == [("GLS", 12345)]


def test_an_explicit_worker_count_wins(fake_suggest) -> None:
    assert runner._gpu_worker_count(("GLS", "BLS"), None, 7) == 7
    assert runner._gpu_worker_count(("GLS",), None, 0) == 1
    assert fake_suggest == []
