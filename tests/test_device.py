"""GPU device helpers (CPU-only behavior; GPU paths exercised on hardware)."""

from __future__ import annotations

from cuperiod.core.backend import cuda_available
from cuperiod.core.device import free_gpu_memory, gpu_info, suggest_gpu_workers


def test_suggest_gpu_workers_at_least_one() -> None:
    assert suggest_gpu_workers("GLS") >= 1


def test_gpu_info_none_without_device() -> None:
    if not cuda_available():
        assert gpu_info() is None


def test_free_gpu_memory_is_safe() -> None:
    free_gpu_memory()  # must be a no-op without a GPU, never raise
