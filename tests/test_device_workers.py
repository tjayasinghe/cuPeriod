"""suggest_gpu_workers: honest arithmetic on fake device info (CORE-B-12).

Every worker process pays a CUDA context (~200-235 MiB measured on an RTX 5070 Ti)
before its working set, and without NVIDIA MPS the processes take turns on the GPU
(measured: GLS throughput stops growing at 4 workers, BLS at 1). The suggestion is the
smallest of what fits in free memory, what the device can use (4 without MPS, 16 with)
and the host CPUs. No GPU is touched here: ``gpu_info`` and the core count are faked.
"""

from __future__ import annotations

import pytest

from cuperiod.core import device
from cuperiod.core.device import GpuInfo, suggest_gpu_workers
from cuperiod.methods.base import get_method

MIB = 1024**2
GIB = 1024**3


def _fake(monkeypatch, *, free: float, mps: bool = False, cores: int = 32) -> None:
    info = GpuInfo(
        device_id=0, name="Fake", total_bytes=int(16 * GIB), free_bytes=int(free),
        mps_enabled=mps,
    )
    monkeypatch.setattr(device, "gpu_info", lambda device_id=0: info)
    monkeypatch.setattr(device, "_host_cores", lambda: cores)


def _per_worker(method: str, grid: int) -> int:
    return device._CONTEXT_BYTES + get_method(method).estimate_device_bytes(grid)


def test_the_footprint_includes_a_cuda_context() -> None:
    working = get_method("GLS").estimate_device_bytes(100_000)
    assert device._estimate_worker_bytes("GLS", 100_000) == 256 * MIB + working
    # An unknown name still gets the context plus a generic working set.
    assert device._estimate_worker_bytes("NoSuchMethod", 1000) > 256 * MIB


def test_memory_bound_count_matches_the_arithmetic(monkeypatch) -> None:
    _fake(monkeypatch, free=1 * GIB, mps=True)
    per = _per_worker("GLS", device._DEFAULT_GRID_HINT)
    expected = int((1 * GIB * 0.8) // per)
    assert expected == 2  # 819 MiB usable / ~400 MiB per worker
    assert suggest_gpu_workers("GLS") == expected
    # A known grid size is used instead of the hint.
    per_small = _per_worker("GLS", 2000)
    assert suggest_gpu_workers("GLS", grid_size=2000) == int((0.8 * GIB) // per_small)


@pytest.mark.parametrize(
    ("mps", "cores", "expected"), [(False, 32, 4), (True, 32, 16), (True, 3, 3)]
)
def test_device_and_host_caps(monkeypatch, mps, cores, expected) -> None:
    _fake(monkeypatch, free=15 * GIB, mps=mps, cores=cores)
    assert suggest_gpu_workers("GLS") == expected
    assert suggest_gpu_workers("BLS") == expected


def test_small_or_full_cards_still_get_one_worker(monkeypatch) -> None:
    _fake(monkeypatch, free=100 * MIB)
    assert suggest_gpu_workers("MHAOV") == 1
    _fake(monkeypatch, free=0)
    assert suggest_gpu_workers("GLS") == 1


def test_count_never_falls_as_free_memory_grows(monkeypatch) -> None:
    previous = 0
    for gib in (0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16):
        _fake(monkeypatch, free=gib * GIB, mps=True)
        count = suggest_gpu_workers("PDM")
        assert 1 <= count <= 16
        assert count >= previous
        previous = count


@pytest.mark.parametrize("headroom", [float("nan"), float("inf"), -0.1, 1.0, 2.0])
def test_bad_headroom_is_refused(monkeypatch, headroom) -> None:
    _fake(monkeypatch, free=8 * GIB)
    with pytest.raises(ValueError, match="headroom"):
        suggest_gpu_workers("GLS", headroom=headroom)


def test_no_gpu_means_one_worker(monkeypatch) -> None:
    monkeypatch.setattr(device, "gpu_info", lambda device_id=0: None)
    assert suggest_gpu_workers("GLS") == 1


def test_mps_detection(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path))
    assert device._mps_active()
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY")
    # The Linux default pipe is only consulted on Linux.
    monkeypatch.setattr(device.sys, "platform", "win32")
    assert not device._mps_active()
    pipe = tmp_path / "control"
    pipe.write_text("")
    monkeypatch.setattr(device, "_MPS_DEFAULT_PIPE", str(tmp_path))
    monkeypatch.setattr(device.sys, "platform", "linux")
    assert device._mps_active()
