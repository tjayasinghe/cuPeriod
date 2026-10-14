"""Backend discovery (cuperiod.core.backend): memoised probes, selectors, DLL shim."""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import cuperiod.core.backend as backend
from cuperiod.methods.base import get_method


def _count_find_spec(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace ``importlib.util.find_spec`` with a recording pass-through."""
    calls: list[str] = []
    real = importlib.util.find_spec

    def counting(name: str, package: str | None = None) -> Any:
        calls.append(name)
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", counting)
    return calls


def test_discovery_is_memoised(monkeypatch: pytest.MonkeyPatch) -> None:
    # Re-probing the import system on every resolution cost ~1.5-3 ms per call
    # (more than a small PDM/CE periodogram); after the first probe, discovery and
    # CPU resolution must not touch importlib again.
    def resolve_all() -> None:
        get_method("GLS").resolve_backend("cpu")
        get_method("PDM").resolve_backend("auto")

    first = backend.available_backends()
    resolve_all()  # first round may import torch/cupy for the device probes
    calls = _count_find_spec(monkeypatch)
    second = backend.available_backends()
    resolve_all()
    assert calls == []
    # Equal content, but a fresh set: a caller mutating it cannot poison the cache.
    assert second == first
    assert second is not first
    second.add("bogus")
    assert "bogus" not in backend.available_backends()


def test_refresh_backends_reprobes(monkeypatch: pytest.MonkeyPatch) -> None:
    backend.available_backends()
    backend.refresh_backends()
    calls = _count_find_spec(monkeypatch)
    assert backend.has_module("numpy") is True
    assert backend.has_module("not_a_real_module_xyz") is False
    assert "numpy" in calls and "not_a_real_module_xyz" in calls
    calls.clear()
    backend.has_module("numpy")
    assert calls == []


def test_cpu_backends_never_probe_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> bool:
        raise AssertionError("a CPU request must not probe for a CUDA device")

    monkeypatch.setattr(backend, "cuda_available", boom)
    import cuperiod.methods.base as base

    monkeypatch.setattr(base, "cuda_available", boom)
    monkeypatch.setattr(base, "available_backends", boom)
    out = backend.cpu_backends()
    assert "numpy" in out and not (out & backend.GPU_BACKEND_NAMES)
    # Explicit CPU requests resolve without any GPU probe.
    assert get_method("GLS").resolve_backend("cpu") == "finufft"
    assert get_method("GLS").resolve_backend("finufft") == "finufft"
    assert get_method("BLS").resolve_backend("numpy") == "numpy"
    assert get_method("PDM").resolve_backend("numpy") == "numpy"


def test_cpu_run_does_not_import_cupy() -> None:
    # A CPU-only run used to import cupy (and load the CUDA DLLs) on its first
    # backend resolution, ~0.3 s in every spawned batch worker.
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        import cuperiod as cup
        from cuperiod.core.backend import array_module
        rng = np.random.default_rng(1)
        t = np.sort(rng.uniform(0, 30, 80))
        y = 12 + 0.1 * np.sin(2 * np.pi * t / 0.7)
        e = np.full(t.size, 0.01)
        cup.get_method("PDM").resolve_backend("cpu")
        cup.get_method("BLS").resolve_backend("cpu")
        cup.periodogram((t, y, e), "GLS", backend="finufft")
        assert array_module(np.zeros(2)) is np
        print("cupy" in sys.modules)
        """
    )
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "False", out.stderr


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("CPU", "cpu"), (" Auto ", "auto"), ("Torch:CUDA", "torch:cuda"), ("gpu", "gpu")],
)
def test_normalize_backend(raw: str, expected: str) -> None:
    assert backend.normalize_backend(raw) == expected


class _S:
    def __init__(self, backend: str) -> None:
        self.backend = backend


def test_requested_backend_falls_back_to_settings() -> None:
    assert backend.requested_backend("auto", None) == "auto"
    assert backend.requested_backend(None, None) == "auto"
    assert backend.requested_backend("auto", _S("astropy")) == "astropy"
    assert backend.requested_backend(None, _S("CPU")) == "cpu"
    # An explicit, non-auto request wins over the settings pin.
    assert backend.requested_backend("finufft", _S("astropy")) == "finufft"
    assert backend.requested_backend(" GPU ", _S("cpu")) == "gpu"


@pytest.mark.parametrize(
    ("requested", "pinned", "expected"),
    [
        ("auto", None, "cpu"),
        (None, None, "cpu"),
        ("auto", "auto", "cpu"),
        ("cpu", None, "cpu"),
        ("finufft", None, "finufft"),
        ("numba", None, "numba"),
        ("torch", None, "torch:cpu"),
        ("torch:cpu", None, "torch:cpu"),
        ("auto", "astropy", "astropy"),
        (None, "torch", "torch:cpu"),
    ],
)
def test_cpu_pool_backend_maps_to_cpu(
    requested: str | None, pinned: str | None, expected: str
) -> None:
    settings = None if pinned is None else _S(pinned)
    assert backend.cpu_pool_backend(requested, settings) == expected


@pytest.mark.parametrize(
    ("requested", "pinned"),
    [
        ("gpu", None),
        ("cupy", None),
        ("cufinufft", None),
        ("torch:cuda", None),
        ("auto", "gpu"),
        (None, "cupy"),
    ],
)
def test_cpu_pool_backend_rejects_gpu(
    requested: str | None, pinned: str | None
) -> None:
    settings = None if pinned is None else _S(pinned)
    with pytest.raises(ValueError, match="device='cpu'"):
        backend.cpu_pool_backend(requested, settings)


def test_cpu_pool_backend_never_resolves_to_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even on a machine where "auto" would take the CUDA path, the selector a CPU
    # pool worker resolves stays on the CPU.
    import cuperiod.methods.base as base

    monkeypatch.setattr(
        base,
        "available_backends",
        lambda: {"numpy", "finufft", "astropy", "numba", "cupy", "cufinufft"},
    )
    monkeypatch.setattr(base, "cuda_available", lambda: True)
    monkeypatch.setattr(base, "torch_gpu_available", lambda: True)
    for name in ("GLS", "BLS", "PDM", "CE", "MHAOV", "STRINGLENGTH", "TLS"):
        m = get_method(name)
        assert m.resolve_backend("auto") == m.gpu_backend
        resolved = m.resolve_backend(backend.cpu_pool_backend("auto"))
        assert not m.is_gpu_backend(resolved), (name, resolved)


def _fake_nvidia_tree(root: Path, *, wheel: bool) -> Path:
    """``root/nvidia/<pkg>/bin`` plus (optionally) an nvidia wheel's dist-info."""
    pkg = root / "nvidia" / ("cufft" if wheel else "evil")
    (pkg / "bin").mkdir(parents=True)
    if wheel:
        (root / "nvidia_cufft_cu12-0.0.dist-info").mkdir()
    return pkg / "bin"


def test_dll_shim_ignores_stray_nvidia_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `nvidia` is a namespace package: a folder of that name first on sys.path (the
    # script or notebook directory) used to be the ONLY directory added, hiding the
    # real CUDA runtime and letting that folder supply cudart/cufft DLLs.
    stray_root = tmp_path / "project"
    site_root = tmp_path / "site"
    stray_bin = _fake_nvidia_tree(stray_root, wheel=False)
    wheel_bin = _fake_nvidia_tree(site_root, wheel=True)
    added: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "add_dll_directory", added.append, raising=False)
    monkeypatch.setattr(backend, "_CUDA_DLL_READY", False)
    monkeypatch.syspath_prepend(str(site_root))
    monkeypatch.syspath_prepend(str(stray_root))
    monkeypatch.delitem(sys.modules, "nvidia", raising=False)
    importlib.invalidate_caches()
    backend.ensure_cuda_dll_path()
    resolved = {str(Path(p).resolve()) for p in added}
    assert str(wheel_bin.resolve()) in resolved
    assert str(stray_bin.resolve()) not in resolved
    assert all(not p.startswith(str(stray_root.resolve())) for p in resolved)
