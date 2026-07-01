"""Shared pytest fixtures and GPU gating."""

from __future__ import annotations

import os

# Windows: torch and numpy/scipy (MKL) each vendor an OpenMP runtime; importing torch
# after numpy aborts with "OMP: Error #15" unless the duplicate load is allowed. We set
# it in the test process (before torch is imported) so the torch-CPU tests run; it is
# harmless elsewhere. The library itself does not set this — see the install docs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Run the Qt GUI tests (tests/gui) headlessly; harmless for the non-Qt tests.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util  # noqa: E402

import pytest  # noqa: E402


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


#: Skip a test unless the GPU stack (cupy + cufinufft) is importable.
requires_gpu = pytest.mark.skipif(
    not (_importable("cupy") and _importable("cufinufft")),
    reason="GPU stack (cupy/cufinufft) not installed",
)

#: Skip a test unless PyTorch (the ``[torch]`` extra) is importable. The torch-CPU path
#: always runs once torch is installed; GPU-device parity adds the marker below.
requires_torch = pytest.mark.skipif(
    not _importable("torch"),
    reason="torch (the [torch] extra) is not installed",
)


def _torch_gpu_available() -> bool:
    if not _importable("torch"):
        return False
    try:
        from cuperiod.core.backend import torch_gpu_available

        return torch_gpu_available()
    except Exception:
        return False


#: Skip a test unless torch sees a non-CPU device (CUDA/ROCm, MPS, or XPU). The portable
#: torch GPU numeric paths are written to spec and CPU-validated; on-hardware parity
#: self-skips until such a device is available (none on the current dev machine/CI).
requires_torch_gpu = pytest.mark.skipif(
    not _torch_gpu_available(),
    reason="no non-CPU torch device (CUDA/ROCm/MPS/XPU) available",
)
