"""Shared pytest fixtures and GPU gating."""

from __future__ import annotations

import os

# Windows: torch and numpy/scipy (MKL) each vendor an OpenMP runtime; importing torch
# after numpy aborts with "OMP: Error #15" unless the duplicate load is allowed. We set
# it in the test process (before torch is imported) so the torch-CPU tests run; it is
# harmless elsewhere. The library itself does not set this — see the install docs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
#: always runs once torch is installed; GPU-device tests add their own skips in PR2.
requires_torch = pytest.mark.skipif(
    not _importable("torch"),
    reason="torch (the [torch] extra) is not installed",
)
