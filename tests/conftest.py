"""Shared pytest fixtures and GPU gating."""

from __future__ import annotations

import importlib.util

import pytest


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
