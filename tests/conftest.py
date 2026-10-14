"""Shared pytest fixtures and GPU gating.

Tests written for a GPU carry the registered ``gpu`` marker, so ``pytest -m gpu`` runs
exactly those and ``pytest -m "not gpu"`` deselects them:

* :data:`requires_gpu` — the CUDA fast path (cupy + cufinufft and a CUDA device).
* :data:`requires_torch_gpu` — a non-CPU torch device (CUDA/ROCm, MPS, XPU).
* :func:`torch_auto_device` — a test that asks for bare ``backend="torch"``, which
  runs on the best torch device: it is GPU-marked only on machines where torch sees
  a GPU, and runs on ``torch:cpu`` (unmarked) elsewhere.

Deselecting is not enough on its own: many unmarked tests use the default
``backend="auto"``, which picks a GPU whenever one is visible. So a run whose ``-m``
expression contains ``not gpu`` also *hides* the GPUs for the whole session:
``CUDA_VISIBLE_DEVICES``/``HIP_VISIBLE_DEVICES`` are set to ``-1`` at configure time
(before anything initialises a CUDA runtime; worker processes the tests spawn inherit
it, and the previous values come back at unconfigure), and every unmarked test sees
cuperiod's device discovery report no CUDA device and a CPU-only torch (which also
covers MPS/XPU, which no environment variable hides). ``backend="auto"`` and bare
``"torch"`` then resolve to the CPU, so ``-m "not gpu"`` keeps the run off a shared
GPU. Without such an expression nothing is hidden, and unmarked ``"auto"`` tests use a
visible GPU as before.

Device checks happen at test setup, not at import: a deselected GPU test never imports
cupy or torch, and a machine with the GPU stack installed but no device *skips* GPU
tests instead of failing them.
"""

from __future__ import annotations

import functools
import os
import sys

# Windows: torch and numpy/scipy (MKL) each vendor an OpenMP runtime; importing torch
# after numpy aborts with "OMP: Error #15" unless the duplicate load is allowed. We set
# it in the test process (before torch is imported) so the torch-CPU tests run; it is
# harmless elsewhere. The library itself does not set this — see the install docs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Run the Qt GUI tests (tests/gui) headlessly; harmless for the non-Qt tests.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Windows: the offscreen platform has no font database of its own and falls back to
# placeholder glyphs whose metrics differ from any real font, so every layout test that
# measures text (minimum form widths, elided labels) would measure the fallback instead.
# Linux offscreen finds the system fonts through fontconfig and needs nothing here.
if sys.platform == "win32":
    os.environ.setdefault(
        "QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    )

# Windows: pyarrow's DLLs must load before the numerical stacks the tests exercise
# first. When pandas first imports pyarrow late in a session (after the torch/numba
# BLS tests, e.g. PreWhitenResult.to_dataframe), loading its native extension dies
# with an access violation that takes the whole run with it; loading it up front, as
# a plain ``import pandas`` at session start would, is stable. Optional dependency.
try:  # noqa: SIM105 - an ImportError is the expected "not installed" case
    import pyarrow  # noqa: E402, F401
except ImportError:
    pass

import importlib.util  # noqa: E402
import re  # noqa: E402
from collections.abc import Callable, Mapping  # noqa: E402
from typing import Any, Final, TypeVar  # noqa: E402

import pytest  # noqa: E402

_F = TypeVar("_F", bound=Callable[..., Any])

#: Environment that hides every device from the CUDA and ROCm runtimes.
_HIDE_GPU_ENV: Final[Mapping[str, str]] = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "HIP_VISIBLE_DEVICES": "-1",
}

#: ``not gpu`` as a whole marker name (``not gpu_extra`` names another marker; the
#: character class is pytest's marker-identifier alphabet).
_NOT_GPU = re.compile(r"(?:^|[\s(])not\s+gpu(?![\w:+\-.\[\]\\/])")

#: The environment values replaced by :func:`hide_gpu_environment` in this session;
#: present only in a run that deselects the ``gpu`` tests.
_HIDDEN_GPU_ENV = pytest.StashKey[dict[str, str | None]]()


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@functools.cache
def _cuda_stack_ready() -> tuple[bool, str]:
    """``(ready, reason)`` for the CUDA fast path: packages *and* a device."""
    if not (_importable("cupy") and _importable("cufinufft")):
        return False, "GPU stack (cupy/cufinufft) not installed"
    try:
        from cuperiod.core.backend import cuda_available

        if not cuda_available():
            return False, "GPU stack installed but no CUDA device is present"
    except Exception as exc:  # noqa: BLE001 - any probe failure means "no GPU"
        return False, f"CUDA probe failed: {exc}"
    return True, ""


@functools.cache
def _torch_gpu_available() -> bool:
    if not _importable("torch"):
        return False
    try:
        from cuperiod.core.backend import torch_gpu_available

        return torch_gpu_available()
    except Exception:
        return False


#: Mark a test as needing the CUDA fast path (cupy + cufinufft and a CUDA device).
#: Deselect with ``-m "not gpu"``; skipped at setup when no device is present.
requires_gpu = pytest.mark.gpu

#: Mark a test as needing a non-CPU torch device (CUDA/ROCm, MPS, or XPU). The portable
#: torch GPU numeric paths are written to spec and CPU-validated; on-hardware parity
#: self-skips until such a device is available.
requires_torch_gpu = pytest.mark.gpu(device="torch")

#: Skip a test unless PyTorch (the ``[torch]`` extra) is importable. The torch-CPU path
#: always runs once torch is installed; GPU-device parity adds the marker above.
requires_torch = pytest.mark.skipif(
    not _importable("torch"),
    reason="torch (the [torch] extra) is not installed",
)

#: Skip a test unless numba (the ``[fast]`` extra) is importable.
requires_numba = pytest.mark.skipif(
    not _importable("numba"),
    reason="numba (the [fast] extra) is not installed",
)


def torch_auto_device(fn: _F) -> _F:
    """Decorate a test that runs bare ``backend="torch"`` (the best torch device).

    Such a test runs everywhere, but lands on the GPU where torch sees one; collection
    adds the ``gpu`` marker there (see :func:`pytest_collection_modifyitems`) so
    ``-m "not gpu"`` keeps a GPU machine's run off the device.
    """
    marked: _F = pytest.mark.torch_auto_device(fn)
    return marked


def _gpu_skip_reason(marker: pytest.Mark) -> str | None:
    """Why a ``gpu``-marked test cannot run here, or ``None`` if it can."""
    device = marker.kwargs.get("device", "cuda")
    if device == "torch":
        if not _torch_gpu_available():
            return "no non-CPU torch device (CUDA/ROCm/MPS/XPU) available"
        return None
    if device == "any":
        return None
    ready, reason = _cuda_stack_ready()
    return None if ready else reason


def deselects_gpu(markexpr: str) -> bool:
    """Whether a ``-m`` expression deselects the ``gpu`` tests (contains ``not gpu``).

    A plain textual test, matching the documented ``-m "not gpu"`` and its
    conjunctions (``"not gpu and not slow"``); an expression that excludes the GPU
    tests some other way just leaves the GPUs visible, as before.
    """
    return _NOT_GPU.search(markexpr or "") is not None


def hide_gpu_environment() -> dict[str, str | None]:
    """Hide the CUDA/ROCm devices from this process and its children.

    Effective only before a CUDA runtime initialises (hence at configure time).
    Returns the replaced values for :func:`restore_environment`.
    """
    saved = {name: os.environ.get(name) for name in _HIDE_GPU_ENV}
    os.environ.update(_HIDE_GPU_ENV)
    return saved


def restore_environment(saved: Mapping[str, str | None]) -> None:
    """Put back environment values saved by :func:`hide_gpu_environment`."""
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def hide_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make cuperiod's device discovery report no GPU, for one test.

    Patches the two seams every resolver reads through ``cuperiod.core.backend``
    (``methods.base``, pre-whitening, the array-API layer and the CLI alike): the
    memoised CUDA probe and the torch device list. This also covers devices no
    environment variable hides (MPS, XPU) and a CUDA runtime that was initialised
    before the session hid the devices.
    """
    import cuperiod.core.backend as backend

    monkeypatch.setattr(backend, "_CUDA_PROBE", False)
    monkeypatch.setattr(
        backend,
        "torch_devices",
        lambda: {"cpu"} if backend.torch_available() else set(),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "torch_auto_device: runs bare backend='torch' (GPU-marked where torch has a "
        "GPU)",
    )
    if deselects_gpu(str(config.getoption("markexpr", default="") or "")):
        config.stash[_HIDDEN_GPU_ENV] = hide_gpu_environment()


def pytest_unconfigure(config: pytest.Config) -> None:
    saved = config.stash.get(_HIDDEN_GPU_ENV, None)
    if saved is not None:
        restore_environment(saved)
        del config.stash[_HIDDEN_GPU_ENV]


@pytest.fixture(autouse=True)
def _keep_unmarked_tests_off_hidden_gpus(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In a run that deselects ``gpu``, resolve ``auto``/bare ``torch`` to the CPU."""
    if _HIDDEN_GPU_ENV not in request.config.stash:
        return
    if request.node.get_closest_marker("gpu") is None:
        hide_gpus(monkeypatch)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # Runs before the -m deselection: a bare-"torch" test is a GPU test only on a
    # machine where torch actually has a GPU (probed once, and only if such a test
    # was collected). In a run that hides the GPUs it runs on torch's CPU device
    # instead, so it stays unmarked and selected.
    if _HIDDEN_GPU_ENV in config.stash:
        return
    auto = [item for item in items if item.get_closest_marker("torch_auto_device")]
    if auto and _torch_gpu_available():
        for item in auto:
            item.add_marker(pytest.mark.gpu(device="any"))


def pytest_runtest_setup(item: pytest.Item) -> None:
    for marker in item.iter_markers("gpu"):
        reason = _gpu_skip_reason(marker)
        if reason is not None:
            pytest.skip(reason)
