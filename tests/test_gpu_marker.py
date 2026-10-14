"""The ``gpu`` marker plumbing in tests/conftest.py.

``pyproject.toml`` registers ``gpu`` with the promise "deselect with -m 'not gpu'", so
every test that runs on a GPU must carry the marker, the device check must skip (not
fail) on a machine with the GPU stack installed but no device, and a run that
deselects ``gpu`` must keep the remaining (``backend="auto"``) tests off the GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import conftest
from conftest import requires_gpu, requires_torch_gpu, torch_auto_device


def _mark_names(fn: Callable[..., Any]) -> list[str]:
    return [m.name for m in getattr(fn, "pytestmark", [])]


def test_requires_gpu_carries_the_gpu_marker() -> None:
    @requires_gpu
    def probe() -> None: ...

    assert "gpu" in _mark_names(probe)


def test_requires_torch_gpu_carries_the_gpu_marker() -> None:
    @requires_torch_gpu
    def probe() -> None: ...

    assert "gpu" in _mark_names(probe)


def test_torch_auto_device_is_marked_for_late_classification() -> None:
    @torch_auto_device
    def probe() -> None: ...

    assert "torch_auto_device" in _mark_names(probe)


def test_cuda_tests_skip_without_a_device(monkeypatch: pytest.MonkeyPatch) -> None:
    # cupy + cufinufft importable but no device: skip with a reason, do not fail.
    monkeypatch.setattr(conftest, "_cuda_stack_ready", lambda: (False, "no device"))
    reason = conftest._gpu_skip_reason(pytest.mark.gpu.mark)
    assert reason is not None and "no device" in reason
    monkeypatch.setattr(conftest, "_cuda_stack_ready", lambda: (True, ""))
    assert conftest._gpu_skip_reason(pytest.mark.gpu.mark) is None


def test_torch_gpu_tests_skip_without_a_torch_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conftest, "_torch_gpu_available", lambda: False)
    assert conftest._gpu_skip_reason(requires_torch_gpu.mark) is not None
    monkeypatch.setattr(conftest, "_torch_gpu_available", lambda: True)
    assert conftest._gpu_skip_reason(requires_torch_gpu.mark) is None


def test_this_module_is_not_gpu_marked(request: pytest.FixtureRequest) -> None:
    assert request.node.get_closest_marker("gpu") is None


# --- a run that deselects gpu keeps every other test off the GPU ----------------------


@pytest.mark.parametrize(
    ("markexpr", "deselects"),
    [
        ("not gpu", True),
        ("not  gpu", True),
        ("not gpu and not slow", True),
        ("slow and not gpu", True),
        ("(not gpu)", True),
        ("", False),
        ("gpu", False),
        ("slow", False),
        ("not slow", False),
        ("not gpu_extra", False),
        ("not gpu-extra", False),
    ],
)
def test_gpu_deselecting_expressions(markexpr: str, deselects: bool) -> None:
    assert conftest.deselects_gpu(markexpr) is deselects


def test_hiding_the_gpus_resolves_auto_to_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cuperiod.core.backend as backend
    from cuperiod.methods.base import get_method, method_names

    # Pretend this machine has a CUDA device and a torch GPU: "auto" takes the GPU.
    monkeypatch.setattr(backend, "_CUDA_PROBE", True)
    monkeypatch.setattr(backend, "torch_devices", lambda: {"cpu", "cuda"})
    assert get_method("PDM").resolve_backend("auto") == "cupy"
    conftest.hide_gpus(monkeypatch)
    assert backend.cuda_available() is False
    assert backend.torch_gpu_available() is False
    assert backend.best_torch_device() == "cpu"
    assert not {"cupy", "cufinufft"} & backend.available_backends()
    for name in method_names():
        method = get_method(name)
        resolved = method.resolve_backend("auto")
        assert not method.is_gpu_backend(resolved), (name, resolved)
        # Bare "torch" runs on the best torch device, now the CPU.
        if method.portable_gpu_backend == "torch":
            assert not method.is_gpu_backend("torch")


def test_hidden_gpu_environment_is_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    saved = conftest.hide_gpu_environment()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["HIP_VISIBLE_DEVICES"] == "-1"
    conftest.restore_environment(saved)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert "HIP_VISIBLE_DEVICES" not in os.environ


_PROBE_TEST = """
import os

import pytest

from cuperiod.core.backend import cuda_available, torch_gpu_available
from cuperiod.methods.base import get_method


def test_the_gpus_are_hidden():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["HIP_VISIBLE_DEVICES"] == "-1"
    assert not cuda_available()
    assert not torch_gpu_available()
    for name in ("GLS", "BLS", "PDM"):
        method = get_method(name)
        assert not method.is_gpu_backend(method.resolve_backend("auto"))


@pytest.mark.torch_auto_device
def test_a_bare_torch_test_stays_selected_on_the_cpu(request):
    assert request.node.get_closest_marker("gpu") is None
"""


def test_a_gpu_deselected_run_hides_the_devices(tmp_path: Path) -> None:
    # End to end: `pytest -m "not gpu"` with this conftest hides the CUDA/ROCm devices
    # from the session (and the worker processes it spawns) and resolves "auto" to
    # the CPU, so an unmarked test cannot land on the shared GPU.
    (tmp_path / "test_probe.py").write_text(_PROBE_TEST, encoding="utf-8")
    tests_dir = Path(conftest.__file__).resolve().parent
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(tests_dir), os.environ.get("PYTHONPATH", "")]
        ),
    }
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("HIP_VISIBLE_DEVICES", None)
    pytest_args = ["-p", "conftest", "-p", "no:cacheprovider", "-q", "-m", "not gpu"]
    out = subprocess.run(
        [sys.executable, "-m", "pytest", *pytest_args, "test_probe.py"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "2 passed" in out.stdout
