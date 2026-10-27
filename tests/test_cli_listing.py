"""CLI: the ``methods`` table and ``doctor``'s torch section say what is true."""

from __future__ import annotations

import importlib
import re

import pytest
from typer.testing import CliRunner

from cuperiod.cli.app import app

runner = CliRunner()

CORE_BACKEND = importlib.import_module("cuperiod.core.backend")


def test_methods_table_columns_line_up() -> None:
    result = runner.invoke(app, ["methods"])
    assert result.exit_code == 0, result.output
    rows = [
        re.match(r"\s+(\S+)\s+(objective|min|max)\b", line)
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    assert all(rows), result.stdout
    names = {m.group(1) for m in rows if m is not None}
    assert {"STRINGLENGTH", "SUPERSMOOTHER", "GLS"} <= names
    # The objective column ends at one position on every row, the long names too.
    assert len({m.end(2) for m in rows if m is not None}) == 1, result.stdout


def _fake_torch(monkeypatch: pytest.MonkeyPatch, devices: set[str]) -> None:
    """Pretend torch is installed with ``devices``; no CUDA fast path."""
    order = ("cuda", "xpu", "mps")
    best = next((d for d in order if d in devices), "cpu")
    monkeypatch.setattr(CORE_BACKEND, "torch_available", lambda: True)
    monkeypatch.setattr(CORE_BACKEND, "torch_devices", lambda: set(devices))
    monkeypatch.setattr(CORE_BACKEND, "best_torch_device", lambda: best)
    monkeypatch.setattr(CORE_BACKEND, "cuda_available", lambda: False)
    methods_base = importlib.import_module("cuperiod.methods.base")
    monkeypatch.setattr(methods_base, "cuda_available", lambda: False)


def _torch_line(output: str, device: str) -> str:
    return next(line for line in output.splitlines() if f"torch:{device}" in line)


def test_doctor_does_not_credit_auto_with_a_cpu_torch_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With only torch's CPU device, backend='auto' resolves to numba/finufft/numpy.
    _fake_torch(monkeypatch, {"cpu"})
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    resolved = result.stdout.split("backend='auto' resolves to:", 1)[1]
    assert "torch" not in resolved
    line = _torch_line(result.stdout, "cpu")
    assert "<- best" in line and "backend='torch'" in line
    assert "auto" not in line.split("<- best", 1)[1]


def test_doctor_credits_auto_when_it_resolves_to_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_torch(monkeypatch, {"cpu", "mps"})
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    resolved = result.stdout.split("backend='auto' resolves to:", 1)[1]
    assert "torch" in resolved
    assert "'auto'" in _torch_line(result.stdout, "mps")
