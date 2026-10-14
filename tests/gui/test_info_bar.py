"""Tests for the run-info bar: device badge, idle/failed/stale states, plain text."""

from __future__ import annotations

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from cuperiod.core.device import GpuInfo
from cuperiod.core.result import Periodogram
from cuperiod.gui.qt import Qt
from cuperiod.gui.widgets import info_bar
from cuperiod.gui.widgets.info_bar import RunInfoBar


def _periodogram(backend: str = "numpy", method: str = "GLS") -> Periodogram:
    freq = np.linspace(0.1, 2.0, 500)
    power = np.exp(-((freq - 0.5) ** 2) / 0.001)
    return Periodogram.from_spectrum(
        method=method,
        backend=backend,
        frequency=freq,
        power=power,
        objective_sense="max",
        n_samples=100,
        baseline=90.0,
    )


class _FakeProbe:
    """Stand-in for ``gpu_info``: a CUDA card that records every probe."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> GpuInfo:
        self.calls += 1
        return GpuInfo(
            device_id=0,
            name="NVIDIA GeForce RTX 4090",
            total_bytes=24 * 1024**3,
            free_bytes=20 * 1024**3,
            mps_enabled=False,
        )


# -- CORE-B-18 / GUI-B-4: the device badge -----------------------------------------


def test_torch_cpu_runs_are_reported_on_the_cpu_without_a_cuda_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _FakeProbe()
    monkeypatch.setattr(info_bar, "gpu_info", probe)
    assert RunInfoBar._device_text("torch:cpu") == "device: CPU"
    assert RunInfoBar._device_text("numba") == "device: CPU"
    # Another torch device is named from the backend itself, never the CUDA card.
    assert RunInfoBar._device_text("torch:mps") == "device: GPU (MPS)"
    assert probe.calls == 0
    # CUDA paths still name the card.
    assert RunInfoBar._device_text("torch:cuda") == "device: RTX 4090 (20/24 GiB)"
    assert RunInfoBar._device_text("cufinufft") == "device: RTX 4090 (20/24 GiB)"
    assert probe.calls == 2


def test_a_torch_cpu_result_shows_cpu_on_a_cuda_machine(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _FakeProbe()
    monkeypatch.setattr(info_bar, "gpu_info", probe)
    bar = RunInfoBar()
    qtbot.addWidget(bar)
    bar.set_result(_periodogram(backend="torch:cpu"))
    assert bar._device.text() == "device: CPU"
    assert probe.calls == 0


def test_set_idle_drops_the_previous_runs_backend_tooltips(qtbot: QtBot) -> None:
    bar = RunInfoBar()
    qtbot.addWidget(bar)
    bar.set_result(_periodogram(backend="numpy"))
    assert bar._backend.toolTip() == "backend: numpy"
    bar.set_idle()
    assert bar._backend.toolTip() == ""
    assert bar._device.toolTip() == ""
    # The static explanations of the other badges are not run data and stay.
    assert "samples" in bar._samples.toolTip().lower()


# -- UX-22: idle states ------------------------------------------------------------


def test_idle_bar_shows_one_line_instead_of_six_dashes(qtbot: QtBot) -> None:
    bar = RunInfoBar()
    qtbot.addWidget(bar)
    bar.show()
    assert bar._method.isVisible()
    assert bar._method.text() == info_bar.IDLE_TEXT
    assert not any(badge.isVisible() for badge in bar._data_badges)
    bar.set_result(_periodogram())
    assert all(badge.isVisible() for badge in bar._data_badges)
    bar.set_idle("not computed yet")
    assert bar._method.text() == "not computed yet"
    assert not any(badge.isVisible() for badge in bar._data_badges)


# -- GUI-B-5 / UX-14: a failure does not blank a result still on screen -------------


def test_a_failed_run_keeps_describing_the_displayed_result(qtbot: QtBot) -> None:
    bar = RunInfoBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.set_result(_periodogram(method="GLS"))
    bar.set_busy(True, "PDM")
    assert "computing" in bar._method.text()
    bar.set_failed("PDM")
    assert bar._method.text() == "GLS"
    assert bar._backend.text() == "backend: numpy"
    assert not bar._busy.isVisible()


def test_a_failed_first_run_says_so(qtbot: QtBot) -> None:
    bar = RunInfoBar()
    qtbot.addWidget(bar)
    bar.set_busy(True, "PDM")
    bar.set_failed("PDM")
    assert bar._method.text() == "PDM · failed"


# -- UX-23: a result that no longer matches the controls is flagged -----------------


def test_stale_flag_marks_the_leading_badge_until_the_next_result(qtbot: QtBot) -> None:
    bar = RunInfoBar()
    qtbot.addWidget(bar)
    bar.set_stale(True)  # nothing displayed: nothing to flag
    assert not bar.is_stale()
    bar.set_result(_periodogram(method="GLS"))
    bar.set_stale(True)
    assert bar.is_stale()
    assert "settings changed" in bar._method.text()
    assert "compute again" in bar._method.toolTip().lower()
    bar.set_result(_periodogram(method="PDM"))
    assert not bar.is_stale()
    assert bar._method.text() == "PDM"


# -- SEC-03 hardening: badges never parse markup -------------------------------------


def test_badges_are_plain_text(qtbot: QtBot) -> None:
    bar = RunInfoBar()
    qtbot.addWidget(bar)
    bar.set_result(_periodogram(method="<b>X</b>"))
    assert bar._method.textFormat() == Qt.TextFormat.PlainText
    assert bar._method.text() == "<b>X</b>"
