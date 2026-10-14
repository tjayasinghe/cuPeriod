"""Image exports never overwrite a file the save dialog did not show, and never block.

A save dialog that does not append the filter's suffix hands back ``plot``; the export
then writes ``plot.png``, a file the dialog's own overwrite prompt never covered. The
views ask before replacing it, through the window's dialog provider (a non-blocking
question), not a modal message box.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

import cuperiod as cup
from cuperiod.gui.dialogs import FakeDialogProvider
from cuperiod.gui.widgets.phased_view import PhasedView
from cuperiod.gui.widgets.spectrum_view import SpectrumView


def _periodogram() -> cup.Periodogram:
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 30.0, 300))
    y = np.sin(2.0 * np.pi * t / 1.7) + rng.normal(0.0, 0.1, t.size)
    result = cup.periodogram((t, y), "GLS", backend="finufft")
    assert isinstance(result, cup.Periodogram)
    return result


def _spectrum(qtbot: QtBot) -> SpectrumView:
    view = SpectrumView()
    qtbot.addWidget(view)
    view.set_periodogram(_periodogram())
    return view


def _phased(qtbot: QtBot) -> PhasedView:
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0.0, 30.0, 300))
    view = PhasedView()
    qtbot.addWidget(view)
    view.set_light_curve(cup.LightCurve.from_arrays(t, np.sin(t), None))
    view.set_fold(1.7, 0.0)
    return view


@pytest.mark.parametrize("make", [_spectrum, _phased])
@pytest.mark.parametrize("replace", [False, True])
def test_existing_png_is_replaced_only_on_a_yes(
    qtbot: QtBot, tmp_path: Path, make, replace: bool
) -> None:
    view = make(qtbot)
    existing = tmp_path / "plot.png"
    existing.write_bytes(b"keep me")
    provider = FakeDialogProvider(answers=[replace])
    view.set_dialog_provider(provider)

    view._write_png(str(tmp_path / "plot"))  # the dialog returned no suffix

    assert [name for name, _ in provider.calls] == ["confirm"]
    title, text, accept = provider.calls[0][1]
    assert "plot.png" in text and accept == "Replace"
    if replace:
        assert existing.read_bytes() != b"keep me"  # an image was written
    else:
        assert existing.read_bytes() == b"keep me"


@pytest.mark.parametrize("make", [_spectrum, _phased])
def test_no_question_when_nothing_would_be_overwritten(
    qtbot: QtBot, tmp_path: Path, make
) -> None:
    view = make(qtbot)
    provider = FakeDialogProvider()
    view.set_dialog_provider(provider)

    view._write_png(str(tmp_path / "fresh"))  # fresh.png does not exist yet
    view._write_png(str(tmp_path / "chosen.png"))  # the dialog already asked

    assert provider.calls == []
    assert (tmp_path / "fresh.png").exists()
    assert (tmp_path / "chosen.png").exists()
