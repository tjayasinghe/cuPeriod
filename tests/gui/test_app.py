"""Tests for the application bootstrap and the console entry point."""

from __future__ import annotations

import os
from typing import Any

import pytest

import cuperiod.gui as gui
from cuperiod.gui import app

# -- GUI-A-20: a closed window must not keep a windowless process alive -------------


def test_entry_point_exits_hard_only_while_a_compute_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", exits.append)

    monkeypatch.setattr(app, "compute_active", lambda: True)
    assert app._finish(3, hard_exit=True) == 3
    assert exits == [3]  # an uninterruptible worker would otherwise hold the process

    exits.clear()
    assert app._finish(0, hard_exit=False) == 0  # embedded use is never killed
    monkeypatch.setattr(app, "compute_active", lambda: False)
    assert app._finish(0, hard_exit=True) == 0  # idle: a normal interpreter exit
    assert exits == []


def test_console_entry_point_asks_for_the_hard_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_run_app(argv: list[str] | None = None, **kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(app, "run_app", fake_run_app)
    with pytest.raises(SystemExit) as info:
        gui.main()
    assert info.value.code == 0
    assert seen == {"exit_when_closed": True}


# -- PySide6 6.12 aborts Python 3.11 ("deallocating None") -----------------------------


@pytest.mark.parametrize(
    ("python", "pyside", "refused"),
    [
        ((3, 11), "6.12.0", True),
        ((3, 11), "6.13.1", True),
        ((3, 11), "6.11.3", False),
        ((3, 12), "6.12.0", False),
        ((3, 11), "dev", False),
    ],
)
def test_known_broken_pyside_is_refused_with_the_fix(
    monkeypatch: pytest.MonkeyPatch,
    python: tuple[int, int],
    pyside: str,
    refused: bool,
) -> None:
    import sys

    import PySide6

    monkeypatch.setattr(sys, "version_info", (*python, 0, "final", 0))
    monkeypatch.setattr(PySide6, "__version__", pyside)
    message = gui._known_broken_qt()
    assert bool(message) is refused
    if refused:
        assert "PySide6<6.12" in message and pyside in message


def test_console_entry_point_stops_before_a_broken_pyside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[bool] = []
    monkeypatch.setattr(app, "run_app", lambda *a, **k: started.append(True) or 0)
    monkeypatch.setattr(gui, "_known_broken_qt", lambda: "PySide6 6.12.0 crashes")
    with pytest.raises(SystemExit) as info:
        gui.main()
    assert info.value.code == "PySide6 6.12.0 crashes"
    assert started == []
