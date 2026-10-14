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
