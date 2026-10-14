"""SessionSettings: defaults, validation and the CUPERIOD_SESSION_ env prefix."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import cuperiod as cup
from cuperiod.core.config import SessionSettings


def test_defaults() -> None:
    s = SessionSettings()
    assert s.recalculate_on_edit is True
    assert s.protect_manual is True
    assert s.undo_depth == 200
    assert s.derived_cache == 8
    assert s.engine_cache == 2
    assert s.collision_rayleigh == 0.25
    assert s.snap_rayleigh == 1.0
    assert s.min_subset_points == 10
    assert s.embed_data_max_points == 200_000
    assert s.max_actions == 100_000
    assert s.max_project_bytes == 64 * 1024 * 1024


def test_environment_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUPERIOD_SESSION_UNDO_DEPTH", "7")
    monkeypatch.setenv("CUPERIOD_SESSION_RECALCULATE_ON_EDIT", "false")
    s = SessionSettings()
    assert s.undo_depth == 7
    assert s.recalculate_on_edit is False


def test_validation() -> None:
    with pytest.raises(ValidationError):
        SessionSettings(undo_depth=0)
    with pytest.raises(ValidationError):
        SessionSettings(collision_rayleigh=float("nan"))
    with pytest.raises(ValidationError):
        SessionSettings(not_a_setting=1)  # type: ignore[call-arg]


def test_prewhiten_settings_are_unchanged() -> None:
    # The session adds a settings class; it does not add fields to PreWhitenSettings.
    assert "undo_depth" not in cup.PreWhitenSettings.model_fields
