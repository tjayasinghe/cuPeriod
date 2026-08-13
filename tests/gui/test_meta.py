"""Tests for method/backend introspection and app-info helpers."""

from __future__ import annotations

from cuperiod.gui.appinfo import app_version, version_label
from cuperiod.gui.meta import (
    backend_options,
    method_display_names,
    multiband_method_names,
)


def test_multiband_method_names() -> None:
    multiband = set(multiband_method_names())
    # The joint-fit methods plus the pooled fold statistics; TLS is single-band only.
    assert multiband == {"GLS", "BLS", "MHAOV", "PDM", "CE", "STRINGLENGTH"}
    assert multiband.issubset(method_display_names())


def test_backend_options_always_include_auto_and_cpu() -> None:
    for name in method_display_names():
        options = backend_options(name)
        assert "auto" in options
        assert "cpu" in options
        assert len(options) == len(set(options))  # no duplicates


def test_version_label_format() -> None:
    label = version_label()
    assert label.startswith("v")
    assert app_version() in label
