"""The docs' shortcut table reads the same registry with or without PySide6.

``docs/_ext/cuperiod_shortcuts.py`` builds the keyboard-shortcut table from
:mod:`cuperiod.gui.commands`. Read the Docs installs no Qt, so the extension then loads
the registry under a stand-in for its Qt imports; this checks that the stand-in path
yields exactly the commands and keys the real module binds.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

EXTENSION = (
    Path(__file__).resolve().parents[1] / "docs" / "_ext" / "cuperiod_shortcuts.py"
)

#: The private name the extension loads the Qt-free copy of the registry under.
STAND_IN = "_cuperiod_docs_commands"

Binding = tuple[str, tuple[str, ...], tuple[Any, ...]]


def _loader() -> Callable[[], Any]:
    """The extension's registry loader, or skip when it is not there as described."""
    if not EXTENSION.is_file():
        pytest.skip(f"{EXTENSION} not found")
    pytest.importorskip("docutils")
    pytest.importorskip("sphinx")
    spec = importlib.util.spec_from_file_location("_test_cuperiod_shortcuts", EXTENSION)
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot load {EXTENSION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = getattr(module, "_registry", None)
    if not callable(loader):
        pytest.skip("the extension has no _registry() loader")
    return loader  # type: ignore[no-any-return]


def _bindings(registry: ModuleType) -> list[Binding]:
    """Command ids with their global key sequences and pane-local letters."""
    return [
        (c.id, tuple(c.keys), tuple(tuple(pair) for pair in c.context))
        for c in registry.COMMANDS
    ]


def test_registry_is_the_same_without_pyside6(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = _loader()
    normal = loader()
    expected = _bindings(normal)
    assert expected and any(keys for _, keys, _ in expected)

    import cuperiod.gui

    # Read the Docs: no PySide6 (nor pyqtgraph, which the GUI extra brings). The
    # registry and the Qt shim must be imported afresh, so drop the cached modules
    # and the package attribute ``from cuperiod.gui import commands`` would reuse.
    monkeypatch.setitem(sys.modules, "PySide6", None)
    monkeypatch.setitem(sys.modules, "pyqtgraph", None)
    for name in ("cuperiod.gui.commands", "cuperiod.gui.qt", STAND_IN):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for attribute in ("commands", "qt"):
        monkeypatch.delattr(cuperiod.gui, attribute, raising=False)
    try:
        stripped = loader()
    finally:
        # Never leave the stand-in copy (or a Qt-less shim) behind for later tests;
        # monkeypatch restores what was there before.
        sys.modules.pop(STAND_IN, None)
        sys.modules.pop("cuperiod.gui.commands", None)
        sys.modules.pop("cuperiod.gui.qt", None)
    assert stripped is not normal  # the Qt-free path ran
    assert stripped.MENUS == normal.MENUS
    assert _bindings(stripped) == expected
