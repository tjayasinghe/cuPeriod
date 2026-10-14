"""Sphinx directive ``shortcut-table``: the GUI's keyboard shortcuts, from the registry.

The table is generated at build time from :data:`cuperiod.gui.commands.COMMANDS` — the
same records the app builds its menus, its context letters and its F1 sheet from — so
the documentation cannot drift from the shortcuts the program actually binds.

Read the Docs installs only the ``docs`` extra, without PySide6. The registry's records
are plain dataclasses and its module touches Qt only inside methods, so when PySide6 is
missing the module is loaded once more under a private name with a stand-in for its Qt
import surface (:mod:`cuperiod.gui.qt`). Keys are printed in Qt's portable form
(``Ctrl+Shift+O``), which is also what the app shows on Windows and Linux.

Usage (MyST)::

    ```{shortcut-table}
    :menus: Pre-whitening, Data
    ```
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from docutils import nodes  # type: ignore[import-untyped]
from docutils.parsers.rst import directives  # type: ignore[import-untyped]
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

_HOSTS = {
    "spectrum": "spectrum",
    "table": "frequency table",
    "lightcurve": "light curve",
}


def _registry() -> Any:
    """The ``cuperiod.gui.commands`` module, with or without PySide6 installed."""
    try:
        from cuperiod.gui import commands
    except ImportError:
        pass
    else:
        return commands
    import cuperiod.gui

    path = Path(cuperiod.gui.__file__).with_name("commands.py")
    stub = types.ModuleType("cuperiod.gui.qt")

    class _QObject:  # the registry class is never instantiated here
        pass

    stub.QObject = _QObject  # type: ignore[attr-defined]
    stub.Signal = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    for name in ("Qt", "QtCore", "QtGui", "QtWidgets"):
        setattr(stub, name, types.SimpleNamespace())
    saved = sys.modules.get("cuperiod.gui.qt")
    sys.modules["cuperiod.gui.qt"] = stub
    try:
        name = "_cuperiod_docs_commands"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Registered before it runs: @dataclass looks its module up in sys.modules.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            sys.modules.pop("cuperiod.gui.qt", None)
        else:
            sys.modules["cuperiod.gui.qt"] = saved
    return module


def _keys(sequences: list[str]) -> nodes.paragraph:
    """``Ctrl+Shift+Z, Ctrl+Y`` as a paragraph of ``<kbd>`` literals."""
    para = nodes.paragraph()
    for i, key in enumerate(sequences):
        if i:
            para += nodes.Text(", ")
        para += nodes.literal(key, key, classes=["kbd"])
    return para


def _cell(content: nodes.Node | str) -> nodes.entry:
    entry = nodes.entry()
    entry += nodes.paragraph(text=content) if isinstance(content, str) else content
    return entry


class ShortcutTable(SphinxDirective):
    """A table of the registry's commands that carry a key: menu, command, keys."""

    has_content = False
    option_spec = {"menus": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        registry = _registry()
        wanted = None
        if "menus" in self.options:
            wanted = {m.strip() for m in self.options["menus"].split(",") if m.strip()}
            unknown = wanted - set(registry.MENUS)
            if unknown:
                raise self.error(f"unknown menu(s) {sorted(unknown)}")
        rows = []
        for command in registry.COMMANDS:
            if wanted is not None and command.menu not in wanted:
                continue
            if not (command.keys or command.context):
                continue
            panes: dict[str, list[str]] = {}
            for host, key in command.context:
                panes.setdefault(_HOSTS[host], []).append(key)
            local = nodes.paragraph()
            for i, (pane, keys) in enumerate(panes.items()):
                if i:
                    local += nodes.Text("; ")
                local.extend(_keys(keys).children)
                local += nodes.Text(f" ({pane})")
            rows.append(
                (command.menu, command.text, _keys(list(command.keys)), local)
            )

        table = nodes.table(classes=["shortcut-table"])
        group = nodes.tgroup(cols=4)
        table += group
        for width in (14, 36, 25, 25):
            group += nodes.colspec(colwidth=width)
        head = nodes.thead()
        row = nodes.row()
        for title in ("Menu", "Command", "Anywhere in the window", "In a focused pane"):
            row += _cell(title)
        head += row
        group += head
        body = nodes.tbody()
        for menu, text, global_keys, local in rows:
            row = nodes.row()
            row += _cell(menu)
            row += _cell(text)
            row += _cell(global_keys)
            row += _cell(local)
            body += row
        group += body
        return [table]


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_directive("shortcut-table", ShortcutTable)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
