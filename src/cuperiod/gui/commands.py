"""The single registry of user commands: actions, menus, shortcuts and their sheet.

Every command is one :class:`Command` record. :class:`CommandRegistry` turns each into
one window-level ``QAction`` (text, tooltip ending in its shortcut, status tip, object
name, accessible name, the global key with ``Qt.WindowShortcut``) that the menu bar, the
session bar and the context menus all share, **plus** one ``QShortcut`` per context
letter, parented to the spectrum, the frequency table or the light-curve view with
``Qt.WidgetWithChildrenShortcut`` and forwarding to the action's ``trigger()``. A
``QAction`` has a single shortcut context, so it cannot carry both a window key and a
pane-local letter; the letters follow the action's enabled state and never fire from
the Options form's line edits, combos or check boxes (they are outside every host).

Invariant (tested over actions *and* context shortcuts): no two enabled bindings share
a key sequence in one context — the window, or one host widget — and ``session.stop``
and ``data.clear_selection`` (both Esc) are never enabled together. F10 is not used.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from cuperiod.gui.qt import QObject, Qt, QtGui, QtWidgets, Signal

#: Panes that carry context letters.
Host = Literal["spectrum", "table", "lightcurve"]

#: When a command is enabled (see :meth:`CommandRegistry.set_state`).
#: ``mutating`` — a session is open and no long job runs; ``session`` — a session is
#: open (read-only, allowed while busy); ``idle`` — no long job runs (no session
#: needed); ``always``; ``stop`` — only while a long job (or a one-shot run) runs;
#: ``selection`` — a session is open and nothing is busy (shares Esc with ``stop``).
Availability = Literal["mutating", "session", "idle", "always", "stop", "selection"]

MENUS: tuple[str, ...] = ("File", "Edit", "Pre-whitening", "Data", "View", "Help")

_HOST_NAMES: Mapping[str, str] = {
    "spectrum": "spectrum",
    "table": "frequency table",
    "lightcurve": "light curve",
}


@dataclass(frozen=True)
class Command:
    """One user command.

    Attributes
    ----------
    id : str
        Stable dotted id (``"session.step"``), also the action's object name stem.
    text : str
        Menu text (no mnemonic ampersands; the accessible name derives from it).
    menu : str
        One of :data:`MENUS`.
    tip : str
        Status-bar tip, one sentence.
    keys : tuple of str
        Global key sequences (``Qt.WindowShortcut``), portable text form.
    context : tuple of (host, key)
        Pane-local letters (``Qt.WidgetWithChildrenShortcut`` on that host).
    availability : str
        See :data:`Availability`.
    checkable : bool
        A tool toggle (exclusive within ``group``).
    group : str
        Exclusive action group name for checkable tools.
    """

    id: str
    text: str
    menu: str
    tip: str
    keys: tuple[str, ...] = ()
    context: tuple[tuple[Host, str], ...] = ()
    availability: Availability = "mutating"
    checkable: bool = False
    group: str = ""

    @property
    def object_name(self) -> str:
        return "action_" + self.id.replace(".", "_")

    @property
    def accessible_name(self) -> str:
        return self.text.rstrip("…").strip()

    def key_text(self) -> str:
        """Human-readable bindings: global keys, then the context letters."""
        parts = [_native(k) for k in self.keys]
        parts += [f"{_native(k)} ({_HOST_NAMES[h]})" for h, k in self.context]
        return ", ".join(parts)

    @property
    def tooltip(self) -> str:
        keys = self.key_text()
        return f"{self.text}  ({keys})" if keys else self.text


def _native(key: str) -> str:
    return QtGui.QKeySequence(key).toString(
        QtGui.QKeySequence.SequenceFormat.NativeText
    )


_LETTERS_SPECTRUM_TABLE: tuple[Host, ...] = ("spectrum", "table")


def _both(key: str) -> tuple[tuple[Host, str], ...]:
    return tuple((h, key) for h in _LETTERS_SPECTRUM_TABLE)


#: The registry table (design 4.5). Order is menu order.
COMMANDS: tuple[Command, ...] = (
    # -- File ------------------------------------------------------------------------
    Command("file.new_session", "New session", "File",
            "Discard the current session's work and start again from the light curve",
            ("Ctrl+N",), availability="idle"),
    Command("file.open_curve", "Open light curve…", "File",
            "Open a light-curve file", ("Ctrl+O",), availability="idle"),
    Command("file.open_project", "Open project…", "File",
            "Open a saved pre-whitening project (*.cupw.json)", ("Ctrl+Shift+O",),
            availability="idle"),
    Command("file.save_project", "Save project", "File",
            "Save the session as a project file", ("Ctrl+S",), availability="session"),
    Command("file.save_project_as", "Save project as…", "File",
            "Save the session under a new project file name", ("Ctrl+Shift+S",),
            availability="session"),
    Command("file.export_script", "Export script…", "File",
            "Write the session's actions as a Python script", ("Ctrl+E",),
            availability="session"),
    Command("file.export_frequencies", "Export frequencies…", "File",
            "Write the frequency table (with errors) to a file",
            availability="session"),
    Command("file.verify_project", "Verify project…", "File",
            "Replay a project's log and compare it with the stored result",
            availability="idle"),
    # -- Edit ------------------------------------------------------------------------
    Command("edit.undo", "Undo", "Edit", "Undo the last action", ("Ctrl+Z",)),
    Command("edit.redo", "Redo", "Edit", "Redo the last undone action",
            ("Ctrl+Shift+Z", "Ctrl+Y")),
    Command("edit.delete", "Delete", "Edit",
            "Remove the selected frequencies, or delete the selected points",
            context=(("table", "Del"), ("lightcurve", "Del"))),
    # -- Pre-whitening ---------------------------------------------------------------
    Command("session.auto", "Auto — run to the stopping rule", "Pre-whitening",
            "Extract frequencies until the stopping rule ends the search",
            ("F5", "Ctrl+Return", "Ctrl+Enter")),
    Command("session.step", "Step", "Pre-whitening",
            "Try the tallest untested peak once (one iteration of Auto)", ("F7",),
            _both("S")),
    Command("session.add_peak", "Add peak (tallest, untested)", "Pre-whitening",
            "Add the tallest untested peak without the acceptance test", ("F6",)),
    Command("session.add_frequency", "Add at frequency…", "Pre-whitening",
            "Type a frequency to add", ("Ctrl+F6",)),
    Command("session.add_at_cursor", "Add peak at cursor", "Pre-whitening",
            "Add the peak nearest the spectrum's pick cursor",
            context=(("spectrum", "A"), ("spectrum", "Ins"))),
    Command("session.calculate", "Calculate (frequencies held)", "Pre-whitening",
            "Re-fit amplitudes and phases with the frequencies held, with errors",
            ("F8",), _both("C")),
    Command("session.improve_all", "Improve all", "Pre-whitening",
            "Re-fit every free frequency, amplitude and phase", ("F9",), _both("I")),
    Command("session.improve_selected", "Improve selected", "Pre-whitening",
            "Re-fit the selected components only", ("Shift+F9",), _both("Shift+I")),
    Command("session.toggle_use", "Use / don't use", "Pre-whitening",
            "Include or exclude the selected components from the model",
            context=(("table", "Space"),)),
    Command("session.fix_frequency", "Fix frequency", "Pre-whitening",
            "Hold the selected frequencies at their values",
            context=(("table", "L"),)),
    Command("session.fix_all", "Fix all three", "Pre-whitening",
            "Hold frequency, amplitude and phase of the selected components",
            context=(("table", "Shift+L"),)),
    Command("session.tie", "Edit tie…", "Pre-whitening",
            "Tie the selected frequency to a combination (e.g. F1+F2)",
            context=(("table", "T"),)),
    Command("session.finalize", "Polish and prune", "Pre-whitening",
            "Final simultaneous fit, then drop components below the S/N limit"),
    Command("session.renumber", "Renumber", "Pre-whitening",
            "Number the components F1..FK in table order"),
    Command("session.stop", "Stop", "Pre-whitening",
            "Stop the running computation; what is done so far is kept", ("Esc",),
            availability="stop"),
    # -- Data ------------------------------------------------------------------------
    Command("data.restore_selected", "Restore selected points", "Data",
            "Bring the selected deleted points back",
            context=(("lightcurve", "U"),)),
    Command("data.restore_all", "Restore all points", "Data",
            "Bring every deleted point back", ("Ctrl+Shift+R",)),
    Command("data.tool_pan", "Pan", "Data",
            "Drag to pan the light curve (Shift-drag selects)",
            context=(("lightcurve", "V"),), availability="session", checkable=True,
            group="tool"),
    Command("data.tool_rect", "Rectangle select", "Data",
            "Drag a rectangle to select points", context=(("lightcurve", "X"),),
            availability="session", checkable=True, group="tool"),
    Command("data.tool_lasso", "Lasso select", "Data",
            "Draw a lasso to select points", context=(("lightcurve", "G"),),
            availability="session", checkable=True, group="tool"),
    Command("data.clear_selection", "Clear selection", "Data",
            "Deselect every point", context=(("lightcurve", "Esc"),),
            availability="selection"),
    # -- View ------------------------------------------------------------------------
    Command("view.focus_spectrum", "Focus spectrum", "View",
            "Move the keyboard focus to the spectrum", ("Ctrl+1",),
            availability="always"),
    Command("view.focus_frequencies", "Focus frequency table", "View",
            "Move the keyboard focus to the frequency table", ("Ctrl+2",),
            availability="always"),
    Command("view.focus_light_curve", "Focus light curve", "View",
            "Move the keyboard focus to the light curve", ("Ctrl+3",),
            availability="always"),
    Command("view.focus_subsets", "Focus subsets", "View",
            "Move the keyboard focus to the subsets panel", ("Ctrl+4",),
            availability="always"),
    Command("view.focus_history", "Focus history", "View",
            "Move the keyboard focus to the history panel", ("Ctrl+5",),
            availability="always"),
    # -- Help ------------------------------------------------------------------------
    Command("view.shortcuts", "Keyboard shortcuts", "Help",
            "List every keyboard shortcut", ("F1",), availability="always"),
)

COMMAND_IDS: frozenset[str] = frozenset(c.id for c in COMMANDS)


def command(command_id: str) -> Command:
    """The registry record ``command_id`` (``KeyError`` when unknown)."""
    for c in COMMANDS:
        if c.id == command_id:
            return c
    raise KeyError(command_id)


def shortcut_sheet(
    commands: Iterable[Command] = COMMANDS,
) -> tuple[tuple[str, ...], ...]:
    """``(menu, command, keys)`` rows for every command that has a binding."""
    return tuple(
        (c.menu, c.text, c.key_text()) for c in commands if c.keys or c.context
    )


def shortcut_sheet_html(
    commands: Iterable[Command] = COMMANDS,
    extra: Iterable[tuple[str, str, str]] = (),
) -> str:
    """The keyboard-shortcut sheet (F1), generated from the registry.

    ``extra`` adds ``(section, command, keys)`` rows the registry does not hold: the
    window's own actions under their menu, and sections of their own (the keys a
    focused plot handles) after the menus.
    """
    sections: dict[str, list[tuple[str, str]]] = {}
    for section, text, keys in (*shortcut_sheet(commands), *extra):
        sections.setdefault(section, []).append((text, keys))
    order = [m for m in MENUS if m in sections]
    order += [s for s in sections if s not in MENUS]
    rows = []
    for section in order:
        rows.append(
            f'<tr><th colspan="2" align="left">{html.escape(section)}</th></tr>'
        )
        for text, keys in sections[section]:
            rows.append(
                f"<tr><td>{html.escape(text)}</td>"
                f"<td><b>{html.escape(keys)}</b></td></tr>"
            )
    return (
        "<table cellspacing='4'><tr><th align='left'>Command</th>"
        "<th align='left'>Keys</th></tr>" + "".join(rows) + "</table>"
    )


@dataclass(frozen=True)
class Binding:
    """One live key binding (for the uniqueness invariant)."""

    context: str  # "window" or the host name
    key: str  # portable text of the QKeySequence
    command_id: str
    enabled: bool


class CommandRegistry(QObject):
    """Builds and owns the ``QAction`` of every :class:`Command`.

    Parameters
    ----------
    window : QWidget
        The top-level window: actions are added to it, so their global keys work
        without a menu.

    Signals
    -------
    triggered(str)
        A command was invoked (menu, button, global key or context letter).
    """

    triggered = Signal(str)

    def __init__(
        self, window: QtWidgets.QWidget, commands: Iterable[Command] = COMMANDS
    ) -> None:
        super().__init__(window)
        self._window = window
        self._commands: dict[str, Command] = {c.id: c for c in commands}
        self._actions: dict[str, QtGui.QAction] = {}
        self._groups: dict[str, QtGui.QActionGroup] = {}
        self._shortcuts: dict[str, list[tuple[str, QtGui.QShortcut]]] = {}
        self._hosts: dict[str, QtWidgets.QWidget] = {}
        self._session = False
        self._busy = False
        for c in self._commands.values():
            self._actions[c.id] = self._make_action(c)
        self.set_state(session=False, busy=False)

    def _make_action(self, c: Command) -> QtGui.QAction:
        action = QtGui.QAction(c.text, self._window)
        action.setObjectName(c.object_name)
        action.setToolTip(c.tooltip)
        action.setStatusTip(c.tip)
        action.setWhatsThis(c.tip)
        # QAction has no accessible name of its own; widgets built from it (buttons)
        # read this property, and the tests read it.
        action.setProperty("accessibleName", c.accessible_name)
        action.setShortcuts([QtGui.QKeySequence(k) for k in c.keys])
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        if c.checkable:
            action.setCheckable(True)
            group = self._groups.get(c.group)
            if group is None:
                group = QtGui.QActionGroup(self)
                group.setExclusive(True)
                self._groups[c.group] = group
            group.addAction(action)
            if c.id == "data.tool_pan":
                action.setChecked(True)
        action.triggered.connect(lambda _=False, cid=c.id: self.triggered.emit(cid))
        self._window.addAction(action)
        return action

    # -- lookup ----------------------------------------------------------------------
    def action(self, command_id: str) -> QtGui.QAction:
        """The ``QAction`` of ``command_id``."""
        return self._actions[command_id]

    def actions(self) -> Mapping[str, QtGui.QAction]:
        return dict(self._actions)

    def commands(self) -> tuple[Command, ...]:
        return tuple(self._commands.values())

    def shortcuts(self, host: str | None = None) -> tuple[QtGui.QShortcut, ...]:
        """The context shortcuts (of one host, or all)."""
        return tuple(
            sc
            for cid, items in self._shortcuts.items()
            for h, sc in items
            if host is None or h == host
        )

    # -- context letters -------------------------------------------------------------
    def attach_context(self, host: Host, widget: QtWidgets.QWidget) -> None:
        """Create the context shortcuts of ``host`` on ``widget`` (once per host)."""
        if host not in _HOST_NAMES:
            raise ValueError(f"unknown host {host!r}")
        if host in self._hosts:
            raise ValueError(f"host {host!r} is already attached")
        self._hosts[host] = widget
        for c in self._commands.values():
            for h, key in c.context:
                if h != host:
                    continue
                action = self._actions[c.id]
                sc = QtGui.QShortcut(QtGui.QKeySequence(key), widget)
                sc.setObjectName(f"shortcut_{host}_{c.id.replace('.', '_')}")
                sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                sc.setAutoRepeat(False)
                sc.setEnabled(action.isEnabled())
                sc.activated.connect(action.trigger)
                action.changed.connect(
                    lambda a=action, s=sc: s.setEnabled(a.isEnabled())
                )
                self._shortcuts.setdefault(c.id, []).append((host, sc))

    # -- enabled state ---------------------------------------------------------------
    def set_state(
        self, *, session: bool | None = None, busy: bool | None = None
    ) -> None:
        """Enable commands for a session being open and/or a long job running.

        The design's state table (4.9): with no session only the ``idle`` / ``always``
        commands work; while a long job runs every mutating command is disabled and
        Stop is enabled; Esc is Stop while busy and *Clear selection* otherwise.
        """
        if session is not None:
            self._session = bool(session)
        if busy is not None:
            self._busy = bool(busy)
        for c in self._commands.values():
            self._actions[c.id].setEnabled(self._available(c.availability))

    def _available(self, availability: Availability) -> bool:
        s, b = self._session, self._busy
        if availability == "always":
            return True
        if availability == "stop":
            return b
        if availability == "idle":
            return not b
        if availability == "session":
            return s
        # "mutating" and "selection": a session, nothing busy
        return s and not b

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def session_open(self) -> bool:
        return self._session

    # -- menus -----------------------------------------------------------------------
    def populate_menus(
        self, menubar: QtWidgets.QMenuBar
    ) -> dict[str, QtWidgets.QMenu]:
        """Add the File, Edit, Pre-whitening, Data, View and Help menus to ``menubar``.

        Existing menus with the same title are reused (actions are appended).
        """
        # Not QAction.menu(): PySide6 hands its result to Python ownership, and the
        # first discarded wrapper deletes the C++ menu.
        existing = {
            m.title().replace("&", ""): m
            for m in menubar.findChildren(
                QtWidgets.QMenu, options=Qt.FindChildOption.FindDirectChildrenOnly
            )
        }
        menus: dict[str, QtWidgets.QMenu] = {}
        for title in MENUS:
            menu = existing.get(title)
            if menu is None:
                menu = menubar.addMenu(title)
                menu.setObjectName("menu_" + title.lower().replace("-", "_"))
            menus[title] = menu
        last: dict[str, str] = {}
        for c in self._commands.values():
            menu = menus[c.menu]
            stem = c.id.split(".", 1)[0]
            if last.get(c.menu) not in (None, stem):
                menu.addSeparator()
            last[c.menu] = stem
            menu.addAction(self._actions[c.id])
        return menus

    def tool_button(
        self, command_id: str, parent: QtWidgets.QWidget | None = None
    ) -> QtWidgets.QToolButton:
        """A button that shows and triggers ``command_id`` (accessible name set)."""
        button = QtWidgets.QToolButton(parent)
        button.setDefaultAction(self._actions[command_id])
        button.setAccessibleName(self._commands[command_id].accessible_name)
        button.setObjectName("button_" + command_id.replace(".", "_"))
        return button

    # -- the invariant ---------------------------------------------------------------
    def bindings(self) -> tuple[Binding, ...]:
        """Every key binding: global keys (context ``"window"``) and context letters."""
        out: list[Binding] = []
        for cid, action in self._actions.items():
            for seq in action.shortcuts():
                out.append(Binding("window", seq.toString(), cid, action.isEnabled()))
        for cid, items in self._shortcuts.items():
            for host, sc in items:
                out.append(
                    Binding(host, sc.key().toString(), cid, sc.isEnabled())
                )
        return tuple(out)

    def conflicts(self) -> tuple[str, ...]:
        """Descriptions of enabled bindings that share a key in one context.

        A context letter also clashes with an *enabled* window key on the same
        sequence: while its host has focus Qt would see two candidates and fire
        neither.
        """
        seen: dict[tuple[str, str], str] = {}
        window: dict[str, str] = {}
        problems: list[str] = []
        for b in self.bindings():
            if not b.enabled:
                continue
            slot = (b.context, b.key)
            if slot in seen and seen[slot] != b.command_id:
                problems.append(
                    f"{b.key} in {b.context}: {seen[slot]} and {b.command_id}"
                )
            seen[slot] = b.command_id
            if b.context == "window":
                window[b.key] = b.command_id
        for b in self.bindings():
            if (
                b.enabled
                and b.context != "window"
                and window.get(b.key, b.command_id) != b.command_id
            ):
                problems.append(
                    f"{b.key} in {b.context} shadows window key of {window[b.key]}"
                )
        return tuple(problems)

    def sheet_html(self, extra: Iterable[tuple[str, str, str]] = ()) -> str:
        """The F1 sheet for this registry's commands (plus ``extra`` rows, see
        :func:`shortcut_sheet_html`)."""
        return shortcut_sheet_html(self._commands.values(), extra)


__all__ = [
    "COMMANDS",
    "COMMAND_IDS",
    "MENUS",
    "Availability",
    "Binding",
    "Command",
    "CommandRegistry",
    "Host",
    "command",
    "shortcut_sheet",
    "shortcut_sheet_html",
]
