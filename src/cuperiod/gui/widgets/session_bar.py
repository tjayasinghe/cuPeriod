"""The session bar above the Frequencies table: the pre-whitening loop at hand.

Buttons for Auto, Step, Add peak, Calculate, Improve all, Improve selected, Delete,
Undo, Redo and Stop, a chip naming the peak Step would try next, and an *Add frequency*
field that takes ``12.34`` (c/d), ``0.081 d`` (a period), ``=F1+F2`` or ``2f1`` (a
combination, added and tied in one undoable step).

Every button is one command of the registry (:mod:`cuperiod.gui.commands`). On its own
the bar emits :attr:`SessionBar.command_requested` with the command id; once
:meth:`SessionBar.bind_commands` hands it the window's registry, each button shows and
triggers the registry's own ``QAction`` — one enabled state, tooltip and shortcut for
the menu, the bar and the keyboard. The field's edits leave as ``edit_requested`` /
``group_requested`` with ``expect_id_epoch`` in every args mapping, like the table's.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from cuperiod.gui.commands import command
from cuperiod.gui.qt import Qt, QtGui, QtWidgets, Signal
from cuperiod.gui.widgets.frequency_table import parse_entry, tie_frequency
from cuperiod.gui.widgets.info_bar import FlowLayout

if TYPE_CHECKING:
    from cuperiod.gui.commands import CommandRegistry
    from cuperiod.gui.viewmodels import CandidateView, ComponentView, SessionView

#: (command id, button text), in bar order.
BUTTONS: tuple[tuple[str, str], ...] = (
    ("session.auto", "Auto"),
    ("session.step", "Step"),
    ("session.add_peak", "Add peak"),
    ("session.calculate", "Calculate"),
    ("session.improve_all", "Improve all"),
    ("session.improve_selected", "Improve selected"),
    ("edit.delete", "Delete"),
    ("edit.undo", "Undo"),
    ("edit.redo", "Redo"),
    ("session.stop", "Stop"),
)

#: Buttons that stay enabled while a long job runs (unbound bar; the registry has its
#: own table for a bound one).
_WHILE_BUSY = frozenset({"session.stop"})

_ADD_TIP = (
    "Add a component: a frequency (12.34 or 12.34 c/d, also µHz), a period "
    "(0.081 d, 2 h), or a combination that is added and tied at once (=F1+F2, 2f1). "
    "Enter adds."
)

_NO_CANDIDATE = "Next peak: –"


def near_combination(
    frequency: float,
    rows: Sequence[ComponentView],
    tolerance: float,
    *,
    max_harmonic: int = 4,
) -> str | None:
    """The simplest low-order combination of active ``rows`` within ``tolerance``.

    Harmonics ``nF1`` (n <= ``max_harmonic``), then ``Fi ± Fj`` and ``2Fi ± Fj``; the
    closest match of the lowest order wins. Display only (the chip's "near F1 + F2"):
    the session's own identification runs on the worker.
    """
    if not (math.isfinite(frequency) and tolerance > 0.0):
        return None
    active = [r for r in rows if r.active and math.isfinite(r.frequency)]
    best: tuple[int, float, str] | None = None

    def consider(order: int, value: float, text: str) -> None:
        nonlocal best
        miss = abs(value - frequency)
        if miss <= tolerance and (best is None or (order, miss) < best[:2]):
            best = (order, miss, text)

    for r in active:
        for n in range(2, max_harmonic + 1):
            consider(n, n * r.frequency, f"{n}{r.label}")
    for a, b in itertools.permutations(active, 2):
        consider(2, a.frequency + b.frequency, f"{a.label} + {b.label}")
        consider(2, a.frequency - b.frequency, f"{a.label} - {b.label}")
        consider(3, 2 * a.frequency + b.frequency, f"2{a.label} + {b.label}")
        consider(3, 2 * a.frequency - b.frequency, f"2{a.label} - {b.label}")
    return None if best is None else best[2]


def candidate_text(
    candidate: CandidateView | None, rows: Sequence[ComponentView], tolerance: float
) -> str:
    """The chip text, e.g. ``Next: 12.34012 c/d · P 0.081037 d · A 0.0039 · S/N ≈
    8.2 · near F1 + F2``."""
    if candidate is None:
        return _NO_CANDIDATE
    snr = candidate.snr_estimate
    parts = [
        f"{candidate.frequency:.7g} c/d",
        f"P {candidate.period:.7g} d",
        f"A {candidate.amplitude:.3g}",
        f"S/N ≈ {snr:.1f}" if math.isfinite(snr) else "S/N ≈ –",
    ]
    near = near_combination(candidate.frequency, rows, tolerance)
    if near is not None:
        parts.append(f"near {near}")
    return "Next: " + " · ".join(parts)


def _rayleigh(view: SessionView) -> float:
    time = view.time
    if time.size < 2:
        return math.nan
    span = float(time.max() - time.min())
    return 1.0 / span if span > 0.0 else math.nan


def next_id(view: SessionView) -> int:
    """The id the session gives the next component it adds.

    Read from the view when it carries ``next_id``; otherwise one past the largest id
    shown, which is right unless the highest ids were removed without a renumbering
    (the tie then names no component and the session refuses it, with a message).
    """
    stated = getattr(view, "next_id", None)
    if isinstance(stated, int) and stated > 0:
        return stated
    return max((r.id for r in view.rows), default=0) + 1


class SessionBar(QtWidgets.QWidget):
    """Buttons, the candidate chip and the Add-frequency field.

    Signals
    -------
    command_requested(str)
        A button was pressed (unbound bar): the registry command id.
    edit_requested(str, object)
        ``("add", {"frequency": f, "expect_id_epoch": None})`` from the field.
    group_requested(object)
        ``(("add", {...}), ("tie", {"id": new, "expression": ...}))`` for a
        combination typed in the field; both args carry ``expect_id_epoch``.
    status_message(str)
        Why an entry was not accepted.
    """

    command_requested = Signal(str)
    edit_requested = Signal(str, object)
    group_requested = Signal(object)
    status_message = Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("session_bar")
        self.setAccessibleName("Pre-whitening session")
        self._view: SessionView | None = None
        self._registry: CommandRegistry | None = None
        self._busy = False
        self._local: dict[str, QtGui.QAction] = {}
        self._buttons: dict[str, QtWidgets.QToolButton] = {}

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # The buttons wrap in a narrow dock (a tool bar would hide the overflow behind
        # a menu), and the layout's minimum height follows the wrap, so the rows
        # never overlap the field below in a short dock.
        buttons = QtWidgets.QWidget()
        flow = FlowLayout(hspacing=4, vspacing=4)
        flow.setContentsMargins(0, 0, 0, 0)
        buttons.setLayout(flow)
        for command_id, text in BUTTONS:
            button = self._make_button(command_id, text)
            flow.addWidget(button)
        outer.addWidget(buttons)

        entry = QtWidgets.QHBoxLayout()
        entry.setSpacing(4)
        label = QtWidgets.QLabel("Add fre&quency:")
        self._field = QtWidgets.QLineEdit()
        self._field.setObjectName("add_frequency_field")
        self._field.setPlaceholderText("12.34 · 0.081 d · =F1+F2 · 2f1")
        self._field.setAccessibleName("Add frequency")
        self._field.setAccessibleDescription(_ADD_TIP)
        self._field.setToolTip(_ADD_TIP + "  (Ctrl+F6)")
        self._field.setStatusTip(_ADD_TIP)
        self._field.setClearButtonEnabled(True)
        self._field.returnPressed.connect(self.submit_entry)
        self._field.textEdited.connect(lambda _text: self._field.setToolTip(_ADD_TIP))
        label.setBuddy(self._field)
        self._add_button = QtWidgets.QPushButton("Add")
        self._add_button.setObjectName("add_frequency_button")
        self._add_button.setAccessibleName("Add the typed frequency")
        self._add_button.setToolTip("Add the frequency, period or combination typed")
        self._add_button.setStatusTip("Add the frequency, period or combination typed")
        self._add_button.clicked.connect(self.submit_entry)
        entry.addWidget(label)
        entry.addWidget(self._field, 1)
        entry.addWidget(self._add_button)
        outer.addLayout(entry)

        self._chip = QtWidgets.QLabel(_NO_CANDIDATE)
        self._chip.setObjectName("candidate_chip")
        self._chip.setAccessibleName("Next candidate")
        self._chip.setToolTip("The peak Step (F7) would try next")
        self._chip.setWordWrap(True)
        self._chip.setTextFormat(Qt.TextFormat.PlainText)
        self._chip.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._chip.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        outer.addWidget(self._chip)
        self._update_enabled()

    def _make_button(self, command_id: str, text: str) -> QtWidgets.QToolButton:
        spec = command(command_id)
        action = QtGui.QAction(text, self)
        action.setObjectName(f"bar_{spec.object_name}")
        action.setToolTip(spec.tooltip)
        action.setStatusTip(spec.tip)
        action.setWhatsThis(spec.tip)
        action.triggered.connect(
            lambda _=False, cid=command_id: self.command_requested.emit(cid)
        )
        self._local[command_id] = action
        button = QtWidgets.QToolButton()
        button.setObjectName("button_" + command_id.replace(".", "_"))
        button.setDefaultAction(action)
        button.setAccessibleName(spec.accessible_name)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._buttons[command_id] = button
        return button

    # -- registry --------------------------------------------------------------------
    def bind_commands(self, registry: CommandRegistry) -> None:
        """Let each button show and trigger the registry's ``QAction``.

        The registry then owns enabled states, tooltips and shortcuts;
        :attr:`command_requested` is no longer emitted. *Add at frequency…*
        (Ctrl+F6) focuses the field.
        """
        self._registry = registry
        for command_id, text in BUTTONS:
            action = registry.action(command_id)
            # A tool button shows the icon text: the short label here, while menus
            # keep the long one.
            action.setIconText(text)
            self._buttons[command_id].setDefaultAction(action)
        registry.action("session.add_frequency").triggered.connect(
            lambda _=False: self.focus_add_field()
        )

    def button(self, command_id: str) -> QtWidgets.QToolButton:
        """The button of ``command_id`` (tests, tab order)."""
        return self._buttons[command_id]

    @property
    def field(self) -> QtWidgets.QLineEdit:
        return self._field

    @property
    def chip(self) -> QtWidgets.QLabel:
        return self._chip

    def focus_add_field(self) -> None:
        """Put the keyboard focus in the Add-frequency field, its text selected."""
        self._field.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._field.selectAll()

    # -- state -----------------------------------------------------------------------
    def set_view(self, view: SessionView) -> None:
        """Show ``view``'s candidate and undo state."""
        self._view = view
        tolerance = _rayleigh(view)
        self._chip.setText(candidate_text(view.candidate, view.rows, tolerance))
        tip = "The peak Step (F7) would try next"
        if view.candidate is not None and view.candidate.hint:
            tip = view.candidate.hint
        self._chip.setToolTip(tip)
        self._update_enabled()

    def clear(self) -> None:
        """No session shown."""
        self._view = None
        self._chip.setText(_NO_CANDIDATE)
        self._chip.setToolTip("The peak Step (F7) would try next")
        self._update_enabled()

    def set_busy(self, busy: bool) -> None:
        """A long job runs: only Stop stays enabled (unbound bar)."""
        self._busy = bool(busy)
        self._update_enabled()

    def _update_enabled(self) -> None:
        view = self._view
        for command_id, action in self._local.items():
            if self._busy:
                enabled = command_id in _WHILE_BUSY
            elif command_id == "session.stop":
                enabled = False
            elif command_id == "edit.undo":
                enabled = view is not None and view.can_undo
            elif command_id == "edit.redo":
                enabled = view is not None and view.can_redo
            else:
                enabled = True
            action.setEnabled(enabled)
        self._field.setEnabled(not self._busy)
        self._add_button.setEnabled(not self._busy)

    # -- the Add-frequency field -----------------------------------------------------
    def submit_entry(self) -> bool:
        """Add what the field holds; on a bad entry say why and keep the text."""
        text = self._field.text()
        try:
            self._submit(text)
        except ValueError as exc:
            message = f"Add frequency: {exc}"
            self._field.setToolTip(message)
            self._field.selectAll()
            self.status_message.emit(message)
            return False
        self._field.clear()
        self._field.setToolTip(_ADD_TIP + "  (Ctrl+F6)")
        return True

    def _submit(self, text: str) -> None:
        if self._busy:
            raise ValueError("busy: wait for the running job, or press Stop (Esc)")
        entry = parse_entry(text)
        view = self._view
        if entry.kind == "frequency":
            self.edit_requested.emit(
                "add", {"frequency": entry.frequency, "expect_id_epoch": None}
            )
            return
        if view is None:
            raise ValueError("a combination needs a session with components")
        frequency = tie_frequency(entry, view.rows)
        new = next_id(view)
        epoch = view.id_epoch
        actions: tuple[tuple[str, dict[str, Any]], ...] = (
            # Added exactly there (no trial fit): the tie decides the frequency.
            ("add", {"frequency": frequency, "refine": "none",
                     "expect_id_epoch": epoch}),
            ("tie", {"id": new, "expression": entry.expression,
                     "expect_id_epoch": epoch}),
        )
        self.group_requested.emit(actions)


__all__ = [
    "BUTTONS",
    "SessionBar",
    "candidate_text",
    "near_combination",
    "next_id",
]
