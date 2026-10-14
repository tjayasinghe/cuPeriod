"""The History dock: the session's action log, and a way back to any state in it.

One line per logged action, as :func:`cuperiod.prewhiten.session.actions.describe`
words it ("Step: accepted F4 = 17.80995 c/d, S/N 143.8", "Tied F7 = F1 + F2"); the
actions of one group — an Auto run, a macro such as *Polish and prune*, an edit that is
several operations — sit under one collapsible node, because Undo moves over them as a
whole. The first line is the session as it was opened.

The head (the state the session is in) is marked ▶ and bold. Undone actions — the redo
tail — are italic and say so. An entry whose state is no longer in memory (after a
project was loaded from a file, or beyond the undo depth) is greyed and says "not in
memory", and its tooltip says how to get it back. Double-click or Enter goes to an
entry's state (:attr:`HistoryPanel.goto_requested`); the right-click menu also offers
*Copy as Python* (:attr:`HistoryPanel.copy_script_requested`, rendered on the worker by
the controller) and *Export script…*.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from cuperiod.gui.commands import command
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, Signal

if TYPE_CHECKING:
    from cuperiod.gui.commands import CommandRegistry
    from cuperiod.gui.viewmodels import HistoryItem, SessionView

#: Tooltip of an entry whose state is not retained (design 2.8).
UNREACHABLE_TIP = (
    "Not in memory: this project was opened from a file (or the action is older than "
    "the undo depth); use File ▸ Verify project to replay its log"
)

_UNDONE_TIP = "Undone: Redo (Ctrl+Shift+Z) or double-click to return to it"
_HEAD_MARK = "▶ "
_ROOT_TEXT = "Session opened"

#: Item role holding the sequence number an item goes to (0 = the opened session).
SEQ_ROLE = Qt.ItemDataRole.UserRole


#: Words for a group's node where the op name is not one: "use_variation,
#: set_variation" read like code.
_OP_WORDS = {
    "add_peak": "add peak",
    "set_active": "use / don't use",
    "set_values": "set values",
    "set_variation": "vary per subset",
    "set_note": "note",
    "delete_points": "delete points",
    "restore_points": "restore points",
    "exclude_subsets": "leave out subsets",
    "define_partition": "new partition",
    "drop_partition": "drop partition",
    "set_subset_weights": "subset weights",
    "use_offsets": "zero points",
    "use_variation": "amplitude/phase partition",
    "use_weights": "weights",
    "configure": "settings",
}


def compact_entry_text(text: str) -> str:
    """One History line without the echo of a log line that says it twice.

    ``"Restore 22 point(s): restored 22 points"`` reads as ``"Restore 22 point(s)"``
    and ``"Improve all: improved all 4 frequencies"`` as ``"Improved all 4
    frequencies"``: when the outcome restates the action (same first letters), the
    longer of the two stays. The full line is the row's tooltip.
    """
    head, sep, outcome = text.partition(": ")
    if not sep or not outcome or outcome[:3].lower() != head[:3].lower():
        return text
    kept = outcome if len(outcome) > len(head) else head
    return kept[:1].upper() + kept[1:]


def _group_title(items: Sequence[HistoryItem]) -> str:
    """The node of a multi-action group: what it was, in a few words."""
    ops = [i.op for i in items]
    steps = sum(op in ("step", "add_peak") for op in ops)
    accepted = sum(i.kind == "accepted" for i in items if i.op in ("step", "add_peak"))
    if set(ops) <= {"step", "polish", "prune", "renumber"} and steps:
        tail = [op for op in ("polish", "prune") if op in ops]
        extra = f", then {' and '.join(tail)}" if tail else ""
        plural = "s" if steps != 1 else ""
        return f"Auto: {steps} step{plural}, {accepted} accepted{extra}"
    if set(ops) <= {"polish", "prune"}:
        return "Polish and prune"
    names: list[str] = []
    for op in ops:
        word = _OP_WORDS.get(op, op.replace("_", " "))
        if word not in names:
            names.append(word)
    return f"{len(items)} actions: {', '.join(names)}"


class HistoryPanel(QtWidgets.QWidget):
    """The action log as a tree, with the head highlighted.

    Signals
    -------
    goto_requested(int)
        Go to the state after entry ``seq`` (0 = as opened); the controller's
        ``session_goto``. Unreachable entries are requested too: the session refuses
        them with its own message.
    copy_script_requested(int)
        *Copy as Python* for entry ``seq``; the controller's ``copy_entry_script``.
    export_script_requested()
        *Export script…* (an unbound panel; a bound one triggers the registry's
        ``file.export_script``).
    """

    goto_requested = Signal(int)
    copy_script_requested = Signal(int)
    export_script_requested = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("history_panel")
        self._shown: tuple[tuple[HistoryItem, ...], int] | None = None
        self._expanded: set[int] = set()
        self._registry: CommandRegistry | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setObjectName("history_tree")
        self._tree.setAccessibleName("Session history")
        self._tree.setAccessibleDescription(
            "Every action of the session. Enter or double-click returns to that "
            "state; Ctrl+C copies the action as Python."
        )
        self._tree.setToolTip(
            "Double-click (or Enter) an action to return to the state after it; "
            "right-click for Copy as Python"
        )
        self._tree.setStatusTip("The session's actions; double-click one to go back")
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["#", "Action"])
        self._tree.setUniformRowHeights(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setAllColumnsShowFocus(True)
        header = self._tree.header()
        if header is not None:
            header.setStretchLastSection(True)
            header.setSectionResizeMode(
                0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        self._tree.itemActivated.connect(self._on_activated)
        self._tree.itemExpanded.connect(self._on_expanded)
        self._tree.itemCollapsed.connect(self._on_collapsed)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        copy = QtGui.QShortcut(
            QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy), self._tree
        )
        copy.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy.activated.connect(self.copy_current)
        layout.addWidget(self._tree, 1)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        spec = command("file.export_script")
        self._export_action = QtGui.QAction("Export script…", self)
        self._export_action.setToolTip(spec.tooltip)
        self._export_action.setStatusTip(spec.tip)
        self._export_action.triggered.connect(
            lambda _=False: self.export_script_requested.emit()
        )
        self._export = QtWidgets.QToolButton()
        self._export.setObjectName("history_export_script")
        self._export.setDefaultAction(self._export_action)
        self._export.setAccessibleName(spec.accessible_name)
        self._export.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        row.addWidget(self._export)
        layout.addLayout(row)
        self.clear()

    # -- registry --------------------------------------------------------------------
    def bind_commands(self, registry: CommandRegistry) -> None:
        """*Export script…* becomes the registry's ``file.export_script``."""
        self._registry = registry
        action = registry.action("file.export_script")
        action.setIconText("Export script…")
        self._export.setDefaultAction(action)

    @property
    def tree(self) -> QtWidgets.QTreeWidget:
        return self._tree

    @property
    def export_button(self) -> QtWidgets.QToolButton:
        return self._export

    # -- rendering -------------------------------------------------------------------
    def clear(self) -> None:
        """No session: an empty log and a disabled export."""
        self._shown = None
        self._expanded.clear()
        self._tree.clear()
        placeholder = QtWidgets.QTreeWidgetItem(["", "No session yet"])
        placeholder.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self._tree.addTopLevelItem(placeholder)
        self._export_action.setEnabled(False)

    def set_view(self, view: SessionView) -> None:
        """Show ``view``'s log; rebuilt only when the log or the head moved."""
        self._export_action.setEnabled(True)
        state = (tuple(view.history), int(view.cursor))
        if state == self._shown:
            return
        self._shown = state
        self._rebuild(*state)

    def _rebuild(self, items: tuple[HistoryItem, ...], cursor: int) -> None:
        bar = self._tree.verticalScrollBar()
        scroll = bar.value() if bar is not None else 0
        head_seq = items[cursor - 1].seq if 0 < cursor <= len(items) else 0
        undone = {i.seq for i in items[cursor:]}
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            root = self._entry_item(
                seq=0, text=_ROOT_TEXT, tip="The session as it was opened",
                head=head_seq == 0, undone=False, reachable=True,
            )
            self._tree.addTopLevelItem(root)
            head_item = root if head_seq == 0 else None
            for group, members in self._groups(items):
                children = [
                    self._history_item(i, head_seq, undone) for i in members
                ]
                if len(members) == 1:
                    self._tree.addTopLevelItem(children[0])
                else:
                    node = self._group_item(group, members, head_seq, undone)
                    node.addChildren(children)
                    self._tree.addTopLevelItem(node)
                    holds_head = any(i.seq == head_seq for i in members)
                    node.setExpanded(group in self._expanded or holds_head)
                for child, item in zip(children, members, strict=True):
                    if item.seq == head_seq:
                        head_item = child
            if head_item is not None:
                self._tree.setCurrentItem(head_item)
        finally:
            self._tree.blockSignals(False)
        if bar is not None:
            bar.setValue(scroll)
        if head_item is not None:
            self._tree.scrollToItem(head_item)

    @staticmethod
    def _groups(
        items: Sequence[HistoryItem],
    ) -> list[tuple[int, list[HistoryItem]]]:
        out: list[tuple[int, list[HistoryItem]]] = []
        for item in items:
            if out and out[-1][0] == item.group:
                out[-1][1].append(item)
            else:
                out.append((item.group, [item]))
        return out

    def _entry_item(
        self, *, seq: int, text: str, tip: str, head: bool, undone: bool,
        reachable: bool,
    ) -> QtWidgets.QTreeWidgetItem:
        label = text
        tips = [tip] if tip else []
        if not reachable:
            label += "  (not in memory)"
            tips.append(UNREACHABLE_TIP)
        elif undone:
            label += "  (undone)"
            tips.append(_UNDONE_TIP)
        if head:
            label = _HEAD_MARK + label
            tips.insert(0, "The current state")
        item = QtWidgets.QTreeWidgetItem(["" if seq == 0 else str(seq), label])
        item.setData(0, SEQ_ROLE, seq)
        tooltip = "\n".join(tips)
        item.setToolTip(0, tooltip)
        item.setToolTip(1, tooltip)
        font = item.font(1)
        font.setBold(head)
        font.setItalic(undone)
        item.setFont(1, font)
        item.setFont(0, font)
        if not reachable:
            grey = QtGui.QBrush(
                self.palette().color(
                    QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text
                )
            )
            item.setForeground(0, grey)
            item.setForeground(1, grey)
        state = "current state" if head else ("undone" if undone else "")
        if not reachable:
            state = "not in memory"
        accessible = f"{text}, {state}" if state else text
        item.setData(1, Qt.ItemDataRole.AccessibleTextRole, accessible)
        return item

    def _history_item(
        self, item: HistoryItem, head_seq: int, undone: set[int]
    ) -> QtWidgets.QTreeWidgetItem:
        text = compact_entry_text(item.text)
        detail = f"{item.op} ({item.kind})" if item.kind else item.op
        return self._entry_item(
            seq=item.seq,
            text=text,
            tip=detail if text == item.text else "\n".join((item.text, detail)),
            head=item.seq == head_seq,
            undone=item.seq in undone,
            reachable=item.reachable,
        )

    def _group_item(
        self,
        group: int,
        members: Sequence[HistoryItem],
        head_seq: int,
        undone: set[int],
    ) -> QtWidgets.QTreeWidgetItem:
        last = members[-1]
        node = self._entry_item(
            seq=last.seq,
            text=_group_title(members),
            tip=f"{len(members)} actions that undo together; double-click goes to "
            "the state after the last",
            head=last.seq == head_seq,
            undone=last.seq in undone,
            reachable=last.reachable,
        )
        node.setText(0, f"{members[0].seq}–{last.seq}")
        node.setData(0, Qt.ItemDataRole.UserRole + 1, group)
        if any(i.seq == head_seq for i in members[:-1]):
            # The head is inside the group (a goto into it): mark the node too.
            font = node.font(1)
            font.setBold(True)
            node.setFont(1, font)
        return node

    # -- interaction -----------------------------------------------------------------
    def _seq_of(self, item: QtWidgets.QTreeWidgetItem | None) -> int | None:
        if item is None:
            return None
        seq = item.data(0, SEQ_ROLE)
        return seq if isinstance(seq, int) else None

    def _on_activated(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        seq = self._seq_of(item)
        if seq is not None:
            self.goto_requested.emit(seq)

    def _on_expanded(self, item: QtWidgets.QTreeWidgetItem) -> None:
        group = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if isinstance(group, int):
            self._expanded.add(group)

    def _on_collapsed(self, item: QtWidgets.QTreeWidgetItem) -> None:
        group = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if isinstance(group, int):
            self._expanded.discard(group)

    def copy_current(self) -> None:
        """*Copy as Python* for the current entry (Ctrl+C in the tree)."""
        seq = self._seq_of(self._tree.currentItem())
        if seq is not None and seq > 0:
            self.copy_script_requested.emit(seq)

    def context_menu(
        self, item: QtWidgets.QTreeWidgetItem | None
    ) -> QtWidgets.QMenu:
        """The right-click menu of ``item`` (shown with ``popup()``)."""
        menu = QtWidgets.QMenu(self._tree)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        menu.setAccessibleName("History actions")
        seq = self._seq_of(item)
        go = menu.addAction("Go to this state\tEnter")
        go.setEnabled(seq is not None)
        go.triggered.connect(
            lambda _=False: self.goto_requested.emit(seq) if seq is not None else None
        )
        copy = menu.addAction("Copy as Python\tCtrl+C")
        copy.setStatusTip("Copy this action as the Python call that repeats it")
        copy.setEnabled(seq is not None and seq > 0)
        copy.triggered.connect(
            lambda _=False: self.copy_script_requested.emit(seq)
            if seq is not None
            else None
        )
        menu.addSeparator()
        export = menu.addAction("Export script…")
        export.setEnabled(self._export.isEnabled())
        export.triggered.connect(lambda _=False: self._export.click())
        return menu

    def _on_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self._tree.itemAt(pos)
        viewport = self._tree.viewport()
        anchor = viewport if viewport is not None else self._tree
        self.context_menu(item).popup(anchor.mapToGlobal(pos))


__all__ = ["SEQ_ROLE", "UNREACHABLE_TIP", "HistoryPanel"]
