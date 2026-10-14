"""The Subsets dock: partitions of the points and the roles they play in the fit.

A *partition* splits the light curve into subsets -- nights, sectors, seasons,
fixed-length chunks, bands, the values of a label column, or hand-drawn time ranges
(Period04's "attributes"). One partition may carry the fitted zero points, one the
relative weights and one the per-subset amplitudes and phases; the three roles are
independent (a zero point per night with amplitudes per season is the normal case).

The dock shows one partition at a time as a table -- whether each subset is used, its
name, size, time span, fitted zero point, relative weight and residual scatter -- and
turns every gesture into a request for the controller:

* deriving a partition (*Split by*, *Cross with…*, the hint's button) or editing one
  (*Merge*, *Split at*, renaming a subset) emits
  :attr:`SubsetsPanel.partition_requested` with a :class:`PartitionRequest`, which the
  controller resolves to a
  :class:`~cuperiod.prewhiten.subsets.Partition` on the compute worker
  (:meth:`PartitionRequest.resolve`) and defines in the session;
* everything else is one session op, :attr:`SubsetsPanel.edit_requested`
  ``(op, args)`` -- ``exclude_subsets``, ``use_offsets`` / ``use_weights`` /
  ``use_variation``, ``set_subset_weights``, ``drop_partition`` -- or a group of ops
  that must undo as one (:attr:`SubsetsPanel.group_requested`).

The *use* box is tri-state and only ever toggles ``exclude_subsets``: checked is a
subset in use with all its points, partially checked one in use with some points
deleted (its tooltip says how many), unchecked one left out. It never deletes or
restores single points, so re-including a night does not resurrect the outliers
lassoed inside it. Selecting rows highlights their points in the light curve
(:attr:`SubsetsPanel.points_highlighted`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, Signal
from cuperiod.gui.widgets.info_bar import FlowLayout
from cuperiod.prewhiten.subsets import (
    ChunksOrigin,
    CrossedOrigin,
    GapsOrigin,
    LabelsOrigin,
    Partition,
    PartitionOrigin,
    RangesOrigin,
    suggest_partition,
)

if TYPE_CHECKING:
    from cuperiod.gui.viewmodels import SessionView, SubsetView

BoolArray = npt.NDArray[np.bool_]

PartitionEdit = Literal["merge", "split_at", "rename"]

#: Longest partition name the session accepts.
_MAX_NAME = 64

_NO_SESSION_HINT = "Open a pre-whitening session to split the data into subsets."
_NO_PARTITION_HINT = (
    "No subsets yet: choose how to split the points (Split by) and press Apply. "
    "Each subset can then have its own zero point, weight or amplitude and phase."
)

#: Table columns, in order.
COLUMNS: tuple[str, ...] = (
    "use",
    "name",
    "N",
    "start",
    "span",
    "zero point ±",
    "weight",
    "resid. rms",
)
_USE, _NAME, _N, _START, _SPAN, _ZERO, _WEIGHT, _RMS = range(len(COLUMNS))

_DASH = "–"

_OFFSETS_TIP = (
    "Fit one zero point per subset of this partition, re-solved with the sinusoids "
    "on every fit"
)


@dataclass(frozen=True)
class PartitionRequest:
    """A partition the Subsets dock asks for, resolved on the compute worker.

    Either a *recipe* (``origin``: gaps, chunks, labels, ranges, crossed) run on the
    session's points under the name ``name``, or an *edit* of the existing partition
    ``name``: ``"merge"`` its ``subsets`` into one, ``"split_at"`` ``time``, or
    ``"rename"`` subset ``subsets[0]`` to ``label``.
    """

    name: str
    origin: PartitionOrigin | None = None
    edit: PartitionEdit | None = None
    subsets: tuple[int, ...] = ()
    time: float | None = None
    label: str = ""

    def resolve(
        self,
        time: Any,
        *,
        partitions: Mapping[str, Partition],
        labels: Mapping[str, Any] | None = None,
    ) -> Partition:
        """The :class:`~cuperiod.prewhiten.subsets.Partition` this request names.

        Parameters
        ----------
        time : array
            The session's point times (``session.data.time``).
        partitions : Mapping[str, Partition]
            The head state's partitions by name.
        labels : Mapping[str, array], optional
            The session's per-point label columns (``session.data.labels``).

        Raises
        ------
        ValueError
            For an edit of a partition that does not exist, a subset index out of
            range, a cut no subset straddles, an invalid label, or a recipe that
            cannot run (a missing column or partition).
        """
        if self.edit is None:
            if self.origin is None:
                raise ValueError("a partition request needs a recipe or an edit")
            return Partition.from_origin(
                self.origin,
                time,
                name=self.name,
                labels=labels,
                partitions=partitions,
            )
        if self.name not in partitions:
            raise ValueError(f"no partition {self.name!r}")
        base = partitions[self.name]
        if self.edit == "merge":
            chosen = sorted(set(self.subsets))
            if len(chosen) < 2:
                raise ValueError("merging needs at least two subsets")
            merged = base
            # Each merge folds the higher index into the first one and shifts every
            # later subset down by one; walking upwards keeps the labels in order.
            for step, index in enumerate(chosen[1:]):
                merged = merged.merged(chosen[0], index - step)
            return merged
        if self.edit == "split_at":
            if self.time is None or not math.isfinite(self.time):
                raise ValueError("splitting needs a finite time")
            return base.split_at(self.time, time)
        if len(self.subsets) != 1:
            raise ValueError("renaming needs exactly one subset")
        return base.renamed(self.subsets[0], self.label)

    def describe(self) -> str:
        """A short phrase for status lines ("split 'night' at 8015.30")."""
        if self.edit == "merge":
            return f"merge {len(self.subsets)} subsets of {self.name!r}"
        if self.edit == "split_at":
            return f"split {self.name!r} at {self.time:.5f}"
        if self.edit == "rename":
            return f"rename a subset of {self.name!r} to {self.label!r}"
        return f"define partition {self.name!r}"


# -- split presets -----------------------------------------------------------------

ParamKind = Literal["gap", "length", "none", "column", "edges"]


@dataclass(frozen=True)
class _Preset:
    key: str
    text: str
    name: str
    param: ParamKind
    default: float
    tip: str


_PRESETS: tuple[_Preset, ...] = (
    _Preset(
        "nights",
        "Nights (time gaps)",
        "night",
        "gap",
        0.3,
        "A new subset at every gap in time longer than the value given (0.3 d "
        "separates the nights of ground-based data)",
    ),
    _Preset(
        "sectors",
        "TESS sectors",
        "sector",
        "none",
        0.0,
        "Runs separated by gaps over 0.5 d, merged up to 28.5 d: one subset per "
        "TESS sector, without a mission calendar",
    ),
    _Preset(
        "quarters",
        "Kepler quarters",
        "quarter",
        "none",
        0.0,
        "Runs separated by gaps over 0.5 d, merged up to 100 d: one subset per "
        "Kepler quarter",
    ),
    _Preset(
        "seasons",
        "Seasons",
        "season",
        "gap",
        60.0,
        "A new subset at every gap longer than the value given (60 d separates "
        "observing seasons)",
    ),
    _Preset(
        "chunks",
        "Every N days",
        "chunk",
        "length",
        10.0,
        "Consecutive chunks of the length given (days); chunks with fewer than 10 "
        "points join a neighbour",
    ),
    _Preset(
        "bands",
        "Bands",
        "band",
        "none",
        0.0,
        "One subset per photometric band (multiband curves)",
    ),
    _Preset(
        "column",
        "Column…",
        "",
        "column",
        0.0,
        "One subset per distinct value of a label column of the data (a site, a "
        "camera, an instrument)",
    ),
    _Preset(
        "ranges",
        "Manual time ranges",
        "range",
        "edges",
        0.0,
        "Time ranges split at the times listed (comma- or space-separated, days)",
    ),
)


def _origin_text(origin: PartitionOrigin, count: int) -> tuple[str, str, str]:
    """``(what it looks like, button text, partition name)`` for a suggestion."""
    if isinstance(origin, GapsOrigin) and origin.max_span is not None:
        return f"TESS-like sectors ({count} sectors)", "Split into sectors", "sector"
    if isinstance(origin, GapsOrigin) and origin.min_gap >= 30.0:
        return f"seasons ({count} seasons)", "Split into seasons", "season"
    return f"nightly data ({count} nights)", "Split into nights", "night"


def _parse_edges(text: str) -> list[float]:
    parts = [p for p in text.replace(";", " ").replace(",", " ").split() if p]
    if not parts:
        raise ValueError("list at least one time to split at")
    try:
        edges = sorted(float(p) for p in parts)
    except ValueError:
        raise ValueError(f"not a list of times: {text!r}") from None
    if not all(math.isfinite(e) for e in edges):
        raise ValueError("split times must be finite")
    if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError("split times must all differ")
    return edges


def _fmt(value: float, spec: str = ".4g") -> str:
    return _DASH if not math.isfinite(value) else format(value, spec)


def rms_weights(rms: Sequence[float]) -> tuple[float, ...]:
    """Relative weights ``1 / rms^2``, normalised to mean one over the subsets that
    have one; a subset without a finite, positive rms gets weight 1.

    The numbers the session's ``subset_rms_weights`` query gives, from the residual
    scatter the dock already shows.
    """
    values = np.asarray(rms, dtype=np.float64)
    weights = np.ones(values.size, dtype=np.float64)
    good = np.isfinite(values) & (values > 0.0)
    if good.any():
        weights[good] = 1.0 / (values[good] * values[good])
        weights[good] /= float(np.mean(weights[good]))
    return tuple(float(w) for w in weights)


class SubsetsPanel(QtWidgets.QWidget):
    """Partitions, their subsets and roles (the Subsets dock, beside Frequencies)."""

    #: A :class:`PartitionRequest` to resolve on the worker and define.
    partition_requested = Signal(object)
    #: ``(op, args)``: one session op (``exclude_subsets``, ``use_offsets``, ...).
    edit_requested = Signal(str, object)
    #: ``{"actions": ((op, args), ...), "expect_id_epoch": None}``: ops that undo as
    #: one (new weights *and* switching them on).
    group_requested = Signal(object)
    #: A boolean mask over the view's points (the selected subsets), or None.
    points_highlighted = Signal(object)
    #: A problem with what the user typed (shown in the status bar).
    message = Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Subsets")
        self._view: SessionView | None = None
        self._label_columns: tuple[str, ...] = ()
        self._partition: str | None = None
        self._rows: list[SubsetView] = []
        #: Per shown row: (excluded, deleted points) as of the last view.
        self._use_state: list[tuple[bool, int]] = []
        self._filling = False
        self._busy = False
        self._time_cursor: float | None = None
        self._suggestion_cache: tuple[object, tuple[PartitionOrigin, int] | None] = (
            None,
            None,
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        # The rows wrap (FlowLayout): as single rows they made the pane 757 px wide,
        # so in a 400 px dock Apply, Remove and two role boxes sat off to the right.
        root.addWidget(self._build_split_row())
        root.addLayout(self._build_partition_row())
        root.addWidget(self._build_roles_row())
        self._roles_summary = QtWidgets.QLabel("")
        self._roles_summary.setObjectName("muted")
        self._roles_summary.setTextFormat(Qt.TextFormat.PlainText)
        self._roles_summary.setWordWrap(True)
        self._roles_summary.setAccessibleName("Roles of the partitions")
        root.addWidget(self._roles_summary)
        root.addLayout(self._build_hint_row())
        self._warning = QtWidgets.QLabel("")
        self._warning.setObjectName("error")
        self._warning.setTextFormat(Qt.TextFormat.PlainText)
        self._warning.setWordWrap(True)
        self._warning.setAccessibleName("Subset warnings")
        self._warning.setVisible(False)
        root.addWidget(self._warning)
        root.addWidget(self._build_table(), 1)
        root.addWidget(self._build_action_row())
        self._empty_hint = QtWidgets.QLabel(_NO_SESSION_HINT)
        self._empty_hint.setObjectName("muted")
        self._empty_hint.setWordWrap(True)
        root.addWidget(self._empty_hint)
        self._on_preset_changed()
        self._sync_enabled()

    # -- construction ------------------------------------------------------------
    @staticmethod
    def _tip(widget: QtWidgets.QWidget, name: str, tip: str) -> None:
        widget.setAccessibleName(name)
        widget.setToolTip(tip)
        widget.setStatusTip(tip)

    @staticmethod
    def _group(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Widgets that wrap as one (a label stays beside its field)."""
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        for widget in widgets:
            row.addWidget(widget)
        return box

    @staticmethod
    def _flow(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """A row that wraps onto more lines when the dock is narrow."""
        host = QtWidgets.QWidget()
        flow = FlowLayout(hspacing=6, vspacing=4)
        flow.setContentsMargins(0, 0, 0, 0)
        host.setLayout(flow)
        for widget in widgets:
            flow.addWidget(widget)
        return host

    def _build_split_row(self) -> QtWidgets.QWidget:
        label = QtWidgets.QLabel("Split &by")
        self._preset = QtWidgets.QComboBox()
        for preset in _PRESETS:
            self._preset.addItem(preset.text, preset.key)
            self._preset.setItemData(
                self._preset.count() - 1, preset.tip, Qt.ItemDataRole.ToolTipRole
            )
        label.setBuddy(self._preset)
        self._tip(
            self._preset,
            "Split by",
            "How to split the points into subsets (a new partition)",
        )
        self._preset.currentIndexChanged.connect(self._on_preset_changed)

        self._param_label = QtWidgets.QLabel("gap")
        self._param = QtWidgets.QDoubleSpinBox()
        self._param.setDecimals(3)
        self._param.setRange(0.001, 100000.0)
        self._param.setSuffix(" d")
        self._param.setKeyboardTracking(False)
        self._param_label.setBuddy(self._param)
        self._column = QtWidgets.QComboBox()
        self._tip(
            self._column,
            "Label column",
            "The per-point label column whose values become the subsets",
        )
        self._edges = QtWidgets.QLineEdit()
        self._edges.setPlaceholderText("split times, e.g. 8015.3, 8030")
        self._tip(
            self._edges,
            "Split times",
            "Times (days) at which to split, separated by commas or spaces",
        )
        self._edges.setMinimumWidth(200)
        self._edges.returnPressed.connect(self._apply_split)

        # "&Name" (not "&as"): Alt+A belongs to Apply.
        name_label = QtWidgets.QLabel("as &name")
        self._name = QtWidgets.QLineEdit()
        self._name.setMaxLength(_MAX_NAME)
        self._name.setMaximumWidth(110)
        name_label.setBuddy(self._name)
        self._tip(
            self._name,
            "Partition name",
            "The name of the new partition (an existing name is replaced, unless a "
            "role uses it)",
        )
        self._name.returnPressed.connect(self._apply_split)
        self._apply = QtWidgets.QPushButton("&Apply")
        self._tip(self._apply, "Apply split", "Define the partition described left")
        self._apply.clicked.connect(self._apply_split)
        return self._flow(
            self._group(label, self._preset),
            self._group(self._param_label, self._param, self._column, self._edges),
            self._group(name_label, self._name, self._apply),
        )

    def _build_partition_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Par&tition")
        self._partition_combo = QtWidgets.QComboBox()
        label.setBuddy(self._partition_combo)
        self._tip(
            self._partition_combo,
            "Partition shown",
            "The partition whose subsets the table lists and the role boxes set",
        )
        self._partition_combo.currentIndexChanged.connect(self._on_partition_changed)
        self._remove = QtWidgets.QPushButton("&Remove")
        self._tip(
            self._remove,
            "Remove partition",
            "Forget this partition (refused while a role uses it or a subset of it "
            "is left out)",
        )
        self._remove.clicked.connect(self._remove_partition)
        row.addWidget(label)
        row.addWidget(self._partition_combo, 1)
        row.addWidget(self._remove)
        return row

    def _build_roles_row(self) -> QtWidgets.QWidget:
        self._offsets = QtWidgets.QCheckBox("&Zero point per subset")
        self._tip(self._offsets, "Zero point per subset", _OFFSETS_TIP)
        self._weights = QtWidgets.QCheckBox("&Weight per subset")
        self._tip(
            self._weights,
            "Weight per subset",
            "Weigh each point by its subset's relative weight (the weight column) "
            "on top of its error",
        )
        self._variation = QtWidgets.QCheckBox("&Use for amplitude/phase")
        self._tip(
            self._variation,
            "Use for amplitude and phase",
            "The partition behind per-subset amplitudes and phases (Variations "
            "dock); switching it resets every per-subset component to common values",
        )
        self._offsets.clicked.connect(lambda on: self._set_role("use_offsets", on))
        self._weights.clicked.connect(lambda on: self._set_role("use_weights", on))
        self._variation.clicked.connect(
            lambda on: self._set_role("use_variation", on)
        )
        return self._flow(self._offsets, self._weights, self._variation)

    def _build_hint_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self._hint = QtWidgets.QLabel("")
        self._hint.setObjectName("muted")
        self._hint.setTextFormat(Qt.TextFormat.PlainText)
        self._hint.setWordWrap(True)
        self._hint_button = QtWidgets.QPushButton("Split into nights")
        self._tip(
            self._hint_button,
            "Apply the suggested split",
            "Define the partition the sampling suggests",
        )
        self._hint_button.clicked.connect(self._apply_hint)
        row.addWidget(self._hint, 1)
        row.addWidget(self._hint_button)
        self._hint.setVisible(False)
        self._hint_button.setVisible(False)
        self._hint_origin: tuple[PartitionOrigin, str] | None = None
        return row

    def _build_table(self) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(0, len(COLUMNS))
        table.setHorizontalHeaderLabels(list(COLUMNS))
        table.setAccessibleName("Subsets of the partition")
        table.setToolTip(
            "Space toggles 'use' (whole subsets in or out of the fit); F2 edits a "
            "name or a weight"
        )
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        table.setTabKeyNavigation(False)
        vheader = table.verticalHeader()
        if vheader is not None:
            vheader.setVisible(False)
        hheader = table.horizontalHeader()
        if hheader is not None:
            hheader.setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            hheader.setStretchLastSection(True)
        tips = {
            _USE: "In the fit: ticked = all points, partly = some points deleted, "
            "empty = the whole subset left out",
            _NAME: "Subset name (editable)",
            _N: "Points in use (of all points in the subset)",
            _START: "First time in the subset",
            _SPAN: "Time span of the subset",
            _ZERO: "Fitted zero point and its 1-sigma error (with 'Zero point per "
            "subset')",
            _WEIGHT: "Relative weight (editable; applies with 'Weight per subset')",
            _RMS: "RMS of the residuals of the points in use",
        }
        for column, tip in tips.items():
            item = table.horizontalHeaderItem(column)
            if item is not None:
                item.setToolTip(tip)
        table.itemChanged.connect(self._on_item_changed)
        table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table = table
        return table

    def _build_action_row(self) -> QtWidgets.QWidget:
        self._rms_weights = QtWidgets.QPushButton("Weights ← 1/rms²")
        self._tip(
            self._rms_weights,
            "Weights from residual scatter",
            "Set each subset's weight to 1/rms² of its residuals (mean one) and "
            "weigh by subset",
        )
        self._rms_weights.clicked.connect(self._apply_rms_weights)
        self._equal = QtWidgets.QPushButton("E&qual weights")
        self._tip(self._equal, "Equal weights", "Give every subset weight 1")
        self._equal.clicked.connect(self._apply_equal_weights)
        self._merge = QtWidgets.QPushButton("&Merge")
        self._tip(self._merge, "Merge subsets", "Merge the selected subsets into one")
        self._merge.clicked.connect(self._merge_selected)
        self._split = QtWidgets.QPushButton("&Split at")
        self._tip(
            self._split,
            "Split at time",
            "Split the subset that straddles the time given: the light curve's "
            "time cursor (click the light curve, or move the cursor with "
            "Left/Right), else the middle of the selected subset",
        )
        self._split.clicked.connect(self._split_at)
        self._split_time = QtWidgets.QDoubleSpinBox()
        self._split_time.setDecimals(5)
        self._split_time.setRange(-1e12, 1e12)
        self._split_time.setKeyboardTracking(False)
        # Its ±1e12 range made it 200 px wide, the widest thing in the pane.
        self._split_time.setMaximumWidth(140)
        self._tip(
            self._split_time,
            "Split time",
            "Time (days) at which 'Split at' cuts a subset in two; a click in the "
            "light curve sets it",
        )
        self._cross = QtWidgets.QToolButton()
        self._cross.setText("&Cross with…")
        self._tip(
            self._cross,
            "Cross with another partition",
            "Every combination of this partition's subsets with another's (site × "
            "night)",
        )
        self._cross.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self._cross_menu = QtWidgets.QMenu(self._cross)
        self._cross.setMenu(self._cross_menu)
        return self._flow(
            self._rms_weights,
            self._equal,
            self._merge,
            self._group(self._split, self._split_time),
            self._cross,
        )

    # -- public API --------------------------------------------------------------
    def set_label_columns(self, names: Sequence[str]) -> None:
        """The per-point label columns of the session's data ("Column…", "Bands")."""
        self._label_columns = tuple(str(n) for n in names)
        self._fill_columns()

    def set_time_cursor(self, time: float | None) -> None:
        """The light curve's time cursor: where *Split at* cuts."""
        self._time_cursor = time if time is not None and math.isfinite(time) else None
        if self._time_cursor is not None:
            self._split_time.setValue(self._time_cursor)

    def set_busy(self, busy: bool) -> None:
        """A long job is running: every control that edits the session pauses."""
        self._busy = bool(busy)
        self._sync_enabled()

    def clear(self) -> None:
        """No session: an empty, disabled dock."""
        self.set_view(None)

    def current_partition(self) -> str | None:
        """The partition the table shows."""
        return self._partition

    def selected_subsets(self) -> tuple[int, ...]:
        """Indices of the selected subsets of :meth:`current_partition`."""
        model = self._table.selectionModel()
        if model is None:
            return ()
        rows = sorted({index.row() for index in model.selectedRows()})
        return tuple(self._rows[r].index for r in rows if r < len(self._rows))

    def set_view(self, view: SessionView | None) -> None:
        """Show ``view``'s partitions; the shown partition and selection survive."""
        self._view = view
        if view is not None:
            extra = getattr(view, "label_columns", None)
            if extra is not None and not self._label_columns:
                self._label_columns = tuple(str(n) for n in extra)
        self._fill_columns()
        partitions = tuple(view.partitions) if view is not None else ()
        wanted = self._partition
        if wanted not in partitions:
            wanted = None
            if view is not None:
                for name in (view.offsets_by, view.variation_by, view.weights_by):
                    if name in partitions:
                        wanted = name
                        break
            if wanted is None and partitions:
                wanted = partitions[0]
        self._filling = True
        try:
            combo = self._partition_combo
            combo.clear()
            for name in partitions:
                count = (
                    sum(1 for s in view.subsets if s.partition == name)
                    if view is not None
                    else 0
                )
                combo.addItem(f"{name} ({count})", name)
            if wanted is not None:
                combo.setCurrentIndex(combo.findData(wanted))
        finally:
            self._filling = False
        self._partition = wanted
        self._show_partition()
        self._show_roles()
        self._show_hint()
        self._show_warnings()
        self._fill_cross_menu()
        # With a session but no partition the pane was an empty table and greyed
        # boxes: say what to do first.
        if view is None:
            self._empty_hint.setText(_NO_SESSION_HINT)
        elif not view.partitions:
            self._empty_hint.setText(_NO_PARTITION_HINT)
        self._empty_hint.setVisible(view is None or not view.partitions)
        self._sync_enabled()

    # -- painting ----------------------------------------------------------------
    def _fill_columns(self) -> None:
        current = self._column.currentData()
        self._column.blockSignals(True)
        self._column.clear()
        for name in self._label_columns:
            self._column.addItem(name, name)
        index = self._column.findData(current)
        self._column.setCurrentIndex(max(index, 0))
        self._column.blockSignals(False)
        model = self._preset.model()
        if isinstance(model, QtGui.QStandardItemModel):
            for row, preset in enumerate(_PRESETS):
                item = model.item(row)
                if item is None:
                    continue
                if preset.key == "bands":
                    item.setEnabled("band" in self._label_columns)
                elif preset.key == "column":
                    item.setEnabled(bool(self._label_columns))

    def _show_partition(self) -> None:
        view = self._view
        name = self._partition
        selected = set(self.selected_subsets())
        rows = (
            sorted(
                (s for s in view.subsets if s.partition == name),
                key=lambda s: s.index,
            )
            if view is not None and name is not None
            else []
        )
        self._rows = rows
        self._use_state = [self._subset_use(s) for s in rows]
        table = self._table
        self._filling = True
        try:
            table.setRowCount(len(rows))
            for r, subset in enumerate(rows):
                self._fill_row(r, subset)
            model = table.selectionModel()
            table.clearSelection()
            if model is not None:
                for r, subset in enumerate(rows):
                    if subset.index in selected:
                        model.select(
                            table.model().index(r, 0),
                            QtCore.QItemSelectionModel.SelectionFlag.Select
                            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                        )
        finally:
            self._filling = False

    def _subset_use(self, subset: SubsetView) -> tuple[bool, int]:
        """``(excluded, deleted points)`` of a subset.

        The view models may name both (``excluded``, ``n_deleted``); otherwise they
        follow from the counts: a subset with no point in use is taken as excluded,
        and the points missing from one in use as deleted.
        """
        excluded = getattr(subset, "excluded", None)
        deleted = getattr(subset, "n_deleted", None)
        if excluded is None:
            excluded = subset.n_points > 0 and subset.n_used == 0
        if deleted is None:
            deleted = 0 if excluded else max(subset.n_points - subset.n_used, 0)
        return bool(excluded), int(deleted)

    def _fill_row(self, r: int, subset: SubsetView) -> None:
        table = self._table
        excluded, deleted = self._use_state[r]
        # The box's state is also spelled out beside it: an empty box alone reads as
        # "nothing here" rather than "left out".
        if excluded:
            check, word = Qt.CheckState.Unchecked, "no"
            state_text = "left out of the fit"
        elif deleted:
            check, word = Qt.CheckState.PartiallyChecked, "partly"
            state_text = f"{deleted} of {subset.n_points} points deleted"
        else:
            check, word = Qt.CheckState.Checked, "yes"
            state_text = "in use, all points"
        use = QtWidgets.QTableWidgetItem(word)
        use.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        use.setCheckState(check)
        use.setToolTip(state_text)
        use.setData(Qt.ItemDataRole.AccessibleTextRole, state_text)
        # The column fits its contents, but the item's size hint leaves out the
        # styled check box, so "yes" / "partly" were elided to "…".
        metrics = table.fontMetrics()
        room = QtCore.QSize(metrics.horizontalAdvance("partly") + 44, metrics.height())
        use.setData(Qt.ItemDataRole.SizeHintRole, room + QtCore.QSize(0, 8))
        table.setItem(r, _USE, use)

        def cell(text: str, editable: bool = False, tip: str = "") -> None:
            item = QtWidgets.QTableWidgetItem(text)
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if editable:
                flags |= Qt.ItemFlag.ItemIsEditable
            item.setFlags(flags)
            if tip:
                item.setToolTip(tip)
            table.setItem(r, column[0], item)
            column[0] += 1

        column = [_NAME]
        cell(subset.label, editable=True)
        n_text = (
            str(subset.n_points)
            if subset.n_used == subset.n_points
            else f"{subset.n_used} of {subset.n_points}"
        )
        cell(n_text)
        cell(_fmt(subset.t_start, ".5f"))
        span = subset.t_stop - subset.t_start
        cell(_DASH if not math.isfinite(span) else f"{span:.3g} d")
        if math.isfinite(subset.zero_point):
            zero = f"{subset.zero_point:.6g}"
            if math.isfinite(subset.zero_point_error):
                zero += f" ± {subset.zero_point_error:.2g}"
            cell(zero)
        else:
            cell(
                _DASH,
                tip="Fitted only while 'Zero point per subset' is ticked for this "
                "partition (and after a fit)",
            )
        cell(_fmt(subset.weight), editable=True)
        cell(_fmt(subset.rms))

    def _show_roles(self) -> None:
        view = self._view
        name = self._partition
        roles = (
            (self._offsets, view.offsets_by if view else None, "zero points"),
            (self._weights, view.weights_by if view else None, "weights"),
            (self._variation, view.variation_by if view else None, "amplitude/phase"),
        )
        for box, owner, _ in roles:
            box.setChecked(name is not None and owner == name)
        if view is None:
            self._roles_summary.setText("")
            return
        parts = [
            f"{label.capitalize()}: {owner if owner else 'off'}"
            for _, owner, label in roles
        ]
        self._roles_summary.setText("  ·  ".join(parts))

    def _suggestion(self) -> tuple[PartitionOrigin, int] | None:
        """The sampling's obvious split and its size, computed once per curve.

        Cheap (a sort and a gap scan; 26 ms on 1e6 points) and cached on the curve's
        identity, so repainting after every edit costs nothing.
        """
        view = self._view
        if view is None or view.time.size < 2:
            return None
        key = (view.key, int(view.time.size), float(view.time[0]), float(view.time[-1]))
        cached_key, cached = self._suggestion_cache
        if cached_key == key:
            return cached
        result: tuple[PartitionOrigin, int] | None = None
        try:
            origin = suggest_partition(view.time)
            if origin is not None:
                count = Partition.from_origin(origin, view.time, name="hint").n_subsets
                result = (origin, count)
        except ValueError:
            result = None
        self._suggestion_cache = (key, result)
        return result

    def _show_hint(self) -> None:
        view = self._view
        suggestion = None
        if view is not None and not [p for p in view.partitions if p != "band"]:
            suggestion = self._suggestion()
        if suggestion is None:
            self._hint_origin = None
            self._hint.setVisible(False)
            self._hint_button.setVisible(False)
            return
        origin, count = suggestion
        looks, button, name = _origin_text(origin, count)
        self._hint_origin = (origin, name)
        self._hint.setText(f"Hint: this looks like {looks}.")
        self._hint_button.setText(button)
        self._hint.setVisible(True)
        self._hint_button.setVisible(True)

    def _show_warnings(self) -> None:
        view = self._view
        if view is None:
            self._warning.setVisible(False)
            return
        low = [r.label for r in view.rows if "low_frequency" in r.flags]
        short = [r.label for r in view.rows if "short_subset" in r.flags]
        lines: list[str] = []
        if low:
            lines.append(
                f"Warning: {', '.join(low)} — under half a cycle per zero-point "
                "subset, so the zero points can absorb the signal."
            )
        if short:
            lines.append(
                f"Warning: {', '.join(short)} — a subset spans less than one cycle, "
                "so its amplitude/phase is poorly defined (merge short subsets)."
            )
        lines.extend(f"Warning: {w}" for w in view.warnings)
        self._warning.setText("\n".join(lines))
        self._warning.setVisible(bool(lines))

    def _fill_cross_menu(self) -> None:
        self._cross_menu.clear()
        view = self._view
        others = (
            [p for p in view.partitions if p != self._partition]
            if view is not None
            else []
        )
        for other in others:
            action = self._cross_menu.addAction(other)
            action.triggered.connect(lambda _c=False, o=other: self._cross_with(o))
        self._cross.setEnabled(bool(others) and not self._busy)

    def _sync_enabled(self) -> None:
        has_view = self._view is not None
        editable = has_view and not self._busy
        has_partition = editable and self._partition is not None
        editors: tuple[QtWidgets.QWidget, ...] = (
            self._preset,
            self._param,
            self._column,
            self._edges,
            self._name,
            self._apply,
            self._hint_button,
        )
        for widget in editors:
            widget.setEnabled(editable)
        self._partition_combo.setEnabled(has_view)
        partition_widgets: tuple[QtWidgets.QWidget, ...] = (
            self._remove,
            self._offsets,
            self._weights,
            self._equal,
            self._split,
            self._split_time,
        )
        for widget in partition_widgets:
            widget.setEnabled(has_partition)
        view = self._view
        settings = view.settings if view is not None else None
        if settings is not None and not getattr(settings, "fit_mean", True):
            self._offsets.setEnabled(False)
            self._offsets.setToolTip(
                "Zero points per subset need a free mean (Options: fit mean)"
            )
        else:
            self._offsets.setToolTip(_OFFSETS_TIP)
        self._variation.setEnabled(has_partition)
        self._rms_weights.setEnabled(
            has_partition and any(math.isfinite(s.rms) for s in self._rows)
        )
        self._merge.setEnabled(has_partition and len(self.selected_subsets()) >= 2)
        self._cross.setEnabled(
            editable
            and view is not None
            and any(p != self._partition for p in view.partitions)
        )
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
            if not editable
            else QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed
        )

    # -- reactions ---------------------------------------------------------------
    def _preset_now(self) -> _Preset:
        key = self._preset.currentData()
        for preset in _PRESETS:
            if preset.key == key:
                return preset
        return _PRESETS[0]

    def _on_preset_changed(self, *_args: object) -> None:
        preset = self._preset_now()
        self._param_label.setVisible(preset.param in ("gap", "length"))
        self._param.setVisible(preset.param in ("gap", "length"))
        self._column.setVisible(preset.param == "column")
        self._edges.setVisible(preset.param == "edges")
        if preset.param in ("gap", "length"):
            self._param_label.setText("&gap" if preset.param == "gap" else "&length")
            self._param.setValue(preset.default)
            self._tip(
                self._param,
                "Gap" if preset.param == "gap" else "Chunk length",
                preset.tip,
            )
        name = preset.name
        if preset.param == "column":
            name = str(self._column.currentData() or "")
        self._name.setText(name)
        self._name.setPlaceholderText(name or "name")

    def _origin(self) -> PartitionOrigin:
        preset = self._preset_now()
        value = float(self._param.value())
        if preset.key == "nights":
            return GapsOrigin(min_gap=value, prefix="N")
        if preset.key == "sectors":
            return GapsOrigin(min_gap=0.5, max_span=28.5, prefix="S")
        if preset.key == "quarters":
            return GapsOrigin(min_gap=0.5, max_span=100.0, prefix="Q")
        if preset.key == "seasons":
            return GapsOrigin(min_gap=value, prefix="S")
        if preset.key == "chunks":
            return ChunksOrigin(length=value, prefix="C")
        if preset.key == "bands":
            return LabelsOrigin(column="band")
        if preset.key == "column":
            column = self._column.currentData()
            if not column:
                raise ValueError("the data has no label column to split by")
            return LabelsOrigin(column=str(column))
        return RangesOrigin(edges=_parse_edges(self._edges.text()))

    def _apply_split(self) -> None:
        if self._view is None or self._busy:
            return
        preset = self._preset_now()
        name = self._name.text().strip() or preset.name
        if preset.param == "column" and not self._name.text().strip():
            name = str(self._column.currentData() or "")
        if not name or len(name) > _MAX_NAME:
            self.message.emit(f"A partition name needs 1 to {_MAX_NAME} characters.")
            return
        try:
            origin = self._origin()
        except (ValueError, ValidationError) as exc:
            self.message.emit(f"Cannot split: {exc}")
            return
        self.partition_requested.emit(PartitionRequest(name=name, origin=origin))

    def _apply_hint(self) -> None:
        if self._hint_origin is None or self._busy:
            return
        origin, name = self._hint_origin
        self.partition_requested.emit(PartitionRequest(name=name, origin=origin))

    def _on_partition_changed(self, *_args: object) -> None:
        if self._filling:
            return
        data = self._partition_combo.currentData()
        self._partition = None if data is None else str(data)
        self._table.clearSelection()
        self._show_partition()
        self._show_roles()
        self._fill_cross_menu()
        self._sync_enabled()
        self.points_highlighted.emit(None)

    def _remove_partition(self) -> None:
        if self._partition is not None and not self._busy:
            self.edit_requested.emit("drop_partition", {"name": self._partition})

    def _set_role(self, op: str, on: bool) -> None:
        if self._partition is None or self._busy:
            return
        self.edit_requested.emit(op, {"name": self._partition if on else None})

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._filling or self._partition is None:
            return
        r = item.row()
        if r >= len(self._rows):
            return
        column = item.column()
        if column == _USE:
            self._toggle_use(r)
        elif column == _NAME:
            self._rename(r, item.text().strip())
        elif column == _WEIGHT:
            self._reweigh(r, item.text().strip())

    def _restore_cell(self, r: int) -> None:
        self._filling = True
        try:
            self._fill_row(r, self._rows[r])
        finally:
            self._filling = False

    def _toggle_use(self, r: int) -> None:
        """Use box clicked: whole subsets in or out (never single points).

        The clicked row decides the direction -- a left-out subset comes back, a
        used one (with or without deleted points) goes out -- and every selected row
        follows it when the clicked row is among them.
        """
        assert self._partition is not None
        excluded, _ = self._use_state[r]
        target = not excluded
        rows = [r]
        model = self._table.selectionModel()
        selected_rows = (
            {i.row() for i in model.selectedRows()} if model is not None else set()
        )
        if r in selected_rows and len(selected_rows) > 1:
            rows = sorted(selected_rows)
        changed = [i for i in rows if self._use_state[i][0] != target]
        # Show the outcome at once (Qt's own cycle would turn a partly-used subset
        # *on*); the view the edit produces repaints it from the session anyway.
        for i in changed:
            self._use_state[i] = (target, self._use_state[i][1])
            self._restore_cell(i)
        if r not in changed:
            self._restore_cell(r)
        if not changed or self._busy:
            return
        self.edit_requested.emit(
            "exclude_subsets",
            {
                "name": self._partition,
                "subsets": [self._rows[i].index for i in changed],
                "excluded": target,
            },
        )

    def _rename(self, r: int, text: str) -> None:
        assert self._partition is not None
        subset = self._rows[r]
        others = {s.label for i, s in enumerate(self._rows) if i != r}
        problem = ""
        if not text or len(text) > _MAX_NAME:
            problem = f"A subset name needs 1 to {_MAX_NAME} characters."
        elif text in others:
            problem = f"Another subset is already called {text!r}."
        if problem or text == subset.label or self._busy:
            if problem:
                self.message.emit(problem)
            self._restore_cell(r)
            return
        self.partition_requested.emit(
            PartitionRequest(
                name=self._partition, edit="rename", subsets=(subset.index,), label=text
            )
        )

    def _reweigh(self, r: int, text: str) -> None:
        assert self._partition is not None
        try:
            weight = float(text)
        except ValueError:
            weight = math.nan
        if not (math.isfinite(weight) and weight > 0.0):
            self.message.emit(f"A weight must be a positive number, not {text!r}.")
            self._restore_cell(r)
            return
        weights = [float(s.weight) for s in self._rows]
        if weights[r] == weight or self._busy:
            self._restore_cell(r)
            return
        weights[r] = weight
        self.edit_requested.emit(
            "set_subset_weights", {"name": self._partition, "weights": weights}
        )

    def _apply_weights(self, weights: Sequence[float]) -> None:
        if self._partition is None or self._view is None or self._busy:
            return
        name = self._partition
        set_op = ("set_subset_weights", {"name": name, "weights": list(weights)})
        if self._view.weights_by == name:
            self.edit_requested.emit(*set_op)
            return
        self.group_requested.emit(
            {
                "actions": (set_op, ("use_weights", {"name": name})),
                "expect_id_epoch": None,
            }
        )

    def _apply_rms_weights(self) -> None:
        self._apply_weights(rms_weights([s.rms for s in self._rows]))

    def _apply_equal_weights(self) -> None:
        if self._partition is None or self._busy:
            return
        self.edit_requested.emit(
            "set_subset_weights",
            {"name": self._partition, "weights": [1.0] * len(self._rows)},
        )

    def _merge_selected(self) -> None:
        chosen = self.selected_subsets()
        if self._partition is None or len(chosen) < 2 or self._busy:
            return
        self.partition_requested.emit(
            PartitionRequest(name=self._partition, edit="merge", subsets=chosen)
        )

    def _split_at(self) -> None:
        if self._partition is None or self._busy:
            return
        self.partition_requested.emit(
            PartitionRequest(
                name=self._partition,
                edit="split_at",
                time=float(self._split_time.value()),
            )
        )

    def _cross_with(self, other: str) -> None:
        if self._partition is None or self._busy:
            return
        name = f"{self._partition}_x_{other}"[:_MAX_NAME]
        self.partition_requested.emit(
            PartitionRequest(
                name=name, origin=CrossedOrigin(a=self._partition, b=other)
            )
        )

    def _on_selection_changed(self) -> None:
        if self._filling:
            return
        chosen = self.selected_subsets()
        self._merge.setEnabled(
            self._partition is not None and len(chosen) >= 2 and not self._busy
        )
        if chosen and self._time_cursor is None:
            first = next(s for s in self._rows if s.index == chosen[0])
            if math.isfinite(first.t_start) and math.isfinite(first.t_stop):
                self._split_time.setValue(0.5 * (first.t_start + first.t_stop))
        self.points_highlighted.emit(self._highlight_mask(chosen))

    def _highlight_mask(self, chosen: Sequence[int]) -> BoolArray | None:
        """The points of the selected subsets, or None (nothing to highlight).

        Exact from the partition's codes when the view carries them; otherwise from
        the subsets' time spans, which is exact for any partition into time ranges
        (nights, sectors, chunks) -- the spans of a label partition overlap, and then
        nothing is highlighted rather than the wrong points.
        """
        view = self._view
        if view is None or not chosen or self._partition is None:
            return None
        codes = self._codes(view, self._partition)
        if codes is not None:
            mask = np.isin(codes, np.asarray(chosen, dtype=np.int64))
            return np.asarray(mask, dtype=np.bool_)
        spans = sorted(
            (s.t_start, s.t_stop) for s in self._rows if math.isfinite(s.t_start)
        )
        if any(b[0] <= a[1] for a, b in zip(spans, spans[1:], strict=False)):
            return None
        mask = np.zeros(view.time.size, dtype=np.bool_)
        for s in self._rows:
            if s.index in chosen and math.isfinite(s.t_start):
                mask |= (view.time >= s.t_start) & (view.time <= s.t_stop)
        return mask

    @staticmethod
    def _codes(view: SessionView, name: str) -> npt.NDArray[np.int64] | None:
        every = getattr(view, "partition_codes", None)
        if isinstance(every, Mapping) and name in every:
            return np.asarray(every[name], dtype=np.int64)
        shown = view.offsets_by or (view.partitions[0] if view.partitions else None)
        if view.point_codes is not None and shown == name:
            return np.asarray(view.point_codes, dtype=np.int64)
        return None



__all__ = ["COLUMNS", "PartitionRequest", "SubsetsPanel", "rms_weights"]
