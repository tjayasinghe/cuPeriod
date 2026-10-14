"""The Variations dock: does a mode's amplitude or phase change between subsets?

Pick components (by default the rows selected in the frequency table), what may vary
(amplitude and phase, amplitude only, phase only) and the partition (nights,
seasons, bands ...), then *Calculate*. The session fits a copy of its model in which
those components take one amplitude and/or phase per subset, every frequency held,
and tests each component's per-subset values for constancy (a Wald statistic on the
successive differences, ``chi2`` with ``S - 1`` degrees of freedom). Nothing in the
session changes until *Apply to model*, which makes the same model the session's own
(``set_variation``, one undoable action).

Two stacked plots share the x axis -- each subset's mid-time, or its label for a
partition that is not a time order (bands) -- with the amplitude per subset (1-sigma
bars, the common value as a dashed line inside a band of its own 1-sigma error) above
the phase difference from the common phase below, switchable to O−C in seconds
(``-dphi / (2 pi f)``). Components are told apart by marker shape as well as colour,
and the table spells out which ones vary (``p < 0.01``) in words.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from cuperiod.gui.qt import Qt, QtWidgets, Signal, pg
from cuperiod.gui.theme import ThemePalette, apply_plot_theme, palette, plot_label_style
from cuperiod.gui.widgets._view_helpers import stroke_width
from cuperiod.gui.widgets.info_bar import FlowLayout

if TYPE_CHECKING:
    from cuperiod.gui.viewmodels import SessionView
    from cuperiod.prewhiten.session.state import VariationResult

VariationMode = Literal["both", "amplitude", "phase"]

#: Significance below which a component is reported as varying.
P_FLAG = 0.01

_SECONDS_PER_DAY = 86400.0

_MODES: tuple[tuple[str, VariationMode], ...] = (
    ("A and φ", "both"),
    ("A only", "amplitude"),
    ("φ only", "phase"),
)

#: Tests table columns.
TEST_COLUMNS: tuple[str, ...] = (
    "component",
    "subsets",
    "A: Wald W",
    "A: p",
    "φ: Wald W",
    "φ: p",
    "ΔBIC",
    "verdict",
)

#: Per-subset table columns.
SUBSET_COLUMNS: tuple[str, ...] = (
    "component",
    "subset",
    "mid-time",
    "N",
    "amplitude ± 1σ",
    "Δφ (rad) ± 1σ",
    "O−C (s) ± 1σ",
)

_DASH = "–"


@dataclass(frozen=True)
class VariationQuery:
    """What *Calculate* asks the controller for (``query_variations``).

    ``ids`` None means every active component.
    """

    ids: tuple[int, ...] | None
    mode: VariationMode
    by: str | None
    improve_frequencies: bool = False


def parse_components(text: str) -> tuple[int, ...]:
    """Component ids from ``"F1, F3"``, ``"1 3"`` or ``"f1;f3"`` (empty: none).

    Raises
    ------
    ValueError
        For anything that is not a component label or number.
    """
    ids: list[int] = []
    for part in re.split(r"[,;\s]+", text.strip()):
        if not part:
            continue
        match = re.fullmatch(r"[Ff]?(\d+)", part)
        if match is None:
            raise ValueError(f"{part!r} is not a component (write F1, F2, ...)")
        value = int(match.group(1))
        if value not in ids:
            ids.append(value)
    return tuple(ids)


def wrap_phase(dphi: Any) -> np.ndarray:
    """Phase differences wrapped to ``(-pi, pi]``."""
    values = np.asarray(dphi, dtype=np.float64)
    wrapped = np.mod(values + np.pi, 2.0 * np.pi) - np.pi
    return np.where(wrapped == -np.pi, np.pi, wrapped)


def o_minus_c(dphi: Any, frequency: float) -> np.ndarray:
    """O−C in seconds of phase differences (radians) of a mode at ``frequency`` (c/d).

    A later arrival is a phase lag: ``O - C = -dphi / (2 pi f)`` days.
    """
    return -np.asarray(dphi, dtype=np.float64) / (2.0 * np.pi * frequency) * (
        _SECONDS_PER_DAY
    )


def _fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return _DASH
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


def _fmt_pm(value: float, error: float, spec: str = ".4g") -> str:
    if not math.isfinite(value):
        return _DASH
    text = format(value, spec)
    if math.isfinite(error):
        spec_e = ".0f" if abs(error) >= 100.0 else ".2g"
        text += f" ± {format(error, spec_e)}"
    return text


class VariationPanel(QtWidgets.QWidget):
    """Per-subset amplitudes and phases with constancy tests (the Variations dock)."""

    #: *Calculate*: a :class:`VariationQuery` for ``controller.query_variations``.
    query_requested = Signal(object)
    #: *Apply to model*: ``("set_variation", args)``, args carrying
    #: ``expect_id_epoch``.
    edit_requested = Signal(str, object)
    #: *Apply to model* when the variation partition must change first:
    #: ``{"actions": (("use_variation", ...), ("set_variation", ...)),
    #: "expect_id_epoch": e}``.
    group_requested = Signal(object)
    #: A problem with what the user typed (shown in the status bar).
    message = Signal(str)

    def __init__(
        self, theme: str = "dark", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setAccessibleName("Amplitude and phase per subset")
        self._pal: ThemePalette = palette(theme)  # type: ignore[arg-type]
        self._view: SessionView | None = None
        self._result: VariationResult | None = None
        self._result_key: object = None
        self._query_key: object = None
        self._busy = False
        self._components_edited = False
        self._filling = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        # The rows wrap (FlowLayout): as single rows they made the pane 664 px wide,
        # so in a 400 px dock the partition picker and the summary were cut off.
        root.addWidget(self._build_query_row())
        root.addWidget(self._build_action_row())
        self._summary = QtWidgets.QLabel(
            "Choose components and a partition, then Calculate."
        )
        self._summary.setObjectName("muted")
        self._summary.setTextFormat(Qt.TextFormat.PlainText)
        self._summary.setWordWrap(True)
        self._summary.setAccessibleName("Variation summary")
        root.addWidget(self._summary)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_plots())
        splitter.addWidget(self._build_tables())
        splitter.setSizes([360, 200])
        root.addWidget(splitter, 1)
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

    def _build_query_row(self) -> QtWidgets.QWidget:
        label = QtWidgets.QLabel("C&omponents")
        self._components = QtWidgets.QLineEdit()
        self._components.setPlaceholderText("selected rows (all if none)")
        label.setBuddy(self._components)
        self._tip(
            self._components,
            "Components to test",
            "Components to test, e.g. F1, F3 (follows the frequency table's "
            "selection; empty = every active component)",
        )
        self._components.textEdited.connect(self._on_components_edited)
        self._components.returnPressed.connect(self.calculate)
        self._components.setMinimumWidth(170)

        mode_label = QtWidgets.QLabel("Var&y")
        self._mode = QtWidgets.QComboBox()
        for text, mode in _MODES:
            self._mode.addItem(text, mode)
        mode_label.setBuddy(self._mode)
        self._tip(
            self._mode,
            "What varies",
            "What may differ between subsets: amplitude and phase, amplitude only "
            "or phase only",
        )

        by_label = QtWidgets.QLabel("pe&r")
        self._by = QtWidgets.QComboBox()
        by_label.setBuddy(self._by)
        self._tip(
            self._by,
            "Partition",
            "The subsets to compare (partitions are made in the Subsets dock)",
        )
        return self._flow(
            self._group(label, self._components),
            self._group(mode_label, self._mode),
            self._group(by_label, self._by),
        )

    def _build_action_row(self) -> QtWidgets.QWidget:
        self._improve = QtWidgets.QCheckBox("Also &improve frequencies")
        self._tip(
            self._improve,
            "Also improve frequencies",
            "Refine the frequencies in the query fit too (slower; by default they "
            "are held, which keeps the phase test honest)",
        )
        self._calculate = QtWidgets.QPushButton("&Calculate")
        self._calculate.setProperty("primary", True)
        self._tip(
            self._calculate,
            "Calculate variations",
            "Fit per-subset amplitudes/phases and test them for constancy; the "
            "session does not change",
        )
        self._calculate.clicked.connect(self.calculate)
        self._apply = QtWidgets.QPushButton("&Apply to model")
        self._tip(
            self._apply,
            "Apply to model",
            "Make the session's model vary the same way (one undoable action)",
        )
        self._apply.clicked.connect(self.apply_to_model)
        lower_label = QtWidgets.QLabel("&Lower plot")
        self._lower = QtWidgets.QComboBox()
        self._lower.addItem("Phase difference (rad)", "phase")
        self._lower.addItem("O−C (s)", "oc")
        lower_label.setBuddy(self._lower)
        self._tip(
            self._lower,
            "Lower plot quantity",
            "Phase difference from the common phase, or the same as an O−C timing "
            "offset in seconds (-Δφ / 2πf)",
        )
        self._lower.currentIndexChanged.connect(self._redraw)
        return self._flow(
            self._improve,
            self._calculate,
            self._apply,
            self._group(lower_label, self._lower),
        )

    def _build_plots(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        style = plot_label_style(self._pal)
        self._amp_plot = pg.PlotWidget()
        self._phase_plot = pg.PlotWidget()
        for plot, name in (
            (self._amp_plot, "Amplitude per subset plot"),
            (self._phase_plot, "Phase or O−C per subset plot"),
        ):
            plot.setMenuEnabled(False)
            plot.showGrid(x=True, y=True, alpha=0.18)
            plot.setAccessibleName(name)
            plot.setToolTip(
                "Drag to pan, wheel to zoom, double-click to reset; markers carry "
                "1-sigma bars"
            )
            layout.addWidget(plot, 1)
        self._amp_plot.setLabel("left", "Amplitude", **style)
        self._phase_plot.setLabel("left", "Δφ (rad)", **style)
        self._phase_plot.setLabel("bottom", "Mid-time (d)", **style)
        self._phase_plot.setXLink(self._amp_plot)
        # The plots share x: one row of tick values (under the lower plot) is enough.
        self._amp_plot.getAxis("bottom").setStyle(showValues=False)
        self._legend = self._amp_plot.addLegend(offset=(8, 8))
        for plot in (self._amp_plot, self._phase_plot):
            apply_plot_theme(plot, self._pal)
        return box

    def _build_tables(self) -> QtWidgets.QTabWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.setAccessibleName("Variation results")
        self._tests = self._table(TEST_COLUMNS, "Constancy tests per component")
        header = self._tests.horizontalHeaderItem(TEST_COLUMNS.index("ΔBIC"))
        if header is not None:
            header.setToolTip(
                "BIC of the per-subset model (every tested component at once) minus "
                "the session's model; below about -10 the variation is real"
            )
        self._per_subset = self._table(SUBSET_COLUMNS, "Values per subset")
        tabs.addTab(self._tests, "Tests")
        tabs.addTab(self._per_subset, "Per subset")
        tabs.setTabToolTip(
            0,
            "Wald test of 'the same in every subset' per component; p < 0.01 is "
            "reported as varying",
        )
        tabs.setTabToolTip(1, "Each component's amplitude and phase in each subset")
        bar = tabs.tabBar()
        if bar is not None:
            bar.setToolTip("The constancy tests, or the values per subset")
            bar.setStatusTip("The constancy tests, or the values per subset")
        self._tabs = tabs
        return tabs

    @staticmethod
    def _table(columns: Sequence[str], name: str) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(list(columns))
        table.setAccessibleName(name)
        table.setToolTip(name)
        table.setStatusTip(name)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
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
        return table

    # -- public API --------------------------------------------------------------
    @property
    def result(self) -> VariationResult | None:
        """The result on display."""
        return self._result

    def set_view(self, view: SessionView | None) -> None:
        """Follow the session: its partitions, components and id epoch.

        A result computed for another session (another key) is dropped; one whose
        components no longer all exist stays visible but cannot be applied.
        """
        previous = self._view
        self._view = view
        partitions = tuple(view.partitions) if view is not None else ()
        current = self._by.currentData()
        wanted = current if current in partitions else None
        if wanted is None and view is not None:
            for name in (view.variation_by, view.offsets_by):
                if name in partitions:
                    wanted = name
                    break
        if wanted is None and partitions:
            wanted = partitions[0]
        self._filling = True
        try:
            self._by.clear()
            for name in partitions:
                self._by.addItem(name, name)
            if wanted is not None:
                self._by.setCurrentIndex(self._by.findData(wanted))
        finally:
            self._filling = False
        stale = view is None or self._result_key != view.key
        if self._result is not None and stale:
            self.set_result(None)
        if previous is not None and (view is None or previous.key != view.key):
            # Another session: its F-numbers name other modes.
            self._components_edited = False
            self._components.clear()
        if view is not None and self._result is not None:
            self._redraw()
        self._sync_enabled()

    def set_selected_components(self, ids: Sequence[int]) -> None:
        """The frequency table's selection: the default components to test.

        Text the user typed wins until the field is cleared.
        """
        if self._components_edited and self._components.text().strip():
            return
        self._components_edited = False
        self._components.setText(", ".join(f"F{int(i)}" for i in ids))

    def set_busy(self, busy: bool) -> None:
        """A long job is running: Apply waits (Calculate may still queue a query)."""
        self._busy = bool(busy)
        self._sync_enabled()

    def clear(self) -> None:
        """No session: nothing to show."""
        self.set_result(None)
        self.set_view(None)

    def calculate(self) -> None:
        """Emit :attr:`query_requested` for the current choice."""
        view = self._view
        if view is None:
            return
        try:
            ids = parse_components(self._components.text())
        except ValueError as exc:
            self.message.emit(str(exc))
            return
        known = {r.id for r in view.rows if r.active}
        unknown = [i for i in ids if i not in known]
        if unknown:
            self.message.emit(
                "No active component "
                + ", ".join(f"F{i}" for i in unknown)
                + " to test."
            )
            return
        by = self._by.currentData()
        if by is None:
            self.message.emit("Define a partition in the Subsets dock first.")
            return
        mode: VariationMode = self._mode.currentData()
        query = VariationQuery(
            ids=ids or None,
            mode=mode,
            by=str(by),
            improve_frequencies=self._improve.isChecked(),
        )
        self._query_key = view.key
        self._summary.setText("Calculating…")
        self.query_requested.emit(query)

    def set_result(self, result: VariationResult | None) -> None:
        """Show a :class:`~cuperiod.prewhiten.session.state.VariationResult`."""
        self._result = result
        self._result_key = (
            None
            if result is None
            else self._query_key
            if self._query_key is not None
            else (self._view.key if self._view is not None else None)
        )
        self._fill_tests()
        self._fill_per_subset()
        self._redraw()
        self._summary.setText(self._summary_text())
        self._sync_enabled()

    def apply_to_model(self) -> None:
        """Make the session vary as the result shown (``set_variation``)."""
        result, view = self._result, self._view
        if result is None or view is None or not self._can_apply():
            return
        args: dict[str, Any] = {
            "ids": list(result.component_ids),
            "mode": result.mode,
            "hold_frequency": not result.frequencies_improved,
        }
        if view.variation_by == result.by:
            self.edit_requested.emit(
                "set_variation", {**args, "expect_id_epoch": view.id_epoch}
            )
            return
        self.group_requested.emit(
            {
                "actions": (
                    ("use_variation", {"name": result.by}),
                    ("set_variation", args),
                ),
                "expect_id_epoch": view.id_epoch,
            }
        )

    def apply_theme(self, theme_palette: ThemePalette) -> None:
        """Re-skin the plots for a new theme."""
        self._pal = theme_palette
        for plot in (self._amp_plot, self._phase_plot):
            apply_plot_theme(plot, theme_palette)
        self._redraw()

    # -- state -------------------------------------------------------------------
    def _on_components_edited(self, _text: str) -> None:
        self._components_edited = True

    def _can_apply(self) -> bool:
        result, view = self._result, self._view
        if result is None or view is None or self._busy:
            return False
        if self._result_key is not None and self._result_key != view.key:
            return False
        if result.by not in view.partitions:
            return False
        ids = {r.id for r in view.rows}
        return all(i in ids for i in result.component_ids)

    def _sync_enabled(self) -> None:
        view = self._view
        has_partition = view is not None and bool(view.partitions)
        has_rows = view is not None and any(r.active for r in view.rows)
        for widget in (self._components, self._mode, self._by, self._improve):
            widget.setEnabled(view is not None)
        self._calculate.setEnabled(has_partition and has_rows)
        if view is None:
            tip = "Open a pre-whitening session first"
        elif not has_partition:
            tip = "Define a partition in the Subsets dock first"
        elif not has_rows:
            tip = "Extract at least one component first"
        else:
            tip = (
                "Fit per-subset amplitudes/phases and test them for constancy; the "
                "session does not change"
            )
        self._calculate.setToolTip(tip)
        can_apply = self._can_apply()
        self._apply.setEnabled(can_apply)
        self._lower.setEnabled(self._frequencies() is not None or self._result is None)
        if self._lower.currentData() == "oc" and self._frequencies() is None:
            self._lower.setCurrentIndex(0)

    def _frequencies(self) -> dict[int, float] | None:
        """Frequency per tested component (for O−C), or None when one is unknown."""
        result, view = self._result, self._view
        if result is None or view is None:
            return None
        out: dict[int, float] = {}
        for cid in result.component_ids:
            row = view.row(cid)
            if row is None or not (math.isfinite(row.frequency) and row.frequency > 0):
                return None
            out[cid] = float(row.frequency)
        return out

    def _summary_text(self) -> str:
        result = self._result
        if result is None:
            return "Choose components and a partition, then Calculate."
        varying = [
            f"F{cid}"
            for k, cid in enumerate(result.component_ids)
            if self._varies(k)
        ]
        dbic = float(result.delta_bic)
        if math.isfinite(dbic):
            verdict = (
                "preferred"
                if dbic < -10.0
                else "not preferred"
                if dbic > 0.0
                else "weakly preferred"
            )
            bic = (
                f"ΔBIC (per-subset − common) = {dbic:.1f}: per-subset model "
                f"{verdict}."
            )
        else:
            bic = "ΔBIC unavailable."
        text = (
            f"{len(result.component_ids)} component(s) over "
            f"{len(result.subset_labels)} subsets of {result.by!r}. {bic} "
        )
        text += (
            f"Varying (p < {P_FLAG:g}): {', '.join(varying)}."
            if varying
            else f"No component varies at p < {P_FLAG:g}."
        )
        if result.frequencies_improved:
            text += " Frequencies were improved in the query fit."
        return text

    def _varies(self, k: int) -> bool:
        result = self._result
        if result is None:
            return False
        ps = [float(result.amplitude_p_value[k]), float(result.phase_p_value[k])]
        return any(math.isfinite(p) and p < P_FLAG for p in ps)

    # -- tables ------------------------------------------------------------------
    def _fill_tests(self) -> None:
        table = self._tests
        result = self._result
        rows = len(result.component_ids) if result is not None else 0
        table.setRowCount(rows)
        if result is None:
            return
        for k, cid in enumerate(result.component_ids):
            a_p = float(result.amplitude_p_value[k])
            p_p = float(result.phase_p_value[k])
            parts = []
            if math.isfinite(a_p) and a_p < P_FLAG:
                parts.append("amplitude varies")
            if math.isfinite(p_p) and p_p < P_FLAG:
                parts.append("phase varies")
            tested = math.isfinite(a_p) or math.isfinite(p_p)
            verdict = (
                ", ".join(parts)
                if parts
                else "consistent with constant"
                if tested
                else "not tested"
            )
            values = (
                f"F{cid}",
                str(len(result.subset_labels)),
                _fmt_pm(float(result.amplitude_statistic[k]), math.nan, ".3g"),
                _fmt_p(a_p),
                _fmt_pm(float(result.phase_statistic[k]), math.nan, ".3g"),
                _fmt_p(p_p),
                _fmt_pm(float(result.delta_bic), math.nan, ".1f"),
                verdict,
            )
            for c, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(text)  # a narrow dock elides the verdict
                if parts:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                table.setItem(k, c, item)

    def _fill_per_subset(self) -> None:
        table = self._per_subset
        result = self._result
        if result is None:
            table.setRowCount(0)
            return
        frequencies = self._frequencies() or {}
        dphi = wrap_phase(result.phase - result.common_phase[:, None])
        n_rows = len(result.component_ids) * len(result.subset_labels)
        table.setRowCount(n_rows)
        r = 0
        for k, cid in enumerate(result.component_ids):
            f = frequencies.get(cid)
            for s, label in enumerate(result.subset_labels):
                err_phi = float(result.phase_error[k, s])
                oc = (
                    _fmt_pm(
                        float(o_minus_c(dphi[k, s], f)),
                        err_phi / (2.0 * np.pi * f) * _SECONDS_PER_DAY,
                        ".4g",
                    )
                    if f is not None
                    else _DASH
                )
                values = (
                    f"F{cid}",
                    label,
                    _fmt_pm(float(result.mid_time[s]), math.nan, ".5f"),
                    str(int(result.n_points[s])),
                    _fmt_pm(
                        float(result.amplitude[k, s]),
                        float(result.amplitude_error[k, s]),
                    ),
                    _fmt_pm(float(dphi[k, s]), err_phi, ".3g"),
                    oc,
                )
                for c, text in enumerate(values):
                    table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
                r += 1

    # -- plots -------------------------------------------------------------------
    def _x(self) -> tuple[np.ndarray, list[tuple[float, str]] | None, str]:
        """x of each subset, tick labels when the axis is categorical, its label.

        Times are drawn from a whole-day epoch: on a Julian-date axis the tick labels
        would otherwise read ``2.45801e+06`` and all look alike.
        """
        result = self._result
        assert result is not None
        mid = np.asarray(result.mid_time, dtype=np.float64)
        ordered = mid.size < 2 or bool(np.all(np.diff(mid) > 0.0))
        if result.by != "band" and ordered and mid.size and np.all(np.isfinite(mid)):
            epoch = math.floor(float(mid[0]))
            label = "Mid-time (d)" if epoch == 0 else f"Mid-time − {epoch} (d)"
            return mid - epoch, None, label
        positions = np.arange(len(result.subset_labels), dtype=np.float64)
        step = max(1, len(result.subset_labels) // 24)
        ticks = [
            (float(i), label)
            for i, label in enumerate(result.subset_labels)
            if i % step == 0
        ]
        return positions, ticks, f"Subset ({result.by})"

    def _redraw(self, *_args: object) -> None:
        for plot in (self._amp_plot, self._phase_plot):
            plot.clear()
        self._legend.clear()
        style = plot_label_style(self._pal)
        result = self._result
        oc = self._lower.currentData() == "oc" and self._frequencies() is not None
        self._phase_plot.setLabel("left", "O−C (s)" if oc else "Δφ (rad)", **style)
        if result is None:
            self._phase_plot.setLabel("bottom", "Mid-time (d)", **style)
            for plot in (self._amp_plot, self._phase_plot):
                plot.getAxis("bottom").setTicks(None)
            return
        x, ticks, x_label = self._x()
        bottom = self._phase_plot.getAxis("bottom")
        top_axis = self._amp_plot.getAxis("bottom")
        bottom.setTicks(None if ticks is None else [ticks])
        top_axis.setTicks(None if ticks is None else [ticks])
        self._phase_plot.setLabel("bottom", x_label, **style)
        frequencies = self._frequencies() or {}
        dphi = wrap_phase(result.phase - result.common_phase[:, None])
        width = stroke_width(self)
        amp_varies = result.mode in ("both", "amplitude")
        phase_varies = result.mode in ("both", "phase")
        for k, cid in enumerate(result.component_ids):
            colour = self._pal.band_color(k)
            symbol = self._pal.band_symbol(k)
            pen = pg.mkPen(colour, width=width)
            name = f"F{cid}"
            if amp_varies:
                self._series(
                    self._amp_plot,
                    x,
                    np.asarray(result.amplitude[k], dtype=np.float64),
                    np.asarray(result.amplitude_error[k], dtype=np.float64),
                    pen,
                    symbol,
                    name,
                )
            common = float(result.common_amplitude[k])
            if math.isfinite(common):
                err = self._common_error(cid, result.amplitude_error[k])
                if math.isfinite(err) and err > 0.0:
                    band = pg.LinearRegionItem(
                        values=(common - err, common + err),
                        orientation="horizontal",
                        movable=False,
                        brush=self._pal.band_color(k, 40),
                        pen=pg.mkPen(None),
                    )
                    band.setZValue(-10)
                    self._amp_plot.addItem(band)
                line = pg.InfiniteLine(
                    pos=common,
                    angle=0,
                    pen=pg.mkPen(colour, width=width, style=Qt.PenStyle.DashLine),
                )
                self._amp_plot.addItem(line)
            if phase_varies:
                y = dphi[k]
                e = np.asarray(result.phase_error[k], dtype=np.float64)
                f = frequencies.get(cid)
                if oc and f is not None:
                    y = o_minus_c(y, f)
                    e = e / (2.0 * np.pi * f) * _SECONDS_PER_DAY
                self._series(self._phase_plot, x, y, e, pen, symbol, None)
                self._phase_plot.addItem(
                    pg.InfiniteLine(
                        pos=0.0,
                        angle=0,
                        pen=pg.mkPen(colour, width=width, style=Qt.PenStyle.DashLine),
                    )
                )
        if not amp_varies:
            self._note(self._amp_plot, "Amplitudes held common (φ only)")
        if not phase_varies:
            self._note(self._phase_plot, "Phases held common (A only)")
        for plot in (self._amp_plot, self._phase_plot):
            plot.enableAutoRange()

    def _common_error(self, cid: int, per_subset: Any) -> float:
        """1-sigma of the common amplitude: the session's own error when it has one
        (a full view), else what a constant amplitude would have from the per-subset
        errors (their inverse-variance combination)."""
        row = self._view.row(cid) if self._view is not None else None
        if row is not None and math.isfinite(row.amplitude_error):
            return float(row.amplitude_error)
        errors = np.asarray(per_subset, dtype=np.float64)
        good = errors[np.isfinite(errors) & (errors > 0.0)]
        if not good.size:
            return math.nan
        return float(1.0 / math.sqrt(float(np.sum(1.0 / (good * good)))))

    def _series(
        self,
        plot: Any,
        x: np.ndarray,
        y: np.ndarray,
        err: np.ndarray,
        pen: Any,
        symbol: str,
        name: str | None,
    ) -> None:
        good = np.isfinite(x) & np.isfinite(y)
        if not good.any():
            return
        e = np.where(np.isfinite(err), err, 0.0)[good]
        plot.addItem(
            pg.ErrorBarItem(x=x[good], y=y[good], top=e, bottom=e, beam=0.0, pen=pen)
        )
        scatter = pg.ScatterPlotItem(
            x=x[good],
            y=y[good],
            symbol=symbol,
            size=8,
            pen=pen,
            brush=pen.color(),
            name=name,
        )
        # A named item joins the plot's legend by itself.
        plot.addItem(scatter)

    def _note(self, plot: Any, text: str) -> None:
        item = pg.TextItem(text, color=self._pal.muted, anchor=(0.5, 0.5))
        plot.addItem(item)
        view_range = plot.viewRange()
        item.setPos(
            0.5 * (view_range[0][0] + view_range[0][1]),
            0.5 * (view_range[1][0] + view_range[1][1]),
        )


__all__ = [
    "P_FLAG",
    "SUBSET_COLUMNS",
    "TEST_COLUMNS",
    "VariationPanel",
    "VariationQuery",
    "o_minus_c",
    "parse_components",
    "wrap_phase",
]
