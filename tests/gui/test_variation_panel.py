"""The Variations dock (spec 4.8): query, plots, tests table and Apply to model.

The views and one result come from a real session on ``backend="numpy"`` (a nightly
curve whose amplitude grows night by night); the rest use fabricated results, so the
plots and tables can be checked against known numbers.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

import cuperiod as cup
from cuperiod.core.config import PreWhitenSettings
from cuperiod.gui.qt import Qt, pg
from cuperiod.gui.theme import palette
from cuperiod.gui.viewmodels import SessionKey, SessionView, build_session_view
from cuperiod.gui.widgets.variation_panel import (
    P_FLAG,
    SUBSET_COLUMNS,
    TEST_COLUMNS,
    VariationPanel,
    VariationQuery,
    o_minus_c,
    parse_components,
    wrap_phase,
)
from cuperiod.prewhiten.session.state import VariationResult
from cuperiod.prewhiten.subsets import Partition

N_NIGHTS = 6
PER_NIGHT = 30
FREQ = 3.7

Live = tuple[cup.PreWhitenSession, SessionView]


def _nightly() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    t = np.concatenate(
        [
            2458000.0 + 2.0 * k + np.sort(rng.uniform(0.0, 0.25, PER_NIGHT))
            for k in range(N_NIGHTS)
        ]
    )
    night = np.repeat(np.arange(N_NIGHTS), PER_NIGHT)
    amplitude = (0.05 + 0.01 * np.arange(N_NIGHTS))[night]
    y = amplitude * np.sin(2 * np.pi * FREQ * (t - 2458000.0)) + rng.normal(
        0.0, 0.003, t.size
    )
    return t, y, np.full(t.size, 0.003)


@pytest.fixture(scope="module")
def live() -> Live:
    session = cup.PreWhitenSession(
        _nightly(),
        backend="numpy",
        settings=PreWhitenSettings(maximum_frequency=10.0),
    )
    session.step()
    session.define_partition(
        Partition.from_gaps(session.data.time, min_gap=0.3, name="night")
    )
    return session, _view(session)


def _view(session: cup.PreWhitenSession, key: str = "star") -> SessionView:
    return build_session_view(
        session, key=SessionKey(key), id_epoch=3, threshold=False
    )


def _result(
    *,
    ids: tuple[int, ...] = (1,),
    mode: str = "both",
    by: str = "night",
    labels: tuple[str, ...] = ("N1", "N2", "N3"),
    mid: tuple[float, ...] = (2458000.1, 2458002.1, 2458004.1),
    amplitude_p: float = 1e-6,
    phase_p: float = 0.5,
) -> VariationResult:
    k, s = len(ids), len(labels)
    return VariationResult(
        component_ids=ids,
        mode=mode,  # type: ignore[arg-type]
        by=by,
        subset_labels=labels,
        mid_time=np.asarray(mid, dtype=np.float64),
        n_points=np.full(s, PER_NIGHT),
        amplitude=np.tile(np.linspace(0.05, 0.07, s), (k, 1)),
        amplitude_error=np.full((k, s), 0.002),
        phase=np.tile(np.array([0.1, 0.3, -0.2])[:s], (k, 1)),
        phase_error=np.full((k, s), 0.05),
        common_amplitude=np.full(k, 0.06),
        common_phase=np.full(k, 0.1),
        amplitude_statistic=np.full(k, 25.0),
        amplitude_p_value=np.full(k, amplitude_p),
        phase_statistic=np.full(k, 1.2),
        phase_p_value=np.full(k, phase_p),
        delta_bic=-14.2,
        frequencies_improved=False,
    )


@pytest.fixture
def panel(qtbot: QtBot, live: Live) -> VariationPanel:
    widget = VariationPanel("light")
    qtbot.addWidget(widget)
    widget.set_view(live[1])
    return widget


def _record(signal: Any) -> list[Any]:
    got: list[Any] = []
    signal.connect(lambda *args: got.append(args if len(args) > 1 else args[0]))
    return got


# -- helpers --------------------------------------------------------------------------


def test_parse_components() -> None:
    assert parse_components("F1, f3;7 F1") == (1, 3, 7)
    assert parse_components("  ") == ()
    with pytest.raises(ValueError, match="not a component"):
        parse_components("F1, G2")


def test_phase_wrapping_and_o_minus_c() -> None:
    np.testing.assert_allclose(
        wrap_phase([0.0, 3 * np.pi / 2, -np.pi, np.pi]), [0, -np.pi / 2, np.pi, np.pi]
    )
    # A quarter-cycle phase lead at 1 c/d arrives a quarter day early.
    assert float(o_minus_c(np.pi / 2, 1.0)) == pytest.approx(-21600.0)


# -- the query ------------------------------------------------------------------------


def test_calculate_asks_for_the_selected_components(panel: VariationPanel) -> None:
    queries = _record(panel.query_requested)
    assert panel._by.currentData() == "night"
    panel.set_selected_components([1])
    assert panel._components.text() == "F1"
    panel._mode.setCurrentIndex(panel._mode.findData("amplitude"))
    panel._improve.setChecked(True)
    panel._calculate.click()
    assert queries == [
        VariationQuery(ids=(1,), mode="amplitude", by="night", improve_frequencies=True)
    ]
    assert panel._summary.text() == "Calculating…"


def test_empty_components_means_every_active_one(panel: VariationPanel) -> None:
    queries = _record(panel.query_requested)
    panel.calculate()
    assert queries[0].ids is None


def test_bad_components_are_reported_not_sent(panel: VariationPanel) -> None:
    queries = _record(panel.query_requested)
    messages = _record(panel.message)
    panel._components.setText("F9")
    panel.calculate()
    panel._components.setText("F1 + F2")
    panel.calculate()
    assert not queries
    assert "F9" in messages[0] and "not a component" in messages[1]


def test_typed_components_win_over_the_table_selection(panel: VariationPanel) -> None:
    panel._components.setText("F1")
    panel._components.textEdited.emit("F1")
    panel.set_selected_components([2])
    assert panel._components.text() == "F1"
    panel._components.clear()
    panel.set_selected_components([2])
    assert panel._components.text() == "F2"


def test_no_partition_no_calculate(qtbot: QtBot, live: Live) -> None:
    panel = VariationPanel()
    qtbot.addWidget(panel)
    assert not panel._calculate.isEnabled()
    panel.set_view(dataclasses.replace(live[1], partitions=(), subsets=()))
    assert not panel._calculate.isEnabled()
    assert "Subsets dock" in panel._calculate.toolTip()


# -- results --------------------------------------------------------------------------


def test_a_real_result_fills_plots_and_tables(
    panel: VariationPanel, live: Live
) -> None:
    session = live[0]
    result = session.variations(None, mode="both", by="night")
    panel.calculate()
    panel.set_result(result)
    tests = panel._tests
    assert [
        tests.horizontalHeaderItem(c).text() for c in range(len(TEST_COLUMNS))
    ] == list(TEST_COLUMNS)
    assert tests.rowCount() == len(result.component_ids)
    assert tests.item(0, 0).text() == "F1"
    assert tests.item(0, 1).text() == str(N_NIGHTS)
    # The amplitude grows night by night: reported in words, not only in a colour.
    assert "amplitude varies" in tests.item(0, 7).text()
    assert tests.item(0, 7).font().bold()
    per = panel._per_subset
    assert [per.horizontalHeaderItem(c).text() for c in range(len(SUBSET_COLUMNS))] == (
        list(SUBSET_COLUMNS)
    )
    assert per.rowCount() == len(result.component_ids) * N_NIGHTS
    assert per.item(0, 1).text() == "N1"
    assert "ΔBIC" in panel._summary.text() and "F1" in panel._summary.text()
    scatters = [
        i for i in panel._amp_plot.listDataItems() if isinstance(i, pg.ScatterPlotItem)
    ]
    assert len(scatters) == 1 and len(scatters[0].data) == N_NIGHTS
    assert panel._apply.isEnabled()


def test_the_tests_table_flags_only_significant_rows(panel: VariationPanel) -> None:
    panel.set_result(_result(ids=(1,), amplitude_p=0.5, phase_p=0.2))
    assert panel._tests.item(0, 7).text() == "consistent with constant"
    assert not panel._tests.item(0, 7).font().bold()
    panel.set_result(_result(ids=(1,), amplitude_p=0.5, phase_p=P_FLAG / 2))
    assert panel._tests.item(0, 7).text() == "phase varies"
    assert panel._tests.item(0, 6).text() == "-14.2"  # the query's ΔBIC
    assert panel._tests.item(0, 3).text() == "0.5"
    assert "-14.2" in panel._summary.text()


def test_the_common_amplitude_is_a_line_in_a_one_sigma_band(
    panel: VariationPanel,
) -> None:
    panel.set_result(_result())
    items = panel._amp_plot.getPlotItem().items
    bands = [i for i in items if isinstance(i, pg.LinearRegionItem)]
    lines = [i for i in items if isinstance(i, pg.InfiniteLine)]
    assert [line.value() for line in lines] == [pytest.approx(0.06)]
    # A light view has no error for the common value: a constant amplitude's own,
    # from the three per-subset errors of 0.002.
    sigma = 0.002 / math.sqrt(3.0)
    assert len(bands) == 1
    np.testing.assert_allclose(bands[0].getRegion(), (0.06 - sigma, 0.06 + sigma))


def test_the_lower_plot_switches_to_o_minus_c(panel: VariationPanel) -> None:
    panel.set_result(_result())
    axis = panel._phase_plot.getAxis("left")
    assert "Δφ" in axis.labelText
    phase_points = _scatter_y(panel._phase_plot)
    np.testing.assert_allclose(phase_points, [0.0, 0.2, -0.3], atol=1e-12)
    panel._lower.setCurrentIndex(panel._lower.findData("oc"))
    assert "O−C (s)" in axis.labelText
    frequency = panel._view.row(1).frequency  # type: ignore[union-attr]
    np.testing.assert_allclose(
        _scatter_y(panel._phase_plot),
        -np.array([0.0, 0.2, -0.3]) / (2 * np.pi * frequency) * 86400.0,
        rtol=1e-9,
        atol=1e-9,
    )
    per = panel._per_subset
    expected = -0.2 / (2 * np.pi * frequency) * 86400
    assert per.item(1, 6).text().startswith(f"{expected:.4g}")


def _scatter_y(plot: Any) -> np.ndarray:
    items = [i for i in plot.listDataItems() if isinstance(i, pg.ScatterPlotItem)]
    assert len(items) == 1
    return np.asarray(items[0].data["y"], dtype=np.float64)


def test_time_axes_start_from_a_whole_day(panel: VariationPanel) -> None:
    panel.set_result(_result())
    assert "Mid-time − 2458000 (d)" in panel._phase_plot.getAxis("bottom").labelText
    items = [
        i for i in panel._amp_plot.listDataItems() if isinstance(i, pg.ScatterPlotItem)
    ]
    np.testing.assert_allclose(items[0].data["x"], [0.1, 2.1, 4.1], atol=1e-6)


def test_bands_are_categories(panel: VariationPanel) -> None:
    panel.set_result(
        _result(by="band", labels=("g", "r", "i"), mid=(5.0, 2.0, 9.0))
    )
    axis = panel._phase_plot.getAxis("bottom")
    assert axis.labelText == "Subset (band)"
    assert axis._tickLevels == [[(0.0, "g"), (1.0, "r"), (2.0, "i")]]


def test_a_mode_that_holds_a_quantity_says_so(panel: VariationPanel) -> None:
    panel.set_result(_result(mode="amplitude", phase_p=math.nan))
    texts = [
        i.toPlainText()
        for i in panel._phase_plot.getPlotItem().items
        if isinstance(i, pg.TextItem)
    ]
    assert texts == ["Phases held common (A only)"]
    items = panel._phase_plot.listDataItems()
    assert not [i for i in items if isinstance(i, pg.ScatterPlotItem)]
    assert panel._tests.item(0, 5).text() == "–"


def test_components_differ_by_shape_as_well_as_colour(
    panel: VariationPanel, live: Live
) -> None:
    view = live[1]
    row = view.rows[0]
    rows = (row, dataclasses.replace(row, id=2, label="F2", frequency=5.1))
    panel.set_view(dataclasses.replace(view, rows=rows))
    panel.set_result(_result(ids=(1, 2)))
    scatters = [
        i for i in panel._amp_plot.listDataItems() if isinstance(i, pg.ScatterPlotItem)
    ]
    symbols = {s.opts["symbol"] for s in scatters}
    assert len(symbols) == 2
    legend = [label.text for _, label in panel._legend.items]
    assert legend == ["F1", "F2"]


# -- applying -------------------------------------------------------------------------


def test_apply_switches_the_partition_and_varies_in_one_action(
    panel: VariationPanel, live: Live
) -> None:
    assert live[1].variation_by is None
    groups = _record(panel.group_requested)
    panel.calculate()
    panel.set_result(_result())
    panel._apply.click()
    assert groups == [
        {
            "actions": (
                ("use_variation", {"name": "night"}),
                ("set_variation", {"ids": [1], "mode": "both", "hold_frequency": True}),
            ),
            "expect_id_epoch": 3,
        }
    ]


def test_apply_on_the_variation_partition_is_one_id_addressed_edit(
    panel: VariationPanel, live: Live
) -> None:
    panel.set_view(dataclasses.replace(live[1], variation_by="night"))
    edits = _record(panel.edit_requested)
    result = dataclasses.replace(_result(mode="phase"), frequencies_improved=True)
    panel.calculate()
    panel.set_result(result)
    panel.apply_to_model()
    assert edits == [
        (
            "set_variation",
            {
                "ids": [1],
                "mode": "phase",
                "hold_frequency": False,
                "expect_id_epoch": 3,
            },
        )
    ]


def test_apply_waits_for_a_matching_session(panel: VariationPanel, live: Live) -> None:
    panel.calculate()
    panel.set_result(_result(ids=(1, 7)))  # F7 is not in the session
    assert not panel._apply.isEnabled()
    panel.set_result(_result())
    assert panel._apply.isEnabled()
    panel.set_busy(True)
    assert not panel._apply.isEnabled()
    panel.set_busy(False)
    # Another session: the result (and the typed components) no longer apply.
    panel._components.setText("F1")
    panel.set_view(_view(live[0], key="other"))
    assert panel.result is None and not panel._apply.isEnabled()
    assert panel._components.text() == ""


def test_theme_switch_keeps_the_result(panel: VariationPanel) -> None:
    panel.set_result(_result())
    panel.apply_theme(palette("dark"))
    assert panel.result is not None
    assert _scatter_y(panel._phase_plot).size == 3


def test_every_control_is_named_and_explained(panel: VariationPanel) -> None:
    controls = [
        panel._components,
        panel._mode,
        panel._by,
        panel._improve,
        panel._calculate,
        panel._apply,
        panel._lower,
        panel._amp_plot,
        panel._phase_plot,
        panel._tests,
        panel._per_subset,
    ]
    for widget in controls:
        assert widget.accessibleName(), widget
    for widget in controls[:7]:
        assert widget.toolTip(), widget
        assert widget.focusPolicy() & Qt.FocusPolicy.TabFocus, widget
