"""The Subsets dock (spec 4.6): partitions, roles, the tri-state use box, requests.

Views come from real sessions on ``backend="numpy"`` (a nightly curve with a zero
point per night, three points deleted in the first night and the sixth night left
out). The dock never touches a session: every test checks what it *asks* for.
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
from cuperiod.gui.qt import Qt, QtCore, QtWidgets
from cuperiod.gui.viewmodels import SessionKey, SessionView, build_session_view
from cuperiod.gui.widgets.subsets_panel import (
    COLUMNS,
    PartitionRequest,
    SubsetsPanel,
    rms_weights,
)
from cuperiod.prewhiten.subsets import (
    ChunksOrigin,
    CrossedOrigin,
    GapsOrigin,
    LabelsOrigin,
    RangesOrigin,
)

N_NIGHTS = 8
PER_NIGHT = 30

Nightly = tuple[cup.PreWhitenSession, SessionView]


def _nightly() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    t = np.concatenate(
        [
            2458000.0 + 2.0 * k + np.sort(rng.uniform(0.0, 0.25, PER_NIGHT))
            for k in range(N_NIGHTS)
        ]
    )
    night = np.repeat(np.arange(N_NIGHTS), PER_NIGHT)
    y = (
        12.0
        + rng.normal(0.0, 0.02, N_NIGHTS)[night]
        + 0.06 * np.sin(2 * np.pi * 3.7 * (t - 2458000.0))
        + rng.normal(0.0, 0.004, t.size)
    )
    return t, y, np.full(t.size, 0.004)


def _session(**kwargs: Any) -> cup.PreWhitenSession:
    return cup.PreWhitenSession(
        _nightly(),
        backend="numpy",
        settings=PreWhitenSettings(maximum_frequency=10.0),
        **kwargs,
    )


def _view(session: cup.PreWhitenSession, key: str = "star") -> SessionView:
    return build_session_view(
        session, key=SessionKey(key), id_epoch=0, threshold=False
    )


def _define(session: cup.PreWhitenSession, request: PartitionRequest) -> None:
    partition = request.resolve(
        session.data.time,
        partitions={p.name: p for p in session.state.partitions},
        labels=session.data.labels,
    )
    session.define_partition(partition)


@pytest.fixture(scope="module")
def nightly() -> Nightly:
    session = _session()
    session.step()
    _define(
        session, PartitionRequest("night", origin=GapsOrigin(min_gap=0.3, prefix="N"))
    )
    session.do("use_offsets", name="night")
    session.do("delete_points", runs=[[3, 6]])
    session.do("exclude_subsets", name="night", subsets=[5], excluded=True)
    return session, _view(session)


@pytest.fixture
def panel(qtbot: QtBot, nightly: Nightly) -> SubsetsPanel:
    widget = SubsetsPanel()
    qtbot.addWidget(widget)
    widget.set_view(nightly[1])
    return widget


def _record(signal: Any) -> list[Any]:
    got: list[Any] = []
    signal.connect(lambda *args: got.append(args if len(args) > 1 else args[0]))
    return got


def _use(panel: SubsetsPanel, row: int) -> QtWidgets.QTableWidgetItem:
    item = panel._table.item(row, 0)
    assert item is not None
    return item


# -- the table ------------------------------------------------------------------------


def test_the_table_lists_the_partition(panel: SubsetsPanel) -> None:
    table = panel._table
    assert [table.horizontalHeaderItem(c).text() for c in range(len(COLUMNS))] == list(
        COLUMNS
    )
    assert table.rowCount() == N_NIGHTS
    assert panel.current_partition() == "night"
    assert table.item(0, 1).text() == "N1"
    assert table.item(0, 2).text() == f"{PER_NIGHT - 3} of {PER_NIGHT}"
    assert table.item(1, 2).text() == str(PER_NIGHT)
    assert table.item(0, 3).text().startswith("2458000.")
    assert table.item(1, 5).text() != "–"  # a fitted zero point
    assert table.item(5, 5).text() == "–"  # the night left out has none
    assert panel._offsets.isChecked() and not panel._weights.isChecked()
    assert "Zero points: night" in panel._roles_summary.text()


def test_the_use_box_is_tri_state_and_spelled_out(panel: SubsetsPanel) -> None:
    assert _use(panel, 0).checkState() == Qt.CheckState.PartiallyChecked
    assert "3 of 30 points deleted" in _use(panel, 0).toolTip()
    assert _use(panel, 0).text() == "partly"
    assert _use(panel, 1).checkState() == Qt.CheckState.Checked
    assert _use(panel, 1).text() == "yes"
    assert _use(panel, 5).checkState() == Qt.CheckState.Unchecked
    assert _use(panel, 5).text() == "no"
    assert _use(panel, 5).data(Qt.ItemDataRole.AccessibleTextRole) == (
        "left out of the fit"
    )


@pytest.mark.parametrize(
    ("row", "clicked", "excluded", "shown"),
    [
        # A partly used subset goes *out* (Qt's own cycle would turn it on) ...
        (0, Qt.CheckState.Checked, True, Qt.CheckState.Unchecked),
        (1, Qt.CheckState.Unchecked, True, Qt.CheckState.Unchecked),
        # ... and a left-out one comes back, keeping its deleted points deleted.
        (5, Qt.CheckState.Checked, False, Qt.CheckState.Checked),
    ],
)
def test_the_use_box_only_ever_excludes_subsets(
    panel: SubsetsPanel,
    row: int,
    clicked: Qt.CheckState,
    excluded: bool,
    shown: Qt.CheckState,
) -> None:
    edits = _record(panel.edit_requested)
    requests = _record(panel.partition_requested)
    _use(panel, row).setCheckState(clicked)  # what Qt does on a click
    assert edits == [
        ("exclude_subsets", {"name": "night", "subsets": [row], "excluded": excluded})
    ]
    assert not requests
    assert _use(panel, row).checkState() == shown


def test_space_toggles_the_use_box(qtbot: QtBot, panel: SubsetsPanel) -> None:
    edits = _record(panel.edit_requested)
    table = panel._table
    table.setFocus()
    table.setCurrentCell(2, 0)
    qtbot.keyClick(table, Qt.Key.Key_Space)
    assert edits and edits[0][0] == "exclude_subsets"
    assert edits[0][1]["subsets"] == [2] and edits[0][1]["excluded"] is True


def test_the_use_box_follows_a_multi_row_selection(panel: SubsetsPanel) -> None:
    edits = _record(panel.edit_requested)
    table = panel._table
    table.selectRow(1)
    table.selectionModel().select(
        table.model().index(3, 0),
        QtCore.QItemSelectionModel.SelectionFlag.Select
        | QtCore.QItemSelectionModel.SelectionFlag.Rows,
    )
    _use(panel, 1).setCheckState(Qt.CheckState.Unchecked)
    assert edits == [
        ("exclude_subsets", {"name": "night", "subsets": [1, 3], "excluded": True})
    ]


def test_renaming_a_subset_is_a_partition_edit(panel: SubsetsPanel) -> None:
    requests = _record(panel.partition_requested)
    messages = _record(panel.message)
    panel._table.item(1, 1).setText("N2")  # unchanged: nothing asked
    panel._table.item(1, 1).setText("N3")  # a name taken by another subset
    assert not requests and "already called" in messages[-1]
    assert panel._table.item(1, 1).text() == "N2"  # reverted
    panel._table.item(1, 1).setText("good night")
    assert requests == [
        PartitionRequest(
            name="night", edit="rename", subsets=(1,), label="good night"
        )
    ]


def test_editing_a_weight_sets_the_partitions_weights(panel: SubsetsPanel) -> None:
    edits = _record(panel.edit_requested)
    messages = _record(panel.message)
    panel._table.item(2, 6).setText("-1")
    assert not edits and "positive number" in messages[-1]
    assert panel._table.item(2, 6).text() == "1"
    panel._table.item(2, 6).setText("2.5")
    weights = [1.0] * N_NIGHTS
    weights[2] = 2.5
    assert edits == [("set_subset_weights", {"name": "night", "weights": weights})]


def test_weights_from_the_residual_scatter_switch_weighting_on(
    panel: SubsetsPanel, nightly: Nightly
) -> None:
    groups = _record(panel.group_requested)
    panel._rms_weights.click()
    rms = [s.rms for s in nightly[1].subsets if s.partition == "night"]
    expected = list(rms_weights(rms))
    assert groups == [
        {
            "actions": (
                ("set_subset_weights", {"name": "night", "weights": expected}),
                ("use_weights", {"name": "night"}),
            ),
            "expect_id_epoch": None,
        }
    ]
    finite = [w for w, r in zip(expected, rms, strict=True) if math.isfinite(r)]
    assert np.mean(finite) == pytest.approx(1.0)
    assert expected[5] == 1.0  # the night without points in use


def test_rms_weights_matches_the_sessions_query(
    nightly: Nightly,
) -> None:
    session, view = nightly
    rms = [s.rms for s in view.subsets if s.partition == "night"]
    np.testing.assert_allclose(
        rms_weights(rms), session.subset_rms_weights("night"), rtol=1e-12
    )


def test_equal_weights(panel: SubsetsPanel) -> None:
    edits = _record(panel.edit_requested)
    panel._equal.click()
    assert edits == [
        ("set_subset_weights", {"name": "night", "weights": [1.0] * N_NIGHTS})
    ]


# -- roles ----------------------------------------------------------------------------


def test_role_boxes_name_the_shown_partition(panel: SubsetsPanel) -> None:
    edits = _record(panel.edit_requested)
    panel._offsets.click()  # zero points per night -> one zero point
    panel._weights.click()
    panel._variation.click()
    assert edits == [
        ("use_offsets", {"name": None}),
        ("use_weights", {"name": "night"}),
        ("use_variation", {"name": "night"}),
    ]


def test_remove_drops_the_partition(panel: SubsetsPanel) -> None:
    edits = _record(panel.edit_requested)
    panel._remove.click()
    assert edits == [("drop_partition", {"name": "night"})]


# -- deriving and editing partitions --------------------------------------------------


def _preset(panel: SubsetsPanel, key: str) -> None:
    panel._preset.setCurrentIndex(panel._preset.findData(key))


def test_split_presets_become_recipes(panel: SubsetsPanel) -> None:
    requests = _record(panel.partition_requested)
    _preset(panel, "nights")
    panel._param.setValue(0.5)
    panel._apply.click()
    _preset(panel, "chunks")
    assert panel._name.text() == "chunk"
    panel._apply.click()
    _preset(panel, "sectors")
    panel._apply.click()
    _preset(panel, "ranges")
    panel._edges.setText("2458004.5, 2458009")
    panel._apply.click()
    assert requests == [
        PartitionRequest("night", origin=GapsOrigin(min_gap=0.5, prefix="N")),
        PartitionRequest("chunk", origin=ChunksOrigin(length=10.0, prefix="C")),
        PartitionRequest(
            "sector", origin=GapsOrigin(min_gap=0.5, max_span=28.5, prefix="S")
        ),
        PartitionRequest("range", origin=RangesOrigin(edges=[2458004.5, 2458009.0])),
    ]


def test_bad_split_input_is_reported_not_sent(panel: SubsetsPanel) -> None:
    requests = _record(panel.partition_requested)
    messages = _record(panel.message)
    _preset(panel, "ranges")
    panel._edges.setText("soon")
    panel._apply.click()
    _preset(panel, "column")  # no label columns: nothing to split by
    panel._apply.click()
    assert not requests and len(messages) == 2


def test_label_columns_enable_the_column_presets(panel: SubsetsPanel) -> None:
    model = panel._preset.model()
    row = panel._preset.findData("column")
    assert not model.item(row).isEnabled()
    panel.set_label_columns(["band", "site"])
    assert model.item(row).isEnabled()
    assert model.item(panel._preset.findData("bands")).isEnabled()
    requests = _record(panel.partition_requested)
    _preset(panel, "column")
    panel._column.setCurrentIndex(panel._column.findData("site"))
    panel._name.setText("")
    panel._apply.click()
    assert requests == [PartitionRequest("site", origin=LabelsOrigin(column="site"))]


def test_merge_and_split_at_ask_for_edits(panel: SubsetsPanel) -> None:
    requests = _record(panel.partition_requested)
    assert not panel._merge.isEnabled()
    table = panel._table
    table.selectRow(1)
    assert panel._split_time.value() == pytest.approx(
        0.5 * (panel._rows[1].t_start + panel._rows[1].t_stop)
    )
    for row in (2, 3):
        table.selectionModel().select(
            table.model().index(row, 0),
            QtCore.QItemSelectionModel.SelectionFlag.Select
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
    assert panel._merge.isEnabled()
    panel._merge.click()
    panel.set_time_cursor(2458004.1)
    panel._split.click()
    assert requests == [
        PartitionRequest("night", edit="merge", subsets=(1, 2, 3)),
        PartitionRequest("night", edit="split_at", time=pytest.approx(2458004.1)),
    ]


def test_requests_resolve_to_the_partitions_they_name(
    nightly: Nightly,
) -> None:
    session = nightly[0]
    partitions = {p.name: p for p in session.state.partitions}
    time = session.data.time

    def resolve(request: PartitionRequest) -> Any:
        return request.resolve(time, partitions=partitions, labels=session.data.labels)

    merged = resolve(PartitionRequest("night", edit="merge", subsets=(1, 2, 3)))
    assert merged.n_subsets == N_NIGHTS - 2
    assert merged.labels[1] == "N2+N3+N4"
    cut = float(np.median(time[PER_NIGHT : 2 * PER_NIGHT]))
    split = resolve(PartitionRequest("night", edit="split_at", time=cut))
    assert split.n_subsets == N_NIGHTS + 1 and split.labels[1:3] == ("N2a", "N2b")
    renamed = resolve(PartitionRequest("night", edit="rename", subsets=(0,), label="x"))
    assert renamed.labels[0] == "x"
    nights = resolve(PartitionRequest("n2", origin=GapsOrigin(min_gap=0.3)))
    assert nights.name == "n2" and nights.n_subsets == N_NIGHTS
    crossed = resolve(
        PartitionRequest("nx", origin=CrossedOrigin(a="night", b="night"))
    )
    assert crossed.n_subsets == N_NIGHTS
    with pytest.raises(ValueError, match="no partition"):
        resolve(PartitionRequest("season", edit="merge", subsets=(0, 1)))
    with pytest.raises(ValueError, match="no subset"):
        resolve(PartitionRequest("night", edit="split_at", time=0.0))


def test_cross_with_lists_the_other_partitions(
    qtbot: QtBot, nightly: Nightly
) -> None:
    session = _session()
    for name, gap in (("night", 0.3), ("week", 3.0)):
        _define(session, PartitionRequest(name, origin=GapsOrigin(min_gap=gap)))
    panel = SubsetsPanel()
    qtbot.addWidget(panel)
    panel.set_view(_view(session))
    assert panel.current_partition() == "night"
    assert panel._cross.isEnabled()
    actions = panel._cross_menu.actions()
    assert [a.text() for a in actions] == ["week"]
    requests = _record(panel.partition_requested)
    actions[0].trigger()
    assert requests == [
        PartitionRequest("night_x_week", origin=CrossedOrigin(a="night", b="week"))
    ]
    # Showing the other partition keeps it across views.
    panel._partition_combo.setCurrentIndex(1)
    panel.set_view(_view(session))
    assert panel.current_partition() == "week"


# -- hint, warnings, highlight ------------------------------------------------------


def test_the_hint_offers_the_obvious_split(qtbot: QtBot) -> None:
    panel = SubsetsPanel()
    qtbot.addWidget(panel)
    panel.set_view(_view(_session(), key="fresh"))
    assert panel._hint.isVisibleTo(panel)
    assert f"nightly data ({N_NIGHTS} nights)" in panel._hint.text()
    requests = _record(panel.partition_requested)
    panel._hint_button.click()
    assert requests == [PartitionRequest("night", origin=GapsOrigin(min_gap=0.3))]


def test_the_hint_is_gone_once_a_partition_exists(panel: SubsetsPanel) -> None:
    assert not panel._hint.isVisibleTo(panel)


def test_warnings_name_low_frequency_and_short_subset_rows(
    panel: SubsetsPanel, nightly: Nightly
) -> None:
    view = nightly[1]
    assert not panel._warning.isVisibleTo(panel)
    row = view.rows[0]
    rows = (
        dataclasses.replace(row, flags=row.flags | {"low_frequency"}),
        dataclasses.replace(row, id=9, label="F9", flags=frozenset({"short_subset"})),
    )
    panel.set_view(dataclasses.replace(view, rows=rows, warnings=("check N6",)))
    text = panel._warning.text()
    assert panel._warning.isVisibleTo(panel)
    assert "F1" in text and "zero points can absorb" in text
    assert "F9" in text and "less than one cycle" in text
    assert "check N6" in text
    assert text.startswith("Warning:")  # said in words, not by colour


def test_selecting_rows_highlights_their_points(
    panel: SubsetsPanel, nightly: Nightly
) -> None:
    masks = _record(panel.points_highlighted)
    panel._table.selectRow(2)
    mask = masks[-1]
    assert mask is not None and mask.shape == nightly[1].time.shape
    expected = np.zeros(N_NIGHTS * PER_NIGHT, dtype=bool)
    expected[2 * PER_NIGHT : 3 * PER_NIGHT] = True
    np.testing.assert_array_equal(mask, expected)
    panel._table.clearSelection()
    assert masks[-1] is None


def test_time_spans_highlight_when_the_view_lacks_the_codes(
    panel: SubsetsPanel, nightly: Nightly
) -> None:
    panel.set_view(dataclasses.replace(nightly[1], point_codes=None))
    masks = _record(panel.points_highlighted)
    panel._table.selectRow(1)
    assert int(masks[-1].sum()) == PER_NIGHT


# -- states and accessibility ---------------------------------------------------------


def test_no_session_means_a_disabled_dock(qtbot: QtBot) -> None:
    panel = SubsetsPanel()
    qtbot.addWidget(panel)
    assert not panel._apply.isEnabled() and not panel._offsets.isEnabled()
    assert panel._empty_hint.isVisibleTo(panel)


def test_busy_pauses_every_edit(panel: SubsetsPanel) -> None:
    panel.set_busy(True)
    for widget in (panel._apply, panel._offsets, panel._remove, panel._equal):
        assert not widget.isEnabled()
    assert (
        panel._table.editTriggers()
        == QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
    )
    edits = _record(panel.edit_requested)
    _use(panel, 1).setCheckState(Qt.CheckState.Unchecked)
    assert not edits
    panel.set_busy(False)
    assert panel._apply.isEnabled()


def test_fit_mean_off_disables_zero_points(
    panel: SubsetsPanel, nightly: Nightly
) -> None:
    view = nightly[1]
    assert view.settings is not None
    settings = view.settings.model_copy(update={"fit_mean": False})
    panel.set_view(dataclasses.replace(view, settings=settings))
    assert not panel._offsets.isEnabled()
    assert "free mean" in panel._offsets.toolTip()


def test_every_control_is_named_explained_and_reachable(panel: SubsetsPanel) -> None:
    controls: list[QtWidgets.QWidget] = [
        panel._preset,
        panel._param,
        panel._column,
        panel._edges,
        panel._name,
        panel._apply,
        panel._partition_combo,
        panel._remove,
        panel._offsets,
        panel._weights,
        panel._variation,
        panel._hint_button,
        panel._table,
        panel._rms_weights,
        panel._equal,
        panel._merge,
        panel._split,
        panel._split_time,
        panel._cross,
    ]
    for widget in controls:
        assert widget.accessibleName(), widget
        assert widget.toolTip(), widget
        assert widget.focusPolicy() & Qt.FocusPolicy.TabFocus, widget
    # Tab walks the dock in reading order: split row, partition, roles, table, actions.
    chain: list[QtWidgets.QWidget] = []
    widget = panel._preset
    for _ in range(60):
        chain.append(widget)
        widget = widget.nextInFocusChain()
        if widget is panel._preset:
            break
    order = [chain.index(w) for w in (panel._preset, panel._apply, panel._offsets)]
    order += [chain.index(w) for w in (panel._table, panel._equal)]
    assert order == sorted(order)
