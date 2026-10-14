"""The frozen view models and their worker-side builder (no Qt involved)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.gui.viewmodels import (
    CandidateView,
    ComponentView,
    MarkerState,
    SessionKey,
    SessionView,
    build_session_view,
    candidate_view,
    component_flags,
    component_views,
    history_items,
    marker_states,
)
from cuperiod.prewhiten.session import report
from cuperiod.prewhiten.session.state import Candidate, Component, Tie


def _data(n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 40.0, n))
    y = (
        0.3 * np.sin(2 * np.pi * 1.3 * t)
        + 0.1 * np.sin(2 * np.pi * 2.71 * t + 1.0)
        + rng.normal(0.0, 0.01, n)
    )
    return t, y, np.full(n, 0.01)


@pytest.fixture
def session() -> cup.PreWhitenSession:
    return cup.PreWhitenSession(_data(), backend="numpy")


def test_session_key_is_hashable_value() -> None:
    assert SessionKey("a", "V") == SessionKey("a", "V")
    assert len({SessionKey("a"), SessionKey("a", ""), SessionKey("a", "B")}) == 2


def test_component_flags_say_what_colour_cannot() -> None:
    c = Component(id=3, frequency=1.0, amplitude=-0.2, phase=0.1, fix_phase=True)
    assert component_flags(c) == {"fixed_phi", "negative_amplitude"}
    tied = dataclasses.replace(
        c, amplitude=0.2, fix_frequency=True, tie=Tie((1, 2), (1, 1)), active=False
    )
    assert component_flags(tied, ("at_bound",)) == {
        "fixed_phi", "tied", "inactive", "at_bound"
    }


def test_marker_shapes_and_selection() -> None:
    def row(i: int, **kw: object) -> ComponentView:
        base = dict(
            id=i, label=f"F{i}", rank=i, frequency=float(i), frequency_error=0.0,
            period=1.0 / i, amplitude=1.0, amplitude_error=0.0, phase=0.0,
            phase_error=0.0, amplitude_rms=None, snr=10.0, fap=0.0, delta_bic=0.0,
            spectrum_amplitude=1.0, amplitude_ratio=1.0, blended=False, active=True,
            fix_frequency=False, fix_amplitude=False, fix_phase=False, tie=None,
            variation="none", combination=None, origin="auto", flags=frozenset(),
            note="", passes=True,
        )
        base.update(kw)
        return ComponentView(**base)  # type: ignore[arg-type]

    rows = [
        row(1),
        row(2, fix_frequency=True),
        row(3, tie="F1 + F2", fix_frequency=True),
        row(4, active=False, rank=None),
    ]
    markers = marker_states(rows, selected=[3])
    assert [m.shape for m in markers] == ["free", "fixed", "tied", "inactive"]
    assert [m.selected for m in markers] == [False, False, True, False]
    assert markers[0] == MarkerState(1, 1.0, "free", False)


def test_candidate_view_hint() -> None:
    assert candidate_view(None) is None
    view = candidate_view(Candidate(2.0, 10, 0.5, 6.25, 1e-5))
    assert view == CandidateView(2.0, 0.5, 0.5, 6.25, view.hint)  # type: ignore[union-attr]
    assert view is not None and "S/N" in view.hint and "F7" in view.hint
    nan = candidate_view(Candidate(2.0, 10, 0.5, math.nan, 1e-5))
    assert nan is not None and "–" in nan.hint


def test_fresh_session_view(session: cup.PreWhitenSession) -> None:
    view = build_session_view(session, key=SessionKey("s"), id_epoch=0)
    assert isinstance(view, SessionView)
    assert view.light and view.result is None
    assert view.rows == () and view.history == () and view.cursor == 0
    assert view.pristine and not view.can_undo and not view.can_redo
    assert view.candidate is not None and abs(view.candidate.frequency - 1.3) < 0.01
    n = session.data.n
    for arr in (view.time, view.value, view.model, view.residuals, view.deleted):
        assert arr.shape == (n,)
    assert not view.deleted.any()
    assert view.threshold is not None
    assert view.threshold.shape == view.residual_spectrum.frequency.shape
    assert view.backend == "numpy" and view.settings is session.settings
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.light = False  # type: ignore[misc]


def test_rows_match_state_and_report(session: cup.PreWhitenSession) -> None:
    session.step()
    session.step()
    state = session.state
    rows = report.rows(session.kernel, state, session.kernel.derive(state))
    views = component_views(state, rows)
    assert [v.label for v in views] == ["F1", "F2"]
    assert [v.rank for v in views] == [1, 2]
    for v, c, r in zip(views, state.components, rows, strict=True):
        assert v.frequency == c.frequency and v.amplitude == c.amplitude
        assert v.phase == c.phase and v.snr == r.snr and v.passes == r.passes
        assert v.period == pytest.approx(1.0 / c.frequency)
        assert math.isnan(v.frequency_error)  # light: no covariance at hand


def test_light_result_only_when_frequencies_move(
    session: cup.PreWhitenSession,
) -> None:
    key = SessionKey("s")
    first = build_session_view(session, key=key, id_epoch=0)
    session.step()
    moved = build_session_view(
        session, key=key, id_epoch=0, previous_frequencies=first.active_frequencies
    )
    assert moved.light and moved.result is not None
    assert len(moved.result.components) == 1
    same = build_session_view(
        session, key=key, id_epoch=0, previous_frequencies=moved.active_frequencies
    )
    assert same.result is None
    progress = build_session_view(
        session, key=key, id_epoch=0, progress=True,
        previous_frequencies=first.active_frequencies,
    )
    assert progress.result is None and progress.candidate is None


def test_model_and_residuals_are_consistent(session: cup.PreWhitenSession) -> None:
    session.step()
    session.step()
    view = build_session_view(session, key=SessionKey("s"), id_epoch=0)
    np.testing.assert_allclose(view.value - view.model, view.residuals, atol=1e-15)
    np.testing.assert_allclose(view.residuals, session.residuals, atol=1e-12)
    assert np.std(view.residuals) < 0.02


def test_full_view_and_history(session: cup.PreWhitenSession) -> None:
    session.run()
    result = session.finalize()
    view = build_session_view(session, key=SessionKey("s"), id_epoch=3, result=result)
    assert not view.light and view.result is result and view.id_epoch == 3
    assert [round(r.frequency, 2) for r in view.rows] == [1.3, 2.71]
    assert view.history and view.cursor == len(view.history)
    assert all(h.reachable for h in view.history)
    assert view.history[0].op == "step" and view.history[0].kind == "accepted"
    assert "Step" in view.history[0].text
    assert "2</b> active" in view.summary_html
    session.undo()
    after = build_session_view(session, key=SessionKey("s"), id_epoch=4)
    assert after.cursor < len(after.history)  # the redo tail is listed
    assert after.can_redo


def test_history_items_from_entries(session: cup.PreWhitenSession) -> None:
    session.step()
    items = history_items(session.history, [True])
    assert len(items) == 1 and items[0].seq == 1 and items[0].reachable
