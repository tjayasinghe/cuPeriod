"""Refusals that name a remedy carry it as a code (``StepReport.hint``).

The messages spell the remedy in the library's keyword arguments ("pass
allow_close=True to add it anyway"), which a GUI user cannot pass; the GUI used to
rewrite them by regular expression. Each such refusal now also carries one of
``allow_close``, ``cascade``, ``exact`` or ``fit_mean``, and its message is unchanged.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import prewhiten_corpus as pc
from cuperiod.prewhiten.session import PreWhitenSession
from cuperiod.prewhiten.session.project import ReportModel
from cuperiod.prewhiten.session.state import (
    REFUSAL_HINTS,
    SessionError,
    StepReport,
)
from cuperiod.prewhiten.subsets import Partition

CASE = pc.case_by_name("triple-last")
SETTINGS = CASE.resolved_settings()


def _three(**settings: Any) -> PreWhitenSession:
    """The triple after three accepted steps: F1 12.34, F2 17.81, F3 24.68 (= 2 F1)."""
    (t, y, e), _, _ = pc.load(CASE)
    cfg = SETTINGS.model_copy(update=settings) if settings else SETTINGS
    session = PreWhitenSession(
        (t, y, e),
        settings=cfg,
        partitions=[Partition.from_ranges(t, [float(np.median(t))], name="half")],
    )
    session.run(max_steps=3)
    assert [c.id for c in session.components] == [1, 2, 3]
    return session


def _refused(
    session: PreWhitenSession, call: Callable[[], StepReport], match: str
) -> StepReport:
    head = session.state
    report = call()
    assert report.kind == "refused", report
    assert re.search(match, report.message), report.message
    assert session.state is head  # nothing committed
    assert report.hint is None or report.hint in REFUSAL_HINTS
    return report


def test_a_collision_on_add_carries_allow_close() -> None:
    session = _three()
    f1 = session.state.component(1).frequency
    report = _refused(
        session,
        lambda: session.add(f1 + 0.001, snap=False),
        r"pass allow_close=True to add it anyway",
    )
    assert report.hint == "allow_close"


def test_a_collision_on_tie_carries_allow_close() -> None:
    session = _three()
    report = _refused(  # 2 F1 lands on F3
        session, lambda: session.tie(2, "2F1"), r"pass allow_close=True to tie it"
    )
    assert report.hint == "allow_close"


def test_a_frequency_outside_the_grid_carries_exact() -> None:
    session = _three()
    report = _refused(
        session,
        lambda: session.add(100.0, snap=False),
        r"outside the frequency grid .*; add it with refine='none'$",
    )
    assert report.hint == "exact"


def test_a_refinement_onto_another_component_carries_exact() -> None:
    session = _three()
    f1 = session.state.component(1).frequency
    baseline = session.kernel.baseline(session.state)
    # Beyond the collision radius (0.25 / T) but inside the resolution guard
    # (1.5 / T), with a refinement box of +-1 / T: the refined frequency always
    # lands within the guard of F1.
    report = _refused(
        session,
        lambda: session.add(f1 + 0.4 / baseline, snap=False),
        r"the refinement moved .* onto F1 .*; add it with refine='none' or "
        r"allow_close=True",
    )
    assert report.hint == "exact"


def test_removing_a_tie_parent_carries_cascade() -> None:
    session = _three()
    assert session.tie(3, "2F1").kind == "edited"
    report = _refused(
        session,
        lambda: session.remove(1),
        r"remove with cascade=True to dissolve the tie",
    )
    assert report.hint == "cascade"


def test_zero_points_without_a_floating_mean_carry_fit_mean() -> None:
    session = _three(fit_mean=False)
    report = _refused(
        session,
        lambda: session.use_offsets("half"),
        r"zero points per subset need fit_mean=True \(configure it first\)",
    )
    assert report.hint == "fit_mean"


def test_refusals_without_a_named_remedy_carry_no_hint() -> None:
    session = _three()
    assert _refused(session, lambda: session.remove(99), "F99").hint is None
    assert _refused(session, lambda: session.add(-1.0, snap=False), "").hint is None
    assert _refused(
        session, lambda: session.do("remove", ids=[1], bogus=True), "bogus"
    ).hint is None
    # Committed reports never carry one.
    assert session.add(40.0, snap=False, refine="none").hint is None


def test_the_hint_round_trips_through_the_report_dict() -> None:
    refused = StepReport(kind="refused", message="m", hint="cascade")
    assert refused.to_dict()["hint"] == "cascade"
    assert StepReport.from_dict(refused.to_dict()) == refused
    edited = StepReport(kind="edited", message="m")
    assert "hint" not in edited.to_dict()  # log entries keep their old form
    assert StepReport.from_dict(edited.to_dict()).hint is None
    # The strict project model accepts a known code and refuses anything else.
    logged = {**edited.to_dict(), "hint": "exact"}
    assert ReportModel.model_validate(logged).hint == "exact"
    with pytest.raises(ValueError, match="hint"):
        ReportModel.model_validate({**edited.to_dict(), "hint": "rm -rf"})


def test_session_error_keeps_its_hint() -> None:
    error = SessionError("m", hint="fit_mean")
    assert error.hint == "fit_mean" and str(error) == "m"
    assert pickle.loads(pickle.dumps(error)).hint == "fit_mean"
    assert SessionError("plain").hint is None
    assert {"allow_close", "cascade", "exact", "fit_mean"} == REFUSAL_HINTS
