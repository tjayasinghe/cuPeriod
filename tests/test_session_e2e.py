"""One interactive session from the automatic run to a saved, scripted project.

A multi-season, multi-night light curve with a combination frequency, a mode whose
amplitude changes from season to season, a zero point per night and a few
outliers. The automatic run finds the strong modes; a manual take-over then deletes
the outliers, empties one night entirely, fits a zero point per night, adds and
ties the combination, fixes a frequency, lets one amplitude vary per season and
improves everything. Undo and redo walk the whole history; the project saves,
loads, verifies, and its exported script reproduces it in a fresh process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.prewhiten.session import PreWhitenSession, project
from cuperiod.prewhiten.session.state import LogEntry, SessionState, StepReport
from cuperiod.prewhiten.subsets import Partition

F1, F2, F4 = 7.3127, 11.8361, 4.4719
F3 = F1 + F2
A1, A2, A3 = 0.030, 0.020, 0.008
A4_SEASON = (0.010, 0.016, 0.022)
NOISE = 0.003
SEASONS, NIGHTS, PER_NIGHT = 3, 12, 50
OUTLIERS = (17, 403, 1222, 1508)
EMPTY_NIGHT = 5
EXCLUDED_NIGHT = 20

SETTINGS = cup.PreWhitenSettings(
    backend="finufft", max_frequencies=3, final_refine=False, prune=False
)
GRID = (2.0, 0.0005, 40000)  # 2..22 c/d: above the nightly zero-point structure


def light_curve() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2026)
    t_parts, season_parts = [], []
    for season in range(SEASONS):
        for night in range(NIGHTS):
            start = 30.0 * season + night * 1.0 + 0.05 * rng.random()
            t_parts.append(start + 0.3 * np.sort(rng.random(PER_NIGHT)))
            season_parts.append(np.full(PER_NIGHT, season))
    t = np.concatenate(t_parts)
    season = np.concatenate(season_parts)
    night = np.repeat(np.arange(SEASONS * NIGHTS), PER_NIGHT)
    tau = t - 30.0
    y = (
        A1 * np.sin(2 * np.pi * F1 * tau + 0.3)
        + A2 * np.sin(2 * np.pi * F2 * tau + 1.9)
        + A3 * np.sin(2 * np.pi * F3 * tau + 4.0)
        + np.asarray(A4_SEASON)[season] * np.sin(2 * np.pi * F4 * tau + 2.5)
        + rng.normal(0.0, 0.004, SEASONS * NIGHTS)[night]
        + rng.normal(0.0, NOISE, t.size)
    )
    y[list(OUTLIERS)] += 0.25
    return {"t": t, "y": y, "e": np.full(t.size, NOISE), "night": night}


def nearest(state: SessionState, frequency: float) -> int:
    """Id of the active component closest to ``frequency``."""
    best = min(state.active, key=lambda c: abs(c.frequency - frequency))
    assert abs(best.frequency - frequency) < 2e-3, (best, frequency)
    return best.id


def committed(report: StepReport) -> StepReport:
    assert report.committed, report
    return report


@pytest.fixture(scope="module")
def data() -> dict[str, np.ndarray]:
    return light_curve()


@pytest.fixture(scope="module")
def takeover(data: dict[str, np.ndarray]) -> dict[str, object]:
    return run_takeover(data)


def run_takeover(data: dict[str, np.ndarray]) -> dict[str, object]:
    """The whole session: heads after each action, for the tests below."""
    session = PreWhitenSession(
        (data["t"], data["y"], data["e"]), settings=SETTINGS, grid=GRID
    )
    heads: list[SessionState] = [session.state]

    def act(report: StepReport) -> StepReport:
        committed(report)
        project.validate_state(session.state, session.kernel.data)
        heads.append(session.state)
        return report

    # -- the automatic run: the three strongest modes --------------------------------
    report = session.run()
    project.validate_state(session.state, session.kernel.data)
    assert len(session.state.active) == 3
    heads.append(session.state)
    auto_entries = len(session.history)
    found = sorted(c.frequency for c in session.state.active)
    assert found == pytest.approx(sorted([F1, F2, F4]), abs=1e-3), (found, report)

    # -- the take-over ----------------------------------------------------------------
    residuals = session.residuals
    outliers = np.flatnonzero(np.abs(residuals) > 8.0 * NOISE)
    assert set(OUTLIERS) <= set(outliers.tolist())
    act(session.delete_points(outliers))
    nights = Partition.from_gaps(session.data.time, min_gap=0.5, name="night")
    assert nights.n_subsets == SEASONS * NIGHTS
    act(session.define_partition(nights))
    # Night EMPTY_NIGHT loses every point: its zero point has no data (NaN).
    emptied = np.flatnonzero(nights.codes == EMPTY_NIGHT)
    act(session.delete_points(emptied))
    act(session.exclude_subsets("night", [nights.labels[EXCLUDED_NIGHT]]))
    act(session.use_offsets("night"))
    # Nightly weights 1/rms^2 of the current residuals (the log keeps the numbers).
    weights = session.subset_rms_weights("night")
    assert len(weights) == SEASONS * NIGHTS and weights[EMPTY_NIGHT] == 1.0
    act(session.set_subset_weights("night", weights))
    act(session.use_weights("night"))
    id1, id2 = nearest(session.state, F1), nearest(session.state, F2)
    id4 = nearest(session.state, F4)
    act(session.add(F3))
    id3 = nearest(session.state, F3)
    act(session.tie(id3, f"F{id1} + F{id2}"))
    act(session.fix(id2, frequency=True))
    seasons = Partition.from_gaps(
        session.data.time, min_gap=10.0, name="season", prefix="S"
    )
    assert seasons.n_subsets == SEASONS
    act(session.define_partition(seasons))
    act(session.use_variation("season"))
    act(session.set_variation(id4, "amplitude"))
    act(session.improve())
    act(session.set_note(id3, "combination F1 + F2"))
    return {
        "session": session,
        "heads": heads,
        "auto_entries": auto_entries,
        "ids": (id1, id2, id3, id4),
        "outliers": outliers,
        "deleted": set(outliers.tolist()) | set(emptied.tolist()),
        "excluded": set(np.flatnonzero(nights.codes == EXCLUDED_NIGHT).tolist()),
    }


def test_the_takeover_recovers_the_planted_model(takeover: dict[str, object]) -> None:
    session = takeover["session"]
    assert isinstance(session, PreWhitenSession)
    id1, id2, id3, id4 = takeover["ids"]  # type: ignore[misc]
    state = session.state
    c1, c2, c3, c4 = (state.component(i) for i in (id1, id2, id3, id4))
    # The tie holds exactly, through the free F1 and the fixed F2.
    assert c3.frequency == pytest.approx(c1.frequency + c2.frequency, rel=1e-13)
    assert c2.fix_frequency and c3.tie is not None
    result = session.result(uncertainty="covariance")
    rows = {s.label: s for s in result.components}
    s1 = rows[f"F{id1}"]
    assert abs(s1.frequency - F1) < 5.0 * s1.frequency_error
    assert abs(s1.amplitude - A1) < 5.0 * s1.amplitude_error
    assert abs(rows[f"F{id3}"].amplitude - A3) < 5.0 * rows[f"F{id3}"].amplitude_error
    # One amplitude per season, close to the planted ones.
    assert c4.variation == "amplitude" and c4.subset_amplitude is not None
    np.testing.assert_allclose(np.abs(c4.subset_amplitude), A4_SEASON, atol=1e-3)
    # A zero point per night; the emptied and the excluded night have none.
    offsets = np.asarray(state.subset_offsets)
    assert offsets.shape == (SEASONS * NIGHTS,)
    assert np.isnan(offsets[EMPTY_NIGHT]) and np.isnan(offsets[EXCLUDED_NIGHT])
    assert np.isfinite(np.delete(offsets, [EMPTY_NIGHT, EXCLUDED_NIGHT])).all()
    deleted, excluded = takeover["deleted"], takeover["excluded"]
    assert isinstance(deleted, set) and isinstance(excluded, set)
    assert set(np.flatnonzero(state.mask.deleted).tolist()) == deleted
    assert result.subsets is not None
    # Residuals at the noise level: the outliers and the zero points are gone.
    used = session.kernel.used(state)
    assert np.std(session.residuals) == pytest.approx(NOISE, rel=0.1)
    assert used.n == SEASONS * NIGHTS * PER_NIGHT - len(deleted | excluded)


def test_undo_and_redo_walk_the_whole_history(takeover: dict[str, object]) -> None:
    session = takeover["session"]
    heads = takeover["heads"]
    assert isinstance(session, PreWhitenSession) and isinstance(heads, list)
    final = session.state
    log = session.history
    # Every manual action is its own undo step; the automatic run is one group.
    walked = [session.state]
    while session.undo():
        walked.append(session.state)
        project.validate_state(session.state, session.kernel.data)
    assert not session.can_undo and session.state.n_iterations == 0
    expected = list(reversed(heads))
    assert len(walked) == len(expected)
    for got, want in zip(walked, expected, strict=True):
        assert got.same_content(want)
    assert session.history == ()
    assert session.history_cursor == 0 and len(session.log_entries) == len(log)
    while session.redo():
        pass
    assert session.state.same_content(final)
    assert session.history == log
    assert session.log_entries[: session.history_cursor] == log
    # The derived fit of a revisited head is the head's own (no stale cache).
    np.testing.assert_allclose(
        session.fit.frequency, [c.frequency for c in final.active], rtol=0, atol=0
    )


def test_save_load_verify(takeover: dict[str, object], tmp_path: Path) -> None:
    session = takeover["session"]
    assert isinstance(session, PreWhitenSession)
    path = tmp_path / "e2e.cupw.json"
    session.save(path, embed_data="always", title="e2e")
    assert not session.is_dirty
    loaded = PreWhitenSession.load(path, verify=True)
    # The stored head comes back as stored; its fit (statistics, zero points) is
    # re-derived on load, so those agree to the rebuild's round-off.
    assert project.divergence(session.state, loaded.state) is None
    np.testing.assert_allclose(
        np.asarray(loaded.state.subset_offsets, dtype=float),
        np.asarray(session.state.subset_offsets, dtype=float),
        rtol=0.0, atol=1e-10,
    )
    assert loaded.state.stats.rss == pytest.approx(session.state.stats.rss, rel=1e-10)
    assert _brief(loaded.history) == _brief(session.history)
    project.validate_state(loaded.state, loaded.kernel.data)
    # The loaded head's numbers are the session's (no drift, no replay needed).
    for a, b in zip(loaded.state.components, session.state.components, strict=True):
        assert a.frequency == b.frequency and a.amplitude == b.amplitude


def _brief(log: tuple[LogEntry, ...]) -> list[tuple[object, ...]]:
    return [(e.seq, e.group, e.op, e.args, e.report.kind, e.report.message)
            for e in log]


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUPERIOD_")}
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def test_exported_script_reproduces_the_session(
    takeover: dict[str, object], tmp_path: Path
) -> None:
    session = takeover["session"]
    assert isinstance(session, PreWhitenSession)
    folder = tmp_path / "scripts"
    folder.mkdir()
    script = session.export_script(folder / "e2e.py", title="e2e take-over")
    saved = tmp_path / "reproduced.cupw.json"
    check = folder / "e2e_check.py"
    check.write_text(
        script.read_text(encoding="utf-8") + f"\nsession.save({str(saved)!r})\n",
        encoding="utf-8",
    )
    done = subprocess.run(
        [sys.executable, str(check)], capture_output=True, text=True, env=_env(),
        cwd=str(tmp_path), timeout=600, check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    reproduced = PreWhitenSession.load(saved)
    assert project.divergence(session.state, reproduced.state) is None
    assert [e.op for e in reproduced.history] == [e.op for e in session.history]


# --- hardening regressions (stage H1) ----------------------------------------------


def _small_session(**kwargs: object) -> PreWhitenSession:
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 20.0, 400))
    y = 0.05 * np.sin(2 * np.pi * 8.0 * t + 0.4) + 0.03 * np.sin(
        2 * np.pi * 5.3 * t + 1.0
    )
    y = y + rng.normal(0.0, 0.004, t.size)
    return PreWhitenSession(
        (t, y, np.full(t.size, 0.004)),
        settings=cup.PreWhitenSettings(backend="finufft", max_frequencies=3),
        grid=(0.5, 0.005, 3000),
        **kwargs,  # type: ignore[arg-type]
    )


def test_prune_without_protection_dissolves_the_ties_of_a_pruned_parent() -> None:
    session = _small_session(
        session_settings=cup.SessionSettings(protect_manual=False)
    )
    committed(session.add(4.0, snap=False, refine="none"))  # noise only: S/N < 4
    committed(session.add(8.0, snap=False, refine="none"))
    committed(session.tie(2, "2F1"))
    report = session.prune()
    assert report.kind == "edited", report
    assert "1 tie to a pruned parent dissolved" in report.message
    (survivor,) = session.state.components
    assert survivor.tie is None and survivor.id == 1  # renumbered
    assert survivor.frequency == pytest.approx(8.0, abs=1e-3)
    project.validate_state(session.state, session.kernel.data)


def test_prune_with_protection_keeps_a_tie_parent() -> None:
    session = _small_session()
    committed(session.add(4.0, snap=False, refine="none"))
    committed(session.add(8.0, snap=False, refine="none"))
    committed(session.tie(2, "2F1"))
    assert session.prune().kind == "noop"
    assert [c.id for c in session.state.components] == [1, 2]


def test_constraint_refusals_name_the_rows_not_fit_positions() -> None:
    session = _small_session()
    committed(session.add(5.3))
    committed(session.add(8.0))
    committed(session.add(11.0, snap=False, refine="none"))
    committed(session.set_active(1, False))  # positions now differ from ids
    report = session.tie(3, "F2", allow_close=True)
    assert report.kind == "refused"
    assert "F2 and F3 have identical frequencies" in report.message, report


def test_fit_constraints_labels_reach_every_message() -> None:
    from cuperiod.prewhiten.constraints import (
        ComponentConstraint,
        ConstraintError,
        FitConstraints,
        FrequencyTie,
    )

    tie = ComponentConstraint(frequency=FrequencyTie(parents=(0,), coefficients=(1,)))
    labelled = FitConstraints(
        components=(ComponentConstraint(), tie), labels=("F7", "F9")
    )
    with pytest.raises(ConstraintError, match="F7 and F9 have identical"):
        labelled.validate(np.array([3.0, 3.0]), 50, fit_mean=True, t_ref=0.0)
    cyclic = FitConstraints(
        components=(
            ComponentConstraint(
                frequency=FrequencyTie(parents=(1,), coefficients=(2,))
            ),
            ComponentConstraint(
                frequency=FrequencyTie(parents=(0,), coefficients=(2,))
            ),
        ),
        labels=("F4", "F6"),
    )
    with pytest.raises(ConstraintError, match="cyclic tie: F4 -> F6 -> F4"):
        cyclic.resolve(np.array([1.0, 2.0]))
    # Without labels: positions, as before.
    plain = FitConstraints(components=(ComponentConstraint(), tie))
    with pytest.raises(ConstraintError, match="F1 and F2 have identical"):
        plain.validate(np.array([3.0, 3.0]), 50, fit_mean=True, t_ref=0.0)


def test_exclude_subsets_names_an_unknown_label() -> None:
    t, y, e, _ = (*light_curve().values(),)
    night = np.repeat(np.arange(SEASONS * NIGHTS), PER_NIGHT).astype(str)
    session = PreWhitenSession(
        (t, y, e), settings=SETTINGS, grid=GRID, labels={"night": night},
        partitions=["night"],
    )
    report = session.exclude_subsets("night", ["X"])
    assert report.kind == "refused"
    assert "'X'" in report.message and "'night'" in report.message
    assert "tuple.index" not in report.message


def test_improve_of_every_row_on_a_plain_state() -> None:
    session = _small_session()
    committed(session.add(5.3))
    committed(session.add(8.0))
    report = session.improve([1, 2])  # nothing held: used to trip an assertion
    assert report.kind == "edited", report


def test_wrappers_refuse_bad_references_instead_of_raising() -> None:
    session = _small_session()
    committed(session.add(5.3))
    for report in (
        session.set_values("Fx", amplitude=0.1),
        session.tie(True, "F1"),  # type: ignore[arg-type]
        session.set_note(None, "x"),  # type: ignore[arg-type]
        session.remove("bogus"),
        session.fix(["F1", "F?"], frequency=True),
        session.improve(["nope"]),
        session.add("not a number"),  # type: ignore[arg-type]
        session.set_subset_weights("night", [[1.0]]),  # type: ignore[list-item]
        session.delete_points([[1, "a"]]),
        session.exclude_subsets("night", [float("inf")]),  # type: ignore[list-item]
    ):
        assert report.kind == "refused", report
    assert len(session.history) == 1
    # numpy integers are ids like any other
    assert session.set_note(np.int64(1), "numpy id").committed


def test_session_settings_are_exported_by_core() -> None:
    from cuperiod.core import SessionSettings, config

    assert SessionSettings is cup.SessionSettings
    assert "SessionSettings" in config.__all__
