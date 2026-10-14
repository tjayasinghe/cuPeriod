"""Project files (*.cupw.json): round trips, replay, verify, and hostile input."""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import os
import pickle
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cuperiod as cup
import prewhiten_corpus as pc
from cuperiod.prewhiten.session import PreWhitenSession, actions, project
from cuperiod.prewhiten.session.state import (
    ProjectError,
    SessionState,
    SessionWarning,
    StepReport,
)
from cuperiod.prewhiten.subsets import Partition

FIXTURES = Path(__file__).parent / "data" / "sessions"
SETTINGS = pc.case_by_name("triple-last").resolved_settings()


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data, _, _ = pc.load(pc.case_by_name("triple-last"))
    t, y, e = data
    assert e is not None
    return t, y, e


def _site(n: int) -> np.ndarray:
    return np.array(["A", "B", "C"] * (n // 3 + 1), dtype=object)[:n]


def _session(**kw: Any) -> PreWhitenSession:
    return PreWhitenSession(_data(), settings=SETTINGS, **kw)


def _ran(**kw: Any) -> PreWhitenSession:
    session = _session(**kw)
    session.run()
    session.finalize()
    return session


def _same_log(a: PreWhitenSession, b: PreWhitenSession) -> None:
    assert [e.to_dict() for e in a.history] == [e.to_dict() for e in b.history]


def _write_csv(path: Path, t: np.ndarray, y: np.ndarray, e: np.ndarray) -> None:
    rows = "\n".join(
        f"{float(a)!r},{float(b)!r},{float(c)!r}"
        for a, b, c in zip(t, y, e, strict=True)
    )
    path.write_text("time,mag,mag_err\n" + rows + "\n", encoding="utf-8")


def _write(path: Path, doc: dict[str, Any]) -> Path:
    path.write_text(json.dumps(doc, allow_nan=False), encoding="utf-8")
    return path


@pytest.fixture
def local_reducer() -> Any:
    """Register test-local reducers; the registry is restored afterwards."""
    from cuperiod.prewhiten.session.session import _load_ops

    _load_ops()  # so a later-stage module cannot re-register over the local one
    saved: dict[str, Any] = {}

    def install(op: str, reducer: Any) -> None:
        if op not in saved:
            saved[op] = actions.reducer(op)
        actions.register(op)(reducer)

    yield install
    for op, previous in saved.items():
        actions.unregister(op)
        if previous is not None:
            actions.register(op)(previous)


# --- round trips ----------------------------------------------------------------------


def test_round_trip_embedded(tmp_path: Path) -> None:
    session = _ran()
    path = tmp_path / "triple.cupw.json"
    session.save(path, title="triple")
    assert not session.is_dirty
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["format"] == project.FORMAT and doc["version"] == project.VERSION
    assert doc["data"]["mode"] == "embedded" and doc["title"] == "triple"
    loaded = PreWhitenSession.load(path)
    assert loaded.state.same_content(session.state)
    _same_log(loaded, session)
    # Float exactness: every value comes back bit for bit.
    for a, b in zip(session.components, loaded.components, strict=True):
        assert (a.frequency, a.amplitude, a.phase) == (
            b.frequency,
            b.amplitude,
            b.phase,
        )
    np.testing.assert_array_equal(loaded.data.time, session.data.time)
    np.testing.assert_array_equal(loaded.data.error, session.data.error)
    assert loaded.t_ref == session.t_ref and loaded.state.grid == session.state.grid
    assert not loaded.is_dirty and not loaded.can_undo
    assert loaded.reachable(len(loaded.history)) and not loaded.reachable(1)
    first, second = session.result(), loaded.result()
    np.testing.assert_array_equal(first.frequency, second.frequency)
    np.testing.assert_array_equal(first.residuals, second.residuals)


def test_round_trip_referenced(tmp_path: Path) -> None:
    t, y, e = _data()
    source = tmp_path / "star.csv"
    _write_csv(source, t, y, e)
    lc = cup.LightCurve.from_file(source)
    session = PreWhitenSession(lc, settings=SETTINGS)
    session.run(max_steps=2)
    path = tmp_path / "star.cupw.json"
    session.save(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["data"]["mode"] == "reference" and doc["data"]["path"] == "star.csv"
    assert "time" not in doc["data"]
    loaded = PreWhitenSession.load(path, verify=True)
    assert loaded.state.same_content(session.state)
    _same_log(loaded, session)
    # The data file moves away: an explicit data= still opens the project.
    source.rename(tmp_path / "elsewhere.csv")
    with pytest.raises(ProjectError, match="not found"):
        PreWhitenSession.load(path)
    again = PreWhitenSession.load(path, data=(t, y, e))
    assert again.state.same_content(session.state)
    # Referencing is impossible now; "never" says why, "auto" embeds.
    with pytest.raises(ProjectError, match="cannot reference"):
        loaded.save(tmp_path / "again.cupw.json", embed_data="never")
    loaded.save(tmp_path / "again.cupw.json")
    doc = json.loads((tmp_path / "again.cupw.json").read_text(encoding="utf-8"))
    assert doc["data"]["mode"] == "embedded"


def test_a_project_outside_the_data_directory_embeds(tmp_path: Path) -> None:
    t, y, e = _data()
    (tmp_path / "data").mkdir()
    source = tmp_path / "data" / "star.csv"
    _write_csv(source, t, y, e)
    session = PreWhitenSession(cup.LightCurve.from_file(source), settings=SETTINGS)
    (tmp_path / "projects").mkdir()
    path = tmp_path / "projects" / "p.cupw.json"
    session.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["data"]["mode"] == "embedded"
    sub = tmp_path / "data" / "p.cupw.json"
    session.save(sub)
    assert json.loads(sub.read_text(encoding="utf-8"))["data"]["mode"] == "reference"


def test_verify_a_finufft_session_and_load_it_on_numpy(tmp_path: Path) -> None:
    pytest.importorskip("finufft")
    session = PreWhitenSession(
        _data(), settings=SETTINGS.model_copy(update={"backend": "finufft"})
    )
    session.run()
    session.finalize()
    path = tmp_path / "f.cupw.json"
    session.save(path)
    verified = PreWhitenSession.load(path, verify=True)
    assert verified.backend == "finufft"
    assert verified.state.same_content(session.state)
    on_numpy = PreWhitenSession.load(path, verify=True, backend="numpy")
    assert on_numpy.backend == "numpy"
    assert on_numpy.state.same_content(session.state)


def test_an_unavailable_backend_falls_back_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session()
    session.run(max_steps=1)
    path = tmp_path / "p.cupw.json"
    session.save(path)
    monkeypatch.setattr(project, "_backend_available", lambda name: False)
    with pytest.warns(SessionWarning, match="not available"):
        loaded = PreWhitenSession.load(path)
    assert loaded.backend in ("numpy", "finufft")
    assert loaded.state.same_content(session.state)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an explicit backend never warns
        PreWhitenSession.load(path, backend="numpy")


def test_hash_mismatch_and_data_override(tmp_path: Path) -> None:
    session = _session()
    session.run(max_steps=2)
    path = tmp_path / "p.cupw.json"
    session.save(path)
    t, y, e = _data()
    with pytest.raises(ProjectError, match="content hash"):
        PreWhitenSession.load(path, data=(t, y + 1e-4, e))
    with pytest.warns(SessionWarning):
        modified = PreWhitenSession.load(
            path, data=(t, y + 1e-4, e), allow_modified_data=True
        )
    assert [c.id for c in modified.components] == [c.id for c in session.components]
    same = PreWhitenSession.load(path, data=cup.LightCurve.from_arrays(t, y, e))
    assert same.state.same_content(session.state)


def test_save_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session()
    path = tmp_path / "p.cupw.json"
    session.save(path)
    before = path.read_bytes()
    session.run(max_steps=1)

    def broken(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        session.save(path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.cupw.json"]
    assert session.is_dirty


def test_embedding_limits(tmp_path: Path) -> None:
    session = _session(session_settings=cup.SessionSettings(embed_data_max_points=10))
    with pytest.raises(ProjectError, match="embed_data_max_points"):
        session.save(tmp_path / "p.cupw.json")
    session.save(tmp_path / "p.cupw.json", embed_data="always")
    with pytest.raises(ProjectError, match="cannot reference"):
        session.save(tmp_path / "q.cupw.json", embed_data="never")
    small = _session(session_settings=cup.SessionSettings(max_project_bytes=1000))
    with pytest.raises(ProjectError, match="max_project_bytes"):
        small.save(tmp_path / "r.cupw.json")
    assert not (tmp_path / "r.cupw.json").exists()


def test_the_frozen_v1_fixture_loads_and_verifies() -> None:
    path = FIXTURES / "v1_minimal.cupw.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    session = PreWhitenSession.load(path, verify=True)
    stored = doc["state"]["components"]
    assert [c.id for c in session.components] == [c["id"] for c in stored]
    assert [c.frequency for c in session.components] == [c["frequency"] for c in stored]
    assert session.backend == "numpy" and len(session.history) == len(doc["log"])
    assert session.history[0].timestamp == doc["log"][0]["timestamp"]
    assert session.result().n_components == len(stored)


def test_an_older_version_goes_through_its_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = json.loads((FIXTURES / "v1_minimal.cupw.json").read_text(encoding="utf-8"))
    # A hand-written "version 0": fix flags as a mapping, no title.
    old = copy.deepcopy(doc)
    old["version"] = 0
    old.pop("title")
    for component in old["state"]["components"]:
        f, a, p = component.pop("fix")
        component["fixed"] = {"frequency": f, "amplitude": a, "phase": p}
    path = _write(tmp_path / "v0.cupw.json", old)
    with pytest.raises(ProjectError, match="no migration from project version 0"):
        PreWhitenSession.load(path)

    def migrate(d: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(d)
        for component in out["state"]["components"]:
            fixed = component.pop("fixed")
            component["fix"] = [fixed["frequency"], fixed["amplitude"], fixed["phase"]]
        return out

    monkeypatch.setitem(project._MIGRATIONS, 0, migrate)
    migrated = PreWhitenSession.load(path, verify=True)
    reference = PreWhitenSession.load(FIXTURES / "v1_minimal.cupw.json")
    assert migrated.state.same_content(reference.state)


def test_environment_immunity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _ran()
    path = tmp_path / "p.cupw.json"
    session.save(path)
    monkeypatch.setenv("CUPERIOD_PREWHITEN_SNR_THRESHOLD", "50")
    monkeypatch.setenv("CUPERIOD_SESSION_PROTECT_MANUAL", "false")
    assert cup.PreWhitenSettings().snr_threshold == 50.0  # the environment is live
    loaded = PreWhitenSession.load(path, verify=True)
    assert loaded.settings.snr_threshold == SETTINGS.snr_threshold
    assert loaded.session_settings.protect_manual is True
    assert loaded.state.same_content(session.state)
    replayed = PreWhitenSession.replay(
        _data(),
        [e.to_dict() for e in session.history],
        settings=loaded.kernel.creation.settings,
        grid=loaded.state.grid,
        t_ref=loaded.t_ref,
        session_settings=loaded.session_settings,
    )
    assert replayed.state.same_content(session.state)
    # A field a file lacks takes its declared default, never the environment's.
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["creation"]["settings"]["snr_threshold"]
    del doc["state"]["settings"]["snr_threshold"]
    del doc["creation"]["session_settings"]["protect_manual"]
    partial = PreWhitenSession.load(_write(tmp_path / "partial.cupw.json", doc))
    assert partial.settings.snr_threshold == 4.0
    assert partial.session_settings.protect_manual is True


def test_stored_statistics_are_recommitted(tmp_path: Path) -> None:
    session = _ran()
    path = tmp_path / "p.cupw.json"
    session.save(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["state"]["stats"]["rss"] = 123.0
    doc["state"]["stats"]["bic"] = None
    doc["state"]["offset"] = 99.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = PreWhitenSession.load(_write(tmp_path / "t.cupw.json", doc))
    assert loaded.state.same_content(session.state)


def test_a_tampered_amplitude_is_replaced_with_a_warning(tmp_path: Path) -> None:
    session = _ran()
    path = tmp_path / "p.cupw.json"
    session.save(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["state"]["components"][0]["amplitude"] *= 1.0 + 1e-6
    with pytest.warns(SessionWarning, match="re-fitted values replace them"):
        loaded = PreWhitenSession.load(_write(tmp_path / "t.cupw.json", doc))
    assert loaded.components[0].amplitude == session.components[0].amplitude
    assert loaded.derived.drift > project.RESTORE_RTOL


def test_from_bands_keeps_its_root_partition(tmp_path: Path) -> None:
    t, y, e = _data()
    bands = cup.MultiBandLightCurve.from_light_curves(
        {
            "V": cup.LightCurve.from_arrays(t[::2], y[::2], e[::2]),
            "B": cup.LightCurve.from_arrays(t[1::2], y[1::2] + 0.3, e[1::2]),
        }
    )
    session = PreWhitenSession.from_bands(bands, settings=SETTINGS)
    session.run(max_steps=2)
    path = tmp_path / "b.cupw.json"
    session.save(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["data"]["mode"] == "embedded" and list(doc["data"]["labels"]) == ["band"]
    loaded = PreWhitenSession.load(path, verify=True)
    (root,) = loaded.state.partitions
    (original,) = session.state.partitions
    assert root.name == "band" and root.labels == original.labels
    np.testing.assert_array_equal(root.codes, original.codes)
    assert loaded.data.labels["band"].tolist() == [
        str(v) for v in session.data.labels["band"]
    ]
    assert loaded.state.same_content(session.state)


def test_a_constrained_head_round_trips(tmp_path: Path, local_reducer: Any) -> None:
    def pin(kernel: Any, state: SessionState, derived: Any, args: Any) -> Any:
        first = state.components[0]
        changed = dataclasses.replace(first, fix_frequency=True, note="pinned")
        new = dataclasses.replace(state, components=(changed, *state.components[1:]))
        return new, None, StepReport(kind="edited", message="F1 frequency fixed")

    local_reducer("fix", pin)
    session = _session()
    session.run(max_steps=3)
    assert session.fix(1, frequency=True).kind == "edited"
    assert session.state.components[0].fix_frequency
    path = tmp_path / "c.cupw.json"
    session.save(path)
    loaded = PreWhitenSession.load(path, verify=True)
    assert loaded.components[0].fix_frequency and loaded.components[0].note == "pinned"
    assert loaded.state.has_model_constraints()
    for a, b in zip(session.components, loaded.components, strict=True):
        assert (a.frequency, a.amplitude, a.phase) == (
            b.frequency,
            b.amplitude,
            b.phase,
        )
    assert loaded.state.stats.rss == pytest.approx(session.state.stats.rss, rel=1e-12)


# --- replay and verify ----------------------------------------------------------------


def _create(session: PreWhitenSession) -> dict[str, Any]:
    creation = session.kernel.creation
    return {
        "settings": creation.settings,
        "grid": creation.grid,
        "t_ref": creation.t_ref,
        "session_settings": creation.session_settings,
    }


def test_replay_reproduces_the_session_bitwise_with_its_groups() -> None:
    session = _session()
    session.run()
    session.finalize()
    replayed = PreWhitenSession.replay(_data(), session.history, **_create(session))
    assert replayed.state.same_content(session.state)
    _same_log(replayed, session)  # timestamps and groups included
    assert replayed.undo() and session.undo()  # the finalize group
    assert replayed.state.same_content(session.state)
    assert replayed.undo() and session.undo()  # the whole run
    assert replayed.state.same_content(session.state) and not replayed.can_undo


def test_replay_names_the_entry_whose_outcome_differs() -> None:
    session = _session()
    session.run()
    log = [e.to_dict() for e in session.history]
    log[-1]["report"]["kind"] = "accepted"
    with pytest.raises(ProjectError, match=r"log entry 4 .*'rejected'.*'accepted'"):
        PreWhitenSession.replay(_data(), log, **_create(session))
    log = [e.to_dict() for e in session.history]
    log[1]["seq"] = 7
    with pytest.raises(ProjectError, match="breaks the sequence"):
        PreWhitenSession.replay(_data(), log, **_create(session))
    log = [e.to_dict() for e in session.history]
    log[0]["args"] = {"seed": 12.3}
    with pytest.raises(ProjectError, match="seed and index"):
        PreWhitenSession.replay(_data(), log, **_create(session))


def test_replay_warns_about_a_seed_that_is_not_the_tallest_peak() -> None:
    session = _session()
    second = session.candidate()
    assert second is not None
    session.run(max_steps=1)
    log = [e.to_dict() for e in session.history]
    # A seed the search would not pick: the second step's peak, attempted first.
    probe = _session()
    probe.run(max_steps=2)
    log[0]["args"] = dict(probe.history[1].args)
    log[0]["report"] = probe.history[1].report.to_dict()
    with pytest.warns(SessionWarning, match="not this run's tallest"):
        PreWhitenSession.replay(_data(), log, **_create(session))


def test_verify_catches_a_head_that_does_not_follow_from_its_log(
    tmp_path: Path,
) -> None:
    session = _ran()
    path = tmp_path / "p.cupw.json"
    session.save(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["state"]["components"][1]["frequency"] *= 1.0 + 1e-7
    tampered = _write(tmp_path / "t.cupw.json", doc)
    with pytest.warns(SessionWarning):  # frequencies are held on a plain load
        PreWhitenSession.load(tampered)
    with (
        warnings.catch_warnings(),
        pytest.raises(ProjectError, match="does not follow from its log"),
    ):
        warnings.simplefilter("ignore", SessionWarning)
        PreWhitenSession.load(tampered, verify=True)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["log"][0]["report"]["frequency"] = 1.0
    with pytest.raises(ProjectError, match="verify failed at log entry 1"):
        PreWhitenSession.load(_write(tmp_path / "u.cupw.json", doc), verify=True)


def test_validate_state_accepts_every_committed_head() -> None:
    session = _session(labels={"site": _site(500)}, partitions=["site"])
    data = session.kernel.data
    project.validate_state(session.state, data)
    while session.state.status != "stopped":
        session.step()
        project.validate_state(session.state, data)
    session.finalize()
    project.validate_state(session.state, data)


# --- hostile files --------------------------------------------------------------------


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A valid project (root partition, three components, a tie) as plain JSON."""
    session = _session(labels={"site": _site(500)}, partitions=["site"])
    session.run(max_steps=3)
    path = tmp_path_factory.mktemp("saved") / "p.cupw.json"
    session.save(path)
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert len(doc["state"]["components"]) == 3
    return doc


def _mutated(doc: dict[str, Any], change: Any) -> dict[str, Any]:
    out = copy.deepcopy(doc)
    change(out)
    return out


def _tie(child: int, parents: list[int]) -> Any:
    def change(d: dict[str, Any]) -> None:
        d["state"]["components"][child]["tie"] = {
            "parents": parents,
            "coefficients": [1] * len(parents),
        }

    return change


def _set(*keys: Any, value: Any) -> Any:
    def change(d: dict[str, Any]) -> None:
        target = d
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    return change


def _site_runs(runs: list[list[int]]) -> Any:
    return _set("state", "partitions", 0, "runs", value=runs)


HOSTILE: dict[str, tuple[Any, str]] = {
    "unknown top-level key": (_set("evil", value=1), "evil"),
    "unknown component key": (_set("state", "components", 0, "evil", value=1), "evil"),
    "wrong type": (_set("data", "n", value="500"), "n"),
    "bool for a number": (_set("state", "offset", value=True), "offset"),
    "time longer than n": (lambda d: d["data"]["time"].append(1.0), "data.time"),
    "labels shorter than n": (lambda d: d["data"]["labels"]["site"].pop(), "labels"),
    "newer version": (_set("version", value=2), "newer cuPeriod"),
    "not a project": (_set("format", value="something"), "not a cuPeriod"),
    "duplicate ids": (_set("state", "components", 1, "id", value=1), "duplicate id"),
    "id at next_id": (_set("state", "next_id", value=3), "next_id"),
    "missing tie parent": (_tie(2, [9]), "parent F9 does not exist"),
    "self tie": (_tie(2, [3]), "tied to itself"),
    "cyclic ties": (lambda d: (_tie(0, [2])(d), _tie(1, [1])(d)), "cyclic tie"),
    "tie coefficient out of range": (
        _set(
            "state",
            "components",
            2,
            "tie",
            value={"parents": [1], "coefficients": [1000]},
        ),
        "coefficient",
    ),
    "overlapping partition runs": (
        _site_runs([[0, 300, 0], [200, 500, 1]]),
        "partitions[0]",
    ),
    "gapped partition runs": (
        _site_runs([[0, 200, 0], [300, 500, 1]]),
        "partitions[0]",
    ),
    "partition code out of range": (
        _site_runs([[0, 250, 0], [250, 500, 7]]),
        "partitions[0]",
    ),
    "partition shorter than the data": (
        _site_runs([[0, 250, 0], [250, 400, 1]]),
        "partitions[0]",
    ),
    "role naming a missing partition": (
        _set("state", "offsets_by", value="night"),
        "offsets_by",
    ),
    "exclusion naming a missing partition": (
        _set("state", "excluded", value=[["night", [0]]]),
        "excluded",
    ),
    "exclusion out of range": (
        _set("state", "excluded", value=[["site", [5]]]),
        "excluded",
    ),
    "subset offsets of the wrong length": (
        lambda d: (
            _set("state", "offsets_by", value="site")(d),
            _set("state", "subset_offsets", value=[0.0, 0.0])(d),
        ),
        "subset_offsets",
    ),
    "subset amplitudes without a partition": (
        _set("state", "components", 0, "subset_amplitude", value=[1.0, 2.0, 3.0]),
        "subset_amplitude",
    ),
    "mask longer than the data": (_set("state", "mask", "n", value=501), "mask"),
    "unsorted mask runs": (
        _set("state", "mask", "deleted", value=[[10, 20], [5, 8]]),
        "mask",
    ),
    "budget beyond the iterations": (
        _set("state", "budget_origin", value=99),
        "budget_origin",
    ),
    "unknown origin kind": (
        _set("state", "partitions", 0, "origin", value={"kind": "pickle"}),
        "origin",
    ),
    "tie text with __import__": (
        lambda d: d["log"].append(
            {
                **d["log"][-1],
                "seq": len(d["log"]) + 1,
                "op": "tie",
                "args": {"id": 1, "expression": "__import__('os')"},
            }
        ),
        "tie",
    ),
    "unknown op": (_set("log", 0, "op", value="eval"), "unknown op"),
    "bad settings value": (
        _set("creation", "settings", "snr_threshold", value=-1.0),
        "snr_threshold",
    ),
    "unknown settings key": (
        _set("state", "settings", "evil", value=1),
        "unknown settings",
    ),
    "content hash not hex": (
        _set("data", "content_sha256", value="x" * 64),
        "content_sha256",
    ),
    "a grid without points": (_set("state", "grid", "nf", value=0), "nf"),
    "a huge grid": (_set("creation", "grid", "nf", value=10**12), "nf"),
}


@pytest.mark.parametrize("case", sorted(HOSTILE))
def test_hostile_documents_raise_project_error(
    case: str, saved: dict[str, Any], tmp_path: Path
) -> None:
    change, match = HOSTILE[case]
    path = _write(tmp_path / "h.cupw.json", _mutated(saved, change))
    with pytest.raises(ProjectError, match=re.escape(match)):
        PreWhitenSession.load(path)


def test_the_unmodified_document_loads(saved: dict[str, Any], tmp_path: Path) -> None:
    session = PreWhitenSession.load(_write(tmp_path / "ok.cupw.json", saved))
    assert len(session.components) == 3 and session.state.partitions[0].name == "site"


@pytest.mark.parametrize(
    "payload",
    [
        b'{"format": "cuperiod-prewhiten-project", "version": 1, "x": NaN}',
        b'{"format": "cuperiod-prewhiten-project", "version": 1, "x": Infinity}',
        b'{"format": "cuperiod-prewhiten-project", "version": 1, "x": 1e400}',
        b"\xff\xfe\x00not utf-8",
        b"[" * 100_000 + b"]" * 100_000,
        pickle.dumps({"format": "cuperiod-prewhiten-project"}),
        b"",
        b"[1, 2, 3]",
        b'{"format": "cuperiod-prewhiten-project", "version": true}',
        b'{"version": 1, "format": "x", "format": "cuperiod-prewhiten-project"}',
    ],
    ids=[
        "nan",
        "infinity",
        "overflow",
        "utf8",
        "nesting",
        "pickle",
        "empty",
        "array",
        "bool-version",
        "duplicate-key",
    ],
)
def test_unreadable_files_raise_project_error(payload: bytes, tmp_path: Path) -> None:
    path = tmp_path / "x.cupw.json"
    path.write_bytes(payload)
    with pytest.raises(ProjectError):
        PreWhitenSession.load(path)


def test_an_oversized_file_is_refused_before_it_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "big.cupw.json"
    with path.open("wb") as fh:
        fh.truncate(65 * 2**20)

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the file was parsed")

    monkeypatch.setattr(json, "loads", never)
    with pytest.raises(ProjectError, match="max_project_bytes"):
        PreWhitenSession.load(path)


def test_a_huge_log_is_refused(saved: dict[str, Any], tmp_path: Path) -> None:
    doc = dict(saved)
    doc["log"] = [0] * 1_000_000
    with pytest.raises(ProjectError, match="max_actions"):
        PreWhitenSession.load(_write(tmp_path / "log.cupw.json", doc))


@pytest.mark.parametrize(
    "reference",
    [
        "/etc/passwd.csv",
        "C:/Windows/star.csv",
        "C:star.csv",
        "../star.csv",
        "sub/../../star.csv",
        "\\\\host\\share\\star.csv",
        "//host/share/star.csv",
        "https://example.org/star.csv",
        "file:///star.csv",
        "star.exe",
        "star.csv.zip",
        "",
    ],
)
def test_data_references_stay_inside_the_project_directory(
    reference: str,
    saved: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = copy.deepcopy(saved)
    data = doc["data"]
    for key in ("time", "value", "error"):
        data.pop(key)
    data["labels"] = {}
    doc["creation"]["label_names"] = []
    data.update(mode="reference", path=reference, columns=None)

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a file outside the project was read")

    monkeypatch.setattr(cup.LightCurve, "from_file", never)
    path = _write(tmp_path / "r.cupw.json", doc)
    with pytest.raises(ProjectError, match="data.path"):
        PreWhitenSession.load(path)


def test_no_pickle_eval_or_exec_in_the_session_package() -> None:
    package = Path(project.__file__).parent
    for source in sorted(package.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                names = set()
            assert not names & {"pickle", "marshal", "shelve", "dill"}, source.name
            if isinstance(node, ast.Name):
                assert node.id not in {"eval", "exec", "__import__"}, source.name
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"eval", "exec"}, source.name


def test_partitions_survive_with_their_origin(tmp_path: Path) -> None:
    t, _, _ = _data()
    nights = Partition.from_gaps(t, name="night", min_gap=0.3)
    session = _session(partitions=[nights])
    path = tmp_path / "n.cupw.json"
    session.save(path)
    loaded = PreWhitenSession.load(path, verify=True)
    (root,) = loaded.state.partitions
    assert root.origin.model_dump() == nights.origin.model_dump()
    np.testing.assert_array_equal(root.codes, nights.codes)
    assert loaded.kernel.creation.partitions[0].name == "night"
