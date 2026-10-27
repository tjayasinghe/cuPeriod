"""Pre-release fixes of the interactive pre-whitening session (1.3.0 gate).

- A component the grid was configured past keeps its own refinement box.
- A project's ``meta.source`` is never trusted: saving or exporting an opened
  project touches no path the file chose (and never a network path).
- ``configure()`` takes numpy scalars, and refuses unknown setting names.
- A constrained project re-opens without a "stored values differ" warning.
- Huge run codes, oversized project grids, fractional point runs and unknown
  ``goto`` targets are refused with a message that says why.
- Exported scripts need nothing beyond ``import cuperiod``.
- ``PreWhitenSettings.downsample_points`` still reads, with a DeprecationWarning.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

import cuperiod as cup
from cuperiod.core.lightcurve import LightCurve
from cuperiod.prewhiten.session import PreWhitenSession, project
from cuperiod.prewhiten.session.script import RENDERERS
from cuperiod.prewhiten.session.state import ProjectError, SessionWarning
from cuperiod.prewhiten.subsets import ManualOrigin, Partition

UNC = "\\\\attacker.invalid\\share\\x.csv"


def _two_modes() -> tuple[np.ndarray, np.ndarray]:
    """0.9876 c/d (A = 0.06) and 6.4321 c/d (A = 0.025) over 60 days."""
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0.0, 60.0, 600))
    y = (
        0.06 * np.sin(2 * np.pi * 0.9876 * t + 0.3)
        + 0.025 * np.sin(2 * np.pi * 6.4321 * t + 1.1)
        + rng.normal(0.0, 0.005, t.size)
    )
    return t, y


def _numpy(**kw: Any) -> cup.PreWhitenSettings:
    return cup.PreWhitenSettings(backend="numpy", **kw)


def _rms(session: PreWhitenSession) -> float:
    return float(np.sqrt(np.mean(session.residuals**2)))


def _sine_session(**kw: Any) -> PreWhitenSession:
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 30.0, 300))
    y = 0.05 * np.sin(2 * np.pi * 3.1 * t) + rng.normal(0.0, 0.004, t.size)
    return PreWhitenSession(
        (t, y, np.full(t.size, 0.004)), settings=_numpy(max_frequencies=3), **kw
    )


# --- a component outside a re-configured grid (bug_sessions-1) ---------------------


@pytest.mark.parametrize(
    ("change", "outside"),
    [({"minimum_frequency": 2.0}, 0), ({"maximum_frequency": 4.0}, 1)],
)
def test_a_component_outside_a_reconfigured_grid_keeps_its_frequency(
    change: dict[str, float], outside: int
) -> None:
    t, y = _two_modes()
    session = PreWhitenSession((t, y), settings=_numpy())
    session.step()
    session.step()
    before = [(c.frequency, c.amplitude) for c in session.components]
    rms = _rms(session)
    assert session.configure(**change).kind == "edited"
    frequencies = np.array([c.frequency for c in session.components])
    low, high = session.kernel.bounds(session.state, frequencies)
    assert np.all(low <= frequencies) and np.all(frequencies <= high)
    width = session.state.settings.refine_bound_rayleigh / session.kernel.baseline(
        session.state
    )
    for label, act in (("improve", session.improve), ("polish", session.polish)):
        assert act().kind == "edited", label
        assert _rms(session) <= rms * 1.001, label
        f, a = (
            session.components[outside].frequency,
            session.components[outside].amplitude,
        )
        assert abs(f - before[outside][0]) < width, label
        assert a == pytest.approx(before[outside][1], rel=0.05), label
    result = session.finalize()
    assert len(result.frequency) == 2
    assert float(result.frequency[outside]) == pytest.approx(
        before[outside][0], abs=width
    )


def test_an_exact_add_below_the_grid_survives_finalize() -> None:
    t, y = _two_modes()
    session = PreWhitenSession((t, y), settings=_numpy(minimum_frequency=2.0))
    assert session.add(0.9876).hint == "exact"
    assert session.add(0.9876, refine="none").kind == "edited"
    assert session.step().kind == "accepted"
    result = session.finalize()
    assert float(result.frequency[0]) == pytest.approx(0.9876, abs=1e-3)
    assert float(result.amplitude[0]) == pytest.approx(0.06, rel=0.05)


def test_the_box_of_a_component_inside_the_grid_is_unchanged() -> None:
    t, y = _two_modes()
    session = PreWhitenSession((t, y), settings=_numpy())
    state = session.state
    engine = session.kernel.engine(state)
    f_min, f_max = float(engine.frequency[0]), float(engine.frequency[-1])
    freqs = np.array([0.5 * f_min, f_min, 3.0, f_max - 1e-3, f_max])
    shift = state.settings.refine_bound_rayleigh / session.kernel.baseline(state)
    low, high = session.kernel.bounds(state, freqs)
    expected_low = np.maximum(freqs - shift, max(0.5 * f_min, 1e-12))
    expected_high = np.minimum(freqs + shift, f_max)
    np.testing.assert_array_equal(low, expected_low)
    np.testing.assert_array_equal(
        high, np.maximum(expected_high, np.nextafter(expected_low, np.inf))
    )


# --- an untrusted meta.source (bug_sessions-2) ---------------------------------------


def _write_csv(path: Path, session: PreWhitenSession) -> None:
    """The session's data as a light-curve file (floats exact)."""
    data = session.data
    assert data.error is not None
    rows = "\n".join(
        f"{float(a)!r},{float(b)!r},{float(c)!r}"
        for a, b, c in zip(data.time, data.value, data.error, strict=True)
    )
    path.write_text("time,mag,mag_err\n" + rows + "\n", encoding="utf-8")


def _watch(monkeypatch: pytest.MonkeyPatch, needle: str) -> list[tuple[str, str]]:
    """Record (and block) every filesystem call on a path containing ``needle``."""
    touched: list[tuple[str, str]] = []

    def guard(name: str) -> None:
        real = getattr(Path, name)

        def method(self: Path, *args: Any, **kwargs: Any) -> Any:
            if needle in str(self):
                touched.append((name, str(self)))
                raise OSError("blocked by the test")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, name, method)

    for name in ("resolve", "is_file", "stat", "exists", "open"):
        guard(name)
    real_from_file = LightCurve.from_file

    def from_file(cls: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
        if needle in str(path):
            touched.append(("LightCurve.from_file", str(path)))
            raise OSError("blocked by the test")
        return real_from_file(path, *args, **kwargs)

    monkeypatch.setattr(LightCurve, "from_file", classmethod(from_file))
    return touched


@pytest.mark.parametrize("where", ["outside", "unc"])
def test_a_project_cannot_name_the_file_a_save_rereads(
    where: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _sine_session()
    session.run()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    path = project_dir / "p.cupw.json"
    session.save(path, embed_data="always")
    if where == "outside":
        # A real file holding exactly these data, outside the project's folder.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        hostile = str(elsewhere / "secret.csv")
        _write_csv(Path(hostile), session)
        needle = "secret.csv"
    else:
        hostile, needle = UNC, "attacker.invalid"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["data"]["meta"] = {"source": hostile}
    path.write_text(json.dumps(doc), encoding="utf-8")

    touched = _watch(monkeypatch, needle)
    loaded = PreWhitenSession.load(path)
    assert "source" not in loaded.data.meta
    assert loaded.data.meta[project.RECORDED_SOURCE] == hostile
    out = tmp_path / "out"
    out.mkdir()
    loaded.save(out / "resaved.cupw.json")
    loaded.export_script(out / "script.py")
    assert touched == []
    resaved = json.loads((out / "resaved.cupw.json").read_text(encoding="utf-8"))
    assert resaved["data"]["mode"] == "embedded"
    assert resaved["data"]["meta"] == {project.RECORDED_SOURCE: hostile}
    script = (out / "script.py").read_text(encoding="utf-8")
    (data_line,) = [line for line in script.splitlines() if line.startswith("DATA =")]
    assert "script.data.csv" in data_line  # written beside the script, not re-read


def test_a_network_source_is_refused_before_any_filesystem_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 30.0, 300))
    y = 0.05 * np.sin(2 * np.pi * 3.1 * t) + rng.normal(0.0, 0.004, t.size)
    lc = LightCurve.from_arrays(t, y, np.full(t.size, 0.004), meta={"source": UNC})
    session = PreWhitenSession(lc, settings=_numpy())
    touched = _watch(monkeypatch, "attacker.invalid")
    path, why = project.source_file(session)
    assert path is None and "UNC" in why
    assert touched == []


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("//host/share/x.csv", "UNC"),
        ("https://example.org/x.csv", "URL"),
        ("C:data.csv", "relative to a drive"),
        ("\\data.csv", "relative to a drive"),
        ("data.csv", ""),
        ("C:\\data\\x.csv", ""),
        ("/data/x.csv", ""),
    ],
)
def test_unusable_source_names(text: str, reason: str) -> None:
    why = project._unusable_source(text)
    assert (reason in why) if reason else why == ""


def test_a_referenced_project_still_references_its_file(tmp_path: Path) -> None:
    csv = tmp_path / "star.csv"
    _write_csv(csv, _sine_session())
    session = PreWhitenSession(LightCurve.from_file(csv), settings=_numpy())
    session.run()
    session.save(tmp_path / "p.cupw.json")
    loaded = PreWhitenSession.load(tmp_path / "p.cupw.json")
    loaded.save(tmp_path / "q.cupw.json")
    doc = json.loads((tmp_path / "q.cupw.json").read_text(encoding="utf-8"))
    assert doc["data"]["mode"] == "reference" and doc["data"]["path"] == "star.csv"


# --- configure() (bug_sessions-3, docs_prewhiten-4) ---------------------------------


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"max_frequencies": np.int64(2)}, {"max_frequencies": 2}),
        ({"snr_threshold": np.float32(3.5)}, {"snr_threshold": 3.5}),
        ({"stop_criteria": ["snr", "bic"]}, {"stop_criteria": ("snr", "bic")}),
    ],
)
def test_configure_takes_numpy_scalars_and_lists(
    updates: dict[str, Any], expected: dict[str, Any]
) -> None:
    session = _sine_session()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "Pydantic serializer warnings" either
        report = session.configure(**updates)
    assert report.kind == "edited", report.message
    for key, value in expected.items():
        assert getattr(session.settings, key) == value
    logged = session.history[-1].args["settings"]
    json.dumps(logged)  # plain JSON data in the log
    assert type(logged[next(iter(updates))]) in (int, float, list)


def test_do_configure_normalises_numpy_scalars() -> None:
    session = _sine_session()
    settings = {**session.settings.model_dump(), "max_frequencies": np.int64(4)}
    report = session.do("configure", settings=settings)
    assert report.kind == "edited", report.message
    logged = session.history[-1].args["settings"]
    json.dumps(logged)
    assert type(logged["max_frequencies"]) is int and logged["max_frequencies"] == 4
    assert session.settings.max_frequencies == 4


@pytest.mark.parametrize(
    "updates",
    [{"snr_treshold": 4.6}, {"snr_treshold": 4.6, "max_frequencies": 10}],
)
def test_configure_refuses_an_unknown_setting(updates: dict[str, Any]) -> None:
    session = _sine_session()
    before = session.settings
    report = session.configure(**updates)
    assert report.kind == "refused"
    assert "unknown settings ['snr_treshold']" in report.message
    assert session.settings == before and session.history == ()


def test_configure_still_refuses_a_bad_value() -> None:
    session = _sine_session()
    report = session.configure(snr_threshold=np.float64(-1.0))
    assert report.kind == "refused" and "snr_threshold" in report.message


# --- re-opening a constrained project (bug_sessions-4) -------------------------------


def _nightly() -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    rng = np.random.default_rng(1)
    night = np.repeat(np.arange(6), 50)
    t = night * 1.0 + 0.3 * rng.random(night.size)
    order = np.argsort(t)
    t, night = t[order], night[order]
    y = (
        0.04 * np.sin(2 * np.pi * 5.21 * t + 0.3)
        + 0.02 * np.sin(2 * np.pi * 8.77 * t + 1.7)
        + 0.01 * np.sin(2 * np.pi * 13.98 * t + 2.2)
        + rng.normal(0.0, 0.003, 6)[night]
        + rng.normal(0.0, 0.004, t.size)
    )
    return (t, y, np.full(t.size, 0.004)), night


def _constrained() -> PreWhitenSession:
    data, night = _nightly()
    session = PreWhitenSession(
        data,
        settings=_numpy(max_frequencies=4, maximum_frequency=25.0),
        labels={"night": night},
        partitions=["night"],
    )
    session.run(max_steps=4)
    assert session.use_offsets("night").kind == "edited"
    assert session.fix([2], amplitude=True).kind == "edited"
    assert session.polish().kind == "edited"
    return session


def test_a_constrained_project_reopens_without_a_warning(tmp_path: Path) -> None:
    session = _constrained()
    path = tmp_path / "mine.cupw.json"
    session.save(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SessionWarning)
        loaded = PreWhitenSession.load(path)
    assert loaded.components == session.components
    assert not loaded.is_dirty


def test_a_tampered_constrained_project_is_still_refitted(tmp_path: Path) -> None:
    session = _constrained()
    path = tmp_path / "mine.cupw.json"
    session.save(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["state"]["components"][0]["amplitude"] *= 1.0 + 1e-3
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.warns(SessionWarning, match="re-fitted values replace them"):
        loaded = PreWhitenSession.load(path)
    assert loaded.components[0].amplitude == pytest.approx(
        session.components[0].amplitude, rel=1e-5
    )


# --- run codes beyond int32 (bug_sessions-5) -----------------------------------------


@pytest.mark.parametrize("code", [1, 2**31, 2**40])
def test_a_run_code_outside_the_labels_is_refused(code: int) -> None:
    session = _sine_session()
    n = session.data.n
    report = session.do(
        "define_partition", name="p", labels=["a"], runs=[[0, n, code]], weights=[1.0]
    )
    assert report.kind == "refused" and "[0, 1)" in report.message
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        Partition.from_runs(n, [(0, n, code)], name="p", labels=["a"])


def test_replay_of_a_huge_run_code_raises_project_error() -> None:
    session = _sine_session()
    n = session.data.n
    partition = Partition.from_runs(n, [(0, n, 0)], name="p", labels=["a"])
    assert session.define_partition(partition).kind == "edited"
    entry = session.history[-1].to_dict()
    entry["args"]["runs"] = [[0, n, 2**40]]
    data = session.data
    with pytest.raises(ProjectError, match="refused on replay"):
        PreWhitenSession.replay(
            (data.time, data.value, data.error), [entry], settings=session.settings
        )


# --- the grid of an untrusted project (bug_sessions-6) -------------------------------


def test_grid_limit() -> None:
    assert project.grid_limit(300) == project.GRID_POINTS_FLOOR
    assert project.grid_limit(1_000_000) == 64_000_000
    assert project.grid_limit(10**8) == project.MAX_GRID_POINTS


@pytest.mark.parametrize("section", ["creation", "state"])
def test_an_oversized_project_grid_is_refused(section: str, tmp_path: Path) -> None:
    session = _sine_session()
    path = tmp_path / "p.cupw.json"
    session.save(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc[section]["grid"]["nf"] = project.GRID_POINTS_FLOOR + 1
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(
        ProjectError, match=rf"{section}\.grid\.nf: .* exceed the limit of 20000000"
    ):
        PreWhitenSession.load(path)


def test_a_session_whose_grid_a_load_would_refuse_is_not_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _sine_session()
    monkeypatch.setattr(project, "GRID_POINTS_FLOOR", 10)
    monkeypatch.setattr(project, "GRID_POINTS_PER_POINT", 0)
    with pytest.raises(ProjectError, match="cannot be saved: creation.grid.nf"):
        session.save(tmp_path / "p.cupw.json")
    assert not (tmp_path / "p.cupw.json").exists()


def test_a_costly_direct_numpy_project_opens_on_nufft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _sine_session()
    session.run()
    assert session.backend == "numpy"
    path = tmp_path / "p.cupw.json"
    session.save(path)
    monkeypatch.setattr(project, "DIRECT_WORK_LIMIT", 1000)
    with pytest.warns(SessionWarning, match="direct 'numpy' backend"):
        loaded = PreWhitenSession.load(path)
    assert loaded.backend == "finufft"
    assert loaded.components == session.components
    with warnings.catch_warnings():
        warnings.simplefilter("error", SessionWarning)
        kept = PreWhitenSession.load(path, backend="numpy")
    assert kept.backend == "numpy"


# --- goto() refusals (bug_sessions-7, docs_prewhiten-13) -----------------------------


def test_goto_says_why_a_state_is_out_of_reach(tmp_path: Path) -> None:
    session = _sine_session(session_settings=cup.SessionSettings(undo_depth=2))
    session.run()
    for value in (2, 3, 2):
        session.configure(max_frequencies=value)
    assert len(session.history) >= 4
    for seq in (999, -5):
        report = session.goto(seq)
        assert report.kind == "refused" and report.message == f"no log entry {seq}"
    report = session.goto(1)
    assert report.kind == "refused"
    assert "no longer in memory" in report.message and "undo_depth" in report.message
    assert "opened from a file" not in report.message
    path = tmp_path / "p.cupw.json"
    session.save(path)
    loaded = PreWhitenSession.load(path)
    report = loaded.goto(1)
    assert report.kind == "refused" and "opened from a file" in report.message
    assert loaded.goto(999).message == "no log entry 999"


# --- fractional point runs (bug_sessions-8) ------------------------------------------


@pytest.mark.parametrize("runs", [[(1.5, 3)], [(1, 3.0)], [(1, 2.5)]])
def test_fractional_run_bounds_are_refused(runs: list[tuple[Any, Any]]) -> None:
    session = _sine_session()
    report = session.delete_points(runs)
    assert report.kind == "refused" and "not a point selection" in report.message
    assert session.history == ()
    assert session.delete_points([(1, 3)]).kind == "edited"
    assert session.restore_points(runs).kind == "refused"


def test_numpy_integer_run_bounds_are_accepted() -> None:
    session = _sine_session()
    report = session.delete_points([(np.int64(1), np.int64(3))])
    assert report.kind == "edited" and session.state.mask.n_deleted == 2


# --- exported scripts use the public API only (docs_prewhiten-14) -------------------


def test_from_runs_takes_a_plain_origin() -> None:
    partition = Partition.from_runs(
        4,
        [(0, 2, 0), (2, 4, 1)],
        name="p",
        labels=["a", "b"],
        origin={"kind": "manual"},
    )
    assert isinstance(partition.origin, ManualOrigin)
    gaps = Partition.from_gaps(np.array([0.0, 0.1, 5.0, 5.1]), min_points=1)
    again = Partition.from_runs(
        4,
        gaps.runs(),
        name=gaps.name,
        labels=gaps.labels,
        origin=gaps.origin.model_dump(mode="json"),
    )
    assert again.origin == gaps.origin
    with pytest.raises(ValidationError):
        Partition.from_runs(
            4, [(0, 4, 0)], name="p", labels=["a"], origin={"kind": "x"}
        )


def test_an_exported_partition_needs_only_import_cuperiod() -> None:
    session = _sine_session()
    nights = Partition.from_gaps(session.data.time, min_gap=0.3, min_points=5)
    assert session.define_partition(nights).kind == "edited"
    text = session.to_script(data_expr="DATA")
    assert "parse_origin" not in text
    imports = [
        line for line in text.splitlines() if line.startswith(("import", "from"))
    ]
    assert imports == ["import numpy as np", "import cuperiod as cup"]
    statement = RENDERERS["define_partition"](session.history[-1].args)
    fresh = _sine_session()
    exec(statement, {"session": fresh, "cup": cup})
    (partition,) = fresh.state.partitions
    np.testing.assert_array_equal(partition.codes, nights.codes)
    assert partition.origin == nights.origin


# --- PreWhitenSettings.downsample_points (api-1) -------------------------------------


def test_reading_downsample_points_warns_and_reads_the_old_default() -> None:
    settings = cup.PreWhitenSettings()
    with pytest.warns(DeprecationWarning, match="downsample_points") as record:
        value = settings.downsample_points
    assert value == 2000
    assert record[0].filename == __file__
    assert "downsample_points" not in cup.PreWhitenSettings.model_fields


def test_the_retired_environment_variable_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUPERIOD_PREWHITEN_DOWNSAMPLE_POINTS", "700")
    with pytest.warns(DeprecationWarning, match="CUPERIOD_PREWHITEN_DOWNSAMPLE_POINTS"):
        settings = cup.PreWhitenSettings()
    assert settings == cup.PreWhitenSettings.model_construct()
