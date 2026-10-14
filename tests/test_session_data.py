"""PreWhitenSession data operations: points, subsets, zero points, weights."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import cuperiod as cup
import oracle_multisine as om
import prewhiten_corpus as pc
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve, stack_bands
from cuperiod.prewhiten.fit import fit_multisine
from cuperiod.prewhiten.session import PreWhitenSession
from cuperiod.prewhiten.session import kernel as kernel_module
from cuperiod.prewhiten.session.ops_data import residual_rms_weights
from cuperiod.prewhiten.session.state import StepReport
from cuperiod.prewhiten.spectrum import SpectrumEngine
from cuperiod.prewhiten.subsets import Partition

CASE = pc.case_by_name("triple-last")
SETTINGS = CASE.resolved_settings()
NIGHTLY = pc.case_by_name("nightly-last")


def _data(case: pc.Case = CASE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (t, y, e), _, _ = pc.load(case)
    assert e is not None
    return t, y, e


def _halves(t: np.ndarray) -> Partition:
    return Partition.from_ranges(t, [float(np.median(t))], name="half")


def _three(**kw: Any) -> PreWhitenSession:
    t, y, e = _data()
    session = PreWhitenSession(
        (t, y, e), settings=SETTINGS, partitions=[_halves(t)], **kw
    )
    session.run(max_steps=3)
    return session


def _refused(
    session: PreWhitenSession, call: Callable[[], StepReport], match: str = ""
) -> StepReport:
    head, log = session.state, session.history
    report = call()
    assert report.kind == "refused", report
    assert re.search(match, report.message), report.message
    assert session.state is head and session.history == log
    return report


def _values(session: PreWhitenSession) -> np.ndarray:
    return np.array(
        [(c.frequency, c.amplitude, c.phase) for c in session.components]
    )


# --- points ---------------------------------------------------------------------------


def test_delete_restore_round_trip() -> None:
    session = _three()
    session.calculate()  # a manual edit, so the budget origin is already reset
    head = session.state
    grid, t_ref = head.grid, session.t_ref
    report = session.delete_points([10, 11, 12, 300])
    assert report.kind == "edited" and {"data", "fit"} <= report.changed
    deleted = session.state
    assert deleted.mask.n_deleted == 4 and session.result().n_deleted == 4
    assert deleted.grid == grid and session.t_ref == t_ref
    assert session.kernel.used(deleted).n == head.stats.n_samples - 4
    assert deleted.stats.n_samples == head.stats.n_samples - 4
    assert session.delete_points([10]).kind == "noop"
    assert session.restore_points([300]).kind == "edited"
    assert session.state.mask.n_deleted == 3
    assert session.restore_points().kind == "edited"
    assert session.state.same_content(head)
    assert session.restore_points().kind == "noop"
    mask = np.zeros(head.mask.n, dtype=bool)
    mask[[5, 6]] = True
    assert session.delete_points(mask).kind == "edited"
    assert session.state.mask.runs() == [(5, 7)]
    assert session.delete_points([(20, 25)]).kind == "edited"
    assert session.state.mask.n_deleted == 7


def test_delete_point_refusals() -> None:
    session = _three()
    n = session.data.n
    _refused(session, lambda: session.delete_points([n]), r"\[0, ")
    _refused(session, lambda: session.delete_points([1.5]), "integers")
    _refused(session, lambda: session.do("delete_points", runs=[]), "runs")
    _refused(session, lambda: session.do("delete_points", runs=[(5, 5)]), "half-open")
    _refused(session, lambda: session.delete_points(np.arange(n - 2)), "min_detections")
    _refused(session, lambda: session.restore_points([n + 1]), r"\[0, ")


def test_delete_then_auto_is_a_fresh_session_on_the_reduced_data() -> None:
    t, y, e = _data()
    session = PreWhitenSession((t, y, e), settings=SETTINGS)
    rng = np.random.default_rng(0)
    gone = np.sort(rng.choice(t.size, 60, replace=False))
    session.delete_points(gone)
    session.run()
    edited = session.finalize()
    keep = np.setdiff1d(np.arange(t.size), gone)
    fresh = PreWhitenSession(
        (t[keep], y[keep], e[keep]),
        settings=SETTINGS,
        grid=session.state.grid,
        t_ref=session.t_ref,
    )
    fresh.run()
    reference = fresh.finalize()
    assert len(edited) == len(reference)
    for a, b in zip(edited, reference, strict=True):
        for name in ("frequency", "amplitude", "phase", "snr", "frequency_error"):
            assert getattr(a, name) == pytest.approx(getattr(b, name), rel=1e-10)
    assert edited.n_deleted == gone.size and edited.n_samples == keep.size


def test_engines_are_built_once_per_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[int] = []

    class Spy(SpectrumEngine):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            built.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(kernel_module, "SpectrumEngine", Spy)
    session = _three()
    assert built == []  # the constructor's engine serves the full sampling
    session.delete_points([1, 2, 3])
    assert len(built) == 1
    session.calculate()
    assert len(built) == 1
    assert session.undo() and session.undo()  # back on the full sampling
    session.kernel.forget()
    session.calculate()
    assert session.redo() is False  # the calculate truncated redo
    session.delete_points([1, 2, 3])
    assert len(built) == 1  # the LRU still holds the reduced sampling's engine


# --- subsets --------------------------------------------------------------------


def test_exclude_then_include_keeps_deleted_points_deleted() -> None:
    session = _three()
    t = session.data.time
    first_half = np.flatnonzero(t < np.median(t))
    session.delete_points(first_half[:5])
    n_used = session.kernel.used(session.state).n
    assert session.exclude_subsets("half", ["R1"]).kind == "edited"
    assert session.state.excluded == (("half", (0,)),)
    assert session.kernel.used(session.state).n == t.size - first_half.size
    assert session.exclude_subsets("half", [0]).kind == "noop"
    assert session.restore_points().kind == "edited"  # restores, never re-includes
    assert session.kernel.used(session.state).n == t.size - first_half.size
    session.delete_points(first_half[:5])
    assert session.exclude_subsets("half", [0], excluded=False).kind == "edited"
    assert not session.state.excluded
    assert session.kernel.used(session.state).n == n_used
    _refused(session, lambda: session.exclude_subsets("half", [7]), "no subset 7")
    _refused(session, lambda: session.exclude_subsets("half", ["X"]), "not in")
    _refused(session, lambda: session.exclude_subsets("moon", [0]), "moon")
    session.exclude_subsets("half", [0])
    _refused(session, lambda: session.exclude_subsets("half", [1]), "min_detections")
    _refused(session, lambda: session.drop_partition("half"), "excluded")


def test_partitions_define_replace_drop() -> None:
    session = _three()
    t = session.data.time
    nights = Partition.from_chunks(t, length=5.0, name="chunk")
    assert session.define_partition(nights).kind == "edited"
    assert session.state.partition("chunk").n_subsets == nights.n_subsets
    assert session.define_partition(nights).kind == "noop"
    wider = Partition.from_chunks(t, length=10.0, name="chunk")
    assert session.define_partition(wider).kind == "edited"
    assert session.state.partition("chunk").n_subsets == wider.n_subsets
    session.use_offsets("chunk")
    _refused(session, lambda: session.define_partition(nights), "in use")
    _refused(session, lambda: session.drop_partition("chunk"), "in use")
    session.use_offsets(None)
    assert session.drop_partition("chunk").kind == "edited"
    _refused(session, lambda: session.drop_partition("chunk"), "chunk")
    bad = {"name": "x", "labels": ["a"], "runs": [(0, 3, 0)], "weights": [1.0]}
    _refused(session, lambda: session.do("define_partition", **bad), "cover")
    origin = {**bad, "runs": [(0, t.size, 0)], "origin": {"kind": "moon"}}
    _refused(session, lambda: session.do("define_partition", **origin), "kind")


def test_define_partition_is_neutral() -> None:
    session = PreWhitenSession(
        _data(), settings=SETTINGS.model_copy(update={"max_frequencies": 2})
    )
    session.run()
    before = session.state
    assert before.status == "stopped"
    session.define_partition(_halves(session.data.time))
    after = session.state
    assert after.status == "stopped" and after.blocked == before.blocked
    assert session.kernel.cached(after) is session.kernel.cached(before)


# --- zero points --------------------------------------------------------------------


def _nightly_offsets() -> tuple[tuple[np.ndarray, ...], np.ndarray, Partition]:
    t, y, e = _data(NIGHTLY)
    rng = np.random.default_rng(5)
    night = np.floor(t - t.min()).astype(int)
    planted = rng.normal(0.0, 0.01, night.max() + 1)
    return (t, y + planted[night], e), planted, Partition.from_gaps(t, name="night")


def test_planted_nightly_zero_points_are_recovered() -> None:
    data, planted, nights = _nightly_offsets()
    session = PreWhitenSession(
        data, settings=NIGHTLY.resolved_settings(), partitions=[nights]
    )
    assert session.use_offsets("night").kind == "edited"
    session.run()
    found = sorted(round(c.frequency, 1) for c in session.components)
    assert found == [9.7, 14.2]
    result = session.result()
    assert result.subsets is not None and result.subsets.offsets_by == "night"
    offsets = result.subsets.offset - 8.0
    assert np.all(np.abs(offsets - planted) < 4.5 * result.subsets.offset_error)
    # The data spectrum is shown with the zero points removed.
    plain = PreWhitenSession(data, settings=NIGHTLY.resolved_settings())
    assert np.max(session.spectrum().amplitude[:50]) < np.max(
        plain.spectrum().amplitude[:50]
    )
    _refused(session, lambda: session.configure(fit_mean=False), "fit_mean")


def test_zero_points_need_a_free_mean_and_flag_low_frequencies() -> None:
    data, _, nights = _nightly_offsets()
    cfg = NIGHTLY.resolved_settings()
    session = PreWhitenSession(
        data, settings=cfg.model_copy(update={"fit_mean": False}), partitions=[nights]
    )
    _refused(session, lambda: session.use_offsets("night"), "fit_mean")
    session = PreWhitenSession(data, settings=cfg, partitions=[nights])
    session.use_offsets("night")
    session.run(max_steps=2)
    assert session.add(1.0, snap=False, refine="none").kind == "edited"
    flags = {c.label: c.flags for c in session.result()}
    assert "low_frequency" in flags["F3"]
    assert "low_frequency" not in flags["F1"]
    assert session.use_offsets(None).kind == "edited"
    assert session.state.subset_offsets is None and session.result().subsets is None


# --- weights --------------------------------------------------------------------------


def test_subset_weights_reach_the_fit_and_the_spectrum() -> None:
    session = _three()
    t, y, e = _data()
    halves = session.state.partition("half")
    assert session.set_subset_weights("half", [1.0, 3.0]).kind == "edited"
    assert session.state.weights_by is None  # not applied until used
    assert session.use_weights("half").kind == "edited"
    omega = np.where(halves.codes == 0, 1.0, 3.0)
    omega = omega / omega.mean()
    used = session.kernel.used(session.state)
    assert used.error_eff is not None
    np.testing.assert_allclose(used.error_eff, e / np.sqrt(omega), rtol=1e-15)
    fit = fit_multisine(
        t, y, e / np.sqrt(omega), session.fit.frequency,
        fit_mean=True, t_ref=session.t_ref, refine="none",
    )
    np.testing.assert_allclose(session.fit.amplitude, fit.amplitude, rtol=1e-9)
    np.testing.assert_allclose(session.fit.phase, fit.phase, rtol=1e-9)
    assert session.set_subset_weights("half", [1.0, 3.0]).kind == "noop"
    _refused(session, lambda: session.set_subset_weights("half", [1.0]), "weights")
    _refused(session, lambda: session.set_subset_weights("half", [1.0, 0.0]), "posit")
    assert session.use_weights(None).kind == "edited"
    assert session.state.is_plain()


def test_rms_weights() -> None:
    session = _three()
    weights = residual_rms_weights(
        session.kernel, session.state, session.derived, "half"
    )
    codes = session.state.partition("half").codes
    r = session.fit.residuals
    expected = np.array([1.0 / np.mean(r[codes == g] ** 2) for g in (0, 1)])
    np.testing.assert_allclose(weights, expected / expected.mean(), rtol=1e-12)
    assert session.set_subset_weights("half", weights).kind == "edited"


# --- roles and partitions -------------------------------------------------------------


def test_switching_the_variation_partition_resets_varying_components() -> None:
    session = _three()
    t = session.data.time
    session.define_partition(Partition.from_chunks(t, length=5.0, name="chunk"))
    session.use_variation("half")
    session.set_variation([1, 2], "amplitude")
    assert session.use_variation("chunk").kind == "edited"
    assert all(c.variation == "none" for c in session.components)
    assert all(c.subset_amplitude is None for c in session.components)
    session.set_variation(1, "both")
    assert session.use_variation(None).kind == "edited"
    assert session.state.component(1).variation == "none"
    assert session.use_variation(None).kind == "noop"
    _refused(session, lambda: session.use_variation("moon"), "moon")


def test_configure_regrids_and_refuses_a_backend_change() -> None:
    session = _three()
    grid = session.state.grid
    _refused(session, lambda: session.configure(backend="finufft"), "backend")
    _refused(session, lambda: session.configure(device="cuda"), "device")
    _refused(session, lambda: session.configure(snr_threshold=-1.0), "snr_threshold")
    _refused(session, lambda: session.do("configure", settings={"bogus": 1}), "bogus")
    assert session.configure().kind == "noop"
    values = _values(session)
    assert session.configure(maximum_frequency=40.0).kind == "edited"
    new_grid = session.state.grid
    assert new_grid != grid and new_grid[0] + new_grid[1] * (new_grid[2] - 1) > 39.0
    top = new_grid[0] + new_grid[1] * (new_grid[2] - 1)
    assert session.spectrum().frequency[-1] == pytest.approx(top)
    np.testing.assert_array_equal(_values(session), values)  # the fit is kept
    assert session.configure(normalization="dft").kind == "edited"
    assert session.state.grid == new_grid
    assert session.state.settings.normalization == "dft"
    assert session.undo() and session.undo() and session.state.grid == grid


# --- multiband ------------------------------------------------------------------------


def test_from_bands_is_stack_bands_plus_a_plain_session() -> None:
    t, y, e = _data()
    half = t.size // 2
    bands = MultiBandLightCurve(
        bands={
            "g": LightCurve(time=t[::2], value=y[::2] + 0.3, error=e[::2]),
            "r": LightCurve(time=t[1::2], value=y[1::2] - 0.2, error=e[1::2]),
        }
    )
    session = PreWhitenSession.from_bands(bands, settings=SETTINGS)
    assert session.state.partitions[0].name == "band"
    assert session.state.partitions[0].labels == ("g", "r")
    assert session.history == ()  # a root partition is never logged
    lc, _ = stack_bands(bands, centre=True)
    plain = PreWhitenSession(lc, settings=SETTINGS)
    session.run()
    plain.run()
    np.testing.assert_array_equal(_values(session), _values(plain))
    assert session.use_offsets("band").kind == "edited"
    offsets = np.asarray(session.state.subset_offsets)
    assert offsets.shape == (2,) and half > 0


# --- per-subset values survive eviction and restore ---------------------------------


@pytest.mark.parametrize(
    ("mode", "attribute"),
    [("amplitude", "subset_amplitude"), ("phase", "subset_phase")],
)
def test_variation_states_rebuild_after_eviction_and_restore(
    mode: Any, attribute: str
) -> None:
    session = _three()
    session.use_variation("half")
    assert session.set_variation(1, mode).kind == "edited"
    head = session.state
    stored = np.asarray(getattr(head.component(1), attribute))
    assert stored.shape == (2,) and np.all(np.isfinite(stored))
    session.set_note(2, "x")
    session.kernel.forget()
    assert session.undo() and session.state is head
    rebuilt = session.kernel.derive(head)
    assert rebuilt.drift <= 1e-9
    detail = rebuilt.fit.detail
    assert detail is not None
    values = detail.subset_amplitude if mode == "amplitude" else detail.subset_phase
    assert values is not None
    np.testing.assert_allclose(values[0], stored, rtol=1e-9, atol=1e-12)
    restored = PreWhitenSession.restore(session.snapshot(), _data())
    assert restored.state.same_content(head)
    assert restored.derived.drift <= 1e-9


# --- error contract and op-class side effects of the data ops -----------------------

DATA_EDITS: dict[str, Callable[[PreWhitenSession], StepReport]] = {
    "delete_points": lambda s: s.delete_points([3, 4]),
    "restore_points": lambda s: (s.delete_points([3]), s.restore_points())[1],
    "exclude_subsets": lambda s: s.exclude_subsets("half", [1]),
    "set_subset_weights": lambda s: s.set_subset_weights("half", [1.0, 2.0]),
    "use_offsets": lambda s: s.use_offsets("half"),
    "use_variation": lambda s: s.use_variation("half"),
    "use_weights": lambda s: s.use_weights("half"),
    "configure": lambda s: s.configure(snr_threshold=3.5),
}


@pytest.mark.parametrize("name", sorted(DATA_EDITS))
def test_data_edits_reset_the_search(name: str) -> None:
    case = pc.case_by_name("drift-blocked")
    t, y, e = _data(case)
    session = PreWhitenSession(
        (t, y, e), settings=case.resolved_settings(), partitions=[_halves(t)]
    )
    session.run()
    before = session.state
    assert before.status == "stopped" and before.blocked
    report = DATA_EDITS[name](session)
    assert report.kind == "edited", report
    after = session.state
    assert after.status == "searching" and after.blocked == ()
    assert after.budget_origin == after.n_iterations == before.n_iterations
    assert not session.is_pristine
    assert session.kernel.cached(after) is not None


# --- end to end ---------------------------------------------------------------------


def test_auto_then_manual_takeover_matches_the_oracle() -> None:
    """Auto run, then by hand: delete outliers, nightly zero points, resume, drop
    spurious terms, tie the combination, fix a frequency, Improve all."""
    rng = np.random.default_rng(41)
    t = np.sort(
        np.concatenate([n + rng.uniform(0.0, 0.33, 35) for n in range(14)])
    ) + 2459000.0
    night = np.floor(t - t.min()).astype(int)
    tau = t - t.min()
    f1, f2 = 9.7, 14.2
    y = (
        8.0
        + 0.015 * np.sin(2 * np.pi * f1 * tau + 2.5)
        + 0.008 * np.sin(2 * np.pi * f2 * tau + 0.4)
        + 0.004 * np.sin(2 * np.pi * (f1 + f2) * tau + 1.3)
    )
    y = y + rng.normal(0.0, 0.006, 14)[night] + rng.normal(0.0, 0.002, t.size)
    outliers = np.array([17, 120, 333])
    y[outliers] += 0.08
    e = np.full(t.size, 0.002)
    cfg = cup.PreWhitenSettings(
        backend="numpy", maximum_frequency=30.0, samples_per_peak=5, max_frequencies=8
    )
    session = PreWhitenSession((t, y, e), settings=cfg)
    session.run()
    assert session.status == "stopped" and len(session.components) == 2
    with session.group():
        assert session.delete_points(outliers).kind == "edited"
        session.define_partition(Partition.from_gaps(t, name="night"))
        assert session.use_offsets("night").kind == "edited"
    session.run()  # Auto resumes on the cleaned data
    spurious = [
        c.id
        for c in session.components
        if min(abs(c.frequency - f) for f in (f1, f2, f1 + f2)) > 0.05
    ]
    if spurious:
        session.remove(*spurious)

    def nearest(f: float) -> int:
        return min(session.components, key=lambda c: abs(c.frequency - f)).id

    i1, i2, i3 = nearest(f1), nearest(f2), nearest(f1 + f2)
    assert len({i1, i2, i3}) == 3
    assert session.tie(i3, f"F{i1} + F{i2}").kind == "edited"
    assert session.fix(i2, frequency=True).kind == "edited"
    assert session.improve().kind == "edited"
    state = session.state
    c1, c2, c3 = (state.component(i) for i in (i1, i2, i3))
    assert c3.frequency == pytest.approx(c1.frequency + c2.frequency, rel=1e-14)

    # The same model by brute force: every free parameter optimised at once.
    used = session.kernel.used(state)
    fit = session.kernel.with_covariance(state, session.derived).fit
    assert used.offset_codes is not None and used.offset_subsets is not None
    assert used.error is not None
    assert fit.detail is not None
    offsets = np.asarray(state.subset_offsets)[used.offset_subsets]
    spec = om.OracleSpec(
        t=used.time, y=used.value, w=1.0 / used.error**2, t_ref=session.t_ref,
        matrix=np.array([[1.0], [0.0], [1.0]]),
        constant=np.array([0.0, c2.frequency, c2.frequency]),
        amp=["common"] * 3, phase=["common"] * 3, offsets=used.offset_codes,
    )
    truth = np.array(
        [c1.frequency]
        + [v for c in (c1, c2, c3) for v in (c.amplitude, c.phase)]
        + list(offsets)
    )
    oracle = om.solve(spec, truth)
    assert np.max(np.abs(oracle.params - truth) / oracle.sigma) < 1e-4
    assert fit.rss == pytest.approx(oracle.rss, rel=1e-10)
    position = {cid: k for k, cid in enumerate(c.id for c in state.active)}
    names = [f"f[{position[i1]}]"] + [
        f"{q}[{position[i]}]" for i in (i1, i2, i3) for q in ("A", "phi")
    ] + [f"offset[{g}]" for g in range(offsets.size)]
    index = [fit.detail.parameter_names.index(n) for n in names]
    sigma = np.sqrt(np.diag(fit.detail.parameter_covariance))[index]
    np.testing.assert_allclose(sigma, oracle.sigma, rtol=1e-5)
    result = session.result()
    assert result.n_deleted == outliers.size
    assert {c.label: c.fixed for c in result}[f"F{i2}"] == "f"
    assert {c.label: c.tied for c in result}[f"F{i3}"] == f"F{i1} + F{i2}"
