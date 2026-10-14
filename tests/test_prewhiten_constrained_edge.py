"""Spec 1.8, "fitted amplitude -> 0 with a non-linear phase".

A constrained component whose phase is a non-linear parameter — a common phase over
per-subset amplitudes, or per-subset phases over a common amplitude — can meet data
with no signal at its frequency. The amplitude then solves to round-off (a few
float64 epsilons of the data's scale) rather than to zero, and the phase it
multiplies is undefined. The spec's behaviour: the trust region copes (no crash, no
NaN value), and the phase — with any frequency only that component sets — is
reported with a NaN error and the ``degenerate`` flag. Before the fix the covariance
divided two rounding errors and reported the phase known to ~0.3 rad, unflagged.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cuperiod.core.config import PreWhitenSettings
from cuperiod.prewhiten import _constrained
from cuperiod.prewhiten.constraints import ComponentConstraint, FitConstraints
from cuperiod.prewhiten.fit import MultiSineFit, fit_multisine
from cuperiod.prewhiten.session import PreWhitenSession
from cuperiod.prewhiten.subsets import Partition
from cuperiod.prewhiten.uncertainty import component_uncertainties

TWO_PI = 2.0 * np.pi
F1, F2 = 4.37, 9.83  # F2 carries no signal


def _data(noise: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two seasons, a 0.02 sinusoid at F1 on a 10.0 offset, nothing at F2."""
    rng = np.random.default_rng(3)
    t = np.sort(np.concatenate([rng.uniform(0, 60, 300), rng.uniform(400, 460, 300)]))
    y = 10.0 + 0.02 * np.sin(TWO_PI * F1 * t + 0.4)
    if noise:
        y = y + rng.normal(0.0, noise, t.size)
    return t, y, (t > 200.0).astype(np.int64)


def _fit(
    t: np.ndarray,
    y: np.ndarray,
    seasons: np.ndarray,
    second: ComponentConstraint,
    refine: str = "none",
) -> MultiSineFit:
    constraints = FitConstraints(
        components=(ComponentConstraint(frequency="fixed"), second),
        variation_groups=seasons,
    )
    return fit_multisine(
        t, y, None, [F1, F2], refine=refine, t_ref=230.0, constraints=constraints
    )


def _assert_values_finite(fit: MultiSineFit) -> None:
    for name in ("frequency", "amplitude", "phase", "residuals"):
        assert np.all(np.isfinite(getattr(fit, name))), name
    assert np.isfinite(fit.offset) and np.isfinite(fit.rss)


CASES = [
    pytest.param("amplitude", "fixed", "none", id="per-subset-A-common-phi"),
    pytest.param("phase", "fixed", "none", id="common-A-per-subset-phi"),
    pytest.param("amplitude", "free", "simultaneous", id="per-subset-A-free-f"),
    pytest.param("phase", "free", "simultaneous", id="per-subset-phi-free-f"),
    pytest.param("phase", "free", "cyclic", id="per-subset-phi-cyclic"),
]


@pytest.mark.parametrize(("variation", "frequency", "refine"), CASES)
def test_a_collapsed_amplitude_leaves_its_phase_nan_and_flagged(
    variation: Any, frequency: Any, refine: str
) -> None:
    t, y, seasons = _data()
    second = ComponentConstraint(frequency=frequency, variation=variation)
    fit = _fit(t, y, seasons, second, refine)
    detail = fit.detail
    assert detail is not None
    _assert_values_finite(fit)  # the trust region copes: no crash, no NaN values
    assert abs(fit.amplitude[1]) < 1e-12  # the amplitude did collapse (round-off)
    assert "degenerate" in detail.flags[1]
    assert np.isnan(fit.phase_error[1])
    assert detail.subset_phase_error is not None
    assert np.all(np.isnan(detail.subset_phase_error[1]))
    if frequency == "free":
        # The frequency of a component with no amplitude is undefined too.
        assert np.isnan(fit.frequency_error[1])
    # The amplitude itself stays measurable: its column does not scale with A.
    assert np.isfinite(fit.amplitude_error[1])
    assert detail.rank < detail.n_columns
    # The live component is untouched by its neighbour's collapse.
    assert "degenerate" not in detail.flags[0]
    assert np.isfinite(fit.amplitude_error[0]) and np.isfinite(fit.phase_error[0])
    assert fit.amplitude[0] == pytest.approx(0.02, rel=1e-9)


def test_constant_data_flags_every_free_phase() -> None:
    t, _, seasons = _data()
    second = ComponentConstraint(frequency="fixed", variation="amplitude")
    fit = _fit(t, np.full(t.size, 10.0), seasons, second)
    detail = fit.detail
    assert detail is not None
    _assert_values_finite(fit)
    assert all("degenerate" in flags for flags in detail.flags)
    assert np.all(np.isnan(fit.phase_error))
    assert np.all(np.isfinite(fit.amplitude_error))


@pytest.mark.parametrize("variation", ["amplitude", "phase"])
def test_a_noise_level_amplitude_is_not_degenerate(variation: Any) -> None:
    # No signal at F2, but noise: the amplitude is small and insignificant, not
    # round-off, so its phase is poorly known (a large but finite error), never NaN.
    t, y, seasons = _data(noise=1e-3)
    second = ComponentConstraint(frequency="fixed", variation=variation)
    fit = _fit(t, y, seasons, second)
    detail = fit.detail
    assert detail is not None
    assert not any("degenerate" in flags for flags in detail.flags)
    assert np.isfinite(fit.phase_error[1]) and fit.phase_error[1] > 0.1
    assert detail.rank == detail.n_columns


def test_a_fixed_amplitude_keeps_its_phase_identifiable_without_signal() -> None:
    # A fixed (non-zero) amplitude cannot collapse: the phase stays a well-defined,
    # finite parameter even where the data carry no signal at that frequency.
    t, y, seasons = _data()
    second = ComponentConstraint(frequency="fixed", amplitude=0.005)
    fit = _fit(t, y, seasons, second)
    detail = fit.detail
    assert detail is not None
    _assert_values_finite(fit)
    assert detail.flags[1] == {"fixed_f", "fixed_a"}
    assert fit.amplitude_error[1] == 0.0
    assert np.isfinite(fit.phase_error[1]) and fit.phase_error[1] > 0.0


@pytest.mark.parametrize("method", ["analytic", "bootstrap"])
@pytest.mark.parametrize("variation", ["amplitude", "phase"])
def test_every_estimator_reports_the_undefined_phase_as_nan(
    method: Any, variation: Any
) -> None:
    # The analytic formulae divide by the (round-off) amplitude and the bootstrap
    # measures replicate scatter: both used to give the undefined phase a number.
    t, y, seasons = _data()
    second = ComponentConstraint(frequency="fixed", variation=variation)
    fit = _fit(t, y, seasons, second)
    errors = component_uncertainties(t, None, fit, method=method, n_resamples=10)
    assert np.isnan(errors.phase[1])
    assert np.isfinite(errors.amplitude[1])
    assert np.isfinite(errors.phase[0]) and np.isfinite(errors.amplitude[0])
    if variation == "phase":
        assert errors.subset_phase is not None
        assert np.all(np.isnan(errors.subset_phase[1]))


@pytest.mark.parametrize("method", ["analytic", "bootstrap"])
def test_estimators_keep_their_numbers_where_the_fit_is_determined(
    method: Any,
) -> None:
    t, y, seasons = _data(noise=1e-3)
    second = ComponentConstraint(frequency="fixed", variation="phase")
    fit = _fit(t, y, seasons, second)
    errors = component_uncertainties(t, None, fit, method=method, n_resamples=10)
    assert np.all(np.isfinite(errors.phase)) and np.all(np.isfinite(errors.amplitude))
    assert errors.subset_phase is not None
    assert np.all(np.isfinite(errors.subset_phase[1]))


def test_ordinary_fits_take_the_unchanged_covariance_path() -> None:
    # With nothing at round-off the inputs come back as the very same objects, so
    # every covariance the oracle tests pin is computed exactly as before.
    setup: Any = SimpleNamespace(y=10.0 + np.zeros(50))
    natural = _constrained._Natural(
        amp=[np.array([0.02]), np.array([1e-4, -2e-4])],
        phase=[np.array([0.1]), np.array([0.2])],
        offsets=np.array([10.0]),
    )
    headline = [(0.02, 0.1, 0.02 + 0j), (1.5e-4, 0.2, 1.5e-4 + 0j)]
    same, heads = _constrained._resolved_amplitudes(setup, natural, headline)
    assert same is natural and heads is headline
    # A round-off entry (here 3e-15 against a scale of 10) is zeroed, the rest pass
    # through, and a zeroed headline keeps its phase.
    natural.amp[1][0] = 3e-15
    collapsed = [(0.02, 0.1, 0.02 + 0j), (2e-15, 0.2, 2e-15 + 0j)]
    zeroed, heads = _constrained._resolved_amplitudes(setup, natural, collapsed)
    assert zeroed.amp[1][0] == 0.0 and zeroed.amp[1][1] == -2e-4
    assert zeroed.amp[0][0] == 0.02
    assert heads == [(0.02, 0.1, 0.02 + 0j), (0.0, 0.2, 0j)]


@pytest.mark.parametrize("estimator", ["covariance", "analytic"])
def test_a_session_reports_the_collapsed_component_as_degenerate(
    estimator: Any,
) -> None:
    t, y, _ = _data()
    session = PreWhitenSession(
        (t, y, np.full(t.size, 1e-3)),
        settings=PreWhitenSettings(maximum_frequency=20.0, uncertainty=estimator),
        partitions=[Partition.from_ranges(t, [200.0], name="season")],
    )
    assert session.step().kind == "accepted"
    assert session.add(F2, snap=False, refine="none").kind == "edited"
    assert session.use_variation("season").kind == "edited"
    for mode in ("amplitude", "phase"):
        assert session.set_variation(2, mode).kind == "edited"
        by_label = {c.label: c for c in session.result()}
        collapsed = by_label["F2"]
        assert "degenerate" in collapsed.flags
        assert np.isnan(collapsed.phase_error)
        assert np.isfinite(collapsed.amplitude) and np.isfinite(collapsed.phase)
        assert "degenerate" not in by_label["F1"].flags
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "collapsed.cupw.json"
        session.save(path)
        reloaded = PreWhitenSession.load(path)
        flags = {c.label: c.flags for c in reloaded.result()}
    assert "degenerate" in flags["F2"]
