"""PreWhitenSession.variations(): per-subset amplitudes and phases, constancy tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.prewhiten.session import PreWhitenSession
from cuperiod.prewhiten.session.state import SessionError
from cuperiod.prewhiten.session.variation import contrast, wald
from cuperiod.prewhiten.subsets import Partition

AMPLITUDES = np.array([0.010, 0.014, 0.008, 0.012])
PHASES = np.array([0.5, 1.0, 0.2, 2.0])
F1, F2 = 7.3, 11.9

SETTINGS = cup.PreWhitenSettings(
    backend="numpy", maximum_frequency=20.0, samples_per_peak=3, max_frequencies=4
)


def _seasons(
    seed: int, *, modulated: bool = True, per_season: int = 150, noise: float = 0.001
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], Partition, float]:
    rng = np.random.default_rng(seed)
    starts = 100.0 * np.arange(4)
    t = np.sort(
        np.concatenate([s + rng.uniform(0.0, 30.0, per_season) for s in starts])
    ) + 2459000.0
    season = ((t - t.min()) // 100.0).astype(int)
    tau = t - t.min()
    amplitude = AMPLITUDES[season] if modulated else np.full(t.size, 0.011)
    phase = PHASES[season] if modulated else np.full(t.size, 0.9)
    y = (
        10.0
        + amplitude * np.sin(2.0 * np.pi * F1 * tau + phase)
        + 0.006 * np.sin(2.0 * np.pi * F2 * tau + 1.0)
        + rng.normal(0.0, noise, t.size)
    )
    e = np.full(t.size, noise)
    seasons = Partition.from_gaps(t, name="season", min_gap=20.0)
    return (t, y, e), seasons, float(t.min())


def _session(seed: int = 7, **kw: Any) -> PreWhitenSession:
    """Both modes added at their true frequencies (held: nothing to refine)."""
    data, seasons, t0 = _seasons(seed, **kw)
    session = PreWhitenSession(
        data, settings=SETTINGS, partitions=[seasons], t_ref=t0
    )
    assert session.add(F1, snap=False, refine="none").component_id == 1
    assert session.add(F2, snap=False, refine="none").component_id == 2
    return session


def test_planted_season_amplitudes_and_phases_are_recovered() -> None:
    session = _session()
    head, log = session.state, session.history
    v = session.variations(mode="both", by="season")
    assert session.state is head and session.history == log  # a query
    assert v.component_ids == (1, 2) and v.by == "season" and v.mode == "both"
    assert v.subset_labels == ("N1", "N2", "N3", "N4")
    np.testing.assert_array_equal(v.n_points, [150, 150, 150, 150])
    assert np.all(np.diff(v.mid_time) > 90.0)
    assert np.all(np.abs(v.amplitude[0] - AMPLITUDES) < 4.0 * v.amplitude_error[0])
    dphi = np.angle(np.exp(1j * (v.phase[0] - PHASES)))
    assert np.all(np.abs(dphi) < 4.0 * v.phase_error[0])
    assert v.amplitude_p_value[0] < 1e-3 and v.phase_p_value[0] < 1e-3
    assert v.amplitude_statistic[0] > v.amplitude_statistic[1]
    assert v.amplitude_p_value[1] > 1e-3 and v.phase_p_value[1] > 1e-3
    assert v.delta_bic < 0.0 and not v.frequencies_improved
    np.testing.assert_array_equal(
        v.common_amplitude, [c.amplitude for c in session.components]
    )
    rows = v.to_table()
    assert len(rows) == 8 and rows[0]["label"] == "F1" and rows[0]["subset"] == "N1"


def test_modes_amplitude_and_phase() -> None:
    session = _session()
    amp = session.variations([1], mode="amplitude", by="season")
    assert amp.component_ids == (1,)
    assert np.all(np.isfinite(amp.amplitude_error[0]))
    assert np.all(np.isnan(amp.phase_error[0])) and np.isnan(amp.phase_p_value[0])
    assert np.ptp(amp.phase[0]) == 0.0  # the common phase, broadcast
    assert amp.amplitude_p_value[0] < 1e-3
    ph = session.variations(["F1"], mode="phase", by="season")
    assert np.all(np.isnan(ph.amplitude_error[0])) and ph.phase_p_value[0] < 1e-3
    improved = session.variations([1], mode="both", by="season",
                                  improve_frequencies=True)
    assert improved.frequencies_improved


def test_set_variation_is_the_persistent_query() -> None:
    session = _session()
    query = session.variations([1], mode="both", by="season")
    session.use_variation("season")
    assert session.set_variation(1, "both").kind == "edited"
    c1 = session.state.component(1)
    assert c1.subset_amplitude is not None and c1.subset_phase is not None
    np.testing.assert_allclose(c1.subset_amplitude, query.amplitude[0], rtol=1e-9)
    np.testing.assert_allclose(c1.subset_phase, query.phase[0], rtol=1e-9)
    # With the partition already the variation role, other varying components keep
    # their per-subset values in a query of a different component.
    again = session.variations([2], mode="amplitude")
    assert again.by == "season" and np.isfinite(again.amplitude_p_value[0])


def test_variation_query_errors() -> None:
    session = _session()
    with pytest.raises(SessionError, match="partition"):
        session.variations(mode="both")
    with pytest.raises(SessionError, match="mode"):
        session.variations(mode="none", by="season")  # type: ignore[arg-type]
    session.set_active(2, False)
    with pytest.raises(SessionError, match="inactive"):
        session.variations([2], mode="both", by="season")
    with pytest.raises(SessionError, match="stopped"):
        session.variations(mode="both", by="season", should_stop=lambda: True)
    with pytest.raises(SessionError, match="moon"):
        session.variations(mode="both", by="moon")


def test_wald_helpers() -> None:
    np.testing.assert_array_equal(contrast(3), [[-1, 1, 0], [0, -1, 1]])
    statistic, p = wald(np.array([1.0, 1.0, 1.0]), np.eye(3))
    assert statistic == 0.0 and p == 1.0
    statistic, p = wald(np.array([0.0, 2.0]), np.eye(2))
    assert statistic == pytest.approx(2.0) and 0.1 < p < 0.2
    # Phases: a jump across 2 pi is a small difference.
    statistic, _ = wald(np.array([0.05, 2 * np.pi - 0.05]), np.eye(2) * 0.01,
                        circular=True)
    assert statistic == pytest.approx(0.01 / 0.02)
    assert np.isnan(wald(np.array([1.0]), np.eye(1))[0])
    assert np.isnan(wald(np.array([1.0, np.nan]), np.eye(2))[0])


def test_wald_p_values_are_uniform_for_constant_modes() -> None:
    p_amplitude = []
    p_phase = []
    for seed in range(200):
        data, seasons, t0 = _seasons(
            1000 + seed, modulated=False, per_season=40, noise=0.003
        )
        session = PreWhitenSession(
            data, settings=SETTINGS, partitions=[seasons], t_ref=t0,
            grid=(5.0, 0.01, 200),
        )
        session.add(F1, snap=False, refine="none")
        session.add(F2, snap=False, refine="none")
        v = session.variations([1], mode="both", by="season")
        p_amplitude.append(v.amplitude_p_value[0])
        p_phase.append(v.phase_p_value[0])
    for values in (np.asarray(p_amplitude), np.asarray(p_phase)):
        assert np.all(np.isfinite(values))
        assert 0.4 < float(np.mean(values)) < 0.6
        assert 0.01 <= float(np.mean(values < 0.05)) <= 0.10
        assert float(np.mean(values < 0.5)) == pytest.approx(0.5, abs=0.1)
