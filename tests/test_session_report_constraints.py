"""Reports of constrained session states: canonical and RMS amplitudes, flags."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
import prewhiten_corpus as pc
from cuperiod.prewhiten.session import PreWhitenSession
from cuperiod.prewhiten.session.report import rows
from cuperiod.prewhiten.subsets import Partition

CASE = pc.case_by_name("triple-last")
SETTINGS = CASE.resolved_settings()


def _three() -> PreWhitenSession:
    (t, y, e), _, _ = pc.load(CASE)
    session = PreWhitenSession((t, y, e), settings=SETTINGS)
    session.run(max_steps=3)
    return session


def _flip_phase(session: PreWhitenSession, component_id: int) -> float:
    """Fix a component's phase pi away from its fitted value; return the original."""
    original = session.state.component(component_id).phase
    with session.group():
        assert session.fix(component_id, phase=True).kind == "edited"
        flipped = float((original + np.pi) % (2.0 * np.pi))
        assert session.set_values(component_id, phase=flipped).kind == "edited"
    return original


def test_a_phase_fixed_pi_away_is_signed_in_the_state_and_canonical_in_the_result() -> (
    None
):
    session = _three()
    a1 = session.state.component(1).amplitude
    phase = _flip_phase(session, 1)
    head = session.state.component(1)
    assert head.amplitude < 0.0 and head.amplitude == pytest.approx(-a1, rel=1e-3)
    assert session.fit.amplitude[0] == head.amplitude  # MultiSineFit is signed too
    result = session.result()
    first = result[0]
    assert first.amplitude > 0.0 and first.amplitude == pytest.approx(-head.amplitude)
    assert first.phase == pytest.approx(phase, abs=1e-9)
    assert "negative_amplitude" in first.flags and first.fixed == "p"
    assert first.snr > SETTINGS.snr_threshold and not first.blended
    assert all(c.amplitude >= 0.0 for c in result)
    # F3 = 2F1 is identified with F1 as its parent: parents are ranked by |A|.
    labels = [m.label for m in result.combinations]
    assert "F3 = 2F1" in labels
    table = rows(session.kernel, session.state, session.derived)
    assert table[0].snr == pytest.approx(first.snr) and table[0].passes


def test_constraint_columns_appear_only_when_a_component_is_constrained() -> None:
    session = _three()
    plain = session.result().to_table()
    assert "fixed" not in plain[0] and "flags" not in plain[0]
    session.fix(2, frequency=True)
    constrained = session.result().to_table()
    assert constrained[1]["fixed"] == "f" and constrained[1]["flags"] == "fixed_f"
    assert constrained[0]["fixed"] == "" and constrained[0]["amplitude_rms"] is None


def test_tie_flags_follow_the_parents() -> None:
    session = _three()
    session.tie(3, "2F1")
    assert session.result()[2].flags == ("tied",)
    session.fix(1, frequency=True)
    row = session.result()[2]
    assert set(row.flags) == {"tied", "fixed_f"} and row.frequency_error == 0.0
    session.fix(1, frequency=False)
    session.set_active(1, False)
    row = session.result()[1]
    assert {"tied", "parent_inactive"} <= set(row.flags)
    assert row.frequency_error == 0.0


# --- a mode whose phase wanders between seasons ------------------------------------

#: Season phases with no linear trend (a trend would be a frequency shift).
WANDER = (0.0, 2.5, 0.3, 2.8)


def _wandering() -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], Partition, float]:
    rng = np.random.default_rng(3)
    t = np.sort(
        np.concatenate([s * 100.0 + rng.uniform(0.0, 30.0, 150) for s in range(4)])
    ) + 2459000.0
    season = ((t - t.min()) // 100.0).astype(int)
    tau = t - t.min()
    y = (
        10.0
        + 0.010 * np.sin(2.0 * np.pi * 7.3 * tau + np.asarray(WANDER)[season])
        + 0.006 * np.sin(2.0 * np.pi * 11.9 * tau + 1.0)
        + rng.normal(0.0, 0.001, t.size)
    )
    e = np.full(t.size, 0.001)
    seasons = Partition.from_gaps(t, name="season", min_gap=20.0)
    return (t, y, e), seasons, float(t.min())


WANDER_SETTINGS = cup.PreWhitenSettings(
    backend="numpy", maximum_frequency=15.0, samples_per_peak=3, max_frequencies=4
)


def test_a_phase_wandering_mode_is_judged_by_its_rms_amplitude() -> None:
    data, seasons, t0 = _wandering()
    session = PreWhitenSession(
        data, settings=WANDER_SETTINGS, partitions=[seasons], t_ref=t0
    )
    session.run(max_steps=2)
    wandering = min(session.components, key=lambda c: abs(c.frequency - 7.3)).id
    steady = next(c.id for c in session.components if c.id != wandering)
    assert session.use_variation("season").kind == "edited"
    report = session.set_variation(wandering, "phase", hold_frequency=False)
    assert report.kind == "edited"
    assert not session.state.component(wandering).fix_frequency
    result = session.finalize()
    by_label = {c.label: c for c in result}
    row = by_label[f"F{wandering}"]
    # Survives the final prune, judged by A_rms (the coherent mean collapses).
    assert row.variation == "phase" and row.amplitude_rms is not None
    assert row.amplitude_rms == pytest.approx(0.010, rel=0.05)
    assert row.amplitude < 0.6 * row.amplitude_rms
    assert row.snr >= WANDER_SETTINGS.snr_threshold
    coherent_snr = row.snr * row.amplitude / row.amplitude_rms
    assert row.snr > coherent_snr
    assert "weak_frequency" in row.flags
    assert by_label[f"F{steady}"].amplitude_rms is None  # common components: None
    assert result.subsets is not None and result.subsets.variation_by == "season"
    phases = result.subsets.phase[[c.label for c in result].index(row.label)]
    # With the frequency free, a linear phase trend and a frequency shift are the same
    # model (hence "weak_frequency"): the jumps are recovered up to a common offset.
    offset = np.angle(np.exp(1j * (np.diff(phases) - np.diff(WANDER))))
    np.testing.assert_allclose(offset, np.mean(offset), atol=0.15)
