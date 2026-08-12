"""g-mode period spacings: comb search, tilted series, échelle, buoyancy radius."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.prewhiten import prewhiten
from cuperiod.prewhiten.spacing import (
    buoyancy_radius,
    echelle,
    find_period_spacing,
    spacing_spectrum,
)
from synth import synthetic_gmode

_SECONDS_PER_DAY = 86400.0


def _comb(n: int = 14, first: float = 0.55, spacing: float = 0.03,
          slope: float = 0.0) -> np.ndarray:
    periods = [first]
    for _ in range(n - 1):
        periods.append(periods[-1] + spacing + slope * periods[-1])
    return np.asarray(periods)


# --- comb search -------------------------------------------------------------


def test_comb_finds_a_perfect_spacing() -> None:
    periods = _comb(spacing=0.03)
    result = spacing_spectrum(periods)
    assert result.best_spacing == pytest.approx(0.03, rel=0.02)
    assert result.best_power > 0.95
    assert result.n_values == periods.size
    assert result.spacing[0] < result.spacing[-1]  # ascending


def test_comb_survives_missing_radial_orders() -> None:
    periods = _comb(n=16, spacing=0.03)
    thinned = np.delete(periods, [4, 9])
    assert spacing_spectrum(thinned).best_spacing == pytest.approx(0.03, rel=0.03)


def test_comb_prefers_the_true_spacing_over_its_sub_multiples() -> None:
    # A comb of dP/2 fits every mode a comb of dP fits, so the two responses tie
    # exactly; only the wider one can be the real spacing.
    periods = _comb(n=14, spacing=0.03)
    result = spacing_spectrum(periods)
    assert result.best_spacing == pytest.approx(0.03, rel=0.02)
    half = np.argmin(np.abs(result.spacing - 0.015))
    assert result.power[half] == pytest.approx(result.best_power, rel=0.01)


def test_comb_response_is_low_for_random_periods() -> None:
    rng = np.random.default_rng(2)
    random_periods = np.sort(rng.uniform(0.5, 1.2, 14))
    assert spacing_spectrum(random_periods).best_power < 0.7


def test_comb_accepts_amplitude_weights() -> None:
    periods = _comb(spacing=0.03)
    weights = np.linspace(1.0, 0.2, periods.size)
    weighted = spacing_spectrum(periods, weights=weights)
    assert weighted.best_spacing == pytest.approx(0.03, rel=0.05)
    assert weighted.to_dict()["n_trials"] == weighted.size


def test_comb_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        spacing_spectrum(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="non-zero range"):
        spacing_spectrum(np.full(5, 1.0))
    with pytest.raises(ValueError, match="minimum_spacing"):
        spacing_spectrum(_comb(), minimum_spacing=1.0, maximum_spacing=0.5)
    with pytest.raises(ValueError, match="same length"):
        spacing_spectrum(_comb(), weights=np.ones(3))


# --- series extraction -------------------------------------------------------


def test_extracts_a_flat_series() -> None:
    periods = _comb(n=12, spacing=0.03)
    series = find_period_spacing(periods)
    assert series is not None
    assert series.n_modes == 12
    assert series.mean_spacing == pytest.approx(0.03, rel=1e-3)
    assert abs(series.slope) < 1e-3
    assert series.rms < 1e-6
    assert np.all(series.multiplicity == 1)


def test_extracts_a_tilted_series_and_recovers_the_slope() -> None:
    periods = _comb(n=16, spacing=0.028, slope=0.008)
    series = find_period_spacing(periods)
    assert series is not None
    assert series.n_modes == 16
    assert series.slope == pytest.approx(0.008, rel=0.05)


def test_bridges_missing_orders_and_records_the_multiplicity() -> None:
    periods = _comb(n=16, spacing=0.03)
    thinned = np.delete(periods, [5, 10])
    series = find_period_spacing(thinned)
    assert series is not None
    assert series.n_modes == 14
    assert 2 in set(series.multiplicity.astype(int))
    assert series.mean_spacing == pytest.approx(0.03, rel=0.02)


def test_intruder_modes_are_excluded_from_the_series() -> None:
    periods = _comb(n=14, spacing=0.03)
    polluted = np.concatenate([periods, [0.4137, 1.2916, 0.6231]])
    # As every caller does, weight by amplitude: the intruders are weak peaks.
    amplitudes = np.concatenate([np.linspace(0.006, 0.002, 14), [3e-4] * 3])
    series = find_period_spacing(polluted, amplitudes)
    assert series is not None
    # Indices are into the *input* array; the intruders sit at the end.
    assert max(series.indices) < periods.size
    assert series.n_modes == periods.size
    assert series.mean_spacing == pytest.approx(0.03, rel=0.01)


def test_a_chain_whose_every_step_skips_an_order_is_rescaled() -> None:
    # Handed a spacing half the truth, the series search must notice that no pair in
    # the chain is consecutive and widen the pattern rather than report dP/2.
    periods = _comb(n=12, spacing=0.03)
    series = find_period_spacing(periods, spacing=0.015)
    assert series is not None
    assert series.mean_spacing == pytest.approx(0.03, rel=1e-3)
    assert np.all(series.multiplicity == 1)


def test_indices_point_back_into_the_unsorted_input() -> None:
    periods = _comb(n=10, spacing=0.03)
    rng = np.random.default_rng(0)
    order = rng.permutation(periods.size)
    series = find_period_spacing(periods[order])
    assert series is not None
    assert np.allclose(np.sort(periods[order][list(series.indices)]), np.sort(periods))


def test_returns_none_when_no_series_is_long_enough() -> None:
    rng = np.random.default_rng(5)
    assert find_period_spacing(np.sort(rng.uniform(0.5, 5.0, 6))) is None
    assert find_period_spacing(np.array([0.5, 0.6])) is None


def test_an_explicit_spacing_skips_the_comb_search() -> None:
    periods = _comb(n=12, spacing=0.03)
    series = find_period_spacing(periods, spacing=0.03)
    assert series is not None and series.n_modes == 12
    assert find_period_spacing(periods, spacing=-1.0) is None


def test_amplitudes_must_match_the_periods() -> None:
    with pytest.raises(ValueError, match="same length"):
        find_period_spacing(_comb(), np.ones(3))


def test_settings_control_the_search() -> None:
    periods = _comb(n=12, spacing=0.03)
    strict = cup.SpacingSettings(min_length=20)
    assert find_period_spacing(periods, settings=strict) is None
    lenient = cup.SpacingSettings(min_length=4, ell=2)
    series = find_period_spacing(periods, settings=lenient)
    assert series is not None and series.ell == 2


def test_series_reporting_helpers() -> None:
    periods = _comb(n=12, spacing=0.03)
    series = find_period_spacing(periods)
    assert series is not None
    assert "Period spacing" in series.summary()
    payload = series.to_dict()
    assert payload["n_modes"] == 12
    assert payload["multiplicity"] == [1] * 11
    predicted = series.predicted_spacing(series.midpoints)
    assert np.allclose(predicted, series.spacings, rtol=0.05)


# --- helpers -----------------------------------------------------------------


def test_buoyancy_radius_follows_the_asymptotic_relation() -> None:
    assert buoyancy_radius(0.03, 1) == pytest.approx(0.03 * np.sqrt(2) * 86400.0)
    assert buoyancy_radius(0.03, 2) == pytest.approx(0.03 * np.sqrt(6) * 86400.0)
    with pytest.raises(ValueError):
        buoyancy_radius(0.03, 0)


def test_echelle_folds_a_perfect_comb_onto_one_ridge() -> None:
    periods = _comb(n=12, spacing=0.03)
    x, y = echelle(periods, 0.03)
    assert np.allclose(y, periods)
    assert float(np.ptp(x)) < 1e-9  # every mode lands on the same phase
    with pytest.raises(ValueError, match="positive"):
        echelle(periods, 0.0)


# --- end to end --------------------------------------------------------------


def test_spacing_recovered_from_a_pre_whitened_g_mode_star() -> None:
    time, value, error, periods = synthetic_gmode(n_modes=14)
    solution = prewhiten(
        (time, value, error),
        settings=cup.PreWhitenSettings(backend="finufft", max_frequencies=20),
    )
    assert solution.n_components >= 12
    independent = solution.independent()
    series = find_period_spacing(
        np.asarray([c.period for c in independent]),
        np.asarray([c.amplitude for c in independent]),
    )
    assert series is not None
    expected = float(np.mean(np.diff(periods)))
    assert series.mean_spacing == pytest.approx(expected, rel=0.05)
    assert series.slope == pytest.approx(0.008, abs=0.004)
    assert series.n_modes >= 12


# --- regressions -------------------------------------------------------------


def test_a_steeply_tilted_series_is_not_discarded() -> None:
    # Regression: the refit loop used to abort whenever the fitted *intercept* went
    # non-positive. The intercept is a nuisance parameter of dP = a + b*P, not a
    # spacing; only a + b*P over the observed range has to be positive, and it is.
    periods = np.cumsum(0.02 + 0.0015 * np.arange(20)) + 0.5
    series = find_period_spacing(periods)
    assert series is not None
    assert series.n_modes == 20
    assert series.intercept <= 0.0 or series.slope > 0.0
    assert np.all(series.predicted_spacing(series.midpoints) > 0.0)
    for gradient in (0.0005, 0.0010, 0.0012, 0.0015):
        tilted = np.cumsum(0.02 + gradient * np.arange(20)) + 0.5
        assert find_period_spacing(tilted) is not None, gradient


def test_a_tilted_series_survives_realistic_scatter() -> None:
    rng = np.random.default_rng(0)
    found = 0
    for _ in range(40):
        periods = np.cumsum(0.02 + 0.0015 * np.arange(20)) + 0.5
        periods = np.sort(periods + rng.normal(0.0, 20.0 / 86400.0, 20))
        if find_period_spacing(periods) is not None:
            found += 1
    assert found == 40


def test_a_wide_amplitude_spread_does_not_promote_past_the_true_spacing() -> None:
    # Regression: the sub-multiple promotion was judged on the amplitude-weighted
    # response, so dropping every other (weak) tooth barely lowered it and the search
    # was promoted to 2x the true spacing — doubling the reported dP and Pi_0.
    periods = 0.5 + 0.03 * np.arange(12)
    amplitudes = np.where(np.arange(12) % 2 == 0, 10.0, 0.5)
    assert spacing_spectrum(periods, weights=amplitudes).best_spacing == pytest.approx(
        0.03, rel=0.02
    )
    series = find_period_spacing(periods, amplitudes)
    assert series is not None
    assert series.mean_spacing == pytest.approx(0.03, rel=0.02)
    assert series.n_modes == 12
    assert series.buoyancy_radius == pytest.approx(
        buoyancy_radius(0.03, 1), rel=0.02
    )
