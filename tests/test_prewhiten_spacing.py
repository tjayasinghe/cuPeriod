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


@pytest.mark.parametrize("intruder", [5.3, 10.0, 25.0, 80.0])
def test_one_far_outlier_does_not_hide_the_comb(intruder: float) -> None:
    # PW-B-1: the default trial range was built on the raw span, so one long period
    # lifted the floor above the true spacing and opened the zero-lag lobe; the comb
    # was reported at ~the span and no series was found.
    periods = np.append(_comb(spacing=0.03), intruder)
    assert spacing_spectrum(periods).best_spacing == pytest.approx(0.03, rel=0.02)
    series = find_period_spacing(periods)
    assert series is not None and series.n_modes == 14
    assert series.mean_spacing == pytest.approx(0.03, rel=1e-3)
    weighted = find_period_spacing(
        periods, np.append(np.linspace(0.006, 0.002, 14), 3e-4)
    )
    assert weighted is not None and weighted.n_modes == 14


def test_a_tilted_comb_survives_a_far_outlier() -> None:
    periods = np.append(_comb(n=16, spacing=0.028, slope=0.008), 17.3)
    series = find_period_spacing(periods)
    assert series is not None and series.n_modes == 16
    assert series.slope == pytest.approx(0.008, rel=0.05)


def test_close_multiplets_keep_their_spacing_in_range() -> None:
    # A robust range must not shrink to the multiplet splitting either (a guard on the
    # PW-B-1 fix: a median-gap rule alone would cap the range at ~8x the splitting).
    from cuperiod.prewhiten.spacing import _default_spacing_range

    centres = _comb(n=10, spacing=0.03)
    for splitting in (0.0013, 0.0023, 0.004):
        periods = np.sort(
            np.concatenate([centres - splitting, centres, centres + splitting])
        )
        low, high = _default_spacing_range(periods)
        assert high == pytest.approx(float(np.ptp(periods)))  # the full range as before
        span = float(np.ptp(periods))
        assert low == pytest.approx(max(0.5 * splitting, span / (10 * periods.size)))
        assert spacing_spectrum(periods).spacing[-1] >= 0.03


def test_spacing_survives_a_slow_signal_in_the_pre_whitened_list() -> None:
    time, value, error, periods = synthetic_gmode(n_modes=14)
    value = value + 0.002 * np.sin(2 * np.pi * (time - time.min()) / 17.3)
    solution = prewhiten(
        (time, value, error),
        settings=cup.PreWhitenSettings(backend="finufft", max_frequencies=20),
    )
    assert max(solution.period) > 10.0  # the slow signal is in the list
    series = find_period_spacing(solution.period, solution.amplitude)
    assert series is not None and series.n_modes >= 12
    expected = float(np.mean(np.diff(periods)))
    assert series.mean_spacing == pytest.approx(expected, rel=0.05)


def test_non_finite_values_keep_their_weights_aligned() -> None:
    # PW-B-8: one NaN value with full-length weights raised 'same length'; a NaN or
    # inf weight made the whole response NaN and returned the full span.
    periods = 0.5 + 0.03 * np.arange(12)
    holed = periods.copy()
    holed[3] = np.nan
    result = spacing_spectrum(holed, weights=np.ones(12))
    assert result.best_spacing == pytest.approx(0.03, rel=0.02)
    assert result.n_values == 11
    for bad in (np.nan, np.inf):
        weights = np.ones(12)
        weights[2] = bad
        result = spacing_spectrum(periods, weights=weights)
        assert result.best_spacing == pytest.approx(0.03, rel=0.02)
        assert np.isfinite(result.best_power)
        assert np.all(np.isfinite(result.power))
    with pytest.raises(ValueError, match="same length"):
        spacing_spectrum(periods, weights=np.ones(11))


def _pick_on_the_full_grid(inverse_spacing, power, values, oversample):
    """The pre-optimisation _pick_spacing: the member response on the whole grid."""
    from cuperiod.prewhiten.spacing import (
        _COMB_MEMBER_TOL,
        _MAX_SUBMULTIPLE,
        _SUBMULTIPLE_RATIO,
        _comb_members,
        _comb_response,
    )

    best = int(np.argmax(power))
    members = _comb_members(values, inverse_spacing[best], _COMB_MEMBER_TOL)
    if members.size < 3:
        return best
    counting = _comb_response(inverse_spacing, members, np.ones_like(members))
    peak = float(counting[best])
    if peak <= 0.0:
        return best
    half_width = max(1, int(oversample))
    chosen = best
    for multiple in range(2, _MAX_SUBMULTIPLE + 1):
        target = inverse_spacing[best] / multiple
        if target < inverse_spacing[0]:
            break
        index = int(np.searchsorted(inverse_spacing, target))
        lo = max(0, index - half_width)
        hi = min(int(inverse_spacing.size), index + half_width + 1)
        if lo >= hi:
            continue
        local = lo + int(np.argmax(counting[lo:hi]))
        if counting[local] >= _SUBMULTIPLE_RATIO * peak:
            chosen = local
    return chosen


@pytest.mark.parametrize("oversample", [1, 20, 40])
def test_windowed_promotion_matches_the_full_grid(oversample: int, monkeypatch) -> None:
    # PERF-PREWHITEN-08: the promotion reads a few windows of the member response, so
    # only those are evaluated now.
    from cuperiod.prewhiten import spacing as spacing_module

    rng = np.random.default_rng(oversample)
    cases = [
        (_comb(n=12, spacing=0.03), None, 0.001),
        (np.concatenate([_comb(spacing=0.03), [0.4137, 1.2916, 0.6231]]),
         np.concatenate([np.linspace(0.006, 0.002, 14), [3e-4] * 3]), None),
        (np.sort(_comb(n=20, spacing=0.02) + rng.normal(0.0, 2e-4, 20)), None, None),
    ]
    rows = {"n": 0}
    original = spacing_module._comb_response

    def counting(inverse_spacing, values, weights):
        rows["n"] += int(inverse_spacing.size)
        return original(inverse_spacing, values, weights)

    for values, weights, minimum in cases:
        v = np.sort(values)
        a = np.ones(v.size) if weights is None else np.abs(weights[np.argsort(values)])
        lo, hi = spacing_module._default_spacing_range(v)
        lo = minimum or lo
        step = 1.0 / (oversample * float(v[-1] - v[0]))
        u = 1.0 / hi + step * np.arange(int(np.ceil((1.0 / lo - 1.0 / hi) / step)) + 1)
        power = original(u, v, a)
        expected = _pick_on_the_full_grid(u, power, v, oversample)
        monkeypatch.setattr(spacing_module, "_comb_response", counting)
        rows["n"] = 0
        assert spacing_module._pick_spacing(u, power, v, oversample) == expected
        assert rows["n"] <= 1 + 5 * (2 * oversample + 1)
        monkeypatch.setattr(spacing_module, "_comb_response", original)


def test_spacing_results_compare_by_identity_without_raising() -> None:
    # PW-B-10
    periods = _comb(n=12, spacing=0.03)
    first, second = spacing_spectrum(periods), spacing_spectrum(periods)
    assert first == first and (first == second) is False
    series_a, series_b = find_period_spacing(periods), find_period_spacing(periods)
    assert series_a is not None and series_b is not None
    assert (series_a == series_b) is False
    assert len({first, second, series_a, series_b}) == 4


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
