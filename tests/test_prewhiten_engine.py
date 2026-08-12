"""The extraction loop: recovery, stopping criteria, guards, and the result object."""

from __future__ import annotations

import json

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import uniform_frequency_grid
from cuperiod.prewhiten import default_prewhiten_grid, prewhiten
from synth import synthetic_pulsator, synthetic_sine


def _settings(**overrides) -> cup.PreWhitenSettings:
    defaults = {"backend": "finufft", "max_frequencies": 8}
    return cup.PreWhitenSettings(**{**defaults, **overrides})


def _noise_only(n: int = 1500, seed: int = 7):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0.0, 27.0, n)) + 2458000.0
    return time, 10.0 + rng.normal(0.0, 0.002, n), np.full(n, 0.002)


# --- recovery ----------------------------------------------------------------


def test_recovers_every_planted_frequency() -> None:
    time, value, error = synthetic_pulsator()
    solution = prewhiten((time, value, error), settings=_settings())
    assert solution.n_components == 3
    recovered = np.sort(solution.frequency)
    assert np.allclose(recovered, [12.34, 17.81, 24.68], atol=2e-3)
    assert np.allclose(np.sort(solution.amplitude)[::-1], [0.012, 0.007, 0.004],
                       rtol=0.1)
    assert np.all(solution.snr > 4.0)
    assert np.all(solution.frequency_error > 0.0)


def test_components_are_ranked_by_extraction_order() -> None:
    time, value, error = synthetic_pulsator()
    solution = prewhiten((time, value, error), settings=_settings())
    assert [c.rank for c in solution.components] == [1, 2, 3]
    assert [c.label for c in solution.components] == ["F1", "F2", "F3"]
    # The strongest planted mode is extracted first.
    assert solution.components[0].amplitude == max(solution.amplitude)


def test_residuals_are_consistent_with_the_reported_model() -> None:
    time, value, error = synthetic_pulsator()
    solution = prewhiten((time, value, error), settings=_settings())
    assert np.allclose(solution.residuals, value - solution.model(time), atol=1e-10)
    assert solution.rms < 2.0 * 0.0015
    assert solution.reduced_chi2 == pytest.approx(1.0, rel=0.2)


def test_works_without_error_bars() -> None:
    time, value, _ = synthetic_pulsator()
    solution = prewhiten((time, value), settings=_settings())
    assert solution.n_components >= 3
    assert np.all(np.isfinite(solution.frequency_error))


def test_a_light_curve_object_and_a_tuple_agree() -> None:
    time, value, error = synthetic_pulsator(n=800)
    settings = _settings(max_frequencies=2)
    from_tuple = prewhiten((time, value, error), settings=settings)
    from_object = prewhiten(
        cup.LightCurve.from_arrays(time, value, error), settings=settings
    )
    assert np.allclose(from_tuple.frequency, from_object.frequency)


# --- stopping criteria -------------------------------------------------------


def test_pure_noise_yields_no_components() -> None:
    solution = prewhiten(_noise_only(), settings=_settings(snr_threshold=4.5))
    assert solution.n_components == 0
    assert "S/N" in solution.stop_reason
    assert solution.components == ()


def test_max_frequencies_caps_the_run() -> None:
    time, value, error = synthetic_pulsator()
    solution = prewhiten((time, value, error), settings=_settings(max_frequencies=2))
    assert solution.n_components == 2
    assert "max_frequencies" in solution.stop_reason


def test_zero_max_frequencies_returns_an_offset_only_solution() -> None:
    time, value, error = synthetic_pulsator(n=400)
    solution = prewhiten((time, value, error), settings=_settings(max_frequencies=0))
    assert solution.n_components == 0
    assert solution.stop_reason == "max_frequencies is 0"
    assert np.isfinite(solution.offset)


def test_a_high_snr_threshold_stops_earlier() -> None:
    time, value, error = synthetic_pulsator()
    strict = prewhiten((time, value, error), settings=_settings(snr_threshold=80.0))
    lenient = prewhiten((time, value, error), settings=_settings(snr_threshold=4.0))
    assert strict.n_components < lenient.n_components


def test_bic_criterion_rejects_noise() -> None:
    solution = prewhiten(
        _noise_only(),
        settings=_settings(stop_criteria=("bic",), min_delta_bic=10.0),
    )
    assert solution.n_components == 0
    assert "BIC" in solution.stop_reason


def test_fap_criterion_accepts_a_strong_signal_and_rejects_noise() -> None:
    time, value, error = synthetic_pulsator(frequencies=(12.34,), amplitudes=(0.02,),
                                            phases=(0.5,))
    strong = prewhiten(
        (time, value, error), settings=_settings(stop_criteria=("fap",))
    )
    assert strong.n_components >= 1
    assert strong.components[0].fap < 1e-3
    quiet = prewhiten(_noise_only(), settings=_settings(stop_criteria=("fap",)))
    assert quiet.n_components == 0


def test_amplitude_floor_is_enforced() -> None:
    time, value, error = synthetic_pulsator()
    solution = prewhiten(
        (time, value, error),
        settings=_settings(stop_criteria=("amplitude",), min_amplitude=0.010),
    )
    assert solution.n_components == 1  # only the 0.012 mag mode clears the floor
    assert "amplitude" in solution.stop_reason


# --- guards ------------------------------------------------------------------


def test_no_two_components_are_unresolved_from_each_other() -> None:
    time, value, error = synthetic_pulsator(n=3000, span=27.0)
    solution = prewhiten((time, value, error), settings=_settings(max_frequencies=12))
    frequencies = np.sort(solution.frequency)
    if frequencies.size > 1:
        rayleigh = 1.0 / solution.baseline
        assert np.min(np.diff(frequencies)) >= 1.5 * rayleigh * 0.999


def test_amplitudes_stay_physical() -> None:
    # Regression: an unbounded refinement could slide one frequency onto another and
    # produce a pair of enormous, mutually cancelling components.
    time, value, error = synthetic_pulsator(n=4000, span=40.0)
    solution = prewhiten((time, value, error), settings=_settings(max_frequencies=15))
    assert np.max(solution.amplitude) < 10.0 * 0.012


def test_pruning_leaves_only_significant_components() -> None:
    time, value, error = synthetic_pulsator(n=2000, span=27.0)
    solution = prewhiten(
        (time, value, error), settings=_settings(max_frequencies=12, prune=True)
    )
    assert np.all(solution.snr >= 4.0)
    assert solution.n_pruned >= 0


def test_too_few_points_raises() -> None:
    time, value, error = synthetic_sine(n=8)
    with pytest.raises(InsufficientDataError, match="min_detections"):
        prewhiten((time, value, error), settings=_settings(min_detections=20))


def test_zero_baseline_raises() -> None:
    time = np.full(50, 2458000.0)
    with pytest.raises(InsufficientDataError):
        prewhiten((time, np.ones(50), np.full(50, 0.01)), settings=_settings())


def test_multiband_input_is_rejected_with_a_useful_message() -> None:
    time, value, error = synthetic_pulsator(n=200)
    mblc = cup.MultiBandLightCurve.from_light_curves(
        {"V": cup.LightCurve.from_arrays(time, value, error)}
    )
    with pytest.raises(ValueError, match="single band"):
        prewhiten(mblc, settings=_settings())


# --- knobs and plumbing ------------------------------------------------------


def test_custom_grid_is_honoured() -> None:
    time, value, error = synthetic_pulsator()
    grid = uniform_frequency_grid(
        float(time.max() - time.min()),
        maximum_frequency=15.0, minimum_frequency=1.0, samples_per_peak=8,
    )
    solution = prewhiten((time, value, error), settings=_settings(), grid=grid)
    assert np.all(solution.frequency < 15.1)  # 17.81 and 24.68 are out of band
    assert solution.spectrum is not None
    assert solution.spectrum.frequency[-1] == pytest.approx(grid.values[-1])


def test_default_grid_spans_one_over_baseline_to_pseudo_nyquist() -> None:
    time, value, error = synthetic_pulsator()
    lc = cup.LightCurve.from_arrays(time, value, error)
    grid = default_prewhiten_grid(lc)
    assert grid.uniform and grid.kind == "frequency"
    assert grid.values[0] == pytest.approx(1.0 / lc.baseline, rel=1e-6)
    assert grid.values[-1] > 24.68


def test_store_spectra_false_drops_the_big_arrays() -> None:
    time, value, error = synthetic_pulsator(n=400)
    solution = prewhiten(
        (time, value, error), settings=_settings(store_spectra=False, max_frequencies=1)
    )
    assert solution.spectrum is None and solution.residual_spectrum is None


def test_backend_argument_overrides_the_setting() -> None:
    time, value, error = synthetic_pulsator(n=400)
    solution = prewhiten(
        (time, value, error), settings=_settings(max_frequencies=1), backend="numpy"
    )
    assert solution.backend == "numpy"


def test_uncertainty_estimators_are_all_selectable() -> None:
    time, value, error = synthetic_pulsator(n=600, span=20.0)
    for method in ("covariance", "analytic", "bootstrap"):
        solution = prewhiten(
            (time, value, error),
            settings=_settings(
                max_frequencies=1, uncertainty=method, n_resamples=10
            ),
        )
        assert solution.uncertainty_method == method
        assert solution.components[0].frequency_error > 0.0


def test_harmonic_is_flagged_as_a_combination() -> None:
    # The default synthetic pulsator plants 24.68 = 2 x 12.34 exactly.
    time, value, error = synthetic_pulsator()
    solution = prewhiten((time, value, error), settings=_settings())
    labels = [c.combination for c in solution.components if c.combination]
    assert any("2F1" in label for label in labels)
    assert len(solution.independent()) == solution.n_components - len(labels)


def test_combinations_can_be_switched_off() -> None:
    time, value, error = synthetic_pulsator()
    solution = prewhiten((time, value, error), settings=_settings(combinations=False))
    assert solution.combinations == ()
    assert all(c.combination is None for c in solution.components)


# --- the result object -------------------------------------------------------


def test_result_serialization_round_trips_through_json() -> None:
    time, value, error = synthetic_pulsator(n=600)
    solution = prewhiten((time, value, error), settings=_settings(max_frequencies=2))
    payload = json.loads(json.dumps(solution.to_dict()))
    assert payload["n_components"] == solution.n_components
    assert payload["components"][0]["label"] == "F1"
    assert payload["stop_reason"] == solution.stop_reason


def test_result_table_and_dataframe_agree() -> None:
    time, value, error = synthetic_pulsator(n=600)
    solution = prewhiten((time, value, error), settings=_settings(max_frequencies=2))
    table = solution.to_table()
    frame = solution.to_dataframe()
    assert len(table) == len(frame) == solution.n_components
    assert frame["frequency"].iloc[0] == pytest.approx(table[0]["frequency"])


def test_result_sequence_protocol_and_summary() -> None:
    time, value, error = synthetic_pulsator(n=600)
    solution = prewhiten((time, value, error), settings=_settings(max_frequencies=2))
    assert len(solution) == solution.n_components
    assert list(solution)[0] is solution[0]
    text = solution.summary()
    assert "Pre-whitening" in text and "F1" in text
    assert "components" in repr(solution)


def test_period_and_its_uncertainty_are_derived_consistently() -> None:
    time, value, error = synthetic_pulsator(n=600)
    solution = prewhiten((time, value, error), settings=_settings(max_frequencies=1))
    component = solution.components[0]
    assert component.period == pytest.approx(1.0 / component.frequency)
    assert component.period_error == pytest.approx(
        component.frequency_error / component.frequency**2
    )


def test_top_level_api_exposes_prewhiten() -> None:
    time, value, error = synthetic_pulsator(n=400)
    solution = cup.prewhiten(
        (time, value, error), settings=cup.PreWhitenSettings(
            backend="finufft", max_frequencies=1
        )
    )
    assert isinstance(solution, cup.PreWhitenResult)


@requires_gpu
def test_gpu_and_cpu_solutions_agree() -> None:
    time, value, error = synthetic_pulsator(n=2000)
    cpu = prewhiten((time, value, error), settings=_settings())
    gpu = prewhiten((time, value, error), settings=_settings(backend="cufinufft"))
    assert gpu.n_components == cpu.n_components
    assert np.allclose(np.sort(gpu.frequency), np.sort(cpu.frequency), rtol=1e-9)
