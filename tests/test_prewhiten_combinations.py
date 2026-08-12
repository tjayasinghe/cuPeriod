"""Combination-frequency identification: correctness, ordering, and honesty."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.prewhiten.combinations import _coefficient_vectors, identify_combinations


def _labels(matches) -> list[str]:
    return [m.label for m in matches]


def test_identifies_a_harmonic_and_a_sum() -> None:
    frequency = np.array([5.0, 7.3, 10.0, 12.3])
    amplitude = np.array([1.0, 0.7, 0.3, 0.2])
    matches = identify_combinations(frequency, amplitude, rayleigh=0.01)
    assert _labels(matches) == ["F3 = 2F1", "F4 = F1 + F2"]
    assert all(m.order == 2 for m in matches)


def test_identifies_a_difference_with_readable_ordering() -> None:
    frequency = np.array([5.0, 7.3, 2.3])
    amplitude = np.array([1.0, 0.7, 0.1])
    (match,) = identify_combinations(frequency, amplitude, rayleigh=0.01)
    assert match.label == "F3 = F2 - F1"
    assert match.coefficients == (-1, 1)  # aligned with parents (F1, F2)
    assert match.parents == (0, 1)


def test_parents_must_be_stronger_than_the_child() -> None:
    # The strongest peak can never be explained by weaker ones.
    frequency = np.array([10.0, 5.0, 5.0000001])
    amplitude = np.array([1.0, 0.5, 0.4])
    matches = identify_combinations(frequency, amplitude, rayleigh=0.01)
    assert all(m.index != 0 for m in matches)


def test_lowest_order_wins_ties() -> None:
    # 10 = 2 x 5 (order 2) and also 5 + 5 ... the order-2 harmonic must be reported.
    frequency = np.array([5.0, 10.0])
    amplitude = np.array([1.0, 0.4])
    (match,) = identify_combinations(frequency, amplitude, rayleigh=0.01, max_order=3)
    assert match.order == 2
    assert match.label == "F2 = 2F1"


def test_tolerance_widens_with_the_propagated_uncertainty() -> None:
    frequency = np.array([5.0, 7.3, 12.32])  # 0.02 off the exact sum
    amplitude = np.array([1.0, 0.7, 0.2])
    tight = identify_combinations(
        frequency, amplitude, rayleigh=1e-6, tolerance_rayleigh=0.25, n_sigma=3.0
    )
    assert tight == ()
    loose = identify_combinations(
        frequency,
        amplitude,
        frequency_error=np.array([0.005, 0.005, 0.005]),
        rayleigh=1e-6,
        n_sigma=3.0,
    )
    assert _labels(loose) == ["F3 = F1 + F2"]


def test_nan_uncertainties_do_not_break_the_search() -> None:
    frequency = np.array([5.0, 7.3, 12.3])
    amplitude = np.array([1.0, 0.7, 0.2])
    matches = identify_combinations(
        frequency,
        amplitude,
        frequency_error=np.array([np.nan, np.nan, np.nan]),
        rayleigh=0.01,
    )
    assert _labels(matches) == ["F3 = F1 + F2"]


def test_chance_rate_is_reported_and_grows_with_the_tolerance() -> None:
    frequency = np.array([5.0, 7.3, 12.3])
    amplitude = np.array([1.0, 0.7, 0.2])
    (tight,) = identify_combinations(
        frequency, amplitude, rayleigh=0.001, tolerance_rayleigh=1.0
    )
    (loose,) = identify_combinations(
        frequency, amplitude, rayleigh=0.1, tolerance_rayleigh=1.0
    )
    assert 0.0 < tight.expected_false < loose.expected_false


def test_negative_predictions_are_not_matched() -> None:
    # A coefficient vector predicting a non-positive frequency is meaningless.
    frequency = np.array([5.0, 7.3, 0.0001])
    amplitude = np.array([1.0, 0.7, 0.1])
    matches = identify_combinations(frequency, amplitude, rayleigh=0.001)
    assert all(m.predicted > 0.0 for m in matches)


def test_custom_labels_flow_into_the_identification() -> None:
    frequency = np.array([5.0, 10.0])
    amplitude = np.array([1.0, 0.4])
    (match,) = identify_combinations(
        frequency, amplitude, rayleigh=0.01, labels=("nu1", "nu2")
    )
    assert match.label == "nu2 = 2nu1"


def test_degenerate_inputs_return_nothing() -> None:
    assert identify_combinations(np.array([5.0]), np.array([1.0])) == ()
    assert (
        identify_combinations(np.array([5.0, 10.0]), np.array([1.0, 0.4]), max_order=0)
        == ()
    )


def test_coefficient_enumeration_is_complete_and_bounded() -> None:
    vectors = list(_coefficient_vectors(2, 2))
    assert (2, 0) in vectors and (1, 1) in vectors and (-1, 1) in vectors
    assert (0, 0) not in vectors
    assert all(sum(abs(c) for c in v) <= 2 for v in vectors)
    assert len(list(_coefficient_vectors(5, 2))) < 200  # not the naive 5^5


def test_to_dict_is_json_friendly() -> None:
    import json

    frequency = np.array([5.0, 10.0])
    amplitude = np.array([1.0, 0.4])
    (match,) = identify_combinations(frequency, amplitude, rayleigh=0.01)
    payload = json.loads(json.dumps(match.to_dict()))
    assert payload["label"] == "F2 = 2F1"
    assert payload["coefficients"] == [2]
    assert payload["predicted"] == pytest.approx(10.0)
