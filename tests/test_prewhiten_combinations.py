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


# --- regressions ---------------------------------------------------------------


def test_a_near_duplicate_is_not_reported_as_an_order_one_combination() -> None:
    # PW-B-7: unit coefficient vectors made "F2 = F1" an order-1 "combination".
    assert identify_combinations(
        np.array([5.0, 5.002]), np.array([1.0, 0.5]), rayleigh=0.01
    ) == ()
    assert (
        identify_combinations(
            np.array([5.0, 10.0]), np.array([1.0, 0.4]), rayleigh=0.01, max_order=1
        )
        == ()
    )


def test_order_one_no_longer_pre_empts_a_real_sum() -> None:
    frequency = np.array([5.0, 7.3, 12.31, 12.3])
    matches = identify_combinations(
        frequency,
        np.array([1.0, 0.7, 0.3, 0.2]),
        frequency_error=np.array([1e-4, 1e-4, 4e-3, 4e-3]),
        rayleigh=0.001,
    )
    by_index = {m.index: m.label for m in matches}
    assert by_index[3] == "F4 = F1 + F2"
    assert all(m.order >= 2 for m in matches)
    assert all(sum(abs(c) for c in v) >= 2 for v in _coefficient_vectors(5, 3))


def test_expected_false_counts_only_positive_in_band_predictions() -> None:
    # PW-B-6: every coefficient vector was counted, although v and -v give at most one
    # positive prediction and most predictions fall outside the searched band.
    frequency = np.array([5.0, 10.0])
    (match,) = identify_combinations(frequency, np.array([1.0, 0.4]), rayleigh=0.01)
    tolerance = 0.25 * 0.01
    assert match.expected_false == pytest.approx(2.0 * tolerance / 5.0)


def test_expected_false_is_calibrated_against_monte_carlo() -> None:
    parents = np.array([5.0, 7.3, 9.1, 11.7, 13.2])
    amplitude = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.01])
    rng = np.random.default_rng(0)
    draws, hits, reported = 3000, 0, []
    for child in rng.uniform(5.0, 13.2, draws):
        matches = identify_combinations(
            np.append(parents, child), amplitude, rayleigh=0.08,
            tolerance_rayleigh=0.25,
        )
        found = [m for m in matches if m.index == 5]
        if found:
            hits += 1
            reported.append(found[0].expected_false)
    measured = hits / draws
    assert 0.02 < measured < 0.07  # ~0.043: order-2 sums/differences/harmonics
    assert np.mean(reported) == pytest.approx(measured, rel=0.5)


def _scalar_reference(frequency, amplitude, sigma, rayleigh, max_order, max_parents):
    """The pre-vectorisation loop (with the order >= 2 and in-band rules)."""
    k = frequency.size
    span = float(frequency.max() - frequency.min())
    floor = 0.25 * rayleigh
    lo, hi = float(frequency.min()), float(frequency.max())
    order_of = np.argsort(amplitude, kind="stable")[::-1]
    out = []
    for position, child in enumerate(order_of):
        if position == 0:
            continue
        parents = tuple(int(i) for i in order_of[: min(position, max_parents)])
        best = None
        window = 0.0
        for coefficients in _coefficient_vectors(len(parents), max_order):
            w = np.asarray(coefficients, dtype=np.float64)
            predicted = float(np.dot(w, frequency[list(parents)]))
            if predicted <= 0.0:
                continue
            parent_variance = np.dot(w**2, sigma[list(parents)] ** 2)
            tolerance = max(
                3.0 * float(np.sqrt(sigma[child] ** 2 + parent_variance)), floor
            )
            if lo <= predicted <= hi:
                window += 2.0 * tolerance
            residual = float(frequency[child]) - predicted
            if tolerance <= 0.0 or abs(residual) > tolerance:
                continue
            order = int(np.sum(np.abs(w)))
            if best is not None and (
                order > best[1] or (order == best[1] and abs(residual) >= abs(best[3]))
            ):
                continue
            best = (coefficients, order, predicted, residual, tolerance)
        if best is not None:
            out.append((int(child), parents, *best, window / span))
    del k
    return sorted(out)


@pytest.mark.parametrize("max_order", [2, 3])
def test_vectorised_search_matches_the_scalar_loop(max_order: int) -> None:
    # PW-B-12: the search is vectorised over the coefficient vectors now.
    rng = np.random.default_rng(max_order)
    for _ in range(25):
        k = int(rng.integers(3, 9))
        parents = rng.uniform(1.0, 8.0, 3)
        children = []
        for _ in range(k - 3):
            a, b = rng.choice(3, 2, replace=False)
            children.append(parents[a] + rng.choice([-1, 1]) * parents[b])
        frequency = np.abs(np.concatenate([parents, children])) + rng.normal(
            0.0, 1e-3, k
        )
        amplitude = np.sort(rng.uniform(0.01, 1.0, k))[::-1]
        sigma = rng.uniform(1e-4, 2e-3, k)
        rayleigh = 0.01
        reference = _scalar_reference(
            frequency, amplitude, sigma, rayleigh, max_order, 5
        )
        matches = identify_combinations(
            frequency, amplitude, frequency_error=sigma, rayleigh=rayleigh,
            max_order=max_order,
        )
        assert [m.index for m in matches] == [r[0] for r in reference]
        for m, r in zip(matches, reference, strict=True):
            assert m.parents == r[1] and m.coefficients == r[2] and m.order == r[3]
            if max_order == 2:
                assert (m.predicted, m.residual, m.tolerance) == (r[4], r[5], r[6])
            else:
                assert m.predicted == pytest.approx(r[4], rel=1e-12)
            assert m.expected_false == pytest.approx(r[7], rel=1e-12)


def test_to_dict_is_json_friendly() -> None:
    import json

    frequency = np.array([5.0, 10.0])
    amplitude = np.array([1.0, 0.4])
    (match,) = identify_combinations(frequency, amplitude, rayleigh=0.01)
    payload = json.loads(json.dumps(match.to_dict()))
    assert payload["label"] == "F2 = 2F1"
    assert payload["coefficients"] == [2]
    assert payload["predicted"] == pytest.approx(10.0)
