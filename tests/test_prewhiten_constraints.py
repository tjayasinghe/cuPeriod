"""Constraint types: tie grammar, tie resolution and the validation table (C0)."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.core.errors import CuPeriodError
from cuperiod.prewhiten.constraints import (
    ComponentConstraint,
    ConstraintError,
    FitConstraints,
    FitStart,
    FrequencyTie,
    flatten_ties,
    format_tie,
    parse_tie,
)

# --- grammar --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "ids", "coefficients"),
    [
        ("f1+f2", (1, 2), (1, 1)),
        ("F1 + F2", (1, 2), (1, 1)),
        ("2f1", (1,), (2,)),
        ("2F1", (1,), (2,)),
        ("2*f1 - f3", (1, 3), (2, -1)),
        ("2 * F1 - F3", (1, 3), (2, -1)),
        ("F1 - F2", (1, 2), (1, -1)),
        ("F1 − F2", (1, 2), (1, -1)),
        ("F3 = F1 + F2", (1, 2), (1, 1)),
        ("f3=f1+f2", (1, 2), (1, 1)),
        ("f1+f1", (1,), (2,)),
        ("f1 + f2 - f2", (1,), (1,)),
        ("-F1 + 3F2", (1, 2), (-1, 3)),
        ("+F12", (12,), (1,)),
        ("  F1\t+\nF2 ", (1, 2), (1, 1)),
        ("10F1 + 10F2", (1, 2), (10, 10)),
    ],
)
def test_parse_tie_accepts(text: str, ids: tuple[int, ...],
                           coefficients: tuple[int, ...]) -> None:
    assert parse_tie(text) == (ids, coefficients)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("0.5f1", "non-integer coefficient '0.5'"),
        ("1.0F1", "non-integer coefficient"),
        ("2e1F1", "non-integer coefficient"),
        ("F1 + 2", "bare number '2'"),
        ("3", "bare number"),
        ("f1 f2", "missing operator"),
        ("F1F2", "missing operator"),
        ("f1 +", "trailing operator"),
        ("f1 -", "trailing operator"),
        ("f1-f1", "sums to zero"),
        ("11F1 + 10F2", "order 21 > 20"),
        ("100F1", "outside 1..99"),
        ("0F1", "outside 1..99"),
        ("F0", "no component 'F0'"),
        ("", "empty tie"),
        ("   ", "empty tie"),
        ("F1 + G2", "unexpected 'G'"),
        ("F1 / F2", "unexpected '/'"),
        ("F1 + + F2", "unexpected '+'"),
        ("F3 = ", "empty right-hand side"),
        ("F1 + F2 = F3", "left-hand side"),
        ("F3 = F1 = F2", "more than one '='"),
        ("F3 = F1 + F3", "cyclic tie: F3 -> F3"),
        ("F1 + " * 50 + "F2", "longer than 200"),
        ("__import__('os')", "unexpected"),
    ],
)
def test_parse_tie_rejects(text: str, fragment: str) -> None:
    with pytest.raises(ConstraintError) as info:
        parse_tie(text)
    assert fragment in str(info.value)


def test_parse_tie_checks_the_edited_component() -> None:
    assert parse_tie("F3 = F1 + F2", child=3) == ((1, 2), (1, 1))
    assert parse_tie("F1 + F2", child=3) == ((1, 2), (1, 1))
    with pytest.raises(ConstraintError, match="defines F4, not the edited"):
        parse_tie("F4 = F1 + F2", child=3)
    with pytest.raises(ConstraintError, match="cyclic tie: F3 -> F3"):
        parse_tie("F1 + F3", child=3)


def test_constraint_error_is_a_cuperiod_value_error() -> None:
    assert issubclass(ConstraintError, CuPeriodError)
    assert issubclass(ConstraintError, ValueError)


@pytest.mark.parametrize(
    "text",
    ["F1 + F2", "2F1 - F3", "F2", "3F4 + F1 - 2F2", "-F1", "-F1 - 2F3", "F10 + F1"],
)
def test_format_parse_round_trip(text: str) -> None:
    ids, coefficients = parse_tie(text)
    rendered = format_tie(ids, coefficients)
    again = parse_tie(rendered)
    assert dict(zip(*again, strict=True)) == dict(zip(ids, coefficients, strict=True))


def test_format_tie_puts_positive_terms_first() -> None:
    assert format_tie((3, 1), (-1, 2)) == "2F1 - F3"
    assert format_tie((1, 2), (1, 1)) == "F1 + F2"
    assert format_tie((1,), (-1,)) == "-F1"
    assert format_tie((1, 2, 3), (1, 0, 1)) == "F1 + F3"


def test_random_strings_only_raise_constraint_errors() -> None:
    rng = np.random.default_rng(5)
    alphabet = list("fF0123456789+-*= .−()/xe") + ["٣", "\U0001f600"]
    for _ in range(3000):
        length = int(rng.integers(0, 24))
        text = "".join(rng.choice(alphabet, size=length))
        try:
            ids, coefficients = parse_tie(text)
        except ConstraintError:
            continue
        assert ids and len(ids) == len(coefficients)
        assert all(n != 0 for n in coefficients)
        assert sum(abs(n) for n in coefficients) <= 20


def test_parse_tie_rejects_non_text() -> None:
    with pytest.raises(ConstraintError):
        parse_tie(12)  # type: ignore[arg-type]


# --- flattening -------------------------------------------------------------------


def test_chains_flatten() -> None:
    ties = {4: ((3, 1), (1, 1), 0.0), 3: ((1,), (2,), 0.0)}
    flat = flatten_ties(ties)
    assert flat[4] == ((1,), (3,), 0.0)
    assert flat[3] == ((1,), (2,), 0.0)


def test_chains_accumulate_constants_and_cancel_terms() -> None:
    ties = {
        5: ((3, 4), (1, -1), 0.5),
        3: ((1, 2), (1, 1), 0.25),
        4: ((1, 2), (1, 1), 0.0),
    }
    parents, coefficients, constant = flatten_ties(ties)[5]
    assert parents == () and coefficients == ()
    assert constant == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("ties", "message"),
    [
        ({3: ((3,), (1,), 0.0)}, "cyclic tie: F3 -> F3"),
        ({3: ((5,), (1,), 0.0), 5: ((3,), (1,), 0.0)}, "cyclic tie: F3 -> F5 -> F3"),
        (
            {2: ((1,), (1,), 0.0), 3: ((4,), (1,), 0.0), 4: ((6,), (2,), 0.0),
             6: ((3,), (1,), 0.0)},
            "cyclic tie: F3 -> F4 -> F6 -> F3",
        ),
    ],
)
def test_cycles_are_named(ties, message: str) -> None:
    with pytest.raises(ConstraintError) as info:
        flatten_ties(ties)
    assert str(info.value) == message


# --- types --------------------------------------------------------------------------


def test_frequency_tie_from_text_uses_positions() -> None:
    tie = FrequencyTie.from_text("F1 + F2")
    assert tie.parents == (0, 1) and tie.coefficients == (1, 1)
    assert tie.text() == "F1 + F2"
    assert FrequencyTie.from_text("F3 = 2F1").parents == (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parents": (0, 1), "coefficients": (1,)},
        {"parents": (0,), "coefficients": (0,)},
        {"parents": (-1,), "coefficients": (1,)},
        {"parents": (0, 0), "coefficients": (1, 1)},
        {"parents": (0,), "coefficients": (1,), "constant": float("nan")},
        {"parents": (0,), "coefficients": (1.5,)},
    ],
)
def test_frequency_tie_rejects(kwargs) -> None:
    with pytest.raises(ConstraintError):
        FrequencyTie(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"amplitude": 0.01, "variation": "amplitude"}, "per-subset amplitude"),
        ({"amplitude": 0.01, "variation": "both"}, "per-subset amplitude"),
        ({"phase": 1.0, "variation": "phase"}, "per-subset phase"),
        ({"phase": 1.0, "variation": "both"}, "per-subset phase"),
        ({"amplitude": 0.0}, "fixed zero amplitude"),
        ({"amplitude": float("inf")}, "finite"),
        ({"phase": float("nan")}, "finite"),
        ({"variation": "season"}, "unknown variation"),
        ({"frequency": "pinned"}, "frequency must be"),
    ],
)
def test_component_constraint_rejects(kwargs, fragment: str) -> None:
    with pytest.raises(ConstraintError, match=fragment):
        ComponentConstraint(**kwargs)


def test_every_legal_class_constructs() -> None:
    legal = [
        {},
        {"phase": 1.0},
        {"amplitude": 0.01},
        {"amplitude": 0.01, "phase": 1.0},
        {"amplitude": 0.0, "phase": 1.0},
        {"variation": "amplitude"},
        {"variation": "amplitude", "phase": 1.0},
        {"variation": "phase"},
        {"variation": "phase", "amplitude": 0.01},
        {"variation": "both"},
    ]
    for kwargs in legal:
        ComponentConstraint(**kwargs)


def test_trivial_and_free() -> None:
    assert FitConstraints().is_trivial()
    assert FitConstraints.free(3).is_trivial()
    assert len(FitConstraints.free(3).components) == 3
    start = FitStart(amplitude=np.ones(2), phase=np.zeros(2))
    assert FitConstraints(components=(ComponentConstraint(),) * 2,
                          start=start).is_trivial()
    assert not FitConstraints(
        components=(ComponentConstraint(frequency="fixed"),)
    ).is_trivial()
    assert not FitConstraints(point_weights=np.ones(3)).is_trivial()
    assert not FitConstraints(offset_groups=np.zeros(3, dtype=int)).is_trivial()
    assert not FitConstraints(variation_groups=np.zeros(3, dtype=int)).is_trivial()


# --- resolution ---------------------------------------------------------------------


def test_resolve_free_fixed_and_tied() -> None:
    freqs = np.array([5.0, 7.0, 99.0, 3.0, 99.0])
    constraints = FitConstraints(
        components=(
            ComponentConstraint(),
            ComponentConstraint(),
            ComponentConstraint(frequency=FrequencyTie((0, 1), (1, 1))),
            ComponentConstraint(frequency="fixed"),
            ComponentConstraint(frequency=FrequencyTie((2, 3), (1, -1), 0.5)),
        )
    )
    resolved = constraints.resolve(freqs)
    assert resolved.status == ("free", "free", "tied", "fixed", "tied")
    assert resolved.free_index.tolist() == [0, 1]
    assert resolved.matrix.tolist() == [[1, 0], [0, 1], [1, 1], [0, 0], [1, 1]]
    assert resolved.constant.tolist() == [0.0, 0.0, 0.0, 3.0, -2.5]
    assert resolved.frequencies(resolved.theta(freqs)).tolist() == [
        5.0, 7.0, 12.0, 3.0, 9.5,
    ]


def test_all_parents_fixed_makes_the_child_constant() -> None:
    constraints = FitConstraints(
        components=(
            ComponentConstraint(frequency="fixed"),
            ComponentConstraint(frequency=FrequencyTie((0,), (2,))),
        )
    )
    resolved = constraints.resolve(np.array([4.0, 0.0]))
    assert resolved.n_free == 0
    assert resolved.matrix.shape == (2, 0)
    assert resolved.constant.tolist() == [4.0, 8.0]
    assert resolved.status == ("fixed", "tied")


def test_resolve_rejects_parents_outside_the_fit_and_cycles() -> None:
    outside = FitConstraints(
        components=(ComponentConstraint(frequency=FrequencyTie((3,), (1,))),)
    )
    with pytest.raises(ConstraintError, match="not in this fit"):
        outside.resolve(np.array([1.0]))
    cyclic = FitConstraints(
        components=(
            ComponentConstraint(frequency=FrequencyTie((1,), (1,))),
            ComponentConstraint(frequency=FrequencyTie((0,), (2,))),
        )
    )
    with pytest.raises(ConstraintError, match="cyclic tie: F1 -> F2 -> F1"):
        cyclic.resolve(np.array([1.0, 2.0]))
    selfish = FitConstraints(
        components=(ComponentConstraint(frequency=FrequencyTie((0,), (2,))),)
    )
    with pytest.raises(ConstraintError, match="cyclic tie: F1 -> F1"):
        selfish.resolve(np.array([1.0]))


# --- validate: every row of the edge-case table that needs no fit ---------------------


def _ok(constraints: FitConstraints, k: int = 2, n: int = 40, *, fit_mean: bool = True,
        t_ref: float | None = 0.0) -> None:
    constraints.validate(np.linspace(5.0, 6.0, k), n, fit_mean=fit_mean, t_ref=t_ref)


def _bad(constraints: FitConstraints, fragment: str, k: int = 2, n: int = 40, *,
         fit_mean: bool = True, t_ref: float | None = 0.0) -> None:
    with pytest.raises(ConstraintError, match=fragment):
        _ok(constraints, k, n, fit_mean=fit_mean, t_ref=t_ref)


def test_validate_accepts_a_full_constraint_set() -> None:
    n = 60
    offsets = np.repeat(np.arange(6), 10)
    seasons = np.repeat(np.arange(3), 20)
    constraints = FitConstraints(
        components=(
            ComponentConstraint(variation="amplitude"),
            ComponentConstraint(phase=1.0),
            ComponentConstraint(frequency=FrequencyTie((0, 1), (1, 1))),
            ComponentConstraint(variation="both", frequency="fixed"),
        ),
        offset_groups=offsets,
        variation_groups=seasons.astype(np.int32),
        point_weights=np.linspace(0.5, 2.0, n),
        start=FitStart(amplitude=np.ones(4), phase=np.zeros(4),
                       subset_amplitude=np.ones((4, 3))),
    )
    constraints.validate(np.array([5.0, 7.0, 12.0, 9.0]), n, fit_mean=True, t_ref=0.0)


def test_validate_fixed_phase_needs_t_ref() -> None:
    constraints = FitConstraints(components=(ComponentConstraint(phase=0.5),) * 2)
    _bad(constraints, "explicit t_ref", t_ref=None)
    _ok(constraints, t_ref=2459000.0)


def test_validate_component_count() -> None:
    _bad(FitConstraints.free(3), "3 component constraints for 2 frequencies")


def test_validate_non_finite_frequencies() -> None:
    with pytest.raises(ConstraintError, match="finite"):
        FitConstraints.free(1).validate(np.array([np.nan]), 10, fit_mean=True,
                                        t_ref=0.0)


def test_validate_non_positive_child() -> None:
    constraints = FitConstraints(
        components=(
            ComponentConstraint(),
            ComponentConstraint(),
            ComponentConstraint(frequency=FrequencyTie((0, 1), (1, -1))),
        )
    )
    with pytest.raises(ConstraintError, match="not positive"):
        constraints.validate(np.array([5.0, 7.0, 1.0]), 40, fit_mean=True, t_ref=0.0)
    constraints.validate(np.array([7.0, 5.0, 1.0]), 40, fit_mean=True, t_ref=0.0)


def test_validate_duplicate_rows() -> None:
    twins = FitConstraints(
        components=(
            ComponentConstraint(),
            ComponentConstraint(frequency=FrequencyTie((0,), (2,))),
            ComponentConstraint(frequency=FrequencyTie((0,), (2,))),
        )
    )
    with pytest.raises(ConstraintError, match="F2 and F3 have identical"):
        twins.validate(np.array([5.0, 0.0, 0.0]), 40, fit_mean=True, t_ref=0.0)
    fixed = ComponentConstraint(frequency="fixed")
    fixed_twins = FitConstraints(components=(fixed, fixed))
    with pytest.raises(ConstraintError, match="identical"):
        fixed_twins.validate(np.array([5.0, 5.0]), 40, fit_mean=True, t_ref=0.0)
    fixed_twins.validate(np.array([5.0, 6.0]), 40, fit_mean=True, t_ref=0.0)


def test_validate_group_codes() -> None:
    _bad(FitConstraints(offset_groups=np.zeros(39, dtype=int)), "one code per point")
    _bad(FitConstraints(offset_groups=np.full(40, 0.0)), "must be integers")
    _bad(FitConstraints(offset_groups=np.full(40, -1)), "negative")
    gap = np.repeat([0, 2], 20)
    _bad(FitConstraints(offset_groups=gap), r"0..S-1 with every group non-empty; "
         r"missing \[1\]")
    _ok(FitConstraints(offset_groups=np.repeat([0, 1], 20)))


def test_validate_offsets_need_fit_mean() -> None:
    _bad(FitConstraints(offset_groups=np.repeat([0, 1], 20)), "fit_mean=True",
         fit_mean=False)


def test_validate_variation_needs_groups() -> None:
    varying = (ComponentConstraint(variation="phase"), ComponentConstraint())
    _bad(FitConstraints(components=varying), "no variation_groups")
    _bad(FitConstraints(components=varying, variation_groups=np.zeros(40, dtype=int)),
         "at least two variation subsets")
    _ok(FitConstraints(components=varying, variation_groups=np.repeat([0, 1], 20)))
    # Groups without any varying component are harmless.
    _ok(FitConstraints(variation_groups=np.zeros(40, dtype=int)))


def test_validate_small_variation_subsets_point_to_merged_small() -> None:
    varying = (ComponentConstraint(variation="both"), ComponentConstraint())
    codes = np.array([0] * 34 + [1] * 6)
    _bad(FitConstraints(components=varying, variation_groups=codes),
         r"variation subsets \[1\] have fewer than 10 points.*merged_small")
    _ok(FitConstraints(components=varying, variation_groups=codes,
                       min_subset_points=5))
    # Points must also outnumber the subset's private parameters.
    many = tuple(ComponentConstraint(variation="both") for _ in range(4))
    _bad(FitConstraints(components=many, variation_groups=codes, min_subset_points=2),
         "fewer than 9 points", k=4)


def test_validate_weights() -> None:
    _bad(FitConstraints(point_weights=np.ones(39)), "one value per point")
    for bad in (0.0, -1.0, np.inf, np.nan):
        _bad(FitConstraints(point_weights=np.r_[np.ones(39), bad]), "finite and pos")


def test_validate_start_shapes() -> None:
    _bad(FitConstraints(start=FitStart(amplitude=np.ones(3), phase=np.zeros(2))),
         r"start.amplitude must have shape \(2,\)")
    varying = (ComponentConstraint(variation="phase"), ComponentConstraint())
    bad_subsets = FitConstraints(
        components=varying, variation_groups=np.repeat([0, 1], 20),
        start=FitStart(np.ones(2), np.zeros(2), subset_phase=np.zeros((2, 3))),
    )
    _bad(bad_subsets, "subset_phase must have shape")


def test_validate_min_subset_points() -> None:
    _bad(FitConstraints(min_subset_points=0), "positive integer")
