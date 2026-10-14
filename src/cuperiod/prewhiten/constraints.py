"""Constraints on a multi-sinusoid fit: fixed values, frequency ties, subsets, weights.

Period04's frequency list lets every component's frequency, amplitude and phase be
"improved" or held, ties a combination frequency to its parents, fits one zero point
per subset of the data and lets amplitude and phase vary from subset to subset. This
module is the vocabulary of that model — plain immutable data, validated here, solved
by :mod:`cuperiod.prewhiten._constrained` when :func:`~cuperiod.prewhiten.fit.\
fit_multisine` receives a non-trivial :class:`FitConstraints`.

Everything lives in one object so a constraint set can be threaded unchanged through
the fit, its uncertainties, an interactive session and its project file:

* :class:`ComponentConstraint` — per sinusoid: a free, fixed or tied frequency; a
  fixed amplitude and/or phase stored *with its value* (a flag without a value could
  not be written down); and which of amplitude and phase take one value per
  variation subset.
* :class:`FrequencyTie` — ``f_child = constant + sum_j n_j f_parent_j`` with integer
  ``n_j``: harmonics and combination frequencies whose value follows their parents.
* :class:`FitConstraints` — the components plus two *independent* partitions of the
  points (zero points per ``offset_groups``, amplitude/phase per
  ``variation_groups``; a zero point per night and an amplitude per season is the
  normal case), relative point weights and optional starting phases.

Ties are written ``"F1 + F2"``, ``"2F1"``, ``"F3 = F1 - F2"`` (:func:`parse_tie`), never
evaluated: one regular-expression tokenizer over a capped input.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, get_args

import numpy as np
import numpy.typing as npt

from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.core.errors import CuPeriodError

#: Which of a component's amplitude and phase take one value per variation subset.
Variation = Literal["none", "amplitude", "phase", "both"]

#: Status of a component's frequency after the ties are resolved.
FrequencyStatus = Literal["free", "fixed", "tied"]

#: Integer group codes of any width (a partition stores ``int32``; callers may hold
#: ``int64``).
GroupCodes = npt.NDArray[np.integer[Any]]

#: Longest tie expression accepted (characters).
MAX_TIE_LENGTH: Final = 200

#: Largest literal coefficient of one term, and largest total order ``sum |n|``.
MAX_TIE_COEFFICIENT: Final = 99
MAX_TIE_ORDER: Final = 20

#: Every flag a component can carry (``FitDetail.flags``, ``Sinusoid.flags``).
FLAGS: Final = frozenset(
    {
        "fixed_f", "fixed_a", "fixed_phi", "tied", "parent_inactive",
        "negative_amplitude", "degenerate", "at_bound", "out_of_band",
        "unresolved", "low_frequency", "weak_frequency", "short_subset",
    }
)


class ConstraintError(CuPeriodError, ValueError):
    """A fit constraint is contradictory, cyclic, or unsupported by the data."""


def _label(index: int) -> str:
    """Display label of a 0-based position (``F1`` is position 0)."""
    return f"F{index + 1}"


# -- tie grammar -------------------------------------------------------------------

_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
    | (?P<label>[Ff]\d+)
    | (?P<number>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+)
    | (?P<sign>[+\-−])
    | (?P<star>\*)
    | (?P<equals>=)
    | (?P<bad>.)
    """,
    re.VERBOSE,
)


def _tokens(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for match in _TOKEN.finditer(text):
        kind = match.lastgroup or "bad"
        value = match.group()
        if kind == "space":
            continue
        if kind == "bad":
            raise ConstraintError(f"unexpected {value!r} in tie {text!r}")
        out.append((kind, value))
    return out


def _label_id(token: str, text: str) -> int:
    ident = int(token[1:])
    if ident < 1:
        raise ConstraintError(f"no component {token!r} in tie {text!r} (F1 is first)")
    return ident


def parse_tie(
    text: str, *, child: int | None = None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Parse a frequency tie into ``(ids, coefficients)``.

    Accepts ``"f1+f2"``, ``"2f1"``, ``"F1 - F2"``, ``"2*f1 - f3"`` and the
    ``"F3 = F1 + F2"`` form; labels are case-insensitive, ``U+2212`` is a minus sign,
    and repeated labels merge (``"f1+f1"`` is ``2F1``). Never evaluates anything.

    Parameters
    ----------
    text : str
        The expression (at most :data:`MAX_TIE_LENGTH` characters).
    child : int, optional
        The id of the component being tied. When given, a left-hand label must name
        it and the right-hand side may not reference it.

    Returns
    -------
    tuple
        Component ids (as written, ``F1`` -> 1) in order of first appearance, and
        their non-zero integer coefficients.

    Raises
    ------
    ConstraintError
        Naming the offending token: a non-integer coefficient (``0.5f1``), a bare
        number, a missing operator (``f1 f2``), a trailing operator, a net-zero result
        (``f1-f1``), a total order ``sum|n|`` above :data:`MAX_TIE_ORDER`, a left-hand
        label that is not ``child``, or a self reference.
    """
    if not isinstance(text, str):
        raise ConstraintError(f"a tie is text, not {type(text).__name__}")
    if len(text) > MAX_TIE_LENGTH:
        raise ConstraintError(f"tie longer than {MAX_TIE_LENGTH} characters")
    tokens = _tokens(text)
    if not tokens:
        raise ConstraintError("empty tie")
    if any(kind == "equals" for kind, _ in tokens):
        if len(tokens) < 2 or tokens[0][0] != "label" or tokens[1][0] != "equals":
            raise ConstraintError(
                f"the left-hand side of {text!r} must be a single label ('F3 = ...')"
            )
        lhs = _label_id(tokens[0][1], text)
        if child is not None and lhs != child:
            raise ConstraintError(
                f"tie {text!r} defines F{lhs}, not the edited component F{child}"
            )
        child = lhs if child is None else child
        tokens = tokens[2:]
        if any(kind == "equals" for kind, _ in tokens):
            raise ConstraintError(f"more than one '=' in tie {text!r}")
        if not tokens:
            raise ConstraintError(f"empty right-hand side in tie {text!r}")
    merged: dict[int, int] = {}
    position = 0
    first = True
    while position < len(tokens):
        sign = 1
        kind, value = tokens[position]
        if kind == "sign":
            sign = 1 if value == "+" else -1
            position += 1
        elif not first:
            raise ConstraintError(f"missing operator before {value!r} in tie {text!r}")
        if position >= len(tokens):
            raise ConstraintError(f"trailing operator in tie {text!r}")
        kind, value = tokens[position]
        coefficient = 1
        if kind == "number":
            if not re.fullmatch(r"\d+", value):
                raise ConstraintError(
                    f"non-integer coefficient {value!r} in tie {text!r}"
                )
            coefficient = int(value)
            if not 1 <= coefficient <= MAX_TIE_COEFFICIENT:
                raise ConstraintError(
                    f"coefficient {value!r} outside 1..{MAX_TIE_COEFFICIENT} "
                    f"in tie {text!r}"
                )
            position += 1
            if position < len(tokens) and tokens[position][0] == "star":
                position += 1
            if position >= len(tokens) or tokens[position][0] != "label":
                raise ConstraintError(
                    f"bare number {value!r} in tie {text!r} (ties are integer "
                    "combinations of components; a constant is not allowed)"
                )
            kind, value = tokens[position]
        if kind != "label":
            raise ConstraintError(f"unexpected {value!r} in tie {text!r}")
        ident = _label_id(value, text)
        if child is not None and ident == child:
            raise ConstraintError(f"cyclic tie: F{child} -> F{child}")
        merged[ident] = merged.get(ident, 0) + sign * coefficient
        position += 1
        first = False
    ids = tuple(i for i, n in merged.items() if n != 0)
    coefficients = tuple(merged[i] for i in ids)
    if not ids:
        raise ConstraintError(f"tie {text!r} sums to zero")
    order = sum(abs(n) for n in coefficients)
    if order > MAX_TIE_ORDER:
        raise ConstraintError(
            f"tie {text!r} has order {order} > {MAX_TIE_ORDER}"
        )
    return ids, coefficients


def format_tie(ids: Sequence[int], coefficients: Sequence[int]) -> str:
    """Inverse of :func:`parse_tie`: ``"F1 + F2"``, ``"2F1 - F3"``.

    Positive terms come first (each group in the given order), coefficients of one are
    omitted, and the minus sign is ASCII.
    """
    if len(ids) != len(coefficients):
        raise ConstraintError("ids and coefficients differ in length")
    terms = [(int(i), int(n)) for i, n in zip(ids, coefficients, strict=True) if n]
    ordered = [t for t in terms if t[1] > 0] + [t for t in terms if t[1] < 0]
    parts: list[str] = []
    for position, (ident, n) in enumerate(ordered):
        magnitude = abs(n)
        term = f"F{ident}" if magnitude == 1 else f"{magnitude}F{ident}"
        if position == 0:
            parts.append(term if n > 0 else f"-{term}")
        else:
            parts.append(f"{'+' if n > 0 else '-'} {term}")
    return " ".join(parts)


TieMap = Mapping[int, tuple[tuple[int, ...], tuple[int, ...], float]]


def flatten_ties(
    ties: TieMap,
    *,
    label: Callable[[int], str] | None = None,
) -> dict[int, tuple[tuple[int, ...], tuple[int, ...], float]]:
    """Substitute tied parents until every parent is untied.

    Parameters
    ----------
    ties : mapping
        ``child -> (parents, coefficients, constant)``; any hashable integer keys
        (session ids or fit positions). A parent that is itself a key is a chain.
    label : callable, optional
        Display name of a key in the cycle message; default ``F{key}`` (keys are
        session ids).

    Returns
    -------
    dict
        The same children, each expressed in untied parents only (merged, zero
        coefficients dropped, constants accumulated). ``F4 = F3 + F1`` with
        ``F3 = 2F1`` becomes ``F4 = 3F1``.

    Raises
    ------
    ConstraintError
        ``"cyclic tie: F3 -> F5 -> F3"`` for a self reference or a cycle.
    """
    name = label if label is not None else (lambda key: f"F{key}")
    resolved: dict[int, tuple[dict[int, int], float]] = {}
    colour: dict[int, int] = {}  # 1 = on the DFS stack, 2 = done

    def visit(node: int, path: list[int]) -> tuple[dict[int, int], float]:
        state = colour.get(node, 0)
        if state == 2:
            return resolved[node]
        if state == 1:
            loop = path[path.index(node):] + [node]
            raise ConstraintError(
                "cyclic tie: " + " -> ".join(name(i) for i in loop)
            )
        colour[node] = 1
        parents, coefficients, constant = ties[node]
        terms: dict[int, int] = {}
        total = float(constant)
        for parent, n in zip(parents, coefficients, strict=True):
            if parent in ties:
                sub_terms, sub_constant = visit(parent, [*path, node])
                total += n * sub_constant
                for p, m in sub_terms.items():
                    terms[p] = terms.get(p, 0) + n * m
            else:
                terms[parent] = terms.get(parent, 0) + n
        colour[node] = 2
        resolved[node] = ({p: m for p, m in terms.items() if m != 0}, total)
        return resolved[node]

    out: dict[int, tuple[tuple[int, ...], tuple[int, ...], float]] = {}
    for child in ties:
        terms, constant = visit(child, [])
        out[child] = (tuple(terms), tuple(terms.values()), constant)
    return out


# -- public types -------------------------------------------------------------------


@dataclass(frozen=True)
class FrequencyTie:
    """``f_child = constant + sum_j coefficients[j] * f[parents[j]]``.

    Attributes
    ----------
    parents : tuple of int
        0-based positions in the ``frequencies`` array of the fit.
    coefficients : tuple of int
        Non-zero integers aligned with ``parents``.
    constant : float, default 0.0
        Cycles/day: the contribution of parents that are not in this fit (a session
        folds an inactive parent's ``n * f`` in here).
    """

    parents: tuple[int, ...]
    coefficients: tuple[int, ...]
    constant: float = 0.0

    def __post_init__(self) -> None:
        if len(self.parents) != len(self.coefficients):
            raise ConstraintError("tie parents and coefficients differ in length")
        for parent in self.parents:
            if isinstance(parent, bool) or int(parent) != parent or parent < 0:
                raise ConstraintError(f"tie parent {parent!r} is not a position")
        for n in self.coefficients:
            if isinstance(n, bool) or int(n) != n or n == 0:
                raise ConstraintError(
                    f"tie coefficient {n!r} is not a non-zero integer"
                )
        if len(set(self.parents)) != len(self.parents):
            raise ConstraintError("a tie names the same parent twice")
        if not np.isfinite(self.constant):
            raise ConstraintError("tie constant must be finite")

    @classmethod
    def from_text(cls, text: str) -> FrequencyTie:
        """Parse ``"F1 + F2"`` — labels are 1-based positions in the fit."""
        ids, coefficients = parse_tie(text)
        return cls(parents=tuple(i - 1 for i in ids), coefficients=coefficients)

    def text(self) -> str:
        """The tie as :func:`format_tie` writes it (positions shown 1-based)."""
        body = format_tie([p + 1 for p in self.parents], self.coefficients)
        if self.constant:
            body = f"{body} + {self.constant!r}" if body else repr(self.constant)
        return body


@dataclass(frozen=True)
class ComponentConstraint:
    """Constraint on one sinusoid; the default is fully free.

    Attributes
    ----------
    frequency : {"free", "fixed"} or FrequencyTie, default "free"
        ``"fixed"`` holds the frequency at the value passed to the fit.
    amplitude : float, optional
        ``None`` = free; a number = fixed at that (signed) value.
    phase : float, optional
        ``None`` = free; a number = fixed, in radians at the fit's *explicit*
        ``t_ref`` (a fixed phase with ``t_ref=None`` is refused).
    variation : {"none", "amplitude", "phase", "both"}, default "none"
        Which of amplitude and phase take one value per variation subset. A
        per-subset quantity cannot be fixed.

    Raises
    ------
    ConstraintError
        For an illegal combination (a fixed per-subset quantity, a fixed zero amplitude
        with a free phase, a non-finite value).
    """

    frequency: Literal["free", "fixed"] | FrequencyTie = "free"
    amplitude: float | None = None
    phase: float | None = None
    variation: Variation = "none"

    def __post_init__(self) -> None:
        if not (
            isinstance(self.frequency, FrequencyTie)
            or self.frequency in ("free", "fixed")
        ):
            raise ConstraintError(
                f"frequency must be 'free', 'fixed' or a FrequencyTie, "
                f"not {self.frequency!r}"
            )
        if self.variation not in get_args(Variation):
            raise ConstraintError(
                f"unknown variation {self.variation!r}; "
                f"choose from {get_args(Variation)}"
            )
        for name in ("amplitude", "phase"):
            value = getattr(self, name)
            if value is not None and not (
                isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool)
                and np.isfinite(value)
            ):
                raise ConstraintError(f"a fixed {name} must be a finite number")
        if self.amplitude is not None and self.variation in ("amplitude", "both"):
            raise ConstraintError(
                "a per-subset amplitude cannot be fixed (variation "
                f"{self.variation!r} with amplitude={self.amplitude!r})"
            )
        if self.phase is not None and self.variation in ("phase", "both"):
            raise ConstraintError(
                "a per-subset phase cannot be fixed (variation "
                f"{self.variation!r} with phase={self.phase!r})"
            )
        if self.amplitude is not None and self.amplitude == 0.0 and self.phase is None:
            raise ConstraintError(
                "a fixed zero amplitude leaves the phase undetermined; fix the phase "
                "too or leave the amplitude free"
            )

    @property
    def is_free(self) -> bool:
        """True for the unconstrained default."""
        return (
            self.frequency == "free"
            and self.amplitude is None
            and self.phase is None
            and self.variation == "none"
        )

    @property
    def varies_amplitude(self) -> bool:
        return self.variation in ("amplitude", "both")

    @property
    def varies_phase(self) -> bool:
        return self.variation in ("phase", "both")


@dataclass(frozen=True, eq=False)
class FitStart:
    """Starting values for the non-linear phases (replaces the linear relaxation).

    Attributes
    ----------
    amplitude, phase : numpy.ndarray
        ``(K,)`` headline values, amplitude signed.
    subset_amplitude, subset_phase : numpy.ndarray, optional
        ``(K, S_v)``; NaN where the quantity does not vary.
    """

    amplitude: FloatArray
    phase: FloatArray
    subset_amplitude: FloatArray | None = None
    subset_phase: FloatArray | None = None


@dataclass(frozen=True, eq=False)
class ResolvedFrequencies:
    """The ties flattened: ``f = constant + matrix @ theta``.

    Attributes
    ----------
    matrix : numpy.ndarray
        ``M``, ``(K, P)``, integer-valued floats.
    constant : numpy.ndarray
        ``f0``, ``(K,)``.
    free_index : numpy.ndarray
        ``(P,)`` component position of each free frequency ``theta_j``.
    status : tuple of str
        ``"free"``, ``"fixed"`` or ``"tied"`` per component.
    """

    matrix: FloatArray
    constant: FloatArray
    free_index: IntArray
    status: tuple[FrequencyStatus, ...]

    @property
    def n_free(self) -> int:
        return int(self.free_index.size)

    def frequencies(self, theta: FloatArray) -> FloatArray:
        """``f0 + M theta``."""
        out: FloatArray = self.constant + self.matrix @ np.asarray(
            theta, dtype=np.float64
        )
        return out

    def theta(self, frequencies: FloatArray) -> FloatArray:
        """The free frequencies read off a full frequency vector."""
        return np.asarray(frequencies, dtype=np.float64)[self.free_index].copy()


def _as_codes(name: str, codes: Any, n_samples: int) -> tuple[IntArray, int]:
    """Validated ``int64`` group codes and their subset count."""
    array = np.asarray(codes)
    if array.shape != (n_samples,):
        raise ConstraintError(
            f"{name} must have one code per point ({n_samples}), got shape "
            f"{array.shape}"
        )
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise ConstraintError(f"{name} must be integers, got {array.dtype}")
    out = array.astype(np.int64, copy=False)
    if out.size == 0:
        return out, 0
    low, high = int(out.min()), int(out.max())
    if low < 0:
        raise ConstraintError(f"{name} contains negative codes")
    counts = np.bincount(out, minlength=high + 1)
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        raise ConstraintError(
            f"{name} must use the codes 0..S-1 with every group non-empty; "
            f"missing {empty[:10].tolist()}"
        )
    return out, high + 1


@dataclass(frozen=True, eq=False)
class FitConstraints:
    """Everything beyond a free multi-sine fit.

    Attributes
    ----------
    components : tuple of ComponentConstraint
        ``()`` = all free; else one per frequency of the fit.
    offset_groups : numpy.ndarray, optional
        ``(N,)`` codes ``0..S_o-1``: one fitted zero point per group (replaces the
        single constant; needs ``fit_mean=True``).
    variation_groups : numpy.ndarray, optional
        ``(N,)`` codes ``0..S_v-1``: the subsets of per-subset amplitudes/phases.
    point_weights : numpy.ndarray, optional
        ``(N,)`` relative multipliers ``omega_i > 0`` of the inverse-variance weights
        (normalised to mean one by the fit).
    min_subset_points : int, default 10
        Fewest points a variation subset may have.
    start : FitStart, optional
        Starting phases, used only when the fit has non-linear phases.
    labels : tuple of str, optional
        Display name of each component in error messages (a session passes its
        ``F{id}`` labels, which differ from the positions once rows are inactive or
        removed); default ``F1, F2, ...`` by position. Never changes a number.
    """

    components: tuple[ComponentConstraint, ...] = ()
    offset_groups: GroupCodes | None = None
    variation_groups: GroupCodes | None = None
    point_weights: FloatArray | None = None
    min_subset_points: int = 10
    start: FitStart | None = field(default=None)
    labels: tuple[str, ...] | None = None

    def _name(self, index: int) -> str:
        """Display label of fit position ``index`` (``labels`` when given)."""
        if self.labels is not None and 0 <= index < len(self.labels):
            return self.labels[index]
        return _label(index)

    @classmethod
    def free(cls, n_components: int) -> FitConstraints:
        """``n_components`` free components and nothing else (trivial)."""
        return cls(components=tuple(ComponentConstraint() for _ in range(n_components)))

    def is_trivial(self) -> bool:
        """True when the fit is the unconstrained v1.3 fit (``start`` is ignored)."""
        return (
            all(c.is_free for c in self.components)
            and self.offset_groups is None
            and self.variation_groups is None
            and self.point_weights is None
        )

    def component(self, index: int) -> ComponentConstraint:
        """The constraint of component ``index`` (free when ``components`` is empty)."""
        if not self.components:
            return ComponentConstraint()
        return self.components[index]

    def expanded(self, n_components: int) -> tuple[ComponentConstraint, ...]:
        """One constraint per component (``()`` expands to all free)."""
        if not self.components:
            return tuple(ComponentConstraint() for _ in range(n_components))
        if len(self.components) != n_components:
            raise ConstraintError(
                f"{len(self.components)} component constraints for "
                f"{n_components} frequencies"
            )
        return self.components

    @property
    def has_variation(self) -> bool:
        return any(c.variation != "none" for c in self.components)

    def resolve(self, frequencies: FloatArray) -> ResolvedFrequencies:
        """Flatten the ties: ``f = f0 + M theta``.

        A fixed frequency is a constant row; a tie to fixed parents moves their
        terms into the constant; chains are substituted. ``theta`` is the vector of
        free frequencies, in component order.

        Raises
        ------
        ConstraintError
            For a parent outside the fit, a self reference or a cycle.
        """
        freqs = np.asarray(frequencies, dtype=np.float64).ravel()
        k = int(freqs.size)
        comps = self.expanded(k)
        free_positions = [i for i, c in enumerate(comps) if c.frequency == "free"]
        column = {pos: j for j, pos in enumerate(free_positions)}
        ties: dict[int, tuple[tuple[int, ...], tuple[int, ...], float]] = {}
        for i, c in enumerate(comps):
            if isinstance(c.frequency, FrequencyTie):
                for parent in c.frequency.parents:
                    if parent >= k:
                        raise ConstraintError(
                            f"{self._name(i)} is tied to fit position {parent + 1}, "
                            f"which is not in this fit of {k} components"
                        )
                    if parent == i:
                        raise ConstraintError(
                            f"cyclic tie: {self._name(i)} -> {self._name(i)}"
                        )
                ties[i] = (
                    c.frequency.parents,
                    c.frequency.coefficients,
                    float(c.frequency.constant),
                )
        flat = flatten_ties(ties, label=self._name)
        p = len(free_positions)
        matrix = np.zeros((k, p), dtype=np.float64)
        constant = np.zeros(k, dtype=np.float64)
        status: list[FrequencyStatus] = []
        for i, c in enumerate(comps):
            if c.frequency == "free":
                matrix[i, column[i]] = 1.0
                status.append("free")
            elif c.frequency == "fixed":
                constant[i] = freqs[i]
                status.append("fixed")
            else:
                parents, coefs, const = flat[i]
                constant[i] = const
                for parent, n in zip(parents, coefs, strict=True):
                    if parent in column:
                        matrix[i, column[parent]] += n
                    else:  # a fixed parent: its value is a constant
                        constant[i] += n * freqs[parent]
                status.append("tied")
        return ResolvedFrequencies(
            matrix=matrix,
            constant=constant,
            free_index=np.asarray(free_positions, dtype=np.int64),
            status=tuple(status),
        )

    def validate(
        self,
        frequencies: FloatArray,
        n_samples: int,
        *,
        fit_mean: bool,
        t_ref: float | None,
    ) -> None:
        """Raise :class:`ConstraintError` for every case the fit cannot honour.

        Checks the component count, the ties (parents in range, no cycle, a positive
        predicted child, no two components with identical frequency rows), the group
        arrays (one code per point, codes ``0..S-1`` all used), ``offset_groups``
        only with ``fit_mean``, varying components only with ``variation_groups`` of
        at least two subsets each holding ``min_subset_points`` and more points than
        its private parameters, positive finite weights, the ``start`` shapes, and a
        fixed phase only with an explicit ``t_ref``.
        """
        freqs = np.asarray(frequencies, dtype=np.float64).ravel()
        k = int(freqs.size)
        comps = self.expanded(k)
        if not np.all(np.isfinite(freqs)):
            raise ConstraintError("frequencies must be finite")
        if t_ref is None and any(c.phase is not None for c in comps):
            raise ConstraintError(
                "a fixed phase needs an explicit t_ref: without one the phase frame "
                "is the weighted epoch of whatever points and weights were passed"
            )
        if isinstance(self.min_subset_points, bool) or self.min_subset_points < 1:
            raise ConstraintError("min_subset_points must be a positive integer")
        resolved = self.resolve(freqs)
        predicted = resolved.frequencies(resolved.theta(freqs))
        for i, status in enumerate(resolved.status):
            if status == "tied" and not predicted[i] > 0.0:
                raise ConstraintError(
                    f"tie puts {self._name(i)} at {predicted[i]:.6g} c/d, which is "
                    "not positive"
                )
        rows = np.column_stack([resolved.matrix, resolved.constant])
        seen: dict[bytes, int] = {}
        for i in range(k):
            key = rows[i].tobytes()
            if key in seen:
                raise ConstraintError(
                    f"{self._name(seen[key])} and {self._name(i)} have identical "
                    "frequencies by construction (duplicate design columns)"
                )
            seen[key] = i

        if self.offset_groups is not None:
            if not fit_mean:
                raise ConstraintError(
                    "offset_groups fit one zero point per group, which needs "
                    "fit_mean=True"
                )
            _as_codes("offset_groups", self.offset_groups, n_samples)
        varying = [i for i, c in enumerate(comps) if c.variation != "none"]
        n_subsets = 0
        if self.variation_groups is not None:
            codes, n_subsets = _as_codes(
                "variation_groups", self.variation_groups, n_samples
            )
            if varying:
                if n_subsets < 2:
                    raise ConstraintError(
                        "per-subset amplitudes or phases need at least two variation "
                        "subsets"
                    )
                private = sum(
                    int(comps[i].varies_amplitude) + int(comps[i].varies_phase)
                    for i in varying
                )
                counts = np.bincount(codes, minlength=n_subsets)
                need = max(int(self.min_subset_points), private + 1)
                small = np.flatnonzero(counts < need)
                if small.size:
                    raise ConstraintError(
                        f"variation subsets {small[:10].tolist()} have fewer than "
                        f"{need} points (min_subset_points={self.min_subset_points}, "
                        f"{private} per-subset parameters); merge them first, e.g. "
                        "with Partition.merged_small()"
                    )
        elif varying:
            raise ConstraintError(
                f"{self._name(varying[0])} varies per subset but no "
                "variation_groups were given"
            )
        if self.point_weights is not None:
            weights = np.asarray(self.point_weights, dtype=np.float64)
            if weights.shape != (n_samples,):
                raise ConstraintError(
                    f"point_weights must have one value per point ({n_samples})"
                )
            if not np.all(np.isfinite(weights) & (weights > 0.0)):
                raise ConstraintError("point_weights must be finite and positive")
        if self.start is not None:
            start = self.start
            for name in ("amplitude", "phase"):
                values = np.asarray(getattr(start, name), dtype=np.float64)
                if values.shape != (k,):
                    raise ConstraintError(
                        f"start.{name} must have shape ({k},), got {values.shape}"
                    )
            for name in ("subset_amplitude", "subset_phase"):
                values_or_none = getattr(start, name)
                if values_or_none is None:
                    continue
                shape = np.shape(values_or_none)
                if self.variation_groups is None or shape != (k, n_subsets):
                    raise ConstraintError(
                        f"start.{name} must have shape ({k}, S_v={n_subsets}), "
                        f"got {shape}"
                    )


@dataclass(frozen=True, eq=False)
class FitDetail:
    """What a constrained fit knows beyond the legacy :class:`MultiSineFit` fields.

    Attributes
    ----------
    constraints : FitConstraints
        The constraints the fit honoured.
    parameter_names : tuple of str
        Names of the free natural parameters in covariance order: ``"f[0]"`` (the
        free frequency of component 0), ``"A[2]"``, ``"A[2|1]"`` (subset 1),
        ``"phi[1|0]"``, ``"offset"``, ``"offset[3]"``.
    parameter_covariance : numpy.ndarray
        ``(n_p, n_p)`` covariance of those parameters (empty without covariance).
    flags : tuple of frozenset of str
        Per component, from :data:`FLAGS`.
    rms_amplitude : numpy.ndarray
        ``(K,)`` ``sqrt(sum_s W_s |z_s|^2 / sum W)``; ``|A|`` for a common component.
    subset_amplitude, subset_amplitude_error : numpy.ndarray, optional
        ``(K, S_v)``; the common value is broadcast where the amplitude does not vary,
        where its error is NaN.
    subset_phase, subset_phase_error : numpy.ndarray, optional
        Likewise for the phases.
    subset_offset, subset_offset_error : numpy.ndarray, optional
        ``(S_o,)`` fitted zero points.
    rank, n_columns : int
        Numerical rank and width of the natural-parameter design at the solution.
    condition : float
        ``|R_00 / R_rr|`` of its column-scaled, column-pivoted QR.
    nfev : int
        Residual evaluations spent by the optimiser.
    """

    constraints: FitConstraints
    parameter_names: tuple[str, ...]
    parameter_covariance: FloatArray
    flags: tuple[frozenset[str], ...]
    rms_amplitude: FloatArray
    subset_amplitude: FloatArray | None
    subset_amplitude_error: FloatArray | None
    subset_phase: FloatArray | None
    subset_phase_error: FloatArray | None
    subset_offset: FloatArray | None
    subset_offset_error: FloatArray | None
    rank: int
    n_columns: int
    condition: float
    nfev: int


__all__ = [
    "FLAGS",
    "ComponentConstraint",
    "ConstraintError",
    "FitConstraints",
    "FitDetail",
    "FitStart",
    "FrequencyStatus",
    "FrequencyTie",
    "GroupCodes",
    "ResolvedFrequencies",
    "Variation",
    "flatten_ties",
    "format_tie",
    "parse_tie",
]
