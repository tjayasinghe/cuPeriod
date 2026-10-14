"""Combination-frequency identification.

A non-linear pulsator's Fourier spectrum is not a list of independent modes: a large
fraction of the peaks are *combinations* :math:`f = \\sum_i n_i f_i` of a few
high-amplitude parents — harmonics ``2f_1``, sums ``f_1 + f_2``, differences
``f_1 - f_2``. Reporting those as independent modes is one of the classic ways to
over-count the mode density of a δ Scuti star, so this module flags them. A
combination has order ``sum|n_i| >= 2``; a frequency that merely coincides with one
stronger component is a near-duplicate, not a combination, and is not reported here.

Two things make the search trustworthy rather than merely suggestive:

**The tolerance is uncertainty-aware.** A match must fall inside
``max(sigma_tolerance, rayleigh_tolerance)`` where the first is
``n_sigma * sqrt(sigma_child^2 + sum_i n_i^2 sigma_i^2)`` — the propagated uncertainty
of the predicted combination — and the second is a fraction of the Rayleigh resolution,
which floors the test when the formal errors are unrealistically small.

**Chance coincidences are counted.** With enough parents and a generous order, *some*
frequency will land within tolerance of *some* combination by luck. Every match carries
the expected number of chance matches of a random frequency in the band of extracted
frequencies, so a user can see immediately whether an identification means anything.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from cuperiod.core._typing import FloatArray, IntArray


@dataclass(frozen=True)
class Combination:
    """One frequency identified as a combination of higher-amplitude parents.

    Attributes
    ----------
    index : int
        Index of the child component within the solution.
    parents : tuple of int
        Indices of the parent components used.
    coefficients : tuple of int
        Integer multipliers aligned with ``parents`` (never all zero).
    order : int
        ``sum(abs(coefficients))`` — 2 for ``2f_1`` or ``f_1 + f_2``, 3 for
        ``2f_1 - f_2``, and so on (always at least 2).
    label : str
        Human-readable identification, e.g. ``"F5 = 2F1 - F2"``.
    predicted : float
        The combination frequency implied by the parents (cycles/day).
    residual : float
        ``observed - predicted`` (cycles/day).
    tolerance : float
        The matching tolerance that was applied (cycles/day).
    expected_false : float
        Expected number of chance matches for a frequency drawn at random from the
        band of extracted frequencies: the summed tolerance windows of the searched
        combinations that predict a positive frequency inside that band, over its
        width. Values approaching (or above) 1 mean the identification carries little
        information.
    """

    index: int
    parents: tuple[int, ...]
    coefficients: tuple[int, ...]
    order: int
    label: str
    predicted: float
    residual: float
    tolerance: float
    expected_false: float

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dict."""
        return {
            "index": self.index,
            "label": self.label,
            "parents": list(self.parents),
            "coefficients": list(self.coefficients),
            "order": self.order,
            "predicted": self.predicted,
            "residual": self.residual,
            "tolerance": self.tolerance,
            "expected_false": self.expected_false,
        }


def _coefficient_vectors(n_parents: int, max_order: int) -> Iterator[tuple[int, ...]]:
    """All integer vectors with ``2 <= sum|n_i| <= max_order``.

    Enumerated depth-first against the remaining order budget, so the count stays modest
    (50 vectors for five parents at order 2) instead of the ``(2m+1)^P`` a naive
    product would produce. Order 1 is left out: a unit vector "explains" a child as a
    copy of one parent, which is a near-duplicate, not a combination, and as the
    lowest order it pre-empted genuine sums and harmonics.
    """

    def walk(
        position: int, budget: int, prefix: tuple[int, ...]
    ) -> Iterator[tuple[int, ...]]:
        if position == n_parents:
            if max_order - budget >= 2:  # a combination of order two or more
                yield prefix
            return
        for value in range(-budget, budget + 1):
            yield from walk(position + 1, budget - abs(value), (*prefix, value))

    if max_order < 2:
        return
    yield from walk(0, max_order, ())


@lru_cache(maxsize=64)
def _coefficient_table(
    n_parents: int, max_order: int
) -> tuple[tuple[tuple[int, ...], ...], FloatArray, FloatArray, IntArray]:
    """``(vectors, matrix, squared, order)`` for one parent count and order.

    The coefficient vectors depend only on how many parents there are, so the
    enumeration and its ``(V, P)`` matrix are built once per pair and reused for every
    child (and every later call); the arrays are read-only.
    """
    vectors = tuple(_coefficient_vectors(n_parents, max_order))
    matrix = np.asarray(vectors, dtype=np.float64).reshape(len(vectors), n_parents)
    squared = matrix * matrix
    order = np.abs(matrix).sum(axis=1).astype(np.int64)
    for array in (matrix, squared, order):
        array.setflags(write=False)
    return vectors, matrix, squared, order


def _format_label(
    child_label: str, parent_labels: tuple[str, ...], coefficients: tuple[int, ...]
) -> str:
    """Render ``F5 = 2F1 - F2`` from labels and integer coefficients.

    Positive terms are written first so a difference reads ``F2 - F1`` rather than
    ``-F1 + F2``.
    """
    used = [
        (label, coefficient)
        for label, coefficient in zip(parent_labels, coefficients, strict=True)
        if coefficient != 0
    ]
    used.sort(key=lambda item: item[1] < 0)
    terms: list[str] = []
    for label, coefficient in used:
        magnitude = abs(coefficient)
        body = f"{magnitude if magnitude != 1 else ''}{label}"
        if not terms:
            terms.append(f"-{body}" if coefficient < 0 else body)
        else:
            terms.append(f"{'-' if coefficient < 0 else '+'} {body}")
    return f"{child_label} = {' '.join(terms)}"


def identify_combinations(
    frequency: FloatArray,
    amplitude: FloatArray,
    *,
    frequency_error: FloatArray | None = None,
    labels: tuple[str, ...] | None = None,
    rayleigh: float = 0.0,
    max_order: int = 2,
    max_parents: int = 5,
    tolerance_rayleigh: float = 0.25,
    n_sigma: float = 3.0,
) -> tuple[Combination, ...]:
    """Flag components that are integer combinations of higher-amplitude parents.

    Parents are the highest-amplitude components: a peak is only ever explained by
    frequencies *stronger* than itself, which is the physical ordering for non-linear
    combinations and also stops the search from explaining a mode by its own harmonics.
    Among all admissible matches the lowest order wins, and ties are broken by the
    smallest frequency residual.

    Parameters
    ----------
    frequency, amplitude : numpy.ndarray
        The extracted components (any order; parents are chosen by amplitude).
    frequency_error : numpy.ndarray, optional
        1-sigma frequency uncertainties, used for the propagated tolerance.
    labels : tuple of str, optional
        Display labels, defaulting to ``F1``, ``F2``, ...
    rayleigh : float, default 0.0
        Rayleigh resolution ``1/T`` (cycles/day); floors the tolerance.
    max_order : int, default 2
        Largest ``sum|n_i|`` considered; below 2 nothing is searched.
    max_parents : int, default 5
        Number of highest-amplitude components usable as parents.
    tolerance_rayleigh : float, default 0.25
        Tolerance floor as a fraction of the Rayleigh resolution.
    n_sigma : float, default 3.0
        Tolerance in units of the propagated frequency uncertainty.

    Returns
    -------
    tuple of Combination
        One entry per identified child, in ascending child index.

    Examples
    --------
    >>> identify_combinations(np.array([5.0, 7.0, 12.0]),        # doctest: +SKIP
    ...                       np.array([1.0, 0.5, 0.1]),
    ...                       rayleigh=0.01)
    (Combination(index=2, label='F3 = F1 + F2', ...),)
    """
    freq = np.asarray(frequency, dtype=np.float64).ravel()
    amp = np.asarray(amplitude, dtype=np.float64).ravel()
    k = int(freq.size)
    if k < 2 or max_order < 2 or max_parents < 1:
        return ()
    sigma = (
        np.zeros(k, dtype=np.float64)
        if frequency_error is None
        else np.nan_to_num(
            np.asarray(frequency_error, dtype=np.float64).ravel(), nan=0.0
        )
    )
    names = labels or tuple(f"F{i + 1}" for i in range(k))
    band_low, band_high = float(freq.min()), float(freq.max())
    span = band_high - band_low
    floor = tolerance_rayleigh * rayleigh

    by_amplitude = np.argsort(amp, kind="stable")[::-1]
    found: list[Combination] = []
    for position, child in enumerate(by_amplitude):
        if position == 0:
            continue
        parents = tuple(int(i) for i in by_amplitude[: min(position, max_parents)])
        vectors, matrix, squared, orders = _coefficient_table(len(parents), max_order)
        if not vectors:
            continue
        parent_freq = freq[list(parents)]
        parent_sigma = sigma[list(parents)]
        # Every coefficient vector at once: predictions, propagated tolerances and
        # residuals as arrays rather than a Python loop of tiny NumPy calls.
        predicted = matrix @ parent_freq
        propagated = np.sqrt(sigma[child] ** 2 + squared @ (parent_sigma**2))
        tolerance = np.maximum(n_sigma * propagated, floor)
        residual = float(freq[child]) - predicted
        # The negated comparisons keep NaN predictions out exactly as the scalar
        # tests did (a NaN fails every comparison).
        positive = ~(predicted <= 0.0)
        hits = np.flatnonzero(
            positive & ~(tolerance <= 0.0) & ~(np.abs(residual) > tolerance)
        )
        if hits.size == 0:
            continue
        # v and -v give one positive prediction between them, and harmonics or sums
        # far outside the extracted frequencies cannot catch a child that lies in it.
        in_band = positive & (predicted >= band_low) & (predicted <= band_high)
        window = float(np.sum(2.0 * tolerance[in_band]))
        # Lowest order first, then the smallest residual, then enumeration order.
        best = int(
            hits[np.lexsort((hits, np.abs(residual[hits]), orders[hits]))[0]]
        )
        coefficients = vectors[best]
        found.append(
            Combination(
                index=int(child),
                parents=parents,
                coefficients=coefficients,
                order=int(orders[best]),
                label=_format_label(
                    names[child], tuple(names[i] for i in parents), coefficients
                ),
                predicted=float(predicted[best]),
                residual=float(residual[best]),
                tolerance=float(tolerance[best]),
                expected_false=window / span if span > 0.0 else float("nan"),
            )
        )
    return tuple(sorted(found, key=lambda c: c.index))


__all__ = ["Combination", "identify_combinations"]
