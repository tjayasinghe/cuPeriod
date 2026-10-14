"""Peak detection, N-best selection, and peak-preserving downsampling.

These operate on a *score to maximize*: the periodogram power for max-objective
methods (GLS/BLS/MHAOV/TLS), or the negated statistic for min-objective methods
(PDM/CE/string-length). Keeping a single maximize convention lets every method share
one peak picker. Non-finite scores never win: they are treated as ``-inf`` so they
can neither be reported nor hide a finite neighbour.

Two selection strategies are provided:

* :func:`select_top_peaks` — greedy by score with a minimum frequency separation, so
  an oversampled grid does not return many samples of the same physical peak.
* :func:`select_diverse_peaks` — additionally skips candidates that are trivially
  alias- or harmonic-related to an already-chosen peak (useful for box searches whose
  harmonic comb would otherwise fill every slot).
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np

from cuperiod.core._typing import Float32Array, FloatArray, IntArray

#: Sidereal-day frequency in cycles/day — the dominant ground-based alias spacing.
SIDEREAL_FREQ_CPD: Final = 1.0027379

#: Solar-day frequency in cycles/day — the spacing an evenly scheduled queue imprints.
SOLAR_FREQ_CPD: Final = 1.0

#: Yearly sidelobe spacing in cycles/day.
YEARLY_FREQ_CPD: Final = 1.0 / 365.25

#: Above this many input samples per output bin the per-bin ``argmax`` loop beats the
#: vectorized ``reduceat`` path (measured crossover ~250; this keeps a margin).
_DOWNSAMPLE_VECTOR_MAX_RATIO: Final = 64


def _sum_is_finite(values: FloatArray) -> bool:
    """Whether every sample is finite, in one cheap pass over the common clean case.

    A sum that overflows only sends a finite array down the callers' (equally correct)
    slow path; the ``inf - inf`` / overflow warnings of the probe itself are muted.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        return bool(np.isfinite(values.sum()))


def _finite_or_minus_inf(score: FloatArray) -> FloatArray:
    """``score`` with each non-finite sample replaced by ``-inf`` (no copy if clean)."""
    if _sum_is_finite(score):
        return score
    return np.where(np.isfinite(score), score, -np.inf)


def local_maxima(score: FloatArray) -> IntArray:
    """Indices of interior local maxima of ``score``.

    A point qualifies if it is strictly greater than its left neighbor and greater
    than or equal to its right neighbor (so flat-topped peaks report their left
    edge). Endpoints are never returned. Non-finite samples compare as ``-inf``: they
    are never maxima themselves and never mask a finite neighbour.

    Parameters
    ----------
    score : numpy.ndarray
        The score to maximize.

    Returns
    -------
    numpy.ndarray
        Int64 indices of local maxima.
    """
    if score.size < 3:
        return np.empty(0, dtype=np.int64)
    return _interior_maxima(_finite_or_minus_inf(score))


def _interior_maxima(safe: FloatArray) -> IntArray:
    """:func:`local_maxima` of a score already free of NaN / ``+inf``."""
    if safe.size < 3:
        return np.empty(0, dtype=np.int64)
    interior = (safe[1:-1] > safe[:-2]) & (safe[1:-1] >= safe[2:])
    return np.flatnonzero(interior).astype(np.int64) + 1


def select_top_peaks(
    frequency: FloatArray,
    score: FloatArray,
    candidates: IntArray,
    n_peaks: int,
    min_separation: float,
) -> IntArray:
    """Greedy highest-score peaks with a minimum frequency separation.

    Parameters
    ----------
    frequency : numpy.ndarray
        Frequency of each grid sample (cycles/day).
    score : numpy.ndarray
        Score to maximize.
    candidates : numpy.ndarray
        Indices to consider (typically :func:`local_maxima` output).
    n_peaks : int
        Maximum number of peaks to return (``<= 0`` returns none).
    min_separation : float
        Minimum allowed frequency gap between any two chosen peaks (cycles/day).

    Returns
    -------
    numpy.ndarray
        Int64 indices of the chosen peaks, in descending score order.
    """
    if n_peaks <= 0:
        return np.empty(0, dtype=np.int64)
    chosen: list[int] = []
    for idx in candidates[np.argsort(score[candidates])[::-1]]:
        if all(
            abs(frequency[idx] - frequency[j]) >= min_separation for j in chosen
        ):
            chosen.append(int(idx))
            if len(chosen) >= n_peaks:
                break
    return np.asarray(chosen, dtype=np.int64)


def is_alias_related(
    f_new: float,
    f_chosen: float,
    tol: float,
    harmonic_max: int,
    *,
    min_separation: float | None = None,
) -> bool:
    """Whether ``f_new`` is trivially related to ``f_chosen``.

    Checks the ground-based alias lattice (sidereal- and solar-day offsets ``k=1,2``
    and the yearly sidelobe) plus harmonics/subharmonics up to ``harmonic_max``.

    Two scales are involved and they are not the same thing. ``min_separation`` is
    the *duplicate-peak* separation (the same peak seen again on an oversampled grid,
    a few Rayleigh widths); ``tol`` is how precisely a frequency relation must hold to
    call two peaks harmonics or aliases, and widens to ``m*tol`` around the ``m``-th
    harmonic. Reusing the separation as ``tol`` turned the harmonic and subharmonic
    bands of one peak into a solid exclusion zone on short baselines.

    Parameters
    ----------
    f_new, f_chosen : float
        Frequencies to compare (cycles/day).
    tol : float
        Alias / harmonic frequency tolerance (cycles/day).
    harmonic_max : int
        Highest harmonic order to test.
    min_separation : float, optional
        Separation below which two frequencies are the same peak (cycles/day).
        Defaults to ``tol``.

    Returns
    -------
    bool
    """
    separation = tol if min_separation is None else min_separation
    if abs(f_new - f_chosen) < separation:
        return True
    for m in range(2, harmonic_max + 1):
        if abs(f_new - m * f_chosen) < m * tol:
            return True
        if abs(f_new - f_chosen / m) < tol:
            return True
    offsets = [k * day for day in (SIDEREAL_FREQ_CPD, SOLAR_FREQ_CPD) for k in (1, 2)]
    offsets.append(YEARLY_FREQ_CPD)
    for off in offsets:
        if (
            abs(f_new - abs(f_chosen - off)) < tol
            or abs(f_new - (f_chosen + off)) < tol
        ):
            return True
    return False


def select_diverse_peaks(
    frequency: FloatArray,
    score: FloatArray,
    candidates: IntArray,
    n_peaks: int,
    tol: float,
    harmonic_max: int,
    *,
    min_separation: float | None = None,
) -> IntArray:
    """Greedy highest-score peaks skipping alias/harmonic-related candidates.

    Parameters
    ----------
    frequency, score, candidates, n_peaks
        As for :func:`select_top_peaks` (``n_peaks <= 0`` returns none).
    tol : float
        Alias / harmonic tolerance passed to :func:`is_alias_related`.
    harmonic_max : int
        Highest harmonic order considered "related".
    min_separation : float, optional
        Duplicate-peak separation passed to :func:`is_alias_related` (defaults to
        ``tol``).

    Returns
    -------
    numpy.ndarray
        Int64 indices of the chosen peaks, in descending score order.
    """
    if n_peaks <= 0:
        return np.empty(0, dtype=np.int64)
    chosen: list[int] = []
    for idx in candidates[np.argsort(score[candidates])[::-1]]:
        f_new = float(frequency[idx])
        if all(
            not is_alias_related(
                f_new, float(frequency[j]), tol, harmonic_max,
                min_separation=min_separation,
            )
            and not is_alias_related(
                float(frequency[j]), f_new, tol, harmonic_max,
                min_separation=min_separation,
            )
            for j in chosen
        ):
            chosen.append(int(idx))
            if len(chosen) >= n_peaks:
                break
    return np.asarray(chosen, dtype=np.int64)


def peak_preserving_downsample(
    frequency: FloatArray,
    power: FloatArray,
    n_points: int,
    *,
    sense: Literal["max", "min"] = "max",
) -> tuple[Float32Array, Float32Array]:
    """Downsample by keeping the most significant sample of each of ``n_points`` bins.

    Unlike striding or averaging, this never loses a peak's apex — the global extreme
    survives exactly — which makes the stored spectrum faithful for plots and quick
    re-analysis while shrinking it for compact batch storage. "Most significant"
    follows ``sense``: the maximum for max-objective statistics, the minimum (the dip)
    for min-objective ones (PDM, CE, string-length). NaN never wins a bin unless the
    whole bin is NaN; ties go to the first sample.

    Parameters
    ----------
    frequency, power : numpy.ndarray
        The full grid (any ordering; bins are index-based).
    n_points : int
        Target number of output points. If the grid is already this small, it is
        returned unchanged (as float32); ``n_points <= 0`` returns empty arrays.
    sense : {"max", "min"}, default "max"
        Whether peaks are maxima or minima of ``power``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(frequency, power)`` as float32 arrays of length ``min(n, n_points)``.
    """
    if sense not in ("max", "min"):
        raise ValueError(f"sense must be 'max' or 'min', got {sense!r}")
    n = int(frequency.size)
    if n_points <= 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    if n <= n_points:
        return frequency.astype(np.float32), power.astype(np.float32)
    score = power if sense == "max" else -power
    if not _sum_is_finite(score):
        score = np.where(np.isnan(score), -np.inf, score)
    edges = np.linspace(0, n, n_points + 1).astype(np.int64)
    if n <= _DOWNSAMPLE_VECTOR_MAX_RATIO * n_points:
        # Every bin is non-empty (the bin width n/n_points exceeds 1), so reduceat
        # gives one maximum per bin; the first sample equal to it is np.argmax's pick.
        starts = edges[:-1]
        bin_max = np.maximum.reduceat(score, starts)
        hits = np.flatnonzero(score == np.repeat(bin_max, np.diff(edges)))
        picks = hits[np.searchsorted(hits, starts)]
    else:
        picks = np.empty(n_points, dtype=np.int64)
        for i in range(n_points):
            lo, hi = int(edges[i]), int(edges[i + 1])
            picks[i] = lo + int(np.argmax(score[lo:hi]))
    return frequency[picks].astype(np.float32), power[picks].astype(np.float32)


__all__ = [
    "SIDEREAL_FREQ_CPD",
    "SOLAR_FREQ_CPD",
    "YEARLY_FREQ_CPD",
    "is_alias_related",
    "local_maxima",
    "peak_preserving_downsample",
    "select_diverse_peaks",
    "select_top_peaks",
]
