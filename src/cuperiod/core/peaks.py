"""Peak detection, N-best selection, and peak-preserving downsampling.

These operate on a *score to maximize*: the periodogram power for max-objective
methods (GLS/BLS/MHAOV/TLS), or the negated statistic for min-objective methods
(PDM/CE/string-length). Keeping a single maximize convention lets every method share
one peak picker.

Two selection strategies are provided:

* :func:`select_top_peaks` — greedy by score with a minimum frequency separation, so
  an oversampled grid does not return many samples of the same physical peak.
* :func:`select_diverse_peaks` — additionally skips candidates that are trivially
  alias- or harmonic-related to an already-chosen peak (useful for box searches whose
  harmonic comb would otherwise fill every slot).
"""

from __future__ import annotations

from typing import Final

import numpy as np

from cuperiod.core._typing import Float32Array, FloatArray, IntArray

#: Sidereal-day frequency in cycles/day — the dominant ground-based alias spacing.
SIDEREAL_FREQ_CPD: Final = 1.0027379

#: Yearly sidelobe spacing in cycles/day.
YEARLY_FREQ_CPD: Final = 1.0 / 365.25


def local_maxima(score: FloatArray) -> IntArray:
    """Indices of interior local maxima of ``score``.

    A point qualifies if it is strictly greater than its left neighbor and greater
    than or equal to its right neighbor (so flat-topped peaks report their left
    edge). Endpoints are never returned.

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
    interior = (score[1:-1] > score[:-2]) & (score[1:-1] >= score[2:])
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
        Maximum number of peaks to return.
    min_separation : float
        Minimum allowed frequency gap between any two chosen peaks (cycles/day).

    Returns
    -------
    numpy.ndarray
        Int64 indices of the chosen peaks, in descending score order.
    """
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
    f_new: float, f_chosen: float, tol: float, harmonic_max: int
) -> bool:
    """Whether ``f_new`` is trivially related to ``f_chosen``.

    Checks the ground-based alias lattice (sidereal-day offsets ``k=1,2`` and the
    yearly sidelobe) plus harmonics/subharmonics up to ``harmonic_max``.

    Parameters
    ----------
    f_new, f_chosen : float
        Frequencies to compare (cycles/day).
    tol : float
        Base frequency tolerance (cycles/day).
    harmonic_max : int
        Highest harmonic order to test.

    Returns
    -------
    bool
    """
    if abs(f_new - f_chosen) < tol:
        return True
    for m in range(2, harmonic_max + 1):
        if abs(f_new - m * f_chosen) < m * tol:
            return True
        if abs(f_new - f_chosen / m) < tol:
            return True
    offsets = [k * SIDEREAL_FREQ_CPD for k in (1, 2)] + [YEARLY_FREQ_CPD]
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
) -> IntArray:
    """Greedy highest-score peaks skipping alias/harmonic-related candidates.

    Parameters
    ----------
    frequency, score, candidates, n_peaks
        As for :func:`select_top_peaks`.
    tol : float
        Frequency tolerance passed to :func:`is_alias_related`.
    harmonic_max : int
        Highest harmonic order considered "related".

    Returns
    -------
    numpy.ndarray
        Int64 indices of the chosen peaks, in descending score order.
    """
    chosen: list[int] = []
    for idx in candidates[np.argsort(score[candidates])[::-1]]:
        f_new = float(frequency[idx])
        if all(
            not is_alias_related(f_new, float(frequency[j]), tol, harmonic_max)
            and not is_alias_related(float(frequency[j]), f_new, tol, harmonic_max)
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
) -> tuple[Float32Array, Float32Array]:
    """Downsample by keeping the max-power sample in each of ``n_points`` index bins.

    Unlike striding or averaging, this never loses a peak's apex — the global maximum
    survives exactly — which makes the stored spectrum faithful for plots and quick
    re-analysis while shrinking it for compact batch storage.

    Parameters
    ----------
    frequency, power : numpy.ndarray
        The full grid (any ordering; bins are index-based).
    n_points : int
        Target number of output points. If the grid is already this small, it is
        returned unchanged (as float32).

    Returns
    -------
    tuple of numpy.ndarray
        ``(frequency, power)`` as float32 arrays of length ``min(n, n_points)``.
    """
    n = int(frequency.size)
    if n <= n_points:
        return frequency.astype(np.float32), power.astype(np.float32)
    edges = np.linspace(0, n, n_points + 1).astype(np.int64)
    freq_out = np.empty(n_points, dtype=np.float32)
    power_out = np.empty(n_points, dtype=np.float32)
    for i in range(n_points):
        lo, hi = int(edges[i]), int(edges[i + 1])
        j = lo + int(np.argmax(power[lo:hi]))
        freq_out[i] = frequency[j]
        power_out[i] = power[j]
    return freq_out, power_out


__all__ = [
    "SIDEREAL_FREQ_CPD",
    "YEARLY_FREQ_CPD",
    "is_alias_related",
    "local_maxima",
    "peak_preserving_downsample",
    "select_diverse_peaks",
    "select_top_peaks",
]
