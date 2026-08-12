"""g-mode period-spacing tools.

High-order gravity modes of the same degree and azimuthal order are asymptotically
equally spaced **in period**, and the spacing carries the physics: its mean value fixes
the buoyancy travel time, its slope traces near-core rotation, and periodic dips in it
reveal chemical-gradient zones left by a retreating convective core. Recovering that
pattern from a frequency list is the step after pre-whitening for γ Dor and SPB stars.

Three tools, in the order they are normally used:

:func:`spacing_spectrum`
    Where is the comb? A scan over trial spacings of
    ``|sum_j a_j exp(2 pi i P_j / dP)|^2`` — the Fourier response of the period list
    treated as a spike train. Sharply peaked at a genuine regular spacing, and unlike a
    histogram of consecutive differences it is unaffected by missing modes.
:func:`find_period_spacing`
    Which modes belong to it? A dynamic-programme search for the longest chain of modes
    whose consecutive spacings follow a *tilted* pattern ``dP(P) = a + b P``, allowing a
    bounded number of missing radial orders to be bridged.
:func:`echelle`
    Does it look right? Period modulo the spacing, the diagnostic plot in which a clean
    series is a near-vertical ridge.

The same machinery works on frequencies, where a regular spacing is the p-mode large
separation or a rotational splitting — pass frequencies instead of periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.config import SpacingSettings

#: Seconds per day, for reporting the buoyancy radius in the usual units.
_SECONDS_PER_DAY: Final = 86400.0

#: Cap on the trial-spacing grid so a pathological input cannot allocate unboundedly.
_MAX_GRID: Final = 1 << 21

#: Elements per transient block of the comb response (8 MB).
_BLOCK_ELEMS: Final = 1 << 20

#: Fraction of the peak response a wider comb must retain to be promoted over it.
_SUBMULTIPLE_RATIO: Final = 0.8

#: Largest integer multiple of the winning spacing tested for the promotion above.
_MAX_SUBMULTIPLE: Final = 6

#: Phase-residual cut (in fractions of a spacing) for "this value is on that comb".
_COMB_MEMBER_TOL: Final = 0.15


@dataclass(frozen=True)
class SpacingSpectrum:
    """Comb response of a set of values against trial spacings.

    Attributes
    ----------
    spacing : numpy.ndarray
        Trial spacings, ascending, in the units of the input (days for periods).
    power : numpy.ndarray
        Response in ``[0, 1]``; 1 means every value lands on the same comb.
    best_spacing, best_power : float
        The tallest response and where it sits.
    n_values : int
        Number of input values.
    """

    spacing: FloatArray
    power: FloatArray
    best_spacing: float
    best_power: float
    n_values: int

    @property
    def size(self) -> int:
        """Number of trial spacings."""
        return int(self.spacing.size)

    def to_dict(self) -> dict[str, Any]:
        """Flatten the summary (not the full arrays) to a plain dict."""
        return {
            "best_spacing": self.best_spacing,
            "best_power": self.best_power,
            "n_values": self.n_values,
            "n_trials": self.size,
        }


@dataclass(frozen=True)
class PeriodSpacingSeries:
    """A chain of modes following one (possibly tilted) period-spacing pattern.

    Attributes
    ----------
    indices : tuple of int
        Positions in the *input* array of the member modes, in ascending period.
    periods : numpy.ndarray
        Member periods (days), ascending.
    spacings : numpy.ndarray
        Observed consecutive differences ``P[i+1] - P[i]`` (days).
    multiplicity : numpy.ndarray
        How many radial orders each observed step spans — 1 for a consecutive pair, 2 if
        one mode is missing, and so on.
    midpoints : numpy.ndarray
        ``(P[i] + P[i+1]) / 2``, the period each spacing is plotted against.
    mean_spacing : float
        Amplitude-unweighted mean of ``spacings / multiplicity`` (days).
    slope, intercept : float
        The fitted tilt ``dP(P) = intercept + slope * P``. A non-zero slope is the
        signature of rotation in the asymptotic g-mode pattern.
    rms : float
        RMS of ``spacings/multiplicity`` about the tilted model (days).
    ell : int
        Spherical degree assumed when converting to a buoyancy radius.
    buoyancy_radius : float
        ``Pi_0 = mean_spacing * sqrt(l(l+1))`` in **seconds**.
    """

    indices: tuple[int, ...]
    periods: FloatArray
    spacings: FloatArray
    multiplicity: FloatArray
    midpoints: FloatArray
    mean_spacing: float
    slope: float
    intercept: float
    rms: float
    ell: int
    buoyancy_radius: float

    @property
    def n_modes(self) -> int:
        """Number of modes in the series."""
        return len(self.indices)

    def predicted_spacing(self, period: FloatArray) -> FloatArray:
        """The tilted model spacing at ``period``."""
        return self.intercept + self.slope * np.asarray(period, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain, JSON-friendly dict."""
        return {
            "n_modes": self.n_modes,
            "indices": list(self.indices),
            "periods": self.periods.tolist(),
            "spacings": self.spacings.tolist(),
            "multiplicity": self.multiplicity.astype(int).tolist(),
            "mean_spacing": self.mean_spacing,
            "slope": self.slope,
            "intercept": self.intercept,
            "rms": self.rms,
            "ell": self.ell,
            "buoyancy_radius": self.buoyancy_radius,
        }

    def summary(self) -> str:
        """A one-paragraph human-readable report."""
        return (
            f"Period spacing: {self.n_modes} modes, "
            f"<dP> = {self.mean_spacing * _SECONDS_PER_DAY:.1f} s "
            f"({self.mean_spacing:.6g} d), slope {self.slope:+.4g}, "
            f"rms {self.rms * _SECONDS_PER_DAY:.1f} s, "
            f"Pi_0(l={self.ell}) = {self.buoyancy_radius:.0f} s"
        )


def buoyancy_radius(mean_spacing: float, ell: int = 1) -> float:
    """Asymptotic buoyancy travel time ``Pi_0`` in seconds.

    In the asymptotic limit ``dP_l = Pi_0 / sqrt(l(l+1))``, so a measured spacing fixes
    ``Pi_0`` once the degree is known. Dipole modes (``l = 1``) dominate γ Dor and SPB
    spectra.

    Parameters
    ----------
    mean_spacing : float
        Mean period spacing in days.
    ell : int, default 1
        Spherical degree.

    Returns
    -------
    float
        ``Pi_0`` in seconds.
    """
    if ell < 1:
        raise ValueError("ell must be >= 1")
    return float(mean_spacing * np.sqrt(ell * (ell + 1)) * _SECONDS_PER_DAY)


def echelle(
    values: FloatArray, spacing: float, *, reference: float = 0.0
) -> tuple[FloatArray, FloatArray]:
    """Échelle coordinates ``(value mod spacing, value)``.

    Parameters
    ----------
    values : numpy.ndarray
        Periods (or frequencies) to fold.
    spacing : float
        The folding spacing, in the same units.
    reference : float, default 0.0
        Offset subtracted before folding — useful to centre the ridge in the panel.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)``: the folded coordinate and the original value.
    """
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    v = np.asarray(values, dtype=np.float64)
    return np.mod(v - reference, spacing), v


def _default_spacing_range(sorted_values: FloatArray) -> tuple[float, float]:
    """``(minimum, maximum)`` trial spacing from the data alone.

    The upper bound is the full range (a two-tooth comb); the lower bound is half the
    smallest observed gap, floored at a tenth of the mean gap so an accidental near-pair
    of modes cannot drag the search into a decade of meaningless fine spacings.
    """
    n = int(sorted_values.size)
    span = float(sorted_values[-1] - sorted_values[0])
    gaps = np.diff(sorted_values)
    gaps = gaps[gaps > 0.0]
    smallest = float(gaps.min()) if gaps.size else span
    return max(0.5 * smallest, span / (10.0 * n)), span


def spacing_spectrum(
    values: FloatArray,
    *,
    weights: FloatArray | None = None,
    minimum_spacing: float | None = None,
    maximum_spacing: float | None = None,
    oversample: int = 20,
) -> SpacingSpectrum:
    """Scan trial spacings for a regular comb in ``values``.

    The response at trial spacing ``dP`` is the squared normalised resultant of the
    phasors ``exp(2 pi i P_j / dP)``: it reaches 1 when every value sits on one comb
    and averages ``1/n`` for values with no common spacing. Because it never looks at
    *consecutive* differences, a missing radial order costs it nothing — the surviving
    modes still land on the comb.

    Parameters
    ----------
    values : numpy.ndarray
        Periods in days (or frequencies; the tool is unit-agnostic).
    weights : numpy.ndarray, optional
        Per-value weights, normally the mode amplitudes, so strong modes dominate.
    minimum_spacing, maximum_spacing : float, optional
        Trial-spacing bounds. Default to half the smallest gap and the full range.
    oversample : int, default 20
        Trial-grid oversampling relative to the natural resolution ``1/range``.

    Returns
    -------
    SpacingSpectrum

    Examples
    --------
    >>> periods = 0.5 + 0.03 * np.arange(12)                      # doctest: +SKIP
    >>> spacing_spectrum(periods).best_spacing                    # doctest: +SKIP
    0.03
    """
    v = np.ascontiguousarray(np.asarray(values, dtype=np.float64)).ravel()
    v = v[np.isfinite(v)]
    if v.size < 3:
        raise ValueError("spacing_spectrum needs at least 3 finite values")
    order = np.argsort(v, kind="stable")
    v = v[order]
    if weights is None:
        a = np.ones(v.size, dtype=np.float64)
    else:
        a = np.asarray(weights, dtype=np.float64).ravel()
        if a.size != order.size:
            raise ValueError("weights and values must have the same length")
        a = np.abs(a[order])
        if not np.any(a > 0.0):
            a = np.ones(v.size, dtype=np.float64)

    span = float(v[-1] - v[0])
    if span <= 0.0:
        raise ValueError("spacing_spectrum needs values with a non-zero range")
    lo, hi = _default_spacing_range(v)
    if minimum_spacing is not None:
        lo = float(minimum_spacing)
    if maximum_spacing is not None:
        hi = float(maximum_spacing)
    if not 0.0 < lo < hi:
        raise ValueError("require 0 < minimum_spacing < maximum_spacing")

    # Uniform in inverse spacing: that is the variable the comb response is periodic in,
    # so a uniform step there resolves every trial equally well.
    step = 1.0 / (max(oversample, 1) * span)
    u_lo, u_hi = 1.0 / hi, 1.0 / lo
    n_trials = int(np.ceil((u_hi - u_lo) / step)) + 1
    if n_trials > _MAX_GRID:
        # Cover the whole requested range at reduced resolution rather than silently
        # truncating it.
        n_trials = _MAX_GRID
        step = (u_hi - u_lo) / (n_trials - 1)
    u = u_lo + step * np.arange(n_trials, dtype=np.float64)
    u = u[u > 0.0]

    power = _comb_response(u, v, a)
    best = _pick_spacing(u, power, v, oversample)
    return SpacingSpectrum(
        spacing=(1.0 / u)[::-1],
        power=power[::-1],
        best_spacing=float(1.0 / u[best]),
        best_power=float(power[best]),
        n_values=int(v.size),
    )


def _comb_response(
    inverse_spacing: FloatArray, values: FloatArray, weights: FloatArray
) -> FloatArray:
    """``|sum_j a_j exp(2 pi i u x_j)|^2 / (sum_j a_j)^2`` on the trial grid."""
    total = float(weights.sum())
    power = np.empty(inverse_spacing.size, dtype=np.float64)
    block = max(1, _BLOCK_ELEMS // max(1, values.size))
    for start in range(0, inverse_spacing.size, block):
        stop = min(start + block, inverse_spacing.size)
        angle = (2.0 * np.pi) * inverse_spacing[start:stop, None] * values[None, :]
        real = (np.cos(angle) * weights).sum(axis=1)
        imag = (np.sin(angle) * weights).sum(axis=1)
        power[start:stop] = (real * real + imag * imag) / (total * total)
    return power


def _comb_members(
    values: FloatArray, inverse_spacing: float, tolerance: float
) -> FloatArray:
    """The values that lie on the comb of spacing ``1/inverse_spacing``.

    The comb's own offset is read off the resultant's phase, so this does not assume the
    teeth pass through zero, and membership is then a plain phase-residual cut.
    """
    phase = values * inverse_spacing
    offset = float(np.angle(np.sum(np.exp(2j * np.pi * phase)))) / (2.0 * np.pi)
    residual = np.abs(((phase - offset + 0.5) % 1.0) - 0.5)
    return values[residual < tolerance]


def _pick_spacing(
    inverse_spacing: FloatArray,
    power: FloatArray,
    values: FloatArray,
    oversample: int,
) -> int:
    """Index of the reported comb peak, promoted past any sub-multiple of itself.

    Every value on a comb of spacing ``dP`` also sits on a comb of ``dP/m``, so a
    regular pattern responds just as strongly at all its sub-multiples and a plain
    ``argmax`` picks between them arbitrarily — reporting ``dP/2`` as the period spacing
    would halve the inferred buoyancy radius.

    The ambiguity is broken by testing the integer *multiples* of the winning spacing.
    Widening a comb by ``m`` keeps every tooth only if the true spacing was ``m``
    times wider; otherwise it drops all but every ``m``-th value and the response
    collapses. So a response at ``m·dP`` still a large fraction of the peak
    (:data:`_SUBMULTIPLE_RATIO`) means ``dP`` was the sub-multiple, and the widest such
    spacing is the one reported.

    ``power`` selects the peak — amplitude-weighted if the caller weighted it, which is
    what suppresses low-amplitude non-members. The promotion, though, is decided on the
    *unweighted* response of that peak's **own members**: it asks "do the modes this
    comb explains also land on the wider one?". Neither a wide amplitude spread — under
    which a weighted response barely notices a widened comb dropping the weak teeth, and
    would promote to a *multiple* of the truth — nor unrelated contaminating peaks,
    which drag an all-values unweighted response around, can distort the decision.
    """
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
            break  # a spacing that wide is outside the searched range
        index = int(np.searchsorted(inverse_spacing, target))
        lo = max(0, index - half_width)
        hi = min(int(inverse_spacing.size), index + half_width + 1)
        if lo >= hi:
            continue
        local = lo + int(np.argmax(counting[lo:hi]))
        if counting[local] >= _SUBMULTIPLE_RATIO * peak:
            chosen = local
    return chosen


def _fit_tilt(
    midpoints: FloatArray, spacings: FloatArray
) -> tuple[float, float, float]:
    """Least-squares ``dP = intercept + slope*P``; gives ``(intercept, slope, rms)``.

    Falls back to a flat pattern when there are too few points to constrain a tilt.
    """
    if midpoints.size == 0:
        return 0.0, 0.0, float("nan")
    if midpoints.size < 3:
        mean = float(np.mean(spacings))
        return mean, 0.0, float(np.std(spacings))
    design = np.stack([np.ones_like(midpoints), midpoints], axis=1)
    solution, *_ = np.linalg.lstsq(design, spacings, rcond=None)
    residual = spacings - design @ solution
    return (
        float(solution[0]),
        float(solution[1]),
        float(np.sqrt(np.mean(residual**2))),
    )


def _step_counts(difference: FloatArray, predicted: FloatArray) -> FloatArray:
    """How many radial orders each observed step spans, at least one.

    ``predicted`` is the tilted model ``a + b P`` evaluated at the step midpoints. Only
    ``a + b P`` over the *observed* range has to be positive — ``a`` alone is a nuisance
    parameter of the parameterisation and is legitimately negative for a steeply tilted
    series — so a non-positive prediction is treated as "cannot tell" (one order) rather
    than allowed to produce a negative step count.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(predicted > 0.0, difference / np.where(
            predicted > 0.0, predicted, 1.0
        ), 1.0)
    return np.maximum(np.round(ratio), 1.0)


def _longest_chain(
    values: FloatArray, intercept: float, slope: float, tolerance: float, max_gap: int
) -> list[int]:
    """Longest chain of ascending values consistent with ``dP = intercept + slope*P``.

    A dynamic programme over the sorted values: ``score[j]`` is the length of the best
    chain ending at ``j``, and a step ``i -> j`` is admissible when the observed
    difference is within ``tolerance`` (a fraction of the local spacing) of an integer
    multiple ``m <= max_gap`` of it. Bridging a missing radial order therefore costs a
    step but not a member.
    """
    n = int(values.size)
    score = np.ones(n, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64)
    for j in range(1, n):
        for i in range(j):
            difference = float(values[j] - values[i])
            if difference <= 0.0:
                continue
            predicted = intercept + slope * 0.5 * float(values[i] + values[j])
            if predicted <= 0.0:
                continue
            steps = int(round(difference / predicted))
            if steps < 1 or steps > max_gap:
                continue
            if abs(difference - steps * predicted) > tolerance * predicted:
                continue
            if score[i] + 1 > score[j]:
                score[j] = score[i] + 1
                parent[j] = i
    end = int(np.argmax(score))
    chain: list[int] = []
    while end >= 0:
        chain.append(end)
        end = int(parent[end])
    chain.reverse()
    return chain


def find_period_spacing(
    periods: FloatArray,
    amplitudes: FloatArray | None = None,
    *,
    settings: SpacingSettings | None = None,
    spacing: float | None = None,
) -> PeriodSpacingSeries | None:
    """Extract the longest tilted period-spacing series from a list of periods.

    The spacing is first located with :func:`spacing_spectrum` (unless supplied), then
    the chain search and the tilt fit are iterated: each pass re-fits
    ``dP(P) = a + b P`` to the current chain and re-runs the search with the improved
    model, so a rotationally tilted pattern is recovered even though the initial guess
    was a single constant spacing.

    Parameters
    ----------
    periods : numpy.ndarray
        Mode periods in days (any order).
    amplitudes : numpy.ndarray, optional
        Mode amplitudes; used to weight the comb search.
    settings : SpacingSettings, optional
        Search controls (bounds, tolerance, bridged gap, minimum length, degree).
    spacing : float, optional
        Skip the comb search and start from this spacing (days).

    Returns
    -------
    PeriodSpacingSeries or None
        ``None`` when no chain reaches ``settings.min_length``.

    Examples
    --------
    >>> series = find_period_spacing(solution.period)              # doctest: +SKIP
    >>> print(series.summary())                                    # doctest: +SKIP
    """
    cfg = settings or SpacingSettings()
    p = np.ascontiguousarray(np.asarray(periods, dtype=np.float64)).ravel()
    finite = np.isfinite(p) & (p > 0.0)
    weights = None
    if amplitudes is not None:
        amp = np.asarray(amplitudes, dtype=np.float64).ravel()
        if amp.size != p.size:
            raise ValueError("amplitudes and periods must have the same length")
        finite &= np.isfinite(amp)
        weights = amp[finite]
    original = np.flatnonzero(finite)
    p = p[finite]
    if p.size < cfg.min_length:
        return None
    order = np.argsort(p, kind="stable")
    p = p[order]
    original = original[order]
    if weights is not None:
        weights = weights[order]

    guess = spacing
    if guess is None:
        comb = spacing_spectrum(
            p,
            weights=weights if cfg.amplitude_weighted else None,
            minimum_spacing=cfg.minimum_spacing,
            maximum_spacing=cfg.maximum_spacing,
            oversample=cfg.oversample,
        )
        guess = comb.best_spacing
    if not np.isfinite(guess) or guess <= 0.0:
        return None

    intercept, slope = float(guess), 0.0
    chain: list[int] = []
    for _ in range(4):
        candidate = _longest_chain(p, intercept, slope, cfg.tolerance, cfg.max_gap)
        if len(candidate) < 2:
            break
        chain = candidate
        members = p[chain]
        difference = np.diff(members)
        midpoints = 0.5 * (members[:-1] + members[1:])
        steps = _step_counts(difference, intercept + slope * midpoints)
        if float(steps.min()) >= 2.0:
            # Not one pair in the chain is a consecutive radial order, which no real
            # series looks like: the working spacing is a sub-multiple of the true one.
            factor = float(steps.min())
            intercept *= factor
            slope *= factor
            continue
        intercept, slope, _ = _fit_tilt(midpoints, difference / steps)
    if len(chain) < cfg.min_length:
        return None

    members = p[chain]
    difference = np.diff(members)
    midpoints = 0.5 * (members[:-1] + members[1:])
    steps = _step_counts(difference, intercept + slope * midpoints)
    unit = difference / steps
    intercept, slope, rms = _fit_tilt(midpoints, unit)
    mean_spacing = float(np.mean(unit))
    return PeriodSpacingSeries(
        indices=tuple(int(original[i]) for i in chain),
        periods=members,
        spacings=difference,
        multiplicity=steps,
        midpoints=midpoints,
        mean_spacing=mean_spacing,
        slope=slope,
        intercept=intercept,
        rms=rms,
        ell=cfg.ell,
        buoyancy_radius=buoyancy_radius(mean_spacing, cfg.ell),
    )


__all__ = [
    "PeriodSpacingSeries",
    "SpacingSpectrum",
    "buoyancy_radius",
    "echelle",
    "find_period_spacing",
    "spacing_spectrum",
]
