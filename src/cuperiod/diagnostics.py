"""Alias diagnostics: is the best peak the true frequency, or a window sidelobe?

Ground-based sampling is not a free choice. A survey observes at night, from one
longitude, when the target is up and the moon is down, which imprints a comb of strong
peaks on the *spectral window* :math:`W(f) = \\sum_j w_j e^{2\\pi i f t_j}` — one per
sidereal day, per synodic month, per year. Because irregular sampling convolves the
true spectrum with that window, a single sinusoid at ``f_true`` appears in the
periodogram as a whole family of peaks at

.. math::

    f = f_{\\rm true} \\pm m f_{\\rm w},

and nothing in the periodogram itself says which member of the family is the star.
Choosing the tallest is a convention, not a measurement: for sparse cadences the
1 cycle/day alias routinely comes within a few percent of the true peak, and a period
quoted without an alias check is a period a referee will ask about.

This module runs that check. It measures the window of *this* light curve's sampling
(no assumed cadence — :func:`cuperiod.spectral_window` on the actual timestamps), reads
off its strongest peaks, predicts where each would place an alias of the frequency being
examined, and reports how well the periodogram actually supports each prediction. The
output is one :class:`AliasReport`, whose :meth:`~AliasReport.summary` prints the
competitor list a period-search paper is expected to show.

Two conventions make the report readable at a glance:

**One score, both objective senses.** Every candidate carries a ``score`` normalized so
that ``1.0`` means "the periodogram likes this frequency exactly as much as the one
being diagnosed" and ``0.0`` means "as bad as the grid gets" — for maximized statistics
(GLS/BLS/MHAOV) and minimized ones (PDM/CE/string-length) alike. See
:class:`AliasCandidate`.

**Harmonics are not ambiguity.** ``2 f_true`` is present in the periodogram of any
non-sinusoidal signal and is expected structure, so harmonics and subharmonics are
reported but never set :attr:`AliasReport.ambiguous`.

Examples
--------
>>> pg = cup.periodogram((t, mag, err), "GLS")           # doctest: +SKIP
>>> report = cup.alias_diagnostics(pg, (t, mag, err))    # doctest: +SKIP
>>> print(report.summary())                              # doctest: +SKIP
>>> report.ambiguous                                     # doctest: +SKIP
True
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.grid import uniform_frequency_grid
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.core.peaks import local_maxima, select_top_peaks
from cuperiod.core.result import Periodogram

#: Sidereal day (cycles/day) — the true spacing of the nightly observing comb.
SIDEREAL_DAY_CPD: Final = 1.00273790935

#: Solar day (cycles/day) — the spacing an evenly-scheduled queue imprints.
SOLAR_DAY_CPD: Final = 1.0

#: Synodic (lunar) month in cycles/day — moon-avoidance scheduling.
SYNODIC_MONTH_CPD: Final = 1.0 / 29.530589

#: Sidereal year in cycles/day — the seasonal visibility window.
YEAR_CPD: Final = 1.0 / 365.25636

#: The classic ground-based window frequencies, used as ``(label, frequency)`` pairs
#: both to name measured window peaks and as the fallback suspects when no light curve
#: is supplied to :func:`alias_diagnostics`.
CLASSIC_WINDOW_FREQUENCIES: Final[tuple[tuple[str, float], ...]] = (
    ("sidereal-day", SIDEREAL_DAY_CPD),
    ("solar-day", SOLAR_DAY_CPD),
    ("synodic-month", SYNODIC_MONTH_CPD),
    ("year", YEAR_CPD),
)

#: Oversampling of the spectral-window grid (samples per Rayleigh width).
_WINDOW_SAMPLES_PER_PEAK: Final = 10

#: Window samples below this many Rayleigh widths belong to the DC peak at ``f -> 0``.
_DC_RAYLEIGH: Final = 2.0

#: How far a predicted alias may sit from a periodogram optimum and still count as
#: matched, in Rayleigh widths.
_MATCH_RAYLEIGH: Final = 3.0


@dataclass(frozen=True)
class WindowPeak:
    """One peak of the spectral window of the sampling.

    Attributes
    ----------
    frequency : float
        Peak frequency in cycles/day. This is an alias *spacing*, not a candidate
        period: it is the offset by which the window replicates every real signal.
    amplitude : float
        ``|W(f)|`` at the peak (dimensionless, ``<= 1``; ``|W| -> 1`` at ``f = 0``).
        NaN for the classic suspects used when no light curve was supplied.
    amplitude_ratio : float
        ``amplitude`` divided by that of the strongest non-DC window peak, so the
        leading peak is ``1.0``. NaN when the amplitudes are unknown.
    """

    frequency: float
    amplitude: float
    amplitude_ratio: float


@dataclass(frozen=True)
class AliasCandidate:
    """A frequency the sampling could confuse with the one being diagnosed.

    Attributes
    ----------
    kind : str
        What relation this candidate has to the examined frequency ``f0``, e.g.
        ``"sidereal-day-alias m=+1 (f_w=1.0027/d)"``, ``"window-alias m=-2
        (f_w=0.033849/d)"``, ``"harmonic n=2"``, ``"subharmonic n=3"``.
    frequency : float
        The *predicted* frequency (cycles/day) — where this relation says a competing
        peak should be.
    matched_frequency : float
        Frequency of the nearest local optimum of the periodogram within
        ``3/baseline`` of ``frequency``, or NaN when the periodogram has no optimum
        there (which is itself the informative outcome: the alias is not populated).
    power : float
        The periodogram statistic at ``matched_frequency`` (NaN when unmatched).
    score : float
        The competitor's strength on a scale where ``1.0`` is "as good as the examined
        peak" and ``0.0`` is "as bad as this periodogram gets"; ``0.0`` exactly when
        unmatched. For ``objective_sense="max"`` it is ``power / best_power``; for
        ``objective_sense="min"``, where a *smaller* statistic is better, it is
        ``(worst - power) / (worst - best_power)`` with ``worst`` the largest finite
        value on the grid. Scores above ``1.0`` are possible and mean the competitor
        beats the frequency being diagnosed.
    separation_rayleigh : float
        ``|matched_frequency - frequency|`` in Rayleigh widths ``1/baseline`` (NaN when
        unmatched). Much above ~1 means the match is loose.
    """

    kind: str
    frequency: float
    matched_frequency: float
    power: float
    score: float
    separation_rayleigh: float

    @property
    def matched(self) -> bool:
        """Whether a periodogram optimum was found near the predicted frequency."""
        return bool(np.isfinite(self.matched_frequency))

    @property
    def is_harmonic(self) -> bool:
        """Whether this is a harmonic/subharmonic rather than a sampling alias."""
        return self.kind.startswith(("harmonic", "subharmonic"))


@dataclass(frozen=True)
class AliasReport:
    """The alias verdict for one periodogram peak.

    Attributes
    ----------
    best_frequency : float
        The frequency that was diagnosed (cycles/day).
    best_power : float
        The periodogram statistic there; the reference every score is measured against.
    rayleigh : float
        Frequency resolution ``1/baseline`` (cycles/day).
    window_peaks : tuple of WindowPeak
        The strongest peaks of the sampling's spectral window, descending in amplitude
        (the classic suspects when no light curve was supplied).
    candidates : tuple of AliasCandidate
        Every predicted competitor, sorted by :attr:`AliasCandidate.score` descending.
    ambiguous : bool
        True when some *non-harmonic* candidate scores at least ``threshold`` — i.e.
        the sampling admits another frequency the data support about as well, and the
        period should not be quoted without a caveat.
    threshold : float
        The score at which a competitor is called serious.
    """

    best_frequency: float
    best_power: float
    rayleigh: float
    window_peaks: tuple[WindowPeak, ...]
    candidates: tuple[AliasCandidate, ...]
    ambiguous: bool
    threshold: float

    @property
    def best_period(self) -> float:
        """The diagnosed frequency as a period in days."""
        return 1.0 / self.best_frequency if self.best_frequency != 0.0 else float("inf")

    def summary(self, *, max_rows: int = 5) -> str:
        """A short human-readable report, one competitor per line.

        Parameters
        ----------
        max_rows : int, default 5
            Maximum number of candidates listed.

        Returns
        -------
        str
            Multi-line text suitable for ``print()``.
        """
        verdict = (
            f"AMBIGUOUS - a non-harmonic competitor scores >= {self.threshold:.2f}"
            if self.ambiguous
            else f"clean - no non-harmonic competitor reaches {self.threshold:.2f}"
        )
        lines = [
            f"Alias check: f = {self.best_frequency:.8g} c/d "
            f"(P = {self.best_period:.8g} d), statistic {self.best_power:.6g}",
            f"  Rayleigh 1/T = {self.rayleigh:.4g} c/d; verdict: {verdict}",
            f"  window peaks: {_format_window_peaks(self.window_peaks)}",
        ]
        if not self.candidates:
            lines.append("  competitors: none in range")
            return "\n".join(lines)
        lines.append(
            f"  {'score':>6} {'predicted':>12} {'matched':>12} {'sep/R':>7}  relation"
        )
        for candidate in self.candidates[:max_rows]:
            matched = (
                f"{candidate.matched_frequency:>12.6g}"
                if candidate.matched
                else f"{'-':>12}"
            )
            separation = (
                f"{candidate.separation_rayleigh:>7.2f}"
                if candidate.matched
                else f"{'-':>7}"
            )
            lines.append(
                f"  {candidate.score:>6.2f} {candidate.frequency:>12.6g} "
                f"{matched} {separation}  {candidate.kind}"
            )
        return "\n".join(lines)


def _format_window_peaks(peaks: Sequence[WindowPeak]) -> str:
    """One-line rendering of the window peaks for :meth:`AliasReport.summary`."""
    if not peaks:
        return "none"
    parts = []
    for peak in peaks:
        ratio = (
            f" ({peak.amplitude_ratio:.2f})"
            if np.isfinite(peak.amplitude_ratio)
            else ""
        )
        parts.append(f"{peak.frequency:.6g} c/d{ratio}")
    return ", ".join(parts)


def _sampling(data: Any) -> tuple[FloatArray, FloatArray | None]:
    """``(time, error)`` of any accepted light-curve input, bands stacked."""
    from cuperiod.api import to_input

    lc = to_input(data)
    if isinstance(lc, MultiBandLightCurve):
        time, _value, error, _band = lc.finite().stacked()
        return time, error
    finite = lc.finite()
    return finite.time, finite.error


def _measured_window_peaks(
    data: Any,
    *,
    max_window_peaks: int,
    window_max_frequency: float,
    backend: str,
) -> tuple[WindowPeak, ...]:
    """Peaks of the spectral window of ``data``'s sampling, strongest first.

    The window is evaluated from one Rayleigh-tenth up to ``window_max_frequency`` at
    ten samples per Rayleigh width, so the daily comb is resolved rather than sampled.
    Everything below :data:`_DC_RAYLEIGH` Rayleigh widths is the DC peak at ``f -> 0``
    (``|W(0)| = 1`` by construction, and it is not an alias spacing) and is dropped.
    Reported peaks are additionally required to be two Rayleigh widths apart so that
    the fine structure of one lobe cannot fill the list.
    """
    from cuperiod.prewhiten.spectrum import spectral_window

    time, error = _sampling(data)
    baseline = float(time.max() - time.min()) if time.size else 0.0
    grid = uniform_frequency_grid(
        baseline,
        maximum_frequency=window_max_frequency,
        samples_per_peak=_WINDOW_SAMPLES_PER_PEAK,
    )
    window = spectral_window(time, error, grid=grid, backend=backend)
    separation = _DC_RAYLEIGH * window.rayleigh
    candidates = local_maxima(window.amplitude)
    if candidates.size:
        keep = np.isfinite(window.amplitude[candidates]) & (
            window.frequency[candidates] >= separation
        )
        candidates = candidates[keep]
    if candidates.size == 0 or max_window_peaks <= 0:
        return ()
    chosen = select_top_peaks(
        window.frequency, window.amplitude, candidates, max_window_peaks, separation
    )
    if chosen.size == 0:
        return ()
    strongest = float(window.amplitude[chosen[0]])
    return tuple(
        WindowPeak(
            frequency=float(window.frequency[i]),
            amplitude=float(window.amplitude[i]),
            amplitude_ratio=(
                float(window.amplitude[i]) / strongest
                if strongest > 0.0
                else float("nan")
            ),
        )
        for i in chosen
    )


def _classic_window_peaks(max_window_peaks: int) -> tuple[WindowPeak, ...]:
    """The textbook ground-based alias spacings, amplitudes unknown."""
    if max_window_peaks <= 0:
        return ()
    return tuple(
        WindowPeak(frequency=f, amplitude=float("nan"), amplitude_ratio=float("nan"))
        for _label, f in CLASSIC_WINDOW_FREQUENCIES[:max_window_peaks]
    )


def _window_label(frequency: float, rayleigh: float) -> str:
    """Name a window peak after the nearest classic suspect it is consistent with.

    A measured peak can only be attributed to (say) the sidereal rather than the solar
    day when the baseline actually resolves the two, so the match must fall inside one
    Rayleigh width; otherwise the peak is reported as a plain ``"window"`` frequency.
    """
    if not np.isfinite(frequency):
        return "window"
    best_label = "window"
    best_gap = rayleigh if np.isfinite(rayleigh) else 0.0
    for label, classic in CLASSIC_WINDOW_FREQUENCIES:
        gap = abs(frequency - classic)
        if gap <= best_gap:
            best_label, best_gap = label, gap
    return best_label


def _predicted_candidates(
    f0: float,
    window_peaks: Sequence[WindowPeak],
    harmonics: Sequence[int],
    rayleigh: float,
    bounds: tuple[float, float],
) -> list[tuple[str, float]]:
    """``(kind, frequency)`` predictions, deduped at one Rayleigh width.

    Aliases come first so that a frequency which is simultaneously an alias and a
    harmonic is reported as the alias — the conservative reading, since only aliases
    can make a period ambiguous. Predictions outside the periodogram's grid, at or
    below zero, or within one Rayleigh width of ``f0`` or of an already-accepted
    prediction are dropped.
    """
    low, high = bounds
    out: list[tuple[str, float]] = []

    def add(kind: str, frequency: float) -> None:
        if not np.isfinite(frequency) or frequency <= 0.0:
            return
        if frequency < low or frequency > high:
            return
        if abs(frequency - f0) < rayleigh:
            return
        if any(abs(frequency - taken) < rayleigh for _kind, taken in out):
            return
        out.append((kind, frequency))

    for peak in window_peaks:
        label = _window_label(peak.frequency, rayleigh)
        tag = f"(f_w={peak.frequency:.5g}/d)"
        for m in (1, 2):
            offset = m * peak.frequency
            add(f"{label}-alias m=+{m} {tag}", f0 + offset)
            add(f"{label}-alias m=-{m} {tag}", abs(f0 - offset))
    for n in harmonics:
        order = int(n)
        if order < 2:
            continue
        add(f"harmonic n={order}", order * f0)
        add(f"subharmonic n={order}", f0 / order)
    return out


def _nearest_index(values: FloatArray, target: float) -> int:
    """Index of the sample of ascending ``values`` closest to ``target``."""
    right = int(np.clip(np.searchsorted(values, target), 0, values.size - 1))
    left = max(right - 1, 0)
    if abs(values[left] - target) <= abs(values[right] - target):
        return left
    return right


def _reference(
    pg: Periodogram, frequency: float | None, rayleigh: float
) -> tuple[float, float]:
    """``(frequency, power)`` of the peak being diagnosed.

    With no explicit frequency this is the periodogram's own best peak (which already
    accounts for the objective sense). With one, the reference statistic is the best
    value within a Rayleigh width of the request, so a frequency quoted to more digits
    than the grid still gets its peak's power rather than a flank sample.
    """
    if frequency is None:
        peaks = pg.best_periods(1)
        if not peaks:
            raise ValueError(
                "alias_diagnostics: the periodogram has no finite samples to diagnose"
            )
        return float(peaks[0].frequency), float(peaks[0].power)
    f0 = float(frequency)
    score = _score_array(pg)
    low = int(np.searchsorted(pg.frequency, f0 - rayleigh, side="left"))
    high = int(np.searchsorted(pg.frequency, f0 + rayleigh, side="right"))
    if high <= low:
        return f0, float(pg.power[_nearest_index(pg.frequency, f0)])
    local = np.where(np.isfinite(score[low:high]), score[low:high], -np.inf)
    return f0, float(pg.power[low + int(np.argmax(local))])


def _score_array(pg: Periodogram) -> FloatArray:
    """The periodogram statistic as a quantity to maximize."""
    return pg.power if pg.objective_sense == "max" else -pg.power


def _scorer(pg: Periodogram, best_power: float) -> Callable[[float], float]:
    """Map a matched statistic onto the competitor score of :class:`AliasCandidate`."""
    if pg.objective_sense == "max":
        reference = best_power

        def maximized(power: float) -> float:
            if not np.isfinite(power) or not np.isfinite(reference) or reference <= 0.0:
                return float("nan")
            return power / reference

        return maximized

    finite = pg.power[np.isfinite(pg.power)]
    worst = float(finite.max()) if finite.size else float("nan")
    span = worst - best_power

    def minimized(power: float) -> float:
        if not np.isfinite(power) or not np.isfinite(span) or span <= 0.0:
            return float("nan")
        return (worst - power) / span

    return minimized


def _optima(pg: Periodogram) -> tuple[FloatArray, FloatArray]:
    """``(frequency, power)`` of every interior local optimum of the periodogram."""
    score = _score_array(pg)
    indices = local_maxima(score)
    if indices.size:
        indices = indices[np.isfinite(score[indices])]
    return pg.frequency[indices], pg.power[indices]


def alias_diagnostics(
    pg: Periodogram,
    data: Any = None,
    *,
    frequency: float | None = None,
    threshold: float = 0.7,
    max_window_peaks: int = 5,
    harmonics: Sequence[int] = (2, 3),
    window_max_frequency: float = 5.5,
    backend: str = "auto",
) -> AliasReport:
    """Check whether a periodogram peak could be a sampling alias.

    The frequency under examination is compared with the family of frequencies the
    sampling cannot distinguish it from: ``f0 +/- m*f_w`` for the strongest peaks
    ``f_w`` of the spectral window (``m = 1, 2``), plus harmonics and subharmonics.
    Each prediction is looked up in the periodogram and scored against the examined
    peak, so the result is not a list of suspicions but a ranked list of how well the
    data actually support each competing frequency.

    Parameters
    ----------
    pg : Periodogram
        Any method's periodogram; both objective senses are handled.
    data : light curve, optional
        The light curve behind ``pg``, in any form :func:`cuperiod.to_input` accepts
        (including a :class:`~cuperiod.MultiBandLightCurve`, whose bands are stacked
        into one sampling). When given, the alias spacings are *measured* from this
        sampling's spectral window. When omitted, the classic ground-based suspects
        (sidereal day, solar day, synodic month, year) stand in and window amplitudes
        are reported as NaN.
    frequency : float, optional
        Frequency to diagnose in cycles/day. Defaults to the periodogram's best peak.
    threshold : float, default 0.7
        Score at or above which a non-harmonic competitor makes the result
        :attr:`~AliasReport.ambiguous`.
    max_window_peaks : int, default 5
        How many window peaks to use (strongest first).
    harmonics : sequence of int, default (2, 3)
        Harmonic orders ``n``: both ``n*f0`` and ``f0/n`` are tested. Orders below 2
        are ignored.
    window_max_frequency : float, default 5.5
        Upper limit of the spectral-window grid in cycles/day. The default covers the
        daily comb through its fifth multiple.
    backend : str, default "auto"
        Backend for the spectral window (see
        :func:`~cuperiod.prewhiten.spectrum.resolve_spectrum_backend`). Unused when
        ``data`` is None.

    Returns
    -------
    AliasReport

    Raises
    ------
    ValueError
        If the periodogram is empty or holds no finite samples.

    Notes
    -----
    Candidates falling outside the periodogram's frequency grid are dropped silently:
    an alias that was never searched cannot compete. Likewise a prediction landing
    within one Rayleigh width of the examined frequency (typically the yearly
    sidelobes of a short baseline) is not a distinguishable alternative and is dropped.

    Examples
    --------
    >>> report = alias_diagnostics(pg, (t, mag, err))       # doctest: +SKIP
    >>> report.ambiguous                                    # doctest: +SKIP
    True
    >>> report.candidates[0].kind                           # doctest: +SKIP
    'solar-day-alias m=-1 (f_w=1/d)'
    """
    if pg.size == 0:
        raise ValueError("alias_diagnostics: the periodogram is empty")
    rayleigh = 1.0 / pg.baseline if pg.baseline > 0.0 else float("inf")
    f0, best_power = _reference(pg, frequency, rayleigh)

    window_peaks = (
        _classic_window_peaks(max_window_peaks)
        if data is None
        else _measured_window_peaks(
            data,
            max_window_peaks=max_window_peaks,
            window_max_frequency=window_max_frequency,
            backend=backend,
        )
    )

    bounds = (float(pg.frequency[0]), float(pg.frequency[-1]))
    predictions = _predicted_candidates(f0, window_peaks, harmonics, rayleigh, bounds)
    optimum_frequency, optimum_power = _optima(pg)
    score_of = _scorer(pg, best_power)
    tolerance = _MATCH_RAYLEIGH * rayleigh

    candidates: list[AliasCandidate] = []
    for kind, predicted in predictions:
        index = (
            _nearest_index(optimum_frequency, predicted)
            if optimum_frequency.size
            else None
        )
        gap = (
            abs(float(optimum_frequency[index]) - predicted)
            if index is not None
            else float("inf")
        )
        if index is None or gap > tolerance:
            candidates.append(
                AliasCandidate(
                    kind=kind,
                    frequency=predicted,
                    matched_frequency=float("nan"),
                    power=float("nan"),
                    score=0.0,
                    separation_rayleigh=float("nan"),
                )
            )
            continue
        power = float(optimum_power[index])
        candidates.append(
            AliasCandidate(
                kind=kind,
                frequency=predicted,
                matched_frequency=float(optimum_frequency[index]),
                power=power,
                score=score_of(power),
                separation_rayleigh=gap / rayleigh,
            )
        )

    candidates.sort(
        key=lambda c: (
            -(c.score if np.isfinite(c.score) else -np.inf),
            c.frequency,
        )
    )
    ambiguous = any(
        not c.is_harmonic and np.isfinite(c.score) and c.score >= threshold
        for c in candidates
    )
    return AliasReport(
        best_frequency=f0,
        best_power=best_power,
        rayleigh=rayleigh,
        window_peaks=window_peaks,
        candidates=tuple(candidates),
        ambiguous=ambiguous,
        threshold=float(threshold),
    )


__all__ = [
    "CLASSIC_WINDOW_FREQUENCIES",
    "SIDEREAL_DAY_CPD",
    "SOLAR_DAY_CPD",
    "SYNODIC_MONTH_CPD",
    "YEAR_CPD",
    "AliasCandidate",
    "AliasReport",
    "WindowPeak",
    "alias_diagnostics",
]
