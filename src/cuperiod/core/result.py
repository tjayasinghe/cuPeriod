"""Result objects: :class:`Peak`, :class:`Periodogram`, :class:`MultiResult`.

A :class:`Periodogram` is the raw spectrum (frequency, period, power) plus any
method-specific per-grid quantities (BLS depth/duration/t0, ...). It is stored in
ascending-frequency order so peak finding is uniform across methods, and it knows
whether its statistic is maximized (GLS/BLS/MHAOV/TLS) or minimized (PDM/CE/string-
length) so :meth:`Periodogram.best_periods` always returns the most significant peaks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from cuperiod.core._typing import Float32Array, FloatArray, IntArray
from cuperiod.core.peaks import (
    local_maxima,
    peak_preserving_downsample,
    select_diverse_peaks,
    select_top_peaks,
)

ObjectiveSense = Literal["max", "min"]


@dataclass(frozen=True)
class Peak:
    """A single periodogram peak.

    Parameters
    ----------
    period : float
        Period in days.
    frequency : float
        Frequency in cycles/day.
    power : float
        The method's statistic at this peak (large = significant for max-objective
        methods; small = significant for min-objective methods).
    rank : int
        1-based significance rank within the result.
    extra : Mapping[str, float], optional
        Method-specific scalars at this peak (e.g. ``depth``, ``duration``, ``t0``,
        ``depth_snr``, ``sde`` for BLS; ``fap`` for GLS).
    """

    period: float
    frequency: float
    power: float
    rank: int
    extra: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int]:
        """Flatten to a plain dict (peak scalars + extras)."""
        out: dict[str, float | int] = {
            "period": self.period,
            "frequency": self.frequency,
            "power": self.power,
            "rank": self.rank,
        }
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class Periodogram:
    """A computed periodogram for a single light curve and method.

    Attributes
    ----------
    method, backend : str
        The method name and the concrete backend that produced this spectrum.
    frequency, period, power : numpy.ndarray
        The raw spectrum, ascending in frequency (``period = 1/frequency``).
    objective_sense : {"max", "min"}
        Whether peaks are maxima or minima of ``power``.
    n_samples : int
        Number of finite light-curve points used.
    baseline : float
        Time baseline in days (sets the Rayleigh resolution ``1/baseline``).
    extras : Mapping[str, numpy.ndarray]
        Per-grid method-specific arrays, aligned with ``frequency``.
    meta : Mapping
        Free-form metadata carried from the light curve.
    """

    method: str
    backend: str
    frequency: FloatArray
    period: FloatArray
    power: FloatArray
    objective_sense: ObjectiveSense
    n_samples: int
    baseline: float
    extras: Mapping[str, FloatArray] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spectrum(
        cls,
        *,
        method: str,
        backend: str,
        frequency: FloatArray,
        power: FloatArray,
        objective_sense: ObjectiveSense,
        n_samples: int,
        baseline: float,
        extras: Mapping[str, FloatArray] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Periodogram:
        """Build a :class:`Periodogram`, sorting everything into ascending frequency."""
        frequency = np.ascontiguousarray(frequency, dtype=np.float64)
        power = np.ascontiguousarray(power, dtype=np.float64)
        order = np.argsort(frequency, kind="stable")
        frequency = frequency[order]
        power = power[order]
        sorted_extras = {
            k: np.ascontiguousarray(v, dtype=np.float64)[order]
            for k, v in (extras or {}).items()
        }
        with np.errstate(divide="ignore"):
            period = 1.0 / frequency
        return cls(
            method=method,
            backend=backend,
            frequency=frequency,
            period=period,
            power=power,
            objective_sense=objective_sense,
            n_samples=n_samples,
            baseline=baseline,
            extras=sorted_extras,
            meta=dict(meta or {}),
        )

    @property
    def size(self) -> int:
        """Number of grid samples."""
        return int(self.frequency.size)

    def _score(self) -> FloatArray:
        """Power converted to a maximize-me score."""
        return self.power if self.objective_sense == "max" else -self.power

    def best_periods(
        self,
        n: int = 10,
        *,
        min_separation_rayleigh: float = 3.0,
        alias_diverse: bool = False,
        harmonic_max: int = 8,
    ) -> list[Peak]:
        """Return the ``n`` most significant peaks.

        Parameters
        ----------
        n : int, default 10
            Maximum number of peaks.
        min_separation_rayleigh : float, default 3.0
            Minimum peak separation in Rayleigh widths (``1/baseline``).
        alias_diverse : bool, default False
            If True, skip candidates alias- or harmonic-related to a chosen peak
            (recommended for box searches).
        harmonic_max : int, default 8
            Highest harmonic order considered when ``alias_diverse`` is set.

        Returns
        -------
        list of Peak
            Ranked most-significant first.
        """
        if self.size == 0:
            return []
        score = self._score()
        candidates = self._peak_candidates(score)
        if candidates.size == 0:
            return []
        tol = (
            min_separation_rayleigh / self.baseline if self.baseline > 0.0 else 0.0
        )
        if alias_diverse:
            chosen = select_diverse_peaks(
                self.frequency, score, candidates, n, max(tol, 1e-12), harmonic_max
            )
        else:
            chosen = select_top_peaks(self.frequency, score, candidates, n, tol)
        return [self._make_peak(int(idx), rank) for rank, idx in enumerate(chosen, 1)]

    def _peak_candidates(self, score: FloatArray) -> IntArray:
        """Finite peak candidates: interior local maxima plus boundary peaks.

        :func:`~cuperiod.core.peaks.local_maxima` excludes the grid endpoints by
        contract, but a true peak can sit at the lowest or highest frequency (a signal
        whose period is comparable to the baseline), so boundary samples that exceed
        their single neighbor are added too. Non-finite scores and non-finite periods
        (e.g. a ``frequency == 0`` sample, period ``inf``) are dropped so they can never
        be reported as a best period.
        """
        candidates = local_maxima(score)
        boundary: list[int] = []
        if self.size == 1:
            boundary.append(0)
        elif self.size >= 2:
            if score[0] >= score[1]:
                boundary.append(0)
            if score[-1] > score[-2]:
                boundary.append(self.size - 1)
        if boundary:
            candidates = np.concatenate(
                [candidates, np.asarray(boundary, dtype=np.int64)]
            )
        if candidates.size:
            ok = np.isfinite(score[candidates]) & np.isfinite(self.period[candidates])
            candidates = candidates[ok]
        if candidates.size == 0:
            finite = np.flatnonzero(np.isfinite(score) & np.isfinite(self.period))
            if finite.size == 0:
                return np.empty(0, dtype=np.int64)
            return finite[[int(np.argmax(score[finite]))]]
        return candidates

    def _make_peak(self, idx: int, rank: int) -> Peak:
        extra = {k: float(v[idx]) for k, v in self.extras.items()}
        return Peak(
            period=float(self.period[idx]),
            frequency=float(self.frequency[idx]),
            power=float(self.power[idx]),
            rank=rank,
            extra=extra,
        )

    def best_period(self) -> float:
        """The single most significant period in days (NaN if none is finite)."""
        peaks = self.best_periods(1)
        return peaks[0].period if peaks else float("nan")

    def downsample(self, n_points: int = 2000) -> tuple[Float32Array, Float32Array]:
        """Peak-preserving downsample of ``(frequency, power)`` to ``n_points``."""
        return peak_preserving_downsample(self.frequency, self.power, n_points)

    def to_dict(
        self, *, n_best: int = 10, include_raw: bool = False
    ) -> dict[str, Any]:
        """Serialize to a plain dict (peaks always; raw spectrum optional)."""
        out: dict[str, Any] = {
            "method": self.method,
            "backend": self.backend,
            "n_samples": self.n_samples,
            "baseline": self.baseline,
            "objective_sense": self.objective_sense,
            "peaks": [p.to_dict() for p in self.best_periods(n_best)],
        }
        if include_raw:
            freq, power = self.downsample()
            out["frequency"] = freq.tolist()
            out["power"] = power.tolist()
        return out


@dataclass(frozen=True)
class MultiResult:
    """Several methods' periodograms for one light curve, keyed by method name."""

    results: Mapping[str, Periodogram]

    def __getitem__(self, method: str) -> Periodogram:
        return self.results[method.upper()]

    def __iter__(self) -> Any:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def keys(self) -> Any:
        """Method names present."""
        return self.results.keys()

    def items(self) -> Any:
        """``(method, Periodogram)`` pairs."""
        return self.results.items()

    def best_periods(self, n: int = 10, **kwargs: Any) -> dict[str, list[Peak]]:
        """The N-best peaks per method."""
        return {k: v.best_periods(n, **kwargs) for k, v in self.results.items()}

    def to_dict(self, *, n_best: int = 10, include_raw: bool = False) -> dict[str, Any]:
        """Serialize every method's result."""
        return {
            k: v.to_dict(n_best=n_best, include_raw=include_raw)
            for k, v in self.results.items()
        }


__all__ = ["MultiResult", "ObjectiveSense", "Peak", "Periodogram"]
