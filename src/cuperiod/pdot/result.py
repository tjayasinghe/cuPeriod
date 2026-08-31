"""Result objects for the period-change (dP/dt) tools.

:class:`PdotSearchResult` is the 2-D analogue of a :class:`~cuperiod.core.result.
Periodogram`: the method's statistic evaluated on a (dP/dt)/P x frequency grid,
plus the refined best cell as a :class:`PdotBest`. The dP/dt axis is stored as
the fractional rate ``pdot_over_p`` = (dP/dt)/P in 1/day, which is the natural
search coordinate (it equals the negative fractional frequency-drift rate
``-fdot/f`` and is what the time-warp scan steps uniformly).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.result import ObjectiveSense, Periodogram

#: Seconds per Julian year: converts a dimensionless dP/dt (day/day) to s/yr.
SECONDS_PER_YEAR: float = 86400.0 * 365.25

#: Days per megayear: converts a dimensionless dP/dt (day/day) to d/Myr.
DAYS_PER_MYR: float = 365.25 * 1e6


@dataclass(frozen=True)
class PdotBest:
    """The best (period, dP/dt) cell of a search, with unit conversions.

    Attributes
    ----------
    period : float
        Best period in days, at the reference epoch ``t_ref``.
    pdot : float
        Best period derivative dP/dt, dimensionless (day/day). Positive means
        the period is lengthening.
    frequency : float
        ``1/period`` in cycles/day (the instantaneous frequency at ``t_ref``).
    fdot : float
        Frequency derivative df/dt in 1/day^2 (``-pdot/period^2``).
    pdot_over_p : float
        Fractional rate (dP/dt)/P in 1/day (the search coordinate).
    power : float
        The method's statistic at the best grid cell (unrefined).
    zero_pdot_power : float
        The best statistic along the dP/dt = 0 row — what a constant-period
        search of the same window would have found. Comparing it with ``power``
        (in the method's objective sense) measures what the extra axis bought.
    t_ref : float
        Reference epoch (same time system as the input, e.g. JD): the epoch at
        which ``period`` holds and around which the quadratic drift is expanded.
    """

    period: float
    pdot: float
    frequency: float
    fdot: float
    pdot_over_p: float
    power: float
    zero_pdot_power: float
    t_ref: float

    @property
    def pdot_s_per_yr(self) -> float:
        """dP/dt in seconds per year (the eclipse-timing unit)."""
        return self.pdot * SECONDS_PER_YEAR

    @property
    def pdot_ms_per_yr(self) -> float:
        """dP/dt in milliseconds per year (the hot-Jupiter decay unit)."""
        return self.pdot * SECONDS_PER_YEAR * 1e3

    @property
    def pdot_d_per_myr(self) -> float:
        """dP/dt in days per megayear (the classical-pulsator unit)."""
        return self.pdot * DAYS_PER_MYR

    @property
    def dlnp_dt_per_yr(self) -> float:
        """Fractional rate d(ln P)/dt per year (the Mira / delta Sct unit)."""
        return self.pdot_over_p * 365.25

    @property
    def timescale_yr(self) -> float:
        """Characteristic period-change timescale ``P / |dP/dt|`` in years."""
        if self.pdot == 0.0:
            return float("inf")
        return abs(self.period / self.pdot) / 365.25

    def to_dict(self) -> dict[str, float]:
        """Flatten to a plain dict (all scalars, converted units included)."""
        return {
            "period": self.period,
            "pdot": self.pdot,
            "pdot_s_per_yr": self.pdot_s_per_yr,
            "pdot_ms_per_yr": self.pdot_ms_per_yr,
            "pdot_d_per_myr": self.pdot_d_per_myr,
            "dlnp_dt_per_yr": self.dlnp_dt_per_yr,
            "frequency": self.frequency,
            "fdot": self.fdot,
            "pdot_over_p": self.pdot_over_p,
            "power": self.power,
            "zero_pdot_power": self.zero_pdot_power,
            "t_ref": self.t_ref,
        }


@dataclass(frozen=True)
class PdotSearchResult:
    """A joint (period, dP/dt) search result for a single light curve.

    Attributes
    ----------
    method, backend : str
        The wrapped periodogram method and the concrete backend that ran it.
    frequency : numpy.ndarray
        Frequency axis (cycles/day), ascending, shared by every row.
    pdot_over_p : numpy.ndarray
        (dP/dt)/P axis (1/day), ascending; contains an exact 0 row unless an
        explicit range excluded it.
    power : numpy.ndarray
        The method's statistic, shape ``(pdot_over_p.size, frequency.size)``.
    objective_sense : {"max", "min"}
        Whether significance means large or small ``power``.
    best : PdotBest
        The refined best cell.
    n_samples : int
        Number of finite light-curve points used.
    baseline : float
        Unwarped time baseline in days.
    t_ref : float
        Reference epoch (weighted mean epoch of the finite points).
    meta : Mapping
        Search provenance plus metadata carried from the light curve.
    """

    method: str
    backend: str
    frequency: FloatArray
    pdot_over_p: FloatArray
    power: FloatArray
    objective_sense: ObjectiveSense
    best: PdotBest
    n_samples: int
    baseline: float
    t_ref: float
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape ``(n_pdot, n_frequency)``."""
        return (int(self.pdot_over_p.size), int(self.frequency.size))

    @property
    def period(self) -> FloatArray:
        """The frequency axis as periods (days), aligned with ``frequency``."""
        with np.errstate(divide="ignore"):
            return 1.0 / self.frequency

    @property
    def pdot(self) -> FloatArray:
        """A ``(n_pdot, n_frequency)`` grid of dP/dt values (day/day).

        The scan steps (dP/dt)/P uniformly, so the dimensionful dP/dt of a cell
        depends on its frequency: ``pdot[j, i] = pdot_over_p[j] / frequency[i]``.
        """
        with np.errstate(divide="ignore"):
            return self.pdot_over_p[:, None] / self.frequency[None, :]

    def _score(self) -> FloatArray:
        return self.power if self.objective_sense == "max" else -self.power

    def pdot_profile(self) -> FloatArray:
        """Best statistic per dP/dt row (length ``n_pdot``), for 1-D plotting."""
        score = self._score()
        idx = np.argmax(score, axis=1)
        return np.asarray(
            self.power[np.arange(score.shape[0]), idx], dtype=np.float64
        )

    def frequency_profile(self) -> FloatArray:
        """Best statistic per frequency column (length ``n_frequency``)."""
        score = self._score()
        idx = np.argmax(score, axis=0)
        return np.asarray(
            self.power[idx, np.arange(score.shape[1])], dtype=np.float64
        )

    def zero_pdot_periodogram(self) -> Periodogram:
        """The dP/dt = 0 row as an ordinary :class:`Periodogram`.

        Directly comparable to a constant-period run of the same method on the
        same window (it *is* one). Uses the row nearest 0 if an explicit range
        excluded the exact zero.
        """
        j = int(np.argmin(np.abs(self.pdot_over_p)))
        return Periodogram.from_spectrum(
            method=self.method,
            backend=self.backend,
            frequency=self.frequency,
            power=self.power[j],
            objective_sense=self.objective_sense,
            n_samples=self.n_samples,
            baseline=self.baseline,
            meta=dict(self.meta),
        )

    def to_dict(self, *, include_map: bool = False) -> dict[str, Any]:
        """Serialize to a plain dict (best cell always; the 2-D map optional)."""
        out: dict[str, Any] = {
            "method": self.method,
            "backend": self.backend,
            "objective_sense": self.objective_sense,
            "n_samples": self.n_samples,
            "baseline": self.baseline,
            "t_ref": self.t_ref,
            "best": self.best.to_dict(),
        }
        if include_map:
            out["frequency"] = self.frequency.tolist()
            out["pdot_over_p"] = self.pdot_over_p.tolist()
            out["power"] = self.power.tolist()
        return out


__all__ = [
    "DAYS_PER_MYR",
    "SECONDS_PER_YEAR",
    "PdotBest",
    "PdotSearchResult",
]
