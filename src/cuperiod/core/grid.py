"""Frequency / period grid construction.

A :class:`GridSpec` is the trial grid a periodogram is evaluated on. Most Fourier
methods (GLS, MHAOV) want a *uniform frequency* grid; box/fold methods (BLS, TLS,
PDM, CE, string-length) want a *period* grid, often log-spaced. :class:`GridSpec`
stores whichever is natural and exposes both views (``frequency`` / ``period``).

The default GLS frequency grid follows the standard Lomb-Scargle heuristic (Rayleigh
resolution ``1/baseline`` divided by an oversampling factor), identical to astropy's
``autofrequency`` so power spectra line up sample-for-sample with the reference.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from cuperiod.core._typing import FloatArray

GridKind = Literal["frequency", "period"]


@dataclass(frozen=True)
class GridSpec:
    """A trial grid in either the frequency or period domain.

    Parameters
    ----------
    kind : {"frequency", "period"}
        Which quantity ``values`` holds.
    values : numpy.ndarray
        Ascending grid samples (cycles/day for frequency, days for period).
    uniform : bool, default False
        True if ``values`` is an arithmetic progression. Uniform frequency grids let
        the NUFFT GLS backend skip re-deriving ``(f0, df, nf)``.
    meta : Mapping, optional
        Method-specific grid parameters (e.g. BLS duration fractions / segments).
    """

    kind: GridKind
    values: FloatArray
    uniform: bool = False
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(np.asarray(self.values, dtype=np.float64))
        if values.ndim != 1:
            raise ValueError(f"grid values must be 1-D, got shape {values.shape}")
        object.__setattr__(self, "values", values)

    @property
    def size(self) -> int:
        """Number of grid samples."""
        return int(self.values.size)

    @property
    def frequency(self) -> FloatArray:
        """The grid as frequencies (cycles/day), ascending."""
        if self.kind == "frequency":
            return self.values
        return (1.0 / self.values)[::-1]

    @property
    def period(self) -> FloatArray:
        """The grid as periods (days), ascending."""
        if self.kind == "period":
            return self.values
        return (1.0 / self.values)[::-1]

    def uniform_frequency_params(self) -> tuple[float, float, int]:
        """Return ``(f0, df, nf)`` for a uniform frequency grid.

        Raises
        ------
        ValueError
            If this is not a frequency grid.
        """
        if self.kind != "frequency":
            raise ValueError("uniform_frequency_params requires a frequency grid")
        freq = self.values
        nf = int(freq.size)
        if nf == 0:
            return 0.0, 0.0, 0
        f0 = float(freq[0])
        df = float(freq[1] - freq[0]) if nf > 1 else 0.0
        return f0, df, nf


def uniform_frequency_grid(
    baseline: float,
    *,
    maximum_frequency: float,
    minimum_frequency: float | None = None,
    samples_per_peak: int = 5,
) -> GridSpec:
    """Build a uniform frequency grid, matching astropy ``autofrequency``.

    Parameters
    ----------
    baseline : float
        Time span ``max(t) - min(t)`` in days.
    maximum_frequency : float
        Highest trial frequency (cycles/day).
    minimum_frequency : float, optional
        Lowest trial frequency. Defaults to one Rayleigh step ``1/(s*baseline)``.
    samples_per_peak : int, default 5
        Oversampling factor; the frequency step is ``1/(samples_per_peak*baseline)``.

    Returns
    -------
    GridSpec
        A uniform frequency grid.
    """
    if baseline <= 0.0:
        raise ValueError("baseline must be positive")
    df = 1.0 / (samples_per_peak * baseline)
    f0 = df if minimum_frequency is None else float(minimum_frequency)
    nf = int(np.ceil((maximum_frequency - f0) / df)) + 1
    nf = max(nf, 1)
    freq = f0 + df * np.arange(nf, dtype=np.float64)
    return GridSpec(kind="frequency", values=freq, uniform=True)


def log_period_grid(
    *,
    minimum_period: float,
    maximum_period: float,
    n_periods: int,
) -> GridSpec:
    """Build a log-spaced period grid (general-purpose; BLS uses its own segments).

    Parameters
    ----------
    minimum_period, maximum_period : float
        Period bounds in days (``minimum_period < maximum_period``).
    n_periods : int
        Number of grid points.

    Returns
    -------
    GridSpec
        A period grid, ascending.
    """
    if not 0.0 < minimum_period < maximum_period:
        raise ValueError("require 0 < minimum_period < maximum_period")
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")
    periods = np.geomspace(minimum_period, maximum_period, n_periods)
    return GridSpec(kind="period", values=periods, uniform=False)


__all__ = ["GridSpec", "GridKind", "log_period_grid", "uniform_frequency_grid"]
