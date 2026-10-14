"""What a period derivative means physically, and how small a one is measurable.

A measured ``dP/dt`` is only the beginning of a claim. This module holds the
three pieces of arithmetic that turn it into (or out of) a physical statement:

* :func:`convert_pdot` — the unit conversions the literature switches between
  (``s/s``, ``s/yr``, ``ms/yr``, ``d/yr``, ``d/Myr``, ``d/cycle``, ``1/yr``);
* :func:`mass_loss_rate` / :func:`mass_transfer_rate` and their inverses — the
  two textbook channels that change an orbital period secularly: isotropic
  (Jeans-mode) mass loss from the system and conservative mass transfer
  between the components;
* :func:`detectable_pdot` — the formal floor of the quadratic-ephemeris method
  for a given timing precision, number of timings and baseline, and the
  matching floor on a light-travel-time amplitude.

Both mass relations follow from Kepler's third law and angular-momentum
bookkeeping alone. For a wind that leaves the system isotropically (carrying
each star's specific angular momentum) the orbit widens as ``a ∝ 1/M`` and

.. math::

    \\frac{\\dot P}{P} = -2\\,\\frac{\\dot M}{M},

so a system losing mass *lengthens* its period: ``dP/dt = 2 P (mdot/M)`` with
``mdot`` the loss rate, e.g. ``2e-6 (P / 1 yr)(mdot / 1e-6 Msun/yr)(Msun / M)``.
For conservative transfer of ``mdot`` from a donor of mass ``M_d`` to an
accretor of mass ``M_a`` (total mass and orbital angular momentum fixed,
``J ∝ M_d M_a sqrt(a)``),

.. math::

    \\frac{\\dot P}{P}
    = 3\\,\\dot M_{\\rm tr}\\left(\\frac{1}{M_d} - \\frac{1}{M_a}\\right)
    = 3\\,\\frac{\\dot M_{\\rm tr}}{M_d}\\,(1 - q),\\qquad q = M_d/M_a ,

the Kwee (1958) relation: transfer from the *less* massive star lengthens the
period, transfer from the *more* massive one shortens it, and equal masses
leave it unchanged. In the limit where one star dominates the mass, both
channels reduce to ``dP/dt ≈ (2–3) P mdot/M``, which is why an eclipse-timing
``dP/dt`` is routinely quoted as a mass-transfer rate.

Routinely, and unreliably. Schaefer (2024, ApJ 966, 155) collected ``dP/dt``
for 52 cataclysmic variables and found 44% of them *positive* — the wrong sign
for magnetic-braking-driven evolution — and the rest more negative than
evolution allows by two orders of magnitude; King & Lasota (2024, "The Perils
of Pdot") conclude that an O−C rate is, in almost every case, dominated by
something other than the secular evolution (a third body's light-travel time,
a magnetic activity cycle, or plain cycle-to-cycle wander). Read the numbers
these functions return as *what the rate would mean if it were secular and
this channel alone produced it* — the physically interesting use is often the
limit a non-detection places, not the rate a detection implies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.pdot.result import DAYS_PER_MYR, SECONDS_PER_YEAR

#: Days per Julian year.
DAYS_PER_YEAR: Final[float] = 365.25

#: Light-travel time across one astronomical unit, in seconds (IAU 2012 au).
AU_LIGHT_SECONDS: Final[float] = 499.004783836

#: Factor from the dimensionless rate (day/day) to each unit that needs no period.
_PDOT_FACTORS: Final[dict[str, float]] = {
    "d/d": 1.0,
    "s/s": 1.0,
    "s/yr": SECONDS_PER_YEAR,
    "ms/yr": SECONDS_PER_YEAR * 1e3,
    "d/yr": DAYS_PER_YEAR,
    "d/Myr": DAYS_PER_MYR,
}

#: Units whose conversion needs the period.
_PER_PERIOD_UNITS: Final[frozenset[str]] = frozenset({"d/cycle", "s/cycle", "1/yr"})


def pdot_units() -> tuple[str, ...]:
    """The unit names :func:`convert_pdot` understands."""
    return tuple(_PDOT_FACTORS) + tuple(sorted(_PER_PERIOD_UNITS))


def _to_dimensionless(value: Any, unit: str, period: float | None) -> FloatArray:
    x = np.asarray(value, dtype=np.float64)
    if unit in _PDOT_FACTORS:
        return x / _PDOT_FACTORS[unit]
    if period is None:
        raise ValueError(f"unit {unit!r} needs period= (days)")
    if unit == "d/cycle":
        return x / period
    if unit == "s/cycle":
        return x / (86400.0 * period)
    # "1/yr": the fractional rate (dP/dt)/P per year
    return x * period / DAYS_PER_YEAR


def _from_dimensionless(x: FloatArray, unit: str, period: float | None) -> FloatArray:
    if unit in _PDOT_FACTORS:
        return x * _PDOT_FACTORS[unit]
    if period is None:
        raise ValueError(f"unit {unit!r} needs period= (days)")
    if unit == "d/cycle":
        return x * period
    if unit == "s/cycle":
        return x * 86400.0 * period
    return x / period * DAYS_PER_YEAR


def convert_pdot(
    value: Any, unit_from: str, unit_to: str, *, period: float | None = None
) -> Any:
    """Convert a period derivative between the units the literature uses.

    Parameters
    ----------
    value : float or array-like
        The rate(s) in ``unit_from``.
    unit_from, unit_to : str
        One of ``"d/d"`` (= ``"s/s"``, the dimensionless rate), ``"s/yr"``,
        ``"ms/yr"``, ``"d/yr"``, ``"d/Myr"``, ``"d/cycle"`` (the per-cycle
        rate ``P dP/dt``), ``"s/cycle"``, or ``"1/yr"`` (the fractional rate
        ``(dP/dt)/P`` per year). The per-cycle and fractional units need
        ``period``.
    period : float, optional
        The period in days, required for the per-cycle and fractional units.

    Returns
    -------
    float or numpy.ndarray
        The rate in ``unit_to`` (a float for scalar input).

    Examples
    --------
    >>> round(convert_pdot(1e-6, "d/yr", "s/s") * 1e9, 3)
    2.738
    >>> round(convert_pdot(-29.35, "ms/yr", "s/s") * 1e10, 2)
    -9.3
    """
    for unit in (unit_from, unit_to):
        if unit not in _PDOT_FACTORS and unit not in _PER_PERIOD_UNITS:
            raise ValueError(
                f"unknown dP/dt unit {unit!r}; choose from {', '.join(pdot_units())}"
            )
    if period is not None and period <= 0.0:
        raise ValueError(f"period must be positive, got {period}")
    out = _from_dimensionless(
        _to_dimensionless(value, unit_from, period), unit_to, period
    )
    return float(out) if out.ndim == 0 else out


# --- mass loss and mass transfer ---------------------------------------------


def _check_positive(name: str, value: Any) -> FloatArray:
    x = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError(f"{name} must be positive and finite")
    return x


def _scalarize(x: FloatArray) -> Any:
    return float(x) if x.ndim == 0 else x


def mass_loss_rate(pdot: Any, period: Any, total_mass: Any) -> Any:
    """The isotropic mass-loss rate a secular ``dP/dt`` would imply.

    Jeans-mode loss — a wind leaving the system without carrying away extra
    orbital angular momentum — widens the orbit and lengthens the period as
    ``(dP/dt)/P = 2 mdot/M``, so ``mdot = (M/2) (dP/dt)/P``.

    Parameters
    ----------
    pdot : float or array-like
        Period derivative, dimensionless (day/day). Positive = lengthening.
    period : float or array-like
        Orbital period in days.
    total_mass : float or array-like
        Total system mass in solar masses.

    Returns
    -------
    float or numpy.ndarray
        The mass-loss rate in solar masses per year, **positive for mass
        leaving the system**. A *negative* value means the period is
        shortening, which no isotropic wind can produce — the sign check that
        Schaefer (2024) found failing for 44% of cataclysmic variables.

    Examples
    --------
    A one-year binary of one solar mass losing ``1e-6`` Msun/yr:

    >>> pdot = pdot_from_mass_loss(1e-6, 365.25, 1.0)
    >>> round(pdot / 1e-6, 6)
    2.0
    >>> round(mass_loss_rate(pdot, 365.25, 1.0) / 1e-6, 6)
    1.0
    """
    p = _check_positive("period", period)
    m = _check_positive("total_mass", total_mass)
    rate = np.asarray(pdot, dtype=np.float64) / p * DAYS_PER_YEAR
    return _scalarize(0.5 * m * rate)


def pdot_from_mass_loss(mass_loss: Any, period: Any, total_mass: Any) -> Any:
    """The secular ``dP/dt`` (day/day) an isotropic wind of ``mass_loss`` produces.

    The inverse of :func:`mass_loss_rate`: ``dP/dt = 2 P (mdot/M)`` with
    ``mdot`` in solar masses per year (positive = loss) and ``P`` in days.
    """
    p = _check_positive("period", period)
    m = _check_positive("total_mass", total_mass)
    rate = 2.0 * np.asarray(mass_loss, dtype=np.float64) / m  # per year
    return _scalarize(rate * p / DAYS_PER_YEAR)


def mass_transfer_rate(pdot: Any, period: Any, m_donor: Any, m_accretor: Any) -> Any:
    """The conservative mass-transfer rate a secular ``dP/dt`` would imply.

    With total mass and orbital angular momentum conserved,
    ``(dP/dt)/P = 3 mdot (1/M_d - 1/M_a)`` (Kwee 1958): transfer from the
    less massive star lengthens the period, transfer from the more massive one
    shortens it.

    Parameters
    ----------
    pdot : float or array-like
        Period derivative, dimensionless (day/day).
    period : float or array-like
        Orbital period in days.
    m_donor, m_accretor : float or array-like
        Component masses in solar masses; the donor is the star assumed to be
        losing mass.

    Returns
    -------
    float or numpy.ndarray
        The transfer rate in solar masses per year, **positive for mass
        flowing from the donor to the accretor**. A negative value means the
        measured sign of ``dP/dt`` is incompatible with that star being the
        donor. NaN when the two masses are equal: conservative transfer between
        equal masses does not change the period at all, so no rate is implied.
    """
    p = _check_positive("period", period)
    md = _check_positive("m_donor", m_donor)
    ma = _check_positive("m_accretor", m_accretor)
    rate = np.asarray(pdot, dtype=np.float64) / p * DAYS_PER_YEAR
    with np.errstate(divide="ignore", invalid="ignore"):
        out = rate / (3.0 * (1.0 / md - 1.0 / ma))
    out = np.where(md == ma, np.nan, out)
    return _scalarize(np.asarray(out, dtype=np.float64))


def pdot_from_mass_transfer(
    mass_transfer: Any, period: Any, m_donor: Any, m_accretor: Any
) -> Any:
    """The secular ``dP/dt`` (day/day) conservative transfer produces.

    The inverse of :func:`mass_transfer_rate`, ``mass_transfer`` in solar
    masses per year (positive = donor to accretor), ``period`` in days.
    """
    p = _check_positive("period", period)
    md = _check_positive("m_donor", m_donor)
    ma = _check_positive("m_accretor", m_accretor)
    rate = 3.0 * np.asarray(mass_transfer, dtype=np.float64) * (1.0 / md - 1.0 / ma)
    return _scalarize(rate * p / DAYS_PER_YEAR)


# --- sensitivity --------------------------------------------------------------


def _design(dt: FloatArray, degree: int, outer_period: float | None) -> FloatArray:
    cols = [dt**k for k in range(degree + 1)]
    if outer_period is not None:
        arg = 2.0 * np.pi * dt / outer_period
        cols += [np.cos(arg), np.sin(arg)]
    return np.column_stack(cols)


def _covariance(x: FloatArray, w: FloatArray) -> FloatArray:
    xtw = x.T * w[None, :]
    return np.asarray(np.linalg.inv(xtw @ x), dtype=np.float64)


@dataclass(frozen=True)
class PdotSensitivity:
    """The formal floor of a quadratic-ephemeris ``dP/dt`` measurement.

    Built by :func:`detectable_pdot`. Everything here is the *formal* error
    from the timing precision alone: real O−C diagrams scatter beyond their
    formal errors (an evolving light-curve shape, spots, genuine wander), so
    treat these as the best case a data set allows, not as a prediction of the
    error bar it will deliver.

    Attributes
    ----------
    period : float
        The period whose derivative is being measured (days).
    timing_error : float
        The (rms) 1-sigma timing error per event or segment (days).
    n_timings : int
        Number of timing points.
    baseline : float
        Time span of the timing points (days).
    degree : int
        Degree of the ephemeris polynomial the rate is read from (2 is the
        classical quadratic).
    sigma_pdot : float
        Formal 1-sigma on ``dP/dt``, dimensionless (day/day).
    sigma_period : float
        Formal 1-sigma on the period from the same fit (days).
    """

    period: float
    timing_error: float
    n_timings: int
    baseline: float
    degree: int
    sigma_pdot: float
    sigma_period: float
    _dt: FloatArray
    _weights: FloatArray

    @property
    def sigma_pdot_s_per_yr(self) -> float:
        """The floor in seconds per year."""
        return self.sigma_pdot * SECONDS_PER_YEAR

    @property
    def sigma_pdot_ms_per_yr(self) -> float:
        """The floor in milliseconds per year."""
        return self.sigma_pdot * SECONDS_PER_YEAR * 1e3

    @property
    def sigma_pdot_d_per_yr(self) -> float:
        """The floor in days per year (the eclipsing-binary survey unit)."""
        return self.sigma_pdot * DAYS_PER_YEAR

    def threshold(self, n_sigma: float = 3.0) -> float:
        """The smallest ``|dP/dt|`` detectable at ``n_sigma`` (day/day)."""
        return float(n_sigma) * self.sigma_pdot

    def sigma_amplitude(self, outer_period: Any) -> Any:
        """Formal 1-sigma on a light-travel-time semi-amplitude (days).

        For a sinusoidal O−C of the given outer period fitted *jointly* with
        the ephemeris polynomial — so an outer period comparable to the
        baseline, which the polynomial partly absorbs, shows a larger floor
        than a short one. The value is the phase-averaged error of the
        semi-amplitude, ``sqrt((var_cos + var_sin)/2)``.
        """
        p2 = np.atleast_1d(np.asarray(outer_period, dtype=np.float64))
        out = np.empty(p2.shape, dtype=np.float64)
        for i, value in enumerate(p2):
            if not np.isfinite(value) or value <= 0.0:
                out[i] = np.nan
                continue
            cov = _covariance(
                _design(self._dt, self.degree, float(value)), self._weights
            )
            k = self.degree + 1
            out[i] = float(np.sqrt(0.5 * (cov[k, k] + cov[k + 1, k + 1])))
        return float(out[0]) if np.ndim(outer_period) == 0 else out

    def to_dict(self) -> dict[str, Any]:
        """Flatten the scalars to a plain dict."""
        return {
            "period": self.period,
            "timing_error": self.timing_error,
            "n_timings": self.n_timings,
            "baseline": self.baseline,
            "degree": self.degree,
            "sigma_pdot": self.sigma_pdot,
            "sigma_pdot_s_per_yr": self.sigma_pdot_s_per_yr,
            "sigma_pdot_ms_per_yr": self.sigma_pdot_ms_per_yr,
            "sigma_pdot_d_per_yr": self.sigma_pdot_d_per_yr,
            "sigma_period": self.sigma_period,
        }


def detectable_pdot(
    period: float,
    timing_error: Any,
    *,
    times: Any | None = None,
    n_timings: int | None = None,
    baseline: float | None = None,
    degree: int = 2,
) -> PdotSensitivity:
    """The formal ``dP/dt`` floor of the quadratic-ephemeris method.

    Given ``n`` timing points of precision ``sigma_t`` spread over a baseline
    ``T``, the weighted least-squares quadratic ephemeris measures the rate
    with a formal error that, for uniform sampling, is

    .. math::

        \\sigma(\\dot P) \\simeq 4\\sqrt{45}\\;\\frac{P\\,\\sigma_t}{T^2\\sqrt{n}}
        \\approx 27\\,\\frac{P\\,\\sigma_t}{T^2\\sqrt{n}} ,

    i.e. it improves with the *square* of the baseline and only with the root
    of the number of timings. The back-of-the-envelope rule "a phase shift of
    0.1 cycle accumulated over the baseline", ``dP/dt ~ 2 (0.1 P) P / T^2``,
    corresponds to a single timing point of error ``0.1 P`` at one end of the
    span; real eclipse timings reach ``~1e-3 P`` and there are many of them,
    which is why survey detections sit at ``~1e-6`` d/yr (``3e-9`` s/s) rather
    than the ``1e-7`` s/s that rule suggests.

    Parameters
    ----------
    period : float
        The period in days.
    timing_error : float or array-like
        1-sigma timing error per point, in days (a scalar, or one per point).
    times : array-like, optional
        The epochs of the timing points (days). When given, ``n_timings`` and
        ``baseline`` are taken from it, and the floor reflects the actual
        sampling — a dense modern season attached to a sparse archival tail is
        a very different design from uniform coverage.
    n_timings, baseline : optional
        Uniform sampling of ``n_timings`` points over ``baseline`` days, used
        when ``times`` is not given.
    degree : int, default 2
        Degree of the ephemeris polynomial; 3 measures the rate in the presence
        of a free cubic (which costs sensitivity but protects against an
        evolving rate — see :attr:`cuperiod.EphemerisFit.pddot`).

    Returns
    -------
    PdotSensitivity

    Examples
    --------
    Fifteen seasonal eclipse timings of a 0.4-day contact binary over 15
    years, each good to 0.001 d:

    >>> s = detectable_pdot(0.4, 0.001, n_timings=15, baseline=15 * 365.25)
    >>> print(f"{s.sigma_pdot_d_per_yr:.1e} d/yr")
    3.0e-08 d/yr
    """
    if not (np.isfinite(period) and period > 0.0):
        raise ValueError(f"period must be positive and finite, got {period}")
    if degree < 1 or degree > 4:
        raise ValueError(f"degree must be between 1 and 4, got {degree}")
    if times is not None:
        t = np.asarray(times, dtype=np.float64).ravel()
        t = t[np.isfinite(t)]
    else:
        if n_timings is None or baseline is None:
            raise ValueError("give either times= or both n_timings= and baseline=")
        if baseline <= 0.0:
            raise ValueError(f"baseline must be positive, got {baseline}")
        t = np.linspace(0.0, float(baseline), int(n_timings))
    if t.size < degree + 2:
        raise ValueError(
            f"need at least {degree + 2} timing points for a degree-{degree} "
            f"ephemeris, got {t.size}"
        )
    sigma = np.array(
        np.broadcast_to(np.asarray(timing_error, dtype=np.float64), t.shape),
        dtype=np.float64,
    )
    if np.any(~np.isfinite(sigma) | (sigma <= 0.0)):
        raise ValueError("timing_error must be positive and finite")
    w = 1.0 / (sigma * sigma)
    t_ref = float(np.dot(w, t) / w.sum())
    dt = t - t_ref
    cov = _covariance(_design(dt, degree, None), w)
    sigma_c2 = float(np.sqrt(cov[2, 2])) if degree >= 2 else float("inf")
    return PdotSensitivity(
        period=float(period),
        timing_error=float(np.sqrt(np.mean(sigma * sigma))),
        n_timings=int(t.size),
        baseline=float(t.max() - t.min()),
        degree=int(degree),
        sigma_pdot=2.0 * float(period) * sigma_c2,
        sigma_period=float(period) * float(np.sqrt(cov[1, 1])),
        _dt=dt,
        _weights=w,
    )


__all__ = [
    "AU_LIGHT_SECONDS",
    "DAYS_PER_YEAR",
    "PdotSensitivity",
    "convert_pdot",
    "detectable_pdot",
    "mass_loss_rate",
    "mass_transfer_rate",
    "pdot_from_mass_loss",
    "pdot_from_mass_transfer",
    "pdot_units",
]
