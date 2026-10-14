"""Light-travel-time orbits in an O-C diagram: the *cyclic* period change.

A close binary with a distant third body orbits their common centre of mass,
and the eclipses arrive early or late by the light-travel time across that
orbit — the light-travel-time effect (LTTE; Irwin 1952). Nothing about the
binary's period actually changes, but the O-C diagram does, and it does so
periodically. The same is true of the quasi-periodic O-C modulation that
magnetic activity cycles produce in convective stars (Applegate 1992). Either
way the O-C is not a parabola, and what a parabola fitted to it returns is not
a period derivative:

* an outer period much longer than the baseline is indistinguishable from a
  quadratic (this is how WASP-4b's "orbital decay" turned out to be a
  companion; Winn & Stefánsson 2025);
* an outer period comparable to or *shorter* than the baseline gives a
  non-monotonic O-C whose quadratic-only ``dP/dt`` is a baseline-weighted
  average of a wobble — its value *and sign* depend on which part of the
  cycle the data happen to cover.

The second regime is common: with a 15-yr baseline, Hong et al. (2022) found
307 of 14,127 OGLE bulge contact binaries with a sinusoidal O-C and 104 more
with a sinusoid on top of a parabola, at outer periods of 5–14 yr, and Hajdu
et al. (2019) 992 LTTE candidates in OGLE-IV. The remedy is to fit the cycle
*jointly* with the ephemeris polynomial, which is what this module does:

.. math::

    \\mathrm{O\\!-\\!C}(t) = \\sum_k c_k\\,\\Delta t^k
      + A\\left[\\frac{1-e^2}{1+e\\cos\\nu}\\,\\sin(\\nu+\\omega)
      + e\\sin\\omega\\right],

Irwin's (1952) light-time orbit with semi-amplitude ``A = a_12 sin i / c``,
eccentricity ``e``, argument of periastron ``ω`` and true anomaly ``ν`` at
the outer period ``P_2``. For ``e = 0`` the bracket is ``sin(2π Δt/P_2 + ω)``
and the model is linear in ``(A cos ω, A sin ω)`` at fixed ``P_2``, so the
search is a *floating-polynomial periodogram*: for every trial outer period
the polynomial and the sinusoid are solved together by weighted least squares
and the chi-squared improvement over the polynomial alone is recorded. The
best trial is then polished by nonlinear least squares, optionally with the
eccentric terms freed.

What comes out is the outer period, the amplitude, the *secular* ``dP/dt``
with the cycle removed (``2 P c_2``), and the mass function
``f(m_3) = (a_12 sin i)^3 / P_2^2`` in solar units, from which a minimum third
mass follows for a given inner mass. Two cautions come with it. The F-test and
ΔBIC compare the joint model with the polynomial *at the best trial*, after
the outer period was searched — with a dozen timing points they are indicative,
not calibrated (Hong et al. classified their sinusoids by Bayes factor *and*
visual inspection). And an O-C alone cannot tell a third body from a magnetic
cycle: Applegate modulation has amplitudes and periods in the same range, and
only an independent handle — radial velocities, eclipse-depth or
brightness variations in phase with the O-C, a mass function that implies an
unseen star brighter than the binary — settles it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.config import PdotSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.pdot.physics import AU_LIGHT_SECONDS, DAYS_PER_YEAR
from cuperiod.pdot.result import SECONDS_PER_YEAR
from cuperiod.pdot.track import EphemerisFit, PhaseTrackResult, _build_ephemeris

_TWO_PI: Final[float] = 2.0 * np.pi
#: Eccentricity ceiling of the polish (Kepler's equation converges comfortably).
_MAX_ECCENTRICITY: Final[float] = 0.95
_KEPLER_TOL: Final[float] = 1e-12
_KEPLER_ITERS: Final[int] = 60
#: Trial outer frequencies per batched linear solve.
_FREQ_CHUNK: Final[int] = 256
#: Hard cap on the trial grid (a timing list this dense wants min_outer_period).
_MAX_TRIALS: Final[int] = 2_000_000
#: Starting eccentricity of the eccentric polish (0 is a stationary edge).
_ECC_START: Final[float] = 0.1


# --- the light-time orbit -----------------------------------------------------


def _true_anomaly(mean_anomaly: FloatArray, e: float) -> FloatArray:
    """Solve Kepler's equation by Newton iteration; returns the true anomaly."""
    m = np.mod(mean_anomaly, _TWO_PI)
    ecc = m + e * np.sin(m)
    for _ in range(_KEPLER_ITERS):
        f = ecc - e * np.sin(ecc) - m
        ecc = ecc - f / (1.0 - e * np.cos(ecc))
        if float(np.max(np.abs(f))) < _KEPLER_TOL:
            break
    half = 0.5 * ecc
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(half), np.sqrt(1.0 - e) * np.cos(half)
    )
    return np.asarray(nu, dtype=np.float64)


def ltte_delay(
    dt: Any,
    outer_period: float,
    amplitude: float,
    omega: float,
    eccentricity: float = 0.0,
    t_periastron: float = 0.0,
) -> FloatArray:
    """Irwin's (1952) light-travel-time delay (days) at ``dt``.

    Parameters
    ----------
    dt : array-like
        Times, in days, relative to the same epoch as ``t_periastron``.
    outer_period : float
        Period of the outer orbit (days).
    amplitude : float
        Semi-amplitude ``a_12 sin i / c`` of the delay (days).
    omega : float
        Argument of periastron of the binary's orbit about the triple's
        centre of mass (radians). For a circular orbit it is the phase of the
        sinusoid at ``t_periastron``.
    eccentricity : float, default 0
        Outer eccentricity; 0 gives ``amplitude * sin(2π (dt - t_peri)/P_2 + ω)``.
    t_periastron : float, default 0
        Epoch of periastron passage, relative to the same epoch as ``dt``.

    Returns
    -------
    numpy.ndarray
        The delay in days: positive when the binary is on the far side of the
        outer orbit and its eclipses arrive late.
    """
    x = np.asarray(dt, dtype=np.float64)
    m = _TWO_PI * (x - t_periastron) / outer_period
    if eccentricity == 0.0:
        return np.asarray(amplitude * np.sin(m + omega), dtype=np.float64)
    e = float(eccentricity)
    nu = _true_anomaly(m, e)
    bracket = (1.0 - e * e) / (1.0 + e * np.cos(nu)) * np.sin(nu + omega) + e * np.sin(
        omega
    )
    return np.asarray(amplitude * bracket, dtype=np.float64)


def _param_names(degree: int, eccentric: bool) -> tuple[str, ...]:
    names = ["outer_period", "amplitude", "omega"]
    if eccentric:
        names += ["eccentricity", "t_periastron"]
    names += [f"c{k}" for k in range(degree + 1)]
    return tuple(names)


def _model(
    params: FloatArray, dt: FloatArray, scaled: FloatArray, degree: int, eccentric: bool
) -> FloatArray:
    """The joint model; polynomial coefficients are in the scaled-time units."""
    p2, amp, om = float(params[0]), float(params[1]), float(params[2])
    if eccentric:
        e, tp = float(params[3]), float(params[4])
        coef = params[5:]
    else:
        e, tp = 0.0, 0.0
        coef = params[3:]
    out = ltte_delay(dt, p2, amp, om, e, tp)
    for k in range(degree + 1):
        out = out + float(coef[k]) * scaled**k
    return out


# --- the floating-polynomial scan ----------------------------------------------


def _scan(
    dt: FloatArray,
    scaled: FloatArray,
    oc: FloatArray,
    w: FloatArray,
    degree: int,
    freqs: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Weighted LSQ of polynomial + sinusoid at every trial frequency.

    Returns ``(chi2, cos_sin, poly)``: the chi-squared per trial, the
    ``(cos, sin)`` coefficients, and the polynomial coefficients (scaled
    units) — everything the polish needs to start from the best trial.
    """
    n, k = dt.size, degree + 3
    base = np.column_stack([scaled**j for j in range(degree + 1)])
    chi2 = np.empty(freqs.size, dtype=np.float64)
    cos_sin = np.empty((freqs.size, 2), dtype=np.float64)
    poly = np.empty((freqs.size, degree + 1), dtype=np.float64)
    for start in range(0, freqs.size, _FREQ_CHUNK):
        f = freqs[start : start + _FREQ_CHUNK]
        arg = _TWO_PI * f[:, None] * dt[None, :]
        x = np.empty((f.size, n, k), dtype=np.float64)
        x[:, :, : degree + 1] = base[None, :, :]
        x[:, :, -2] = np.cos(arg)
        x[:, :, -1] = np.sin(arg)
        xw = x * w[None, :, None]
        ata = np.einsum("mnk,mnl->mkl", xw, x)
        atb = np.einsum("mnk,n->mk", xw, oc)
        try:
            coef = np.linalg.solve(ata, atb[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coef = np.stack(
                [np.linalg.lstsq(x[i] * np.sqrt(w)[:, None], oc * np.sqrt(w),
                                 rcond=None)[0] for i in range(f.size)]
            )
        resid = oc[None, :] - np.einsum("mnk,mk->mn", x, coef)
        sl = slice(start, start + f.size)
        chi2[sl] = np.einsum("mn,n->m", resid * resid, w)
        cos_sin[sl] = coef[:, -2:]
        poly[sl] = coef[:, : degree + 1]
    return chi2, cos_sin, poly


def _refine_frequency(freqs: FloatArray, chi2: FloatArray, j: int) -> float:
    """Parabolic refinement of the chi-squared minimum in frequency."""
    if j <= 0 or j >= freqs.size - 1:
        return float(freqs[j])
    ym, y0, yp = float(chi2[j - 1]), float(chi2[j]), float(chi2[j + 1])
    denom = ym - 2.0 * y0 + yp
    if denom <= 0.0:
        return float(freqs[j])
    offset = 0.5 * (ym - yp) / denom
    step = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    return float(freqs[j] + np.clip(offset, -1.0, 1.0) * step)


def _polish(
    x0: FloatArray,
    dt: FloatArray,
    scaled: FloatArray,
    oc: FloatArray,
    sigma: FloatArray,
    degree: int,
    eccentric: bool,
    p2_bounds: tuple[float, float],
) -> tuple[FloatArray, FloatArray, float]:
    """Nonlinear weighted least squares from ``x0``; returns (params, cov, chi2)."""
    from scipy.optimize import least_squares

    def residuals(p: FloatArray) -> FloatArray:
        return (oc - _model(p, dt, scaled, degree, eccentric)) / sigma

    lo = [p2_bounds[0], 0.0, -np.inf]
    hi = [p2_bounds[1], np.inf, np.inf]
    if eccentric:
        lo += [0.0, -p2_bounds[1]]
        hi += [_MAX_ECCENTRICITY, p2_bounds[1]]
    lo += [-np.inf] * (degree + 1)
    hi += [np.inf] * (degree + 1)
    start = np.clip(np.asarray(x0, dtype=np.float64), lo, hi)
    fixed = p2_bounds[0] == p2_bounds[1]
    if fixed:
        # least_squares needs lo < hi; hold the outer period by a hair-thin box.
        eps = 1e-12 * max(1.0, abs(p2_bounds[0]))
        lo[0], hi[0] = p2_bounds[0] - eps, p2_bounds[0] + eps
        start[0] = p2_bounds[0]
    result = least_squares(
        residuals, start, bounds=(lo, hi), method="trf", x_scale="jac",
        max_nfev=2000,
    )
    params = np.asarray(result.x, dtype=np.float64)
    if fixed:
        params[0] = p2_bounds[0]
    chi2 = float(np.dot(result.fun, result.fun))
    jac = np.asarray(result.jac, dtype=np.float64)
    if fixed:
        jac = jac[:, 1:]
    try:
        cov_free = np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        cov_free = np.full((jac.shape[1], jac.shape[1]), np.nan)
    if fixed:
        cov = np.full((params.size, params.size), np.nan)
        cov[1:, 1:] = cov_free
    else:
        cov = cov_free
    return params, np.asarray(cov, dtype=np.float64), chi2


# --- results ------------------------------------------------------------------


@dataclass(frozen=True)
class LTTEFit:
    """A light-travel-time orbit fitted jointly with an ephemeris polynomial.

    Attributes
    ----------
    period_ref : float
        The reference period the O-C values are relative to (days).
    t_ref : float
        Expansion epoch of the polynomial; also the epoch the circular
        model's phase is quoted at (same time system as the input).
    outer_period, outer_period_err : float
        Period of the outer orbit (days) and its formal 1-sigma (NaN when the
        outer period was held fixed).
    amplitude, amplitude_err : float
        Semi-amplitude ``a_12 sin i / c`` of the delay (days).
    omega, omega_err : float
        Argument of periastron (radians, in ``(-π, π]``); for a circular fit,
        the phase of the sinusoid at ``t_ref``.
    eccentricity, eccentricity_err : float
        Outer eccentricity (0 with NaN error for a circular fit).
    t_periastron, t_periastron_err : float
        Epoch of periastron passage (same time system as the input); ``t_ref``
        with NaN error for a circular fit.
    coefficients, coefficient_errors : numpy.ndarray
        Polynomial ``(c0, c1, c2, ...)`` of ``O-C = sum c_k dt**k`` (days, dt
        in days) and their formal errors.
    pdot, pdot_err : float
        The *secular* period derivative ``2 * period_ref * c2`` (day/day) with
        the cycle removed, and its formal error; 0 for ``degree = 1``.
    chi2, chi2_reduced : float
        Weighted chi-squared and per-dof value of the joint fit.
    n_used, n_params : int
        Number of O-C points and of fitted parameters.
    aic, bic : float
        ``chi2 + 2k`` and ``chi2 + k ln n`` (the convention of
        :class:`~cuperiod.EphemerisFit`).
    delta_bic : float
        ``null.bic - bic``: positive favours the cycle over the polynomial
        alone; > 10 is conventionally strong, > 4.6 is a Bayes factor of 10.
    f_test_p : float
        p-value of the cyclic terms against the polynomial-only model (3 extra
        parameters circular, 5 eccentric). Optimistic: the outer period was
        searched, and with few points the F-distribution is only indicative.
    n_cycles : float
        Baseline of the O-C points over the outer period. Below ~1 the cycle
        and the polynomial are strongly degenerate.
    at_grid_edge : bool
        The best trial sat at an end of the outer-period grid: the true cycle
        may lie beyond it.
    eccentric : bool
        Whether the eccentric terms were fitted.
    covariance : numpy.ndarray
        Formal covariance of the parameters in ``param_names`` order (physical
        units: days for periods, amplitudes, epochs and polynomial coefficients
        per ``day**k``; radians for ``omega``).
    param_names : tuple of str
        Names of the fitted parameters, in ``covariance`` order.
    """

    period_ref: float
    t_ref: float
    outer_period: float
    outer_period_err: float
    amplitude: float
    amplitude_err: float
    omega: float
    omega_err: float
    eccentricity: float
    eccentricity_err: float
    t_periastron: float
    t_periastron_err: float
    coefficients: FloatArray
    coefficient_errors: FloatArray
    pdot: float
    pdot_err: float
    chi2: float
    chi2_reduced: float
    n_used: int
    n_params: int
    aic: float
    bic: float
    delta_bic: float
    f_test_p: float
    n_cycles: float
    at_grid_edge: bool
    eccentric: bool
    covariance: FloatArray
    param_names: tuple[str, ...]

    @property
    def degree(self) -> int:
        """Degree of the ephemeris polynomial."""
        return int(self.coefficients.size - 1)

    @property
    def outer_period_yr(self) -> float:
        """Outer period in years."""
        return self.outer_period / DAYS_PER_YEAR

    @property
    def outer_period_err_yr(self) -> float:
        """Formal 1-sigma on the outer period, in years."""
        return self.outer_period_err / DAYS_PER_YEAR

    @property
    def amplitude_s(self) -> float:
        """Semi-amplitude in seconds (the unit eclipse-timing papers quote)."""
        return self.amplitude * 86400.0

    @property
    def amplitude_err_s(self) -> float:
        """Formal 1-sigma on the semi-amplitude, in seconds."""
        return self.amplitude_err * 86400.0

    @property
    def a_sini_au(self) -> float:
        """Projected semi-major axis ``a_12 sin i`` of the binary's outer orbit (au)."""
        return self.amplitude_s / AU_LIGHT_SECONDS

    @property
    def mass_function(self) -> float:
        """The mass function ``(m_3 sin i)^3 / (m_1 + m_2 + m_3)^2`` in solar masses.

        ``f = (a_12 sin i)^3 / P_2^2`` with ``a`` in au and ``P_2`` in years.
        """
        return self.a_sini_au**3 / self.outer_period_yr**2

    @property
    def error_inflation(self) -> float:
        """``sqrt(chi2_reduced)`` floored at 1: the factor honest errors carry."""
        return float(np.sqrt(max(1.0, self.chi2_reduced)))

    @property
    def pdot_err_scaled(self) -> float:
        """:attr:`pdot_err` inflated by :attr:`error_inflation`."""
        return self.pdot_err * self.error_inflation

    @property
    def pdot_s_per_yr(self) -> float:
        """Secular dP/dt in seconds per year."""
        return self.pdot * SECONDS_PER_YEAR

    @property
    def pdot_d_per_yr(self) -> float:
        """Secular dP/dt in days per year."""
        return self.pdot * DAYS_PER_YEAR

    def minimum_third_mass(self, inner_mass: float) -> float:
        """The third-body mass (solar masses) for ``sin i = 1``.

        Solves ``m_3^3 / (m_12 + m_3)^2 = f`` for the given total inner mass;
        any inclination below 90° means a heavier companion.
        """
        from scipy.optimize import brentq

        f = self.mass_function
        if not np.isfinite(f) or f <= 0.0 or inner_mass < 0.0:
            return float("nan")

        def g(m: float) -> float:
            return m**3 - f * (inner_mass + m) ** 2

        upper = 4.0 * (f + inner_mass) + 1e-9
        return float(brentq(g, 0.0, upper))

    def _dt(self, time: Any) -> FloatArray:
        return np.asarray(time, dtype=np.float64) - self.t_ref

    def ltte_part(self, time: Any) -> FloatArray:
        """The cyclic (light-time) part of the model at ``time`` (days)."""
        return ltte_delay(
            self._dt(time),
            self.outer_period,
            self.amplitude,
            self.omega,
            self.eccentricity,
            self.t_periastron - self.t_ref,
        )

    def secular_part(self, time: Any) -> FloatArray:
        """The polynomial (ephemeris) part of the model at ``time`` (days)."""
        dt = self._dt(time)
        out = np.zeros_like(dt)
        for k, c in enumerate(self.coefficients):
            out += c * dt**k
        return out

    def predict(self, time: Any) -> FloatArray:
        """The fitted O-C (days) at ``time`` (same time system as the input)."""
        return self.secular_part(time) + self.ltte_part(time)

    def period_at(self, time: Any) -> FloatArray:
        """The *apparent* instantaneous period at ``time`` (days).

        Events observed at ``t = t_intrinsic + O-C(t)`` recur with period
        ``period_ref * (1 + d(O-C)/dt)``: a light-time orbit makes the observed
        period wobble about the secular law even though the binary's own period
        does not change. Compare with :func:`cuperiod.period_profile`.
        """
        t = np.asarray(time, dtype=np.float64)
        h = 1e-3 * self.outer_period
        slope = (self.predict(t + h) - self.predict(t - h)) / (2.0 * h)
        return np.asarray(self.period_ref * (1.0 + slope), dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dict (scalars, plus the coefficient lists)."""
        return {
            "period_ref": self.period_ref,
            "t_ref": self.t_ref,
            "outer_period": self.outer_period,
            "outer_period_err": self.outer_period_err,
            "outer_period_yr": self.outer_period_yr,
            "outer_period_err_yr": self.outer_period_err_yr,
            "amplitude": self.amplitude,
            "amplitude_err": self.amplitude_err,
            "amplitude_s": self.amplitude_s,
            "amplitude_err_s": self.amplitude_err_s,
            "omega": self.omega,
            "omega_err": self.omega_err,
            "eccentricity": self.eccentricity,
            "eccentricity_err": self.eccentricity_err,
            "t_periastron": self.t_periastron,
            "t_periastron_err": self.t_periastron_err,
            "coefficients": self.coefficients.tolist(),
            "coefficient_errors": self.coefficient_errors.tolist(),
            "pdot": self.pdot,
            "pdot_err": self.pdot_err,
            "pdot_err_scaled": self.pdot_err_scaled,
            "pdot_s_per_yr": self.pdot_s_per_yr,
            "pdot_d_per_yr": self.pdot_d_per_yr,
            "a_sini_au": self.a_sini_au,
            "mass_function": self.mass_function,
            "chi2": self.chi2,
            "chi2_reduced": self.chi2_reduced,
            "n_used": self.n_used,
            "n_params": self.n_params,
            "aic": self.aic,
            "bic": self.bic,
            "delta_bic": self.delta_bic,
            "f_test_p": self.f_test_p,
            "n_cycles": self.n_cycles,
            "at_grid_edge": self.at_grid_edge,
            "eccentric": self.eccentric,
            "degree": self.degree,
        }


@dataclass(frozen=True)
class LTTESearchResult:
    """The outer-period scan of an O-C diagram, and the fit at its best trial.

    Attributes
    ----------
    outer_period : numpy.ndarray
        Trial outer periods (days), from the longest to the shortest.
    delta_chi2 : numpy.ndarray
        Chi-squared improvement of polynomial + sinusoid over the polynomial
        alone at each trial (the floating-polynomial periodogram).
    best : LTTEFit
        The polished fit at the best trial.
    null : EphemerisFit
        The polynomial-only fit of the same degree — the quadratic ephemeris
        when ``degree = 2`` — whose ``pdot`` is what a parabola alone would
        have reported.
    t_ref : float
        Reference epoch of both fits.
    baseline : float
        Span of the O-C points (days).
    degree : int
        Degree of the ephemeris polynomial.
    eccentric : bool
        Whether the eccentric terms were fitted at the best trial.
    n_used : int
        Number of O-C points.
    meta : Mapping
        Provenance (the grid settings and the input kind).
    """

    outer_period: FloatArray
    delta_chi2: FloatArray
    best: LTTEFit
    null: EphemerisFit
    t_ref: float
    baseline: float
    degree: int
    eccentric: bool
    n_used: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def delta_bic(self) -> float:
        """``null.bic - best.bic``: positive favours the cycle."""
        return self.best.delta_bic

    @property
    def f_test_p(self) -> float:
        """p-value of the cyclic terms against the polynomial-only model."""
        return self.best.f_test_p

    @property
    def n_cycles(self) -> float:
        """Baseline over the best outer period."""
        return self.best.n_cycles

    @property
    def pdot_change(self) -> float:
        """``best.pdot - null.pdot``: how much the cycle biased the parabola's rate."""
        return self.best.pdot - self.null.pdot

    def to_dict(self, *, include_spectrum: bool = False) -> dict[str, Any]:
        """Flatten to a plain dict; the spectrum is included on request."""
        out: dict[str, Any] = {
            "t_ref": self.t_ref,
            "baseline": self.baseline,
            "degree": self.degree,
            "eccentric": self.eccentric,
            "n_used": self.n_used,
            "delta_bic": self.delta_bic,
            "f_test_p": self.f_test_p,
            "n_cycles": self.n_cycles,
            "pdot_change": self.pdot_change,
            "best": self.best.to_dict(),
            "null": self.null.to_dict(),
        }
        if include_spectrum:
            out["outer_period"] = self.outer_period.tolist()
            out["delta_chi2"] = self.delta_chi2.tolist()
        return out


# --- the search -----------------------------------------------------------------


def _resolve_input(
    data: Any,
    period: float | None,
    t0: float | None,
    errors: Any | None,
    oc: Any | None,
) -> tuple[FloatArray, FloatArray, FloatArray | None, float, str]:
    """Turn any accepted input into ``(t, oc, sigma_or_None, period_ref, kind)``."""
    if isinstance(data, PhaseTrackResult):
        if oc is not None or errors is not None:
            raise ValueError("oc= and errors= are taken from the PhaseTrackResult")
        t = np.asarray(data.t_mid, dtype=np.float64)
        return (
            t,
            np.asarray(data.delay, dtype=np.float64),
            np.asarray(data.delay_err, dtype=np.float64),
            float(data.period),
            "phase_track",
        )
    if period is None or period <= 0.0:
        raise ValueError("period= (days, positive) is required for array input")
    t = np.asarray(data, dtype=np.float64).ravel()
    sigma: FloatArray | None = None
    if errors is not None:
        sigma = np.asarray(errors, dtype=np.float64).ravel()
        if sigma.size != t.size:
            raise ValueError("errors and times length mismatch")
    if oc is not None:
        values = np.asarray(oc, dtype=np.float64).ravel()
        if values.size != t.size:
            raise ValueError("oc and times length mismatch")
        kind = "oc"
    else:
        # Event times: nearest-integer cycles, as fit_ephemeris does.
        epoch = float(np.nanmin(t)) if t0 is None else float(t0)
        cycles = np.round((t - epoch) / period)
        values = t - epoch - period * cycles
        kind = "times"
    keep = np.isfinite(t) & np.isfinite(values)
    if sigma is not None:
        keep &= np.isfinite(sigma)
    order = np.argsort(t[keep], kind="stable")
    t, values = t[keep][order], values[keep][order]
    if sigma is not None:
        sigma = sigma[keep][order]
        if np.any(sigma <= 0.0):
            raise ValueError("errors must be positive")
    return t, values, sigma, float(period), kind


def _f_test(delta: float, k_extra: int, chi2_full: float, dof_full: int) -> float:
    from scipy.stats import f as f_dist

    if dof_full < 1 or k_extra < 1:
        return float("nan")
    if chi2_full <= 0.0:
        return 0.0 if delta > 0.0 else float("nan")
    stat = (max(0.0, delta) / k_extra) / (chi2_full / dof_full)
    return float(f_dist.sf(stat, k_extra, dof_full))


def ltte_search(
    data: Any,
    *,
    period: float | None = None,
    t0: float | None = None,
    errors: Any | None = None,
    oc: Any | None = None,
    outer_period: float | None = None,
    min_outer_period: float | None = None,
    max_outer_period: float | None = None,
    samples_per_peak: int | None = None,
    degree: int | None = None,
    eccentric: bool = False,
    settings: PdotSettings | None = None,
) -> LTTESearchResult:
    """Search an O-C diagram for a light-travel-time orbit (a cyclic O-C).

    Fits, at every trial outer period, an ephemeris polynomial *and* a
    sinusoid to the O-C points by weighted least squares; refines the best
    trial by nonlinear least squares (optionally with Irwin's eccentric
    terms); and reports the joint fit next to the polynomial-only one, so the
    secular ``dP/dt`` is measured with the cycle removed rather than averaged
    over it.

    Parameters
    ----------
    data : PhaseTrackResult or array-like
        A :func:`cuperiod.phase_track` result (its segment delays are the
        O-C), or a 1-D array of times: event times (minima/maxima, cycles
        assigned against ``period``/``t0`` like :func:`cuperiod.fit_ephemeris`)
        when ``oc`` is not given, or the epochs of the ``oc`` values when it is.
    period : float, optional
        Reference period in days (required for array input).
    t0 : float, optional
        Trial epoch for cycle assignment of event times; defaults to the first.
    errors : array-like, optional
        1-sigma timing errors (days) for array input. Without them every point
        gets the same error, set from the scatter about the polynomial-only fit.
    oc : array-like, optional
        O-C values (days) at the epochs in ``data``.
    outer_period : float, optional
        Hold the outer period fixed at this value (days) and fit only the
        amplitude, phase and polynomial — to test a published orbit, or a
        magnetic-cycle period seen elsewhere. Skips the scan.
    min_outer_period, max_outer_period : float, optional
        Outer-period range of the scan (days). Defaults: the larger of four
        median point spacings and 20 reference periods, up to the baseline
        divided by ``settings.ltte_min_cycles`` (one full cycle by default —
        longer orbits are degenerate with the polynomial).
    samples_per_peak : int, optional
        Oversampling of the outer-frequency grid relative to ``1/baseline``
        (``settings.ltte_samples_per_peak``).
    degree : int, optional
        Degree of the ephemeris polynomial fitted with the cycle
        (``settings.ltte_degree``; 2 is the classical parabola + sinusoid,
        1 a sinusoid on a linear ephemeris).
    eccentric : bool, default False
        Free the eccentricity and periastron epoch in the polish. Needs well
        over a dozen points to mean anything; the circular model is what a
        survey classification uses.
    settings : PdotSettings, optional
        Source of the defaults above and of ``ltte_min_points``.

    Returns
    -------
    LTTESearchResult

    Raises
    ------
    InsufficientDataError
        Fewer O-C points than ``settings.ltte_min_points`` (or than the model
        needs), or a baseline too short to hold one cycle above the resolvable
        floor.

    Examples
    --------
    >>> track = cup.phase_track((t, mag, err), period=0.4448)      # doctest: +SKIP
    >>> res = cup.ltte_search(track)                               # doctest: +SKIP
    >>> res.best.outer_period_yr, res.best.amplitude_s, res.delta_bic  # doctest: +SKIP
    >>> res.best.pdot, res.null.pdot     # secular vs parabola-only   # doctest: +SKIP

    Notes
    -----
    The scan is linear at each trial (the sinusoid enters through ``A cos ω``
    and ``A sin ω``), which is why it is cheap and alias-free in the usual
    sense; the eccentric polish is a local refinement from the circular
    solution, tried from four periastron phases. The polynomial is fitted in
    baseline-scaled time and converted back, so decades-long timing lists do
    not lose precision to the ``dt**2`` conditioning.
    """
    psettings = settings or PdotSettings()
    deg = psettings.ltte_degree if degree is None else int(degree)
    if deg < 1 or deg > 3:
        raise ValueError(f"degree must be 1, 2 or 3, got {deg}")
    spp = psettings.ltte_samples_per_peak if samples_per_peak is None else int(
        samples_per_peak
    )
    if spp < 1:
        raise ValueError(f"samples_per_peak must be >= 1, got {spp}")

    t, values, sigma, period_ref, kind = _resolve_input(data, period, t0, errors, oc)
    n_params = deg + 1 + (5 if eccentric else 3)
    floor = max(int(psettings.ltte_min_points), n_params + 2)
    if t.size < floor:
        raise InsufficientDataError(
            f"ltte_search: need at least {floor} O-C points for a degree-{deg} "
            f"{'eccentric' if eccentric else 'circular'} fit, got {t.size}"
        )
    baseline = float(t[-1] - t[0])
    if baseline <= 0.0:
        raise InsufficientDataError("ltte_search: the O-C points span no time")

    if sigma is None:
        # Equal weights, scaled from the polynomial-only residuals.
        resid = values - np.polyval(np.polyfit(t - t.mean(), values, deg), t - t.mean())
        scale = max(float(np.std(resid)), 1e-9 * period_ref)
        sigma = np.full(t.size, scale, dtype=np.float64)
    w = 1.0 / (sigma * sigma)
    t_ref = float(np.dot(w, t) / w.sum())
    dt = t - t_ref
    time_scale = max(0.5 * baseline, 1e-300)
    scaled = dt / time_scale

    null = _build_ephemeris(
        dt, values, sigma, period_ref=period_ref, t_ref=t_ref, degree=deg
    )

    # -- the trial grid ------------------------------------------------------
    spacing = float(np.median(np.diff(t))) if t.size > 1 else baseline
    lo = (
        max(4.0 * spacing, 20.0 * period_ref)
        if min_outer_period is None
        else float(min_outer_period)
    )
    hi = (
        baseline / psettings.ltte_min_cycles
        if max_outer_period is None
        else float(max_outer_period)
    )
    if outer_period is not None:
        if outer_period <= 0.0:
            raise ValueError(f"outer_period must be positive, got {outer_period}")
        freqs = np.array([1.0 / outer_period], dtype=np.float64)
        p2_bounds = (float(outer_period), float(outer_period))
    else:
        if not (lo > 0.0 and hi > lo):
            raise InsufficientDataError(
                "ltte_search: no outer period is resolvable — the scan range "
                f"[{lo:.4g}, {hi:.4g}] d is empty (baseline {baseline:.4g} d, "
                f"median spacing {spacing:.4g} d); pass min_outer_period / "
                "max_outer_period, or lower settings.ltte_min_cycles"
            )
        df = 1.0 / (spp * baseline)
        n_trials = int(np.ceil((1.0 / lo - 1.0 / hi) / df)) + 1
        if n_trials > _MAX_TRIALS:
            raise ValueError(
                f"ltte_search: {n_trials} trial outer periods; raise "
                "min_outer_period or lower samples_per_peak"
            )
        freqs = np.linspace(1.0 / hi, 1.0 / lo, max(2, n_trials))
        p2_bounds = (lo, hi)

    chi2_grid, cos_sin, poly = _scan(dt, scaled, values, w, deg, freqs)
    j = int(np.argmin(chi2_grid))
    at_edge = outer_period is None and (j == 0 or j == freqs.size - 1)
    f_best = (
        _refine_frequency(freqs, chi2_grid, j) if outer_period is None else freqs[0]
    )
    a_cos, b_sin = float(cos_sin[j, 0]), float(cos_sin[j, 1])
    amp0 = float(np.hypot(a_cos, b_sin))
    omega0 = float(np.arctan2(a_cos, b_sin))  # a cos θ + b sin θ = A sin(θ + ω)
    x0 = np.concatenate([[1.0 / f_best, amp0, omega0], poly[j]])

    # -- polish --------------------------------------------------------------
    params, cov, chi2 = _polish(x0, dt, scaled, values, sigma, deg, False, p2_bounds)
    if eccentric:
        best: tuple[FloatArray, FloatArray, float] | None = None
        p2_c, amp_c, om_c = float(params[0]), float(params[1]), float(params[2])
        for k in range(4):
            # Same circular phase, four periastron placements around the cycle.
            start = np.concatenate(
                [
                    [p2_c, amp_c, om_c + 0.5 * np.pi * k, _ECC_START, 0.25 * k * p2_c],
                    params[3:],
                ]
            )
            trial = _polish(start, dt, scaled, values, sigma, deg, True, p2_bounds)
            if best is None or trial[2] < best[2]:
                best = trial
        assert best is not None
        params, cov, chi2 = best

    # -- unpack --------------------------------------------------------------
    n_poly = deg + 1
    names = _param_names(deg, eccentric)
    scale_vec = np.ones(params.size, dtype=np.float64)
    scale_vec[-n_poly:] = [1.0 / time_scale**k for k in range(n_poly)]
    params_phys = params * scale_vec
    cov_phys = cov * scale_vec[:, None] * scale_vec[None, :]
    errs = np.sqrt(np.clip(np.diag(cov_phys), 0.0, None))
    omega = float((params_phys[2] + np.pi) % _TWO_PI - np.pi)
    if eccentric:
        ecc, ecc_err = float(params_phys[3]), float(errs[3])
        tp = t_ref + float(params_phys[4]) % float(params_phys[0])
        tp_err = float(errs[4])
    else:
        ecc, ecc_err, tp, tp_err = 0.0, float("nan"), t_ref, float("nan")
    coef = np.asarray(params_phys[-n_poly:], dtype=np.float64)
    coef_err = np.asarray(errs[-n_poly:], dtype=np.float64)
    c2 = float(coef[2]) if deg >= 2 else 0.0
    c2_err = float(coef_err[2]) if deg >= 2 else 0.0

    n = int(t.size)
    k_full = int(params.size) - (1 if outer_period is not None else 0)
    dof = max(1, n - k_full)
    k_extra = k_full - n_poly
    bic = chi2 + k_full * float(np.log(n))
    fit = LTTEFit(
        period_ref=period_ref,
        t_ref=t_ref,
        outer_period=float(params_phys[0]),
        outer_period_err=float(errs[0]) if outer_period is None else float("nan"),
        amplitude=float(params_phys[1]),
        amplitude_err=float(errs[1]),
        omega=omega,
        omega_err=float(errs[2]),
        eccentricity=ecc,
        eccentricity_err=ecc_err,
        t_periastron=tp,
        t_periastron_err=tp_err,
        coefficients=coef,
        coefficient_errors=coef_err,
        pdot=2.0 * period_ref * c2,
        pdot_err=2.0 * period_ref * c2_err,
        chi2=chi2,
        chi2_reduced=chi2 / dof,
        n_used=n,
        n_params=k_full,
        aic=chi2 + 2.0 * k_full,
        bic=bic,
        delta_bic=null.bic - bic,
        f_test_p=_f_test(null.chi2 - chi2, k_extra, chi2, dof),
        n_cycles=baseline / float(params_phys[0]),
        at_grid_edge=bool(at_edge),
        eccentric=bool(eccentric),
        covariance=np.asarray(cov_phys, dtype=np.float64),
        param_names=names,
    )
    meta = {
        "input": kind,
        "samples_per_peak": spp,
        "min_outer_period": float(1.0 / freqs[-1]),
        "max_outer_period": float(1.0 / freqs[0]),
        "n_trials": int(freqs.size),
        "fixed_outer_period": outer_period is not None,
    }
    return LTTESearchResult(
        outer_period=1.0 / freqs,
        delta_chi2=null.chi2 - chi2_grid,
        best=fit,
        null=null,
        t_ref=t_ref,
        baseline=baseline,
        degree=deg,
        eccentric=bool(eccentric),
        n_used=n,
        meta=meta,
    )


__all__ = ["LTTEFit", "LTTESearchResult", "ltte_delay", "ltte_search"]
