"""Segmented template-phase tracking and quadratic-ephemeris fits.

:func:`phase_track` is an automated O-C (observed minus calculated) analysis:
it fits one multi-harmonic template to the whole light curve at a fixed period,
splits the data into time segments, measures the *time shift* of the template
that best matches each segment (with a Fisher-information uncertainty), and
fits the shifts against a quadratic ephemeris. A constant period derivative
makes the O-C curve a parabola,

    O-C(t) = c0 + c1*dt + c2*dt**2,   dt = t - t_ref,
    dP/dt  = 2 * P * c2,    P_true ~ P * (1 + c1),

(positive dP/dt: events arrive progressively late), so ``c2`` *is* the
measurement, an F-test against the linear model its significance.

:func:`fit_ephemeris` exposes the same quadratic fit for user-supplied timing
lists (times of minima/maxima from the literature or from dense follow-up).

Unlike :func:`cuperiod.pdot_search` (detection over a grid), this is the
*measurement* tool: it propagates per-point uncertainties into ``dP/dt``. Chain
them: search first, then track at the found period. A drift left in the fold
would smear the template, so the tracker re-derives it on drift-corrected
times until self-consistent (``template_iterations``); when the accumulated
drift exceeds a cycle or so, start it at the search's value (``pdot=``) so the
first template is already sharp. Either way the delays and fits are reported
against the constant-period ephemeris.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import PdotSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.lightcurve import LightCurve
from cuperiod.pdot.models import PeriodModel, get_model
from cuperiod.pdot.result import SECONDS_PER_YEAR

#: Shift-scan sampling: the coarse grid must resolve the sharpest template
#: feature (scale P / (2 * harmonics)), with margin.
_SCAN_PER_HARMONIC: int = 32
_SCAN_MIN: int = 128
#: Newton polish iterations after the parabolic seed.
_NEWTON_STEPS: int = 6


# --- template ----------------------------------------------------------------


@dataclass(frozen=True)
class PhaseTemplate:
    """A fixed-frequency multi-harmonic light-curve template.

    Model: ``offset + sum_m a_m*cos(2*pi*m*f0*dt) + b_m*sin(2*pi*m*f0*dt)``
    with ``dt`` referenced to the track's ``t_ref``.
    """

    frequency: float
    offset: float
    cos_coeffs: FloatArray
    sin_coeffs: FloatArray
    chi2_reduced: float

    @property
    def n_harmonics(self) -> int:
        """Number of harmonics."""
        return int(self.cos_coeffs.size)

    @property
    def amplitudes(self) -> FloatArray:
        """Per-harmonic amplitudes ``sqrt(a_m**2 + b_m**2)``."""
        return np.hypot(self.cos_coeffs, self.sin_coeffs)

    def __call__(self, dt: FloatArray, shift: float = 0.0) -> FloatArray:
        """Evaluate the template at ``dt`` (days), optionally time-shifted."""
        return self.offset + self._shape(dt - shift)

    def _shape(self, x: FloatArray) -> FloatArray:
        m = np.arange(1, self.n_harmonics + 1, dtype=np.float64)
        ang = (2.0 * np.pi * self.frequency) * x[:, None] * m[None, :]
        return np.cos(ang) @ self.cos_coeffs + np.sin(ang) @ self.sin_coeffs

    def _shape_deriv(self, x: FloatArray) -> FloatArray:
        """d(shape)/dx — the light-curve slope, for Fisher uncertainties."""
        m = np.arange(1, self.n_harmonics + 1, dtype=np.float64)
        omega = 2.0 * np.pi * self.frequency * m
        ang = (2.0 * np.pi * self.frequency) * x[:, None] * m[None, :]
        return np.sin(ang) @ (-omega * self.cos_coeffs) + np.cos(ang) @ (
            omega * self.sin_coeffs
        )


def _design(dt: FloatArray, f0: float, n_harmonics: int) -> FloatArray:
    m = np.arange(1, n_harmonics + 1, dtype=np.float64)
    ang = (2.0 * np.pi * f0) * dt[:, None] * m[None, :]
    return np.hstack([np.ones((dt.size, 1)), np.cos(ang), np.sin(ang)])


def _fit_template_order(
    dt: FloatArray, y: FloatArray, w: FloatArray, f0: float, n_harmonics: int
) -> tuple[PhaseTemplate, float]:
    """Weighted LSQ template of a given order; returns (template, chi2)."""
    x = _design(dt, f0, n_harmonics)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)
    resid = y - x @ coef
    chi2 = float(np.dot(w, resid * resid))
    dof = max(1, dt.size - (1 + 2 * n_harmonics))
    tpl = PhaseTemplate(
        frequency=f0,
        offset=float(coef[0]),
        cos_coeffs=np.asarray(coef[1 : 1 + n_harmonics], dtype=np.float64),
        sin_coeffs=np.asarray(coef[1 + n_harmonics :], dtype=np.float64),
        chi2_reduced=chi2 / dof,
    )
    return tpl, chi2


def fit_template(
    dt: FloatArray,
    y: FloatArray,
    w: FloatArray,
    f0: float,
    *,
    n_harmonics: int | None = None,
    max_harmonics: int = 10,
    weighted: bool = True,
) -> PhaseTemplate:
    """Fit the fixed-frequency template, BIC-selecting the order if not given.

    With per-point errors (``weighted``), BIC = chi2 + k*ln(N); without, the
    Gaussian profile form ``N*ln(RSS/N) + k*ln(N)`` is used.
    """
    n = dt.size
    if n_harmonics is not None:
        return _fit_template_order(dt, y, w, f0, n_harmonics)[0]
    best: PhaseTemplate | None = None
    best_bic = np.inf
    top = max(1, min(max_harmonics, (n - 2) // 2))
    log_n = np.log(n)
    for order in range(1, top + 1):
        tpl, chi2 = _fit_template_order(dt, y, w, f0, order)
        k = 1 + 2 * order
        if weighted:
            bic = chi2 + k * log_n
        else:
            # w is 1/N each: chi2 is then RSS/N exactly.
            bic = n * np.log(max(chi2, np.finfo(float).tiny)) + k * log_n
        if bic < best_bic - 1e-12:
            best, best_bic = tpl, bic
    assert best is not None
    return best


# --- per-segment shift fit ---------------------------------------------------


@dataclass(frozen=True)
class SegmentShift:
    """One segment's measured template delay (its O-C point)."""

    t_mid: float
    delay: float
    delay_err: float
    n_points: int
    duration: float
    chi2_reduced: float


def _fit_segment_shift(
    dt: FloatArray,
    y: FloatArray,
    w: FloatArray,
    template: PhaseTemplate,
    *,
    free_offset: bool,
    error_scale: float,
) -> tuple[float, float, float]:
    """Best time shift of the fixed template for one segment.

    Coarse chi-squared scan over one full period, parabolic seed, Newton
    polish (Gauss-Newton curvature), Fisher sigma from the slope of the
    template at the data points (offset-marginalized when ``free_offset``).
    Returns ``(delay, delay_err, chi2_reduced)``; the delay is in
    ``[-P/2, P/2)`` — cycle assignment happens at the track level.
    """
    period = 1.0 / template.frequency
    n_scan = max(_SCAN_MIN, _SCAN_PER_HARMONIC * template.n_harmonics)
    shifts = (np.arange(n_scan, dtype=np.float64) / n_scan - 0.5) * period
    sum_w = float(w.sum())

    # chi2 on the coarse grid, vectorized over shifts: (n_scan, n_pts).
    shape = np.empty((n_scan, dt.size), dtype=np.float64)
    for k, s in enumerate(shifts):
        shape[k] = template._shape(dt - s)
    resid0 = y[None, :] - template.offset - shape
    if free_offset:
        c = (resid0 @ w) / sum_w
        resid0 = resid0 - c[:, None]
    chi2_grid = (resid0 * resid0) @ w
    k0 = int(np.argmin(chi2_grid))

    # Parabolic seed on the (periodic) grid.
    km, kp = (k0 - 1) % n_scan, (k0 + 1) % n_scan
    sm, s0, sp = chi2_grid[km], chi2_grid[k0], chi2_grid[kp]
    denom = sm - 2.0 * s0 + sp
    step = period / n_scan
    off = 0.5 * (sm - sp) / denom if denom > 0 else 0.0
    delay = shifts[k0] + step * float(np.clip(off, -0.5, 0.5))

    # Newton (Gauss-Newton) polish with analytic derivatives.
    for _ in range(_NEWTON_STEPS):
        x = dt - delay
        model = template.offset + template._shape(x)
        slope = -template._shape_deriv(x)  # d(model)/d(delay)
        resid = y - model
        if free_offset:
            resid = resid - float(np.dot(w, resid)) / sum_w
            slope = slope - float(np.dot(w, slope)) / sum_w
        grad = -2.0 * float(np.dot(w, resid * slope))
        curv = 2.0 * float(np.dot(w, slope * slope))
        if curv <= 0.0:
            break
        move = -grad / curv
        delay += float(np.clip(move, -step, step))
    delay = (delay + 0.5 * period) % period - 0.5 * period

    x = dt - delay
    slope = -template._shape_deriv(x)
    resid = y - template.offset - template._shape(x)
    n_free = 1
    if free_offset:
        resid = resid - float(np.dot(w, resid)) / sum_w
        slope = slope - float(np.dot(w, slope)) / sum_w
        n_free = 2
    chi2 = float(np.dot(w, resid * resid))
    dof = max(1, dt.size - n_free)
    fisher = float(np.dot(w, slope * slope))
    if fisher <= 0.0:
        return delay, np.inf, chi2 / dof
    sigma = error_scale / np.sqrt(fisher)
    return delay, sigma, chi2 / dof


# --- quadratic ephemeris fit -------------------------------------------------


@dataclass(frozen=True)
class EphemerisFit:
    """A weighted polynomial fit to O-C delays.

    Degree 1 is the constant-period ephemeris and degree 2 the classical
    quadratic (constant ``dP/dt``). Degree 3 and above add no new *rate* — the
    instantaneous ``pdot`` is still ``2*period_ref*c2`` because higher terms
    vanish from the second derivative at ``t_ref`` — but they absorb curvature
    that a changing rate puts into the O-C, which both de-biases ``pdot`` and,
    through :attr:`pddot`, measures the change itself.

    Attributes
    ----------
    period_ref : float
        The reference period the O-C values are relative to (days).
    t_ref : float
        Expansion epoch of the polynomial (same time system as the input).
    coefficients : numpy.ndarray
        ``(c0, c1, c2, ...)`` of ``O-C = sum_k c_k * dt**k`` (days, dt in days).
    covariance : numpy.ndarray
        Covariance of the coefficients (from the formal input errors).
    pdot : float
        ``2 * period_ref * c2`` (0 for a linear fit), day/day.
    pdot_err : float
        Formal 1-sigma on ``pdot``; multiply by ``sqrt(chi2_reduced)`` when the
        fit is poor (documented, not applied silently — see
        :attr:`pdot_err_scaled`).
    pddot : float
        Second period derivative ``d2P/dt2`` at ``t_ref`` in 1/day, from
        ``period_ref * (6*c3 + 8*c2**2)``; NaN below degree 3. Nonzero means the
        rate itself is evolving: ``pddot = 0`` is the constant-dP/dt model and
        ``pddot = pdot**2/period_ref`` the exponential one.
    pddot_err : float
        Formal 1-sigma on :attr:`pddot` (NaN below degree 3).
    period_correction : float
        ``period_ref * c1``: the linear-term period offset (days).
    t0_correction : float
        ``c0``: the epoch offset (days).
    chi2, chi2_reduced : float
        Weighted chi-squared and per-dof value.
    aic, bic : float
        Information criteria ``chi2 + 2k`` and ``chi2 + k*ln(n)``. Model
        comparison convention: ``linear.bic - quadratic.bic`` positive favors
        the quadratic (dP/dt) model; > 10 is conventionally strong.
    n_used : int
        Number of O-C points.
    f_test_p : float
        p-value of the quadratic term against the linear model (NaN for the
        linear fit itself, or when dof does not allow the test).
    worst_phase_residual : float
        Largest ``|O-C| / period`` seen during cycle assignment (NaN when the
        fit was built from delays rather than event times). Values approaching
        0.5 mean the nearest-integer cycle count is at risk (Sterken 2005's
        classic O-C ambiguity).
    """

    period_ref: float
    t_ref: float
    coefficients: FloatArray
    covariance: FloatArray
    pdot: float
    pdot_err: float
    period_correction: float
    t0_correction: float
    chi2: float
    chi2_reduced: float
    aic: float
    bic: float
    n_used: int
    f_test_p: float
    worst_phase_residual: float = float("nan")
    pddot: float = float("nan")
    pddot_err: float = float("nan")

    @property
    def degree(self) -> int:
        """Polynomial degree (1 = linear, 2 = quadratic)."""
        return int(self.coefficients.size - 1)

    @property
    def pdot_err_scaled(self) -> float:
        """:attr:`pdot_err` inflated by ``sqrt(chi2_reduced)`` when > 1.

        The honest error bar whenever the O-C scatter exceeds the formal timing
        errors — which is the norm rather than the exception on real data, where
        the light-curve shape evolves, spots migrate, or the drift is not
        actually the model being fitted. Equal to :attr:`pdot_err` for a fit
        that already matches its errors.
        """
        return self.pdot_err * float(np.sqrt(max(1.0, self.chi2_reduced)))

    @property
    def implied_model(self) -> str:
        """Which period law the fitted ``pddot`` favours (degree >= 3 only).

        ``"constant-pdot"`` when ``pddot`` is consistent with 0,
        ``"exponential"`` when it is consistent with ``pdot**2/period_ref``,
        ``"neither"`` when it is more than 1 sigma from both, and ``"unknown"``
        below degree 3 or without a usable error.
        """
        if not np.isfinite(self.pddot) or not np.isfinite(self.pddot_err):
            return "unknown"
        if self.pddot_err <= 0.0:
            return "unknown"
        expected = self.pdot * self.pdot / self.period_ref
        d_const = abs(self.pddot) / self.pddot_err
        d_exp = abs(self.pddot - expected) / self.pddot_err
        if d_const <= 1.0 and d_const <= d_exp:
            return "constant-pdot"
        if d_exp <= 1.0:
            return "exponential"
        return "neither"

    @property
    def pdot_s_per_yr(self) -> float:
        """dP/dt in seconds per year."""
        return self.pdot * SECONDS_PER_YEAR

    @property
    def pdot_ms_per_yr(self) -> float:
        """dP/dt in milliseconds per year (the transit-timing unit)."""
        return self.pdot * SECONDS_PER_YEAR * 1e3

    @property
    def dlnp_dt_per_yr(self) -> float:
        """Fractional rate d(ln P)/dt per year (the Mira / delta Sct unit)."""
        return (self.pdot / self.period_ref) * 365.25

    def predict(self, time: FloatArray) -> FloatArray:
        """The fitted O-C (days) at ``time`` (same time system as the input)."""
        dt = np.asarray(time, dtype=np.float64) - self.t_ref
        out = np.zeros_like(dt)
        for k, c in enumerate(self.coefficients):
            out += c * dt**k
        return out

    def to_dict(self) -> dict[str, Any]:
        """Flatten the scalars to a plain dict."""
        return {
            "degree": self.degree,
            "pdot": self.pdot,
            "pdot_err": self.pdot_err,
            "pdot_err_scaled": self.pdot_err_scaled,
            "pdot_s_per_yr": self.pdot_s_per_yr,
            "pdot_ms_per_yr": self.pdot_ms_per_yr,
            "dlnp_dt_per_yr": self.dlnp_dt_per_yr,
            "pddot": self.pddot,
            "pddot_err": self.pddot_err,
            "period_correction": self.period_correction,
            "t0_correction": self.t0_correction,
            "chi2_reduced": self.chi2_reduced,
            "aic": self.aic,
            "bic": self.bic,
            "n_used": self.n_used,
            "f_test_p": self.f_test_p,
            "worst_phase_residual": self.worst_phase_residual,
            "implied_model": self.implied_model,
            "period_ref": self.period_ref,
            "t_ref": self.t_ref,
        }


def _poly_fit(
    dt: FloatArray, oc: FloatArray, sigma: FloatArray, degree: int
) -> tuple[FloatArray, FloatArray, float]:
    """Weighted polynomial LSQ; returns (coefficients, covariance, chi2)."""
    x = np.vander(dt, degree + 1, increasing=True)
    w = 1.0 / (sigma * sigma)
    xtw = x.T * w[None, :]
    cov = np.linalg.inv(xtw @ x)
    coef = cov @ (xtw @ oc)
    resid = oc - x @ coef
    chi2 = float(np.dot(w, resid * resid))
    return np.asarray(coef, dtype=np.float64), np.asarray(cov, dtype=np.float64), chi2


def _build_ephemeris(
    dt: FloatArray,
    oc: FloatArray,
    sigma: FloatArray,
    *,
    period_ref: float,
    t_ref: float,
    degree: int,
    f_test_p: float = float("nan"),
) -> EphemerisFit:
    """One weighted polynomial ephemeris fit of any degree >= 1.

    ``pdot`` reads the quadratic coefficient (degree >= 2) and ``pddot`` the
    cubic (degree >= 3): expanding ``O-C = dt - tau(dt)`` for a period with
    ``P(dt) = P0 + Pdot*dt + Pddot*dt**2/2`` gives ``c2 = Pdot/(2*P0)`` and
    ``c3 = Pddot/(6*P0) - Pdot**2/(3*P0**2)``, which invert to the expressions
    below. Higher terms are fitted but not interpreted.
    """
    coef, cov, chi2 = _poly_fit(dt, oc, sigma, degree)
    k = degree + 1
    n = int(dt.size)
    dof = max(1, n - k)
    c2 = float(coef[2]) if degree >= 2 else 0.0
    c2_err = float(np.sqrt(cov[2, 2])) if degree >= 2 else 0.0
    pddot = float("nan")
    pddot_err = float("nan")
    if degree >= 3:
        c3 = float(coef[3])
        # Pddot/P = 6*c3 + 8*c2**2; propagate through the (c2, c3) block.
        pddot = period_ref * (6.0 * c3 + 8.0 * c2 * c2)
        g2, g3 = 16.0 * c2, 6.0
        var = (
            g2 * g2 * float(cov[2, 2])
            + g3 * g3 * float(cov[3, 3])
            + 2.0 * g2 * g3 * float(cov[2, 3])
        )
        pddot_err = period_ref * float(np.sqrt(max(0.0, var)))
    return EphemerisFit(
        period_ref=period_ref,
        t_ref=t_ref,
        coefficients=coef,
        covariance=cov,
        pdot=2.0 * period_ref * c2,
        pdot_err=2.0 * period_ref * c2_err,
        period_correction=period_ref * float(coef[1]),
        t0_correction=float(coef[0]),
        chi2=chi2,
        chi2_reduced=chi2 / dof,
        aic=chi2 + 2.0 * k,
        bic=chi2 + k * float(np.log(n)),
        n_used=n,
        f_test_p=f_test_p,
        pddot=pddot,
        pddot_err=pddot_err,
    )


def _ephemeris_from_delays(
    t: FloatArray, oc: FloatArray, sigma: FloatArray, period_ref: float
) -> tuple[EphemerisFit, EphemerisFit]:
    """Linear and quadratic ephemeris fits to O-C delays, with the F-test."""
    if t.size < 3:
        raise InsufficientDataError(
            f"ephemeris fit: need at least 3 O-C points, got {t.size}"
        )
    w = 1.0 / (sigma * sigma)
    t_ref = float(np.dot(w, t) / w.sum())
    dt = t - t_ref

    def build(degree: int, f_test_p: float) -> EphemerisFit:
        return _build_ephemeris(
            dt,
            oc,
            sigma,
            period_ref=period_ref,
            t_ref=t_ref,
            degree=degree,
            f_test_p=f_test_p,
        )

    linear = build(1, float("nan"))
    dof_quad = t.size - 3
    if dof_quad < 1:
        raise InsufficientDataError(
            "ephemeris fit: need at least 4 O-C points for the quadratic term"
        )
    quad = build(2, float("nan"))
    from scipy.stats import f as f_dist

    delta = max(0.0, linear.chi2 - quad.chi2)
    f_stat = delta / (quad.chi2 / dof_quad) if quad.chi2 > 0.0 else np.inf
    p = float(f_dist.sf(f_stat, 1, dof_quad))
    quad = build(2, p)
    return linear, quad


def fit_ephemeris(
    times: Any,
    *,
    period: float,
    t0: float | None = None,
    errors: Any | None = None,
) -> EphemerisFit:
    """Quadratic-ephemeris fit to a list of event times (minima/maxima).

    Assigns each event the nearest integer cycle of the trial ephemeris
    ``t0 + period*E``, forms ``O-C = t - t0 - period*E``, and fits the
    quadratic model. Feed it literature timing lists to reproduce classical
    eclipse-timing dP/dt measurements.

    Parameters
    ----------
    times : array-like
        Event times (days; any consistent epoch, e.g. HJD/BJD).
    period : float
        Trial period (days).
    t0 : float, optional
        Trial epoch of zero phase; defaults to the first event.
    errors : array-like, optional
        1-sigma timing errors (days). ``None`` uses equal weights and an
        RMS-derived scale.

    Returns
    -------
    EphemerisFit
        The quadratic fit (its ``f_test_p`` compares against the linear model).

    Notes
    -----
    Cycle assignment by nearest integer silently fails when the true drift
    accumulates more than half a period between *consecutive* events — the
    classical O-C ambiguity. Densely sampled timing lists are safe; sparse
    ones should be inspected (a wrong branch shows up as a sawtooth O-C).
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    t = np.sort(t[np.isfinite(t)])
    if t.size < 4:
        raise InsufficientDataError(
            f"fit_ephemeris: need at least 4 finite event times, got {t.size}"
        )
    if period <= 0.0:
        raise ValueError(f"period must be positive, got {period}")
    epoch = float(t[0]) if t0 is None else float(t0)
    cycles = np.round((t - epoch) / period)
    oc = t - epoch - period * cycles
    if errors is None:
        scale = max(float(np.std(oc)), 1e-9 * period)
        sigma = np.full(t.size, scale, dtype=np.float64)
    else:
        sigma = np.asarray(errors, dtype=np.float64).ravel()
        if sigma.size != t.size:
            raise ValueError("errors and times length mismatch")
        if np.any(~np.isfinite(sigma) | (sigma <= 0.0)):
            raise ValueError("errors must be finite and positive")
    _, quad = _ephemeris_from_delays(t, oc, sigma, period)
    return replace(
        quad, worst_phase_residual=float(np.max(np.abs(oc)) / period)
    )


# --- phase_track -------------------------------------------------------------


@dataclass(frozen=True)
class PhaseTrackResult:
    """An automated O-C analysis of one light curve.

    Attributes
    ----------
    period : float
        The reference period the delays are measured against (days).
    pdot_seed : float
        The starting guess for the bulk drift (the ``pdot=`` argument). Only
        an initialization: the delays and fits are always reported against
        the *constant-period* ephemeris, whatever the seed.
    model : str
        Period law used to unwind the seed drift before templating.
    t_ref : float
        Reference epoch of the phase model (weighted mean epoch).
    template : PhaseTemplate
        The fixed multi-harmonic template.
    t_mid, delay, delay_err : numpy.ndarray
        Per-segment weighted mid-epochs, measured template delays (days, =
        O-C), and their 1-sigma errors.
    n_points : numpy.ndarray
        Points per segment.
    linear_fit, quadratic_fit : EphemerisFit
        The competing ephemeris models over the segment delays.
    wrap_risk : bool
        True when consecutive segment delays jump by more than 0.3 period —
        the cycle-count assignment may have slipped; re-run with fewer/longer
        segments or seed ``pdot=`` from :func:`cuperiod.pdot_search`.
    meta : Mapping
        Provenance plus light-curve metadata.
    """

    period: float
    pdot_seed: float
    model: str
    t_ref: float
    template: PhaseTemplate
    t_mid: FloatArray
    delay: FloatArray
    delay_err: FloatArray
    n_points: FloatArray
    linear_fit: EphemerisFit
    quadratic_fit: EphemerisFit
    wrap_risk: bool
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def pdot(self) -> float:
        """Measured dP/dt (day/day), against the constant-period ephemeris."""
        return self.quadratic_fit.pdot

    @property
    def pdot_err(self) -> float:
        """Formal 1-sigma on :attr:`pdot` (from the quadratic fit)."""
        return self.quadratic_fit.pdot_err

    @property
    def pdot_err_scaled(self) -> float:
        """:attr:`pdot_err` inflated by ``sqrt(chi2_reduced)`` when > 1.

        Prefer this on real data: an evolving light-curve shape leaves the O-C
        scattered well beyond the formal per-segment timing errors.
        """
        return self.quadratic_fit.pdot_err_scaled

    @property
    def pdot_s_per_yr(self) -> float:
        """Total dP/dt in seconds per year."""
        return self.pdot * SECONDS_PER_YEAR

    @property
    def f_test_p(self) -> float:
        """p-value of the quadratic (dP/dt) term against the linear model."""
        return self.quadratic_fit.f_test_p

    @property
    def n_segments(self) -> int:
        """Number of usable segments."""
        return int(self.t_mid.size)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalars plus the O-C table."""
        return {
            "period": self.period,
            "pdot": self.pdot,
            "pdot_err": self.pdot_err,
            "pdot_err_scaled": self.pdot_err_scaled,
            "pdot_s_per_yr": self.pdot_s_per_yr,
            "pdot_seed": self.pdot_seed,
            "model": self.model,
            "f_test_p": self.f_test_p,
            "wrap_risk": self.wrap_risk,
            "t_ref": self.t_ref,
            "n_segments": self.n_segments,
            "n_harmonics": self.template.n_harmonics,
            "t_mid": self.t_mid.tolist(),
            "delay": self.delay.tolist(),
            "delay_err": self.delay_err.tolist(),
            "linear_fit": self.linear_fit.to_dict(),
            "quadratic_fit": self.quadratic_fit.to_dict(),
        }


def _segment_bounds(
    time: FloatArray, period: float, settings: PdotSettings
) -> list[slice]:
    """Contiguous, points-balanced segments honoring the floor constraints."""
    n = time.size
    baseline = float(time[-1] - time[0])
    by_points = n // settings.min_segment_points
    by_cycles = int(baseline / (settings.min_segment_cycles * period))
    n_seg = min(by_points, max(1, by_cycles), 20)
    if settings.n_segments is not None:
        n_seg = min(settings.n_segments, by_points)
    if n_seg < 3:
        raise InsufficientDataError(
            "phase_track: cannot form 3 usable segments "
            f"({n} points, {baseline:.1f} d baseline, period {period} d); "
            "relax min_segment_points / min_segment_cycles"
        )
    edges = np.linspace(0, n, n_seg + 1).astype(np.int64)
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def phase_track(
    data: Any,
    *,
    period: float,
    pdot: float = 0.0,
    model: str | PeriodModel | None = None,
    settings: PdotSettings | None = None,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
) -> PhaseTrackResult:
    """Measure dP/dt from the drift of the light curve's phase over time.

    Fits one fixed multi-harmonic template at ``period`` (order BIC-selected
    unless pinned in ``settings``), measures the template's best time shift in
    each of a set of contiguous time segments, and fits the shifts against
    linear and quadratic ephemerides. The quadratic coefficient is the period
    derivative; the F-test between the two models is its significance.

    Parameters
    ----------
    data : various
        The light curve (single-band), in any form :func:`cuperiod.to_input`
        accepts.
    period : float
        Reference period in days (from a periodogram or the literature).
    pdot : float, default 0.0
        Starting guess for the bulk dP/dt (day/day). Purely an initialization
        for the template — pass :func:`cuperiod.pdot_search`'s ``best.pdot``
        when the accumulated drift is large enough (roughly a cycle) that a
        zero-start template would wash out. The delays, fits, and F-test are
        always reported against the constant-period ephemeris.
    model : str, optional
        Period law used to remove the bulk drift before templating
        (:mod:`cuperiod.pdot.models`); match it to the ``model=`` of the
        :func:`cuperiod.pdot_search` that produced ``pdot``. It affects only how
        the seed drift is unwound — the reported O-C is always measured against
        a constant period, so the returned delays stay model-independent.
    settings : PdotSettings, optional
        Tracking knobs (segmentation, harmonics, per-segment offsets).
    columns, domain
        Passed to :func:`cuperiod.to_input` for table inputs.

    Returns
    -------
    PhaseTrackResult
        Per-segment O-C delays and the fitted ephemerides.

    Examples
    --------
    >>> res = phase_track((t, mag, err), period=0.5213)     # doctest: +SKIP
    >>> res.pdot_s_per_yr, res.pdot_err, res.f_test_p       # doctest: +SKIP
    """
    from cuperiod.api import to_input
    from cuperiod.pdot.scan import weighted_epoch

    psettings = settings or PdotSettings()
    pmodel = get_model(model)
    if period <= 0.0:
        raise ValueError(f"period must be positive, got {period}")
    lc = to_input(data, columns=columns, domain=domain)
    if not isinstance(lc, LightCurve):
        raise ValueError(
            "phase_track works on a single band; pass one band of the "
            "MultiBandLightCurve (multi-band tracking is not supported yet)"
        )
    finite = lc.finite()
    if finite.n < psettings.min_detections:
        raise InsufficientDataError(
            f"phase_track: {finite.n} finite points < min_detections "
            f"{psettings.min_detections}"
        )
    order = np.argsort(finite.time, kind="stable")
    time = finite.time[order]
    y = finite.value[order]
    weighted = finite.error is not None
    if finite.error is not None:
        err = finite.error[order]
        w = 1.0 / (err * err)
    else:
        w = np.full(time.size, 1.0 / time.size, dtype=np.float64)

    t_ref = weighted_epoch(finite)
    dt = time - t_ref

    # Track once at the given bulk rate, then (template_iterations times)
    # fold the measured residual rate back into the bulk and re-derive the
    # template on drift-corrected times. A drift left in the fold smears the
    # template — shrinking the measured rate and inflating its chi-squared —
    # so the loop makes the template self-consistent with the measurement.
    # Passing pdot= from pdot_search starts it at (or near) the fixed point.
    bulk = pdot
    passes = 1
    result = _track_pass(dt, time, y, w, weighted, period, bulk, psettings, pmodel)
    for _ in range(psettings.template_iterations):
        correction = result.quadratic.pdot
        if abs(correction) <= 0.1 * result.quadratic.pdot_err:
            break
        bulk += correction
        result = _track_pass(
            dt, time, y, w, weighted, period, bulk, psettings, pmodel
        )
        passes += 1

    # Report against the *constant-period* ephemeris, whatever bulk drift
    # ended up in the warp: the removed drift is deterministic, so add its
    # delay back and refit. The O-C then shows the classical parabola, and
    # the F-test always answers "is the drift real?" — while the delays
    # themselves were measured against the sharp, drift-corrected template.
    delay = result.delay
    linear, quad = result.linear, result.quadratic
    if bulk != 0.0:
        dt_mid = result.t_mid - t_ref
        # The delay the removed drift itself produces against a constant
        # period: dt - tau(dt) (exactly (bulk/2P)*dt^2 + h.o.t.).
        delay = result.delay + pmodel.delay(dt_mid, bulk / period)
        linear, quad = _ephemeris_from_delays(
            result.t_mid, delay, result.delay_err, period
        )

    meta = {
        "phase_track": {
            "n_harmonics": result.template.n_harmonics,
            "segment_offsets": psettings.segment_offsets,
            "pdot_bulk_removed": bulk,
            "template_passes": passes,
            "model": pmodel.name,
        },
        **dict(finite.meta),
    }
    return PhaseTrackResult(
        period=period,
        pdot_seed=pdot,
        model=pmodel.name,
        t_ref=t_ref,
        template=result.template,
        t_mid=result.t_mid,
        delay=delay,
        delay_err=result.delay_err,
        n_points=result.n_points,
        linear_fit=linear,
        quadratic_fit=quad,
        wrap_risk=result.wrap_risk,
        meta=meta,
    )


@dataclass(frozen=True)
class _TrackPass:
    """One template-and-measure pass at a fixed bulk rate."""

    template: PhaseTemplate
    t_mid: FloatArray
    delay: FloatArray
    delay_err: FloatArray
    n_points: FloatArray
    linear: EphemerisFit
    quadratic: EphemerisFit
    wrap_risk: bool


def _track_pass(
    dt: FloatArray,
    time: FloatArray,
    y: FloatArray,
    w: FloatArray,
    weighted: bool,
    period: float,
    bulk_pdot: float,
    psettings: PdotSettings,
    pmodel: PeriodModel,
) -> _TrackPass:
    # Remove the bulk drift: on these warped times the reference model is a
    # strictly periodic signal at `period` (see the scan module docstring).
    dtw = pmodel.warp(dt, bulk_pdot / period)

    template = fit_template(
        dtw,
        y,
        w,
        1.0 / period,
        n_harmonics=psettings.harmonics,
        max_harmonics=psettings.max_harmonics,
        weighted=weighted,
    )
    # Without per-point errors the Fisher sigma needs a noise scale; with poor
    # per-point errors it should not *shrink* below the fit quality. Inflate by
    # sqrt(chi2_reduced) of the global template fit when > 1.
    error_scale = float(np.sqrt(max(1.0, template.chi2_reduced)))
    if not weighted:
        # w sums to 1, so the Fisher sum is the *mean* squared slope; the true
        # sigma is rms_noise/sqrt(sum slope^2) = (rms/sqrt(N))/sqrt(mean slope^2).
        rms = float(
            np.sqrt(np.dot(w, (y - template(dtw)) ** 2) / max(1e-300, w.sum()))
        )
        error_scale = rms / np.sqrt(time.size)

    shifts: list[SegmentShift] = []
    for sl in _segment_bounds(time, period, psettings):
        seg_t, seg_y, seg_w = dtw[sl], y[sl], w[sl]
        if seg_t.size < max(5, psettings.min_segment_points // 2):
            continue
        delay, sigma, chi2r = _fit_segment_shift(
            seg_t,
            seg_y,
            seg_w,
            template,
            free_offset=psettings.segment_offsets,
            error_scale=error_scale,
        )
        if not np.isfinite(sigma):
            continue
        t_mid = float(np.dot(seg_w, time[sl]) / seg_w.sum())
        shifts.append(
            SegmentShift(
                t_mid=t_mid,
                delay=delay,
                delay_err=sigma,
                n_points=int(seg_t.size),
                duration=float(time[sl][-1] - time[sl][0]),
                chi2_reduced=chi2r,
            )
        )
    if len(shifts) < 3:
        raise InsufficientDataError(
            f"phase_track: only {len(shifts)} usable segments; "
            "relax the segmentation settings"
        )

    # Cycle continuity: unwrap each delay to the branch nearest its predecessor.
    delays = np.array([s.delay for s in shifts], dtype=np.float64)
    for k in range(1, delays.size):
        delays[k] -= period * np.round((delays[k] - delays[k - 1]) / period)
    steps = np.abs(np.diff(delays))
    wrap_risk = bool(steps.size and float(steps.max()) > 0.3 * period)

    t_mids = np.array([s.t_mid for s in shifts], dtype=np.float64)
    delay_errs = np.array([s.delay_err for s in shifts], dtype=np.float64)
    n_points = np.array([s.n_points for s in shifts], dtype=np.float64)
    linear, quad = _ephemeris_from_delays(t_mids, delays, delay_errs, period)
    return _TrackPass(
        template=template,
        t_mid=t_mids,
        delay=delays,
        delay_err=delay_errs,
        n_points=n_points,
        linear=linear,
        quadratic=quad,
        wrap_risk=wrap_risk,
    )


__all__ = [
    "EphemerisFit",
    "PhaseTrackResult",
    "SegmentShift",
    "PhaseTemplate",
    "fit_ephemeris",
    "fit_template",
    "phase_track",
]
