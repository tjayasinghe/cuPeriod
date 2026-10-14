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
from typing import TYPE_CHECKING, Any

import numpy as np

from cuperiod.core._coerce import as_float_array
from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import PdotSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.lightcurve import LightCurve, raise_if_unusable
from cuperiod.pdot.models import PeriodModel, get_model
from cuperiod.pdot.result import SECONDS_PER_YEAR

if TYPE_CHECKING:  # ltte imports this module; the annotation needs no runtime import.
    from cuperiod.pdot.ltte import LTTESearchResult

#: Shift-scan sampling: the coarse grid must resolve the sharpest template
#: feature (scale P / (2 * harmonics)), with margin.
_SCAN_PER_HARMONIC: int = 32
_SCAN_MIN: int = 128
#: Newton polish iterations after the parabolic seed.
_NEWTON_STEPS: int = 6
#: A template whose harmonics outside the multiples of ``m`` carry less than this
#: fraction of its power is ``m``-fold symmetric: a shift by ``P/m`` is nearly as
#: good as the true one, so segment delays are unwrapped by ``P/m``.
_SYMMETRY_POWER_FRACTION: float = 0.05
#: Symmetry orders tested, highest first (2 = the equal-minima contact binary).
_SYMMETRY_ORDERS: tuple[int, ...] = (4, 3, 2)
#: Fewest segments (O-C points) a track can use: the quadratic ephemeris has
#: three parameters, and its F-test against the linear one needs a spare degree
#: of freedom.
_MIN_SEGMENTS: int = 4


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

    def _shape_and_deriv(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """:meth:`_shape` and :meth:`_shape_deriv` from one cos/sin evaluation.

        The arithmetic is exactly theirs (same angle, same products), so the
        values are bit-identical; only the trigonometry is shared.
        """
        m = np.arange(1, self.n_harmonics + 1, dtype=np.float64)
        omega = 2.0 * np.pi * self.frequency * m
        ang = (2.0 * np.pi * self.frequency) * x[:, None] * m[None, :]
        cos, sin = np.cos(ang), np.sin(ang)
        shape = cos @ self.cos_coeffs + sin @ self.sin_coeffs
        deriv = sin @ (-omega * self.cos_coeffs) + cos @ (omega * self.sin_coeffs)
        return shape, deriv


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


#: Condition number of the weighted top-order design above which the nested-QR
#: order ladder is not trusted (sampling commensurate with the period makes high
#: harmonics alias onto low ones) and every order is fitted on its own instead.
_LADDER_MAX_COND: float = 1e8


def _ladder_chi2(
    dt: FloatArray, y: FloatArray, w: FloatArray, f0: float, top: int
) -> FloatArray | None:
    """Weighted chi-squared of every template order ``1..top`` from one QR.

    With the design columns ordered ``[1, cos1, sin1, cos2, sin2, ...]`` the
    order-``k`` model is exactly the leading ``2k+1`` columns, so a single QR of
    the weighted design augmented with the weighted data column gives every
    nested fit at once: the residual of the leading-``m`` fit is the norm of the
    data column's ``R`` entries below row ``m``. One ``O(N*top**2)``
    factorization replaces ``top`` separate least-squares solves on freshly
    built designs. Returns ``None`` when the design is too ill-conditioned for
    the nested residuals to be trusted; the caller then fits each order itself.
    """
    n = int(dt.size)
    m = np.arange(1, top + 1, dtype=np.float64)
    ang = (2.0 * np.pi * f0) * dt[:, None] * m[None, :]
    sw = np.sqrt(w)
    aug = np.empty((n, 2 * top + 2), dtype=np.float64)
    aug[:, 0] = sw
    aug[:, 1 : 2 * top + 1 : 2] = np.cos(ang) * sw[:, None]
    aug[:, 2 : 2 * top + 1 : 2] = np.sin(ang) * sw[:, None]
    aug[:, -1] = y * sw
    # mode="r" returns R alone (the stubs type every mode as the Q, R pair).
    r: FloatArray = np.asarray(np.linalg.qr(aug, mode="r"), dtype=np.float64)
    if not np.all(np.isfinite(r)):
        return None
    if float(np.linalg.cond(r[:, :-1])) > _LADDER_MAX_COND:
        return None
    tail = r[:, -1] ** 2
    # chi2 of the leading-m fit = sum of tail[m:]; reverse cumulative sum.
    below = np.cumsum(tail[::-1])[::-1]
    return np.asarray(below[1 + 2 * np.arange(1, top + 1)], dtype=np.float64)


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

    The order is chosen with the *scale-free* BIC ``N*ln(chi2/N) + k*ln(N)``,
    which depends on the weights only up to an overall factor, whether or not
    per-point errors were supplied. The weighted form ``chi2 + k*ln(N)`` is
    right only when the quoted errors are right: survey photometry routinely
    understates them several-fold, every extra harmonic then looks like a
    bargain, the selection runs to ``max_harmonics``, and the noise-fitted
    harmonics inflate the Fisher information of every segment delay (the same
    failure :func:`cuperiod.period_profile` guards against). The returned
    template is fitted with the original weights, so its ``chi2_reduced`` still
    measures the fit against the quoted errors.

    Parameters
    ----------
    dt, y, w : numpy.ndarray
        Times relative to the reference epoch (days), values, and weights.
    f0 : float
        Template frequency (cycles/day).
    n_harmonics : int, optional
        Pin the order instead of selecting it.
    max_harmonics : int, default 10
        Largest order the selection may pick.
    weighted : bool, default True
        Kept for backward compatibility. The selection no longer depends on it:
        for equal weights the scale-free form is the Gaussian profile BIC
        ``N*ln(RSS/N) + k*ln(N)`` up to a constant, as before.
    """
    del weighted  # the scale-free criterion needs no branch (see above)
    n = dt.size
    if n_harmonics is not None:
        return _fit_template_order(dt, y, w, f0, n_harmonics)[0]
    top = max(1, min(max_harmonics, (n - 2) // 2))
    ladder = _ladder_chi2(dt, y, w, f0, top)
    if ladder is None:
        ladder = np.array(
            [_fit_template_order(dt, y, w, f0, k)[1] for k in range(1, top + 1)],
            dtype=np.float64,
        )
    # A noise-free curve leaves chi2 at rounding level, where its logarithm is
    # noise: floor it at a tiny fraction of the data's own weighted power so
    # such orders tie and the cheapest wins.
    floor = max(np.finfo(np.float64).tiny, 1e-24 * float(np.dot(w, y * y)))
    log_n = float(np.log(n))
    best_order, best_bic = 1, np.inf
    for order in range(1, top + 1):
        chi2 = max(float(ladder[order - 1]), floor)
        bic = n * float(np.log(chi2 / n)) + (1 + 2 * order) * log_n
        if bic < best_bic - 1e-12:
            best_order, best_bic = order, bic
    return _fit_template_order(dt, y, w, f0, best_order)[0]


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


def _shift_scan_chi2(
    dt: FloatArray,
    y: FloatArray,
    w: FloatArray,
    template: PhaseTemplate,
    shifts: FloatArray,
    *,
    free_offset: bool,
) -> FloatArray:
    """Weighted chi-squared of the template shifted by each of ``shifts``.

    A time shift ``s`` only rotates each harmonic's ``(a_m, b_m)`` pair,

        a_m*cos(w_m*(x - s)) + b_m*sin(w_m*(x - s))
            = (a_m*cos(w_m*s) - b_m*sin(w_m*s)) * cos(w_m*x)
            + (a_m*sin(w_m*s) + b_m*cos(w_m*s)) * sin(w_m*x),

    so with the fixed basis ``B = [cos(w_m*x), sin(w_m*x)]`` and rotated
    coefficients ``q(s)`` the whole scan is
    ``chi2(s) = r'Wr - 2*q(s)'B'Wr + q(s)'(B'WB)q(s)``, ``r = y - offset``. The
    basis is evaluated once (``N x 2H``) instead of once per trial shift, and
    no ``(n_shifts, N)`` temporaries are built. A free per-segment offset is
    the weighted mean of the residual, which is linear in ``q``: centring ``r``
    and every basis column on their weighted means marginalizes it exactly.

    The value is a difference of large terms, good to ~1e-14 of the grid's
    range; that is ample for picking the best cell and seeding the parabola,
    and the Newton polish that follows evaluates the residuals directly.
    """
    m = np.arange(1, template.n_harmonics + 1, dtype=np.float64)
    two_pi_f = 2.0 * np.pi * template.frequency
    ang = two_pi_f * dt[:, None] * m[None, :]
    basis = np.hstack([np.cos(ang), np.sin(ang)])
    resid = y - template.offset
    if free_offset:
        sum_w = float(w.sum())
        resid = resid - float(np.dot(w, resid)) / sum_w
        basis = basis - (w @ basis)[None, :] / sum_w
    weighted_basis = basis * w[:, None]
    gram = weighted_basis.T @ basis
    proj = weighted_basis.T @ resid
    rr = float(np.dot(w, resid * resid))
    s_ang = two_pi_f * shifts[:, None] * m[None, :]
    cs, ss = np.cos(s_ang), np.sin(s_ang)
    a, b = template.cos_coeffs, template.sin_coeffs
    q = np.hstack([a * cs - b * ss, a * ss + b * cs])  # (n_shifts, 2H)
    quad = np.einsum("ki,ki->k", q @ gram, q)
    return np.asarray(rr - 2.0 * (q @ proj) + quad, dtype=np.float64)


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

    chi2_grid = _shift_scan_chi2(dt, y, w, template, shifts, free_offset=free_offset)
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
        shape, deriv = template._shape_and_deriv(dt - delay)
        model = template.offset + shape
        slope = -deriv  # d(model)/d(delay)
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

    shape, deriv = template._shape_and_deriv(dt - delay)
    slope = -deriv
    resid = y - template.offset - shape
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
        p-value of the quadratic term against the linear model. NaN for the
        linear fit itself and whenever no test is possible: too few degrees of
        freedom, a non-finite chi-squared, or O-C values both models fit
        exactly (a noise-free linear ephemeris has nothing for the quadratic
        term to explain).
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
    def pddot_err_scaled(self) -> float:
        """:attr:`pddot_err` inflated by ``sqrt(chi2_reduced)`` when > 1.

        The same honest-error inflation as :attr:`pdot_err_scaled`, for the rate
        of change of the rate; NaN below degree 3, where no ``pddot`` is fitted.
        :attr:`implied_model` still reads the formal :attr:`pddot_err`.
        """
        return self.pddot_err * float(np.sqrt(max(1.0, self.chi2_reduced)))

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
    def pdot_d_per_yr(self) -> float:
        """dP/dt in days per year (the eclipsing-binary survey unit)."""
        return self.pdot * 365.25

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
            "pdot_d_per_yr": self.pdot_d_per_yr,
            "dlnp_dt_per_yr": self.dlnp_dt_per_yr,
            "pddot": self.pddot,
            "pddot_err": self.pddot_err,
            "pddot_err_scaled": self.pddot_err_scaled,
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
    p = _f_test(linear.chi2 - quad.chi2, 1, quad.chi2, dof_quad)
    return linear, replace(quad, f_test_p=p)


def _f_test(delta: float, k_extra: int, chi2_full: float, dof_full: int) -> float:
    """p-value of ``k_extra`` nested extra parameters (the classical F-test).

    ``delta`` is the chi-squared the extra parameters remove and ``chi2_full``
    the fuller model's own, on ``dof_full`` degrees of freedom. NaN whenever no
    test is possible: no degrees of freedom, a non-finite chi-squared (a failed
    fit), or a fit that is exact under *both* models (nothing to explain, so
    nothing to be significant about). An exact fuller model that the smaller one
    misses is the one infinitely significant case, ``p = 0``.
    """
    from scipy.stats import f as f_dist

    if dof_full < 1 or k_extra < 1:
        return float("nan")
    if not (np.isfinite(delta) and np.isfinite(chi2_full)):
        return float("nan")
    if chi2_full <= 0.0:
        return 0.0 if delta > 0.0 else float("nan")
    stat = (max(0.0, delta) / k_extra) / (chi2_full / dof_full)
    return float(f_dist.sf(stat, k_extra, dof_full))


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
        Event times (days; any consistent epoch, e.g. HJD/BJD), in any order.
        Non-finite (or masked) times are dropped together with their errors.
    period : float
        Trial period (days); positive and finite.
    t0 : float, optional
        Trial epoch of zero phase; defaults to the first event.
    errors : array-like, optional
        1-sigma timing errors (days), one per entry of ``times`` and in the
        same order. ``None`` gives every event the same error, estimated a
        posteriori from the scatter about the fitted quadratic (so the
        quadratic's ``chi2_reduced`` is 1 by construction and ``pdot_err`` is
        the ordinary unweighted least-squares error).

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
    if not (np.isfinite(period) and period > 0.0):
        raise ValueError(f"period must be positive and finite, got {period}")
    if t0 is not None and not np.isfinite(t0):
        raise ValueError(f"t0 must be finite, got {t0}")
    t_all = as_float_array(times, name="times", time=True).ravel()
    sigma_all: FloatArray | None = None
    if errors is not None:
        sigma_all = as_float_array(errors, name="errors").ravel()
        if sigma_all.size != t_all.size:
            raise ValueError("errors and times length mismatch")
    # One mask and one ordering for both arrays: each error stays with its event.
    keep = np.flatnonzero(np.isfinite(t_all))
    keep = keep[np.argsort(t_all[keep], kind="stable")]
    t = t_all[keep]
    if t.size < 4:
        raise InsufficientDataError(
            f"fit_ephemeris: need at least 4 finite event times, got {t.size}"
        )
    epoch = float(t[0]) if t0 is None else float(t0)
    cycles = np.round((t - epoch) / period)
    oc = t - epoch - period * cycles
    if sigma_all is None:
        sigma = np.full(t.size, _quadratic_scatter(t, oc, period), dtype=np.float64)
    else:
        sigma = sigma_all[keep]
        if np.any(~np.isfinite(sigma) | (sigma <= 0.0)):
            raise ValueError("errors must be finite and positive")
    _, quad = _ephemeris_from_delays(t, oc, sigma, period)
    return replace(
        quad, worst_phase_residual=float(np.max(np.abs(oc)) / period)
    )


def _quadratic_scatter(t: FloatArray, oc: FloatArray, period: float) -> float:
    """A-posteriori timing error: the RMS about an unweighted quadratic O-C fit.

    The raw O-C spread is the wrong scale: it holds the linear term left by an
    imperfect trial period/epoch and the very parabola being measured, so
    ``pdot_err`` would measure the trend (tens of times too large, and changing
    with the trial period) rather than the timing noise. The residual RMS on
    ``n - 3`` degrees of freedom is the standard unweighted least-squares
    noise estimate. Times are centred and scaled to ``[-1, 1]`` first so the
    fit stays well conditioned over long baselines; the residual is unchanged
    by that reparameterization. Floored at ``1e-9 * period`` so exact data still
    give a finite fit.
    """
    x = t - float(np.mean(t))
    half = float(np.max(np.abs(x)))
    if half > 0.0:
        x = x / half
    design = np.vander(x, 3, increasing=True)
    coef, *_ = np.linalg.lstsq(design, oc, rcond=None)
    resid = oc - design @ coef
    dof = max(1, t.size - 3)
    rms = float(np.sqrt(float(np.dot(resid, resid)) / dof))
    return max(rms, 1e-9 * period)


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
        True when consecutive segment delays jump by more than 0.3 of the
        unwrap period (``period / symmetry_order``) — the cycle-count
        assignment may have slipped; re-run with fewer/longer segments or seed
        ``pdot=`` from :func:`cuperiod.pdot_search`.
    meta : Mapping
        Provenance plus light-curve metadata.
    symmetry_order : int
        ``m`` when the template is ``m``-fold symmetric (harmonics outside the
        multiples of ``m`` carry under 5% of its power) and 1 otherwise. A
        contact binary with equal minima is 2-fold symmetric: a shift by half
        a period fits almost as well as the true one, and a segment's best
        shift can land on either branch. Delays are therefore unwrapped to
        the branch nearest their predecessor in steps of ``period / m`` —
        cycle continuity is the only information that separates the branches
        — and the O-C is only defined modulo that step.
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
    symmetry_order: int = 1

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
    def pdot_d_per_yr(self) -> float:
        """Total dP/dt in days per year."""
        return self.pdot * 365.25

    @property
    def f_test_p(self) -> float:
        """p-value of the quadratic (dP/dt) term against the linear model."""
        return self.quadratic_fit.f_test_p

    @property
    def n_segments(self) -> int:
        """Number of usable segments."""
        return int(self.t_mid.size)

    def ltte_search(self, **kwargs: Any) -> LTTESearchResult:
        """Search these delays for a cyclic (light-travel-time) O-C.

        A convenience for :func:`cuperiod.ltte_search` on this result; the
        keyword arguments are passed through. A quadratic alone averages over
        a cycle shorter than the baseline, so when the O-C is not monotonic
        this is where the *secular* rate should be read from.
        """
        from cuperiod.pdot.ltte import ltte_search

        return ltte_search(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalars plus the O-C table."""
        return {
            "period": self.period,
            "pdot": self.pdot,
            "pdot_err": self.pdot_err,
            "pdot_err_scaled": self.pdot_err_scaled,
            "pdot_s_per_yr": self.pdot_s_per_yr,
            "pdot_d_per_yr": self.pdot_d_per_yr,
            "pdot_seed": self.pdot_seed,
            "model": self.model,
            "f_test_p": self.f_test_p,
            "wrap_risk": self.wrap_risk,
            "symmetry_order": self.symmetry_order,
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
    if n_seg < _MIN_SEGMENTS:
        knobs = (
            f"raise n_segments (set to {settings.n_segments}) or lower "
            "min_segment_points"
            if settings.n_segments is not None
            else "relax min_segment_points / min_segment_cycles"
        )
        raise InsufficientDataError(
            f"phase_track: cannot form {_MIN_SEGMENTS} usable segments "
            f"({n} points, {baseline:.1f} d baseline, period {period} d; the "
            f"quadratic ephemeris needs {_MIN_SEGMENTS} O-C points); {knobs}"
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
    domain: Domain | str | None = None,
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
    if not (np.isfinite(period) and period > 0.0):
        raise ValueError(f"period must be positive and finite, got {period}")
    if not np.isfinite(pdot):
        raise ValueError(f"pdot must be finite, got {pdot}")
    lc = to_input(data, columns=columns, domain=domain)
    if not isinstance(lc, LightCurve):
        raise ValueError(
            "phase_track works on a single band; pass one band of the "
            "MultiBandLightCurve (multi-band tracking is not supported yet)"
        )
    raise_if_unusable(lc, "phase_track")
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
            "symmetry_order": result.symmetry_order,
            "unwrap_period": period / result.symmetry_order,
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
        symmetry_order=result.symmetry_order,
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
    symmetry_order: int


def _symmetry_order(template: PhaseTemplate) -> int:
    """The template's fold symmetry ``m`` (1 when none reaches the threshold)."""
    power = np.asarray(template.amplitudes, dtype=np.float64) ** 2
    total = float(power.sum())
    if total <= 0.0:
        return 1
    harmonic = np.arange(1, power.size + 1)
    for m in _SYMMETRY_ORDERS:
        if m > power.size:
            continue
        outside = float(power[harmonic % m != 0].sum())
        if outside < _SYMMETRY_POWER_FRACTION * total:
            return m
    return 1


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
    if len(shifts) < _MIN_SEGMENTS:
        raise InsufficientDataError(
            f"phase_track: only {len(shifts)} usable segments (the quadratic "
            f"ephemeris needs {_MIN_SEGMENTS}); relax the segmentation settings"
        )

    # Cycle continuity: unwrap each delay to the branch nearest its predecessor.
    # A near-symmetric template (equal-minima contact binary: power in the
    # even harmonics only) fits a shift by P/2 almost as well as the true one,
    # and a segment's global best lands on either branch by noise — so the
    # unwrap step is the template's ambiguity period P/m, not P.
    symmetry = _symmetry_order(template)
    unwrap = period / symmetry
    delays = np.array([s.delay for s in shifts], dtype=np.float64)
    for k in range(1, delays.size):
        delays[k] -= unwrap * np.round((delays[k] - delays[k - 1]) / unwrap)
    steps = np.abs(np.diff(delays))
    wrap_risk = bool(steps.size and float(steps.max()) > 0.3 * unwrap)

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
        symmetry_order=symmetry,
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
