"""Parametric period-vs-time laws fitted to a measured ``P(t)`` track.

:mod:`cuperiod.pdot.models` covers the two laws a *search* can warp by, because
they have a single rate parameter. This module covers the rest: laws with a
characteristic epoch or exponent, which cannot be a one-dimensional scan axis
but are exactly what the literature fits to a sequence of periods measured in
consecutive time intervals (:func:`cuperiod.period_profile`).

That two-step recipe — measure ``P`` per interval, then fit a law to the
periods — is how period changes are actually published for systems whose rate
is not constant, and reproducing a published number generally means reproducing
both steps.

Registered laws, all with ``dt = t - t_ref`` except where ``t0`` appears:

``constant``
    ``P = P0``. The null hypothesis.
``linear``
    ``P = P0 + Pdot*dt``. Constant ``dP/dt``; matches
    ``models.constant-pdot``.
``exponential``
    ``P = P0*exp(a*dt)``. Constant ``d(ln P)/dt``; matches
    ``models.exponential``.
``singular``
    ``P = P0*exp(b/(t - t0))``. A finite-time singularity at ``t0``: the rate
    accelerates without bound as ``t`` approaches it. This is Tylenda et al.
    (2011) Eq. (1), the description of V1309 Sco's orbital decay through its
    merger, and its decay timescale has the closed form
    ``P/|dP/dt| = (t - t0)**2 / b``.
``pejcha``
    ``P = P0*exp((amplitude/(1-delta)) * |t - t0|**(1-delta))``. Pejcha (2014)
    Eq. (12),
    obtained by integrating a mass-loss rate ``~|t - t0|**(-delta)``. It
    contains ``singular`` as the ``delta = 2`` case (``amplitude = b`` for a
    ``t0`` after the data) and lets the exponent float instead.

``t0`` is constrained to lie *outside* the observed span, and no closer to it
than the median spacing of the period points — a singularity inside the data
would mean the period passed through zero or infinity mid-run, and one within a
spacing of the edge is a spike on a single measurement rather than a trend the
track can resolve.

The two singular laws are not fitted in their published coordinates. About the
mid-epoch ``t_c`` of the data, with ``s = t - t_c`` and ``u = 1/(t0 - t_c)``,

    ``P = P_c * exp(a * g(s))``,  ``g = s/(1 - u*s)`` for ``singular``,

where ``P_c`` is the period at ``t_c`` and ``a`` its logarithmic rate there
(``pejcha``'s ``g`` is the same function of ``s``, ``u`` and ``delta``). The
point of the rewrite: a ``t0`` after the data and one before it are the two
signs of the single bounded coordinate ``u``, joined smoothly at ``u = 0``,
where the law *is* the exponential one with ``t0`` at infinity. In the
published coordinates the two branches had to be fitted separately, and on any
track with a trend the wrong one crept towards ``t0 = ±inf`` — a point it could
never reach — until the evaluation budget ran out, which cost seconds per fit.
The epoch floor becomes the box ``|u| <= 1/(span/2 + floor)``.

Fits are weighted least squares on the periods themselves, with inverse-error
weights, which is the convention these published fits use.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.errors import InsufficientDataError

#: Days per Julian year, for reporting decay timescales.
DAYS_PER_YEAR: Final[float] = 365.25

#: Evaluation budget of a bounded (singular-law) fit. The regular coordinates
#: converge in a few tens of evaluations; this is a safety net, not a knob.
_MAX_NFEV: Final[int] = 2000

#: Guard range of ``pejcha``'s exponent (see :func:`_fit_bounds`).
_DELTA_BOUNDS: Final[tuple[float, float]] = (1.05, 8.0)

_TINY: Final[float] = float(np.finfo(np.float64).tiny)


@dataclass(frozen=True)
class _Frame:
    """Constants of one track that every law evaluation needs (computed once)."""

    lo: float
    hi: float
    span: float
    floor: float

    @classmethod
    def of(cls, t: FloatArray) -> _Frame:
        return cls(
            lo=float(t.min()),
            hi=float(t.max()),
            span=_span(t),
            floor=_epoch_floor(t),
        )

    @property
    def mid(self) -> float:
        """Origin of the singular laws' regular coordinates."""
        return 0.5 * (self.lo + self.hi)

    @property
    def u_max(self) -> float:
        """Largest ``|u| = 1/|t0 - t_c|``: ``t0`` exactly one floor off the data."""
        return 1.0 / (0.5 * self.span + self.floor)


@dataclass(frozen=True)
class _Law:
    """Internal description of one period law.

    ``predict``/``dlogp`` work in *model coordinates* ``q`` with ``x`` the time
    from the law's origin: the published parameters for ``constant``,
    ``linear`` and ``exponential`` (origin ``t_ref``), and ``(P_c, a, u[,
    delta])`` for the singular laws (origin ``t_c``, see the module docstring).
    """

    name: str
    param_names: tuple[str, ...]
    description: str
    #: ``(x, q) -> P(t)``
    predict: Callable[[FloatArray, FloatArray], FloatArray]
    #: ``(x, q) -> d(ln P)/dt``
    dlogp: Callable[[FloatArray, FloatArray], FloatArray]
    #: ``(x, P, w, frame) -> starting vectors in optimizer coordinates``
    starts: Callable[[FloatArray, FloatArray, FloatArray, _Frame], list[FloatArray]]
    #: Whether the law has a singular epoch ``t0`` (and the regular coordinates).
    singular_epoch: bool = False


def _safe_log(x: FloatArray) -> FloatArray:
    floored = np.maximum(np.asarray(x, dtype=np.float64), 1e-300)
    return np.asarray(np.log(floored), dtype=np.float64)


def _wls(design: FloatArray, y: FloatArray, w: FloatArray) -> FloatArray:
    """Weighted linear least squares, tolerant of a singular normal matrix."""
    xtw = design.T * w[None, :]
    try:
        coef = np.linalg.solve(xtw @ design, xtw @ y)
    except np.linalg.LinAlgError:
        sw = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(design * sw, y * sw.ravel(), rcond=None)
    return np.asarray(coef, dtype=np.float64)


# --- the singular laws in regular coordinates ---------------------------------


def _sing_g(x: FloatArray, u: float) -> FloatArray:
    """``s/(1 - u*s)``: ``b/(t - t0)`` less its value at ``t_c``, over ``a``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(x / (1.0 - u * x), dtype=np.float64)


def _pej_g(x: FloatArray, u: float, delta: float) -> FloatArray:
    """Pejcha's exponent less its value at ``t_c``, over ``a`` (``-> s`` as u -> 0).

    ``[1 - |1 - u*s|**(1-delta)] / ((1-delta)*u)``, evaluated through
    ``expm1``/``log1p`` so it stays exact as ``u`` approaches 0, where the law
    becomes the exponential one.
    """
    x = np.asarray(x, dtype=np.float64)
    if u == 0.0:
        return x.copy()
    one_m = 1.0 - delta
    ux = u * x
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # ln|1 - u*s|: log1p while u*s is small (where 1 - u*s would round),
        # the plain log further out -- including past t0, mirrored like |t - t0|.
        log_gap = np.where(
            ux < 0.5, np.log1p(-np.minimum(ux, 0.5)), np.log(np.abs(1.0 - ux))
        )
        return np.asarray(-np.expm1(one_m * log_gap) / (one_m * u), dtype=np.float64)


def _physical(law: _Law, q: FloatArray) -> FloatArray:
    """Model coordinates -> the published parameters (``params`` order)."""
    if not law.singular_epoch:
        return np.array(q, dtype=np.float64)
    pc, a, u = float(q[0]), float(q[1]), float(q[2])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv_u = np.float64(1.0) / np.float64(u)
        t0_off = float(inv_u)  # t0 - t_c; +-inf in the exponential limit
        if law.name == "singular":
            b = float(-a * inv_u * inv_u)
            p0 = float(pc * np.exp(-a * inv_u))
            return np.array([p0, b, t0_off], dtype=np.float64)
        delta = float(q[3])
        amp = float(-a * np.sign(u) * np.abs(inv_u) ** delta)
        p0 = float(pc * np.exp(a * inv_u / (1.0 - delta)))
        return np.array([p0, amp, t0_off, delta], dtype=np.float64)


def _physical_jacobian(law: _Law, q: FloatArray) -> FloatArray:
    """``d(published parameters) / d(model coordinates)`` at ``q``.

    The full matrix, not a diagonal of one-parameter multipliers: ``pejcha``'s
    amplitude depends on ``a``, ``u`` *and* ``delta``, and dropping the cross
    terms understated its error several-fold.
    """
    k = len(law.param_names)
    if not law.singular_epoch:
        return np.eye(k)
    phys = _physical(law, q)
    p0 = float(phys[0])
    a, u = float(q[1]), float(q[2])
    jac = np.zeros((k, k), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv_u = np.float64(1.0) / np.float64(u)
        jac[2, 2] = -inv_u * inv_u  # t0 = t_c + 1/u
        if law.name == "singular":
            jac[0, 0] = np.exp(-a * inv_u)
            jac[0, 1] = -p0 * inv_u
            jac[0, 2] = p0 * a * inv_u * inv_u
            jac[1, 1] = -inv_u * inv_u
            jac[1, 2] = 2.0 * a * inv_u**3
            return jac
        delta = float(q[3])
        one_m = 1.0 - delta
        amp = float(phys[1])
        jac[0, 0] = np.exp(a * inv_u / one_m)
        jac[0, 1] = p0 * inv_u / one_m
        jac[0, 2] = -p0 * a * inv_u * inv_u / one_m
        jac[0, 3] = p0 * a * inv_u / (one_m * one_m)
        jac[1, 1] = -np.sign(u) * np.abs(inv_u) ** delta
        jac[1, 2] = a * delta * np.abs(inv_u) ** (delta + 1.0)
        jac[1, 3] = amp * np.log(np.abs(inv_u))  # -A ln|u|
        jac[3, 3] = 1.0
    return jac


# --- the individual laws ------------------------------------------------------


def _start_constant(
    x: FloatArray, p: FloatArray, w: FloatArray, frame: _Frame
) -> list[FloatArray]:
    return [np.array([float(np.dot(w, p) / w.sum())], dtype=np.float64)]


def _start_linear(
    x: FloatArray, p: FloatArray, w: FloatArray, frame: _Frame
) -> list[FloatArray]:
    design = np.stack([np.ones_like(x), x], axis=1)
    return [_wls(design, p, w)]


def _start_exponential(
    x: FloatArray, p: FloatArray, w: FloatArray, frame: _Frame
) -> list[FloatArray]:
    design = np.stack([np.ones_like(x), x], axis=1)
    coef = _wls(design, _safe_log(p), w * p * p)
    return [np.array([math.exp(float(coef[0])), float(coef[1])], dtype=np.float64)]


#: Starting ``u/u_max``: a singular epoch half a span beyond each end.
_START_V: Final[tuple[float, float]] = (0.5, -0.5)


def _start_singular(
    x: FloatArray, p: FloatArray, w: FloatArray, frame: _Frame
) -> list[FloatArray]:
    # At fixed u the law is linear in (ln P_c, a): one weighted fit per side.
    out: list[FloatArray] = []
    for v in _START_V:
        design = np.stack([np.ones_like(x), _sing_g(x, v * frame.u_max)], axis=1)
        coef = _wls(design, _safe_log(p), w * p * p)
        out.append(
            np.array(
                [math.exp(float(coef[0])), float(coef[1]) * frame.span, v],
                dtype=np.float64,
            )
        )
    return out


def _start_pejcha(
    x: FloatArray, p: FloatArray, w: FloatArray, frame: _Frame
) -> list[FloatArray]:
    # delta = 2 reduces to `singular`: the natural start.
    return [
        np.array([q[0], q[1], q[2], 2.0], dtype=np.float64)
        for q in _start_singular(x, p, w, frame)
    ]


def _make_registry() -> dict[str, _Law]:
    def const_p(x: FloatArray, q: FloatArray) -> FloatArray:
        return np.full(np.shape(x), float(q[0]), dtype=np.float64)

    def const_d(x: FloatArray, q: FloatArray) -> FloatArray:
        return np.zeros(np.shape(x), dtype=np.float64)

    def lin_p(x: FloatArray, q: FloatArray) -> FloatArray:
        return np.asarray(q[0] + q[1] * x, dtype=np.float64)

    def lin_d(x: FloatArray, q: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(q[1] / lin_p(x, q), dtype=np.float64)

    def exp_p(x: FloatArray, q: FloatArray) -> FloatArray:
        return np.asarray(q[0] * np.exp(q[1] * x), dtype=np.float64)

    def exp_d(x: FloatArray, q: FloatArray) -> FloatArray:
        return np.full(np.shape(x), float(q[1]), dtype=np.float64)

    def sing_p(x: FloatArray, q: FloatArray) -> FloatArray:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(q[0] * np.exp(q[1] * _sing_g(x, float(q[2]))))

    def sing_d(x: FloatArray, q: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gap = 1.0 - float(q[2]) * np.asarray(x, dtype=np.float64)
            return np.asarray(q[1] / (gap * gap), dtype=np.float64)

    def pej_p(x: FloatArray, q: FloatArray) -> FloatArray:
        with np.errstate(over="ignore", invalid="ignore"):
            g = _pej_g(x, float(q[2]), float(q[3]))
            return np.asarray(q[0] * np.exp(q[1] * g), dtype=np.float64)

    def pej_d(x: FloatArray, q: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gap = 1.0 - float(q[2]) * np.asarray(x, dtype=np.float64)
            return np.asarray(
                q[1] * np.sign(gap) * np.abs(gap) ** (-float(q[3])), dtype=np.float64
            )

    return {
        "constant": _Law(
            "constant", ("period",), "P = P0", const_p, const_d, _start_constant
        ),
        "linear": _Law(
            "linear",
            ("period", "pdot"),
            "P = P0 + Pdot*(t - t_ref); constant dP/dt",
            lin_p,
            lin_d,
            _start_linear,
        ),
        "exponential": _Law(
            "exponential",
            ("period", "dlnp_dt"),
            "P = P0*exp(a*(t - t_ref)); constant d(ln P)/dt",
            exp_p,
            exp_d,
            _start_exponential,
        ),
        "singular": _Law(
            "singular",
            ("period", "b", "t0"),
            "P = P0*exp(b/(t - t0)); Tylenda et al. (2011) Eq. 1",
            sing_p,
            sing_d,
            _start_singular,
            singular_epoch=True,
        ),
        "pejcha": _Law(
            "pejcha",
            ("period", "amplitude", "t0", "delta"),
            "P = P0*exp((amplitude/(1-delta))*|t-t0|**(1-delta)); Pejcha (2014) Eq. 12",
            pej_p,
            pej_d,
            _start_pejcha,
            singular_epoch=True,
        ),
    }


_REGISTRY: Final[dict[str, _Law]] = _make_registry()

#: Laws tried by ``law="auto"``, cheapest first. ``pejcha`` is excluded by
#: default: its extra exponent needs a well-sampled track to be meaningful, and
#: on short baselines it merely re-fits ``singular`` with a wandering ``delta``.
AUTO_LAWS: Final[tuple[str, ...]] = ("constant", "linear", "exponential", "singular")


def available_laws() -> tuple[str, ...]:
    """Names of the registered period laws."""
    return tuple(_REGISTRY)


def _law(name: str) -> _Law:
    try:
        return _REGISTRY[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown period law {name!r}; choose from {sorted(_REGISTRY)}"
        ) from None


# --- the fit result -----------------------------------------------------------


@dataclass(frozen=True)
class PeriodLawFit:
    """One period law fitted to a measured ``P(t)`` track.

    Attributes
    ----------
    law : str
        Registry name of the fitted law.
    description : str
        Its functional form, for captions and reports.
    params : dict
        Fitted parameters by name. ``period`` is ``P0``; ``t0`` (when present)
        is in the input time system, *not* an offset. A singular fit whose
        epoch ran off to infinity (the exponential limit) reports ``t0 = ±inf``.
    errors : dict
        Formal 1-sigma parameter errors from the covariance at the solution,
        already inflated by ``sqrt(chi2_reduced)`` when the fit is poor (or, for
        unweighted points, scaled by the residual scatter). NaN for a parameter
        the points do not constrain, or one that rests on a constraint (a
        ``t0`` held at the epoch floor, an exponent at its guard): neither has
        a Gaussian error, and 0.0 would claim a perfect measurement.
    t_ref : float
        Expansion epoch for the laws that use one (``linear``, ``exponential``).
    chi2, chi2_reduced : float
        Weighted chi-squared and its per-degree-of-freedom value. For
        unweighted points (``weighted`` false) the common error is estimated
        from the residuals, so ``chi2_reduced`` is 1 by construction.
    aic, bic : float
        ``chi2 + 2k`` and ``chi2 + k*ln(n)``; for unweighted points, the
        concentrated-likelihood ``n*ln(RSS/n) + 2k`` and ``n*ln(RSS/n) +
        k*ln(n)``. Lower is better.
    bic_scaled : float
        What :func:`compare_period_laws` and ``law="auto"`` rank on:
        ``chi2/s + k*ln(n)``, with ``s`` the ``chi2_reduced`` of the most
        flexible free fit in the compared set, floored at 1 (unfloored, and on
        the residual sum of squares, for unweighted points). With honest errors
        ``s = 1`` and this *is* ``bic``; with errors understated ``k``-fold, as
        period tracks of real stars routinely are, the plain ``bic``
        multiplies every chi-squared difference by ``k**2`` and lets noise buy
        a period change, while ``bic_scaled`` does not. Its differences carry
        the usual reading (above ~10 decisive). Defined relative to the compared
        set, so NaN on a fit made alone with a named ``law``.
    rms : float
        Unweighted RMS of the period residuals (days) — the intuitive companion
        to ``chi2_reduced``, and immune to mis-scaled input errors.
    n_used, n_params : int
        Number of period points and free parameters.
    converged : bool
        Whether the optimizer reported success.
    at_bound : bool
        The fit ran onto a constraint instead of finding a trend: its variation
        is dominated by the end of the track — the law changes by more over
        the last (or first) resolution element, one median point spacing, than
        over the whole rest of the span, i.e. a singular epoch parked on the
        floor just outside the data, absorbing one discrepant point — or
        ``pejcha``'s exponent reached an end of its numerical guard range. Such
        a fit has not found an interior optimum, and neither its parameters nor
        its BIC mean what they normally would — a three-parameter spike will
        happily out-score a constant on a track that is not changing at all.
        :func:`compare_period_laws` ranks these below every fit that is not
        pinned, whatever their BIC.
    weighted : bool
        Whether ``period_err`` was given (``False``: equal weights, with the
        error scale estimated from the residuals).
    """

    law: str
    description: str
    params: dict[str, float]
    errors: dict[str, float]
    t_ref: float
    chi2: float
    chi2_reduced: float
    aic: float
    bic: float
    rms: float
    n_used: int
    n_params: int
    converged: bool
    at_bound: bool = False
    bic_scaled: float = float("nan")
    weighted: bool = True
    # Model coordinates and their origin (see _Law); ``params`` carries the
    # same fit in published form, so equality does not need them -- and an
    # ndarray in a dataclass's generated __eq__ makes ``==`` raise.
    _vector: FloatArray = field(
        repr=False, compare=False, default_factory=lambda: np.zeros(0)
    )
    _origin: float = field(default=float("nan"), repr=False, compare=False)

    # -- evaluation ---------------------------------------------------------------

    def period_at(self, time: Any) -> FloatArray:
        """Predicted period (days) at ``time``."""
        t = np.atleast_1d(np.asarray(time, dtype=np.float64))
        return _REGISTRY[self.law].predict(self._shift(t), self._vector)

    def dlnp_dt_at(self, time: Any) -> FloatArray:
        """Predicted fractional rate ``d(ln P)/dt`` (1/day) at ``time``."""
        t = np.atleast_1d(np.asarray(time, dtype=np.float64))
        return _REGISTRY[self.law].dlogp(self._shift(t), self._vector)

    def pdot_at(self, time: Any) -> FloatArray:
        """Predicted ``dP/dt`` (day/day) at ``time``."""
        return self.period_at(time) * self.dlnp_dt_at(time)

    def timescale_yr_at(self, time: Any) -> FloatArray:
        """Predicted decay timescale ``P/|dP/dt|`` in years at ``time``.

        For the ``singular`` law this is exactly ``(t - t0)**2 / b`` — the
        closed form that makes an inspiral's timescale collapse quadratically as
        the singular epoch approaches.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(
                1.0 / (np.abs(self.dlnp_dt_at(time)) * DAYS_PER_YEAR),
                dtype=np.float64,
            )

    def fractional_change(self, start: float, end: float) -> float:
        """``P(end)/P(start) - 1``: the headline total fractional change.

        Integrating the law rather than multiplying a rate by a baseline, so it
        stays correct when the rate is not constant — which is the whole reason
        for fitting a law in the first place.
        """
        lo, hi = self.period_at(np.array([start, end], dtype=np.float64))
        if lo == 0.0:
            return float("nan")
        return float(hi / lo - 1.0)

    def _shift(self, t: FloatArray) -> FloatArray:
        """Time from the origin of the model coordinates."""
        origin = self._origin if np.isfinite(self._origin) else self.t_ref
        return t - origin

    @property
    def singular_epoch(self) -> float:
        """The fitted ``t0``, or NaN for laws that do not have one."""
        return self.params.get("t0", float("nan"))

    def to_dict(self, *, prefix: str = "") -> dict[str, Any]:
        """Flatten to a plain dict, optionally with a column-name prefix."""
        out: dict[str, Any] = {
            "law": self.law,
            "chi2": self.chi2,
            "chi2_reduced": self.chi2_reduced,
            "aic": self.aic,
            "bic": self.bic,
            "bic_scaled": self.bic_scaled,
            "rms": self.rms,
            "n_used": self.n_used,
            "n_params": self.n_params,
            "converged": self.converged,
            "at_bound": self.at_bound,
            "weighted": self.weighted,
            "t_ref": self.t_ref,
        }
        for name, value in self.params.items():
            out[name] = value
            out[f"{name}_err"] = self.errors.get(name, float("nan"))
        return {f"{prefix}{k}": v for k, v in out.items()} if prefix else out


# --- fitting ------------------------------------------------------------------


def _span(t: FloatArray) -> float:
    return max(float(t.max() - t.min()), 1e-6)


def _epoch_floor(t: FloatArray) -> float:
    """Closest a singular epoch may sit to the data: the median point spacing.

    Keeping ``t0`` merely *outside* the observed span is not enough. Nothing
    stops it converging to within a hair of the nearest point, and there the
    divergence is confined between that point and its neighbour: ``b/(t - t0)``
    becomes a spike on one measurement, which buys a large chi-squared drop for
    no describable behaviour. That is not a period law, it is an outlier
    absorber with two free parameters, and left unchecked it lets ``singular``
    beat ``constant`` on a track that is not changing at all.

    The median spacing of the points is the shortest interval the track can
    resolve, so a singularity closer than that to the data is unmeasurable by
    construction. It is a floor, not a prior: a genuinely imminent singularity
    — the case this law exists for — sits many spacings away and is untouched.
    """
    if t.size < 2:
        return _span(t)
    return max(float(np.median(np.diff(np.sort(t)))), 1e-9)


def _coordinate_scale(law: _Law, frame: _Frame) -> FloatArray:
    """Optimizer coordinates -> model coordinates, elementwise.

    The singular laws are optimized in ``(P_c, a*span, u/u_max[, delta])``:
    every coordinate order-unity whatever the units of time, and the epoch
    floor a fixed box ``[-1, 1]``.
    """
    scale = np.ones(len(law.param_names), dtype=np.float64)
    if law.singular_epoch:
        scale[1] = 1.0 / frame.span
        scale[2] = frame.u_max
    return scale


def _fit_bounds(law: _Law, n: int) -> tuple[FloatArray, FloatArray]:
    """Box bounds in optimizer coordinates.

    ``u/u_max`` is boxed to ``[-1, 1]``, the epoch floor. The law is singular
    at ``delta = 1`` (the exponent's prefactor blows up) and a very large
    exponent is indistinguishable from a step, so the exponent is boxed well
    away from both.
    """
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    if law.singular_epoch:
        lo[2], hi[2] = -1.0, 1.0
    if "delta" in law.param_names:
        i = law.param_names.index("delta")
        lo[i], hi[i] = _DELTA_BOUNDS
    return lo, hi


def _at_bound(
    law: _Law,
    q: FloatArray,
    err: FloatArray,
    frame: _Frame,
    model: FloatArray,
) -> bool:
    """Has the fit run onto a constraint instead of finding a trend?

    Two independent checks, because a singular law fails in two ways.

    *Edge domination.* A finite-time singularity placed just outside the data
    puts almost all of its variation into the first (or last) resolution
    element of the track: ``b/(t - t0)`` becomes a spike on one measurement,
    which buys a large chi-squared drop while describing nothing. The test
    evaluates the *law* (in ``ln P``) at both ends and one epoch floor -- the
    median point spacing -- inside them: if either end element moves it
    further than half of the whole swing, it is not a trend. That measures the
    law against the track's resolution, not against wherever the last two
    samples happen to fall: a long final gap must not make a genuine inspiral
    look edge-dominated, nor a short one hide a spike. A genuine inspiral
    passes comfortably, its steepening spread over many resolution elements.
    Algebraically it flags a ``t0`` within about one floor of its own floor.

    *Exponent at its guard.* ``pejcha``'s exponent range has no physical
    meaning, it only keeps the optimizer away from the ``delta = 1`` pole and
    out of the numerically hopeless tail, so a fit that reaches an end of it has
    been carried there rather than measured.
    """
    if not law.singular_epoch:
        return False
    f = frame.floor
    if frame.hi - frame.lo > 2.0 * f:
        probe = np.array([frame.lo, frame.lo + f, frame.hi - f, frame.hi])
        m = np.asarray(law.predict(probe - frame.mid, q), dtype=np.float64)
    else:  # too few resolution elements for the probe: the sampled model
        m = np.asarray(model, dtype=np.float64)
        m = m[[0, 1, -2, -1]] if m.size >= 3 else m
    if m.size == 4 and np.all(np.isfinite(m)):
        if np.all(m > 0.0):
            m = np.log(m)
        swing = abs(float(m[3] - m[0]))
        edge = max(abs(float(m[1] - m[0])), abs(float(m[3] - m[2])))
        if swing > 0.0 and edge > 0.5 * swing:
            return True
    if "delta" in law.param_names:
        lo, hi = _fit_bounds(law, len(law.param_names))
        j = law.param_names.index("delta")
        delta, sigma = float(q[j]), float(err[j])
        margin = max(sigma if np.isfinite(sigma) else 0.0,
                     1e-2 * float(hi[j] - lo[j]))
        if delta - margin <= lo[j] or delta + margin >= hi[j]:
            return True
    return False


def _rank_key(fit: PeriodLawFit) -> tuple[bool, float]:
    """Sort key: fits pinned against a constraint rank below every free fit."""
    key = fit.bic_scaled if np.isfinite(fit.bic_scaled) else fit.bic
    return (fit.at_bound, key)


def _common_chi2(fit: PeriodLawFit) -> float:
    """The fit's chi-squared on the scale shared by the compared set.

    The weighted chi-squared when errors were given; otherwise the residual sum
    of squares, i.e. a chi-squared for a common unit error.
    """
    if fit.weighted:
        return fit.chi2
    return float(fit.n_used) * fit.rms * fit.rms


def _ranked(fits: Sequence[PeriodLawFit]) -> list[PeriodLawFit]:
    """The fits with ``bic_scaled`` filled on one common scale, best first.

    The scale ``s`` is the ``chi2_reduced`` of the most flexible fit that is
    free (converged and not ``at_bound``): with honest errors it is about 1 and
    changes nothing (it is floored at 1, so the ranking is then the plain BIC);
    with errors understated ``k``-fold it is about ``k**2`` and takes back the
    factor by which they inflate every chi-squared difference. For unweighted
    points it is the residual variance itself, unfloored, which makes the
    criterion independent of the units of the periods.
    """
    if not fits:
        return []
    pool = [f for f in fits if not f.at_bound and f.converged] or [
        f for f in fits if not f.at_bound
    ] or list(fits)
    ref = max(pool, key=lambda f: (f.n_params, -_common_chi2(f)))
    dof = max(1, ref.n_used - ref.n_params)
    scale = _common_chi2(ref) / dof
    scale = max(1.0, scale) if ref.weighted else max(scale, _TINY)
    out = [
        replace(
            f,
            bic_scaled=float(
                _common_chi2(f) / scale + f.n_params * math.log(max(f.n_used, 1))
            ),
        )
        for f in fits
    ]
    return sorted(out, key=_rank_key)


def _unit_error(t: FloatArray, p: FloatArray) -> float:
    """A common error for unweighted points, for the optimizer's sake only.

    The fitted parameters do not depend on a common error, but the optimizer's
    termination tests do: one day, the old stand-in, puts every residual near
    1e-6 and the chi-squared near 1e-10. The scatter about a straight line is
    the right order of magnitude for any law here.
    """
    if p.size >= 3:
        design = np.stack([np.ones_like(t), t - float(np.mean(t))], axis=1)
        coef, *_ = np.linalg.lstsq(design, p, rcond=None)
        scale = float(np.sqrt(np.mean((p - design @ coef) ** 2)))
    else:
        scale = float(np.std(p))
    floor = 1e-12 * max(float(np.max(np.abs(p))), 1.0) if p.size else 1.0
    return max(scale, floor) if np.isfinite(scale) else floor


def _prepare(
    time: Any, period: Any, period_err: Any | None, t_ref: float | None
) -> tuple[FloatArray, FloatArray, FloatArray, bool, float]:
    """Validate and clean the inputs: ``(t, p, sigma, weighted, t_ref)``."""
    t = np.asarray(time, dtype=np.float64).ravel()
    p = np.asarray(period, dtype=np.float64).ravel()
    if t.size != p.size:
        raise ValueError("time and period must have the same length")
    weighted = period_err is not None
    if period_err is None:
        sigma = np.ones_like(p)
    else:
        sigma = np.asarray(period_err, dtype=np.float64).ravel()
        if sigma.size != p.size:
            raise ValueError("period_err and period must have the same length")
        sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, np.nan)
    good = np.isfinite(t) & np.isfinite(p) & (p > 0.0) & np.isfinite(sigma)
    t, p, sigma = t[good], p[good], sigma[good]
    if t.size == 0:
        raise InsufficientDataError("fit_period_law: no usable period points")
    order = np.argsort(t, kind="stable")
    t, p, sigma = t[order], p[order], sigma[order]
    if not weighted:
        sigma = np.full(p.size, _unit_error(t, p), dtype=np.float64)
    w = 1.0 / (sigma * sigma)
    ref = float(np.dot(w, t) / w.sum()) if t_ref is None else float(t_ref)
    return t, p, sigma, weighted, ref


def fit_period_law(
    time: Any,
    period: Any,
    period_err: Any | None = None,
    *,
    law: str = "auto",
    t_ref: float | None = None,
    scale_errors: bool = True,
) -> PeriodLawFit:
    """Fit a parametric ``P(t)`` law to periods measured in time intervals.

    Parameters
    ----------
    time : array-like
        Epoch of each period measurement (days; typically the interval's
        weighted mean epoch, as :func:`cuperiod.period_profile` returns).
    period : array-like
        Measured periods (days).
    period_err : array-like, optional
        1-sigma period errors. ``None`` weights every point equally and
        estimates the common error from the residuals: the parameter errors
        then carry the residual scatter, and the laws are compared with a
        criterion that does not depend on the units of the periods (see
        :attr:`PeriodLawFit.bic_scaled`).
    law : str, default "auto"
        A name from :func:`available_laws`, or ``"auto"`` to fit every law in
        :data:`AUTO_LAWS` and return the best by :func:`compare_period_laws`'s
        ranking.
    t_ref : float, optional
        Expansion epoch for the laws that use one. Defaults to the weighted mean
        epoch of ``time``. Ignored by ``singular`` / ``pejcha``, which carry
        their own ``t0``.
    scale_errors : bool, default True
        Inflate the parameter errors by ``sqrt(chi2_reduced)`` when the fit is
        poor. The period points of a real star scatter well beyond their formal
        errors, so this is on by default. Without ``period_err`` there is no
        formal error to inflate and the residual scatter is always used.

    Returns
    -------
    PeriodLawFit

    Examples
    --------
    >>> prof = cup.period_profile(lc, period=1.43)              # doctest: +SKIP
    >>> fit = cup.fit_period_law(prof.t_mid, prof.period,
    ...                          prof.period_err, law="singular")  # doctest: +SKIP
    >>> fit.params["t0"], fit.fractional_change(t[0], t[-1])    # doctest: +SKIP
    """
    if law != "auto":
        _law(law)  # an unknown name is a caller error, before any work
    t, p, sigma, weighted, ref = _prepare(time, period, period_err, t_ref)

    if law == "auto":
        fits = [
            f
            for name in AUTO_LAWS
            if (f := _fit_or_skip(name, t, p, sigma, ref, scale_errors, weighted))
            is not None
        ]
        if not fits:
            raise InsufficientDataError(
                "fit_period_law: no law could be fitted to these period points"
            )
        return _ranked(fits)[0]

    fit = _try_fit(law, t, p, sigma, ref, scale_errors, weighted)
    if fit is None:
        need = len(_law(law).param_names)
        raise InsufficientDataError(
            f"fit_period_law: {t.size} period points cannot constrain "
            f"law={law!r} ({need} free parameters)"
        )
    return fit


def _fit_or_skip(
    name: str,
    t: FloatArray,
    p: FloatArray,
    sigma: FloatArray,
    t_ref: float,
    scale_errors: bool,
    weighted: bool,
) -> PeriodLawFit | None:
    """:func:`_try_fit`, with a numerically degenerate law skipped, not raised."""
    try:
        return _try_fit(name, t, p, sigma, t_ref, scale_errors, weighted)
    except (InsufficientDataError, np.linalg.LinAlgError):
        return None


def _errors(
    law: _Law,
    jac: FloatArray,
    z: FloatArray,
    q: FloatArray,
    zscale: FloatArray,
    active: FloatArray,
) -> FloatArray:
    """Formal 1-sigma errors of the published parameters (before inflation).

    The covariance of the optimizer coordinates comes from the Gauss-Newton
    approximation, by SVD so a degenerate direction cannot blow up the rest;
    it is carried to the published parameters through the full Jacobian of the
    reparametrization. A published parameter that depends on a coordinate the
    points do not constrain (weight in a dropped singular direction), or on one
    held by its bound, gets NaN: the covariance says nothing true about it,
    and the 0.0 that a truncated direction otherwise yields reads as a perfect
    measurement.
    """
    k = len(law.param_names)
    try:
        _, s, vt = np.linalg.svd(np.asarray(jac, dtype=np.float64), full_matrices=False)
    except (np.linalg.LinAlgError, ValueError):
        return np.full(k, np.nan)
    if s.size != k or not (np.isfinite(s[0]) and s[0] > 0.0):
        return np.full(k, np.nan)
    keep = s > s[0] * 1e-12
    cov_z = (vt[keep].T / (s[keep] ** 2)) @ vt[keep]
    unconstrained = np.sum(vt[~keep] ** 2, axis=0) > 1e-8
    lo, hi = _fit_bounds(law, k)
    boxed = np.isfinite(lo) & np.isfinite(hi)
    tol = 1e-6 * np.where(boxed, hi - lo, 1.0)
    pinned = (active != 0) | (boxed & ((z - lo <= tol) | (hi - z <= tol)))
    g = _physical_jacobian(law, q) * zscale[None, :]
    with np.errstate(invalid="ignore", over="ignore"):
        var = np.einsum("ij,jk,ik->i", g, cov_z, g)
        err = np.sqrt(np.clip(var, 0.0, np.inf))
    shaky = unconstrained | pinned
    depends = (np.abs(np.nan_to_num(g, nan=1.0)) > 0.0).astype(np.float64)
    err[(depends @ shaky.astype(np.float64)) > 0.0] = np.nan
    return np.asarray(err, dtype=np.float64)


def _try_fit(
    name: str,
    t: FloatArray,
    p: FloatArray,
    sigma: FloatArray,
    t_ref: float,
    scale_errors: bool,
    weighted: bool = True,
) -> PeriodLawFit | None:
    """Fit one law; ``None`` when it cannot be constrained by these points."""
    from scipy.optimize import least_squares

    spec = _law(name)
    k = len(spec.param_names)
    n = int(t.size)
    if n <= k:
        return None
    w = 1.0 / (sigma * sigma)
    frame = _Frame.of(t)
    origin = frame.mid if spec.singular_epoch else t_ref
    x = t - origin
    zscale = _coordinate_scale(spec, frame)
    bounds = _fit_bounds(spec, k)

    def residual(z: FloatArray) -> FloatArray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            model = spec.predict(x, z * zscale)
            out = (np.asarray(model, dtype=np.float64) - p) / sigma
        return np.where(np.isfinite(out), out, 1e6)

    best: Any = None
    for start in spec.starts(x, p, w, frame):
        z0 = np.clip(np.asarray(start, dtype=np.float64), bounds[0], bounds[1])
        if not np.all(np.isfinite(z0)):
            continue
        try:
            if spec.singular_epoch:
                sol = least_squares(
                    residual, z0, bounds=bounds, method="trf", max_nfev=_MAX_NFEV
                )
            else:
                sol = least_squares(residual, z0, method="lm")
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(sol.cost) and (best is None or sol.cost < best.cost):
            best = sol
    if best is None:
        return None

    z = np.asarray(best.x, dtype=np.float64)
    q = z * zscale
    chi2 = float(2.0 * best.cost)
    dof = max(1, n - k)
    chi2_red = chi2 / dof
    if weighted:
        inflate = math.sqrt(max(1.0, chi2_red)) if scale_errors else 1.0
    else:
        # No errors were given: the residual scatter is the only scale there is.
        inflate = math.sqrt(chi2_red)
    active = np.asarray(
        getattr(best, "active_mask", np.zeros(k)), dtype=np.float64
    )
    err = _errors(spec, best.jac, z, q, zscale, active) * inflate

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        model = np.asarray(spec.predict(x, q), dtype=np.float64)
    resid = model - p
    rss = float(np.dot(resid, resid))
    if weighted:
        chi2_out, chi2_red_out = chi2, chi2_red
        aic = chi2 + 2.0 * k
        bic = chi2 + k * math.log(n)
    else:
        # Errors estimated from these residuals: chi2_reduced is 1 by
        # construction (0 for an exact fit) and the criteria are the
        # concentrated likelihood's, which do not depend on the units.
        chi2_out = float(dof) if rss > 0.0 else 0.0
        chi2_red_out = 1.0 if rss > 0.0 else 0.0
        log_l = n * math.log(max(rss, _TINY) / n)
        aic = log_l + 2.0 * k
        bic = log_l + k * math.log(n)

    phys = _physical(spec, q)
    if spec.singular_epoch:
        phys[2] = origin + phys[2]  # t0 back in the input time system
    return PeriodLawFit(
        law=name,
        description=spec.description,
        params=dict(zip(spec.param_names, (float(v) for v in phys), strict=True)),
        errors=dict(zip(spec.param_names, (float(v) for v in err), strict=True)),
        t_ref=t_ref,
        chi2=chi2_out,
        chi2_reduced=chi2_red_out,
        aic=aic,
        bic=bic,
        rms=float(np.sqrt(rss / n)),
        n_used=n,
        n_params=k,
        converged=bool(best.success),
        at_bound=_at_bound(spec, q, err, frame, model),
        weighted=weighted,
        _vector=q,
        _origin=float(origin),
    )


def compare_period_laws(
    time: Any,
    period: Any,
    period_err: Any | None = None,
    *,
    laws: Sequence[str] = AUTO_LAWS,
    t_ref: float | None = None,
    scale_errors: bool = True,
) -> list[PeriodLawFit]:
    """Fit several period laws to the same track and rank them.

    Parameters
    ----------
    time, period, period_err
        As :func:`fit_period_law`.
    laws : sequence of str, default :data:`AUTO_LAWS`
        Laws to fit (a single name is accepted too, and ``"auto"`` stands for
        :data:`AUTO_LAWS`). Laws that cannot be constrained by the available
        points are skipped rather than raising; an unknown name is a mistake
        and raises ``ValueError``.
    t_ref, scale_errors
        As :func:`fit_period_law`.

    Returns
    -------
    list of PeriodLawFit
        Best first: fits whose parameters sit on a constraint (``at_bound``) are
        ranked below every fit that is free, whatever their BIC, and the rest are
        ordered by :attr:`PeriodLawFit.bic_scaled` -- the BIC with every
        chi-squared divided by the excess scatter of the most flexible free law,
        so errors that understate the real scatter of the track cannot buy a
        period change. A ``bic_scaled`` gap above ~10 over the runner-up is
        conventionally decisive; a gap of 2 is not worth reporting as a
        preference.

    Raises
    ------
    ValueError
        An unknown law name, or arrays of different lengths.
    """
    names = (laws,) if isinstance(laws, str) else tuple(laws)
    expanded: list[str] = []
    for name in names:
        group = AUTO_LAWS if name == "auto" else (name,)
        for item in group:
            _law(item)
            if item not in expanded:
                expanded.append(item)
    t, p, sigma, weighted, ref = _prepare(time, period, period_err, t_ref)
    fits = [
        f
        for name in expanded
        if (f := _fit_or_skip(name, t, p, sigma, ref, scale_errors, weighted))
        is not None
    ]
    return _ranked(fits)


__all__ = [
    "AUTO_LAWS",
    "DAYS_PER_YEAR",
    "PeriodLawFit",
    "available_laws",
    "compare_period_laws",
    "fit_period_law",
]
