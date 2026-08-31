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
    contains ``singular`` as the ``delta = 2`` case (``amplitude = b``) and lets
    the exponent float instead.

``t0`` is constrained to lie *outside* the observed span, and no closer to it
than the median spacing of the period points — a singularity inside the data
would mean the period passed through zero or infinity mid-run, and one within a
spacing of the edge is a spike on a single measurement rather than a trend the
track can resolve. It is fitted as the log of that standoff, so the optimizer
can reach neither. Both branches (``t0`` after the data, the inspiral case; and
before it) are tried and the better chi-squared kept.

Fits are weighted least squares on the periods themselves, with inverse-error
weights, which is the convention these published fits use.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.errors import InsufficientDataError

#: Days per Julian year, for reporting decay timescales.
DAYS_PER_YEAR: Final[float] = 365.25


@dataclass(frozen=True)
class _Law:
    """Internal description of one period law."""

    name: str
    param_names: tuple[str, ...]
    description: str
    #: ``(t, params) -> P(t)``
    predict: Callable[[FloatArray, FloatArray], FloatArray]
    #: ``(t, params) -> d(ln P)/dt``
    dlogp: Callable[[FloatArray, FloatArray], FloatArray]
    #: ``(t, P, w, t_ref) -> list of starting vectors`` (several = several branches)
    starts: Callable[[FloatArray, FloatArray, FloatArray, float], list[FloatArray]]
    #: Names that are fitted as ``log`` offsets from the data edge (the ``t0``s).
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


# --- the individual laws ------------------------------------------------------


def _start_constant(
    t: FloatArray, p: FloatArray, w: FloatArray, t_ref: float
) -> list[FloatArray]:
    return [np.array([float(np.dot(w, p) / w.sum())], dtype=np.float64)]


def _start_linear(
    t: FloatArray, p: FloatArray, w: FloatArray, t_ref: float
) -> list[FloatArray]:
    design = np.stack([np.ones_like(t), t - t_ref], axis=1)
    return [_wls(design, p, w)]


def _start_exponential(
    t: FloatArray, p: FloatArray, w: FloatArray, t_ref: float
) -> list[FloatArray]:
    design = np.stack([np.ones_like(t), t - t_ref], axis=1)
    coef = _wls(design, _safe_log(p), w * p * p)
    return [np.array([math.exp(float(coef[0])), float(coef[1])], dtype=np.float64)]


def _epoch_candidates(t: FloatArray) -> list[float]:
    """Trial singular epochs: one span beyond each end of the data."""
    lo, hi = float(t.min()), float(t.max())
    span = max(hi - lo, 1e-6)
    return [hi + 0.5 * span, lo - 0.5 * span]


def _start_singular(
    t: FloatArray, p: FloatArray, w: FloatArray, t_ref: float
) -> list[FloatArray]:
    out: list[FloatArray] = []
    for t0 in _epoch_candidates(t):
        with np.errstate(divide="ignore", invalid="ignore"):
            basis = 1.0 / (t - t0)
        design = np.stack([np.ones_like(t), basis], axis=1)
        coef = _wls(design, _safe_log(p), w * p * p)
        out.append(
            np.array(
                [math.exp(float(coef[0])), float(coef[1]), t0], dtype=np.float64
            )
        )
    return out


def _start_pejcha(
    t: FloatArray, p: FloatArray, w: FloatArray, t_ref: float
) -> list[FloatArray]:
    # delta = 2 reduces to `singular` with amplitude = b: the natural start.
    return [
        np.array([q[0], q[1], q[2], 2.0], dtype=np.float64)
        for q in _start_singular(t, p, w, t_ref)
    ]


def _make_registry() -> dict[str, _Law]:
    def const_p(t: FloatArray, q: FloatArray) -> FloatArray:
        return np.full(np.shape(t), float(q[0]), dtype=np.float64)

    def const_d(t: FloatArray, q: FloatArray) -> FloatArray:
        return np.zeros(np.shape(t), dtype=np.float64)

    def lin_p(t: FloatArray, q: FloatArray) -> FloatArray:
        return np.asarray(q[0] + q[1] * t, dtype=np.float64)

    def lin_d(t: FloatArray, q: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(q[1] / lin_p(t, q), dtype=np.float64)

    def exp_p(t: FloatArray, q: FloatArray) -> FloatArray:
        return np.asarray(q[0] * np.exp(q[1] * t), dtype=np.float64)

    def exp_d(t: FloatArray, q: FloatArray) -> FloatArray:
        return np.full(np.shape(t), float(q[1]), dtype=np.float64)

    def sing_p(t: FloatArray, q: FloatArray) -> FloatArray:
        return np.asarray(q[0] * np.exp(q[1] / (t - q[2])), dtype=np.float64)

    def sing_d(t: FloatArray, q: FloatArray) -> FloatArray:
        d = np.asarray(t, dtype=np.float64) - q[2]
        return np.asarray(-q[1] / (d * d), dtype=np.float64)

    def pej_p(t: FloatArray, q: FloatArray) -> FloatArray:
        gap = np.abs(np.asarray(t, dtype=np.float64) - q[2])
        return np.asarray(
            q[0] * np.exp((q[1] / (1.0 - q[3])) * gap ** (1.0 - q[3])),
            dtype=np.float64,
        )

    def pej_d(t: FloatArray, q: FloatArray) -> FloatArray:
        diff = np.asarray(t, dtype=np.float64) - q[2]
        return np.asarray(
            q[1] * np.sign(diff) * np.abs(diff) ** (-q[3]), dtype=np.float64
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
        is in the input time system, *not* an offset.
    errors : dict
        Formal 1-sigma parameter errors from the covariance at the solution,
        already inflated by ``sqrt(chi2_reduced)`` when the fit is poor.
    t_ref : float
        Expansion epoch for the laws that use one (``linear``, ``exponential``).
    chi2, chi2_reduced : float
        Weighted chi-squared and its per-degree-of-freedom value.
    aic, bic : float
        ``chi2 + 2k`` and ``chi2 + k*ln(n)``. Lower is better; the ``bic``
        difference is what :func:`compare_period_laws` ranks on.
    rms : float
        Unweighted RMS of the period residuals (days) — the intuitive companion
        to ``chi2_reduced``, and immune to mis-scaled input errors.
    n_used, n_params : int
        Number of period points and free parameters.
    converged : bool
        Whether the optimizer reported success.
    at_bound : bool
        The fit ran onto a constraint instead of finding a trend: its variation
        is dominated by a single step at one end of the track (a singular epoch
        parked just outside the data, absorbing one discrepant point), or
        ``pejcha``'s exponent reached an end of its numerical guard range. Such
        a fit has not found an interior optimum, and neither its parameters nor
        its BIC mean what they normally would — a three-parameter spike will
        happily out-score a constant on a track that is not changing at all.
        :func:`compare_period_laws` ranks these below every fit that is not
        pinned, whatever their BIC.
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
    _vector: FloatArray = field(repr=False, default_factory=lambda: np.zeros(0))

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
        """Laws without a ``t0`` are expanded about ``t_ref``; the rest are not."""
        if "t0" in _REGISTRY[self.law].param_names:
            return t
        return t - self.t_ref

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
            "rms": self.rms,
            "n_used": self.n_used,
            "n_params": self.n_params,
            "converged": self.converged,
            "at_bound": self.at_bound,
            "t_ref": self.t_ref,
        }
        for name, value in self.params.items():
            out[name] = value
            out[f"{name}_err"] = self.errors.get(name, float("nan"))
        return {f"{prefix}{k}": v for k, v in out.items()} if prefix else out


# --- fitting ------------------------------------------------------------------


def _pack(law: _Law, vec: FloatArray, t: FloatArray) -> FloatArray:
    """Physical parameters -> optimizer coordinates.

    Two reconditionings, both undone by :func:`_unpack`:

    * ``t0`` becomes ``log`` of its *standoff* beyond the nearer data edge — the
      offset less :func:`_epoch_floor` — so the optimizer can reach neither the
      inside of the observed span nor its immediate edge.
    * ``pejcha``'s ``amplitude`` carries units of ``day**(delta-1)``, so its magnitude
      swings by many orders as ``delta`` moves and the Jacobian becomes
      hopelessly scaled. Fitting ``amplitude * T**(1-delta)`` — the same coefficient
      with the gap measured in units of the baseline — keeps it order-unity for
      every exponent.
    """
    if not law.singular_epoch:
        return np.array(vec, dtype=np.float64)
    out = np.array(vec, dtype=np.float64)
    i = law.param_names.index("t0")
    hi, lo = float(t.max()), float(t.min())
    gap = out[i] - hi if out[i] > hi else lo - out[i]
    if "delta" in law.param_names:
        out[1] = out[1] * _span(t) ** (1.0 - out[law.param_names.index("delta")])
    out[i] = math.log(max(gap - _epoch_floor(t), 1e-6))
    return out


def _unpack(law: _Law, vec: FloatArray, t: FloatArray, after: bool) -> FloatArray:
    """Optimizer coordinates -> physical parameters (see :func:`_pack`)."""
    if not law.singular_epoch:
        return np.array(vec, dtype=np.float64)
    out = np.array(vec, dtype=np.float64)
    i = law.param_names.index("t0")
    gap = _epoch_floor(t) + math.exp(min(float(out[i]), 700.0))
    out[i] = (float(t.max()) + gap) if after else (float(t.min()) - gap)
    if "delta" in law.param_names:
        out[1] = out[1] * _span(t) ** (out[law.param_names.index("delta")] - 1.0)
    return out


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


def _at_bound(
    law: _Law, vector: FloatArray, err: FloatArray, model: FloatArray, t: FloatArray
) -> bool:
    """Has the fit run onto a constraint instead of finding a trend?

    Two independent checks, because a singular law fails in two ways.

    *Edge domination.* A finite-time singularity placed just outside the data
    puts almost all of its variation into the first (or last) step of the
    track: ``b/(t - t0)`` becomes a spike on one measurement, which buys a large
    chi-squared drop while describing nothing. The test is scale-free and needs
    no tolerance — if a single end step moves the model further than the whole
    rest of the track does, it is not a trend. A genuine inspiral fails this
    test comfortably: its steepening is spread over many points, and the largest
    single step is a small fraction of the total swing.

    *Exponent at its guard.* ``pejcha``'s exponent range has no physical
    meaning, it only keeps the optimizer away from the ``delta = 1`` pole and
    out of the numerically hopeless tail, so a fit that reaches an end of it has
    been carried there rather than measured.
    """
    if not law.singular_epoch:
        return False
    m = np.asarray(model, dtype=np.float64)
    if m.size >= 3 and np.all(np.isfinite(m)):
        swing = float(m.max() - m.min())
        edge = max(abs(float(m[1] - m[0])), abs(float(m[-1] - m[-2])))
        if swing > 0.0 and edge > 0.5 * swing:
            return True
    if "delta" in law.param_names:
        lo, hi = _fit_bounds(law, len(law.param_names))
        j = law.param_names.index("delta")
        delta, sigma = float(vector[j]), float(err[j])
        margin = max(sigma if np.isfinite(sigma) else 0.0,
                     1e-2 * float(hi[j] - lo[j]))
        if delta - margin <= lo[j] or delta + margin >= hi[j]:
            return True
    return False


def _rank_key(fit: PeriodLawFit) -> tuple[bool, float]:
    """Sort key: fits pinned against a constraint rank below every free fit."""
    return (fit.at_bound, fit.bic)


def _fit_bounds(law: _Law, n: int) -> tuple[FloatArray, FloatArray]:
    """Box bounds in packed coordinates; only ``delta`` is actually constrained.

    The law is singular at ``delta = 1`` (the exponent's prefactor blows up) and
    a very large exponent is indistinguishable from a step, so the exponent is
    boxed well away from both.
    """
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    if "delta" in law.param_names:
        i = law.param_names.index("delta")
        lo[i], hi[i] = 1.05, 8.0
    return lo, hi


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
        1-sigma period errors. ``None`` weights every point equally.
    law : str, default "auto"
        A name from :func:`available_laws`, or ``"auto"`` to fit every law in
        :data:`AUTO_LAWS` and return the lowest-BIC one.
    t_ref : float, optional
        Expansion epoch for the laws that use one. Defaults to the weighted mean
        epoch of ``time``. Ignored by ``singular`` / ``pejcha``, which carry
        their own ``t0``.
    scale_errors : bool, default True
        Inflate the parameter errors by ``sqrt(chi2_reduced)`` when the fit is
        poor. The period points of a real star scatter well beyond their formal
        errors, so this is on by default.

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

    t = np.asarray(time, dtype=np.float64).ravel()
    p = np.asarray(period, dtype=np.float64).ravel()
    if t.size != p.size:
        raise ValueError("time and period must have the same length")
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
    w = 1.0 / (sigma * sigma)
    ref = float(np.dot(w, t) / w.sum()) if t_ref is None else float(t_ref)

    if law == "auto":
        fits = [
            f
            for name in AUTO_LAWS
            if (f := _try_fit(name, t, p, sigma, ref, scale_errors)) is not None
        ]
        if not fits:
            raise InsufficientDataError(
                "fit_period_law: no law could be fitted to these period points"
            )
        return min(fits, key=_rank_key)

    fit = _try_fit(law, t, p, sigma, ref, scale_errors)
    if fit is None:
        spec = _REGISTRY.get(law)
        need = len(spec.param_names) if spec is not None else 0
        raise InsufficientDataError(
            f"fit_period_law: {t.size} period points cannot constrain "
            f"law={law!r} ({need} free parameters)"
        )
    return fit


def _try_fit(
    name: str,
    t: FloatArray,
    p: FloatArray,
    sigma: FloatArray,
    t_ref: float,
    scale_errors: bool,
) -> PeriodLawFit | None:
    """Fit one law; ``None`` when it cannot be constrained by these points."""
    from scipy.optimize import least_squares

    try:
        spec = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown period law {name!r}; choose from {sorted(_REGISTRY)}"
        ) from None
    k = len(spec.param_names)
    if t.size <= k:
        return None
    w = 1.0 / (sigma * sigma)
    shifted = t if "t0" in spec.param_names else t - t_ref

    best: tuple[float, Any, FloatArray] | None = None
    for start in spec.starts(t, p, w, t_ref):
        after = bool(
            spec.singular_epoch
            and start[spec.param_names.index("t0")] > t.max()
        )

        def residual(x: FloatArray, _after: bool = after) -> FloatArray:
            q = _unpack(spec, x, t, _after)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                model = spec.predict(shifted, q)
            out = (np.asarray(model, dtype=np.float64) - p) / sigma
            return np.where(np.isfinite(out), out, 1e6)

        try:
            if spec.singular_epoch:
                sol = least_squares(
                    residual,
                    _pack(spec, start, t),
                    bounds=_fit_bounds(spec, k),
                    method="trf",
                    max_nfev=4000,
                )
            else:
                sol = least_squares(residual, _pack(spec, start, t), method="lm")
        except (ValueError, np.linalg.LinAlgError):
            continue
        chi2 = float(2.0 * sol.cost)
        if not np.isfinite(chi2):
            continue
        if best is None or chi2 < best[0]:
            best = (chi2, sol, _unpack(spec, sol.x, t, after))
    if best is None:
        return None

    chi2, sol, vector = best
    dof = max(1, t.size - k)
    chi2_red = chi2 / dof
    inflate = math.sqrt(max(1.0, chi2_red)) if scale_errors else 1.0
    # Covariance from the Gauss-Newton approximation at the solution. The
    # optimizer works in packed coordinates, so a log-offset t0's error is
    # rescaled by d(t0)/d(log gap) = gap.
    try:
        jac = np.asarray(sol.jac, dtype=np.float64)
        _, s, vt = np.linalg.svd(jac, full_matrices=False)
        keep = s > max(s[0], 1e-300) * 1e-12
        cov = (vt[keep].T / (s[keep] ** 2)) @ vt[keep]
        err_packed = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    except (np.linalg.LinAlgError, IndexError, ValueError):
        err_packed = np.full(k, np.nan)
    # Packed -> physical errors: both reconditionings in _pack are monotone
    # one-parameter maps, so their derivative is a plain multiplier.
    err = np.array(err_packed, dtype=np.float64) * inflate
    if spec.singular_epoch:
        i = spec.param_names.index("t0")
        gap = abs(float(vector[i]) - (t.max() if vector[i] > t.max() else t.min()))
        err[i] = err[i] * max(gap - _epoch_floor(t), 0.0)
        if "delta" in spec.param_names:
            delta = float(vector[spec.param_names.index("delta")])
            err[1] = err[1] * _span(t) ** (delta - 1.0)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        model = np.asarray(spec.predict(shifted, vector), dtype=np.float64)
    resid = model - p
    return PeriodLawFit(
        law=name,
        description=spec.description,
        params=dict(zip(spec.param_names, (float(v) for v in vector), strict=True)),
        errors=dict(zip(spec.param_names, (float(v) for v in err), strict=True)),
        t_ref=t_ref,
        chi2=chi2,
        chi2_reduced=chi2_red,
        aic=chi2 + 2.0 * k,
        bic=chi2 + k * math.log(t.size),
        rms=float(np.sqrt(np.mean(resid * resid))),
        n_used=int(t.size),
        n_params=k,
        converged=bool(sol.success),
        at_bound=_at_bound(spec, vector, err, model, t),
        _vector=vector,
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
    """Fit several period laws to the same track and rank them by BIC.

    Parameters
    ----------
    time, period, period_err
        As :func:`fit_period_law`.
    laws : sequence of str, default :data:`AUTO_LAWS`
        Laws to fit. Ones that cannot be constrained by the available points are
        skipped rather than raising.
    t_ref, scale_errors
        As :func:`fit_period_law`.

    Returns
    -------
    list of PeriodLawFit
        Best first: fits whose parameters sit on a constraint (``at_bound``) are
        ranked below every fit that is free, whatever their BIC, and the rest are
        ordered by BIC. A ``bic`` gap above ~10 over the runner-up is
        conventionally decisive; a gap of 2 is not worth reporting as a
        preference.
    """
    fits = []
    for name in laws:
        try:
            fits.append(
                fit_period_law(
                    time,
                    period,
                    period_err,
                    law=name,
                    t_ref=t_ref,
                    scale_errors=scale_errors,
                )
            )
        except (InsufficientDataError, ValueError):
            continue
    return sorted(fits, key=_rank_key)


__all__ = [
    "AUTO_LAWS",
    "DAYS_PER_YEAR",
    "PeriodLawFit",
    "available_laws",
    "compare_period_laws",
    "fit_period_law",
]
