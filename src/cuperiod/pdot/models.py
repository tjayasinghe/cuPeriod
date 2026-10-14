"""Period-evolution models: how ``P`` is allowed to change with time.

Every model here shares one property that makes the whole subsystem work: the
accumulated phase factorizes as

    phi(t) = f0 * tau(dt),      dt = t - t_ref,

with a warp ``tau`` that does **not** depend on the trial frequency. That is
what lets :func:`cuperiod.pdot_search` turn any registered periodogram into a
period-drift search — one warp per trial rate, the frequency axis untouched —
and what lets :mod:`cuperiod.pdot.track` express an O-C curve as
``dt - tau(dt)`` without ever committing to a Taylor expansion.

All models are parameterized by the same scalar

    a = (dP/dt) / P   evaluated at ``t_ref``, in 1/day,

so ``a`` means the same thing in each, the models agree to first order, and a
search grid in ``a`` can be reinterpreted under a different model without
rescaling. They differ in how the rate *evolves*:

``constant-pdot``
    ``P(t) = P0 * (1 + a*dt)`` — dP/dt is constant, the classical quadratic
    ephemeris and the model behind PHOEBE's ``dpdt``. The period reaches zero
    at ``dt = -1/a``, so the warp has a finite domain.

``exponential``
    ``P(t) = P0 * exp(a*dt)`` — d(ln P)/dt is constant, i.e. the *fractional*
    rate rather than the absolute one. This is the standard description of
    mass-transfer-driven period change and of a binary spiralling in, where
    each e-folding takes the same time however short the period has become. It
    has no singular epoch: the warp is defined for every ``dt``.

The two are indistinguishable over a short baseline and diverge in the *cubic*
term of the O-C curve, which is exactly what a residual ladder
(:func:`cuperiod.ephemeris_ladder`) measures:

    constant-pdot :  O-C = (a/2)*dt^2 - (a^2/3)*dt^3 + ...
    exponential   :  O-C = (a/2)*dt^2 - (a^2/6)*dt^3 + ...

A light curve whose O-C keeps shedding residual when a cubic term is allowed is
telling you the rate itself is changing; see
:func:`cuperiod.period_profile` for the model-free alternative of measuring
``P`` and ``dP/dt`` independently in consecutive intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from cuperiod.core._typing import FloatArray

#: Model name used when the caller does not choose one. The classical quadratic
#: ephemeris: what "dP/dt" means in the eclipse-timing literature.
DEFAULT_MODEL: Final[str] = "constant-pdot"


@dataclass(frozen=True)
class PeriodModel:
    """One period-evolution law, in the frequency-independent warp form.

    Attributes
    ----------
    name : str
        Registry key (``"constant-pdot"`` / ``"exponential"``).
    description : str
        One-line human-readable summary, used in reports and plot labels.
    singular : bool
        Whether the warp has a finite domain (the period reaching zero).
    """

    name: str
    description: str
    singular: bool

    # -- the warp ----------------------------------------------------------------

    def warp(self, dt: FloatArray, a: float) -> FloatArray:
        """Warped time ``tau`` such that the phase is exactly ``f0 * tau``.

        Parameters
        ----------
        dt : numpy.ndarray
            Time since the reference epoch (days).
        a : float
            Fractional rate ``(dP/dt)/P`` at the reference epoch (1/day).

        Returns
        -------
        numpy.ndarray
            ``tau`` in days; equal to ``dt`` when ``a == 0``.

        Raises
        ------
        ValueError
            If ``a`` is not finite (a NaN rate warped every time to NaN, and an
            infinite one was reported as a period passing through zero).
        """
        if not np.isfinite(a):
            raise ValueError(f"the rate a = (dP/dt)/P must be finite, got {a}")
        if a == 0.0:
            return np.asarray(dt, dtype=np.float64)
        return self._warp(np.asarray(dt, dtype=np.float64), a)

    def _warp(self, dt: FloatArray, a: float) -> FloatArray:  # pragma: no cover
        raise NotImplementedError

    def delay(self, dt: FloatArray, a: float) -> FloatArray:
        """O-C delay (days) of this model against a constant-period ephemeris.

        ``dt - tau(dt)``: the time by which events run late (positive) relative
        to a clock ticking at the reference period. Its leading term is
        ``(a/2)*dt**2`` for every model.
        """
        return np.asarray(dt, dtype=np.float64) - self.warp(dt, a)

    def period_at(self, dt: FloatArray, period: float, a: float) -> FloatArray:
        """Instantaneous period (days) at ``dt``, given ``P(t_ref) = period``."""
        return period * self._period_factor(np.asarray(dt, dtype=np.float64), a)

    def _period_factor(self, dt: FloatArray, a: float) -> FloatArray:
        raise NotImplementedError  # pragma: no cover

    def pdot_at(self, dt: FloatArray, period: float, a: float) -> FloatArray:
        """Instantaneous ``dP/dt`` (day/day) at ``dt``.

        Constant for ``constant-pdot``; proportional to ``P(t)`` for the
        exponential model, which is what makes an inspiral accelerate.
        """
        return a * self._pdot_factor(np.asarray(dt, dtype=np.float64), a) * period

    def _pdot_factor(self, dt: FloatArray, a: float) -> FloatArray:
        raise NotImplementedError  # pragma: no cover

    def max_abs_rate(self, dt_extent: float) -> float:
        """Largest ``|a|`` this model can represent over ``+-dt_extent`` days.

        Finite for singular models (the period must stay positive over the
        span), and a loose sanity bound otherwise.
        """
        if dt_extent <= 0.0:
            return float("inf")
        return (1.0 if self.singular else 8.0) / dt_extent

    def fractional_change(self, dt_start: float, dt_end: float, a: float) -> float:
        """``P(dt_end)/P(dt_start) - 1``: the total fractional period change.

        The headline number of a period-change study ("the period shortened by
        1.2% over six years"), computed under this model rather than by
        multiplying a rate by a baseline — which is only correct to first order.
        """
        one = np.array([dt_start, dt_end], dtype=np.float64)
        lo, hi = self._period_factor(one, a)
        if lo == 0.0:
            return float("nan")
        return float(hi / lo - 1.0)


class _ConstantPdot(PeriodModel):
    """``P(t) = P0*(1 + a*dt)``; ``tau = ln(1 + a*dt)/a``."""

    def _warp(self, dt: FloatArray, a: float) -> FloatArray:
        arg = a * dt
        if np.any(arg <= -1.0):
            raise ValueError(
                "constant-pdot warp: 1 + a*dt <= 0 for some point — the period "
                "passes through zero inside the span; narrow the rate range or "
                "use model='exponential', which has no singular epoch"
            )
        return np.asarray(np.log1p(arg) / a, dtype=np.float64)

    def _period_factor(self, dt: FloatArray, a: float) -> FloatArray:
        return np.asarray(1.0 + a * dt, dtype=np.float64)

    def _pdot_factor(self, dt: FloatArray, a: float) -> FloatArray:
        return np.ones_like(np.asarray(dt, dtype=np.float64))


class _Exponential(PeriodModel):
    """``P(t) = P0*exp(a*dt)``; ``tau = (1 - exp(-a*dt))/a``."""

    def _warp(self, dt: FloatArray, a: float) -> FloatArray:
        # -expm1(-x)/a == (1 - exp(-x))/a, accurate as x -> 0.
        return np.asarray(-np.expm1(-a * dt) / a, dtype=np.float64)

    def _period_factor(self, dt: FloatArray, a: float) -> FloatArray:
        return np.asarray(np.exp(a * dt), dtype=np.float64)

    def _pdot_factor(self, dt: FloatArray, a: float) -> FloatArray:
        return np.asarray(np.exp(a * dt), dtype=np.float64)


_REGISTRY: Final[dict[str, PeriodModel]] = {
    "constant-pdot": _ConstantPdot(
        name="constant-pdot",
        description="P(t) = P0*(1 + a*dt); dP/dt constant (quadratic ephemeris)",
        singular=True,
    ),
    "exponential": _Exponential(
        name="exponential",
        description="P(t) = P0*exp(a*dt); d(ln P)/dt constant",
        singular=False,
    ),
}

#: Accepted aliases, so callers can say what they mean.
_ALIASES: Final[dict[str, str]] = {
    "constant": "constant-pdot",
    "constant_pdot": "constant-pdot",
    "linear": "constant-pdot",
    "quadratic": "constant-pdot",
    "exp": "exponential",
    "constant-fractional": "exponential",
}


def get_model(name: str | PeriodModel | None) -> PeriodModel:
    """Look up a period-evolution model by name.

    Parameters
    ----------
    name : str or PeriodModel or None
        ``"constant-pdot"`` (default), ``"exponential"``, an alias
        (``"quadratic"``, ``"exp"``, ...), a model instance, or ``None``.

    Returns
    -------
    PeriodModel
    """
    if isinstance(name, PeriodModel):
        return name
    if name is None:
        return _REGISTRY[DEFAULT_MODEL]
    key = str(name).strip().lower().replace(" ", "-")
    key = _ALIASES.get(key, _ALIASES.get(key.replace("-", "_"), key))
    try:
        return _REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"unknown period-evolution model {name!r}; "
            f"choose from {sorted(_REGISTRY)}"
        ) from None


def available_models() -> tuple[str, ...]:
    """Names of the registered period-evolution models."""
    return tuple(sorted(_REGISTRY))


__all__ = [
    "DEFAULT_MODEL",
    "PeriodModel",
    "available_models",
    "get_model",
]
