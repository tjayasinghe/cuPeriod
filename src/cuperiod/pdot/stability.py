"""The Eddington-Plakidis test: random period wander vs. secular change.

A smooth-looking O-C curve is not evidence of a smooth period change: a star
whose period fluctuates *randomly* from cycle to cycle accumulates a random
walk in the times of maximum, and a parabola will happily fit the result
(Sterken 2005 demonstrates this with literal dice rolls). Eddington & Plakidis
(1929, MNRAS 90, 65) built the classical discriminator. If ``a(E)`` is the O-C
delay at cycle ``E`` and ``u(x) = |a(E+x) - a(E)|`` the accumulated delay
between two maxima ``x`` cycles apart, then under random per-cycle fluctuation

    <u(x)>**2 = 2*alpha**2 + x*epsilon**2,

where ``alpha`` is the mean (absolute) measurement error of one timing and
``epsilon`` the mean (absolute) random fluctuation of the period per cycle: a
line in ``x`` whose slope measures the wander and whose intercept the timing
noise. A significant slope says the O-C is (at least partly) a random walk and
a parabolic dP/dt fit alone should not be trusted.

Two refinements over the classical recipe, both load-bearing:

- A genuine secular dP/dt also grows ``u(x)``, so the quadratic trend is
  removed first (``detrend=2``; Csornyei et al. 2025 make the need explicit).
- Polynomial detrending *absorbs part of the walk itself* — a random walk
  correlates strongly with smooth polynomials, which is Sterken's warning in
  reverse — biasing the naive slope low by tens of percent. This estimator
  therefore fits the *exact* expectation: with ``A = I - H`` the detrending
  projector, ``E[u(i,j)**2]`` is a known linear combination of the per-cycle
  walk variance and the timing variance (propagated through ``A`` and the
  walk covariance ``min(E_i, E_j)``), and those two variances are solved by
  weighted least squares. It reduces to the classical line when detrending
  distorts nothing, and stays unbiased when it does.

``alpha``/``epsilon`` are reported in the 1929 mean-absolute convention
(Gaussian sigma times ``sqrt(2/pi)``); ``alpha_sigma``/``epsilon_sigma`` give
the Gaussian sigmas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.errors import InsufficientDataError

#: Fewest pairs a lag bin needs before its mean enters the fit.
_MIN_PAIRS: int = 5

#: Largest event count for the exact O(N^2) expectation matrices (~200 MB).
_MAX_EVENTS: int = 5000

_MEAN_ABS: float = float(np.sqrt(2.0 / np.pi))


@dataclass(frozen=True)
class EddingtonPlakidisResult:
    """Result of the Eddington-Plakidis fluctuation test.

    Attributes
    ----------
    alpha : float
        Mean absolute timing error of a single time of maximum (days). 0.0
        if the fitted timing variance came out negative. Weakly constrained
        when the wander dominates (its information lives only at zero lag) —
        ``epsilon`` is the primary output of this test.
    epsilon : float
        Mean absolute random period fluctuation per cycle (days). 0.0 if the
        fitted walk variance came out negative.
    alpha_err, epsilon_err : float
        Formal 1-sigma uncertainties (NaN where the parameter was clipped).
    slope_significance : float
        t-statistic of the per-cycle walk variance: how significantly the
        accumulated delays grow with lag. Above ~3, random cycle-to-cycle
        fluctuation is present.
    lags : numpy.ndarray
        Cycle separations with at least the minimum pair count.
    mean_u : numpy.ndarray
        Root-mean-square accumulated delay per lag (days), after detrending.
    n_pairs : numpy.ndarray
        Number of event pairs behind each lag bin.
    detrend : int
        Polynomial degree removed from the O-C before differencing.
    """

    alpha: float
    epsilon: float
    alpha_err: float
    epsilon_err: float
    slope_significance: float
    lags: FloatArray
    mean_u: FloatArray
    n_pairs: FloatArray
    detrend: int

    @property
    def alpha_sigma(self) -> float:
        """Gaussian-sigma equivalent of ``alpha`` (days)."""
        return self.alpha / _MEAN_ABS

    @property
    def epsilon_sigma(self) -> float:
        """Gaussian-sigma equivalent of ``epsilon`` (days)."""
        return self.epsilon / _MEAN_ABS

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalars plus the lag profile."""
        return {
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "alpha_err": self.alpha_err,
            "epsilon_err": self.epsilon_err,
            "alpha_sigma": self.alpha_sigma,
            "epsilon_sigma": self.epsilon_sigma,
            "slope_significance": self.slope_significance,
            "detrend": self.detrend,
            "lags": self.lags.tolist(),
            "mean_u": self.mean_u.tolist(),
            "n_pairs": self.n_pairs.tolist(),
        }


def eddington_plakidis(
    times: Any,
    *,
    period: float,
    t0: float | None = None,
    max_lag: int | None = None,
    detrend: int = 2,
) -> EddingtonPlakidisResult:
    """Test a timing series for random cycle-to-cycle period fluctuations.

    Assigns cycle numbers against the trial ephemeris (like
    :func:`cuperiod.fit_ephemeris`), removes a polynomial trend of degree
    ``detrend``, bins the squared accumulated delays by cycle separation, and
    solves the timing variance and per-cycle walk variance from their exact
    detrending-corrected expectations (see the module docstring).

    Parameters
    ----------
    times : array-like
        Times of maximum/minimum light (days).
    period : float
        Trial period (days).
    t0 : float, optional
        Trial epoch; defaults to the first event.
    max_lag : int, optional
        Largest cycle separation used; default is a third of the span in
        cycles, capped at 100.
    detrend : int, default 2
        Polynomial degree removed from the O-C first. 2 removes a secular
        quadratic (a real dP/dt) so it cannot masquerade as wander; use 1 to
        test against a linear ephemeris only.

    Returns
    -------
    EddingtonPlakidisResult

    Examples
    --------
    >>> ep = eddington_plakidis(maxima, period=331.0)      # doctest: +SKIP
    >>> ep.epsilon, ep.slope_significance                  # doctest: +SKIP

    Notes
    -----
    The classical estimator with an exact-expectation correction; for
    likelihood-based model selection between jitter and period-change models
    see Koen (2006, MNRAS 365, 489). Cycle numbering shares
    :func:`cuperiod.fit_ephemeris`'s nearest-integer caveat.
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    t = np.sort(t[np.isfinite(t)])
    if period <= 0.0:
        raise ValueError(f"period must be positive, got {period}")
    if not 0 <= detrend <= 4:
        raise ValueError(f"detrend must be between 0 and 4, got {detrend}")
    if t.size < max(8, detrend + 4):
        raise InsufficientDataError(
            f"eddington_plakidis: need at least {max(8, detrend + 4)} event "
            f"times, got {t.size}"
        )
    epoch = float(t[0]) if t0 is None else float(t0)
    cycles = np.round((t - epoch) / period)
    oc = t - epoch - period * cycles
    # Duplicate cycle assignments (two timings of the same event) are averaged.
    uniq, inverse = np.unique(cycles, return_inverse=True)
    if uniq.size < t.size:
        dup_sums = np.zeros(uniq.size)
        dup_counts = np.zeros(uniq.size)
        np.add.at(dup_sums, inverse, oc)
        np.add.at(dup_counts, inverse, 1.0)
        oc = dup_sums / dup_counts
        cycles = uniq
    n = cycles.size
    if n > _MAX_EVENTS:
        raise InsufficientDataError(
            f"eddington_plakidis: {n} distinct cycles exceeds the exact-"
            f"estimator cap of {_MAX_EVENTS}; thin the timing list"
        )

    span = int(cycles[-1] - cycles[0])
    top = max(3, min(span // 3, 100))
    if max_lag is not None:
        top = min(int(max_lag), span)
    if top < 3:
        raise InsufficientDataError(
            "eddington_plakidis: too few distinct cycles for a lag profile"
        )

    # Detrending projector A = I - H and the detrended delays.
    basis = np.vander(cycles - cycles.mean(), detrend + 1, increasing=True)
    q, _ = np.linalg.qr(basis)
    a_mat = np.eye(n) - q @ q.T
    resid = a_mat @ oc

    # Exact expectations: with walk covariance M_ij = min(E_i, E_j) (cycles
    # since the first event) and white timing noise, the detrended covariance
    # is  s_eps^2 * A M A^T + s_alpha^2 * A A^T,  so every pair's E[u^2] is
    # linear in the two variances.
    e_rel = cycles - cycles[0]
    walk_cov = np.minimum.outer(e_rel, e_rel)
    b_walk = a_mat @ walk_cov @ a_mat.T
    b_noise = a_mat @ a_mat.T

    ii, jj = np.triu_indices(n, k=1)
    lag_flat = (cycles[jj] - cycles[ii]).astype(np.int64)
    keep = (lag_flat >= 1) & (lag_flat <= top)
    ii, jj, lag_flat = ii[keep], jj[keep], lag_flat[keep]
    if lag_flat.size == 0:
        raise InsufficientDataError("eddington_plakidis: no usable pairs")
    d = resid[jj] - resid[ii]
    u2 = d * d
    dw = np.diag(b_walk)
    dn = np.diag(b_noise)
    g_walk = dw[ii] + dw[jj] - 2.0 * b_walk[ii, jj]
    g_noise = dn[ii] + dn[jj] - 2.0 * b_noise[ii, jj]

    sums_u2 = np.bincount(lag_flat, weights=u2, minlength=top + 1)
    sums_gw = np.bincount(lag_flat, weights=g_walk, minlength=top + 1)
    sums_gn = np.bincount(lag_flat, weights=g_noise, minlength=top + 1)
    counts = np.bincount(lag_flat, minlength=top + 1)
    good = np.flatnonzero(counts >= _MIN_PAIRS)
    good = good[good >= 1]
    if good.size < 3:
        raise InsufficientDataError(
            "eddington_plakidis: fewer than 3 lag bins with "
            f"{_MIN_PAIRS}+ pairs; supply more event times or raise max_lag"
        )
    m = counts[good].astype(np.float64)
    y = sums_u2[good] / m
    x_mat = np.stack([sums_gn[good] / m, sums_gw[good] / m], axis=1)

    # Weighted LSQ for (sigma_alpha^2, sigma_eps^2). A bin mean of squared
    # Gaussian differences has variance ~ 2*E[u^2]^2/m, so after a count-
    # weighted first pass the fit is reweighted once by the model-predicted
    # variance — without this, the large-lag bins (huge u^2, huge scatter)
    # drown the timing-noise term.
    weights = m.copy()
    coef = np.zeros(2)
    cov = np.eye(2)
    for _ in range(2):
        xtw = x_mat.T * weights[None, :]
        cov = np.asarray(np.linalg.inv(xtw @ x_mat), dtype=np.float64)
        coef = np.asarray(cov @ (xtw @ y), dtype=np.float64)
        floor = 0.05 * float(np.median(y))
        predicted = np.maximum(x_mat @ coef, max(floor, np.finfo(float).tiny))
        weights = m / (2.0 * predicted * predicted)
    fit_resid = y - x_mat @ coef
    dof = max(1, good.size - 2)
    s2 = float(np.dot(weights, fit_resid * fit_resid)) / dof
    cov = cov * s2
    var_alpha, var_eps = float(coef[0]), float(coef[1])
    var_alpha_err = float(np.sqrt(cov[0, 0]))
    var_eps_err = float(np.sqrt(cov[1, 1]))

    sigma_alpha = float(np.sqrt(var_alpha)) if var_alpha > 0.0 else 0.0
    sigma_eps = float(np.sqrt(var_eps)) if var_eps > 0.0 else 0.0
    alpha = sigma_alpha * _MEAN_ABS
    epsilon = sigma_eps * _MEAN_ABS
    # d(sigma)/d(var) = 1/(2*sigma); the mean-abs factor carries through.
    alpha_err = (
        _MEAN_ABS * var_alpha_err / (2.0 * sigma_alpha)
        if sigma_alpha > 0.0
        else float("nan")
    )
    epsilon_err = (
        _MEAN_ABS * var_eps_err / (2.0 * sigma_eps)
        if sigma_eps > 0.0
        else float("nan")
    )
    significance = var_eps / var_eps_err if var_eps_err > 0.0 else float("inf")

    return EddingtonPlakidisResult(
        alpha=alpha,
        epsilon=epsilon,
        alpha_err=alpha_err,
        epsilon_err=epsilon_err,
        slope_significance=float(significance),
        lags=good.astype(np.float64),
        mean_u=np.sqrt(y),
        n_pairs=m,
        detrend=detrend,
    )


__all__ = ["EddingtonPlakidisResult", "eddington_plakidis"]
